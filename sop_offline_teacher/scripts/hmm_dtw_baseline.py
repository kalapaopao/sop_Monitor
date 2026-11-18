"""不用 TCN 的两种时序方案（HMM / DTW 原型），与 TCN 同口径对比。

为什么提供这两个：
  · 用户明确要求不使用 TCN（父项目里"骨架+检测+状态机+TCN"试过没做成）
  · 但 SOP 步骤识别**必须**有时序上下文，否则只能做逐帧分类（v2 的 MLP 只有 48.5%）
  · 这两条都不含可训练深度时序网络：
      HMM  —— 发射概率用每类高斯（对角协方差）+ 转移矩阵从训练周期统计，Viterbi 解码
      DTW  —— 每类做一个"模板序列"，滑窗与模板做动态时间规整，取最小距离

评估口径与 TCN 完全一致：7 个完整周期 leave-one-cycle-out、同特征(v3)、同 invalid 屏蔽。

用法：
    python scripts/hmm_dtw_baseline.py --pose web/public/data/local_pose_full_v2.json \
        --labels web/public/data/frame_labels_full.json --clip-offset 0
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_pose_baseline import CLASSES, C2I, cycle_of, report  # noqa: E402

LOG0 = -1e3


def gauss_emissions(Xtr, ytr, Xte, min_std=1e-3):
    """每类对角高斯 → 对数发射概率（标准化后计算，稳定）。"""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    emits = np.full((len(Xte), len(CLASSES)), LOG0, dtype=np.float64)
    for ci in range(len(CLASSES)):
        m = ytr == ci
        if m.sum() < 5:
            continue
        cm, cs = Ztr[m].mean(0), Ztr[m].std(0) + min_std
        ll = -0.5 * (((Zte - cm) / cs) ** 2 + np.log(cs ** 2)).sum(1)
        emits[:, ci] = ll
    return emits


def mlp_probs(Xtr, ytr, Xte, epochs=300, seed=0, hidden=64):
    """逐帧 MLP（无时序）→ 返回测试帧的类别概率。"""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Ztr = ((Xtr - mu) / sd).astype(np.float32)
    Zte = ((Xte - mu) / sd).astype(np.float32)
    K = len(CLASSES)
    model = nn.Sequential(nn.Linear(Ztr.shape[1], hidden), nn.ReLU(), nn.Dropout(0.2),
                          nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, K))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    cnt = Counter(ytr.tolist())
    w = torch.tensor([len(ytr) / (K * max(1, cnt.get(i, 0))) for i in range(K)], dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=w)
    xt, yt = torch.tensor(Ztr), torch.tensor(ytr)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = lossf(model(xt), yt)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(Zte))
        return torch.log_softmax(logits, dim=1).numpy()


def trans_matrix(ytr, cyc_tr, alpha=0.5):
    """从训练序列统计转移（按周期的连续帧，allowed 帧之间视为连续）。"""
    order = np.argsort(cyc_tr)
    cnt = Counter()
    prev = None
    for i in order:
        cur = int(ytr[i])
        if prev is not None:
            cnt[(prev, cur)] += 1
        prev = cur
    K = len(CLASSES)
    T = np.full((K, K), alpha, dtype=np.float64)
    for (a, b), c in cnt.items():
        T[a, b] += c
    T = T / T.sum(1, keepdims=True)
    return np.log(T)


def viterbi(emits, logT):
    n, K = emits.shape
    dp = np.full((n, K), -np.inf)
    bp = np.zeros((n, K), dtype=np.int32)
    dp[0] = emits[0]
    for t in range(1, n):
        for k in range(K):
            cand = dp[t - 1] + logT[:, k]
            j = int(np.argmax(cand))
            dp[t, k] = cand[j] + emits[t, k]
            bp[t, k] = j
    path = np.zeros(n, dtype=np.int32)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(n - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]
    return path


def dtw(a, b):
    na, nb = len(a), len(b)
    d = np.full((na + 1, nb + 1), np.inf)
    d[0, 0] = 0.0
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            cost = np.linalg.norm(a[i - 1] - b[j - 1])
            d[i, j] = cost + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
    return d[na, nb] / (na + nb)


def dtw_classify(Xtr, ytr, Xte, win=15, n_proto=8):
    """每类取若干模板（训练集中该类连续片段重采样到 n_proto 帧），滑窗最小 DTW 距离。"""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    protos = {}
    for ci in range(len(CLASSES)):
        idx = np.where(ytr == ci)[0]
        if len(idx) < win:
            continue
        # 取若干互不重叠的窗口作为模板
        picks = []
        step = max(win, len(idx) // 4)
        for s in range(0, len(idx) - win + 1, step):
            seg = Ztr[idx[s:s + win]]
            k = np.linspace(0, len(seg) - 1, n_proto).astype(int)
            picks.append(seg[k])
            if len(picks) >= 4:
                break
        if picks:
            protos[ci] = np.stack(picks)
    pred = np.zeros(len(Zte), dtype=np.int32)
    for t in range(len(Zte)):
        lo = max(0, t - win + 1)
        seg = Zte[lo:t + 1]
        if len(seg) < 5:
            pred[t] = C2I["idle_auxiliary"]
            continue
        k = np.linspace(0, len(seg) - 1, n_proto).astype(int)
        q = seg[k]
        best, bc = np.inf, C2I["idle_auxiliary"]
        for ci, ps in protos.items():
            for p in ps:
                v = dtw(q, p)
                if v < best:
                    best, bc = v, ci
        pred[t] = bc
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", type=Path, default=Path("web/public/data/local_pose_full_v2.json"))
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--gt", type=Path, default=Path("web/public/data/gt_segments.json"))
    ap.add_argument("--stations", type=Path, default=Path("configs/stations.json"))
    ap.add_argument("--clip-offset", type=float, default=0.0)
    ap.add_argument("--min-cycle-frames", type=int, default=30)
    ap.add_argument("--no-station", action="store_true", help="消融：不使用固定工位区特征")
    ap.add_argument("--methods", default="hmm,dtw")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--save-pred", type=Path, default=None,
                    help="导出逐帧预测（供 eval_segments/publish_p3_baseline 使用）")
    ap.add_argument("--save-for", default="mlp_hmm", help="为哪个方法导出预测")
    a = ap.parse_args()

    from pose_features_v3 import build_features_v3
    t, X, dbase = build_features_v3(a.pose, a.stations, use_station=not a.no_station)
    labj = json.loads(a.labels.read_text(encoding="utf-8-sig"))
    frames = labj["frames"]
    n = min(len(t), len(frames))
    t, X, frames = t[:n], X[:n], frames[:n]
    y = np.array([C2I.get(f["label"], -1) for f in frames])
    mask = np.array([f["label"] != "invalid" for f in frames])
    print(f"帧 {n} ｜ 特征 {dbase} 维（v3）｜ 屏蔽 invalid {int((~mask).sum())}")

    gt = json.loads(a.gt.read_text(encoding="utf-8-sig"))
    cyc = np.array([cycle_of(float(x) + a.clip_offset, gt["cycles"]) for x in t])
    usable = mask & (y >= 0) & (cyc >= 0)
    cnt = Counter(cyc[usable].tolist())
    cycles_ok = [c for c, k in sorted(cnt.items()) if k >= a.min_cycle_frames]
    print("各周期可用帧：", dict(sorted(cnt.items())))

    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    summary = {}
    for method in methods:
        all_t, all_p, per_cycle = [], [], {}
        for c in cycles_ok:
            tr = np.where(usable & (cyc != c))[0]
            te = np.where(usable & (cyc == c))[0]
            if len(tr) < 50 or len(te) < 20:
                continue
            if method == "hmm":
                emits = gauss_emissions(X[tr], y[tr], X[te])
                logT = trans_matrix(y[tr], cyc[tr])
                pred = viterbi(emits, logT)
            elif method == "mlp_hmm":
                emits = mlp_probs(X[tr], y[tr], X[te])
                logT = trans_matrix(y[tr], cyc[tr])
                pred = viterbi(emits, logT)
            elif method == "mlp":
                emits = mlp_probs(X[tr], y[tr], X[te])
                pred = emits.argmax(1)
            elif method == "dtw":
                pred = dtw_classify(X[tr], y[tr], X[te])
            else:
                raise SystemExit(f"未知方法 {method}")
            acc = float((pred == y[te]).mean())
            per_cycle[c] = acc
            all_t.append(t[te]); all_p.append(pred)
            print(f"  [{method}] 留出周期 {c}: {len(tr)} 训练 / {len(te)} 验证 → {acc:.1%}")
        if not all_t:
            continue
        yt = y[np.concatenate([np.where(usable & (cyc == c))[0] for c in per_cycle])]
        yp = np.concatenate(all_p)
        acc, cm = report(yt, yp, f"{method.upper()} 留一周期交叉验证")
        summary[method] = {"frame_accuracy": round(float(acc), 4),
                           "per_cycle": per_cycle,
                           "confusion": cm.tolist()}
        if a.save_pred and method == a.save_for:
            a.save_pred.parent.mkdir(parents=True, exist_ok=True)
            a.save_pred.write_text(json.dumps({
                "classes": CLASSES, "feat": "v3", "model": method,
                "clip_offset": a.clip_offset, "frame_accuracy": round(float(acc), 4),
                "frames": [{"t": float(tt), "true": int(yy), "pred": int(pp)}
                           for tt, yy, pp in zip(np.concatenate(all_t), yt, yp)]},
                ensure_ascii=False, indent=1), encoding="utf-8")
            print("写出预测", a.save_pred)
        print(f"\n**{method.upper()} 帧级 {acc:.1%}**\n")

    if a.out and summary:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"classes": CLASSES, "feat": "v3", "results": summary},
                                    ensure_ascii=False, indent=1), encoding="utf-8")
        print("写出", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
