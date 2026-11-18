"""TCN 训练/评估（第 2 步）：与 MLP 版同口径的 7 周期留一验证。

与 MLP 版的唯一区别：模型看了 15 帧（1.5 秒）窗口，而不是单帧。
其余全部保持可比：同样的特征（pose_features_v2，105 维）、同样的标签规则、
同样的 invalid 屏蔽、同样的 skip_partial 与周期划分。

用法：
    python scripts/train_tcn.py --pose web/public/data/local_pose_full.json \
        --labels web/public/data/frame_labels_full.json --clip-offset 0 \
        --feat v2 --win 15 --epochs 200 \
        --save-pred api_runs/pred_tcn.json --out api_runs/baseline_tcn.json
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_pose_baseline import CLASSES, C2I, cycle_of, report  # noqa: E402
from tcn_model import TCN, count_params  # noqa: E402


def build_sequences(X: np.ndarray, win: int):
    """因果窗口：为每帧构造 [t-win+1, t] 的特征序列，左侧用首帧复制填充。"""
    n, d = X.shape
    pad = np.repeat(X[:1], win - 1, axis=0)
    Xp = np.concatenate([pad, X], axis=0)          # (win-1 + n, d)
    idx = np.arange(n)[:, None] + np.arange(win)[None, :]
    return Xp[idx].astype(np.float32)              # (n, win, d)


def train_predict(Xall, y, tr_idx, te_idx, win, epochs, seed=0,
                  loss_kind='ce', focal_gamma=2.0, oversample=False):
    torch.manual_seed(seed)
    dev = 'cpu'
    model = TCN(Xall.shape[2], len(CLASSES)).to(dev)   # shape[2]=特征维（shape[1] 是窗口长度）
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    cnt = Counter(y[tr_idx].tolist())
    w = torch.tensor([len(tr_idx) / (len(CLASSES) * max(1, cnt.get(i, 0)))
                      for i in range(len(CLASSES))], dtype=torch.float32, device=dev)
    lossf = nn.CrossEntropyLoss(weight=w)

    # 为了效率：整段前向，用 mask 取需要的位置
    xt = torch.tensor(Xall, device=dev)             # (N, win, D)
    yt = torch.tensor(y, device=dev)
    tr_mask = torch.zeros(len(y), dtype=torch.bool, device=dev)
    tr_mask[torch.tensor(tr_idx, device=dev)] = True
    bs = 512
    if oversample:
        cnt_arr = np.array([cnt.get(k, 0) for k in range(len(CLASSES))], dtype=np.float64)
        w_per = np.where(cnt_arr > 0,
                        (cnt_arr.sum() / (len(CLASSES) * np.maximum(cnt_arr, 1))) ** 0.5, 0.0)
        reps = np.maximum(1, np.round(w_per[tr_idx] / max(1e-6, w_per.min()))).astype(int)
        order = np.repeat(np.array(tr_idx), reps)
        print(f'    [oversample] 训练索引 {len(tr_idx)} → {len(order)}')
    else:
        order = np.array(tr_idx)
    for ep in range(epochs):
        model.train()
        np.random.default_rng(ep).shuffle(order)
        for i in range(0, len(order), bs):
            b = torch.tensor(order[i:i + bs], device=dev)
            opt.zero_grad()
            out = model(xt[b])                      # (B, win, C)
            logits = out[:, -1, :]                  # 只用窗口最后一帧
            if loss_kind == 'focal':
                ce = nn.functional.cross_entropy(logits, yt[b], weight=w, reduction='none')
                pt = torch.exp(-ce)
                loss = ((1 - pt) ** focal_gamma * ce).mean()
            else:
                loss = lossf(logits, yt[b])
            loss.backward()
            opt.step()
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(te_idx), 1024):
            b = torch.tensor(np.array(te_idx[i:i + 1024]), device=dev)
            out = model(xt[b])
            preds.append(out[:, -1, :].argmax(1).cpu().numpy())
    return np.concatenate(preds), model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pose', type=Path, default=Path('web/public/data/local_pose_full.json'))
    ap.add_argument('--labels', type=Path, required=True)
    ap.add_argument('--gt', type=Path, default=Path('web/public/data/gt_segments.json'))
    ap.add_argument('--stations', type=Path, default=Path('configs/stations.json'))
    ap.add_argument('--clip-offset', type=float, default=0.0)
    ap.add_argument('--feat', choices=['v2', 'v3'], default='v2',
                    help='v2=手部特征(105维) v3=手部+身体/手臂(153维，手部失败时用 body 腕兜底)')
    ap.add_argument('--appearance', type=Path, default=None,
                    help='手周外观特征 json（hand_crop_features.py 产物），追加到骨架特征后')
    ap.add_argument('--no-station', action='store_true',
                    help='消融：不使用固定工位区特征（衡量模型对 ROI 框的依赖）')
    ap.add_argument('--win', type=int, default=15)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--min-cycle-frames', type=int, default=30)
    ap.add_argument('--loss', choices=['ce', 'focal'], default='ce',
                    help='focal 抑制大量容易的 idle 帧主导梯度，提升稀有类召回')
    ap.add_argument('--focal-gamma', type=float, default=2.0)
    ap.add_argument('--oversample', action='store_true',
                    help='按稀有类过采样（S1/S2/S6 帧数少，重采样让它们多参与训练）')
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--save-pred', type=Path, default=None)
    a = ap.parse_args()

    if a.feat == 'v3':
        from pose_features_v3 import build_features_v3
        t, X, dbase = build_features_v3(a.pose, a.stations, use_station=not a.no_station)
    else:
        from pose_features_v2 import build_features
        t, X, dbase = build_features(a.pose, a.stations)
    if a.appearance and a.appearance.exists():
        appj = json.loads(a.appearance.read_text(encoding='utf-8-sig'))
        rows_a = [list(r.get('left') or []) + list(r.get('right') or []) for r in appj['frames']]
        adim = max(len(r) for r in rows_a)          # 不同帧可能长度不一（缺失帧用短零向量）
        amap = {round(float(r['t']), 2): np.array((ra + [0.0] * adim)[:adim], dtype=np.float32)
                for r, ra in zip(appj['frames'], rows_a)}
        A = np.stack([amap.get(round(float(x), 2), np.zeros(adim, dtype=np.float32)) for x in t])
        X = np.concatenate([X, A], axis=1)
        dbase += adim
        print(f'[appearance] 追加 {adim} 维 → 总 {X.shape[1]} 维')
    labj = json.loads(a.labels.read_text(encoding='utf-8-sig'))
    frames = labj['frames']
    n = min(len(t), len(frames))
    t, X, frames = t[:n], X[:n], frames[:n]
    y = np.array([C2I.get(f['label'], -1) for f in frames])
    mask = np.array([f['label'] != 'invalid' for f in frames])
    print(f'[feat {a.feat}] 帧 {n} ｜ 特征 {dbase} 维 → 窗口 {a.win} 帧（{a.win * 0.1:.1f}s）｜ '
          f'屏蔽 invalid {int((~mask).sum())}')

    gt = json.loads(a.gt.read_text(encoding='utf-8-sig'))
    cycles = gt['cycles']
    cyc = np.array([cycle_of(float(x) + a.clip_offset, cycles) for x in t])
    usable = mask & (y >= 0) & (cyc >= 0)
    cnt = Counter(cyc[usable].tolist())
    print('各周期可用帧：', dict(sorted(cnt.items())))
    cycles_ok = [c for c, k in sorted(cnt.items()) if k >= a.min_cycle_frames]

    Xs = build_sequences(X, a.win)
    proto = TCN(Xs.shape[2], len(CLASSES))
    print(f'TCN 参数量 {count_params(proto):,} ｜ 感受野 {proto.receptive_field()} 帧 '
          f'({proto.receptive_field() * 0.1:.1f}s)')

    all_true, all_pred, all_t, per_cycle = [], [], [], {}
    for c in cycles_ok:
        tr = np.where(usable & (cyc != c))[0]
        te = np.where(usable & (cyc == c))[0]
        if len(tr) < 50 or len(te) < 20:
            continue
        pred, _ = train_predict(Xs, y, tr, te, a.win, a.epochs,
                                 loss_kind=a.loss, focal_gamma=a.focal_gamma,
                                 oversample=a.oversample)
        acc = float((pred == y[te]).mean())
        per_cycle[c] = acc
        print(f'\n--- 留出周期 {c}：训练 {len(tr)} 帧 / 验证 {len(te)} 帧 → 帧级 {acc:.1%}')
        all_true.append(y[te]); all_pred.append(pred); all_t.append(t[te])

    if not all_true:
        sys.exit('没有可用周期')
    yt = np.concatenate(all_true); yp = np.concatenate(all_pred); tt = np.concatenate(all_t)
    acc, cm = report(yt, yp, f'TCN 留一周期交叉验证（窗口 {a.win} 帧）')
    print('\n逐周期帧级准确率：', {k: f'{v:.1%}' for k, v in per_cycle.items()})
    print(f'\n**TCN 基线：帧级 {acc:.1%}**（与 MLP 同口径、无泄漏）')

    if a.save_pred:
        a.save_pred.parent.mkdir(parents=True, exist_ok=True)
        order = np.argsort(tt)
        a.save_pred.write_text(json.dumps({
            'classes': CLASSES, 'feat': a.feat, 'model': 'tcn', 'win': a.win, 'loss': a.loss, 'oversample': a.oversample,
            'clip_offset': a.clip_offset, 'frame_accuracy': round(float(acc), 4),
            'frames': [{'t': float(tt[i]), 'true': int(yt[i]), 'pred': int(yp[i])}
                       for i in order]}, ensure_ascii=False, indent=1), encoding='utf-8')
        print('写出预测', a.save_pred)

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({
            'model': 'tcn', 'win': a.win, 'feat': a.feat, 'loss': a.loss, 'oversample': a.oversample,
            'frame_accuracy': round(float(acc), 4), 'per_cycle': per_cycle,
            'confusion': cm.tolist(), 'classes': CLASSES}, ensure_ascii=False, indent=1),
            encoding='utf-8')
        print('写出', a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
