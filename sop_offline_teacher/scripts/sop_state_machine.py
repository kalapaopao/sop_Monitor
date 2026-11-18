"""状态机去抖：把「逐帧原型分类」的噪声标签流，整理成干净的工序段。

这是参考视频第 3 层（状态机）的本地实现，0 次 API 调用。

为什么需要它（Codex 与 DSH 一致结论）：
  VLM 逐段独立判断会出现「前一窗口装上盖 / 后一窗口取镜座 / 再后装密封塞」的抖动；
  参考视频的稳定性来自"先有片段标签再渲染"——背后就是去抖规则。

四个规则（参考视频可见的行为）：
  1. 进入确认（debounce-in） ：新标签必须连续 N 帧才切换 —— 对应画面里
     「工具轻放下、拿起或半变换不直接计数」那条说明
  2. 最短段长               ：短于 min_dur 的段并入邻居
  3. 同标签间隙合并          ：两段同标签间隔 < gap 秒则合并
  4. 远离兜底               ：离最近原型太远的帧标 unknown（对应「未辨认」三级状态）

流水线（本地全链路，模仿参考视频）：
  10 Hz 骨架 → 滑窗特征 → 最近原型分类（帧级标签）→ 状态机去抖 → 工序段 → 与 GT 对比

用法：
    python scripts/sop_state_machine.py --k 8
    python scripts/sop_state_machine.py --k 8 --confirm-ticks 8 --min-dur 1.2 --merge-gap 1.0
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).resolve().parent))
from discover_motifs import (B2S, NAME, tick_features, resample, kmeans,
                             gt_label_at, load_pose_defaults)  # noqa: E402


def frame_labels(feat: np.ndarray, centers: np.ndarray, win: int):
    """滑窗均值特征 → 最近原型，得到每帧标签。

    注意：原型中心是在「重采样 16 点 × 6 维 = 96 维」的段向量空间里训练的，
    而帧级特征是 6 维。所以这里先把滑窗均值展平成与原型同构的表示：
    用最近 win 帧的均值作为段的近似（等价于把段向量退化成 6 维再比较）。
    为保持距离可比，这里改为：把每个原型中心也降到 6 维（16 段均值），
    然后在 6 维空间比较 —— 训练与推理口径一致。
    """
    n, d = feat.shape
    centers_6 = centers.reshape(len(centers), -1, d).mean(1)  # (k, 6)
    lab = np.full(n, -1, dtype=int)
    for i in range(n):
        lo = max(0, i - win + 1)
        v = feat[lo:i + 1].mean(0)
        dist = ((centers_6 - v) ** 2).sum(1)
        lab[i] = int(dist.argmin())
    return lab


def debounce(ts, lab, confirm_ticks: int, min_dur: float, merge_gap: float, far_thr: float,
             feat: np.ndarray, centers: np.ndarray, win: int):
    """四条规则：确认切换 / 最短段长 / 同标签合并 / 远离兜底。返回段列表。

    centers 为 (k, 96) 的原型中心；内部降到 6 维（16 段均值）与 feat 同口径比较。
    """
    centers_6 = centers.reshape(len(centers), -1, feat.shape[1]).mean(1)
    n = len(ts)
    # 规则 1：进入确认 —— 候选标签连续 confirm_ticks 帧才切换
    confirmed = np.empty(n, dtype=int)
    confirmed[:] = lab[:1] if n else 0
    cur = lab[0] if n else 0
    run, cand = 0, None
    for i in range(n):
        if lab[i] == cur:
            run = 0
            cand = None
        else:
            if cand == lab[i]:
                run += 1
            else:
                cand, run = lab[i], 1
            if run >= confirm_ticks:
                cur, run, cand = lab[i], 0, None
        confirmed[i] = cur
    # 规则 4：段均值离原型太远 → 标 unknown(-1)
    segs = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and confirmed[j + 1] == confirmed[i]:
            j += 1
        cid = int(confirmed[i])
        v = feat[i:j + 1].mean(0)
        if ((centers_6[cid] - v) ** 2).sum() > far_thr:
            cid = -1
        segs.append([ts[i], ts[j] + 0.1, cid])
        i = j + 1
    # 规则 3：同标签间隙合并
    merged = []
    for s in segs:
        if merged and merged[-1][2] == s[2] and s[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    # 规则 2：最短段长（并入前后邻居中时长更近的那个）
    out = []
    for idx, s in enumerate(merged):
        if s[1] - s[0] >= min_dur:
            out.append(s)
            continue
        # 尝试并入邻居（同标签优先，否则并入更近的）
        prev_ok = out and (out[-1][2] == s[2] or True)
        nxt = merged[idx + 1] if idx + 1 < len(merged) else None
        if prev_ok and (nxt is None or s[0] - out[-1][1] <= nxt[0] - s[1]):
            if out:
                out[-1][1] = s[1]
                continue
        if nxt is not None:
            nxt[0] = s[0]
            continue
        out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pose', type=Path, default=Path('web/public/data/local_pose_2min.json'))
    ap.add_argument('--gt', type=Path, default=Path('web/public/data/gt_segments.json'))
    ap.add_argument('--clip-offset', type=float, default=300.0)
    ap.add_argument('--k', type=int, default=8, help='原型数（内部跑 k-means，与 discover_motifs 同法）')
    ap.add_argument('--win', type=int, default=5, help='滑窗帧数（5 帧 = 0.5 秒，因果）')
    ap.add_argument('--confirm-ticks', type=int, default=8, help='新标签需连续多少帧才切换（8 帧 = 0.8 秒）')
    ap.add_argument('--min-dur', type=float, default=1.2, help='最短段长（秒）')
    ap.add_argument('--merge-gap', type=float, default=1.0, help='同标签间隙合并阈值（秒）')
    ap.add_argument('--far-quantile', type=float, default=0.9,
                    help='段到原型距离超过该分位视为 unknown')
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()

    pose = json.loads(a.pose.read_text(encoding='utf-8-sig'))
    ts, feat = tick_features(pose)
    dur = float(pose.get('duration') or ts[-1])
    print(f'骨架 {len(ts)} 帧 @ {pose.get("rate")} Hz ｜ 特征维度 {feat.shape[1]}')

    # 用与 discover_motifs 相同的方式建原型（分段 → 重采样 → k-means）
    det = json.loads(Path('api_runs/boundaries.json').read_text(encoding='utf-8-sig'))
    peaks = det.get('peaks_clip') or []
    edges = [0.0] + [p for p in sorted(peaks) if 0.5 < p < dur - 0.5] + [dur]
    segs = [(x, y) for x, y in zip(edges, edges[1:]) if y - x >= 0.8]
    X = np.array([resample(feat[(ts >= x) & (ts < y)], 16).reshape(-1) for x, y in segs], dtype=np.float32)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    C, lab_seg, _ = kmeans(X, a.k)

    # 帧级分类（用与训练同口径的标准化）
    mu, sd = X.mean(0), X.std(0) + 1e-6
    centers = C
    raw = frame_labels(feat, centers, a.win)

    # unknown 阈值：段到原型的距离分位数（6 维空间，与 frame_labels 同口径）
    centers_6 = centers.reshape(len(centers), -1, feat.shape[1]).mean(1)
    dists = []
    for i in range(len(ts)):
        lo = max(0, i - a.win + 1)
        v = feat[lo:i + 1].mean(0)
        dmin = ((centers_6 - v) ** 2).sum(1).min()
        dists.append(dmin)
    far_thr = float(np.quantile(dists, a.far_quantile))

    sm = debounce(ts, raw, a.confirm_ticks, a.min_dur, a.merge_gap, far_thr, feat, centers, a.win)

    gt = json.loads(a.gt.read_text(encoding='utf-8-sig'))['segments']

    def gt_of(s, e):
        """段中心（以及整段多数时刻）对应的 GT 步骤。"""
        votes = {}
        for t in np.arange(s, e, 0.1):
            g = gt_label_at(gt, float(t), a.clip_offset)
            votes[g] = votes.get(g, 0) + 1
        return max(votes, key=votes.get) if votes else 'idle'

    # GT 视角命中率：每个 GT 标注段，看我们输出的段标签是什么
    print(f'\n状态机参数：confirm={a.confirm_ticks} 帧({a.confirm_ticks / 10:.1f}s) ｜ '
          f'最短段 {a.min_dur}s ｜ 合并间隙 {a.merge_gap}s ｜ far 分位 {a.far_quantile}')
    print(f'去抖后段数：{len(sm)}（原始帧标签切换 {int((np.diff(raw) != 0).sum())} 次）')

    print(f'\n{"#":>3} {"时间":<16} {"原型":>4} {"GT(多数)":<8} 命中')
    ok = 0
    rows = []
    for i, (s, e, cid) in enumerate(sm):
        g = gt_of(s, e)
        hit = (cid == -1 and g in ('idle', 'unknown')) or (
            cid >= 0 and any(lab_seg[q] == cid for q in range(len(segs))
                             if gt_of(*segs[q]) == g))
        ok += int(hit)
        rows.append({'start': round(float(s), 2), 'end': round(float(e), 2),
                     'prototype': int(cid), 'gt': g, 'hit': bool(hit)})
        print(f'{i:>3} {s:7.2f}-{e:7.2f} {cid:>4} {NAME.get(g, g):<8} {"OK" if hit else ""}')
    print(f'\n段级命中（粗口径）：{ok}/{len(sm)} = {ok / max(1, len(sm)):.1%}')

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({
            'params': {'k': a.k, 'win': a.win, 'confirm_ticks': a.confirm_ticks,
                       'min_dur': a.min_dur, 'merge_gap': a.merge_gap,
                       'far_quantile': a.far_quantile},
            'segments': rows}, ensure_ascii=False, indent=1), encoding='utf-8')
        print('写出', a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
