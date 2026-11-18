"""用骨架/手部几何在本地找「动作切换点」（变化点检测），完全不调用大模型。

动机（有实测支撑）：
  你那 120 秒里 GT 段时长中位数只有 2.36 秒，而 8 帧/30 秒的采样间隔是 4.28 秒，
  55% 的步骤比采样间隔还短 —— 大模型在 4 秒一帧的条件下"看不见"它们，边界必然粗。
  但骨架是 10 Hz（0.1 秒一帧），做"手有没有突然改变运动状态"这件事精度高得多。

做法：
  1) 每个 10 Hz tick 取特征：左右手腕坐标、手部包围盒尺寸、两手间距、运动速度
  2) 计算滑窗内的"特征变化强度"（多通道速度 + 尺寸变化）
  3) 局部极大值 + 最小间隔 → 候选边界
  4) 与人工 GT 边界比对（±tol），给出召回/精确/时间误差

用法：
    python scripts/detect_boundaries.py --pose web/public/data/local_pose_2min.json \
        --gt web/public/data/gt_segments.json --tol 1.0
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def point_center(pts):
    if not pts:
        return None
    xs = [p[0] for p in pts if p[2] > 0.18]
    ys = [p[1] for p in pts if p[2] > 0.18]
    if not xs:
        return None
    return sum(xs) / len(xs), sum(ys) / len(ys)


def hand_extent(pts):
    if not pts:
        return None
    xs = [p[0] for p in pts if p[2] > 0.18]
    ys = [p[1] for p in pts if p[2] > 0.18]
    if len(xs) < 2:
        return None
    return ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5


def build_features(frames):
    feats = []
    for f in frames:
        lc = point_center(f.get('left_hand'))
        rc = point_center(f.get('right_hand'))
        le = hand_extent(f.get('left_hand'))
        re = hand_extent(f.get('right_hand'))
        feats.append({'t': float(f['timestamp']), 'lc': lc, 'rc': rc, 'le': le, 're': re})
    return feats


def change_signal(feats, win=5, clip_h=1080):
    """多通道变化强度：手位置速度、手尺寸变化、两手间距变化。返回 [(t, score)]。"""
    ts = [f['t'] for f in feats]
    n = len(feats)

    def get(i, k):
        return feats[i].get(k)

    sig = []
    for i in range(n):
        j = min(n - 1, i + win)
        if j == i:
            sig.append((ts[i], 0.0))
            continue
        dt = max(1e-3, ts[j] - ts[i])
        acc = []
        for k in ('lc', 'rc'):
            a, b = get(i, k), get(j, k)
            if a and b:
                acc.append((((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5) / dt / clip_h * 100)
        for k in ('le', 're'):
            a, b = get(i, k), get(j, k)
            if a and b:
                acc.append(abs(a - b) / clip_h * 100)
        sig.append((ts[i], sum(acc) / len(acc) if acc else 0.0))
    return sig


def smooth(sig, k=5):
    vals = [v for _t, v in sig]
    out = []
    for i in range(len(vals)):
        lo, hi = max(0, i - k), min(len(vals), i + k + 1)
        out.append((sig[i][0], sum(vals[lo:hi]) / (hi - lo)))
    return out


def pick_peaks(sig, min_gap=1.0, k=1.6):
    """取局部极大值，阈值 = 中位数 + k×MAD，并保证最小间隔（间隔内保留更强的那个）。"""
    vals = sorted(v for _t, v in sig)
    med = vals[len(vals) // 2]
    mad = sorted(abs(v - med) for v in vals)[len(vals) // 2] or 1e-6
    thr = med + k * mad
    cand = []
    for i in range(1, len(sig) - 1):
        t, v = sig[i]
        if v < thr:
            continue
        if v >= sig[i - 1][1] and v > sig[i + 1][1]:
            cand.append((v, t))
    cand.sort(reverse=True)          # 最强的优先占位
    peaks = []
    for _v, t in cand:
        if all(abs(t - p) >= min_gap for p in peaks):
            peaks.append(t)
    return sorted(peaks), thr, med, mad


def eval_edges(gt_edges, pred, tol):
    used, errs = set(), []
    for g in gt_edges:
        best = None
        for i, p in enumerate(pred):
            if i in used:
                continue
            d = abs(p - g)
            if d <= tol and (best is None or d < best[1]):
                best = (i, d)
        if best:
            used.add(best[0])
            errs.append(best[1])
    errs.sort()
    return len(errs), len(pred) - len(errs), (errs[len(errs) // 2] if errs else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pose', type=Path, default=Path('web/public/data/local_pose_2min.json'))
    ap.add_argument('--gt', type=Path, default=Path('web/public/data/gt_segments.json'))
    ap.add_argument('--clip-offset', type=float, default=300.0)
    ap.add_argument('--tol', type=float, default=1.0)
    ap.add_argument('--min-gap', type=float, default=1.0)
    ap.add_argument('--k', type=float, default=1.6, help='阈值 = 中位数 + k×MAD')
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()

    pose = json.loads(a.pose.read_text(encoding='utf-8-sig'))
    frames = pose.get('frames') or []
    print(f'骨架帧 {len(frames)}；采样率 {pose.get("rate")} Hz；时长 {pose.get("duration")} s')

    feats = build_features(frames)
    sig = smooth(change_signal(feats))
    peaks, thr, med, mad = pick_peaks(sig, a.min_gap, a.k)
    print(f'变化强度阈值 {thr:.3f}（中位 {med:.3f} + {a.k}×MAD {mad:.3f}）→ 候选边界 {len(peaks)} 个')
    print('候选边界（片段时间）:', [round(p, 1) for p in peaks[:30]])

    g = json.loads(a.gt.read_text(encoding='utf-8-sig'))
    lo, hi = a.clip_offset, a.clip_offset + float(pose.get('duration') or 120)
    gt_edges = sorted({round(e, 3) for s in g['segments']
                       for e in (s['start'], s['end']) if lo <= e <= hi})
    pred = sorted(p + a.clip_offset for p in peaks)
    hit, extra, med_err = eval_edges(gt_edges, pred, a.tol)
    print(f'\nGT 边界 {len(gt_edges)} 条（v2 时间 {lo:.0f}–{hi:.0f}s）')
    print(f'命中 {hit} 条（召回 {hit / max(1, len(gt_edges)):.1%}）｜多余候选 {extra} 条'
          f'（精确 {hit / max(1, len(pred)):.1%}）｜命中误差中位 '
          f'{med_err:.2f}s' if med_err is not None else '')
    print('\nGT 边界逐条：')
    for ge in gt_edges:
        near = min(pred, key=lambda p: abs(p - ge)) if pred else None
        d = abs(near - ge) if near is not None else None
        flag = 'OK ' if d is not None and d <= a.tol else '   '
        print(f'  {flag} GT {ge:7.2f}  最近候选 {near if near is None else round(near, 2)}'
              f'  误差 {d if d is None else round(d, 2)}')

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({'peaks_clip': peaks, 'peaks_v2': pred,
                                     'threshold': thr, 'tol': a.tol,
                                     'gt_edges': gt_edges}, ensure_ascii=False, indent=1),
                         encoding='utf-8')
        print('写出', a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
