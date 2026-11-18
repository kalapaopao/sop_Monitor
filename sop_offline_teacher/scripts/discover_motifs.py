"""无监督动作原型发现：完全不调用大模型、不需要新标注，验证"重复性"这条路走不走得通。

对应你的想法：一个人重复做同一套动作 → 手部运动模式会重复 → 把重复的模式聚成簇，
每簇就是一个"动作原型"，人工只给它起一次名字（K 次劳动，覆盖整个数据集）。

做法：
  1) 用 10 Hz 骨架构造每个时刻的运动特征（按人体高度归一，换人/远近不变）
  2) 用变化点检测得到的边界切段（本地、0 成本）
  3) 每段特征序列重采样到固定长度 → 标准化 → 拼成向量
  4) k-means 聚类（多次随机初始化取最优）
  5) 与人工 GT 做交叉表，算"纯度"（每簇里最多的那个 GT 类占比之和）

用法：
    python scripts/discover_motifs.py --k 6
    python scripts/discover_motifs.py --k 8 --gt web/public/data/gt_segments.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

B2S = {'Remove_eyepiece_assembly': 'S1', 'Remove_the_upper_cover_of_the_eyepiece': 'S2',
       'Assemble_the_mirror_base_cover': 'S3', 'Assemble_the_mirror_base_sealing_plug': 'S4',
       'Fasten_mirror_base': 'S5', 'Fasten_tripod_strap': 'S6'}
NAME = {'S1': '取镜座', 'S2': '取上盖', 'S3': '装上盖', 'S4': '装密封塞',
        'S5': '紧固', 'S6': '系扎带', 'idle': '空闲', 'unknown': '未知'}


def center(pts):
    if not pts:
        return None
    xs = [p[0] for p in pts if p[2] > 0.18]
    ys = [p[1] for p in pts if p[2] > 0.18]
    return (sum(xs) / len(xs), sum(ys) / len(ys)) if xs else None


def extent(pts):
    if not pts:
        return None
    xs = [p[0] for p in pts if p[2] > 0.18]
    ys = [p[1] for p in pts if p[2] > 0.18]
    if len(xs) < 3:
        return None
    return float(np.hypot(max(xs) - min(xs), max(ys) - min(ys)))


def tick_features(pose):
    """每个 10 Hz tick 的 6 维特征，全部按人体框高度归一。"""
    frames = pose['frames']
    ts, rows = [], []
    prev = {'lc': None, 'rc': None}
    for f in frames:
        t = float(f['timestamp'])
        lc, rc = center(f.get('left_hand')), center(f.get('right_hand'))
        le, re = extent(f.get('left_hand')), extent(f.get('right_hand'))
        bbox = f.get('person') or f.get('person_box') or f.get('bbox')
        h = None
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            h = abs(float(bbox[3]) - float(bbox[1])) or None
        h = h or 900.0
        def vel(cur, key):
            p = prev[key]
            prev[key] = cur
            if p is None or cur is None:
                return 0.0
            return float(np.hypot(cur[0] - p[0], cur[1] - p[1])) / h * 10.0   # 每秒、按高度归一
        vl = vel(lc, 'lc')
        vr = vel(rc, 'rc')
        d = float(np.hypot(lc[0] - rc[0], lc[1] - rc[1])) / h if (lc and rc) else 0.0
        rows.append([
            vl, vr,
            (le or 0.0) / h, (re or 0.0) / h,
            d,
            ((lc[1] + rc[1]) / 2 / h) if (lc and rc) else 0.0,   # 手的高度（画面纵向相对位置）
        ])
        ts.append(t)
    return np.array(ts), np.array(rows, dtype=np.float32)


def resample(seq, n=16):
    if len(seq) == 0:
        return np.zeros((n, seq.shape[1] if seq.ndim == 2 else 0), dtype=np.float32)
    if len(seq) == 1:
        return np.repeat(seq, n, axis=0)
    idx = np.linspace(0, len(seq) - 1, n)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(seq) - 1)
    w = (idx - lo)[:, None]
    return ((1 - w) * seq[lo] + w * seq[hi]).astype(np.float32)


def kmeans(X, k, iters=200, restarts=20, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        C = X[rng.choice(len(X), k, replace=False)]
        for _ in range(iters):
            d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
            lab = d.argmin(1)
            newC = np.array([X[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        inertia = float(((X - C[lab]) ** 2).sum())
        if best is None or inertia < best[0]:
            best = (inertia, C, lab)
    return best[1], best[2], best[0]


def gt_label_at(gt_segs, clip_t, offset):
    v2 = clip_t + offset
    for s in gt_segs:
        if s['start'] - 1e-6 <= v2 <= s['end'] + 1e-6:
            return B2S.get(s['behavior'], 'unknown')
    return 'idle'


def load_pose_defaults(pose):
    """给其他脚本复用：返回 (ts, feat)。"""
    return tick_features(pose)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pose', type=Path, default=Path('web/public/data/local_pose_2min.json'))
    ap.add_argument('--boundaries', type=Path, default=Path('api_runs/boundaries.json'))
    ap.add_argument('--gt', type=Path, default=Path('web/public/data/gt_segments.json'))
    ap.add_argument('--clip-offset', type=float, default=300.0)
    ap.add_argument('--k', type=int, default=6)
    ap.add_argument('--resample', type=int, default=16)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()

    pose = json.loads(a.pose.read_text(encoding='utf-8-sig'))
    ts, feat = tick_features(pose)
    print(f'骨架 {len(ts)} 帧 @ {pose.get("rate")} Hz，特征维度 {feat.shape[1]}')

    dur = float(pose.get('duration') or ts[-1])
    peaks = json.loads(a.boundaries.read_text(encoding='utf-8-sig')).get('peaks_clip') or []
    edges = [0.0] + [p for p in sorted(peaks) if 0.5 < p < dur - 0.5] + [dur]
    segs = [(x, y) for x, y in zip(edges, edges[1:]) if y - x >= 0.8]
    print(f'用本地检测边界切出 {len(segs)} 段（>=0.8 秒）')

    X = []
    for x, y in segs:
        m = (ts >= x) & (ts < y)
        seq = feat[m]
        v = resample(seq, a.resample).reshape(-1)
        X.append(v)
    X = np.array(X, dtype=np.float32)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)

    C, lab, inertia = kmeans(X, a.k)
    print(f'k-means k={a.k} 完成，inertia={inertia:.1f}')

    gt = json.loads(a.gt.read_text(encoding='utf-8-sig'))['segments']
    labels = [gt_label_at(gt, (x + y) / 2, a.clip_offset) for x, y in segs]

    print(f'\n{"簇":>3} {"段数":>4}  时间分布                         GT 构成')
    order = sorted(range(a.k), key=lambda j: -int((lab == j).sum()))
    for j in order:
        idx = [i for i in range(len(segs)) if lab[i] == j]
        if not idx:
            continue
        spans = ', '.join(f'{segs[i][0]:.0f}-{segs[i][1]:.0f}' for i in idx[:6])
        comp = {}
        for i in idx:
            comp[labels[i]] = comp.get(labels[i], 0) + 1
        comps = ' '.join(f'{NAME.get(k, k)}×{v}' for k, v in sorted(comp.items(), key=lambda x: -x[1]))
        print(f'{j:>3} {len(idx):>4}  {spans:<30} {comps}')

    # 纯度：每簇取最多的 GT 类，累加
    def purity(only_labeled=False):
        hit = tot = 0
        for j in range(a.k):
            idx = [i for i in range(len(segs))
                   if lab[i] == j and (not only_labeled or labels[i] != 'idle')]
            if not idx:
                continue
            comp = {}
            for i in idx:
                comp[labels[i]] = comp.get(labels[i], 0) + 1
            hit += max(comp.values())
            tot += len(idx)
        return hit, tot

    hit_all, tot_all = purity(False)
    hit_lab, tot_lab = purity(True)
    print(f'\n聚类纯度（含空闲段）：{hit_all}/{tot_all} = {hit_all / max(1, tot_all):.1%}')
    print(f'聚类纯度（只算 GT 有标注的段）：{hit_lab}/{tot_lab} = {hit_lab / max(1, tot_lab):.1%}'
          '  ← 这个才反映"动作本身能不能被运动特征分开"')
    print('参考：这也是"人工只给 K 个原型命名"能达到的上限，因为同一簇内的段会被赋予同一个名字')

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({
            'k': a.k, 'segments': [{'start': x, 'end': y, 'cluster': int(c), 'gt': g}
                                   for (x, y), c, g in zip(segs, lab, labels)],
            'purity_all': hit_all / max(1, tot_all), 'purity_labeled': hit_lab / max(1, tot_lab)}, ensure_ascii=False, indent=1), encoding='utf-8')
        print('写出', a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
