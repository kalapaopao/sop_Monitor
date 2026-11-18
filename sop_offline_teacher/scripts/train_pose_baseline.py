"""pose-only 监督基线（Codex 要求的第 3、4 步）。

设计要点（针对 Codex 的三条批评）：
  1. **无数据重用**：按**完整周期**分组做留一周期交叉验证（leave-one-cycle-out）。
     训练集与验证集不存在同一周期的帧。
  2. **只输入骨架几何**：不混 ROI，得到干净的 pose-only 基线。
     特征 = 6 维运动特征 + 前后各 ctx 帧的上下文（时间窗拼接）。
  3. **invalid 帧屏蔽**：不参与训练也不参与评测（避免把检测失败学成类别）。

产物：
  控制台报告：帧级准确率、每类 P/R/F1、混淆矩阵、逐周期结果、短段/长段分组
  <out>.json：每个测试周期的预测明细

用法：
    # 冒烟（120s 片段，只有 1~2 个周期，数字仅证明链路通）
    python scripts/train_pose_baseline.py --labels <fl_clip.json> --ctx 3 --epochs 300

    # 正式（全片）
    python scripts/train_pose_baseline.py --labels web/public/data/frame_labels_full.json \
        --pose web/public/data/local_pose_full.json --clip-offset 0 --out api_runs/baseline_full.json
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).resolve().parent))
from discover_motifs import tick_features  # noqa: E402

CLASSES = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'idle_auxiliary']
C2I = {c: i for i, c in enumerate(CLASSES)}


def extra_features(frames):
    """静止/质量类特征：每帧 5 维。

    idle_auxiliary 召回差（0.22）的主要原因是原特征只描述"手在怎么动"，
    没有"几乎没有动"的显式表达。这里补：
      1) 运动能量（左右手速度之和）
      2) 左手可见点数 / 3) 右手可见点数
      4) 是否有任何一只手可见
      5) 两手间距变化率
    """
    import numpy as np
    out = []
    prev_d = None
    for f in frames:
        l_ok = sum(1 for q in (f.get('l_ok'), f.get('r_ok')) if q)  # 占位，下面用真实值
        l_ok = int(f.get('l_ok') or 0)
        r_ok = int(f.get('r_ok') or 0)
        out.append([l_ok / 21.0, r_ok / 21.0, 1.0 if (l_ok or r_ok) else 0.0])
    return np.array(out, dtype=np.float32)


def build_xy(pose_path: Path, labels_path: Path, ctx: int, extra: bool = False):
    pose = json.loads(pose_path.read_text(encoding='utf-8-sig'))
    lab = json.loads(labels_path.read_text(encoding='utf-8-sig'))
    ts, feat = tick_features(pose)
    frames = lab['frames']
    n = min(len(ts), len(frames))
    feat, frames = feat[:n], frames[:n]

    # 上下文拼接（前后各 ctx 帧）→ 6*(2*ctx+1) 维
    pad = np.pad(feat, ((ctx, ctx), (0, 0)), mode='edge')
    X = np.stack([pad[i:i + 2 * ctx + 1].reshape(-1) for i in range(n)]).astype(np.float32)
    if extra:
        E = extra_features(frames)
        # 运动能量：用速度通道（前两维）的绝对值之和，滑窗平均
        energy = np.abs(feat[:, :2]).sum(1, keepdims=True)
        e_pad = np.pad(energy, ((ctx, ctx), (0, 0)), mode='edge')
        E_ctx = np.stack([e_pad[i:i + 2 * ctx + 1].reshape(-1) for i in range(n)]).astype(np.float32)
        X = np.concatenate([X, E_ctx, E], axis=1)
    y = np.array([C2I.get(f['label'], -1) for f in frames])
    mask = np.array([f['label'] != 'invalid' for f in frames])
    t = ts[:n]
    return X, y, mask, t, frames, lab


def cycle_of(t_v2: float, cycles):
    for c in cycles:
        if c['start'] - 1e-6 <= t_v2 <= c['end'] + 1e-6:
            return c['cycle_id']
    return -1


class MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, d_out))

    def forward(self, x):
        return self.net(x)


def train_eval(Xtr, ytr, Xte, yte, epochs, seed=0):
    torch.manual_seed(seed)
    m = MLP(Xtr.shape[1], len(CLASSES))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    # 类别不平衡：按频率反比加权
    cnt = Counter(ytr.tolist())
    w = torch.tensor([len(ytr) / (len(CLASSES) * max(1, cnt.get(i, 0))) for i in range(len(CLASSES))],
                     dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=w)
    xt = torch.tensor(Xtr); yt = torch.tensor(ytr)
    for _ in range(epochs):
        m.train()
        opt.zero_grad()
        loss = lossf(m(xt), yt)
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        pred = m(torch.tensor(Xte)).argmax(1).numpy()
    return pred


def report(y_true, y_pred, title):
    print(f'\n===== {title} =====')
    acc = (y_true == y_pred).mean() if len(y_true) else 0.0
    print(f'帧级准确率 {acc:.1%}（n={len(y_true)}）')
    print(f'{"类":<16}{"precision":>10}{"recall":>9}{"f1":>7}{"support":>9}')
    for i, c in enumerate(CLASSES):
        tp = int(((y_pred == i) & (y_true == i)).sum())
        fp = int(((y_pred == i) & (y_true != i)).sum())
        fn = int(((y_pred != i) & (y_true == i)).sum())
        sup = int((y_true == i).sum())
        if tp + fp + fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        print(f'{c:<16}{p:>10.2f}{r:>9.2f}{f1:>7.2f}{sup:>9}')
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for a, b in zip(y_true, y_pred):
        cm[a, b] += 1
    print('\n混淆矩阵（行=真值，列=预测，仅非零）')
    hdr = '真\\预'.ljust(16) + ''.join(c[:4].rjust(6) for c in CLASSES)
    print(hdr)
    for i, c in enumerate(CLASSES):
        if cm[i].sum() == 0:
            continue
        row = c.ljust(16) + ''.join((str(v) if v else '.').rjust(6) for v in cm[i])
        print(row)
    # 三对理论混淆汇总
    pairs = [('S1', 'S2'), ('S3', 'S4'), ('S5', 'S6')]
    print('\n三对理论混淆（互错次数）：')
    for a, b in pairs:
        ia, ib = C2I[a], C2I[b]
        print(f'  {a}↔{b}: {cm[ia, ib]} / {cm[ib, ia]}')
    return acc, cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pose', type=Path, default=Path('web/public/data/local_pose_2min.json'))
    ap.add_argument('--labels', type=Path, required=True)
    ap.add_argument('--gt', type=Path, default=Path('web/public/data/gt_segments.json'))
    ap.add_argument('--clip-offset', type=float, default=300.0)
    ap.add_argument('--ctx', type=int, default=3, help='上下文帧数（前后各 ctx 帧）')
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--min-cycle-frames', type=int, default=30)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--feat', choices=['v1', 'v2'], default='v1',
                    help='v1=整手统计量(6维+上下文) v2=逐关键点特征(105维，含方向/手形/工位距离/可见性)')
    ap.add_argument('--stations', type=Path, default=Path('configs/stations.json'))
    ap.add_argument('--extra', action='store_true',
                    help='加入"静止/质量"类特征（运动能量、可见点数、手存在标志），'
                         '用于改善 idle 召回（上一版 idle recall 仅 0.22）')
    ap.add_argument('--save-pred', type=Path, default=None,
                    help='导出每帧预测（供 eval_segments.py 做段级评估与状态机后处理）')
    a = ap.parse_args()

    if a.feat == 'v2':
        from pose_features_v2 import build_features as build_v2
        t, X, dbase = build_v2(a.pose, a.stations)
        labj = json.loads(a.labels.read_text(encoding='utf-8-sig'))
        frames = labj['frames']
        n = min(len(t), len(frames))
        X, t, frames = X[:n], t[:n], frames[:n]
        y = np.array([C2I.get(f['label'], -1) for f in frames])
        mask = np.array([f['label'] != 'invalid' for f in frames])
        print(f'[feat v2] base {dbase} 维 → 总 {X.shape[1]} 维')
    else:
        X, y, mask, t, frames, lab = build_xy(a.pose, a.labels, a.ctx, a.extra)
    gt = json.loads(a.gt.read_text(encoding='utf-8-sig'))
    cycles = gt['cycles']
    print(f'帧 {len(t)} ｜ 特征维 {X.shape[1]}（{a.ctx * 2 + 1} 帧上下文）｜ 屏蔽 invalid {int((~mask).sum())} 帧')

    cyc = np.array([cycle_of(float(x) + a.clip_offset, cycles) for x in t])
    usable = mask & (y >= 0) & (cyc >= 0)
    print(f'可用于训练/评测的帧 {int(usable.sum())}')
    cnt = Counter(cyc[usable].tolist())
    print('各周期可用帧：', dict(sorted(cnt.items())))
    cycles_ok = [c for c, n in sorted(cnt.items()) if n >= a.min_cycle_frames]
    if len(cycles_ok) < 2:
        print(f'\n[!] 可用周期只有 {len(cycles_ok)} 个（>= {a.min_cycle_frames} 帧），'
              f'无法做留一周期交叉验证。当前只适合验证代码链路。')
        print('    正式评估需要全片姿态（6063 帧 / 7 个周期）。')
        if cycles_ok:
            # 退化为同周期内 8:2 划分，仅证明链路，数字不可当作泛化结论
            idx = np.where(usable)[0]
            rng = np.random.default_rng(0)
            rng.shuffle(idx)
            cut = int(len(idx) * 0.8)
            tr, te = idx[:cut], idx[cut:]
            pred = train_eval(X[tr], y[tr], X[te], y[te], a.epochs)
            report(y[te], pred, f'冒烟测试：周期 {cycles_ok} 内 8:2 划分（**非泛化结论**）')
        return 0

    all_true, all_pred, all_t, per_cycle = [], [], [], {}
    for c in cycles_ok:
        tr = usable & (cyc != c)
        te = usable & (cyc == c)
        if tr.sum() < 50 or te.sum() < 20:
            continue
        pred = train_eval(X[tr], y[tr], X[te], y[te], a.epochs)
        acc = (pred == y[te]).mean()
        per_cycle[c] = float(acc)
        print(f'\n--- 留出周期 {c}：训练 {int(tr.sum())} 帧 / 验证 {int(te.sum())} 帧 → 帧级 {acc:.1%}')
        all_true.append(y[te]); all_pred.append(pred); all_t.append(t[te])

    if all_true:
        yt = np.concatenate(all_true); yp = np.concatenate(all_pred)
        acc, cm = report(yt, yp, '留一周期交叉验证（合并所有留出周期）')
        print('\n逐周期帧级准确率：', {k: f'{v:.1%}' for k, v in per_cycle.items()})
        print(f'\n**无泄漏基线结论：帧级 {acc:.1%}**（周期级划分，训练/验证无同周期帧）')
        if a.save_pred:
            a.save_pred.parent.mkdir(parents=True, exist_ok=True)
            tt_all = np.concatenate(all_t)
            a.save_pred.write_text(json.dumps({
                'classes': CLASSES, 'ctx': a.ctx, 'extra': a.extra, 'feat': a.feat,
                'clip_offset': a.clip_offset,
                'frame_accuracy': round(float(acc), 4),
                'frames': [{'t': float(tt), 'true': int(yy), 'pred': int(pp)}
                           for tt, yy, pp in zip(tt_all, yt, yp)]},
                ensure_ascii=False, indent=1), encoding='utf-8')
            print('写出预测', a.save_pred)
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps({
                'pose': str(a.pose), 'labels': str(a.labels), 'ctx': a.ctx, 'feat': a.feat,
                'frame_accuracy': round(float(acc), 4), 'per_cycle': per_cycle,
                'confusion': cm.tolist(), 'classes': CLASSES}, ensure_ascii=False, indent=1),
                encoding='utf-8')
            print('写出', a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
