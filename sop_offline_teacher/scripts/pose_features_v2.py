"""逐关键点骨架特征 v2（第 1 步：替代原来的 6 维整手统计量）。

设计原则：
  · **按人体框归一化**（用 person bbox 的宽高），换人/远近不变
  · **保留方向信息**（x/y 分开，不只有模长）
  · **保留手形信息**（指尖相对手腕、张开度）——区分"捏、扶、推"
  · **保留手到工位的距离**（读 configs/stations.json 的区域中心）
  · **显式可见性**（每只手 3 态：有效点够多 / 部分可见 / 未检出）

输出维度：base 34 维 + 两个时间差分（Δ3 帧、Δ6 帧）= 102 维
（不用"前后各 3 帧直接拼接"，那会把时间顺序丢掉，改用差分更有物理意义）
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

P_THR = 0.18                      # 关键点可信阈值（与 run_local_pose 一致）
TIP_IDS = [4, 8, 12, 16, 20]      # 五指尖（21 点手部 layout）
WRIST_ID = 0


def _pts(hand):
    return [(p[0], p[1], p[2]) for p in (hand or []) if len(p) >= 3]


def _valid(hand):
    return [(x, y) for x, y, s in _pts(hand) if s > P_THR]


def _wrist(hand):
    p = _pts(hand)
    if p and p[WRIST_ID][2] > P_THR:
        return p[WRIST_ID][0], p[WRIST_ID][1]
    return None


def _tip_centroid(hand):
    p = _pts(hand)
    tips = [p[i] for i in TIP_IDS if i < len(p) and p[i][2] > P_THR]
    if not tips:
        return None
    return sum(t[0] for t in tips) / len(tips), sum(t[1] for t in tips) / len(tips)


def _spread(hand):
    v = _valid(hand)
    if len(v) < 3:
        return None
    xs = [q[0] for q in v]
    ys = [q[1] for q in v]
    return (max(xs) - min(xs)) + (max(ys) - min(ys))


def load_station_centers(path: Path):
    """读固定工位区域，返回 [(名称, cx_norm, cy_norm)]（归一化到画面）。"""
    try:
        d = json.loads(path.read_text(encoding='utf-8-sig'))
        st = (d.get('stations') or [{}])[0]
        out = []
        for r in st.get('regions') or []:
            x0, y0, x1, y1 = r['rect_norm']
            out.append((r.get('name', r.get('id', '?')), (x0 + x1) / 2, (y0 + y1) / 2))
        return out
    except Exception:
        return []


def build_features(pose_path: Path, stations_path: Path | None = None,
                   frame_w: float = 1920.0, frame_h: float = 1080.0):
    pose = json.loads(Path(pose_path).read_text(encoding='utf-8-sig'))
    stations = load_station_centers(stations_path) if stations_path else []
    ts, rows = [], []
    prev = {'lw': None, 'rw': None}

    for f in pose['frames']:
        t = float(f['timestamp'])
        bb = (f.get('person') or {}).get('bbox')
        if bb:
            x0, y0, x1, y1 = [float(v) for v in bb]
        else:
            x0, y0, x1, y1 = 0.0, 0.0, frame_w, frame_h
        bw = max(1.0, x1 - x0)
        bh = max(1.0, y1 - y0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

        def norm(x, y):
            return (x - cx) / bh, (y - cy) / bh      # 用高度归一，保持长宽比

        lh, rh = f.get('left_hand'), f.get('right_hand')
        lw, rw = _wrist(lh), _wrist(rh)
        lt, rt = _tip_centroid(lh), _tip_centroid(rh)
        lsp, rsp = _spread(lh), _spread(rh)
        ln, rn = len(_valid(lh)), len(_valid(rh))

        def hand_feats(w, tip, sp, n, side):
            # 1) 手腕位置（相对人体中心，按高度归一）
            a = norm(*w) if w else (0.0, 0.0)
            # 2) 指尖相对手腕（手指指向）
            if w and tip:
                b = ((tip[0] - w[0]) / bh, (tip[1] - w[1]) / bh)
                bd = (b[0] ** 2 + b[1] ** 2) ** 0.5
            else:
                b, bd = (0.0, 0.0), 0.0
            # 3) 张开度
            s = (sp / bh) if sp else 0.0
            # 4) 手腕速度（每秒、按高度归一）
            pv = prev['lw' if side == 'l' else 'rw']
            if w and pv:
                vx, vy = (w[0] - pv[0]) / bh * 10.0, (w[1] - pv[1]) / bh * 10.0
            else:
                vx, vy = 0.0, 0.0
            prev['lw' if side == 'l' else 'rw'] = w
            # 5) 可见性（三态编码 + 有效点数比例）
            ratio = n / 21.0
            vis_full, vis_part = (1.0, 0.0) if n >= 12 else ((0.0, 1.0) if n >= 4 else (0.0, 0.0))
            # 6) 到各工位区中心的距离（归一化）
            sd = []
            if w:
                wx, wy = w[0] / frame_w, w[1] / frame_h
                for _nm, sx, sy in stations[:3]:
                    sd.append(float(np.hypot(wx - sx, wy - sy)))
                while len(sd) < 3:
                    sd.append(0.0)
            else:
                sd = [0.0, 0.0, 0.0]
            return [a[0], a[1], b[0], b[1], bd, s, vx, vy,
                    float(np.hypot(vx, vy)), ratio, vis_full, vis_part, *sd]

        lf = hand_feats(lw, lt, lsp, ln, 'l')
        rf = hand_feats(rw, rt, rsp, rn, 'r')
        # 全局
        two = 0.0
        mid = [0.0, 0.0]
        if lw and rw:
            two = float(np.hypot(lw[0] - rw[0], lw[1] - rw[1]) / bh)
            m = norm((lw[0] + rw[0]) / 2, (lw[1] + rw[1]) / 2)
            mid = [m[0], m[1]]
        aspect = bh / bw
        both = 1.0 if (ln >= 4 and rn >= 4) else 0.0
        rows.append(lf + rf + [two, mid[0], mid[1], aspect, both])
        ts.append(t)

    base = np.array(rows, dtype=np.float32)
    # 时间差分（因果：只用过去），补方向与速度变化
    def lag(k):
        out = np.zeros_like(base)
        if len(base) > k:
            out[k:] = base[k:] - base[:-k]
        return out
    X = np.concatenate([base, lag(3), lag(6)], axis=1)
    return np.array(ts), X, base.shape[1]


if __name__ == '__main__':
    ap_pose = sys.argv[1] if len(sys.argv) > 1 else 'web/public/data/local_pose_full.json'
    ts, X, d_base = build_features(Path(ap_pose), Path('configs/stations.json'))
    print(f'帧 {len(ts)} ｜ base 维 {d_base} ｜ 总维 {X.shape[1]}')
    print('前 3 帧特征（截断）:\n', np.round(X[:3, :8], 3))
