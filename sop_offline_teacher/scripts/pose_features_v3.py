"""骨架特征 v3：在 v2（手部）基础上加入身体/手臂特征，专治"手部检测失败时一无所有"。

诊断依据：
  · v2 版在片段 0–45 秒（v2 300–345s）全判成 idle
  · 该区间 **双手均检测失败的帧占 36.9%**（全片平均 13.7%）
  · RTMW 其实同时输出身体 133 点，但旧提取脚本只存了两只手 → 手失败时只有 bbox
  · 新提取（local_pose_full_v2.json）存了 body：肩/肘/腕/髋，86% 的帧可用

v3 新增（相对 v2 的 35 维 base）：
  每侧手臂 6 维 ×2：
    · 肩→腕 向量（肩为原点，按人体高归一）——手臂整体指向
    · 肘角（肩-肘-腕夹角）——伸直/弯曲，区分"伸手取件"vs"压合"
    · 腕速（用 body 腕点，手部失败时仍可用）
    · 手部失败标志（1=该侧手点不可用，用 body 腕兜底）
  全局 2 维：双肩宽度（尺度参考）、身体朝向（髋左右中点向量）

兜底逻辑（关键）：
  手部 21 点有效 → 用 v2 的手部特征
  手部失败但 body 腕有效 → 用 body 腕当作手腕位置（速度/方向仍可算），并置"兜底标志"
  两者都没有 → 全零 + 标志位
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 复用 v2 的常量与工具
from pose_features_v2 import (P_THR, TIP_IDS, WRIST_ID, _pts, _valid, _wrist,
                              _tip_centroid, _spread, load_station_centers)


def _bp(body, key):
    v = (body or {}).get(key)
    if not v or len(v) < 3 or v[2] <= P_THR:
        return None
    return v[0], v[1]


def build_features_v3(pose_path: Path, stations_path: Path | None = None,
                      frame_w: float = 1920.0, frame_h: float = 1080.0,
                      use_station: bool = True):
    """use_station=False 时不使用固定工位区特征（消融"换工位/重框 ROI"的依赖）。"""
    pose = json.loads(Path(pose_path).read_text(encoding='utf-8-sig'))
    stations = load_station_centers(stations_path) if (stations_path and use_station) else []
    ts, rows = [], []
    prev = {'lw': None, 'rw': None, 'blw': None, 'brw': None}

    for f in pose['frames']:
        t = float(f['timestamp'])
        bb = (f.get('person') or {}).get('bbox')
        if bb:
            x0, y0, x1, y1 = [float(v) for v in bb]
        else:
            x0, y0, x1, y1 = 0.0, 0.0, frame_w, frame_h
        bw_ = max(1.0, x1 - x0)
        bh = max(1.0, y1 - y0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2  # noqa: F841
        body = f.get('body') or {}

        def norm(x, y):
            return (x - cx) / bh, (y - cy) / bh

        lh, rh = f.get('left_hand'), f.get('right_hand')
        lw, rw = _wrist(lh), _wrist(rh)
        lt, rt = _tip_centroid(lh), _tip_centroid(rh)
        lsp, rsp = _spread(lh), _spread(rh)
        ln, rn = len(_valid(lh)), len(_valid(rh))

        # ---- v2 的手部特征（保持可比） ----
        def hand_feats(w, tip, sp, n, side):
            a = norm(*w) if w else (0.0, 0.0)
            if w and tip:
                b = ((tip[0] - w[0]) / bh, (tip[1] - w[1]) / bh)
                bd = (b[0] ** 2 + b[1] ** 2) ** 0.5
            else:
                b, bd = (0.0, 0.0), 0.0
            s = (sp / bh) if sp else 0.0
            pv = prev['lw' if side == 'l' else 'rw']
            if w and pv:
                vx, vy = (w[0] - pv[0]) / bh * 10.0, (w[1] - pv[1]) / bh * 10.0
            else:
                vx, vy = 0.0, 0.0
            prev['lw' if side == 'l' else 'rw'] = w
            ratio = n / 21.0
            vis_full, vis_part = (1.0, 0.0) if n >= 12 else ((0.0, 1.0) if n >= 4 else (0.0, 0.0))
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

        # ---- v3 新增：手臂/身体特征 ----
        sl, sr = _bp(body, 'shoulder_l'), _bp(body, 'shoulder_r')
        el, er = _bp(body, 'elbow_l'), _bp(body, 'elbow_r')
        wl, wr = _bp(body, 'wrist_l'), _bp(body, 'wrist_r')
        hl, hr = _bp(body, 'hip_l'), _bp(body, 'hip_r')

        def arm_feats(sh, elb, wr_b, hand_w, side):
            # 兜底：手部腕点优先，否则用 body 腕点
            w_use = hand_w or wr_b
            fb = 0.0 if hand_w else (1.0 if wr_b else 0.0)
            if sh and w_use:
                v = ((w_use[0] - sh[0]) / bh, (w_use[1] - sh[1]) / bh)
                vd = float(np.hypot(*v))
            else:
                v, vd = (0.0, 0.0), 0.0
            # 肘角（肩-肘-腕）；无肘点时给 0
            ang = 0.0
            if sh and elb and w_use:
                a = np.array([sh[0] - elb[0], sh[1] - elb[1]])
                b = np.array([w_use[0] - elb[0], w_use[1] - elb[1]])
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if na > 1e-6 and nb > 1e-6:
                    ang = float(np.clip(a @ b / (na * nb), -1, 1))
            # body 腕速度
            pv = prev['blw' if side == 'l' else 'brw']
            if w_use and pv:
                vx, vy = (w_use[0] - pv[0]) / bh * 10.0, (w_use[1] - pv[1]) / bh * 10.0
            else:
                vx, vy = 0.0, 0.0
            prev['blw' if side == 'l' else 'brw'] = w_use
            return [v[0], v[1], vd, ang, vx, vy, fb]

        la = arm_feats(sl, el, wl, lw, 'l')
        ra = arm_feats(sr, er, wr, rw, 'r')

        # 全局：肩宽（尺度）、身体朝向
        sw = ((sr[0] - sl[0]) / bh) if (sl and sr) else 0.0
        facing = 0.0
        if hl and hr:
            facing = float(np.clip((hr[0] - hl[0]) / bh, -2, 2))

        two = 0.0
        mid = [0.0, 0.0]
        lw_u = lw or wl
        rw_u = rw or wr
        if lw_u and rw_u:
            two = float(np.hypot(lw_u[0] - rw_u[0], lw_u[1] - rw_u[1]) / bh)
            m = norm((lw_u[0] + rw_u[0]) / 2, (lw_u[1] + rw_u[1]) / 2)
            mid = [m[0], m[1]]
        aspect = bh / bw_
        both = 1.0 if (ln >= 4 and rn >= 4) else 0.0

        rows.append(lf + rf + la + ra + [two, mid[0], mid[1], aspect, both, sw, facing])
        ts.append(t)

    base = np.array(rows, dtype=np.float32)
    # 缺失值前向填充（避免零值突变；仅对连续小段有效）
    for j in range(base.shape[1]):
        col = base[:, j]
        for i in range(1, len(col)):
            if col[i] == 0.0 and col[i - 1] != 0.0:
                col[i] = col[i - 1]
        base[:, j] = col

    def lag(k):
        out = np.zeros_like(base)
        if len(base) > k:
            out[k:] = base[k:] - base[:-k]
        return out

    X = np.concatenate([base, lag(3), lag(6)], axis=1)
    return np.array(ts), X, base.shape[1]


if __name__ == '__main__':
    ap_pose = sys.argv[1] if len(sys.argv) > 1 else 'web/public/data/local_pose_full_v2.json'
    ts, X, d = build_features_v3(Path(ap_pose), Path('configs/stations.json'))
    print(f'帧 {len(ts)} ｜ base 维 {d} ｜ 总维 {X.shape[1]}')
