"""从骨架产出「行为事件」：停留、双手稳定、往返取料、离开作业位。

为什么用这些事件，而不是继续做动作分类
--------------------------------------
一个装配步骤跨工人真正不变的不是手腕轨迹，而是**这件事在物理上发生了**：
  · 手伸进取料位 → 拿了件（往返一次）
  · 双手在装配位上稳定保持一段时间 → 正在对位/压合/施力
  · 双手长时间保持同一构型（≥ 8 秒）→ 正在紧固（工具在受力）
工人怎么甩手腕、用哪只手、快还是慢，都不改变上面三件事。

所以本脚本只输出事件，不做分类：
    still      单侧手腕在 0.5 秒窗口内位移很小（抑制 RTMW 抖动）
    dwell      连续 still ≥ --dwell-min 秒
    two_stable 左右手同时 dwell，且两手相对构型也稳定
    reach      手腕进入某个状态框并停留 ≥ --reach-min 秒（取料/放料候选）
    away       双手都不在作业框内，持续 ≥ --away-min 秒（中断/走动/备料）

每条事件都带数值证据（时长、相对构型、所在区域），可人工核对。

用法
----
    .venv\\Scripts\\python.exe scripts/build_behavior_events.py \
        --pose web/public/data/local_pose_full_v2.json \
        --stations configs/stations.json \
        --out api_runs/behavior_events
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P_THR = 0.18
WRIST_ID = 0


def _pt(v):
    if not v or len(v) < 3 or v[2] <= P_THR:
        return None
    return float(v[0]), float(v[1])


def wrist_of(frame, side, fw, fh):
    hand = frame.get(f"{side}_hand")
    if hand and len(hand) > WRIST_ID:
        q = _pt(hand[WRIST_ID])
        if q:
            return q[0] / fw, q[1] / fh, "hand"
    q = _pt((frame.get("body") or {}).get(f"wrist_{side[0]}"))
    if q:
        return q[0] / fw, q[1] / fh, "body"
    return None


def in_rect(x, y, rect):
    x0, y0, x1, y1 = rect
    return x0 <= x <= x1 and y0 <= y <= y1


def runs(mask):
    """布尔序列 → 连续 True 区间 [i0, i1)。"""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


class Track:
    """单手的时间序列 + 抗抖速度。"""

    def __init__(self, side, samples, lag=0.5):
        self.side = side
        self.t = np.array([s[0] for s in samples], dtype=np.float64)
        self.x = np.array([s[1] for s in samples], dtype=np.float64)
        self.y = np.array([s[2] for s in samples], dtype=np.float64)
        self.index = {round(float(t), 6): i for i, t in enumerate(self.t)}
        n = len(self.t)
        # 速度：用 lag 秒的位移（比逐帧差分抗抖得多）
        self.speed = np.full(n, np.nan)
        j = 0
        for i in range(n):
            while j < i and self.t[i] - self.t[j] > lag + 0.3:
                j += 1
            if j < i and 0.2 <= self.t[i] - self.t[j] <= lag + 0.3:
                dt = self.t[i] - self.t[j]
                self.speed[i] = float(np.hypot(self.x[i] - self.x[j], self.y[i] - self.y[j])) / dt

    def pos_at(self, t):
        i = self.index.get(round(float(t), 6))
        return None if i is None else (float(self.x[i]), float(self.y[i]))

    def slice_positions(self, t0, t1):
        m = (self.t >= t0) & (self.t <= t1)
        return self.x[m], self.y[m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", type=Path, required=True)
    ap.add_argument("--stations", type=Path, default=Path("configs/stations.json"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frame-w", type=float, default=1920.0)
    ap.add_argument("--frame-h", type=float, default=1080.0)
    ap.add_argument("--still-thr", type=float, default=0.08,
                    help="0.5s 窗口内的位移速度阈值（画面归一化单位/秒）")
    ap.add_argument("--dwell-min", type=float, default=2.0, help="停留最短时长（秒）")
    ap.add_argument("--reach-min", type=float, default=0.4, help="进入状态框的最短停留（秒）")
    ap.add_argument("--away-min", type=float, default=1.5, help="离开作业框的最短时长（秒）")
    a = ap.parse_args()

    pose = json.loads(a.pose.read_text(encoding="utf-8-sig"))
    frames = pose["frames"]
    cfg = json.loads(a.stations.read_text(encoding="utf-8-sig"))
    st = (cfg.get("stations") or [{}])[0]
    boxes = {r["id"]: r["rect_norm"] for r in st.get("regions") or [] if r.get("enabled", True)}
    for k in ("state_asm", "state_feed", "state_out"):
        if k not in boxes:
            raise SystemExit(f"stations.json 缺少 {k} 区域")

    tracks = {}
    all_times = []
    for side in ("left", "right"):
        samples = []
        for f in frames:
            t = float(f["timestamp"])
            q = wrist_of(f, side, a.frame_w, a.frame_h)
            if q is not None:
                samples.append((t, q[0], q[1]))
        tracks[side] = Track(side, samples)
    all_times = sorted({float(f["timestamp"]) for f in frames})

    events = []

    # ---- dwell：单侧连续低速 ----
    dwells = []
    for side, tr in tracks.items():
        still = (tr.speed < a.still_thr)
        for i0, i1 in runs(list(still)):
            t0, t1 = float(tr.t[i0]), float(tr.t[i1 - 1])
            if t1 - t0 < a.dwell_min:
                continue
            cx = float(np.median(tr.x[i0:i1])); cy = float(np.median(tr.y[i0:i1]))
            d = {"type": "dwell", "side": side, "t0": round(t0, 2), "t1": round(t1, 2),
                 "dur": round(t1 - t0, 2), "cx": round(cx, 3), "cy": round(cy, 3),
                 "jitter": round(float(np.std(tr.x[i0:i1]) + np.std(tr.y[i0:i1])), 4),
                 "regions": [k for k, r in boxes.items() if in_rect(cx, cy, r)]}
            dwells.append(d)
            events.append(d)

    # ---- two_stable：左右手同时 dwell，且相对构型稳定 ----
    two = []
    for dl in [d for d in dwells if d["side"] == "left"]:
        for dr in [d for d in dwells if d["side"] == "right"]:
            lo, hi = max(dl["t0"], dr["t0"]), min(dl["t1"], dr["t1"])
            if hi - lo < max(2.0, a.dwell_min):
                continue
            tl = tracks["left"]; trk = tracks["right"]
            common = [t for t in all_times if lo <= t <= hi
                      and tl.pos_at(t) is not None and trk.pos_at(t) is not None]
            if len(common) < 5:
                continue
            vec = np.array([(trk.pos_at(t)[0] - tl.pos_at(t)[0],
                             trk.pos_at(t)[1] - tl.pos_at(t)[1]) for t in common])
            rel_std = float(vec.std(axis=0).mean())
            ev = {"type": "two_stable", "t0": round(lo, 2), "t1": round(hi, 2),
                  "dur": round(hi - lo, 2),
                  "left": [dl["cx"], dl["cy"]], "right": [dr["cx"], dr["cy"]],
                  "rel_dx": round(float(vec[:, 0].mean()), 3),
                  "rel_dy": round(float(vec[:, 1].mean()), 3),
                  "rel_std": round(rel_std, 4),
                  "in_asm": in_rect(dl["cx"], dl["cy"], boxes["state_asm"])
                            and in_rect(dr["cx"], dr["cy"], boxes["state_asm"])}
            two.append(ev)
            events.append(ev)

    # ---- reach：手腕进入状态框并停留 ----
    reaches = []
    for side, tr in tracks.items():
        for rid in ("state_feed", "state_out"):
            r = boxes[rid]
            ins = [in_rect(float(x), float(y), r) for x, y in zip(tr.x, tr.y)]
            for i0, i1 in runs(ins):
                if i1 - i0 < 2:
                    continue
                t0, t1 = float(tr.t[i0]), float(tr.t[i1 - 1])
                if t1 - t0 < a.reach_min:
                    continue
                ev = {"type": "reach", "zone": rid, "side": side,
                      "t0": round(t0, 2), "t1": round(t1, 2), "dur": round(t1 - t0, 2)}
                reaches.append(ev)
                events.append(ev)

    # ---- away：双手都不在作业框内 ----
    tl, trk = tracks["left"], tracks["right"]
    flags = []
    for t in all_times:
        hit = False
        for tr in (tl, trk):
            p = tr.pos_at(t)
            if p and in_rect(p[0], p[1], boxes["state_asm"]):
                hit = True
                break
        flags.append(hit)
    aways = []
    for i0, i1 in runs([not v for v in flags]):
        if i1 - i0 < 2:
            continue
        t0, t1 = all_times[i0], all_times[i1 - 1]
        if t1 - t0 < a.away_min:
            continue
        ev = {"type": "away", "t0": round(t0, 2), "t1": round(t1, 2), "dur": round(t1 - t0, 2)}
        aways.append(ev)
        events.append(ev)

    events.sort(key=lambda e: (e["t0"], e["type"], e.get("side") or ""))
    a.out.mkdir(parents=True, exist_ok=True)
    doc = {
        "pose": str(a.pose), "frame": [a.frame_w, a.frame_h],
        "params": {"still_thr": a.still_thr, "dwell_min": a.dwell_min,
                   "reach_min": a.reach_min, "away_min": a.away_min},
        "regions": boxes,
        "counts": {"dwell": len(dwells), "two_stable": len(two),
                   "reach": len(reaches), "away": len(aways)},
        "events": events,
    }
    (a.out / "events.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"腕点：左 {len(tl.t)} 右 {len(trk.t)}")
    print(f"事件：dwell {len(dwells)} ｜ two_stable {len(two)} ｜ reach {len(reaches)} ｜ away {len(aways)}")
    print("\n=== two_stable（双手稳定构型；紧固候选）===")
    print(f"{'t0':>7}{'t1':>7}{'dur':>7}{'rel_dx':>8}{'rel_dy':>8}{'rel_std':>9}{'in_asm':>7}")
    for e in two:
        print(f"{e['t0']:>7.1f}{e['t1']:>7.1f}{e['dur']:>7.1f}{e['rel_dx']:>8.3f}"
              f"{e['rel_dy']:>8.3f}{e['rel_std']:>9.3f}{str(e['in_asm']):>7}")
    print("\n=== reach（进入取料/放料位，前 40 条）===")
    print(f"{'t0':>7}{'t1':>7}{'dur':>6}{'zone':>12}{'side':>7}")
    for e in reaches[:40]:
        print(f"{e['t0']:>7.1f}{e['t1']:>7.1f}{e['dur']:>6.1f}{e['zone']:>12}{e['side']:>7}")
    if len(reaches) > 40:
        print(f"  …另有 {len(reaches) - 40} 条")
    print(f"\n写入 {a.out / 'events.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
