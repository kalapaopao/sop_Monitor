"""用工人的手实际去过哪里，反推「料位 / 装配位 / 成品位」应该画在哪。

动机
----
现有固定 ROI 是为「注意力引导」画的，不是为「物件状态观测」画的。
实测 `取料区域` 矩形平均 72.5% 被人框覆盖 —— 它盖在人身上。
直接在上面做状态观测，测到的是人的运动。

而"工人会去哪里取料、在哪里作业、把成品放到哪"，**在 pose 数据里就有**：
把左右手腕的落点在画面上累积成热力图，热区的形状直接就是工位布局：

    高停留、低往返  → 装配位（手按住/拧螺丝的地方）
    低停留、多次往返 → 取料位（伸手进去拿出来，停留很短）
    每周期出现一次   → 成品位

本脚本给出这张热力图 + 峰值坐标，用来**数据驱动地**定状态观测框，
并把现有 ROI 一起画出来做对照（能直接看出原框是否盖在热区上）。

用法
----
    .venv\\Scripts\\python.exe scripts\\probe_hand_zones.py \
        --pose web/public/data/local_pose_full_v2.json \
        --stations configs/stations.json \
        --out api_runs/hand_zones
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
SPARK = " .:-=+*#%@"
COLS, ROWS = 20, 12


def _pt(v):
    if not v or len(v) < 3 or v[2] <= P_THR:
        return None
    return float(v[0]), float(v[1])


def wrist_of(frame, side):
    hand = frame.get(f"{side}_hand")
    if hand and len(hand) > WRIST_ID:
        q = _pt(hand[WRIST_ID])
        if q:
            return q, "hand"
    q = _pt((frame.get("body") or {}).get(f"wrist_{side[0]}"))
    if q:
        return q, "body"
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", type=Path, required=True)
    ap.add_argument("--stations", type=Path, default=Path("configs/stations.json"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frame-w", type=float, default=1920.0)
    ap.add_argument("--frame-h", type=float, default=1080.0)
    ap.add_argument("--dwell-min", type=float, default=3.0,
                    help="连续停留超过多少秒算「作业位」（低移动速度）")
    ap.add_argument("--speed-thr", type=float, default=0.20,
                    help="停留判定的速度阈值（画面归一化单位/秒）；RTMW 手腕抖动约 0.1~0.2")
    a = ap.parse_args()

    pose = json.loads(a.pose.read_text(encoding="utf-8-sig"))
    frames = pose["frames"]

    heat = np.zeros((ROWS, COLS), dtype=np.float64)
    heat_l = np.zeros((ROWS, COLS), dtype=np.float64)
    heat_r = np.zeros((ROWS, COLS), dtype=np.float64)
    series = {"left": [], "right": []}   # 每只手单独一条序列（不能混着算速度）
    for f in frames:
        t = float(f["timestamp"])
        for side, arr in (("left", heat_l), ("right", heat_r)):
            p, _src = wrist_of(f, side)
            if not p:
                continue
            nx = min(max(p[0] / a.frame_w, 0.0), 0.9999)
            ny = min(max(p[1] / a.frame_h, 0.0), 0.9999)
            c, r = int(nx * COLS), int(ny * ROWS)
            arr[r, c] += 1
            heat[r, c] += 1
            series[side].append((t, nx, ny))
    total = heat.sum()
    if total <= 0:
        raise SystemExit("pose 里没有有效腕点")
    heat_n = heat / total

    def grid(title, mat):
        hi = mat.max() if mat.max() > 0 else 1.0
        print(f"\n=== {title} ===")
        print("      " + "".join(f"{c / COLS:.2f}"[2] for c in range(COLS)) + "   (列 = 宽度等分)")
        for r in range(ROWS):
            line = "".join(SPARK[int(round(min(1.0, mat[r, c] / hi) * (len(SPARK) - 1)))]
                           for c in range(COLS))
            print(f"  y{(r + 0.5) / ROWS:.2f} {line}")

    print(f"腕点总数 {int(total)} ｜ 左 {int(heat_l.sum())} ｜ 右 {int(heat_r.sum())}")
    grid("左右手合计的落点热力图", heat_n)
    grid("左手", heat_l / max(1.0, heat_l.sum()))
    grid("右手", heat_r / max(1.0, heat_r.sum()))

    # ---- 停留分析：每只手单独算速度，连续低速 ≥ dwell-min 秒算一次停留 ----
    dwell_segments = []
    for side, seq in series.items():
        seq.sort()
        cur = None
        for i in range(1, len(seq)):
            t, x, y = seq[i]
            t0, x0, y0 = seq[i - 1]
            dt = t - t0
            if dt <= 0 or dt > 0.5:          # 断档 → 收尾
                if cur:
                    dwell_segments.append(cur)
                    cur = None
                continue
            speed = float(np.hypot(x - x0, y - y0)) / dt
            if speed < a.speed_thr:
                if cur is None:
                    cur = {"t0": t0, "t1": t, "sx": x, "sy": y, "n": 1, "side": side}
                else:
                    cur["t1"] = t
                    cur["sx"] += x; cur["sy"] += y; cur["n"] += 1
            else:
                if cur:
                    dwell_segments.append(cur)
                    cur = None
        if cur:
            dwell_segments.append(cur)
    dwell = [d for d in dwell_segments if (d["t1"] - d["t0"]) >= a.dwell_min]
    print(f"\n=== 停留片段（单侧手腕速度<{a.speed_thr}/s 且持续 ≥ {a.dwell_min}s）：{len(dwell)} 段 ===")
    print(f"{'start':>7}{'end':>7}{'dur':>7}{'手':>5}{'cx':>7}{'cy':>7}   rect_norm(以停留中心为中心的 0.16 宽方框)")
    zones = []
    for d in sorted(dwell, key=lambda z: -(z["t1"] - z["t0"]))[:20]:
        cx = d["sx"] / d["n"]; cy = d["sy"] / d["n"]
        z = {"t0": round(d["t0"], 2), "t1": round(d["t1"], 2), "side": d["side"],
             "dur": round(d["t1"] - d["t0"], 2), "cx": round(cx, 3), "cy": round(cy, 3),
             "rect_norm": [round(max(0.0, cx - 0.08), 3), round(max(0.0, cy - 0.08), 3),
                           round(min(1.0, cx + 0.08), 3), round(min(1.0, cy + 0.08), 3)]}
        zones.append(z)
        print(f"{z['t0']:>7.1f}{z['t1']:>7.1f}{z['dur']:>7.1f}{z['side']:>5}"
              f"{cx:>7.3f}{cy:>7.3f}   {z['rect_norm']}")

    # ---- 与现有 ROI 对照 ----
    print("\n=== 现有 ROI 与热区的对照（该框内腕点占比 vs 该框面积占比）===")
    cfg = json.loads(a.stations.read_text(encoding="utf-8-sig"))
    st = (cfg.get("stations") or [{}])[0]
    report = []
    print(f"{'id':<12}{'名称':<10}{'type':<7}{'框内腕点占比':>14}{'框面积占比':>12}{'倍数':>8}")
    for r in st.get("regions") or []:
        x0, y0, x1, y1 = r["rect_norm"]
        c0, c1 = int(x0 * COLS), max(int(x0 * COLS) + 1, int(np.ceil(x1 * COLS)))
        r0, r1 = int(y0 * ROWS), max(int(y0 * ROWS) + 1, int(np.ceil(y1 * ROWS)))
        share = float(heat_n[r0:r1, c0:c1].sum())
        area = float((x1 - x0) * (y1 - y0))
        lift = share / area if area > 0 else 0.0
        report.append({"id": r["id"], "name": r.get("name"), "type": r.get("type"),
                       "wrist_share": round(share, 4), "area_share": round(area, 4),
                       "lift": round(lift, 3)})
        print(f"{r['id']:<12}{r.get('name', ''):<10}{r.get('type', ''):<7}"
              f"{share:>14.4f}{area:>12.4f}{lift:>8.2f}")
    print("\n读法：倍数 ≈1 表示框内腕点密度和画面平均一样（框没有选中任何手停留的地方）；")
    print("      倍数 >2 才是真的框住了作业区。")

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "hand_zones.json").write_text(json.dumps({
        "pose": str(a.pose), "frame": [a.frame_w, a.frame_h],
        "grid": [ROWS, COLS], "wrist_points": int(total),
        "heat": [[round(float(heat_n[r, c]), 6) for c in range(COLS)] for r in range(ROWS)],
        "dwell_zones": zones,
        "roi_vs_heat": report,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n写入 {a.out / 'hand_zones.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
