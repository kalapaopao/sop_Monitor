"""物件状态观测：把固定工位区变成可解释的「这个位子上有没有件」序列。

为什么要有这一步
----------------
现有链路（骨架 → MLP/TCN → HMM）学的是"这个人怎么动"，所以换个人就掉分。
但一个装配步骤真正不变的是**物件的状态转移**：件从取料位消失、在装配位出现、
成品出现在放料位。这些是物理事实，与工人手法无关。

本脚本只做一件事：把三个固定工位区在每个 tick 的外观，压缩成几个标量，
并且带上"工人的手是不是在这个区域里"——用来区分"区域变了是因为人挡着"
还是"因为件变了"。

指标（每个区域每个 tick）
------------------------
diff      与背景模型的平均绝对灰度差（越大越"不寻常"）
fg        与背景模型差值超过 --fg-thr 的像素占比（越大越"有东西/被动过"）
hand_in   是否有手腕落在这个矩形内（左右手 + body 腕兜底）
bright    区域平均灰度（排查灯光/曝光漂移用）

背景模型 = 该区域在整段视频上按 --bg-rate 抽样的逐像素中位数。
它代表"这个位子最平常的样子"，因此 diff/fg 表达的是"偏离平常有多远"。
**不假设**取料位初始有件、也不假设装配位初始为空——这一点由下游根据
diff 的实际数值和 GT 核对后再决定阈值（先测量、后定规则）。

用法
----
    .venv\\Scripts\\python.exe scripts/station_state_obs.py \
        --video vedio/v2.mp4 \
        --stations configs/stations.json \
        --pose web/public/data/local_pose_full_v2.json \
        --out api_runs/state_obs

    # 快速冒烟（只跑前 60 秒）
    ... --limit 60

    # 只观测部分区域
    ... --regions state_feed,state_asm
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P_THR = 0.18  # 关键点可信阈值（与 run_local_pose.py / pose_features_*.py 一致）
WRIST_ID = 0
SPARK = " .:-=+*#%@"


# ---------------------------------------------------------------- 基础工具


def region_box(rect_norm, w, h):
    x0, y0, x1, y1 = rect_norm
    bx0 = int(round(x0 * w)); by0 = int(round(y0 * h))
    bx1 = int(round(x1 * w)); by1 = int(round(y1 * h))
    bx0 = max(0, min(bx0, w - 2)); by0 = max(0, min(by0, h - 2))
    bx1 = max(bx0 + 2, min(bx1, w)); by1 = max(by0 + 2, min(by1, h))
    return [bx0, by0, bx1, by1]


def _pt(v):
    if not v or len(v) < 3 or v[2] <= P_THR:
        return None
    return float(v[0]), float(v[1])


def wrist_of(frame, side):
    """优先用手部 21 点的腕点；手部不可用时退到 body 腕点。返回 (x, y, src)。"""
    hand = frame.get(f"{side}_hand")
    if hand:
        p = hand[WRIST_ID] if len(hand) > WRIST_ID else None
        q = _pt(p)
        if q:
            return q[0], q[1], "hand"
    b = (frame.get("body") or {}).get(f"wrist_{side[0]}")
    q = _pt(b)
    if q:
        return q[0], q[1], "body"
    return None


def in_box(pt, box):
    x0, y0, x1, y1 = box
    return x0 <= pt[0] <= x1 and y0 <= pt[1] <= y1


def load_ticks(pose_path: Path, start: float, limit: float):
    pose = json.loads(Path(pose_path).read_text(encoding="utf-8-sig"))
    frames = pose["frames"]
    ticks = []
    for f in frames:
        t = float(f["timestamp"])
        if t < start - 1e-6:
            continue
        if limit > 0 and t > start + limit:
            break
        ticks.append({
            "t": t,
            "lw": wrist_of(f, "left"),
            "rw": wrist_of(f, "right"),
            "bbox": (f.get("person") or {}).get("bbox"),
        })
    return ticks, pose


def cover_ratio(box, bbox):
    """工人框覆盖了区域面积的比例（0~1）。用来判断这个区域还能不能看到物件。"""
    if not bbox:
        return 0.0
    x0, y0, x1, y1 = box
    bx0, by0, bx1, by1 = [float(v) for v in bbox]
    ix0, iy0 = max(x0, bx0), max(y0, by0)
    ix1, iy1 = min(x1, bx1), min(y1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return float((ix1 - ix0) * (iy1 - iy0)) / float((x1 - x0) * (y1 - y0))


def sparkline(values, width=300, vmax=None):
    """把一条序列压成 ASCII 强度条，便于在没有画图的环境里肉眼核对。"""
    n = len(values)
    if n == 0:
        return ""
    idx = np.linspace(0, n - 1, min(width, n)).round().astype(int)
    v = np.asarray([values[i] for i in idx], dtype=float)
    hi = vmax if vmax else (np.nanmax(v) if np.nanmax(v) > 0 else 1.0)
    out = []
    for x in v:
        if not np.isfinite(x):
            out.append("?")
            continue
        k = int(round(max(0.0, min(1.0, x / hi)) * (len(SPARK) - 1)))
        out.append(SPARK[k])
    return "".join(out)


def _resample_mask(mask, width):
    """把布尔序列重采样到 ASCII 条的字符数（与 sparkline 用同一组下标）。"""
    n = len(mask)
    if n == 0:
        return []
    idx = np.linspace(0, n - 1, min(width, n)).round().astype(int)
    return [bool(mask[i]) for i in idx]


def gt_cycles(gt_path: Path):
    d = json.loads(Path(gt_path).read_text(encoding="utf-8-sig"))
    return d.get("cycles") or [], d.get("segments") or []


# ---------------------------------------------------------------- 主流程


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--stations", type=Path, default=Path("configs/stations.json"))
    ap.add_argument("--pose", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--regions", default="", help="逗号分隔的区域 id；空=所有 type=state 的区域")
    ap.add_argument("--work-size", type=int, default=96, help="区域工作分辨率（只用来看状态，不需要清晰）")
    ap.add_argument("--bg-rate", type=float, default=1.0, help="背景模型抽样频率 Hz")
    ap.add_argument("--fg-thr", type=float, default=15.0, help="判为前景的灰度差阈值")
    ap.add_argument("--limit", type=float, default=0.0, help="处理时长（0=到结尾）")
    ap.add_argument("--start", type=float, default=0.0, help="起始秒（默认 0）")
    ap.add_argument("--gt", type=Path, default=Path("web/public/data/gt_segments.json"))
    ap.add_argument("--spark", type=int, default=300, help="诊断条形宽度（字符数），0=不打印")
    a = ap.parse_args()

    cfg = json.loads(a.stations.read_text(encoding="utf-8-sig"))
    station = (cfg.get("stations") or [None])[0]
    if not station:
        raise SystemExit("stations.json 里没有工位")

    want = {s.strip() for s in a.regions.split(",") if s.strip()}
    regions = [r for r in (station.get("regions") or [])
               if r.get("enabled", True) and (r["id"] in want if want else r.get("type") == "state")]
    if not regions:
        raise SystemExit("没有可观测的区域（检查 stations.json 的 type=state 或 --regions）")

    ticks, pose = load_ticks(a.pose, a.start, a.limit)
    if not ticks:
        raise SystemExit("pose 里没有 tick")

    cap = cv2.VideoCapture(str(a.video))
    if not cap.isOpened():
        raise SystemExit(f"打不开视频 {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    boxes = {r["id"]: region_box(r["rect_norm"], fw, fh) for r in regions}
    S = a.work_size

    n = len(ticks)
    crops = {r["id"]: np.zeros((n, S, S), dtype=np.uint8) for r in regions}
    ok_flag = np.zeros(n, dtype=bool)

    print(f"视频 {a.video} ｜ {fw}x{fh} @ {fps:.3f}fps ｜ tick {n} 个 ｜ 区域 {len(regions)} 个")
    print("解码中…（顺序 grab，单趟）")
    t0 = time.time()
    cursor = -1
    for i, tk in enumerate(ticks):
        want_f = int(round(tk["t"] * fps))
        while cursor < want_f:
            if not cap.grab():
                break
            cursor += 1
        ok, frame = cap.retrieve() if cursor >= want_f else (False, None)
        if not ok or frame is None:
            continue
        ok_flag[i] = True
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for r in regions:
            x0, y0, x1, y1 = boxes[r["id"]]
            patch = gray[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            crops[r["id"]][i] = cv2.resize(patch, (S, S), interpolation=cv2.INTER_AREA)
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{n}  {time.time() - t0:.0f}s")
    cap.release()
    print(f"解码完成 {time.time() - t0:.0f}s ｜ 成功 {int(ok_flag.sum())}/{n}")

    # ---- 背景模型：整段抽样中位数 ----
    step = max(1, int(round(10.0 / max(a.bg_rate, 0.05))))  # pose 是 10 Hz
    sample_idx = np.arange(0, n, step)
    sample_idx = sample_idx[ok_flag[sample_idx]]
    if len(sample_idx) < 5:
        raise SystemExit("成功帧太少，无法建立背景模型")
    bg = {r["id"]: np.median(crops[r["id"]][sample_idx].astype(np.float32), axis=0)
          for r in regions}
    print(f"背景模型：每区 {len(sample_idx)} 个抽样帧的中位数")

    # ---- 逐 tick 指标 ----
    obs = []
    series = {r["id"]: {"diff": np.zeros(n), "fg": np.zeros(n), "bright": np.zeros(n),
                        "hand_in": np.zeros(n, dtype=bool), "cover": np.zeros(n),
                        "fg_quiet": np.full(n, np.nan)} for r in regions}
    for i, tk in enumerate(ticks):
        if not ok_flag[i]:
            continue
        hands = [p for p in (tk["lw"], tk["rw"]) if p]
        entry = {"t": round(tk["t"], 2), "regions": {}}
        for r in regions:
            rid = r["id"]
            g = crops[rid][i].astype(np.float32)
            d = np.abs(g - bg[rid])
            diff = float(d.mean())
            fg = float((d > a.fg_thr).mean())
            bright = float(g.mean())
            cov = cover_ratio(boxes[rid], tk["bbox"])
            hit = any(in_box(h, boxes[rid]) for h in hands)
            series[rid]["diff"][i] = diff
            series[rid]["fg"][i] = fg
            series[rid]["bright"][i] = bright
            series[rid]["cover"][i] = cov
            series[rid]["hand_in"][i] = hit
            if not hit:
                # 「手不在这个区域里」的读数才算干净的物件状态观测
                series[rid]["fg_quiet"][i] = fg
            entry["regions"][rid] = {
                "diff": round(diff, 2), "fg": round(fg, 4),
                "hand_in": bool(hit), "cover": round(cov, 3), "bright": round(bright, 1),
            }
        obs.append(entry)

    a.out.mkdir(parents=True, exist_ok=True)
    doc = {
        "video": str(a.video), "fps": round(float(fps), 6), "frame": [fw, fh],
        "pose": str(a.pose), "work_size": S, "bg_rate": a.bg_rate, "fg_thr": a.fg_thr,
        "bg_sample_frames": int(len(sample_idx)),
        "regions": [{"id": r["id"], "name": r.get("name"), "type": r.get("type"),
                     "rect_px": boxes[r["id"]], "rect_norm": r["rect_norm"], "color": r.get("color")}
                    for r in regions],
        "ticks": obs,
    }
    out_json = a.out / "region_obs.json"
    out_json.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")

    # 后续脚本用 npz，读得快、体积小
    np.savez_compressed(
        a.out / "region_obs.npz",
        t=np.array([tk["t"] for tk in ticks], dtype=np.float32),
        ok=ok_flag,
        **{f"{rid}__{k}": v for rid in series for k, v in series[rid].items()},
    )

    # ---- 诊断 ----
    print(f"\n写入 {out_json}")
    print(f"写入 {a.out / 'region_obs.npz'}")
    print("\n=== 区域信号分布（整段）===")
    print(f"{'区域':<12}{'名称':<14}{'diff p50':>9}{'diff p95':>9}"
          f"{'fg p50':>8}{'fg p95':>8}{'安静fg p50':>11}{'手在内':>8}{'人框覆盖':>9}")
    for r in regions:
        rid = r["id"]
        ok = ok_flag
        d = series[rid]["diff"][ok]
        f = series[rid]["fg"][ok]
        q = series[rid]["fg_quiet"][ok]
        q = q[np.isfinite(q)]
        h = series[rid]["hand_in"][ok]
        cov = series[rid]["cover"][ok]
        q50 = f"{np.median(q):.3f}" if q.size else "  n/a"
        print(f"{rid:<12}{r.get('name', ''):<14}{np.median(d):>9.2f}"
              f"{np.percentile(d, 95):>9.2f}"
              f"{np.median(f):>8.3f}{np.percentile(f, 95):>8.3f}"
              f"{q50:>11}{h.mean():>8.3f}{cov.mean():>9.3f}")

    # ---- 与 GT 周期对齐（如果有）----
    cycles, segs = ([], [])
    if a.gt.exists():
        cycles, segs = gt_cycles(a.gt)
    if cycles and a.spark:
        tarr = np.array([tk["t"] for tk in ticks])
        print("\n=== 逐区域 ASCII（字符=强度；H=手在该区域内，此时读数不可用于判断物件）===")
        for r in regions:
            rid = r["id"]
            fg = series[rid]["fg"].copy()
            hand = series[rid]["hand_in"].copy()
            fg[~ok_flag] = np.nan
            hi = np.nanpercentile(fg, 99) if np.isfinite(fg).any() else 1.0
            hi = max(hi, 1e-6)
            chars = sparkline(fg, a.spark, vmax=hi)
            gated = "".join("H" if h else c for c, h in zip(chars, _resample_mask(hand, len(chars))))
            print(f"\n[{rid} {r.get('name', '')}]  fg（0=与平常一致，满格=偏离最大）")
            print("  " + gated)
            mid = a.spark // 2 * (tarr[-1] - tarr[0]) / a.spark + tarr[0]
            print(f"  时间: {tarr[0]:.0f}s{' ' * max(1, a.spark // 2 - 10)}{mid:.0f}s"
                  f"{' ' * max(1, a.spark // 2 - 10)}{tarr[-1]:.0f}s")
        if segs:
            print("\n=== GT 段位置（人工核对用，不是模型输出）===")
            for c in cycles:
                in_c = [s for s in segs if s.get("cycle_id") == c.get("cycle_id")]
                if not in_c:
                    continue
                desc = " ".join(f"{s['name_cn']}@{s['start']:.0f}-{s['end']:.0f}" for s in in_c)
                print(f"  c{c['cycle_id']} {c['start']:.0f}-{c['end']:.0f}s: {desc}")

    print("\n下一步：先人工核对上面的信号与 GT 时间位置，再定解码阈值。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
