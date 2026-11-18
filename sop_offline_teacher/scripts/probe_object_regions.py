"""探测「物件在画面的哪些位置真正地出现/消失」，用于数据驱动地定状态观测框。

为什么需要这一步
----------------
现有 3 个固定 ROI 是为「注意力引导」画的，不是为「物件状态观测」画的。
实测：`取料区域` 矩形平均 72.5% 面积被工人框覆盖 —— 它在人身上，不在料盘上。
在这样的框里算"区域外观变化"，测到的是人的运动，不是物件的状态。

本脚本换一个问法：
    把整幅画面切成粗网格，逐格统计「**排除工人框之后**，该格与它自己
    长期中位数相比发生了变化的像素比例」，以及该格被工人覆盖的比例。

于是每个格子的含义变成：
    object_activity 高 + cover 低  →  这里真的有东西在出现/消失（料盘、成品位）
    object_activity 高 + cover 高  →  人在动（手在桌面区作业）
    object_activity 低 + cover 低  →  背景/墙面（对状态观测无用）

输出两个 ASCII 网格，可直接肉眼判断该把状态框画在哪。

用法
----
    .venv\\Scripts\\python.exe scripts\\probe_object_regions.py \
        --video vedio/v2.mp4 \
        --pose web/public/data/local_pose_full_v2.json \
        --out api_runs/object_probe
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

SPARK = " .:-=+*#%@"


def spark(v, hi):
    if not np.isfinite(v):
        return "?"
    if hi <= 0:
        return SPARK[0]
    k = int(round(max(0.0, min(1.0, v / hi)) * (len(SPARK) - 1)))
    return SPARK[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--pose", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rate", type=float, default=1.0, help="抽样频率 Hz（背景与统计都用它）")
    ap.add_argument("--work-w", type=int, default=192, help="工作宽度")
    ap.add_argument("--fg-thr", type=float, default=15.0)
    ap.add_argument("--cols", type=int, default=16)
    ap.add_argument("--rows", type=int, default=9)
    a = ap.parse_args()

    pose = json.loads(a.pose.read_text(encoding="utf-8-sig"))
    dur = float(pose.get("duration") or pose["frames"][-1]["timestamp"])
    cap = cv2.VideoCapture(str(a.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = a.work_w
    H = max(2, int(round(W * fh / fw)))

    times = np.arange(0.0, dur, 1.0 / max(a.rate, 0.05))
    # person bbox 查表：pose 10 Hz，取最近的 tick
    pframe = pose["frames"]
    ptimes = np.array([float(f["timestamp"]) for f in pframe])

    print(f"视频 {a.video} ｜ {fw}x{fh} @ {fps:.3f}fps ｜ 时长 {dur:.1f}s ｜ 抽样 {len(times)} 帧 @ {a.rate}Hz")
    t0 = time.time()
    frames = np.zeros((len(times), H, W), dtype=np.uint8)
    bboxes = [None] * len(times)
    cursor = -1
    for i, t in enumerate(times):
        want = int(round(t * fps))
        while cursor < want:
            if not cap.grab():
                break
            cursor += 1
        ok, frame = cap.retrieve() if cursor >= want else (False, None)
        if not ok or frame is None:
            continue
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames[i] = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA)
        j = int(np.argmin(np.abs(ptimes - t)))
        bboxes[i] = (pframe[j].get("person") or {}).get("bbox")
    cap.release()
    print(f"解码完成 {time.time() - t0:.0f}s")

    valid = frames.any(axis=(1, 2))
    if valid.sum() < 10:
        raise SystemExit("有效帧太少")
    bg = np.median(frames[valid].astype(np.float32), axis=0)

    # 每帧：排除人框后的变化像素
    ch_obj = np.zeros((len(times), H, W), dtype=np.float32)
    ch_all = np.zeros((len(times), H, W), dtype=np.float32)
    person = np.zeros((len(times), H, W), dtype=bool)
    for i in range(len(times)):
        if not valid[i]:
            continue
        d = np.abs(frames[i].astype(np.float32) - bg)
        m = (d > a.fg_thr)
        ch_all[i] = m.astype(np.float32)
        if bboxes[i]:
            bx0, by0, bx1, by1 = [float(v) for v in bboxes[i]]
            px0 = int(np.clip(bx0 / fw * W, 0, W - 1)); px1 = int(np.clip(bx1 / fw * W, 0, W))
            py0 = int(np.clip(by0 / fh * H, 0, H - 1)); py1 = int(np.clip(by1 / fh * H, 0, H))
            pm = np.zeros((H, W), dtype=bool)
            pm[py0:py1, px0:px1] = True
            person[i] = pm
            ch_obj[i] = m & ~pm
        else:
            ch_obj[i] = m

    # 粗网格聚合
    ys = np.linspace(0, H, a.rows + 1).round().astype(int)
    xs = np.linspace(0, W, a.cols + 1).round().astype(int)
    obj = np.zeros((a.rows, a.cols))
    cov = np.zeros((a.rows, a.cols))
    persist = np.zeros((a.rows, a.cols))
    for r in range(a.rows):
        for c in range(a.cols):
            sl = (slice(ys[r], max(ys[r] + 1, ys[r + 1])), slice(xs[c], max(xs[c] + 1, xs[c + 1])))
            o = ch_obj[valid][(slice(None),) + sl]
            p = person[valid][(slice(None),) + sl]
            obj[r, c] = float(o.mean())
            cov[r, c] = float(p.mean())
            persist[r, c] = float((o.mean(axis=(1, 2)) > 0.05).mean())

    a.out.mkdir(parents=True, exist_ok=True)
    doc = {
        "video": str(a.video), "frame": [fw, fh], "work": [W, H], "rate": a.rate,
        "fg_thr": a.fg_thr, "grid": [a.rows, a.cols],
        "cells": [{"row": r, "col": c,
                   "rect_norm": [xs[c] / W, ys[r] / H, xs[c + 1] / W, ys[r + 1] / H],
                   "obj_activity": round(obj[r, c], 4),
                   "person_cover": round(cov[r, c], 4),
                   "persist": round(persist[r, c], 4)}
                  for r in range(a.rows) for c in range(a.cols)],
    }
    (a.out / "object_probe.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")

    def grid_print(title, mat, fmt, hi):
        print(f"\n=== {title} ===（列 = 画面宽度 {a.cols} 等分，行 = 高度 {a.rows} 等分）")
        print("     " + "".join(f"{(xs[c] / W):>2.0%}"[:1] for c in range(a.cols)))
        for r in range(a.rows):
            line = "".join(spark(float(mat[r, c]), hi) for c in range(a.cols))
            print(f"  y{(ys[r] / H):.2f} {line}   " + fmt(mat[r, :]).strip())

    grid_print("物件活动度 obj_activity（已排除人框；高=这里真有东西在变）", obj,
               lambda row: f"max {row.max():.3f}", max(obj.max(), 1e-6))
    grid_print("工人框覆盖 person_cover（高=这格基本被挡）", cov,
               lambda row: f"max {row.max():.3f}", max(cov.max(), 1e-6))
    grid_print("持续物件状态 persist（该格变化像素>5% 的时间占比）", persist,
               lambda row: f"max {row.max():.3f}", max(persist.max(), 1e-6))

    print("\n=== 候选状态框（obj_activity 高 且 person_cover 低）===")
    print(f"{'row':>4}{'col':>4}{'obj':>8}{'cover':>8}{'persist':>9}   rect_norm")
    cand = sorted(
        [(r, c) for r in range(a.rows) for c in range(a.cols)
         if obj[r, c] > 0.08 and cov[r, c] < 0.25],
        key=lambda rc: -obj[rc[0], rc[1]])
    for r, c in cand[:12]:
        print(f"{r:>4}{c:>4}{obj[r, c]:>8.3f}{cov[r, c]:>8.3f}{persist[r, c]:>9.3f}   "
              f"[{xs[c] / W:.3f}, {ys[r] / H:.3f}, {xs[c + 1] / W:.3f}, {ys[r + 1] / H:.3f}]")
    if not cand:
        print("  （没有满足条件的格子：说明画面里没有稳定的物件位，或阈值需要调整）")
    print(f"\n写入 {a.out / 'object_probe.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
