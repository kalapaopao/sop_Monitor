"""从参考视频里抽帧，供 OCR 使用（放大 2 倍，小字才认得出来）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--every", type=float, default=5.0, help="每多少秒抽一帧")
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--max", type=int, default=0, help="最多抽多少帧，0=不限")
    a = ap.parse_args()

    cap = cv2.VideoCapture(str(a.video))
    if not cap.isOpened():
        raise SystemExit(f"打不开 {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = n / max(fps, 1e-6)
    times = []
    t = 0.0
    while t < dur:
        times.append(round(t, 2))
        t += a.every
    if a.max:
        times = times[:a.max]
    a.out.mkdir(parents=True, exist_ok=True)
    print(f"{a.video.name} {dur:.1f}s ｜ 抽 {len(times)} 帧 ｜ 放大 {a.scale}x")

    cursor = -1
    saved = 0
    for ts in times:
        want = int(round(ts * fps))
        while cursor < want:
            if not cap.grab():
                break
            cursor += 1
        ok, frame = cap.retrieve() if cursor >= want else (False, None)
        if not ok or frame is None:
            continue
        if a.scale != 1.0:
            frame = cv2.resize(frame, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_CUBIC)
        name = f"{int(round(ts * 10)):05d}_{ts:06.1f}s.png"
        cv2.imwrite(str(a.out / name), frame)
        saved += 1
    cap.release()
    print(f"写入 {saved} 张 → {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
