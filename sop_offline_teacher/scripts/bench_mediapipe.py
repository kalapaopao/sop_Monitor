"""单独测 MediaPipe HandLandmarker 的纯推理耗时（排除视频 seek 干扰）。

背景：整段跑 400 tick 用了 58s（145ms/tick），需要判断慢在推理还是慢在读帧。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

VIDEO = Path("vedio/v2.mp4")
MODEL = Path("model/mediapipe/hand_landmarker.task")


def main():
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision

    opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    lm = vision.HandLandmarker.create_from_options(opts)

    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_MSEC, 400 * 1000)     # c4：有装配与紧固
    frames = []
    for _ in range(30):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    print(f"取到 {len(frames)} 帧")

    def to_img(f, scale=1.0):
        if scale != 1.0:
            f = cv2.resize(f, (int(f.shape[1] * scale), int(f.shape[0] * scale)),
                           interpolation=cv2.INTER_AREA)
        return mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB))

    ts = 0

    def next_ts():
        nonlocal ts
        ts += 100
        return ts

    for scale in (1.0, 0.5):
        imgs = [to_img(f, scale) for f in frames]
        for im in imgs[:5]:
            lm.detect_for_video(im, next_ts())
        t0 = time.perf_counter()
        n = 0
        for im in imgs:
            lm.detect_for_video(im, next_ts())
            n += 1
        ms = (time.perf_counter() - t0) / n * 1000
        res = lm.detect_for_video(imgs[0], next_ts())
        print(f"  scale={scale:<4} 纯推理 {ms:6.2f} ms/帧 ｜ 本帧检出 {len(res.hand_landmarks)} 只手")

    # 读帧成本（顺序 grab+retrieve，不做 seek）
    cap = cv2.VideoCapture(str(VIDEO))
    t0 = time.perf_counter()
    n = 0
    for _ in range(100):
        ok, f = cap.grab()
        if not ok:
            break
        ok2, f2 = cap.retrieve()
        n += 1
    grab_ms = (time.perf_counter() - t0) / max(1, n) * 1000
    cap.release()

    # seek 成本（现有脚本的写法：每 tick set POS_MSEC）
    cap = cv2.VideoCapture(str(VIDEO))
    t0 = time.perf_counter()
    for i in range(100):
        cap.set(cv2.CAP_PROP_POS_MSEC, (400 + i * 0.1) * 1000)
        ok, f = cap.read()
    seek_ms = (time.perf_counter() - t0) / 100 * 1000
    cap.release()

    print(f"\n  顺序 grab+retrieve  {grab_ms:6.2f} ms/帧")
    print(f"  每帧 seek+read      {seek_ms:6.2f} ms/帧  ← 现有脚本的写法")
    print("\n结论：seek 是主要开销时，离线批处理应改成顺序解码。")
    lm.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
