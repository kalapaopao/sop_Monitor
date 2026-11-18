"""一次解码同时跑 RTMW（top-down）和 MediaPipe（single-stage），并量化"为什么手不稳"。

为什么合并成一个脚本
--------------------
分两次跑要解码两遍视频。实测每帧 seek 约 130ms、顺序解码约 6ms，
所以一次解码跑两个模型能省一半时间，而且**两个模型看到的是同一帧**，
对比才有意义。

复用 run_local_pose.py 的 crop_person / decode / keep_hand，保证几何与现有链路完全一致
（letterbox 留边、pad=0.1、掌骨比例过滤都不重写一套，避免出现"两套口径"）。

额外产出：keep_hand 的拒绝原因统计
----------------------------------
`keep_hand` 有三个拒绝条件，侧视角下最容易踩的是"贴画面边缘"和"点太散"。
这里逐条计数，直接回答"侧偏的手为什么不稳"。

用法
----
    .venv\\Scripts\\python.exe scripts/extract_hands_dual.py \
        --video vedio/v_3/side2.mp4 \
        --person-model model/yolo/best.onnx \
        --pose-model model/rtmw_wholebody_256x192.onnx \
        --mp-model model/mediapipe/hand_landmarker.task \
        --out api_runs/v3_side2 --rate 10
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "api_teacher"))
import run_local_pose as RLP  # noqa: E402  复用裁剪/解码/过滤，保证与现有链路同一口径

P_THR = 0.18


def reason_of(points, frame_shape, score_thr=0.18, min_extent=30.0, border=8.0, max_palm_ratio=5.0):
    """复刻 keep_hand 的判定，但返回拒绝原因而不是布尔值。"""
    valid = [p for p in points if p[2] > score_thr]
    if len(valid) < 4:
        return "点数不足"
    xs = [p[0] for p in valid]
    ys = [p[1] for p in valid]
    extent = max(max(xs) - min(xs), max(ys) - min(ys))
    h, w = frame_shape[:2]
    near = any(x < border or y < border or x > w - border or y > h - border
               for x, y in zip(xs, ys))
    if extent < min_extent:
        return "点太散"
    if near:
        return "贴画面边缘"
    r = RLP.palm_ratio(points, score_thr)
    if max_palm_ratio and r is not None and r > max_palm_ratio:
        return "掌骨比例异常"
    return "ok"


def extent_px(pts):
    if not pts:
        return 0.0
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--person-model", type=Path, required=True)
    ap.add_argument("--pose-model", type=Path, required=True)
    ap.add_argument("--mp-model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rate", type=float, default=10.0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=0.0)
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO person 置信度")
    ap.add_argument("--mp-conf", type=float, default=0.15)
    ap.add_argument("--no-mp", action="store_true", help="只跑 RTMW（快速冒烟）")
    ap.add_argument("--max-palm-ratio", type=float, default=5.0,
                    help="keep_hand 的掌骨比例上限；0=关闭。正面调到 5.0 的经验值，"
                         "侧面因透视会误杀真手，需要重新定")
    ap.add_argument("--border", type=float, default=8.0,
                    help="keep_hand 的贴边阈值（像素）；手靠近画面边缘时会被判为假手，"
                         "侧机位很容易踩到，可放宽")
    ap.add_argument("--bbox-fallback", type=float, default=0.0,
                    help=">0 时：YOLO 本帧没框到人，就复用最近 N 秒内最后一个框继续跑 RTMW。"
                         "这是对『人一动手就丢』最直接的补法；0=关闭")
    ap.add_argument("--timing", action="store_true", help="打印各阶段耗时")
    a = ap.parse_args()

    import onnxruntime as ort
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision
    from ultralytics import YOLO

    cap = cv2.VideoCapture(str(a.video))
    if not cap.isOpened():
        raise SystemExit(f"打不开 {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_s = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    end = min(total_s, a.start + a.duration) if a.duration > 0 else total_s
    ticks = []
    t = a.start
    while t < end:
        ticks.append(round(t, 3))
        t += 1.0 / a.rate

    person = YOLO(str(a.person_model), task="detect", verbose=False)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    pose = ort.InferenceSession(str(a.pose_model), so, providers=["CPUExecutionProvider"])
    pin = pose.get_inputs()[0].name

    lm = None
    if not a.no_mp:
        lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(a.mp_model)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=a.mp_conf,
            min_hand_presence_confidence=a.mp_conf,
            min_tracking_confidence=0.2,
        ))

    print(f"{a.video.name} {W}x{H} @{fps:.3f} ｜ tick {len(ticks)} 个 @{a.rate}Hz")
    rows = []
    rej = {"左手": {}, "右手": {}}
    t0 = time.time()
    acc = {"decode": 0.0, "yolo": 0.0, "crop": 0.0, "rtmw": 0.0, "post": 0.0, "mp": 0.0, "n": 0}
    cursor = -1
    mp_ts = 0
    last_box, last_box_t = None, -1e9
    for k, ts in enumerate(ticks):
        _t = time.perf_counter()
        want_f = int(round(ts * fps))
        while cursor < want_f:
            if not cap.grab():
                break
            cursor += 1
        ok, frame = cap.retrieve() if cursor >= want_f else (False, None)
        acc["decode"] += time.perf_counter() - _t
        acc["n"] += 1
        if not ok or frame is None:
            rows.append({"timestamp": ts, "person": None, "left_hand": [], "right_hand": [],
                         "body": {}, "mp_left": [], "mp_right": [], "rej": {}})
            continue

        row = {"timestamp": ts, "person": None, "left_hand": [], "right_hand": [],
               "body": {}, "mp_left": [], "mp_right": []}
        # ---- RTMW（top-down）----
        _t = time.perf_counter()
        res = next(iter(person.predict(frame, conf=a.conf, verbose=False)))
        acc["yolo"] += time.perf_counter() - _t
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.empty((0, 4))
        scores = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.empty(0)
        row["rej"] = {}
        if len(boxes):
            i = int(np.argmax(scores))
            box = boxes[i].tolist()
            box_src = "yolo"
            last_box, last_box_t = box, ts
        elif a.bbox_fallback > 0 and last_box is not None and (ts - last_box_t) <= a.bbox_fallback:
            # 本帧 YOLO 没框到人，但人不会瞬移：复用最近的框继续跑 RTMW。
            # 这是对"人一动手/一动身就丢手"最直接的补法。
            box = last_box
            box_src = "fallback"
        else:
            box = None
            box_src = "none"
        row["box_src"] = box_src
        if box is not None:
            _t = time.perf_counter()
            tensor, meta = RLP.crop_person(frame, box, 0.1)
            acc["crop"] += time.perf_counter() - _t
            if tensor is not None:
                _t = time.perf_counter()
                pts = RLP.decode(pose.run(None, {pin: tensor}), meta)
                acc["rtmw"] += time.perf_counter() - _t
                _t = time.perf_counter()
                lh_raw, rh_raw = pts[91:112], pts[112:133]
                for name, raw in (("left_hand", lh_raw), ("right_hand", rh_raw)):
                    r = reason_of(raw, frame.shape, border=a.border,
                                  max_palm_ratio=a.max_palm_ratio)
                    row["rej"][name] = r
                    if r == "ok":
                        row[name] = raw
                    else:
                        rej["左手" if name == "left_hand" else "右手"][r] = \
                            rej["左手" if name == "left_hand" else "右手"].get(r, 0) + 1
                row["person"] = {"bbox": box,
                                 "score": (float(scores[i]) if len(boxes) else 0.0),
                                 "src": box_src}
                def _p(idx):
                    q = pts[idx]
                    return [float(v) for v in np.asarray(q).ravel()[:3]]
                row["body"] = {kk: _p(ii) for kk, ii in (
                    ("shoulder_l", 5), ("shoulder_r", 6), ("elbow_l", 7), ("elbow_r", 8),
                    ("wrist_l", 9), ("wrist_r", 10), ("hip_l", 11), ("hip_r", 12))}
                acc["post"] += time.perf_counter() - _t

        # ---- MediaPipe（single-stage，全图）----
        if lm is not None:
            _t = time.perf_counter()
            mp_ts += 100
            img = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            r = lm.detect_for_video(img, mp_ts)
            dets = []
            for j, hands in enumerate(r.hand_landmarks):
                lab = "Right"
                sc = 0.0
                if r.handedness and j < len(r.handedness) and r.handedness[j]:
                    cat = r.handedness[j][0]
                    lab, sc = cat.category_name, float(cat.score)
                pts = [[float(p.x) * W, float(p.y) * H, 1.0] for p in hands]
                if len(pts) == 21 and extent_px(pts) >= 20:
                    dets.append((sc, lab, pts))
            dets.sort(key=lambda z: -z[0])
            slot = {"Left": None, "Right": None}
            for sc, lab, pts in dets:
                if slot[lab] is None:
                    slot[lab] = pts
                elif slot["Right" if lab == "Left" else "Left"] is None:
                    slot["Right" if lab == "Left" else "Left"] = pts
            row["mp_left"] = slot["Left"] or []
            row["mp_right"] = slot["Right"] or []
            acc["mp"] += time.perf_counter() - _t
        rows.append(row)
        if (k + 1) % 1000 == 0:
            print(f"  {k + 1}/{len(ticks)}  {time.time() - t0:.0f}s")
    cap.release()
    if lm is not None:
        lm.close()
    print(f"提取完成 {time.time() - t0:.0f}s")

    n = len(rows)
    if a.timing and acc["n"]:
        print(f"\n=== 每阶段平均耗时（{acc['n']} tick）===")
        tot = 0.0
        for k in ("decode", "yolo", "crop", "rtmw", "post", "mp"):
            ms = acc[k] / acc["n"] * 1000
            tot += ms
            print(f"  {k:<8}{ms:8.2f} ms")
        print(f"  {'合计':<8}{tot:8.2f} ms ｜ 实际 {1000 * (time.time() - t0) / acc['n']:.2f} ms")
    r_one = sum(1 for r in rows if r["left_hand"] or r["right_hand"])
    r_both = sum(1 for r in rows if r["left_hand"] and r["right_hand"])
    m_one = sum(1 for r in rows if r["mp_left"] or r["mp_right"])
    m_both = sum(1 for r in rows if r["mp_left"] and r["mp_right"])
    f_one = sum(1 for r in rows if (r["left_hand"] or r["right_hand"]
                                    or r["mp_left"] or r["mp_right"]))
    print(f"\n=== 覆盖率（{n} tick）===")
    print(f"{'通道':<14}{'至少一只':>10}{'双手':>9}")
    print(f"{'RTMW':<14}{r_one / n:>10.1%}{r_both / n:>9.1%}")
    if lm is not None:
        print(f"{'MediaPipe':<14}{m_one / n:>10.1%}{m_both / n:>9.1%}")
        print(f"{'两者合并':<14}{f_one / n:>10.1%}{'—':>9}")

    # 手的大小（侧视角下会变小，这是"不稳"的直接原因之一）
    rl = [extent_px(r["left_hand"]) for r in rows if r["left_hand"]]
    rr = [extent_px(r["right_hand"]) for r in rows if r["right_hand"]]
    if rl:
        print(f"\nRTMW 手框边长 p10/p50/p90：左 {np.percentile(rl, 10):.0f}/{np.median(rl):.0f}/"
              f"{np.percentile(rl, 90):.0f} px ｜ 右 {np.percentile(rr, 10):.0f}/{np.median(rr):.0f}/"
              f"{np.percentile(rr, 90):.0f} px" if rr else "")

    # 拒绝原因
    if rej["左手"] or rej["右手"]:
        print(f"\n=== RTMW 手被 keep_hand 拒绝的原因（次数）===")
        for side in ("左手", "右手"):
            tot = sum(rej[side].values())
            if tot:
                detail = " ".join(f"{k}:{v}" for k, v in sorted(rej[side].items(), key=lambda z: -z[1]))
                print(f"  {side} 共 {tot} 次 → {detail}")

    # 抖动
    def jitter(key):
        vals = []
        prev = None
        for r in rows:
            p = r[key]
            if p and prev and len(p) == len(prev):
                e = max(1.0, extent_px(p))
                vals.append(float(np.mean([abs(p[i][0] - prev[i][0]) + abs(p[i][1] - prev[i][1])
                                           for i in range(len(p))])) / 2.0 / e)
            prev = p if p else prev
        return float(np.median(vals)) if vals else None

    print(f"\n=== 帧间抖动（按手框边长归一，越低越稳）===")
    print(f"  RTMW 左手 {jitter('left_hand')}  右手 {jitter('right_hand')}")
    print(f"  MediaPipe 左手 {jitter('mp_left')}  右手 {jitter('mp_right')}")

    a.out.mkdir(parents=True, exist_ok=True)
    base = {"video": str(a.video), "fps": round(float(fps), 6), "duration": round(float(total_s), 3),
            "rate": a.rate}
    (a.out / "rtmw.json").write_text(json.dumps(
        {**base, "source": "rtmw_topdown",
         "frames": [{"timestamp": r["timestamp"], "person": r["person"],
                     "left_hand": r["left_hand"], "right_hand": r["right_hand"],
                     "body": r["body"], "reject": r.get("rej", {})} for r in rows]},
        ensure_ascii=False), encoding="utf-8")
    if lm is not None:
        (a.out / "mediapipe.json").write_text(json.dumps(
            {**base, "source": "mediapipe_hand_landmarker",
             "frames": [{"timestamp": r["timestamp"], "person": None,
                         "left_hand": r["mp_left"], "right_hand": r["mp_right"], "body": {}}
                        for r in rows]}, ensure_ascii=False), encoding="utf-8")
    print(f"\n写入 {a.out / 'rtmw.json'}" + (f" 与 {a.out / 'mediapipe.json'}" if lm else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
