"""用 MediaPipe HandLandmarker 提取手部 21 点，输出与现有 pose JSON 同格式。

为什么要换
----------
现有链路是 top-down：YOLO 必须先框到人，RTMW 才在人的裁图里找手。
实测 5992 帧里：无 person 框 12.0%，而这些帧**结构上必然没有手**；
"有框但手没出来"只占 1.7%。也就是说手缺失的 87.7% 来自人框漏检。

MediaPipe HandLandmarker 是 single-stage：palm 检测器直接在全图上找手掌，
再对手掌小图回归 21 点，**不需要 person 框**。还自带跨帧跟踪。

输出格式
--------
与 `web/public/data/local_pose_full_v2.json` 完全一致：
    {"frames":[{"timestamp":t,"person":null|{"bbox":..,"score":..},
                "left_hand":[[x,y,score]×21],"right_hand":[...],"body":{}}]}
差别：person 为 null（本模型不产出人框），body 为空（不产出身体点），
并在顶层写 "source":"mediapipe_hand_landmarker" 标明来源。

左右手的标签坑（必须实测确认，不能照文档猜）
--------------------------------------------
MediaPipe 的 handedness 是**按输入图像已镜像（自拍）**这个假设给出的。
监控视频通常不镜像，所以原始 label 很可能是反的。
本脚本提供 --handedness raw|flip|auto：
  auto 会和一份参考 json（RTMW）比对两者哪种与参考更一致，再决定，
  并把两种一致率都打印出来 —— 以数据定，不靠猜。

用法
----
    .venv\\Scripts\\python.exe scripts/run_mediapipe_hands.py \
        --video vedio/v2.mp4 --model model/mediapipe/hand_landmarker.task \
        --output web/public/data/local_pose_mp.json --rate 10
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


def build_landmarker(model_path: Path, num_hands: int, det: float, pres: float, track: float):
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision

    opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=det,
        min_hand_presence_confidence=pres,
        min_tracking_confidence=track,
    )
    return vision.HandLandmarker.create_from_options(opts)


def hand_extent(pts, score_thr):
    v = [(p[0], p[1]) for p in pts if p[2] > score_thr]
    if len(v) < 4:
        return 0.0
    xs = [p[0] for p in v]
    ys = [p[1] for p in v]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=0.0, help="0=到片尾")
    ap.add_argument("--rate", type=float, default=10.0)
    ap.add_argument("--num-hands", type=int, default=2)
    ap.add_argument("--det-conf", type=float, default=0.3)
    ap.add_argument("--pres-conf", type=float, default=0.3)
    ap.add_argument("--track-conf", type=float, default=0.3)
    ap.add_argument("--handedness", choices=["raw", "flip", "auto"], default="auto")
    ap.add_argument("--reference", type=Path, default=None,
                    help="参考 pose json（RTMW 产出），用于 auto 判定左右手与稳定性对比")
    ap.add_argument("--min-extent", type=float, default=20.0, help="21 点外接框小于此值视为假检测（像素）")
    ap.add_argument("--crop", default="", help="先裁作业区再送模型，格式 x0,y0,x1,y1（归一化）。"
                                             "留空=全图。理由：MediaPipe palm 检测内部把输入缩到 192，"
                                             "1080p 里 100px 的手缩完只剩 ~10px，必须先放大。")
    ap.add_argument("--crop-size", type=int, default=640, help="裁出来的区域放大到多少像素再送模型")
    a = ap.parse_args()

    import mediapipe as mp

    cap = cv2.VideoCapture(str(a.video))
    if not cap.isOpened():
        raise SystemExit(f"打不开视频 {a.video}")
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

    crop_box = None
    if a.crop.strip():
        parts = [float(v) for v in a.crop.split(",")]
        if len(parts) != 4:
            raise SystemExit("--crop 需要 4 个归一化数：x0,y0,x1,y1")
        crop_box = (int(parts[0] * W), int(parts[1] * H), int(parts[2] * W), int(parts[3] * H))
        print(f"先裁作业区 {crop_box}（像素）→ 放大到 {a.crop_size}² 再送模型")

    lm = build_landmarker(a.model, a.num_hands, a.det_conf, a.pres_conf, a.track_conf)
    print(f"视频 {a.video} {W}x{H} @{fps:.3f} ｜ tick {len(ticks)} 个 @{a.rate}Hz")
    print(f"MediaPipe {mp.__version__} ｜ 模型 {a.model.name} ｜ 最多 {a.num_hands} 只手")

    frames_raw = []          # 存原始 label，后面再决定要不要翻
    t0 = time.time()
    # 顺序 grab 到目标帧，避免每 tick 一次 seek（实测 seek 约 130ms/次，比推理还贵 8 倍）
    cursor = -1
    for k, ts in enumerate(ticks):
        want_f = int(round(ts * fps))
        while cursor < want_f:
            if not cap.grab():
                break
            cursor += 1
        ok, frame = cap.retrieve() if cursor >= want_f else (False, None)
        if not ok or frame is None:
            frames_raw.append({"timestamp": ts, "Left": [], "Right": [],
                               "LeftS": 0.0, "RightS": 0.0})
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if crop_box is not None:
            cx0, cy0, cx1, cy1 = crop_box
            sub = frame[cy0:cy1, cx0:cx1]
            if sub.size == 0:
                frames_raw.append({"timestamp": ts, "Left": [], "Right": [],
                                   "LeftS": 0.0, "RightS": 0.0})
                continue
            rgb = cv2.cvtColor(cv2.resize(sub, (a.crop_size, a.crop_size),
                                          interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = lm.detect_for_video(img, int(round(ts * 1000)))
        # 先收集所有检出的手，再分配左右槽位。
        # 不能按 label 直接去重：MediaPipe 偶尔把两只手都标成同一侧，
        # 那样会把真正存在的第二只手丢掉（实测双手覆盖率会被压到 4%）。
        dets = []
        for i, hands in enumerate(res.hand_landmarks):
            lab = "Right"
            sc = 0.0
            if res.handedness and i < len(res.handedness) and res.handedness[i]:
                cat = res.handedness[i][0]
                lab = cat.category_name
                sc = float(cat.score)
            if crop_box is not None:
                cx0, cy0, cx1, cy1 = crop_box
                cw, ch = max(1, cx1 - cx0), max(1, cy1 - cy0)
                pts = [[cx0 + float(p.x) * cw, cy0 + float(p.y) * ch,
                        float(getattr(p, "visibility", 1.0) or 1.0)] for p in hands]
            else:
                pts = [[float(p.x) * W, float(p.y) * H,
                        float(getattr(p, "visibility", 1.0) or 1.0)] for p in hands]
            if len(pts) < 21:
                continue
            if hand_extent(pts, 0.0) < a.min_extent:
                continue
            dets.append((sc, lab, pts))
        dets.sort(key=lambda z: -z[0])
        got = {"Left": None, "Right": None, "LeftS": 0.0, "RightS": 0.0}
        for sc, lab, pts in dets:
            if got[lab] is None:
                got[lab], got[lab + "S"] = pts, sc
            else:
                other = "Right" if lab == "Left" else "Left"
                if got[other] is None:
                    got[other], got[other + "S"] = pts, sc
        frames_raw.append({"timestamp": ts,
                           "Left": got["Left"] or [], "LeftS": got["LeftS"],
                           "Right": got["Right"] or [], "RightS": got["RightS"]})
        if (k + 1) % 1000 == 0:
            print(f"  {k + 1}/{len(ticks)}  {time.time() - t0:.0f}s")
    lm.close()
    cap.release()
    print(f"提取完成 {time.time() - t0:.0f}s")

    # ---- 左右手标签判定 ----
    mode = a.handedness
    ref = None
    ref_by_t = {}
    if a.reference and a.reference.exists():
        ref = json.loads(a.reference.read_text(encoding="utf-8-sig"))
        for f in ref["frames"]:
            ref_by_t[round(float(f["timestamp"]), 3)] = f

    def centroid(pts):
        return None if not pts else (float(np.mean([p[0] for p in pts])),
                                     float(np.mean([p[1] for p in pts])))

    def agreement(flip):
        """与参考 json 比：同一 tick 上，本方法判为左手的那只，离参考左手更近还是右手更近。"""
        if not ref_by_t:
            return None
        ok = tot = 0
        for fr in frames_raw:
            r = ref_by_t.get(round(fr["timestamp"], 3))
            if not r:
                continue
            rl = centroid(r.get("left_hand") or [])
            rr = centroid(r.get("right_hand") or [])
            if rl is None and rr is None:
                continue
            mp_l = (fr["Right"] if flip else fr["Left"])
            c = centroid(mp_l)
            if c is None:
                continue
            dl = 1e9 if rl is None else ((c[0] - rl[0]) ** 2 + (c[1] - rl[1]) ** 2) ** 0.5
            dr = 1e9 if rr is None else ((c[0] - rr[0]) ** 2 + (c[1] - rr[1]) ** 2) ** 0.5
            tot += 1
            ok += int(dl <= dr)
        return (ok / tot) if tot else None

    if mode == "auto":
        ag_raw = agreement(False)
        ag_flip = agreement(True)
        print(f"\n=== 左右手标签判定（与 RTMW 参考比）===")
        print(f"  不翻（raw） ：MediaPipe 的 Left 与 RTMW 左手一致率 {ag_raw:.1%}" if ag_raw is not None
              else "  无参考，跳过 auto 判定，按 raw 处理")
        print(f"  翻转(flip)  ：一致率 {ag_flip:.1%}" if ag_flip is not None else "")
        if ag_raw is not None and ag_flip is not None:
            flip = ag_flip > ag_raw
            print(f"  → 采用 {'flip（说明监控视频未镜像，MediaPipe label 是反的）' if flip else 'raw'}")
        else:
            flip = False
    else:
        flip = (mode == "flip")
        print(f"\n左右手：强制 {mode}")

    rows = []
    for fr in frames_raw:
        L = fr["Right"] if flip else fr["Left"]
        R = fr["Left"] if flip else fr["Right"]
        LS = fr["RightS"] if flip else fr["LeftS"]
        RS = fr["LeftS"] if flip else fr["RightS"]
        rows.append({"timestamp": fr["timestamp"], "person": None,
                     "left_hand": L, "right_hand": R, "body": {},
                     "_ls": LS, "_rs": RS})

    # ---- 统计 ----
    n = len(rows)
    both = sum(1 for r in rows if r["left_hand"] and r["right_hand"])
    one = sum(1 for r in rows if r["left_hand"] or r["right_hand"])
    print(f"\n=== MediaPipe 手检测覆盖（{n} tick）===")
    print(f"  双手   {both / n:6.1%}")
    print(f"  至少一只 {one / n:6.1%}")
    print(f"  没有    {(n - one) / n:6.1%}")

    out = a.output
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {"video": str(a.video), "fps": round(float(fps), 6), "duration": round(float(total_s), 3),
           "rate": a.rate, "pad": 0.0, "source": "mediapipe_hand_landmarker",
           "model": str(a.model), "handedness_flip": bool(flip),
           "params": {"num_hands": a.num_hands, "det_conf": a.det_conf,
                      "pres_conf": a.pres_conf, "track_conf": a.track_conf,
                      "min_extent": a.min_extent},
           "frames": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]}
    out.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    print(f"\n写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
