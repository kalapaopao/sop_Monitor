"""实测这条链路每一环的推理耗时（本机 CPU / GPU），用于算 NX 多路预算。

为什么必须实测
--------------
"换一个更好的模型"很容易，但最终要跑在 Jetson NX 上、还要 4 路。
所以每加一步都要知道它吃掉多少 ms，否则方案是空谈。

测量项
------
  · YOLO(person) 640       —— 现有链路第 1 步
  · RTMW 192x256 onnx      —— 现有链路第 2 步（top-down，依赖上一步的框）
  · MobileNetV3-Small 112  —— 新增的「物件身份」通道（CPU / CUDA 各测一次）
  · 合起来每秒要跑多少次、占多少核

用法
----
    .venv\\Scripts\\python.exe scripts\\bench_inference.py \
        --video vedio/v2.mp4 --person-model model/yolo/best.onnx \
        --pose-model model/rtmw_wholebody_256x192.onnx \
        --obj-weights E:/.../mobilenet_v3_small-047dcff4.pth
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

MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)


def timeit(fn, warmup=5, iters=30):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--person-model", type=Path, required=True)
    ap.add_argument("--pose-model", type=Path, required=True)
    ap.add_argument("--obj-weights", type=Path, default=None)
    ap.add_argument("--iters", type=int, default=30)
    a = ap.parse_args()

    cap = cv2.VideoCapture(str(a.video))
    cap.set(cv2.CAP_PROP_POS_MSEC, 340 * 1000)   # 取一段有动作的帧
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("抽帧失败")
    H, W = frame.shape[:2]
    print(f"测试帧 {W}x{H} ｜ iters={a.iters}")

    res = {}

    # ---- 1) YOLO person ----
    from ultralytics import YOLO
    person = YOLO(str(a.person_model), task="detect", verbose=False)
    _ = person.predict(frame, conf=0.25, verbose=False)

    def run_yolo():
        r = next(iter(person.predict(frame, conf=0.25, verbose=False)))
        return r.boxes

    ms_yolo = timeit(run_yolo, 3, a.iters)
    r = next(iter(person.predict(frame, conf=0.25, verbose=False)))
    nb = 0 if r.boxes is None else len(r.boxes)
    res["yolo_person"] = ms_yolo
    print(f"YOLO(person) 640          {ms_yolo:7.2f} ms  （本帧检出 {nb} 个人）")

    # ---- 2) RTMW ----
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    pose = ort.InferenceSession(str(a.pose_model), so, providers=["CPUExecutionProvider"])
    inp = pose.get_inputs()[0].name
    canvas = cv2.resize(frame, (192, 256))
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
    tensor = np.transpose((rgb - MEAN) / STD, (2, 0, 1))[None].astype(np.float32)

    ms_pose = timeit(lambda: pose.run(None, {inp: tensor}), 3, a.iters)
    res["rtmw"] = ms_pose
    print(f"RTMW 192x256 onnx         {ms_pose:7.2f} ms")

    # ---- 3) 物件分类器 ----
    if a.obj_weights and Path(a.obj_weights).exists():
        import torch
        import torch.nn as nn
        from torchvision.models import mobilenet_v3_small
        m = mobilenet_v3_small(weights=None)
        sd = torch.load(str(a.obj_weights), map_location="cpu", weights_only=True)
        m.load_state_dict(sd, strict=False)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, 5)
        m.eval()
        x = torch.randn(1, 3, 112, 112)
        with torch.no_grad():
            ms_obj_cpu = timeit(lambda: m(x), 5, a.iters)
        res["obj_mobilenet_cpu"] = ms_obj_cpu
        print(f"MobileNetV3-S 112 (CPU)   {ms_obj_cpu:7.2f} ms")
        if torch.cuda.is_available():
            m2 = m.cuda()
            x2 = x.cuda()
            with torch.no_grad():
                ms_obj_gpu = timeit(lambda: m2(x2), 10, a.iters * 3)
            res["obj_mobilenet_cuda"] = ms_obj_gpu
            print(f"MobileNetV3-S 112 (CUDA)  {ms_obj_gpu:7.2f} ms  ← 本机独显，NX 上不可直接类比")

    # ---- 4) 裁图 + 解码开销（纯 OpenCV，可忽略但要算） ----
    ms_crop = timeit(lambda: cv2.resize(frame[300:460, 900:1080], (112, 112)), 5, a.iters * 3)
    res["crop_resize"] = ms_crop
    print(f"裁图+resize 112           {ms_crop:7.2f} ms")

    # ---- 汇总 ----
    base = res["yolo_person"] + res["rtmw"]
    obj_all = res.get("obj_mobilenet_cpu", 0.0) * 2 + res["crop_resize"] * 2
    print(f"\n=== 单帧单路汇总（CPU）===")
    print(f"现有链路（YOLO+RTMW）              {base:7.2f} ms")
    if a.obj_weights:
        print(f"若每帧对两只手都跑物件分类         +{obj_all:6.2f} ms  → 合计 {base + obj_all:.2f} ms")
        print(f"若物件分类降到 5 Hz（每 2 帧一次）  +{obj_all / 2:6.2f} ms  → 合计 {base + obj_all / 2:.2f} ms")
        print(f"若只在模型不确定的帧才跑（估 30%）  +{obj_all * 0.3:6.2f} ms  → 合计 {base + obj_all * 0.3:.2f} ms")

    print(f"\n=== 多路 / 多频率推算（按本机 CPU 实测，NX 上按 3 倍慢估）===")
    print(f"{'场景':<22}{'本机 CPU':>12}{'NX@3x 估':>12}{'占 6 核比例':>14}")
    configs = [
        ("1 路 @10Hz 现有链路", 1, 10, base),
        ("4 路 @5Hz 现有链路", 4, 5, base),
        ("4 路 @10Hz 现有链路", 4, 10, base),
    ]
    if a.obj_weights:
        configs += [
            ("4 路 @5Hz + 物件(全帧)", 4, 5, base + obj_all),
            ("4 路 @10Hz + 物件(门控)", 4, 10, base + obj_all * 0.3),
        ]
    for name, streams, hz, ms in configs:
        cores = streams * hz * ms / 1000.0
        nx = cores * 3
        print(f"{name:<22}{cores:>11.2f}c{nx:>11.2f}c{nx / 6:>13.0%}")

    print("\n注意：以上是桌面 CPU 实测；Jetson NX 按 3 倍慢是**假设**，必须在 NX 上重测。")
    print("      若走 TensorRT / FP16，NX 上通常还能再快 2~3 倍，同样未验证。")

    out = Path("api_runs/bench")
    out.mkdir(parents=True, exist_ok=True)
    (out / "inference_ms.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n写入 {out / 'inference_ms.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
