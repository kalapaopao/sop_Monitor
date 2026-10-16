#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
mkdir -p "$ROOT/build" "$ROOT/output"

"$TRTEXEC" --onnx="$ROOT/models/yolo_20261007_latest/best.raw_trt2.onnx" \
  --saveEngine="$ROOT/build/yolo.plan" --fp16 --workspace=2048 --buildOnly \
  2>&1 | tee "$ROOT/output/trtexec_yolo.log"

"$TRTEXEC" --onnx="$ROOT/models/yolox_tiny_8xb8-300e_humanart-6f3252f9.raw_trt.onnx" \
  --saveEngine="$ROOT/build/yolox.plan" --fp16 --workspace=1024 --buildOnly \
  2>&1 | tee "$ROOT/output/trtexec_yolox.log"

"$TRTEXEC" --onnx="$ROOT/models/rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.onnx" \
  --saveEngine="$ROOT/build/rtmw.plan" --fp16 --workspace=2048 \
  --minShapes=input:1x3x256x192 --optShapes=input:1x3x256x192 \
  --maxShapes=input:1x3x256x192 --buildOnly \
  2>&1 | tee "$ROOT/output/trtexec_rtmw.log"

