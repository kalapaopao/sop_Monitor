#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"
engine=models/joint_20261018/joint_fp16.plan
if [ -e "$engine" ]; then
  echo "Engine already exists: $engine; preserve it and choose a new version directory for a rebuild." >&2
  exit 1
fi
mkdir -p output/joint_20261019
test -s build/libsop_silu_plugin.so
if [ ! -f models/joint_20261018/best.silu.onnx ]; then
  python3 tools/prepare_silu_plugin.py models/joint_20261018/best.raw.onnx models/joint_20261018/best.silu.onnx
fi
python3 tools/build_joint_strict.py models/joint_20261018/best.silu.onnx "$engine" \
  --plugin-library build/libsop_silu_plugin.so \
  2>&1 | tee output/joint_20261019/build-engine.log
