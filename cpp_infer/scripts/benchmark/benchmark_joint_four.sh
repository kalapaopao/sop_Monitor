#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:-output/joint_20261019}"
mkdir -p "$OUT"
tegrastats --interval 500 > "$OUT/tegrastats.log" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
inputs=(--source ../test_vedio/03.mp4 --second-source ../test_vedio/03.mp4
        --third-source ../test_vedio/07.mp4 --fourth-source ../test_vedio/07.mp4
        --model-dir models --start-sec 0 --duration 30)
for mode in headless gui; do
  flags=()
  if [ "$mode" = headless ]; then flags+=(--headless); fi
  for detector in baseline joint; do
    engine=()
    executable=./build/sop_nx_dual_probe
    if [ "$detector" = joint ]; then
      executable=./build/sop_nx_joint_probe
      engine=(--yolo-engine models/joint_20261018/joint_fp16.plan)
    fi
    name="${detector}_${mode}"
    "$executable" "${inputs[@]}" "${flags[@]}" "${engine[@]}" \
      --metrics "$OUT/$name.csv" > "$OUT/$name.log" 2>&1
    tail -n 4 "$OUT/$name.log"
  done
done
