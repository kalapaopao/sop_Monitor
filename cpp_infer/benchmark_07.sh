#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
OUT="${1:-output/pipeline_07_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
sha256sum ../test_vedio/07.mp4 build/yolo.plan build/yolox.plan build/rtmw.plan > "$OUT/assets.sha256"
for pass in 1 2; do
  modes="serial pipeline parallel"
  if [ "$pass" = 2 ]; then modes="parallel pipeline serial"; fi
  for mode in $modes; do
    name="${mode}_111_run${pass}"
    ./build/sop_nx_infer --source ../test_vedio/07.mp4 --model-dir models \
      --mode "$mode" --headless --start-sec 180 --max-frames 330 --warmup-frames 30 \
      --metrics "$OUT/$name.csv" > "$OUT/$name.log" 2>&1
    tail -n 1 "$OUT/$name.log"
  done
done
for mode in serial pipeline parallel; do
  name="${mode}_223"
  ./build/sop_nx_infer --source ../test_vedio/07.mp4 --model-dir models \
    --mode "$mode" --headless --start-sec 180 --max-frames 330 --warmup-frames 30 \
    --obb-stride 2 --person-stride 2 --pose-stride 3 \
    --metrics "$OUT/$name.csv" > "$OUT/$name.log" 2>&1
  tail -n 1 "$OUT/$name.log"
done
printf 'Reports: %s/%s\n' "$ROOT" "$OUT"
