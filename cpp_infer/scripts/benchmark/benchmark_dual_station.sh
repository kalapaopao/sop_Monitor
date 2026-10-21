#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:-output/dual_03_07_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
tegrastats --interval 500 > "$OUT/tegrastats-dual.log" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
for mode in headless gui; do
  flags=()
  if [ "$mode" = headless ]; then flags+=(--headless); fi
  ./build/sop_nx_dual_probe --source ../test_vedio/03.mp4 --second-source ../test_vedio/07.mp4 \
    --model-dir models --start-sec 180 --duration 30 \
    --metrics "$OUT/dual_$mode.csv" "${flags[@]}" > "$OUT/dual_$mode.log" 2>&1
  tail -n 2 "$OUT/dual_$mode.log"
done
