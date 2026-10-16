#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
OUT="${1:-output/optimization_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
children=()
cleanup() {
  for pid in "${children[@]}"; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM
tegrastats --interval 500 > "$OUT/tegrastats.log" &
children+=("$!")
monitor=${children[0]}
# Reverse order on pass 2 reduces a simple warm-cache/order bias.
for pass in 1 2; do
  paths="full preview"
  if [ "$pass" = 2 ]; then paths="preview full"; fi
  for path in $paths; do
    flags=()
    if [ "$path" = preview ]; then flags+=(--preview-first); fi
    name="gui_${path}_${pass}"
    ./build/sop_nx_infer --source ../test_vedio/07.mp4 --model-dir models \
      --mode parallel --start-sec 180 --max-frames 330 --warmup-frames 30 \
      --metrics "$OUT/$name.csv" "${flags[@]}" > "$OUT/$name.log" 2>&1
    tail -n 1 "$OUT/$name.log"
  done
done
# Independent processes are a capacity probe, not the proposed shared-model architecture.
for count in 1 2 3; do
  available=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  if [ "$available" -lt 2200000 ]; then
    printf 'Insufficient available RAM for %s-process probe: %s kB\n' "$count" "$available" >&2
    exit 1
  fi
  pids=()
  for ((stream=1; stream<=count; stream++)); do
    name="capacity_${count}_${stream}"
    ./build/sop_nx_infer --source ../test_vedio/07.mp4 --model-dir models \
      --mode parallel --realtime --headless --start-sec 180 --max-frames 500 \
      --warmup-frames 75 --metrics "$OUT/$name.csv" > "$OUT/$name.log" 2>&1 &
    pids+=("$!")
    children+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  children=("$monitor")
  if [ "$failed" = 1 ]; then echo "Capacity probe failed: $count" >&2; exit 1; fi
  for ((stream=1; stream<=count; stream++)); do tail -n 1 "$OUT/capacity_${count}_${stream}.log"; done
done
