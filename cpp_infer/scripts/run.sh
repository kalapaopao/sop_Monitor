#!/usr/bin/env bash
# 一键启动 sop_nx_batch_probe
#
# 用法:
#   bash scripts/run.sh              默认 --preset 4lane
#   bash scripts/run.sh 2lane        --preset 2lane
#   bash scripts/run.sh 1lane        --preset 1lane
#
# 追加任意参数（会覆盖 preset 里的值）:
#   bash scripts/run.sh 4lane --duration 10
#   bash scripts/run.sh 1lane --headless=false 是无效的，用下面方式看画面：
#   bash scripts/run.sh 1lane 03.mp4 07.mp4        # 位置参数直接给视频
#
set -euo pipefail

CV_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$CV_ROOT/build/sop_nx_batch_probe"

if [ ! -x "$BIN" ]; then
  echo "可执行文件不存在: $BIN" >&2
  echo "请先编译: cd $CV_ROOT/build && make -j2 sop_nx_batch_probe" >&2
  exit 1
fi

PRESET="${1:-4lane}"
shift || true

# 如果第一个参数是存在的文件/目录，当作位置参数而不是 preset
if [ -e "$PRESET" ] || [[ "$PRESET" == */* ]]; then
  set -- "$PRESET" "$@"
  PRESET=""
fi

ARGS=()
if [ -n "$PRESET" ]; then
  ARGS+=(--preset "$PRESET")
fi
ARGS+=("$@")

# 需要显示画面时自动补 DISPLAY：命令行显式给了 --headless 或 preset 里写了 headless: true 都算
WANTS_DISPLAY=1
if [[ " ${ARGS[*]} " == *" --headless "* ]]; then
  WANTS_DISPLAY=0
elif [ -n "$PRESET" ]; then
  for f in "$CV_ROOT/config/$PRESET.yaml" "config/$PRESET.yaml" "$PRESET.yaml"; do
    if [ -f "$f" ] && grep -qE '^[[:space:]]*headless:[[:space:]]*true' "$f"; then
      WANTS_DISPLAY=0
      break
    fi
  done
fi

if [ "$WANTS_DISPLAY" = 1 ] && [ -z "${DISPLAY:-}" ]; then
  export DISPLAY=:0
  export XAUTHORITY="${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}"
  echo "[run] 未指定 --headless，已自动设置 DISPLAY=:0"
fi

echo "[run] preset : ${PRESET:-（无，使用位置参数）}"
echo "[run] 参数   : ${ARGS[*]:-（全默认）}"
echo "[run] 命令   : build/sop_nx_batch_probe ${ARGS[*]}"
echo

# 程序约定从 build/ 启动（--model-dir、--source 的相对基准）
cd "$CV_ROOT/build"
exec ./sop_nx_batch_probe "${ARGS[@]}"
