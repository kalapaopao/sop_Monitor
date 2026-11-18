"""把姿态 JSON 瘦身成前端只需要的样子（只留时间戳 + 左右手点）。

前端画「左手/右手识别片段」只用 timestamp 与 left_hand/right_hand 是否有效，
person/body/来源标记都不用。10Hz 全片 605s 的融合文件有 17MB，
裁到需要的时间段并去掉冗余字段后一般只有 1MB 量级，浏览器加载快很多。

用法
----
    .venv\\Scripts\\python.exe scripts/slim_pose_for_web.py \
        --pose web/public/data/local_pose_side2_fused.json \
        --output web/public/data/local_pose_side2_web.json --duration 120
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--duration", type=float, default=0.0, help="只保留前 N 秒，0=全部")
    a = ap.parse_args()

    doc = json.loads(a.pose.read_text(encoding="utf-8-sig"))
    frames = doc.get("frames") or []
    if a.duration > 0:
        frames = [f for f in frames if float(f["timestamp"]) <= a.duration + 1e-6]

    slim = [{"timestamp": f["timestamp"],
             "left_hand": [[round(p[0], 1), round(p[1], 1), round(p[2], 2)] for p in (f.get("left_hand") or [])],
             "right_hand": [[round(p[0], 1), round(p[1], 1), round(p[2], 2)] for p in (f.get("right_hand") or [])]}
            for f in frames]

    out = {"video": doc.get("video"), "fps": doc.get("fps"),
           "duration": doc.get("duration"), "rate": doc.get("rate", 10),
           "source": doc.get("source", ""), "slim": True, "frames": slim}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    before = a.pose.stat().st_size / 1e6
    after = a.output.stat().st_size / 1e6
    print(f"{len(frames)} tick ｜ {before:.1f} MB -> {after:.1f} MB")
    print(f"写入 {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
