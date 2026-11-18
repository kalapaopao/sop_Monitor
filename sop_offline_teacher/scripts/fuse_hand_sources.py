"""合并两套手部识别：主通道优先，主通道缺失时才用备用通道。

为什么不是二选一
----------------
实测（全片 5992 tick）：

    RTMW（现用 top-down）      至少一只手 86.3% ｜ 双手 66.1% ｜ 抖动 0.0732
    MediaPipe（全图 single-stage）至少一只手 61.5% ｜ 双手  4.2%* ｜ 抖动 0.0419
    *该 4.2% 有实现瑕疵，修正后见下方复测

MediaPipe 覆盖明显更差（因为 palm 检测器内部把输入缩到 192，1080p 里 100px 的手缩完太小），
**但在 RTMW 完全没有 person 框的 717 个 tick（12.0%）上，MediaPipe 能补出 40.2%** ——
那些 tick 的 RTMW 覆盖率是 0，所以这部分是纯增益。

所以正确做法不是替换，而是：
    主通道（RTMW）有手 → 用主通道
    主通道没手      → 用备用通道（MediaPipe）
并在输出里逐帧标注来源，便于核对与回退。

用法
----
    .venv\\Scripts\\python.exe scripts/fuse_hand_sources.py \
        --primary web/public/data/local_pose_full_v2.json --primary-name RTMW \
        --fallback web/public/data/local_pose_mp.json   --fallback-name MediaPipe \
        --output web/public/data/local_pose_fused.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load(path):
    d = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    by = {round(float(f["timestamp"]), 2): f for f in d["frames"]}
    return d, by


def has(f, sides=("left_hand", "right_hand")):
    return any((f or {}).get(s) for s in sides)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", type=Path, required=True)
    ap.add_argument("--primary-name", default="primary")
    ap.add_argument("--fallback", type=Path, required=True)
    ap.add_argument("--fallback-name", default="fallback")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--smooth", type=float, default=0.0,
                    help=">0 时对备用通道的手做时间中值平滑（帧）。备用通道抖动小，一般不用开")
    a = ap.parse_args()

    dp, tp = load(a.primary)
    df, tf = load(a.fallback)
    all_t = sorted(set(tp) | set(tf))

    out_frames = []
    n_prim = n_fb = n_none = 0
    for t in all_t:
        p = tp.get(t)
        f = tf.get(t)
        src = "none"
        frame = None
        if has(p):
            frame = {"timestamp": t,
                     "person": (p or {}).get("person"),
                     "left_hand": p.get("left_hand") or [],
                     "right_hand": p.get("right_hand") or [],
                     "body": p.get("body") or {}}
            src = "primary"
            n_prim += 1
        elif has(f):
            frame = {"timestamp": t, "person": None,
                     "left_hand": f.get("left_hand") or [],
                     "right_hand": f.get("right_hand") or [],
                     "body": {}}
            src = "fallback"
            n_fb += 1
        else:
            frame = {"timestamp": t, "person": (p or {}).get("person"),
                     "left_hand": [], "right_hand": [], "body": (p or {}).get("body") or {}}
            n_none += 1
        frame["source"] = src
        out_frames.append(frame)

    n = len(out_frames)
    one = sum(1 for x in out_frames if x["left_hand"] or x["right_hand"])
    both = sum(1 for x in out_frames if x["left_hand"] and x["right_hand"])

    def cov(doc_by, ts):
        one_ = sum(1 for t in ts if has(doc_by.get(t)))
        both_ = sum(1 for t in ts if (doc_by.get(t) or {}).get("left_hand") and (doc_by.get(t) or {}).get("right_hand"))
        return one_ / len(ts), both_ / len(ts)

    p_one, p_both = cov(tp, all_t)
    f_one, f_both = cov(tf, all_t)

    print(f"tick {n}")
    print(f"\n{'通道':<16}{'至少一只手':>12}{'双手':>10}")
    print(f"{a.primary_name:<16}{p_one:>12.1%}{p_both:>10.1%}")
    print(f"{a.fallback_name:<16}{f_one:>12.1%}{f_both:>10.1%}")
    print(f"{'合并（本脚本）':<16}{one / n:>12.1%}{both / n:>10.1%}")
    print(f"\n来源分布：主通道 {n_prim} ({n_prim / n:.1%}) ｜ "
          f"备用补入 {n_fb} ({n_fb / n:.1%}) ｜ 仍然没有 {n_none} ({n_none / n:.1%})")
    print(f"净增益：至少一只手 {one / n - p_one:+.1%}，双手 {both / n - p_both:+.1%}")

    doc = {
        "video": dp.get("video"), "fps": dp.get("fps"),
        "duration": dp.get("duration"),
        "rate": dp.get("rate", 10),
        "pad": dp.get("pad", 0.0),
        "source": f"fused:{a.primary_name}+{a.fallback_name}",
        "fuse_policy": "主通道有手则用主通道，否则用备用通道",
        "coverage": {"primary_one": round(p_one, 4), "fallback_one": round(f_one, 4),
                     "fused_one": round(one / n, 4), "fused_both": round(both / n, 4)},
        "frames": out_frames,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    print(f"\n写入 {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
