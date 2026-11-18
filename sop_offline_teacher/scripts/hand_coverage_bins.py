"""按时间分箱统计手部检出率，定位"哪一段根本看不到手"。

用法
----
    .venv\\Scripts\\python.exe scripts/hand_coverage_bins.py \
        --pose api_runs/v3_side2_full/rtmw.json --bin 30 --limit 600
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SPARK = " .:-=+*#%@"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", type=Path, required=True)
    ap.add_argument("--bin", type=float, default=30.0)
    ap.add_argument("--limit", type=float, default=0.0)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    d = json.loads(a.pose.read_text(encoding="utf-8-sig"))
    fr = d["frames"]
    ts = np.array([f["timestamp"] for f in fr])
    has_bb = np.array([bool((f.get("person") or {}).get("bbox")) for f in fr])
    l = np.array([bool(f.get("left_hand")) for f in fr])
    r = np.array([bool(f.get("right_hand")) for f in fr])
    anyh = l | r
    both = l & r
    tmax = ts.max()
    if a.limit > 0:
        tmax = min(tmax, a.limit)

    edges = np.arange(0, tmax + a.bin, a.bin)
    print(f"{a.label or a.pose.name} ｜ {len(fr)} tick ｜ 0~{tmax:.0f}s ｜ 每 {a.bin:.0f}s 一箱")
    print(f"\n{'区间':>14}{'有框':>8}{'至少一只':>10}{'双手':>8}   {'覆盖率条':<24}")
    worst = []
    for i in range(len(edges) - 1):
        t0, t1 = edges[i], edges[i + 1]
        if t0 >= tmax:
            break
        m = (ts >= t0) & (ts < t1)
        n = int(m.sum())
        if n == 0:
            continue
        bb = has_bb[m].mean()
        one = anyh[m].mean()
        tw = both[m].mean()
        bar = "".join(SPARK[int(round(one * (len(SPARK) - 1)))] for _ in range(int(round(one * 24))))
        worst.append((one, t0, t1))
        print(f"{t0:6.0f}-{t1:5.0f}s{bb:>8.0%}{one:>10.0%}{tw:>8.0%}   {bar}")

    worst.sort()
    print(f"\n最差的 5 段（至少一只手覆盖率从低到高）：")
    for one, t0, t1 in worst[:5]:
        print(f"  {t0:.0f}-{t1:.0f}s  仅 {one:.0%}")
    print(f"\n整体：有框 {has_bb.mean():.1%} ｜ 至少一只手 {anyh.mean():.1%} ｜ 双手 {both.mean():.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
