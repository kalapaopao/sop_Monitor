"""拆解"为什么两段视频的手部覆盖率差 20 个点"。

起因：用户指出 v2.mp4 与 side2.mp4 **都是侧面拍摄**，v2 只是封面看起来像正面。
所以之前"正面 vs 侧面"的解释不成立，必须换成可量化的分解：

    coverage = P(有 person 框) × P(有手 | 有框)
             + P(无 person 框) × 0        ← top-down 结构上必然为 0

把两个因子分别算出来，就能看清差异来自"YOLO 框不到人"还是"框到了但手没出来"。

用法
----
    .venv\\Scripts\\python.exe scripts/diagnose_hand_gap.py \
        --a web/public/data/local_pose_full_v2.json --a-name v2 \
        --b api_runs/v3_side2_full/rtmw.json --b-name side2
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
    return d["frames"]


def analyze(frames, name):
    n = len(frames)
    bb = [bool((f.get("person") or {}).get("bbox")) for f in frames]
    lh = [bool(f.get("left_hand")) for f in frames]
    rh = [bool(f.get("right_hand")) for f in frames]
    anyh = [a or b for a, b in zip(lh, rh)]
    both = [a and b for a, b in zip(lh, rh)]
    n_bb = sum(bb)
    n_no = n - n_bb
    cov_all = sum(anyh) / n
    cov_given = (sum(1 for i in range(n) if bb[i] and anyh[i]) / n_bb) if n_bb else 0.0
    both_given = (sum(1 for i in range(n) if bb[i] and both[i]) / n_bb) if n_bb else 0.0

    # 手框边长（像素）
    sizes = []
    for f in frames:
        for key in ("left_hand", "right_hand"):
            p = f.get(key) or []
            if len(p) >= 4:
                xs = [q[0] for q in p]
                ys = [q[1] for q in p]
                sizes.append(max(max(xs) - min(xs), max(ys) - min(ys)))
    med = float(np.median(sizes)) if sizes else 0.0

    return {
        "name": name, "n": n,
        "bbox_rate": n_bb / n,
        "cov_all": cov_all,
        "cov_given_bbox": cov_given,
        "both_given_bbox": both_given,
        "no_bbox": n_no,
        "hand_px_median": med,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--b-name", default="B")
    a = ap.parse_args()

    fa = load(a.a)
    fb = load(a.b)
    ra = analyze(fa, a.a_name)
    rb = analyze(fb, a.b_name)

    print(f"{'':<24}{a.a_name:>12}{a.b_name:>12}{'差值':>10}")
    rows = [
        ("tick 数", "n", "{:.0f}", False),
        ("YOLO 有 person 框比例", "bbox_rate", "{:.1%}", True),
        ("至少一只手覆盖率", "cov_all", "{:.1%}", True),
        ("有框时的覆盖率", "cov_given_bbox", "{:.1%}", True),
        ("有框时的双手覆盖率", "both_given_bbox", "{:.1%}", True),
        ("手框边长中位(px)", "hand_px_median", "{:.0f}", False),
    ]
    for label, key, fmt, is_pct in rows:
        va, vb = ra[key], rb[key]
        d = vb - va
        ds = f"{d:+.1%}" if is_pct else f"{d:+.0f}"
        print(f"{label:<24}{fmt.format(va):>12}{fmt.format(vb):>12}{ds:>10}")

    print(f"\n=== 覆盖率差距的来源分解 ===")
    for r in (ra, rb):
        lost = 1.0 - r["bbox_rate"]
        print(f"{r['name']:<8} 总体覆盖 {r['cov_all']:.1%} = "
              f"有框{ r['bbox_rate']:.1%}×{r['cov_given_bbox']:.1%} "
              f"；因无框直接丢失 {lost:.1%}")
    gap = rb["cov_all"] - ra["cov_all"]
    from_bbox = (rb["bbox_rate"] - ra["bbox_rate"]) * ra["cov_given_bbox"]
    resid = gap - from_bbox
    print(f"\n覆盖率差 {gap:+.1%}：其中来自「YOLO 框到人的比例」约 {from_bbox:+.1%}，"
          f"其余 {resid:+.1%} 来自「有框时手能不能出来」")
    print("（二阶交叉项已并入其余项，量级很小）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
