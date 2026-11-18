"""对比两套手部识别的稳定性：覆盖 / 抖动 / 左右手一致性 / 闪烁。

为什么要单独比"稳定性"
----------------------
覆盖率只说明"有没有检出来"，但用户要的是**稳定**：
帧间点不乱跳、左右手不来回换、同一个状态能持续。
所以这里给四个量：

  coverage   至少一只 / 双手 的比例（按 GT 步骤分组）
  jitter     相邻两帧同一关键点的位移（按手框边长归一化），中位数越低越稳
  side_flip  与参考模型相比，左右手判定不一致的帧占比
  flicker    手"出现→消失→出现"的次数 / 分钟，越低越稳

用法
----
    .venv\\Scripts\\python.exe scripts/compare_hand_models.py \
        --a web/public/data/local_pose_full_v2.json --a-name RTMW \
        --b web/public/data/local_pose_mp.json --b-name MediaPipe \
        --gt web/public/data/gt_segments.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BEH = {
    "Remove_eyepiece_assembly": "S1 取镜座",
    "Remove_the_upper_cover_of_the_eyepiece": "S2 取上盖",
    "Assemble_the_mirror_base_cover": "S3 装上盖",
    "Assemble_the_mirror_base_sealing_plug": "S4 装密封塞",
    "Fasten_mirror_base": "S5 紧固",
    "Fasten_tripod_strap": "S6 系扎带",
}


def load(path):
    d = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    by_t = {}
    for f in d["frames"]:
        by_t[round(float(f["timestamp"]), 2)] = f
    return d, by_t


def cents(pts):
    if not pts:
        return None
    v = [p for p in pts if len(p) >= 3 and p[2] > 0.18] or pts
    if not v:
        return None
    return float(np.mean([p[0] for p in v])), float(np.mean([p[1] for p in v]))


def extent(pts):
    if not pts:
        return 0.0
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def stats(doc, by_t, times):
    n = len(times)
    both = one = 0
    jit = []
    flips = 0
    trans = 0
    prev_present = None
    prev_pts = {"left": None, "right": None}
    prev_t = None
    for t in times:
        f = by_t.get(t)
        lh = (f or {}).get("left_hand") or []
        rh = (f or {}).get("right_hand") or []
        if lh and rh:
            both += 1
        if lh or rh:
            one += 1
        present = bool(lh or rh)
        if prev_present is not None and present != prev_present:
            trans += 1
        prev_present = present
        if f:
            for side, pts in (("left", lh), ("right", rh)):
                pp = prev_pts[side]
                if pts and pp and len(pts) == len(pp):
                    e = max(1.0, extent(pts))
                    d = np.mean([abs(pts[i][0] - pp[i][0]) + abs(pts[i][1] - pp[i][1])
                                 for i in range(len(pts))]) / 2.0
                    jit.append(d / e)
                prev_pts[side] = pts
    span_min = (times[-1] - times[0]) / 60.0 if len(times) > 1 else 1.0
    return {
        "coverage_one": one / n if n else 0,
        "coverage_both": both / n if n else 0,
        "jitter": float(np.median(jit)) if jit else None,
        "transitions_per_min": trans / span_min if span_min > 0 else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--b-name", default="B")
    ap.add_argument("--gt", type=Path, default=None)
    a = ap.parse_args()

    da, ta = load(a.a)
    db, tb = load(a.b)
    common = sorted(set(ta) & set(tb))
    print(f"{a.a_name}: {len(ta)} tick ｜ {a.b_name}: {len(tb)} tick ｜ 共同 tick {len(common)}")

    sa = stats(da, ta, common)
    sb = stats(db, tb, common)

    # 左右手一致性
    agree = tot = 0
    for t in common:
        fa, fb = ta[t], tb[t]
        for side in ("left_hand", "right_hand"):
            ca, cb = cents(fa.get(side) or []), cents(fb.get(side) or [])
            if ca is None and cb is None:
                continue
            tot += 1
            if (ca is None) != (cb is None):
                continue
            agree += int(((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5 < 200)
    sa["side_agree"] = agree / tot if tot else None
    sb["side_agree"] = None

    print(f"\n=== 总体稳定性（同一批 {len(common)} 个 tick）===")
    print(f"{'指标':<22}{a.a_name:>14}{a.b_name:>14}")
    print(f"{'至少一只手覆盖':<22}{sa['coverage_one']:>13.1%}{sb['coverage_one']:>14.1%}")
    print(f"{'双手覆盖':<22}{sa['coverage_both']:>13.1%}{sb['coverage_both']:>14.1%}")
    ja = f"{sa['jitter']:.4f}" if sa["jitter"] is not None else "n/a"
    jb = f"{sb['jitter']:.4f}" if sb["jitter"] is not None else "n/a"
    print(f"{'帧间抖动(归一化,越低越稳)':<22}{ja:>14}{jb:>14}")
    print(f"{'出现/消失次数每分钟':<22}{sa['transitions_per_min']:>13.1f}{sb['transitions_per_min']:>14.1f}")
    if sa.get("side_agree") is not None:
        print(f"{'与对方左右手一致率':<22}{sa['side_agree']:>13.1%}{'（基准）':>14}")

    # ---- 关键补充：在 A 没有 person 框的帧上，B 能不能补上？----
    # 这些帧 A 结构上必然没有手（覆盖 = 0），所以 B 只要能检出就是净增益。
    if any("person" in (ta[t] or {}) for t in common[:50]):
        nb = [t for t in common if not ((ta[t] or {}).get("person") or {}).get("bbox")]
        print(f"\n=== 在 {a.a_name} 没有 person 框的 {len(nb)} 个 tick 上 ===")
        if nb:
            ca = sum(1 for t in nb if (ta[t].get("left_hand") or ta[t].get("right_hand"))) / len(nb)
            cb = sum(1 for t in nb if (tb[t].get("left_hand") or tb[t].get("right_hand"))) / len(nb)
            print(f"  {a.a_name:<12} 至少一只手覆盖 {ca:>6.1%}（结构上基本为 0）")
            print(f"  {b_name if (b_name := a.b_name) else 'B':<12} 至少一只手覆盖 {cb:>6.1%}"
                  f"  ← 这部分是纯增益")
            print(f"  占全片比例 {len(nb) / len(common):.1%}")

    # 按 GT 步骤分组
    if a.gt and a.gt.exists():
        gt = json.loads(a.gt.read_text(encoding="utf-8-sig"))
        spans = []
        for s in gt["segments"]:
            if s.get("nested_short") or s["behavior"] not in BEH:
                continue
            spans.append((s["start"], s["end"], BEH[s["behavior"]]))
        print(f"\n=== 按 GT 步骤分组：至少一只手覆盖 ===")
        print(f"{'步骤':<14}{'帧数':>6}{a.a_name:>12}{a.b_name:>12}{'差':>8}")
        for name in [v for v in BEH.values()]:
            ts = [t for (s0, s1, nm) in spans if nm == name for t in common if s0 <= t < s1]
            if not ts:
                continue
            ca = sum(1 for t in ts if (ta[t].get("left_hand") or ta[t].get("right_hand"))) / len(ts)
            cb = sum(1 for t in ts if (tb[t].get("left_hand") or tb[t].get("right_hand"))) / len(ts)
            print(f"{name:<14}{len(ts):>6}{ca:>12.1%}{cb:>12.1%}{cb - ca:>+8.1%}")

    out = Path("api_runs/hand_compare.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"a": str(a.a), "b": str(a.b), "n": len(common),
                               "a_stats": sa, "b_stats": sb}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
