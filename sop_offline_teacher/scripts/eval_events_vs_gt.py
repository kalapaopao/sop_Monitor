"""把本地事件候选段与人工 GT 对齐评估（诚实口径，不用帧级准确率糊弄）。

口径说明
--------
· GT 来自 BORIS 人工标注（38 段 / 6 类行为 / 7 个周期），是**行为契约**，不是"标准作业"。
· 候选段的类别是**粗类**（PICK/ASSEMBLE/FASTEN/OTHER/IDLE），与本项目的六步表是
  多对一关系（S1/S2→PICK，S3/S4→ASSEMBLE，S5→FASTEN）。因此这里只评：
    ① 段级 IoU（同类里 GT 段被覆盖得怎么样）
    ② 边界命中率（±tol 秒，不要求类名一致 —— 这才是分工真正看的东西）
    ③ 紧固段专项（最长、最确定的一步）
    ④ GT 步骤时长被非空闲候选覆盖的比例
· 所有数字都标明来源文件，可复算。

用法
----
    .venv\\Scripts\\python.exe scripts\\eval_events_vs_gt.py \
        --proposal api_runs/events_local/segments_proposal.json \
        --gt web/public/data/gt_segments.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GT_TO_COARSE = {
    "Remove_eyepiece_assembly": "PICK",
    "Remove_the_upper_cover_of_the_eyepiece": "PICK",
    "Assemble_the_mirror_base_cover": "ASSEMBLE",
    "Assemble_the_mirror_base_sealing_plug": "ASSEMBLE",
    "Fasten_mirror_base": "FASTEN",
    "Fasten_tripod_strap": "OTHER",
}
STEP_NAME = {
    "Remove_eyepiece_assembly": "S1 取镜座组件",
    "Remove_the_upper_cover_of_the_eyepiece": "S2 取镜座上盖",
    "Assemble_the_mirror_base_cover": "S3 组装上盖",
    "Assemble_the_mirror_base_sealing_plug": "S4 装密封塞",
    "Fasten_mirror_base": "S5 紧固",
    "Fasten_tripod_strap": "S6 系扎带",
}


def iou(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposal", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--tol", type=float, default=2.0, help="边界命中容差（秒）")
    ap.add_argument("--range", nargs=2, type=float, default=None, metavar=("T0", "T1"),
                    help="只评某个时间区间（如 300 420）")
    ap.add_argument("--json-out", type=Path, default=None)
    a = ap.parse_args()

    prop = json.loads(a.proposal.read_text(encoding="utf-8-sig"))
    gt = json.loads(a.gt.read_text(encoding="utf-8-sig"))

    segs = prop["segments"]
    gtsegs = [s for s in gt["segments"] if not s.get("nested_short")]
    if a.range:
        t0, t1 = a.range
        segs = [s for s in segs if s["end"] > t0 and s["start"] < t1]
        gtsegs = [s for s in gtsegs if s["end"] > t0 and s["start"] < t1]

    print(f"候选段 {len(segs)} ｜ GT 段 {len(gtsegs)}（已剔除 nested_short）"
          f"{f' ｜ 区间 {a.range[0]}-{a.range[1]}s' if a.range else ''}")
    print(f"来源：{a.proposal} vs {a.gt}")

    # ---------- ① 段级 IoU（同类） ----------
    print("\n=== ① 段级 IoU（只和同类比；阈值 IoU≥0.3 记为命中）===")
    print(f"{'GT 步骤':<18}{'段数':>5}{'命中':>5}{'IoU中位':>9}{'时长合计':>10}{'被覆盖':>9}")
    total_hit = total_n = 0
    for beh, coarse in GT_TO_COARSE.items():
        gs = [s for s in gtsegs if s["behavior"] == beh]
        if not gs:
            continue
        ps = [s for s in segs if s["coarse_class"] == coarse]
        hits, ious = 0, []
        for g in gs:
            best = max((iou(g["start"], g["end"], p["start"], p["end"]) for p in ps), default=0.0)
            ious.append(best)
            if best >= 0.3:
                hits += 1
        dur = sum(g["end"] - g["start"] for g in gs)
        covered = sum(
            sum(max(0.0, min(g["end"], p["end"]) - max(g["start"], p["start"]))
                for p in ps if p["coarse_class"] != "IDLE")
            for g in gs)
        covered = min(covered, dur)
        total_hit += hits; total_n += len(gs)
        print(f"{STEP_NAME[beh]:<18}{len(gs):>5}{hits:>5}{sorted(ious)[len(ious) // 2]:>9.2f}"
              f"{dur:>10.1f}{covered / dur if dur else 0:>9.1%}")
    print(f"{'合计':<18}{total_n:>5}{total_hit:>5}{'':>9}{'':>10}"
          f"{total_hit / total_n if total_n else 0:>9.1%}  ← 命中率")

    # ---------- ② 边界命中率（不看类名） ----------
    print(f"\n=== ② 边界命中率（±{a.tol}s，不要求类名一致）===")
    gt_b = sorted({round(s["start"], 1) for s in gtsegs} | {round(s["end"], 1) for s in gtsegs})
    pr_b = sorted({round(s["start"], 1) for s in segs if s["coarse_class"] != "IDLE"}
                  | {round(s["end"], 1) for s in segs if s["coarse_class"] != "IDLE"})
    hit_g = [g for g in gt_b if any(abs(g - p) <= a.tol for p in pr_b)]
    hit_p = [p for p in pr_b if any(abs(g - p) <= a.tol for g in gt_b)]
    errs = [min(abs(g - p) for p in pr_b) for g in gt_b] if pr_b else []
    print(f"GT 边界 {len(gt_b)} 条，命中 {len(hit_g)} → 召回 {len(hit_g) / len(gt_b):.1%}")
    print(f"候选边界 {len(pr_b)} 条，命中 {len(hit_p)} → 精确 {len(hit_p) / len(pr_b) if pr_b else 0:.1%}")
    if errs:
        errs.sort()
        print(f"GT 边界到最近候选的时间误差：中位 {errs[len(errs) // 2]:.2f}s ｜ 90% {errs[int(len(errs) * 0.9)]:.2f}s")

    # ---------- ③ 紧固专项 ----------
    print("\n=== ③ 紧固（S5）专项：GT vs 候选 ===")
    print(f"{'GT 区间':<20}{'GT 时长':>8}{'候选区间':>20}{'候选时长':>9}{'IoU':>7}{'起止误差':>14}")
    fasten_gt = [s for s in gtsegs if s["behavior"] == "Fasten_mirror_base"]
    fasten_pr = [s for s in segs if s["coarse_class"] == "FASTEN"]
    used = set()
    rows = []
    for g in fasten_gt:
        best, bi = 0.0, None
        for i, p in enumerate(fasten_pr):
            v = iou(g["start"], g["end"], p["start"], p["end"])
            if v > best:
                best, bi = v, i
        if bi is None:
            print(f"{g['start']:.1f}-{g['end']:.1f}{'':<8}{g['end'] - g['start']:>8.1f}"
                  f"{'（无）':>20}{'':>9}{0.0:>7.2f}")
            rows.append({"gt": [g["start"], g["end"]], "pred": None, "iou": 0.0})
            continue
        p = fasten_pr[bi]; used.add(bi)
        print(f"{g['start']:.1f}-{g['end']:.1f}{'':<8}{g['end'] - g['start']:>8.1f}"
              f"{p['start']:>9.1f}-{p['end']:.1f}{p['end'] - p['start']:>9.1f}"
              f"{best:>7.2f}{p['start'] - g['start']:>7.1f}/{(p['end'] - g['end']):>6.1f}")
        rows.append({"gt": [g["start"], g["end"]], "pred": [p["start"], p["end"]], "iou": round(best, 3)})
    extra = [fasten_pr[i] for i in range(len(fasten_pr)) if i not in used]
    if extra:
        print(f"\n多出来的紧固候选 {len(extra)} 段（GT 里没有对应）：")
        for p in extra:
            print(f"  {p['start']:.1f}-{p['end']:.1f}  {p['end'] - p['start']:.1f}s")
    print(f"\n紧固段数：GT {len(fasten_gt)} ｜ 候选 {len(fasten_pr)}")

    # ---------- ④ 覆盖 ----------
    print("\n=== ④ GT 步骤时长被「非空闲」候选覆盖的比例 ===")
    gt_dur = sum(s["end"] - s["start"] for s in gtsegs)
    cov = 0.0
    for g in gtsegs:
        for p in segs:
            if p["coarse_class"] == "IDLE":
                continue
            cov += max(0.0, min(g["end"], p["end"]) - max(g["start"], p["start"]))
    print(f"GT 总时长 {gt_dur:.1f}s ｜ 被覆盖 {min(cov, gt_dur):.1f}s → {min(cov, gt_dur) / gt_dur:.1%}")

    if a.json_out:
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        a.json_out.write_text(json.dumps({
            "proposal": str(a.proposal), "gt": str(a.gt), "tol": a.tol,
            "range": a.range,
            "fasten": rows,
            "boundary": {"gt": len(gt_b), "recall": len(hit_g) / len(gt_b) if gt_b else 0,
                         "pred": len(pr_b), "precision": len(hit_p) / len(pr_b) if pr_b else 0},
            "coverage": min(cov, gt_dur) / gt_dur if gt_dur else 0,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n写入 {a.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
