"""段级、按类的 P/R/F1 评估（这才是业务真正关心的口径，不是帧级准确率）。

约定
----
· GT = 人工 BORIS 标注（6 类行为 → step_id，剔除 nested_short）
· 预测 = 任何带 step_id 的 segments_proposal.json
· 匹配 = 同类、IoU ≥ --iou 即算命中（一对一贪心，按 IoU 从大到小）
· 同时给出：GT 步骤被"同类且 IoU 达标"命中的比例（这是最直白的可用性指标）

用法
----
    .venv\\Scripts\\python.exe scripts\\eval_segments_per_class.py \
        --proposal api_runs/segs_pred/segments_proposal.json \
        --gt web/public/data/gt_segments.json --range 300 420
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BEH_TO_STEP = {
    "Remove_eyepiece_assembly": "S1",
    "Remove_the_upper_cover_of_the_eyepiece": "S2",
    "Assemble_the_mirror_base_cover": "S3",
    "Assemble_the_mirror_base_sealing_plug": "S4",
    "Fasten_mirror_base": "S5",
    "Fasten_tripod_strap": "S6",
}
STEP_CN = {"S1": "取镜座", "S2": "取上盖", "S3": "装上盖", "S4": "装密封塞",
           "S5": "紧固", "S6": "系扎带", "idle": "空闲"}


def iou(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposal", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--tol", type=float, default=1.5)
    ap.add_argument("--range", nargs=2, type=float, default=None, metavar=("T0", "T1"))
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    prop = json.loads(a.proposal.read_text(encoding="utf-8-sig"))
    segs = [s for s in prop["segments"] if s.get("step_id")]
    gtraw = json.loads(a.gt.read_text(encoding="utf-8-sig"))

    gts = []
    for s in gtraw["segments"]:
        if s.get("nested_short"):
            continue
        sid = BEH_TO_STEP.get(s["behavior"])
        if not sid:
            continue
        gts.append({"start": s["start"], "end": s["end"], "step_id": sid})
    gts.sort(key=lambda x: x["start"])

    if a.range:
        t0, t1 = a.range
        segs = [s for s in segs if s["end"] > t0 and s["start"] < t1]
        gts = [g for g in gts if g["end"] > t0 and g["start"] < t1]

    print(f"{('[' + a.label + '] ') if a.label else ''}候选 {len(segs)} 段 ｜ GT {len(gts)} 段"
          f"{f' ｜ 区间 {a.range[0]:.0f}-{a.range[1]:.0f}s' if a.range else ''}")

    # 同类配对，按 IoU 从大到小贪心
    pairs = []
    for gi, g in enumerate(gts):
        for pi, p in enumerate(segs):
            if p["step_id"] != g["step_id"]:
                continue
            v = iou(g["start"], g["end"], p["start"], p["end"])
            if v >= a.iou:
                pairs.append((v, gi, pi))
    pairs.sort(reverse=True)
    used_g, used_p, matched = set(), set(), []
    for v, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi); used_p.add(pi)
        matched.append((v, gi, pi))

    print(f"\n=== 按类（同类 + IoU≥{a.iou}）===")
    print(f"{'类':<10}{'GT':>4}{'命中':>5}{'召回':>7}{'精确':>7}{'F1':>6}")
    tot_g = tot_m = 0
    per_class = {}
    for sid in ("S1", "S2", "S3", "S4", "S5", "S6"):
        gc = [i for i, g in enumerate(gts) if g["step_id"] == sid]
        pc = [i for i, p in enumerate(segs) if p["step_id"] == sid]
        if not gc and not pc:
            continue
        m = sum(1 for _v, gi, pi in matched if gts[gi]["step_id"] == sid)
        rec = m / len(gc) if gc else 0.0
        pre = m / len(pc) if pc else 0.0
        f1 = 2 * pre * rec / (pre + rec) if (pre + rec) else 0.0
        per_class[sid] = {"gt": len(gc), "pred": len(pc), "hit": m,
                          "recall": round(rec, 3), "precision": round(pre, 3), "f1": round(f1, 3)}
        tot_g += len(gc); tot_m += m
        print(f"{sid + ' ' + STEP_CN[sid]:<10}{len(gc):>4}{m:>5}{rec:>7.1%}{pre:>7.1%}{f1:>6.2f}")

    print(f"\n**GT 步骤被同类命中：{tot_m}/{tot_g} = {tot_m / tot_g:.1%}**（IoU≥{a.iou}）")
    durs = [v for v, _g, _p in matched]
    if durs:
        durs.sort()
        print(f"命中段 IoU：中位 {durs[len(durs) // 2]:.2f} ｜ 最小 {durs[0]:.2f} ｜ 最大 {durs[-1]:.2f}")

    # 边界（相邻 GT 段的接缝处）
    gt_b = sorted({round(g["start"], 1) for g in gts} | {round(g["end"], 1) for g in gts})
    pr_b = sorted({round(p["start"], 1) for p in segs} | {round(p["end"], 1) for p in segs})
    hg = [g for g in gt_b if any(abs(g - p) <= a.tol for p in pr_b)]
    hp = [p for p in pr_b if any(abs(g - p) <= a.tol for g in gt_b)]
    errs = sorted(min(abs(g - p) for p in pr_b) for g in gt_b) if pr_b else [0]
    print(f"\n边界（±{a.tol}s）：GT {len(gt_b)} 条 → 召回 {len(hg) / len(gt_b):.1%} ｜ "
          f"候选 {len(pr_b)} 条 → 精确 {len(hp) / len(pr_b) if pr_b else 0:.1%} ｜ "
          f"中位误差 {errs[len(errs) // 2]:.2f}s")

    if a.json_out:
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        a.json_out.write_text(json.dumps({
            "proposal": str(a.proposal), "iou_thr": a.iou, "tol": a.tol, "range": a.range,
            "per_class": per_class,
            "gt_steps_hit": {"hit": tot_m, "total": tot_g, "rate": round(tot_m / tot_g, 4)},
            "matched_iou_median": round(durs[len(durs) // 2], 3) if durs else None,
            "boundary": {"recall": round(len(hg) / len(gt_b), 4) if gt_b else None,
                         "precision": round(len(hp) / len(pr_b), 4) if pr_b else None,
                         "median_err": round(errs[len(errs) // 2], 2)},
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"写入 {a.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
