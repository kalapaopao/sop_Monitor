"""把教师自由产出的候选段与人工 GT 做**边界对齐评分**（不要求类别名一致）。

为什么不用"分类准确率"评自由分段：
  自由分段的产物是"时间区间 + 描述"，描述措辞不可能与 GT 类名逐字相同。
  真正要回答的是"人还需要改多少"，所以评三件事：
    1. 边界命中率：GT 的每个边界（start/end）附近 ±tol 内是否有候选边界
    2. 段级 IoU：每个 GT 段与候选段的最大 IoU（衡量"大致找到了这段动作"）
    3. 段数偏差：模型段数 vs GT 段数（多了说明切碎，少了说明合并过度）

用法：
    python scripts/eval_boundaries.py --gt web/public/data/gt_segments.json \
        --proposal api_runs/proposal_clip/segments_proposal.json --tol 1.0
    # 只评某个时间范围（例如只评 300–420s 的片段）
    python scripts/eval_boundaries.py ... --range 300 420
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def iou(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def boundary_hit(gt_edges, prop_edges, tol):
    """返回 (命中数, 命中明细)。gt_edges/prop_edges 为已排序的时间点列表。"""
    used = set()
    hits = []
    for g in gt_edges:
        best = None
        for i, p in enumerate(prop_edges):
            if i in used:
                continue
            d = abs(p - g)
            if d <= tol and (best is None or d < best[1]):
                best = (i, d)
        if best is not None:
            used.add(best[0])
            hits.append((g, prop_edges[best[0]], best[1]))
        else:
            hits.append((g, None, None))
    return hits, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', type=Path, required=True)
    ap.add_argument('--proposal', type=Path, required=True)
    ap.add_argument('--tol', type=float, default=1.0, help='边界命中容差（秒）')
    ap.add_argument('--range', nargs=2, type=float, default=None, help='只评这个时间范围')
    a = ap.parse_args()

    gt = json.loads(a.gt.read_text(encoding='utf-8-sig'))
    segs = gt['segments']
    if a.range:
        r0, r1 = a.range
        segs = [s for s in segs if s['end'] > r0 and s['start'] < r1]
    prop = json.loads(a.proposal.read_text(encoding='utf-8-sig'))
    psegs = prop.get('segments') or prop.get('steps') or []
    if a.range:
        r0, r1 = a.range
        psegs = [s for s in psegs if s['end'] > r0 and s['start'] < r1]

    print(f'GT 段 {len(segs)} 个 ｜ 候选段 {len(psegs)} 个 ｜ 容差 ±{a.tol}s'
          + (f' ｜ 范围 {a.range[0]:.0f}–{a.range[1]:.0f}s' if a.range else ''))
    if not segs or not psegs:
        sys.exit('有一边为空，无法评分')

    gt_edges = sorted([x for s in segs for x in (s['start'], s['end'])])
    pr_edges = sorted([x for s in psegs for x in (s['start'], s['end'])])
    hits, used = boundary_hit(gt_edges, pr_edges, a.tol)
    hit_n = sum(1 for _g, p, _d in hits if p is not None)
    recall = hit_n / len(gt_edges)
    precision = len(used) / len(pr_edges) if pr_edges else 0.0
    errs = [d for _g, _p, d in hits if d is not None]
    errs.sort()

    print(f'\n边界命中：GT 边界 {len(gt_edges)} 条，命中 {hit_n} 条 → 召回 {recall:.1%}')
    print(f'          候选边界 {len(pr_edges)} 条，被匹配 {len(used)} 条 → 精确 {precision:.1%}')
    if errs:
        print(f'          命中边界的时间误差：中位 {errs[len(errs) // 2]:.2f}s  '
              f'最大 {errs[-1]:.2f}s')

    print('\n段级 IoU（每个 GT 段找最佳候选段）：')
    ious = []
    for s in segs:
        best = max(psegs, key=lambda p: iou(s['start'], s['end'], p['start'], p['end']))
        v = iou(s['start'], s['end'], best['start'], best['end'])
        ious.append(v)
        flag = 'OK ' if v >= 0.5 else ('~  ' if v >= 0.3 else 'X  ')
        name = s.get('name_cn') or s.get('behavior') or s.get('label')
        print(f'  {flag}GT {s["start"]:7.1f}-{s["end"]:7.1f} ({s["dur"]:5.1f}s) {name:<12}'
              f' → 候选 {best["start"]:7.1f}-{best["end"]:7.1f}  IoU={v:.2f}')
    ious.sort()
    hit50 = sum(1 for v in ious if v >= 0.5)
    print(f'\nIoU>=0.5 的 GT 段：{hit50}/{len(ious)} = {hit50 / len(ious):.1%}'
          f' ｜ IoU 中位 {ious[len(ious) // 2]:.2f}')
    print(f'段数：GT {len(segs)} vs 候选 {len(psegs)}'
          f'（候选偏多说明切碎，偏少说明合并过度）')

    print('\n结论怎么看：')
    print('  · 边界召回高 + IoU 高 → 候选可直接给人工微调（改名字、并/拆少数段）')
    print('  · 边界召回低 → 模型没找到动作切换点，需要更密 stride 或更强证据')
    print('  · IoU 低但边界召回高 → 找到了切换点，但段划分粒度与人工不一致（可接受，人工重排）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
