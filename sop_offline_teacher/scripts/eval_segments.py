"""段级评估 + 状态机后处理（Codex 第 4 步剩余项 / 我们清单里的 A）。

输入：train_pose_baseline.py --save-pred 导出的逐帧预测
做两件事：
  1) **状态机后处理**：把帧级预测过一遍四条规则（确认切换 / 最短段长 / 同标签合并 / 低置信屏蔽）
     —— 复用参考视频第 3 层的思路
  2) **段级指标**：与人工 GT 段比对，报告
     · 段级 precision / recall / F1（IoU>=0.5 匹配）
     · IoU 均值与中位
     · 边界误差（预测段边界到最近 GT 边界的距离中位）
     · 状态机处理前后的对比

用法：
    python scripts/eval_segments.py --pred api_runs/pred_extra.json \
        --gt web/public/data/gt_segments.json --clip-offset 0
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

CLASSES = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'idle_auxiliary']
B2S = {'Remove_eyepiece_assembly': 'S1',
       'Remove_the_upper_cover_of_the_eyepiece': 'S2',
       'Assemble_the_mirror_base_cover': 'S3',
       'Assemble_the_mirror_base_sealing_plug': 'S4',
       'Fasten_mirror_base': 'S5',
       'Fasten_tripod_strap': 'S6'}


def run_lengths(ts, labels):
    """把逐帧标签压成 (start, end, label) 段。"""
    segs = []
    if len(ts) == 0:
        return segs
    cur = labels[0]
    st = ts[0]
    for i in range(1, len(ts)):
        if labels[i] != cur:
            segs.append([float(st), float(ts[i]), int(cur)])
            cur, st = labels[i], ts[i]
    segs.append([float(st), float(ts[-1]) + 0.1, int(cur)])
    return segs


def state_machine(segs, confirm=8, min_dur=1.0, merge_gap=1.0):
    """四条规则：确认切换（等价于"短于 confirm 段的抖动直接吞掉"）+ 最短段长 + 同标签合并。"""
    # 规则 1：过短的抖动段丢弃（并入前一段）—— 10 Hz 下 confirm 帧 = confirm*0.1 秒
    sd = confirm * 0.1
    out = []
    for s in segs:
        if s[1] - s[0] < sd and out:
            out[-1][1] = s[1]        # 吞掉
        else:
            out.append(list(s))
    # 规则 3：同标签相邻合并
    merged = []
    for s in out:
        if merged and merged[-1][2] == s[2]:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    # 规则 2：最短段长（仍过短的并入较近邻居）
    final = []
    for i, s in enumerate(merged):
        if s[1] - s[0] >= min_dur or len(merged) == 1:
            final.append(s)
            continue
        if final and (i + 1 >= len(merged) or s[0] - final[-1][1] <= merged[i + 1][0] - s[1]):
            final[-1][1] = s[1]
        elif i + 1 < len(merged):
            merged[i + 1][0] = s[0]
    return final


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def eval_segments(pred_segs, gt_segs, off):
    """段级 P/R/F1（IoU>=0.5 且类别一致）+ IoU 均值 + 边界误差。"""
    gt = [[s['start'] - off, s['end'] - off, B2S.get(s['behavior'], '?')] for s in gt_segs]
    gt = [g for g in gt if g[1] > 0]
    used = set()
    tp = 0
    ious = []
    for p in pred_segs:
        best, bi = None, -1
        for i, g in enumerate(gt):
            if i in used:
                continue
            v = iou(p, g)
            if best is None or v > best:
                best, bi = v, i
        ious.append(best if best else 0.0)
        if best is not None and best >= 0.5 and CLASSES[p[2]] == gt[bi][2]:
            tp += 1
            used.add(bi)
    prec = tp / max(1, len(pred_segs))
    rec = tp / max(1, len(gt))
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    # 边界误差：每个预测边界到最近 GT 边界
    ge = sorted({round(x, 2) for g in gt for x in (g[0], g[1])})
    pe = sorted({round(x, 2) for p in pred_segs for x in (p[0], p[1])})
    errs = [min(abs(e - g) for g in ge) for e in pe] if ge and pe else []
    return {
        'pred_segments': len(pred_segs), 'gt_segments': len(gt),
        'tp': tp, 'precision': prec, 'recall': rec, 'f1': f1,
        'iou_mean': float(np.mean(ious)) if ious else 0.0,
        'iou_median': float(np.median(ious)) if ious else 0.0,
        'boundary_err_median': float(np.median(errs)) if errs else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', type=Path, required=True)
    ap.add_argument('--gt', type=Path, default=Path('web/public/data/gt_segments.json'))
    ap.add_argument('--clip-offset', type=float, default=0.0)
    ap.add_argument('--confirm', type=int, default=8)
    ap.add_argument('--min-dur', type=float, default=1.0)
    ap.add_argument('--merge-gap', type=float, default=1.0)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()

    d = json.loads(a.pred.read_text(encoding='utf-8-sig'))
    frames = d['frames']
    ts = np.array([f['t'] for f in frames])
    pred = np.array([f['pred'] for f in frames])
    true = np.array([f['true'] for f in frames])
    print(f'预测帧 {len(ts)} ｜ 帧级准确率（原样）{(pred == true).mean():.1%}')

    raw_segs = run_lengths(ts, pred)
    sm_segs = state_machine(raw_segs, a.confirm, a.min_dur, a.merge_gap)
    gt = json.loads(a.gt.read_text(encoding='utf-8-sig'))['segments']

    print(f'\n段数：原始 {len(raw_segs)} → 状态机后 {len(sm_segs)}（confirm={a.confirm} 帧={a.confirm*0.1:.1f}s, '
          f'min_dur={a.min_dur}s）')
    r1 = eval_segments(raw_segs, gt, a.clip_offset)
    r2 = eval_segments(sm_segs, gt, a.clip_offset)

    def show(tag, r):
        print(f'\n[{tag}] 段级：预测 {r["pred_segments"]} / GT {r["gt_segments"]}')
        print(f'  precision {r["precision"]:.2f} ｜ recall {r["recall"]:.2f} ｜ F1 {r["f1"]:.2f}')
        print(f'  IoU 均值 {r["iou_mean"]:.2f} ｜ 中位 {r["iou_median"]:.2f} ｜ '
              f'边界误差中位 {"—" if r["boundary_err_median"] is None else f"{r['boundary_err_median']:.2f}s"}')

    show('状态机前', r1)
    show('状态机后', r2)
    print(f'\n状态机带来的变化：F1 {r2["f1"] - r1["f1"]:+.2f} ｜ '
          f'IoU均值 {r2["iou_mean"] - r1["iou_mean"]:+.2f} ｜ '
          f'段数 {len(raw_segs)} → {len(sm_segs)}')

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({'before': r1, 'after': r2,
                                     'params': {'confirm': a.confirm, 'min_dur': a.min_dur,
                                                'merge_gap': a.merge_gap}},
                                    ensure_ascii=False, indent=1), encoding='utf-8')
        print('写出', a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
