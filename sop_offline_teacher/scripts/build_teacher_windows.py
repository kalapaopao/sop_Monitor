"""把人工真值 gt_segments.json 变成教师模型的评估窗口。

窗口规则（对应已确认的标注规则）：
  · 正样本：取段中部，窗口 2 s（段时长 <2 s 时用整段，并标记 kind=short）
  · 嵌套短段（nested_short=true）：时间归属长段，**不生成独立窗口**，单独统计
  · idle 负样本：只取"距最近事件 > --idle-margin（默认 2s）"的中段，窗口 2 s
  · 每个窗口给出 3 帧的时间点（均匀分布），供教师模型一次窗口多帧输入

用法：
    # 全片窗口（38 段）
    python scripts/build_teacher_windows.py --gt web/public/data/gt_segments.json \
        --out web/public/data/teacher_windows.json

    # 只取已有姿态/ROI 证据的那 120 秒（片段 0–120s，即 v2 的 300–420s）
    python scripts/build_teacher_windows.py --gt web/public/data/gt_segments.json \
        --only-clip --out web/public/data/teacher_windows_clip.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def frames_of(start, end, n=3):
    """在窗口内均匀取 n 帧（避开两端 10%）。"""
    if n <= 1:
        return [round((start + end) / 2, 3)]
    lo = start + (end - start) * 0.1
    hi = end - (end - start) * 0.1
    step = (hi - lo) / (n - 1)
    return [round(lo + step * i, 3) for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--win', type=float, default=2.0, help='正样本窗口长度（秒）')
    ap.add_argument('--idle-margin', type=float, default=2.0, help='idle 窗口距最近事件的留白（秒）')
    ap.add_argument('--frames', type=int, default=3, help='每窗送几帧')
    ap.add_argument('--only-clip', action='store_true', help='只保留落在测试片段内的窗口')
    ap.add_argument('--mode', choices=['center', 'state', 'inside'], default='center',
                    help='center=取段中部固定窗口（首轮用，适合持续性动作）；'
                         'state=向段外各扩 0.6s（**已证明会把邻居动作框进来，勿用**）；'
                         'inside=严格在段内取"首/中/末"三帧（适合状态变化型动作，靠首末对比判状态）')
    a = ap.parse_args()

    gt = json.loads(a.gt.read_text(encoding='utf-8-sig'))
    segs = gt['segments']
    clip_off, clip_dur = gt.get('clip_offset', 0.0), gt.get('clip_duration', 0.0)
    clip_end = clip_off + clip_dur

    def in_clip(s, e):
        return (not a.only_clip) or (e > clip_off and s < clip_end)

    windows = []
    skipped = []

    # ---- 正样本 ----
    for s in segs:
        if s.get('nested_short'):
            skipped.append({'reason': 'nested_short', 'behavior': s['behavior'],
                            'start': s['start'], 'end': s['end'], 'dur': s['dur']})
            continue
        dur = s['dur']
        if a.mode == 'inside':
            # 严格在段内：用"段首 / 段中 / 段末"三帧做前后对比。
            # 不能往段外扩——实测那样会把邻居动作的帧当成这个动作的帧喂进去（标签污染）。
            inset = 0.15 if dur >= 0.6 else max(0.02, dur * 0.2)
            ws, we = s['start'], s['end']
            mid = (s['start'] + s['end']) / 2
            fr = [round(s['start'] + inset, 3), round(mid, 3), round(s['end'] - inset, 3)]
            frames = sorted(set(fr))
            kind = 'inside'
        elif a.mode == 'state':
            # 已弃用：会跨到邻居动作
            pre = 0.6 if dur >= 1.0 else 1.2
            ws = max(0.0, s['start'] - pre)
            we = min(gt['duration'], s['end'] + pre)
            mid = (s['start'] + s['end']) / 2
            fr = [round(ws + min(0.2, (s['start'] - ws) / 2), 3), round(mid, 3),
                  round(we - min(0.2, (we - s['end']) / 2), 3)]
            frames = sorted(set(fr))
            kind = 'state'
        elif dur >= a.win:
            c = (s['start'] + s['end']) / 2
            ws, we = c - a.win / 2, c + a.win / 2
            kind = 'normal'
            frames = frames_of(ws, we, a.frames)
        else:
            ws, we = s['start'], s['end']
            kind = 'short'
            frames = frames_of(ws, we, a.frames)
        if not in_clip(ws, we):
            continue
        windows.append({
            'idx': len(windows), 'kind': kind, 'label': s['behavior'], 'name_cn': s['name_cn'],
            'start': round(ws, 3), 'end': round(we, 3), 'win_dur': round(we - ws, 3),
            'frames': frames,
            'seg_start': round(s['start'], 3), 'seg_end': round(s['end'], 3),
            'src_seg': s['idx'], 'cycle_id': s.get('cycle_id'),
            'short': bool(s.get('short')), 'clip': bool(s.get('in_clip')),
        })

    # ---- idle 负样本 ----
    occ = sorted((s['start'], s['end']) for s in segs)
    merged = []
    for s, e in occ:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    gaps = []
    prev_end = 0.0
    for s, e in merged:
        gaps.append((prev_end, s))
        prev_end = e
    gaps.append((prev_end, gt['duration']))

    for gs, ge in gaps:
        usable_s, usable_e = gs + a.idle_margin, ge - a.idle_margin
        if usable_e - usable_s < a.win:
            continue
        c = (usable_s + usable_e) / 2
        ws, we = c - a.win / 2, c + a.win / 2
        if not in_clip(ws, we):
            continue
        windows.append({
            'idx': len(windows), 'kind': 'idle', 'label': 'idle', 'name_cn': '空闲/无事件',
            'start': round(ws, 3), 'end': round(we, 3), 'win_dur': round(we - ws, 3),
            'frames': frames_of(ws, we, a.frames),
            'src_seg': None, 'cycle_id': None, 'short': False, 'clip': False,
        })

    windows.sort(key=lambda w: w['start'])
    for i, w in enumerate(windows):
        w['idx'] = i
        # clip 相对时间
        w['clip_start'] = round(w['start'] - clip_off, 3)
        w['clip_end'] = round(w['end'] - clip_off, 3)
        w['clip_frames'] = [round(t - clip_off, 3) for t in w['frames']]

    out = {
        'source_gt': str(a.gt).replace('\\', '/'),
        'only_clip': a.only_clip,
        'win': a.win, 'idle_margin': a.idle_margin, 'frames_per_window': a.frames,
        'clip_offset': clip_off, 'clip_duration': clip_dur,
        'counts': {
            'total': len(windows),
            'by_label': {},
            'by_kind': {},
            'skipped': len(skipped),
        },
        'windows': windows,
        'skipped': skipped,
    }
    for w in windows:
        out['counts']['by_label'][w['label']] = out['counts']['by_label'].get(w['label'], 0) + 1
        out['counts']['by_kind'][w['kind']] = out['counts']['by_kind'].get(w['kind'], 0) + 1

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')

    print(f'窗口 {len(windows)} 个（跳过 {len(skipped)} 段：nested_short）')
    print('按标签：')
    for k, v in sorted(out['counts']['by_label'].items(), key=lambda x: -x[1]):
        print(f'   {k:<46} {v:>3}')
    print('按类型：', out['counts']['by_kind'])
    print(f'每窗 {a.frames} 帧 → 共 {len(windows) * a.frames} 张图')
    if a.only_clip:
        print(f'（仅片段内：v2 {clip_off:.0f}–{clip_end:.0f}s）')
    print('写出：', a.out)


if __name__ == '__main__':
    main()
