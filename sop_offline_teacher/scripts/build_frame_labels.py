"""把人工 GT 段转成「帧级标签」，并冻结标签规则（Codex 要求的第 2 步）。

标签规则（冻结，不再改）：
  1. **长段优先**：GT 里存在嵌套短段（如"取上盖 0.76s"完全嵌在"取镜座 4.68s"里）。
     帧级标签只能有一个值 → 取**最长的那段**。被覆盖的短段单独记录到 `nested_short`，不进训练。
  2. **三类标签**：
        S1..S6           真正的 SOP 步骤（来自 GT）
        idle_auxiliary   没有 GT 标注的帧（流程外/备料/等待/辅助）
        invalid          姿态质量不足的帧（**训练时屏蔽，不当作类别**）
  3. **invalid 判定**（可调）：
        该帧两手都不可见，或左右手可见点 < min_points
     理由：如果让模型把"检测失败"学成动作类别，它会把遮挡误判成"空闲"。

产物：
    <out>.json        每帧: t / label / source(GT段idx或gap) / quality
    统计打印：各类帧数、嵌套短段清单、invalid 比例

用法：
    python scripts/build_frame_labels.py --pose web/public/data/local_pose_full.json \
        --gt web/public/data/gt_segments.json --clip-offset 0 --out web/public/data/frame_labels_full.json
    # 片段（120s）用法：--clip-offset 300
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

B2S = {'Remove_eyepiece_assembly': 'S1',
       'Remove_the_upper_cover_of_the_eyepiece': 'S2',
       'Assemble_the_mirror_base_cover': 'S3',
       'Assemble_the_mirror_base_sealing_plug': 'S4',
       'Fasten_mirror_base': 'S5',
       'Fasten_tripod_strap': 'S6'}
VALID_PTS = 0.18          # 关键点可信阈值（与 run_local_pose 一致）
CLASSES = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'idle_auxiliary']
MASKED = 'invalid'


def hand_ok(pts, min_points=6):
    if not pts:
        return 0
    return sum(1 for p in pts if p[2] > VALID_PTS)


def resolve_gt(segs):
    """长段优先：返回 (主段列表, 被覆盖的嵌套短段列表)。"""
    kept, nested = [], []
    for s in segs:
        covered = any(
            o is not s and (o['end'] - o['start']) > (s['end'] - s['start'])
            and o['start'] <= s['start'] + 1e-6 and o['end'] >= s['end'] - 1e-6
            for o in segs)
        (nested if covered else kept).append(s)
    return kept, nested


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pose', type=Path, required=True)
    ap.add_argument('--gt', type=Path, default=Path('web/public/data/gt_segments.json'))
    ap.add_argument('--clip-offset', type=float, default=300.0,
                    help='姿态文件时间轴相对 v2 的偏移：片段用 300，全片用 0')
    ap.add_argument('--min-points', type=int, default=6, help='每只手至少几个可信点才算有效')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()

    pose = json.loads(a.pose.read_text(encoding='utf-8-sig'))
    frames = pose['frames']
    gt = json.loads(a.gt.read_text(encoding='utf-8-sig'))['segments']

    kept, nested = resolve_gt(gt)
    print(f'GT 段 {len(gt)} → 主段 {len(kept)}，被覆盖的嵌套短段 {len(nested)}')
    for s in nested:
        print(f'  [nested_short] {s["start"]:.2f}-{s["end"]:.2f}s '
              f'{B2S.get(s["behavior"], "?")} ({s["dur"]:.2f}s) 已被更长段覆盖，不进帧级训练')

    labels, stats = [], {}
    invalid_n = 0
    for f in frames:
        t = float(f['timestamp'])
        v2 = t + a.clip_offset
        # 质量判定
        l_ok = hand_ok(f.get('left_hand'), a.min_points)
        r_ok = hand_ok(f.get('right_hand'), a.min_points)
        quality = 'ok' if (l_ok >= a.min_points or r_ok >= a.min_points) else 'poor'
        if quality == 'poor':
            lab, src = MASKED, 'quality'
            invalid_n += 1
        else:
            hit = next((s for s in kept if s['start'] - 1e-6 <= v2 <= s['end'] + 1e-6), None)
            if hit:
                lab, src = B2S.get(hit['behavior'], 'unknown'), f'gt:{hit["idx"]}'
            else:
                lab, src = 'idle_auxiliary', 'gap'
        labels.append({'t': round(t, 2), 'label': lab, 'source': src,
                       'quality': quality, 'l_ok': l_ok, 'r_ok': r_ok})
        stats[lab] = stats.get(lab, 0) + 1

    total = len(labels)
    print(f'\n帧数 {total} ｜ invalid {invalid_n} = {invalid_n / max(1, total):.1%}（训练时屏蔽）')
    print('各类帧数：')
    for k in CLASSES + [MASKED]:
        if stats.get(k):
            print(f'  {k:<16} {stats[k]:>6}  {stats[k] / total:6.1%}')

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        'pose': str(a.pose), 'gt': str(a.gt), 'clip_offset': a.clip_offset,
        'rules': {
            'nested': 'long_segment_wins；嵌套短段记入 nested_short 且不进帧级训练',
            'classes': CLASSES, 'masked': MASKED,
            'invalid_rule': f'左右手可信点均 < {a.min_points}',
            'valid_pt_threshold': VALID_PTS,
        },
        'nested_short': [{'start': s['start'], 'end': s['end'], 'behavior': s['behavior'],
                          'dur': s['dur']} for s in nested],
        'class_counts': stats, 'frames': labels}, ensure_ascii=False, indent=1), encoding='utf-8')
    print('\n写出', a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
