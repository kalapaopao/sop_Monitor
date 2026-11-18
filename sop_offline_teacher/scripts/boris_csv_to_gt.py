"""把 BORIS 导出的 CSV 转成统一的人工真值 gt_segments.json。

规则（已由负责人确认）：
  1. 短段（<= --short-sec，默认 0.8s）**算真实动作**，保留为训练正样本，只打 `short=true` 标记；
  2. 嵌套时**长段优先**：完全被更长段包含的段打 `nested_short=true`，其时间归属长段；
  3. idle 窗口只取"距最近事件 > --idle-margin（默认 2s）"的中段，避免边界污染（见 build_teacher_windows.py）。

另外记录与测试片段的换算关系：实测 `v2_5m_7m_h264.mp4` 就是 `v2.mp4` 的 300–420s，
即 clip_t = v2_t - 300（±0.5s）。落在该窗口的段会同时给出 clip 相对时间。

用法：
    python scripts/boris_csv_to_gt.py --csv vedio/event/event.csv \
        --video vedio/v2.mp4 --clip-offset 300 --clip-duration 120 \
        --out web/public/data/gt_segments.json
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

NAME_CN = {
    'Remove_eyepiece_assembly': '取镜座组件',
    'Remove_the_upper_cover_of_the_eyepiece': '取镜座上盖',
    'Assemble_the_mirror_base_cover': '组装镜座上盖',
    'Assemble_the_mirror_base_sealing_plug': '组装镜座密封塞',
    'Fasten_mirror_base': '紧固镜子镜座',
    'Fasten_tripod_strap': '系上镜座扎带',
}


def read_csv(path: Path):
    rows = []
    with path.open(encoding='utf-8-sig', newline='') as h:
        for r in csv.DictReader(h):
            b = (r.get('Behavior') or '').strip()
            if not b:
                continue
            rows.append(((r.get('Behavior type') or '').strip().upper(),
                         float(r.get('Time') or 0), b,
                         (r.get('Media duration (s)') or '').strip(),
                         (r.get('FPS') or '').strip(),
                         (r.get('Media file name') or '').strip()))
    return rows


def pair(rows):
    segs, open_at = [], {}
    for ty, t, b, *_ in rows:
        if ty == 'START':
            open_at[b] = t
        elif ty in ('STOP', 'VIDEO_END') and b in open_at:
            segs.append({'behavior': b, 'start': open_at.pop(b), 'end': t})
    segs.sort(key=lambda s: s['start'])
    for i, s in enumerate(segs):
        s['idx'] = i
        s['dur'] = round(s['end'] - s['start'], 3)
        s['name_cn'] = NAME_CN.get(s['behavior'], s['behavior'])
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path('web/public/data/gt_segments.json'))
    ap.add_argument('--video', default='vedio/v2.mp4')
    ap.add_argument('--short-sec', type=float, default=0.8, help='短段阈值（仍算真实动作）')
    ap.add_argument('--clip-offset', type=float, default=300.0, help='测试片段在源视频中的起点')
    ap.add_argument('--clip-duration', type=float, default=120.0)
    a = ap.parse_args()

    rows = read_csv(a.csv)
    segs = pair(rows)
    if not segs:
        raise SystemExit('CSV 里没有可配对的事件')

    duration = float(rows[0][3]) if rows[0][3] else 0.0
    fps = float(rows[0][4]) if rows[0][4] else 0.0

    # 嵌套：被更长段完全包含
    for s in segs:
        s['nested_short'] = any(
            o['idx'] != s['idx'] and o['dur'] > s['dur'] + 1e-6
            and o['start'] <= s['start'] + 1e-6 and o['end'] >= s['end'] - 1e-6
            for o in segs)
        s['short'] = s['dur'] <= a.short_sec

    # 周期：以模板末步（紧固）的结束为周期闭合点，一个周期 = 取镜座→取上盖→装上盖→装密封塞→紧固
    # 用紧固做锚点更稳：取镜座偶尔在一个周期里出现两次，会把周期切碎
    anchor = 'Fasten_mirror_base'
    ends = [s['end'] for s in segs if s['behavior'] == anchor]
    cycles = []
    prev = 0.0
    for i, en in enumerate(ends):
        cycles.append({'cycle_id': i, 'anchor': anchor, 'start': round(prev, 3), 'end': round(en, 3),
                       'complete': i + 1 < len(ends)})
        prev = en
    if prev < duration - 1.0:
        cycles.append({'cycle_id': len(cycles), 'anchor': anchor, 'start': round(prev, 3),
                       'end': round(duration, 3), 'complete': False})
    for s in segs:
        s['cycle_id'] = next((c['cycle_id'] for c in cycles
                              if c['start'] <= s['start'] < c['end']), None)
        # clip 相对时间
        cs, ce = s['start'] - a.clip_offset, s['end'] - a.clip_offset
        s['in_clip'] = (ce > 0) and (cs < a.clip_duration)
        s['clip_start'] = round(max(0.0, cs), 3) if s['in_clip'] else None
        s['clip_end'] = round(min(a.clip_duration, ce), 3) if s['in_clip'] else None

    per = {}
    for s in segs:
        d = per.setdefault(s['behavior'], {'name_cn': s['name_cn'], 'count': 0, 'durs': [],
                                           'short': 0, 'in_clip': 0})
        d['count'] += 1
        d['durs'].append(s['dur'])
        d['short'] += int(s['short'])
        d['in_clip'] += int(s['in_clip'])
    for d in per.values():
        v = sorted(d.pop('durs'))
        d['median_dur'] = round(v[len(v) // 2], 2)
        d['min_dur'] = round(v[0], 2)
        d['max_dur'] = round(v[-1], 2)

    out = {
        'source': str(a.csv).replace('\\', '/'),
        'video': a.video,
        'duration': duration,
        'fps_effective': round(fps, 4),
        'clip_offset': a.clip_offset,
        'clip_duration': a.clip_duration,
        'rules': {'short_sec': a.short_sec, 'short_is_real': True,
                  'nested': 'long segment wins, short one keeps nested_short flag'},
        'behaviors': per,
        'cycles': cycles,
        'segments': segs,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')

    n_clip = sum(1 for s in segs if s['in_clip'])
    print(json.dumps({'out': str(a.out), 'segments': len(segs), 'cycles': len(cycles),
                      'short': sum(1 for s in segs if s['short']),
                      'nested_short': sum(1 for s in segs if s['nested_short']),
                      'in_clip': n_clip}, ensure_ascii=False))
    print(f'{"行为":<28}{"段数":>4}{"中位s":>7}{"最短":>6}{"最长":>7}{"短段":>5}{"在片段内":>7}')
    for b, d in per.items():
        print(f'{d["name_cn"]:<28}{d["count"]:>4}{d["median_dur"]:>7.2f}{d["min_dur"]:>6.2f}'
              f'{d["max_dur"]:>7.2f}{d["short"]:>5}{d["in_clip"]:>7}')


if __name__ == '__main__':
    main()
