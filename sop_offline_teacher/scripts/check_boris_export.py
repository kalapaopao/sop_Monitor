"""校验 BORIS 导出的 CSV（State 事件），给出训练前必须修掉的问题。

检查项：
  1. START / STOP 是否成对（BORIS 用 VIDEO_END 代替越过片尾的 STOP）
  2. 同类事件是否重叠、是否倒序
  3. 过短片段（默认 <1s，通常是误触或重复按键）
  4. 时间是否越界（超出媒体时长）
  5. 片段之间的空档分布（用于决定"合并还是拆段"的阈值）
  6. 行为定义粒度提示：片段中位时长异常长时，提醒确认标签是否过粗

用法：
    python scripts/check_boris_export.py project/data/evnets/*.csv
    python scripts/check_boris_export.py <dir> --min-dur 1.0
"""
from __future__ import annotations
import argparse, csv, glob, os, sys
from collections import defaultdict

# Windows 控制台默认是 GBK，遇到 [注意] 之类的字符会直接抛异常
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def read_events(path):
    """返回 (行为, 类型, 时间) 列表，忽略空行与无行为行。"""
    out = []
    with open(path, encoding='utf-8-sig', newline='') as h:
        for r in csv.DictReader(h):
            b = (r.get('Behavior') or '').strip()
            if not b:
                continue
            ty = (r.get('Behavior type') or '').strip().upper()
            try:
                t = float(r.get('Time') or 0)
            except ValueError:
                t = 0.0
            out.append((b, ty, t, (r.get('Comment') or '').strip()))
    return out


def media_duration(path):
    with open(path, encoding='utf-8-sig', newline='') as h:
        for r in csv.DictReader(h):
            d = (r.get('Media duration (s)') or '').strip()
            if d:
                try:
                    return float(d)
                except ValueError:
                    return None
    return None


def pair_segments(events):
    """按行为配对 START/STOP(或 VIDEO_END)。返回 (segs, problems)。"""
    segs, problems = [], []
    open_at = {}
    for beh, ty, t, note in events:
        if ty == 'START':
            if beh in open_at:
                problems.append(f'{beh}: 在 {t:.2f}s 重复 START（上一次 {open_at[beh]:.2f}s 未结束）')
            open_at[beh] = t
        elif ty in ('STOP', 'VIDEO_END'):
            if beh not in open_at:
                problems.append(f'{beh}: {t:.2f}s 出现 {ty} 但前面没有 START')
                continue
            start = open_at.pop(beh)
            if t < start:
                problems.append(f'{beh}: {start:.2f}s→{t:.2f}s 起止倒序')
                continue
            segs.append({'behavior': beh, 'start': start, 'end': t,
                         'dur': t - start, 'end_type': ty, 'note': note})
    for beh, start in open_at.items():
        problems.append(f'{beh}: {start:.2f}s 的 START 没有对应 STOP（到片尾也未闭合）')
    segs.sort(key=lambda s: s['start'])
    return segs, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+', help='CSV 文件或目录')
    ap.add_argument('--min-dur', type=float, default=1.0, help='过短片段阈值(秒)')
    a = ap.parse_args()

    files = []
    for p in a.paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, '*.csv')))
        else:
            files += sorted(glob.glob(p))
    if not files:
        sys.exit('没找到 CSV')

    all_segs, all_gaps, total_problems = [], [], 0
    print(f'{"文件":<34}{"时长s":>8}{"片段":>5}{"中位s":>8}{"过短":>5}{"重叠":>5}  问题')
    print('-' * 96)
    for f in files:
        ev = read_events(f)
        dur = media_duration(f)
        segs, problems = pair_segments(ev)

        # 同类重叠检查
        overlaps = 0
        by_beh = defaultdict(list)
        for s in segs:
            by_beh[s['behavior']].append(s)
        for beh, lst in by_beh.items():
            lst.sort(key=lambda s: s['start'])
            for x, y in zip(lst, lst[1:]):
                if y['start'] < x['end']:
                    overlaps += 1
                    problems.append(f'{beh}: {x["start"]:.1f}-{x["end"]:.1f} 与 {y["start"]:.1f}-{y["end"]:.1f} 重叠')

        if dur:
            for s in segs:
                if s['end'] > dur + 0.5:
                    problems.append(f'{s["behavior"]}: 结束 {s["end"]:.1f}s 超出媒体时长 {dur:.1f}s')

        short = [s for s in segs if s['dur'] < a.min_dur]
        for s in short:
            problems.append(f'{s["behavior"]}: {s["start"]:.2f}-{s["end"]:.2f}s 仅 {s["dur"]:.2f}s（疑似误触）')

        durs = sorted(s['dur'] for s in segs)
        med = durs[len(durs) // 2] if durs else 0
        name = os.path.basename(f)
        print(f'{name[:33]:<34}{(dur or 0):>8.1f}{len(segs):>5}{med:>8.1f}{len(short):>5}{overlaps:>5}  {"OK" if not problems else str(len(problems)) + " 条"}')

        all_segs += segs
        for x, y in zip(segs, segs[1:]):
            all_gaps.append(y['start'] - x['end'])
        total_problems += len(problems)
        for p in problems:
            print(f'    ! {p}')

    print('-' * 96)
    if all_segs:
        durs = sorted(s['dur'] for s in all_segs)
        gaps = sorted(g for g in all_gaps if g >= 0)
        print(f'片段总数 {len(all_segs)}，时长 中位 {durs[len(durs)//2]:.1f}s '
              f'(p10 {durs[len(durs)//10]:.1f} / p90 {durs[len(durs)*9//10]:.1f} / 最短 {durs[0]:.2f} / 最长 {durs[-1]:.1f})')
        if gaps:
            print(f'相邻片段间隔 中位 {gaps[len(gaps)//2]:.1f}s '
                  f'(最短 {gaps[0]:.1f} / 最长 {gaps[-1]:.1f})；'
                  f'<2s 占 {sum(1 for g in gaps if g < 2)/len(gaps)*100:.0f}%')
            print(f'→ 合并/拆段参考：间隔 <2s 基本是同一动作内的停顿；>5s 多为换了动作或换了工件')
        med = durs[len(durs)//2]
        if med > 10:
            print(f'[注意] 片段中位时长达 {med:.1f}s，比"单个动作"常见的 2~5s 长很多。'
                  f'请确认行为定义是否已把"取工具+对准+作业+放下"整段算作一个事件；'
                  f'若是，则它与 SOP 步骤不是同一粒度，训练前要重新划分。')
    print(f'\n共发现 {total_problems} 条需要处理的问题')
    return 0 if total_problems == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
