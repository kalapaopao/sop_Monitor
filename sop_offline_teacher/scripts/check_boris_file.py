"""校验 .boris 项目文件（BORIS 7.0 JSON）是否结构合法、能否被 BORIS 打开。

用于导出后自检，避免"导出的文件 BORIS 打不开"或"事件错位"。

检查项：
  1. 必需顶层键是否齐全、project_format_version 是否 7.0
  2. behaviors_conf：key 唯一、type 合法、code 非空
  3. observations：file 结构、events 元组长度为 6、时间升序
  4. State 事件是否成对闭合（同一 code 出现偶数次）
  5. 帧号与时间是否自洽（frame ≈ round(t × fps)，容差 1 帧）
  6. 事件是否越界（超过 media_info.length）

用法：
    python scripts/check_boris_file.py <file.boris> [--fps 24.702]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from collections import Counter

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REQUIRED_TOP = ('time_format', 'project_date', 'project_name', 'project_description',
                'project_format_version', 'subjects_conf', 'behaviors_conf', 'observations',
                'behavioral_categories', 'behavioral_categories_config', 'independent_variables',
                'coding_map', 'behaviors_coding_map', 'converters')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path', type=Path)
    ap.add_argument('--fps', type=float, default=None, help='不给则从 media_info 推断')
    a = ap.parse_args()

    problems = []
    try:
        d = json.loads(a.path.read_text(encoding='utf-8-sig'))
    except Exception as exc:
        print(f'[X] 不是合法 JSON：{exc}')
        return 2

    missing = [k for k in REQUIRED_TOP if k not in d]
    if missing:
        problems.append(f'缺顶层键: {missing}')
    if str(d.get('project_format_version')) != '7.0':
        problems.append(f"project_format_version = {d.get('project_format_version')!r}，期望 '7.0'")

    beh = d.get('behaviors_conf') or {}
    codes = []
    keys = []
    for idx, v in beh.items():
        if not v.get('code'):
            problems.append(f'behaviors_conf[{idx}] 的 code 为空')
        if v.get('type') not in ('State event', 'Point event'):
            problems.append(f'behaviors_conf[{idx}] 的 type 非法: {v.get("type")!r}')
        codes.append(v.get('code'))
        keys.append(v.get('key'))
    if len(set(keys)) != len(keys):
        problems.append(f'behaviors_conf 的 key 有重复: {keys}')

    total_events = 0
    for name, ob in (d.get('observations') or {}).items():
        files = (ob.get('file') or {}).get('1') or []
        if not files:
            problems.append(f'观测 {name} 没有关联媒体文件')
        ev = ob.get('events') or []
        total_events += len(ev)
        bad_len = [e for e in ev if not isinstance(e, list) or len(e) != 6]
        if bad_len:
            problems.append(f'观测 {name} 有 {len(bad_len)} 个事件元组不是 6 元')
        times = [e[0] for e in ev if isinstance(e, list) and e]
        if times != sorted(times):
            problems.append(f'观测 {name} 的事件时间不是升序')
        cnt = Counter(e[2] for e in ev if isinstance(e, list) and len(e) == 6)
        odd = {c: n for c, n in cnt.items() if n % 2}
        if odd:
            problems.append(f'观测 {name} 有未闭合的 State 事件: {odd}')
        unknown = {c for c in cnt if c not in codes}
        if unknown:
            problems.append(f'观测 {name} 用到未定义的行为代码: {unknown}')
        fps = a.fps
        if fps is None:
            try:
                vp = files[0]
                fps = (ob.get('media_info') or {}).get('fps', {}).get(vp) or None
            except Exception:
                fps = None
        if fps:
            off = [abs(e[5] - round(e[0] * fps)) for e in ev if isinstance(e, list) and len(e) == 6]
            bad = [i for i, o in enumerate(off) if o > 1]
            if bad:
                problems.append(f'观测 {name} 有 {len(bad)} 个事件帧号与时间不自洽（>1 帧）')
        try:
            dur = float((ob.get('media_info') or {}).get('length', {}).get(files[0], 0)) if files else 0
        except Exception:
            dur = 0
        if dur:
            over = [e for e in ev if isinstance(e, list) and len(e) == 6 and e[0] > dur + 0.5]
            if over:
                problems.append(f'观测 {name} 有 {len(over)} 个事件超出媒体时长 {dur:.2f}s')

    n_seg = total_events // 2
    print(f'文件: {a.path.name}')
    print(f'项目: {d.get("project_name")} | 版本 {d.get("project_format_version")} | '
          f'行为 {len(beh)} 类 | 观测 {len(d.get("observations") or {})} 个 | 事件 {total_events} 个（≈{n_seg} 段）')
    if problems:
        print(f'\n[!] 发现 {len(problems)} 个问题：')
        for p in problems:
            print('   -', p)
        return 1
    print('\n[OK] 结构合法：键齐全、事件 6 元、时间升序、State 成对闭合、帧号自洽')
    return 0


if __name__ == '__main__':
    sys.exit(main())
