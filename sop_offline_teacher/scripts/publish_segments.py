"""把一次教师跑出来的结果「发布」到前端，让网页可以直接切换查看。

解决的问题：每次跑完大模型，结果都在 api_runs/ 里，网页读的是固定文件，
必须手工拷贝才能看。本脚本一步完成：

    api_runs/<运行目录>/segments_with_step.json 或 segments_proposal.json
        ↓  转成前端契约（steps 数组 + coverage），时间从 v2 轴换到片段时间轴
    web/public/data/runs/<运行名>.json            ← 每次运行各存一份
    web/public/data/runs/index.json               ← 运行清单（网页据此出现下拉框）
    web/public/data/segments.json                 ← 同时设为"当前默认"

用法：
    # 发布已有的映射结果
    python scripts/publish_segments.py --from api_runs/mapped_clean --name mapped_clean

    # 发布自由分段结果
    python scripts/publish_segments.py --from api_runs/free_clean --name free_clean

    # 只看会写什么，不落盘
    python scripts/publish_segments.py --from api_runs/free_clean --name x --dry-run
"""
from __future__ import annotations
import argparse, json, shutil, sys
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DATA_DIR = Path('web/public/data')
RUNS_DIR = DATA_DIR / 'runs'
BACKUP_DIR = Path('api_runs/_backup')
DEFAULT_DURATION = 119.99          # v2_5m_7m_pose.mp4 的有效时长
DEFAULT_CLIP_OFFSET = 300.0        # clip_t = v2_t - 300

# step_id → 调色板索引（与前端 PALETTE 对应）
COLOR_BY_STEP = {'S1': 0, 'S2': 1, 'S3': 2, 'S4': 3, 'S5': 4, 'S6': 5,
                 'idle': 7, 'unknown': 6}


def load_segments(src: Path):
    """优先读映射后的结果（带 step_id），否则读原始自由分段。"""
    cand = [src / 'segments_with_step.json', src / 'segments_proposal.json']
    if src.is_file():
        cand = [src]
    for p in cand:
        if p.exists():
            d = json.loads(p.read_text(encoding='utf-8-sig'))
            return p, d
    raise SystemExit(f'{src} 里没有 segments_with_step.json / segments_proposal.json')


def enrich_from_windows(segs, src: Path, clip_offset: float = 0.0):
    """旧版自由分段的 proposal 里没有 left_hand/right_hand，从 windows_freeform.json 补。

    做法：取每段中心时间，找覆盖它的窗口，用该窗口的左手/右手描述补上。
    新版权流水线（L1-L3）已经直接写在 proposal 里，这里只会补空值。
    """
    wf = src / 'windows_freeform.json'
    if not wf.exists():
        return 0
    wins = (json.loads(wf.read_text(encoding='utf-8-sig')).get('windows') or [])
    if not wins:
        return 0
    filled = 0
    for s in segs:
        # 段是 v2 时间轴，窗口是片段时间轴，查表前要换算
        center = (float(s['start']) + float(s['end'])) / 2 - clip_offset
        cand = [w for w in wins if w['start'] <= center <= w['end']] or \
               [min(wins, key=lambda w: abs((w['start'] + w['end']) / 2 - center))]
        w = cand[0]
        segs_in_w = w.get('segments') or []
        first = segs_in_w[0] if segs_in_w else w
        for k_src, k_dst in (('left_hand', 'left_hand'), ('right_hand', 'right_hand'),
                             ('state_change', 'state_change')):
            if not s.get(k_dst) and first.get(k_src):
                s[k_dst] = first[k_src]
                filled += 1
    return filled


def to_frontend(doc, src_file: Path, clip_offset: float, duration: float, name: str,
                video_name: str = 'v2_5m_7m_pose.mp4', pose_name: str = '', station: str = ''):
    segs = doc.get('segments') or doc.get('steps') or []   # 兼容 enriched 文件的 steps 键
    steps, dropped = [], 0
    for s in segs:
        st = float(s['start']) - clip_offset
        en = float(s['end']) - clip_offset
        st, en = max(0.0, st), min(duration, en)
        if en - st <= 0.05:
            dropped += 1
            continue
        step_id = s.get('step_id') or 'unknown'
        steps.append({
            'idx': len(steps),
            'step_id': step_id,
            'name': s.get('step_name') or s.get('name') or s.get('action') or step_id,
            'start': round(st, 3),
            'end': round(en, 3),
            'kind': 'step',
            'source': f'teacher:{name}',
            'color': COLOR_BY_STEP.get(step_id, 6),
            'need_review': bool(s.get('need_review')),
            'action': s.get('action') or '',
            'evidence': s.get('evidence_i_used') or '',
            'state_change': s.get('state_change') or 'none',
            'left_hand': s.get('left_hand') or '',
            'right_hand': s.get('right_hand') or '',
            'confidence': s.get('confidence'),
        })
    steps.sort(key=lambda x: x['start'])
    for i, s in enumerate(steps):
        s['idx'] = i
    coverage = [steps[0]['start'], steps[-1]['end']] if steps else [0.0, 0.0]
    out = {
        'version': 1,
        'video': video_name,
        'duration': duration,
        'source': f"teacher-run:{name}",
        'note': (f"来自 {src_file.as_posix()}；共 {len(steps)} 段"
                 f"{'，其中 ' + str(sum(1 for x in steps if x['need_review'])) + ' 段需人工复核' if any(x['need_review'] for x in steps) else ''}"
                 f"{'；' + str(dropped) + ' 段完全落在片段外已丢弃' if dropped else ''}"),
        'coverage': [round(coverage[0], 3), round(coverage[1], 3)],
        'steps': steps,
    }
    if pose_name:
        out['pose'] = pose_name
    if station:
        out['station'] = station
    return out


def publish(src: Path, name: str, clip_offset=DEFAULT_CLIP_OFFSET,
            duration=DEFAULT_DURATION, dry_run=False,
            video_name='v2_5m_7m_pose.mp4', pose_name='', station=''):
    src_file, doc = load_segments(src)
    # 补全左右手/状态：先看本目录，再看该结果记录的来源目录
    # （映射后的 segments_with_step.json 里 source_proposal 指回 free_clean，那里才有 windows_freeform.json）
    cand_dirs = [src if src.is_dir() else src.parent]
    sp = doc.get('source_proposal')
    if sp:
        cand_dirs.append(Path(sp).parent)
    filled = 0
    for d in cand_dirs:
        filled = enrich_from_windows(doc.get('segments') or [], d, clip_offset)
        if filled:
            print(f'从 {d}/windows_freeform.json 补全了 {filled} 个字段（左右手/状态）')
            break
    out = to_frontend(doc, src_file, clip_offset, duration, name, video_name, pose_name, station)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_path = RUNS_DIR / f'{name}.json'

    index_path = RUNS_DIR / 'index.json'
    index = json.loads(index_path.read_text(encoding='utf-8')) if index_path.exists() else {'version': 1, 'runs': []}
    entry = {
        'name': name,
        'file': f'{name}.json',
        'source_file': src_file.as_posix(),
        'steps': len(out['steps']),
        'need_review': sum(1 for s in out['steps'] if s['need_review']),
        'coverage': out['coverage'],
        'published_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    index['runs'] = [r for r in index['runs'] if r['name'] != name] + [entry]
    index['runs'].sort(key=lambda r: r['published_at'])
    index['active'] = name

    print(f'来源 {src_file}')
    print(f'段数 {len(out["steps"])} ｜ 需复核 {entry["need_review"]} ｜ 覆盖 {out["coverage"][0]}–{out["coverage"][1]}s')
    for s in out['steps'][:12]:
        flag = '!' if s['need_review'] else ' '
        print(f'  {flag} {s["start"]:6.2f}-{s["end"]:6.2f}s  {s["step_id"]:<7} {s["name"][:22]}')
    if len(out['steps']) > 12:
        print(f'  … 另有 {len(out["steps"]) - 12} 段')

    if dry_run:
        print('dry-run：未写文件')
        return out, entry

    run_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding='utf-8')

    # 设为当前默认，并先备份旧文件（可回退）
    active_path = DATA_DIR / 'segments.json'
    if active_path.exists():
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        shutil.copy2(active_path, BACKUP_DIR / f'segments_{stamp}.json')
    active_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')

    (DATA_DIR / 'segments_meta.json').write_text(json.dumps(
        {'active': name, 'runs': len(index['runs']), 'published_at': entry['published_at']},
        ensure_ascii=False, indent=1), encoding='utf-8')

    print(f'\n写入 {run_path}')
    print(f'写入 {index_path}（共 {len(index["runs"])} 个运行）')
    print(f'写入 {active_path}')
    print('刷新网页即可在「结果来源」里切换。')
    return out, entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from', dest='src', type=Path, required=True)
    ap.add_argument('--name', required=True, help='运行名（会作为下拉项显示）')
    ap.add_argument('--clip-offset', type=float, default=DEFAULT_CLIP_OFFSET)
    ap.add_argument('--duration', type=float, default=DEFAULT_DURATION)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--video-name', default='v2_5m_7m_pose.mp4',
                    help='web/public/video 下的文件名，写进 run JSON 的 video 字段')
    ap.add_argument('--pose', default='', help='data 下的姿态文件名，例如 local_pose_side2_web.json')
    ap.add_argument('--station', default='', help='机位 id（configs/stations.json 里的 station），决定用哪套 ROI')
    a = ap.parse_args()
    publish(a.src, a.name, a.clip_offset, a.duration, a.dry_run,
            a.video_name, a.pose, a.station)
    return 0


if __name__ == '__main__':
    sys.exit(main())
