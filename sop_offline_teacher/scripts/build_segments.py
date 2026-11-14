"""把动作语义结果统一成前端用的 segments.json（单一时间基准 0–视频时长）。

现在支持两种输入：
  ablation  读 web/public/data/ablation_14_{A,B,C}.json（只覆盖前 30 秒的三组对照）
  teacher   读教师模型输出的段列表（未来 --sop 跑出来的 segments）

设计目的：前端只认一种 schema，以后教师模型跑全片，只要重新生成这个文件，
页面代码一行都不用改。

用法：
    python scripts/build_segments.py --source ablation --run A \
        --video web/public/video/v2_5m_7m_pose.mp4 \
        --pose web/public/data/local_pose_2min.json \
        --out web/public/data/segments.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

KIND_COLOR = {'step': 0, 'fallback': 8, 'idle': 9}


def load_duration(pose_path: Path) -> float:
    return float(json.loads(pose_path.read_text(encoding='utf-8'))['duration'])


def from_ablation(path: Path, run: str, duration: float):
    data = json.loads(path.read_text(encoding='utf-8'))[run]
    steps = []
    for i, item in enumerate(data):
        try:
            detail = json.loads(item.get('response') or '{}')
        except Exception:
            detail = {}
        name = ' · '.join(x for x in (detail.get('phase'), detail.get('object')) if x)
        if not name:
            name = detail.get('action') or f'窗口 {i}'
        steps.append({
            'idx': i,
            'step_id': f'W{i:02d}',
            'name': name,
            'start': round(float(item['start']), 3),
            'end': round(float(item['end']), 3),
            'kind': 'step',
            'source': 'ablation',
        })
    coverage = [steps[0]['start'], steps[-1]['end']] if steps else [0, 0]
    return steps, coverage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', choices=['ablation'], default='ablation')
    ap.add_argument('--run', default='A')
    ap.add_argument('--data-dir', type=Path, default=Path('web/public/data'))
    ap.add_argument('--pose', type=Path, default=Path('web/public/data/local_pose_2min.json'))
    ap.add_argument('--video', default='v2_5m_7m_pose.mp4')
    ap.add_argument('--out', type=Path, default=Path('web/public/data/segments.json'))
    a = ap.parse_args()

    duration = load_duration(a.pose)
    if a.source == 'ablation':
        steps, coverage = from_ablation(a.data_dir / f'ablation_14_{a.run}.json', a.run, duration)
        note = f'来自 ablation_14_{a.run}（固定 3 秒窗口，仅覆盖前 {coverage[1]:.0f} 秒）'
    else:
        raise SystemExit('unsupported source')

    # 相邻完全重复的段合并，避免时间轴上出现"看起来是同一步却被切成两块"
    merged = []
    for s in steps:
        prev = merged[-1] if merged else None
        if prev and prev['name'] == s['name'] and abs(prev['end'] - s['start']) < 1e-6:
            prev['end'] = s['end']
            prev['merged'] = prev.get('merged', 1) + 1
            continue
        merged.append(s)
    for i, s in enumerate(merged):
        s['idx'] = i
        s['color'] = i % 8

    out = {
        'version': 1,
        'video': a.video,
        'duration': round(duration, 3),
        'source': a.source,
        'note': note,
        'coverage': [round(coverage[0], 3), round(coverage[1], 3)],
        'steps': merged,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({'steps': len(merged), 'coverage': out['coverage'],
                      'duration': out['duration'], 'out': str(a.out)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
