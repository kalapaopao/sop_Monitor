"""把模型产出的工序片段导出成 BORIS 7.0 项目文件（.boris），可直接双击打开。

格式依据：BORIS 7.0 的 .boris 是 JSON（不是 SQLite），结构实测自
`BORIS_Data/project/sop_installment_02.boris`：

  {
    "time_format": "hh:mm:ss", "project_date", "project_name", "project_description",
    "project_format_version": "7.0", "subjects_conf": {},
    "behaviors_conf": {"0": {"type":"State event","key":"a","code":...,"description":...,
                             "color":"","category":"","modifiers":"","excluded":"","coding map":""}},
    "observations": {"<名字>": {"file": {"1": [路径], "2": [], ...}, "type":"MEDIA", "date":...,
                                "description":"", "time offset":0.0,
                                "events": [[秒, 主体, 行为code, 修饰, "", 帧号], ...], ...}},
    ...
  }

关键点：
  · State 事件是**成对**的：同一个 code 出现两次 = 开始 / 结束
  · 事件元组固定 6 元：`[时间(秒), 主体, 行为代码, 修饰, 空, 帧号]`
  · 帧号 = round(秒 × 有效帧率)，有效帧率用 frames/duration（实测 v2.mp4 是 24.702，不是标称 25）
  · 事件必须按时间升序

用法：
    python scripts/export_boris.py --segments web/public/data/segments.json \
        --video E:/study/.../vedio/v2_5m_7m_h264.mp4 --fps 24.702 \
        --name sop_installment_03 --out web/public/data/export.boris
"""
from __future__ import annotations
import argparse, json
from datetime import datetime
from pathlib import Path

KEY_SEQ = 'abcdefghijklmnopqrstuvwxyz'


def build(segments, video_path, fps, name, description='', with_hands=None, duration=None):
    """segments: [{'step_id','name','start','end'}, ...]；with_hands: {left:[(s,e)], right:[(s,e)]}"""
    behaviors = []          # (code, description)
    seen = {}

    def reg(code, desc):
        if code not in seen:
            seen[code] = len(behaviors)
            behaviors.append((code, desc))
        return code

    events = []
    for seg in segments:
        step_id = str(seg.get('step_id') or seg.get('id') or 'step')
        reg(step_id, str(seg.get('name') or step_id))
        events.append((float(seg['start']), step_id))
        events.append((float(seg['end']), step_id))

    if with_hands:
        for side, code, desc in (('left', 'hand_left_visible', '左手可见'),
                                 ('right', 'hand_right_visible', '右手可见')):
            segs = with_hands.get(side) or []
            if not segs:
                continue
            reg(code, desc)
            for s, e in segs:
                events.append((float(s), code))
                events.append((float(e), code))

    events.sort(key=lambda x: x[0])

    behaviors_conf = {}
    for i, (code, desc) in enumerate(behaviors):
        behaviors_conf[str(i)] = {
            'type': 'State event',
            'key': KEY_SEQ[i] if i < len(KEY_SEQ) else '',
            'code': code,
            'description': desc,
            'color': '', 'category': '', 'modifiers': '', 'excluded': '', 'coding map': '',
        }

    vp = str(video_path).replace('\\', '/')
    media_info = {
        'length': {vp: float(duration) if duration else 0.0},
        'fps': {vp: 25.0},
        'hasVideo': {vp: True},
        'hasAudio': {vp: False},
        'offset': {'1': 0.0},
        'display': {vp: 'Nothing'},
    }
    obs = {
        'file': {'1': [vp], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': []},
        'type': 'MEDIA',
        'date': datetime.now().isoformat(timespec='milliseconds'),
        'description': '',
        'time offset': 0.0,
        'events': [[round(t, 3), '', code, '', '', int(round(t * fps))] for t, code in events],
        'observation time interval': [0, 0],
        'independent_variables': {},
        'visualize_spectrogram': False,
        'visualize_waveform': False,
        'media_creation_date_as_offset': False,
        'media_scan_sampling_duration': 0,
        'image_display_duration': 1,
        'close_behaviors_between_videos': False,
        'media_info': media_info,
    }

    return {
        'time_format': 'hh:mm:ss',
        'project_date': datetime.now().isoformat(timespec='seconds'),
        'project_name': name,
        'project_description': description or name,
        'project_format_version': '7.0',
        'subjects_conf': {},
        'behaviors_conf': behaviors_conf,
        'observations': {name: obs},
        'behavioral_categories': [],
        'behavioral_categories_config': [],
        'independent_variables': {},
        'coding_map': {},
        'behaviors_coding_map': [],
        'converters': {},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--segments', type=Path, required=True, help='segments.json（含 steps）')
    ap.add_argument('--video', required=True, help='视频绝对路径（写进 observations.file）')
    ap.add_argument('--fps', type=float, default=24.702, help='有效帧率 = frames/duration')
    ap.add_argument('--name', default='sop_export', help='项目名/观测名')
    ap.add_argument('--description', default='')
    ap.add_argument('--pose', type=Path, default=None, help='给了就顺带导出左右手可见段')
    ap.add_argument('--hands', action='store_true', help='导出左右手可见段（需 --pose）')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()

    seg = json.loads(a.segments.read_text(encoding='utf-8-sig'))
    steps = seg.get('steps') or []

    with_hands = None
    if a.hands and a.pose:
        pose = json.loads(a.pose.read_text(encoding='utf-8-sig'))
        rate = float(pose.get('rate') or 10)
        half = 1 / (2 * rate)
        gap = 0.6
        MINP, THR = 4, 0.18

        def build_segs(key):
            out, cur = [], None
            for r in pose['frames']:
                pts = [p for p in (r.get(key) or []) if p[2] > THR]
                if len(pts) < MINP:
                    continue
                t0 = max(0.0, r['timestamp'] - half); t1 = r['timestamp'] + half
                if cur is None:
                    cur = [t0, t1]
                elif t0 - cur[1] <= gap:
                    cur[1] = t1
                else:
                    out.append(cur); cur = [t0, t1]
            if cur:
                out.append(cur)
            return out

        with_hands = {'left': build_segs('left_hand'), 'right': build_segs('right_hand')}

    payload = build(steps, a.video, a.fps, a.name, a.description,
                    with_hands=with_hands, duration=seg.get('duration'))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')

    n_ev = len(payload['observations'][a.name]['events'])
    print(json.dumps({'out': str(a.out), 'behaviors': len(payload['behaviors_conf']),
                      'events': n_ev, 'segments': len(steps),
                      'hands': bool(with_hands)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
