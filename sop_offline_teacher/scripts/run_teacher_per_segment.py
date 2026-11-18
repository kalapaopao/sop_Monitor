"""分工式流程：边界由本地骨架检测给出，大模型只负责「给这一段判 SOP 步骤」。

为什么这样分工（有实测依据）：
  · 大模型在 8 帧/30 秒（4.28 秒一帧）下，边界平均偏 3.5 秒，且 55% 的 GT 段比采样间隔还短 → 看不见。
  · 本地骨架变化点检测（10 Hz）在 ±2 秒容差下召回 69.6%，命中误差中位 0.53 秒，且 0 成本。
  · 因此：时间定位交给骨架，语义判定交给 VLM —— 每段只看**这一段内部**的几帧，
    帧间隔可以从 4 秒缩短到 0.5~1 秒，模型判步骤的难度大幅下降。

两种边界来源：
  --from-gt        用人工标注的边界（用来测「命名能力的上界」，把边界问题完全剔除）
  --from-detect    用 scripts/detect_boundaries.py 输出的候选边界（真实部署形态）

用法：
    # 先看成本
    python scripts/run_teacher_per_segment.py --from-detect --detected api_runs/boundaries.json \
        --video web/public/video/v2_5m_7m_pose.mp4 --out api_runs/perseg --dry-run

    # 测命名上界（用 GT 边界 + 干净画面）
    $env:DASHSCOPE_API_KEY="..."
    python scripts/run_teacher_per_segment.py --from-gt --pose web/public/data/local_pose_2min.json \
        --video web/public/video/v2_5m_7m_pose.mp4 --out api_runs/perseg_gt --publish-name perseg_gt
"""
from __future__ import annotations
import argparse, base64, json, os, sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DEFAULT_STEPS = Path('configs/sop_steps.json')
DEFAULT_VOCAB = Path('configs/teacher_vocab.json')

PROMPT = """你是工业装配现场的 SOP 判定员。下面给出的是**同一段动作**内的 {n} 张画面，
按时间先后排列，覆盖 {start:.1f}–{end:.1f} 秒（这一段已经被切好，你**不需要**再分段）。

【本工位的 SOP 步骤表】（只能从下表 + 兜底里选一个）
{step_table}
兜底：{fallback}

【重要】这些步骤之间**没有固定先后顺序**。不要因为上一步是 S1 就倾向选 S2；
只根据这段画面本身的三要素判定，不符合任何一步就选 unknown。

【判定方法】用三要素对号：**手在做什么 + 物件状态 + 工具是否接触**。
画面里可能存在与 SOP 无关的物体（{oos}），它们不算 SOP 物件。

【只输出 JSON】
{{"step_id":"S3","action":"简体中文≤30字","state_change":"none",
  "evidence_i_used":"依据画面哪里（≤25字）","left_hand":"≤15字","right_hand":"≤15字",
  "confidence":0.8}}

要求：三个中文描述必须为简体中文；state_change 只能取 {states}；拿不准就选 unknown 并把 confidence 填 0。"""


def load_steps(path: Path):
    d = json.loads(path.read_text(encoding='utf-8-sig'))
    lines = []
    for s in d.get('steps') or []:
        c = s.get('criteria') or {}
        lines.append(f"{s['step_id']} {s['name_cn']}：手={c.get('hand','—')}；"
                     f"物件={c.get('object','—')}；工具={c.get('tool','—')}")
    fb = ' / '.join(f"{x['step_id']}（{x['name_cn']}）" for x in (d.get('fallback_steps') or []))
    names = {s['step_id']: s['name_cn'] for s in (d.get('steps') or [])}
    for x in d.get('fallback_steps') or []:
        names[x['step_id']] = x['name_cn']
    allowed = set(names)
    return {'table': '\n'.join(lines), 'fallback': fb or 'unknown',
            'allowed': allowed, 'names': names}


def b64(p: Path):
    return 'data:image/jpeg;base64,' + base64.b64encode(Path(p).read_bytes()).decode()


def grab(video: Path, times, out_dir: Path, quality=88):
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    out_dir.mkdir(parents=True, exist_ok=True)
    res = []
    for t in times:
        if t < 0 or (total and t * fps > total - 2):
            res.append(None)
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, frame = cap.read()
        if not ok:
            res.append(None)
            continue
        p = out_dir / f'{t:08.2f}.jpg'
        if not p.exists():
            cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        res.append(p)
    cap.release()
    return res


def parse(txt):
    t = (txt or '').strip()
    if t.startswith('```'):
        parts = t.split('```')
        t = parts[1].lstrip('json').strip() if len(parts) > 1 else t
    try:
        return json.loads(t)
    except Exception:
        return {}


class RoiIndex:
    """按时间取最近的手部裁剪图（左右手各一张）。"""

    def __init__(self, index_path: Path):
        self.frames, self.ts = [], []
        if index_path and index_path.exists():
            d = json.loads(index_path.read_text(encoding='utf-8-sig'))
            self.frames = d.get('frames') or []
            self.ts = [f['t'] for f in self.frames]

    def nearest_with(self, t, key, tol=0.6):
        if not self.ts:
            return None, None
        import bisect
        i = min(max(bisect.bisect_left(self.ts, t), 0), len(self.ts) - 1)
        best = None
        for step in range(len(self.ts)):
            for j in (i - step, i + step):
                if 0 <= j < len(self.ts) and abs(self.ts[j] - t) <= tol:
                    if (self.frames[j].get(key) or {}).get('file'):
                        d = abs(self.ts[j] - t)
                        if best is None or d < best[0]:
                            best = (d, self.frames[j])
            if best is not None:
                break
        return (best[1], best[1]['t']) if best else (None, None)


def segments_from_gt(gt_path: Path, clip_offset: float, duration: float, skip_partial=True):
    """从人工边界取段。

    skip_partial：丢掉跨越片段边界、被截断的段（例如 269.6–301.0 的紧固段，
    在 300–420 片段里只剩 0.96 秒尾巴）。这种段判错不代表模型不行，算进准确率会误导。
    """
    g = json.loads(gt_path.read_text(encoding='utf-8-sig'))
    out, skipped = [], 0
    for s in g['segments']:
        if s['end'] <= clip_offset or s['start'] >= clip_offset + duration:
            continue                      # 完全不在片段内，不算"跳过"
        if skip_partial and (s['start'] < clip_offset or s['end'] > clip_offset + duration):
            skipped += 1                  # 压在片段边界上、被截断的段
            continue
        st, en = s['start'] - clip_offset, s['end'] - clip_offset
        out.append({'start': max(0.0, st), 'end': min(duration, en), 'gt': s['behavior']})
    out.sort(key=lambda x: x['start'])
    if skipped:
        print(f'（已跳过 {skipped} 个被片段边界截断的段，避免不公平计分）')
    return out


def segments_from_detect(det: Path, clip_offset: float, duration: float):
    d = json.loads(det.read_text(encoding='utf-8-sig'))
    peaks = sorted(p for p in d.get('peaks_v2') or [])
    edges = [clip_offset] + [p for p in peaks if clip_offset < p < clip_offset + duration] + [clip_offset + duration]
    segs = []
    for a, b in zip(edges, edges[1:]):
        if b - a < 0.6:
            continue
        segs.append({'start': round(a - clip_offset, 3), 'end': round(b - clip_offset, 3), 'gt': None})
    return segs


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--from-gt', action='store_true')
    src.add_argument('--from-detect', type=Path, help='detect_boundaries.py 的输出 json')
    ap.add_argument('--detected', type=Path, default=None, help='（兼容写法）同 --from-detect')
    ap.add_argument('--gt', type=Path, default=Path('web/public/data/gt_segments.json'))
    ap.add_argument('--steps', type=Path, default=DEFAULT_STEPS)
    ap.add_argument('--vocab', type=Path, default=DEFAULT_VOCAB)
    ap.add_argument('--video', type=Path, default=Path('web/public/video/v2_5m_7m_pose.mp4'))
    ap.add_argument('--clip-offset', type=float, default=300.0)
    ap.add_argument('--duration', type=float, default=119.99)
    ap.add_argument('--start', type=float, default=0.0, help='只跑片段的这一段（秒），用于分段验收')
    ap.add_argument('--end', type=float, default=0.0, help='0=到片段末尾')
    ap.add_argument('--frames', type=int, default=3, help='每段取几帧（段内均匀取）')
    ap.add_argument('--src-mode', choices=['clean', 'multi'], default='clean',
                    help='clean=只发全局画面；multi=全局画面 + 左右手周围放大图'
                         '（诊断显示所有错误都指向"分不清手里那个东西"，放大图正是针对它）')
    ap.add_argument('--roi-index', type=Path, default=Path('web/public/data/roi/roi_index.json'))
    ap.add_argument('--roi-dir', type=Path, default=Path('web/public/data/roi'))
    ap.add_argument('--min-seg', type=float, default=0.0, help='跳过短于此的段（GT 里有 0.28s 的段）')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--model', default=os.getenv('DASHSCOPE_MODEL', 'qwen3-vl-plus'))
    ap.add_argument('--publish-name', default='')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    steps = load_steps(a.steps)
    vocab = json.loads(a.vocab.read_text(encoding='utf-8-sig'))
    oos = ' / '.join(vocab.get('out_of_scope_objects') or []) or '（无）'
    states = ' | '.join(vocab.get('states') or [])

    if a.from_gt:
        segs = segments_from_gt(a.gt, a.clip_offset, a.duration)
        mode = 'gt'
    else:
        det = a.from_detect or a.detected
        segs = segments_from_detect(det, a.clip_offset, a.duration)
        mode = 'detect'
    if a.min_seg:
        segs = [s for s in segs if s['end'] - s['start'] >= a.min_seg]
    # 只跑片段中的一段（例如「后一分钟」：--start 60）
    r0 = max(0.0, a.start)
    r1 = min(a.duration, a.end) if a.end > 0 else a.duration
    if r0 > 0 or r1 < a.duration:
        kept = []
        for s in segs:
            st, en = max(s['start'], r0), min(s['end'], r1)
            if en - st < 0.5:
                continue
            kept.append({**s, 'start': round(st, 3), 'end': round(en, 3)})
        segs = kept
        print(f'限定区间 {r0:.1f}–{r1:.1f}s')
    if a.limit:
        segs = segs[:a.limit]
    print(f'边界来源 {mode} ｜ 段数 {len(segs)} ｜ 每段 {a.frames} 帧 → '
          f'{len(segs) * a.frames} 张图 / {len(segs)} 次调用')
    for s in segs[:6]:
        print(f'   {s["start"]:7.2f}-{s["end"]:7.2f}s ({s["end"] - s["start"]:5.2f}s) {s.get("gt") or ""}')

    frames_dir = a.out / 'frames'
    roi = RoiIndex(a.roi_index) if a.src_mode == 'multi' else None
    plans, hand_filled = [], 0
    for s in segs:
        dur = s['end'] - s['start']
        inset = min(0.12, dur * 0.1)
        lo, hi = s['start'] + inset, s['end'] - inset
        times = [round(lo + (hi - lo) * (i + 0.5) / a.frames, 3) if a.frames > 1 else (lo + hi) / 2
                 for i in range(a.frames)]
        paths = grab(a.video, times, frames_dir)
        imgs = []
        for t, fp in zip(times, paths):
            if fp:
                imgs.append(('global', t, fp))
        if not imgs:
            continue
        if roi is not None:
            mid = (s['start'] + s['end']) / 2
            for side in ('left', 'right'):
                fe, ft = roi.nearest_with(mid, side, 0.6)
                if fe and fe.get(side) and (a.roi_dir / fe[side]['file']).exists():
                    imgs.append((side, ft if ft is not None else mid, a.roi_dir / fe[side]['file']))
                    hand_filled += 1
        plans.append({'seg': s, 'imgs': imgs})

    total_imgs = sum(len(p['imgs']) for p in plans)
    print(f'输入模式 {a.src_mode} ｜ 图片合计 {total_imgs} 张'
          f'（平均每段 {total_imgs / max(1, len(plans)):.1f} 张）'
          + (f'｜手部放大图补到 {hand_filled} 张' if a.src_mode == 'multi' else ''))
    if a.dry_run:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / 'plan.json').write_text(json.dumps({
            'mode': mode, 'src_mode': a.src_mode,
            'segments': [{'start': p['seg']['start'], 'end': p['seg']['end'],
                          'images': [[k, round(t, 2), str(f)] for k, t, f in p['imgs']]}
                         for p in plans]}, ensure_ascii=False, indent=1), encoding='utf-8')
        print('dry-run：已写计划，未调用 API')
        return 0

    key = os.getenv('DASHSCOPE_API_KEY')
    if not key:
        print('缺少 DASHSCOPE_API_KEY')
        return 2
    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=os.getenv('DASHSCOPE_BASE_URL',
                    'https://dashscope.aliyuncs.com/compatible-mode/v1'))

    results = []
    for i, p in enumerate(plans):
        s = p['seg']
        prompt = PROMPT.format(n=len(p['imgs']), start=s['start'], end=s['end'],
                               step_table=steps['table'], fallback=steps['fallback'],
                               oos=oos, states=states)
        content = [{'type': 'text', 'text': prompt}]
        for kind, t, fp in p['imgs']:
            if kind == 'global':
                label = f'整体画面（时间 {t:.2f} 秒）'
            elif kind == 'left':
                label = f'左手周围放大图（时间 {t:.2f} 秒，看左手拿着什么）'
            else:
                label = f'右手周围放大图（时间 {t:.2f} 秒，看右手拿着什么）'
            content.append({'type': 'text', 'text': label})
            content.append({'type': 'image_url', 'image_url': {'url': b64(fp)}})
        try:
            r = client.chat.completions.create(model=a.model, temperature=0.0,
                                               messages=[{'role': 'user', 'content': content}])
            raw = r.choices[0].message.content
        except Exception as exc:
            raw = f'__ERROR__ {exc}'
        d = parse(raw)
        sid = str(d.get('step_id') or 'unknown').strip()
        if sid not in steps['allowed']:
            sid = 'unknown'
        results.append({'start': s['start'], 'end': s['end'], 'gt_behavior': s.get('gt'),
                        'step_id': sid, 'step_name': steps['names'].get(sid, ''),
                        'action': d.get('action', ''), 'state_change': d.get('state_change', 'none'),
                        'evidence_i_used': d.get('evidence_i_used', ''),
                        'left_hand': d.get('left_hand', ''), 'right_hand': d.get('right_hand', ''),
                        'confidence': d.get('confidence'), 'raw': raw[:400]})
        print(f'  [{i + 1}/{len(plans)}] {s["start"]:6.2f}-{s["end"]:6.2f}s → {sid:<7} '
              f'{results[-1]["step_name"][:10]:<10} {results[-1]["action"][:26]}')

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'segments_proposal.json').write_text(json.dumps({
        'source': f'teacher-per-segment-{mode}', 'boundary_mode': mode,
        'note': '边界来自本地骨架检测/人工标注，大模型只做每段的 SOP 步骤判定',
        'segments': results}, ensure_ascii=False, indent=1), encoding='utf-8')

    if a.publish_name:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from publish_segments import publish
            print('\n发布到前端…')
            publish(a.out, a.publish_name, clip_offset=0.0, duration=a.duration)
        except Exception as exc:
            print(f'[!] 发布失败（结果已写入 {a.out}）：{exc}')

    # 与人工 GT 对照：按每段中心时刻落在哪个 GT 段来比对（两种边界来源都适用）
    cfg = json.loads(a.steps.read_text(encoding='utf-8-sig'))
    b2s = {x['boris_behavior']: x['step_id'] for x in cfg['steps'] if x.get('boris_behavior')}
    gt_all = json.loads(a.gt.read_text(encoding='utf-8-sig'))['segments']

    def gt_at(clip_t):
        v2 = clip_t + a.clip_offset
        for x in gt_all:
            if x['start'] - 1e-6 <= v2 <= x['end'] + 1e-6:
                return b2s.get(x['behavior'], 'unknown')
        return 'idle'

    ok = tot = 0
    nested = 0
    print(f'\n与人工标注对照（按段中心所在 GT 段，边界来源 {mode}，输入 {a.src_mode}）：')
    for r in results:
        center = (r['start'] + r['end']) / 2
        want = gt_at(center)
        good = r['step_id'] == want
        # 嵌套段：GT 里「取上盖」嵌在「取镜座」内，同一批画面会被两次计分 → 单独标出
        inner = any(
            (x['start'] - a.clip_offset) <= r['start'] + 1e-6
            and r['end'] <= (x['end'] - a.clip_offset) + 1e-6
            and x['behavior'] != r['gt_behavior']          # 只比较 behavior 名，别拿 step_id 比
            for x in gt_all)
        if inner:
            nested += 1
        tag = ' [嵌套]' if inner else ''
        ok += int(good)
        tot += 1
        print(f'  {"OK " if good else "   "} {r["start"]:6.2f}-{r["end"]:6.2f}s '
              f'GT={want:<7} 判={r["step_id"]:<7}{tag} {r["action"][:20]}')
    print(f'→ 一致率 {ok}/{tot} = {ok / max(1, tot):.1%}'
          + (f'（其中 {nested} 段是 GT 嵌套段，会重复计分）' if nested else '')
          + '（detect 模式下边界误差也会算进这个数字）')
    print('\n写出', a.out / 'segments_proposal.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
