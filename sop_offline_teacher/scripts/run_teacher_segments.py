"""教师模型"自由分段"跑整段视频，已实现 L1/L2/L3 三层约束。

L1 提示词：角色 + 任务（**切分**而不是一窗一标签）+ 判据（什么算完成、看不清怎么办）
L2 结构化：强制 `segments` 数组 + 每段 start/end 秒 + 固定枚举 + 严格 JSON
L3 词表：物件白名单 + 禁用同义词（配置在 `configs/teacher_vocab.json`，改配置不改代码）

输入模式（回答"叠加骨架和框到底帮忙还是干扰"）：
  clean   ：只发干净画面
  overlay ：发烧了骨架/手框/ROI 虚线的画面
  multi   ：干净全局 + 左手周 + 右手周 + 各固定区小图（多图输入）

产物：
  windows_freeform.json   每窗：raw 原文 / parsed 解析 / flags 质量标记
  segments_proposal.json  合并后的候选段（可导入 BORIS、可用 eval_boundaries 打分）

用法：
    # 看提示词（不花钱、不需要 Key）
    python scripts/run_teacher_segments.py --print-prompt

    # 算成本
    python scripts/run_teacher_segments.py --src-mode clean --video web/public/video/v2_5m_7m_pose.mp4 \
        --clean-video vedio/v2.mp4 --clean-offset 300 --duration 120 --stride 30 --frames 8 \
        --out-offset 300 --out api_runs/free_clean --dry-run

    # 真跑
    $env:DASHSCOPE_API_KEY="..."
    python scripts/run_teacher_segments.py --src-mode clean ... --out api_runs/free_clean
"""
from __future__ import annotations
import argparse, base64, bisect, json, os, sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DEFAULT_VOCAB = Path('configs/teacher_vocab.json')
DEFAULT_STEPS = Path('configs/sop_steps.json')

PROMPT = """你是工业装配现场的 SOP 动作教师。下面按时间先后给出同一工位的 {n} 张画面，
覆盖视频的 {start:.1f}–{end:.1f} 秒。

【本工位的 SOP 步骤表】（step_id 只能从下表 + 兜底里选，不得自创）
{step_table}

兜底类别：{fallback}

【可用物件白名单】（只能使用这些名词，禁止自创同义词）
{whitelist}
若画面里出现白名单以外的物体，写「{unknown_object}」，不要猜名字。

【画面里还存在与 SOP 无关、但真实存在的物体】
{oos}
- 这些物体**确实在工位里**，可以提及，但不要把它们当成 SOP 物件；
- 若当前动作只是在操作它们（例如拿起扫码枪、拆塑料袋），step_id 必须选兜底类别；
- 只有当它们被用于当前 SOP 步骤时，才算 SOP 物件。

【任务】
在这段画面里**切分动作并判定 SOP 步骤**：
- 中途换了步骤就分成多段，每段给出起止秒数（必须在 {start:.1f}–{end:.1f} 范围内）；
- 整段同一步骤只输出一段；
- 用「三要素」判定：**手在做什么 + 物件状态 + 工具是否接触**，三者与步骤表一致才算该步骤；
- 判不准就选 unknown，**不要为了凑满而硬选**。

【判断要求】
1. step_id：只能取上面表格里的 id 或兜底 id。
2. action：简体中文 ≤30 字，必须对应「手的动作 + 物件状态」。
3. state_change：只能取 {states}；只有物件状态真的改变才填前三者。
4. evidence_i_used：说明你依据画面哪里判断、以及关键三要素里哪一条成立（简体中文 ≤25 字）。
   找不到依据：step_id 选 unknown，confidence 填 0。
5. left_hand / right_hand：简体中文 ≤15 字；看不到写「未看到」。
6. 只输出 JSON，不要解释，不要 Markdown 代码块。

【输出格式】
{{"segments":[{{"start":{start:.1f},"end":{end:.1f},"step_id":"S3","action":"中文一句话",
  "state_change":"none","evidence_i_used":"中文依据",
  "left_hand":"中文","right_hand":"中文","confidence":0.8}}]}}"""

RETRY_SUFFIX = """

【上一次回答不合格，请修正以下问题后重新输出】
{problems}
仍然只输出 JSON。"""

IMAGE_LABELS = {
    'global': '整体画面（完整画面，用于判断人在做什么）',
    'left': '左手周围放大图（看左手拿着什么）',
    'right': '右手周围放大图（看右手拿着什么）',
}


# ===== L3：词表与校验 ============================================================

def load_vocab(path: Path):
    v = json.loads(path.read_text(encoding='utf-8-sig'))
    for k in ('whitelist', 'states'):
        if not v.get(k):
            raise SystemExit(f'{path} 缺少 {k}')
    v.setdefault('banned_synonyms', {})
    v.setdefault('unknown_action', '无法确认')
    v.setdefault('unknown_object', '未识别物件')
    v.setdefault('min_seg_sec', 0.4)
    v.setdefault('max_seg_sec', 40.0)
    v.setdefault('require_evidence', True)
    return v


def has_cjk(s):
    return any('\u4e00' <= ch <= '\u9fff' for ch in str(s or ''))


def load_steps(path: Path):
    """读取 SOP 步骤表，生成给模型看的「三要素判据」文本（方案 A 的核心素材）。"""
    d = json.loads(path.read_text(encoding='utf-8-sig'))
    lines = []
    for s in d.get('steps') or []:
        c = s.get('criteria') or {}
        exp = s.get('expected_sec') or []
        dur = f'，典型时长 {exp[0]:g}~{exp[1]:g} 秒' if len(exp) == 2 else ''
        lines.append(f"{s['step_id']} {s['name_cn']}：手={c.get('hand', '—')}；"
                     f"物件={c.get('object', '—')}；工具={c.get('tool', '—')}{dur}")
    fb = ' / '.join(f"{x['step_id']}（{x['name_cn']}）" for x in (d.get('fallback_steps') or []))
    allowed = [s['step_id'] for s in (d.get('steps') or [])] + \
              [x['step_id'] for x in (d.get('fallback_steps') or [])]
    names = {s['step_id']: s['name_cn'] for s in (d.get('steps') or [])}
    for x in d.get('fallback_steps') or []:
        names[x['step_id']] = x['name_cn']
    return {'table': '\n'.join(lines), 'fallback': fb or 'unknown', 'allowed': allowed, 'names': names}


def build_prompt(vocab, start, end, n, steps=None):
    return PROMPT.format(n=n, start=start, end=end,
                         step_table=(steps or {}).get('table', '（未提供步骤表）'),
                         fallback=(steps or {}).get('fallback', 'unknown'),
                         whitelist=' / '.join(vocab['whitelist']),
                         oos=' / '.join(vocab.get('out_of_scope_objects') or []) or '（无）',
                         states=' | '.join(vocab['states']),
                         unknown_object=vocab['unknown_object'],
                         unknown_action=vocab['unknown_action'])


def parse_segments(raw):
    """L2：解析 `segments` 数组；兼容旧的单对象输出。返回 (segments, 解析错误)。"""
    txt = (raw or '').strip()
    if txt.startswith('```'):
        parts = txt.split('```')
        txt = parts[1].lstrip('json').strip() if len(parts) > 1 else txt
    try:
        d = json.loads(txt)
    except Exception as exc:
        return [], f'JSON 解析失败: {exc}'
    segs = d.get('segments') if isinstance(d, dict) else None
    if segs is None and isinstance(d, dict):          # 兼容旧格式
        segs = [d]
    if not isinstance(segs, list) or not segs:
        return [], 'segments 不是非空数组'
    out = []
    for i, s in enumerate(segs):
        if not isinstance(s, dict):
            continue
        out.append({
            'start': s.get('start'), 'end': s.get('end'),
            'step_id': str(s.get('step_id') or '').strip(),
            'action': str(s.get('action') or ''),
            'state_change': str(s.get('state_change') or 'none').lower(),
            'evidence_i_used': str(s.get('evidence_i_used') or ''),
            'left_hand': str(s.get('left_hand') or ''),
            'right_hand': str(s.get('right_hand') or ''),
            'confidence': s.get('confidence'),
        })
    return out, '' if out else 'segments 里没有可用对象'


def validate_segments(segs, vocab, win_start, win_end, steps=None):
    """L2+L3+方案A：结构 / 枚举 / 中文 / 依据 / 白名单 / 时长 / 越界 / step_id 合法性。"""
    problems = []
    flags = {'schema': True, 'lang': True, 'evidence': True, 'whitelist': True,
             'range': True, 'step_id': True}
    banned = vocab['banned_synonyms']
    allowed = set((steps or {}).get('allowed') or [])
    for i, s in enumerate(segs):
        tag = f'segments[{i}]'
        if allowed and s.get('step_id') not in allowed:
            problems.append(f'{tag} step_id 非法: {s.get("step_id")!r}，只能是 {sorted(allowed)}')
            flags['step_id'] = False
        st, en = s.get('start'), s.get('end')
        if not isinstance(st, (int, float)) or not isinstance(en, (int, float)):
            problems.append(f'{tag} start/end 不是数字'); flags['schema'] = False; continue
        if en <= st:
            problems.append(f'{tag} end({en}) <= start({st})'); flags['schema'] = False
        if st < win_start - 1.0 or en > win_end + 1.0:
            problems.append(f'{tag} 超出窗口 {win_start:.1f}–{win_end:.1f}'); flags['range'] = False
        dur = min(en, win_end) - max(st, win_start)
        if dur < vocab['min_seg_sec']:
            problems.append(f'{tag} 时长 {dur:.2f}s < 最短 {vocab["min_seg_sec"]}s'); flags['range'] = False
        if dur > vocab['max_seg_sec']:
            problems.append(f'{tag} 时长 {dur:.2f}s > 最长 {vocab["max_seg_sec"]}s'); flags['range'] = False
        if s['state_change'] not in vocab['states']:
            problems.append(f'{tag} state_change 非法: {s["state_change"]!r}'); flags['schema'] = False
        if not has_cjk(s['action']):
            problems.append(f'{tag} action 不是中文'); flags['lang'] = False
        if vocab['require_evidence'] and not s['evidence_i_used'].strip():
            problems.append(f'{tag} 缺 evidence_i_used'); flags['evidence'] = False
        # L3 白名单：白名单外的关键词若出现在 action 的描述里，要求同时出现白名单名词或「未识别物件」
        act = s['action']
        hit_banned = [w for w in banned if w in act]
        named_ok = any(w in act for w in vocab['whitelist']) or vocab['unknown_object'] in act
        if hit_banned and not named_ok:
            problems.append(f'{tag} action 用了白名单外同义词 {hit_banned}，且未给白名单物件')
            flags['whitelist'] = False
    return problems, flags


# ===== 输入素材 ==================================================================

def b64(path: Path):
    return 'data:image/jpeg;base64,' + base64.b64encode(Path(path).read_bytes()).decode()


def probe_end(video: Path):
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    hi = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
    lo = 0.0
    for _ in range(22):
        mid = (lo + hi) / 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(mid * fps)))
        ok, _f = cap.read()
        if ok:
            lo = mid
        else:
            hi = mid
    cap.release()
    return lo


def grab(video: Path, times, out_dir: Path, max_t, quality=86, prefix=''):
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for t in times:
        t = min(t, max_t)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, frame = cap.read()
        if not ok:
            out.append(None)
            continue
        p = out_dir / f'{prefix}{t:08.2f}.jpg'
        if not p.exists():
            cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        out.append(p)
    cap.release()
    return out


class RoiIndex:
    def __init__(self, index_path: Path):
        self.frames, self.ts = [], []
        if index_path and index_path.exists():
            d = json.loads(index_path.read_text(encoding='utf-8-sig'))
            self.frames = d.get('frames') or []
            self.ts = [f['t'] for f in self.frames]

    def nearest_with(self, t, key, tol=0.6):
        if not self.ts:
            return None, None
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

    def nearest(self, t, tol=0.4):
        if not self.ts:
            return None
        i = min(max(bisect.bisect_left(self.ts, t), 0), len(self.ts) - 1)
        if i > 0 and abs(self.ts[i - 1] - t) <= abs(self.ts[i] - t):
            i -= 1
        return self.frames[i] if abs(self.ts[i] - t) <= tol else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', type=Path, default=Path('web/public/video/v2_5m_7m_pose.mp4'))
    ap.add_argument('--src-mode', choices=['clean', 'overlay', 'multi'], default='clean')
    ap.add_argument('--clean-video', type=Path, default=Path('vedio/v2.mp4'))
    ap.add_argument('--clean-offset', type=float, default=300.0)
    ap.add_argument('--roi-index', type=Path, default=Path('web/public/data/roi/roi_index.json'))
    ap.add_argument('--roi-dir', type=Path, default=Path('web/public/data/roi'))
    ap.add_argument('--station-index', type=Path,
                    default=Path('web/public/data/station_roi/station_roi_index.json'))
    ap.add_argument('--station-dir', type=Path, default=Path('web/public/data/station_roi'))
    ap.add_argument('--vocab', type=Path, default=DEFAULT_VOCAB)
    ap.add_argument('--steps', type=Path, default=DEFAULT_STEPS,
                    help='SOP 步骤表（方案A：把三要素判据写进提示词，直接输出 step_id）')
    ap.add_argument('--no-step', action='store_true', help='关掉步骤判定，退回纯自由描述')
    ap.add_argument('--start', type=float, default=0.0)
    ap.add_argument('--duration', type=float, default=0.0, help='0=到片尾')
    ap.add_argument('--stride', type=float, default=30.0,
                    help='每个窗口覆盖多少秒。切分模式下建议 20~30s（窗口内要能装下多个动作）')
    ap.add_argument('--frames', type=int, default=8, help='每窗帧数（切分模式建议 8~12）')
    ap.add_argument('--out-offset', type=float, default=0.0)
    ap.add_argument('--out', type=Path, default=None,
                    help='结果输出目录（--print-prompt 时可省略）')
    ap.add_argument('--publish-name', default='',
                    help='跑完自动发布到网页（发布名，会出现在前端「结果来源」下拉里）')
    ap.add_argument('--publish-duration', type=float, default=119.99,
                    help='发布到前端时使用的片段时间轴长度（默认 119.99，对应 v2_5m_7m_pose.mp4）')
    ap.add_argument('--model', default=os.getenv('DASHSCOPE_MODEL', 'qwen3-vl-plus'))
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--print-prompt', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    vocab = load_vocab(a.vocab)
    steps = None if a.no_step else (load_steps(a.steps) if a.steps.exists() else None)
    if steps is None:
        print(f'[提示] 未启用 SOP 步骤判定（--no-step 或找不到 {a.steps}），只输出自由描述')

    if a.print_prompt:
        p = build_prompt(vocab, a.start, a.start + (a.duration or 30), a.frames, steps)
        print('=' * 72); print(p); print('=' * 72)
        print(f'\n字符数 {len(p)} ｜ 白名单 {len(vocab["whitelist"])} 项 ｜ '
              f'禁用同义词 {len(vocab["banned_synonyms"])} 个 ｜ 状态枚举 {vocab["states"]}')
        return 0

    overlay_end = probe_end(a.video)
    t0 = max(0.0, a.start)
    t1 = min(overlay_end, a.start + a.duration) if a.duration > 0 else overlay_end
    if t1 <= t0:
        sys.exit(f'区间非法：{t0}–{t1}')

    wins, t = [], t0
    while t < t1 - 1e-6:
        we = min(t + a.stride, t1)
        inner = [round(t + (we - t) * (i + 0.5) / a.frames, 3) for i in range(a.frames)]
        wins.append({'idx': len(wins), 'start': round(t, 3), 'end': round(we, 3), 'frames': inner})
        t = we
    if a.limit:
        wins = wins[:a.limit]
    print(f'模式 {a.src_mode} ｜ 区间 {t0:.1f}–{t1:.1f}s ｜ stride {a.stride}s ｜ '
          f'帧/窗 {a.frames} ｜ 窗口 {len(wins)}')

    out_frames = a.out / 'frames'
    roi_idx, st_idx = RoiIndex(a.roi_index), RoiIndex(a.station_index)
    st_meta = {}
    if a.station_index.exists():
        st_meta = {r['id']: r.get('name', r['id'])
                   for r in (json.loads(a.station_index.read_text(encoding='utf-8-sig')).get('regions') or [])}

    plans = []
    for w in wins:
        imgs = []
        if a.src_mode in ('overlay', 'clean'):
            src = a.video if a.src_mode == 'overlay' else a.clean_video
            times = w['frames'] if a.src_mode == 'overlay' else [a.clean_offset + x for x in w['frames']]
            max_t = overlay_end if a.src_mode == 'overlay' else probe_end(a.clean_video)
            pre = 'ov_' if a.src_mode == 'overlay' else 'cl_'
            for ts, p in zip(w['frames'], grab(src, times, out_frames, max_t, prefix=pre)):
                if p:
                    imgs.append(('global', ts, p))
        else:
            mid = (w['start'] + w['end']) / 2
            gp = grab(a.clean_video, [a.clean_offset + mid], out_frames,
                      probe_end(a.clean_video), prefix='cl_')
            if gp and gp[0]:
                imgs.append(('global', mid, gp[0]))
            for side in ('left', 'right'):
                fe, ft = roi_idx.nearest_with(mid, side, 0.6)
                if fe and fe.get(side) and (a.roi_dir / fe[side]['file']).exists():
                    imgs.append((side, ft if ft is not None else mid, a.roi_dir / fe[side]['file']))
            se = st_idx.nearest(mid)
            if se:
                for rid, ent in (se.get('regions') or {}).items():
                    if ent and (a.station_dir / ent['file']).exists():
                        imgs.append((f'station:{st_meta.get(rid, rid)}', mid, a.station_dir / ent['file']))
        if imgs:
            plans.append({'w': w, 'imgs': imgs})

    total_imgs = sum(len(p['imgs']) for p in plans)
    print(f'窗口 {len(plans)} 个 ｜ 图片 {total_imgs} 张 ｜ 平均每窗 '
          f'{total_imgs / max(1, len(plans)):.1f} 张 ｜ 调用 {len(plans)} 次')

    if a.dry_run:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / 'plan.json').write_text(json.dumps({
            'mode': a.src_mode, 'range': [t0, t1], 'stride': a.stride, 'frames': a.frames,
            'windows': [{'idx': p['w']['idx'], 'start': p['w']['start'], 'end': p['w']['end'],
                         'images': [[k, round(ts, 2), str(f)] for k, ts, f in p['imgs']]}
                        for p in plans]}, ensure_ascii=False, indent=1), encoding='utf-8')
        print('dry-run：已写计划，未调用 API')
        return 0

    key = os.getenv('DASHSCOPE_API_KEY')
    if not key:
        print('缺少 DASHSCOPE_API_KEY（可先用 --print-prompt / --dry-run）')
        return 2
    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=os.getenv('DASHSCOPE_BASE_URL',
                    'https://dashscope.aliyuncs.com/compatible-mode/v1'))

    rows, retried, need_review = [], 0, 0
    for i, p in enumerate(plans):
        w = p['w']
        base = build_prompt(vocab, w['start'], w['end'], len(p['imgs']), steps)
        img_content = []
        for kind, ts, fp in p['imgs']:
            label = IMAGE_LABELS.get(kind, f'固定区域「{kind.split(":", 1)[-1]}」的放大图')
            img_content.append({'type': 'text', 'text': f'{label}（时间 {ts:.2f} 秒）'})
            img_content.append({'type': 'image_url', 'image_url': {'url': b64(fp)}})

        raw, segs, problems, flags, attempt_used = '', [], [], {}, 0
        for attempt in (1, 2):
            attempt_used = attempt
            prompt = base if attempt == 1 else base + RETRY_SUFFIX.format(
                problems='\n'.join(f'- {x}' for x in problems[:6]) or '- 输出不符合要求')
            try:
                r = client.chat.completions.create(
                    model=a.model, temperature=0.0,
                    messages=[{'role': 'user', 'content': [{'type': 'text', 'text': prompt}] + img_content}])
                raw = r.choices[0].message.content
            except Exception as exc:
                raw = f'__ERROR__ {exc}'
            segs, perr = parse_segments(raw)
            if perr:
                problems, flags = [perr], {'schema': False}
                if attempt == 1:
                    retried += 1
                    continue
                break
            problems, flags = validate_segments(segs, vocab, w['start'], w['end'], steps)
            hard = [x for x in problems if 'JSON' in x or '中文' in x or '数组' in x]
            if hard and attempt == 1:
                retried += 1
                continue
            break

        bad = bool(problems)
        need_review += int(bad)
        rows.append({'idx': w['idx'], 'start': w['start'], 'end': w['end'],
                     'imgs': len(p['imgs']), 'attempts': attempt_used,
                     'segments': segs, 'problems': problems, 'flags': flags,
                     'need_review': bad, 'raw': raw[:1500]})
        brief = '；'.join(f'{s["start"]:.1f}-{s["end"]:.1f} {s["action"][:18]}' for s in segs[:3])
        mark = 'OK ' if not problems else f'!{len(problems)}'
        print(f'  [{i + 1}/{len(plans)}] {w["start"]:7.1f}-{w["end"]:7.1f}s 图{p["imgs"]} {mark} {brief[:70]}')

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'windows_freeform.json').write_text(json.dumps({
        'video': str(a.video), 'mode': a.src_mode, 'range': [t0, t1],
        'stride': a.stride, 'frames': a.frames, 'vocab': str(a.vocab),
        'retried': retried, 'need_review': need_review, 'windows': rows},
        ensure_ascii=False, indent=1), encoding='utf-8')

    # 合并成候选段（新格式已是"分段结果"，直接汇总；旧格式退回按 phase_key 合并）
    proposal = []
    for r in rows:
        for s in r['segments']:
            st = max(r['start'], float(s['start'])); en = min(r['end'], float(s['end']))
            if en - st <= 0:
                continue
            proposal.append({'start': round(st + a.out_offset, 3), 'end': round(en + a.out_offset, 3),
                             'step_id': s.get('step_id') or '',
                             'step_name': ((steps or {}).get('names') or {}).get(s.get('step_id'), ''),
                             'action': s['action'], 'state_change': s['state_change'],
                             'evidence_i_used': s['evidence_i_used'],
                             'left_hand': s['left_hand'], 'right_hand': s['right_hand'],
                             'confidence': s['confidence'],
                             'from_window': r['idx'], 'need_review': r['need_review']})
    proposal.sort(key=lambda x: x['start'])
    for i, s in enumerate(proposal):
        s['idx'] = i
        s['dur'] = round(s['end'] - s['start'], 2)
    (a.out / 'segments_proposal.json').write_text(json.dumps({
        'video': str(a.video), 'source': f'teacher-l123-{a.src_mode}',
        'note': 'L1 切分问法 + L2 结构校验 + L3 词表白名单；need_review=true 的段建议人工核对',
        'needs_review': sum(1 for s in proposal if s['need_review']),
        'segments': proposal}, ensure_ascii=False, indent=1), encoding='utf-8')

    print(f'\n候选段 {len(proposal)} 个（需人工复核 {sum(1 for s in proposal if s["need_review"])} 个）'
          f' ｜ 重试 {retried} 次 ｜ 有问题的窗口 {need_review}/{len(rows)}')
    for s in proposal[:20]:
        print(f'  {s["start"]:7.1f}-{s["end"]:7.1f}s ({s["dur"]:5.1f}s) '
              f'{s["state_change"]:<9} {s["action"][:34]}')

    if a.publish_name:
        # 一步发布到前端：写完 api_runs 后立刻生成 web/public/data/runs/<name>.json
        # 网页刷新（或点「重新读取清单」）即可在「结果来源」里看到这次结果。
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from publish_segments import publish as publish_run
            print('\n发布到前端…')
            publish_run(a.out, a.publish_name, clip_offset=a.out_offset,
                        duration=a.publish_duration)
        except Exception as exc:
            print(f'[!] 发布失败（结果仍已写入 {a.out}）：{exc}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
