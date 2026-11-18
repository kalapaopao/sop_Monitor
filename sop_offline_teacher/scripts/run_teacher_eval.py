"""用教师模型（Qwen-VL）对 GT 窗口做分类，第一次给出带真值的准确率与混淆矩阵。

与旧 `run_ablation.py` 的区别：
  · 窗口来自人工真值（gt_segments.json → teacher_windows.json），不是固定 3 s 切
  · **固定词表**：只允许从 6 个行为 + idle 里选，禁止自由命名
  · 强制输出单个 step_id，便于直接算混淆矩阵
  · 分证据档位 `--evidence none|skeleton|roi`，用于 P2 消融

用法：
    # 不调 API，只检查能抽出多少帧、会发多少次请求
    python scripts/run_teacher_eval.py --windows web/public/data/teacher_windows.json \
        --video vedio/v2.mp4 --out api_runs/teacher_eval --dry-run

    # 真正调用（需要 DASHSCOPE_API_KEY）
    $env:DASHSCOPE_API_KEY="..."
    python scripts/run_teacher_eval.py --windows ... --video vedio/v2.mp4 \
        --out api_runs/teacher_eval --evidence none
"""
from __future__ import annotations
import argparse, base64, json, os, sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

VOCAB = [
    ('Remove_eyepiece_assembly', '取镜座组件（取出并查看外观）'),
    ('Remove_the_upper_cover_of_the_eyepiece', '取镜座上盖（取出并查看外观）'),
    ('Assemble_the_mirror_base_cover', '组装镜座上盖（对上盖并压合到位）'),
    ('Assemble_the_mirror_base_sealing_plug', '组装镜座密封塞（对准孔位压入）'),
    ('Fasten_mirror_base', '紧固镜子镜座（持工具持续施力）'),
    ('Fasten_tripod_strap', '系上镜座扎带（绕带并收紧）'),
    ('idle', '空闲/非装配动作（等待、走动、与装配无关）'),
]

PROMPT = """你是工业装配现场的 SOP 动作分类器。你会看到同一工位的连续画面（按时间顺序）。

请把这段画面归类到下面 7 个类别中的**一个**，只能从表里选，禁止自创：

{vocab}

判定要求：
1. 只依据画面里能直接看到的证据；看不清时选 idle，不要猜。
2. 关注：左右手各自拿着什么、在做什么；工具是否与工件接触；工件状态是否变化。
3. 只输出 JSON，不要解释、不要 Markdown 代码块。

{state_hint}

输出格式：
{{"step_id": "上面 7 个之一", "confidence": 0-1 的小数, "reason": "不超过 20 字的中文依据"}}

{evidence}"""

STATE_HINT = """这三帧是**同一个动作的开始、中间、结束**。请重点比较**第一帧与最后一帧**：
工件是否发生了位置变化（例如零件从手里移到了夹具上、上盖是否已经贴合到位、密封塞是否已经压入）。
只有动作真正完成（状态发生改变）时才算该步骤；只是拿着、路过、或在准备，应选 idle 或其他更贴切的类别。"""


def b64_image(path: Path):
    return 'data:image/jpeg;base64,' + base64.b64encode(path.read_bytes()).decode()


def probe_seekable_end(video: Path, lo=0.0, hi=None):
    """二分探测实际能 seek 读取的最后时刻。

    实测 v2.mp4 容器声明 606.30s / nb_frames=14977，但 seek >599.12s 一律 read 失败
    （尾部约 7 秒不可解码）。不处理会让末尾窗口抽帧失败并被静默跳过。
    """
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if hi is None:
        hi = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
    lo = float(lo)
    for _ in range(22):
        mid = (lo + hi) / 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(mid * fps)))
        ok, _frame = cap.read()
        if ok:
            lo = mid
        else:
            hi = mid
    cap.release()
    return lo


def grab_frames(video: Path, times, out_dir: Path, quality=88, max_t=None):
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if max_t is not None:
        times = [min(t, max_t) for t in times]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, frame = cap.read()
        if not ok:
            paths.append(None)
            continue
        p = out_dir / f'{t:08.2f}.jpg'
        if not p.exists():
            cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        paths.append(p)
    cap.release()
    return paths


def evidence_text(kind: str, window, pose_rows, roi_index, stations):
    """把结构化证据转成给模型的文字（P2 消融用）。"""
    if kind == 'none':
        return ''
    lines = ['结构化证据：']
    if kind in ('skeleton', 'roi') and pose_rows:
        t0, t1 = window['start'], window['end']
        sel = [r for r in pose_rows if t0 <= r['timestamp'] <= t1]
        if sel:
            def mean_pts(k):
                pts = [p for r in sel for p in (r.get(k) or []) if p[2] > 0.18]
                if not pts:
                    return None
                return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
            L, R = mean_pts('left_hand'), mean_pts('right_hand')
            vis_l = sum(1 for r in sel if len(r.get('left_hand') or []) == 21) / len(sel)
            vis_r = sum(1 for r in sel if len(r.get('right_hand') or []) == 21) / len(sel)
            lines.append(f'- 窗口内 {len(sel)} 个姿态采样点（10Hz）')
            lines.append(f'- 左手可见比例 {vis_l:.2f}，右手可见比例 {vis_r:.2f}')
            if L and R:
                lines.append(f'- 两手平均距离 {((L[0]-R[0])**2 + (L[1]-R[1])**2) ** 0.5:.0f} 像素')
    if kind == 'roi' and stations:
        names = '、'.join(r['name'] for r in (stations.get('regions') or [])[:3])
        lines.append(f'- 已配置固定工位区域：{names}（画面里会有对应位置，请注意这些区域里的工件状态变化）')
    return '\n'.join(lines) if len(lines) > 1 else ''


def parse_pred(text: str):
    txt = (text or '').strip()
    if txt.startswith('```'):
        txt = txt.split('```')[1].lstrip('json').strip() if '```' in txt else txt
    try:
        d = json.loads(txt)
    except Exception:
        d = {}
        for code, _ in VOCAB:
            if code in (text or ''):
                d = {'step_id': code, 'confidence': 0.0, 'reason': 'fallback-parse'}
                break
    code = d.get('step_id')
    if code not in [c for c, _ in VOCAB]:
        code = 'idle'
        d['parse_note'] = 'invalid-or-missing step_id'
    d['step_id'] = code
    return d


def metrics(preds):
    labels = [c for c, _ in VOCAB]
    cm = {t: {p: 0 for p in labels} for t in labels}
    ok = 0
    for p in preds:
        t, q = p['truth'], p['pred']
        cm[t][q] += 1
        ok += int(t == q)
    n = len(preds)
    per = {}
    for l in labels:
        tp = cm[l][l]
        fp = sum(cm[t][l] for t in labels if t != l)
        fn = sum(cm[l][q] for q in labels if q != l)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[l] = {'tp': tp, 'fp': fp, 'fn': fn,
                  'precision': round(prec, 3), 'recall': round(rec, 3), 'f1': round(f1, 3)}
    return {'n': n, 'accuracy': round(ok / n, 3) if n else 0.0, 'confusion': cm, 'per_class': per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', type=Path, required=True)
    ap.add_argument('--video', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--evidence', choices=['none', 'skeleton', 'roi'], default='none')
    ap.add_argument('--pose', type=Path, default=Path('web/public/data/local_pose_2min.json'))
    ap.add_argument('--stations', type=Path, default=Path('configs/stations.json'))
    ap.add_argument('--model', default=os.getenv('DASHSCOPE_MODEL', 'qwen3-vl-plus'))
    ap.add_argument('--limit', type=int, default=0, help='只跑前 N 个窗口（省钱试跑）')
    ap.add_argument('--dry-run', action='store_true', help='不调 API，只抽帧并打印计划')
    ap.add_argument('--labels', default='', help='只跑这些标签（逗号分隔），用于省钱的小样本验证')
    a = ap.parse_args()

    wj = json.loads(a.windows.read_text(encoding='utf-8-sig'))
    windows = wj['windows']
    if a.labels:
        want = {x.strip() for x in a.labels.split(',') if x.strip()}
        windows = [w for w in windows if w['label'] in want]
        print(f'按 --labels 过滤后剩 {len(windows)} 个窗口：{sorted(want)}')
    if a.limit:
        windows = windows[:a.limit]

    frames_dir = a.out / 'frames'
    max_seek = probe_seekable_end(a.video)
    declared = None
    try:
        import cv2
        _cap = cv2.VideoCapture(str(a.video))
        declared = _cap.get(cv2.CAP_PROP_FRAME_COUNT) / (_cap.get(cv2.CAP_PROP_FPS) or 25.0)
        _cap.release()
    except Exception:
        pass
    tail = ''
    if declared and declared - max_seek > 1:
        tail = f'（容器声明 {declared:.2f}s，尾部不可解码）'
    print(f'可 seek 边界 {max_seek:.2f}s{tail}')
    plans, missing, clamped = [], 0, 0
    for w in windows:
        if max(w['frames'], default=0) > max_seek:
            clamped += 1
        paths = grab_frames(a.video, w['frames'], frames_dir, max_t=max_seek)
        if any(p is None for p in paths):
            missing += 1
            continue
        plans.append({'window': w, 'frames': [str(p) for p in paths]})

    vocab_txt = '\n'.join(f'- {c}：{d}' for c, d in VOCAB)
    print(f'窗口 {len(windows)} 个 → 成功抽帧 {len(plans)} 个'
          f'（失败 {missing}，帧时间被夹到可解码边界的 {clamped}）')
    print(f'请求数 = {len(plans)}，图片数 = {sum(len(p["frames"]) for p in plans)}')

    if a.dry_run:
        manifest = a.out / 'request_manifest.json'
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({
            'video': str(a.video), 'evidence': a.evidence, 'model': a.model,
            'window_count': len(plans),
            'requests': [{'idx': i, 'window': p['window'], 'frames': p['frames']}
                         for i, p in enumerate(plans)],
            'prompt_vocab': vocab_txt,
        }, ensure_ascii=False, indent=1), encoding='utf-8')
        print('dry-run：已写出请求清单', manifest, '（未调用 API）')
        return 0

    key = os.getenv('DASHSCOPE_API_KEY')
    if not key:
        print('缺少 DASHSCOPE_API_KEY，无法调用。先设置环境变量，或加 --dry-run 只抽帧。')
        return 2
    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=os.getenv('DASHSCOPE_BASE_URL',
                    'https://dashscope.aliyuncs.com/compatible-mode/v1'))

    pose_rows = json.loads(a.pose.read_text(encoding='utf-8-sig'))['frames'] if a.pose.exists() else None
    stations = json.loads(a.stations.read_text(encoding='utf-8-sig')) if a.stations.exists() else None
    stations = (stations or {}).get('stations', [{}])[0] if stations else None

    preds = []
    for i, p in enumerate(plans):
        w = p['window']
        ev = evidence_text(a.evidence, w, pose_rows, None, stations) if a.evidence != 'none' else ''
        hint = STATE_HINT if w['kind'] == 'inside' else ''
        content = [{'type': 'text', 'text': PROMPT.format(
            vocab=vocab_txt, evidence=ev or '（无额外结构化证据）', state_hint=hint)}]
        for t, fp in zip(w['frames'], p['frames']):
            content.append({'type': 'text', 'text': f'画面时间 {t:.2f} 秒'})
            content.append({'type': 'image_url', 'image_url': {'url': b64_image(Path(fp))}})
        try:
            r = client.chat.completions.create(model=a.model, temperature=0.0,
                                               messages=[{'role': 'user', 'content': content}])
            raw = r.choices[0].message.content
        except Exception as exc:
            raw = f'__ERROR__ {exc}'
        d = parse_pred(raw)
        preds.append({'idx': i, 'window': w, 'truth': w['label'], 'pred': d['step_id'],
                      'confidence': d.get('confidence'), 'reason': d.get('reason'),
                      'kind': w['kind'], 'raw': raw[:800]})
        print(f'  [{i + 1}/{len(plans)}] {w["start"]:7.1f}-{w["end"]:7.1f}s '
              f'truth={w["label"][:22]:<22} pred={d["step_id"][:22]}')

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'predictions.json').write_text(json.dumps(preds, ensure_ascii=False, indent=1),
                                            encoding='utf-8')
    m_all = metrics(preds)
    m_norm = metrics([p for p in preds if p['kind'] != 'short'] or preds)
    m_short = metrics([p for p in preds if p['kind'] == 'short']) if any(p['kind'] == 'short' for p in preds) else None
    summary = {'evidence': a.evidence, 'model': a.model,
               'all': m_all, 'exclude_short': m_norm, 'only_short': m_short}
    (a.out / 'metrics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                        encoding='utf-8')

    print(f'\n总体准确率 {m_all["accuracy"]:.3f}（n={m_all["n"]}）')
    print(f'剔除短段后 {m_norm["accuracy"]:.3f}', end='')
    if m_short:
        print(f' | 仅短段 {m_short["accuracy"]:.3f}（n={m_short["n"]}）')
    else:
        print()
    print('\n逐类：')
    for k, v in m_all['per_class'].items():
        if v['tp'] + v['fn'] + v['fp']:
            print(f'  {k:<46} P={v["precision"]:.2f} R={v["recall"]:.2f} F1={v["f1"]:.2f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
