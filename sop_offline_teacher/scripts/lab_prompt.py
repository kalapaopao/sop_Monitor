"""最小练手脚本：只做一件事——把一小段画面发给教师模型，打印「原文 → 解析 → 校验」。

这个脚本是为"自己动手改提示词"准备的，刻意写得很短、很直白：
  · 顶部三个常量（可用物件白名单 / 提示词 / 校验规则）就是你要改的地方；
  · `--print-prompt` 只打印最终提示词，不花钱、不需要 API Key，改完立刻能看效果；
  · 真跑时只发 1~3 帧，一次调用，方便快速对比。

用法：
    # ① 只看看提示词长什么样（不花钱）
    python scripts/lab_prompt.py --print-prompt

    # ② 真跑一次（需要 DASHSCOPE_API_KEY）
    python scripts/lab_prompt.py --video vedio/v2.mp4 --start 312 --frames 3

    # ③ 换个时间段再跑（也可以改 WHITELIST / PROMPT 后重跑对比）
    python scripts/lab_prompt.py --video vedio/v2.mp4 --start 330 --frames 3 --save api_runs/lab/330.json
"""
from __future__ import annotations
import argparse, base64, json, os, sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ===== ① 你要改的东西（改完用 --print-prompt 立刻看效果）=========================

# 可用物件白名单：模型只能使用这些名词，出现别的就判不合格。
# 试着加一个「镜片」或删掉「夹具」，再跑同一个时间段，看输出怎么变。
WHITELIST = ['镜座组件', '镜座上盖', '镜子镜座', '镜座扎带', '镜座密封塞', '螺丝刀', '夹具', '手']

# 允许的 state_change 取值
STATES = ['inserted', 'removed', 'fastened', 'none']

PROMPT = """你是工业装配现场的 SOP 动作教师。下面按时间先后给出同一工位的 {n} 张画面，
覆盖视频的 {start:.1f}–{end:.1f} 秒。

【可用物件白名单】（只能使用这些名词，禁止自创同义词）
{whitelist}
若画面里出现白名单以外的物体，写「未识别物件」，不要猜名字。

【任务】在这段画面里切分动作：如果中途换了动作，给出每段的起止秒数；整段同一个动作就只给一段。

【判断要求】
1. action：简体中文，≤30 字，必须对应「手的动作 + 物件状态」。
2. state_change：只能取 {states}。只有物件状态真的改变才填前三者，否则 none。
3. evidence_i_used：写出依据画面里哪一处判断（≤20 字）；找不到依据就把 action 写「无法确认」、confidence 填 0。
4. left_hand / right_hand：简体中文 ≤15 字；看不到写「未看到」。
5. 只输出 JSON，不要解释，不要 Markdown 代码块。

【输出格式】
{{"segments":[{{"start":12.0,"end":18.4,"action":"中文一句话","state_change":"none",
 "evidence_i_used":"中文","left_hand":"中文","right_hand":"中文","confidence":0.8}}]}}"""

# ===== ② 下面一般不用改 =========================================================


def build_prompt(start, end, n):
    return PROMPT.format(n=n, start=start, end=end,
                         whitelist=' / '.join(WHITELIST),
                         states=' | '.join(STATES))


def frames_at(video: Path, start: float, end: float, n: int):
    """在 [start, end] 内均匀取 n 帧，返回 [(时间, 图片路径)]。"""
    import cv2
    out_dir = Path('api_runs/lab/frames')
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    picks = []
    for i in range(n):
        t = start + (end - start) * (i + 0.5) / n if n > 1 else (start + end) / 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, frame = cap.read()
        if not ok:
            continue
        p = out_dir / f'{t:08.2f}.jpg'
        if not p.exists():
            cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        picks.append((round(t, 2), p))
    cap.release()
    return picks


def b64(p: Path):
    return 'data:image/jpeg;base64,' + base64.b64encode(p.read_bytes()).decode()


def validate(parsed):
    """校验：JSON 结构 / 枚举 / 白名单 / 中文 / 依据字段。返回问题列表。"""
    problems = []
    segs = parsed.get('segments')
    if not isinstance(segs, list) or not segs:
        return ['segments 不是非空数组'], 0
    for i, s in enumerate(segs):
        tag = f'segments[{i}]'
        if not isinstance(s.get('start'), (int, float)) or not isinstance(s.get('end'), (int, float)):
            problems.append(f'{tag} 缺少数值型 start/end')
        if s.get('state_change') not in STATES:
            problems.append(f'{tag} state_change 非法：{s.get("state_change")!r}')
        if not any('\u4e00' <= ch <= '\u9fff' for ch in str(s.get('action', ''))):
            problems.append(f'{tag} action 不是中文')
        if not str(s.get('evidence_i_used', '')).strip():
            problems.append(f'{tag} 缺 evidence_i_used（这条最容易暴露编造）')
        # 白名单：只检查"白名单以外的名词"是否被写进 action，用最小实现避免误伤
        act = str(s.get('action', ''))
        for banned in ['外壳', '电机', '盖板', '线束', '扫码枪', '塑料袋', '支架']:
            if banned in act and banned not in ''.join(WHITELIST):
                problems.append(f'{tag} action 里出现白名单外的名词：{banned}')
    return problems, len(segs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', type=Path, default=Path('vedio/v2.mp4'))
    ap.add_argument('--start', type=float, default=312.0, help='窗口起点（v2 的秒）')
    ap.add_argument('--end', type=float, default=None, help='窗口终点，默认 start+6')
    ap.add_argument('--frames', type=int, default=3)
    ap.add_argument('--print-prompt', action='store_true', help='只打印提示词，不调 API')
    ap.add_argument('--save', type=Path, default=None, help='把原文+解析结果存成 json')
    ap.add_argument('--model', default=os.getenv('DASHSCOPE_MODEL', 'qwen3-vl-plus'))
    a = ap.parse_args()

    end = a.end if a.end is not None else a.start + 6.0
    prompt = build_prompt(a.start, end, a.frames)

    if a.print_prompt:
        print('=' * 70)
        print(prompt)
        print('=' * 70)
        print(f'\n字符数 {len(prompt)} ｜ 白名单 {len(WHITELIST)} 项 ｜ 状态枚举 {STATES}')
        print('改完提示词可以再跑一次本命令，不花钱。')
        return 0

    key = os.getenv('DASHSCOPE_API_KEY')
    if not key:
        print('缺少 DASHSCOPE_API_KEY。想先看提示词可加 --print-prompt（不需要 Key）。')
        return 2

    picks = frames_at(a.video, a.start, end, a.frames)
    if not picks:
        sys.exit('抽帧失败，检查 --video / --start')
    print(f'抽帧 {len(picks)} 张：' + '、'.join(f'{t}s' for t, _ in picks))

    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=os.getenv('DASHSCOPE_BASE_URL',
                    'https://dashscope.aliyuncs.com/compatible-mode/v1'))
    content = [{'type': 'text', 'text': prompt}]
    for t, p in picks:
        content.append({'type': 'text', 'text': f'画面时间 {t} 秒'})
        content.append({'type': 'image_url', 'image_url': {'url': b64(p)}})

    resp = client.chat.completions.create(model=a.model, temperature=0.0,
                                          messages=[{'role': 'user', 'content': content}])
    raw = resp.choices[0].message.content
    print('\n--- 模型原文 ---')
    print(raw[:1500])

    txt = raw.strip()
    if txt.startswith('```'):
        parts = txt.split('```')
        txt = parts[1].lstrip('json').strip() if len(parts) > 1 else txt
    try:
        parsed = json.loads(txt)
    except Exception as exc:
        print(f'\n[!] JSON 解析失败：{exc}')
        parsed = {}

    problems, n = validate(parsed) if parsed else (['无法解析，跳过校验'], 0)
    print('\n--- 解析结果 ---')
    print(json.dumps(parsed, ensure_ascii=False, indent=1)[:1500])
    print(f'\n--- 校验：{n} 段，{len(problems)} 个问题 ---')
    for p in problems:
        print('  -', p)
    if not problems:
        print('  OK：结构、枚举、中文、依据字段、白名单都通过')

    if a.save:
        a.save.parent.mkdir(parents=True, exist_ok=True)
        a.save.write_text(json.dumps({'window': [a.start, end], 'prompt': prompt,
                                      'raw': raw, 'parsed': parsed, 'problems': problems},
                                     ensure_ascii=False, indent=1), encoding='utf-8')
        print('\n已保存到', a.save)
    return 0


if __name__ == '__main__':
    sys.exit(main())
