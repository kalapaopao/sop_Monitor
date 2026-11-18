# ⚠️ 遗留脚本（旧路线：让模型自由命名动作，无 SOP 步骤表）。
# 已被取代：
#   - 自由分段/直出 step_id  → scripts/run_teacher_segments.py
#   - 逐段判定（GT/本地边界）  → scripts/run_teacher_per_segment.py
# 保留仅为历史参考。新工作请勿基于本文件。

"""Run Qwen3-VL-Plus on sampled video frames and render a review page."""
import argparse
import base64
import json
import os
from pathlib import Path

import cv2
from openai import OpenAI


PROMPT = """你是工业现场动作分析教师模型。分析连续视频帧，自己命名动作，不使用固定动作列表。
请分别描述当前动作、动作阶段、左手、右手、工具、工件、手与目标的关系、相比前一窗口的变化。
只把画面直接支持的内容写入 observed，把合理推断写入 inferred，无法确认写入 unknown。
输出 JSON，字段为 action、phase、left_hand、right_hand、tool、object、relation、change、observed、inferred、unknown。
不要把“疑似拧紧”写成已确认的“拧紧”。"""


def sample_frames(video: Path, out: Path, start_offset: float, duration: float, stride: float, count: int):
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    frames = []
    index = 0
    start = start_offset
    end_time = start_offset + duration
    while start < end_time:
        offsets = [stride * i / max(1, count - 1) for i in range(count)]
        paths = []
        for offset in offsets:
            timestamp = start + offset
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            path = out / f"frame_{index:04d}_{int(offset * 1000):04d}.jpg"
            cv2.imwrite(str(path), frame)
            paths.append((timestamp, path))
        if paths:
            frames.append((start, min(start + stride, end_time), paths))
        start += stride
        index += 1
    cap.release()
    return frames


def image_data(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def call_model(client: OpenAI, model: str, start: float, end: float, frames):
    content = [{"type": "text", "text": f"窗口 {start:.1f}-{end:.1f} 秒。\n{PROMPT}"}]
    for timestamp, path in frames:
        content.append({"type": "text", "text": f"画面时间 {timestamp:.2f} 秒"})
        content.append({"type": "image_url", "image_url": {"url": image_data(path)}})
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0.1,
    )
    return response.choices[0].message.content


def render_report(out: Path, video: Path, results):
    cards = []
    for item in results:
        images = "".join(f'<img src="frames/{Path(p).name}">' for p in item["frame_paths"])
        cards.append(f'<article><h2>{item["start"]:.0f}s–{item["end"]:.0f}s</h2><div class="imgs">{images}</div><pre>{item["response"]}</pre></article>')
    html = """<!doctype html><meta charset="utf-8"><title>API教师视频分析</title>
<style>body{font-family:Arial,'Microsoft YaHei';background:#07111e;color:#e8f0f7;margin:20px}video{width:75%;background:#000}article{background:#132238;padding:14px;margin:14px 0;border-radius:10px}h2{color:#f4c15d}.imgs{display:flex;gap:8px}.imgs img{width:30%;height:130px;object-fit:cover}pre{white-space:pre-wrap;background:#0b1522;padding:10px}</style>
<h1>Qwen3-VL-Plus API教师分析</h1><p>视频播放器与动作片段结果对照。</p>
<video controls src="VIDEO_SRC"></video>CARDS""".replace("VIDEO_SRC", video.as_uri()).replace("CARDS", "".join(cards))
    (out / "report.html").write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--stride", type=float, default=2.0)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--model", default=os.getenv("DASHSCOPE_MODEL", "qwen3-vl-plus"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise SystemExit("DASHSCOPE_API_KEY is missing")
    args.output.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"], base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    sampled = sample_frames(args.video, args.output / "frames", args.start, args.duration, args.stride, args.frames)
    results = []
    for start, end, frames in sampled:
        response = call_model(client, args.model, start, end, frames)
        results.append({"start": start, "end": end, "response": response, "frame_paths": [str(p) for _, p in frames]})
        (args.output / "raw_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"completed {start:.1f}-{end:.1f}s", flush=True)
    render_report(args.output, args.video, results)


if __name__ == "__main__":
    main()
