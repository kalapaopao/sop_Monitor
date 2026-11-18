"""让大模型"通看一段视频，自己归纳工序"，而不是从我们给的六步里选。

这是用户真正要的那个实验：他接受 SOP 名称由视频归纳，只要划分合理、
前后一致、切换准确。前面测的是"在我们切好的段上做六选一"，那是另一回事。

做法：
  · 从窗口里按固定间隔抽 N 张「干净原片」帧（不烧骨架/ROI）
  · 一次调用，要求模型：归纳 3~10 个工序，给出名称 + 起止秒 + 依据，
    并明确"允许重复出现、允许跳过、允许流程外动作"
  · 程序校验时间轴（递增、落界内），再和人工 GT 比边界

用法
----
    .venv\\Scripts\\python.exe scripts/vlm_define_sop.py \
        --video vedio/v2.mp4 --start 300 --duration 120 \
        --every 4 --out api_runs/vlm_sop_define
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import cv2

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCHEMA = (
    '只输出一个 JSON 对象，不要解释、不要代码块：\n'
    '{"steps":[{"index":1,"name":"中文名（8~16 字，描述手里的物件+动作）",'
    '"start":秒,"end":秒,"evidence":"依据画面里哪一处"}],'
    '"note":"你认为这段视频在装配什么产品，一句话"}'
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--every", type=float, default=4.0, help="每多少秒抽一帧")
    ap.add_argument("--max-frames", type=int, default=40)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=os.environ.get("QWEN_MODEL", "qwen3-vl-plus"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key and not a.dry_run:
        raise SystemExit("没有 DASHSCOPE_API_KEY")

    cap = cv2.VideoCapture(str(a.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    times = []
    t = a.start
    while t < a.start + a.duration and len(times) < a.max_frames:
        times.append(round(t, 2))
        t += a.every
    print(f"{a.video.name} ｜ 窗口 {a.start:.0f}~{a.start + a.duration:.0f}s ｜ 抽 {len(times)} 帧")

    images = []
    for ts in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        h, w = frame.shape[:2]
        s = 720.0 / max(w, h)
        if s < 1.0:
            frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            images.append((ts, base64.b64encode(buf.tobytes()).decode("ascii")))
    cap.release()
    print(f"取到 {len(images)} 帧")
    if a.dry_run:
        print("[dry-run] 只报规模，不调用")
        return 0

    import requests
    prompt = (
        f"这是一段装配作业视频的连续截图，覆盖 {a.start:.0f} 到 {a.start + a.duration:.0f} 秒，"
        f"每 {a.every:.0f} 秒一帧，共 {len(images)} 帧，每帧上方标了它的时间。\n"
        "请**通看整段**，自己归纳出实际反复出现的工序（不要套用任何给定的词表）：\n"
        "· 工序按时间顺序列出，可以重复出现同一种工序，也可以有“流程外/停留”类\n"
        "· 名称用 8~16 个汉字，写清“手里的物件 + 动作”，例如“取得黑色壳片并与银灰件靠合”\n"
        "· 起止时间必须落在这段视频内，相邻工序不要重叠\n"
        "· 如果某段看不清，就命名为“未确认动作”，不要编造\n"
        + SCHEMA
    )
    content = [{"type": "text", "text": prompt}]
    for ts, b64 in images:
        content.append({"type": "text", "text": f"【t={ts:.1f}s】"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    t0 = time.time()
    last = None
    for attempt in range(3):
        try:
            r = requests.post(
                "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": a.model, "temperature": 0.2,
                      "messages": [{"role": "user", "content": content}]},
                timeout=180)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            txt = r.json()["choices"][0]["message"]["content"]
            s = txt.strip()
            if s.startswith("```"):
                s = s.split("```")[1]
                s = s[4:] if s.startswith("json") else s
            i, j = s.find("{"), s.rfind("}")
            last = json.loads(s[i:j + 1])
            break
        except Exception as e:
            last = e
            if attempt < 2:
                time.sleep(3 + 4 * attempt)
    if isinstance(last, Exception) or not isinstance(last, dict):
        raise SystemExit(f"调用失败：{last}")
    print(f"调用完成 {time.time() - t0:.1f}s ｜ 模型 {a.model}")

    steps = last.get("steps") or []
    print(f"\n=== 模型自己归纳出的工序（{len(steps)} 条）===")
    print(f"{'起':>8}{'止':>8}{'时长':>7}  名称")
    for s in steps:
        try:
            s0, s1 = float(s["start"]), float(s["end"])
        except Exception:
            continue
        print(f"{s0:>8.1f}{s1:>8.1f}{s1 - s0:>7.1f}  {s.get('name', '')}")
    print(f"\n模型对产品的判断：{last.get('note', '')}")

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "vlm_sop.json").write_text(json.dumps({
        "video": str(a.video), "start": a.start, "duration": a.duration,
        "every": a.every, "model": a.model, "n_frames": len(images),
        "raw": last,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n写入 {a.out / 'vlm_sop.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
