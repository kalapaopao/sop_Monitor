"""两个补充实验：① 给易混淆反例能不能纠正判错  ② 原生视频输入的帧数上限在哪

实验一（negative）：复现 Codex 记录的具体错误——v2 76~82 秒问「中央白布定位座上
有没有工件」，基线问法与加"反例提示"的问法对比，看答案会不会翻转。
正确参考取 Codex 的人工画面复核（工件仍在手中、定位座为空）。

实验二（videolimit）：原生视频输入比多图便宜约 2 倍/帧，但 150 帧会 500。
逐档试 25/50/75/100 帧，找出还返回 200 的上限。

用法
----
    .venv\\Scripts\\python.exe scripts/vlm_followup_experiments.py --mode negative
    .venv\\Scripts\\python.exe scripts/vlm_followup_experiments.py --mode videolimit
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

BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


def post(payload, key, timeout=180):
    import requests
    return requests.post(BASE, headers={"Authorization": f"Bearer {key}"},
                         json=payload, timeout=timeout)


def frame_b64(video: Path, t: float, max_side=896):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
    ok, fr = cap.read()
    cap.release()
    if not ok or fr is None:
        return None
    h, w = fr.shape[:2]
    s = max_side / max(w, h)
    if s < 1.0:
        fr = cv2.resize(fr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None


def ask_images(model, key, imgs, prompt):
    content = [{"type": "text", "text": prompt}]
    for t, b64 in imgs:
        content.append({"type": "text", "text": f"【t={t:.1f}s】"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    r = post({"model": model, "temperature": 0.0,
              "messages": [{"role": "user", "content": content}]}, key)
    if r.status_code != 200:
        return r.status_code, r.text[:200], None
    j = r.json()
    return 200, j["choices"][0]["message"]["content"], j.get("usage")


def mode_negative(a, key):
    video = Path("vedio/v2.mp4")
    times = [76.0, 78.0, 80.0, 82.0]
    imgs = [(t, b) for t in times for b in [frame_b64(video, t)] if b]
    print(f"取到 {len(imgs)} 帧（v2 {times[0]:.0f}~{times[-1]:.0f}s，干净原片）")
    print("人工参考（Codex 复核）：工件仍在操作者手中，中央白布定位座为空")

    q = ("这是同一工位连续几秒的画面。请只回答一个问题："
         "中央那块白布定位座上当前有没有工件？\n"
         '只输出 JSON：{"jig_state":"occupied|empty","evidence":"依据哪一处"}')

    hint = ("\n\n提示（易混淆点）：操作者手里拿着工件、还没放到定位座上时，"
            "定位座应判为 empty。不要因为画面里出现了工件就判 occupied——"
            "要判断的是【定位座上】有没有。")

    results = {}
    for tag, prompt in (("基线", q), ("加反例提示", q + hint)):
        st, txt, usage = ask_images(a.model, key, imgs, prompt)
        print(f"\n--- {tag} ---  HTTP {st}")
        print("  " + str(txt)[:400].replace("\n", " "))
        if usage:
            print(f"  tokens: prompt {usage.get('prompt_tokens')} / completion {usage.get('completion_tokens')}")
        results[tag] = {"status": st, "answer": txt, "usage": usage}
        time.sleep(2)

    out = Path("api_runs/vlm_followup")
    out.mkdir(parents=True, exist_ok=True)
    (out / "negative_test.json").write_text(json.dumps(
        {"video": str(video), "times": times, "human_reference": "定位座为空；工件在手中",
         "model": a.model, "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n写入 {out / 'negative_test.json'}")
    return 0


def mode_videolimit(a, key):
    video = Path("vedio/v2.mp4")
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.release()
    out = Path("api_runs/vlm_followup")
    out.mkdir(parents=True, exist_ok=True)
    log = []
    for n in (25, 50, 75, 100):
        step = 4.0                      # 每 4 秒一帧，和 Codex 的口径一致
        times = [80.0 + i * step for i in range(n)]
        imgs = []
        ok_frames = 0
        cap = cv2.VideoCapture(str(video))
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, fr = cap.read()
            if not ok or fr is None:
                continue
            h, w = fr.shape[:2]
            s = 448.0 / max(w, h)
            fr = cv2.resize(fr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            ok2, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 72])
            if ok2:
                imgs.append(base64.b64encode(buf.tobytes()).decode("ascii"))
                ok_frames += 1
        cap.release()
        # 原生视频：Qwen 兼容模式按"视频帧序列"送
        content = [{"type": "text", "text": "画面是一段装配作业。用一句话说这段大致在做什么，不要展开。"}]
        content.append({"type": "video", "video": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}} for b in imgs
        ]})
        st, txt, usage = None, None, None
        try:
            r = post({"model": a.model, "temperature": 0.0,
                      "messages": [{"role": "user", "content": content}]}, key, timeout=240)
            st = r.status_code
            txt = (r.json().get("choices", [{}])[0].get("message", {}).get("content")
                   if st == 200 else r.text[:160])
            usage = r.json().get("usage") if st == 200 else None
        except Exception as e:
            st, txt = "EXC", f"{type(e).__name__}: {e}"[:160]
        vt = (usage or {}).get("prompt_tokens_details", {}).get("video_tokens")
        print(f"  {ok_frames:>4} 帧 → HTTP {st}  video_tokens={vt}  {str(txt)[:80]}")
        log.append({"frames": ok_frames, "status": st, "video_tokens": vt,
                    "usage": usage, "answer": txt})
        if st != 200:
            break
        time.sleep(2)
    (out / "videolimit.json").write_text(json.dumps(
        {"video": str(video), "every_sec": 4.0, "model": a.model, "runs": log},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n写入 {out / 'videolimit.json'}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["negative", "videolimit"], required=True)
    ap.add_argument("--model", default=os.environ.get("QWEN_MODEL", "qwen3-vl-plus"))
    a = ap.parse_args()
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise SystemExit("没有 DASHSCOPE_API_KEY")
    return mode_negative(a, key) if a.mode == "negative" else mode_videolimit(a, key)


if __name__ == "__main__":
    sys.exit(main())
