"""片段语义补全：本地模型负责"切好边界 + 判好 step_id"，Qwen 只负责"描述这一段"。

为什么这样分工（有实测依据）：
  · 让 Qwen 自己切边界+判六步：clean 18.2%、multi 9.1%（已证伪）
  · 本地 MLP+HMM：帧级 88.6%（边界与 step_id 已可信）
  · 让 Qwen 描述"一个边界已确定的片段"，任务难度大幅下降，且它的语言能力正好用在这

Qwen 被明确禁止：改 step_id、改边界。它只输出五个字段：
  action / left_hand{action,holding,target} / right_hand{action,holding,target}
  / state_change / evidence / consistent（是否与给定步骤一致）

用法：
    # 0 成本：只看会发什么
    python scripts/enrich_segments_qwen.py --run web/public/data/runs/p3_mlp_hmm.json --dry-run

    # 真跑（每个片段 1 次调用）
    $env:DASHSCOPE_API_KEY="..."
    python scripts/enrich_segments_qwen.py --run web/public/data/runs/p3_mlp_hmm.json \
        --out api_runs/p3_enriched --publish-name p3_enriched
"""
from __future__ import annotations
import argparse, base64, json, os, sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROMPT = """你是工业装配现场的动作描述员。下面给出同一个工位的 {n} 张画面，
覆盖 {start:.2f}–{end:.2f} 秒，这是**一个已经被切好的动作片段**。

【已知信息（不要改动，也不要质疑边界）】
- 这一步的 SOP 编号：{step_id}
- 这一步的 SOP 名称：{step_name}
- 判定标准：{criteria}

【你的任务】只描述这一段画面里实际发生了什么，输出五个字段：

1. action：简体中文 ≤30 字，写"手在做什么 + 物件状态"
2. left_hand：{{"action":"≤12字，左手在做什么","holding":"≤10字，左手拿着什么；空手写'空手'","target":"≤10字，左手作用于什么"}}
3. right_hand：同上结构，写右手
4. state_change：只能取 inserted | removed | fastened | none（物件状态是否真的改变）
5. evidence：≤25 字，你依据画面哪里判断的
6. consistent：true / false —— **画面看起来是否真的像这个 SOP 步骤**；
   如果看起来更像别的动作或看不清，填 false，并在 evidence 里说明

【硬性要求】
- 只能用简体中文描述，禁止英文单词（JSON 的键名除外）
- 手里拿的东西看不清就写"未识别"，不要猜品牌/型号
- 只输出 JSON，不要解释，不要 Markdown 代码块

输出格式：
{{"action":"…","left_hand":{{"action":"…","holding":"…","target":"…"}},
 "right_hand":{{"action":"…","holding":"…","target":"…"}},
 "state_change":"none","evidence":"…","consistent":true}}"""


def b64(p: Path):
    return "data:image/jpeg;base64," + base64.b64encode(Path(p).read_bytes()).decode()


def grab(video: Path, times, out_dir: Path, quality=88):
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    out_dir.mkdir(parents=True, exist_ok=True)
    res = []
    for t in times:
        if t < 0 or (n and t * fps > n - 2):
            res.append(None)
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, frame = cap.read()
        if not ok:
            res.append(None)
            continue
        p = out_dir / f"{t:08.2f}.jpg"
        if not p.exists():
            cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        res.append(p)
    cap.release()
    return res


def parse(txt):
    t = (txt or "").strip()
    if t.startswith("```"):
        parts = t.split("```")
        t = parts[1].lstrip("json").strip() if len(parts) > 1 else t
    try:
        return json.loads(t)
    except Exception:
        return {}


def hand_obj(v):
    if isinstance(v, dict):
        return {"action": str(v.get("action") or ""),
                "holding": str(v.get("holding") or ""),
                "target": str(v.get("target") or "")}
    return {"action": str(v or ""), "holding": "", "target": ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="已发布的 run json（含 steps）")
    ap.add_argument("--steps", type=Path, default=Path("configs/sop_steps.json"))
    ap.add_argument("--video", type=Path, default=Path("web/public/video/v2_5m_7m_pose.mp4"))
    ap.add_argument("--frames", type=int, default=2, help="每段取几帧")
    ap.add_argument("--min-dur", type=float, default=1.5, help="短于此的段不做补全（信息不足）")
    ap.add_argument("--skip-idle", action="store_true", help="跳过 idle 段（不花调用）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model", default=os.getenv("DASHSCOPE_MODEL", "qwen3-vl-plus"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--publish-name", default="")
    a = ap.parse_args()

    run = json.loads(a.run.read_text(encoding="utf-8-sig"))
    steps_cfg = json.loads(a.steps.read_text(encoding="utf-8-sig"))
    crit = {}
    for s in steps_cfg.get("steps", []) + steps_cfg.get("fallback_steps", []):
        c = s.get("criteria") or {}
        crit[s["step_id"]] = (f"手={c.get('hand', '—')}；物件={c.get('object', '—')}；"
                              f"工具={c.get('tool', '—')}") if c else "无"

    segs = []
    for st in run.get("steps") or []:
        dur = float(st["end"]) - float(st["start"])
        if dur < a.min_dur:
            continue
        if a.skip_idle and st.get("step_id") == "idle":
            continue
        segs.append(st)
    if a.limit:
        segs = segs[:a.limit]
    total_imgs = len(segs) * a.frames
    print(f"待补全片段 {len(segs)} 个 ｜ 每段 {a.frames} 帧 → {total_imgs} 张图 / {len(segs)} 次调用")
    print(f"视频 {a.video} ｜ 跳过短于 {a.min_dur}s 的段" + ("（且跳过 idle）" if a.skip_idle else ""))

    frames_dir = a.out / "frames" if a.out else Path("api_runs/enrich_frames")
    plans = []
    for st in segs:
        s, e = float(st["start"]), float(st["end"])
        inset = min(0.15, (e - s) * 0.1)
        lo, hi = s + inset, e - inset
        times = [round(lo + (hi - lo) * (i + 0.5) / a.frames, 3) for i in range(a.frames)]
        paths = grab(a.video, times, frames_dir)
        if all(p is None for p in paths):
            continue
        plans.append({"seg": st, "times": times, "paths": paths})

    if a.dry_run:
        out = a.out or Path("api_runs/enrich_plan")
        out.mkdir(parents=True, exist_ok=True)
        (out / "plan.json").write_text(json.dumps({
            "run": str(a.run), "video": str(a.video), "frames": a.frames,
            "segments": [{"start": p["seg"]["start"], "end": p["seg"]["end"],
                          "step_id": p["seg"].get("step_id"),
                          "times": [round(t, 2) for t in p["times"]],
                          "images": [str(x) if x else None for x in p["paths"]]}
                         for p in plans]}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"dry-run：计划写到 {out / 'plan.json'}（未调用 API）")
        for p in plans[:6]:
            print(f"  {p['seg']['start']:6.2f}-{p['seg']['end']:6.2f}s "
                  f"{p['seg'].get('step_id'):<8} 帧 {[round(t,2) for t in p['times']]}")
        return 0

    key = os.getenv("DASHSCOPE_API_KEY")
    if not key:
        print("缺少 DASHSCOPE_API_KEY（可先 --dry-run 看计划）")
        return 2
    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=os.getenv("DASHSCOPE_BASE_URL",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1"))

    enriched, inconsistent = [], 0
    for i, p in enumerate(plans):
        st = p["seg"]
        sid = st.get("step_id") or "unknown"
        prompt = PROMPT.format(n=len([x for x in p["paths"] if x]),
                               start=float(st["start"]), end=float(st["end"]),
                               step_id=sid, step_name=st.get("name") or sid,
                               criteria=crit.get(sid, "无"))
        content = [{"type": "text", "text": prompt}]
        for t, fp in zip(p["times"], p["paths"]):
            if not fp:
                continue
            content.append({"type": "text", "text": f"画面时间 {t:.2f} 秒"})
            content.append({"type": "image_url", "image_url": {"url": b64(fp)}})
        try:
            r = client.chat.completions.create(model=a.model, temperature=0.0,
                                               messages=[{"role": "user", "content": content}])
            raw = r.choices[0].message.content
        except Exception as exc:
            raw = f"__ERROR__ {exc}"
        d = parse(raw)
        ok = bool(d.get("consistent", True))
        inconsistent += int(not ok)
        item = {**st,
                "action": d.get("action") or st.get("action") or "",
                "left_hand": hand_obj(d.get("left_hand")),
                "right_hand": hand_obj(d.get("right_hand")),
                "state_change": d.get("state_change") or "none",
                "evidence": d.get("evidence") or "",
                "qwen_consistent": ok,
                "need_review": bool(st.get("need_review")) or (not ok)}
        enriched.append(item)
        print(f"  [{i+1}/{len(plans)}] {st['start']:6.2f}-{st['end']:6.2f}s {sid:<8} "
              f"{'OK ' if ok else '冲突'} {str(item['action'])[:26]}")

    out = a.out or Path("api_runs/enriched")
    out.mkdir(parents=True, exist_ok=True)
    result = {"version": 1, "video": run.get("video"), "duration": run.get("duration"),
              "source": f"enriched:{a.run.stem}",
              "note": f"边界与 step_id 来自本地模型；action/左右手/证据由 Qwen 补全"
                      f"（{len(enriched)} 段，{inconsistent} 段被标为与步骤不一致）",
              "coverage": run.get("coverage"),
              "steps": enriched}
    (out / "segments_enriched.json").write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                                encoding="utf-8")
    print(f"\n补全 {len(enriched)} 段，其中 {inconsistent} 段 Qwen 认为与给定步骤不一致（已标 need_review）")
    print("写出", out / "segments_enriched.json")

    if a.publish_name:
        from publish_segments import publish
        print("\n发布到前端…")
        publish(out / "segments_enriched.json", a.publish_name,
                clip_offset=0.0, duration=float(run.get("duration") or 119.99))
    return 0


if __name__ == "__main__":
    sys.exit(main())
