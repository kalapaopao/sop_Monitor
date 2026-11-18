"""给已确定的候选段补语义：action / 左右手 / state_change / evidence。

分工（这是本项目的关键设计决策，不要改回去）
--------------------------------------------
本地事件规则**只负责**：边界 + 粗类（PICK / ASSEMBLE / FASTEN / OTHER / IDLE）+ 数值证据。
本脚本**才负责**：把这一段用中文说清楚 —— 而且在两种模式下都**不准改 step_id、不准改边界**。

    --provider local   确定性模板兜底，0 成本、可离线、可复现。
                       它不会替你做"手里是镜座还是上盖"的判断，
                       这类段一律 need_review=True，如实标出来。
    --provider qwen    调用 Qwen-VL 看段内 2 帧图，输出结构化 JSON。
                       这是**唯一**能回答"手里那个东西是什么"的环节。
                       没有 DASHSCOPE_API_KEY 时本模式直接报错退出，不会伪造结果。

用法
----
    # 本地兜底（不需要 Key）
    .venv\\Scripts\\python.exe scripts\\enrich_segments.py \
        --proposal api_runs/events_local/segments_proposal.json \
        --out api_runs/events_enriched --provider local

    # Qwen 语义确认（需要 Key）
    $env:DASHSCOPE_API_KEY="..."
    .venv\\Scripts\\python.exe scripts\\enrich_segments.py \
        --proposal api_runs/events_local/segments_proposal.json \
        --out api_runs/events_enriched --provider qwen \
        --video web/public/video/v2_5m_7m_pose.mp4 --clip-offset 300
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 按 step_id 的本地模板：只描述"这一步的标准动作"，不编造手里是什么物件。
# 六步表里本来就有 criteria，这里的话术与 configs/sop_steps.json 保持一致。
STEP_TEXT = {
    "S1": {"action": "抓取镜座组件并移到作业面",
           "left_hand": "扶稳料盘/工装", "right_hand": "抓取镜座组件", "state_change": "removed"},
    "S2": {"action": "捏住上盖边缘取下镜座上盖",
           "left_hand": "扶住镜座", "right_hand": "捏住上盖边缘并取下", "state_change": "removed"},
    "S3": {"action": "双手对位上盖并压合到位",
           "left_hand": "扶稳镜座并对位", "right_hand": "压合上盖", "state_change": "inserted"},
    "S4": {"action": "捏住密封塞对准孔位压入",
           "left_hand": "扶稳镜座", "right_hand": "指压密封塞入位", "state_change": "inserted"},
    "S5": {"action": "持电动螺丝刀在紧固点连续施力",
           "left_hand": "扶稳镜座、限制位移", "right_hand": "持工具紧固", "state_change": "tightened"},
    "S6": {"action": "一手扶持、一手绕扎带并收紧扣合",
           "left_hand": "扶持镜座", "right_hand": "绕带并收紧", "state_change": "tightened"},
    "idle": {"action": "双手离开作业面，等待或准备下一件",
             "left_hand": "无操作", "right_hand": "无操作", "state_change": "none"},
}
# 粗类兜底（给 build_candidate_segments.py 产出的 PICK/ASSEMBLE/... 用）
LOCAL_TEXT = {
    "PICK": {"action": "伸手进入取料位并取出一个零件", "left_hand": "扶稳工位", "right_hand": "伸入取料位抓取零件", "state_change": "removed"},
    "ASSEMBLE": {"action": "在作业面上对位并安装零件", "left_hand": "扶稳镜座组件", "right_hand": "对位并压入零件", "state_change": "inserted"},
    "FASTEN": {"action": "持工具在紧固点持续施力", "left_hand": "扶稳镜座", "right_hand": "持电动螺丝刀连续紧固", "state_change": "tightened"},
    "OTHER": {"action": "在作业面之外完成一段持续动作（流程外或未纳入 SOP 表的步骤）", "left_hand": "参与该动作", "right_hand": "参与该动作", "state_change": "unknown"},
    "IDLE": {"action": "双手离开作业面，等待或准备下一件", "left_hand": "无操作", "right_hand": "无操作", "state_change": "none"},
}


def local_template(seg):
    sid = seg.get("step_id")
    if sid in STEP_TEXT:
        return STEP_TEXT[sid]
    return LOCAL_TEXT.get(seg.get("coarse_class"), LOCAL_TEXT["IDLE"])

QWEN_SCHEMA_HINT = (
    '只输出一个 JSON 对象，不要解释，不要 markdown 代码块。字段必须齐全：\n'
    '{"step_id":"S1|S2|S3|S4|S5|S6|idle|unknown",'
    '"action":"中文，一句话描述这段在做什么",'
    '"left_hand":{"action":"中文","holding":"中文","target":"中文"},'
    '"right_hand":{"action":"中文","holding":"中文","target":"中文"},'
    '"state_change":"none|inserted|removed|tightened|unknown",'
    '"evidence":"中文，指出你依据画面里哪一处判断",'
    '"confidence":0.0}'
)


def load_frames(video: Path, times: list[float]):
    """按时间点抽帧，返回 [(t, base64_jpeg)]。视频不存在时返回空。"""
    if not video or not video.exists():
        return []
    import cv2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return []
    out = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        h, w = frame.shape[:2]
        scale = 640.0 / max(w, h)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            continue
        out.append((t, base64.b64encode(buf.tobytes()).decode("ascii")))
    cap.release()
    return out


def hand_crops(video: Path, pose: Path, times: list[float], out_size: int = 224,
               pad: float = 1.8, min_pts: int = 4):
    """按姿态关键点裁「手周小图」，返回 [(t, side, base64_jpeg)]。

    这是 Codex 指出的第一个实现缺陷：原来送进大模型的是整帧缩到 640，
    手在这个尺度下只有几十像素，取料/接触/松手这种短过程根本看不出来。
    这里直接按 21 点外接框裁到 224x224，手是主要占比。

    标注沿用 keep_hand 的口径：有效点少于 min_pts 视为没有这只手。
    """
    P_THR = 0.18
    if not video or not video.exists():
        return []
    if not pose or not pose.exists():
        return []
    import cv2
    doc = json.loads(Path(pose).read_text(encoding="utf-8-sig"))
    frames = doc.get("frames", [])
    # 姿态时间戳在 0.1s 栅格上（7.2/7.3…），查询时间却是任意小数（7.23）。
    # 直接字典查会全部落空 —— 之前就是这样导致手周特写几乎为 0。
    # 这里改成"最近邻 + 容差"查找。
    ts = [float(f["timestamp"]) for f in frames]
    tol = 0.08
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return []
    out = []
    for t in times:
        q = float(t)
        if not ts:
            break
        k = min(range(len(ts)), key=lambda i: abs(ts[i] - q))
        if abs(ts[k] - q) > tol:
            continue
        f = frames[k]
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, q) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        H, W = frame.shape[:2]
        for side in ("left", "right"):
            pts = [(p[0], p[1]) for p in (f.get(f"{side}_hand") or [])
                   if len(p) >= 3 and p[2] > P_THR]
            if len(pts) < min_pts:
                continue
            xs = [q[0] for q in pts]
            ys = [q[1] for q in pts]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            half = max(max(x1 - x0, y1 - y0) * pad, 24.0) / 2
            ix0 = int(max(0, cx - half)); iy0 = int(max(0, cy - half))
            ix1 = int(min(W, cx + half)); iy1 = int(min(H, cy + half))
            if ix1 - ix0 < 12 or iy1 - iy0 < 12:
                continue
            crop = frame[iy0:iy1, ix0:ix1]
            crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_CUBIC)
            ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                continue
            out.append((t, side, base64.b64encode(buf.tobytes()).decode("ascii")))
    cap.release()
    return out


def call_qwen(images, seg, step_table, allow_resegment=False):
    """调用 DashScope OpenAI 兼容端点。返回解析后的 dict。"""
    import requests

    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise SystemExit("没有 DASHSCOPE_API_KEY：qwen 模式需要它。"
                         "本脚本不会伪造 Qwen 结果，请先设置环境变量，或改用 --provider local。")
    base = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model = os.environ.get("QWEN_MODEL", "qwen3-vl-plus")

    steps = "\n".join(f"- {s['step_id']} {s['name_cn']}：手={s['criteria']['hand']}；"
                      f"物件={s['criteria']['object']}；工具={s['criteria']['tool']}"
                      for s in step_table)
    prompt = (
        "你在核验一段装配作业视频片段。本地规则已经把边界和粗类判定好了，"
        f"粗类={seg['coarse_class']}（PICK=取料 / ASSEMBLE=装配 / FASTEN=紧固 / OTHER=流程外 / IDLE=空闲），"
        f"段时长 {seg['end'] - seg['start']:.1f} 秒。\n"
        "你的任务只有两件：\n"
        "1) 看画面判断这一步落在哪个 step_id；\n"
        "2) 用中文描述这段动作以及左右手各自在做什么。\n"
        "禁止改动段数（除非下面允许）；禁止编造品牌型号；看不清就填 unknown，不要猜。\n"
        "图片说明：带「全局画面」的是整帧缩小图，用来看人在哪、整体在干什么；"
        "带「左/右 手周特写」的是按手部关键点裁出来放大的小图，用来看手里到底拿着什么、"
        "手指在做什么。判断手里物件时以小图为准。\n\n"
        "SOP 步骤表：\n" + steps + "\n\n"
        "允许出现的物件：镜座组件、镜座上盖、镜子镜座、镜座扎带、镜座密封塞、电动螺丝刀、"
        "螺丝、料盘、桌面。除此之外的一律写“未识别物件”。\n\n"
        + QWEN_SCHEMA_HINT
    )
    if allow_resegment:
        prompt += (
            "\n\n边界修正（可选）：如果你认为本地给的这一段其实包含多个不同步骤，"
            "或边界明显偏了，请额外输出：\n"
            '{"sub_segments":[{"start":秒,"end":秒,"step_id":"Sx","reason":"依据哪张图"}]}\n'
            f"必须落在父段 {seg['start']:.2f}~{seg['end']:.2f} 秒之内，严格递增不重叠，"
            "每段至少 0.2 秒。没把握就不要输出这个字段。"
        )
    content = [{"type": "text", "text": prompt}]
    for item in images:
        # images 里每项是 (t, label, b64)；label 会写进提示词，让模型知道哪张是什么
        t, label, b64 = item
        content.append({"type": "text", "text": f"【图】t={t:.2f}s {label}"})
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    # 网络偶发超时很常见；重试 3 次并退避，避免一次抖动毁掉整轮
    import time as _time
    last_err = None
    for attempt in range(3):
        try:
            resp = requests.post(
                f"{base.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": model, "messages": [{"role": "user", "content": content}],
                      "temperature": 0.1},
                timeout=90,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Qwen 调用失败 HTTP {resp.status_code}: {resp.text[:300]}")
            txt = resp.json()["choices"][0]["message"]["content"]
            s = txt.strip()
            if s.startswith("```"):
                s = s.split("```")[1]
                if s.startswith("json"):
                    s = s[4:]
            i, j = s.find("{"), s.rfind("}")
            if i < 0 or j <= i:
                raise RuntimeError(f"Qwen 没有返回 JSON：{txt[:200]}")
            return json.loads(s[i:j + 1])
        except Exception as e:      # 网络超时 / 5xx / JSON 解析失败
            last_err = e
            if attempt < 2:
                _time.sleep(2 + 3 * attempt)
    raise RuntimeError(f"重试 3 次仍失败：{last_err}")


def _validate_subsegments(subs, parent, valid_ids=None, min_dur=0.2):
    """校验大模型提出的边界修正（拆段/调边界），不合法就整条丢弃。

    规则：
      · 必须落在父段 [start, end] 之内，允许小幅越界（截断到父段）
      · 必须严格递增、互不重叠、每段时长 ≥ min_dur
      · step_id 必须合法
    任何一条不满足就返回 []，宁可不改，也不让模型把时间轴搞乱。
    """
    if not isinstance(subs, list) or not subs:
        return []
    p0, p1 = float(parent["start"]), float(parent["end"])
    out = []
    for s in subs:
        try:
            a = max(p0, float(s["start"]))
            b = min(p1, float(s["end"]))
        except Exception:
            return []
        sid = s.get("step_id")
        if valid_ids is not None and sid not in valid_ids:
            return []
        if b - a < min_dur:
            return []
        out.append({"start": round(a, 2), "end": round(b, 2),
                    "step_id": sid, "reason": str(s.get("reason", ""))[:80]})
    out.sort(key=lambda x: x["start"])
    for i in range(1, len(out)):
        if out[i]["start"] < out[i - 1]["end"] - 1e-6:
            return []
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposal", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--provider", choices=["local", "qwen"], default="local")
    ap.add_argument("--video", type=Path, default=None, help="qwen 模式抽帧用")
    ap.add_argument("--pose", type=Path, default=None,
                    help="qwen 模式的姿态 json；给了它才会裁「手周特写」一起送进去。"
                         "没有它就退回整帧，等于放弃短过程证据")
    ap.add_argument("--allow-resegment", action="store_true",
                    help="允许大模型对边界提出拆段/调整（结果经程序校验，不合法的整条丢弃）")
    ap.add_argument("--sop", type=Path, default=Path("configs/sop_steps.json"))
    ap.add_argument("--clip-offset", type=float, default=0.0,
                    help="proposal 是绝对时间时，抽帧要减掉这个偏移（视频是片段时用）")
    ap.add_argument("--frames", type=int, default=2, help="每段最少抽几帧（短段用）")
    ap.add_argument("--max-frames", type=int, default=8,
                    help="每段最多抽几帧；默认按每 5 秒 1 帧自适应，上限受此约束")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 段（省钱试跑）")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    prop = json.loads(a.proposal.read_text(encoding="utf-8-sig"))
    step_table = json.loads(a.sop.read_text(encoding="utf-8-sig"))["steps"]
    segs = prop["segments"]
    if a.limit:
        segs = segs[:a.limit]

    if a.provider == "qwen" and not a.dry_run and not os.environ.get("DASHSCOPE_API_KEY"):
        raise SystemExit("没有 DASHSCOPE_API_KEY：qwen 模式需要它。"
                         "本脚本不会伪造 Qwen 结果，请先设置环境变量，或改用 --provider local。")

    enriched = []
    n_qwen = 0

    def build_doc():
        return {
            "source": prop.get("source", ""),
            "enrich_provider": a.provider,
            "enrich_note": ("本地模板：只补话术，不判断物件；PICK/ASSEMBLE 一律 need_review"
                            if a.provider == "local" else "Qwen-VL 逐段语义确认（含手周特写）"),
            "timeline": prop.get("timeline", ""),
            "duration": prop.get("duration"),
            "step_id_source": "configs/sop_steps.json",
            "segments": enriched,
        }

    def save_progress():
        # 增量落盘：网络抖动/超时的时候不至于整轮白跑
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "segments_with_step.json").write_text(
            json.dumps(build_doc(), ensure_ascii=False, indent=1), encoding="utf-8")

    for k, s in enumerate(segs):
        e = dict(s)
        if a.provider == "local":
            tpl = local_template(s)
            e.update({
                "action": tpl["action"] + f"（{s['end'] - s['start']:.1f} 秒）",
                "left_hand": tpl["left_hand"],
                "right_hand": tpl["right_hand"],
                "state_change": tpl["state_change"],
                "semantic_source": "local-template",
                # 本地模板分不清"手里是镜座还是上盖"，这类必须人工/Qwen 复核
                "need_review": s["need_review"] or s.get("step_id") in ("S2", "S6", "S1", "S4"),
            })
        else:
            # 抽帧数自适应段长：长段只取 2 帧会抓到不代表整段的瞬间
            # （实测 28 秒紧固段取 2 帧，正好抓工人去料盒取螺丝 → 被误判成"取料"）
            span = max(0.2, s["end"] - s["start"])
            n_frames = int(min(a.max_frames, max(a.frames, (span + 4.9) // 5)))
            times = [s["start"] + span * f / (n_frames + 1) for f in range(1, n_frames + 1)]
            times = [t - a.clip_offset for t in times]
            imgs = []
            # ① 全局画面（缩小）：给出"人在哪、整体在干什么"
            for t, b64 in (load_frames(a.video, times) if a.video else []):
                imgs.append((t, "全局画面（缩小，看整体）", b64))
            # ② 手周小图（放大到 224）：Codex 指出原来只有整帧，
            #    手在 640 宽的画面里只有几十像素，取料/接触/松手根本看不出来。
            if a.pose:
                for t, side, b64 in hand_crops(a.video, a.pose, times):
                    imgs.append((t, f"{'左' if side == 'left' else '右'}手周特写", b64))
            n_hand = sum(1 for x in imgs if "手周特写" in x[1])
            if a.dry_run:
                print(f"[dry-run] {s['start']:7.1f}-{s['end']:7.1f} {s['step_id']:<6} "
                      f"图 {len(imgs)} 张（全局 {len(imgs) - n_hand} + 手周 {n_hand}）"
                      f" 时间 {[round(t, 2) for t in times]}")
                enriched.append(e)
                continue
            if not imgs:
                raise SystemExit(f"没能从 {a.video} 抽到帧（检查 --video / --pose / --clip-offset）")
            r = call_qwen(imgs, s, step_table, allow_resegment=a.allow_resegment)
            n_qwen += 1
            sid = r.get("step_id")
            valid = {x["step_id"] for x in step_table} | {"idle", "unknown"}
            if sid not in valid:
                sid = s["step_id"]
                r["confidence"] = min(float(r.get("confidence") or 0.3), 0.3)
            e.update({
                "step_id": sid,
                "step_name": next((x["name_cn"] for x in step_table if x["step_id"] == sid),
                                  s["step_name"]),
                "action": r.get("action", ""),
                "left_hand": r.get("left_hand", ""),
                "right_hand": r.get("right_hand", ""),
                "state_change": r.get("state_change", "none"),
                "evidence": (s["evidence"] + " ｜ Qwen：" + str(r.get("evidence", ""))),
                "evidence_i_used": (s["evidence"] + " ｜ Qwen：" + str(r.get("evidence", ""))),
                "confidence": r.get("confidence"),
                "semantic_source": "qwen",
                "n_images": len(imgs),
                "n_hand_crops": sum(1 for x in imgs if "手周特写" in x[1]),
                # Qwen 自己说不像 → 要求复核
                "need_review": s["need_review"] or not r.get("consistent", True),
            })
            # ③ 允许大模型提出边界修正（拆段/并段）。Codex 指出的第二个缺陷：
            #    原来只能整段选一个 step_id，段里含"取盖—对位—压合"时错误边界被保留。
            if a.allow_resegment:
                subs = r.get("sub_segments") or []
                fixed = _validate_subsegments(subs, s)
                if fixed:
                    e["sub_segments"] = fixed
                    e["need_review"] = True   # 被模型改过边界的段一律要求复核
        enriched.append(e)
        if a.provider == "qwen" and not a.dry_run:
            save_progress()          # 每段存一次，超时不丢进度
            print(f"  [{len(enriched)}/{len(segs)}] {s['start']:.1f}-{s['end']:.1f}s "
                  f"→ {e.get('step_id')} {e.get('step_name', '')}", flush=True)

    doc = build_doc()
    a.out.mkdir(parents=True, exist_ok=True)
    # 文件名是 publish_segments.py 的输入契约（它只认 segments_with_step / segments_proposal），
    # 不代表"一定是 Qwen 补的"——具体来源见文件内的 enrich_provider 字段。
    out_path = a.out / "segments_with_step.json"
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"provider={a.provider} ｜ 段数 {len(enriched)} ｜ Qwen 调用 {n_qwen}")
    print(f"需要复核 {sum(1 for e in enriched if e['need_review'])} 段")
    print(f"写入 {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
