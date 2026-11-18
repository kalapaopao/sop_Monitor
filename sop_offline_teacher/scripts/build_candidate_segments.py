"""把行为事件合成候选工序段（本地规则，零模型、零 API 成本）。

输入  api_runs/behavior_events/events.json   （dwell / two_stable / reach / away）
输出  api_runs/events_local/segments_proposal.json

规则（全部显式、可核对）
------------------------
1. **紧固段 FASTEN**：`two_stable` 里双手都在作业面上半区、相对构型稳定
   （rel_std ≤ --rel-std）的那些，按间隔 ≤ --merge-gap 合并；
   合并后跨度 ≥ --fasten-span 才算一段紧固作业。
   依据：紧固是"稳定施力 + 小范围换位"的交替，单次稳定只有 2~6 s，
   合并后与实测 GT 的 27.6~41.4 s 是同量级。
2. **取料事件 PICK**：手腕进入取料盘（role=tray）并停留 ≥ reach_min 的那些，
   间隔 ≤ --pick-gap 的合并成一次取料。
3. **装配段 ASSEMBLE**：手在作业面（上半区或下半区）但不属于上面两类的时段。
4. **其余 IDLE**。

映射到六步契约（**不合并类别**，ambiguity 用 need_review 标注）
--------------------------------------------------------------
    FASTEN  → S5 紧固          （边界由本地事件给出，置信度高）
    PICK    → S1 取镜座组件     （S1/S2 只能靠"手里是什么"区分 → need_review=True）
    ASSEMBLE→ S3 组装镜座上盖   （S3/S4 只能靠"手里是什么"区分 → need_review=True）
    IDLE    → idle

同时输出 `business_view`（取料 / 装配 / 紧固 / 空闲）——这是给业务看的派生视图，
**不覆盖** canonical 六步，符合"保留六类真值、另建业务视图"的要求。

用法
----
    .venv\\Scripts\\python.exe scripts/build_candidate_segments.py \
        --events api_runs/behavior_events/events.json \
        --out api_runs/events_local
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

STEP_OF_CLASS = {"FASTEN": "S5", "PICK": "S1", "ASSEMBLE": "S3", "IDLE": "idle",
                 "OTHER": "unknown"}
NAME_OF_CLASS = {"FASTEN": "紧固镜子镜座", "PICK": "取镜座组件",
                 "ASSEMBLE": "组装镜座上盖", "IDLE": "循环外停留与准备",
                 "OTHER": "流程外或插入式局部处理"}
REVIEW = {"FASTEN": False, "PICK": True, "ASSEMBLE": True, "IDLE": False, "OTHER": True}
BIZ = {"FASTEN": "紧固", "PICK": "取料", "ASSEMBLE": "装配", "IDLE": "空闲", "OTHER": "其他"}
TICK = 0.1


def merge_spans(spans, gap):
    """按时间排序后合并间隔 ≤ gap 的区间。"""
    if not spans:
        return []
    spans = sorted(spans)
    out = [list(spans[0])]
    for s, e in spans[1:]:
        if s - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(a, b) for a, b in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--stations", type=Path, default=Path("configs/stations.json"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--merge-gap", type=float, default=8.0, help="紧固稳定段之间的最大间隔（秒）")
    ap.add_argument("--pick-gap", type=float, default=3.0, help="取料事件合并间隔（秒）")
    ap.add_argument("--fasten-span", type=float, default=8.0, help="合并后最短紧固跨度（秒）")
    ap.add_argument("--rel-std", type=float, default=0.05, help="双手相对构型稳定度上限")
    ap.add_argument("--min-dur", type=float, default=1.5, help="最短段长，短于此并入邻居")
    ap.add_argument("--max-pick", type=float, default=8.0,
                    help="取料事件的最长合理跨度（取料是短暂伸手，超过它的不算取料）")
    ap.add_argument("--pose", type=Path, default=None,
                    help="给定时直接用姿态算『手在作业面』掩码，比用 dwell 事件更连续")
    ap.add_argument("--duration", type=float, default=0.0, help="时间轴总长（0=取事件里最大值）")
    a = ap.parse_args()

    doc = json.loads(a.events.read_text(encoding="utf-8-sig"))
    cfg = json.loads(a.stations.read_text(encoding="utf-8-sig"))
    regions = {r["id"]: r for r in cfg["stations"][0]["regions"] if r.get("enabled", True)}
    roles = {r.get("role"): r for r in regions.values() if r.get("role")}
    for need in ("tray", "work_upper", "work_lower"):
        if need not in roles:
            raise SystemExit(f"stations.json 缺少 role={need} 的区域")

    ev = doc["events"]
    duration = a.duration or max(e["t1"] for e in ev) + 5.0

    # ---- 1) 紧固段 ----
    stable = [e for e in ev if e["type"] == "two_stable"
              and e.get("in_asm") and e["rel_std"] is not None and e["rel_std"] <= a.rel_std]
    raw_spans = [(e["t0"], e["t1"]) for e in stable]
    merged = merge_spans(raw_spans, a.merge_gap)
    fasten = []
    for t0, t1 in merged:
        if t1 - t0 < a.fasten_span:
            continue
        inside = [e for e in stable if e["t0"] >= t0 - 0.01 and e["t1"] <= t1 + 0.01]
        rel_std = float(np.mean([e["rel_std"] for e in inside])) if inside else 9.9
        rel_dx = float(np.mean([e["rel_dx"] for e in inside])) if inside else 0.0
        rel_dy = float(np.mean([e["rel_dy"] for e in inside])) if inside else 0.0
        fasten.append({"t0": float(t0), "t1": float(t1), "n_stable": len(inside),
                       "rel_std": round(rel_std, 4), "rel_dx": round(rel_dx, 3),
                       "rel_dy": round(rel_dy, 3)})

    # ---- 2) 取料事件（短暂的伸手；过长的其实是手停在料位附近，不算取料）----
    tray_id = roles["tray"]["id"]
    reaches = [e for e in ev if e["type"] == "reach" and e["zone"] == tray_id]
    pick = [s for s in merge_spans([(e["t0"], e["t1"]) for e in reaches], a.pick_gap)
            if s[1] - s[0] <= a.max_pick]

    # ---- 3) 时间轴：装配掩码优先铺底，再用取料/紧固覆盖 ----
    n = int(np.ceil(duration / TICK)) + 1
    cls = np.array(["IDLE"] * n, dtype=object)

    def paint(t0, t1, name):
        i0 = max(0, int(round(t0 / TICK)))
        i1 = min(n, int(round(t1 / TICK)))
        if i1 > i0:
            cls[i0:i1] = name

    # 装配：手在作业面上半区或下半区
    if a.pose and a.pose.exists():
        pose = json.loads(a.pose.read_text(encoding="utf-8-sig"))
        wu, wl = roles["work_upper"]["rect_norm"], roles["work_lower"]["rect_norm"]

        def _pt(v):
            return (float(v[0]), float(v[1])) if v and len(v) >= 3 and v[2] > 0.18 else None

        fw, fh = float(doc.get("frame", [1920, 1080])[0]), float(doc.get("frame", [1920, 1080])[1])
        for f in pose["frames"]:
            t = float(f["timestamp"])
            hit = False
            for side in ("left", "right"):
                hand = f.get(f"{side}_hand")
                p = _pt(hand[0]) if hand else None
                if p is None:
                    p = _pt((f.get("body") or {}).get(f"wrist_{side[0]}"))
                if p is None:
                    continue
                x, y = p[0] / fw, p[1] / fh
                for r in (wu, wl):
                    if r[0] <= x <= r[2] and r[1] <= y <= r[3]:
                        hit = True
                        break
                if hit:
                    break
            if hit:
                paint(t, t + 1.0 / 10.0, "ASSEMBLE")
    else:
        for e in ev:
            if e["type"] == "dwell" and any(k in e.get("regions", [])
                                            for k in (roles["work_upper"]["id"],
                                                      roles["work_lower"]["id"])):
                paint(e["t0"], e["t1"], "ASSEMBLE")
    for t0, t1 in pick:
        paint(t0, t1, "PICK")
    for f in fasten:
        paint(f["t0"], f["t1"], "FASTEN")

    # ---- 4) 压缩成段，并迭代吸收过短段 ----
    def compress(arr):
        segs, i = [], 0
        while i < len(arr):
            j = i
            while j < len(arr) and arr[j] == arr[i]:
                j += 1
            segs.append([i * TICK, min(duration, j * TICK), str(arr[i])])
            i = j
        return segs

    segs = compress(cls)
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for k, s in enumerate(segs):
            if s[1] - s[0] >= a.min_dur:
                continue
            if k == 0:
                segs[1][0] = s[0]
            elif k == len(segs) - 1:
                segs[k - 1][1] = s[1]
            else:
                left, right = segs[k - 1], segs[k + 1]
                if (left[1] - left[0]) >= (right[1] - right[0]):
                    left[1] = s[1]
                else:
                    right[0] = s[0]
            segs.pop(k)
            changed = True
            break
    # 合并相邻同类
    merged_same = []
    for s in segs:
        if merged_same and merged_same[-1][2] == s[2]:
            merged_same[-1][1] = s[1]
        else:
            merged_same.append(s)
    keep = merged_same

    # ---- 4b) IDLE 里如果手其实一直在动，就不是空闲，而是「流程外动作」----
    # 依据：GT 的「系上镜座扎带」(329.6-341.6s) 就落在作业区之外，
    # 若直接归 idle 会把一个真实动作吞掉。这里用姿态速度把它分出来。
    if a.pose and a.pose.exists():
        pose2 = json.loads(a.pose.read_text(encoding="utf-8-sig"))
        fw2, fh2 = float(doc["frame"][0]), float(doc["frame"][1])
        tt, vv = [], []
        for side in ("left", "right"):
            prev = None
            for f in pose2["frames"]:
                hand = f.get(f"{side}_hand")
                p = _pt(hand[0]) if hand else None
                if p is None:
                    q = _pt((f.get("body") or {}).get(f"wrist_{side[0]}"))
                    p = q
                t = float(f["timestamp"])
                if p is None or prev is None or not (0.05 <= t - prev[0] <= 0.3):
                    prev = (t, p) if p is not None else None
                    continue
                x, y = p[0] / fw2, p[1] / fh2
                px, py = prev[1][0] / fw2, prev[1][1] / fh2
                tt.append(t)
                vv.append(float(np.hypot(x - px, y - py)) / (t - prev[0]))
                prev = (t, p)
        if tt:
            tt = np.array(tt); vv = np.array(vv)
            for s in keep:
                if s[2] != "IDLE":
                    continue
                m = (tt >= s[0]) & (tt < s[1])
                if m.sum() >= 5 and float(np.median(vv[m])) > 0.12:
                    s[2] = "OTHER"

    out = []
    for t0, t1, name in keep:
        step = STEP_OF_CLASS[name]
        if name == "FASTEN":
            f = next((x for x in fasten if abs(x["t0"] - t0) < 0.35), None)
            evidence = (f"双手在作业面上半区保持稳定构型 {f['n_stable']} 次、合并跨度 "
                        f"{t1 - t0:.1f}s；相对构型波动 {f['rel_std']:.4f}（右手约在左手上方 "
                        f"{abs(f['rel_dy']):.3f} 画面高）") if f else "双手稳定构型合并"
            conf = 0.8
        elif name == "PICK":
            evidence = f"手腕进入取料盘区域（role=tray）并离开，停留 {t1 - t0:.1f}s"
            conf = 0.6
        elif name == "ASSEMBLE":
            evidence = f"手在作业面区域活动但未形成稳定施力构型，持续 {t1 - t0:.1f}s"
            conf = 0.5
        elif name == "OTHER":
            evidence = (f"双手不在作业面区域内但仍在持续运动（中位速度>0.12/s），"
                        f"持续 {t1 - t0:.1f}s —— 可能是流程外动作或未纳入 SOP 表的步骤")
            conf = 0.4
        else:
            evidence = "双手均不在作业面区域内"
            conf = 0.5
        out.append({
            "start": round(t0, 2), "end": round(t1, 2),
            "coarse_class": name,
            "step_id": step,
            "step_name": NAME_OF_CLASS[name],
            "action": "",
            "evidence": evidence,
            "evidence_i_used": evidence,
            "state_change": "none",
            "left_hand": "", "right_hand": "",
            "confidence": conf,
            "need_review": REVIEW[name],
            "business": BIZ[name],
        })

    a.out.mkdir(parents=True, exist_ok=True)
    proposal = {
        "source": "scripts/build_candidate_segments.py（本地事件规则，0 次 API 调用）",
        "timeline": "v2.mp4 绝对时间",
        "duration": round(duration, 2),
        "params": {k: getattr(a, k) for k in
                   ("merge_gap", "pick_gap", "fasten_span", "rel_std", "min_dur")},
        "events_used": doc["counts"],
        "segments": out,
        "business_view": [{"start": s["start"], "end": s["end"], "label": s["business"]}
                          for s in out],
    }
    (a.out / "segments_proposal.json").write_text(
        json.dumps(proposal, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"紧固稳定段 {len(stable)} → 合并 {len(merged)} → 成段 {len(fasten)}")
    print(f"取料事件 {len(reaches)} → 合并 {len(pick)}")
    print(f"总计 {len(out)} 段（时长 {duration:.1f}s）")
    print(f"\n{'start':>7}{'end':>7}{'dur':>7}  {'class':<9}{'step':<6}{'review':>7}")
    for s in out:
        print(f"{s['start']:>7.1f}{s['end']:>7.1f}{s['end'] - s['start']:>7.1f}  "
              f"{s['coarse_class']:<9}{s['step_id']:<6}{str(s['need_review']):>7}")
    print(f"\n写入 {a.out / 'segments_proposal.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
