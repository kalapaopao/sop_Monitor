"""把已有的逐帧预测压成工序段（段级后处理），并与其它分段法同口径对比。

为什么改用这个
--------------
早先一版用「手腕停留 / 往返」启发式直接从骨架造分段，绕过了项目里已有的
帧级分类器（`api_runs/pred_mlp_hmm.json`，7 周期留一、无泄漏、帧级 88.62%）。
结果是分类信息被丢光：120 秒窗口里约 12 个人工步骤只命中 1~2 个，
边界中位误差 5 秒以上。属于明确的路线错误。

本脚本换回来：**帧级预测是主信号，分段只是后处理**。
逐帧预测已经包含 S1~S6 的区分能力，缺的只是"把抖动抹平成段"和"处理短段"。

后处理（全部显式）
------------------
1. 补齐到 0.1s 栅格（invalid 帧被屏蔽，用最近可见帧补）
2. 迟滞：新标签连续 confirm 个 tick 才切换（实测 HMM 已平滑，此项通常无影响）
3. 抹平：过短段并入相邻段。类别有时长先验时用先验，
   **不再统一用全局阈值**——否则 0.2~2.0 秒的 S2 会被必然抹掉
4. 剔除：超出 `configs/sop_steps.json` 的 expected_sec（上下限各留余量）视为误判
5. 合并相邻同类

用法
----
    .venv\\Scripts\\python.exe scripts\\build_segments_from_pred.py \
        --pred api_runs/pred_mlp_hmm.json --out api_runs/segs_pred \
        --confirm 8 --min-dur 1.2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NAME_CN = {"S1": "取镜座组件", "S2": "取镜座上盖", "S3": "组装镜座上盖",
           "S4": "组装镜座密封塞", "S5": "紧固镜子镜座", "S6": "系上镜座扎带",
           "idle_auxiliary": "循环外停留与准备"}
TICK = 0.1


def hysteresis(labels, confirm):
    if not len(labels):
        return labels
    out = list(labels)
    cur = labels[0]
    run_val, run_len = labels[0], 0
    for i, v in enumerate(labels):
        if v == run_val:
            run_len += 1
        else:
            run_val, run_len = v, 1
        if run_val != cur and run_len >= confirm:
            cur = run_val
            for k in range(i - run_len + 1, i + 1):
                out[k] = cur
        elif run_val == cur:
            out[i] = cur
    return out


def compress(labels, times):
    segs, i = [], 0
    while i < len(labels):
        j = i
        while j < len(labels) and labels[j] == labels[i]:
            j += 1
        segs.append([times[i], times[j - 1] + TICK, labels[i]])
        i = j
    return segs


def hysteresis_confirmed(labels, confirm, expected=None):
    """Hold the current class until sufficient evidence, preserving known short classes."""
    if not len(labels):
        return []
    current = labels[0]
    candidate, count = current, 0
    output = []
    for index, value in enumerate(labels):
        output.append(current)
        if value == current:
            candidate, count = current, 0
            continue
        count = count + 1 if value == candidate else 1
        candidate = value
        required = max(1, confirm)
        if expected and value in expected:
            required = min(required, max(1, int(np.ceil(expected[value][0] / TICK))))
        if count >= required:
            current = value
            output[index - count + 1:index + 1] = [current] * count
            count = 0
    return output


def _absorb_one(segs, k):
    if k == 0:
        segs[1][0] = segs[0][0]
    elif k == len(segs) - 1:
        segs[k - 1][1] = segs[k][1]
    else:
        left, right = segs[k - 1], segs[k + 1]
        if (left[1] - left[0]) >= (right[1] - right[0]):
            left[1] = segs[k][1]
        else:
            right[0] = segs[k][0]
    segs.pop(k)


def absorb_short(segs, min_dur, expected=None, tol=0.5):
    """抹平过短段。类别有时长先验时用先验，避免把真实短动作抹掉。"""
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for k, s in enumerate(segs):
            step = s[2]
            thr = min_dur
            if expected and step in expected:
                thr = min(min_dur, expected[step][0] * tol)
            if s[1] - s[0] >= thr:
                continue
            _absorb_one(segs, k)
            changed = True
            break
    return segs


def prune_by_expected(segs, expected, tol=0.5):
    """剔除长度不合 expected_sec 的段（视为误判）。依据是项目已有契约，不是现场调参。"""
    if not expected:
        return segs
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for k, s in enumerate(segs):
            step = s[2]
            if step == "idle_auxiliary" or step not in expected:
                continue
            lo, hi = expected[step]
            d = s[1] - s[0]
            if d >= lo * tol and d <= hi * 1.5:
                continue
            _absorb_one(segs, k)
            changed = True
            break
    return segs


def merge_same(segs):
    out = []
    for s in segs:
        if out and out[-1][2] == s[2]:
            out[-1][1] = s[1]
        else:
            out.append(list(s))
    return out


def build(pred_path, confirm, min_dur, expected=None, prune=True, strict_hysteresis=False):
    d = json.loads(Path(pred_path).read_text(encoding="utf-8-sig"))
    classes = d["classes"]
    fr = d["frames"]
    t = np.array([f["t"] for f in fr])
    lab = np.array([classes[f["pred"]] for f in fr], dtype=object)

    grid = np.arange(float(t[0]), float(t[-1]) + TICK / 2, TICK)
    idx = np.clip(np.searchsorted(t, grid), 0, len(t) - 1)
    prev = np.clip(idx - 1, 0, len(t) - 1)
    choose = np.where(np.abs(t[prev] - grid) < np.abs(t[idx] - grid), prev, idx)
    labels = lab[choose]

    if strict_hysteresis:
        labels = np.array(hysteresis_confirmed(list(labels), confirm, expected), dtype=object)
    else:
        labels = np.array(hysteresis(list(labels), confirm), dtype=object)
    segs = compress(labels, list(grid))
    segs = absorb_short(segs, min_dur, expected)
    segs = merge_same(segs)
    if prune and expected:
        segs = prune_by_expected(segs, expected)
        segs = merge_same(segs)
    return segs, classes, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--confirm", type=int, default=8, help="迟滞确认 tick 数（10Hz，8=0.8s）")
    ap.add_argument("--min-dur", type=float, default=1.2, help="最短段长（秒）")
    ap.add_argument("--sop", type=Path, default=Path("configs/sop_steps.json"))
    ap.add_argument("--no-prune", action="store_true", help="关闭 expected_sec 过滤")
    ap.add_argument("--strict-hysteresis", action="store_true", help="启用实际确认切换，并按类别最短时长保护短步骤；默认保持旧结果")
    a = ap.parse_args()

    expected = {}
    if a.sop.exists():
        for s in json.loads(a.sop.read_text(encoding="utf-8-sig"))["steps"]:
            expected[s["step_id"]] = tuple(s["expected_sec"])

    segs, classes, d = build(a.pred, a.confirm, a.min_dur,
                             expected=expected, prune=not a.no_prune, strict_hysteresis=a.strict_hysteresis)

    out = []
    for s0, s1, cl in segs:
        n_tick = int(round((s1 - s0) / TICK))
        out.append({
            "start": round(float(s0), 2), "end": round(float(s1), 2),
            "dur": round(float(s1 - s0), 2),
            "coarse_class": "IDLE" if cl == "idle_auxiliary" else "STEP",
            "step_id": "idle" if cl == "idle_auxiliary" else cl,
            "step_name": NAME_CN.get(cl, cl),
            "action": "",
            "evidence": (f"逐帧预测→迟滞(confirm={a.confirm})→类别先验抹平(min_dur={a.min_dur})"
                         f"→expected_sec 剔除；该段 {n_tick} tick"),
            "evidence_i_used": "",
            "state_change": "none",
            "left_hand": "", "right_hand": "",
            "confidence": None,
            "need_review": cl in ("S2", "S6"),
            "business": "空闲" if cl == "idle_auxiliary" else "工序",
        })
    for x in out:
        x["evidence_i_used"] = x["evidence"]

    a.out.mkdir(parents=True, exist_ok=True)
    doc = {
        "source": (f"scripts/build_segments_from_pred.py ← {a.pred}"
                   f"（{d['model']}，帧级 {d['frame_accuracy']:.2%}）"),
        "timeline": "v2.mp4 绝对时间",
        "duration": round(float(d["frames"][-1]["t"]) + 5, 2),
        "params": {"confirm": a.confirm, "min_dur": a.min_dur, "prune": not a.no_prune, "strict_hysteresis": a.strict_hysteresis},
        "segments": out,
        "business_view": [{"start": s["start"], "end": s["end"], "label": s["business"]}
                          for s in out],
    }
    (a.out / "segments_proposal.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"帧级来源 {a.pred}（{d['model']}，{d['frame_accuracy']:.2%}）")
    print(f"confirm={a.confirm} min_dur={a.min_dur} prune={not a.no_prune} → {len(out)} 段")
    print(f"\n{'start':>7}{'end':>7}{'dur':>7}  {'step':<7}{'name':<12}")
    for s in out[:30]:
        print(f"{s['start']:>7.1f}{s['end']:>7.1f}{s['dur']:>7.1f}  {s['step_id']:<7}{s['step_name']:<12}")
    if len(out) > 30:
        print(f"  …另有 {len(out) - 30} 段")
    print(f"\n写入 {a.out / 'segments_proposal.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
