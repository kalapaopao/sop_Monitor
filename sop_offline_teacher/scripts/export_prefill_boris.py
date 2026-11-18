"""把本地模型（MLP+HMM）的预测导出成 BORIS 工程文件，用于「预填 + 人工改错」。

复用 scripts/export_boris.py 里已验证的 BORIS 7.0 结构（避免自己拼写出错）。

用法：
    python scripts/export_prefill_boris.py --pred api_runs/pred_mlp_hmm.json \
        --video vedio/v2.mp4 --out api_runs/prefill_full.boris --skip-idle
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_segments import run_lengths, state_machine  # noqa: E402
from export_boris import build  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CLASSES = ["S1", "S2", "S3", "S4", "S5", "S6", "idle_auxiliary"]
NAME = {"S1": "取镜座组件", "S2": "取镜座上盖", "S3": "组装镜座上盖",
        "S4": "组装镜座密封塞", "S5": "紧固镜子镜座", "S6": "系上镜座扎带",
        "idle_auxiliary": "循环外停留与准备"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--video", type=Path, default=Path("vedio/v2.mp4"))
    ap.add_argument("--fps", type=float, default=24.702, help="有效帧率 = 总帧数/时长")
    ap.add_argument("--duration", type=float, default=599.2)
    ap.add_argument("--confirm", type=int, default=8)
    ap.add_argument("--min-dur", type=float, default=2.0)
    ap.add_argument("--skip-idle", action="store_true")
    ap.add_argument("--name", default="prefill")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    d = json.loads(a.pred.read_text(encoding="utf-8-sig"))
    fr = d["frames"]
    ts = np.array([f["t"] for f in fr])
    pred = np.array([f["pred"] for f in fr])
    raw = run_lengths(ts, pred)
    sm = state_machine(raw, a.confirm, a.min_dur, 1.0)
    segs = [{"step_id": CLASSES[c], "name": NAME.get(CLASSES[c], CLASSES[c]),
             "start": s, "end": e} for s, e, c in sm]
    if a.skip_idle:
        segs = [s for s in segs if s["step_id"] != "idle_auxiliary"]

    print(f"逐帧预测 {len(ts)} 帧 → 原始段 {len(raw)} → 去抖 {len(sm)} 段"
          f"{'（已跳过 idle）' if a.skip_idle else ''}")
    cnt = Counter(s["step_id"] for s in segs)
    print("段构成：", dict(cnt))

    doc = build(segs, a.video, a.fps, a.name,
                description="本地模型预填（MLP+HMM 88.6%）——请在 BORIS 中删除/调整错误段后导出 CSV",
                with_hands=None, duration=a.duration)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    n_ev = len(doc["observations"][a.name]["events"])
    print(f"写出 {a.out}（{len(doc['behaviors_conf'])} 类行为 / {n_ev} 事件 = {n_ev // 2} 段）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
