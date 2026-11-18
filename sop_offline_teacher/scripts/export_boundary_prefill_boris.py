"""冷启动预填：**完全不需要任何标注**，用骨架变化点找出候选边界，导出成 BORIS 工程。

为什么需要它（用户指出我上一条的漏洞）：
  之前那个 prefill 用的模型是"用同一段视频的标注训练出来的" → 对全新视频/新产线无效。
  冷启动阶段没有任何本地模型，只能用**无监督**方法。

本脚本只做一件事：找边界，**不判类别**。
  · 输入：10 Hz 骨架（无需任何标注）
  · 输出：BORIS 工程，每段名字是占位符 `STEP?`
  · 你要做的：在 BORIS 里逐段改名字（键盘操作，很快）+ 删掉误切

实测（detect_boundaries，±2s 容差）：边界召回 69.6%、误差中位 0.53 秒。

用法：
    python scripts/export_boundary_prefill_boris.py \
        --pose web/public/data/local_pose_full_v2.json \
        --video vedio/v2.mp4 --out api_runs/prefill_boundaries.boris
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_boundaries import build_features, change_signal, smooth, pick_peaks  # noqa: E402
from export_boris import build  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", type=Path, default=Path("web/public/data/local_pose_full_v2.json"))
    ap.add_argument("--video", type=Path, default=Path("vedio/v2.mp4"))
    ap.add_argument("--fps", type=float, default=24.702)
    ap.add_argument("--k", type=float, default=1.2, help="变化强度阈值系数（越大候选越少）")
    ap.add_argument("--min-gap", type=float, default=1.5, help="两个边界的最小间隔（秒）")
    ap.add_argument("--min-dur", type=float, default=1.5, help="最短段长（秒）")
    ap.add_argument("--placeholder", default="STEP?", help="占位行为名，BORIS 里再改")
    ap.add_argument("--name", default="prefill_boundaries")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    pose = json.loads(a.pose.read_text(encoding="utf-8-sig"))
    frames = pose["frames"]
    dur = float(pose.get("duration") or frames[-1]["timestamp"])
    print(f"骨架 {len(frames)} 帧 ｜ 时长 {dur:.1f}s ｜ 无监督（不使用任何标注）")

    feats = build_features(frames)
    sig = smooth(change_signal(feats))
    peaks, thr, med, mad = pick_peaks(sig, a.min_gap, a.k)
    print(f"变化强度阈值 {thr:.3f} → 候选边界 {len(peaks)} 个")

    edges = [0.0] + [p for p in peaks if 0.5 < p < dur - 0.5] + [dur]
    segs = []
    for s, e in zip(edges, edges[1:]):
        if e - s < a.min_dur:
            continue
        segs.append({"step_id": a.placeholder, "name": "待命名", "start": s, "end": e})
    print(f"切出候选段 {len(segs)} 个（最短 {a.min_dur}s 已过滤）")

    doc = build(segs, a.video, a.fps, a.name,
                description="无监督边界预填：只切了边界，没有判类别。请在 BORIS 里逐段命名/合并/删除。",
                with_hands=None, duration=dur)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    n_ev = len(doc["observations"][a.name]["events"])
    print(f"写出 {a.out}（{n_ev} 事件 = {n_ev // 2} 段，全部占位名 {a.placeholder}）")
    print()
    print("下一步（BORIS 里）：")
    print("  1) 打开这个文件")
    print("  2) 在 Behaviors coding map 里把 %s 改成真实步骤名（S1~S6）" % a.placeholder)
    print("  3) 逐段检查：删掉误切的、合并切碎的、拖动边界")
    print("  4) 导出 CSV → python scripts/check_boris_export.py <CSV目录>")
    print("  5) python scripts/boris_csv_to_gt.py --csv <CSV> → 训练")
    return 0


if __name__ == "__main__":
    sys.exit(main())
