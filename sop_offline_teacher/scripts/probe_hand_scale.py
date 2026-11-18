"""可行性探针：手部 ROI 方案在本视频上到底剩多少像素、多少覆盖率。

要回答的三个问题
----------------
1. 手在 1920x1080 里有多大？——决定「手周裁图」里还剩多少有效分辨率。
   如果手只有 40~60 像素，裁成 96x96 或 128x128 只是把模糊放大，CNN/YOLO 学不到外观。
2. 每个 SOP 步骤期间，手被检测到的比例是多少？——「每个人装配总有一只手」
   在物理上成立，但**检测得到**是另一件事。手被工件挡住、被身体挡住时没有关键点。
3. 短动作（S2 0.2~1.8s）期间手可见吗？——这决定短动作能不能靠物件身份区分。

用法
----
    .venv\\Scripts\\python.exe scripts\\probe_hand_scale.py \
        --pose web/public/data/local_pose_full_v2.json \
        --gt web/public/data/gt_segments.json --out api_runs/hand_scale
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P_THR = 0.18
BEH_TO_STEP = {
    "Remove_eyepiece_assembly": "S1",
    "Remove_the_upper_cover_of_the_eyepiece": "S2",
    "Assemble_the_mirror_base_cover": "S3",
    "Assemble_the_mirror_base_sealing_plug": "S4",
    "Fasten_mirror_base": "S5",
    "Fasten_tripod_strap": "S6",
}


def pts(hand):
    return [(p[0], p[1]) for p in (hand or []) if len(p) >= 3 and p[2] > P_THR]


def hand_box(hand):
    v = pts(hand)
    if len(v) < 4:
        return None
    xs = [p[0] for p in v]
    ys = [p[1] for p in v]
    return min(xs), min(ys), max(xs), max(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    pose = json.loads(a.pose.read_text(encoding="utf-8-sig"))
    gt = json.loads(a.gt.read_text(encoding="utf-8-sig"))
    frames = pose["frames"]

    sizes_l, sizes_r = [], []
    have = {"both": 0, "one": 0, "none": 0}
    per_frame = []
    for f in frames:
        t = float(f["timestamp"])
        bl = hand_box(f.get("left_hand"))
        br = hand_box(f.get("right_hand"))
        n = (bl is not None) + (br is not None)
        have[("none", "one", "both")[n]] += 1
        for b, acc in ((bl, sizes_l), (br, sizes_r)):
            if b:
                acc.append(max(b[2] - b[0], b[3] - b[1]))
        per_frame.append({"t": t, "l": bl is not None, "r": br is not None,
                          "sl": (max(bl[2] - bl[0], bl[3] - bl[1]) if bl else None),
                          "sr": (max(br[2] - br[0], br[3] - br[1]) if br else None)})

    tot = len(frames)
    print(f"帧数 {tot}（10 Hz，{frames[0]['timestamp']:.1f}~{frames[-1]['timestamp']:.1f}s）")
    print(f"\n=== 手检测覆盖 ===")
    print(f"  双手都检出 {have['both'] / tot:>6.1%}")
    print(f"  只检出一只 {have['one'] / tot:>6.1%}")
    print(f"  一只都没有 {have['none'] / tot:>6.1%}")

    def q(name, arr, frame_px=1920):
        if not arr:
            print(f"  {name}: 无样本")
            return
        arr = np.array(arr)
        qs = np.percentile(arr, [10, 50, 90])
        # 把手框外扩 50% 作为 ROI 边长，看裁到 96/128 后是否是放大
        roi = qs[1] * 1.5
        print(f"  {name}: 手框边长 p10/p50/p90 = {qs[0]:.0f}/{qs[1]:.0f}/{qs[2]:.0f} px"
              f"  ｜ 外扩 1.5 倍后 ROI 边长 {roi:.0f} px"
              f"  ｜ 裁到 128 是 {128 / roi:.2f}× 放大")

    print(f"\n=== 手的大小（取 21 点外接框的长边，画面 {pose.get('video', '')}）===")
    q("左手", sizes_l)
    q("右手", sizes_r)

    # ---- 按 GT 步骤统计手可见率 ----
    print(f"\n=== 各 SOP 步骤期间的手可见率（GT 段内逐帧统计）===")
    print(f"{'步骤':<6}{'GT 段数':>8}{'帧数':>7}{'双手':>8}{'至少一只':>10}{'手框中位':>10}")
    ts = np.array([r["t"] for r in per_frame])
    out_stats = {}
    for beh, sid in BEH_TO_STEP.items():
        gs = [s for s in gt["segments"] if s["behavior"] == beh and not s.get("nested_short")]
        if not gs:
            continue
        mask = np.zeros(tot, dtype=bool)
        for s in gs:
            mask |= (ts >= s["start"]) & (ts < s["end"])
        n = int(mask.sum())
        if not n:
            continue
        both = sum(1 for i in np.where(mask)[0] if per_frame[i]["l"] and per_frame[i]["r"])
        one = sum(1 for i in np.where(mask)[0] if per_frame[i]["l"] or per_frame[i]["r"])
        szs = [x for i in np.where(mask)[0] for x in (per_frame[i]["sl"], per_frame[i]["sr"]) if x]
        med = float(np.median(szs)) if szs else 0.0
        out_stats[sid] = {"segs": len(gs), "frames": n, "both": both, "one": one, "hand_px": med}
        print(f"{sid:<6}{len(gs):>8}{n:>7}{both / n:>8.1%}{one / n:>10.1%}{med:>10.0f}")

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "hand_scale.json").write_text(json.dumps({
        "pose": str(a.pose),
        "coverage": {"both": have["both"] / tot, "one": have["one"] / tot,
                     "none": have["none"] / tot},
        "hand_px_close": {"left": sorted(sizes_l), "right": sorted(sizes_r)},
        "per_step": out_stats,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n写入 {a.out / 'hand_scale.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
