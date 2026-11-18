"""按「手 + 手周 ROI」建物件识别数据集，并做一次可分离性基线。

思路（这是本轮的关键设计）
--------------------------
1. **手提供 WHERE**：RTMW 的 21 个手部点取外接框 + 外扩，就是 ROI。
   免标注、免物体检测器、换产线不用重画框。实测 ROI 边长 135~164 px，
   裁到 128 基本是原生分辨率（见 api_runs/hand_scale）。
2. **已有 SOP 段提供 WHAT 的标签**：GT 里的「取镜座组件」段内，手里那个东西
   就是镜座组件。于是**每一帧被检出的手，自动得到一个带类别的手周小图**，
   不需要为物件单独标一个框。这是零额外标注。
3. **物件身份把数据跨步骤汇总**：上盖同时出现在 S2(取) 和 S3(装)，
   密封塞在 S4，镜座在 S1、S5。合并成"物件类"后，S2 那可怜的 46 帧样本
   可以借用 S3 的 512 帧——这正是短动作数据稀缺的解药。

⚠️ **标签口径限制（必须说清，否则会误导）**
本脚本的 `obj_class` 是**由 SOP 段推导出来的**，不是人看着某只手标出来的：
同一段里左右两只手都会被赋成同一个物件类。例如 S5「紧固」段里，
一只手握电动螺丝刀、另一只手在扶工件，但两只手的裁图都会标成"螺丝刀"。

所以在这份数据上得到的分类准确率，衡量的是
**「这张手周图来自哪个 SOP 阶段」**，**不能**当作
「这只手里拿的是什么」的准确率。要得到后者，需要逐手标注或接触关系判定。

字段名保留 `obj_class` 以兼容旧结果，但输出里同时写
`obj_class_from_step: true` 与 `label_is_hand_specific: false`，避免被误用。

本脚本做两件事：
  A. 按上述规则导出裁图 + 索引（obj_class / step_id / cycle / hand_px）
  B. 在导出的数据上跑一个 leave-one-cycle-out 的最近质心基线，
     给出**每个物件类的可分离性**——如果连廉价特征都能分开，
     换真 CNN 只会更好；如果分不开，说明 ROI 分辨率或标签有问题，先别训 YOLO。

用法
----
    .venv\\Scripts\\python.exe scripts\\build_hand_object_set.py \
        --video vedio/v2.mp4 \
        --pose web/public/data/local_pose_full_v2.json \
        --gt web/public/data/gt_segments.json \
        --out api_runs/hand_objects --size 128 --pad 1.5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P_THR = 0.18

# 步骤内「手里是什么」——由 configs/sop_steps.json 的三要素推得，不是猜的
STEP_TO_OBJECT = {
    "S1": "镜座组件",
    "S2": "上盖",
    "S3": "上盖",
    "S4": "密封塞",
    "S5": "螺丝刀",     # 紧固时手里是工具，不是工件
    "S6": "扎带",
}
BEH_TO_STEP = {
    "Remove_eyepiece_assembly": "S1",
    "Remove_the_upper_cover_of_the_eyepiece": "S2",
    "Assemble_the_mirror_base_cover": "S3",
    "Assemble_the_mirror_base_sealing_plug": "S4",
    "Fasten_mirror_base": "S5",
    "Fasten_tripod_strap": "S6",
}


def hand_box(hand):
    v = [(p[0], p[1]) for p in (hand or []) if len(p) >= 3 and p[2] > P_THR]
    if len(v) < 4:
        return None
    xs = [p[0] for p in v]
    ys = [p[1] for p in v]
    return min(xs), min(ys), max(xs), max(ys)


def expand(box, pad, w, h):
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = max(x1 - x0, y1 - y0) * pad
    half = max(s, 24.0) / 2
    return (int(max(0, cx - half)), int(max(0, cy - half)),
            int(min(w, cx + half)), int(min(h, cy + half)))


def feats(bgr):
    """廉价外观描述子（仅用于可分离性基线，不是最终模型）。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    hist = cv2.calcHist([h], [0], None, [8], [0, 180]).ravel()
    hist = hist / (hist.sum() + 1e-6)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    f = list(hist)
    f += [h.mean() / 180, h.std() / 180, s.mean() / 255, s.std() / 255,
          v.mean() / 255, v.std() / 255,
          float((s > 60).mean()), float((v < 60).mean()),
          float(edges.mean() / 255)]
    # 4x4 网格平均亮度，保留一点空间结构
    small = cv2.resize(gray, (4, 4), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    f += list(small.ravel())
    return np.array(f, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--pose", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--pad", type=float, default=1.5)
    ap.add_argument("--not-found", type=int, default=5)
    ap.add_argument("--step-stride", type=str, default="S5:3",
                    help="对某些步骤降采样，格式 S5:3 表示每 3 帧取 1 帧（长动作帧太多）")
    ap.add_argument("--no-save", action="store_true", help="只算特征，不落盘裁图")
    a = ap.parse_args()

    stride = {}
    for part in (a.step_stride or "").split(","):
        if ":" in part:
            k, v = part.split(":")
            stride[k.strip()] = max(1, int(v))

    gt = json.loads(a.gt.read_text(encoding="utf-8-sig"))
    spans = []
    for s in gt["segments"]:
        sid = BEH_TO_STEP.get(s["behavior"])
        if not sid or s.get("nested_short"):
            continue
        spans.append((s["start"], s["end"], sid, s.get("cycle_id")))
    spans.sort()

    pose = json.loads(a.pose.read_text(encoding="utf-8-sig"))
    frames = pose["frames"]

    cap = cv2.VideoCapture(str(a.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    img_dir = a.out / "crops"
    if not a.no_save:
        img_dir.mkdir(parents=True, exist_ok=True)
    else:
        a.out.mkdir(parents=True, exist_ok=True)

    # 只处理落在 GT 段内的帧
    wanted = []
    seen = {}
    for f in frames:
        t = float(f["timestamp"])
        for (s0, s1, sid, cyc) in spans:
            if s0 <= t < s1:
                k = (sid, cyc, int(t * 10))
                n = seen.get((sid, cyc), 0)
                seen[(sid, cyc)] = n + 1
                st = stride.get(sid, 1)
                if n % st == 0:
                    wanted.append((t, f, sid, cyc))
                break

    print(f"视频 {a.video} {W}x{H} @{fps:.3f} ｜ GT 段 {len(spans)} ｜ 待处理帧 {len(wanted)}")
    t0 = time.time()
    rows = []
    cursor = -1
    for t, f, sid, cyc in wanted:
        want = int(round(t * fps))
        while cursor < want:
            if not cap.grab():
                break
            cursor += 1
        ok, frame = cap.retrieve() if cursor >= want else (False, None)
        if not ok or frame is None:
            continue
        for side in ("left", "right"):
            b = hand_box(f.get(f"{side}_hand"))
            if b is None:
                continue
            x0, y0, x1, y1 = expand(b, a.pad, W, H)
            if x1 - x0 < 12 or y1 - y0 < 12:
                continue
            crop = frame[y0:y1, x0:x1]
            crop = cv2.resize(crop, (a.size, a.size), interpolation=cv2.INTER_AREA)
            name = f"{sid}_c{cyc}_{int(t * 10):05d}_{side[0]}.jpg"
            if not a.no_save:
                cv2.imwrite(str(img_dir / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            rows.append({
                "t": round(t, 2), "side": side, "step_id": sid, "cycle": cyc,
                "obj_class": STEP_TO_OBJECT[sid],
                "hand_px": round(max(b[2] - b[0], b[3] - b[1]), 1),
                "file": name, "_f": feats(crop),
            })
    cap.release()
    print(f"裁图完成 {time.time() - t0:.0f}s ｜ 样本 {len(rows)}")

    # ---- 类别分布 ----
    from collections import Counter
    print("\n=== 样本分布 ===")
    print(f"{'物件类':<10}{'样本':>7}{'来自步骤':>22}{'手边框中位':>11}")
    for cls in sorted({r["obj_class"] for r in rows}):
        sub = [r for r in rows if r["obj_class"] == cls]
        src = Counter(r["step_id"] for r in sub)
        med = float(np.median([r["hand_px"] for r in sub]))
        print(f"{cls:<10}{len(sub):>7}   {dict(src)!s:<20}{med:>11.0f}")

    # ---- leave-one-cycle-out 最近质心基线 ----
    X = np.array([r["_f"] for r in rows], dtype=np.float32)
    y = np.array([r["obj_class"] for r in rows])
    cyc = np.array([r["cycle"] for r in rows])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    classes = sorted(set(y))
    print(f"\n=== 可分离性基线：最近质心 + 留一周期（{len(classes)} 类，{len(Z)} 样本）===")
    print(f"{'留出周期':>8}{'测试样本':>9}{'准确率':>8}{'预测分布':>28}")
    accs = []
    for c in sorted(set(cyc)):
        tr = cyc != c
        te = cyc == c
        if te.sum() == 0 or tr.sum() == 0:
            continue
        cent = {k: Z[tr][y[tr] == k].mean(0) for k in classes if (y[tr] == k).any()}
        pred = []
        for z in Z[te]:
            best, bd = None, 1e9
            for k, cvec in cent.items():
                d = float(np.linalg.norm(z - cvec))
                if d < bd:
                    bd, best = d, k
            pred.append(best)
        acc = float((np.array(pred) == y[te]).mean())
        accs.append(acc)
        dist = Counter(pred)
        print(f"{c:>8}{int(te.sum()):>9}{acc:>8.1%}   {dict(dist)!s:<26}")
    chance = 1.0 / max(1, len(classes))
    print(f"\n整体留一周期平均准确率 {np.mean(accs):.1%}（随机基线 {chance:.1%}）")
    print("说明：这是**极廉价特征**的下限，只用来判断『物件外观里有没有可用的信息量』。")
    print("      真 CNN（MobileNetV3 等）只会更好；若这里就接近随机，先怀疑 ROI 分辨率或标签，别急着训 YOLO。")

    for r in rows:
        r.pop("_f", None)
    (a.out / "index.json").write_text(json.dumps({
        "video": str(a.video), "pose": str(a.pose), "gt": str(a.gt),
        "size": a.size, "pad": a.pad, "step_to_object": STEP_TO_OBJECT,
        # 标签口径：obj_class 是由 SOP 段推导的，同一段里左右手会被赋成同一个物件类。
        # 因此在这个数据集上得到的准确率衡量的是「裁图来自哪个 SOP 阶段」，
        # 不是「这只手里拿的是什么」。见文件头说明。
        "obj_class_from_step": True,
        "label_is_hand_specific": False,
        "n": len(rows), "items": rows,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n写入 {a.out / 'index.json'}")
    if not a.no_save:
        print(f"写入裁图目录 {img_dir}（{len(rows)} 张）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
