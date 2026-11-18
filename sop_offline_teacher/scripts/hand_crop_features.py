"""手周外观特征（廉价版，不需要预训练 CNN）：给每帧的左右手各算 16 维。

动机：
  pose-only 特征（v3）已到 85.5%，但「取镜座/取上盖」与「空闲」仍混，
  因为两者手部运动都慢；区别在于**手里有没有东西、东西是什么颜色/形状**。
  完整方案是 MobileNetV3 编码 ROI（需预训练权重/更多算力），
  这里先用确定性的低成本外观描述子验证"外观是否有用"：

每只手 16 维：
    1-3   ROI 平均 BGR
    4-6   ROI 标准差 BGR
    7-10  4 段色调直方图（粗分，抑制光照）
    11    亮度
    12    边缘密度（Sobel 阈值化后的比例）
    13-14 前景占比（与中值色差 > 阈值 的比例）
    15    手臂/物体的粗细代理（列边缘间距）
    16    是否有效手（0/1）

用法：
    python scripts/hand_crop_features.py --pose web/public/data/local_pose_full_v2.json \
        --video vedio/v2.mp4 --out web/public/data/hand_appearance_full.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import cv2
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

P_THR = 0.18
CROP = 96          # 裁剪边长（像素，原图尺度按人体高自适应）


def wrist_of(hand):
    if hand and len(hand) > 0 and hand[0][2] > P_THR:
        return float(hand[0][0]), float(hand[0][1])
    return None


def hand_extent(hand):
    v = [(p[0], p[1]) for p in (hand or []) if p[2] > P_THR]
    if len(v) < 3:
        return None
    xs = [p[0] for p in v]
    ys = [p[1] for p in v]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def describe(patch):
    """16 维外观描述子。"""
    if patch is None or patch.size == 0:
        return [0.0] * 16
    p = patch.astype(np.float32)
    b, g, r = p[:, :, 0], p[:, :, 1], p[:, :, 2]
    feats = [float(b.mean()), float(g.mean()), float(r.mean()),
             float(b.std()), float(g.std()), float(r.std())]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0].astype(np.float32), hsv[:, :, 1].astype(np.float32), hsv[:, :, 2].astype(np.float32)
    hist = np.histogram(h, bins=4, range=(0, 180))[0].astype(np.float32)
    hist = hist / max(1.0, hist.sum())
    feats += hist.tolist()
    feats.append(float(v.mean()))
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    feats.append(float((mag > 60).mean()))
    med = np.median(p.reshape(-1, 3), axis=0)
    d = np.abs(p - med).sum(2)
    fg = float((d > 60).mean())
    feats.append(fg)
    cols = (mag > 60).sum(0) > mag.shape[0] * 0.4
    feats.append(float(cols.mean()))
    feats.append(1.0)
    return [round(float(x), 4) for x in feats[:16]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pose', type=Path, required=True)
    ap.add_argument('--video', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--min-side', type=int, default=48)
    ap.add_argument('--max-side', type=int, default=320)
    a = ap.parse_args()

    pose = json.loads(a.pose.read_text(encoding='utf-8-sig'))
    frames = pose['frames']
    cap = cv2.VideoCapture(str(a.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    rows = []
    for i, f in enumerate(frames):
        t = float(f['timestamp'])
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, img = cap.read()
        if not ok:
            rows.append({'t': round(t, 2), 'left': [0.0] * 16, 'right': [0.0] * 16})
            continue
        bb = (f.get('person') or {}).get('bbox')
        bh = (float(bb[3]) - float(bb[1])) if bb else 900.0
        out = {}
        for side in ('left', 'right'):
            hand = f.get(f'{side}_hand')
            w = wrist_of(hand)
            if w is None:
                out[side] = [0.0] * 16
                continue
            ext = hand_extent(hand) or bh * 0.08
            side_px = int(np.clip(ext * 1.8, a.min_side, a.max_side))
            x0 = int(max(0, w[0] - side_px / 2)); y0 = int(max(0, w[1] - side_px / 2))
            patch = img[y0:y0 + side_px, x0:x0 + side_px]
            out[side] = describe(patch if patch.size else None)
        rows.append({'t': round(t, 2), 'left': out['left'], 'right': out['right']})
        if (i + 1) % 1000 == 0:
            print(f'  {i + 1}/{len(frames)} 帧', flush=True)
    cap.release()

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({'video': str(a.video), 'desc_dim': 16, 'frames': rows},
                                ensure_ascii=False), encoding='utf-8')
    print(f'写出 {a.out}（{len(rows)} 帧 × 32 维）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
