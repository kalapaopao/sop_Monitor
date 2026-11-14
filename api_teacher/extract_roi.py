"""从 RTMW 手部关键点裁出左右手 ROI 小图（零标注的"物体侧证据"）。

设计要点：
- 手框完全由 21 个关键点算出来，不需要任何人标框，也不需要额外检测模型；
- 裁图时按外接框中心取正方形并外扩（默认 40%），保证把"手里那个东西"一起框进来；
- 姿态是 5 Hz 均匀采样，所以默认走**顺序解码**而不是逐帧 seek——快 10 倍以上。

用法：
    python api_teacher/extract_roi.py \
        --video vedio/v2_5m_7m_h264.mp4 \
        --pose web/public/data/local_pose_2min.json \
        --out web/public/data/roi --size 96 --pad 0.4

产出：
    web/public/data/roi/0000_012.4_L.jpg ...     左右手小图
    web/public/data/roi/roi_index.json          每帧的手框 + 图片路径 + 有效性
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import cv2
import numpy as np

SCORE_THR = 0.18      # 与 run_local_pose.py / render_pose_video.py 保持一致
MIN_POINTS = 4


def hand_box(points, frame_w, frame_h, pad=0.4, max_side_ratio=0.4):
    """21 个关键点 → 正方形手框（含外扩，剔除离群点）。

    直接用 min/max 会被个别"飞点"带偏：实测最大框边长到过 963 px（画面一半），
    手只占框面积 12%，等于裁了一大片背景。所以先按中位数 + MAD 剔除离群点。
    """
    valid = [p for p in points if p[2] > SCORE_THR]
    if len(valid) < MIN_POINTS:
        return None
    xs = np.asarray([p[0] for p in valid], np.float32)
    ys = np.asarray([p[1] for p in valid], np.float32)

    # 用中位数中心 + MAD 剔除飞点（只保留 3σ 内的点）
    cx0, cy0 = float(np.median(xs)), float(np.median(ys))
    dx = np.abs(xs - cx0); dy = np.abs(ys - cy0)
    mad_x = float(np.median(dx)) * 1.4826 + 1e-6
    mad_y = float(np.median(dy)) * 1.4826 + 1e-6
    keep = (dx <= 3.0 * mad_x) & (dy <= 3.0 * mad_y)
    if keep.sum() < MIN_POINTS:
        keep = np.ones(len(xs), bool)
    xs, ys = xs[keep], ys[keep]

    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    cx = (x0 + x1) / 2.0; cy = (y0 + y1) / 2.0
    side = max(x1 - x0, y1 - y0)
    side = max(side, 16.0)                                  # 退化手（点挤在一起）给个下限
    side = min(side, min(frame_w, frame_h) * max_side_ratio)  # 上限，避免半个画面
    side *= (1.0 + 2.0 * pad)                               # 外扩，把手边工具/工件带进来

    half = side / 2.0
    bx0 = max(0, int(round(cx - half))); by0 = max(0, int(round(cy - half)))
    bx1 = min(frame_w, int(round(cx + half))); by1 = min(frame_h, int(round(cy + half)))
    if bx1 - bx0 < 8 or by1 - by0 < 8:
        return None
    return [bx0, by0, bx1, by1]


def is_uniform(ts, tol=0.1):
    if len(ts) < 3:
        return False
    d = np.diff(np.asarray(ts, np.float32))
    med = float(np.median(d))
    return med > 0 and bool(np.all(np.abs(d - med) <= med * tol))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', type=Path, required=True)
    ap.add_argument('--pose', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--size', type=int, default=96, help='输出边长（正方形）')
    ap.add_argument('--pad', type=float, default=0.4, help='ROI 裁图的外扩比例（要带进手里的工具/工件）')
    ap.add_argument('--limit', type=int, default=0, help='只处理前 N 帧，0=全部')
    ap.add_argument('--jpeg-quality', type=int, default=88)
    ap.add_argument('--box-pad', type=float, default=0.10,
                    help='回写到姿态 JSON 的显示框外扩比例（画面上要贴合手，不能和 ROI 一样大）')
    ap.add_argument('--write-boxes', action='store_true',
                    help='把 left_box / right_box 回写进 --pose 指向的姿态 JSON（前端据此在画面上画手框）')
    a = ap.parse_args()

    pose = json.loads(a.pose.read_text(encoding='utf-8'))
    rows = pose['frames']
    if a.limit:
        rows = rows[:a.limit]
    a.out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(a.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ts = [r['timestamp'] for r in rows]
    sequential = is_uniform(ts)
    print(json.dumps({'frames': len(rows), 'fps': fps, 'seq': sequential, 'size': a.size, 'pad': a.pad}))

    index = []
    saved = 0
    cursor = -1  # 顺序解码时已读到的帧号
    for i, row in enumerate(rows):
        t = row['timestamp']
        if sequential:
            want = int(round(t * fps))
            while cursor < want:
                ok = cap.grab()
                if not ok:
                    break
                cursor += 1
            ok, frame = cap.retrieve() if cursor >= want else (False, None)
        else:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
        if not ok or frame is None:
            index.append({'i': i, 't': t, 'left': None, 'right': None, 'error': 'decode'})
            continue

        entry = {'i': i, 't': t, 'left': None, 'right': None}
        for key, tag in (('left_hand', 'L'), ('right_hand', 'R')):
            box = hand_box(row.get(key) or [], fw, fh, a.pad)
            if a.write_boxes:
                # 显示框用更小的外扩，画面上才贴合手
                row['left_box' if tag == 'L' else 'right_box'] = hand_box(
                    row.get(key) or [], fw, fh, a.box_pad)
            if box is None:
                continue
            x0, y0, x1, y1 = box
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (a.size, a.size), interpolation=cv2.INTER_AREA)
            name = f'{i:04d}_{t:06.1f}_{tag}.jpg'
            cv2.imwrite(str(a.out / name), crop, [cv2.IMWRITE_JPEG_QUALITY, a.jpeg_quality])
            entry['left' if tag == 'L' else 'right'] = {'file': name, 'box': box}
            saved += 1
        index.append(entry)

    (a.out / 'roi_index.json').write_text(
        json.dumps({'video': str(a.video), 'size': a.size, 'pad': a.pad,
                    'score_thr': SCORE_THR, 'min_points': MIN_POINTS,
                    'frames': index}, ensure_ascii=False), encoding='utf-8')

    if a.write_boxes:
        pose['hand_box'] = {'size': a.size, 'roi_pad': a.pad, 'display_pad': a.box_pad,
                            'score_thr': SCORE_THR, 'min_points': MIN_POINTS, 'max_side_ratio': 0.4}
        a.pose.write_text(json.dumps(pose, ensure_ascii=False), encoding='utf-8')
        print(json.dumps({'boxes_written': str(a.pose), 'frames': len(rows)}))
    left_ok = sum(1 for e in index if e.get('left'))
    right_ok = sum(1 for e in index if e.get('right'))
    print(json.dumps({'crops': saved, 'left_frames': left_ok, 'right_frames': right_ok,
                      'out': str(a.out)}))


if __name__ == '__main__':
    main()
