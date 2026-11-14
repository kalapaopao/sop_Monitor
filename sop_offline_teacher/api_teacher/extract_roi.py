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


def hand_box(points, frame_w, frame_h, pad=0.4, max_side_ratio=0.4, robust=True):
    """21 个关键点 → 正方形手框（含外扩）。

    robust=True  （ROI 裁图用）：先按中位数 + MAD 剔除离群点，避免被"飞点"带偏，
                                实测裁图最大框到过 963 px（画面一半）。
    robust=False （画面显示框用）：用全部有效点的范围，保证框住整只手、不被 MAD 切掉指尖，
                                只用 max_side_ratio 兜住极端情况。
    """
    valid = [p for p in points if p[2] > SCORE_THR]
    if len(valid) < MIN_POINTS:
        return None
    xs = np.asarray([p[0] for p in valid], np.float32)
    ys = np.asarray([p[1] for p in valid], np.float32)

    if robust:
        cx0, cy0 = float(np.median(xs)), float(np.median(ys))
        dx = np.abs(xs - cx0); dy = np.abs(ys - cy0)
        mad_x = float(np.median(dx)) * 1.4826 + 1e-6
        mad_y = float(np.median(dy)) * 1.4826 + 1e-6
        keep = (dx <= 3.0 * mad_x) & (dy <= 3.0 * mad_y)
        if keep.sum() >= MIN_POINTS:
            xs, ys = xs[keep], ys[keep]

    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    cx = (x0 + x1) / 2.0; cy = (y0 + y1) / 2.0
    side = max(x1 - x0, y1 - y0)
    side = max(side, 16.0)                                    # 退化手（点挤在一起）给个下限
    side = min(side, min(frame_w, frame_h) * max_side_ratio)  # 上限，避免半个画面
    side *= (1.0 + 2.0 * pad)                                 # 外扩，把手边工具/工件带进来

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


def hand_extent(points, score_thr=SCORE_THR):
    """稳健的手部尺度：MAD 剔离群后的长边边长 + 中心。无效返回 None。

    注意：这是"手的尺寸"，不是"手 + 手持物"的尺寸。手持物伸出去多少
    与工具/工件的实际大小有关，固定百分比外扩永远适配不了，
    所以 ROI 边界改用「k × 本尺度」并做时间平滑，见 context_roi()。
    """
    valid = [p for p in points if p[2] > score_thr]
    if len(valid) < MIN_POINTS:
        return None
    xs = np.asarray([p[0] for p in valid], np.float32)
    ys = np.asarray([p[1] for p in valid], np.float32)
    cx0, cy0 = float(np.median(xs)), float(np.median(ys))
    dx = np.abs(xs - cx0); dy = np.abs(ys - cy0)
    mad_x = float(np.median(dx)) * 1.4826 + 1e-6
    mad_y = float(np.median(dy)) * 1.4826 + 1e-6
    keep = (dx <= 3.0 * mad_x) & (dy <= 3.0 * mad_y)
    if keep.sum() >= MIN_POINTS:
        xs, ys = xs[keep], ys[keep]
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    return {'extent': max(x1 - x0, y1 - y0), 'cx': (x0 + x1) / 2.0, 'cy': (y0 + y1) / 2.0}


def smooth_scale(rows, key, win_sec=1.6):
    """对手部尺度与中心做**高斯加权均值**平滑，返回与 rows 等长的序列。

    手张开/握拳会让关键点外接框在 59~648 px 之间跳（10 倍），
    直接用瞬时值当 ROI 边长，同一只手会被裁成完全不同的比例，
    CNN 学到的外观特征就不可比。

    这里刻意用高斯加权均值而不是中值：中值是"分段常数"，相邻采样点经常取到
    完全相同的值，插值后仍然是一段一段静止，看起来卡顿（实测零位移帧占 26~32%）。
    均值则每个采样点都略有变化，配合渲染时插值，运动是连续的。
    底层每个样本的尺度已经过 MAD 剔离群，所以均值不会被"飞点"带偏。
    """
    stats = [hand_extent(r.get(key) or []) for r in rows]
    ts = np.asarray([r['timestamp'] for r in rows], np.float32)
    out = []
    half = win_sec / 2.0
    sigma = max(win_sec / 4.0, 1e-3)
    for i, cur in enumerate(stats):
        if cur is None:
            out.append(None)
            continue
        lo = int(np.searchsorted(ts, ts[i] - half))
        hi = int(np.searchsorted(ts, ts[i] + half, side='right'))
        exts, cxs, cys, wts = [], [], [], []
        for j in range(lo, hi):
            s = stats[j]
            if s is None:
                continue
            w = float(np.exp(-0.5 * ((ts[j] - ts[i]) / sigma) ** 2))
            exts.append(s['extent']); cxs.append(s['cx']); cys.append(s['cy']); wts.append(w)
        if not wts:
            out.append(cur)
            continue
        wsum = sum(wts)
        out.append({
            'extent': float(sum(e * w for e, w in zip(exts, wts)) / wsum),
            'cx': float(sum(c * w for c, w in zip(cxs, wts)) / wsum),
            'cy': float(sum(c * w for c, w in zip(cys, wts)) / wsum),
        })
    return out, stats


def context_roi(smoothed, frame_w, frame_h, person_h=None,
                person_k=0.35, hand_k=2.2, side_min=180, side_max=700):
    """上下文 ROI：覆盖"手 + 手持工具/工件"。

    边长取自两个尺度里较大的那个：
      · person_k × 人体框高度 —— 人体尺度比手部稳定得多，几乎不抖，
        且随机位远近自动缩放（工具/工件的绝对尺寸本来就和人相关）；
      · hand_k × 平滑后的手部尺度 —— 兜底（没检测到人时用）。
    """
    if not smoothed:
        return None
    cands = [smoothed['extent'] * hand_k]
    if person_h:
        cands.append(person_h * person_k)
    side = float(min(max(max(cands), side_min), side_max))
    half = side / 2.0
    bx0 = int(round(smoothed['cx'] - half)); by0 = int(round(smoothed['cy'] - half))
    bx1 = int(round(smoothed['cx'] + half)); by1 = int(round(smoothed['cy'] + half))
    # 贴边时把框整体平移回画面内，避免裁图出现黑边导致尺度失真
    if bx0 < 0: bx1 -= bx0; bx0 = 0
    if by0 < 0: by1 -= by0; by0 = 0
    if bx1 > frame_w: bx0 -= (bx1 - frame_w); bx1 = frame_w
    if by1 > frame_h: by0 -= (by1 - frame_h); by1 = frame_h
    bx0 = max(0, bx0); by0 = max(0, by0)
    if bx1 - bx0 < 8 or by1 - by0 < 8:
        return None
    return [bx0, by0, bx1, by1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', type=Path, required=True)
    ap.add_argument('--pose', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--size', type=int, default=96, help='输出边长（正方形）')
    ap.add_argument('--pad', type=float, default=0.4, help='ROI 裁图的外扩比例（要带进手里的工具/工件）')
    ap.add_argument('--limit', type=int, default=0, help='只处理前 N 帧，0=全部')
    ap.add_argument('--jpeg-quality', type=int, default=88)
    ap.add_argument('--person-k', type=float, default=0.35, help='ROI 边长 = k x 人体框高度（主尺度）')
    ap.add_argument('--hand-k', type=float, default=2.2, help='ROI 边长下限 = k x 平滑手部尺度')
    ap.add_argument('--scale-window', type=float, default=1.6, help='手部尺度时间平滑窗口(秒)')
    ap.add_argument('--scale-min', type=float, default=180, help='ROI 边长下限(px)')
    ap.add_argument('--scale-max', type=float, default=700, help='ROI 边长上限(px)')
    ap.add_argument('--box-pad', type=float, default=0.32,
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

    # 计算左右手的平滑尺度（时间中值），用于上下文 ROI 的边长与中心
    smooth = {key: smooth_scale(rows, key, a.scale_window)[0] for key in ('left_hand', 'right_hand')}
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
            # 上下文 ROI：用平滑尺度算，覆盖"手 + 手持工具/工件"
            person = row.get('person')
            person_h = None
            if person and person.get('bbox'):
                person_h = float(person['bbox'][3] - person['bbox'][1])
            roi = context_roi(smooth[key][i], fw, fh, person_h,
                              a.person_k, a.hand_k, a.scale_min, a.scale_max)
            if a.write_boxes:
                # 手框：不剔离群点（否则指尖会被切掉），只表示"手在哪"
                row['left_box' if tag == 'L' else 'right_box'] = hand_box(
                    row.get(key) or [], fw, fh, a.box_pad, robust=False)
                # 上下文 ROI 也写进去，前端/渲染脚本据此画出模型实际看的范围
                row['left_roi' if tag == 'L' else 'right_roi'] = roi
            if roi is None:
                continue
            x0, y0, x1, y1 = roi
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (a.size, a.size), interpolation=cv2.INTER_AREA)
            name = f'{i:04d}_{t:06.1f}_{tag}.jpg'
            cv2.imwrite(str(a.out / name), crop, [cv2.IMWRITE_JPEG_QUALITY, a.jpeg_quality])
            entry['left' if tag == 'L' else 'right'] = {'file': name, 'box': roi}
            saved += 1
        index.append(entry)

    (a.out / 'roi_index.json').write_text(
        json.dumps({'video': str(a.video), 'size': a.size, 'kind': 'context_roi',
                    'person_k': a.person_k, 'hand_k': a.hand_k, 'scale_window': a.scale_window,
                    'scale_min': a.scale_min, 'scale_max': a.scale_max,
                    'score_thr': SCORE_THR, 'min_points': MIN_POINTS,
                    'frames': index}, ensure_ascii=False), encoding='utf-8')

    if a.write_boxes:
        pose['hand_box'] = {'size': a.size, 'display_pad': a.box_pad,
                            'person_k': a.person_k, 'hand_k': a.hand_k, 'scale_window': a.scale_window,
                            'score_thr': SCORE_THR, 'min_points': MIN_POINTS, 'max_side_ratio': 0.4}
        a.pose.write_text(json.dumps(pose, ensure_ascii=False), encoding='utf-8')
        print(json.dumps({'boxes_written': str(a.pose), 'frames': len(rows)}))
    left_ok = sum(1 for e in index if e.get('left'))
    right_ok = sum(1 for e in index if e.get('right'))
    print(json.dumps({'crops': saved, 'left_frames': left_ok, 'right_frames': right_ok,
                      'out': str(a.out)}))


if __name__ == '__main__':
    main()
