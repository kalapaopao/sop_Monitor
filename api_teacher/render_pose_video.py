"""把 RTMW 手骨架烧进原视频，输出可直接在网页播放的叠加效果视频。

这样网页里看到的骨架和画面对应同一帧，不存在任何“两路画面叠加/时间不同步”的问题；
关键点按 5 Hz 采样，渲染时会按相邻两帧线性插值，做到 25 fps 连续跟随。

用法：
    python api_teacher/render_pose_video.py \
        --video vedio/v2_5m_7m_h264.mp4 \
        --pose web/public/data/local_pose_2min.json \
        --output web/public/video/v2_5m_7m_pose.mp4

需要 ffmpeg（默认取 PATH 里的 ffmpeg，可用 --ffmpeg 指定绝对路径）。
"""
from __future__ import annotations
import argparse, json, shutil, subprocess, sys
from pathlib import Path
import cv2, numpy as np

LEFT_COLOR = (190, 214, 85)    # #55d6be
RIGHT_COLOR = (97, 162, 244)   # #f4a261
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
         (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
         (0, 17), (17, 18), (18, 19), (19, 20)]
SCORE_THR = 0.18


def load_pose(path: Path):
    data = json.loads(path.read_text(encoding='utf-8'))
    rows = data['frames']
    ts = np.array([r['timestamp'] for r in rows], np.float32)
    def pack(key):
        arr = np.full((len(rows), 21, 3), np.nan, np.float32)
        for i, r in enumerate(rows):
            pts = r.get(key) or []
            if len(pts) == 21:
                arr[i] = np.array(pts, np.float32)
        return arr
    return data, ts, pack('left_hand'), pack('right_hand')


def hand_at(ts, arr, t):
    """取 t 时刻的手部关键点；相邻两帧都有效时线性插值，否则退化为最近的有效帧（限 0.25s）。"""
    if len(ts) == 0:
        return None
    i = min(max(int(np.searchsorted(ts, t)), 0), len(ts) - 1)
    if i == 0:
        return arr[0] if ts[0] - 0.25 <= t and not np.all(np.isnan(arr[0])) else None
    a, b = i - 1, i
    if np.all(np.isnan(arr[a])) or np.all(np.isnan(arr[b])):
        j = a if abs(ts[a] - t) <= abs(ts[b] - t) else b
        if abs(ts[j] - t) <= 0.25 and not np.all(np.isnan(arr[j])):
            return arr[j]
        return None
    span = float(ts[b] - ts[a])
    w = 0.0 if span <= 0 else float(np.clip((t - ts[a]) / span, 0.0, 1.0))
    return arr[a] * (1 - w) + arr[b] * w


def draw_hand(frame, pts, color, label):
    valid = [p for p in pts if p[2] > SCORE_THR]
    if len(valid) < 4:
        return
    for a, b in EDGES:
        if pts[a][2] > SCORE_THR and pts[b][2] > SCORE_THR:
            cv2.line(frame, (int(pts[a][0]), int(pts[a][1])), (int(pts[b][0]), int(pts[b][1])), color, 3, cv2.LINE_AA)
    for p in valid:
        cv2.circle(frame, (int(p[0]), int(p[1])), 4, color, -1, cv2.LINE_AA)
    x, y = int(valid[0][0]) + 8, int(valid[0][1]) - 10
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 5, cv2.LINE_AA)
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)


def draw_box(frame, box, color, label, dashed=False, thickness=2):
    """画框。dashed=True 用于"上下文 ROI"（模型实际看的范围），实线用于手框。"""
    if not box:
        return
    x0, y0, x1, y1 = [int(v) for v in box]
    if dashed:
        # OpenCV 没有虚线矩形，手工画四段的短划
        def seg(p, q, step=18):
            px, py = p; qx, qy = q
            length = max(abs(qx - px), abs(qy - py))
            n = max(1, length // step)
            for k in range(0, n, 2):
                a = k / n; b = min(1.0, (k + 1) / n)
                cv2.line(frame, (int(px + (qx - px) * a), int(py + (qy - py) * a)),
                         (int(px + (qx - px) * b), int(py + (qy - py) * b)), color, thickness, cv2.LINE_AA)
        seg((x0, y0), (x1, y0)); seg((x1, y0), (x1, y1))
        seg((x1, y1), (x0, y1)); seg((x0, y1), (x0, y0))
    else:
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)
    ty = max(20, y0 - 8)
    cv2.putText(frame, label, (x0 + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 4, cv2.LINE_AA)
    cv2.putText(frame, label, (x0 + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def box_at(rows, ts, idx_key, t, max_gap=0.35):
    """取 t 时刻的框，相邻两帧都有效时**线性插值**坐标。

    早期版本这里直接取最近帧、不插值，理由是"框是离散判定"——这个理由只对
    "有没有框"成立，对"框在哪、多大"不成立。姿态只有 5 Hz，视频 24.7 fps，
    约 5 个视频帧共用同一个姿态样本，所以不插值的框看起来每 0.2 秒卡一下。
    现在：两端都有效 → 按时间线性插值；只有一端有效 → 用最近的那个（限 max_gap）。
    """
    if len(ts) == 0:
        return None
    i = int(np.searchsorted(ts, t))
    if i <= 0:
        b = rows[0].get(idx_key)
        return b if (b and ts[0] - max_gap <= t) else None
    if i >= len(ts):
        b = rows[-1].get(idx_key)
        return b if (b and t - ts[-1] <= max_gap) else None
    a, bb = i - 1, i
    pa, pb = rows[a].get(idx_key), rows[bb].get(idx_key)
    if pa and pb:
        span = float(ts[bb] - ts[a])
        w = 0.0 if span <= 0 else float(np.clip((t - ts[a]) / span, 0.0, 1.0))
        return [pa[k] * (1 - w) + pb[k] * w for k in range(4)]
    if pa and t - ts[a] <= max_gap:
        return pa
    if pb and ts[bb] - t <= max_gap:
        return pb
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', type=Path, required=True)
    ap.add_argument('--pose', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--ffmpeg', default=None)
    ap.add_argument('--crf', type=int, default=20)
    ap.add_argument('--start', type=float, default=0)
    ap.add_argument('--duration', type=float, default=0, help='0 表示到视频结尾')
    ap.add_argument('--boxes', action='store_true',
                    help='把左右手框也烧进画面（框来自姿态 JSON 的 left_box/right_box）')
    a = ap.parse_args()

    ffmpeg = a.ffmpeg or shutil.which('ffmpeg')
    if not ffmpeg:
        sys.exit('找不到 ffmpeg，请用 --ffmpeg 指定路径')

    data, ts, left, right = load_pose(a.pose)
    rows = data['frames']
    has_boxes = bool(a.boxes) and any(r.get('left_box') or r.get('right_box') for r in rows[:50])
    if a.boxes and not has_boxes:
        print('WARN: 姿态 JSON 里没有 left_box/right_box，先跑 extract_roi.py --write-boxes')
    cap = cv2.VideoCapture(str(a.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first = int(a.start * fps)
    last = total_frames if a.duration <= 0 else min(total_frames, int((a.start + a.duration) * fps))
    if first:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)

    a.output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, '-y', '-loglevel', 'error',
           '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', f'{fps}', '-i', '-',
           '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', str(a.crf),
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(a.output)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    drawn = 0
    boxed = 0
    for idx in range(first, last):
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        lp, rp = hand_at(ts, left, t), hand_at(ts, right, t)
        if lp is not None:
            draw_hand(frame, lp, LEFT_COLOR, 'L'); drawn += 1
        if rp is not None:
            draw_hand(frame, rp, RIGHT_COLOR, 'R'); drawn += 1
        if has_boxes:
            lb = box_at(rows, ts, 'left_box', t)
            rb = box_at(rows, ts, 'right_box', t)
            lr = box_at(rows, ts, 'left_roi', t)
            rr = box_at(rows, ts, 'right_roi', t)
            if lr:
                draw_box(frame, lr, LEFT_COLOR, 'L roi', dashed=True, thickness=2); boxed += 1
            if rr:
                draw_box(frame, rr, RIGHT_COLOR, 'R roi', dashed=True, thickness=2); boxed += 1
            if lb:
                draw_box(frame, lb, LEFT_COLOR, 'L'); boxed += 1
            if rb:
                draw_box(frame, rb, RIGHT_COLOR, 'R'); boxed += 1
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    code = proc.wait()
    cap.release()
    print(json.dumps({'output': str(a.output), 'frames': last - first,
                      'hand_draws': drawn, 'box_draws': boxed, 'ffmpeg_exit': code}))


if __name__ == '__main__':
    main()
