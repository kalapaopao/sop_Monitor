"""Generate local person boxes and RTMW whole-body hand keypoints for the UI.

关键点坐标必须能直接画回原视频帧上，所以这里的 crop/letterbox 与反算必须严格成对：
crop 时按比例缩放后居中贴在 192x256 画布上（留下 left/top 黑边），decode 时必须先扣掉
这两个留边再除以 scale，否则关键点会整体偏移（此前版本漏了这一步，实测最多偏 57 px）。
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2, numpy as np, onnxruntime as ort
from ultralytics import YOLO

INPUT_W, INPUT_H = 192, 256
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)


def crop_person(frame, box, pad=0.1):
    """缩放 + 居中 letterbox 到 192x256，返回张量与逆变换所需的几何参数。"""
    h_img, w_img = frame.shape[:2]
    x0, y0, x1, y1 = [float(v) for v in box]
    if pad > 0:
        pad_w, pad_h = (x1 - x0) * pad, (y1 - y0) * pad
        x0 -= pad_w; x1 += pad_w; y0 -= pad_h; y1 += pad_h
    x0 = max(0.0, min(x0, w_img - 1)); x1 = max(x0 + 1, min(x1, float(w_img)))
    y0 = max(0.0, min(y0, h_img - 1)); y1 = max(y0 + 1, min(y1, float(h_img)))
    crop = frame[int(round(y0)):int(round(y1)), int(round(x0)):int(round(x1))]
    if crop.size == 0:
        return None, None
    h, w = crop.shape[:2]
    scale = min(INPUT_W / w, INPUT_H / h)
    rw = max(1, round(w * scale)); rh = max(1, round(h * scale))
    canvas = np.zeros((INPUT_H, INPUT_W, 3), np.uint8)
    left = (INPUT_W - rw) // 2; top = (INPUT_H - rh) // 2
    canvas[top:top + rh, left:left + rw] = cv2.resize(crop, (rw, rh))
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = (rgb - MEAN) / STD
    meta = {'x0': float(x0), 'y0': float(y0), 'scale': float(scale),
            'left': float(left), 'top': float(top), 'w': int(w), 'h': int(h),
            'rw': int(rw), 'rh': int(rh)}
    return np.transpose(rgb, (2, 0, 1))[None].astype(np.float32), meta


def decode(outputs, meta):
    """SimCC 输出（分箱步长 0.5 px）→ 原视频像素坐标。"""
    sx, sy = outputs[0][0], outputs[1][0]
    n = min(len(sx), 133)
    points = []
    for i in range(n):
        cx = float(np.argmax(sx[i]) / 2.0)
        cy = float(np.argmax(sy[i]) / 2.0)
        score = float((np.max(sx[i]) + np.max(sy[i])) / 2)
        x = meta['x0'] + (cx - meta['left']) / meta['scale']
        y = meta['y0'] + (cy - meta['top']) / meta['scale']
        points.append([x, y, score])
    return points


PALM_BONES = [(0, 1), (0, 5), (0, 9), (0, 13), (0, 17)]  # 腕点 → 5 个掌指关节


def palm_ratio(points, score_thr=0.18):
    """掌骨长度 max/min。正常手 5 根掌骨长度接近（实测 p50≈2.8、p90≈5.0）；
    关键点撒成"星形"时该比值会明显变大，可作为假检测的判据。"""
    pts = {i: np.array([p[0], p[1]], np.float32) for i, p in enumerate(points) if p[2] > score_thr}
    lengths = [float(np.linalg.norm(pts[a] - pts[b])) for a, b in PALM_BONES if a in pts and b in pts]
    if len(lengths) < 4:
        return None
    lo = min(lengths)
    if lo < 1e-3:
        return 1e9
    return max(lengths) / lo


def keep_hand(points, frame_shape, score_thr=0.18, min_extent=30.0, border=8.0, max_palm_ratio=5.0):
    """过滤模型在遮挡/出框时给出的假检测。

    三种情况视为无效：21 个点几乎重合、整只手贴画面边缘、掌骨长度比例离谱（撒成星形）。
    返回过滤后的点列表，无效时返回空列表。
    """
    valid = [p for p in points if p[2] > score_thr]
    if len(valid) < 4:
        return []
    xs = [p[0] for p in valid]; ys = [p[1] for p in valid]
    extent = max(max(xs) - min(xs), max(ys) - min(ys))
    h, w = frame_shape[:2]
    near_border = any(x < border or y < border or x > w - border or y > h - border for x, y in zip(xs, ys))
    if extent < min_extent or near_border:
        return []
    ratio = palm_ratio(points, score_thr)
    if max_palm_ratio and ratio is not None and ratio > max_palm_ratio:
        return []
    return points


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--person-model', type=Path, required=True)
    p.add_argument('--pose-model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--start', type=float, default=0)
    p.add_argument('--duration', type=float, default=30)
    p.add_argument('--rate', type=float, default=5)
    p.add_argument('--pad', type=float, default=0.1, help='人物框外扩比例，避免手贴边被裁掉')
    p.add_argument('--max-palm-ratio', type=float, default=5.0, help='掌骨长度 max/min 上限，超过视为假检测；0=不过滤')
    a = p.parse_args()

    cap = cv2.VideoCapture(str(a.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    video_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    total = min(video_duration, a.start + a.duration)
    person = YOLO(str(a.person_model), task='detect')
    pose = ort.InferenceSession(str(a.pose_model), providers=['CPUExecutionProvider'])
    rows = []
    t = a.start
    while t < total:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, frame = cap.read()
        if not ok:
            break
        result = next(iter(person.predict(frame, conf=.25, verbose=False)))
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.empty((0, 4))
        scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.empty(0)
        if len(boxes):
            i = int(np.argmax(scores))
            box = boxes[i].tolist()
            tensor, meta = crop_person(frame, box, a.pad)
            if tensor is None:
                left_pts = right_pts = []
            else:
                points = decode(pose.run(None, {pose.get_inputs()[0].name: tensor}), meta)
                left_pts = keep_hand(points[91:112], frame.shape, max_palm_ratio=a.max_palm_ratio)
                right_pts = keep_hand(points[112:133], frame.shape, max_palm_ratio=a.max_palm_ratio)
            rows.append({'timestamp': round(t, 3), 'person': {'bbox': box, 'score': float(scores[i])},
                         'left_hand': left_pts, 'right_hand': right_pts})
        else:
            rows.append({'timestamp': round(t, 3), 'person': None, 'left_hand': [], 'right_hand': []})
        t += 1 / a.rate
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({'video': str(a.video), 'fps': fps, 'duration': video_duration,
                                    'rate': a.rate, 'pad': a.pad, 'frames': rows},
                                   ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'frames': len(rows), 'output': str(a.output)}))


if __name__ == '__main__':
    main()
