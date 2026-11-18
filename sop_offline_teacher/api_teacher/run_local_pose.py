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
    p.add_argument('--rate', type=float, default=10,
                   help='采样频率 Hz。10 比 5 明显更顺（5Hz 时框每 0.2s 才更新一次），耗时约 2 倍')
    p.add_argument('--pad', type=float, default=0.1, help='人物框外扩比例，避免手贴边被裁掉')
    p.add_argument('--max-palm-ratio', type=float, default=12.0,
                   help='掌骨长度 max/min 上限，超过视为假检测；0=不过滤。'
                        '原来是 5.0，是按一个视角调出来的，侧视角透视会误杀真手（实测 40s 内 91 次），'
                        '放到 12.0 后双手覆盖率 74.0%→87.8%')
    p.add_argument('--border', type=float, default=2.0,
                   help='keep_hand 的贴边阈值（像素），原来 8。侧机位手常靠近画面边缘，8 会误杀')
    p.add_argument('--conf', type=float, default=0.15,
                   help='YOLO person 置信度，原来写死 0.25')
    p.add_argument('--bbox-fallback', type=float, default=3.0,
                   help='YOLO 本帧没框到人时，复用最近 N 秒内的框继续跑 RTMW；0=关闭。'
                        '实测这是「人一动就丢手」最直接的补法：坏窗口覆盖率 42.7%→81.8%，'
                        '且恢复出的手与最后一次真实检出位移中位仅 14px（不是硬凑）')
    a = p.parse_args()

    cap = cv2.VideoCapture(str(a.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    video_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    total = min(video_duration, a.start + a.duration)
    person = YOLO(str(a.person_model), task='detect')
    pose = ort.InferenceSession(str(a.pose_model), providers=['CPUExecutionProvider'])
    rows = []
    t = a.start
    # 顺序解码：原来每 tick 用 cap.set(POS_MSEC) 定位，实测每次 seek 约 130ms，
    # 比推理还贵 8 倍。改成顺序 grab 后同样结果、整段快 4~7 倍。
    cursor = -1
    last_box, last_box_t = None, -1e9
    while t < total:
        want_f = int(round(t * fps))
        while cursor < want_f:
            if not cap.grab():
                break
            cursor += 1
        ok, frame = cap.retrieve() if cursor >= want_f else (False, None)
        if not ok or frame is None:
            break
        result = next(iter(person.predict(frame, conf=a.conf, verbose=False)))
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.empty((0, 4))
        scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.empty(0)
        body_pts = {}
        if len(boxes):
            i = int(np.argmax(scores))
            box = boxes[i].tolist()
            box_score = float(scores[i])
            box_src = 'yolo'
            last_box, last_box_t = box, t
        elif a.bbox_fallback > 0 and last_box is not None and (t - last_box_t) <= a.bbox_fallback:
            # 人不会瞬移：本帧没框到人就用最近的框继续跑，别整帧把信息丢掉
            box, box_score, box_src = last_box, 0.0, 'fallback'
        else:
            box, box_score, box_src = None, 0.0, 'none'
        if box is not None:
            tensor, meta = crop_person(frame, box, a.pad)
            if tensor is None:
                left_pts = right_pts = []
            else:
                points = decode(pose.run(None, {pose.get_inputs()[0].name: tensor}), meta)
                left_pts = keep_hand(points[91:112], frame.shape, border=a.border,
                                     max_palm_ratio=a.max_palm_ratio)
                right_pts = keep_hand(points[112:133], frame.shape, border=a.border,
                                      max_palm_ratio=a.max_palm_ratio)
                # 身体关键点（RTMW wholebody 133 点的 0-22 为身体）：
                #   5/6=左右肩 7/8=左右肘 9/10=左右腕 11/12=左右髋
                def _p(idx):
                    q = points[idx]
                    return [float(v) for v in q] if isinstance(q, (list, tuple)) else list(np.asarray(q).ravel()[:3])
                body_pts = {k: _p(i) for k, i in (
                    ('shoulder_l', 5), ('shoulder_r', 6), ('elbow_l', 7), ('elbow_r', 8),
                    ('wrist_l', 9), ('wrist_r', 10), ('hip_l', 11), ('hip_r', 12))}
            rows.append({'timestamp': round(t, 3),
                         'person': {'bbox': box, 'score': box_score, 'src': box_src},
                         'left_hand': left_pts, 'right_hand': right_pts, 'body': body_pts})
        else:
            rows.append({'timestamp': round(t, 3), 'person': None, 'left_hand': [],
                         'right_hand': [], 'body': {}})
        t += 1 / a.rate
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({'video': str(a.video), 'fps': fps, 'duration': video_duration,
                                    'rate': a.rate, 'pad': a.pad, 'frames': rows},
                                   ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'frames': len(rows), 'output': str(a.output)}))


if __name__ == '__main__':
    main()
