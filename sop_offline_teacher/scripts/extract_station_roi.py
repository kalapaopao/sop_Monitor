"""按 configs/stations.json 里的固定 ROI 裁图，供模型"重点关注"使用。

要点（与"裁剪掉画面"的区别）：
- 这里裁出的固定 ROI 只是**额外一路输入**；全局画面仍然完整进入模型，
  所以"人有没有走过来、整个人在不在画面里"这类信息不会丢。
- 每个区域按 tick 裁一张小图（默认 128×128），下游由共享图像编码器转成特征向量，
  再和骨架、手周 ROI 特征一起送入时序模型。
- type=state 的区域额外算一个"变化强度"（与参考帧逐像素差分的平均绝对值），
  用来判断"这个工位上有没有东西被放上去/被拿走"，这个指标可解释、可验证。

用法：
    python scripts/extract_station_roi.py \
        --video vedio/v2_5m_7m_h264.mp4 \
        --stations configs/stations.json \
        --pose web/public/data/local_pose_2min.json \
        --out web/public/data/station_roi --size 128
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2
import numpy as np


def region_box(rect_norm, w, h):
    x0, y0, x1, y1 = rect_norm
    bx0 = int(round(x0 * w)); by0 = int(round(y0 * h))
    bx1 = int(round(x1 * w)); by1 = int(round(y1 * h))
    bx0 = max(0, min(bx0, w - 2)); by0 = max(0, min(by0, h - 2))
    bx1 = max(bx0 + 2, min(bx1, w)); by1 = max(by0 + 2, min(by1, h))
    return [bx0, by0, bx1, by1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', type=Path, required=True)
    ap.add_argument('--stations', type=Path, default=Path('configs/stations.json'))
    ap.add_argument('--pose', type=Path, default=None, help='用它的时间戳作为 tick（保证与骨架/手周 ROI 对齐）')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--size', type=int, default=128)
    ap.add_argument('--rate', type=float, default=10.0, help='没有 --pose 时的采样频率')
    ap.add_argument('--duration', type=float, default=0, help='没有 --pose 时的时长，0=整段')
    ap.add_argument('--jpeg-quality', type=int, default=88)
    a = ap.parse_args()

    cfg = json.loads(a.stations.read_text(encoding='utf-8-sig'))
    station = (cfg.get('stations') or [None])[0]
    if not station:
        raise SystemExit('stations.json 里没有工位')
    regions = [r for r in (station.get('regions') or []) if r.get('enabled', True)]
    if not regions:
        raise SystemExit('该工位没有启用的区域')

    cap = cv2.VideoCapture(str(a.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_s = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps

    if a.pose:
        pose = json.loads(a.pose.read_text(encoding='utf-8-sig'))
        ticks = [float(r['timestamp']) for r in pose['frames']]
    else:
        dur = a.duration if a.duration > 0 else total_s
        n = max(1, int(round(dur * a.rate)))
        ticks = [i / a.rate for i in range(n)]

    a.out.mkdir(parents=True, exist_ok=True)
    boxes = {r['id']: region_box(r['rect_norm'], fw, fh) for r in regions}
    ref = {r['id']: None for r in regions}   # 参考帧（state 区域用来算变化强度）

    index = []
    saved = 0
    cursor = -1
    for i, t in enumerate(ticks):
        want = int(round(t * fps))
        while cursor < want:                 # 顺序解码，比逐帧 seek 快一个量级
            ok = cap.grab()
            if not ok:
                break
            cursor += 1
        ok, frame = cap.retrieve() if cursor >= want else (False, None)
        if not ok or frame is None:
            index.append({'i': i, 't': t, 'error': 'decode'})
            continue

        entry = {'i': i, 't': round(t, 3), 'regions': {}}
        for r in regions:
            rid = r['id']; x0, y0, x1, y1 = boxes[rid]
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                entry['regions'][rid] = None
                continue
            small = cv2.resize(crop, (a.size, a.size), interpolation=cv2.INTER_AREA)
            name = f"{i:05d}_{t:06.1f}_{rid}.jpg"
            cv2.imwrite(str(a.out / name), small, [cv2.IMWRITE_JPEG_QUALITY, a.jpeg_quality])
            saved += 1

            info = {'file': name, 'box': [x0, y0, x1, y1], 'type': r.get('type', 'focus')}
            if r.get('type') == 'state':
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.int16)
                if ref[rid] is None:
                    ref[rid] = gray
                    info['change'] = 0.0
                else:
                    info['change'] = round(float(np.abs(gray - ref[rid]).mean()), 2)
            entry['regions'][rid] = info
        index.append(entry)

    cap.release()
    (a.out / 'station_roi_index.json').write_text(json.dumps({
        'video': str(a.video), 'station': station.get('id'), 'station_name': station.get('name'),
        'size': a.size, 'frame': [fw, fh], 'regions': regions, 'frames': index,
    }, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'ticks': len(ticks), 'crops': saved, 'regions': len(regions), 'out': str(a.out)}))


if __name__ == '__main__':
    main()
