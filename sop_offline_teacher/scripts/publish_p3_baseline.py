"""把 P3 pose-only 基线的逐帧预测，发成前端可切换的「结果来源」。

流程：pred_extra.json（全片逐帧预测）→ 截取片段窗口(300–420s) → 状态机去抖
     → 写成 segments_proposal.json → 交给 publish_segments.py 发布

这样网页上能直接看到"纯本地、零 API"的模型效果，与 VLM 的几个 run 并排比较。

用法：
    python scripts/publish_p3_baseline.py --pred api_runs/pred_extra.json --name p3_pose
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_segments import CLASSES, run_lengths, state_machine  # noqa: E402

NAME_CN = {'S1': '取镜座组件', 'S2': '取镜座上盖', 'S3': '组装镜座上盖',
           'S4': '组装镜座密封塞', 'S5': '紧固镜子镜座', 'S6': '系上镜座扎带',
           'idle_auxiliary': '循环外停留与准备'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', type=Path, required=True)
    ap.add_argument('--name', default='p3_pose')
    ap.add_argument('--clip-offset', type=float, default=300.0)
    ap.add_argument('--duration', type=float, default=119.99)
    ap.add_argument('--confirm', type=int, default=8)
    ap.add_argument('--min-dur', type=float, default=2.0)
    ap.add_argument('--out', type=Path, default=Path('api_runs/p3_pose'))
    a = ap.parse_args()

    d = json.loads(a.pred.read_text(encoding='utf-8-sig'))
    lo, hi = a.clip_offset, a.clip_offset + a.duration
    frames = [f for f in d['frames'] if lo <= f['t'] <= hi]
    if not frames:
        sys.exit(f'{a.pred} 里没有落在 {lo}–{hi}s 的帧')
    print(f'取片段内帧 {len(frames)} 帧（{lo:.0f}–{hi:.0f}s）')

    import numpy as np
    ts = np.array([f['t'] for f in frames])
    pred = np.array([f['pred'] for f in frames])
    true = np.array([f['true'] for f in frames])
    acc = float((pred == true).mean())
    print(f'片段内帧级准确率 {acc:.1%}')

    raw = run_lengths(ts, pred)
    sm = state_machine(raw, a.confirm, a.min_dur, 1.0)
    print(f'段数：原始 {len(raw)} → 状态机后 {len(sm)}')

    segs = []
    for i, (s, e, lab) in enumerate(sm):
        cname = CLASSES[int(lab)]
        segs.append({
            'start': round(max(0.0, s - a.clip_offset), 3),
            'end': round(min(a.duration, e - a.clip_offset), 3),
            'step_id': cname if cname.startswith('S') else 'idle',
            'step_name': NAME_CN.get(cname, cname),
            'action': f'纯本地骨架模型判定为「{NAME_CN.get(cname, cname)}」',
            'state_change': 'none', 'evidence_i_used': '10Hz 骨架几何特征 + 2 层 MLP + 状态机去抖',
            'left_hand': '', 'right_hand': '',
            'confidence': None, 'need_review': False,
        })
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'segments_proposal.json').write_text(json.dumps({
        'source': 'p3-pose-baseline',
        'note': f'pose-only 无泄漏基线（7 周期留一验证，帧级 {acc:.1%}）+ 状态机去抖',
        'segments': segs}, ensure_ascii=False, indent=1), encoding='utf-8')

    from publish_segments import publish
    publish(a.out, a.name, clip_offset=0.0, duration=a.duration)
    return 0


if __name__ == '__main__':
    sys.exit(main())
