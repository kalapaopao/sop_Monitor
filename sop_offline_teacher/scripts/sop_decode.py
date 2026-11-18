"""方案 C：在「每段每步分数」上做序列解码（Viterbi + 顺序约束 + 时长先验）。

两种输入模式：
  --mode keyword  从描述文本用关键词打分（退路，不需要 API）
  --mode step     直接用方案 A 输出的 step_id + confidence 作为观测分（推荐）

为什么需要它：
  模型逐段独立判断，会把工序顺序判错（S3 判在 S1 前面）。
  装配工序有强顺序性和时长先验，用这两条可以把孤立错误拉回来。

转移概率怎么来（重要）：
  从人工 GT 统计，但用 --exclude-range 排除被评估区间，避免"用答案改答案"。

时长先验怎么来：
  configs/sop_steps.json 里每个步骤的 expected_sec（例如 S5 典型 20~45 秒）。
  落在区间内加分、偏离扣分，权重由 --prior-weight 控制。

用法：
    # 方案 A 的输出（含 step_id/confidence）+ 顺序与时长约束
    python scripts/sop_decode.py --mode step --proposal api_runs/run_A/segments_proposal.json \
        --gt web/public/data/gt_segments.json --exclude-range 300 420 \
        --out api_runs/decoded_A

    # 没有 step_id 时退回关键词模式
    python scripts/sop_decode.py --mode keyword --proposal api_runs/free_clean/segments_proposal.json \
        --gt web/public/data/gt_segments.json --exclude-range 300 420 --out api_runs/decoded_kw
"""
from __future__ import annotations
import argparse, json, math, sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SOP_IDS = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
EXTRA = ['idle', 'unknown']
ALL_STATES = SOP_IDS + EXTRA


def load_cfg(path: Path):
    d = json.loads(path.read_text(encoding='utf-8-sig'))
    b2s = {s['boris_behavior']: s['step_id'] for s in d['steps'] if s.get('boris_behavior')}
    order = {}
    expected = {}
    for s in d['steps']:
        order[s['step_id']] = s.get('order', 99)
        if len(s.get('expected_sec') or []) == 2:
            expected[s['step_id']] = tuple(s['expected_sec'])
    return d, b2s, order, expected


def transitions_from_gt(gt_path: Path, b2s: dict, exclude=None, gap_min=0.5):
    """从 GT 统计转移，并把标注之间的空隙也作为 idle 状态纳入时间序列。

    踩过的坑：早期版本只统计"有标注的段"，于是 idle 从不出现，
    解码器几乎不可能输出 idle，导致"空隙越多越吃亏"（实测把命中率从 30.8% 拉到 23.1%）。
    """
    g = json.loads(gt_path.read_text(encoding='utf-8-sig'))
    segs = [s for s in sorted(g['segments'], key=lambda x: x['start'])
            if not (exclude and s['end'] > exclude[0] and s['start'] < exclude[1])]
    seq, prev_end = [], None
    for s in segs:
        if prev_end is not None and s['start'] - prev_end >= gap_min:
            seq.append('idle')
        seq.append(b2s.get(s['behavior'], 'unknown'))
        prev_end = s['end']
    return Counter(zip(seq, seq[1:])), len(seq)


def log_trans(cnt):
    """拉普拉斯平滑后的 log P(next|cur)，未见转移给小概率而非 0。"""
    totals = Counter()
    for (a, _b), c in cnt.items():
        totals[a] += c
    table = {}
    for a in ALL_STATES:
        tot = totals.get(a, 0) + len(ALL_STATES) * 0.5
        for b in ALL_STATES:
            table[(a, b)] = math.log((cnt.get((a, b), 0) + 0.5) / tot)
    return table


def viterbi(emissions, trans, gamma=0.0):
    if not emissions:
        return []
    prev = {st: emissions[0].get(st, 0.0) for st in ALL_STATES}
    back = []
    for i in range(1, len(emissions)):
        cur, bk = {}, {}
        for st in ALL_STATES:
            best, arg = -1e18, None
            for pst, psc in prev.items():
                val = psc + gamma * trans.get((pst, st), -6.0)
                if val > best:
                    best, arg = val, pst
            cur[st] = best + emissions[i].get(st, 0.0)
            bk[st] = arg
        back.append(bk)
        prev = cur
    last = max(prev, key=prev.get)
    path = [last]
    for bk in reversed(back):
        path.append(bk[path[-1]])
    return list(reversed(path))


def dur_prior(seg, sid, expected, weight):
    """时长先验：落在 expected_sec 内给 +weight，偏离给 -weight。"""
    if not weight or sid not in expected:
        return 0.0
    lo, hi = expected[sid]
    dur = float(seg['end']) - float(seg['start'])
    if lo <= dur <= hi:
        return weight
    # 允许一定弹性：明显超出才罚
    if dur < lo * 0.5 or dur > hi * 2.0:
        return -weight
    return 0.0


def neighbors_of(sid, order, enabled=False):
    """相邻步骤。默认关闭（enabled=False）：它隐含"步骤有序"的假设，
    取消强制顺序后不再使用。"""
    if not enabled or sid not in order:
        return []
    o = order[sid]
    return [k for k, v in order.items() if abs(v - o) == 1]


def emissions_step(segs, order, expected, prior_weight, neighbor_boost=False):
    out = []
    for s in segs:
        sid = str(s.get('step_id') or '').strip()
        conf = s.get('confidence')
        conf = float(conf) if isinstance(conf, (int, float)) else 0.8
        em = {k: 0.0 for k in ALL_STATES}
        if sid in em:
            em[sid] += conf * 2.0 + dur_prior(s, sid, expected, prior_weight)
            for nb in neighbors_of(sid, order, neighbor_boost):
                em[nb] += conf * 0.6 + dur_prior(s, nb, expected, prior_weight * 0.5)
        else:
            em['unknown'] += 1.0
        out.append(em)
    return out


def emissions_keyword(segs, steps_path, vocab_path, order, expected, prior_weight):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from map_to_sop import load_steps, score_step
    _, steps, _, _ = load_steps(steps_path)
    banned, oos = {}, []
    if vocab_path.exists():
        v = json.loads(vocab_path.read_text(encoding='utf-8-sig'))
        banned = v.get('banned_synonyms') or {}
        oos = v.get('out_of_scope_objects') or []
    out = []
    for s in segs:
        text = ' '.join(str(s.get(k) or '') for k in ('action', 'evidence_i_used', 'left_hand', 'right_hand'))
        em = {k: 0.0 for k in ALL_STATES}
        for st in steps:
            if st['step_id'] in SOP_IDS:
                sc, _h, _b = score_step(text, st, banned, oos)
                em[st['step_id']] = float(sc) + dur_prior(s, st['step_id'], expected, prior_weight)
        top = max(em[k] for k in SOP_IDS)
        if top == 0:
            em['idle'] = 0.5 if any(w in text for w in oos) else 0.0
            em['unknown'] = 0.5
        out.append(em)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--proposal', type=Path, required=True)
    ap.add_argument('--mode', choices=['step', 'keyword'], default='step')
    ap.add_argument('--gt', type=Path, default=Path('web/public/data/gt_segments.json'))
    ap.add_argument('--steps', type=Path, default=Path('configs/sop_steps.json'))
    ap.add_argument('--vocab', type=Path, default=Path('configs/teacher_vocab.json'))
    ap.add_argument('--exclude-range', nargs=2, type=float, default=None,
                    help='统计转移时排除的区间（填被评估区间，避免作弊）')
    ap.add_argument('--gamma', type=float, default=0.0,
                    help='顺序约束权重。默认 0=完全不做顺序约束（按要求取消强制顺序）。'
                         '实测在固定窗口 + 描述与 GT 不符的数据上，加顺序反而更差')
    ap.add_argument('--prior-weight', type=float, default=0.0,
                    help='时长先验权重。默认 0：当前结果是固定 6 秒窗口切分，'
                         '用「整段时长」比 expected_sec 会系统性误罚 S5（20~45s）。'
                         '只在「先检测动作段再做判定」的流程里才建议开启')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()

    cfg, b2s, order, expected = load_cfg(a.steps)
    cnt, n = transitions_from_gt(a.gt, b2s, tuple(a.exclude_range) if a.exclude_range else None)
    trans = log_trans(cnt)
    print(f'转移统计：GT 排除 {a.exclude_range or "无"} 后剩 {n} 段 → {len(cnt)} 种转移')
    if cnt:
        print('  最常见：', '、'.join(f'{x}->{y}×{c}' for (x, y), c in cnt.most_common(6)))

    doc = json.loads(a.proposal.read_text(encoding='utf-8-sig'))
    segs = doc.get('segments') or []
    if a.mode == 'step':
        ems = emissions_step(segs, order, expected, a.prior_weight)
    else:
        ems = emissions_keyword(segs, a.steps, a.vocab, order, expected, a.prior_weight)

    path = viterbi(ems, trans, a.gamma)
    g = json.loads(a.gt.read_text(encoding='utf-8-sig'))
    gt_segs = g['segments']

    def gt_at(t):
        for s in gt_segs:
            if s['start'] - 1e-6 <= t <= s['end'] + 1e-6:
                return b2s.get(s['behavior'], 'unknown')
        return 'idle'

    def raw_choice(em):
        return max(em, key=em.get)

    print(f'\n解码 {len(segs)} 段（模式 {a.mode}，gamma={a.gamma}，时长先验={a.prior_weight}）')
    print(f'{"时间":>16} {"原始":>8} {"解码":>8} {"GT":>8}  描述')
    ok_raw = ok_dec = 0
    for s, em, dec in zip(segs, ems, path):
        center = (float(s['start']) + float(s['end'])) / 2
        raw, gt = raw_choice(em), gt_at(center)
        ok_raw += int(raw == gt)
        ok_dec += int(dec == gt)
        if dec == gt and raw != gt:
            mark = '  <-- 被解码修正'
        elif dec != gt and raw == gt:
            mark = '  <-- 被解码弄错'
        elif dec != gt:
            mark = '  <-- 仍错'
        else:
            mark = ''
        print(f'{s["start"]:7.1f}-{s["end"]:6.1f} {raw:>8} {dec:>8} {gt:>8}  '
              f'{str(s.get("action") or "")[:24]}{mark}')
    tot = max(1, len(segs))
    print(f'\n按「模型段对齐 GT」口径：原始 {ok_raw}/{tot} = {ok_raw / tot:.1%} → '
          f'解码 {ok_dec}/{tot} = {ok_dec / tot:.1%}')

    # GT 视角（更公平）：每个 GT 段是否被覆盖它的模型段判对
    def gt_view(use_decoded):
        ok, details = 0, []
        for gseg in gt_segs:
            if a.exclude_range and not (gseg['end'] > a.exclude_range[0] and gseg['start'] < a.exclude_range[1]):
                continue
            want = b2s.get(gseg['behavior'], 'unknown')
            pick = None
            for s, em, dec in zip(segs, ems, path):
                inter = min(gseg['end'], s['end']) - max(gseg['start'], s['start'])
                if inter > 0 and inter / max(1e-6, gseg['end'] - gseg['start']) >= 0.5:
                    pick = (dec if use_decoded else raw_choice(em))
                    break
            hit = pick == want
            ok += int(hit)
            details.append((gseg['start'], gseg['end'], want, pick, hit))
        return ok, len(details), details

    ok_raw_g, n_g, det_raw = gt_view(False)
    ok_dec_g, _, det_dec = gt_view(True)
    print(f'按「GT 段是否判对」口径：原始 {ok_raw_g}/{n_g} = {ok_raw_g / max(1, n_g):.1%} → '
          f'解码 {ok_dec_g}/{n_g} = {ok_dec_g / max(1, n_g):.1%}')
    print('\n逐 GT 段：')
    for (s0, e0, want, _p, _h), (_, _, _, pick, hit) in zip(det_raw, det_dec):
        print(f'  GT {s0:7.1f}-{e0:7.1f} 应为 {want:<7} 解码后 {pick if pick else "(无覆盖)":<8}'
              f' {"OK" if hit else ""}')

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'decoded.json').write_text(json.dumps({
        'proposal': str(a.proposal), 'mode': a.mode, 'gamma': a.gamma,
        'prior_weight': a.prior_weight, 'exclude_range': a.exclude_range,
        'agreement_segmentwise': [ok_raw / tot, ok_dec / tot],
        'agreement_gtview': [ok_raw_g / max(1, n_g), ok_dec_g / max(1, n_g)],
        'segments': [{'start': s['start'], 'end': s['end'],
                      'raw': raw_choice(em), 'decoded': dec,
                      'gt': gt_at((float(s['start']) + float(s['end'])) / 2),
                      'action': s.get('action', '')}
                     for s, em, dec in zip(segs, ems, path)]}, ensure_ascii=False, indent=1),
        encoding='utf-8')
    print('\n写出', a.out / 'decoded.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
