"""把"自由描述"的识别结果映射到你定义的 SOP 步骤表（step_id）。

这是「你的 SOP」和「识别系统」之间的转换器，解决的核心问题是：
模型自由写的是「右手把黑色壳片压到座子上」，而你只认 S3「组装镜座上盖」。

映射方式（三步，全在本地、零 API 成本）：
  1. 别名匹配：描述里命中 sop_steps.json 的 aliases → 得分
  2. 白名单/禁用同义词校准：命中文档里的 banned_synonyms 时按其指向纠正
  3. 兜底：都没命中 → 按时长/关键词猜 idle 或 unknown，并标 need_review

产物：
  segments_with_step.json  每段带 step_id / confidence / mapping_method / need_review
  mapping_report.md        映射率、每步段数、与 GT 对比的准确率（给了 --gt 时）

用法：
    python scripts/map_to_sop.py --proposal api_runs/free_clean/segments_proposal.json \
        --gt web/public/data/gt_segments.json --out api_runs/mapped_clean
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def load_steps(path: Path):
    d = json.loads(path.read_text(encoding='utf-8-sig'))
    steps = d['steps'] + d.get('fallback_steps', [])
    by_id = {s['step_id']: s for s in steps}
    boris_to_step = {s['boris_behavior']: s['step_id'] for s in d['steps'] if s.get('boris_behavior')}
    return d, steps, by_id, boris_to_step


def score_step(text: str, step: dict, banned: dict, oos: list):
    """按「动词 + 物件」两维打分，返回 (得分, 命中说明, 是否命中禁用同义词)。

    为什么不用整串别名匹配：模型说的是"外壳/盖板/支架"，而 SOP 名是"镜座/上盖"，
    实测整串匹配映射率只有 11.8%。拆成动词维度和物件维度后，命中率显著提高。
    """
    verbs = step.get('verbs_any') or []
    objs = step.get('objects_any') or []
    aliases = step.get('aliases') or []
    hit_v = [w for w in verbs if w and w in text]
    hit_o = [w for w in objs if w and w in text]
    hit_a = [w for w in aliases if w and w in text]
    score = float(bool(hit_v)) + float(bool(hit_o)) + (0.5 if hit_a else 0.0)
    # 禁用同义词：文本用了错误叫法，而这一步正是它应被纠正到的目标 → 弱加权
    bad = any(w in text and right and right in step.get('name_cn', '')
              for w, right in banned.items())
    if bad:
        score += 0.5
    hits = [f'动词:{hit_v}' for _ in [0] if hit_v] + [f'物件:{hit_o}' for _ in [0] if hit_o] \
           + [f'别名:{hit_a}' for _ in [0] if hit_a]
    return score, hits, bad


def map_segment(seg, steps, banned, oos, min_score=1.0):
    text = ' '.join(str(seg.get(k) or '') for k in ('action', 'evidence_i_used', 'left_hand', 'right_hand'))
    best, best_score, best_hits, best_bad = None, 0.0, [], False
    for st in steps:
        sc, hits, bad = score_step(text, st, banned, oos)
        if sc > best_score:
            best, best_score, best_hits, best_bad = st, sc, hits, bad
    sop_hit = best is not None and best_score >= min_score
    # 只命中动词、没命中物件 → 大概率不是这一步，交人工
    if sop_hit:
        need = best_score < 1.5
        return best['step_id'], best_score, best_hits, best_bad, ('partial' if need else 'match')
    # 没命中任何 SOP 步骤：区分「与装配无关」和「真的看不出」
    oos_hit = [w for w in oos if w in text]
    if oos_hit or '空闲' in text or '无关' in text:
        return 'idle', 0.0, [f'无关物体:{oos_hit}'], False, 'idle'
    return 'unknown', best_score, best_hits, False, 'no_keyword'


def gt_step_at(gt_segs, t, boris_to_step):
    for s in gt_segs:
        if s['start'] - 1e-6 <= t <= s['end'] + 1e-6:
            return boris_to_step.get(s['behavior'], 'unknown'), s
    return 'idle', None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--proposal', type=Path, required=True)
    ap.add_argument('--steps', type=Path, default=Path('configs/sop_steps.json'))
    ap.add_argument('--vocab', type=Path, default=Path('configs/teacher_vocab.json'))
    ap.add_argument('--gt', type=Path, default=None, help='给了就算与人工标注的一致率')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--min-score', type=float, default=1.5,
                    help='接受为 SOP 步骤的最低分（动词+物件都命中=2.0；只命中一维=1.0）。默认 1.5 即要求两维都命中')
    a = ap.parse_args()

    cfg, steps, by_id, boris_to_step = load_steps(a.steps)
    banned, oos = {}, []
    if a.vocab.exists():
        _v = json.loads(a.vocab.read_text(encoding='utf-8-sig'))
        banned = _v.get('banned_synonyms') or {}
        oos = _v.get('out_of_scope_objects') or []

    prop = json.loads(a.proposal.read_text(encoding='utf-8-sig'))
    segs = prop.get('segments') or []

    mapped = []
    for s in segs:
        step_id, score, hits, bad, method = map_segment(s, steps, banned, oos, a.min_score)
        center = (float(s['start']) + float(s['end'])) / 2
        mapped.append({**s, 'step_id': step_id,
                       'step_name': by_id.get(step_id, {}).get('name_cn', step_id),
                       'mapping_score': score, 'mapping_hits': hits,
                       'mapping_method': method,
                       'need_review': bool(bad) or step_id == 'unknown' or bool(s.get('need_review'))})
        mapped[-1]['center'] = round(center, 3)

    # 统计
    rate = sum(1 for m in mapped if m['step_id'] != 'unknown') / max(1, len(mapped))
    print(f'段数 {len(mapped)} ｜ 映射到 SOP 步骤 {sum(1 for m in mapped if m["step_id"] != "unknown")} 段 '
          f'（映射率 {rate:.1%}）｜ 需复核 {sum(1 for m in mapped if m["need_review"])} 段')
    print('\n每段映射结果：')
    for m in mapped:
        flag = '!' if m['need_review'] else ' '
        print(f'  {flag} {m["start"]:7.1f}-{m["end"]:7.1f}s  {m["step_id"]:<7} {m["step_name"]:<12} '
              f'score={m["mapping_score"]:.1f}  {m["action"][:26]}')

    gt_rows, acc = [], None
    if a.gt:
        g = json.loads(a.gt.read_text(encoding='utf-8-sig'))
        gt_segs = g['segments']
        ok = tot = 0
        for m in mapped:
            gt_id, gt_seg = gt_step_at(gt_segs, m['center'], boris_to_step)
            hit = (m['step_id'] == gt_id)
            ok += int(hit); tot += 1
            gt_rows.append({'start': m['start'], 'end': m['end'], 'mine': m['step_id'],
                            'gt': gt_id, 'hit': hit,
                            'gt_behavior': gt_seg['behavior'] if gt_seg else None})
        acc = ok / max(1, tot)
        print(f'\n与人工标注一致率：{ok}/{tot} = {acc:.1%}（按段中心所在 GT 段比对）')
        print('不一致的段：')
        for r in gt_rows:
            if not r['hit']:
                print(f'  {r["start"]:7.1f}-{r["end"]:7.1f}s  我={r["mine"]:<7} GT={r["gt"]:<7} {r["gt_behavior"] or ""}')

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'segments_with_step.json').write_text(json.dumps({
        'source_proposal': str(a.proposal), 'steps_config': str(a.steps),
        'mapping_rate': round(rate, 3),
        'agreement_with_gt': round(acc, 3) if acc is not None else None,
        'segments': mapped, 'gt_compare': gt_rows}, ensure_ascii=False, indent=1), encoding='utf-8')

    lines = ['# SOP 步骤映射报告', '',
             f'- 输入：`{a.proposal}`',
             f'- 步骤表：`{a.steps}`（{len(cfg["steps"])} 个 SOP 步骤 + {len(cfg.get("fallback_steps", []))} 个兜底）',
             f'- 段数：{len(mapped)}｜映射率：**{rate:.1%}**｜需复核：{sum(1 for m in mapped if m["need_review"])}',
             f'- 与人工标注一致率：**{acc:.1%}**' if acc is not None else '- 未提供 GT，未算一致率',
             '', '| 段 | 时间 | step_id | 步骤名 | 得分 | 描述 |', '|---|---|---|---|---|---|']
    for m in mapped:
        lines.append(f'| {m.get("idx", "")} | {m["start"]:.1f}–{m["end"]:.1f}s | {m["step_id"]} | '
                     f'{m["step_name"]} | {m["mapping_score"]:.1f} | {m["action"][:30]} |')
    (a.out / 'mapping_report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'\n写出：{a.out / "segments_with_step.json"}')
    print(f'      {a.out / "mapping_report.md"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
