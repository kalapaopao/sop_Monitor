"""客观检验参考实现的 SOP 是否"稳定地跟着画面走"。

用户的标准很明确：只看 SOP 步骤能不能对上实时画面，手/工具描述是加分项。
那我就不看描述，只看两件事：

1. **是不是自定义 SOP**：名称是不是针对该产品/工位的具体说法（而不是通用类别）
2. **是不是稳地跟着画面**：相邻 5 秒采样点之间，SOP 名称是否保持稳定
   —— 如果每一帧都在换名字，说明它是"逐帧生成"，不是"分段"；
      如果长时间保持同一个名字、只在少数时刻切换，才说明它真的在做工序分段。

OCR 会认错字，所以名字比较用相似度（difflib），不是严格相等。

用法
----
    .venv\\Scripts\\python.exe scripts/check_ref_sop_stability.py --ocr api_runs/ref_ocr_all.jsonl
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def clean_name(s: str) -> str:
    """只留中文/字母数字，去掉 OCR 常混进来的标点噪声。"""
    s = norm(s)
    s = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]", "", s)
    return s


def similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr", type=Path, required=True)
    ap.add_argument("--sim", type=float, default=0.72,
                    help="相邻两帧 SOP 名相似度高于此值视为同一段（OCR 容错）")
    a = ap.parse_args()

    rows = [json.loads(x) for x in a.ocr.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = [r for r in rows if r.get("lines")]

    seq = []   # (file, t, name)
    for r in rows:
        lines = [norm(x) for x in r["lines"]]
        i = next((k for k, v in enumerate(lines) if "标准SOP" in v or "SOP描" in v), None)
        if i is None:
            continue
        name = None
        for v in lines[i + 1:]:
            if "当前动作" in v:
                break
            if v and not re.search(r"\d{2}[:：]\d{2}", v) and 2 <= len(clean_name(v)) <= 20:
                name = clean_name(v)
                break
        if name:
            t = float(re.search(r"_(\d+\.\d)s", r["file"]).group(1))
            seq.append((r["file"], t, name))

    if not seq:
        raise SystemExit("没解析出任何 SOP 名称")

    segs = []          # [start_t, end_t, 代表名]
    changes = []
    for f, t, nm in sorted(seq, key=lambda x: x[1]):
        if not segs or similar(segs[-1][2], nm) < a.sim:
            if segs:
                changes.append((t, segs[-1][2], nm))
            segs.append([t, t, nm])
        else:
            segs[-1][1] = t

    n = len(seq)
    same = sum(1 for i in range(1, n) if similar(seq[i - 1][2], seq[i][2]) >= a.sim)
    span = seq[-1][1] - seq[0][1]

    print(f"采样 {n} 帧（每 5 秒）｜覆盖 {span:.0f} 秒")
    print(f"相邻帧 SOP 名称保持一致的比例：{same}/{n - 1} = {same / max(1, n - 1):.1%}")
    print(f"由此推出 SOP 段数：{len(segs)} 段（平均 {span / max(1, len(segs)):.0f} 秒/段）")
    print(f"\n=== 推断出的 SOP 段落 ===")
    print(f"{'起':>7}{'止':>7}{'时长':>7}  名称（OCR 原文）")
    for s0, s1, nm in segs:
        print(f"{s0:>7.0f}{s1:>7.0f}{s1 - s0 + 5:>7.0f}  {nm}")

    print(f"\n=== 每次切换的位置（说明它真的在找工序边界）===")
    for t, before, after in changes[:30]:
        print(f"  {t:>6.0f}s  「{before[:16]}」 → 「{after[:16]}」")
    if len(changes) > 30:
        print(f"  …另有 {len(changes) - 30} 次")

    print(f"\n=== 判断 ===")
    stab = same / max(1, n - 1)
    if stab > 0.75:
        print(f"稳定性 {stab:.0%}：**是分段，不是逐帧生成**。名称能长时间保持，只在 {len(changes)} 处切换。")
    elif stab > 0.5:
        print(f"稳定性 {stab:.0%}：介于两者之间，可能做了分段但边界较碎。")
    else:
        print(f"稳定性 {stab:.0%}：更像逐帧/逐窗口生成，名称频繁变化。")

    # 自定义程度：名称长度与具体性
    lens = [len(x[2]) for x in segs]
    print(f"名称长度：中位 {sorted(lens)[len(lens) // 2]} 字（越长越像针对具体工位的自定义说法）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
