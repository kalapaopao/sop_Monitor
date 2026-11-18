"""从参考实现的 OCR 结果里抽出它的字段结构和 SOP 词表。

背景：用户认为这段参考实现的输出质量很高。我无法看画面，只能靠 OCR 的文字
来判断——所以要先把"它到底输出哪些字段、用了哪些 SOP 名称"客观列出来，
再和自己的方案逐项对比。

OCR 常见毛病：中文字之间被插入空格、个别字认错（描述→描还/描达）。
所以这里先去掉所有空格再做锚点匹配，并允许少量字形误差。

用法
----
    .venv\\Scripts\\python.exe scripts/analyze_ref_ocr.py --ocr api_runs/ref_ocr_all.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def norm(s: str) -> str:
    """去掉所有空白，OCR 会给中文逐字插空格。"""
    return re.sub(r"\s+", "", s or "")


def is_anchor(line, *keys):
    n = norm(line)
    return any(k in n for k in keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr", type=Path, required=True)
    a = ap.parse_args()

    rows = []
    for ln in a.ocr.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rows.append(json.loads(ln))
        except Exception:
            continue
    ok = [r for r in rows if r.get("lines")]
    print(f"帧 {len(rows)}，有文字 {len(ok)}")

    sop_names = Counter()
    sop_codes = Counter()
    action_examples = []
    hand_device_missing = 0
    hand_device_total = 0
    visibility = Counter()
    frames_seen = 0

    for r in ok:
        lines = [norm(x) for x in r["lines"]]
        # 定位「标准SOP描述」区块：取其后到「当前动作描述」之间的非空行
        i = next((k for k, v in enumerate(lines) if "标准SOP" in v or "SOP描" in v), None)
        if i is None:
            continue
        frames_seen += 1
        block = []
        for v in lines[i + 1:]:
            if "当前动作" in v or "当前蔀" in v:
                break
            if v:
                block.append(v)
        if block:
            name = block[0]
            # 名称多为 4~12 个汉字，去掉明显是代码/时间的行
            if not re.search(r"\d{2}:\d{2}", name) and 2 <= len(name) <= 16:
                sop_names[name] += 1
            for v in block[1:]:
                m = re.match(r"^([A-Za-z_]{2,12}\d*|[A-Z]\d+|[A-Za-z_]+)", v)
                if m and not v.startswith("00"):
                    sop_codes[m.group(1)] += 1

        # 当前动作描述
        j = next((k for k, v in enumerate(lines) if "当前动作" in v or "当莉动作" in v), None)
        if j is not None:
            for v in lines[j + 1:j + 3]:
                if v and not re.search(r"\d{2}:\d{2}", v) and len(v) > 6:
                    if len(action_examples) < 400:
                        action_examples.append(v)
                    break

        # 器件 / 目标 / 可见性
        for v in lines:
            if v.startswith("器件") or v.startswith("目标") or "件：" in v[:4]:
                hand_device_total += 1
                if "未" in v or "床" in v:      # 未确认（OCR 常把"未"认成"床"）
                    hand_device_missing += 1
            for w in ("可见", "局部可见", "未确认", "遮挡", "不可见"):
                if w in v:
                    visibility[w] += 1

    print(f"\n=== 他们出现的标准 SOP 名称（{len(sop_names)} 种）===")
    for name, c in sop_names.most_common(40):
        print(f"  {c:>4}  {name}")
    if sop_codes:
        print(f"\n=== 步骤代码（共 {len(sop_codes)} 种）===")
        print("  " + "  ".join(f"{k}×{v}" for k, v in sop_codes.most_common(20)))
    print(f"\n=== 当前动作描述样例（{len(action_examples)} 条）===")
    for x in action_examples[:12]:
        print(f"  · {x}")
    print(f"\n=== 器件/目标 字段 ===")
    print(f"  出现 {hand_device_total} 次，其中含「未/床(未)」= 未确认 {hand_device_missing} 次"
          f"（{hand_device_missing / hand_device_total:.1%}）" if hand_device_total else "  未出现")
    print(f"\n=== 可见性用词 ===")
    for k, v in visibility.most_common():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
