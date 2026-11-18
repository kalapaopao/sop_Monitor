"""模拟浏览器保存 ROI 的完整往返，验证刷新后能读回。

为什么需要这个脚本
------------------
前端保存 = 读整份 stations.json → 只替换当前机位 → POST 整份 → 回读校验。
只测 GET/POST 各自可用，说明不了"改完刷新还在"。
这里完整走一遍：GET → 改坐标 → POST → 再 GET（等价于刷新）→ 比对坐标。

用法
----
    .venv\\Scripts\\python.exe scripts/verify_roi_roundtrip.py --station ST02 --base http://127.0.0.1:5174
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=15) as r:
        return json.loads(r.read().decode("utf-8"))


def post(base, path, doc):
    body = json.dumps(doc, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(base + path, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:5174")
    ap.add_argument("--station", default="ST02")
    ap.add_argument("--no-restore", action="store_true")
    a = ap.parse_args()

    orig = get(a.base, "/api/stations")
    idx = next((i for i, s in enumerate(orig["stations"]) if s["id"] == a.station), None)
    if idx is None:
        raise SystemExit(f"找不到机位 {a.station}")
    st0 = orig["stations"][idx]
    if not st0.get("regions"):
        raise SystemExit(f"机位 {a.station} 没有区域可测")
    r0 = st0["regions"][0]
    before = list(r0["rect_norm"])
    print(f"机位 {a.station} ｜ 区域 {r0['id']} ｜ 原始坐标 {before}")

    # 模拟"用户在界面上微调了 0.01"
    moved = [round(v + 0.01, 4) if i in (0, 2) else v for i, v in enumerate(before)]
    doc = copy.deepcopy(orig)
    doc["stations"][idx]["regions"][0]["rect_norm"] = moved

    out = post(a.base, "/api/stations", doc)
    print(f"POST ok={out.get('ok')} verified={out.get('verified')}")
    print(f"      file={out.get('file')}")
    print(f"      summary={out.get('summary')}")

    # 再读一次 = 刷新页面
    back = get(a.base, "/api/stations")
    st1 = next(s for s in back["stations"] if s["id"] == a.station)
    got = st1["regions"][0]["rect_norm"]
    ok = all(abs(x - y) < 1e-6 for x, y in zip(got, moved))
    print(f"\n刷新后读回坐标 {got}  → 与写入一致: {ok}")
    others = [(s["id"], len(s.get('regions') or [])) for s in back["stations"]]
    print(f"其它机位是否还在: {others}")

    log = Path("api_runs/roi_save_log.jsonl")
    if log.exists():
        lines = log.read_text(encoding="utf-8").strip().split("\n")
        print(f"\n保存日志 api_runs/roi_save_log.jsonl 共 {len(lines)} 条，最后一条：")
        print("  " + lines[-1])
    else:
        print("\n⚠ 没有生成 api_runs/roi_save_log.jsonl（检查 vite.config.js 是否已重启生效）")

    if not a.no_restore:
        doc2 = copy.deepcopy(back)
        i2 = next(i for i, s in enumerate(doc2["stations"]) if s["id"] == a.station)
        doc2["stations"][i2]["regions"][0]["rect_norm"] = before
        post(a.base, "/api/stations", doc2)
        chk = next(s for s in get(a.base, "/api/stations")["stations"] if s["id"] == a.station)
        print(f"\n已还原测试坐标 → {chk['regions'][0]['rect_norm']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
