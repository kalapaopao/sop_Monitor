"""标注质量体检 v2：按预测时间点筛（修正上一版按 GT 起始时间步进的网格错配）。"""
import json
import pathlib
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

C = ["S1", "S2", "S3", "S4", "S5", "S6", "idle_auxiliary"]
B2S = {"Remove_eyepiece_assembly": "S1",
       "Remove_the_upper_cover_of_the_eyepiece": "S2",
       "Assemble_the_mirror_base_cover": "S3",
       "Assemble_the_mirror_base_sealing_plug": "S4",
       "Fasten_mirror_base": "S5",
       "Fasten_tripod_strap": "S6"}
NAME = {"S1": "取镜座", "S2": "取上盖", "S3": "装上盖", "S4": "装密封塞",
        "S5": "紧固", "S6": "系扎带", "idle_auxiliary": "空闲"}

gt = json.loads(pathlib.Path("web/public/data/gt_segments.json").read_text(encoding="utf-8"))["segments"]
pred = json.loads(pathlib.Path("api_runs/pred_mlp_hmm.json").read_text(encoding="utf-8"))
pts = [(float(f["t"]), C[f["pred"]]) for f in pred["frames"]]

rows = []
for s in sorted(gt, key=lambda x: x["start"]):
    want = B2S.get(s["behavior"], "?")
    votes = Counter(v for t, v in pts if s["start"] - 1e-6 <= t <= s["end"] + 1e-6)
    n = sum(votes.values())
    if n == 0:
        continue
    top, topn = votes.most_common(1)[0]
    rows.append({"start": s["start"], "end": s["end"], "dur": s["dur"], "gt": want,
                 "model": top, "share": topn / n, "agree": votes.get(want, 0) / n, "n": n})

print("[一致性最低的 15 段 → 疑似标注可疑]")
print("  v2时间范围         时长    标注      模型多数    占比  一致率  帧数")
for r in sorted(rows, key=lambda r: r["agree"])[:15]:
    print("  %7.1f–%7.1f %6.2fs %-8s → %-8s %5.0f%% %5.0f%% %4d" % (
        r["start"], r["end"], r["dur"], NAME.get(r["gt"], r["gt"]),
        NAME.get(r["model"], r["model"]), r["share"] * 100, r["agree"] * 100, r["n"]))

print()
print("[统计] 有预测的 GT 段 %d 个：一致率≥80%% %d 个，50–80%% %d 个，<50%% %d 个" % (
    len(rows),
    sum(1 for r in rows if r["agree"] >= 0.8),
    sum(1 for r in rows if 0.5 <= r["agree"] < 0.8),
    sum(1 for r in rows if r["agree"] < 0.5)))

gaps, prev = [], None
for s in sorted(gt, key=lambda x: x["start"]):
    if prev is not None and s["start"] - prev > 0.5:
        gaps.append((prev, s["start"]))
    prev = s["end"]
if prev:
    gaps.append((prev, 599.2))
tot = sum(b - a for a, b in gaps)
print("[统计] 未标注空隙 %d 段 / 合计 %.0f 秒（%.0f%% 全片），最长 %.1f 秒" % (
    len(gaps), tot, tot / 599.2 * 100, max(b - a for a, b in gaps)))
print("[统计] 短于 1 秒的 GT 段：%d 个" % sum(1 for s in gt if s["dur"] < 1.0))
