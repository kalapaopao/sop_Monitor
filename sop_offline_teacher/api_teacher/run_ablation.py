"""Run three Qwen evidence ablations on the same video windows."""
import argparse, base64, gzip, json, os
from pathlib import Path
import cv2
from openai import OpenAI

PROMPT = """你是工业现场动作分析教师模型。根据连续画面和可用证据，自由命名当前最具体的动作。
分别描述 action、phase、left_hand、right_hand、tool、object、relation、change、unknown。
只写画面或证据直接支持的内容，无法确认写 unknown。只输出 JSON。"""

def b64(path):
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()

def frames(video, out, start, duration, stride, count):
    cap=cv2.VideoCapture(str(video)); result=[]; i=0; t=start
    out.mkdir(parents=True, exist_ok=True)
    while t < start + duration:
        group=[]
        for j in range(count):
            ts=t + stride*j/max(1,count-1)
            cap.set(cv2.CAP_PROP_POS_MSEC, ts*1000); ok, frame=cap.read()
            if ok:
                p=out/f"frame_{i:04d}_{j:02d}.jpg"; cv2.imwrite(str(p),frame); group.append((ts,p))
        if group: result.append((t,min(t+stride,start+duration),group))
        t += stride; i += 1
    cap.release(); return result

def read_rows(path):
    with gzip.open(path,"rt",encoding="utf-8") as f: return [json.loads(x) for x in f]

def evidence(rows, start, end, mode):
    if mode == "A": return {}
    selected=[r for r in rows if start <= float(r.get("timestamp_sec",0)) < end]
    persons=[r for r in selected if r.get("selected_person_index") is not None]
    out={"frames":len(selected),"person_frames":len(persons)}
    if mode == "B":
        out["person_present"]=bool(persons)
        return out
    out["hand_detections"]=sum(sum(d.get("label")=="hand" for d in r.get("detections",[])) for r in selected)
    last=persons[-1] if persons else None
    if last:
        p=last["persons"][last["selected_person_index"]].get("keypoints",[])
        for name,idx in (("left_wrist",9),("right_wrist",10)):
            if idx < len(p): out[name]=[round(float(p[idx][0]),1),round(float(p[idx][1]),1),round(float(p[idx][2]),2)]
        out["hand_keypoints_available"]=len(p)>=133
    return out

def call(client, model, start, end, group, ev):
    content=[{"type":"text","text":f"窗口 {start:.1f}-{end:.1f} 秒。\n{PROMPT}\n结构化证据：{json.dumps(ev,ensure_ascii=False)}"}]
    for ts,p in group:
        content += [{"type":"text","text":f"画面时间 {ts:.2f} 秒"},{"type":"image_url","image_url":{"url":b64(p)}}]
    r=client.chat.completions.create(model=model,messages=[{"role":"user","content":content}],temperature=.1)
    return r.choices[0].message.content

def render(out, video, cases):
    rows=[]
    mode = next((name for name in "ABC" if cases.get(name)), None)
    if mode is None:
        raise ValueError("no ablation results to render")
    for i in range(len(cases[mode])):
        start,end=cases[mode][i]["start"],cases[mode][i]["end"]
        cards=[]
        item=cases[mode][i]
        images="".join(f"<img src='frames/{Path(p).name}'>" for p in item.get("frame_paths", []))
        cards.append(f"<section><h3>{mode}组</h3><div class='frames'>{images}</div><pre>{item['response']}</pre><small>证据：{json.dumps(item['evidence'],ensure_ascii=False)}</small></section>")
        rows.append(f"<article><h2>{start:.1f}s–{end:.1f}s</h2>{''.join(cards)}</article>")
    html="""<!doctype html><meta charset=utf-8><title>A/B/C 证据对照</title><style>
body{font-family:Arial,'Microsoft YaHei';background:#07111e;color:#eaf2f8;margin:24px}video{width:70%;background:#000}article{background:#132238;padding:16px;margin:16px 0;border-radius:10px}section{display:inline-block;vertical-align:top;width:96%;margin:8px 0;background:#0b1727;padding:10px;min-height:180px}.frames{display:flex;gap:8px}.frames img{width:30%;height:150px;object-fit:cover}h2{color:#f4c15d}h3{color:#55d6be}pre{white-space:pre-wrap;min-height:100px}small{color:#aab9c8}</style><h1>Qwen 三组证据对照</h1><p>原始视频编码在部分浏览器中可能黑屏，下面的窗口截图仍可直接核对画面。</p><video controls src="VIDEO"></video>CARDS""".replace("VIDEO",video.resolve().as_uri()).replace("CARDS","".join(rows))
    (out/"ablation_report.html").write_text(html,encoding="utf-8")

def main():
    p=argparse.ArgumentParser(); p.add_argument("video",type=Path); p.add_argument("--observations",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--mode",choices=["A","B","C"],required=True); p.add_argument("--duration",type=float,default=30); p.add_argument("--start",type=float,default=0); p.add_argument("--stride",type=float,default=3); p.add_argument("--frames",type=int,default=3); p.add_argument("--model",default=os.getenv("DASHSCOPE_MODEL","qwen3-vl-plus")); a=p.parse_args()
    if not os.getenv("DASHSCOPE_API_KEY"): raise SystemExit("DASHSCOPE_API_KEY is missing")
    client=OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"],base_url=os.getenv("DASHSCOPE_BASE_URL","https://dashscope.aliyuncs.com/compatible-mode/v1"))
    groups=frames(a.video,a.output/"frames",a.start,a.duration,a.stride,a.frames); rows=read_rows(a.observations); cases={m:[] for m in "ABC"}
    for start,end,group in groups:
        ev=evidence(rows,start,end,a.mode); response=call(client,a.model,start,end,group,ev); cases[a.mode].append({"start":start,"end":end,"evidence":ev,"response":response,"frame_paths":[str(x[1]) for x in group]}); print(f"{a.mode} {start:.1f}-{end:.1f}s",flush=True)
    (a.output/"ablation_results.json").write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding="utf-8"); render(a.output,a.video,cases)
if __name__ == "__main__": main()
