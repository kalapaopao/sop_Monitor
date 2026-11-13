from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "api_runs"
OUT = RUNS / "ablation_dashboard.html"


def load(name: str) -> dict:
    path = RUNS / name / "ablation_results.json"
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    payload = {name: load(name)[name[-1]] for name in ("ablation_14_A", "ablation_14_B", "ablation_14_C")}
    html = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>动作分割结果 · 三次运行对齐看板</title>
<style>
:root{--bg:#07111e;--panel:#112238;--panel2:#0c1a2b;--line:#213850;--text:#eaf2f8;--muted:#8fa5b8;--cyan:#55d6be;--orange:#f4a261;--red:#ff5c61}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Arial,'Microsoft YaHei',sans-serif}
.app{max-width:1500px;margin:0 auto;padding:18px}.top{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}.top h1{font-size:20px;margin:0}.top small{color:var(--muted)}
.tabs{display:flex;gap:8px}.tab{border:1px solid var(--line);background:var(--panel2);color:var(--muted);padding:8px 18px;border-radius:8px;cursor:pointer}.tab.active{color:#06131f;background:var(--cyan);font-weight:bold}
.layout{display:grid;grid-template-columns:minmax(650px,1.65fr) minmax(390px,.8fr);gap:14px}.left,.card{background:var(--panel);border:1px solid var(--line);border-radius:10px}.left{padding:10px}video{display:block;width:100%;aspect-ratio:16/9;background:#000;border-radius:7px;object-fit:contain}.controls{display:flex;gap:10px;align-items:center;margin:10px 2px;color:var(--muted);font-size:13px}.controls input{flex:1;accent-color:var(--cyan)}.time{min-width:115px;text-align:right;color:var(--text)}
.counter{display:flex;align-items:center;gap:20px;padding:14px 4px;border-top:1px solid var(--line)}.counter strong{font-size:48px;color:var(--cyan);font-weight:400}.counter b{color:#dbe7f2;font-size:16px}.counter span{color:var(--muted);font-size:12px;line-height:1.7}
.cards{display:grid;gap:10px}.card{padding:12px;min-height:124px}.card h3{font-size:13px;margin:0 0 7px;color:var(--cyan)}.card .main{font-size:15px;color:#f6c778;line-height:1.5}.card p{margin:5px 0;color:var(--muted);font-size:12px;line-height:1.45}.card.warn h3{color:var(--orange)}
.timeline{margin-top:10px;background:#0b1b2b;border:1px solid var(--line);border-radius:10px;padding:10px}.timeline h3{font-size:13px;margin:0 0 7px;color:var(--muted)}.trackrow{display:flex;align-items:center;height:19px;margin:3px 0}.tracklabel{width:54px;color:#b6c6d5;font-size:12px}.bar{height:14px;display:flex;gap:2px;flex:1}.seg{height:100%;min-width:2px;background:#8496a8;border-radius:1px;cursor:pointer}.seg.active{outline:2px solid white;z-index:2}.seg.tight{background:#d9ae60}.seg.left{background:#43c7c0}.seg.right{background:#df9b61}.legend{display:flex;gap:14px;color:var(--muted);font-size:11px;margin-top:7px}.dot{display:inline-block;width:22px;height:6px;border-radius:2px;background:#29415b;margin-right:4px}.dot.active{background:var(--cyan)}.dot.tight{background:#f4c15d}
@media(max-width:900px){.layout{grid-template-columns:1fr}.top{align-items:flex-start;gap:10px;flex-direction:column}}
</style></head><body><main class="app">
<div class="top"><div><h1>动作分割结果 · 三次运行对齐看板</h1><small>周期工序 → 当前动作 / 左手动作 / 右手动作 / 操作器件与目标</small></div><div class="tabs" id="tabs"></div></div>
<div class="layout"><section class="left"><video id="video" controls preload="metadata" poster="ablation_14_30s/frames/frame_0000_00.jpg" src="../vedio/v2_5m_7m.mp4"></video><div id="videoFallback" style="display:none;padding:10px;color:#f4c15d;background:#261d12;border-radius:6px;font-size:13px"><img id="fallbackFrame" src="ablation_14_30s/frames/frame_0000_00.jpg" style="width:100%;max-height:360px;object-fit:contain;display:block;margin-bottom:8px">浏览器无法解码 5–7 分钟片段。上方图片来自本次 30 秒结果抽帧，可继续与右侧结果核对。</div><div class="controls"><span>时间</span><input id="slider" type="range" min="0" max="30" step="0.1" value="0"><span id="time" class="time">00:00.0 / 00:30.0</span></div><div class="counter"><strong id="count">0</strong><div><b>放钉关联紧固件计数</b><br><span>由当前运行结果按时间片段关联展示。每个片段对应左侧视频内容。</span></div></div></section><section class="cards"><article class="card"><h3>标准 SOP 描述</h3><div id="sop" class="main"></div><p id="sopMeta"></p></article><article class="card"><h3>当前动作描述</h3><div id="action" class="main"></div><p id="change"></p></article><article class="card"><h3>左手</h3><div id="left" class="main"></div><p id="leftMeta"></p></article><article class="card"><h3>右手</h3><div id="right" class="main"></div><p id="rightMeta"></p></article></section></div>
<section class="timeline"><h3>SOP 装配采样时间轴 · 点击色块跳转视频与右侧结果</h3><div class="trackrow"><span class="tracklabel">工序</span><div id="barProcess" class="bar"></div></div><div class="trackrow"><span class="tracklabel">左手</span><div id="barLeft" class="bar"></div></div><div class="trackrow"><span class="tracklabel">右手</span><div id="barRight" class="bar"></div></div><div class="legend"><span><i class="dot active"></i>当前片段</span><span><i class="dot tight"></i>紧固阶段</span><span>当前运行：<b id="runLabel">A</b></span></div></section>
</main><script>const DATA = __DATA__;
const runs={A:DATA.ablation_14_A,B:DATA.ablation_14_B,C:DATA.ablation_14_C};let run='A',idx=0;
const $=id=>document.getElementById(id), video=$('video'), slider=$('slider');
function parsed(item){try{return JSON.parse(item.response||'{}')}catch{return {}}}
function fmt(t){return '00:'+Number(t).toFixed(1).padStart(4,'0')}
function renderTabs(){ $('tabs').innerHTML=['A','B','C'].map(x=>`<button class="tab ${x===run?'active':''}" data-run="${x}">${x} 运行</button>`).join('');document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{run=b.dataset.run;idx=0;render()}) }
function renderBar(items){let duration=Math.max(1,items.length?items[items.length-1].end:30);let processColors=['#8496a8','#9bb5ca','#b8a7d5','#d9ae60','#86b5d6','#a4c6a8'];let make=(kind)=>items.map((x,i)=>{let p=parsed(x),tight=/screw|螺丝|fasten|tighten|拧紧/i.test((p.action||'')+' '+(p.phase||''));let text=kind==='process'?(p.phase||p.action||'未知工序'):kind==='left'?(p.left_hand||'未知'): (p.right_hand||'未知');let color=kind==='process'?`background:${processColors[i%processColors.length]}`:'';return `<span class="seg ${kind==='left'?'left ':kind==='right'?'right ':tight?'tight ':''}${i===idx?'active':''}" style="flex:${Math.max(.1,(x.end-x.start)/duration)};${color}" title="${x.start}s-${x.end}s ${text}" data-i="${i}"></span>`}).join('');$('barProcess').innerHTML=make('process');$('barLeft').innerHTML=make('left');$('barRight').innerHTML=make('right');document.querySelectorAll('.seg').forEach(s=>s.onclick=()=>{idx=+s.dataset.i;render()})}
function render(){let items=runs[run]||[],x=items[idx]||{},p=parsed(x),ev=x.evidence||{};slider.max=items.length?items[items.length-1].end:30;slider.value=x.start||0;$('time').textContent=`${fmt(x.start||0)} / ${fmt(slider.max)}`;$('runLabel').textContent=run;$('sop').textContent=`${p.phase||'未知阶段'} · ${p.object||'目标未识别'}`;$('sopMeta').textContent=`片段 ${fmt(x.start||0)} — ${fmt(x.end||0)} · ${ev.person_present===false?'未检测到人员':'人员可见'}`;$('action').textContent=p.action||'未知动作';$('change').textContent=p.change||'变化：未给出';$('left').textContent=p.left_hand||'未知';$('leftMeta').textContent=p.object?`器件：${p.tool||'未知'} · 目标：${p.object}`:'器件：未知 · 目标：未知';$('right').textContent=p.right_hand||'未知';$('rightMeta').textContent=`器件：${p.tool||'未知'} · 关系：${p.relation||'未知'}`;$('count').textContent=(items.slice(0,idx+1).filter(y=>/screw|螺丝|fasten|tighten|拧紧/i.test((parsed(y).action||'')+' '+(parsed(y).phase||''))).length);$('fallbackFrame').src=`ablation_14_30s/frames/frame_${String(idx).padStart(4,'0')}_00.jpg`;renderBar(items);renderTabs();if(video.duration&&Math.abs(video.currentTime-(x.start||0))>.2)video.currentTime=x.start||0}
slider.oninput=()=>{let t=+slider.value,items=runs[run]||[];idx=Math.min(items.length-1,Math.max(0,items.findIndex(x=>t>=x.start&&t<x.end)));if(idx<0)idx=items.length-1;render()};video.ontimeupdate=()=>{let t=video.currentTime,items=runs[run]||[];let i=items.findIndex(x=>t>=x.start&&t<x.end);if(i>=0&&i!==idx){idx=i;render()}};video.onerror=()=>{video.style.display='none';$('videoFallback').style.display='block'};render();</script></body></html>'''
    OUT.write_text(html.replace("__DATA__", json.dumps(payload, ensure_ascii=False)), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
