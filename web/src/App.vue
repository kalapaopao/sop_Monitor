<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'

// 手骨架判定口径：与渲染视频使用的阈值保持一致
const HAND_SCORE = 0.18
const HAND_MIN_POINTS = 4
// 相邻识别片段间隔小于该值就合并，避免单帧漏检把进度条切碎
const HAND_MERGE_GAP = 0.35
// 骨架已经由 api_teacher/render_pose_video.py 烧进画面，网页只负责播放
const VIDEO_SRC = '/video/v2_5m_7m_pose.mp4'

const runs = ref({ A: [], B: [], C: [] })
const selectedRun = ref('A')
const selectedIndex = ref(0)
const video = ref(null)
const videoError = ref(false)
const poseDuration = ref(0)
const poseRate = ref(5)
const handSegments = ref({ left: [], right: [] })
const currentSec = ref(0)
const loadError = ref('')

const items = computed(() => runs.value[selectedRun.value] || [])
const current = computed(() => items.value[selectedIndex.value] || {})
const detail = computed(() => { try { return JSON.parse(current.value.response || '{}') } catch { return {} } })
const currentTime = computed(() => Number(current.value.start || 0))
const duration = computed(() => poseDuration.value || Number(items.value.at(-1)?.end || 30))
const count = computed(() => items.value.slice(0, selectedIndex.value + 1).filter(item => /screw|螺丝|fasten|tighten|拧紧/i.test(`${item.response}`)).length)

const trackRows = computed(() => ([
  { key: 'left', label: '左手', cls: 'left', segs: handSegments.value.left },
  { key: 'right', label: '右手', cls: 'right', segs: handSegments.value.right },
]))
const playheadPct = computed(() => pct(currentSec.value))
const coverageText = computed(() => {
  const total = duration.value || 1
  const summarize = segs => {
    const sec = segs.reduce((sum, seg) => sum + (seg[1] - seg[0]), 0)
    return `${sec.toFixed(0)}s / ${Math.round((sec / total) * 100)}%`
  }
  return { left: summarize(handSegments.value.left), right: summarize(handSegments.value.right) }
})

function pct(sec) {
  const total = duration.value || 1
  return Math.min(100, Math.max(0, (Number(sec) || 0) / total * 100))
}
function fmt(sec) {
  // 先取到“十分之一秒”整数再换算，避免 119.99 秒被四舍五入成 1:60.0
  const tenths = Math.max(0, Math.floor((Number(sec) || 0) * 10))
  const m = Math.floor(tenths / 600)
  return `${m}:${((tenths - m * 600) / 10).toFixed(1).padStart(4, '0')}`
}

// 把 5Hz 的关键点帧序列压成 [start, end] 识别片段
function buildSegments(rows, key) {
  const segs = []
  const half = 1 / (2 * (poseRate.value || 5))
  let cur = null
  for (const row of rows) {
    const pts = row[key] || []
    const valid = pts.filter(p => p[2] > HAND_SCORE).length >= HAND_MIN_POINTS
    if (!valid) {
      if (cur) { segs.push(cur); cur = null }
      continue
    }
    const t0 = Math.max(0, row.timestamp - half)
    const t1 = row.timestamp + half
    if (!cur) cur = [t0, t1]
    else if (t0 - cur[1] <= HAND_MERGE_GAP) cur[1] = t1
    else { segs.push(cur); cur = [t0, t1] }
  }
  if (cur) segs.push(cur)
  return segs
}

function setIndex(index) {
  selectedIndex.value = index
  if (video.value && !videoError.value) video.value.currentTime = currentTime.value
}
function syncVideo() {
  const el = video.value
  if (!el) return
  const time = el.currentTime
  currentSec.value = time
  const index = items.value.findIndex(item => time >= item.start && time < item.end)
  if (index >= 0) selectedIndex.value = index
}
// 点击识别片段进度条跳转
function seekFromBar(event) {
  if (videoError.value) return
  const box = event.currentTarget.getBoundingClientRect()
  if (!box.width) return
  const ratio = Math.min(1, Math.max(0, (event.clientX - box.left) / box.width))
  const target = Math.min(ratio * duration.value, Math.max(0, duration.value - 0.05))
  currentSec.value = target
  if (video.value) video.value.currentTime = target
}

let rafId = 0
let pendingSeek = null
function loop() {
  rafId = requestAnimationFrame(loop)
  const el = video.value
  if (!el || el.paused || el.ended) return
  currentSec.value = el.currentTime
}
function onLoadedMetadata() {
  const el = video.value
  if (!el) return
  if (pendingSeek != null) {
    el.currentTime = Math.min(pendingSeek, Math.max(0, el.duration - 0.05))
    pendingSeek = null
  }
  currentSec.value = el.currentTime || 0
}

onMounted(async () => {
  rafId = requestAnimationFrame(loop)
  const seekParam = Number(new URLSearchParams(location.search).get('t'))
  if (Number.isFinite(seekParam) && seekParam > 0) pendingSeek = seekParam
  try {
    const [a, b, c] = await Promise.all(['A', 'B', 'C'].map(key => fetch(`/data/ablation_14_${key}.json`).then(r => r.json())))
    runs.value = { A: a.A || [], B: b.B || [], C: c.C || [] }
    const pose = await fetch('/data/local_pose_2min.json').then(r => r.json())
    const rows = pose.frames || []
    poseDuration.value = Number(pose.duration || 0)
    poseRate.value = Number(pose.rate || 5)
    handSegments.value = { left: buildSegments(rows, 'left_hand'), right: buildSegments(rows, 'right_hand') }
  } catch (err) {
    loadError.value = `分析数据加载失败：${err?.message || err}`
  }
  if (!poseDuration.value && video.value?.duration) poseDuration.value = video.value.duration
})

onBeforeUnmount(() => {
  if (rafId) cancelAnimationFrame(rafId)
  rafId = 0
})
</script>

<template>
  <main class="app">
    <header><div><h1>动作分割结果 · SOP 装配分析</h1><p>周期工序 → 当前动作 / 左手动作 / 右手动作 / 操作器件与目标</p></div><nav><button v-for="run in ['A','B','C']" :key="run" :class="{active: selectedRun === run}" @click="selectedRun=run;selectedIndex=0">{{ run }} 运行</button></nav></header>
    <section class="layout">
      <div class="left-panel">
        <div class="video-wrap">
          <video v-show="!videoError" ref="video" controls preload="metadata" poster="/data/frame_0000_00.jpg" :src="VIDEO_SRC" @timeupdate="syncVideo" @seeked="syncVideo" @loadedmetadata="onLoadedMetadata" @error="videoError=true" />
          <div v-if="videoError" class="fallback"><img :src="`/data/frame_${String(selectedIndex).padStart(4,'0')}_00.jpg`" /><p>视频解码失败，当前显示本次分析抽帧。</p></div>
        </div>
        <div class="counter"><strong>{{ count }}</strong><div><b>放钉关联紧固件计数</b><p>按视觉结果关联展示。当前数据仍是固定采样窗口。</p></div></div>
      </div>
      <aside class="cards">
        <article><h3>标准 SOP 描述</h3><strong>{{ detail.phase || '未知阶段' }} · {{ detail.object || '目标未识别' }}</strong><p>片段 {{ current.start?.toFixed(2) }} — {{ current.end?.toFixed(2) }} 秒</p></article>
        <article><h3>当前动作描述</h3><strong>{{ detail.action || '未知动作' }}</strong><p>{{ detail.change || '变化：未给出' }}</p></article>
        <article><h3>左手</h3><strong>{{ detail.left_hand || '未知' }}</strong><p>器件：{{ detail.tool || '未知' }} · 目标：{{ detail.object || '未知' }}</p></article>
        <article class="orange"><h3>右手</h3><strong>{{ detail.right_hand || '未知' }}</strong><p>关系：{{ detail.relation || '未知' }}</p></article>
      </aside>
    </section>
    <section class="timeline">
      <h2>SOP 装配采样时间轴 <span class="timeline-note">以下「工序」为 0–30 秒抽样窗口的等宽分段</span></h2>
      <div class="track"><label>工序</label><div class="segments"><button v-for="(item,index) in items" :key="index" :class="['segment', 'phase', {selected:index===selectedIndex}]" :style="{flex: Math.max(.1, item.end-item.start)}" :title="`${item.start}–${item.end}s ${detail.phase || ''}`" @click="setIndex(index)" /></div></div>
      <div class="hand-bar">
        <div class="hb-head-row"><b>左右手识别片段</b><span>{{ fmt(currentSec) }} / {{ fmt(duration) }} · 左手 {{ coverageText.left }} · 右手 {{ coverageText.right }}</span></div>
        <div class="hb-tracks">
          <div class="hb-row" v-for="row in trackRows" :key="row.key">
            <span class="hb-label">{{ row.label }}</span>
            <div class="hb-rail">
              <i v-for="(seg, i) in row.segs" :key="i" class="hb-seg" :class="row.cls" :style="{ left: pct(seg[0]) + '%', width: Math.max(0.4, pct(seg[1] - seg[0])) + '%' }" />
            </div>
          </div>
          <div class="hb-play" @click="seekFromBar" title="点击跳转到对应时间"><i :style="{ left: playheadPct + '%' }" /></div>
        </div>
        <p class="hb-legend">青色 = 左手识别到关键点，橙色 = 右手识别到关键点，深灰 = 该时段未识别；点击进度条可跳转</p>
        <p v-if="loadError" class="hb-warn">{{ loadError }}</p>
      </div>
    </section>
  </main>
</template>
