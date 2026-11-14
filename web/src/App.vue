<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'

// 手骨架判定口径：与生成视频/ROI 的脚本保持一致
const HAND_SCORE = 0.18
const HAND_MIN_POINTS = 4
const HAND_MERGE_GAP = 0.35
// 骨架已由 api_teacher/render_pose_video.py 烧进画面；手框由 extract_roi.py 写入姿态 JSON
const VIDEO_SRC = '/video/v2_5m_7m_pose.mp4'
const PALETTE = ['#7b8fd4', '#8fbf6a', '#57b8c9', '#d9a05b', '#b48ad4', '#6fae8f', '#c98b8b', '#a8b0c0']

const runs = ref({ A: [], B: [], C: [] })
const segments = ref({ steps: [], duration: 0, coverage: [0, 0], note: '' })
const selectedRun = ref('A')
const selectedIndex = ref(0)
const video = ref(null)
const overlay = ref(null)
const videoError = ref(false)
const poseRows = ref([])
const poseDuration = ref(0)
const poseRate = ref(5)
const handSegments = ref({ left: [], right: [] })
const currentSec = ref(0)
const loadError = ref('')

const items = computed(() => runs.value[selectedRun.value] || [])
const current = computed(() => items.value[selectedIndex.value] || {})
const detail = computed(() => { try { return JSON.parse(current.value.response || '{}') } catch { return {} } })
const duration = computed(() => poseDuration.value || segments.value.duration || Number(items.value.at(-1)?.end || 30))
const count = computed(() => items.value.slice(0, selectedIndex.value + 1).filter(item => /screw|螺丝|fasten|tighten|拧紧/i.test(`${item.response}`)).length)

const stepSegs = computed(() => segments.value.steps || [])
const activeStep = computed(() => stepSegs.value.find(s => currentSec.value >= s.start && currentSec.value < s.end) || null)
const activeStepName = computed(() => activeStep.value?.name || '')

const tlRows = computed(() => ([
  { key: 'step', label: '工序', cls: 'step', segs: stepSegs.value },
  { key: 'left', label: '左手', cls: 'left', segs: handSegments.value.left.map(([start, end]) => ({ start, end })) },
  { key: 'right', label: '右手', cls: 'right', segs: handSegments.value.right.map(([start, end]) => ({ start, end })) },
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
const coverageNote = computed(() => {
  const [a, b] = segments.value.coverage || [0, 0]
  return `${a.toFixed(0)}–${b.toFixed(0)} 秒有工序数据（${segments.value.note || '抽样窗口'}）`
})
// 当前播放位置是否有抽样出来的动作语义结果（卡片只在覆盖范围内有意义）
const inCoverage = computed(() => {
  const [a, b] = segments.value.coverage || [0, 0]
  return currentSec.value >= a && currentSec.value <= b
})

function pct(sec) {
  const total = duration.value || 1
  return Math.min(100, Math.max(0, (Number(sec) || 0) / total * 100))
}
function fmt(sec) {
  const tenths = Math.max(0, Math.floor((Number(sec) || 0) * 10))
  const m = Math.floor(tenths / 600)
  return `${m}:${((tenths - m * 600) / 10).toFixed(1).padStart(4, '0')}`
}
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
function poseRowAt(time) {
  const rows = poseRows.value
  if (!rows.length) return null
  let lo = 0; let hi = rows.length - 1
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (rows[mid].timestamp < time) lo = mid + 1
    else hi = mid
  }
  const next = rows[lo]; const prev = rows[lo - 1]
  if (prev && Math.abs(prev.timestamp - time) <= Math.abs(next.timestamp - time)) return prev
  return next
}

function setIndex(index) {
  selectedIndex.value = index
  if (video.value && !videoError.value) video.value.currentTime = Number(items.value[index]?.start || 0)
}
function seekTo(sec) {
  const target = Math.min(Math.max(0, sec), Math.max(0, duration.value - 0.05))
  currentSec.value = target
  if (video.value && !videoError.value) video.value.currentTime = target
  drawBoxes(target)
}
function seekFromTimeline(event) {
  const box = event.currentTarget.getBoundingClientRect()
  if (!box.width) return
  const ratio = Math.min(1, Math.max(0, (event.clientX - box.left) / box.width))
  seekTo(ratio * duration.value)
}
function seekToStep(seg) {
  seekTo(Number(seg.start) || 0)
}

// 在画面上画左右手框（骨架已经烧进视频，这里只补实时框）
function drawBoxes(time) {
  const canvas = overlay.value; const el = video.value
  if (!canvas || !el || !el.videoWidth) return
  const cssW = el.clientWidth, cssH = el.clientHeight
  if (!cssW || !cssH) return
  const dpr = window.devicePixelRatio || 1
  const w = Math.round(cssW * dpr), h = Math.round(cssH * dpr)
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h }
  const ctx = canvas.getContext('2d')
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, cssW, cssH)
  const row = poseRowAt(time)
  if (!row || Math.abs(row.timestamp - time) > 0.6) return
  const sx = cssW / el.videoWidth, sy = cssH / el.videoHeight
  const drawBox = (box, color, label) => {
    if (!box) return
    const x = box[0] * sx, y = box[1] * sy, bw = (box[2] - box[0]) * sx, bh = (box[3] - box[1]) * sy
    ctx.strokeStyle = color; ctx.lineWidth = 2
    ctx.strokeRect(x, y, bw, bh)
    ctx.font = 'bold 12px Arial'
    const tw = ctx.measureText(label).width
    ctx.fillStyle = color
    ctx.fillRect(x, Math.max(0, y - 15), tw + 8, 15)
    ctx.fillStyle = '#06131f'
    ctx.fillText(label, x + 4, Math.max(11, y - 4))
  }
  drawBox(row.left_box, '#55d6be', 'L')
  drawBox(row.right_box, '#f4a261', 'R')
}

function syncVideo() {
  const el = video.value
  if (!el) return
  currentSec.value = el.currentTime
  const idx = items.value.findIndex(item => el.currentTime >= item.start && el.currentTime < item.end)
  if (idx >= 0) selectedIndex.value = idx
  drawBoxes(el.currentTime)
}

let rafId = 0
let resizeObserver = null
let pendingSeek = null
function loop() {
  rafId = requestAnimationFrame(loop)
  const el = video.value
  if (!el || el.paused || el.ended) return
  currentSec.value = el.currentTime
  drawBoxes(el.currentTime)
}
function onLoadedMetadata() {
  const el = video.value
  if (!el) return
  if (pendingSeek != null) { el.currentTime = Math.min(pendingSeek, Math.max(0, el.duration - 0.05)); pendingSeek = null }
  currentSec.value = el.currentTime || 0
  drawBoxes(el.currentTime || 0)
}

onMounted(async () => {
  rafId = requestAnimationFrame(loop)
  if (video.value && 'ResizeObserver' in window) {
    resizeObserver = new ResizeObserver(() => drawBoxes(video.value?.currentTime || 0))
    resizeObserver.observe(video.value)
  }
  const seekParam = Number(new URLSearchParams(location.search).get('t'))
  if (Number.isFinite(seekParam) && seekParam > 0) pendingSeek = seekParam
  try {
    const [a, b, c, seg, pose] = await Promise.all([
      ...['A', 'B', 'C'].map(k => fetch(`/data/ablation_14_${k}.json`).then(r => r.json())),
      fetch('/data/segments.json').then(r => r.json()).catch(() => null),
      fetch('/data/local_pose_2min.json').then(r => r.json()),
    ])
    runs.value = { A: a.A || [], B: b.B || [], C: c.C || [] }
    if (seg) segments.value = seg
    const rows = pose.frames || []
    poseRows.value = rows
    poseDuration.value = Number(pose.duration || 0)
    poseRate.value = Number(pose.rate || 5)
    handSegments.value = { left: buildSegments(rows, 'left_hand'), right: buildSegments(rows, 'right_hand') }
  } catch (err) {
    loadError.value = `数据加载失败：${err?.message || err}`
  }
  if (!poseDuration.value && video.value?.duration) poseDuration.value = video.value.duration
  drawBoxes(video.value?.currentTime || 0)
})

onBeforeUnmount(() => {
  if (rafId) cancelAnimationFrame(rafId)
  rafId = 0
  if (resizeObserver) resizeObserver.disconnect()
})
</script>

<template>
  <main class="app">
    <header>
      <div><h1>动作分割结果 · SOP 装配分析</h1><p>周期工序 → 当前动作 / 左手动作 / 右手动作 / 操作器件与目标</p></div>
      <nav><button v-for="run in ['A','B','C']" :key="run" :class="{active: selectedRun === run}" @click="selectedRun = run; selectedIndex = 0">{{ run }} 运行</button></nav>
    </header>

    <section class="layout">
      <div class="left-panel">
        <div class="video-wrap">
          <video v-show="!videoError" ref="video" controls preload="metadata" poster="/data/frame_0000_00.jpg" :src="VIDEO_SRC" @timeupdate="syncVideo" @seeked="syncVideo" @loadedmetadata="onLoadedMetadata" @error="videoError = true" />
          <canvas ref="overlay" class="pose-overlay" v-show="!videoError" />
          <div v-if="videoError" class="fallback"><img :src="`/data/frame_${String(selectedIndex).padStart(4,'0')}_00.jpg`" /><p>视频解码失败，当前显示分析抽帧。</p></div>
        </div>
        <div class="counter"><strong>{{ count }}</strong><div><b>放钉关联紧固件计数</b><p>按视觉结果关联展示。当前数据仍是固定采样窗口。</p></div></div>
      </div>

      <aside class="cards">
        <p v-if="!inCoverage" class="card-warn">该时刻（{{ fmt(currentSec) }}）没有抽样的动作语义结果，卡片内容不适用；仅 {{ coverageNote }}</p>
        <article><h3>标准 SOP 描述</h3><strong>{{ detail.phase || '未知阶段' }} · {{ detail.object || '目标未识别' }}</strong><p>片段 {{ current.start?.toFixed(2) }} — {{ current.end?.toFixed(2) }} 秒</p></article>
        <article><h3>当前动作描述</h3><strong>{{ detail.action || '未知动作' }}</strong><p>{{ detail.change || '变化：未给出' }}</p></article>
        <article><h3>左手</h3><strong>{{ detail.left_hand || '未知' }}</strong><p>器件：{{ detail.tool || '未知' }} · 目标：{{ detail.object || '未知' }}</p></article>
        <article class="orange"><h3>右手</h3><strong>{{ detail.right_hand || '未知' }}</strong><p>关系：{{ detail.relation || '未知' }}</p></article>
      </aside>
    </section>

    <section class="timeline">
      <h2>时间轴
        <span class="timeline-note">三道轨道共用同一坐标轴 0 – {{ fmt(duration) }}，白色播放头联动；{{ coverageNote }}</span>
      </h2>
      <div class="tl">
        <div class="tl-rows">
          <div class="tl-row" v-for="row in tlRows" :key="row.key">
            <span class="tl-label">{{ row.label }}</span>
            <div class="tl-rail">
              <i v-for="(seg, i) in row.segs" :key="i"
                 class="tl-seg" :class="[row.cls, { on: row.key === 'step' && activeStep && seg.idx === activeStep.idx }]"
                 :style="{ left: pct(seg.start) + '%', width: Math.max(0.3, pct(seg.end - seg.start)) + '%',
                           background: row.key === 'step' ? PALETTE[(seg.color ?? i) % PALETTE.length] : '' }"
                 :title="`${seg.name || row.label}  ${Number(seg.start).toFixed(1)}–${Number(seg.end).toFixed(1)}s`"
                 @click.stop="row.key === 'step' ? seekToStep(seg) : seekTo(seg.start)" />
            </div>
          </div>
        </div>
        <div class="tl-play" @click="seekFromTimeline" title="点击跳转到对应时间"><i :style="{ left: playheadPct + '%' }" /></div>
      </div>
      <p class="tl-legend">
        <span><i class="dot cy"></i>左手识别片段 {{ coverageText.left }}</span>
        <span><i class="dot or"></i>右手识别片段 {{ coverageText.right }}</span>
        <span><i class="dot st"></i>工序段 {{ stepSegs.length }} 段</span>
        <span>当前工序：<b>{{ activeStepName || '该时段无工序数据' }}</b></span>
      </p>
      <p v-if="loadError" class="hb-warn">{{ loadError }}</p>
    </section>
  </main>
</template>
