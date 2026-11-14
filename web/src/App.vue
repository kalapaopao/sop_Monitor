<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from 'vue'

// 手骨架判定口径：与生成视频/ROI 的脚本保持一致
const HAND_SCORE = 0.18
const HAND_MIN_POINTS = 4
const HAND_MERGE_GAP = 0.6   // 10 Hz 采样下短时遮挡更明显，合并间隔调大避免碎片过多
// 骨架已由 api_teacher/render_pose_video.py 烧进画面；手框由 extract_roi.py 写入姿态 JSON
const VIDEO_SRC = '/video/v2_5m_7m_pose.mp4'
const PALETTE = ['#7b8fd4', '#8fbf6a', '#57b8c9', '#d9a05b', '#b48ad4', '#6fae8f', '#c98b8b', '#a8b0c0']

const runs = ref({ A: [], B: [], C: [] })
const segments = ref({ steps: [], duration: 0, coverage: [0, 0], note: '' })
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

// ===== 固定 ROI 编辑器 =====
// 这些区域是「工位配置」：坐标归一化保存，不裁剪画面，只作为额外输入通道，
// 让模型在保留全局画面的同时重点关注这些区域。
const ROI_PALETTE = ['#7bd6be', '#f4a261', '#8fb8ff', '#d59bf4', '#c9d66b', '#7fd1e0']
const editing = ref(false)
const regions = ref([])
const roiCanvas = ref(null)
const saving = ref(false)
const saveMsg = ref('')
const saveBad = ref(false)
const stationMeta = ref({ id: 'ST01', name: '默认工位', frame: [1920, 1080] })
let dragging = false
let draft = null

function normFromEvent(event) {
  const c = roiCanvas.value
  if (!c) return null
  const box = c.getBoundingClientRect()
  if (!box.width || !box.height) return null
  const x = Math.min(1, Math.max(0, (event.clientX - box.left) / box.width))
  const y = Math.min(1, Math.max(0, (event.clientY - box.top) / box.height))
  return { x, y }
}
function fitCanvas() {
  const c = roiCanvas.value; const el = video.value
  if (!c || !el) return null
  const w = el.clientWidth, h = el.clientHeight
  if (!w || !h) return null
  const dpr = window.devicePixelRatio || 1
  const bw = Math.round(w * dpr), bh = Math.round(h * dpr)
  if (c.width !== bw || c.height !== bh) { c.width = bw; c.height = bh }
  const ctx = c.getContext('2d')
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  return { w, h, ctx }
}
function drawRois() {
  const fit = fitCanvas()
  if (!fit) return
  const { w, h, ctx } = fit
  ctx.clearRect(0, 0, w, h)
  const stroke = (r, color, dashed, label) => {
    const x = r[0] * w, y = r[1] * h, rw = (r[2] - r[0]) * w, rh = (r[3] - r[1]) * h
    ctx.save()
    ctx.setLineDash(dashed ? [8, 6] : [])
    ctx.strokeStyle = color
    ctx.lineWidth = 2
    ctx.strokeRect(x, y, rw, rh)
    ctx.setLineDash([])
    ctx.fillStyle = 'rgba(7,17,30,.72)'
    ctx.font = 'bold 12px Arial'
    const tw = ctx.measureText(label).width
    ctx.fillRect(x, Math.max(0, y - 16), tw + 10, 16)
    ctx.fillStyle = color
    ctx.fillText(label, x + 5, Math.max(11, y - 4))
    ctx.restore()
  }
  for (const r of regions.value) {
    if (!r.enabled) continue
    stroke(r.rect_norm, r.color, r.type === 'state', r.name || '区域')
  }
  if (draft) stroke(draft, '#ffffff', true, '新区域')
}
function onDown(event) {
  if (!editing.value) return
  const p = normFromEvent(event)
  if (!p) return
  dragging = true
  draft = [p.x, p.y, p.x, p.y]
  roiCanvas.value?.setPointerCapture?.(event.pointerId)
  drawRois()
}
function onMove(event) {
  if (!dragging || !draft) return
  const p = normFromEvent(event)
  if (!p) return
  draft = [Math.min(draft[0], p.x), Math.min(draft[1], p.y), Math.max(draft[0], p.x), Math.max(draft[1], p.y)]
  drawRois()
}
function onUp(event) {
  if (!dragging) return
  dragging = false
  const p = normFromEvent(event)
  if (draft && p) {
    draft = [Math.min(draft[0], p.x), Math.min(draft[1], p.y), Math.max(draft[0], p.x), Math.max(draft[1], p.y)]
  }
  const d = draft
  draft = null
  if (d && (d[2] - d[0]) > 0.02 && (d[3] - d[1]) > 0.02) {
    const n = regions.value.length
    regions.value.push({
      id: 'r' + Date.now().toString(36),
      name: '区域 ' + (n + 1),
      type: 'focus',
      rect_norm: d.map(v => Math.round(v * 10000) / 10000),
      color: ROI_PALETTE[n % ROI_PALETTE.length],
      enabled: true,
    })
    saveMsg.value = '已添加 ' + regions.value.length + ' 个区域（还没保存）'
    saveBad.value = false
  }
  drawRois()
}
function removeRegion(id) {
  regions.value = regions.value.filter(r => r.id !== id)
  saveMsg.value = '已删除（还没保存）'
  saveBad.value = false
  drawRois()
}
function clearRegions() {
  regions.value = []
  saveMsg.value = '已清空（还没保存）'
  saveBad.value = false
  drawRois()
}
async function loadStations() {
  try {
    const data = await fetch('/api/stations').then(r => r.json())
    const st = (data.stations || [])[0]
    if (st) {
      stationMeta.value = { id: st.id || 'ST01', name: st.name || '默认工位', frame: st.frame || [1920, 1080] }
      regions.value = (st.regions || []).map(r => ({ ...r }))
    }
  } catch (err) {
    saveMsg.value = '读取配置失败：' + err.message
    saveBad.value = true
  }
}
async function saveStations() {
  saving.value = true
  saveMsg.value = ''
  saveBad.value = false
  try {
    const payload = {
      version: 1,
      stations: [{ ...stationMeta.value, regions: regions.value }],
    }
    const res = await fetch('/api/stations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    const out = await res.json()
    if (!res.ok) throw new Error(out.error || res.statusText)
    saveMsg.value = '已保存 ' + out.stations + ' 个工位到 ' + out.file
  } catch (err) {
    saveMsg.value = '保存失败：' + err.message
    saveBad.value = true
  } finally {
    saving.value = false
  }
}
// 开发期调试钩子：只在本机 dev build 里挂到 window，方便自动化测试画框流程
if (import.meta.env.DEV) {
  window.__roi = { regions, editing, getDraft: () => draft, getDragging: () => dragging }
}

function toggleEdit() {
  editing.value = !editing.value
  if (editing.value) {
    video.value?.pause()
    nextTick(drawRois)
  } else {
    drawRois()
  }
}

function pct(sec) {
  const total = duration.value || 1
  return Math.min(100, Math.max(0, (Number(sec) || 0) / total * 100))
}
function fmt(sec) {
  const tenths = Math.max(0, Math.floor((Number(sec) || 0) * 10))
  const m = Math.floor(tenths / 600)
  return `${m}:${((tenths - m * 600) / 10).toFixed(1).padStart(4, '0')}`
}
// 把 10Hz 的关键点帧序列压成 [start, end] 识别片段。
// 注意：遇到无效帧时**不能立刻收尾**，否则下面的 MERGE_GAP 永远没有机会合并，
// 短时遮挡会把时间轴切成一堆碎条（此前就是这个 bug：gap 调多大都没用）。
function buildSegments(rows, key) {
  const segs = []
  const half = 1 / (2 * (poseRate.value || 10))
  let cur = null
  for (const row of rows) {
    const pts = row[key] || []
    const valid = pts.filter(p => p[2] > HAND_SCORE).length >= HAND_MIN_POINTS
    if (!valid) continue                       // 先挂着，交给下一个有效帧决定是否断开
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
  if (video.value && !videoError.value) video.value.currentTime = Number(items.value[index]?.start || 0)
}
function seekTo(sec) {
  const target = Math.min(Math.max(0, sec), Math.max(0, duration.value - 0.05))
  currentSec.value = target
  if (video.value && !videoError.value) video.value.currentTime = target
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


function syncVideo() {
  const el = video.value
  if (!el) return
  currentSec.value = el.currentTime
  const idx = items.value.findIndex(item => el.currentTime >= item.start && el.currentTime < item.end)
  if (idx >= 0) selectedIndex.value = idx
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
  if (pendingSeek != null) { el.currentTime = Math.min(pendingSeek, Math.max(0, el.duration - 0.05)); pendingSeek = null }
  currentSec.value = el.currentTime || 0
}

onMounted(async () => {
  rafId = requestAnimationFrame(loop)
  window.addEventListener('resize', drawRois)
  loadStations()
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
    poseDuration.value = Number(pose.duration || 0)
    poseRate.value = Number(pose.rate || 5)
    handSegments.value = { left: buildSegments(rows, 'left_hand'), right: buildSegments(rows, 'right_hand') }
  } catch (err) {
    loadError.value = `数据加载失败：${err?.message || err}`
  }
  if (!poseDuration.value && video.value?.duration) poseDuration.value = video.value.duration
})

onBeforeUnmount(() => {
  if (rafId) cancelAnimationFrame(rafId)
  rafId = 0
  window.removeEventListener('resize', drawRois)
})
</script>

<template>
  <main class="app">
    <header>
      <div><h1>动作分割结果 · SOP 装配分析</h1><p>周期工序 → 当前动作 / 左手动作 / 右手动作 / 操作器件与目标</p></div>
      <div class="head-right">
        <nav><button v-for="run in ['A','B','C']" :key="run" :class="{active: selectedRun === run}" @click="selectedRun = run; selectedIndex = 0">{{ run }} 运行</button></nav>
        <button class="roi-toggle" :class="{ active: editing }" @click="toggleEdit">{{ editing ? '退出 ROI 编辑' : 'ROI 编辑' }}</button>
      </div>
    </header>

    <section class="layout">
      <div class="left-panel">
        <div class="video-wrap" :class="{ editing }">
          <video v-show="!videoError" ref="video" controls preload="metadata" poster="/data/frame_0000_00.jpg" :src="VIDEO_SRC" @timeupdate="syncVideo" @seeked="syncVideo" @loadedmetadata="onLoadedMetadata" @error="videoError = true" />
          <canvas
            ref="roiCanvas"
            class="roi-layer"
            v-show="editing"
            @pointerdown="onDown"
            @pointermove="onMove"
            @pointerup="onUp"
            @pointerleave="onUp"
          />
          <div v-if="videoError" class="fallback"><img :src="`/data/frame_${String(selectedIndex).padStart(4,'0')}_00.jpg`" /><p>视频解码失败，当前显示分析抽帧。</p></div>
        </div>
        <div v-if="editing" class="roi-panel">
          <p class="roi-hint">在画面上按住左键拖拽即可画框。这些区域是<strong>工位配置</strong>：不裁剪画面，而是作为额外输入让模型重点关注。</p>
          <div class="roi-list">
            <div v-for="r in regions" :key="r.id" class="roi-item" :class="{ off: !r.enabled }">
              <i class="sw" :style="{ background: r.color }"></i>
              <input v-model="r.name" class="roi-name" placeholder="区域名称" />
              <select v-model="r.type" class="roi-type">
                <option value="focus">注意力引导</option>
                <option value="state">状态判定</option>
              </select>
              <label class="roi-en"><input type="checkbox" v-model="r.enabled" /> 启用</label>
              <button class="roi-del" @click="removeRegion(r.id)">删除</button>
            </div>
            <p v-if="!regions.length" class="roi-empty">还没有区域，拖拽画面开始画第一个。</p>
          </div>
          <div class="roi-actions">
            <button @click="saveStations" :disabled="saving">{{ saving ? '保存中…' : '保存到 configs/stations.json' }}</button>
            <button @click="clearRegions">清空</button>
            <span class="roi-msg" :class="{ bad: saveBad }">{{ saveMsg }}</span>
          </div>
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
