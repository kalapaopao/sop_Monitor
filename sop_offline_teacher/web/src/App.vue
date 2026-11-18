<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from 'vue'

// 手骨架判定口径：与生成视频/ROI 的脚本保持一致
const HAND_SCORE = 0.18
const HAND_MIN_POINTS = 4
const HAND_MERGE_GAP = 0.6   // 10 Hz 采样下短时遮挡更明显，合并间隔调大避免碎片过多
// 骨架已由 api_teacher/render_pose_video.py 烧进画面；手框由 extract_roi.py 写入姿态 JSON
// 视频与姿态都可由「结果来源」里的 run 覆盖（run JSON 里带 video / pose 字段即可），
// 不带时用下面的默认值 —— 保持原有行为不变。
const DEFAULT_VIDEO = '/video/v2_5m_7m_pose.mp4'
const DEFAULT_POSE = '/data/local_pose_2min.json'
const VIDEO_SRC = ref(DEFAULT_VIDEO)
const poseSrc = ref(DEFAULT_POSE)
// 封面只在默认视频上使用。换视频后必须清掉，否则播放器仍显示旧视频的第一帧，
// 看起来像是"没换成新视频"（Vue 改 :src 不会自动丢弃 poster）。
const DEFAULT_POSTER = '/data/frame_0000_00.jpg'
const VIDEO_POSTER = ref(DEFAULT_POSTER)
const PALETTE = ['#7b8fd4', '#8fbf6a', '#57b8c9', '#d9a05b', '#b48ad4', '#6fae8f', '#c98b8b', '#a8b0c0']

// 「结果来源」：每次教师跑完用 scripts/publish_segments.py 发布，
// 网页读 runs/index.json 自动出现下拉项，不用改页面代码。
const runs = ref([])
const activeRun = ref('')
const runNote = ref('')

// ===== 工序段编辑（在网页上微调模型预填的段）=====
const stepOptions = ref([])
const editMode = ref(false)
const editIdx = ref(-1)
const editMsg = ref('')

async function loadStepOptions() {
  if (Array.isArray(segments.value.step_options)) {
    stepOptions.value = segments.value.step_options
    return
  }
  try {
    const d = await fetch('/api/steps', { cache: 'no-store' }).then(r => r.json())
    stepOptions.value = d.steps || []
  } catch {
    stepOptions.value = []
  }
}
function toggleSegEdit() {
  editMode.value = !editMode.value
  editIdx.value = -1
  editMsg.value = editMode.value ? '点时间轴上的工序段选中它' : ''
  if (editMode.value && !stepOptions.value.length) loadStepOptions()
}
const editSeg = computed(() => (editIdx.value >= 0 ? (segments.value.steps || [])[editIdx.value] : null))
function pickSeg(seg) {
  if (!editMode.value) { seekToStep(seg); return }
  editIdx.value = seg.idx ?? 0
  editMsg.value = ''
}
function refreshSegMeta() {
  const st = segments.value.steps || []
  st.forEach((s, k) => { s.idx = k })
  segments.value.coverage = st.length ? [st[0].start, st[st.length - 1].end] : [0, 0]
}
function setSegStepId(id) {
  const s = editSeg.value
  if (!s) return
  s.step_id = id
  const o = stepOptions.value.find(x => x.step_id === id)
  if (o) s.name = o.name_cn
  s.need_review = false
  editMsg.value = `已改为 ${id}`
}
function nudge(key, d) {
  const s = editSeg.value
  if (!s) return
  const v = Math.max(0, Math.min(duration.value, Number(s[key]) + d))
  s[key] = Math.round(v * 100) / 100
  if (s.end <= s.start) s.end = Math.round((s.start + 0.1) * 100) / 100
  refreshSegMeta()
  editMsg.value = `已调整 ${key === 'start' ? '起点' : '终点'}`
}
function delSeg() {
  const i = editIdx.value
  if (i < 0) return
  segments.value.steps.splice(i, 1)
  refreshSegMeta()
  editIdx.value = Math.min(i, (segments.value.steps || []).length - 1)
  editMsg.value = '已删除该段'
}
async function saveEdited() {
  try {
    const r = await fetch('/api/runs', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, cache: 'no-store',
      body: JSON.stringify({ name: activeRun.value, doc: segments.value, steps: segments.value.steps }),
    })
    const o = await r.json()
    if (!r.ok) throw new Error(o.error || r.statusText)
    editMsg.value = `已保存 ${o.steps} 段 → ${o.file}（原有文件备份为 .bak.json）`
    await reloadRuns(activeRun.value)
  } catch (e) {
    editMsg.value = `保存失败：${e?.message || e}`
  }
}

// ===== 鼠标拖拽编辑：拖两端改边界，拖中间整体平移 =====
const drag = ref(null)

function timeFromEvent(e, railEl) {
  const r = railEl.getBoundingClientRect()
  const p = Math.max(0, Math.min(1, (e.clientX - r.left) / Math.max(1, r.width)))
  return Math.round(p * duration.value * 100) / 100
}

// ===== 轨道层（剪映式叠层）=====
function segLayer(sg) { return Number(sg.layer ?? 0) }
function overlaps(a, b) {
  return Number(a.start) < Number(b.end) - 1e-6 && Number(b.start) < Number(a.end) - 1e-6
}
function sameLayerOthers(seg) {
  const l = segLayer(seg)
  return (segments.value.steps || []).filter(x => x !== seg && segLayer(x) === l)
}
// 拖两端时：把边界夹在同层邻段之间，绝不产生覆盖
function clampStart(seg, t) {
  const prev = sameLayerOthers(seg).filter(o => Number(o.end) <= Number(seg.end) + 1e-6)
    .sort((a, b) => Number(a.end) - Number(b.end)).pop()
  const lo = prev ? Number(prev.end) : 0
  return Math.max(lo, Math.min(t, Number(seg.end) - 0.2))
}
function clampEnd(seg, t) {
  const next = sameLayerOthers(seg).filter(o => Number(o.start) >= Number(seg.start) - 1e-6)
    .sort((a, b) => Number(a.start) - Number(b.start))[0]
  const hi = next ? Number(next.start) : duration.value
  return Math.min(hi, Math.max(t, Number(seg.start) + 0.2))
}
// 松手后定层：当前层若与别人重叠，就往上抬一层，直到不重叠
function settleLayer(seg) {
  for (let guard = 0; guard < 8; guard++) {
    const clash = (segments.value.steps || []).filter(o => o !== seg && segLayer(o) === segLayer(seg) && overlaps(o, seg))
    if (!clash.length) return segLayer(seg)
    seg.layer = Math.max(...clash.map(segLayer)) + 1
  }
  return segLayer(seg)
}
// 拖动时让视频与白色播放头跟随鼠标（节流，避免 seek 风暴）
let lastScrubAt = 0
function scrubTo(t) {
  const now = performance.now()
  if (now - lastScrubAt < 70) return
  lastScrubAt = now
  seekTo(t)
}

function onSegPointerDown(e, seg) {
  if (!editMode.value) return
  editIdx.value = seg.idx ?? 0
  const railEl = e.currentTarget.closest('.tl-rail')
  if (!railEl) return
  const cls = e.target.classList
  const mode = cls.contains('h-l') ? 'start' : cls.contains('h-r') ? 'end' : 'move'
  drag.value = {
    seg, mode, railEl, moved: false,
    startX: e.clientX, startT: timeFromEvent(e, railEl),
    origStart: Number(seg.start), origEnd: Number(seg.end),
  }
  window.addEventListener('pointermove', onSegPointerMove)
  window.addEventListener('pointerup', onSegPointerUp, { once: true })
  e.preventDefault()
  e.stopPropagation()
}

function onSegPointerMove(e) {
  const d = drag.value
  if (!d) return
  if (Math.abs(e.clientX - d.startX) > 3) d.moved = true
  let t = timeFromEvent(e, d.railEl)
  const minDur = 0.2
  if (d.mode === "start") {
    t = snapTime(t, d.seg.idx, d.railEl)
    d.seg.start = clampStart(d.seg, Math.max(0, Math.min(t, d.origEnd - minDur)))
    scrubTo(d.seg.start)
  } else if (d.mode === "end") {
    t = snapTime(t, d.seg.idx, d.railEl)
    d.seg.end = clampEnd(d.seg, Math.min(duration.value, Math.max(t, d.origStart + minDur)))
    scrubTo(d.seg.end)
  } else {
    const shift = t - d.startT
    let s0 = d.origStart + shift
    let e0 = d.origEnd + shift
    // 磁吸：让起点或终点贴到目标
    const candS = snapTime(s0, d.seg.idx, d.railEl)
    if (snapLine.value !== null) { e0 += candS - s0; s0 = candS }
    else {
      const candE = snapTime(e0, d.seg.idx, d.railEl)
      if (snapLine.value !== null) { s0 += candE - e0; e0 = candE }
    }
    if (s0 < 0) { e0 -= s0; s0 = 0 }
    if (e0 > duration.value) { s0 -= (e0 - duration.value); e0 = duration.value }
    d.seg.start = Math.round(Math.max(0, s0) * 100) / 100
    d.seg.end = Math.round(Math.min(duration.value, e0) * 100) / 100
    scrubTo(d.seg.start)
  }
  d.seg.start = Math.round(d.seg.start * 100) / 100
  d.seg.end = Math.round(d.seg.end * 100) / 100
  refreshSegMeta()
}

function onSegPointerUp() {
  window.removeEventListener('pointermove', onSegPointerMove)
  const d = drag.value
  drag.value = null
  if (d && d.moved) {
    const layer = d.mode === "move" ? settleLayer(d.seg) : segLayer(d.seg)
    editMsg.value = `已拖动：${d.seg.start.toFixed(2)}–${d.seg.end.toFixed(2)}s`
      + (layer > 0 ? `（有重叠，已叠到第 ${layer + 1} 层）` : "") + "（记得保存）"
  }
}

function addSegAtPlayhead() {
  const st = segments.value.steps || []
  const t0 = Math.round(currentSec.value * 100) / 100
  const t1 = Math.round(Math.min(duration.value, t0 + 3) * 100) / 100
  st.push({
    idx: st.length, step_id: 'unknown', name: '流程外或插入式局部处理',
    start: t0, end: Math.max(t0 + 0.2, t1), kind: 'step',
    source: 'manual', color: 6, need_review: true, layer: 0,
    action: '（人工新增，待填写）', evidence: '', state_change: 'none',
    left_hand: '', right_hand: '',
  })
  st.sort((a, b) => a.start - b.start)
  refreshSegMeta()
  editIdx.value = st.findIndex(s => s.start === t0)
  editMsg.value = '已在播放头处新增一段，可拖拽调整'
}

// 当前时间是否落在这次结果的覆盖范围之外（用于提示 + 提供跳转按钮）
const outOfCoverage = computed(() => {
  const [c0, c1] = segments.value.coverage || [0, 0]
  if (!(c1 > c0)) return false
  return currentSec.value < c0 || currentSec.value > c1
})

async function reloadRuns(prefer) {
  try {
    const idx = await fetch(`/data/runs/index.json?ts=${Date.now()}`, { cache: 'no-store' }).then(r => r.json())
    runs.value = idx.runs || []
    if (!runs.value.length) {
      activeRun.value = ''
      runNote.value = '暂无发布的运行结果（先跑 scripts/publish_segments.py）'
      return
    }
    const want = prefer || idx.active || runs.value[runs.value.length - 1].name
    const hit = runs.value.find(r => r.name === want) || runs.value[runs.value.length - 1]
    activeRun.value = hit.name
    // 必须真的去加载这一次运行：除了工序段，run 还可能自带视频与姿态数据。
    // 以前这里只设 activeRun，导致首屏视频/姿态停在默认值，
    // 出现"卡片是新的、视频还是旧那一个"的现象。
    await switchRun()
    return
  } catch (err) {
    runs.value = []
    runNote.value = `未找到 runs/index.json（${err?.message || err}）`
  }
}

async function switchRun() {
  const r = runs.value.find(x => x.name === activeRun.value)
  if (!r) return
  try {
    const data = await fetch(`/data/runs/${r.file}`, { cache: 'no-store' }).then(x => x.json())
    segments.value = data
    stepOptions.value = Array.isArray(data.step_options) ? data.step_options : []
    if (editMode.value && !stepOptions.value.length) await loadStepOptions()
    if (activeStep.value) activeStep.value = null
    // run 可以自带视频与姿态数据（例如换一套手部识别的产物），不带就沿用默认值
    const useVideo = data.video || 'v2_5m_7m_pose.mp4'
    videoError.value = false
    VIDEO_SRC.value = `/video/${useVideo}`
    VIDEO_POSTER.value = data.video ? '' : DEFAULT_POSTER
    await switchPose(data.pose ? `/data/${data.pose}` : DEFAULT_POSE)
    // ROI 是按机位分开存的：run 声明了 station 就跟着切过去；
    // 没声明则沿用当前机位（不能拿 A 机位的框去量 B 机位的画面）。
    await loadStations(data.station || undefined)
    runNote.value = `${r.steps} 段${r.need_review ? `，${r.need_review} 段待复核` : ''}` +
      `｜覆盖 ${r.coverage[0].toFixed(1)}–${r.coverage[1].toFixed(1)}s｜发布 ${r.published_at}` +
      `｜视频 ${useVideo}`
    // 当前时间不在这次结果的覆盖范围内时，自动跳到覆盖起点，
    // 否则用户会停在无数据区看到旧 ablation 内容，误以为"没生效"。
    const [c0, c1] = data.coverage || [0, 0]
    if (c1 > c0 && (currentSec.value < c0 || currentSec.value > c1)) seekTo(c0 + 0.1)
  } catch (err) {
    runNote.value = `读取 ${r.file} 失败：${err?.message || err}`
  }
}

const segments = ref({ steps: [], duration: 0, coverage: [0, 0], note: '' })
const selectedIndex = ref(0)
const video = ref(null)
const videoError = ref(false)
const poseDuration = ref(0)
const poseRate = ref(5)
const handSegments = ref({ left: [], right: [] })
const currentSec = ref(0)
const loadError = ref('')
const poseFrames = ref([])
// ===== BORIS 导出 =====
const exportOpen = ref(false)
const exportVideo = ref('E:/study/SOP_train_model_resource/sop_offline_teacher/web/public/video/v2_5m_7m_pose.mp4')
const exportFps = ref(24.702)
const exportHands = ref(false)
const exportMsg = ref('')

const duration = computed(() => poseDuration.value || segments.value.duration || 120)

function applyPose(pose) {
  const rows = (pose && pose.frames) || []
  poseFrames.value = rows
  poseDuration.value = Number((pose && pose.duration) || 0)
  poseRate.value = Number((pose && pose.rate) || 5)
  handSegments.value = { left: buildSegments(rows, 'left_hand'), right: buildSegments(rows, 'right_hand') }
}

async function switchPose(src) {
  if (!src || src === poseSrc.value) return
  try {
    const p = await fetch(src, { cache: 'no-store' }).then(r => r.json())
    poseSrc.value = src
    applyPose(p)
  } catch (e) {
    runNote.value = `姿态数据 ${src} 加载失败：${e?.message || e}`
  }
}

const stepSegs = computed(() => segments.value.steps || [])
const activeStep = computed(() => stepSegs.value.find(s => currentSec.value >= s.start && currentSec.value < s.end) || null)
const activeStepName = computed(() => activeStep.value?.name || '')

// 右侧四张卡片：**只读当前「结果来源」发布的运行结果**。
// 不再回退旧的 ablation 数据（那会导致"工序轨是新的、卡片是旧的"这种错位）。
const cardReady = computed(() => !!activeStep.value)
const cardTitle = computed(() => {
  if (!cardReady.value) return '该时刻无结果'
  const st = activeStep.value
  return (st.step_id && st.step_id !== 'unknown')
    ? `${st.step_id} · ${st.name}`
    : `未映射到 SOP 步骤（${st.name || '无描述'}）`
})
const cardSpan = computed(() => cardReady.value
  ? `片段 ${activeStep.value.start.toFixed(2)} — ${activeStep.value.end.toFixed(2)} 秒`
  : `当前时刻 ${fmt(currentSec.value)} 不在任何结果段内`)
const cardAction = computed(() => cardReady.value
  ? (activeStep.value.action || '（该段无动作描述）')
  : '拖动播放头到有结果的区间')
const cardActionMeta = computed(() => {
  if (!cardReady.value) return '或换一个「结果来源」'
  const st = activeStep.value
  return st.evidence ? `依据：${st.evidence}` : `状态变化：${st.state_change || 'none'}`
})
// 左右手字段兼容两种形态：
//   字符串（早期模型只给一句话）
//   结构化 {action, holding, target}（Qwen 片段补全后的形态）
function handAction(v) {
  if (!v) return ''
  return typeof v === 'string' ? v : (v.action || '')
}
function handMeta(v) {
  if (!v || typeof v === 'string') return ''
  const parts = []
  if (v.holding) parts.push(`持握：${v.holding}`)
  if (v.target) parts.push(`作用于：${v.target}`)
  return parts.join(' · ')
}
const cardLeft = computed(() => {
  if (!cardReady.value) return '—'
  return handAction(activeStep.value.left_hand) || '（模型未给出）'
})
const cardLeftMeta = computed(() => {
  if (!cardReady.value) return '无数据'
  const m = handMeta(activeStep.value.left_hand)
  const extra = `状态变化：${activeStep.value.state_change || 'none'}` +
    (activeStep.value.need_review ? ' · 该段待人工复核' : '')
  return m ? `${m}　${extra}` : extra
})
const cardRight = computed(() => {
  if (!cardReady.value) return '—'
  return handAction(activeStep.value.right_hand) || '（模型未给出）'
})
const cardRightMeta = computed(() => {
  if (!cardReady.value) return '无数据'
  const m = handMeta(activeStep.value.right_hand)
  const extra = `来源：${activeRun.value || '当前运行'}` +
    (activeStep.value.need_review ? ' · 待人工复核' : '')
  return m ? `${m}　${extra}` : extra
})

// 标注模式：只留工序一条轨、把条拉高，专门用来拖拽编辑
const annotMode = ref(false)
function toggleAnnot() {
  annotMode.value = !annotMode.value
  editMode.value = annotMode.value
  editIdx.value = -1
  editMsg.value = annotMode.value
    ? '标注模式：拖动两端改边界、拖中间平移、双击空白处插入新段'
    : ''
  if (annotMode.value && !stepOptions.value.length) loadStepOptions()
}

// 剪映式插入：在 t 处插入 dur 秒的段
//   若 t 落在某段内部 → 先把该段拆成两段，再把 t 之后的所有段整体右移（ripple）
function insertSegAt(t, dur = 3) {
  const st = segments.value.steps || []
  const t0 = Math.max(0, Math.round(t * 100) / 100)
  const d = Math.max(0.3, Math.round(dur * 100) / 100)
  // 1) 拆开被插入点穿过的段
  const out = []
  for (const sg of st) {
    if (t0 > Number(sg.start) + 1e-6 && t0 < Number(sg.end) - 1e-6) {
      out.push({ ...sg, end: t0 })
      out.push({ ...sg, idx: sg.idx + 0.5, start: t0 })
    } else {
      out.push({ ...sg })
    }
  }
  // 2) t0 之后的段整体右移 d（给新段腾位置）
  for (const sg of out) {
    if (Number(sg.start) >= t0 - 1e-6) {
      sg.start = Math.round((Number(sg.start) + d) * 100) / 100
      sg.end = Math.round((Number(sg.end) + d) * 100) / 100
    } else if (Number(sg.end) > t0 + 1e-6) {
      sg.end = t0
    }
  }
  // 3) 插入新段
  out.push({
    idx: out.length, step_id: 'unknown', name: '流程外或插入式局部处理',
    start: t0, end: Math.min(duration.value, Math.round((t0 + d) * 100) / 100),
    kind: 'step', source: 'manual', color: 6, need_review: true, layer: 0,
    action: '（人工新增，待填写）', evidence: '', state_change: 'none',
    left_hand: '', right_hand: '',
  })
  out.sort((a, b) => a.start - b.start)
  // 4) 裁掉超出片尾的
  const kept = out.filter(sg => Number(sg.start) < duration.value - 0.05)
  for (const sg of kept) sg.end = Math.min(duration.value, Number(sg.end))
  const final = kept.filter(sg => Number(sg.end) - Number(sg.start) > 0.05)
  segments.value.steps = final
  refreshSegMeta()
  editIdx.value = final.findIndex(sg => Math.abs(Number(sg.start) - t0) < 1e-6)
  editMsg.value = `已在 ${t0.toFixed(2)}s 插入 ${d}s 片段（后面片段整体右移）`
}

// ===== 磁吸：吸附到相邻段边缘 / 播放头 / 0 / 片尾 =====
const snapLine = ref(null)      // 吸附时显示的参考线位置（秒）
const SNAP_PX = 10              // 吸附半径（像素）

function snapTargets(excludeIdx) {
  const ts = [0, duration.value, Math.round(currentSec.value * 100) / 100]
  for (const sg of segments.value.steps || []) {
    if (sg.idx === excludeIdx) continue
    ts.push(Math.round(Number(sg.start) * 100) / 100, Math.round(Number(sg.end) * 100) / 100)
  }
  return ts
}

function snapTime(t, excludeIdx, railEl) {
  const w = railEl ? railEl.getBoundingClientRect().width : 0
  const tol = w > 0 ? (duration.value * SNAP_PX) / w : 0.25
  let best = null
  for (const c of snapTargets(excludeIdx)) {
    const d = Math.abs(c - t)
    if (d <= tol && (best === null || d < best.d)) best = { t: c, d }
  }
  if (best) { snapLine.value = best.t; return Math.round(best.t * 100) / 100 }
  snapLine.value = null
  return t
}

function onRailDblClick(e) {
  if (!annotMode.value || !e) return
  const railEl = (e.currentTarget && e.currentTarget.closest)
    ? e.currentTarget.closest('.tl-rail')
    : (e.target && e.target.closest ? e.target.closest('.tl-rail') : null)
  if (!railEl) return
  insertSegAt(timeFromEvent(e, railEl))
}

const annotRows = computed(() => {
  const st = stepSegs.value || []
  const maxL = st.reduce((m, sg) => Math.max(m, Number(sg.layer ?? 0)), 0)
  const rows = []
  for (let l = 0; l <= maxL; l++) {
    rows.push({
      key: `step-${l}`, label: l === 0 ? '工序' : `工序 L${l + 1}`, cls: 'step', layer: l,
      segs: st.filter(sg => Number(sg.layer ?? 0) === l),
    })
  }
  return rows
})
const tlRows = computed(() => {
  if (annotMode.value) return annotRows.value
  return [
    { key: 'step', label: '工序', cls: 'step', layer: 0, segs: stepSegs.value },
    { key: 'left', label: '左手', cls: 'left', segs: handSegments.value.left.map(([start, end]) => ({ start, end })) },
    { key: 'right', label: '右手', cls: 'right', segs: handSegments.value.right.map(([start, end]) => ({ start, end })) },
  ]
})
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
const loadCount = ref(0)
const loadedAt = ref('')
const stationMeta = ref({ id: 'ST01', name: '默认工位', frame: [1920, 1080] })
// 多机位支持：configs/stations.json 里每个机位一个 station，互不覆盖。
// stationsDoc 保存整份文档，保存时只替换当前机位，不能像以前那样
// 用单个 station 覆盖整个数组（那样会把别的机位的 ROI 删掉）。
const stationsDoc = ref(null)
const stationId = ref('')
const stationChoices = ref([])
const stationsFile = ref('configs/stations.json')
const savedAtText = ref('')
// 请求序号：只接受最后一次发起的 /api/stations 响应。
// 首屏会先后触发两次加载（先默认机位、再按 run 里声明的机位），
// 如果不丢弃过期响应，先发后到的那次会把刚切好的 regions 覆盖掉，
// 现象就是"刷新后 ROI 有时对有时不对 / 改了记不住"。
let stationReqSeq = 0
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
// 数值微调单个坐标。拖拽只能新建框，改不了已有框的位置，
// 所以给每个区域 4 个输入框做精确设置（并立刻标成"未保存"）。
function setCoord(r, i, raw) {
  const v = Number(raw)
  if (!Number.isFinite(v)) return
  const clamped = Math.min(1, Math.max(0, v))
  const next = [...r.rect_norm]
  next[i] = Math.round(clamped * 10000) / 10000
  // 保证 x0<x1、y0<y1
  if (next[2] - next[0] < 0.01) next[2] = Math.min(1, next[0] + 0.01)
  if (next[3] - next[1] < 0.01) next[3] = Math.min(1, next[1] + 0.01)
  r.rect_norm = next
  saveMsg.value = '坐标已改，点下面按钮保存（还没保存）'
  saveBad.value = false
  drawRois()
}
function removeRegion(id) {  regions.value = regions.value.filter(r => r.id !== id)
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
async function loadStations(preferId) {
  const seq = ++stationReqSeq
  try {
    const data = await fetch('/api/stations', { cache: 'no-store' }).then(r => r.json())
    if (seq !== stationReqSeq) return   // 有更新的请求在跑，本次结果作废
    stationsDoc.value = data
    const list = data.stations || []
    stationChoices.value = list.map(s => ({ id: s.id, name: s.name || s.id, camera: s.camera || '' }))
    const want = preferId || stationId.value
    const st = list.find(s => s.id === want) || list[0]
    if (st) {
      stationId.value = st.id || 'ST01'
      stationMeta.value = {
        id: st.id || 'ST01',
        name: st.name || '默认工位',
        frame: st.frame || [1920, 1080],
        camera: st.camera || '',
      }
      regions.value = (st.regions || []).map(r => ({ ...r }))
      savedAtText.value = st.saved_at || '（该机位还没保存过）'
    }
    loadedAt.value = new Date().toLocaleTimeString()
    loadCount.value = regions.value.length
    // 载入后立刻画一次，让"已保存的区域"在刷新后能直接看到（不必先进编辑模式）
    await nextTick()
    drawRois()
  } catch (err) {
    if (seq !== stationReqSeq) return
    saveMsg.value = '读取配置失败：' + err.message
    saveBad.value = true
  }
}
async function saveStations() {
  saving.value = true
  saveMsg.value = ''
  saveBad.value = false
  try {
    // 只替换当前机位，其余机位原样带回去
    const doc = stationsDoc.value && Array.isArray(stationsDoc.value.stations)
      ? JSON.parse(JSON.stringify(stationsDoc.value))
      : { version: 4, stations: [] }
    if (!doc.stations.length) doc.stations.push({ id: stationId.value || 'ST01' })
    let idx = doc.stations.findIndex(s => s.id === (stationId.value || doc.stations[0].id))
    if (idx < 0) idx = 0
    doc.stations[idx] = {
      ...doc.stations[idx],
      ...stationMeta.value,
      id: doc.stations[idx].id || stationMeta.value.id || 'ST01',
      regions: regions.value,
      saved_at: new Date().toISOString().replace('T', ' ').slice(0, 19),
    }
    doc.version = doc.version || 4
    const res = await fetch('/api/stations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(doc),
      cache: 'no-store',
    })
    const out = await res.json()
    if (!res.ok) throw new Error(out.error || res.statusText)
    if (out.file) stationsFile.value = out.file
    // 保存后再回读一次，确认磁盘上真的是这份内容（而不是"以为保存成功"）
    const back = await fetch('/api/stations', { cache: 'no-store' }).then(r => r.json())
    const stBack = (back.stations || []).find(s => s.id === doc.stations[idx].id) || {}
    const n = (stBack.regions || []).length
    const expect = regions.value.length
    if (n !== expect) {
      saveBad.value = true
      saveMsg.value = `保存后回读不一致：磁盘 ${n} 个区域，页面 ${expect} 个（请检查 ${out.file}）`
      return
    }
    stationsDoc.value = back
    savedAtText.value = stBack.saved_at || ''
    saveMsg.value = `已保存 ${expect} 个区域 → 工位 ${doc.stations[idx].id}（${stBack.name || ''}）｜文件 ${out.file}`
  } catch (err) {
    saveBad.value = true
    saveMsg.value = '保存失败：' + (err?.message || err)
  } finally {
    saving.value = false
  }
}
// 开发期调试钩子：只在本机 dev build 里挂到 window，方便自动化测试画框流程
if (import.meta.env.DEV) {
  window.__roi = { regions, editing, getDraft: () => draft, getDragging: () => dragging }
}

// ===== BORIS 7.0 导出 =====
// 格式依据：BORIS 的 .boris 是 JSON；State 事件成对出现，
// 事件元组固定 6 元 [秒, 主体, 行为代码, 修饰, "", 帧号]。
const BORIS_KEYS = 'abcdefghijklmnopqrstuvwxyz'

function handSegsFor(rows, key) {
  const out = []; let cur = null
  const half = 1 / (2 * (poseRate.value || 10))
  for (const r of rows) {
    const pts = r[key] || []
    if (pts.filter(p => p[2] > HAND_SCORE).length < HAND_MIN_POINTS) continue
    const t0 = Math.max(0, r.timestamp - half), t1 = r.timestamp + half
    if (!cur) cur = [t0, t1]
    else if (t0 - cur[1] <= HAND_MERGE_GAP) cur[1] = t1
    else { out.push(cur); cur = [t0, t1] }
  }
  if (cur) out.push(cur)
  return out
}

function buildBoris(opts = {}) {
  // opts 仅供自动化测试覆盖，正常调用不传参
  const fps = Number(opts.fps ?? exportFps.value) || 24.702
  const wantHands = opts.hands ?? exportHands.value
  const behaviors = []
  const seen = new Map()
  const reg = (code, desc) => { if (!seen.has(code)) { seen.set(code, behaviors.length); behaviors.push({ code, desc }) } }
  const events = []
  for (const seg of stepSegs.value) {
    const code = String(seg.step_id || seg.id || 'step')
    reg(code, String(seg.name || code))
    events.push([Number(seg.start), code])
    events.push([Number(seg.end), code])
  }
  if (wantHands && poseFrames.value.length) {
    for (const [side, code, desc] of [['left_hand', 'hand_left_visible', '左手可见'],
                                      ['right_hand', 'hand_right_visible', '右手可见']]) {
      const segs = handSegsFor(poseFrames.value, side)
      if (!segs.length) continue
      reg(code, desc)
      for (const [a, b] of segs) { events.push([a, code]); events.push([b, code]) }
    }
  }
  events.sort((x, y) => x[0] - y[0])

  const behaviors_conf = {}
  behaviors.forEach((b, i) => {
    behaviors_conf[String(i)] = {
      type: 'State event', key: BORIS_KEYS[i] || '', code: b.code, description: b.desc,
      color: '', category: '', modifiers: '', excluded: '', 'coding map': '',
    }
  })

  const vp = String(opts.video ?? exportVideo.value ?? '').replace(/\\/g, '/')
  const name = 'sop_export_' + new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')
  const now = new Date().toISOString()
  const obs = {
    file: { 1: [vp], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [] },
    type: 'MEDIA', date: now, description: '', 'time offset': 0.0,
    events: events.map(([t, code]) => [Math.round(t * 1000) / 1000, '', code, '', '', Math.round(t * fps)]),
    'observation time interval': [0, 0],
    independent_variables: {},
    visualize_spectrogram: false, visualize_waveform: false,
    media_creation_date_as_offset: false, media_scan_sampling_duration: 0,
    image_display_duration: 1, close_behaviors_between_videos: false,
    media_info: {
      length: { [vp]: poseDuration.value || segments.value.duration || 0 },
      fps: { [vp]: 25.0 }, hasVideo: { [vp]: true }, hasAudio: { [vp]: false },
      offset: { 1: 0.0 }, display: { [vp]: 'Nothing' },
    },
  }
  return {
    time_format: 'hh:mm:ss', project_date: now,
    project_name: name, project_description: name, project_format_version: '7.0',
    subjects_conf: {}, behaviors_conf, observations: { [name]: obs },
    behavioral_categories: [], behavioral_categories_config: [], independent_variables: {},
    coding_map: {}, behaviors_coding_map: [], converters: {},
  }
}

function exportBoris() {
  const data = buildBoris()
  const evCount = Object.values(data.observations)[0].events.length
  if (!evCount) { exportMsg.value = '没有可导出的工序片段'; return }
  const text = JSON.stringify(data, null, 1)
  const blob = new Blob([text], { type: 'application/json;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const aEl = document.createElement('a')
  aEl.href = url
  aEl.download = data.project_name + '.boris'
  document.body.appendChild(aEl); aEl.click(); aEl.remove()
  setTimeout(() => URL.revokeObjectURL(url), 2000)
  exportMsg.value = '已导出 ' + Object.keys(data.behaviors_conf).length + ' 类行为 / ' + (evCount / 2) + ' 段，可直接用 BORIS 打开'
}

function toggleExport() {
  exportOpen.value = !exportOpen.value
}

// 开发期调试钩子：便于自动化比对导出结果
if (import.meta.env.DEV) {
  window.__boris = { build: (o) => buildBoris(o) }
  window.__edit = {
    segments, annotMode, editMode, editIdx, snapLine,
    toggleAnnot, insertSegAt, snapTime, refreshSegMeta, drag,
  }
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
  // selectedIndex 现在只用于"视频解码失败时显示对应抽帧图"，直接由当前秒数推出
  selectedIndex.value = Math.max(0, Math.floor(el.currentTime))
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
  drawRois()   // 视频尺寸就绪后再画一次，保证刷新后能立刻看到已保存的区域
}

onMounted(async () => {
  rafId = requestAnimationFrame(loop)
  window.addEventListener('resize', drawRois)
  const seekParam = Number(new URLSearchParams(location.search).get('t'))
  if (Number.isFinite(seekParam) && seekParam > 0) pendingSeek = seekParam
  // 顺序必须是：先读一次配置（保证面板有内容）→ 再加载「结果来源」
  // （它可能自带视频/姿态，并声明自己的机位）。两次都 await，避免并发覆盖。
  await loadStations()
  await reloadRuns(new URLSearchParams(location.search).get('run') || undefined)
  try {
    if (!activeRun.value) {
      const seg = await fetch('/data/segments.json').then(r => r.json()).catch(() => null)
      if (seg) segments.value = seg
    }
    if (poseSrc.value === DEFAULT_POSE) {
      const pose = await fetch(DEFAULT_POSE).then(r => r.json())
      applyPose(pose)
    }
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
        <button class="roi-toggle" :class="{ active: editing }" @click="toggleEdit">{{ editing ? '退出 ROI 编辑' : 'ROI 编辑' }}</button>
        <button class="roi-toggle" :class="{ active: exportOpen }" @click="toggleExport">{{ exportOpen ? '收起导出' : '导出 BORIS' }}</button>
      </div>
    </header>

    <section class="layout">
      <div class="left-panel">
        <div class="video-wrap" :class="{ editing }">
          <video v-show="!videoError" ref="video" controls preload="metadata" :poster="VIDEO_POSTER || null" :src="VIDEO_SRC" @loadstart="videoError = false" @timeupdate="syncVideo" @seeked="syncVideo" @loadedmetadata="onLoadedMetadata" @error="videoError = true" />
          <canvas
            ref="roiCanvas"
            class="roi-layer"
            :class="{ readonly: !editing }"
            v-show="editing || (regions.length > 0 && segments.show_rois !== false)"
            @pointerdown="onDown"
            @pointermove="onMove"
            @pointerup="onUp"
            @pointerleave="onUp"
          />
          <div v-if="videoError" class="fallback"><img :src="`/data/frame_${String(selectedIndex).padStart(4,'0')}_00.jpg`" /><p>视频解码失败，当前显示分析抽帧。</p></div>
        </div>
        <div v-if="editing" class="roi-panel">
          <p class="roi-status" :class="{ bad: loadCount === 0 }">
            <strong>机位</strong>
            <select v-model="stationId" class="roi-station" @change="loadStations(stationId)">
              <option v-for="s in stationChoices" :key="s.id" :value="s.id">{{ s.name }}（{{ s.id }}）</option>
            </select>
            <template v-if="stationMeta.camera">｜录像 {{ stationMeta.camera }}</template>
            <br />
            当前已加载 {{ loadCount }} 个区域<template v-if="loadedAt">（读取于 {{ loadedAt }}）</template>
            <template v-if="loadCount === 0">：这个机位还没保存过区域，画完记得点下面的保存按钮</template>
            <br />
            文件 <code>{{ stationsFile }}</code>
            <template v-if="savedAtText">｜该机位最后保存 {{ savedAtText }}</template>
          </p>
          <p class="roi-hint">在画面上按住左键拖拽即可画框。ROI 是<strong>按机位</strong>分开保存的：每个机位一套坐标，互不覆盖。这些区域是工位配置：不裁剪画面，而是作为额外输入让模型重点关注。</p>
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
              <!-- 数值微调：拖拽只能新建，改不了已有框的位置，这里给精确输入 -->
              <div class="roi-nums">
                <span class="roi-nums-label">坐标 x0,y0,x1,y1</span>
                <input v-for="(v, i) in r.rect_norm" :key="i" class="roi-num" type="number"
                       step="0.001" min="0" max="1" :value="v"
                       @change="setCoord(r, i, $event.target.value)" />
              </div>
            </div>
            <p v-if="!regions.length" class="roi-empty">还没有区域，拖拽画面开始画第一个。</p>
          </div>
          <div class="roi-actions">
            <button @click="saveStations" :disabled="saving">{{ saving ? '保存中…' : '保存到 configs/stations.json' }}</button>
            <button @click="clearRegions">清空</button>
            <span class="roi-msg" :class="{ bad: saveBad }">{{ saveMsg }}</span>
          </div>
        </div>
      </div>

      <aside class="cards">
        <p v-if="!cardReady" class="card-warn">
          该时刻（{{ fmt(currentSec) }}）不在当前结果的任何段内。{{ coverageNote }}
        </p>
        <article><h3>{{ segments.sop_catalog?.length ? '候选 SOP 步骤' : '标准 SOP 描述' }}</h3><strong>{{ cardTitle }}</strong><p>{{ cardSpan }}</p></article>
        <article><h3>当前动作描述</h3><strong>{{ cardAction }}</strong><p>{{ cardActionMeta }}</p>
          <details v-if="activeStep?.children?.length">
            <summary>查看本步骤的细动作</summary>
            <button v-for="(child, i) in activeStep.children" :key="i" class="run-reload" @click="seekTo(child.start)">
              {{ child.name }}（{{ child.start.toFixed(1) }}–{{ child.end.toFixed(1) }}s）
            </button>
          </details>
        </article>
        <article><h3>左手</h3><strong>{{ cardLeft }}</strong><p>{{ cardLeftMeta }}</p></article>
        <article class="orange"><h3>右手</h3><strong>{{ cardRight }}</strong><p>{{ cardRightMeta }}</p></article>
      </aside>
    </section>

    <section v-if="exportOpen" class="export-panel">
      <h2>导出标注给 BORIS
        <span class="timeline-note">BORIS 7.0 项目文件（.boris，JSON）；State 事件成对，事件元组 [秒, 主体, 行为代码, 修饰, 空, 帧号]</span>
      </h2>
      <div class="ex-row">
        <label>视频绝对路径</label>
        <input v-model="exportVideo" class="ex-input" placeholder="E:/.../v2_5m_7m_pose.mp4" />
      </div>
      <div class="ex-row">
        <label>有效帧率</label>
        <input v-model.number="exportFps" type="number" step="0.001" class="ex-num" />
        <span class="ex-hint">= 总帧数 / 时长（当前视频 2964 / 119.99 = 24.702，不是标称 25）</span>
      </div>
      <div class="ex-row">
        <label>附带</label>
        <label class="ex-check"><input type="checkbox" v-model="exportHands" /> 同时导出左右手可见段</label>
        <span class="ex-hint">导出后建议用 scripts/check_boris_file.py 校验一次</span>
      </div>
      <div class="ex-row">
        <button class="ex-go" @click="exportBoris">下载 .boris</button>
        <span class="ex-msg">{{ exportMsg }}</span>
      </div>
    </section>

    <section class="timeline">
      <h2>时间轴
        <span class="timeline-note">三道轨道共用同一坐标轴 0 – {{ fmt(duration) }}，白色播放头联动；{{ coverageNote }}</span>
      </h2>
      <div class="run-bar" v-if="runs.length">
        <label>结果来源
          <select v-model="activeRun" @change="switchRun">
            <option v-for="r in runs" :key="r.name" :value="r.name">
              {{ r.label || r.name }}（{{ r.steps }} 段{{ r.need_review ? `，${r.need_review} 段待复核` : '' }}）
            </option>
          </select>
        </label>
        <span class="run-note">{{ runNote }}</span>
        <button class="run-reload" @click="reloadRuns(activeRun)" title="重新读取 runs/index.json，用于跑完新结果后不刷新页面">重新读取清单</button>
        <button class="run-reload" :class="{ on: annotMode }" @click="toggleAnnot"
                title="标注模式：只留工序一轨并拉高，方便拖拽/插入/平移片段">标注模式</button>
        <button class="run-reload" :class="{ on: editMode }" @click="toggleSegEdit"
                title="在网页上微调工序段：点段选中 → 改步骤名 / 挪边界 / 删除 → 保存">工序段编辑</button>
        <button v-if="outOfCoverage" class="run-reload" @click="seekTo(segments.coverage[0] + 0.1)">
          跳到有数据的 {{ segments.coverage[0].toFixed(1) }}s
        </button>
      </div>
      <details v-if="segments.sop_catalog?.length" class="sop-catalog" open>
        <summary>模型归纳的候选 SOP（{{ segments.sop_catalog.length }} 步）</summary>
        <p>步骤来自三位工人的视频归纳，可点击时间轴核对。{{ segments.catalog_note }}</p>
        <table>
          <thead><tr><th>步骤</th><th>画面判据</th><th>工人差异</th></tr></thead>
          <tbody><tr v-for="s in segments.sop_catalog" :key="s.step_id">
            <td>{{ s.step_id }} · {{ s.name }}</td><td>{{ s.visual_definition }}</td><td>{{ s.worker_variations || '待复核' }}</td>
          </tr></tbody>
        </table>
        <a v-if="segments.examples_url" :href="segments.examples_url" target="_blank" rel="noopener">查看三个视频的工序示例片段</a>
      </details>
      <div v-if="editMode" class="seg-edit">
        <template v-if="editSeg">
          <b>第 {{ editIdx + 1 }} 段</b>
          <span>{{ editSeg.start.toFixed(2) }} – {{ editSeg.end.toFixed(2) }}s</span>
          <select :value="editSeg.step_id" @change="setSegStepId($event.target.value)">
            <option v-for="o in stepOptions" :key="o.step_id" :value="o.step_id">
              {{ o.step_id }} · {{ o.name_cn }}
            </option>
          </select>
          <button @click="nudge('start', -0.1)">起点 −0.1</button>
          <button @click="nudge('start', 0.1)">起点 +0.1</button>
          <button @click="nudge('end', -0.1)">终点 −0.1</button>
          <button @click="nudge('end', 0.1)">终点 +0.1</button>
          <button class="danger" @click="delSeg">删除该段</button>
          <button @click="addSegAtPlayhead">在播放头新增段</button>
          <button class="primary" @click="saveEdited">保存到 runs/</button>
        </template>
        <span v-else>点时间轴上的工序段选中它</span>
        <span class="seg-msg">{{ editMsg }}</span>
      </div>
      <p v-if="outOfCoverage" class="run-warn">
        当前时刻（{{ fmt(currentSec) }}）不在这次结果的覆盖范围
        （{{ segments.coverage[0].toFixed(1) }}–{{ segments.coverage[1].toFixed(1) }} 秒），
        右侧卡片会回退显示旧数据；点右边按钮或拖动播放头到该区间才能看到这次结果。
      </p>
      <div class="tl" :class="{ annot: annotMode }">
        <div class="tl-rows">
          <div class="tl-row" v-for="row in tlRows" :key="row.key">
            <span class="tl-label">{{ row.label }}</span>
            <div class="tl-rail" @click="seekFromTimeline"
                 @dblclick="row.cls === 'step' ? onRailDblClick($event) : null">
              <i v-for="(seg, i) in row.segs" :key="i"
                 class="tl-seg" :class="[row.cls, { on: row.cls === 'step' && activeStep && seg.idx === activeStep.idx }]"
                 :style="{ left: pct(seg.start) + '%', width: Math.max(0.3, pct(seg.end - seg.start)) + '%',
                           background: row.cls === 'step' ? PALETTE[(seg.color ?? i) % PALETTE.length] : '' }"
                 :title="`${seg.name || row.label}  ${Number(seg.start).toFixed(1)}–${Number(seg.end).toFixed(1)}s`"
                 @click.stop="row.cls === 'step' ? pickSeg(seg) : seekTo(seg.start)"
                 @pointerdown="row.cls === 'step' && editMode ? onSegPointerDown($event, seg) : null">
                <span v-if="annotMode && row.cls === 'step'" class="name">{{ seg.step_id }}</span>
                <template v-if="editMode && row.cls === 'step'">
                  <span class="h h-l" title="拖动改起点"></span>
                  <span class="h h-r" title="拖动改终点"></span>
                </template>
              </i>
            </div>
          </div>
        </div>
        <div v-if="snapLine !== null" class="snap-line" :style="{ left: pct(snapLine) + '%' }" />
        <div class="tl-play"><i :style="{ left: playheadPct + '%' }" /></div>
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
