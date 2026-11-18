import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const here = path.dirname(fileURLToPath(import.meta.url))
const STATIONS_FILE = path.resolve(here, '..', 'configs', 'stations.json')

/**
 * 只在本机 dev server 上提供固定 ROI 配置的读写接口。
 * 浏览器不能直接写文件，所以由 dev server 代写；生产环境（NX）上配置文件是随程序分发的，
 * 这个中间件不会打包进 dist。
 */
function stationsApi() {
  return {
    name: 'stations-api',
    configureServer(server) {
      // 不要用 import.meta.url 推路径：Vite 会把配置文件打包到临时目录（.vite.config.js.*.tmpdir），
      // 那时 '..' 会解析到 web/ 而不是项目根，读写会落到另一个文件上。
      // server.config.root 恒等于 vite 的根目录（web/），用它最稳。
      const root = server.config.root || here
      const file = path.resolve(root, '..', 'configs', 'stations.json')
      console.log('[stations-api] 配置文件:', file)
      server.middlewares.use('/api/stations', (req, res, next) => {
        // 这个接口读的是磁盘上的实时配置，绝不能被浏览器缓存，
        // 否则"保存了但刷新还是旧的"（实测踩过）。
        res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate')
        res.setHeader('Pragma', 'no-cache')
        if (req.method === 'GET') {
          try {
            const txt = fs.existsSync(file) ? fs.readFileSync(file, 'utf8') : '{"version":1,"stations":[]}'
            res.setHeader('Content-Type', 'application/json; charset=utf-8')
            res.end(txt)
          } catch (err) {
            res.statusCode = 500
            res.end(JSON.stringify({ error: String(err) }))
          }
          return
        }
        if (req.method === 'POST') {
          let body = ''
          req.on('data', chunk => { body += chunk })
          req.on('end', () => {
            try {
              // Windows 记事本/PowerShell 写出的 JSON 常带 BOM，先剥掉再解析
              const data = JSON.parse(body.replace(/^\uFEFF/, ''))
              if (!data || !Array.isArray(data.stations)) throw new Error('stations 必须是数组')
              fs.mkdirSync(path.dirname(file), { recursive: true })
              fs.writeFileSync(file, JSON.stringify(data, null, 2), 'utf8')
              // 写完立刻回读，确认真的落盘（而不是"以为保存成功"）
              const back = JSON.parse(fs.readFileSync(file, 'utf8'))
              const summary = (back.stations || []).map(s => ({
                id: s.id, name: s.name, regions: (s.regions || []).length,
                saved_at: s.saved_at || null,
              }))
              // 留痕：把每次保存的摘要追加到 api_runs/roi_save_log.jsonl，
              // 以后出现"保存了但刷新是旧的"可以直接看这条日志判断是前端没发、还是后端没写。
              try {
                const logDir = path.resolve(root, '..', 'api_runs')
                fs.mkdirSync(logDir, { recursive: true })
                fs.appendFileSync(
                  path.resolve(logDir, 'roi_save_log.jsonl'),
                  JSON.stringify({ at: new Date().toISOString(), file, summary }) + '\n',
                  'utf8')
              } catch (e) { /* 日志失败不影响保存 */ }
              res.setHeader('Content-Type', 'application/json; charset=utf-8')
              res.end(JSON.stringify({ ok: true, file, stations: (back.stations || []).length,
                                       summary, verified: true }))
            } catch (err) {
              res.statusCode = 400
              res.end(JSON.stringify({ error: String(err) }))
            }
          })
          return
        }
        next()
      })
    },
  }
}

/**
 * /api/steps   GET  → 返回 configs/sop_steps.json 的步骤表（前端下拉用）
 * /api/runs    POST → 把网页里编辑后的工序段写回 web/public/data/runs/<name>.json
 *                     并同步 index.json（只改动段数/待复核数）
 */
function editorApi() {
  return {
    name: 'editor-api',
    configureServer(server) {
      const root = server.config.root || here
      const stepsFile = path.resolve(root, '..', 'configs', 'sop_steps.json')
      const runsDir = path.resolve(root, 'public', 'data', 'runs')
      server.middlewares.use('/api/steps', (req, res) => {
        res.setHeader('Cache-Control', 'no-store')
        try {
          const d = JSON.parse(fs.readFileSync(stepsFile, 'utf8'))
          const steps = [...(d.steps || []), ...(d.fallback_steps || [])]
            .map(s => ({ step_id: s.step_id, name_cn: s.name_cn }))
          res.setHeader('Content-Type', 'application/json; charset=utf-8')
          res.end(JSON.stringify({ steps }))
        } catch (err) {
          res.statusCode = 500
          res.end(JSON.stringify({ error: String(err) }))
        }
      })
      server.middlewares.use('/api/runs', (req, res) => {
        res.setHeader('Cache-Control', 'no-store')
        if (req.method !== 'POST') { res.statusCode = 405; res.end('{}'); return }
        let body = ''
        req.on('data', c => { body += c })
        req.on('end', () => {
          try {
            const data = JSON.parse(body.replace(/^\uFEFF/, ''))
            const name = String(data.name || '').replace(/[^\w.-]/g, '')
            if (!name || !Array.isArray(data.steps)) throw new Error('name/steps 缺失')
            const file = path.join(runsDir, `${name}.json`)
            // 备份原文件，可回退
            if (fs.existsSync(file)) {
              fs.copyFileSync(file, path.join(runsDir, `${name}.bak.json`))
            }
            const doc = { ...data.doc, steps: data.steps }
            fs.writeFileSync(file, JSON.stringify(doc, null, 1), 'utf8')
            // 同步 index
            const idxFile = path.join(runsDir, 'index.json')
            if (fs.existsSync(idxFile)) {
              const idx = JSON.parse(fs.readFileSync(idxFile, 'utf8'))
              const entry = (idx.runs || []).find(r => r.name === name)
              if (entry) {
                entry.steps = data.steps.length
                entry.need_review = data.steps.filter(s => s.need_review).length
                entry.published_at = new Date().toLocaleString('sv-SE').replace('T', ' ')
                fs.writeFileSync(idxFile, JSON.stringify(idx, null, 1), 'utf8')
              }
            }
            res.setHeader('Content-Type', 'application/json; charset=utf-8')
            res.end(JSON.stringify({ ok: true, file, steps: data.steps.length }))
          } catch (err) {
            res.statusCode = 400
            res.end(JSON.stringify({ error: String(err) }))
          }
        })
      })
    },
  }
}

export default defineConfig({
  plugins: [vue(), stationsApi(), editorApi()],
  // 显式绑定 IPv4 回环地址并锁定端口。
  // 之前 Vite 只监听 ::1（IPv6），因此 http://127.0.0.1:5173/ 连接会被拒绝，
  // 只有解析到 IPv6 的 localhost 才能打开页面，这是“网站打不开”的直接原因之一。
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    // 不要让 Vite 去 watch public/video 下的大 mp4：
    // 这些文件在渲染/转码时会被占用，chokidar 会抛 EBUSY 并让整个 dev server 退出
    // （实测 2026-11-17 就是这样崩的：watch side3_discovered_h264.mp4 → EBUSY）。
    // 视频是静态资源，本来也不需要热更新。
    watch: {
      ignored: [
        '**/public/video/**',
        '**/public/data/station_roi/**',
        '**/public/data/roi/**',
        '**/node_modules/**',
        '**/dist/**',
      ],
    },
  },
})
