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
      server.middlewares.use('/api/stations', (req, res, next) => {
        if (req.method === 'GET') {
          try {
            const txt = fs.existsSync(STATIONS_FILE) ? fs.readFileSync(STATIONS_FILE, 'utf8') : '{"version":1,"stations":[]}'
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
              fs.mkdirSync(path.dirname(STATIONS_FILE), { recursive: true })
              fs.writeFileSync(STATIONS_FILE, JSON.stringify(data, null, 2), 'utf8')
              res.setHeader('Content-Type', 'application/json; charset=utf-8')
              res.end(JSON.stringify({ ok: true, file: STATIONS_FILE, stations: data.stations.length }))
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

export default defineConfig({
  plugins: [vue(), stationsApi()],
  // 显式绑定 IPv4 回环地址并锁定端口。
  // 之前 Vite 只监听 ::1（IPv6），因此 http://127.0.0.1:5173/ 连接会被拒绝，
  // 只有解析到 IPv6 的 localhost 才能打开页面，这是“网站打不开”的直接原因之一。
  server: { host: '127.0.0.1', port: 5173, strictPort: true },
})
