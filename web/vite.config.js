import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  // 显式绑定 IPv4 回环地址并锁定端口。
  // 之前 Vite 只监听 ::1（IPv6），因此 http://127.0.0.1:5173/ 连接会被拒绝，
  // 只有解析到 IPv6 的 localhost 才能打开页面，这是“网站打不开”的直接原因之一。
  server: { host: '127.0.0.1', port: 5173, strictPort: true },
})
