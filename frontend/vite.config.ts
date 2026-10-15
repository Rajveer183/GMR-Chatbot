import net from 'node:net'
import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

const env = loadEnv(process.env.NODE_ENV || 'development', process.cwd(), '')

// The backend (uvicorn) may run on 8001 or 8000.
// Vite's proxy only accepts a single target, so probe the candidates and
// proxy to whichever one is actually listening.
const API_HOST = env.VITE_API_HOST || process.env.VITE_API_HOST || '127.0.0.1'
const CANDIDATE_PORTS = (env.VITE_API_PORT || process.env.VITE_API_PORT || '8001,8000')
  .split(',')
  .map((p) => Number(p.trim()))
  .filter((p) => Number.isInteger(p) && p > 0)

const targetFor = (port: number) => `http://${API_HOST}:${port}`

function isListening(port: number, timeout = 300): Promise<boolean> {
  return new Promise((resolve) => {
    const socket = net.connect({ host: API_HOST, port })
    const done = (ok: boolean) => {
      socket.destroy()
      resolve(ok)
    }
    socket.setTimeout(timeout)
    socket.once('connect', () => done(true))
    socket.once('timeout', () => done(false))
    socket.once('error', () => done(false))
  })
}

let activeTarget = targetFor(CANDIDATE_PORTS[0])
let probing: Promise<void> | null = null

function refreshTarget() {
  if (probing) return probing
  probing = (async () => {
    for (const port of CANDIDATE_PORTS) {
      if (await isListening(port)) {
        const next = targetFor(port)
        if (next !== activeTarget) {
          console.log(`[proxy] /api -> ${next}`)
          activeTarget = next
        }
        return
      }
    }
  })().finally(() => {
    probing = null
  })
  return probing
}

// Pick a target before the server starts, and keep re-checking so the backend
// can be started (or restarted on the other port) after Vite is already up.
await refreshTarget()

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: Number(env.PORT || env.VITE_PORT || process.env.PORT || process.env.VITE_PORT) || 5174,
    strictPort: false,
    proxy: {
      '/api': {
        target: activeTarget,
        changeOrigin: true,
        bypass(_req, _res, options) {
          // Runs before each proxied request: point at the live backend.
          options.target = activeTarget
          void refreshTarget()
          return undefined
        },
        configure(proxy) {
          proxy.on('error', (err) => {
            console.error(`[proxy] ${activeTarget} failed: ${err.message}`)
            void refreshTarget()
          })
        },
      },
    },
  },
})
