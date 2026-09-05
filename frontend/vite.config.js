import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Built straight into the Flask app's static folder. Everything is bundled -
// no CDN, because the lab PC may have no internet.
export default defineConfig({
  plugins: [react()],
  base: '/',
  build: {
    outDir: '../gradient_bench/api/static',
    emptyOutDir: true,
    assetsDir: 'assets',
  },
  server: { proxy: { '/api': 'http://127.0.0.1:5051' } },
})
