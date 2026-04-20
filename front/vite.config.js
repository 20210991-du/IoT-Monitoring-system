import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 매설배관 AI 통합관제 시스템 — Vite 설정
// - 1920×1080 고정 캔버스 프로토타입(CDN React + 브라우저 Babel)을
//   Vite + React + ESM 구조로 번들링합니다.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
  },
  preview: {
    port: 4173,
  },
});
