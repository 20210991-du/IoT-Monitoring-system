import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App.jsx";
import "./styles/global.css";
import "leaflet/dist/leaflet.css";

// 뷰포트 기반 레이아웃 — 캔버스는 CSS 로 100vw × 100vh 를 채우고,
// 대시보드는 CSS 변수/미디어쿼리로 1920/1440/1280/1024 구간에 맞춰 압축됩니다.
// 1024×720 미만 뷰포트에서는 global.css 의 min-width/height 가 스크롤을 유발합니다.

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
  </StrictMode>
);
