import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App.jsx";
import "./styles/global.css";
import "leaflet/dist/leaflet.css";

// 뷰포트 기반 레이아웃 — 캔버스는 CSS 로 100vw × 100vh 를 채우고,
// 내부 요소들은 position:absolute + flex/grid 로 자연스럽게 재배치됩니다.
// 1280×720 미만 뷰포트에서는 global.css 의 min-width/height 가 스크롤을 유발합니다.

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
  </StrictMode>
);
