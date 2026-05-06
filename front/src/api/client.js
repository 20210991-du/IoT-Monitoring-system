// API client — FastAPI 백엔드(http://localhost:8000) 연동
const BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function get(path) {
  const res = await fetch(`${BASE}${path}`, { signal: AbortSignal.timeout(8000) });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export const fetchHealth    = () => get("/api/health");
export const fetchDevices   = () => get("/api/devices");
export const fetchAnomalies = () => get("/api/anomalies");
export const fetchInsights  = () => get("/api/insights");

export async function predictDevice(deviceId) {
  const res = await fetch(`${BASE}/api/predict/${deviceId}`, {
    method: "POST",
    signal: AbortSignal.timeout(15000),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

// ──────────────────────────────────────────────────────────────
// 장비 배열 → MapPanel용 markers 배열 변환
// 백엔드 좌표: lat 37.5±0.175, lng 127.0±0.3
// 맵 좌표: x/y 정규화 [0, 1]
// ──────────────────────────────────────────────────────────────
const LAT_MIN = 37.15, LAT_MAX = 37.85;
const LNG_MIN = 126.4, LNG_MAX = 127.6;
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const avg   = (arr) => arr.reduce((a, b) => a + b, 0) / arr.length;

export function devicesToMarkers(devices) {
  const toX = (lng) => clamp((lng - LNG_MIN) / (LNG_MAX - LNG_MIN), 0.05, 0.95);
  const toY = (lat) => clamp(1 - (lat - LAT_MIN) / (LAT_MAX - LAT_MIN), 0.05, 0.95);

  // 모든 장비를 개별 single 마커로 표시 (클러스터링 X)
  // 색·아이콘은 status 별 (정상/위험/이상의심/통신장애)
  return devices
    .filter((d) => d.lat != null && d.lng != null)
    .map((d) => ({
      id:           `s-${d.deviceId}`,
      kind:         "single",
      lat:          d.lat,
      lng:          d.lng,
      x:            toX(d.lng),
      y:            toY(d.lat),
      status:       d.status,
      node:         d.deviceId,
      label:        d.label || "",
      mse:          d.mse || 0,
      zone:         d.zone,
      contribution: d.contribution || [],
    }));
}
