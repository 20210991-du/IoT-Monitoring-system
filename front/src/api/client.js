// API client — Mac Studio Express server.js 와 same-origin 통신
//   기본은 빈 문자열 → 사이트를 서빙하는 같은 origin (/api/*) 호출
//   개발 모드 (vite dev :5173) 에서는 VITE_API_BASE_URL=http://localhost:5050 권장
const BASE = import.meta.env.VITE_API_BASE_URL || "";

async function get(path) {
  const res = await fetch(`${BASE}${path}`, { signal: AbortSignal.timeout(8000) });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export const fetchHealth       = () => get("/api/health");
export const fetchDevices      = () => get("/api/devices");
export const fetchAnomalies    = () => get("/api/anomalies");
export const fetchInsights     = () => get("/api/insights");
export const fetchAnnouncement = () => get("/api/announcement");   // 배너 공지 (공개 GET)

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
// 백엔드 좌표: 군산 (lat 35.80~36.00, lng 126.45~126.83)
//   - 새만금방조제 (lng 126.477) ~ 대야 (lng 126.810)
//   - 단말 lat/lng 가 있으면 leaflet 이 그대로 사용 → 아래 정규화는 fallback
// 맵 좌표: x/y 정규화 [0, 1]  (mock data shape 호환용)
// ──────────────────────────────────────────────────────────────
const LAT_MIN = 35.80, LAT_MAX = 36.00;
const LNG_MIN = 126.45, LNG_MAX = 126.83;
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
      label:        d.aiRisk ? `AI ${d.aiRisk}` : d.label || "",
      mse:          d.aiMse ?? d.mse ?? null,
      threshold:    d.aiThreshold ?? d.threshold ?? null,
      aiRatio:      d.aiRatio ?? (d.aiMse != null && d.aiThreshold > 0 ? d.aiMse / d.aiThreshold : null),
      zone:         d.zone,
      location:     d.location ?? null,
      updatedAt:    d.updatedAt ?? null,
      contribution: d.contribution || [],
    }));
}
