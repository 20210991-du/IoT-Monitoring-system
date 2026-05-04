import { useEffect, useRef } from "react";
import L from "leaflet";

const TILES = {
  light: {
    url: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  },
  dark: {
    url: "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
    attribution: '&copy; <a href="https://carto.com/">CARTO</a>',
  },
  satellite: {
    url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attribution: "Tiles &copy; Esri",
  },
};

const LAT_MIN = 37.15, LAT_MAX = 37.85, LNG_MIN = 126.4, LNG_MAX = 127.6;
function toLatLng(m) {
  if (m.lat != null && m.lng != null) return [m.lat, m.lng];
  return [
    LAT_MIN + (1 - (m.y ?? 0.5)) * (LAT_MAX - LAT_MIN),
    LNG_MIN + (m.x ?? 0.5) * (LNG_MAX - LNG_MIN),
  ];
}

// 핀 안에 들어갈 status 별 흰색 아이콘 (24×24 viewBox 기준 path)
const PIN_INNER_ICON = {
  // 위험: alert + 굵게 강조 (anomaly 와 같은 path, 색만 다름)
  critical: `<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><path d="M12 9v4"/><circle cx="12" cy="17" r="1.1" fill="white" stroke="none"/>`,
  // 이상 의심: alert (삼각형 + ! )
  anomaly: `<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><path d="M12 9v4"/><circle cx="12" cy="17" r="0.9" fill="white" stroke="none"/>`,
  // 관찰 필요: eye
  warn:    `<path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7-11-7-11-7z"/><circle cx="12" cy="12" r="3"/>`,
  // 정상: check
  normal:  `<path d="M5 12.5 10 17l9-11"/>`,
  // 통신 장애: wifi_off (사선)
  offline: `<path d="M2 2l20 20"/><path d="M8.5 16.5a5 5 0 0 1 7 0"/><path d="M2 8.8A17 17 0 0 1 7 6"/><path d="M22 8.8A17 17 0 0 0 16 5.8"/><circle cx="12" cy="20" r="1.2" fill="white" stroke="none"/>`,
};

function makeIcon(status) {
  const color =
    status === "critical" ? "#991b1b" :
    status === "anomaly"  ? "#ef4444" :
    status === "warn"     ? "#f59e0b" : "#4f46e5";
  const iconPaths = PIN_INNER_ICON[status] || "";
  const inner = iconPaths
    ? `<g transform="translate(9 7) scale(0.583)" stroke="white" stroke-width="2.4" fill="none" stroke-linecap="round" stroke-linejoin="round">${iconPaths}</g>`
    : `<circle cx="16" cy="14" r="5" fill="rgba(255,255,255,0.92)"/>`;
  const svg = `<svg width="32" height="40" viewBox="0 0 32 40" xmlns="http://www.w3.org/2000/svg" style="filter:drop-shadow(0 3px 4px rgba(0,0,0,0.35))"><path d="M16 0 C 8 0 2 6 2 14 C 2 24 16 40 16 40 C 16 40 30 24 30 14 C 30 6 24 0 16 0 Z" fill="${color}" stroke="rgba(255,255,255,0.85)" stroke-width="1.5"/>${inner}</svg>`;
  return L.divIcon({ html: svg, className: "", iconSize: [32, 40], iconAnchor: [16, 40], popupAnchor: [0, -44] });
}

// 클러스터 색상 — status 별
const CLUSTER_COLOR = {
  normal:  "#10b981",  // 정상 → 초록
  offline: "#64748b",  // 통신장애 → 회색
  mixed:   "#4f46e5",  // 혼합 (정상+장애) → 보라
};

function makeClusterIcon(count, status = "mixed") {
  const color = CLUSTER_COLOR[status] || CLUSTER_COLOR.mixed;
  const iconPaths = PIN_INNER_ICON[status] || "";
  // normal: 카운트 배지 X → 핀 head 중심(20, 16)에 정확히 정렬
  // offline: 카운트 배지 O → 배지와 겹치지 않게 좌하 offset
  const tx = status === "normal" ? 13 : 10;
  const ty = status === "normal" ? 10 : 9;
  // mixed (정상+장애 혼합) 는 3 dots 유지, normal/offline 은 해당 아이콘
  const inner = iconPaths
    ? `<g transform="translate(${tx} ${ty}) scale(0.583)" stroke="white" stroke-width="2.8" fill="none" stroke-linecap="round" stroke-linejoin="round">${iconPaths}</g>`
    : `<circle cx="20" cy="13" r="2.2" fill="rgba(255,255,255,0.95)"/><circle cx="14" cy="21" r="2.2" fill="rgba(255,255,255,0.95)"/><circle cx="26" cy="21" r="2.2" fill="rgba(255,255,255,0.95)"/>`;
  // 핀 모양 + 상태 아이콘 (+ 카운트 배지 — 정상 클러스터는 생략)
  const badge = status === "normal"
    ? ""
    : `<circle cx="32" cy="6" r="7" fill="white" stroke="${color}" stroke-width="1.5"/><text x="32" y="9.5" font-family="-apple-system,system-ui,sans-serif" font-weight="800" font-size="9" text-anchor="middle" fill="${color}">${count}</text>`;
  const svg = `<svg width="40" height="48" viewBox="0 0 40 48" xmlns="http://www.w3.org/2000/svg" style="filter:drop-shadow(0 3px 5px rgba(0,0,0,0.35))">
    <path d="M20 0 C 11 0 4 7 4 16 C 4 27 20 48 20 48 C 20 48 36 27 36 16 C 36 7 29 0 20 0 Z" fill="${color}" stroke="rgba(255,255,255,0.85)" stroke-width="1.5"/>
    ${inner}
    ${badge}
  </svg>`;
  return L.divIcon({
    html: svg,
    className: "",
    iconSize: [40, 48],
    iconAnchor: [20, 48],
    popupAnchor: [0, -48],
  });
}

function popupHtml(m) {
  const color = m.status === "anomaly" ? "#ef4444" : "#f59e0b";
  const contrib = (m.contribution || []).slice(0, 2).map((c, i) => {
    const bg  = i === 0 ? (m.status === "anomaly" ? "rgba(239,68,68,0.12)" : "rgba(245,158,11,0.12)") : "#f1f5f9";
    const col = i === 0 ? color : "#64748b";
    const bd  = i === 0 ? (m.status === "anomaly" ? "rgba(239,68,68,0.3)" : "rgba(245,158,11,0.3)") : "#e2e8f0";
    return `<span style="font-size:9px;font-weight:700;padding:2px 6px;border-radius:4px;background:${bg};color:${col};border:1px solid ${bd}">${c.sensor} ${c.pct}%</span>`;
  }).join("");
  return `<div style="min-width:190px;font-family:system-ui,sans-serif">
    <div style="font-family:JetBrains Mono,monospace;font-weight:700;font-size:13px;color:${color}">${m.node}</div>
    <div style="font-size:12px;font-weight:600;margin-top:4px;color:#555">${m.label || ""}</div>
    <div style="margin-top:8px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px">
      <div><div style="font-size:9px;color:#888;margin-bottom:1px">MSE</div><div style="font-weight:700;font-size:12px">${m.mse != null ? m.mse.toFixed(3) : "-"}</div></div>
      <div><div style="font-size:9px;color:#888;margin-bottom:1px">구역</div><div style="font-weight:700;font-size:12px">${m.zone || "-"}</div></div>
      <div><div style="font-size:9px;color:#888;margin-bottom:1px">상태</div><div style="font-weight:700;font-size:12px">${m.status === "anomaly" ? "이상" : "관찰"}</div></div>
    </div>
    ${contrib ? `<div style="margin-top:8px;display:flex;gap:4px;flex-wrap:wrap">${contrib}</div>` : ""}
  </div>`;
}

export function MapPanel({ markers, onMarker, mapStyle, focus, fitTrigger }) {
  const containerRef    = useRef(null);
  const mapRef          = useRef(null);
  const tileRef         = useRef(null);
  const leafletMarkers  = useRef([]);
  const markerByNode    = useRef(new Map());
  const mapStyleRef     = useRef(mapStyle);

  // 맵 초기화 (한 번만)
  useEffect(() => {
    const el = containerRef.current;
    if (!el || mapRef.current) return;

    const map = L.map(el, {
      center: [37.5665, 126.9780],
      zoom: 11,
      zoomControl: false,
    });
    mapRef.current = map;

    const t = TILES[mapStyleRef.current] || TILES.light;
    tileRef.current = L.tileLayer(t.url, { attribution: t.attribution }).addTo(map);

    return () => {
      map.remove();
      mapRef.current  = null;
      tileRef.current = null;
      leafletMarkers.current = [];
    };
  }, []);

  // 타일 교체 (mapStyle 변경 시)
  useEffect(() => {
    mapStyleRef.current = mapStyle;
    const map = mapRef.current;
    if (!map) return;
    if (tileRef.current) map.removeLayer(tileRef.current);
    const t = TILES[mapStyle] || TILES.light;
    tileRef.current = L.tileLayer(t.url, { attribution: t.attribution }).addTo(map);
  }, [mapStyle]);

  // 마커 갱신
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    leafletMarkers.current.forEach((lm) => lm.remove());
    leafletMarkers.current = [];
    markerByNode.current.clear();

    markers.forEach((m) => {
      const pos  = toLatLng(m);
      const icon = m.kind === "cluster" ? makeClusterIcon(m.count, m.status) : makeIcon(m.status);
      const lm   = L.marker(pos, { icon }).addTo(map);
      if (m.kind === "single") {
        lm.bindPopup(popupHtml(m));
        lm.on("click", () => onMarker && onMarker(m));
        if (m.node) markerByNode.current.set(m.node, lm);
      }
      leafletMarkers.current.push(lm);
    });
  }, [markers, onMarker]);

  // 외부 focus 요청 → flyTo + 매칭 single 마커 popup
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !focus || focus.lat == null || focus.lng == null) return;
    map.flyTo([focus.lat, focus.lng], 15, { duration: 0.8 });
    if (focus.node) {
      const lm = markerByNode.current.get(focus.node);
      if (lm) {
        // flyTo 끝난 뒤 팝업 열기 (대략 800ms 후)
        const id = setTimeout(() => lm.openPopup(), 820);
        return () => clearTimeout(id);
      }
    }
  }, [focus]);

  // 외부 fit 요청 → 모든 마커가 보이도록 flyToBounds
  useEffect(() => {
    if (!fitTrigger) return;
    const map = mapRef.current;
    if (!map) return;
    map.closePopup();
    if (markers.length === 0) return;
    if (markers.length === 1) {
      const [lat, lng] = toLatLng(markers[0]);
      map.flyTo([lat, lng], 14, { duration: 0.8 });
      return;
    }
    const bounds = L.latLngBounds(markers.map((m) => toLatLng(m)));
    if (bounds.isValid()) {
      map.flyToBounds(bounds, { padding: [50, 50], duration: 0.8, maxZoom: 13 });
    }
  }, [fitTrigger, markers]);

  return (
    <div
      ref={containerRef}
      style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}
    />
  );
}
