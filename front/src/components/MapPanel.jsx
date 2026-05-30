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

// 군산 영역 — 시범 운영 사이트 (SITE_ID=2). 새만금 서쪽 ~ 대야 동쪽까지.
const LAT_MIN = 35.80, LAT_MAX = 36.00, LNG_MIN = 126.45, LNG_MAX = 126.83;
// 초기 지도 중심 — 군산 시청 부근 (대다수 단말 분포 중심)
const MAP_INITIAL_CENTER = [35.96, 126.70];
const MAP_INITIAL_ZOOM   = 11;
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

// 상태별 핀 채움색 (정상/위험/이상의심/통신장애)
const PIN_FILL = { critical: "#dc2626", warn: "#f59e0b", normal: "#10b981", offline: "#64748b" };

// 모든 단말 = 동일한 원형 핀 + 상태별 아이콘. selected 일 때 확대 + 링 (카카오/네이버식 선택 강조).
function makeIcon(status, selected = false) {
  const color = PIN_FILL[status] || "#4f46e5";
  // critical 은 PIN_INNER_ICON 에 별도 없음 → anomaly(경보) 아이콘 사용
  const paths = PIN_INNER_ICON[status] || PIN_INNER_ICON.anomaly || "";

  // 선택 시 = 물방울(teardrop) 핀 + 확대 (꼭지점이 위치에 닿고 위로 솟음)
  if (selected) {
    const tinner = paths
      ? `<g transform="translate(9 7) scale(0.583)" stroke="white" stroke-width="2.4" fill="none" stroke-linecap="round" stroke-linejoin="round">${paths}</g>`
      : `<circle cx="16" cy="14" r="5" fill="rgba(255,255,255,0.92)"/>`;
    const tripple = status === "critical"
      ? `<circle cx="16" cy="14" r="13" fill="none" stroke="#dc2626" stroke-width="2.4"><animate attributeName="r" values="13;22" dur="1.6s" repeatCount="indefinite"/><animate attributeName="opacity" values="0.85;0" dur="1.6s" repeatCount="indefinite"/></circle>`
      : "";
    const tsvg = `<svg width="32" height="40" viewBox="0 0 32 40" xmlns="http://www.w3.org/2000/svg" style="overflow:visible;filter:drop-shadow(0 3px 4px rgba(0,0,0,0.35))">${tripple}<path d="M16 0 C 8 0 2 6 2 14 C 2 24 16 40 16 40 C 16 40 30 24 30 14 C 30 6 24 0 16 0 Z" fill="${color}" stroke="white" stroke-width="1.5"/>${tinner}</svg>`;
    return L.divIcon({ html: tsvg, className: "", iconSize: [32, 40], iconAnchor: [16, 40], popupAnchor: [0, -44] });
  }

  // 기본 = 동그라미 핀 (16px, 모든 상태 동일)
  const D = 16, r = D / 2;
  const pad  = status === "critical" ? 16 : 5;      // ripple 여유
  const size = D + pad * 2;
  const c    = size / 2;
  const sc   = (D * 0.52) / 24;
  const it   = (c - 12 * sc).toFixed(2);
  const inner = paths
    ? `<g transform="translate(${it} ${it}) scale(${sc.toFixed(3)})" stroke="white" stroke-width="2.6" fill="none" stroke-linecap="round" stroke-linejoin="round">${paths}</g>`
    : `<circle cx="${c}" cy="${c}" r="${(r * 0.38).toFixed(1)}" fill="white"/>`;
  const ripple = status === "critical"
    ? `<circle cx="${c}" cy="${c}" r="${r}" fill="none" stroke="#dc2626" stroke-width="2.4">
         <animate attributeName="r"       values="${r};${r + pad}" dur="1.6s" repeatCount="indefinite"/>
         <animate attributeName="opacity" values="0.85;0"          dur="1.6s" repeatCount="indefinite"/>
       </circle>`
    : "";
  const svg = `<svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}" xmlns="http://www.w3.org/2000/svg" style="overflow:visible;filter:drop-shadow(0 2px 4px rgba(0,0,0,0.3))">
    ${ripple}
    <circle cx="${c}" cy="${c}" r="${r}" fill="${color}" stroke="white" stroke-width="1"/>
    ${inner}
  </svg>`;
  return L.divIcon({ html: svg, className: "", iconSize: [size, size], iconAnchor: [c, c], popupAnchor: [0, -(r + 4)] });
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

const STATUS_DISPLAY = {
  critical: { ko: "위험",      color: "#991b1b", bg: "rgba(220,38,38,0.12)",  bd: "rgba(220,38,38,0.3)"  },
  warn:     { ko: "이상 의심", color: "#b45309", bg: "rgba(245,158,11,0.12)", bd: "rgba(245,158,11,0.3)" },
  normal:   { ko: "정상",      color: "#047857", bg: "rgba(16,185,129,0.12)", bd: "rgba(16,185,129,0.3)" },
  offline:  { ko: "통신 장애", color: "#475569", bg: "rgba(100,116,139,0.12)", bd: "rgba(100,116,139,0.3)" },
};

function formatMse(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return "-";
  if (Math.abs(n) >= 0.01) return n.toFixed(4);
  return n.toFixed(6);
}

function formatAiRatio(m) {
  const ratio = Number.isFinite(Number(m.aiRatio))
    ? Number(m.aiRatio)
    : m.mse != null && Number(m.threshold) > 0
      ? Number(m.mse) / Number(m.threshold)
      : null;
  if (!Number.isFinite(ratio)) return "-";
  return ratio >= 100 ? `x${Math.round(ratio)}` : `x${ratio.toFixed(2)}`;
}

function featureLabel(name) {
  return String(name || "")
    .replace(/_dev24$/u, " 편차")
    .replace(/_diff1$/u, " 변화")
    .replace(/_/gu, " ");
}

function fmtWhen(v) {
  if (!v) return null;
  const d = new Date(v);
  if (isNaN(d.getTime())) return null;
  return d.toLocaleString("ko-KR", {
    timeZone: "Asia/Seoul", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit",
  });
}

// 마커 팝업 = 장비명만 (상태별 색상). 모든 상세 정보(상태·위치·측정·AI/MSE)는 우측 슬라이드 사이드바에서 (사용자 요청).
function popupHtml(m) {
  const sd = STATUS_DISPLAY[m.status] || STATUS_DISPLAY.warn;
  return `<div data-popup-node="${m.node}" style="font-family:system-ui,sans-serif;cursor:pointer;padding:2px 4px;white-space:nowrap" title="클릭하면 단말 상세 사이드바 열림">
    <span style="font-family:JetBrains Mono,monospace;font-weight:700;font-size:13px;color:${sd.color}">${m.node}</span>
  </div>`;
}

export function MapPanel({ markers, onMarker, mapStyle, focus, fitTrigger, boundsRequest, onMapClick, deselectTrigger }) {
  const containerRef    = useRef(null);
  const mapRef          = useRef(null);
  const tileRef         = useRef(null);
  const leafletMarkers  = useRef([]);
  const markerByNode    = useRef(new Map());
  const selectedNodeRef = useRef(null);   // 현재 선택(확대)된 단말 node
  const mapStyleRef     = useRef(mapStyle);
  const hasInitialFit   = useRef(false);   // markers 첫 로드 시 1회 fitBounds 트리거
  const onMapClickRef   = useRef(null);
  onMapClickRef.current = onMapClick;

  // 선택 해제 — 확대된(물방울) 마커를 동그라미로 되돌리고 선택 ref 초기화
  const deselectMarker = () => {
    const prev = selectedNodeRef.current;
    if (!prev) return;
    const plm = markerByNode.current.get(prev);
    if (plm && plm._single) plm.setIcon(makeIcon(plm._mStatus, false));
    selectedNodeRef.current = null;
  };

  // 맵 초기화 (한 번만)
  useEffect(() => {
    const el = containerRef.current;
    if (!el || mapRef.current) return;

    const map = L.map(el, {
      center: MAP_INITIAL_CENTER,   // 군산 시청 부근 (5/20 수정 — 이전 서울 좌표 버그)
      zoom: MAP_INITIAL_ZOOM,
      zoomControl: false,
    });
    mapRef.current = map;

    const t = TILES[mapStyleRef.current] || TILES.light;
    tileRef.current = L.tileLayer(t.url, { attribution: t.attribution }).addTo(map);

    // 빈 지도(마커 외) 클릭 → 선택 해제 + 사이드바 닫기 (카카오/네이버식)
    map.on("click", () => {
      deselectMarker();
      if (onMapClickRef.current) onMapClickRef.current();
    });

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
      lm._mStatus = m.status;
      if (m.kind === "single") {
        lm._single = true;
        lm.on("click", () => onMarker && onMarker(m));
        if (m.node) markerByNode.current.set(m.node, lm);
      }
      leafletMarkers.current.push(lm);
    });
    // 선택 상태 유지 (markers 갱신 후 재적용)
    if (selectedNodeRef.current) {
      const slm = markerByNode.current.get(selectedNodeRef.current);
      if (slm && slm._single) slm.setIcon(makeIcon(slm._mStatus, true));
    }

    // markers 가 처음 채워진 시점에 1회 fitBounds — 군산 단말이 다 화면 안에 들어오게
    if (!hasInitialFit.current && markers.length > 0) {
      const positions = markers.map(toLatLng).filter(([la, ln]) => la != null && ln != null);
      if (positions.length > 0) {
        try {
          map.fitBounds(positions, { padding: [40, 40], maxZoom: 13 });
          hasInitialFit.current = true;
        } catch {}
      }
    }
  }, [markers, onMarker]);

  // 외부 focus 요청 → 오프셋 재중심(우측 사이드바에 안 가리게) + 선택 마커 확대
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !focus || focus.lat == null || focus.lng == null) return;
    const Z = 15;
    // 핀을 화면 중앙이 아니라 '사이드바를 뺀 좌측 영역' 중앙에 두기 위해 중심을 동쪽으로 오프셋
    const target = map.unproject(map.project([focus.lat, focus.lng], Z).add([200, 0]), Z);
    map.flyTo(target, Z, { duration: 0.8 });
    if (focus.node) {
      // 이전 선택 마커 원복
      const prev = selectedNodeRef.current;
      if (prev && prev !== focus.node) {
        const plm = markerByNode.current.get(prev);
        if (plm && plm._single) plm.setIcon(makeIcon(plm._mStatus, false));
      }
      const lm = markerByNode.current.get(focus.node);
      if (lm) {
        if (lm._single) lm.setIcon(makeIcon(lm._mStatus, true));   // 선택 → 물방울 확대
        selectedNodeRef.current = focus.node;
      }
    }
  }, [focus]);

  // 외부 deselect 요청 (사이드바 X 버튼 등으로 닫힘) → 선택 마커 원복
  useEffect(() => {
    if (!deselectTrigger) return;
    deselectMarker();
  }, [deselectTrigger]);

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

  // 외부 bounds 요청 (예: 챗봇 응답이 여러 노드 언급 시) → 해당 좌표들 fitBounds
  useEffect(() => {
    if (!boundsRequest || !boundsRequest.coords || boundsRequest.coords.length === 0) return;
    const map = mapRef.current;
    if (!map) return;
    map.closePopup();
    if (boundsRequest.coords.length === 1) {
      const [lat, lng] = boundsRequest.coords[0];
      map.flyTo([lat, lng], 14, { duration: 0.8 });
      return;
    }
    const bounds = L.latLngBounds(boundsRequest.coords);
    if (bounds.isValid()) {
      map.flyToBounds(bounds, { padding: [60, 60], duration: 0.8, maxZoom: 13 });
    }
  }, [boundsRequest]);

  return (
    <div
      ref={containerRef}
      style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}
    />
  );
}
