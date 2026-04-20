// Custom SVG "map" — stylized topology with roads, water, district blobs.
// Works in light, dark, and satellite modes.

function Marker({ m, theme, active, onClick }) {
  const color =
    m.status === "anomaly" ? "var(--err)" :
    m.status === "warn"    ? "var(--warn)" :
                             "var(--brand)";
  const ring = m.status === "anomaly";
  const style = {
    position: "absolute",
    left: `${m.x * 100}%`,
    top: `${m.y * 100}%`,
    transform: "translate(-50%, -100%)",
    cursor: "pointer",
  };
  if (m.kind === "cluster") {
    return (
      <div
        style={{ ...style, transform: "translate(-50%, -50%)" }}
        onClick={() => onClick && onClick(m)}
      >
        <div style={{
          width: 44, height: 44, borderRadius: "50%",
          background: "var(--brand)",
          boxShadow: "0 0 0 6px rgba(79,70,229,0.22), 0 8px 24px -6px rgba(79,70,229,0.5)",
          color: "#fff",
          display: "grid", placeItems: "center",
          fontWeight: 800, fontSize: 16, fontFamily: "Space Grotesk",
          border: "2px solid rgba(255,255,255,0.8)",
        }}>
          {m.count}
        </div>
      </div>
    );
  }
  return (
    <div style={style} onClick={() => onClick && onClick(m)}>
      <div style={{ position: "relative", width: 32, height: 40 }}>
        {ring && (
          <div style={{
            position: "absolute", left: 6, top: 20, width: 20, height: 20,
            borderRadius: "50%", background: color, opacity: 0.4,
            animation: "pulse-ring 1.6s ease-out infinite",
          }} />
        )}
        <svg
          width="32"
          height="40"
          viewBox="0 0 32 40"
          style={{
            filter: active
              ? `drop-shadow(0 0 8px ${color})`
              : "drop-shadow(0 3px 4px rgba(0,0,0,0.25))",
          }}
        >
          <path
            d="M16 0 C 8 0 2 6 2 14 C 2 24 16 40 16 40 C 16 40 30 24 30 14 C 30 6 24 0 16 0 Z"
            fill={color}
            stroke="rgba(255,255,255,0.85)"
            strokeWidth="1.5"
          />
          <circle cx="16" cy="14" r="5" fill="rgba(255,255,255,0.92)" />
          {m.status === "anomaly" && <circle cx="16" cy="14" r="2.2" fill={color} />}
        </svg>
      </div>
    </div>
  );
}

function MapSVG({ style }) {
  return (
    <svg
      viewBox="0 0 1000 560"
      preserveAspectRatio="xMidYMid slice"
      style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}
    >
      <defs>
        <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
          <path d="M 40 0 L 0 0 0 40" fill="none" stroke={style.grid} strokeWidth="0.5" />
        </pattern>
        <linearGradient id="water" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0%" stopColor={style.water1} />
          <stop offset="100%" stopColor={style.water2} />
        </linearGradient>
        <radialGradient id="district" cx="0.5" cy="0.5" r="0.6">
          <stop offset="0%" stopColor={style.district} stopOpacity="0.9" />
          <stop offset="100%" stopColor={style.district} stopOpacity="0.1" />
        </radialGradient>
      </defs>

      <rect width="1000" height="560" fill={style.land} />
      <rect width="1000" height="560" fill="url(#grid)" opacity="0.5" />

      <ellipse cx="260" cy="180" rx="220" ry="140" fill="url(#district)" opacity="0.6" />
      <ellipse cx="680" cy="320" rx="280" ry="170" fill="url(#district)" opacity="0.5" />
      <ellipse cx="500" cy="450" rx="260" ry="120" fill="url(#district)" opacity="0.4" />

      <path
        d="M -20 300 Q 150 260 300 310 T 600 340 T 900 300 T 1020 320 L 1020 360 Q 900 340 600 380 T 300 350 T 150 310 T -20 340 Z"
        fill="url(#water)"
      />
      <path
        d="M -20 300 Q 150 260 300 310 T 600 340 T 900 300 T 1020 320"
        fill="none"
        stroke={style.waterEdge}
        strokeWidth="1"
        opacity="0.6"
      />

      <g fill="none" stroke={style.roadMajor} strokeWidth="3.5" strokeLinecap="round" opacity="0.9">
        <path d="M 0 120 L 1000 180" />
        <path d="M 0 480 L 1000 460" />
        <path d="M 200 0 L 260 560" />
        <path d="M 720 0 L 680 560" />
      </g>
      <g fill="none" stroke={style.roadMajorCenter} strokeWidth="1" strokeDasharray="6 10" opacity="0.7">
        <path d="M 0 120 L 1000 180" />
        <path d="M 0 480 L 1000 460" />
        <path d="M 200 0 L 260 560" />
        <path d="M 720 0 L 680 560" />
      </g>

      <g fill="none" stroke={style.roadMinor} strokeWidth="1.6" opacity="0.8">
        <path d="M 80 60 L 400 240 L 360 480" />
        <path d="M 420 40 L 520 260 L 780 380 L 940 540" />
        <path d="M 880 80 L 820 220 L 940 420" />
        <path d="M 60 380 L 320 380 L 480 500" />
        <path d="M 500 80 L 620 80 L 640 200" />
        <path d="M 740 500 L 900 500" />
        <path d="M 120 200 L 280 240" />
        <path d="M 560 180 L 780 140" />
        <path d="M 380 320 L 620 320" />
      </g>

      <g fill="none" stroke={style.rail} strokeWidth="2" opacity="0.7">
        <path d="M 0 260 Q 300 220 600 260 T 1000 280" strokeDasharray="2 4" />
      </g>

      <g fill={style.block} opacity="0.55">
        <rect x="130" y="100" width="70" height="40" rx="2" />
        <rect x="230" y="80" width="50" height="60" rx="2" />
        <rect x="320" y="110" width="45" height="35" rx="2" />
        <rect x="80" y="180" width="90" height="40" rx="2" />
        <rect x="200" y="210" width="60" height="40" rx="2" />
        <rect x="540" y="200" width="80" height="50" rx="2" />
        <rect x="640" y="240" width="60" height="45" rx="2" />
        <rect x="750" y="200" width="70" height="50" rx="2" />
        <rect x="440" y="400" width="80" height="55" rx="2" />
        <rect x="540" y="420" width="60" height="50" rx="2" />
        <rect x="620" y="390" width="70" height="45" rx="2" />
        <rect x="100" y="420" width="90" height="55" rx="2" />
        <rect x="800" y="440" width="90" height="40" rx="2" />
      </g>

      <g fontFamily="Noto Sans KR" fontWeight="700" fontSize="11" fill={style.textFaint} opacity="0.8">
        <text x="170" y="80">성북구</text>
        <text x="520" y="130">중구</text>
        <text x="760" y="210">강동구</text>
        <text x="120" y="430">영등포구</text>
        <text x="620" y="460">송파구</text>
      </g>
      <g fontFamily="JetBrains Mono" fontSize="9" fill={style.textFaint} opacity="0.7">
        <text x="440" y="325">한강</text>
      </g>
    </svg>
  );
}

const MAP_STYLES = {
  light: {
    land: "#e8eef6", grid: "#c8d3e4",
    water1: "#9fc5e8", water2: "#7aafdb", waterEdge: "#6a9bc6",
    roadMajor: "#ffffff", roadMajorCenter: "#ffc96b", roadMinor: "#ffffff",
    rail: "#94a3b8", block: "#d2dbe9", district: "#a9b9d1", textFaint: "#1e3a5f",
    overlay: "rgba(255,255,255,0.0)",
  },
  dark: {
    land: "#0f1b33", grid: "#1e2e52",
    water1: "#1a3764", water2: "#102548", waterEdge: "#2a4d84",
    roadMajor: "#2a3e68", roadMajorCenter: "#6a7ab0", roadMinor: "#1e2e52",
    rail: "#4a5a82", block: "#1a2848", district: "#223a6a", textFaint: "#9fb0d8",
    overlay: "rgba(79,70,229,0.04)",
  },
  satellite: {
    land: "#2a3826", grid: "#3a4a33",
    water1: "#1a2a4a", water2: "#0f1f3a", waterEdge: "#2a3e5e",
    roadMajor: "#4a3a2a", roadMajorCenter: "#c8a060", roadMinor: "#3a2e22",
    rail: "#5a4632", block: "#3d4a32", district: "#445a33", textFaint: "#e8dcc0",
    overlay: "rgba(0,0,0,0.1)",
  },
};

export function MapPanel({ markers, activeId, onMarker, mapStyle, theme }) {
  const style = MAP_STYLES[mapStyle] || MAP_STYLES.light;
  return (
    <div style={{ position: "absolute", inset: 0, overflow: "hidden", borderRadius: "inherit" }}>
      <MapSVG style={style} />
      {mapStyle === "satellite" && (
        <div style={{
          position: "absolute", inset: 0, pointerEvents: "none",
          background: "radial-gradient(ellipse at 40% 30%, transparent 0%, rgba(0,0,0,0.35) 100%)",
        }} />
      )}

      <div style={{ position: "absolute", inset: 0, pointerEvents: "none", overflow: "hidden" }}>
        <div style={{
          position: "absolute", top: 0, bottom: 0, width: 60,
          background: `linear-gradient(90deg, transparent, ${theme === "dark" ? "rgba(124,116,255,0.18)" : "rgba(79,70,229,0.15)"}, transparent)`,
          animation: "scan-line 6s linear infinite",
        }} />
      </div>

      {markers.map((m) => (
        <Marker key={m.id} m={m} theme={theme} active={activeId === m.id} onClick={onMarker} />
      ))}
    </div>
  );
}
