import { useState, useEffect, useRef, useCallback } from "react";
import { Icons } from "../components/Icons.jsx";
import { signIn } from "../lib/authMock.js";

function fmtTime(d = new Date()) {
  return d.toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

const INITIAL_LOG = [
  { id: "boot-1", time: fmtTime(new Date(Date.now() - 4000)), kind: "ok", text: "SYS: 시스템 시작 · AI 엔진 초기화",     tail: "OK" },
  { id: "boot-2", time: fmtTime(new Date(Date.now() - 2000)), kind: "ai", text: "AI: LSTM-AutoEncoder 모델 로드 완료"            },
  { id: "boot-3", time: fmtTime(),                            kind: "ai", text: "AI: 백엔드 연결 확인 중..."                     },
];

// ── 테마 감지 ─────────────────────────────────────────────
function useTheme() {
  const [isDark, setIsDark] = useState(
    () => document.documentElement.dataset.theme === "dark"
  );
  useEffect(() => {
    const obs = new MutationObserver(() =>
      setIsDark(document.documentElement.dataset.theme === "dark")
    );
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => obs.disconnect();
  }, []);
  return isDark;
}

// ── 다크/라이트 팔레트 ────────────────────────────────────
const DARK = {
  bg0: "#060c18",
  bg1: "#0c1628",
  ink: "#e9eef9",
  inkSoft: "rgba(233,238,249,0.62)",
  inkMuted: "rgba(233,238,249,0.38)",
  inkFaint: "rgba(233,238,249,0.14)",
  brand: "#7c74ff",
  brand2: "#c084fc",
  ok: "#34d399",
  err: "#f87171",
  glassBg: "rgba(17,26,46,0.82)",
  glassBd: "rgba(124,116,255,0.22)",
  blobA: "rgba(79,70,229,.38)",
  blobB: "rgba(192,132,252,.24)",
  blobC: "rgba(20,184,166,.10)",
};

const LIGHT = {
  bg0: "#f0f4ff",
  bg1: "#ffffff",
  ink: "#0f172a",
  inkSoft: "rgba(15,23,42,0.65)",
  inkMuted: "rgba(15,23,42,0.42)",
  inkFaint: "rgba(15,23,42,0.12)",
  brand: "#4f46e5",
  brand2: "#7c3aed",
  ok: "#059669",
  err: "#dc2626",
  glassBg: "rgba(255,255,255,0.82)",
  glassBd: "rgba(79,70,229,0.16)",
  blobA: "rgba(79,70,229,.07)",
  blobB: "rgba(139,92,246,.05)",
  blobC: "transparent",
};

// ── 부팅 로그 ─────────────────────────────────────────────
const BOOT_LINES = [
  { text: "관제 시스템 초기화 중...",                  delay: 0    },
  { text: "LSTM-Autoencoder 가중치 복원 완료",         delay: 280  },
  { text: "IoT 네트워크 브리지 연결 완료",             delay: 580  },
  { text: "55개 감시 노드 상태 확인 완료",             delay: 900  },
  { text: "실시간 데이터 스트림 활성화",               delay: 1200 },
  { text: "보안 인증 모듈 준비 완료",                  delay: 1480 },
  { text: "시스템 준비 완료.  로그인 화면 로드 중...", delay: 1750 },
];


// ── 파티클 캔버스 ─────────────────────────────────────────
function ParticleCanvas({ isDark }) {
  const ref = useRef(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx      = canvas.getContext("2d");
    let raf;
    const mouse    = { x: -9999, y: -9999 };
    const sparkles = [];

    const hBase   = isDark ? 240 : 222;
    const hRange  = isDark ? 65  : 35;
    const aBase   = isDark ? 0.15 : 0.05;
    const aRange  = isDark ? 0.55 : 0.16;
    const brandRGB = isDark ? "124,116,255" : "79,70,229";

    const resize = () => { canvas.width = window.innerWidth; canvas.height = window.innerHeight; };
    resize();
    window.addEventListener("resize", resize);

    const onMove = (e) => {
      mouse.x = e.clientX; mouse.y = e.clientY;
      if (Math.random() < 0.4) {
        sparkles.push({
          x: e.clientX + (Math.random() - 0.5) * 12,
          y: e.clientY + (Math.random() - 0.5) * 12,
          vx: (Math.random() - 0.5) * 1.6,
          vy: -Math.random() * 2.0 - 0.4,
          r:  Math.random() * 2.2 + 0.5,
          a:  isDark ? 0.8 : 0.35,
          h:  hBase + Math.random() * hRange,
          life: 1,
        });
        if (sparkles.length > 90) sparkles.splice(0, 25);
      }
    };
    window.addEventListener("mousemove", onMove);

    const pts = Array.from({ length: 60 }, () => ({
      x:  Math.random() * window.innerWidth,
      y:  Math.random() * window.innerHeight,
      vx: (Math.random() - 0.5) * 0.35,
      vy: (Math.random() - 0.5) * 0.35,
      r:  Math.random() * 1.6 + 0.4,
      a:  Math.random() * aRange + aBase,
      h:  hBase + Math.random() * hRange,
    }));

    const tick = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      for (let i = 0; i < pts.length; i++) {
        for (let j = i + 1; j < pts.length; j++) {
          const dx = pts[i].x - pts[j].x, dy = pts[i].y - pts[j].y;
          const d  = Math.sqrt(dx * dx + dy * dy);
          if (d < 130) {
            ctx.beginPath();
            ctx.strokeStyle = `rgba(${brandRGB},${(1 - d / 130) * (isDark ? 0.10 : 0.05)})`;
            ctx.lineWidth = 0.5;
            ctx.moveTo(pts[i].x, pts[i].y); ctx.lineTo(pts[j].x, pts[j].y);
            ctx.stroke();
          }
        }
      }

      pts.forEach((p) => {
        const dx = p.x - mouse.x, dy = p.y - mouse.y;
        const d  = Math.sqrt(dx * dx + dy * dy);
        if (d < 110 && d > 0) {
          const f = (110 - d) / 110;
          p.vx += (dx / d) * f * 0.22; p.vy += (dy / d) * f * 0.22;
        }
        p.vx *= 0.97; p.vy *= 0.97;
        p.x  += p.vx;  p.y  += p.vy;
        if (p.x < 0) p.x = canvas.width;  if (p.x > canvas.width)  p.x = 0;
        if (p.y < 0) p.y = canvas.height; if (p.y > canvas.height) p.y = 0;

        const sat  = isDark ? 85 : 70;
        const lum  = isDark ? 70 : 50;
        const gSize = isDark ? p.r * 5 : p.r * 4;
        const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, gSize);
        g.addColorStop(0, `hsla(${p.h},${sat}%,${lum}%,${p.a})`);
        g.addColorStop(1, `hsla(${p.h},${sat}%,${lum}%,0)`);
        ctx.beginPath(); ctx.fillStyle = g;
        ctx.arc(p.x, p.y, gSize, 0, Math.PI * 2); ctx.fill();
        ctx.beginPath();
        ctx.fillStyle = `hsla(${p.h},100%,${isDark ? 88 : 40}%,${p.a * (isDark ? 1.5 : 1.2)})`;
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2); ctx.fill();
      });

      for (let i = sparkles.length - 1; i >= 0; i--) {
        const s = sparkles[i];
        s.x += s.vx; s.y += s.vy; s.vy += 0.06; s.life -= 0.038;
        if (s.life <= 0) { sparkles.splice(i, 1); continue; }
        ctx.save();
        ctx.globalAlpha = s.life * s.a;
        ctx.beginPath();
        ctx.fillStyle = `hsla(${s.h},90%,${isDark ? 80 : 45}%,1)`;
        ctx.arc(s.x, s.y, s.r * s.life, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }

      raf = requestAnimationFrame(tick);
    };
    tick();
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
      window.removeEventListener("mousemove", onMove);
    };
  }, [isDark]);

  return <canvas ref={ref} aria-hidden style={{ position: "absolute", inset: 0, width: "100%", height: "100%", zIndex: 0, pointerEvents: "none" }} />;
}

// ── SVG 데이터 흐름선 ──────────────────────────────────────
function DataFlowLines({ isDark }) {
  const stroke1 = isDark ? "#7c74ff" : "#4f46e5";
  const stroke2 = isDark ? "#c084fc" : "#7c3aed";
  return (
    <svg aria-hidden viewBox="0 0 1440 900" preserveAspectRatio="xMidYMid slice"
      style={{ position: "absolute", inset: 0, width: "100%", height: "100%", zIndex: 0, pointerEvents: "none", opacity: isDark ? 0.22 : 0.09 }}>
      <defs>
        <style>{`
          @keyframes fl1{from{stroke-dashoffset:900}to{stroke-dashoffset:0}}
          @keyframes fl2{from{stroke-dashoffset:700}to{stroke-dashoffset:0}}
          @keyframes fl3{from{stroke-dashoffset:1100}to{stroke-dashoffset:0}}
          .fl1{stroke-dasharray:220 680;animation:fl1 5s linear infinite}
          .fl2{stroke-dasharray:160 540;animation:fl2 6.5s linear infinite;animation-delay:-2s}
          .fl3{stroke-dasharray:110 990;animation:fl3 8s linear infinite;animation-delay:-4s}
          .fl4{stroke-dasharray:190 810;animation:fl3 7s linear infinite;animation-delay:-1.5s}
          .fl5{stroke-dasharray:130 670;animation:fl1 9s linear infinite;animation-delay:-5s}
        `}</style>
      </defs>
      <path className="fl1" d="M-100 180 C250 180 450 80 650 280 S1050 480 1600 180"  fill="none" stroke={stroke1} strokeWidth="1"/>
      <path className="fl2" d="M-100 620 C280 620 520 420 720 520 S1120 720 1600 520"  fill="none" stroke={stroke2} strokeWidth="0.8"/>
      <path className="fl3" d="M220 -50 C220 200 320 400 420 600 S520 800 620 950"     fill="none" stroke={stroke1} strokeWidth="0.6"/>
      <path className="fl4" d="M1180 -50 C1080 200 880 300 780 500 S680 700 780 950"   fill="none" stroke={stroke2} strokeWidth="1"/>
      <path className="fl5" d="M0 420 C380 370 780 470 1180 400 S1560 440 1800 420"   fill="none" stroke={stroke1} strokeWidth="0.7"/>
    </svg>
  );
}

// ── 부팅 시퀀스 ───────────────────────────────────────────
function BootScreen({ onDone, C, isDark }) {
  const [idx,      setIdx]      = useState(0);
  const [visible,  setVisible]  = useState(true);
  const [progress, setProgress] = useState(0);
  const [exiting,  setExiting]  = useState(false);

  useEffect(() => {
    const timers = BOOT_LINES.map((line, i) =>
      setTimeout(() => {
        setVisible(false);
        setTimeout(() => {
          setIdx(i);
          setProgress(Math.round(((i + 1) / BOOT_LINES.length) * 100));
          setVisible(true);
        }, 120);
      }, line.delay)
    );
    timers.push(setTimeout(() => { setExiting(true); setTimeout(onDone, 480); }, 2350));
    return () => timers.forEach(clearTimeout);
  }, [onDone]);

  const current = BOOT_LINES[idx];
  const isLast  = idx === BOOT_LINES.length - 1;

  return (
    <div style={{
      position: "fixed", inset: 0, background: C.bg0, zIndex: 100,
      display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center",
      fontFamily: "'JetBrains Mono','Courier New',monospace",
      opacity: exiting ? 0 : 1, transition: "opacity 480ms ease",
    }}>
      <style>{`
        @keyframes _bootFade { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:none} }
        @keyframes _pulseDot { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.4;transform:scale(1.6)} }
        @keyframes _blink    { 0%,100%{opacity:1} 50%{opacity:0} }
      `}</style>

      <div aria-hidden style={{
        position: "absolute", inset: 0, pointerEvents: "none",
        backgroundImage: `radial-gradient(circle,${isDark ? "rgba(124,116,255,.06)" : "rgba(79,70,229,.04)"} 1px,transparent 1px)`,
        backgroundSize: "40px 40px",
      }} />

      <div style={{ position: "relative", zIndex: 1, width: 480, textAlign: "center" }}>
        {/* 로고 */}
        <div style={{ marginBottom: 48 }}>
          <div style={{
            display: "inline-flex", alignItems: "center", gap: 12,
            padding: "10px 24px", border: `1px solid ${C.glassBd}`,
            borderRadius: 8, background: isDark ? "rgba(124,116,255,.06)" : "rgba(79,70,229,.04)",
          }}>
            <span style={{ width: 8, height: 8, borderRadius: "50%", background: C.brand, boxShadow: `0 0 20px ${C.brand}`, display: "inline-block", animation: "_pulseDot 1.2s ease-in-out infinite" }} />
            <span style={{ fontSize: 13, fontWeight: 700, color: C.ink, letterSpacing: "0.07em" }}>AI 통합관제 시스템 v2.1.0</span>
          </div>
        </div>

        {/* 현재 메시지 한 줄 */}
        <div style={{ height: 28, marginBottom: 36, display: "flex", alignItems: "center", justifyContent: "center", gap: 10 }}>
          <span style={{ color: isDark ? "rgba(124,116,255,.5)" : "rgba(79,70,229,.45)", fontSize: 13 }}>›</span>
          <span style={{
            fontSize: 13, color: isLast ? C.brand : C.inkSoft,
            opacity: visible ? 1 : 0,
            transform: visible ? "translateY(0)" : "translateY(6px)",
            transition: "opacity 120ms ease, transform 120ms ease",
          }}>
            {current?.text}
          </span>
          <span style={{
            display: "inline-block", width: 7, height: 14,
            background: C.brand, borderRadius: 1, verticalAlign: "middle",
            animation: "_blink .65s ease-in-out infinite",
          }} />
        </div>

        {/* 프로그레스 바 */}
        <div style={{ height: 2, borderRadius: 999, background: isDark ? "rgba(124,116,255,.12)" : "rgba(79,70,229,.10)", overflow: "hidden", marginBottom: 10 }}>
          <div style={{
            height: "100%", borderRadius: 999,
            background: `linear-gradient(90deg,${C.brand},${C.brand2})`,
            width: `${progress}%`, transition: "width 260ms ease",
            boxShadow: `0 0 8px ${C.brand}`,
          }} />
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: C.inkMuted }}>
          <span>시스템 초기화 중...</span>
          <span style={{ color: progress === 100 ? C.ok : C.brand, fontWeight: 700 }}>{progress}%</span>
        </div>
      </div>
    </div>
  );
}

// ── 실시간 시스템 로그 (Dashboard LogLine과 동일 포맷) ────
const LOG_KIND_COLOR = (C, isDark) => ({
  ok:   C.ok,
  ai:   C.brand,
  alert: C.err,
  warn: isDark ? "#fbbf24" : "#d97706",
  auth: C.inkMuted,
  data: C.inkSoft,
});

function LiveLog({ C, isDark }) {
  const [lines, setLines] = useState(() => [...INITIAL_LOG]);
  const seenIds = useRef(new Set(INITIAL_LOG.map((l) => l.id)));

  const push = useCallback((newLines) => {
    const fresh = newLines.filter((l) => !seenIds.current.has(l.id));
    if (!fresh.length) return;
    fresh.forEach((l) => seenIds.current.add(l.id));
    setLines((prev) => [...prev, ...fresh].slice(-5));
  }, []);

  useEffect(() => {
    const BASE = import.meta.env.VITE_API_BASE_URL || "";
    const get = (path) =>
      fetch(BASE + path, { signal: AbortSignal.timeout(5000) })
        .then((r) => (r.ok ? r.json() : null))
        .catch(() => null);

    // 최초 헬스 체크
    get("/api/health").then((d) => {
      push([{
        id: "health-init",
        time: fmtTime(),
        kind: d ? "ok" : "warn",
        text: d ? "SYS: 백엔드 연결 완료" : "SYS: 백엔드 오프라인 — 데모 모드",
        tail: d ? "OK" : null,
      }]);
    });

    // 이상 탐지 폴링
    const t1 = setInterval(async () => {
      const data = await get("/api/anomalies");
      if (!data?.length) return;
      const item    = data[Math.floor(Math.random() * data.length)];
      const isAlert = item.status === "critical";
      push([{
        id:   `anml-${item.deviceId}-${Date.now()}`,
        time: fmtTime(),
        kind: isAlert ? "alert" : "warn",
        text: `AI: ${item.deviceId} ${isAlert ? "이상 감지" : "이상 의심"} (MSE ${item.mse?.toFixed(3) ?? "-"})`,
        tail: isAlert ? "ALERT" : "WARN",
      }]);
    }, 9000);

    // 디바이스 상태 폴링
    const t2 = setInterval(async () => {
      const data = await get("/api/devices");
      if (!data?.length) return;
      const item    = data[Math.floor(Math.random() * data.length)];
      const kindMap = { normal: "ok", critical: "alert", warn: "warn", offline: "warn" };
      const textMap = { normal: "정상 수신", critical: "위험 확인 필요", warn: "이상 의심", offline: "통신 장애" };
      push([{
        id:   `dev-${item.deviceId}-${Date.now()}`,
        time: fmtTime(),
        kind: kindMap[item.status] || "data",
        text: `DATA: ${item.deviceId} ${textMap[item.status] ?? item.status}`,
        tail: item.status === "normal" ? "OK" : undefined,
      }]);
    }, 13000);

    return () => { clearInterval(t1); clearInterval(t2); };
  }, [push]);

  const kColors = LOG_KIND_COLOR(C, isDark);
  const bd = isDark ? "rgba(124,116,255,.10)" : "rgba(79,70,229,.10)";

  return (
    <div style={{
      marginTop: 36, maxWidth: 460,
      border: `1px solid ${bd}`,
      borderRadius: 12,
      background: isDark ? "rgba(0,0,0,.28)" : "rgba(255,255,255,.60)",
      backdropFilter: "blur(12px)",
      overflow: "hidden",
    }}>
      {/* 헤더 */}
      <div style={{
        padding: "7px 14px",
        borderBottom: `1px solid ${bd}`,
        display: "flex", alignItems: "center", gap: 8,
        fontSize: 11, color: C.inkMuted,
        fontFamily: "'JetBrains Mono','Courier New',monospace",
      }}>
        <span style={{ width: 5, height: 5, borderRadius: "50%", background: C.ok, boxShadow: `0 0 6px ${C.ok}`, display: "inline-block", animation: "_pulseDot 2s ease-in-out infinite" }} />
        실시간 시스템 로그
        <span style={{ marginLeft: "auto", color: C.inkFaint, fontSize: 10 }}>{lines.length} EVENTS</span>
      </div>

      {/* 고정 높이 로그 영역 — 레이아웃 점프 없음 */}
      <div style={{
        height: 130,              // 5줄 × 26px
        padding: "6px 0",
        overflow: "hidden",
        display: "flex", flexDirection: "column", justifyContent: "flex-end",
      }}>
        {lines.map((line, i) => {
          const color  = kColors[line.kind] || C.inkMuted;
          const bgLine = line.kind === "alert"
            ? `${C.err}10`
            : line.kind === "warn" ? `${kColors.warn}10` : "transparent";
          return (
            <div key={line.id} style={{
              display: "flex", alignItems: "baseline", gap: 8,
              padding: "2px 14px", borderRadius: 4,
              background: bgLine,
              fontSize: 11,
              fontFamily: "'JetBrains Mono','Courier New',monospace",
              animation: i === lines.length - 1 ? "_logFadeIn 250ms ease both" : "none",
              flexShrink: 0,
            }}>
              <span style={{ color, fontWeight: 700, flexShrink: 0, fontSize: 10 }}>[{line.time}]</span>
              <span style={{
                flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                color: (line.kind === "alert" || line.kind === "warn") ? color : C.inkSoft,
                fontWeight: line.kind === "alert" ? 700 : 400,
              }}>{line.text}</span>
              {line.tail && (
                <span style={{ color, fontWeight: 700, flexShrink: 0, fontSize: 10 }}>{line.tail}</span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── 타이핑 훅 ─────────────────────────────────────────────
function useTyping(text, speed = 75, delay = 500) {
  const [out, setOut] = useState("");
  useEffect(() => {
    let i = 0;
    const t = setTimeout(() => {
      const iv = setInterval(() => { i++; setOut(text.slice(0, i)); if (i >= text.length) clearInterval(iv); }, speed);
      return () => clearInterval(iv);
    }, delay);
    return () => clearTimeout(t);
  }, [text]);
  return out;
}

// ── 카운팅 훅 ─────────────────────────────────────────────
function useCount(target, duration = 1600, delay = 900) {
  const [val, setVal] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => {
      const start = performance.now();
      const tick  = (now) => {
        const p = Math.min((now - start) / duration, 1);
        const e = p === 1 ? 1 : 1 - Math.pow(2, -10 * p);
        setVal(Math.round(e * target));
        if (p < 1) requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    }, delay);
    return () => clearTimeout(t);
  }, [target, duration, delay]);
  return val;
}

// ── 글리치 훅 ─────────────────────────────────────────────
function useGlitch(baseInterval = 6000) {
  const [glitch, setGlitch] = useState(false);
  useEffect(() => {
    let outerTimer, innerTimer;
    const schedule = () => {
      outerTimer = setTimeout(() => {
        setGlitch(true);
        innerTimer = setTimeout(() => { setGlitch(false); schedule(); }, 90 + Math.random() * 110);
      }, baseInterval + Math.random() * 4000);
    };
    schedule();
    return () => { clearTimeout(outerTimer); clearTimeout(innerTimer); };
  }, [baseInterval]);
  return glitch;
}

// ── 입력 필드 ──────────────────────────────────────────────
function Field({ icon, label, focused, filled, C, children }) {
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ fontSize: 12, fontWeight: 600, color: C.inkSoft, marginBottom: 8, letterSpacing: "-0.01em" }}>{label}</div>
      <div style={{
        display: "flex", alignItems: "center", gap: 10,
        padding: "0 14px", height: 46, borderRadius: 12,
        background: "rgba(128,128,128,0.05)",
        border: `1px solid ${focused ? C.brand : filled ? `${C.ok}55` : C.inkFaint}`,
        boxShadow: focused ? `0 0 0 3px ${C.brand}22` : filled ? `0 0 0 2px ${C.ok}14` : "none",
        transition: "border-color 200ms, box-shadow 200ms",
      }}>
        <span style={{ color: focused ? C.brand : filled ? C.ok : C.inkMuted, display: "grid", placeItems: "center", transition: "color 200ms" }}>{icon}</span>
        {children}
        {filled && !focused && (
          <span style={{ color: C.ok, flexShrink: 0, display: "grid", placeItems: "center", animation: "_fadeUp 220ms ease both" }}>
            <Icons.check size={14} />
          </span>
        )}
      </div>
    </div>
  );
}

// ── 메인 ──────────────────────────────────────────────────
export function Login({ onLogin, onSignUp, prefillId }) {
  const isDark = useTheme();
  const C      = isDark ? DARK : LIGHT;

  const [booting,   setBooting]   = useState(true);
  const [id,        setId]        = useState(prefillId || "admin");
  const [pw,        setPw]        = useState(prefillId ? "" : "11111111");
  const [remember,  setRemember]  = useState(true);
  const [loading,   setLoading]   = useState(false);
  const [focus,     setFocus]     = useState(null);
  const [error,     setError]     = useState("");
  const [errStatus, setErrStatus] = useState(null);
  const [failCount, setFailCount] = useState(0);
  const [idFilled,  setIdFilled]  = useState(!!id);
  const [pwFilled,  setPwFilled]  = useState(!!pw);

  const typedText   = useTyping("통합관제 시스템", 75, 500);
  const nodeCount   = useCount(55, 1800, 900);
  const sensorCount = useCount(6,  1200, 1100);
  const glitch      = useGlitch(6000);

  const inputStyle = { flex: 1, background: "transparent", border: "none", outline: "none", color: C.ink, fontSize: 14, fontWeight: 500, padding: "12px 0", fontFamily: "inherit" };

  // 3D 카드 틸트
  const cardRef = useRef(null);
  const onCardMove = (e) => {
    const el = cardRef.current; if (!el) return;
    const r  = el.getBoundingClientRect();
    const mx = (e.clientX - r.left) / r.width;
    const my = (e.clientY - r.top)  / r.height;
    el.style.transform = `perspective(900px) rotateX(${(0.5 - my) * 9}deg) rotateY(${(mx - 0.5) * 14}deg) translateZ(6px)`;
    el.style.setProperty("--mx", `${mx * 100}%`);
    el.style.setProperty("--my", `${my * 100}%`);
  };
  const onCardLeave = () => {
    const el = cardRef.current; if (!el) return;
    el.style.transform = "perspective(900px) rotateX(0deg) rotateY(0deg) translateZ(0px)";
    el.style.setProperty("--mx", "50%"); el.style.setProperty("--my", "50%");
  };

  // 버튼 ripple
  const btnRef = useRef(null);
  const doRipple = (e) => {
    const btn = btnRef.current; if (!btn || loading) return;
    const r    = btn.getBoundingClientRect();
    const size = Math.max(r.width, r.height) * 2.5;
    const span = document.createElement("span");
    span.style.cssText = `position:absolute;width:${size}px;height:${size}px;` +
      `left:${e.clientX - r.left - size / 2}px;top:${e.clientY - r.top - size / 2}px;` +
      `border-radius:50%;background:rgba(255,255,255,0.22);transform:scale(0);` +
      `animation:_ripple 620ms ease-out forwards;pointer-events:none;`;
    btn.appendChild(span);
    setTimeout(() => span.remove(), 700);
  };

  const submit = (e) => {
    e && e.preventDefault();
    setLoading(true); setError(""); setErrStatus(null);
    setTimeout(() => {
      const res = signIn({ id, pw, remember });
      if (!res.ok) {
        setLoading(false); setError(res.error); setErrStatus(res.status || null);
        if (!res.status) setFailCount((c) => c + 1);
        return;
      }
      setFailCount(0); onLogin && onLogin(res.user);
    }, 600);
  };

  const FULL = "통합관제 시스템";
  const done = typedText.length >= FULL.length;

  const gradTitle = isDark
    ? "linear-gradient(120deg,#7c74ff 0%,#c084fc 50%,#f0abfc 100%)"
    : "linear-gradient(120deg,#4f46e5 0%,#7c3aed 50%,#a855f7 100%)";

  return (
    <div style={{ position: "absolute", inset: 0, background: C.bg0, color: C.ink, overflow: "hidden", display: "flex", flexDirection: "column", transition: "background 300ms ease, color 300ms ease" }}>

      <style>{`
        @keyframes _ripple   { to { transform:scale(1); opacity:0 } }
        @keyframes _pulseDot { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.4;transform:scale(1.6)} }
        @keyframes _float    { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-5px)} }
        @keyframes _scanline { 0%{top:-4px} 100%{top:100vh} }
        @keyframes _gradShift{ 0%,100%{background-position:0% 50%} 50%{background-position:100% 50%} }
        @keyframes _blink    { 0%,100%{opacity:1} 50%{opacity:0} }
        @keyframes _fadeUp    { from{opacity:0;transform:translateY(14px)} to{opacity:1;transform:none} }
        @keyframes _logFadeIn { from{opacity:0} to{opacity:1} }
        @keyframes _glitch1  { 0%{clip-path:inset(8% 0 80% 0);transform:translate(-3px,1px)}
                               50%{clip-path:inset(55% 0 20% 0);transform:translate(3px,-1px)}
                               100%{clip-path:inset(30% 0 55% 0);transform:translate(-2px,0)} }
        @keyframes _glitch2  { 0%{clip-path:inset(70% 0 5% 0);transform:translate(3px,0)}
                               50%{clip-path:inset(15% 0 65% 0);transform:translate(-3px,1px)}
                               100%{clip-path:inset(45% 0 35% 0);transform:translate(2px,-1px)} }
        @keyframes spin      { to { transform:rotate(360deg) } }
        .card-glow::before {
          content:''; position:absolute; inset:0; border-radius:20px; pointer-events:none; z-index:0;
          background: radial-gradient(340px circle at var(--mx,50%) var(--my,50%), ${isDark ? "rgba(124,116,255,.18)" : "rgba(79,70,229,.10)"}, transparent 68%);
        }
        .card-3d {
          transition: transform 220ms cubic-bezier(.23,1,.32,1), box-shadow 220ms;
          transform-style: preserve-3d;
        }
      `}</style>

      {booting && <BootScreen onDone={() => setBooting(false)} C={C} isDark={isDark} />}

      <ParticleCanvas isDark={isDark} />
      <DataFlowLines isDark={isDark} />

      {/* gradient blobs */}
      <div aria-hidden style={{
        position: "absolute", inset: 0, zIndex: 0,
        background: isDark
          ? `radial-gradient(1200px 700px at 16% 22%, ${C.blobA}, transparent 62%),
             radial-gradient(900px 600px at 84% 78%, ${C.blobB}, transparent 60%),
             radial-gradient(600px 500px at 52% 95%, ${C.blobC}, transparent 55%),
             linear-gradient(180deg, ${C.bg0} 0%, ${C.bg1} 100%)`
          : `radial-gradient(900px 600px at 18% 20%, ${C.blobA}, transparent 55%),
             radial-gradient(700px 500px at 82% 78%, ${C.blobB}, transparent 50%),
             linear-gradient(160deg, #eef2ff 0%, ${C.bg0} 40%, #f5f0ff 100%)`,
      }} />

      {/* dot grid */}
      <div aria-hidden style={{
        position: "absolute", inset: 0, zIndex: 0, pointerEvents: "none",
        backgroundImage: `radial-gradient(circle,${isDark ? "rgba(124,116,255,.07)" : "rgba(79,70,229,.05)"} 1px,transparent 1px)`,
        backgroundSize: "32px 32px",
        maskImage: "radial-gradient(ellipse 85% 85% at 50% 50%, black 30%, transparent 100%)",
        WebkitMaskImage: "radial-gradient(ellipse 85% 85% at 50% 50%, black 30%, transparent 100%)",
      }} />

      {/* scanline */}
      <div aria-hidden style={{ position: "absolute", inset: 0, zIndex: 0, pointerEvents: "none", overflow: "hidden" }}>
        <div style={{
          position: "absolute", left: 0, right: 0, height: 3,
          background: `linear-gradient(transparent, ${isDark ? "rgba(124,116,255,.07)" : "rgba(79,70,229,.04)"}, transparent)`,
          animation: "_scanline 10s linear infinite",
        }} />
      </div>

      {/* ── Header ── */}
      <header style={{ position: "relative", zIndex: 1, padding: "30px 52px 0", display: "flex", alignItems: "center", justifyContent: "space-between", flexShrink: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%", background: C.brand, boxShadow: `0 0 14px ${C.brand}`, animation: "_pulseDot 2s ease-in-out infinite", display: "inline-block" }} />
          <span style={{ fontSize: 15, fontWeight: 700, letterSpacing: "-0.02em", color: C.ink }}>AI 기반 지능형 통합관제 시스템</span>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          {[{ label: "AI 정상", color: C.ok }, { label: "실시간 감시중", color: C.brand }].map((b) => (
            <div key={b.label} style={{
              display: "flex", alignItems: "center", gap: 6,
              padding: "5px 13px", borderRadius: 999,
              background: `${b.color}18`, border: `1px solid ${b.color}42`,
              fontSize: 11, fontWeight: 600, color: b.color,
            }}>
              <span style={{ width: 5, height: 5, borderRadius: "50%", background: b.color, boxShadow: `0 0 6px ${b.color}`, animation: "_pulseDot 2s ease-in-out infinite", display: "inline-block" }} />
              {b.label}
            </div>
          ))}
        </div>
      </header>

      {/* ── Main ── */}
      <main style={{
        position: "relative", zIndex: 1, flex: 1,
        display: "flex", alignItems: "center", justifyContent: "space-between",
        gap: 80, padding: "50px 120px 40px", minHeight: 0,
      }}>

        {/* 좌측 */}
        <div style={{ flex: "1 1 auto", maxWidth: 620, animation: "_fadeUp 650ms ease both" }}>

          <div style={{
            display: "inline-flex", alignItems: "center", gap: 8,
            padding: "6px 14px", borderRadius: 999,
            background: `${C.brand}14`, border: `1px solid ${C.brand}38`,
            color: C.brand, fontSize: 12, fontWeight: 600, letterSpacing: "-0.01em",
            marginBottom: 28, animation: "_float 4s ease-in-out infinite",
          }}>
            <span style={{ width: 5, height: 5, borderRadius: "50%", background: C.brand, animation: "_pulseDot 1.6s ease-in-out infinite", display: "inline-block" }} />
            호서대학교 시원팀 · 2026 캡스톤디자인
          </div>

          {/* 타이틀 + 글리치 */}
          <h1 style={{ margin: 0, fontSize: 58, fontWeight: 800, lineHeight: 1.06, letterSpacing: "-0.035em", color: C.ink }}>
            AI 기반 지능형
            <br />
            <span style={{ position: "relative", display: "inline-block" }}>
              <span style={{ background: gradTitle, backgroundSize: "200% 200%", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", animation: "_gradShift 4s ease-in-out infinite" }}>
                {typedText}
              </span>
              {glitch && (
                <>
                  <span aria-hidden style={{ position: "absolute", inset: 0, overflow: "hidden", background: "linear-gradient(120deg,#ff5555,#ff00cc)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", animation: "_glitch1 60ms steps(1) infinite", pointerEvents: "none" }}>{typedText}</span>
                  <span aria-hidden style={{ position: "absolute", inset: 0, overflow: "hidden", background: "linear-gradient(120deg,#00ffee,#0066ff)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", animation: "_glitch2 60ms steps(1) infinite", pointerEvents: "none" }}>{typedText}</span>
                </>
              )}
            </span>
            {!done && (
              <span style={{ display: "inline-block", width: 3, height: "0.85em", background: C.brand, marginLeft: 4, verticalAlign: "middle", animation: "_blink 0.75s ease-in-out infinite", borderRadius: 2 }} />
            )}
          </h1>

          <p style={{ margin: "20px 0 0", fontSize: 16, lineHeight: 1.7, color: C.inkSoft, fontWeight: 400, maxWidth: 440 }}>
            분산된 IoT 센서 데이터를 통합 관제,<br />
            AI가 이상을 사전에 포착합니다.
          </p>

          <div style={{ display: "flex", gap: 14, marginTop: 40 }}>
            {[
              { val: nodeCount,   unit: "개", label: "감시 노드",   color: C.brand  },
              { val: sensorCount, unit: "종", label: "센서 타입",   color: C.brand2 },
              { val: "24/7",      unit: "",   label: "실시간 수집", color: C.ok     },
            ].map((s) => (
              <div key={s.label}
                style={{
                  padding: "14px 20px", borderRadius: 14,
                  background: isDark ? "rgba(255,255,255,.03)" : "rgba(255,255,255,.6)",
                  border: `1px solid ${isDark ? "rgba(255,255,255,.07)" : "rgba(0,0,0,.07)"}`,
                  backdropFilter: "blur(10px)", minWidth: 88,
                  transition: "border-color 200ms, background 200ms, transform 200ms",
                  cursor: "default",
                  boxShadow: isDark ? "none" : "0 2px 8px rgba(0,0,0,.06)",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = `${s.color}55`;
                  e.currentTarget.style.background   = `${s.color}0d`;
                  e.currentTarget.style.transform    = "translateY(-3px)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = isDark ? "rgba(255,255,255,.07)" : "rgba(0,0,0,.07)";
                  e.currentTarget.style.background   = isDark ? "rgba(255,255,255,.03)" : "rgba(255,255,255,.6)";
                  e.currentTarget.style.transform    = "translateY(0)";
                }}
              >
                <div style={{ fontSize: 28, fontWeight: 800, letterSpacing: "-0.03em", color: s.color, lineHeight: 1, fontVariantNumeric: "tabular-nums" }}>
                  {s.val}{s.unit}
                </div>
                <div style={{ fontSize: 11, color: C.inkMuted, marginTop: 5, fontWeight: 500 }}>{s.label}</div>
              </div>
            ))}
          </div>

          <LiveLog C={C} isDark={isDark} />
        </div>

        {/* 우측 카드 */}
        <div
          ref={cardRef}
          className="card-glow card-3d"
          onMouseMove={onCardMove}
          onMouseLeave={onCardLeave}
          style={{
            position: "relative",
            flex: "0 0 420px", padding: 36, borderRadius: 20,
            background: C.glassBg,
            border: `1px solid ${C.glassBd}`,
            backdropFilter: "blur(28px) saturate(1.4)",
            WebkitBackdropFilter: "blur(28px) saturate(1.4)",
            boxShadow: isDark
              ? "0 40px 80px -30px rgba(0,0,0,.9), 0 0 0 1px rgba(255,255,255,.03) inset"
              : "0 20px 60px -20px rgba(79,70,229,.15), 0 4px 20px rgba(0,0,0,.08), 0 0 0 1px rgba(255,255,255,.9) inset",
            animation: "_fadeUp 720ms ease both", animationDelay: "90ms",
          }}
        >
          <div style={{
            position: "absolute", top: 0, left: "15%", right: "15%", height: 1,
            background: `linear-gradient(90deg,transparent,${C.brand}55,transparent)`,
            borderRadius: 1,
          }} />

          <div style={{ position: "relative", zIndex: 1 }}>
            <div style={{ marginBottom: 28 }}>
              <div style={{ fontSize: 24, fontWeight: 800, letterSpacing: "-0.025em", marginBottom: 6, color: C.ink }}>관제 시스템 접속</div>
              <div style={{ fontSize: 13, color: C.inkSoft }}>운영자 계정으로 로그인해 주세요.</div>
            </div>

            <form onSubmit={submit}>
              <Field icon={<Icons.user size={16} />} label="운영자 ID" focused={focus === "id"} filled={idFilled && focus !== "id"} C={C}>
                <input
                  value={id}
                  onChange={(e) => { setId(e.target.value); setIdFilled(false); }}
                  onFocus={() => setFocus("id")}
                  onBlur={() => { setFocus(null); setIdFilled(!!id); }}
                  style={inputStyle} placeholder="ID 를 입력하세요" autoComplete="username"
                />
              </Field>

              <Field icon={<Icons.lock size={16} />} label="비밀번호" focused={focus === "pw"} filled={pwFilled && focus !== "pw"} C={C}>
                <input
                  type="password" value={pw}
                  onChange={(e) => { setPw(e.target.value); setPwFilled(false); }}
                  onFocus={() => setFocus("pw")}
                  onBlur={() => { setFocus(null); setPwFilled(!!pw); }}
                  style={inputStyle} placeholder="비밀번호를 입력하세요" autoComplete="current-password"
                />
              </Field>

              <div style={{ display: "flex", alignItems: "center", marginBottom: 22, marginTop: 4 }}>
                <label style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer", fontSize: 13, color: C.inkSoft, userSelect: "none" }}
                  onClick={() => setRemember((v) => !v)}>
                  <span style={{
                    width: 16, height: 16, borderRadius: 5,
                    background: remember ? `linear-gradient(135deg,${C.brand},${C.brand2})` : "transparent",
                    border: `1px solid ${remember ? C.brand : C.inkFaint}`,
                    display: "grid", placeItems: "center",
                    transition: "all 160ms ease",
                  }}>
                    {remember && <Icons.check size={10} color="#fff" />}
                  </span>
                  로그인 상태 유지
                </label>
              </div>

              {error && (() => {
                const isPending  = errStatus === "pending";
                const isRejected = errStatus === "rejected";
                const tone = isPending
                  ? { bg: "rgba(245,158,11,.10)", bd: "rgba(245,158,11,.30)", fg: "#f59e0b", label: "승인 대기" }
                  : isRejected
                    ? { bg: "rgba(100,116,139,.10)", bd: "rgba(100,116,139,.30)", fg: "#64748b", label: "반려" }
                    : { bg: `${C.err}14`, bd: `${C.err}38`, fg: C.err, label: null };
                return (
                  <div style={{
                    fontSize: 12, color: tone.fg, marginBottom: 14,
                    padding: tone.label ? "10px 12px" : "8px 12px",
                    borderRadius: 8, background: tone.bg,
                    border: `1px solid ${tone.bd}`, lineHeight: 1.55,
                    animation: "_fadeUp 240ms ease both",
                  }}>
                    {tone.label && (
                      <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
                        <span style={{ width: 5, height: 5, borderRadius: "50%", background: tone.fg }} />
                        <span style={{ fontWeight: 700, letterSpacing: "0.02em" }}>{tone.label}</span>
                      </div>
                    )}
                    {error}
                  </div>
                );
              })()}

              <button
                ref={btnRef} type="submit" disabled={loading} onClick={doRipple}
                style={{
                  position: "relative", overflow: "hidden",
                  width: "100%", height: 50, borderRadius: 12, border: "none",
                  background: loading
                    ? `${C.brand}70`
                    : `linear-gradient(135deg,${C.brand} 0%,${C.brand2} 100%)`,
                  backgroundSize: "200% 200%",
                  color: "#fff", fontSize: 15, fontWeight: 700, letterSpacing: "-0.01em",
                  boxShadow: loading ? "none" : `0 12px 32px -8px ${C.brand}55, 0 0 0 1px rgba(255,255,255,.08) inset`,
                  display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
                  transition: "transform 140ms, box-shadow 180ms",
                  cursor: loading ? "not-allowed" : "pointer",
                  animation: loading ? "none" : "_gradShift 3s ease-in-out infinite",
                }}
                onMouseEnter={(e) => { if (!loading) e.currentTarget.style.transform = "translateY(-2px)"; }}
                onMouseLeave={(e) => { e.currentTarget.style.transform = "translateY(0)"; }}
                onMouseDown={(e)  => { if (!loading) e.currentTarget.style.transform = "translateY(1px)"; }}
              >
                {loading ? (
                  <>
                    <span style={{ width: 16, height: 16, borderRadius: "50%", border: "2px solid rgba(255,255,255,.3)", borderTopColor: "#fff", animation: "spin .7s linear infinite" }} />
                    인증 중...
                  </>
                ) : (
                  <>로그인<Icons.chevron_right size={16} /></>
                )}
              </button>

              <div style={{
                marginTop: 18, paddingTop: 18, borderTop: `1px solid ${C.inkFaint}`,
                display: "flex", justifyContent: "space-between", alignItems: "center",
                fontSize: 13, color: C.inkSoft,
              }}>
                <span style={{
                  fontSize: 12,
                  color: failCount >= 3 ? (isDark ? "#fbbf24" : "#d97706") : "transparent",
                  fontWeight: failCount >= 3 ? 600 : 400,
                  transition: "color 240ms", pointerEvents: "none", userSelect: "none",
                }}>
                  {failCount >= 3 ? "비밀번호는 관리자에게 문의" : "·"}
                </span>
                <a onClick={(e) => { e.preventDefault(); onSignUp && onSignUp(); }} href="#"
                  style={{ color: C.brand, fontWeight: 600, cursor: "pointer" }}>
                  회원가입
                </a>
              </div>
            </form>
          </div>
        </div>
      </main>

      {/* ── Footer ── */}
      <footer style={{
        position: "relative", zIndex: 1,
        padding: "0 56px 26px", textAlign: "center",
        fontSize: 12, color: C.inkMuted, letterSpacing: "-0.01em", flexShrink: 0,
      }}>
        © 2026 시원팀 · 호서대학교 컴퓨터공학부 캡스톤디자인
      </footer>
    </div>
  );
}
