import { useState, useEffect, useRef, useMemo, Fragment } from "react";
import { createPortal } from "react-dom";
import { Icons } from "../components/Icons.jsx";
import { MapPanel } from "../components/MapPanel.jsx";
import { devicesToMarkers } from "../api/client.js";
import { useWeather } from "../lib/weather.js";
import { useGuestbook, GuestbookList } from "../components/Guestbook.jsx";

const statusChip = (status) => {
  const map = {
    normal:   { ko: "정상", fg: "#047857", bg: "rgba(16,185,129,0.14)", bd: "rgba(16,185,129,0.3)" },
    critical: { ko: "이상", fg: "#fff",     bg: "#dc2626",                bd: "#991b1b" },
    anomaly:  { ko: "이상", fg: "#b91c1c", bg: "rgba(239,68,68,0.12)",   bd: "rgba(239,68,68,0.3)" },
    warn:     { ko: "관찰", fg: "#b45309", bg: "rgba(245,158,11,0.14)",  bd: "rgba(245,158,11,0.3)" },
    offline:  { ko: "통신 장애", fg: "#475569", bg: "rgba(100,116,139,0.14)", bd: "rgba(100,116,139,0.3)" },
  };
  return map[status] || map.normal;
};

// 시간 문자열을 한국어 12시간제("오후 7:42")로 정규화.
//  - 이미 "오전/오후 ..." 형식이면 그대로 통과 (신규 메시지)
//  - 옛 "HH:MM" / "HH:MM:SS" 24시간제 저장값이면 변환 (기존 localStorage/DB 메시지)
//  - 파싱 불가하면 원본 반환.
function to12h(t) {
  if (!t || typeof t !== "string") return t;
  if (t.includes("오전") || t.includes("오후")) return t; // 이미 오전/오후
  const m = t.match(/^(\d{1,2}):(\d{2})/);
  if (!m) return t;
  const h = parseInt(m[1], 10);
  if (isNaN(h) || h > 23) return t;
  const ampm = h < 12 ? "오전" : "오후";
  let h12 = h % 12; if (h12 === 0) h12 = 12;
  return `${ampm} ${h12}:${m[2]}`;
}

function Kpi({ label, value, accent, icon, delta, active, onClick, danger }) {
  // '이상'(danger=true) 카드는 0건이 아닐 때 정적 빨강 그림자로만 강조 (깜빡임 펄스는 제거).
  // 모든 카드는 자신의 status accent 색을 숫자에 적용 → 시각적 일관성.
  const alarming = danger && value > 0;
  const valueFg  = value > 0 ? accent : "var(--ink-3)";
  const iconCol  = accent;
  return (
    <button
      onClick={onClick}
      style={{
        position: "relative", flex: 1, minWidth: 0, height: "var(--dash-kpi-row)", textAlign: "left",
        background: "var(--bg-elev)", borderRadius: 16,
        border: `1px solid ${active ? accent : "var(--line)"}`,
        padding: "var(--dash-kpi-pad)",
        boxShadow: alarming
          ? `0 0 0 1px rgba(220,38,38,0.18), 0 6px 16px -6px rgba(220,38,38,0.25), var(--shadow-card)`
          : (active ? `0 0 0 3px ${accent}22, var(--shadow-card)` : "var(--shadow-card)"),
        cursor: "pointer", transition: "all 180ms",
        overflow: "hidden",
      }}
    >
      <div style={{
        position: "absolute", left: 0, top: 0, bottom: 0,
        width: alarming ? 5 : 4,
        background: accent,
      }} />
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6 }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: "var(--ink-3)", letterSpacing: "0.02em" }}>{label}</div>
        <div style={{ color: iconCol, opacity: 0.95 }}>{icon}</div>
      </div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
        <div className="num" style={{ fontSize: 44, fontWeight: 700, lineHeight: 1, color: valueFg }}>{value}</div>
        <div style={{ fontSize: 13, color: "var(--ink-3)", fontWeight: 500 }}>대</div>
        {delta && (
          <div className="mono" style={{
            marginLeft: "auto", fontSize: 11, fontWeight: 700,
            color: delta.startsWith("+") ? "var(--err)" : "var(--ok)",
          }}>{delta}</div>
        )}
      </div>
    </button>
  );
}

function KPIRow({ active, setActive, counts }) {
  const items = [
    { k: "all",      label: "총 장비",   value: counts.all,      accent: "var(--brand)", icon: <Icons.box size={18} /> },
    { k: "normal",   label: "정상",      value: counts.normal,   accent: "var(--ok)",    icon: <Icons.check size={18} /> },
    { k: "warn",     label: "관찰",      value: counts.warn,     accent: "var(--warn)",  icon: <Icons.eye size={18} /> },
    { k: "critical", label: "이상",      value: counts.critical, accent: "#dc2626",      icon: <Icons.alert size={18} />, danger: true },
    { k: "offline",  label: "통신 장애", value: counts.offline,  accent: "var(--ink-3)", icon: <Icons.wifi_off size={18} /> },
  ];
  return (
    <div style={{ display: "flex", gap: "var(--dash-kpi-gap)", height: "100%" }}>
      {items.map((i) => (
        <Kpi key={i.k} {...i} active={active === i.k} onClick={() => setActive(i.k === active ? null : i.k)} />
      ))}
    </div>
  );
}

function PanelHeader({ children, right, pad = "16px 20px" }) {
  return (
    <div style={{
      display: "flex", alignItems: "center", justifyContent: "space-between",
      gap: 12,    // children 과 right 가 붙지 않게 (LogPanel 필터 칩 ↔ ← AI 탐지로 간격)
      flexWrap: "wrap",
      padding: pad, borderBottom: "1px solid var(--line-soft)",
    }}>
      {children}
      {right}
    </div>
  );
}

function Panel({ children, style, className }) {
  return (
    <div
      className={className}
      style={{
        background: "var(--bg-elev)", borderRadius: 16,
        border: "1px solid var(--line)",
        boxShadow: "var(--shadow-card)",
        overflow: "hidden", ...style,
      }}
    >
      {children}
    </div>
  );
}

// 세그먼트 토글 공통 — 활성 칸을 실측(offsetLeft/Width)해 흰 알약을 밀착 + 전환 시 슬라이드(스프링)·젤리(빨려들어갔다 튀어나옴)
// items: [{ label, dot? }], activeIdx: 현재 칸, onSelect(i, item): 다른 칸 클릭 시
function SegmentedToggle({ items, activeIdx, onSelect, pad = "4px 12px" }) {
  const segRefs = useRef([]);
  const [box, setBox] = useState(null);
  useEffect(() => {
    const measure = () => { const el = segRefs.current[activeIdx]; if (el) setBox({ left: el.offsetLeft, width: el.offsetWidth }); };
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [activeIdx, items.length]);
  return (
    <div style={{
      position: "relative", display: "inline-flex", flexShrink: 0,
      padding: 3, borderRadius: 999,
      background: "var(--bg-sunk)", border: "1px solid var(--line)",
    }}>
      {box && (
        <div aria-hidden style={{
          position: "absolute", top: 3, bottom: 3, zIndex: 0,
          left: box.left, width: box.width,
          transition: "left 440ms cubic-bezier(0.34, 1.56, 0.64, 1), width 440ms cubic-bezier(0.34, 1.56, 0.64, 1)",
        }}>
          {/* 안쪽 젤리 — key 변경으로 칸 전환마다 재생(빨려들어갔다 튀어나옴) */}
          <div key={activeIdx} className="pill-jelly" style={{
            width: "100%", height: "100%", borderRadius: 999,
            background: "var(--bg-elev)", boxShadow: "var(--shadow-card)",
          }} />
        </div>
      )}
      {items.map((it, i) => {
        const active = i === activeIdx;
        return (
          <button key={it.label} ref={(el) => (segRefs.current[i] = el)} type="button" title={it.label}
            onClick={() => { if (!active && onSelect) onSelect(i, it); }}
            style={{
              position: "relative", zIndex: 1,
              display: "inline-flex", alignItems: "center", gap: 5,
              padding: pad, borderRadius: 999, border: "none", background: "transparent",
              color: active ? "var(--ink)" : "var(--ink-3)",
              fontSize: 12, fontWeight: 700,
              cursor: active ? "default" : "pointer", whiteSpace: "nowrap",
              transition: "color 200ms ease",
            }}
            onMouseEnter={(e) => { if (!active) e.currentTarget.style.color = "var(--ink)"; }}
            onMouseLeave={(e) => { if (!active) e.currentTarget.style.color = "var(--ink-3)"; }}
          >
            {it.dot && <span style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--ok)", animation: "pulse-dot 1.2s infinite", flexShrink: 0 }} />}
            {it.label}
          </button>
        );
      })}
    </div>
  );
}

// 전체 장비 현황 요약 ⇄ 시스템 로그 세그먼트 토글 (5/30 — 두 뷰를 한 토글로 스위칭)
function PanelViewToggle({ logOpen, onToggleLog }) {
  return (
    <SegmentedToggle
      items={[{ label: "장비 현황 요약" }, { label: "시스템 로그" }]}
      activeIdx={logOpen ? 1 : 0}
      onSelect={() => { if (onToggleLog) onToggleLog(); }}
    />
  );
}

// 시간(h) → "N일 M시간" 변환 (Gemini 5/26 — 687h 같은 큰 숫자 직관성 부족)
function fmtHoursShort(h) {
  if (h == null || !isFinite(h)) return "—";
  const n = Math.floor(Number(h));
  if (n < 24) return `${n}시간`;
  const days = Math.floor(n / 24);
  const hours = n % 24;
  return hours === 0 ? `${days}일` : `${days}일 ${hours}시간`;
}

function formatMse(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return "—";
  if (Math.abs(n) >= 0.01) return n.toFixed(4);
  return n.toFixed(6);
}

function formatAiRatio(item) {
  const ratio = Number.isFinite(Number(item.aiRatio))
    ? Number(item.aiRatio)
    : item.mse != null && item.threshold > 0
      ? Number(item.mse) / Number(item.threshold)
      : null;
  if (!Number.isFinite(ratio)) return null;
  return `${Math.round(ratio * 100)}%`;
}

function featureLabel(name) {
  return String(name || "")
    .replace(/_dev24$/u, " 편차")
    .replace(/_diff1$/u, " 변화")
    .replace(/_/gu, " ");
}

function anomalyLabel(label) {
  return String(label || "")
    .replace(/_dev24(?= |$)/gu, " 편차")
    .replace(/_diff1(?= |$)/gu, " 변화")
    .replace(/_/gu, " ");
}

// (actionFor 제거됨 — AI 탐지 카드 '조치' 줄 미표시, 2026-05-30 요청)

// 이두현 3구간 상태 미터 (Threshold Indicator Bar). 정상 0~70% · 관찰 70~100% · 이상 100%↑.
// 구간을 1/3 폭으로 등분(범주 가독성 우선) + ▲ 현재 포인터. 이상은 무한대라 100~250%를 마지막 칸에 매핑 후 우측 고정.
function RatioGauge({ ratio, compact = false }) {
  const r = Number(ratio);
  if (!Number.isFinite(r)) return null;
  const pct = Math.round(r * 100);
  let pos;
  if (pct <= 70)       pos = (pct / 70) * 33.33;
  else if (pct <= 100) pos = 33.33 + ((pct - 70) / 30) * 33.33;
  else                 pos = 66.66 + Math.min((pct - 100) / 150, 1) * 33.34;
  pos = Math.max(0, Math.min(97, pos));            // 좌측 끝(0)~우측(97). 위치·색은 부모(AiAnalysis)가 카운트업으로 구동.
  const isN = pct < 70, isW = pct >= 70 && pct < 100, isC = pct >= 100;
  const col = isC ? "var(--err)" : isW ? "var(--warn)" : "var(--ok)";
  return (
    <div style={{ margin: compact ? 0 : "3px 0 8px" }}>
      {/* 구간 라벨 (현재 구간만 색 강조) */}
      <div style={{ display: "flex", fontSize: 9, fontWeight: 700, color: "var(--ink-4)" }}>
        <span style={{ width: "33.33%", textAlign: "center", color: isN ? "var(--ok)" : undefined,   fontWeight: isN ? 800 : 700 }}>정상</span>
        <span style={{ width: "33.33%", textAlign: "center", color: isW ? "var(--warn)" : undefined, fontWeight: isW ? 800 : 700 }}>관찰</span>
        <span style={{ width: "33.34%", textAlign: "center", color: isC ? "var(--err)" : undefined,  fontWeight: isC ? 800 : 700 }}>이상</span>
      </div>
      {/* 3색 구간 바 + 경계 눈금 */}
      <div style={{ position: "relative", height: 8, borderRadius: 999, overflow: "hidden", display: "flex", marginTop: 2 }}>
        <div style={{ width: "33.33%", background: "var(--ok)" }} />
        <div style={{ width: "33.33%", background: "var(--warn)" }} />
        <div style={{ width: "33.34%", background: "var(--err)" }} />
        <div style={{ position: "absolute", left: "33.33%", top: 0, bottom: 0, width: 2, background: "var(--bg-elev)" }} />
        <div style={{ position: "absolute", left: "66.66%", top: 0, bottom: 0, width: 2, background: "var(--bg-elev)" }} />
      </div>
      {/* ▲ 현재 포인터 (compact: % 텍스트 생략 — 정보 줄에 이미 표시) */}
      <div style={{ position: "relative", height: compact ? 10 : 20 }}>
        <div style={{ position: "absolute", left: `${pos}%`, transform: "translateX(-50%)", textAlign: "center", whiteSpace: "nowrap" }}>
          <div style={{ fontSize: 7, color: col, lineHeight: "8px", "--glow": `${(2 + Math.min(pct / 500, 1) * 14).toFixed(1)}px`, animation: "arrow-pulse 1.6s ease-in-out infinite" }}>▲</div>
          {!compact && <div style={{ fontSize: 9, fontWeight: 800, color: col, lineHeight: "9px", marginTop: 3 }}>{pct}%</div>}
        </div>
      </div>
    </div>
  );
}

// AI 분석 박스 — 단말 선택 시 0 → 실제값 카운트업. % 숫자·게이지 화살표·구간색이 함께 움직임.
function AiAnalysis({ item }) {
  const mse   = item.aiMse ?? item.mse;
  const th    = item.aiThreshold ?? item.threshold;
  const ratio = item.aiRatio ?? (th > 0 ? mse / th : null);
  const targetPct = ratio != null ? Math.round(Number(ratio) * 100) : null;

  const [animPct, setAnimPct] = useState(0);
  useEffect(() => {
    if (targetPct == null) { setAnimPct(0); return; }
    setAnimPct(0);
    let raf, start = null;
    const DUR = 1500;                                  // 카운트업 시간 (ms)
    const ease = (t) => 1 - Math.pow(1 - t, 3);        // ease-out-cubic
    const step = (ts) => {
      if (start == null) start = ts;
      const t = Math.min((ts - start) / DUR, 1);
      setAnimPct(targetPct * ease(t));
      if (t < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [targetPct, item.deviceId]);

  if (mse == null) return null;
  const shown = Math.round(animPct);
  const col = shown >= 100 ? "var(--err)" : shown >= 70 ? "var(--warn)" : "var(--ok)";

  return (
    <div style={{ marginBottom: 10, padding: "9px 14px 4px", borderRadius: 10, background: "var(--bg-sunk)", border: "1px solid var(--line-soft)" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", columnGap: 18, rowGap: 4 }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: "var(--ink-3)" }}>AI 분석</div>
        <span className="mono" style={{ fontSize: 15, fontWeight: 700, color: col }}>{targetPct != null ? `${shown}%` : "-"}</span>
      </div>
      {targetPct != null && (
        <div style={{ marginTop: 4 }}>
          <RatioGauge ratio={animPct / 100} compact />
        </div>
      )}
    </div>
  );
}

// 기여도 칩 한 줄 — 넘치면 호버 시 천천히 왼쪽으로 스크롤(마퀴) 후 복귀.
//   wrap 으로 2줄 되던 걸 nowrap 고정. JS 로 overflow 양 측정해 정확히 그만큼만 이동.
function ContribChips({ contribution, color }) {
  const viewRef = useRef(null);
  const trackRef = useRef(null);
  const [shift, setShift] = useState(0);     // 호버 시 이동할 px (overflow 양)
  const [hover, setHover] = useState(false);

  const measure = () => {
    const v = viewRef.current, t = trackRef.current;
    if (!v || !t) return;
    setShift(Math.max(0, t.scrollWidth - v.clientWidth));
  };
  useEffect(() => { measure(); }, [contribution]);

  // 이동 거리에 비례한 시간(천천히): 40px/s, 최소 0.6s, 최대 6s
  const dur = shift > 0 ? Math.min(6, Math.max(0.6, shift / 40)) : 0;

  return (
    <div
      ref={viewRef}
      onMouseEnter={() => { measure(); setHover(true); }}
      onMouseLeave={() => setHover(false)}
      style={{ overflow: "hidden", width: "100%" }}
    >
      <div
        ref={trackRef}
        style={{
          display: "flex", gap: 4, width: "max-content",
          transform: hover && shift > 0 ? `translateX(-${shift}px)` : "translateX(0)",
          // 호버: 천천히 스크롤(dur, linear) / 떼면: 즉각 복귀(0.1s, ease-out)
          transition: hover && shift > 0
            ? `transform ${dur}s linear`
            : "transform 0.1s ease-out",
        }}
      >
        {contribution.map((c, i) => (
          <span
            key={i}
            style={{
              fontSize: 9, fontWeight: 600,
              padding: "2px 6px", borderRadius: 4,
              background: i === 0
                ? color.replace("var(--err)", "rgba(239,68,68,0.15)").replace("var(--warn)", "rgba(245,158,11,0.15)")
                : "var(--bg-elev)",
              color: i === 0 ? color : "var(--ink-3)",
              border: `1px solid ${i === 0 ? color : "var(--line)"}`,
              whiteSpace: "nowrap", flexShrink: 0,
            }}
          >
            {featureLabel(c.sensor)} {c.pct}%
          </span>
        ))}
      </div>
    </div>
  );
}

function AnomalyCard({ item, onClick, kind, highlighted = false }) {
  const color = kind === "offline" ? "var(--ink-3)" : kind === "warn" ? "var(--warn)" : "var(--err)";
  // 통신 두절(offline) 카드는 label 이 "통신 두절..." 로 시작 → 우측 박스 = 두절 일/시간
  // 일반 anomaly 는 threshold 대비 배수 표시. 퍼센트는 극단값에서 너무 과장되어 보임.
  const isOffline = typeof item.label === "string" && item.label.startsWith("통신 두절");
  const ratioText = isOffline ? null : formatAiRatio(item);
  return (
    <div
      onClick={() => onClick(item)}
      role="button"
      tabIndex={0}
      style={{
        display: "block", width: "100%", textAlign: "left",
        padding: "10px 12px",
        background: highlighted ? "var(--brand-wash)" : "var(--bg-sunk)",
        border: highlighted ? "1px solid var(--brand-wash)" : "1px solid var(--line-soft)",
        borderRadius: 10, marginBottom: 6,
        boxShadow: highlighted ? "0 0 0 3px var(--brand-wash)" : "none",
        transition: "all 200ms", cursor: "pointer",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 10, marginBottom: 8 }}>
        <div style={{ minWidth: 0, flex: 1, display: "flex", alignItems: "baseline", gap: 8 }}>
          <span className="mono" style={{ fontSize: 11, fontWeight: 700, color: "var(--ink)", whiteSpace: "nowrap", flexShrink: 0 }}>{item.node}</span>
          <span style={{
            fontSize: 11, color, fontWeight: 600,
            whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", minWidth: 0,
          }}>
            {anomalyLabel(item.label)}
          </span>
        </div>
        <div style={{ textAlign: "right", flexShrink: 0 }}>
          {isOffline ? (
            <>
              <div className="mono" style={{ fontSize: 9, color: "var(--ink-4)", letterSpacing: "0.05em" }}>두절</div>
              <div style={{ fontSize: 13, fontWeight: 700, color, lineHeight: 1, whiteSpace: "nowrap" }}>{fmtHoursShort(item.mse)}</div>
            </>
          ) : ratioText != null ? (
            <>
              <div className="mono" style={{ fontSize: 16, fontWeight: 700, color, lineHeight: 1 }}>{ratioText}</div>
            </>
          ) : (
            <>
              <div className="mono" style={{ fontSize: 9, color: "var(--ink-4)", letterSpacing: "0.05em" }}>MSE</div>
              <div className="mono" style={{ fontSize: 14, fontWeight: 700, color, lineHeight: 1 }}>{formatMse(item.mse)}</div>
            </>
          )}
        </div>
      </div>
      {item.contribution && item.contribution.length > 0 && (
        <ContribChips contribution={item.contribution} color={color} />
      )}
    </div>
  );
}

function AIPanels({ onAnalyze, anomalies, watch, commOutage = [], focusNode }) {
  const listRef  = useRef(null);
  const cardRefs = useRef({});
  const [flash, setFlash] = useState(null);
  // 이상 + 관찰 + 통신두절 통합 리스트. 심각도 우선, 같은 그룹은 threshold 대비 배수 내림차순.
  const combined = [
    ...anomalies.map((a) => ({ ...a, _kind: "anomaly" })),
    ...watch.map((w) => ({ ...w, _kind: "warn" })),
    ...commOutage.map((o) => ({ ...o, mse: o.hoursSilent, _kind: "offline" })),
  ].sort((a, b) => {
    const rank = (k) => k === "anomaly" ? 3 : k === "warn" ? 2 : 1;
    const kindOrder = rank(b._kind) - rank(a._kind);
    if (kindOrder !== 0) return kindOrder;
    const ar = Number.isFinite(Number(a.aiRatio)) ? Number(a.aiRatio) : (a.threshold > 0 ? a.mse / a.threshold : 0);
    const br = Number.isFinite(Number(b.aiRatio)) ? Number(b.aiRatio) : (b.threshold > 0 ? b.mse / b.threshold : 0);
    return br - ar;
  });

  // 지도 사이드바에서 단말 선택 시 → 해당 카드로 스크롤 + 잠깐 강조 (목록에 없는 정상 단말이면 무시)
  useEffect(() => {
    if (!focusNode) return;
    const el = cardRefs.current[focusNode];
    const c  = listRef.current;
    if (!el || !c) return;
    const top = el.getBoundingClientRect().top - c.getBoundingClientRect().top + c.scrollTop - 8;
    c.scrollTo({ top: Math.max(0, top), behavior: "smooth" });
    setFlash(focusNode);
    const t = setTimeout(() => setFlash(null), 300);
    return () => clearTimeout(t);
  }, [focusNode]);

  return (
    <Panel style={{ height: "100%", display: "flex", flexDirection: "column", minHeight: 0 }}>
      <PanelHeader pad="8px 18px">
        <div style={{ display: "flex", alignItems: "center", gap: 10, minHeight: 30 }}>
          {/* 항목은 하나지만 옆 패널(장비 현황 요약 | 시스템 로그) 세그먼트 토글과 디자인 통일 */}
          <SegmentedToggle items={[{ label: "AI 탐지 목록" }]} activeIdx={0} />
        </div>
      </PanelHeader>
      <div ref={listRef} className="scroll" style={{ padding: 12, flex: 1, overflowY: "auto", minHeight: 0 }}>
        {combined.map((a) => (
          <div key={a.node} ref={(el) => { cardRefs.current[a.node] = el; }}>
            <AnomalyCard item={a} kind={a._kind} onClick={onAnalyze} highlighted={flash === a.node} />
          </div>
        ))}
      </div>
    </Panel>
  );
}

// 핵심 수치·장비 자동 강조 (TB24-XXXXXX, NN%, N시간, 제N구역 등)
function highlightBody(text) {
  const pattern = /(TB24-[\w-]+|\d+\.\d{3,4}|\d+\.?\d*%|\d+시간|\d+분|제\d+구역|\d+mV|\d+mA|\d+dBm)/g;
  const parts = text.split(pattern);
  return parts.map((part, i) =>
    pattern.test(part) ? (
      <strong key={i} style={{ color: "var(--ink)", fontWeight: 700 }}>{part}</strong>
    ) : (
      <span key={i}>{part}</span>
    )
  );
}


function Metric({ label, value, color, mono }) {
  return (
    <div>
      <div style={{ fontSize: 10, color: "var(--ink-4)", fontWeight: 500 }}>{label}</div>
      <div className={mono ? "mono" : ""} style={{ fontSize: 12, fontWeight: 700, color: color || "var(--ink)" }}>{value}</div>
    </div>
  );
}

function MarkerPopup({ m, onClose }) {
  const color = m.status === "anomaly" ? "var(--err)" : "var(--warn)";
  return (
    <div style={{
      position: "absolute",
      left: `${m.x * 100}%`, top: `${m.y * 100}%`,
      transform: "translate(-50%, calc(-100% - 50px))",
      background: "var(--bg-elev)",
      border: "1px solid var(--line-soft)",
      borderRadius: 12,
      boxShadow: "0 20px 40px -10px rgba(0,0,0,0.25)",
      padding: 14, minWidth: 240,
      animation: "slide-in-up 220ms ease",
      zIndex: 10,
      color: "var(--ink)",
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "start", marginBottom: 8 }}>
        <div>
          <div className="mono" style={{ fontSize: 13, fontWeight: 700, color: "var(--ink)" }}>{m.node}</div>
          <div style={{ fontSize: 11, color, fontWeight: 600, marginTop: 2 }}>{m.label}</div>
        </div>
        <button onClick={onClose} style={{ color: "var(--ink-3)" }}><Icons.close size={14} /></button>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10, marginTop: 8 }}>
        <Metric label="MSE" value={formatMse(m.mse)} color={color} />
        <Metric label="구역" value={m.zone} />
        <Metric label="상태" value={m.status === "critical" ? "이상" : "관찰"} color={color} />
      </div>
      <div style={{
        marginTop: 10, padding: "8px 10px", borderRadius: 8,
        background: "var(--brand-wash)", color: "var(--brand)",
        fontSize: 11, fontWeight: 600, display: "flex", alignItems: "center", gap: 6,
      }}>
        <Icons.sparkle size={12} />
        AI 이상 스코어 적용 · 실시간 분석 중
      </div>
      <div style={{
        position: "absolute", left: "50%", bottom: -6,
        transform: "translateX(-50%) rotate(45deg)",
        width: 12, height: 12, background: "var(--bg-elev)",
        borderRight: "1px solid var(--line-soft)",
        borderBottom: "1px solid var(--line-soft)",
      }} />
    </div>
  );
}

function MapPanelWrap({ markers, onMarker, mapStyle, setMapStyle, focus, fitTrigger, boundsRequest, showNormal, setShowNormal, autoKpiSec = 0, onCancelAutoKpi, onMapClick, deselectTrigger }) {
  return (
    <Panel style={{ position: "relative", height: "100%", isolation: "isolate" }}>
      <MapPanel markers={markers} onMarker={onMarker} mapStyle={mapStyle} focus={focus} fitTrigger={fitTrigger} boundsRequest={boundsRequest} onMapClick={onMapClick} deselectTrigger={deselectTrigger} />

      {/* Legend */}
      <div className="glass-soft" style={{
        position: "absolute", left: 16, top: 16, zIndex: 1000,
        border: "1px solid var(--line-soft)", borderRadius: 10,
        padding: "10px 14px",
        boxShadow: "0 8px 24px -10px rgba(0,0,0,0.2)",
        color: "var(--ink)",
        pointerEvents: "none",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--ok)", animation: "pulse-dot 1.2s infinite" }} />
          <div style={{ fontSize: 12, fontWeight: 800 }}>GIS 관제</div>
        </div>
        <div style={{ display: "flex", gap: 12, fontSize: 10, color: "var(--ink-3)" }}>
          <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#10b981" }} />정상
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <span style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--warn)" }} />관찰
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#991b1b" }} />이상
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#64748b" }} />통신 장애
          </span>
        </div>
      </div>



      {/* 지도 스타일 스위처는 Header 설정 아이콘 드롭다운으로 이동 (2026-05-04) */}
    </Panel>
  );
}

function MiniTable({ data, onRowClick }) {
  const [hoverId, setHoverId] = useState(null);

  return (
    <div className="scroll" style={{ overflow: "auto", height: "100%" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13, tableLayout: "fixed" }}>
        <colgroup>
          <col style={{ width: 130 }} />
          <col style={{ width: 130 }} />
          <col />
          <col style={{ width: 96 }} />
        </colgroup>
        <thead>
          <tr style={{ position: "sticky", top: 0, background: "var(--bg-elev)", zIndex: 1 }}>
            {["시설명", "장비명", "설치위치", "상태"].map((h, i) => {
              const align = i === 3 ? "center" : "left";
              return (
                <th
                  key={h}
                  style={{
                    textAlign: align,
                    padding: "8px 16px",
                    fontWeight: 700, fontSize: 12, color: "var(--ink-3)",
                    borderBottom: "1px solid var(--line)",
                    background: "var(--bg-elev)",
                    whiteSpace: "nowrap",
                    letterSpacing: "-0.01em",
                  }}
                >
                  {h}
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {data.map((r, idx) => {
            const c = statusChip(r.status);
            const isHover = hoverId === r.id;
            const isZebra = idx % 2 === 1;
            return (
              <tr
                key={r.id}
                onClick={() => onRowClick && onRowClick(r)}
                onMouseEnter={() => setHoverId(r.id)}
                onMouseLeave={() => setHoverId(null)}
                style={{
                  borderBottom: "1px solid var(--line-soft)",
                  cursor: "pointer",
                  background: isHover
                    ? "var(--brand-wash)"
                    : isZebra
                    ? "var(--bg-sunk)"
                    : "transparent",
                  transition: "background 120ms",
                }}
              >
                <td className="mono" style={{
                  padding: "11px 16px", fontWeight: 700,
                  color: "var(--ink)", letterSpacing: "-0.01em",
                }}>
                  {r.facilityId}
                </td>
                <td className="mono" style={{
                  padding: "11px 16px", color: "var(--ink-2)",
                  fontWeight: 500,
                }}>
                  {r.deviceId}
                </td>
                <td style={{
                  padding: "11px 16px", color: "var(--ink-2)",
                  overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                }}>
                  {r.location}
                </td>
                <td style={{ padding: "11px 16px", textAlign: "center" }}>
                  {/* 필 박스 제거 — 글자색만. KPI 카드 색과 일치 (정상=var(--ok) · 관찰=var(--warn) · 이상=#dc2626) */}
                  <span style={{
                    fontSize: 13, fontWeight: 700, whiteSpace: "nowrap",
                    color: ({ "정상": "var(--ok)", "관찰": "var(--warn)", "이상": "#dc2626" })[c.ko] || (c.fg === "#fff" ? c.bg : c.fg),
                  }}>
                    {c.ko}
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// KPI 활성 상태별 헤더 라벨/색상
const TABLE_HEADER_BY_KPI = {
  all:      { ko: "전체",      bar: "var(--brand)", chipBg: "var(--brand-wash)",         chipFg: "var(--brand)" },
  normal:   { ko: "정상",      bar: "#10b981",       chipBg: "rgba(16,185,129,0.12)",     chipFg: "#047857"      },
  critical: { ko: "이상",      bar: "#dc2626",       chipBg: "rgba(220,38,38,0.12)",      chipFg: "#991b1b"      },
  warn:     { ko: "관찰",      bar: "var(--warn)",   chipBg: "rgba(245,158,11,0.14)",     chipFg: "#b45309"      },
  offline:  { ko: "통신 장애", bar: "#64748b",       chipBg: "rgba(100,116,139,0.14)",    chipFg: "#475569"      },
};

function TableSummary({ data, onRowClick, activeKpi, logOpen, onToggleLog }) {
  const [query, setQuery] = useState("");
  const filtered = data.filter(
    (r) =>
      !query ||
      r.facilityId.toLowerCase().includes(query.toLowerCase()) ||
      r.deviceId.toLowerCase().includes(query.toLowerCase()) ||
      r.location.includes(query)
  );
  const meta = TABLE_HEADER_BY_KPI[activeKpi] || TABLE_HEADER_BY_KPI.all;
  return (
    <Panel style={{ height: "100%", display: "flex", flexDirection: "column" }}>
      <PanelHeader
        pad="8px 18px"
        right={
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            {/* 시스템 로그 토글은 헤더 좌측 세그먼트 컨트롤(PanelViewToggle)로 이동 (5/30) */}
            <div style={{
              display: "flex", alignItems: "center", gap: 8,
              padding: "5px 12px", borderRadius: 10,
              background: "var(--bg-sunk)", border: "1px solid var(--line)",
              width: "clamp(160px, 16vw, 240px)",
            }}>
              <Icons.search size={13} color="var(--ink-4)" />
              <input
                placeholder="장비 검색..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                style={{
                  flex: 1, border: "none", outline: "none", background: "transparent",
                  fontSize: 12,
                }}
              />
            </div>
          </div>
        }
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          {onToggleLog
            ? <PanelViewToggle logOpen={logOpen} onToggleLog={onToggleLog} />
            : <div style={{ fontSize: 14, fontWeight: 700 }}>{meta.ko} 장비 현황 요약</div>}
          <span style={{
            fontSize: 10, fontWeight: 700, padding: "2px 8px",
            background: meta.chipBg, color: meta.chipFg, borderRadius: 999,
            transition: "background 200ms ease, color 200ms ease",
          }}>
            {filtered.length}개
          </span>
        </div>
      </PanelHeader>
      <div style={{ flex: 1, overflow: "hidden" }}>
        <MiniTable data={filtered} onRowClick={onRowClick} />
      </div>
    </Panel>
  );
}

function LogLine({ line }) {
  const colors = {
    ok: "var(--ok)",
    data: "var(--ink-2)",
    alert: "var(--err)",
    ai: "var(--brand)",
    auth: "var(--ink-3)",
    warn: "var(--warn)",
  };
  return (
    <div
      className="mono"
      style={{
        padding: "4px 10px", marginBottom: 3,
        fontSize: 11, display: "flex", gap: 10, alignItems: "center",
        animation: "slide-in-up 220ms ease",
      }}
    >
      <span style={{ color: colors[line.kind] || "var(--ink-3)", fontWeight: 700, flexShrink: 0 }}>
        [{line.time}]
      </span>
      <span style={{
        flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
        color: line.kind === "alert" || line.kind === "warn" ? colors[line.kind] : "var(--ink-2)",
        fontWeight: line.kind === "alert" ? 700 : 400,
      }}>
        {line.text}
      </span>
      {line.tail && (
        <span style={{ color: colors[line.kind] || "var(--ok)", fontWeight: 700, flexShrink: 0 }}>
          {line.tail}
        </span>
      )}
    </div>
  );
}

// kind 필터 칩 정의 — LogLine 의 색과 일치
const LOG_KINDS = [
  { k: "ok",    label: "정상",  color: "var(--ok)"    },
  { k: "ai",    label: "AI",   color: "var(--brand)" },
  { k: "alert", label: "이상",  color: "var(--err)"   },
  { k: "warn",  label: "경고",  color: "var(--warn)"  },
];

function LogPanel({ lines, onToggleLog }) {
  const [query, setQuery] = useState("");
  const [hiddenKinds, setHiddenKinds] = useState(() => new Set());
  // kind 별 카운트 (모든 lines 기준 — 필터 적용 전)
  const kindCounts = useMemo(() => {
    const c = { ok: 0, ai: 0, alert: 0, warn: 0 };
    for (const l of lines) { if (c[l.kind] !== undefined) c[l.kind]++; }
    return c;
  }, [lines]);
  // kind + search 필터 + 전역 시간정렬(최신이 위)
  // lines 는 외부 가상이벤트와 polling 데이터가 append 된 순서라 정렬이 안 됨 →
  // 표시 직전 ts 내림차순으로 통합 정렬해야 가상이벤트가 실 DB 로그를 가리지 않음.
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return lines
      .filter((l) => {
        if (hiddenKinds.has(l.kind)) return false;
        if (q && !(l.text || "").toLowerCase().includes(q)) return false;
        return true;
      })
      .sort((a, b) => {
        const ta = a.ts ? new Date(a.ts).getTime() : 0;
        const tb = b.ts ? new Date(b.ts).getTime() : 0;
        return tb - ta;   // 최신 → 과거
      });
  }, [lines, query, hiddenKinds]);
  const toggleKind = (k) => {
    setHiddenKinds((prev) => {
      const next = new Set(prev);
      next.has(k) ? next.delete(k) : next.add(k);
      return next;
    });
  };
  const allVisible = hiddenKinds.size === 0;
  return (
    <Panel style={{ height: "100%", display: "flex", flexDirection: "column" }}>
      {/* PanelHeader — 제목·카운트 | (우측) 검색창 + ← 장비 현황 요약
          검색창을 헤더로 올려 로그 목록 표시 공간 확보 (5/30 사용자 요청) */}
      <PanelHeader
        pad="8px 18px"
        right={
          <div style={{ display: "flex", alignItems: "center", gap: 8, flexShrink: 0 }}>
            {/* 종류 필터 칩 제거 (5/30 사용자 요청) — 검색창만 유지 */}
            <div style={{
              display: "flex", alignItems: "center", gap: 8,
              padding: "5px 12px", borderRadius: 10,
              background: "var(--bg-sunk)", border: "1px solid var(--line)",
              width: "clamp(150px, 14vw, 200px)",
            }}>
              <Icons.search size={13} color="var(--ink-4)" />
              <input
                placeholder="로그 검색..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                style={{
                  flex: 1, border: "none", outline: "none", background: "transparent",
                  fontSize: 12,
                }}
              />
            </div>
            {/* ← 장비 현황 요약 버튼은 헤더 좌측 세그먼트 토글로 이동 (5/30) */}
          </div>
        }
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10, flex: 1, minWidth: 0 }}>
          <PanelViewToggle logOpen={true} onToggleLog={onToggleLog} />
          {/* 시스템 로그 개수 칩 제거 (사용자 요청) — 합성 로그 줄 수라 정보 가치 낮음 */}
        </div>
      </PanelHeader>
      {/* 검색창은 헤더로 이동 (5/30) — 별도 검색 행 제거하여 로그 목록 공간 확보 */}
      <div className="scroll" style={{
        padding: 10, flex: 1, overflow: "auto",
        background: "var(--bg-sunk)",
      }}>
        {filtered.length === 0 && (query || !allVisible) ? (
          <div style={{
            padding: 16, textAlign: "center",
            fontSize: 11, color: "var(--ink-4)",
          }}>
            {query
              ? `"${query}" 와 일치하는 로그 없음`
              : "선택된 종류의 로그 없음"}
          </div>
        ) : (() => {
          // 인접 라인의 날짜(YYYY-MM-DD) 변경 지점에 separator 삽입
          const today = new Date(); today.setHours(0, 0, 0, 0);
          const fmtDay = (d) => `${d.getMonth() + 1}월 ${d.getDate()}일`;
          const dayKey = (d) => `${d.getFullYear()}-${d.getMonth()}-${d.getDate()}`;
          const todayKey = dayKey(today);
          const yKey = (() => { const y = new Date(today); y.setDate(y.getDate() - 1); return dayKey(y); })();

          let prevKey = null;
          const out = [];
          for (const l of filtered) {
            if (!l.ts) {
              // ts 없으면 separator 판정 불가 — 그냥 라인만
              out.push(<LogLine key={l.id} line={l} />);
              continue;
            }
            const d = new Date(l.ts);
            const k = dayKey(d);
            if (k !== prevKey) {
              const label = k === todayKey ? `${fmtDay(d)} (오늘)`
                          : k === yKey     ? `${fmtDay(d)} (어제)`
                          :                  `${d.getFullYear()}년 ${fmtDay(d)}`;
              out.push(
                <div key={`sep-${k}`} style={{
                  display: "flex", alignItems: "center", gap: 10,
                  margin: "10px 4px 6px",
                  fontSize: 9, fontWeight: 700, color: "var(--ink-4)",
                  letterSpacing: "0.05em",
                }}>
                  <div style={{ flex: 1, height: 1, background: "var(--line-soft)" }} />
                  <span style={{ whiteSpace: "nowrap" }}>{label}</span>
                  <div style={{ flex: 1, height: 1, background: "var(--line-soft)" }} />
                </div>
              );
              prevKey = k;
            }
            out.push(<LogLine key={l.id} line={l} />);
          }
          return out;
        })()}
      </div>
    </Panel>
  );
}

// ── AI 챗봇 (mock) ─────────────────────────────────────────
//   현재 LLM 미연동 — 키워드/노드 ID 매칭 기반 응답.
//   실제 백엔드 연결 시 mockAIResponse → fetch("/api/chat") 으로 교체 예정.

const STATUS_KO_BY_KEY = { normal: "정상", critical: "이상", warn: "관찰", offline: "통신 장애" };

function mockAIResponse(input, ctx = {}) {
  const equipment = ctx.equipment || [];
  const text = (input || "").trim();
  const lower = text.toLowerCase();

  // 1) 노드 ID 직접 조회
  const nodeMatch = text.match(/TB24-[A-Z0-9-]+/i);
  if (nodeMatch) {
    const node = nodeMatch[0].toUpperCase();
    const eq = equipment.find((e) => e.deviceId === node);
    if (!eq) return `${node} 는 등록된 장비가 아닙니다.`;
    const lines = [
      `📍 ${node} (${eq.zone || "-"})`,
      `• 상태: ${STATUS_KO_BY_KEY[eq.status] || eq.status}`,
      `• MSE: ${formatMse(eq.aiMse ?? eq.mse)} (임계 ${formatMse(eq.aiThreshold ?? eq.threshold)})`,
      `• 최근 라벨: ${eq.aiRisk ? `AI ${eq.aiRisk}` : eq.label || "정상"}`,
      eq.contribution?.length ? `• 기여도 1순위: ${eq.contribution[0].sensor} ${eq.contribution[0].pct}%` : null,
      `• 마지막 갱신: ${eq.updatedAt || "—"}`,
    ].filter(Boolean);
    return lines.join("\n");
  }

  // 2) 이상/관찰 키워드 → 현재 목록 (이두현 모델 등급)
  if (/위험|이상|critical/.test(lower)) {
    const c = equipment.filter((e) => e.status === "critical");
    if (c.length === 0) return "현재 이상 단계 장비가 없습니다.";
    return `🚨 이상 ${c.length}건:\n${c.map((e) => `• ${e.deviceId} · ${e.zone} — ${e.label}`).join("\n")}`;
  }
  if (/관찰|의심|anomaly|watch|warn/.test(lower)) {
    const w = equipment.filter((e) => e.status === "warn");
    if (w.length === 0) return "현재 관찰 단계 장비가 없습니다.";
    return `⚠️ 관찰 ${w.length}건:\n${w.slice(0, 6).map((e) => `• ${e.deviceId} · ${e.zone} — ${e.label}`).join("\n")}`;
  }
  if (/장애|offline|통신/.test(lower)) {
    const o = equipment.filter((e) => e.status === "offline");
    if (o.length === 0) return "현재 통신 장애 장비가 없습니다.";
    return `📵 통신 장애 ${o.length}건:\n${o.map((e) => `• ${e.deviceId} · ${e.zone}`).join("\n")}`;
  }
  if (/요약|상태|summary|현황/.test(lower)) {
    const c = { critical: 0, warn: 0, normal: 0, offline: 0 };
    equipment.forEach((e) => { if (c[e.status] !== undefined) c[e.status]++; });
    return `📊 전체 ${equipment.length}대\n• 이상 ${c.critical}대 · 관찰 ${c.warn}대\n• 통신장애 ${c.offline}대 · 정상 ${c.normal}대`;
  }

  // 3) 도메인 용어 설명
  if (/방식전위|방식 전위|cathodic/.test(lower)) {
    return "방식전위(P/S Potential)는 매설배관의 전기방식 효율 지표입니다. 일반 기준 -850 mV 이하면 양호, 초과하면 부식 진행 가능. 본 시스템은 일교차·AC 유입과 함께 추세 분석.";
  }
  if (/희생전류|희생양극|sacrificial/.test(lower)) {
    return "희생양극은 배관 대신 부식되어 본 배관을 보호합니다. 희생전류가 점차 감소하면 양극 소모 또는 접속부 불량 가능성. 1mA 이하면 교체 검토 필요.";
  }
  if (/ac\s*유입|ac\b/.test(lower)) {
    return "AC 유입은 인접 송전선·전철 등에서 유도되는 교류 전압. 200 mV 이상은 가속 부식 위험, 500 mV 이상은 즉각 차폐 또는 배수장치 점검 필요.";
  }
  if (/통신품질|dbm|rssi/.test(lower)) {
    return "통신 품질은 노드 신호 세기(dBm). -65 이상 양호, -75 이하 주의, -85 이하 통신 두절 임박. 게이트웨이 위치·안테나 점검.";
  }
  if (/임계|threshold|mse/.test(lower)) {
    return "MSE 임계값은 단말별로 다릅니다. 현재 MSE가 threshold의 70% 미만이면 정상, 70~100%이면 관찰, 100% 초과이면 이상으로 분류합니다.";
  }

  // 4) 도움말
  if (/도움|help|\?$|메뉴/.test(lower)) {
    return "사용 예시:\n• 'TB24-250448' 특정 장비 조회\n• '이상' / '관찰' / '장애' 현재 목록\n• '요약' 전체 상태\n• '방식전위' / '희생전류' / 'AC유입' 도메인 설명";
  }

  // 5) fallback
  return `"${text}" — 아직 LLM 미연동 상태라 일반 응답이 어렵습니다.\n노드 ID(예: TB24-250448) 또는 도메인 키워드로 질문해 주세요. '도움'을 입력하면 사용법을 안내합니다.`;
}

// 컨텍스트 추출 (equipment + weather → LLM 시스템 프롬프트용)
function buildChatContext(equipment, weather) {
  const counts = { all: equipment.length, normal: 0, critical: 0, warn: 0, offline: 0 };
  const criticalNodes = [];
  const warnNodes = [];
  const offlineNodes = [];       // 단순 ID 리스트 (기존 호환)
  const offlineDetails = [];     // 신규: 마지막 측정 시각 + 두절 시간
  const trends = []; // 위험·이상 의심 노드의 12시간 MSE 추이
  equipment.forEach((e) => {
    if (counts[e.status] !== undefined) counts[e.status]++;
    if (e.status === "critical") criticalNodes.push(e.deviceId);
    else if (e.status === "warn") warnNodes.push(e.deviceId);
    else if (e.status === "offline") {
      offlineNodes.push(e.deviceId);
      offlineDetails.push({
        deviceId:     e.deviceId,
        zone:         e.zone,
        location:     e.location,
        facilityId:   e.facilityId,
        updatedAt:    e.updatedAt,     // 마지막 측정 시각 (ISO)
        hoursSilent:  e.hoursSilent,   // 몇 시간 끊겼는지
      });
    }
    // 위험·이상 의심 노드만 trend 포함 (토큰 절약)
    if ((e.status === "critical" || e.status === "warn") && Array.isArray(e.mseHistory)) {
      trends.push({
        deviceId: e.deviceId,
        zone: e.zone,
        label: e.label,
        status: e.status,
        mse: e.mse,
        mseHistory: e.mseHistory,
      });
    }
  });
  // 시각 정보
  const now = new Date();
  const nowText = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-${String(now.getDate()).padStart(2, "0")} ${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}`;
  // 날씨 (있을 때만) — precip(강수 mm)/humidity(상대습도 %)도 LLM 컨텍스트로 전달
  const weatherCtx = weather && !weather.stale
    ? {
        temp: weather.temp, ko: weather.ko, code: weather.code, time: weather.time,
        precip: weather.precip ?? null, humidity: weather.humidity ?? null,
      }
    : null;
  return { counts, criticalNodes, warnNodes, offlineNodes, offlineDetails, trends, nowText, weather: weatherCtx };
}

async function callLLM(message, context, history) {
  const ctrl = new AbortController();
  const timeout = setTimeout(() => ctrl.abort(), 60_000);
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, context, history }),
      signal: ctrl.signal,
    });
    clearTimeout(timeout);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || "unknown");
    return { ok: true, reply: data.reply, model: data.model };
  } catch (err) {
    clearTimeout(timeout);
    return { ok: false, error: err.message };
  }
}

// SSE 스트리밍 호출
//   onDelta(piece, acc)              — 토큰 단위 콜백
//   onTool({round,name,args})        — 서버가 도구 호출했을 때 알림 (function calling)
//   onSession({sessionId})           — 서버가 부여한 chat_sessions.id
//   onDone(payload)                  — 종료 콜백
//   onError(err)                     — 에러
async function callLLMStream(message, context, history, sessionId, demoMode, model, { onDelta, onTool, onSession, onDone, onError, signal, webSearch, clientId }) {
  let acc = "";
  // 무응답 가드 — STALL_MS 동안 새 데이터가 한 조각도 안 오면 abort.
  //   Ollama 첫 토큰 스톨·도구 지연·터널(SSE) 끊김 등으로 '생성 중' 인디케이터가
  //   영구 고착되는 것을 방지. 활성 스트림은 매 청크마다 타이머를 리셋하므로
  //   정상 응답(도구 라운드 포함)은 끊기지 않는다.
  const ctrl = new AbortController();
  // 27b 는 32GB 에서 prefill/첫토큰이 느려 60s 를 넘길 때가 있음 → mock 폴백 방지 위해 더 길게 (서버 타임아웃 180s 안쪽)
  const STALL_MS = model === "qwen3.5:27b" ? 1_200_000 : (model && model.startsWith("gpt-")) ? 120_000 : 60_000;
  let stallTimer = null;
  const armStall = () => {
    if (stallTimer) clearTimeout(stallTimer);
    stallTimer = setTimeout(() => ctrl.abort(), STALL_MS);
  };
  if (signal) {
    if (signal.aborted) ctrl.abort();
    else signal.addEventListener("abort", () => ctrl.abort(), { once: true });
  }
  try {
    armStall();   // fetch 응답 자체가 안 와도 가드
    const res = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, context, history, sessionId: sessionId || undefined, demo: !!demoMode, model, webSearch: !!webSearch, clientId: clientId || undefined }),
      signal: ctrl.signal,
    });
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      armStall();   // 데이터 받을 때마다 무응답 타이머 리셋
      buf += decoder.decode(value, { stream: true });
      // SSE block: event: X\ndata: Y\n\n
      let sep;
      while ((sep = buf.indexOf("\n\n")) !== -1) {
        const block = buf.slice(0, sep);
        buf = buf.slice(sep + 2);
        const lines = block.split("\n");
        let event = "message";
        let data  = "";
        for (const ln of lines) {
          if (ln.startsWith("event:")) event = ln.slice(6).trim();
          else if (ln.startsWith("data:")) data += ln.slice(5).trim();
        }
        if (!data) continue;
        let payload;
        try { payload = JSON.parse(data); } catch { continue; }
        if (event === "delta" && payload.text) {
          acc += payload.text;
          onDelta && onDelta(payload.text, acc);
        } else if (event === "tool") {
          // 서버가 DB 도구 호출 시작 (function calling)
          onTool && onTool(payload);
        } else if (event === "session") {
          // 서버가 부여한 chat_sessions.id
          onSession && onSession(payload);
        } else if (event === "done") {
          onDone && onDone(payload);
          return { ok: true, reply: payload.reply || acc.trim(), sessionId: payload.sessionId };
        } else if (event === "error") {
          throw new Error(payload.message || "stream error");
        }
      }
    }
    // 스트림이 자연 종료 (done 이벤트 없이) — 누적된 acc 반환
    onDone && onDone({ reply: acc.trim() });
    return { ok: true, reply: acc.trim() };
  } catch (err) {
    onError && onError(err);
    return { ok: false, error: err.message };
  } finally {
    if (stallTimer) clearTimeout(stallTimer);
  }
}

// 챗봇 응답에서 단일 status 키워드 감지 (정확히 1개일 때만 반환)
function detectKpiFromReply(text) {
  if (!text) return null;
  const flags = {
    critical: /위험|이상/.test(text),
    warn:     /관찰/.test(text),
    offline:  /통신\s*장애|통신\s*두절/.test(text),
    normal:   /정상\b|정상\s/.test(text),
  };
  const hits = Object.entries(flags).filter(([_, v]) => v).map(([k]) => k);
  return hits.length === 1 ? hits[0] : null;
}

// 채팅 히스토리 localStorage 키 + 한도
const CHAT_STORAGE_KEY = "siwon.chat.history";
const CHAT_SESSION_KEY = "siwon.chat.session_id";
const CHAT_LAST_ACTIVE_KEY = "siwon.chat.last_active_date";   // 자정 자동 새 세션용 (YYYY-MM-DD)
const SHARED_LOGIN_IDS = new Set(["siwon"]);   // 공유/공개 계정 — 계정 동기화 제외(브라우저-로컬 유지). 백엔드 SHARED_ACCOUNTS 와 일치
const CHAT_MAX_KEEP = 60; // 최근 60개 메시지만 보관

// 로컬(브라우저 timezone) YYYY-MM-DD 키
function localDateKey(d) {
  const x = d instanceof Date ? d : new Date(d);
  return `${x.getFullYear()}-${String(x.getMonth() + 1).padStart(2, "0")}-${String(x.getDate()).padStart(2, "0")}`;
}
function todayKey() { return localDateKey(new Date()); }

function loadChatHistory(key = CHAT_STORAGE_KEY) {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return null;
    const arr = JSON.parse(raw);
    if (!Array.isArray(arr) || arr.length === 0) return null;
    // 형식 검증
    return arr.filter((m) => m && typeof m.text === "string" && (m.role === "ai" || m.role === "user"));
  } catch { return null; }
}

function saveChatHistory(messages, key = CHAT_STORAGE_KEY) {
  try {
    const trimmed = messages.slice(-CHAT_MAX_KEEP);
    localStorage.setItem(key, JSON.stringify(trimmed));
  } catch { /* ignore quota */ }
}

// 세션 목록 fetch (헤더 드롭다운용)
async function fetchChatSessions() {
  try {
    const r = await fetch("/api/chat/sessions");
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const d = await r.json();
    return d.ok ? (d.sessions || []) : [];
  } catch { return []; }
}
async function fetchChatSession(id) {
  try {
    const r = await fetch(`/api/chat/sessions/${id}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const d = await r.json();
    return d.ok ? d : null;
  } catch { return null; }
}
// 세션 제목 + 메시지 본문 통합 검색. 실패 시 null (호출부에서 제목 폴백 유지)
async function searchChatSessions(q) {
  try {
    const r = await fetch(`/api/chat/search?q=${encodeURIComponent(q)}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const d = await r.json();
    return d.ok ? (d.sessions || []) : null;
  } catch { return null; }
}
// 매치 주변(±radius)만 잘라 스니펫 생성 + 양끝 생략부호
function snippetAround(text, term, radius = 42) {
  if (!text) return "";
  const i = text.toLowerCase().indexOf(term.toLowerCase());
  if (i < 0) return text.length > radius * 2 ? text.slice(0, radius * 2) + "…" : text;
  const start = Math.max(0, i - radius);
  const end = Math.min(text.length, i + term.length + radius);
  return (start > 0 ? "…" : "") + text.slice(start, end) + (end < text.length ? "…" : "");
}
// 매치어를 <mark> 로 감싼 노드 배열 (대소문자 무시)
function highlightTerm(text, term) {
  if (!term || !text) return text;
  const out = [];
  const lc = text.toLowerCase(), t = term.toLowerCase();
  let from = 0, idx;
  while ((idx = lc.indexOf(t, from)) !== -1) {
    if (idx > from) out.push(text.slice(from, idx));
    out.push(<mark key={idx} style={{ background: "var(--brand-wash)", color: "var(--brand)", padding: "0 1px", borderRadius: 3 }}>{text.slice(idx, idx + term.length)}</mark>);
    from = idx + term.length;
  }
  if (from < text.length) out.push(text.slice(from));
  return out;
}
async function deleteChatSession(id) {
  try {
    const r = await fetch(`/api/chat/sessions/${id}`, { method: "DELETE" });
    return r.ok;
  } catch { return false; }
}
async function renameChatSession(id, title) {
  try {
    const r = await fetch(`/api/chat/sessions/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    });
    return r.ok;
  } catch { return false; }
}
async function pinChatSession(id, pinned) {
  try {
    const r = await fetch(`/api/chat/sessions/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pinned: pinned ? 1 : 0 }),
    });
    return r.ok;
  } catch { return false; }
}

// 도구 호출 칩 — 비슷한 도구를 카테고리로 묶어 "{카테고리} 도구 호출" 로 표시
const TOOL_CATEGORY = {
  list_devices: "단말", get_device_detail: "단말", get_device_history: "단말",
  get_aggregate: "단말", find_devices_by_value: "단말", get_zone_summary: "단말",
  compare_devices: "단말", get_recent_changes: "단말", get_summary: "단말",
  search_devices_by_location: "위치", geocode_location: "위치", find_devices_near: "위치",
  get_alarms: "알람", get_maintenance_log: "알람",
  get_predictions: "AI", get_ai_model_info: "AI",
  execute_safe_sql: "DB", describe_table: "DB",
  get_weather_forecast: "날씨", get_weather_history: "날씨",
};

const CHAT_MODELS = [
  { value: "qwen3.5:9b",  label: "Qwen 빠름",   hint: "속도 빠름" },
  { value: "qwen3:14b",   label: "Qwen 균형",   hint: "속도 중간" },
  { value: "qwen3.5:27b", label: "Qwen 고품질", hint: "매우 느림" },
  { value: "gpt-4o-mini", label: "GPT 빠름",   hint: "외부 · 빠름" },
  { value: "gpt-5",       label: "GPT 고품질", hint: "외부 · 고품질" },
  { value: "gpt-5.5",     label: "GPT 최고품질", hint: "외부 · 최고품질" },
];

// 첨부 이미지 → 다운스케일(최대 변 1200px) PNG data URL. 용량 억제.
function imageFileToDataURL(file, maxPx = 1200) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const img = new Image();
      img.onload = () => {
        let w = img.width, h = img.height;
        if (Math.max(w, h) > maxPx) { const s = maxPx / Math.max(w, h); w = Math.round(w * s); h = Math.round(h * s); }
        const cv = document.createElement("canvas");
        cv.width = w; cv.height = h;
        cv.getContext("2d").drawImage(img, 0, 0, w, h);
        try { resolve(cv.toDataURL("image/png")); } catch (e) { reject(e); }
      };
      img.onerror = reject;
      img.src = reader.result;
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function ChatPanel({ equipment = [], weather = null, onBotReply, onAutoKpi, demoMode = false, autoMessage = null, onAutoConsumed, user = null }) {
  // 계정 스코프 — 로그인 개인 계정만 서버 동기화+WS; 게스트(siwon)/익명은 브라우저-로컬(현행)
  const accountScoped = !!user && !SHARED_LOGIN_IDS.has(user.id);
  const histKey = accountScoped ? `${CHAT_STORAGE_KEY}::${user.id}` : CHAT_STORAGE_KEY;
  const sessKey = accountScoped ? `${CHAT_SESSION_KEY}::${user.id}` : CHAT_SESSION_KEY;
  const initialTime = (() => { const d = new Date(); return d.toLocaleTimeString("ko-KR", { timeZone: "Asia/Seoul", hour: "numeric", minute: "2-digit" }); })();
  const greeting = { role: "ai", text: "안녕하세요. AI 관제 도우미입니다.\n노드 ID 또는 키워드(이상/관찰/방식전위 등)로 질문해 주세요.", time: initialTime, dateKey: todayKey() };
  const [messages, setMessages] = useState(() => loadChatHistory(histKey) || [greeting]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [llmActive, setLlmActive] = useState(null); // null=미확인, true=LLM, false=mock
  // 관리자 문의방 (별도 모드 — AI 지원 답변 + 관리자 전달)
  const [inquiryMode, setInquiryMode] = useState(false);
  const [inquiryKind, setInquiryKind] = useState("question");
  const [inquiryChannel, setInquiryChannel] = useState("admin");   // admin(관리자 문의) | developer(개발자 문의·포폴)
  const [inquiryMsgs, setInquiryMsgs] = useState([]);
  const [inquirySending, setInquirySending] = useState(false);
  const [inquiryImages, setInquiryImages] = useState([]);   // 문의 첨부 이미지 배열 (최대 5장, data URL)
  const [replyToast, setReplyToast] = useState(null);       // 방 안 볼 때 답변 도착 토스트 { target }
  const [flashQid, setFlashQid] = useState(null);           // 답변 클릭 시 강조할 질문 id
  const [replyTarget, setReplyTarget] = useState(null);     // 카톡식 답장 대상 { text, role }
  const [sessionId, setSessionId] = useState(() => {
    try { const v = localStorage.getItem(sessKey); return v ? Number(v) : null; } catch { return null; }
  });
  const [lastActiveDate, setLastActiveDate] = useState(() => {
    try { return localStorage.getItem(CHAT_LAST_ACTIVE_KEY) || null; } catch { return null; }
  });
  const [showSessions, setShowSessions] = useState(false);
  const [sessions, setSessions] = useState([]);
  const [sessionsLoading, setSessionsLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState(() => {
    try { return localStorage.getItem("siwon.chat.model") || "qwen3.5:9b"; } catch { return "qwen3.5:9b"; }
  });
  const [modelMenuOpen, setModelMenuOpen] = useState(false);
  const modelBtnRef = useRef(null);
  const [webSearch, setWebSearch] = useState(false);   // 🌐 웹검색 토글 (DuckDuckGo)
  const [promptsOpen, setPromptsOpen] = useState(false); // + 추천문구 팝업
  const [promptIndex, setPromptIndex] = useState(0);     // 추천 질문 목록 하이라이트 (↑/↓)
  const [cmdPrompt, setCmdPrompt] = useState(null);      // 선택한 추천 질문(스킬) 객체 {title,prompt,icon,arg}. 칩으로 표시, 입력칸엔 인자를 받아 전송 시 결합
  const taRef = useRef(null);                            // textarea 자동높이
  const abortRef = useRef(null);   // 진행 중 스트리밍 중지용 AbortController
  const curModel = CHAT_MODELS.find((m) => m.value === selectedModel) || CHAT_MODELS[0];
  const provider = curModel.value.startsWith("gpt-") ? "gpt" : "local";   // 로컬 LLM / GPT 그룹
  const visibleModels = CHAT_MODELS.filter((m) => (m.value.startsWith("gpt-") ? "gpt" : "local") === provider);
  const switchProvider = (p) => {
    if (p === provider || sending) return;
    let next = null;
    try { next = localStorage.getItem(p === "gpt" ? "siwon.chat.gptModel" : "siwon.chat.localModel"); } catch {}
    if (!next || !CHAT_MODELS.some((m) => m.value === next)) next = p === "gpt" ? "gpt-4o-mini" : "qwen3:14b";
    setSelectedModel(next);
    try { localStorage.setItem("siwon.chat.model", next); } catch {}
    // 메뉴는 열어둔다 — 전환된 그룹의 모델을 바로 고를 수 있게
  };
  const listRef = useRef(null);
  const stickBottomRef = useRef(true);   // 맨아래 고정 상태인지 — onScroll 로 추적, 전송/스트리밍 추적용
  const [guestbookOpen, setGuestbookOpen] = useState(false);   // 단톡방(방명록) 모드 — 토글 4번째 칸
  const gb = useGuestbook(guestbookOpen);
  const [gbImage, setGbImage] = useState(null);                // 라운지(공개문의) 사진 첨부 — 단일 data URL(살균·리사이즈)
  const isGbAdmin = !!user && (user.role === "admin" || user.role === "superadmin");
  useEffect(() => {
    if (!modelMenuOpen) return;
    const onDoc = (e) => { if (modelBtnRef.current && !modelBtnRef.current.contains(e.target)) setModelMenuOpen(false); };
    const onEsc = (e) => { if (e.key === "Escape") setModelMenuOpen(false); };
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onEsc);
    return () => { document.removeEventListener("mousedown", onDoc); document.removeEventListener("keydown", onEsc); };
  }, [modelMenuOpen]);

  // 입력이 비면 textarea 높이 리셋 (전송 후 여러 줄 흔적 제거)
  useEffect(() => { const t = taRef.current; if (t && !input) t.style.height = "auto"; }, [input]);

  // 답변 토스트 5초 후 자동 사라짐
  useEffect(() => { if (!replyToast) return; const t = setTimeout(() => setReplyToast(null), 5000); return () => clearTimeout(t); }, [replyToast]);

  // 생성 중 실시간 표시 (경과초 · 토큰) — sending 동안 300ms 틱
  const [genTokens, setGenTokens] = useState(0);
  const [, setGenTick] = useState(0);
  const genStartRef = useRef(0);
  useEffect(() => {
    if (!sending) return;
    genStartRef.current = Date.now();
    setGenTokens(0);
    const id = setInterval(() => setGenTick((t) => t + 1), 300);
    return () => clearInterval(id);
  }, [sending]);

  // B1. 자정 자동 새 세션 — 마운트 시 한 번 체크
  //   localStorage 의 last_active_date 가 오늘과 다르면 = 다음 날 챗봇 첫 진입
  //   → sessionId / messages 리셋 후 오늘 키 저장
  useEffect(() => {
    const today = todayKey();
    if (lastActiveDate && lastActiveDate !== today) {
      // 다른 날 → 새 세션 시작
      setMessages([greeting]);
      setSessionId(null);
      try {
        localStorage.removeItem(CHAT_SESSION_KEY);
        localStorage.removeItem(CHAT_STORAGE_KEY);
      } catch {}
    }
    if (lastActiveDate !== today) {
      setLastActiveDate(today);
      try { localStorage.setItem(CHAT_LAST_ACTIVE_KEY, today); } catch {}
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 드롭다운 열기 → 세션 목록 로드
  const openSessionsList = async () => {
    setShowSessions((s) => !s);
    if (!showSessions) {
      setSessionsLoading(true);
      const list = await fetchChatSessions();
      setSessions(list);
      setSessionsLoading(false);
    }
  };

  // 세션 클릭 → 메시지 로드
  const loadSession = async (sid) => {
    setShowSessions(false);
    if (sending) return;
    const d = await fetchChatSession(sid);
    if (!d) return;
    const msgs = (d.messages || []).map((m) => {
      const t = m.createdAt ? new Date(m.createdAt) : new Date();
      return {
        role: m.role,
        text: m.text,
        time: t.toLocaleTimeString("ko-KR", { timeZone: "Asia/Seoul", hour: "numeric", minute: "2-digit" }),
        dateKey: localDateKey(t),    // C1. day-divider 용
      };
    });
    if (msgs.length === 0) return;
    setMessages(msgs);
    setSessionId(sid);
    try { localStorage.setItem(CHAT_SESSION_KEY, String(sid)); } catch {}
    setLlmActive(true);   // 영구 저장된 세션은 LLM 기록
  };

  // 로그인 개인 계정의 서버 대화 로드 (마운트/로그인/WS 재연결 시). loadSession 매핑 재사용.
  const loadCurrentServerSession = async () => {
    if (sending) return;
    try {
      const r = await fetch("/api/chat/sessions/current", { credentials: "same-origin" });
      const d = await r.json();
      if (!d?.ok || !d.session || !(d.messages || []).length) return;   // 없으면 greeting 유지
      const msgs = d.messages.map((m) => {
        const t = m.createdAt ? new Date(m.createdAt) : new Date();
        return { role: m.role, text: m.text, time: t.toLocaleTimeString("ko-KR", { timeZone: "Asia/Seoul", hour: "numeric", minute: "2-digit" }), dateKey: localDateKey(t) };
      });
      setMessages(msgs);
      setSessionId(d.session.id);
      try { localStorage.setItem(sessKey, String(d.session.id)); } catch {}
      setLlmActive(true);
    } catch {}
  };

  // WS 핸들러 클로저가 항상 현재값을 보도록 ref 동기화
  const connIdRef = useRef(null);
  const sessionIdRef = useRef(sessionId);
  sessionIdRef.current = sessionId;
  const inquiryModeRef = useRef(inquiryMode);
  inquiryModeRef.current = inquiryMode;
  const inquiryChannelRef = useRef(inquiryChannel);
  inquiryChannelRef.current = inquiryChannel;
  const reloadInquiryRef = useRef(null);   // openInquiry 정의 후 할당(TDZ 회피) — WS 문의 이벤트 시 새로고침용

  // 마운트/로그인 시 대화 동기화 — 계정=서버(/sessions/current, 기기 간 동기화), 익명/공유=현행 localStorage.
  useEffect(() => {
    if (accountScoped) loadCurrentServerSession();
    else if (sessionId) loadSession(sessionId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [accountScoped, user?.id]);

  // 로그인 개인 계정 → WebSocket 연결. 같은 계정 다른 화면에 새 메시지 실시간 반영. (게스트/익명 미연결)
  useEffect(() => {
    if (!accountScoped) return;
    let stop = false, retry = 0, timer = null, ws = null;
    const connect = () => {
      if (stop) return;
      const proto = location.protocol === "https:" ? "wss:" : "ws:";
      try { ws = new WebSocket(`${proto}//${location.host}/ws/chat`); } catch { return; }
      ws.onopen = () => { retry = 0; };
      ws.onmessage = (ev) => {
        let msg; try { msg = JSON.parse(ev.data); } catch { return; }
        if (msg.type === "hello") { connIdRef.current = msg.connId; return; }
        if (msg.type === "chat:user" || msg.type === "chat:ai") {
          const role = msg.type === "chat:ai" ? "ai" : "user";
          if (msg.sessionId && sessionIdRef.current && msg.sessionId !== sessionIdRef.current) return;  // 다른 세션 무시
          setMessages((cur) => {
            const last = cur[cur.length - 1];
            if (last && last.role === role && last.text === msg.text) return cur;   // tail dedup
            const now = new Date();
            const base = { role, text: msg.text, time: now.toLocaleTimeString("ko-KR", { timeZone: "Asia/Seoul", hour: "numeric", minute: "2-digit" }), dateKey: todayKey() };
            return [...cur, role === "ai" ? { ...base, meta: { model: msg.model } } : base];
          });
          if (msg.sessionId && !sessionIdRef.current) { setSessionId(msg.sessionId); try { localStorage.setItem(sessKey, String(msg.sessionId)); } catch {} }
        }
        // 문의 실시간 — 다른 화면의 새 문의 / 관리자·개발자 답변. 현재 그 문의방을 보고 있으면 새로고침.
        if (msg.type === "inquiry:new" || msg.type === "inquiry:reply") {
          const viewing = inquiryModeRef.current && inquiryChannelRef.current === msg.target;
          if (viewing && reloadInquiryRef.current) {
            reloadInquiryRef.current(msg.target);   // 그 방 보고 있으면 즉시 새로고침
          } else if (msg.type === "inquiry:reply") {
            setReplyToast({ target: msg.target, ts: msg.ts || Date.now() });   // 안 보고 있으면 토스트
          }
        }
      };
      ws.onerror = () => { try { ws.close(); } catch {} };
      ws.onclose = () => {
        if (stop) return;
        retry = Math.min(retry + 1, 6);
        timer = setTimeout(() => { connect(); loadCurrentServerSession(); }, Math.min(1000 * 2 ** retry, 30000) + Math.random() * 500);
      };
    };
    connect();
    return () => { stop = true; if (timer) clearTimeout(timer); try { ws && ws.close(); } catch {} };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [accountScoped, user?.id]);

  // 세션 삭제
  const removeSession = async (sid, e) => {
    if (e) { e.stopPropagation(); e.preventDefault(); }
    const ok = await deleteChatSession(sid);
    if (ok) {
      setSessions((s) => s.filter((x) => x.id !== sid));
      // 현재 세션이 삭제됐으면 새 세션 시작
      if (sid === sessionId) {
        setMessages([greeting]);
        setSessionId(null);
        try { localStorage.removeItem(CHAT_SESSION_KEY); } catch {}
      }
    }
  };

  // 새 세션 시작 (SessionSidebar onNew) — '새 대화' 버튼과 동일 리셋. 미정의 시 화면 크래시.
  const startNewSession = () => {
    if (sending) return;
    setMessages([greeting]);
    setLlmActive(null);
    setSessionId(null);
    setShowSessions(false);
    try {
      localStorage.removeItem(CHAT_STORAGE_KEY);
      localStorage.removeItem(CHAT_SESSION_KEY);
    } catch {}
  };

  // 세션 이름 변경 (SessionSidebar onRename) — 로컬 목록 즉시 갱신 + best-effort 저장(엔드포인트 없으면 무시)
  const renameSession = async (sid, title) => {
    if (!sid || !title) return;
    setSessions((s) => s.map((x) => (x.id === sid ? { ...x, title, name: title } : x)));
    try {
      await fetch(`/api/chat/sessions/${sid}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title }),
      });
    } catch {}
  };

  // 세션 고정/해제 (SessionSidebar onPin) — 낙관적 토글 + 고정 우선 재정렬
  const togglePinSession = async (sid, e) => {
    if (e) { e.stopPropagation(); e.preventDefault(); }
    const cur = sessions.find((x) => x.id === sid);
    const next = cur && (cur.pinned ? 1 : 0) ? 0 : 1;
    setSessions((s) => {
      const updated = s.map((x) => (x.id === sid ? { ...x, pinned: next } : x));
      // 고정(pinned=1) 먼저, 그 안에서 updated_at 내림차순
      return [...updated].sort((a, b) => {
        if ((b.pinned ? 1 : 0) !== (a.pinned ? 1 : 0)) return (b.pinned ? 1 : 0) - (a.pinned ? 1 : 0);
        return new Date(b.updated_at || 0) - new Date(a.updated_at || 0);
      });
    });
    await pinChatSession(sid, next);
  };

  // 메시지 변할 때마다 저장 + 자동 스크롤.
  // 맨아래 고정(stickBottomRef) 상태일 때만 따라 내림 — 전송/생성 중엔 따라가고, 사용자가 위로 올리면 멈춤.
  // (렌더 후 거리측정은 새 메시지로 콘텐츠가 커지면 오판되므로, onScroll 로 미리 기록한 ref 를 사용)
  useEffect(() => {
    const c = listRef.current;
    if (c && stickBottomRef.current) c.scrollTop = c.scrollHeight;
    saveChatHistory(messages, histKey);
  }, [messages, sending]);


  // send(e, q?) — q 지정 시 그것 우선 전송 (빠른 질문 카드용), 없으면 input state 사용
  const send = async (e, q) => {
    e && e.preventDefault();
    const trimmed = (q != null ? q : input).trim();
    if (!trimmed || sending) return;

    // B1. 자정 가드 (send 중 자정 넘긴 경우 대비) — 새 날이면 sessionId 리셋
    const today = todayKey();
    const isNewDay = !!lastActiveDate && lastActiveDate !== today;
    if (isNewDay && sessionId) {
      setSessionId(null);
      try { localStorage.removeItem(CHAT_SESSION_KEY); } catch {}
    }
    if (lastActiveDate !== today) {
      setLastActiveDate(today);
      try { localStorage.setItem(CHAT_LAST_ACTIVE_KEY, today); } catch {}
    }

    const now = new Date();
    const time = now.toLocaleTimeString("ko-KR", { timeZone: "Asia/Seoul", hour: "numeric", minute: "2-digit" });
    stickBottomRef.current = true;   // 전송 시 맨아래로 따라 내리도록 고정
    const newUserMsg = { role: "user", text: trimmed, time, dateKey: today };
    const r = new Date();
    const rtime = r.toLocaleTimeString("ko-KR", { timeZone: "Asia/Seoul", hour: "numeric", minute: "2-digit" });
    // 사용자 메시지 + 빈 AI 메시지(스트리밍 채워질 자리) 동시 추가
    //   isNewDay 면 base = [greeting] 로 (이전 메시지 자르고 새 세션 시작)
    setMessages((m) => {
      const base = isNewDay ? [greeting] : m;
      return [...base, newUserMsg, { role: "ai", text: "", time: rtime, streaming: true, dateKey: today }];
    });
    setInput("");
    setSending(true);

    const ctx = buildChatContext(equipment, weather);
    const historyForLLM = [...messages, newUserMsg].slice(-12);

    // LLM 스트리밍 시도
    let finalReply = "";
    let usedLLM = false;
    let donePayload = null;
    let stopped = false;
    let timedOut = false;
    let streamedSoFar = "";
    const t0 = Date.now();

    const controller = new AbortController();
    abortRef.current = controller;
    const stream = await callLLMStream(trimmed, ctx, historyForLLM, sessionId, demoMode, selectedModel, {
      webSearch,
      clientId: connIdRef.current,
      onSession: (info) => {
        // 서버가 새 세션 발급 또는 기존 세션 확인 — localStorage 에 저장
        if (info?.sessionId && info.sessionId !== sessionId) {
          setSessionId(info.sessionId);
          try { localStorage.setItem(sessKey, String(info.sessionId)); } catch {}
        }
      },
      onDelta: (_piece, acc) => {
        streamedSoFar = acc;
        setGenTokens((n) => n + 1);
        // AI 메시지(마지막)의 text 누적 갱신
        setMessages((m) => {
          const arr = m.slice();
          const last = arr[arr.length - 1];
          if (last && last.role === "ai" && last.streaming) {
            arr[arr.length - 1] = { ...last, text: acc };
          }
          return arr;
        });
      },
      onTool: (info) => {
        // 서버가 DB 도구 호출 — "🔧 list_devices 조회 중..." UI 표시
        setMessages((m) => {
          const arr = m.slice();
          const last = arr[arr.length - 1];
          if (last && last.role === "ai" && last.streaming) {
            const toolCalls = [...(last.toolCalls || []), info];
            arr[arr.length - 1] = { ...last, toolCalls };
          }
          return arr;
        });
      },
      onDone: (payload) => {
        finalReply = (payload && payload.reply) || "";
        donePayload = payload || null;
      },
      signal: controller.signal,
    });

    abortRef.current = null;
    if (stream.ok) {
      finalReply = stream.reply || finalReply;
      usedLLM = true;
    } else if (controller.signal.aborted) {
      // 사용자가 중지 — 지금까지 받은 부분 응답 유지 (mock 대체 안 함)
      finalReply = streamedSoFar || "(중지됨)";
      usedLLM = true;
      stopped = true;
    } else if (Date.now() - t0 > 30_000) {
      // 오래 끌다 실패 = 모델이 느려 시간 초과 (진짜 '미연동' mock 과 구분)
      timedOut = true;
      finalReply = (streamedSoFar ? streamedSoFar + "\n\n" : "")
        + "⏱ 응답이 너무 오래 걸려 중단됐습니다. 현재 **고품질(27b)** 은 이 서버(32GB)에서 도구를 여러 번 쓰는 분석 질문에 수 분이 걸리거나 시간 초과될 수 있어요. **균형(14b)** 모델을 권장합니다.";
    } else {
      // 스트리밍 실패 → mock fallback (진짜 미연동/연결 실패)
      finalReply = mockAIResponse(trimmed, { equipment });
    }

    // 마지막 메시지를 최종 결과로 확정 (streaming 플래그 해제, toolCalls 보존, meta 부착)
    const elapsedMs = Date.now() - t0;
    setMessages((m) => {
      const arr = m.slice();
      // 중지: 이번 턴의 AI 메시지 + 직전 사용자 메시지를 함께 제거 (대화 흔적 남기지 않음)
      if (stopped) {
        if (arr.length && arr[arr.length - 1].role === "ai") arr.pop();
        if (arr.length && arr[arr.length - 1].role === "user") arr.pop();
        return arr;
      }
      const last = arr[arr.length - 1];
      if (last && last.role === "ai") {
        arr[arr.length - 1] = {
          role: "ai",
          text: finalReply,
          time: rtime,
          toolCalls: last.toolCalls || [],
          meta: stream.ok ? {
            elapsedMs,
            rounds: donePayload?.rounds,
            tokens: donePayload?.tokens,
            model:  donePayload?.model,
          } : timedOut ? { elapsedMs, fallback: "timeout" }
            : { elapsedMs, fallback: "mock" },
        };
      }
      return arr;
    });
    setLlmActive(usedLLM);
    setSending(false);
    if (stopped) return;   // 중지 — 버블 제거됨, 후처리(지도 줌·KPI) 생략

    // 응답에서 노드 ID 추출 → 지도 자동 zoom
    //   실제 단말 체계 TB24-250xxx + 데모 단말 DEMO-### 모두 매칭 (대소문자 무시).
    const matches = (finalReply || "").match(/(?:TB24-[A-Za-z0-9]+|DEMO-[A-Za-z0-9]+)/g) || [];
    const nodes = [...new Set(matches.map((s) => s.toUpperCase()))];
    if (nodes.length > 0 && onBotReply) onBotReply(nodes);

    // 응답에서 단일 status 추출 → 자동 KPI 필터 (30초)
    if (onAutoKpi) {
      const kpi = detectKpiFromReply(finalReply);
      onAutoKpi(kpi);
    }
  };

  // ── 문의방 핸들러 (admin: 관리자 문의 / developer: 개발자 문의·포폴) ──
  const ttime = (d) => { try { return new Date(d).toLocaleTimeString("ko-KR", { timeZone: "Asia/Seoul", hour: "numeric", minute: "2-digit" }); } catch { return ""; } };
  // 답변 말풍선 클릭 → 그 질문으로 스크롤 + 통통 바운스 (반복 클릭 시 재생되도록 null→qid)
  const scrollToQuestion = (qid) => {
    const el = document.getElementById(`inqq-${qid}`);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
    setFlashQid(null);
    setTimeout(() => setFlashQid(qid), 20);
    setTimeout(() => setFlashQid((c) => (c === qid ? null : c)), 900);
  };
  // 말풍선 답장 버튼 → 인용 대상 지정 + 입력창 포커스
  const handleReplyTo = (msg) => { setReplyTarget({ text: (msg.text || "사진").slice(0, 300), role: msg.role }); setTimeout(() => taRef.current?.focus(), 0); };
  const openInquiry = async (channel = "admin") => {
    setInquiryChannel(channel);
    setInquiryMode(true);
    const greet = channel === "developer"
      ? { role: "ai", text: "👨‍💻 개발자 문의방입니다.\n이 프로젝트가 어떤 기능이고 어떻게 만들어졌는지 무엇이든 물어보세요. 제가 설명드리고, 필요하면 개발자(박지훈)가 직접 답변도 답니다.", time: initialTime }
      : { role: "ai", text: "📩 상담원 문의방입니다.\n사용 중 불편한 점이나 버그를 남겨 주세요. 접수되면 관리자에게 전달되고, 제가 먼저 도와드립니다.", time: initialTime };
    const msgs = [greet];
    try {
      const r = await fetch(`/api/inquiries/mine?target=${channel}`).then((x) => x.json());
      if (r?.ok) for (const q of (r.inquiries || [])) {
        const _qt = ttime(q.createdAt);
        const _dk = localDateKey(q.createdAt);   // 날짜 구분선용
        const _imgs = Array.isArray(q.images) ? q.images : (q.image ? [q.image] : []);
        for (const _src of _imgs) msgs.push({ role: "user", image: _src, text: "", time: _qt, dateKey: _dk, qid: q.id });   // 사진 먼저(각각)
        if (q.message) msgs.push({ role: "user", text: q.message, time: _qt, dateKey: _dk, qid: q.id, quote: q.replyQuote || null });  // 글자 다음
        if (q.botReply) msgs.push({ role: "ai", text: q.botReply, time: ttime(q.createdAt), dateKey: _dk, replyTo: q.id, meta: { model: channel === "developer" ? "AI 설명" : (q.kind === "bug" ? "버그 접수" : "문의 접수") } });
        if (q.adminReply) msgs.push({ role: "ai", text: q.adminReply, time: ttime(q.createdAt), dateKey: _dk, human: true, replyTo: q.id, meta: { model: channel === "developer" ? "개발자 답변" : "관리자 답변" } });
      }
    } catch { /* 비어있어도 진행 */ }
    setInquiryMsgs(msgs);
  };
  reloadInquiryRef.current = openInquiry;   // WS 문의 이벤트 시 현재 문의방 새로고침
  const submitInquiry = async (e) => {
    if (e) e.preventDefault();
    const text = input.trim();
    const imgs = inquiryImages;
    if ((!text && imgs.length === 0) || inquirySending) return;
    const channel = inquiryChannel, kind = inquiryKind;
    const quoteText = replyTarget?.text || null;
    setInput(""); setInquiryImages([]); setReplyTarget(null);
    stickBottomRef.current = true;   // 전송 시 맨아래로 따라 내리도록 고정
    const _now = ttime(Date.now());
    const _dk = todayKey();   // 날짜 구분선용
    setInquiryMsgs((m) => {
      const add = [];
      imgs.forEach((src) => add.push({ role: "user", image: src, text: "", time: _now, dateKey: _dk }));   // 사진 말풍선 먼저(각각)
      if (text) add.push({ role: "user", text, time: _now, dateKey: _dk, quote: quoteText });   // 글자 말풍선 다음
      return [...m, ...add];
    });
    setInquirySending(true);
    try {
      const r = await fetch("/api/inquiries", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target: channel, kind, message: text, images: imgs, replyQuote: quoteText, clientId: connIdRef.current }),
      }).then((x) => x.json());
      const reply = (r && r.reply) || (channel === "developer" ? "질문이 접수되었습니다. 개발자가 확인 후 답변드립니다. 🙇" : "문의가 접수되었습니다. 관리자가 확인 후 반영하겠습니다. 🙇");
      setInquiryMsgs((m) => [...m, { role: "ai", text: reply, time: ttime(Date.now()), dateKey: todayKey(), meta: { model: channel === "developer" ? "AI 설명" : (kind === "bug" ? "버그 접수" : "문의 접수") } }]);
    } catch {
      setInquiryMsgs((m) => [...m, { role: "ai", text: "접수 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.", time: ttime(Date.now()), dateKey: todayKey() }]);
    } finally { setInquirySending(false); }
  };

  // 외부 (AI 탐지 카드 클릭 등) 에서 autoMessage 가 들어오면 자동 전송. send 후 onAutoConsumed 호출로 부모 reset.
  useEffect(() => {
    if (autoMessage && !sending) {
      setInquiryMode(false);   // 문의 모드여도 AI 관제 도우미로 전환 → '상세 분석' 결과가 그 창에 표시되도록
      send(null, autoMessage);
      onAutoConsumed && onAutoConsumed();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoMessage]);

  const chatView = inquiryMode ? inquiryMsgs : messages;
  const busy = inquiryMode ? inquirySending : sending;
  const stopBtn = sending && !inquiryMode;   // 빨간 중지 버튼은 일반 스트리밍일 때만
  // 봇 아바타 — 일반=chatbot, 관리자 문의=상담원, 개발자 문의=개발자
  const botAvatar = inquiryMode
    ? (inquiryChannel === "developer" ? "/avatars/developer.png" : "/avatars/agent.png")
    : "/chatbot.png";
  const botLabel = inquiryMode
    ? (inquiryChannel === "developer" ? "개발자" : "상담원")
    : "AI 관제 도우미";
  // 추천 질문 항목 (부모 보유 — 목록·↑/↓ 네비·Enter 선택용). 아이콘 · 요약단어 · 상세설명 · 전송질문.
  const quickPromptItems = [
    // 문의 채널 (목록 최상단) — 선택 시 프롬프트 전송이 아니라 문의방을 연다 (channel 필드로 구분)
    { img: "/avatars/siwon.png", title: "시원팀 공개문의", desc: "프로젝트·팀에게 질문 — AI 페르소나가 답변", lounge: true },
    { img: "/avatars/agent.png",     title: "상담원 문의", desc: "문의·버그 신고 — 관리자에게 전달", channel: "admin" },
    // ── 추천 질문 50선 (매설배관 음극방식·AI(XAI)·운영 도메인 — 예리한 진단/분석 질문) ──
    { icon: Icons.crosshair, title: "우선 점검", desc: "위험도 순 TOP 5 + 근거·조치", prompt: "전체 단말을 AI 위험도 순으로 정렬해 즉시 점검이 필요한 TOP 5와 각 단말의 근거 수치(방식전위·AC유입·MSE)·권장 조치를 표로 정리해줘" },
    { icon: Icons.alert, title: "즉시 조치", desc: "당장 출동해야 할 단말만", prompt: "지금 당장 현장 출동이 필요한 위험 단말만 골라 위치와 즉시 조치를 알려줘" },
    { icon: Icons.pin, title: "점검 동선", desc: "같은 구역 묶어 동선 추천", prompt: "이상·관찰 단말을 같은 구역끼리 묶어 한 번에 점검할 최적 동선을 제안해줘" },
    { icon: Icons.clock, title: "방치 경고", desc: "오래 관찰로 방치된 단말", prompt: "관찰 상태로 가장 오래 방치된 단말 TOP 5와 경과 기간·조치 시급도를 알려줘" },
    { icon: Icons.cpu, title: "AI 근거", desc: "이상 판정 MSE·기여 피처 분해", prompt: "'이상' 단말의 LSTM-AutoEncoder 재구성오차(MSE)와 AI 기준 대비를 제시하고, 어떤 센서 피처가 가장 크게 기여했는지 분해해 근거를 설명해줘" },
    { icon: Icons.eye, title: "오탐 의심", desc: "알람 없이 AI만 이상", prompt: "알람은 없는데 AI만 '이상'으로 본 단말을 찾아 오탐 가능성과 추가 확인 포인트를 평가해줘" },
    { icon: Icons.alert, title: "미탐 점검", desc: "AI 정상인데 수치 경계", prompt: "AI는 '정상'이라 했지만 실측값이 위험 경계에 있는 단말을 찾아 놓친 위험이 없는지 점검해줘" },
    { icon: Icons.filter, title: "임계 타당성", desc: "이상 임계값 과민/둔감", prompt: "현재 AI 이상 임계값이 과민하거나 둔감한지 MSE 분포로 평가하고 적정 임계 조정안을 제시해줘" },
    { icon: Icons.activity, title: "MSE 급증", desc: "재구성오차 급등 단말", prompt: "최근 재구성오차(MSE)가 급증한 단말과 그 시점·동반된 센서 변화를 정리해줘" },
    { icon: Icons.cpu, title: "모델 신뢰", desc: "판정 경계·신뢰 낮음", prompt: "AI 판정이 경계에 가까워 신뢰가 낮은 단말을 추려 추가 점검이 필요한지 알려줘" },
    { icon: Icons.activity, title: "방식전위", desc: "-850mV 미달 부식위험", prompt: "방식전위가 방식기준 -850mV를 충족하지 못하는 단말을 찾아 부식 위험도·점검 우선순위·조치를 정리해줘" },
    { icon: Icons.zap, title: "과방식", desc: "-1200mV 초과 도막손상", prompt: "방식전위가 과방식(-1200mV 이하) 영역인 단말을 찾아 도막 손상·수소취화 위험을 평가해줘" },
    { icon: Icons.trend, title: "방식 추세", desc: "방식전위 지속 악화", prompt: "방식전위가 최근 지속적으로 악화되는 단말을 찾아 조기 경보와 원인 가설을 제시해줘" },
    { icon: Icons.zap, title: "교류 간섭", desc: "AC유입 과다 교류부식", prompt: "AC유입(교류전압)이 과다한 단말을 찾아 교류부식 위험을 평가하고 배류 등 대책·점검 우선순위를 제시해줘" },
    { icon: Icons.zap, title: "미주전류", desc: "직류 미주전류 의심", prompt: "방식전위·전류 패턴으로 직류 미주전류(stray current) 간섭이 의심되는 단말을 찾아줘" },
    { icon: Icons.crosshair, title: "AC 핫스팟", desc: "배류 보강 필요 구간", prompt: "AC 간섭이 집중된 핫스팟을 식별하고 배류시설 보강이 필요한 구간을 알려줘" },
    { icon: Icons.activity, title: "양극 소모", desc: "희생전류 저하 수명임박", prompt: "희생전류 저하로 희생양극 수명이 임박한 단말을 찾아 교체 우선순위를 알려줘" },
    { icon: Icons.activity, title: "전류 이상", desc: "희생전류 비정상 변동", prompt: "희생전류가 비정상적으로 급변하는 단말과 가능한 원인을 정리해줘" },
    { icon: Icons.sun, title: "온도 영향", desc: "온도급변·방식 상관", prompt: "온도 급변과 방식전위 이상이 함께 나타난 단말을 찾아 상관관계를 설명해줘" },
    { icon: Icons.activity, title: "습도 상관", desc: "습도 상승 부식가속", prompt: "습도가 높은 단말 중 부식 가속이 의심되는 단말을 찾아줘" },
    { icon: Icons.refresh, title: "센서 드리프트", desc: "비물리적 값 변동", prompt: "측정값이 물리적으로 설명되지 않게 드리프트하는 센서 고장 의심 단말을 찾아줘" },
    { icon: Icons.database, title: "결측 점검", desc: "측정 결측·신뢰도", prompt: "측정 결측·지연이 잦아 데이터 신뢰도가 낮은 단말을 찾아 보정 필요성을 알려줘" },
    { icon: Icons.wifi_off, title: "통신 장애", desc: "두절 위치·시간", prompt: "통신 장애 단말의 위치, 마지막 측정 시각, 두절 경과 시간을 정리해줘" },
    { icon: Icons.clock, title: "측정 지연", desc: "가장 오래된 측정", prompt: "마지막 측정이 가장 오래된 단말 TOP 5와 경과 시간을 알려줘" },
    { icon: Icons.check, title: "가용성", desc: "24h 측정 누락률", prompt: "최근 24시간 측정 누락률이 높은 단말을 찾아 가용성 문제를 진단해줘" },
    { icon: Icons.trend, title: "선제 예측", desc: "관찰→이상 전이 예측", prompt: "최근 추세로 곧 '이상'으로 전이될 가능성이 높은 '관찰' 단말을 예측하고 선제 점검 대상·이유를 알려줘" },
    { icon: Icons.trend, title: "급변 감지", desc: "24h 최대 변동", prompt: "최근 24시간 가장 가파르게 변한 단말 TOP 5와 변화량을 정리해줘" },
    { icon: Icons.sun, title: "계절성", desc: "온도변화 위험 상승", prompt: "계절·온도 변화로 향후 위험이 오를 가능성이 큰 단말을 사전 경보해줘" },
    { icon: Icons.layers, title: "구역 비교", desc: "구역별 이상 비율", prompt: "구역별 이상·관찰 비율을 비교해 가장 취약한 구역과 그 특징을 알려줘" },
    { icon: Icons.crosshair, title: "이상 군집", desc: "이상 밀집 핫스팟", prompt: "이상 단말이 지리적으로 밀집한 군집(핫스팟)을 식별해줘" },
    { icon: Icons.box, title: "시설별", desc: "시설 유형별 경향", prompt: "시설 유형(방조제·정문·교차로 등)별 위험 경향을 비교 분석해줘" },
    { icon: Icons.layers, title: "관제 보고", desc: "전체 종합+리스크·조치", prompt: "전체 상태를 관제 보고 형식으로 요약해줘 — 이상/관찰/정상/통신장애 집계, 핵심 리스크 단말, 오늘의 권장 조치 3가지" },
    { icon: Icons.list, title: "교대 인수", desc: "인계용 현황 요약", prompt: "교대 인계용으로 현재 상황과 주의 깊게 볼 단말을 간결히 요약해줘" },
    { icon: Icons.mail, title: "일일 브리핑", desc: "오늘 변화·신규·해소", prompt: "오늘의 변화를 브리핑해줘 — 신규 이상, 해소된 단말, 주의할 전이 건" },
    { icon: Icons.briefcase, title: "임원 보고", desc: "비전문가용 1분 요약", prompt: "비전문가도 이해하게 현재 안전 상태와 핵심 리스크·조치를 1분 분량으로 요약해줘" },
    { icon: Icons.search, title: "단말 심층", desc: "특정 ID 종합 진단", arg: "단말 ID", prompt: "내가 지정할 단말 ID의 모든 지표를 종합 진단해줘 — 먼저 어떤 단말인지 물어봐줘" },
    { icon: Icons.clock, title: "변화 이력", desc: "단말 상태 타임라인", arg: "단말 ID", prompt: "특정 단말의 최근 상태 변화 타임라인을 정리해줘 — ID를 먼저 물어봐줘" },
    { icon: Icons.layers, title: "비교 진단", desc: "두 단말 왜 다른가", arg: "비교할 두 단말 ID", prompt: "비슷해 보이는 두 단말을 비교해 왜 하나만 이상인지 차이를 설명해줘 — 두 ID를 물어봐줘" },
    { icon: Icons.pin, title: "이웃 영향", desc: "인접 단말 동반 위험", arg: "단말 ID", prompt: "특정 단말 주변 인접 단말이 함께 위험한지(군집성)를 확인해줘 — ID를 물어봐줘" },
    { icon: Icons.zap, title: "AC 초과", desc: "AC 기준 초과 폭", prompt: "AC유입 기준을 초과한 단말과 초과 폭을 큰 순으로 정리해줘" },
    { icon: Icons.filter, title: "경계값", desc: "기준에 아슬한 정상", prompt: "'정상'이지만 위험 기준에 가장 근접한 '아슬한' 단말 TOP 5를 뽑아줘" },
    { icon: Icons.alert, title: "복합 위험", desc: "둘 이상 지표 동시 악화", prompt: "방식전위·AC·MSE 등 둘 이상 지표가 동시에 나쁜 복합 위험 단말을 찾아줘" },
    { icon: Icons.check, title: "정상 검증", desc: "의심스러운 정상 역검증", prompt: "'정상' 분류 중 통계적으로 의심스러운 단말을 역검증해 오분류 가능성을 평가해줘" },
    { icon: Icons.crosshair, title: "조치 우선", desc: "효과 대비 우선순위", prompt: "위험 대비 조치 효과가 큰 순서로 처리 우선순위를 제안해줘" },
    { icon: Icons.clock, title: "점검 주기", desc: "위험도 기반 주기", prompt: "위험도에 따라 단말별 권장 점검 주기를 제안해줘" },
    { icon: Icons.refresh, title: "재발 단말", desc: "조치 후 재이상", prompt: "조치 후 다시 이상으로 돌아온(재발) 단말을 찾아줘" },
    { icon: Icons.database, title: "데이터 신선도", desc: "예측 스냅샷 최신성", prompt: "현재 AI 예측 스냅샷이 얼마나 최신인지, 갱신이 필요한지 알려줘" },
    { icon: Icons.trend, title: "추세 요약", desc: "최근 7일 위험 추이", prompt: "최근 7일 전체 위험 추이(증가/감소)와 변곡점을 요약해줘" },
    { icon: Icons.eye, title: "관찰 정리", desc: "관찰 단말 우선순위", prompt: "관찰 단말을 AI 기준 대비가 높은 순으로 정리하고 원인 피처를 붙여줘" },
    { icon: Icons.sparkle, title: "용어 설명", desc: "도메인 용어 쉽게", prompt: "방식전위·교류부식·희생양극 등 핵심 용어를 비전문가용으로 쉽게 설명해줘" },
  ];
  // 슬래시 명령 모드 — input 이 '/' 로 시작하면 추천 질문 팝업을 열고, '/' 뒤 글자로 필터링
  const slashMode = input.startsWith("/");
  const promptQuery = slashMode ? input.slice(1).trim().toLowerCase() : "";
  // 시원팀 공개문의(라운지) 전용 추천문구 — AI 페르소나에게 묻는 질문(스킬 칩으로 동작: 선택 → 덧붙임 입력 → 전송)
  //  · 아바타(img)가 붙은 항목은 담당 AI 페르소나에게 자동 라우팅됨 (이두현=AI, 이재헌=DB·PM, 박지훈=대시보드·통합)
  //  · arg 가 있으면 입력칸이 '인자' 입력칸으로 동작 (관제 도우미 추천질문과 동일 메커니즘)
  const loungePrompts = [
    // ── 프로젝트 전반 ──
    { icon: Icons.sparkle, title: "프로젝트 소개", desc: "이 시스템이 뭐예요?", prompt: "이 프로젝트가 어떤 시스템인지 한눈에 소개해줘" },
    { icon: Icons.briefcase, title: "핵심 가치", desc: "차별점·강점이 뭐예요?", prompt: "이 프로젝트의 핵심 가치와 다른 관제 시스템과의 차별점이 뭐예요?" },
    { icon: Icons.layers, title: "기술 스택", desc: "프론트·백·AI·DB·인프라", prompt: "이 프로젝트의 전체 기술 스택(프론트·백엔드·AI·DB·인프라)을 정리해줘" },
    { icon: Icons.box, title: "전체 흐름", desc: "센서→AI→대시보드", prompt: "센서 데이터가 수집되고 AI를 거쳐 대시보드에 뜨기까지 전체 흐름을 설명해줘" },
    // ── AI 이상탐지 (이두현) ──
    { img: "/avatars/lee_duhyeon.png", title: "AI 이상탐지", desc: "어떤 모델·판정 기준?", prompt: "AI 이상탐지는 어떤 모델을 쓰고 위험을 어떻게 판정해요?" },
    { img: "/avatars/lee_duhyeon.png", title: "왜 이 모델?", desc: "분류 대신 AutoEncoder", prompt: "왜 분류 모델 대신 LSTM AutoEncoder를 선택했어요?" },
    { img: "/avatars/lee_duhyeon.png", title: "3단계 판정", desc: "정상·관찰·이상", prompt: "정상·관찰·이상 3단계로 나눈 이유와 '관찰' 단계의 의미가 뭐예요?" },
    { img: "/avatars/lee_duhyeon.png", title: "임계치 설계", desc: "고정값·장비별 기준", prompt: "이상 임계치를 고정값으로, 그것도 장비마다 다르게 설정한 이유가 뭐예요?" },
    { img: "/avatars/lee_duhyeon.png", title: "AI 신뢰도", desc: "결과를 믿어도 되나", prompt: "AI 결과에 신뢰도(ai_reliability) 값을 같이 출력하는 이유가 뭐예요?" },
    // ── 데이터·DB·PM (이재헌) ──
    { img: "/avatars/lee_jaeheon.png", title: "데이터·DB", desc: "출처·동기화", prompt: "데이터는 어디서 오고 DB는 어떻게 동기화해요?" },
    { img: "/avatars/lee_jaeheon.png", title: "센서·통신", desc: "어떤 센서·누가 제공", prompt: "어떤 센서 데이터를 수집하고 IoT 센서 통신은 누가 제공했어요?" },
    { img: "/avatars/lee_jaeheon.png", title: "PM·일정", desc: "프로젝트 관리 방식", prompt: "PM으로서 일정과 팀 협업은 어떻게 관리했어요?" },
    // ── 대시보드·통합 (박지훈) ──
    { img: "/avatars/park.png", title: "대시보드·지도", desc: "화면 어떻게 만들었나", prompt: "대시보드와 지도는 무엇으로 어떻게 만들었어요?" },
    { img: "/avatars/park.png", title: "통합 아키텍처", desc: "파트들을 어떻게 연결", prompt: "프론트·백엔드·AI 파트를 어떻게 하나의 시스템으로 통합했어요?" },
    // ── 팀·과정 ──
    { icon: Icons.user, title: "팀 역할", desc: "누가 뭘 맡았나", prompt: "팀원들이 각자 어떤 역할을 맡았어요?" },
    { icon: Icons.alert, title: "어려웠던 점", desc: "가장 큰 도전", prompt: "프로젝트에서 가장 어려웠던 점은 뭐였어요?" },
  ];
  // 상담원 문의(문의·버그 신고) 전용 추천 템플릿 — 선택 시 입력칸에 채워짐(빈칸 채워 제출). kind 가 있으면 문의 종류도 설정.
  const inquiryPrompts = [
    { icon: Icons.alert,    title: "버그 신고",       desc: "오작동·에러 신고",   kind: "bug",      prompt: "[버그 신고]\n- 어느 화면/기능: \n- 무엇을 하다가: \n- 어떤 문제가 났는지: \n- 재현 방법(있다면): " },
    { icon: Icons.eye,      title: "화면 오류·깨짐",  desc: "레이아웃·표시 문제", kind: "bug",      prompt: "[화면 오류]\n- 어느 화면: \n- 브라우저/기기: \n- 어떻게 보이는지(증상): " },
    { icon: Icons.database, title: "데이터·수치 오류", desc: "값이 이상해요",       kind: "bug",      prompt: "[데이터 오류]\n- 어느 장비/화면: \n- 표시된 값: \n- 기대한 값: " },
    { icon: Icons.sparkle,  title: "기능 제안",       desc: "이런 기능이 있으면", kind: "question", prompt: "[기능 제안]\n- 원하는 기능: \n- 왜 필요한지(기대효과): " },
    { icon: Icons.settings, title: "개선 요청",       desc: "더 편했으면",       kind: "question", prompt: "[개선 요청]\n- 어디를: \n- 어떻게 바뀌면 좋을지: " },
    { icon: Icons.search,   title: "사용법 문의",     desc: "어떻게 쓰나요?",     kind: "question", prompt: "[사용법 문의]\n- 무엇을 하고 싶은지: \n- 어디서 막히는지: " },
    { icon: Icons.lock,     title: "계정·로그인",     desc: "로그인·권한 문제",   kind: "question", prompt: "[계정·로그인 문의]\n- 증상: \n- 계정 ID(선택): " },
    { icon: Icons.mail,     title: "기타 문의",       desc: "그 외 무엇이든",     kind: "question", prompt: "[문의]\n- 내용: " },
  ];
  const promptSource = guestbookOpen ? loungePrompts : inquiryMode ? inquiryPrompts : quickPromptItems;
  const filteredPrompts = promptQuery
    ? promptSource.filter((it) => (it.title + " " + it.desc + " " + (it.prompt || "")).toLowerCase().includes(promptQuery))
    : promptSource;
  // @멘션 모드(라운지 전용) — 입력 끝에 '@글자'가 있으면 팀원 페르소나 목록을 띄움(/ 와 동일 UX). 라우팅은 짧은 이름(박지훈 등) 매칭.
  const mentionMatch = guestbookOpen ? input.match(/(?:^|\s)@([^\s@]*)$/) : null;
  const mentionMode = !!mentionMatch;
  const mentionQuery = mentionMatch ? mentionMatch[1].toLowerCase() : "";
  const LOUNGE_FE_KEYS = ["park", "lee_jaeheon", "lee_duhyeon"];
  const mentionItems = mentionMode
    ? (gb.personas || [])
        .filter((p) => LOUNGE_FE_KEYS.includes(p.key))
        .map((p) => { const short = (p.name || "").replace(/^AI\s*/, ""); return { img: p.avatar, title: short, desc: p.lane || p.tone || "담당 페르소나", mention: short }; })
        .filter((m) => !mentionQuery || (m.title + " " + m.desc).toLowerCase().includes(mentionQuery) || (mentionQuery.length >= 2 && m.title.slice(0, 2) === mentionQuery.slice(0, 2)))   // 오타 허용: 앞 2글자 일치(이두헌→이두현)
    : [];
  // 팝업에 띄울 항목/선택 핸들러 — 멘션 모드면 팀원, 아니면 추천(슬래시/＋)
  const popupItems = mentionMode ? mentionItems : filteredPrompts;
  const showPrompts = promptsOpen || slashMode || mentionMode;
  // 선택지 팝업 활성(슬래시/멘션) — Enter/클릭이 '전송'이 아니라 '항목 선택' → 전송 버튼은 비활성으로 표시
  const cmdActive = showPrompts && (slashMode || mentionMode) && popupItems.length > 0;
  // 추천 질문/문의 항목 선택 — channel 있으면 문의방 열기, 추천 질문은 (전송 대신) 입력칸에 전체 문장 채우기
  const selectPrompt = (it) => {
    if (!it) return;
    setPromptsOpen(false);
    if (it.lounge) { setCmdPrompt(null); setInput(""); setGuestbookOpen(true); return; }   // 시원팀 공개문의로 전환
    if (it.channel) { setCmdPrompt(null); setInput(""); openInquiry(it.channel); return; }
    if (inquiryMode) {   // 상담원 문의 추천 = 템플릿 채움(빈칸 채워 제출). 문의 종류(kind)도 함께 설정.
      setCmdPrompt(null);
      if (it.kind) setInquiryKind(it.kind);
      setInput(it.prompt || it.title || "");
      requestAnimationFrame(() => { const ta = taRef.current; if (ta) { ta.focus(); ta.style.height = "auto"; ta.style.height = Math.min(ta.scrollHeight, 140) + "px"; } });
      return;
    }
    // 추천 질문을 '스킬 칩'으로 보관(아이콘·제목·전체 프롬프트·인자 힌트). 입력칸은 비워서 '인자(대상·조건)'를 받는다.
    // 라운지(시원팀 공개문의)도 동일 메커니즘 — 칩 선택 → 덧붙임 입력 → doSend 에서 결합해 AI 페르소나에게 전송.
    setCmdPrompt(it);
    setInput("");
    requestAnimationFrame(() => {
      const ta = taRef.current;
      if (ta) { ta.focus(); ta.style.height = "auto"; ta.style.height = Math.min(ta.scrollHeight, 140) + "px"; }
    });
  };
  // @멘션 선택 — 파란 칩(cmdPrompt 재사용)으로 표시. 입력의 '@쿼리'는 제거, 메시지는 칩 뒤에 작성. 전송 시 "@이름 메시지"로 합쳐 라우팅.
  const selectMention = (it) => {
    if (!it) return;
    setPromptsOpen(false);
    const rest = input.replace(/(^|\s)@([^\s@]*)$/, "$1").replace(/\s+$/, "");
    setCmdPrompt({ title: `@${it.mention}`, prompt: `@${it.mention}`, mention: true });
    setInput(rest);
    requestAnimationFrame(() => { const ta = taRef.current; if (ta) { ta.focus(); ta.style.height = "auto"; ta.style.height = Math.min(ta.scrollHeight, 140) + "px"; } });
  };
  // 팝업 항목 선택 — 멘션이면 팀원 삽입, 아니면 추천 선택
  const popupPick = (it) => (mentionMode ? selectMention(it) : selectPrompt(it));
  // 전송 — 스킬 칩(cmdPrompt)이 있으면 스킬 프롬프트 + 입력한 인자(대상·조건)를 결합해 보냄 (인자 없으면 스킬만)
  const doSend = (e) => {
    if (guestbookOpen) {
      // 라운지: 스킬 칩(추천 질문)이 있으면 질문 + 덧붙인 입력을 결합해 전송, 없으면 입력칸 그대로. 사진(gbImage) 동봉 가능.
      if (cmdPrompt !== null) {
        const arg = input.trim();
        let full = cmdPrompt.prompt;
        if (cmdPrompt.mention) full = arg ? `${cmdPrompt.prompt} ${arg}` : cmdPrompt.prompt;   // @멘션 칩 → "@이름 메시지"
        else if (arg) full = cmdPrompt.arg
          ? `${cmdPrompt.prompt} (${cmdPrompt.arg}: ${arg})`
          : `${cmdPrompt.prompt}\n\n덧붙임: ${arg}`;
        setCmdPrompt(null); setInput("");
        gb.send(full, !!user, gbImage, selectedModel); setGbImage(null); gb.stopTyping();
        return;
      }
      const t = input.trim();
      if (t || gbImage) { gb.send(t, !!user, gbImage, selectedModel); setInput(""); setGbImage(null); gb.stopTyping(); }
      return;
    }
    if (busy) return;
    if (cmdPrompt !== null) {
      const arg = input.trim();
      let full = cmdPrompt.prompt;
      if (arg) full = cmdPrompt.arg
        ? `${cmdPrompt.prompt}\n\n(${cmdPrompt.arg}: ${arg} — 이미 지정됨, 다시 묻지 말고 바로 분석)`
        : `${cmdPrompt.prompt}\n\n추가 조건: ${arg}`;
      setCmdPrompt(null); setInput("");
      send(null, full);
      return;
    }
    if (!input.trim()) return;
    (inquiryMode ? submitInquiry : send)(e);
  };
  return (
    <Panel style={{ height: "100%", display: "flex", flexDirection: "column", minHeight: 0, position: "relative" }}>
      <PanelHeader
        pad="8px 18px"
        right={(() => {
          return (
            <div style={{ display: "flex", alignItems: "center", gap: 6, position: "relative" }}>

              {/* 헤더 새 대화(새로고침) 버튼 제거 (5/31) — 세션 사이드바 '＋ 새 대화' 로 대체 */}
              {/* LLM 연결 상태 배지 제거 (사용자 요청 — 상단 'AI 연동됨' 표시와 중복) */}

            </div>
          );
        })()}
      >
        <div style={{ position: "relative", display: "flex", alignItems: "center", justifyContent: "center", gap: 8, minHeight: 30, flex: 1 }}>
          {/* 세션 목록 토글 버튼 제거(2026-06-01) — 사이드바/핸들러/백엔드 코드는 보존(향후 채팅 확장용),
              진입로만 차단해 showSessions 가 항상 false → 사이드바 미노출. 되살리려면 이 버튼만 복구. */}
          {/* 모드 토글 — AI 챗봇(일반) / 관리자 문의 / 개발자 문의 (장비현황 세그먼트 토글 스타일) */}
          <SegmentedToggle
            pad="5px 11px"
            items={[{ label: "AI 관제 도우미" }, { label: "상담원 문의" }, { label: "시원팀 공개문의" }]}
            activeIdx={guestbookOpen ? 2 : (!inquiryMode ? 0 : 1)}
            onSelect={(i) => {
              // 방(탭) 전환 시 작성 중이던 입력 초기화 — 사용자 요청 (텍스트·스킬/멘션 칩·첨부 사진·답장)
              setInput(""); setCmdPrompt(null); setGbImage(null); setInquiryImages([]); setReplyTarget(null); setPromptsOpen(false);
              if (i === 2) { setGuestbookOpen(true); }
              else { setGuestbookOpen(false); if (i === 0) setInquiryMode(false); else openInquiry("admin"); }
            }}
          />
          {/* 접속 표시 — 시원팀 공개문의(라운지)일 때만, 헤더 오른쪽 끝에 고정(토글은 중앙 유지) */}
          {guestbookOpen && (
            <span style={{ position: "absolute", right: 2, top: "50%", transform: "translateY(-50%)", display: "inline-flex", alignItems: "center", gap: 5, fontSize: 11, color: "var(--ink-3)", whiteSpace: "nowrap" }}>
              <span style={{ width: 7, height: 7, borderRadius: "50%", background: gb.connected ? "var(--ok)" : "var(--ink-4)" }} />{gb.connected ? `접속 ${gb.online}` : "연결 중…"}
            </span>
          )}
        </div>
      </PanelHeader>

      {guestbookOpen ? <GuestbookList gb={gb} isGuest={!user} isAdmin={isGbAdmin} me={user ? { name: user.name, role: user.role } : null} ChatMessage={ChatMessage} DayDivider={DayDivider} /> : (
      <div ref={listRef} className="scroll"
        onScroll={(e) => { const c = e.currentTarget; stickBottomRef.current = (c.scrollHeight - c.scrollTop - c.clientHeight) < 80; }}
        style={{
        flex: 1, overflow: "auto",
        padding: "12px 12px clamp(150px, 30vh, 380px)",
        background: "var(--bg-sunk)",
        display: "flex", flexDirection: "column", gap: 8,
      }}>
        {chatView.map((m, i) => {
          const prevKey = i > 0 ? chatView[i - 1]?.dateKey : null;
          // 첫 메시지(i===0)도 날짜 구분선 표시 — 대화 시작 시 연도+날짜 노출
          const showDivider = m.dateKey && (i === 0 || m.dateKey !== prevKey);   // 미데이트 인트로(상담원 greeting) 다음 첫 날짜 메시지에도 구분선
          // 같은 사람(role)·같은 분(time)·같은 날이 이어지면 시간은 그 묶음의 '마지막' 메시지에만 표시 (분 중복 제거)
          const nextM = chatView[i + 1];
          const hideTime = !!nextM && nextM.role === m.role && nextM.time === m.time && nextM.dateKey === m.dateKey;
          return (
            <Fragment key={i}>
              {showDivider && <DayDivider dateKey={m.dateKey} />}
              <div
                id={m.qid ? `inqq-${m.qid}` : undefined}
                onClick={m.replyTo ? () => scrollToQuestion(m.replyTo) : undefined}
                title={m.replyTo ? "이 답변의 질문으로 이동" : undefined}
                style={{
                  cursor: m.replyTo ? "pointer" : "default",
                  borderRadius: 12, transition: "background 300ms ease",
                  background: m.qid && flashQid === m.qid ? "var(--brand-wash)" : "transparent",
                  animation: m.qid && flashQid === m.qid ? "reply-bounce 0.6s ease" : "none",
                }}
              >
                <ChatMessage message={m} botAvatar={botAvatar} botLabel={botLabel} hideTime={hideTime} onReply={inquiryMode ? handleReplyTo : undefined} />
              </div>
            </Fragment>
          );
        })}
        {/* 스트리밍 중엔 마지막 AI 메시지의 깜빡 커서가 visual feedback 역할 */}
        {busy && chatView[chatView.length - 1]?.role !== "ai" && <ChatTyping botAvatar={botAvatar} />}
      </div>
      )}

      {/* 컴포저 — GPT식 둥근 카드 (textarea + 하단 액션줄). 대화 위에 띄워 글래스가 메시지를 비추도록 absolute + 투명 */}
      <div style={{ position: "absolute", left: 0, right: 0, bottom: 0, padding: "8px 10px 10px", background: "transparent" }}>

        {/* + 추천 질문 팝업 (위로) */}
        {showPrompts && (
          <div className="scroll glass-input" style={{
            position: "absolute", left: 12, right: 12, bottom: "calc(100% + 2px)", zIndex: 40,
            borderRadius: 22, padding: 8,
            maxHeight: "min(260px, 40vh)", overflowY: "auto",
          }}>
            {mentionMode && (
              <div style={{ padding: "4px 10px 6px", fontSize: 11, fontWeight: 700, color: "var(--ink-3)" }}>팀원 멘션 — 콕 집어 물어볼 페르소나 선택</div>
            )}
            {popupItems.length === 0 ? (
              <div style={{ padding: "8px 10px 12px", fontSize: 12, color: "var(--ink-3)" }}>{mentionMode ? "해당하는 팀원이 없어요" : "일치하는 추천 질문이 없습니다"}</div>
            ) : (
              <QuickPrompts items={popupItems} highlightIndex={promptIndex} onHover={setPromptIndex} onPick={popupPick} disabled={false} />
            )}
          </div>
        )}

        {/* 입력 카드 */}
        <form onSubmit={(e) => {
            e.preventDefault();
            if (showPrompts && (slashMode || mentionMode) && popupItems.length) {
              popupPick(popupItems[promptIndex] || popupItems[0]);
              return;
            }
            doSend(e);
          }} className="glass-input" style={{
          display: "flex", flexDirection: "column", gap: 7,
          padding: "10px 12px 9px", borderRadius: 22,
        }}>
          {/* 답장 인용 미리보기 — 카톡식 (↳ 원문 · ✕) */}
          {replyTarget && (
            <div style={{
              display: "flex", alignItems: "center", gap: 8,
              padding: "6px 8px", marginBottom: 2, borderRadius: 10,
              background: "var(--bg-sunk)", borderLeft: "3px solid var(--brand)",
            }}>
              <span style={{ fontSize: 13, color: "var(--brand)", flexShrink: 0 }}>↳</span>
              <span style={{ flex: 1, minWidth: 0, fontSize: 12, color: "var(--ink-3)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{replyTarget.text}</span>
              <button type="button" onClick={() => setReplyTarget(null)} title="답장 취소" style={{ flexShrink: 0, border: "none", background: "transparent", color: "var(--ink-4)", cursor: "pointer", display: "grid", placeItems: "center", padding: 2 }}>
                <Icons.close size={12} />
              </button>
            </div>
          )}
          {/* 스킬(추천 질문) + 인자 입력 — 한 줄 인라인: 파란 스킬 글자 + 그 뒤 일반 입력 */}
          <div style={{ display: "flex", alignItems: "baseline", gap: 6, width: "100%" }}>
            {cmdPrompt && (
              <span style={{ flexShrink: 0, color: "var(--brand)", fontWeight: 700, fontSize: 13.5, lineHeight: 1.5, whiteSpace: "nowrap" }}>
                {cmdPrompt.title}
              </span>
            )}
            <textarea
            ref={taRef}
            className="scroll"
            value={input}
            onChange={(e) => {
              const v = e.target.value;
              setInput(v);
              // '/' 명령 또는 '@' 멘션 입력 시 — 입력할 때마다 첫 항목으로 하이라이트 리셋
              if (v.startsWith("/") || /(?:^|\s)@[^\s@]*$/.test(v)) setPromptIndex(0);
              if (guestbookOpen) { if (v.trim()) gb.sendTyping(user ? user.name : (gb.guestName || "게스트")); else gb.stopTyping(); }   // 단톡방 '입력 중' 신호
              const t = e.target; t.style.height = "auto"; t.style.height = Math.min(t.scrollHeight, 140) + "px";
            }}
            onKeyDown={(e) => {
              // ── 스킬 칩(cmdPrompt) 활성 시 입력칸은 '인자' 입력. 인자가 비어 있을 때 Backspace 면 스킬 칩 제거 ──
              if (cmdPrompt !== null && e.key === "Backspace" && !input && !e.nativeEvent.isComposing) {
                e.preventDefault(); setCmdPrompt(null);
                const t = taRef.current; if (t) t.style.height = "auto";
                return;
              }
              // 추천/멘션 목록 열림 → ↑/↓ 이동, Enter 선택, Esc 닫기 (슬래시·@멘션 공통)
              if (showPrompts) {
                const n = popupItems.length;
                if (e.key === "ArrowDown") { e.preventDefault(); if (n) setPromptIndex((i) => (i + 1) % n); return; }
                if (e.key === "ArrowUp")   { e.preventDefault(); if (n) setPromptIndex((i) => (i - 1 + n) % n); return; }
                if (e.key === "Escape")    { e.preventDefault(); setPromptsOpen(false); if (slashMode) setInput(""); else if (mentionMode) setInput(input.replace(/(^|\s)@[^\s@]*$/, "$1")); return; }
                if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing && n) {
                  e.preventDefault();
                  popupPick(popupItems[promptIndex]);
                  return;   // 명령/멘션 모드에선 그대로 전송하지 않고 항목 선택
                }
              }
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); doSend(e); }
            }}
            onPaste={async (e) => {
              if (!inquiryMode) return;   // 문의 모드에서만 이미지 붙여넣기
              const items = e.clipboardData && e.clipboardData.items;
              if (!items) return;
              for (const it of items) {
                if (it.type && it.type.indexOf("image/") === 0) {
                  const f = it.getAsFile();
                  if (f) { e.preventDefault(); try { const _d = await imageFileToDataURL(f, 1200); setInquiryImages((arr) => arr.length >= 5 ? arr : [...arr, _d]); } catch { /* 무시 */ } }
                  return;
                }
              }
            }}
            placeholder={
              guestbookOpen ? (cmdPrompt ? "덧붙일 내용 입력 (없으면 그대로 전송)" : "/ 를 입력해 추천 질문 받기 , @ 를 입력해 팀원 지정하기")
              : inquiryMode ? "/ 를 입력해 추천 양식 받기"
              : cmdPrompt ? ""   // 스킬 활성 시 안내 문구 없음 (사용자 요청)
              : "/ 를 입력해 추천 질문 받기"
            }
            rows={1}
            style={{
              flex: 1, minWidth: 0, padding: 0, resize: "none", border: "none", outline: "none", background: "transparent",
              fontSize: 13.5, lineHeight: 1.5, color: "var(--ink)", fontWeight: 400, fontFamily: "inherit",
              minHeight: 22, maxHeight: 140, overflowY: "auto",
            }}
          />
          </div>
          {inquiryMode && inquiryImages.length > 0 && (
            <div style={{ display: "flex", flexWrap: "wrap", gap: 10, padding: "4px 2px 0" }}>
              {inquiryImages.map((src, i) => (
                <div key={i} style={{ position: "relative", display: "inline-block" }}>
                  <img src={src} alt="첨부 미리보기" style={{ display: "block", height: 56, borderRadius: 8, border: "1px solid var(--line)" }} />
                  <button type="button" onClick={() => setInquiryImages((arr) => arr.filter((_, j) => j !== i))} aria-label="첨부 제거"
                    style={{
                      position: "absolute", top: -7, right: -7,
                      width: 20, height: 20, borderRadius: "50%", padding: 0,
                      display: "grid", placeItems: "center",
                      background: "var(--err)", color: "#fff",
                      border: "2px solid var(--bg-elev)", boxShadow: "0 1px 4px rgba(0,0,0,0.25)",
                      fontSize: 12, fontWeight: 800, lineHeight: 1, cursor: "pointer",
                    }}>×</button>
                </div>
              ))}
            </div>
          )}
          {/* 라운지 사진 첨부 미리보기 (썸네일 + 제거) */}
          {guestbookOpen && gbImage && (
            <div style={{ display: "flex", gap: 10, padding: "4px 2px 0" }}>
              <div style={{ position: "relative", display: "inline-block" }}>
                <img src={gbImage} alt="첨부 미리보기" style={{ display: "block", height: 56, borderRadius: 8, border: "1px solid var(--line)" }} />
                <button type="button" onClick={() => setGbImage(null)} aria-label="첨부 제거"
                  style={{
                    position: "absolute", top: -7, right: -7,
                    width: 20, height: 20, borderRadius: "50%", padding: 0,
                    display: "grid", placeItems: "center",
                    background: "var(--err)", color: "#fff",
                    border: "2px solid var(--bg-elev)", boxShadow: "0 1px 4px rgba(0,0,0,0.25)",
                    fontSize: 12, fontWeight: 800, lineHeight: 1, cursor: "pointer",
                  }}>×</button>
              </div>
            </div>
          )}
          {/* 하단 액션줄 */}
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            {/* 추천 질문/양식 토글 — 모든 모드(관제·라운지·상담원 문의) 공통. 공통 버튼이라 항상 맨 왼쪽 고정(통일감) */}
            {(
              <button type="button" onClick={() => { setPromptsOpen((o) => !o); setPromptIndex(0); }} title="추천 질문"
                style={{
                  width: 34, height: 34, borderRadius: "50%", display: "grid", placeItems: "center",
                  border: "none",
                  background: promptsOpen ? "var(--brand-wash)" : "transparent",
                  color: promptsOpen ? "var(--brand)" : "var(--ink-3)", cursor: "pointer", flexShrink: 0,
                }}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" style={{ flexShrink: 0 }}>
                  <line x1="12" y1="5" x2="12" y2="19" />
                  <line x1="5" y1="12" x2="19" y2="12" />
                </svg>
              </button>
            )}
            {/* 상담원 문의 이미지 첨부 — '+' 뒤로 배치(공통 버튼 정렬 통일) */}
            {inquiryMode && (
              <label title="이미지 첨부 (PNG/JPG · 최대 5장)" style={{
                width: 34, height: 34, borderRadius: "50%", display: "grid", placeItems: "center", position: "relative",
                background: inquiryImages.length ? "var(--brand-wash)" : "transparent",
                color: inquiryImages.length ? "var(--brand)" : "var(--ink-3)", cursor: "pointer", flexShrink: 0,
              }}>
                <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="3" y="3" width="18" height="18" rx="2" /><circle cx="8.5" cy="8.5" r="1.5" /><path d="M21 15l-5-5L5 21" />
                </svg>
                {inquiryImages.length > 0 && (
                  <span style={{ position: "absolute", top: -2, right: -2, minWidth: 15, height: 15, padding: "0 3px", borderRadius: 999, background: "var(--brand)", color: "#fff", fontSize: 9, fontWeight: 800, display: "grid", placeItems: "center", border: "1px solid var(--bg-elev)" }}>{inquiryImages.length}</span>
                )}
                <input type="file" accept="image/png,image/jpeg" multiple style={{ display: "none" }}
                  onChange={async (e) => {
                    const files = Array.from(e.target.files || []);
                    e.target.value = "";
                    for (const f of files) {
                      try { const _d = await imageFileToDataURL(f, 1200); setInquiryImages((arr) => arr.length >= 5 ? arr : [...arr, _d]); } catch { /* 무시 */ }
                    }
                  }} />
              </label>
            )}
            {/* 라운지(공개문의) 사진 첨부 — 모델/웹검색이 있던 자리. 단일 이미지, 캔버스 재인코딩으로 살균·1200px 리사이즈 */}
            {!inquiryMode && guestbookOpen && (
              <label title="사진 첨부 (PNG/JPG · 1장)" style={{
                width: 34, height: 34, borderRadius: "50%", display: "grid", placeItems: "center", position: "relative",
                background: gbImage ? "var(--brand-wash)" : "transparent",
                color: gbImage ? "var(--brand)" : "var(--ink-3)", cursor: "pointer", flexShrink: 0,
              }}>
                <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="3" y="3" width="18" height="18" rx="2" /><circle cx="8.5" cy="8.5" r="1.5" /><path d="M21 15l-5-5L5 21" />
                </svg>
                <input type="file" accept="image/png,image/jpeg" style={{ display: "none" }}
                  onChange={async (e) => {
                    const f = (e.target.files || [])[0];
                    e.target.value = "";
                    if (f) { try { const _d = await imageFileToDataURL(f, 1200); setGbImage(_d); } catch { /* 무시 */ } }
                  }} />
              </label>
            )}
            {!inquiryMode && !guestbookOpen && (
              <button type="button" onClick={() => setWebSearch((w) => !w)} title={webSearch ? "웹검색 켜짐 (DuckDuckGo)" : "웹검색 끄기/켜기"}
                style={{
                  height: 34, borderRadius: 999, display: "inline-flex", alignItems: "center", gap: 6, justifyContent: "center",
                  padding: webSearch ? "0 12px" : 0, width: webSearch ? "auto" : 34,
                  border: "none",
                  background: webSearch ? "var(--brand)" : "transparent",
                  color: webSearch ? "#fff" : "var(--ink-3)", cursor: "pointer", flexShrink: 0,
                  transition: "background 140ms, width 140ms",
                }}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
                  <circle cx="12" cy="12" r="9" />
                  <line x1="3" y1="12" x2="21" y2="12" />
                  <ellipse cx="12" cy="12" rx="4" ry="9" />
                </svg>{webSearch && <span style={{ fontSize: 11.5, fontWeight: 700 }}>검색</span>}
              </button>
            )}
            {/* 모델칩 — [로컬|GPT] 토글 + 모델 팝업. 라운지(공개문의)에선 선택 모델이 AI 페르소나 답변에 적용(사용자 요청, GPT 공개비용 수용). */}
            {!inquiryMode && (
              <div ref={modelBtnRef} style={{ position: "relative", flexShrink: 0 }}>
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => { if (!busy) setModelMenuOpen((o) => !o); }}
                  title={provider === "gpt" ? "챗봇 모델 — GPT (외부 · OpenAI)" : "챗봇 모델 — 로컬 (Mac Studio Ollama)"}
                  style={{
                    display: "inline-flex", alignItems: "center",
                    height: 32, padding: "0 8px", borderRadius: 999,
                    background: "transparent", border: "none",
                    cursor: busy ? "not-allowed" : "pointer", opacity: busy ? 0.55 : 1, whiteSpace: "nowrap",
                  }}
                >
                  <span style={{ fontSize: 11.5, fontWeight: 700, color: "var(--ink-2)" }}>{curModel.label}</span>
                </button>
                {modelMenuOpen && (
                  <div style={{
                    position: "absolute", bottom: "calc(100% + 8px)", left: "50%", transform: "translateX(-50%)", zIndex: 60,
                    minWidth: 214, padding: 6,
                    background: "var(--bg-elev)", border: "1px solid var(--line)", borderRadius: 14,
                    boxShadow: "0 -10px 30px -10px rgba(15,23,42,0.30)",
                  }}>
                    {/* 말풍선 꼬리 — 팝업 하단 정중앙 */}
                    <div style={{
                      position: "absolute", bottom: -6, left: "50%", marginLeft: -6, width: 12, height: 12,
                      background: "var(--bg-elev)",
                      borderRight: "1px solid var(--line)", borderBottom: "1px solid var(--line)",
                      transform: "rotate(45deg)",
                    }} />
                    {visibleModels.map((m) => {
                      const active = m.value === selectedModel;
                      return (
                        <button key={m.value} type="button"
                          onClick={() => {
                            setSelectedModel(m.value);
                            try { localStorage.setItem("siwon.chat.model", m.value); localStorage.setItem(m.value.startsWith("gpt-") ? "siwon.chat.gptModel" : "siwon.chat.localModel", m.value); } catch {}
                            setModelMenuOpen(false);
                          }}
                          onMouseEnter={(e) => { if (!active) e.currentTarget.style.background = "var(--bg-sunk)"; }}
                          onMouseLeave={(e) => { if (!active) e.currentTarget.style.background = "transparent"; }}
                          style={{
                            display: "flex", flexDirection: "column", alignItems: "flex-start", gap: 2, width: "100%", textAlign: "left",
                            padding: "7px 10px", borderRadius: 7, border: "none", background: active ? "var(--brand-wash)" : "transparent", cursor: "pointer",
                          }}>
                          <span style={{ fontSize: 12.5, fontWeight: 700, color: active ? "var(--brand)" : "var(--ink)", display: "inline-flex", alignItems: "center", gap: 5 }}>
                            {m.label}{active && <span style={{ fontSize: 10 }}>✓</span>}
                          </span>
                          <span style={{ fontSize: 10, color: "var(--ink-4)", fontFamily: "JetBrains Mono, ui-monospace, monospace" }}>{m.value} · {m.hint}</span>
                        </button>
                      );
                    })}
                    {/* 로컬/GPT 토글 — 목록 아래 (간격 띄움) */}
                    <div style={{ display: "flex", gap: 2, padding: 2, marginTop: 12, borderRadius: 999, background: "var(--bg-sunk)", border: "1px solid var(--line)" }}>
                      {[["local", "로컬"], ["gpt", "GPT"]].map(([p, lbl]) => {
                        const on = provider === p;
                        return (
                          <button key={p} type="button" onClick={() => switchProvider(p)}
                            title={p === "gpt" ? "GPT (외부 · OpenAI)" : "로컬 LLM (Mac Studio Ollama)"}
                            style={{
                              flex: 1, padding: "5px 0", borderRadius: 999, border: "none", fontSize: 11.5, fontWeight: 700,
                              background: on ? "var(--bg-elev)" : "transparent", color: on ? "var(--ink)" : "var(--ink-3)",
                              boxShadow: on ? "var(--shadow-card)" : "none", cursor: on ? "default" : "pointer", transition: "all 140ms ease",
                            }}>
                            {lbl}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            )}
            <div style={{ flex: 1 }} />
            <button
              type={stopBtn ? "button" : "submit"}
              onClick={stopBtn ? () => { try { abortRef.current?.abort(); } catch {} } : undefined}
              disabled={stopBtn ? false : (busy || cmdActive || (!cmdPrompt && !input.trim() && !(guestbookOpen && gbImage)))}
              title={stopBtn ? "생성 중지" : cmdActive ? "추천 항목 선택 모드" : "전송"}
              style={{
                width: 34, height: 34, borderRadius: "50%", display: "grid", placeItems: "center", border: "none", flexShrink: 0,
                background: stopBtn ? "var(--err)" : (busy || cmdActive || (!cmdPrompt && !input.trim() && !(guestbookOpen && gbImage))) ? "var(--line)" : "var(--brand)",
                color: "#fff", cursor: stopBtn ? "pointer" : (busy || cmdActive || (!cmdPrompt && !input.trim() && !(guestbookOpen && gbImage))) ? "not-allowed" : "pointer",
                transition: "background 140ms",
              }}>
              {stopBtn
                ? <span style={{ width: 10, height: 10, borderRadius: 2, background: "#fff" }} />
                : <span style={{ fontSize: 17, fontWeight: 800, lineHeight: 1 }}>↑</span>}
            </button>
          </div>
        </form>
      </div>

      {/* 좌측 사이드바 — 세션 목록 (오버레이, ChatPanel 안에 갇힘)
          showSessions=true 일 때 좌측에서 슬라이드 인. 외부 클릭/ESC 로 닫힘. */}
      <SessionSidebar
        open={showSessions}
        loading={sessionsLoading}
        sessions={sessions}
        activeSessionId={sessionId}
        onClose={() => setShowSessions(false)}
        onPick={(id) => { setInquiryMode(false); loadSession(id); }}
        onDelete={removeSession}
        onNew={() => { setInquiryMode(false); startNewSession(); }}
        onRename={renameSession}
        onPin={togglePinSession}
        onOpenInquiry={(channel) => { openInquiry(channel); setShowSessions(false); }}
        inquiryActive={inquiryMode ? inquiryChannel : null}
      />
      {/* 답변 도착 토스트 — 방 안 볼 때 (클릭 시 그 방 열기) */}
      {replyToast && createPortal(
        <div
          onClick={() => { openInquiry(replyToast.target); setReplyToast(null); }}
          style={{
            position: "fixed", right: 24, bottom: 24, zIndex: 9999,
            display: "flex", alignItems: "center", gap: 11,
            padding: "12px 16px", borderRadius: 14, cursor: "pointer",
            background: "var(--bg-elev)", border: "1px solid var(--line)",
            boxShadow: "0 14px 36px -10px rgba(15,23,42,0.32)",
            animation: "slide-in-up 220ms ease", maxWidth: 320,
          }}
        >
          <span style={{ width: 34, height: 34, borderRadius: 10, flexShrink: 0, display: "grid", placeItems: "center", fontSize: 17, background: "var(--brand)", color: "#fff" }}>💬</span>
          <span style={{ minWidth: 0 }}>
            <span style={{ display: "block", fontSize: 13, fontWeight: 700, color: "var(--ink)" }}>{replyToast.target === "developer" ? "개발자" : "상담원"}이 답변했어요</span>
            <span style={{ display: "block", fontSize: 11.5, color: "var(--brand)", fontWeight: 600, marginTop: 1 }}>탭하여 보기 →</span>
          </span>
        </div>,
        document.body
      )}
    </Panel>
  );
}

// 세션 목록 사이드바 — ChatPanel 안 absolute 오버레이
//   open=true 시 좌측에서 슬라이드 인 (translateX 220ms).
//   ESC 또는 외부 클릭으로 닫힘. 세션 클릭 시 자동 닫힘 (onPick 안에서 처리).
function SessionSidebar({ open, loading, sessions, activeSessionId, onClose, onPick, onDelete, onNew, onRename, onPin, onOpenInquiry, inquiryActive }) {
  const ref = useRef(null);
  const [query, setQuery] = useState("");
  const [editingId, setEditingId] = useState(null);
  const [editText, setEditText] = useState("");
  const [menu, setMenu] = useState(null);   // 세션 행 ⋯ 메뉴: { id, s, pinned, top, left } | null
  const [search, setSearch] = useState(null);   // 백엔드 통합검색 결과: { q, sessions } | null

  // ESC 닫기 (단, 이름 변경 중이면 편집만 취소)
  useEffect(() => {
    if (!open) return;
    const onKey = (e) => {
      if (e.key !== "Escape") return;
      if (editingId != null) { setEditingId(null); return; }
      onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose, editingId]);

  // 외부 클릭 닫기 (사이드바 영역 밖)
  useEffect(() => {
    if (!open) return;
    const onDoc = (e) => {
      if (menu != null) return;   // 행 ⋯ 메뉴 열려 있으면 사이드바 닫지 않음 (메뉴 백드롭이 처리)
      if (ref.current && !ref.current.contains(e.target)) onClose();
    };
    const t = setTimeout(() => document.addEventListener("mousedown", onDoc), 0);
    return () => {
      clearTimeout(t);
      document.removeEventListener("mousedown", onDoc);
    };
  }, [open, onClose, menu]);

  // 패널 닫히면 검색/편집 상태 초기화
  useEffect(() => { if (!open) { setQuery(""); setEditingId(null); setMenu(null); setSearch(null); } }, [open]);

  // 통합검색(제목+본문) — 220ms 디바운스. 입력 비면 해제(평상 목록으로 복귀)
  useEffect(() => {
    const term = query.trim();
    if (!term) { setSearch(null); return; }
    let cancelled = false;
    const t = setTimeout(async () => {
      const r = await searchChatSessions(term);
      if (!cancelled && r != null) setSearch({ q: term, sessions: r });
    }, 220);
    return () => { cancelled = true; clearTimeout(t); };
  }, [query]);

  const term = query.trim();
  const searchMode = !!term;
  const searchReady = !!(search && search.q === term);   // 현재어에 대한 백엔드 결과 도착?
  // 표시 목록: 검색 중이면 백엔드 결과(제목+본문), 응답 전엔 로드된 세션 제목으로 즉시 필터(폴백)
  const displaySessions = !searchMode
    ? (sessions || [])
    : searchReady
      ? search.sessions
      : (sessions || []).filter((s) => (s.title || `세션 #${s.id}`).toLowerCase().includes(term.toLowerCase()));
  // 그룹: 검색 중엔 헤더 없이 평면(__flat). 평상시엔 고정됨 + 최신순 평면
  let grouped = null;
  if (open) {
    if (searchMode) {
      grouped = displaySessions.length ? { "__flat": displaySessions } : {};
    } else {
      const pinnedList = displaySessions.filter((s) => s.pinned);
      const rest = displaySessions
        .filter((s) => !s.pinned)
        .sort((a, b) => new Date(b.updated_at || 0) - new Date(a.updated_at || 0));
      // 고정됨이 있으면 나머지에 "최근" 헤더를 붙여 두 그룹을 분리, 없으면 헤더 없는 평면
      grouped = pinnedList.length
        ? { "고정됨": pinnedList, ...(rest.length ? { "최근": rest } : {}) }
        : (rest.length ? { "__flat": rest } : {});
    }
  }

  const beginEdit = (s, e) => {
    if (e) { e.stopPropagation(); e.preventDefault(); }
    setEditingId(s.id);
    setEditText(s.title || `세션 #${s.id}`);
  };
  const commitEdit = async (sid) => {
    const t = editText.trim();
    if (t && onRename) await onRename(sid, t);
    setEditingId(null);
  };

  return (
    <>
      {/* 스크림 — 사이드바 열릴 때 뒤 채팅 디밍 (클릭 시 닫힘) */}
      <div
        onClick={onClose}
        style={{
          position: "absolute", inset: 0, zIndex: 29,
          background: "rgba(15,23,42,0.32)",
          opacity: open ? 1 : 0,
          pointerEvents: open ? "auto" : "none",
          transition: "opacity 220ms ease",
        }}
      />
    <div
      ref={ref}
      style={{
        position: "absolute", left: 0, top: 0, bottom: 0,
        width: 280, zIndex: 30,
        background: "var(--bg)",
        border: "1px solid var(--line)",
        borderRadius: 16,
        boxShadow: open ? "20px 0 50px -12px rgba(15,23,42,0.45)" : "none",
        transform: open ? "translateX(0)" : "translateX(-100%)",
        transition: "transform 220ms ease, box-shadow 220ms ease",
        display: "flex", flexDirection: "column",
        overflow: "hidden",
      }}
    >
      {/* 헤더 */}
      <div style={{
        padding: "12px 14px",
        borderBottom: "1px solid var(--line)",
        display: "flex", alignItems: "center", justifyContent: "flex-start",
      }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: "var(--ink)" }}>
          AI 관제 도우미 세션
        </div>
      </div>

      {/* 새 대화 + 검색 */}
      <div style={{ padding: "10px 12px", borderBottom: "1px solid var(--line-soft)", display: "flex", flexDirection: "column", gap: 8 }}>
        <button
          onClick={() => { if (onNew) onNew(); }}
          style={{
            display: "flex", alignItems: "center", justifyContent: "center", gap: 6,
            width: "100%", padding: "8px 10px", borderRadius: 8,
            background: "var(--brand)", color: "#fff", border: "none",
            fontSize: 12, fontWeight: 700, cursor: "pointer",
            transition: "filter 140ms",
          }}
          onMouseEnter={(e) => { e.currentTarget.style.filter = "brightness(1.08)"; }}
          onMouseLeave={(e) => { e.currentTarget.style.filter = "none"; }}
        >
          <Icons.plus size={13} /> 새 대화
        </button>
        <div style={{
          display: "flex", alignItems: "center", gap: 6,
          padding: "6px 9px", borderRadius: 8,
          background: "var(--bg-elev)", border: "1px solid var(--line)",
        }}>
          <Icons.search size={12} />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="세션·대화 내용 검색"
            style={{
              flex: 1, border: "none", outline: "none", background: "transparent",
              fontSize: 12, color: "var(--ink)",
            }}
          />
          {query && (
            <button
              onClick={() => setQuery("")}
              title="검색 지우기"
              style={{ background: "transparent", border: "none", color: "var(--ink-4)", cursor: "pointer", padding: 0, display: "grid", placeItems: "center" }}
            ><Icons.close size={11} /></button>
          )}
        </div>
      </div>

      {/* 목록 */}
      <div className="scroll" style={{ flex: 1, overflow: "auto" }}>
        {/* 고정 — 문의 채널 (항상 최상단): 상담원 문의. (개발자 문의는 '시원팀 공개문의'로 통합되어 제거됨) */}
        {!searchMode && (
          <div style={{ display: "flex", gap: 8, padding: "8px 12px", borderBottom: "1px solid var(--line)" }}>
            {[
              { ch: "admin",     icon: "📩",   img: "/avatars/agent.png",     title: "상담원 문의", tip: "문의 · 버그 신고 — 관리자에게 전달" },
            ].map((it) => {
              const active = inquiryActive === it.ch;
              return (
                <button
                  key={it.ch}
                  type="button"
                  title={it.tip}
                  onClick={() => onOpenInquiry && onOpenInquiry(it.ch)}
                  style={{
                    flex: 1, cursor: "pointer",
                    display: "flex", alignItems: "center", justifyContent: "center", gap: 6,
                    padding: "7px 8px", borderRadius: 9,
                    background: active ? "var(--brand-wash)" : "var(--bg-elev)",
                    border: "1px solid " + (active ? "var(--brand)" : "var(--line)"),
                    transition: "background 140ms, border-color 140ms",
                  }}
                  onMouseOver={(e) => { if (!active) e.currentTarget.style.borderColor = "var(--brand)"; }}
                  onMouseOut={(e)  => { if (!active) e.currentTarget.style.borderColor = "var(--line)"; }}
                >
                  <img src={it.img} alt="" style={{ width: 22, height: 22, borderRadius: "50%", objectFit: "cover", flexShrink: 0 }} />
                  <span style={{ fontSize: 11.5, fontWeight: 700, whiteSpace: "nowrap", color: active ? "var(--brand)" : "var(--ink)" }}>{it.title}</span>
                </button>
              );
            })}
          </div>
        )}
        {loading && (
          <div style={{ padding: 14, fontSize: 11, color: "var(--ink-4)" }}>불러오는 중...</div>
        )}
        {!loading && !searchMode && (!sessions || sessions.length === 0) && (
          <div style={{ padding: 14, fontSize: 11, color: "var(--ink-4)" }}>저장된 세션 없음</div>
        )}
        {searchMode && displaySessions.length === 0 && (
          <div style={{ padding: 14, fontSize: 11, color: "var(--ink-4)" }}>
            {searchReady ? "검색 결과 없음" : "검색 중…"}
          </div>
        )}
        {!loading && grouped && Object.entries(grouped).map(([label, items]) => {
          if (items.length === 0) return null;
          return (
            <div key={label}>
              {label !== "__flat" && (
                <div style={{
                  position: "sticky", top: 0, zIndex: 1,
                  display: "flex", alignItems: "center", justifyContent: "center", gap: 6,
                  padding: "7px 12px",
                  fontSize: 11.5, fontWeight: 800, letterSpacing: "0.01em",
                  color: "var(--ink)",
                  background: "var(--chat-group-bg)",
                  borderBottom: "1px solid var(--line)",
                }}>
                  {label === "고정됨" && <Icons.pin size={11} color="var(--err)" />}
                  <span>{label}</span>
                  <span style={{
                    position: "absolute", right: 12, top: "50%", transform: "translateY(-50%)",
                    fontSize: 10, fontWeight: 700, color: "var(--ink-4)",
                    background: "var(--bg-elev)", border: "1px solid var(--line)",
                    borderRadius: 999, padding: "0 7px", lineHeight: "16px",
                  }}>{items.length}</span>
                </div>
              )}
              {items.map((s) => {
                const isActive = s.id === activeSessionId;
                const isEditing = s.id === editingId;
                const dt = s.updated_at ? new Date(s.updated_at) : null;
                let dtLabel = "";
                if (dt) {
                  const tz = { timeZone: "Asia/Seoul" };
                  const time = dt.toLocaleTimeString("ko-KR", { ...tz, hour: "numeric", minute: "2-digit" });
                  const sameDay = dt.toLocaleDateString("en-CA", tz) === new Date().toLocaleDateString("en-CA", tz);
                  dtLabel = sameDay ? time : `${dt.toLocaleDateString("en-US", { ...tz, month: "numeric", day: "numeric" })} ${time}`;
                }
                return (
                  <div
                    key={s.id}
                    onClick={() => { if (!isEditing) { onPick(s.id); onClose(); } }}
                    className="session-row"
                    style={{
                      padding: "9px 12px",
                      borderBottom: "1px solid var(--line)",
                      background: isActive ? "rgba(79,70,229,0.08)" : "transparent",
                      cursor: isEditing ? "default" : "pointer",
                      display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8,
                    }}
                    onMouseOver={(e) => { if (!isActive && !isEditing) e.currentTarget.style.background = "var(--bg-elev)"; }}
                    onMouseOut={(e)  => { if (!isActive && !isEditing) e.currentTarget.style.background = "transparent"; }}
                  >
                    <div style={{ flex: 1, minWidth: 0 }}>
                      {isEditing ? (
                        <input
                          autoFocus
                          value={editText}
                          onChange={(e) => setEditText(e.target.value)}
                          onClick={(e) => e.stopPropagation()}
                          onKeyDown={(e) => {
                            if (e.key === "Enter") { e.preventDefault(); commitEdit(s.id); }
                            else if (e.key === "Escape") { e.preventDefault(); setEditingId(null); }
                          }}
                          onBlur={() => commitEdit(s.id)}
                          maxLength={60}
                          style={{
                            width: "100%", boxSizing: "border-box",
                            padding: "3px 6px", borderRadius: 6,
                            border: "1px solid var(--brand)", outline: "none",
                            background: "var(--bg-elev)", color: "var(--ink)",
                            fontSize: 12, fontWeight: 600,
                          }}
                        />
                      ) : (
                        <>
                          <div style={{
                            fontSize: 12, fontWeight: isActive ? 700 : 500,
                            color: isActive ? "var(--brand)" : "var(--ink)",
                            overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                          }}>
                            {searchMode ? highlightTerm(s.title || `세션 #${s.id}`, term) : (s.title || `세션 #${s.id}`)}
                          </div>
                          <div style={{ fontSize: 9, color: "var(--ink-4)", marginTop: 2 }}>
                            {dtLabel} · 메시지 {s.messageCount || 0}
                          </div>
                          {searchMode && s.matchSnippet && (
                            <div style={{
                              fontSize: 9.5, color: "var(--ink-4)", marginTop: 3, lineHeight: 1.4,
                              overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                            }}>
                              {highlightTerm(snippetAround(s.matchSnippet, term), term)}
                            </div>
                          )}
                        </>
                      )}
                    </div>
                    {isEditing ? (
                      <button
                        onMouseDown={(e) => { e.preventDefault(); e.stopPropagation(); commitEdit(s.id); }}
                        title="저장 (Enter)"
                        style={{ background: "transparent", border: "none", color: "var(--brand)", cursor: "pointer", padding: 2, display: "grid", placeItems: "center" }}
                      ><Icons.check size={13} /></button>
                    ) : (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          if (menu && menu.id === s.id) { setMenu(null); return; }
                          const r = e.currentTarget.getBoundingClientRect();
                          const top = Math.min(r.bottom + 4, window.innerHeight - 178);
                          setMenu({ id: s.id, s, pinned: s.pinned, top, left: r.right - 170 });
                        }}
                        title="더보기"
                        aria-label="세션 메뉴"
                        style={{
                          background: menu && menu.id === s.id ? "var(--bg-sunk)" : "transparent",
                          border: "none", color: "var(--ink-3)", cursor: "pointer",
                          padding: "0 6px", height: 24, borderRadius: 6,
                          fontSize: 18, lineHeight: 1, opacity: 0.65, flexShrink: 0,
                        }}
                        onMouseOver={(e) => { e.currentTarget.style.opacity = 1; e.currentTarget.style.background = "var(--bg-sunk)"; }}
                        onMouseOut={(e) => { e.currentTarget.style.opacity = 0.65; e.currentTarget.style.background = (menu && menu.id === s.id) ? "var(--bg-sunk)" : "transparent"; }}
                      >⋯</button>
                    )}
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>

      {/* 세션 행 ⋯ 메뉴 (고정 / 이름 바꾸기 / 삭제) */}
      {menu && createPortal((
        <>
          <div onClick={() => setMenu(null)} style={{ position: "fixed", inset: 0, zIndex: 50 }} />
          <div style={{
            position: "fixed", top: menu.top, left: Math.max(8, menu.left), zIndex: 51,
            minWidth: 172, padding: 5,
            background: "var(--bg-elev)", border: "1px solid var(--line)", borderRadius: 12,
            boxShadow: "0 14px 36px -10px rgba(15,23,42,0.4)",
          }}>
            {[
              { ic: <Icons.pin size={13} color={menu.pinned ? "var(--brand)" : "var(--ink-3)"} />, label: menu.pinned ? "고정 해제" : "상단 고정", act: () => onPin && onPin(menu.id) },
              { ic: <Icons.pencil size={13} color="var(--ink-3)" />, label: "이름 바꾸기", act: () => beginEdit(menu.s) },
            ].map((it, i) => (
              <button
                key={i}
                onClick={() => { it.act(); setMenu(null); }}
                onMouseOver={(e) => { e.currentTarget.style.background = "var(--bg-sunk)"; }}
                onMouseOut={(e) => { e.currentTarget.style.background = "transparent"; }}
                style={{
                  display: "flex", alignItems: "center", gap: 9, width: "100%", textAlign: "left",
                  padding: "8px 11px", border: "none", background: "transparent", cursor: "pointer",
                  borderRadius: 8, fontSize: 12.5, fontWeight: 600, color: "var(--ink)",
                }}
              >{it.ic}{it.label}</button>
            ))}
            <div style={{ height: 1, background: "var(--line-soft)", margin: "4px 6px" }} />
            <button
              onClick={() => { const id = menu.id; setMenu(null); onDelete && onDelete(id); }}
              onMouseOver={(e) => { e.currentTarget.style.background = "rgba(220,38,38,0.08)"; }}
              onMouseOut={(e) => { e.currentTarget.style.background = "transparent"; }}
              style={{
                display: "flex", alignItems: "center", gap: 9, width: "100%", textAlign: "left",
                padding: "8px 11px", border: "none", background: "transparent", cursor: "pointer",
                borderRadius: 8, fontSize: 12.5, fontWeight: 700, color: "#dc2626",
              }}
            ><Icons.close size={13} color="#dc2626" />삭제</button>
          </div>
        </>
      ), document.body)}
    </div>
    </>
  );
}

// C1. 메시지 day-divider — 같은 세션 안에서 날짜 변경 지점 표시.
//   라벨: "5월 26일 (수)" 가운데 + 좌우 회색 라인.
function DayDivider({ dateKey }) {
  if (!dateKey) return null;
  const d = new Date(dateKey + "T00:00:00");
  if (isNaN(d.getTime())) return null;
  const y = d.getFullYear();
  const m = d.getMonth() + 1;
  const day = d.getDate();
  const dow = "일월화수목금토"[d.getDay()];
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 8,
      padding: "6px 4px 2px",
      fontSize: 10, fontWeight: 700, color: "var(--ink-4)",
      letterSpacing: "0.04em",
    }}>
      <div style={{ flex: 1, height: 1, background: "var(--line-soft)" }} />
      <span>{y}년 {m}월 {day}일 ({dow})</span>
      <div style={{ flex: 1, height: 1, background: "var(--line-soft)" }} />
    </div>
  );
}

// 추천 질문 — 입력창 위 팝업에 세로 목록으로. ↑/↓ 하이라이트(부모 onKeyDown) + Enter 선택, 클릭·호버 지원.
// 항목(items)은 부모가 보유 — 화살표 네비 시 부모가 선택 prompt 를 알아야 하기 때문.
function QuickPrompts({ items = [], highlightIndex = 0, onHover, onPick, disabled }) {
  const activeRef = useRef(null);
  // 키보드 ↑/↓ 로 하이라이트가 화면 밖 항목으로 이동하면 팝업이 함께 스크롤되도록 (block:nearest = 보이면 그대로, 벗어나면 최소 스크롤)
  useEffect(() => { activeRef.current?.scrollIntoView({ block: "nearest" }); }, [highlightIndex]);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      {items.map((item, idx) => {
        const active = idx === highlightIndex;
        const Icon = item.icon || Icons.sparkle;
        return (
          <button
            key={idx}
            ref={active ? activeRef : null}
            type="button"
            onClick={() => !disabled && onPick(item)}
            onMouseEnter={() => onHover && onHover(idx)}
            disabled={disabled}
            style={{
              display: "flex", alignItems: "center", gap: 11, width: "100%",
              padding: "9px 10px", borderRadius: 9, border: "none", textAlign: "left",
              background: active ? "var(--brand-wash)" : "transparent",
              cursor: disabled ? "not-allowed" : "pointer", opacity: disabled ? 0.5 : 1,
              transition: "background 70ms ease-out",
            }}
          >
            {/* 아이콘 → 요약 단어 → 상세 설명 (가로 순) */}
            <span style={{ flexShrink: 0, display: "grid", placeItems: "center", color: active ? "var(--brand)" : "var(--ink-3)", transition: "color 70ms ease-out" }}>
              {item.img
                ? <img src={item.img} alt="" style={{ width: 22, height: 22, borderRadius: "50%", objectFit: "cover", display: "block" }} />
                : <Icon size={18} />}
            </span>
            <span style={{ fontSize: 13, fontWeight: 700, flexShrink: 0, whiteSpace: "nowrap", color: active ? "var(--brand)" : "var(--ink)" }}>
              {item.title}
            </span>
            <span style={{ fontSize: 12, color: "var(--ink-3)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", minWidth: 0 }}>
              {item.desc}
            </span>
          </button>
        );
      })}
    </div>
  );
}

// 간단 inline 마크다운 파서: **굵게**, `코드`, [추정] 노란 배지
//  - 외부 의존성 없이 React 노드 배열 반환
//  - LLM 이 자주 쓰는 굵게 강조 (** **) 만 잘 처리하면 충분
//  - [추정] 은 시스템 프롬프트 응답 규칙 2번 (자문 Q5 반영) — 도구 데이터로
//    확인 안 된 결론에 LLM 이 붙이는 라벨. 노란 배지로 시각적 분리.
function renderInlineMD(text) {
  if (!text) return null;
  const tokens = [];
  // 패턴: **굵게**  |  `코드`  |  [추정]
  const re = /(\*\*([^*]+)\*\*|`([^`]+)`|\[추정\])/g;
  let last = 0, m, key = 0;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) tokens.push(text.slice(last, m.index));
    if (m[2] != null) {
      tokens.push(<strong key={key++} style={{ fontWeight: 800 }}>{m[2]}</strong>);
    } else if (m[3] != null) {
      tokens.push(
        <code key={key++} style={{
          fontFamily: "JetBrains Mono, ui-monospace, monospace",
          fontSize: "0.92em",
          padding: "1px 5px",
          borderRadius: 4,
          background: "rgba(0,0,0,0.06)",
        }}>{m[3]}</code>
      );
    } else {
      // [추정] 배지 — 도구 데이터 미확인 결론 시각화
      tokens.push(
        <span key={key++} style={{
          display: "inline-block",
          padding: "1px 6px",
          marginRight: 3,
          fontSize: "0.78em",
          fontWeight: 700,
          letterSpacing: "0.02em",
          background: "rgba(245,158,11,0.18)",
          color: "#b45309",
          border: "1px solid rgba(245,158,11,0.45)",
          borderRadius: 4,
          verticalAlign: "baseline",
        }}>추정</span>
      );
    }
    last = m.index + m[0].length;
  }
  if (last < text.length) tokens.push(text.slice(last));
  return tokens;
}

function ChatMessage({ message, botAvatar = "/chatbot.png", botLabel = "AI 관제 도우미", onReply, hideTime = false, onAuthorClick }) {
  const isAi = message.role === "ai";
  const [hov, setHov] = useState(false);
  return (
    <div
      onMouseEnter={() => onReply && setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
      display: "flex",
      flexDirection: isAi ? "row" : "row-reverse",
      gap: 8,
      alignItems: isAi ? "flex-start" : "flex-end",
    }}>
      {isAi && (
        <div className={message.human ? "neon-green" : undefined} onClick={onAuthorClick} title={onAuthorClick ? "프로필 보기" : undefined} style={{
          width: 34, height: 34, borderRadius: "50%",
          background: "linear-gradient(135deg, #4f46e5, #8b83ff)",
          color: "#fff",
          display: "grid", placeItems: "center",
          flexShrink: 0,
          marginTop: 1,
          overflow: "hidden",
          cursor: onAuthorClick ? "pointer" : "default",
        }}>
          <img src={botAvatar} alt="AI" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
        </div>
      )}
      <div style={{ maxWidth: "min(85%, 360px)", display: "flex", flexDirection: "column", gap: 3 }}>
        {isAi && (
          <div onClick={onAuthorClick} title={onAuthorClick ? "프로필 보기" : undefined} style={{ fontSize: 11, fontWeight: 600, color: "var(--ink-3)", paddingLeft: 2, marginBottom: 1, cursor: onAuthorClick ? "pointer" : "default" }}>
            {botLabel}
          </div>
        )}
        <div style={{
          padding: "8px 11px",
          background: isAi ? "var(--chat-ai-bg)" : "var(--chat-user-bg)",
          color: isAi ? "var(--chat-ai-fg)" : "#FFFFFF",
          border: "none",
          borderRadius: isAi ? "18px 18px 18px 4px" : "18px 18px 4px 18px",
          fontSize: 12.5,
          lineHeight: 1.55,
          whiteSpace: "pre-wrap",
          wordBreak: "break-word",
          boxShadow: isAi ? "none" : "0 6px 14px rgba(79,70,229,0.18)",
          ...(isAi && message.streaming && !message.text && !(message.toolCalls?.length)
            ? { alignSelf: "flex-start", width: "fit-content", padding: "9px 12px" }
            : {}),
        }}>
          {/* 답장 인용 — 카톡식 (원문 일부를 말풍선 상단에) */}
          {message.quote && (
            <div style={{
              borderLeft: `3px solid ${isAi ? "rgba(0,0,0,0.18)" : "rgba(255,255,255,0.55)"}`,
              paddingLeft: 8, marginBottom: 6, fontSize: 11.5, lineHeight: 1.4,
              opacity: 0.72, maxHeight: 46, overflow: "hidden",
              whiteSpace: "pre-wrap", wordBreak: "break-word",
            }}>{message.quote}</div>
          )}
          {/* 도구 호출 칩 (function calling) — 스트리밍 중이거나 호출이력 있을 때 표시
              상태 분기:
                · 스트리밍 中 + 마지막 칩 = 보라 + pulse (조회 중)
                · 스트리밍 中 + 그 외 칩 = 회색 + ✓ prefix (직전 호출 완료)
                · 스트리밍 완료 후 = 회색 + ✓ prefix (모두 완료, 흔적) */}
          {isAi && Array.isArray(message.toolCalls) && message.toolCalls.length > 0 && (
            <div style={{
              display: "flex", flexWrap: "wrap", rowGap: 2, columnGap: 12,
              marginBottom: message.text ? 6 : 0,
            }}>
              {(() => {
                // 같은 카테고리 도구 호출은 하나로 묶고 횟수 표시
                //   (14× "AI 도구 호출" → "AI 도구 14회 호출"). 진행 중인 마지막 호출의 카테고리만 pulse.
                const groups = [];
                const at = {};
                const lastIdx = message.toolCalls.length - 1;
                message.toolCalls.forEach((tc, i) => {
                  const cat = TOOL_CATEGORY[tc.name] || tc.name;
                  if (at[cat] == null) { at[cat] = groups.length; groups.push({ cat, count: 0, active: false }); }
                  groups[at[cat]].count += 1;
                  if (message.streaming && i === lastIdx) groups[at[cat]].active = true;
                });
                return groups.map((g, idx) => (
                  <span key={idx} style={{
                    display: "inline-flex", alignItems: "center", gap: 4,
                    fontSize: 10.5, lineHeight: 1.3, fontWeight: 700,
                    color: g.active ? "var(--brand)" : "var(--ok)",
                    whiteSpace: "nowrap",
                    transition: "color 200ms",
                  }}>
                    {g.active
                      ? <span style={{ width: 5, height: 5, borderRadius: "50%", background: "var(--brand)", animation: "blink-fade 1.2s ease-in-out infinite", flexShrink: 0 }} />
                      : <Icons.check size={11} color="var(--ok)" />}
                    <span>{g.cat + " 도구 " + (g.count > 1 ? g.count + "회 호출" : "호출")}</span>
                  </span>
                ));
              })()}
            </div>
          )}
          {/* AI 응답 생성 중 인디케이터 — streaming + 아직 텍스트·도구 X */}
          {isAi && message.streaming && !message.text && !(message.toolCalls?.length) ? (
            <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
              <span style={{ width: 5, height: 5, borderRadius: "50%", background: "var(--ink-3)", animation: "pulse-dot 1.2s 0s infinite" }} />
              <span style={{ width: 5, height: 5, borderRadius: "50%", background: "var(--ink-3)", animation: "pulse-dot 1.2s 0.2s infinite" }} />
              <span style={{ width: 5, height: 5, borderRadius: "50%", background: "var(--ink-3)", animation: "pulse-dot 1.2s 0.4s infinite" }} />
            </span>
          ) : (
            <>
              {renderInlineMD(message.text)}
              {message.streaming && (
                <span style={{
                  display: "inline-block",
                  width: 2, height: 14, marginLeft: 3, borderRadius: 1,
                  verticalAlign: "text-bottom",
                  background: "var(--ink-3)", opacity: 0.7,
                  animation: "blink 0.9s step-start infinite",
                }} />
              )}
            </>
          )}
          {message.image && (
            <img
              src={message.image}
              alt="첨부 이미지"
              loading="lazy"
              style={{
                display: "block", maxWidth: "100%", maxHeight: 240,
                marginTop: (message.text || message.toolCalls?.length) ? 6 : 0,
                borderRadius: 10, border: "1px solid rgba(0,0,0,0.10)",
              }}
            />
          )}
        </div>
        <div style={{
          fontSize: 9, color: "#94A3B8",
          textAlign: isAi ? "left" : "right",
          paddingLeft: isAi ? 4 : 0,
          paddingRight: isAi ? 0 : 4,
          display: "flex",
          justifyContent: isAi ? "space-between" : "flex-end",
          gap: 8,
        }}>
          {!hideTime && <span>{to12h(message.time)}</span>}
          {/* 모델 · 토큰 · 걸린시간 — 시간과 분리된 생성 메타 그룹 */}
          {isAi && message.meta && !message.streaming && (() => {
            const parts = [];
            if (message.meta.model) parts.push(message.meta.model);
            if (message.meta.tokens) {
              const p = Number(message.meta.tokens.prompt) || 0;
              const c = Number(message.meta.tokens.completion) || 0;
              const nf = (n) => n.toLocaleString("en-US");
              if (p > 0)      parts.push(`${nf(p + c)}tok`);
              else if (c > 0) parts.push(`${nf(c)}tok`);
            }
            if (message.meta.elapsedMs != null) parts.push(`${(message.meta.elapsedMs / 1000).toFixed(1)}s`);
            if (message.meta.fallback) parts.push(message.meta.fallback);
            return parts.length ? <span style={{ opacity: 0.45 }}>{parts.join(" · ")}</span> : null;
          })()}
        </div>
      </div>
      {onReply && (
        <button
          type="button"
          onClick={(e) => { e.stopPropagation(); onReply(message); }}
          title="답장"
          style={{
            alignSelf: "center", flexShrink: 0,
            width: 26, height: 26, borderRadius: "50%",
            display: "grid", placeItems: "center", border: "1px solid var(--line)",
            background: "var(--bg-elev)", color: "var(--ink-3)", cursor: "pointer",
            fontSize: 13, lineHeight: 1,
            opacity: hov ? 1 : 0, transition: "opacity 140ms",
            pointerEvents: hov ? "auto" : "none",
          }}
        >↩</button>
      )}
    </div>
  );
}

function ChatTyping({ botAvatar = "/chatbot.png" }) {
  return (
    <div style={{ display: "flex", gap: 8, alignItems: "flex-end" }}>
      <div style={{
        width: 34, height: 34, borderRadius: "50%",
        background: "linear-gradient(135deg, #4f46e5, #8b83ff)",
        color: "#fff",
        display: "grid", placeItems: "center",
        flexShrink: 0,
        overflow: "hidden",
      }}>
        <img src={botAvatar} alt="AI" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
      </div>
      <div style={{
        padding: "10px 14px",
        background: "var(--bg-elev)",
        border: "1px solid var(--line)",
        borderRadius: 12,
        borderBottomLeftRadius: 4,
        display: "flex", gap: 4, alignItems: "center",
      }}>
        <span style={{ width: 5, height: 5, borderRadius: "50%", background: "var(--ink-3)", animation: "pulse-dot 1.2s 0s infinite" }} />
        <span style={{ width: 5, height: 5, borderRadius: "50%", background: "var(--ink-3)", animation: "pulse-dot 1.2s 0.2s infinite" }} />
        <span style={{ width: 5, height: 5, borderRadius: "50%", background: "var(--ink-3)", animation: "pulse-dot 1.2s 0.4s infinite" }} />
      </div>
    </div>
  );
}

function fmtTime(d) {
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}:${String(d.getSeconds()).padStart(2, "0")}`;
}

function useLogStream(externalEvents = []) {
  // 초기값: 빈 배열 — 부팅 하드코딩 메시지 제거 (실데이터가 polling 으로 즉시 채워짐, 5/30)
  const [lines, setLines] = useState([]);
  const processedIds = useRef(new Set());
  const latestTsRef  = useRef(null);   // 백엔드 polling 증분용

  // (A) 페이지 진입 시 App.jsx 가 생성한 즉시 이벤트 추가
  useEffect(() => {
    if (!externalEvents || externalEvents.length === 0) return;
    const fresh = externalEvents.filter((e) => !processedIds.current.has(e.id));
    if (fresh.length === 0) return;
    fresh.forEach((e) => processedIds.current.add(e.id));
    setLines((prev) => [...prev.slice(-200), ...fresh]);
  }, [externalEvents]);

  // (B) 30초 polling — /api/log-events 영구 저장 데이터 누적
  useEffect(() => {
    let aborted = false;
    const poll = async () => {
      try {
        const url = latestTsRef.current
          ? `/api/log-events?after=${encodeURIComponent(latestTsRef.current)}&limit=100`
          : `/api/log-events?limit=100`;
        const r = await fetch(url);
        if (!r.ok || aborted) return;
        const d = await r.json();
        if (!d?.ok || aborted) return;
        // 증분 cursor — 서버가 준 nextCursor(KST DATETIME 문자열)를 파싱 없이 그대로 사용.
        // (e.ts 를 new Date 로 재파싱하면 타임존 변환 오차로 매번 9시간치를 중복 조회함)
        if (d.nextCursor) latestTsRef.current = d.nextCursor;
        if (!Array.isArray(d.events) || d.events.length === 0) return;
        const fresh = d.events.filter((e) => !processedIds.current.has(e.id));
        if (fresh.length === 0) return;
        fresh.forEach((e) => processedIds.current.add(e.id));
        // 시간순 (오래된 → 최신) 으로 누적
        const sorted = fresh.slice().sort((x, y) => new Date(x.ts).getTime() - new Date(y.ts).getTime());
        setLines((prev) => [...prev.slice(-200), ...sorted]);
      } catch { /* silent — 다음 주기에 재시도 */ }
    };
    poll();                                  // 즉시 1회
    const t = setInterval(poll, 30000);      // 30초 주기
    return () => { aborted = true; clearInterval(t); };
  }, []);

  return lines;
}

function buildTrendPath(mse, threshold) {
  const thW = threshold || 0.409;
  const thA = thW * 1.5;
  const H = 140, W = 640, N = 21;
  const toY = (v) => Math.max(3, Math.min(H - 3, H - Math.max(0, Math.min(1, v)) * H));
  const startV = Math.max(0.02, mse * 0.12);
  const pts = Array.from({ length: N }, (_, i) => {
    const t = i / (N - 1);
    const ease = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
    const v = startV + (mse - startV) * ease;
    const noise = (Math.sin(i * 2.31 + mse * 7.3) * 0.018 + Math.cos(i * 1.71 + mse * 3.7) * 0.012) * mse;
    return [Math.round(t * W), Math.round(toY(v + noise))];
  });
  const lineD = pts.map(([x, y], i) => `${i === 0 ? "M" : "L"} ${x} ${y}`).join(" ");
  return {
    lineD,
    areaD: lineD + ` L ${W} ${H} L 0 ${H} Z`,
    lastX: pts[N - 1][0],
    lastY: pts[N - 1][1],
    yW: toY(thW),
    yA: toY(thA),
    thW, thA,
  };
}

// AnalysisModal 폐기 (5/26) — AI 탐지 카드 클릭은 챗봇 메시지 푸쉬로 대체.
// App.jsx 에서도 import + 호출 제거됨. 호환을 위해 빈 컴포넌트 export 유지 (안 쓰이지만).
export function AnalysisModal() { return null; }

// ────────────────────────────────────────────────
// 방식전위 트렌드 차트 (6시간, 30분 간격 × 13점)
// 안전 범위: -850mV ~ -1200mV (국내 음극방식 기준)
// ────────────────────────────────────────────────
function VoltTrendChart({ item }) {
  if (item.status === "offline" || item.volt == null || item.volt === 0) {
    return (
      <div style={{
        height: 160, display: "flex", alignItems: "center", justifyContent: "center",
        color: "var(--ink-4)", fontSize: 12, fontFamily: "JetBrains Mono",
      }}>
        통신 두절 — 데이터 없음
      </div>
    );
  }

  const volt  = item.volt;
  const seed  = Math.abs(item.id ?? 1) + 1;
  const N     = 13;                // 30분 × 13 = 6시간
  const SAFE_HI = -850;            // 방식 최소 기준 (이보다 양극이면 위험)
  const SAFE_LO = -1200;           // 과방식 기준

  // 차트 레이아웃
  const CX = 50, CY = 12, CW = 390, CH = 110;
  const vTop = -150, vBot = -1450; // Y축 표시 범위
  const toY = (v) =>
    CY + Math.max(0, Math.min(CH, (vTop - v) / (vTop - vBot) * CH));
  const toX = (i) => CX + (i / (N - 1)) * CW;

  // 트렌드 데이터 생성 — 현재 volt 값으로 수렴하는 곡선
  const startVolt =
    item.status === "normal"
      ? volt + Math.sin(seed * 0.7) * 20
      : Math.max(-1100, Math.min(-880, -950 + Math.sin(seed * 0.7) * 40));

  const pts = Array.from({ length: N }, (_, i) => {
    const t    = i / (N - 1);
    const ease = t * t * (3 - 2 * t);
    const v    = startVolt + (volt - startVolt) * ease;
    const noise = Math.sin(i * 2.31 + seed * 0.73) * 10
                + Math.cos(i * 1.71 + seed * 1.13) * 6;
    return Math.round(v + noise);
  });

  const pathD = pts
    .map((v, i) => `${i === 0 ? "M" : "L"} ${toX(i).toFixed(1)} ${toY(v).toFixed(1)}`)
    .join(" ");
  const areaD = `${pathD} L ${toX(N - 1).toFixed(1)} ${(CY + CH).toFixed(1)} L ${toX(0).toFixed(1)} ${(CY + CH).toFixed(1)} Z`;

  // 현재 전위 상태색
  const color = volt > SAFE_HI ? "var(--err)" : volt < SAFE_LO ? "var(--warn)" : "var(--ok)";
  const stopC = volt > SAFE_HI ? "#ef4444"    : volt < SAFE_LO ? "#f59e0b"    : "#10b981";

  // 임계선 Y 좌표
  const ySafeHi = toY(SAFE_HI); // -850mV
  const ySafeLo = toY(SAFE_LO); // -1200mV
  const gradId  = `vg-${seed}`;

  // Y축 레이블
  const yLabels = [-300, -500, -850, -1000, -1200, -1400];
  // X축 시간 레이블
  const xLabels = [
    { i: 0, t: "6h전" }, { i: 2, t: "5h" }, { i: 4, t: "4h" },
    { i: 6, t: "3h"  }, { i: 8, t: "2h" }, { i: 10, t: "1h" }, { i: 12, t: "현재" },
  ];

  const lastX = toX(N - 1);
  const lastY = toY(volt);

  return (
    <svg viewBox={`0 0 ${CX + CW + 8} ${CY + CH + 30}`} style={{ width: "100%", height: "100%" }}>
      <defs>
        <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={stopC} stopOpacity="0.28" />
          <stop offset="100%" stopColor={stopC} stopOpacity="0" />
        </linearGradient>
      </defs>

      {/* 구역 배경 (위험 / 안전 / 과방식) */}
      <rect x={CX} y={CY}       width={CW} height={ySafeHi - CY}          fill="rgba(239,68,68,0.06)" />
      <rect x={CX} y={ySafeHi}  width={CW} height={ySafeLo - ySafeHi}     fill="rgba(16,185,129,0.07)" />
      <rect x={CX} y={ySafeLo}  width={CW} height={CY + CH - ySafeLo}     fill="rgba(245,158,11,0.06)" />

      {/* 임계선 */}
      <line x1={CX} y1={ySafeHi} x2={CX + CW} y2={ySafeHi}
        stroke="var(--err)"  strokeWidth="1" strokeDasharray="4 3" opacity="0.55" />
      <line x1={CX} y1={ySafeLo} x2={CX + CW} y2={ySafeLo}
        stroke="var(--warn)" strokeWidth="1" strokeDasharray="4 3" opacity="0.55" />
      {/* 임계값 레이블 */}
      <text x={CX + CW + 3} y={ySafeHi + 3}  fontSize="7" fontFamily="JetBrains Mono" fill="var(--err)"  fontWeight="700">-850</text>
      <text x={CX + CW + 3} y={ySafeLo + 3}  fontSize="7" fontFamily="JetBrains Mono" fill="var(--warn)" fontWeight="700">-1200</text>

      {/* X축 그리드 & 레이블 */}
      {xLabels.map(({ i, t }) => (
        <g key={i}>
          <line x1={toX(i)} y1={CY} x2={toX(i)} y2={CY + CH}
            stroke="var(--line-soft)" strokeWidth="0.5" />
          <text x={toX(i)} y={CY + CH + 14}
            fontSize="8" fontFamily="JetBrains Mono" fill="var(--ink-4)"
            textAnchor="middle">
            {t}
          </text>
        </g>
      ))}

      {/* Y축 레이블 */}
      {yLabels.map((v) => (
        <text key={v} x={CX - 4} y={toY(v) + 3}
          fontSize="7.5" fontFamily="JetBrains Mono" fill="var(--ink-4)"
          textAnchor="end">
          {v}
        </text>
      ))}
      <text x={10} y={CY + CH / 2} fontSize="7.5" fontFamily="JetBrains Mono"
        fill="var(--ink-4)" textAnchor="middle"
        transform={`rotate(-90, 10, ${CY + CH / 2})`}>
        mV
      </text>

      {/* 면적 */}
      <path d={areaD} fill={`url(#${gradId})`} />
      {/* 추이선 */}
      <path d={pathD} fill="none" stroke={color} strokeWidth="1.8" strokeLinejoin="round" />

      {/* 현재 포인트 */}
      <circle cx={lastX} cy={lastY} r="4" fill={color} />
      <circle cx={lastX} cy={lastY} r="8" fill="none" stroke={color} strokeWidth="1.2" opacity="0.45">
        <animate attributeName="r" values="4;10;4" dur="2s" repeatCount="indefinite" />
        <animate attributeName="opacity" values="0.6;0;0.6" dur="2s" repeatCount="indefinite" />
      </circle>
      {/* 현재값 말풍선 */}
      <rect x={lastX - 26} y={lastY - 20} width="52" height="14" rx="4" fill={color} />
      <text x={lastX} y={lastY - 10} fontSize="8" fontFamily="JetBrains Mono" fontWeight="700"
        fill="#fff" textAnchor="middle">
        {volt}mV
      </text>
    </svg>
  );
}

// 사이드바 — 마커 popup 클릭 시 지도 영역 안 우측에 슬라이드 인 (5/26 사용자 결정).
// 백드롭 없음 — 지도/챗봇 보면서 동시 확인 가능.
// AI 분석 / 위험도 / 기여도 등 모두 제거 — 챗봇 패널이 그 역할 담당.
function DashboardEquipmentDrawer({ item, onClose, onDetailRequest, closing }) {
  const MIN_W = 420;   // 최대 너비는 지도 셀 폭까지 동적(onBarDown에서 계산)
  const [w, setW]   = useState(MIN_W);     // 패널 너비 (좌측 바 드래그로 리사이즈)
  const [tx, setTx] = useState(MIN_W);     // 패널 translateX (시작: 화면 밖 우측)
  const [resizing, setResizing] = useState(false);
  const resizeRef  = useRef(null);
  const mountedRef = useRef(false);
  // 마운트 시 슬라이드 인
  useEffect(() => { const id = requestAnimationFrame(() => setTx(0)); return () => cancelAnimationFrame(id); }, []);
  // closing prop 변화 → 슬라이드 아웃(닫기) / 인(단말 swap 복귀). 초기 마운트는 스킵.
  useEffect(() => {
    if (!mountedRef.current) { mountedRef.current = true; return; }
    setTx(closing ? w + 60 : 0);   // 현재 너비 + 그림자 여유만큼 우측으로 슬라이드 아웃
  }, [closing]);

  if (!item) return null;
  const c = statusChip(item.status);
  const lastMeasuredText = item.updatedAt
    ? new Date(item.updatedAt).toLocaleString("ko-KR", { timeZone: "Asia/Seoul", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" })
    : null;

  // 좌측 바: 드래그 = 손을 따라오는 동작감 + 놓으면 끝까지 스냅(좌→풀, 우→기본), 클릭 = 닫기
  const onBarDown = (e) => {
    if (closing || e.button === 2) return;
    const bar = e.currentTarget;
    const cell = bar.parentElement?.parentElement;                      // 바 → 패널 → 지도 셀
    const maxW = cell ? Math.max(MIN_W, cell.clientWidth - 12) : 1100;  // 지도 셀 폭까지 확장 허용
    try { bar.setPointerCapture(e.pointerId); } catch {}               // 즉시 캡처 → 좁은 바 밖으로 나가도 move 수신
    resizeRef.current = { startX: e.clientX, startW: w, active: false, maxW, lastDx: 0 };
  };
  const onBarMove = (e) => {
    const d = resizeRef.current;
    if (!d) return;
    const dx = e.clientX - d.startX;
    d.lastDx = dx;
    if (!d.active) {
      if (Math.abs(dx) < 4) return;   // 4px 미만 = 아직 클릭으로 간주
      d.active = true; setResizing(true);
    }
    if (d.startW <= MIN_W && dx > 0) {
      setW(MIN_W);                                              // 중간상태에서 우로 끌기 → 닫는 동작
      setTx(dx);                                                // 패널을 우측으로 밀어냄(동작감)
    } else {
      setTx(0);
      setW(Math.min(d.maxW, Math.max(MIN_W, d.startW - dx)));   // 드래그 중 손을 따라옴
    }
  };
  const onBarUp = () => {
    const d = resizeRef.current;
    resizeRef.current = null;
    if (!d) return;
    setResizing(false);                                         // 트랜지션 복귀
    if (d.active && d.lastDx < 0) {                             // 좌로 끌기 → 최대로 펼침
      requestAnimationFrame(() => setW(d.maxW));
      return;
    }
    // 클릭 또는 우로 끌기 → 한 단계 접기 (최대→중간, 중간→닫기)
    if (d.startW > MIN_W) requestAnimationFrame(() => setW(MIN_W));   // 최대 → 중간
    else onClose();                                                  // 중간 → 닫기
  };

  return (
    <div className="glass-panel" style={{
      position: "absolute", right: 0, top: 0, bottom: 0, width: w, zIndex: 30,
      border: "1px solid var(--line)",
      borderRadius: 16,
      overflow: "hidden",
      boxShadow: "-12px 0 30px -12px rgba(0,0,0,0.22)",
      display: "flex", flexDirection: "column",
      transform: `translateX(${tx}px)`,
      transition: resizing ? "none" : "transform 260ms cubic-bezier(0.22,1,0.36,1), width 280ms cubic-bezier(0.22,1,0.36,1)",
    }}>
      {/* 좌측 가운데 닫기 바 — 클릭하면 사이드바 닫힘 (사실상 닫기 버튼) */}
      <div
        onPointerDown={onBarDown}
        onPointerMove={onBarMove}
        onPointerUp={onBarUp}
        title="왼쪽으로 끌기: 펼치기 · 오른쪽으로 끌기·클릭: 접기/닫기"
        style={{
          position: "absolute", left: 5, top: "50%", transform: "translateY(-50%)",
          width: 7, height: 64, borderRadius: 999,
          background: "var(--ink-3)", opacity: resizing ? 0.7 : 0.45,
          cursor: "ew-resize",
          userSelect: "none", touchAction: "none", zIndex: 5,
          transition: "opacity 140ms",
        }}
      />
      <div style={{
        padding: 18, borderBottom: "1px solid var(--line-soft)",
      }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "start", gap: 8 }}>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
              <span className="mono" style={{ fontSize: 16, fontWeight: 800, color: ({ "정상": "var(--ok)", "관찰": "var(--warn)", "이상": "#dc2626" })[c.ko] || (c.fg === "#fff" ? c.bg : c.fg) }}>{item.deviceId}</span>
              <span style={{
                fontSize: 12, fontWeight: 800,
                color: ({ "정상": "var(--ok)", "관찰": "var(--warn)", "이상": "#dc2626" })[c.ko] || (c.fg === "#fff" ? c.bg : c.fg),
              }}>
                {c.ko}
              </span>
            </div>
            <div className="mono" style={{ fontSize: 11, color: "var(--ink-3)" }}>{item.facilityId} · {item.zone}</div>
            {item.location && <div style={{ fontSize: 12, color: "var(--ink-2)", marginTop: 6 }}>{item.location}</div>}
          </div>
          {/* 상단 우측: 상세 분석 pill(보조 액션) + 닫기 */}
          <div style={{ display: "flex", alignItems: "center", gap: 8, flexShrink: 0 }}>
            {onDetailRequest && (
              <button
                className="attn-shine"
                onClick={() => onDetailRequest(item.deviceId)}
                title="AI 관제 도우미가 이 단말을 분석·요약해 드려요"
                style={{
                  display: "inline-flex", alignItems: "center", gap: 5,
                  padding: "7px 14px", borderRadius: "16px 16px 4px 16px",
                  background: "var(--chat-user-bg)",
                  border: "1px solid var(--brand)",
                  color: "#fff",
                  fontSize: 11, fontWeight: 700,
                  cursor: "pointer",
                  whiteSpace: "nowrap",
                  boxShadow: "0 2px 7px -1px rgba(79,70,229,0.45)",
                  transition: "all 140ms ease",
                }}
                onMouseEnter={(e) => { e.currentTarget.style.filter = "brightness(0.93)"; e.currentTarget.style.boxShadow = "0 5px 12px -2px rgba(79,70,229,0.5)"; }}
                onMouseLeave={(e) => { e.currentTarget.style.filter = "brightness(1)"; e.currentTarget.style.boxShadow = "0 2px 7px -1px rgba(79,70,229,0.45)"; }}
              >
                <img src="/chatbot.png" alt="" style={{ width: 16, height: 16, borderRadius: "50%", objectFit: "cover", flexShrink: 0 }} />
                상세 분석
              </button>
            )}
          </div>
        </div>
      </div>
      <div className="scroll" style={{ padding: 18, overflowY: "auto", flex: 1 }}>
        <AiAnalysis key={item.deviceId} item={item} />
        {item.volt != null && (
          <>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 8 }}>
              <div style={{ fontSize: 11, fontWeight: 700, color: "var(--ink-3)" }}>실시간 측정값</div>
              {lastMeasuredText && (
                <div className="mono" style={{ fontSize: 10, color: "var(--ink-4)" }}>
                  마지막 측정 {lastMeasuredText}
                </div>
              )}
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 8, marginBottom: 8 }}>
              {[
                { l: "방식전위", v: `${item.volt}mV` },
                { l: "AC 유입", v: `${(item.ac ?? 0).toLocaleString()}mV` },
                { l: "희생전류",  v: `${item.sacrificial ?? 0}mA` },
                { l: "온도",     v: `${item.temp ?? "-"}°C` },
                { l: "습도", v: `${item.hum ?? "-"}%` },
                { l: "통신품질", v: item.commOk ? `${item.commDbm}dBm` : "단절", a: !item.commOk ? "var(--err)" : null },
              ].map((s) => (
                <div key={s.l} style={{ padding: "10px 12px", borderRadius: 8, background: "var(--bg-sunk)", border: "1px solid var(--line-soft)" }}>
                  <div style={{ fontSize: 10, color: "var(--ink-3)" }}>{s.l}</div>
                  <div className="mono" style={{ fontSize: 17, fontWeight: 700, marginTop: 2, color: s.a || "var(--ink)" }}>{s.v}</div>
                </div>
              ))}
            </div>
          </>
        )}

        {/* 트렌드 차트 제거 (5/26 사용자 결정 — 전체 장비 페이지에서 조회 가능) */}

        {item.status === "offline" && (
          <div style={{
            padding: "12px 14px", borderRadius: 10,
            background: "rgba(100,116,139,0.10)",
            border: "1px solid rgba(100,116,139,0.25)",
            fontSize: 12, color: "var(--ink-2)",
          }}>
            통신 두절 단말 — 실시간 측정 데이터 없음
            {item.updatedAt && <><br/>마지막 측정: {new Date(item.updatedAt).toLocaleString("ko-KR", { timeZone: "Asia/Seoul" })}</>}
          </div>
        )}
      </div>
    </div>
  );
}

export function Dashboard({ user = null, mapStyle, setMapStyle, theme, autoPlay = true, equipment = [], markers = [], anomalies = [], watch = [], commOutage = [], aiEvents = [], demoMode = false, logOpen = false, onLogClose, onToggleLog }) {
  const [activeKpi, setActiveKpi] = useState(null);
  const [drawer, setDrawer] = useState(null);
  const [focused, setFocused] = useState(null); // {lat, lng, node, ts}
  const [fitTrigger, setFitTrigger] = useState(0); // 카운터: 변할 때마다 지도 fit
  const [boundsRequest, setBoundsRequest] = useState(null); // { coords, ts } — 챗봇이 여러 노드 언급 시
  const [showNormal, setShowNormal] = useState(true);        // 지도 위 정상 핀 토글
  const weather = useWeather();                              // 챗봇 컨텍스트용 날씨
  const [autoKpiSec, setAutoKpiSec] = useState(0);           // AI 자동 필터 잔여 초 (0 = 비활성)
  const autoKpiTimer = useRef(null);                         // setTimeout 핸들
  const autoKpiTick  = useRef(null);                         // setInterval 핸들 (1초)
  // logOpen / ESC 처리는 App.jsx 로 이동 (헤더 알약 버튼이 토글, 5/26)

  const counts = useMemo(() => {
    const c = { all: equipment.length, normal: 0, critical: 0, anomaly: 0, warn: 0, offline: 0 };
    equipment.forEach((e) => { if (c[e.status] !== undefined) c[e.status]++; });
    return c;
  }, [equipment]);

  const tableData = useMemo(() => {
    if (!activeKpi || activeKpi === "all") return equipment;
    return equipment.filter((e) => e.status === activeKpi);
  }, [activeKpi, equipment]);

  const filteredMarkers = useMemo(() => {
    let base = markers;
    if (activeKpi && activeKpi !== "all") {
      base = devicesToMarkers(equipment.filter((e) => e.status === activeKpi));
    }
    // 정상 핀 토글 (활성 KPI 가 'normal' 일 때는 토글 무시 — 명시적 선택)
    if (!showNormal && activeKpi !== "normal") {
      base = base.filter((m) => m.status !== "normal");
    }
    return base;
  }, [activeKpi, equipment, markers, showNormal]);

  // 노드 ID 로 지도 포커싱 (장비 lat/lng 우선, markers fallback)
  const focusByNode = (node) => {
    if (!node) return;
    const eq = equipment.find((e) => e.deviceId === node);
    if (eq && eq.lat != null && eq.lng != null) {
      setFocused({ lat: eq.lat, lng: eq.lng, node, ts: Date.now() });
      return;
    }
    const mk = markers.find((m) => m.node === node);
    if (mk && mk.lat != null && mk.lng != null) {
      setFocused({ lat: mk.lat, lng: mk.lng, node, ts: Date.now() });
      return;
    }
    // 둘 다 없으면 콘솔 경고 (개발 편의)
    console.warn(`[focusByNode] 노드 위치를 찾을 수 없습니다: ${node}`);
  };

  // 표 row 클릭: AI 탐지 카드와 동일 동작으로 통일 — 지도 포커스 + 우측 상세 사이드바 열기
  const handleRowClick = (eq) => {
    if (!eq || !eq.deviceId) return;
    const full = equipment.find((e) => e.deviceId === eq.deviceId) || eq;
    selectDevice(full);
    focusByNode(eq.deviceId);
  };

  // 챗봇에 자동 전송할 메시지 (AI 탐지 카드 클릭 → "TB24-XXX 분석" 푸쉬, 5/26)
  const [chatAutoMessage, setChatAutoMessage] = useState(null);
  // 마커 popup 클릭 → 지도 영역 안 사이드바에 표시할 단말 (5/26)
  const [sidebarDevice, setSidebarDevice] = useState(null);
  const [sidebarClosing, setSidebarClosing] = useState(false);  // 닫힘 슬라이드아웃 애니메이션 중
  const [deselectTick, setDeselectTick] = useState(0);          // 지도 선택 마커 해제 신호
  // 사이드바 닫기 — 우측 슬라이드아웃 후 언마운트 + 지도 마커 선택 해제
  const closeSidebar = () => {
    setDeselectTick(Date.now());
    setSidebarClosing(true);
    setTimeout(() => { setSidebarDevice(null); setSidebarClosing(false); }, 230);
  };
  const sidebarDeviceRef = useRef(null);
  sidebarDeviceRef.current = sidebarDevice;
  // 단말 선택 — 이미 열려있고 '다른' 단말이면 빠른 슬라이드아웃→인 으로 바뀐 느낌
  const selectDevice = (eq) => {
    if (!eq) return;
    const cur = sidebarDeviceRef.current;
    if (cur && cur.deviceId !== eq.deviceId) {
      setSidebarClosing(true);
      setTimeout(() => { setSidebarDevice(eq); setSidebarClosing(false); }, 160);
    } else {
      setSidebarDevice(eq);
    }
  };

  // AI 탐지 카드 클릭 (5/26 변경): 사이드바 열기 + 지도 포커스. 챗봇 푸쉬는 사이드바 안 '상세 분석' 버튼으로 옮김.
  const handleAnalyze = (item) => {
    if (!item || !item.node) return;
    const eq = equipment.find((e) => e.deviceId === item.node);
    if (eq) selectDevice(eq);
    focusByNode(item.node);
  };

  // 사이드바 안 '상세 분석' 버튼 클릭 → 챗봇 패널에 자동 메시지 푸쉬
  const handleDetailRequest = (deviceId) => {
    if (!deviceId) return;
    setChatAutoMessage(`${deviceId} 의 현재 상태, 위험 요인, 권장 조치를 정리해줘`);
  };

  // 지도 마커 클릭: 사이드바 열기/갱신 + 지도 포커스 (AI카드·표·팝업과 동일 통일). popup 도 Leaflet 자동 표시.
  const handleMarkerClick = (m) => {
    if (!m || !m.node) return;
    const eq = equipment.find((x) => x.deviceId === m.node);
    if (eq) selectDevice(eq);
    focusByNode(m.node);
  };

  // popup 클릭 delegation — popup HTML 의 data-popup-node 감지
  useEffect(() => {
    const onClick = (e) => {
      const target = e.target.closest && e.target.closest("[data-popup-node]");
      if (!target) return;
      const node = target.getAttribute("data-popup-node");
      const eq = equipment.find((x) => x.deviceId === node);
      if (eq) setSidebarDevice(eq);
    };
    document.addEventListener("click", onClick);
    return () => document.removeEventListener("click", onClick);
  }, [equipment]);

  // 사용자가 KPI 직접 클릭: 활성 토글 + 지도 fit + AI 자동 타이머 취소 + 열린 사이드바 닫기
  const handleKpiClick = (newActive) => {
    cancelAutoKpi();
    setActiveKpi(newActive);
    setFitTrigger(Date.now());
    if (sidebarDeviceRef.current) closeSidebar();   // KPI 전환 시 선택 단말/사이드바 자동 닫기
  };

  // AI 자동 필터 취소 (타이머 + 카운트다운 정리)
  const cancelAutoKpi = () => {
    if (autoKpiTimer.current) { clearTimeout(autoKpiTimer.current); autoKpiTimer.current = null; }
    if (autoKpiTick.current)  { clearInterval(autoKpiTick.current);  autoKpiTick.current  = null; }
    setAutoKpiSec(0);
  };

  // 챗봇 응답에서 status 추출 → 자동 KPI 활성 (30초 후 자동 해제)
  const handleAutoKpi = (kpi) => {
    // 타이머/카운트다운 제거 — 챗봇이 status 언급 시 해당 KPI 필터만 적용(자동 복귀·칩 없음)
    cancelAutoKpi();
    setActiveKpi(kpi);
    setFitTrigger(Date.now());
  };

  // 언마운트 시 타이머 정리
  useEffect(() => () => cancelAutoKpi(), []);

  // 챗봇 응답에서 노드 ID 들이 언급되면 지도 자동 zoom
  //  - 1개: flyTo + 팝업
  //  - 2개+: fitBounds (모두 한 화면에)
  const fitToNodes = (nodes) => {
    if (!nodes || nodes.length === 0) return;
    const coords = nodes
      .map((n) => equipment.find((e) => e.deviceId === n))
      .filter((eq) => eq && eq.lat != null && eq.lng != null)
      .map((eq) => [eq.lat, eq.lng, eq.deviceId]);
    if (coords.length === 0) return;
    if (coords.length === 1) {
      setFocused({ lat: coords[0][0], lng: coords[0][1], node: coords[0][2], ts: Date.now() });
    } else {
      setBoundsRequest({ coords: coords.map(([lat, lng]) => [lat, lng]), ts: Date.now() });
    }
  };

  const lines = useLogStream(aiEvents);

  return (
    <>
      {/*
        통합 3-row grid:
          row 1 (auto)             — 좌: KPI                / 우: AI 탐지(span 1-2)
          row 2 (1.2fr, min 360)   — 좌: 지도               / 우: AI 탐지(이어짐)
          row 3 (1fr, min 280)     — 좌: 표+로그            / 우: AI 조치 권고
        AI 탐지가 row 1+2 를 span 하여 우측 분기점이 좌측의 지도/표 분기점과 정확히 일치.
      */}
      {/*
        2026-05-22 옴니솔루션 피드백 반영 — 챗봇 영역 확장 (사이즈 작다는 의견):
          row 1 (112px 고정)       — 좌: KPI                / 우: AI 챗봇 (span 1-3, 거대)
          row 2 (1.2fr, min 300)   — 좌: 지도               /     AI 챗봇 이어짐
          row 3 (1fr, min 240)     — 좌: 표 + AI 탐지       /     AI 챗봇 이어짐
        실시간 시스템 로그 → 우상단 floating 버튼 토글로 분리 (드로어).
        AI 탐지 → 옛 LogPanel 자리(좌측 row 3 우측 영역) 로 이동.
        AI 챗봇 → 우측 전체 column 차지 (옛 AIPanels + 옛 ChatPanel 자리 합쳐서).
      */}
      <div className="dashboard-grid" style={{
        position: "absolute", left: 0, right: 0, top: 0, bottom: 0,
        padding: "var(--dash-pad) var(--dash-pad) 18px",   /* 하단만 18px 고정 — 카드 바로 아래 카피라이트 푸터가 붙어 보이도록 간격 축소 */
        display: "grid",
        gridTemplateColumns: "minmax(0, 1fr) var(--dash-chat-col)",
        gridTemplateRows: "var(--dash-kpi-row) minmax(var(--dash-map-min), 1.2fr) minmax(var(--dash-bottom-min), 1fr)",
        gap: "var(--dash-gap)",
        minHeight: 0,
        overflow: "auto",
      }}>
        {/* (col 1, row 1) — KPI */}
        <div style={{ gridColumn: 1, gridRow: 1, minHeight: 0 }}>
          <KPIRow active={activeKpi} setActive={handleKpiClick} counts={counts} />
        </div>

        {/* (col 2, row 1~3) — AI 챗봇 (전체 우측 column · 옴니 5/22 확장) */}
        <div style={{ gridColumn: 2, gridRow: "1 / span 3", minHeight: 0 }}>
          <ChatPanel
            user={user}
            equipment={equipment}
            weather={weather}
            onBotReply={fitToNodes}
            onAutoKpi={handleAutoKpi}
            demoMode={demoMode}
            autoMessage={chatAutoMessage}
            onAutoConsumed={() => setChatAutoMessage(null)}
          />
        </div>

        {/* (col 1, row 2) — 지도 + 사이드바 (5/26 마커 popup 클릭 시 사이드바 슬라이드인) */}
        <div style={{ gridColumn: 1, gridRow: 2, minHeight: 0, position: "relative", overflow: "hidden" }}>
          <MapPanelWrap
            markers={filteredMarkers}
            onMarker={handleMarkerClick}
            mapStyle={mapStyle}
            setMapStyle={setMapStyle}
            focus={focused}
            fitTrigger={fitTrigger}
            boundsRequest={boundsRequest}
            showNormal={showNormal}
            setShowNormal={setShowNormal}
            autoKpiSec={autoKpiSec}
            onCancelAutoKpi={cancelAutoKpi}
            onMapClick={closeSidebar}
            deselectTrigger={deselectTick}
          />
          {sidebarDevice && (
            <DashboardEquipmentDrawer
              item={sidebarDevice}
              closing={sidebarClosing}
              onClose={closeSidebar}
              onDetailRequest={handleDetailRequest}
            />
          )}
        </div>

        {/* (col 1, row 3) — 표 + (AI 탐지 ⇄ 시스템 로그 swap, 5/26)
            헤더 알약 클릭 시 in-place swap. 드로어 X. */}
        <div className="dashboard-bottom-grid" style={{
          gridColumn: 1, gridRow: 3,
          display: "grid",
          gridTemplateColumns: "var(--dash-table-col) var(--dash-ai-col)",
          gap: "var(--dash-gap)", minHeight: 0,
        }}>
          {/* 좌측 셀: 전체 장비 현황 요약 ⇄ 실시간 시스템 로그 (버튼이 있는 패널 자체가 swap, 5/30) */}
          {logOpen
            ? <LogPanel lines={lines} onToggleLog={onToggleLog} />
            : <TableSummary data={tableData} onRowClick={handleRowClick} activeKpi={activeKpi} logOpen={logOpen} onToggleLog={onToggleLog} />}
          {/* 우측 셀: AI 탐지 목록 고정 */}
          <AIPanels anomalies={anomalies} watch={watch} commOutage={commOutage} onAnalyze={handleAnalyze} focusNode={sidebarDevice?.deviceId} />
        </div>
      </div>

      <DashboardEquipmentDrawer item={drawer} onClose={() => setDrawer(null)} />
    </>
  );
}
