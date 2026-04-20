// App chrome: header, sub-nav, emergency banner, footer status bar
import { useState, useEffect } from "react";
import { Icons } from "./Icons.jsx";

export function useClock() {
  const [now, setNow] = useState(() => new Date(2026, 2, 26, 14, 42, 5));
  useEffect(() => {
    const t = setInterval(() => setNow((n) => new Date(n.getTime() + 1000)), 1000);
    return () => clearInterval(t);
  }, []);
  return now;
}

export function fmtClock(d) {
  const yr = d.getFullYear(), mo = d.getMonth() + 1, day = d.getDate();
  const h = String(d.getHours()).padStart(2, "0");
  const m = String(d.getMinutes()).padStart(2, "0");
  const s = String(d.getSeconds()).padStart(2, "0");
  return `${yr}년 ${mo}월 ${day}일 ${h}:${m}:${s}`;
}

function LogoMark() {
  return (
    <div style={{
      width: 40, height: 40, borderRadius: 10,
      background: "linear-gradient(135deg, var(--brand) 0%, #7c3aed 100%)",
      display: "grid", placeItems: "center",
      boxShadow: "0 6px 14px -4px rgba(79,70,229,0.5)",
    }}>
      {/* 통합 로고: 파이프 단면(외곽 원) + AI 신호 파형 + 중앙 감지 노드 */}
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
        <circle cx="12" cy="12" r="7.5" stroke="#fff" strokeWidth="1" opacity="0.35" />
        <path d="M4 12 Q7 8 10 12" stroke="#fff" strokeWidth="1.8" strokeLinecap="round" />
        <path d="M14 12 Q17 16 20 12" stroke="#fff" strokeWidth="1.8" strokeLinecap="round" />
        <circle cx="12" cy="12" r="2" fill="#fff" />
      </svg>
    </div>
  );
}

const iconBtn = {
  width: 32, height: 32, borderRadius: 8,
  display: "grid", placeItems: "center",
  color: "var(--ink-3)",
};

export function Header({ onLogout }) {
  const now = useClock();
  return (
    <div style={{
      position: "absolute", left: 0, right: 0, top: 0, height: 56,
      background: "var(--bg-elev)", borderBottom: "1px solid var(--line)",
      display: "flex", alignItems: "center", justifyContent: "space-between",
      padding: "0 32px", zIndex: 40,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
        <div style={{ fontSize: 17, fontWeight: 800, letterSpacing: "-0.01em" }}>
          AI 기반 지능형 통합관제 시스템
        </div>
        <div style={{ width: 1, height: 18, background: "var(--line)", margin: "0 8px" }} />
        <div className="mono" style={{ fontSize: 13, color: "var(--ink-3)" }}>
          {fmtClock(now)}
        </div>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
        <button title="알림" style={iconBtn}>
          <div style={{ position: "relative" }}>
            <Icons.bell />
            <span style={{
              position: "absolute", right: -3, top: -3, width: 8, height: 8,
              background: "var(--err)", borderRadius: "50%",
              border: "2px solid var(--bg-elev)",
            }} />
          </div>
        </button>
        <button title="설정" style={iconBtn}><Icons.settings /></button>
        <div style={{
          display: "flex", alignItems: "center", gap: 12,
          padding: "4px 6px 4px 14px", border: "1px solid var(--line)", borderRadius: 10,
        }}>
          <div style={{ textAlign: "right", lineHeight: 1.25 }}>
            <div style={{ fontSize: 12, fontWeight: 700 }}>관리자 시원팀</div>
            <div style={{ fontSize: 10, color: "var(--ink-3)" }}>운영팀 팀장</div>
          </div>
          <div onClick={onLogout} title="로그아웃" style={{
            width: 32, height: 32, borderRadius: "50%", cursor: "pointer",
            background: "linear-gradient(135deg, #f59e0b, #f97316)",
            display: "grid", placeItems: "center",
            color: "#fff", fontWeight: 800, fontSize: 12,
            boxShadow: "0 0 0 2px var(--bg-sunk)",
          }}>시원</div>
        </div>
      </div>
    </div>
  );
}

export function SubNav({ tab, setTab }) {
  const tabs = [
    { k: "dashboard", ko: "대시보드" },
    { k: "equipment", ko: "전체 장비 현황" },
  ];
  return (
    <div style={{
      position: "absolute", left: 0, right: 0, top: 56, height: 48,
      background: "var(--bg-elev)", borderBottom: "1px solid var(--line)",
      display: "flex", alignItems: "stretch", padding: "0 32px", gap: 8, zIndex: 40,
    }}>
      {tabs.map((t) => {
        const active = tab === t.k;
        return (
          <button key={t.k} onClick={() => setTab(t.k)} style={{
            position: "relative",
            padding: "0 20px", display: "flex", alignItems: "center", gap: 8,
            fontSize: 14, fontWeight: 700,
            color: active ? "var(--brand)" : "var(--ink-3)",
          }}>
            {t.ko}
            {active && (
              <span style={{
                position: "absolute", left: 0, right: 0, bottom: -1, height: 2,
                background: "var(--brand)", borderRadius: 2,
              }} />
            )}
          </button>
        );
      })}
      <div style={{ flex: 1 }} />
      <span style={{
        display: "flex", alignItems: "center", gap: 6,
        fontSize: 13, fontWeight: 600, color: "var(--ink-2)",
      }}>
        <span style={{ fontSize: 16 }}>⛅</span>
        <span>흐림</span>
        <span className="mono" style={{ color: "var(--ink-3)" }}>12°C</span>
      </span>
    </div>
  );
}

export function EmergencyBanner({ onDismiss, onOpen }) {
  return (
    <div style={{
      position: "absolute", left: 0, right: 0, top: 104, height: 40,
      background: "linear-gradient(90deg, #f59e0b 0%, #f97316 100%)",
      boxShadow: "0 4px 12px -4px rgba(245,158,11,0.5)",
      display: "flex", alignItems: "center", justifyContent: "space-between",
      padding: "0 32px", color: "#fff", zIndex: 35,
      overflow: "hidden",
    }}>
      <div style={{
        position: "absolute", inset: 0,
        background: "repeating-linear-gradient(45deg, transparent, transparent 20px, rgba(255,255,255,0.04) 20px, rgba(255,255,255,0.04) 40px)",
        pointerEvents: "none",
      }} />
      <div style={{ display: "flex", alignItems: "center", gap: 12, zIndex: 1 }}>
        <div style={{
          width: 22, height: 22,
          animation: "pulse-dot 1.4s infinite",
        }}>
          <Icons.alert size={22} />
        </div>
        <div style={{ fontSize: 14, fontWeight: 700 }}>
          2026-03-26 [경고] 제2구역 매설배관 방식전위 급격한 변동 감지 — 즉각적인 현장 점검이 필요합니다.
        </div>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 16, zIndex: 1 }}>
        <div className="mono" style={{ fontSize: 11, opacity: 0.9, letterSpacing: "0.04em" }}>
          발생 시각 14:22:05
        </div>
        <button onClick={onOpen} style={{
          padding: "4px 14px", borderRadius: 999,
          background: "rgba(255,255,255,0.22)", color: "#fff",
          fontSize: 12, fontWeight: 700,
          border: "1px solid rgba(255,255,255,0.4)",
        }}>자세히 보기</button>
        <button onClick={onDismiss} title="닫기" style={{ color: "#fff", opacity: 0.85 }}>
          <Icons.close size={16} />
        </button>
      </div>
    </div>
  );
}

export function FooterBar() {
  return (
    <div style={{
      position: "absolute", left: 0, right: 0, bottom: 0, height: 32,
      background: "var(--bg-elev)", borderTop: "1px solid var(--line)",
      display: "flex", alignItems: "center", justifyContent: "space-between",
      padding: "0 32px", zIndex: 30,
    }}>
      <div className="mono" style={{ display: "flex", alignItems: "center", gap: 24, fontSize: 10, fontWeight: 700, letterSpacing: "0.08em" }}>
        <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--brand)" }} />
          <span style={{ color: "var(--ink-3)" }}>MODEL: LSTM-AUTOENCODER V3</span>
        </span>
        <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--ink-4)" }} />
          <span style={{ color: "var(--ink-3)" }}>THRESHOLD: 0.00409</span>
        </span>
        <span style={{ color: "var(--ink-4)" }}>· GPU util 34% · INFER 12ms</span>
      </div>
      <div className="mono" style={{ display: "flex", alignItems: "center", gap: 24, fontSize: 10, fontWeight: 700, letterSpacing: "0.08em" }}>
        <span style={{ display: "flex", alignItems: "center", gap: 6, color: "var(--ok)" }}>
          <Icons.check size={10} /> SYSTEM HEALTH: NORMAL
        </span>
        <span style={{ color: "var(--ink-4)" }}>|</span>
        <span style={{ color: "var(--ink-3)" }}>SERVER: ASIA-KR-01 · 128.4.17.2</span>
      </div>
    </div>
  );
}
