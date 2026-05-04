// App chrome: header, sub-nav, emergency banner, footer status bar
import { useState, useEffect, useRef } from "react";
import { Icons } from "./Icons.jsx";
import { getBeep, setBeep, playBeep } from "../lib/userPrefs.js";
import { ProfileModal } from "./UserModals.jsx";

// 역할별 라벨/그라디언트 (Header 아바타 · 드롭다운 톤)
const ROLE_META = {
  admin:    { label: "관리자", grad: "linear-gradient(135deg, #8b5cf6, #6d28d9)", glow: "rgba(139,92,246,0.45)" },
  operator: { label: "관제사", grad: "linear-gradient(135deg, #38bdf8, #0284c7)", glow: "rgba(56,189,248,0.45)" },
  viewer:   { label: "뷰어",   grad: "linear-gradient(135deg, #94a3b8, #64748b)", glow: "rgba(148,163,184,0.4)"  },
  guest:    { label: "게스트", grad: "linear-gradient(135deg, #fbbf24, #d97706)", glow: "rgba(245,158,11,0.45)" },
};

export function useClock() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000);
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

export function Header({ onLogout, user, setUser, theme, setTheme, mapStyle, setMapStyle }) {
  const now = useClock();
  const [open, setOpen] = useState(false);          // 사용자 카드 드롭다운
  const [settingsOpen, setSettingsOpen] = useState(false); // 설정 드롭다운
  const [profileOpen, setProfileOpen]   = useState(null); // null | "profile" | "password"
  const [beep, setBeepState] = useState(() => getBeep());
  const wrapRef     = useRef(null);
  const settingsRef = useRef(null);

  // 외부 클릭 시 드롭다운들 닫기
  useEffect(() => {
    if (!open && !settingsOpen) return;
    const onDoc = (e) => {
      if (open && wrapRef.current && !wrapRef.current.contains(e.target)) setOpen(false);
      if (settingsOpen && settingsRef.current && !settingsRef.current.contains(e.target)) setSettingsOpen(false);
    };
    const onKey = (e) => {
      if (e.key === "Escape") {
        setOpen(false);
        setSettingsOpen(false);
      }
    };
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, [open, settingsOpen]);

  const toggleBeep = () => {
    const next = !beep;
    setBeepState(next);
    setBeep(next);
    if (next) {
      // 켰을 때 미리듣기
      setTimeout(() => playBeep("ok"), 60);
    }
  };

  const pickTheme = (t) => {
    setTheme && setTheme(t);
    // 드롭다운 유지 — 즉시 반영 결과 보면서 더 고를 수 있게
  };
  const pickMap = (s) => {
    setMapStyle && setMapStyle(s);
  };

  const roleMeta = ROLE_META[user?.role] || ROLE_META.operator;
  const displayName = user?.name || "사용자";
  const initial = (displayName.trim()[0] || "U").toUpperCase();

  const handleLogout = () => {
    setOpen(false);
    onLogout && onLogout();
  };

  return (
    <div style={{
      position: "absolute", left: 0, right: 0, top: 0, height: 56,
      background: "var(--bg-elev)", borderBottom: "1px solid var(--line)",
      display: "flex", alignItems: "center", justifyContent: "space-between",
      padding: "0 32px", zIndex: 50,   // SubNav(40)·Banner(35) 위로
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
        {/* ── 설정 아이콘 (클릭 → 드롭다운: 테마/비프) ── */}
        <div ref={settingsRef} style={{ position: "relative" }}>
          <button
            type="button"
            onClick={() => { setSettingsOpen((v) => !v); setOpen(false); }}
            aria-expanded={settingsOpen}
            title="설정"
            style={{
              ...iconBtn,
              background: settingsOpen ? "var(--bg-sunk)" : "transparent",
              color: settingsOpen ? "var(--brand)" : "var(--ink-3)",
              border: settingsOpen ? "1px solid var(--brand)" : "1px solid transparent",
              transition: "all 140ms ease",
              cursor: "pointer",
            }}
          >
            <Icons.settings />
          </button>

          {settingsOpen && (
            <div
              role="menu"
              style={{
                position: "absolute", right: 0, top: "calc(100% + 8px)",
                width: 280,
                background: "var(--bg-elev)",
                border: "1px solid var(--line)",
                borderRadius: 14,
                boxShadow: "0 24px 60px -12px rgba(0,0,0,0.28), 0 0 0 1px rgba(0,0,0,0.02)",
                overflow: "hidden",
                animation: "slide-in-up 180ms ease both",
                zIndex: 60,
              }}
            >
              <div style={{
                padding: "10px 14px 8px",
                fontSize: 11, fontWeight: 700, letterSpacing: "0.04em",
                color: "var(--ink-3)", textTransform: "uppercase",
                borderBottom: "1px solid var(--line)",
              }}>
                테마
              </div>
              <div style={{ padding: 8, display: "flex", gap: 6 }}>
                <ThemeChip
                  active={theme === "light"}
                  icon={<Icons.sun size={14} />}
                  label="밝게"
                  onClick={() => pickTheme("light")}
                />
                <ThemeChip
                  active={theme === "dark"}
                  icon={<Icons.moon size={14} />}
                  label="어둡게"
                  onClick={() => pickTheme("dark")}
                />
              </div>

              <div style={{
                padding: "10px 14px 8px",
                fontSize: 11, fontWeight: 700, letterSpacing: "0.04em",
                color: "var(--ink-3)", textTransform: "uppercase",
                borderTop: "1px solid var(--line)",
                borderBottom: "1px solid var(--line)",
              }}>
                지도 표시
              </div>
              <div style={{ padding: 8, display: "flex", gap: 6 }}>
                <ThemeChip
                  active={mapStyle === "light"}
                  icon={<Icons.sun size={14} />}
                  label="밝게"
                  onClick={() => pickMap("light")}
                />
                <ThemeChip
                  active={mapStyle === "dark"}
                  icon={<Icons.moon size={14} />}
                  label="어둡게"
                  onClick={() => pickMap("dark")}
                />
                <ThemeChip
                  active={mapStyle === "satellite"}
                  icon={<Icons.satellite size={14} />}
                  label="위성"
                  onClick={() => pickMap("satellite")}
                />
              </div>

              <div style={{
                padding: "10px 14px 8px",
                fontSize: 11, fontWeight: 700, letterSpacing: "0.04em",
                color: "var(--ink-3)", textTransform: "uppercase",
                borderTop: "1px solid var(--line)",
                borderBottom: "1px solid var(--line)",
              }}>
                알림
              </div>
              <div style={{ padding: 6 }}>
                <ToggleRow
                  icon={<Icons.bell size={14} />}
                  label="비프음"
                  hint={beep ? "켜짐" : "꺼짐"}
                  on={beep}
                  onClick={toggleBeep}
                />
              </div>

            </div>
          )}
        </div>

        {/* ── 사용자 카드 (클릭 → 드롭다운) ─────────────── */}
        <div ref={wrapRef} style={{ position: "relative" }}>
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            aria-expanded={open}
            title="계정 메뉴"
            style={{
              display: "flex", alignItems: "center", gap: 10,
              padding: "4px 8px 4px 14px",
              border: `1px solid ${open ? "var(--brand)" : "var(--line)"}`,
              borderRadius: 10,
              background: open ? "var(--bg-sunk)" : "transparent",
              cursor: "pointer",
              transition: "border-color 140ms ease, background 140ms ease",
            }}
          >
            <div style={{ textAlign: "right", lineHeight: 1.25 }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: "var(--ink)" }}>
                {displayName}
              </div>
              <div style={{ fontSize: 10, color: "var(--ink-3)" }}>
                {roleMeta.label}{user?.id ? ` · ${user.id}` : ""}
              </div>
            </div>
            <div style={{
              width: 32, height: 32, borderRadius: "50%",
              background: roleMeta.grad,
              display: "grid", placeItems: "center",
              color: "#fff", fontWeight: 800, fontSize: 13,
              boxShadow: `0 0 0 2px var(--bg-sunk)`,
              flexShrink: 0,
            }}>{initial}</div>
            <span style={{
              color: "var(--ink-3)",
              transform: open ? "rotate(90deg)" : "rotate(0deg)",
              transition: "transform 160ms ease",
              display: "grid", placeItems: "center",
            }}>
              <Icons.chevron_right size={14} />
            </span>
          </button>

          {open && (
            <div
              role="menu"
              style={{
                position: "absolute", right: 0, top: "calc(100% + 8px)",
                width: 280,
                background: "var(--bg-elev)",
                border: "1px solid var(--line)",
                borderRadius: 14,
                boxShadow: "0 24px 60px -12px rgba(0,0,0,0.28), 0 0 0 1px rgba(0,0,0,0.02)",
                overflow: "hidden",
                animation: "slide-in-up 180ms ease both",
                zIndex: 60,
              }}
            >
              {/* 상단 프로필 영역 */}
              <div style={{
                display: "flex", alignItems: "center", gap: 11,
                padding: "12px 14px",
                borderBottom: "1px solid var(--line)",
                background: "linear-gradient(180deg, var(--bg-sunk), transparent)",
              }}>
                <div style={{
                  width: 36, height: 36, borderRadius: "50%",
                  background: roleMeta.grad,
                  display: "grid", placeItems: "center",
                  color: "#fff", fontWeight: 800, fontSize: 14,
                  boxShadow: `0 4px 12px -3px ${roleMeta.glow}`,
                  flexShrink: 0,
                }}>
                  {initial}
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{
                    display: "flex", alignItems: "center", gap: 6,
                    fontSize: 13, fontWeight: 700, color: "var(--ink)",
                    overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                  }}>
                    <span style={{ overflow: "hidden", textOverflow: "ellipsis" }}>{displayName}</span>
                    <span style={{
                      padding: "1px 6px", borderRadius: 999,
                      fontSize: 9, fontWeight: 700, letterSpacing: "0.02em",
                      color: "#fff", background: roleMeta.grad, flexShrink: 0,
                    }}>
                      {roleMeta.label}
                    </span>
                  </div>
                  <div style={{
                    fontSize: 11, color: "var(--ink-3)", marginTop: 2,
                    overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                  }}>
                    {user?.id}
                  </div>
                </div>
              </div>

              {/* 메뉴 영역 */}
              <div style={{ padding: 6 }}>
                <MenuItem
                  icon={<Icons.user size={14} />}
                  label="내 정보"
                  onClick={() => { setOpen(false); setProfileOpen("profile"); }}
                />
                <MenuItem
                  icon={<Icons.lock size={14} />}
                  label="비밀번호 변경"
                  onClick={() => { setOpen(false); setProfileOpen("password"); }}
                />
                <div style={{ height: 1, background: "var(--line)", margin: "4px 6px" }} />
                <MenuItem
                  icon={<span style={{ display: "grid", placeItems: "center" }}>↩</span>}
                  label="로그아웃"
                  tone="err"
                  onClick={handleLogout}
                />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── 내 정보 모달 (프로필 / 비밀번호 탭 통합) ── */}
      {profileOpen && (
        <ProfileModal
          user={user}
          defaultTab={profileOpen}
          onClose={() => setProfileOpen(null)}
          onUpdate={(u) => setUser && setUser(u)}
        />
      )}
    </div>
  );
}

function ThemeChip({ active, icon, label, onClick }) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        flex: 1,
        display: "flex", alignItems: "center", justifyContent: "center", gap: 6,
        padding: "9px 10px", borderRadius: 8,
        border: `1px solid ${active ? "var(--brand)" : "var(--line)"}`,
        background: active ? "rgba(99,102,241,0.10)" : "transparent",
        color: active ? "var(--brand)" : "var(--ink-2)",
        fontSize: 12, fontWeight: 700,
        cursor: "pointer",
        transition: "all 140ms ease",
      }}
    >
      {icon}{label}
    </button>
  );
}

function ToggleRow({ icon, label, hint, on, onClick }) {
  const [hover, setHover] = useState(false);
  return (
    <button
      type="button"
      onClick={onClick}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        width: "100%",
        display: "flex", alignItems: "center", gap: 10,
        padding: "9px 10px", borderRadius: 8,
        background: hover ? "var(--bg-sunk)" : "transparent",
        border: "none",
        cursor: "pointer",
        transition: "background 120ms ease",
        textAlign: "left",
      }}
    >
      <span style={{
        width: 20, height: 20, display: "grid", placeItems: "center",
        color: on ? "var(--brand)" : "var(--ink-3)",
      }}>
        {icon}
      </span>
      <span style={{ flex: 1, fontSize: 13, fontWeight: 600, color: "var(--ink)" }}>{label}</span>
      {hint && (
        <span style={{ fontSize: 10, color: on ? "var(--brand)" : "var(--ink-4)", fontWeight: 600 }}>
          {hint}
        </span>
      )}
      {/* 토글 스위치 */}
      <span style={{
        position: "relative", width: 32, height: 18, borderRadius: 999,
        background: on ? "linear-gradient(135deg, #4f46e5, #8b83ff)" : "var(--line)",
        transition: "background 200ms ease",
        flexShrink: 0,
      }}>
        <span style={{
          position: "absolute", top: 2, left: on ? 16 : 2,
          width: 14, height: 14, borderRadius: "50%",
          background: "#fff",
          boxShadow: "0 1px 3px rgba(0,0,0,0.25)",
          transition: "left 200ms ease",
        }} />
      </span>
    </button>
  );
}

function MenuItem({ icon, label, onClick, disabled, hint, tone }) {
  const [hover, setHover] = useState(false);
  const color = disabled
    ? "var(--ink-4)"
    : tone === "err" ? "#dc2626" : "var(--ink)";
  const bg = !disabled && hover
    ? (tone === "err" ? "rgba(239,68,68,0.08)" : "var(--bg-sunk)")
    : "transparent";
  return (
    <button
      type="button"
      role="menuitem"
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onClick={disabled ? undefined : onClick}
      disabled={disabled}
      style={{
        width: "100%",
        display: "flex", alignItems: "center", gap: 10,
        padding: "9px 10px", borderRadius: 8,
        background: bg, border: "none",
        color, fontSize: 13, fontWeight: 600,
        cursor: disabled ? "not-allowed" : "pointer",
        transition: "background 120ms ease",
        textAlign: "left",
      }}
    >
      <span style={{
        width: 20, height: 20, display: "grid", placeItems: "center",
        color: disabled ? "var(--ink-4)" : tone === "err" ? "#dc2626" : "var(--ink-3)",
      }}>
        {icon}
      </span>
      <span style={{ flex: 1 }}>{label}</span>
      {hint && (
        <span style={{ fontSize: 10, color: "var(--ink-4)", fontWeight: 500 }}>
          {hint}
        </span>
      )}
    </button>
  );
}

const API_STATUS_STYLE = {
  mock:    { dot: "var(--ink-4)", text: "MOCK",      textColor: "var(--ink-4)" },
  loading: { dot: "var(--warn)", text: "연결 중…",   textColor: "var(--warn)"  },
  ok:      { dot: "var(--ok)",   text: "AI 연동됨",  textColor: "var(--ok)"    },
  error:   { dot: "var(--err)",  text: "오프라인",   textColor: "var(--err)"   },
};

export function SubNav({ tab, setTab, apiStatus = "mock", user, pendingCount = 0 }) {
  const baseTabs = [
    { k: "dashboard", ko: "대시보드" },
    { k: "equipment", ko: "전체 장비 현황" },
  ];
  const tabs = user?.role === "admin"
    ? [...baseTabs, { k: "users", ko: "관리자 페이지", badge: pendingCount }]
    : baseTabs;
  const st = API_STATUS_STYLE[apiStatus] || API_STATUS_STYLE.mock;
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
            {!!t.badge && t.badge > 0 && (
              <span style={{
                display: "inline-flex", alignItems: "center", justifyContent: "center",
                minWidth: 18, height: 18, padding: "0 5px", borderRadius: 999,
                background: "linear-gradient(135deg, #f59e0b, #f97316)",
                color: "#fff", fontSize: 10, fontWeight: 800,
                boxShadow: "0 2px 6px rgba(245,158,11,0.45)",
                animation: "pulse-dot 1.6s infinite",
              }}>
                {t.badge}
              </span>
            )}
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
        fontSize: 11, fontWeight: 700, color: st.textColor,
        fontFamily: "JetBrains Mono, monospace",
        marginRight: 16,
      }}>
        <span style={{
          width: 6, height: 6, borderRadius: "50%", background: st.dot,
          animation: apiStatus === "ok" ? "pulse-dot 2s infinite" : "none",
        }} />
        {st.text}
      </span>
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
