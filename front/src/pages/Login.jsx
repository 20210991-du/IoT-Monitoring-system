import { useState } from "react";
import { Icons } from "../components/Icons.jsx";
import { signIn } from "../lib/authMock.js";

/* ── AI 기반 지능형 통합관제 시스템 · 로그인 ───────────────────
 *  Modern dark · glass morphism · 한글 우선 copy
 *  2026-04-20 재설계 (이전 버전은 git history 참조)
 *  Flex-based layout: header / main(left|right) / footer 3-row
 */

const COLORS = {
  // 로그인은 전체 dark 고정 — 앱 테마와 독립
  bg0: "#05070f",
  bg1: "#0c1024",
  ink: "#f1f3ff",
  inkSoft: "rgba(241,243,255,0.62)",
  inkMuted: "rgba(241,243,255,0.38)",
  inkFaint: "rgba(241,243,255,0.14)",
  brand: "#8b83ff",
  brand2: "#c084fc",
  ok: "#34d399",
  err: "#fb7185",
  glassBg: "rgba(18, 22, 48, 0.55)",
  glassBorder: "rgba(139,131,255,0.22)",
};

function Field({ icon, label, children, focused, onFocus, onBlur }) {
  return (
    <div style={{ marginBottom: 16 }}>
      <div
        style={{
          fontSize: 12,
          fontWeight: 600,
          color: COLORS.inkSoft,
          marginBottom: 8,
          letterSpacing: "-0.01em",
        }}
      >
        {label}
      </div>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          padding: "0 14px",
          height: 46,
          borderRadius: 12,
          background: "rgba(255,255,255,0.035)",
          border: `1px solid ${focused ? COLORS.brand : COLORS.inkFaint}`,
          boxShadow: focused ? `0 0 0 3px rgba(139,131,255,0.14)` : "none",
          transition: "border-color 180ms ease, box-shadow 180ms ease",
        }}
      >
        <span style={{ color: focused ? COLORS.brand : COLORS.inkMuted, display: "grid", placeItems: "center" }}>
          {icon}
        </span>
        <div style={{ flex: 1, display: "flex", alignItems: "center" }} onFocus={onFocus} onBlur={onBlur}>
          {children}
        </div>
      </div>
    </div>
  );
}

const inputStyle = {
  flex: 1,
  width: "100%",
  background: "transparent",
  border: "none",
  outline: "none",
  color: COLORS.ink,
  fontSize: 14,
  fontWeight: 500,
  padding: "12px 0",
  fontFamily: "inherit",
};

export function Login({ onLogin, onSignUp, prefillId }) {
  const [id, setId] = useState(prefillId || "admin");
  const [pw, setPw] = useState(prefillId ? "" : "11111111");
  const [remember, setRemember] = useState(true);
  const [loading, setLoading] = useState(false);
  const [focus, setFocus] = useState(null);
  const [error, setError] = useState("");
  const [errStatus, setErrStatus] = useState(null);   // null | "pending" | "rejected"
  const [failCount, setFailCount] = useState(0);      // ID 없음 / 비번 틀림 누적 (pending/rejected 제외)

  const submit = (e) => {
    e && e.preventDefault();
    setLoading(true);
    setError("");
    setErrStatus(null);
    setTimeout(() => {
      const res = signIn({ id, pw, remember });
      if (!res.ok) {
        setLoading(false);
        setError(res.error);
        setErrStatus(res.status || null);
        // pending/rejected 는 시스템 상태 문제 → 카운트 X
        // 비번 까먹은 사람을 위한 hint 발동만 카운트
        if (!res.status) setFailCount((c) => c + 1);
        return;
      }
      setFailCount(0);
      onLogin && onLogin(res.user);
    }, 600);
  };

  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        background: COLORS.bg0,
        color: COLORS.ink,
        overflow: "hidden",
        display: "flex",
        flexDirection: "column",
      }}
    >
      {/* ── 배경 · 소프트 gradient blob 2개 + 미세 노이즈 ── */}
      <div
        aria-hidden
        style={{
          position: "absolute",
          inset: 0,
          background: `
            radial-gradient(1200px 700px at 18% 22%, rgba(79,70,229,0.35), transparent 62%),
            radial-gradient(900px 600px at 82% 78%, rgba(192,132,252,0.22), transparent 60%),
            linear-gradient(180deg, ${COLORS.bg0} 0%, ${COLORS.bg1} 100%)
          `,
          zIndex: 0,
        }}
      />
      <div
        aria-hidden
        style={{
          position: "absolute",
          inset: 0,
          opacity: 0.35,
          mixBlendMode: "overlay",
          backgroundImage:
            "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  0 0 0 0.08 0'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>\")",
          zIndex: 0,
        }}
      />

      {/* ── Header : 브랜드 · 상태 ── */}
      <header
        style={{
          position: "relative",
          zIndex: 1,
          padding: "40px 56px 0",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          flexShrink: 0,
        }}
      >
        <div style={{ fontSize: 16, fontWeight: 700, letterSpacing: "-0.02em", color: COLORS.ink }}>
          AI 기반 지능형 통합관제 시스템
        </div>
      </header>

      {/* ── Main : 좌 카피 + 우 로그인 카드 ── */}
      <main
        style={{
          position: "relative",
          zIndex: 1,
          flex: 1,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 80,
          padding: "80px 120px 40px",
          minHeight: 0,
        }}
      >
        {/* ── 좌측 ── */}
        <div
          style={{
            flex: "1 1 auto",
            maxWidth: 620,
            animation: "slide-in-up 620ms ease both",
          }}
        >
          <div
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 8,
              padding: "6px 12px",
              borderRadius: 999,
              background: "rgba(139,131,255,0.12)",
              border: "1px solid rgba(139,131,255,0.25)",
              color: COLORS.brand,
              fontSize: 12,
              fontWeight: 600,
              letterSpacing: "-0.01em",
              marginBottom: 24,
            }}
          >
            <span style={{ width: 5, height: 5, borderRadius: "50%", background: COLORS.brand }} />
            호서대학교 시원팀 · 2026 캡스톤디자인
          </div>

          <h1
            style={{
              margin: 0,
              fontSize: 60,
              fontWeight: 800,
              lineHeight: 1.05,
              letterSpacing: "-0.035em",
              color: COLORS.ink,
            }}
          >
            AI 기반 지능형
            <br />
            <span
              style={{
                background: "linear-gradient(120deg, #8b83ff 0%, #c084fc 60%, #f0abfc 100%)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
              }}
            >
              통합관제 시스템
            </span>
          </h1>

          <p
            style={{
              margin: "22px 0 0",
              fontSize: 16,
              lineHeight: 1.6,
              color: COLORS.inkSoft,
              fontWeight: 400,
              maxWidth: 480,
            }}
          >
            분산된 IoT 센서 데이터를 통합 관제,
            <br />
            AI가 이상을 사전에 포착합니다.
          </p>
        </div>

        {/* ── 우측 · 로그인 카드 (glass) ── */}
        <div
          style={{
            flex: "0 0 420px",
            padding: 36,
            borderRadius: 20,
            background: COLORS.glassBg,
            border: `1px solid ${COLORS.glassBorder}`,
            backdropFilter: "blur(24px) saturate(1.3)",
            WebkitBackdropFilter: "blur(24px) saturate(1.3)",
            boxShadow: "0 40px 80px -30px rgba(0,0,0,0.8), 0 0 0 1px rgba(255,255,255,0.03) inset",
            animation: "slide-in-up 700ms ease both",
            animationDelay: "80ms",
          }}
        >
          <div style={{ marginBottom: 26 }}>
            <div
              style={{
                fontSize: 24,
                fontWeight: 800,
                color: COLORS.ink,
                letterSpacing: "-0.025em",
                marginBottom: 6,
              }}
            >
              관제 시스템 접속
            </div>
            <div style={{ fontSize: 13, color: COLORS.inkSoft }}>
              운영자 계정으로 로그인해 주세요.
            </div>
          </div>

          <form onSubmit={submit}>
            <Field
              icon={<Icons.user size={16} />}
              label="운영자 ID"
              focused={focus === "id"}
            >
              <input
                value={id}
                onChange={(e) => setId(e.target.value)}
                onFocus={() => setFocus("id")}
                onBlur={() => setFocus(null)}
                style={inputStyle}
                placeholder="ID 를 입력하세요"
                autoComplete="username"
              />
            </Field>
            <Field
              icon={<Icons.lock size={16} />}
              label="비밀번호"
              focused={focus === "pw"}
            >
              <input
                type="password"
                value={pw}
                onChange={(e) => setPw(e.target.value)}
                onFocus={() => setFocus("pw")}
                onBlur={() => setFocus(null)}
                style={inputStyle}
                placeholder="비밀번호를 입력하세요"
                autoComplete="current-password"
              />
            </Field>

            <div
              style={{
                display: "flex",
                alignItems: "center",
                marginBottom: 22,
                marginTop: 4,
              }}
            >
              <label
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  cursor: "pointer",
                  fontSize: 13,
                  color: COLORS.inkSoft,
                  userSelect: "none",
                }}
                onClick={() => setRemember(!remember)}
              >
                <span
                  style={{
                    width: 16,
                    height: 16,
                    borderRadius: 5,
                    background: remember ? "linear-gradient(135deg, #4f46e5, #8b83ff)" : "transparent",
                    border: `1px solid ${remember ? COLORS.brand : COLORS.inkFaint}`,
                    display: "grid",
                    placeItems: "center",
                    transition: "all 160ms ease",
                  }}
                >
                  {remember && <Icons.check size={10} color="#fff" />}
                </span>
                로그인 상태 유지
              </label>
            </div>

            {error && (() => {
              const isPending  = errStatus === "pending";
              const isRejected = errStatus === "rejected";
              const tone = isPending
                ? { bg: "rgba(245,158,11,0.10)", bd: "rgba(245,158,11,0.30)", fg: "#fbbf24", label: "승인 대기" }
                : isRejected
                  ? { bg: "rgba(100,116,139,0.10)", bd: "rgba(100,116,139,0.30)", fg: "#94a3b8", label: "반려" }
                  : { bg: "rgba(251,113,133,0.08)", bd: "rgba(251,113,133,0.22)", fg: COLORS.err, label: null };
              return (
                <div
                  style={{
                    fontSize: 12,
                    color: tone.fg,
                    marginBottom: 14,
                    padding: tone.label ? "10px 12px" : "8px 12px",
                    borderRadius: 8,
                    background: tone.bg,
                    border: `1px solid ${tone.bd}`,
                    lineHeight: 1.55,
                  }}
                >
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
              type="submit"
              disabled={loading}
              style={{
                width: "100%",
                height: 48,
                borderRadius: 12,
                background: loading
                  ? "rgba(79,70,229,0.45)"
                  : "linear-gradient(135deg, #4f46e5 0%, #8b83ff 100%)",
                color: "#fff",
                fontSize: 15,
                fontWeight: 700,
                letterSpacing: "-0.01em",
                boxShadow: loading
                  ? "none"
                  : "0 12px 30px -8px rgba(79,70,229,0.55), 0 0 0 1px rgba(255,255,255,0.06) inset",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: 8,
                transition: "transform 140ms ease, box-shadow 180ms ease, opacity 140ms ease",
                cursor: loading ? "not-allowed" : "pointer",
              }}
            >
              {loading ? (
                <>
                  <span
                    style={{
                      width: 16,
                      height: 16,
                      borderRadius: "50%",
                      border: "2px solid rgba(255,255,255,0.3)",
                      borderTopColor: "#fff",
                      animation: "spin 0.7s linear infinite",
                    }}
                  />
                  인증 중...
                </>
              ) : (
                <>
                  로그인
                  <Icons.chevron_right size={16} />
                </>
              )}
            </button>

            <div
              style={{
                marginTop: 18,
                paddingTop: 18,
                borderTop: `1px solid ${COLORS.inkFaint}`,
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                fontSize: 13,
                color: COLORS.inkSoft,
              }}
            >
              <span style={{
                fontSize: 12,
                color: failCount >= 3 ? "#fbbf24" : "transparent",
                fontWeight: failCount >= 3 ? 600 : 400,
                transition: "color 240ms ease",
                pointerEvents: "none",
                userSelect: "none",
              }}>
                {failCount >= 3 ? "비밀번호는 관리자에게 문의" : "·"}
              </span>
              <a
                onClick={(e) => { e.preventDefault(); onSignUp && onSignUp(); }}
                href="#"
                style={{ color: COLORS.brand, fontWeight: 600, cursor: "pointer" }}
              >
                회원가입
              </a>
            </div>
          </form>
        </div>
      </main>

      {/* ── Footer ── */}
      <footer
        style={{
          position: "relative",
          zIndex: 1,
          padding: "0 56px 28px",
          textAlign: "center",
          fontSize: 12,
          color: COLORS.inkMuted,
          letterSpacing: "-0.01em",
          flexShrink: 0,
        }}
      >
        © 2026 시원팀 · 호서대학교 컴퓨터공학부 캡스톤디자인
      </footer>
    </div>
  );
}
