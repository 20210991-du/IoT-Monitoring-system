import { useState } from "react";
import { Icons } from "../components/Icons.jsx";
import { signUp, ROLE_LABEL, RULES } from "../lib/authMock.js";

/* ── AI 기반 지능형 통합관제 시스템 · 회원가입 ───────────────
 *  Login.jsx 와 동일 dark · glass morphism 톤.
 *  카드 폭만 살짝 넓혀(460) 폼 항목 6개 수용.
 *  2026-05-04 신규 (mock 인증, 실 API 교체 예정).
 */

const COLORS = {
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

function Field({ icon, label, hint, error, children, focused, onFocus, onBlur }) {
  const showError = !!error;
  return (
    <div style={{ marginBottom: 14 }}>
      <div
        style={{
          display: "flex",
          alignItems: "baseline",
          justifyContent: "space-between",
          marginBottom: 6,
        }}
      >
        <div style={{ fontSize: 12, fontWeight: 600, color: COLORS.inkSoft, letterSpacing: "-0.01em" }}>
          {label}
        </div>
        {hint && !showError && (
          <div style={{ fontSize: 11, color: COLORS.inkMuted }}>{hint}</div>
        )}
      </div>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          padding: "0 14px",
          height: 44,
          borderRadius: 12,
          background: "rgba(255,255,255,0.035)",
          border: `1px solid ${
            showError ? COLORS.err : focused ? COLORS.brand : COLORS.inkFaint
          }`,
          boxShadow: focused
            ? `0 0 0 3px rgba(139,131,255,0.14)`
            : showError
              ? "0 0 0 3px rgba(251,113,133,0.12)"
              : "none",
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
      {showError && (
        <div style={{ fontSize: 11, color: COLORS.err, marginTop: 5, letterSpacing: "-0.01em" }}>
          {error}
        </div>
      )}
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
  padding: "11px 0",
  fontFamily: "inherit",
};

const selectStyle = {
  ...inputStyle,
  appearance: "none",
  WebkitAppearance: "none",
  cursor: "pointer",
};

export function SignUp({ onSuccess, onBackToLogin }) {
  const [form, setForm] = useState({
    id: "",
    pw: "",
    pw2: "",
    name: "",
    role: "operator",
  });
  const [focus, setFocus] = useState(null);
  const [errors, setErrors] = useState({});
  const [globalError, setGlobalError] = useState("");
  const [loading, setLoading] = useState(false);
  const [done, setDone] = useState(false);

  const update = (k) => (e) => {
    const v = e?.target ? e.target.value : e;
    setForm((s) => ({ ...s, [k]: v }));
    if (errors[k]) setErrors((s) => ({ ...s, [k]: undefined }));
    if (globalError) setGlobalError("");
  };

  const submit = (e) => {
    e && e.preventDefault();
    setLoading(true);
    setErrors({});
    setGlobalError("");

    // 약간의 지연으로 인증중 느낌 (mock)
    setTimeout(() => {
      const res = signUp(form);
      if (!res.ok) {
        setLoading(false);
        if (res.field) setErrors({ [res.field]: res.error });
        else setGlobalError(res.error);
        return;
      }
      setDone(true);
      setTimeout(() => onSuccess && onSuccess(form.id), 2400);
    }, 500);
  };

  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        background: COLORS.bg0,
        color: COLORS.ink,
        overflow: "auto",
        display: "flex",
        flexDirection: "column",
      }}
    >
      {/* ── 배경 ── */}
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

      {/* ── Header ── */}
      <header
        style={{
          position: "relative",
          zIndex: 1,
          padding: "32px 56px 0",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          flexShrink: 0,
        }}
      >
        <div style={{ fontSize: 16, fontWeight: 700, letterSpacing: "-0.02em", color: COLORS.ink }}>
          AI 기반 지능형 통합관제 시스템
        </div>
        <a
          onClick={(e) => { e.preventDefault(); onBackToLogin && onBackToLogin(); }}
          href="#"
          style={{ fontSize: 13, color: COLORS.inkSoft, fontWeight: 500, cursor: "pointer" }}
        >
          ← 로그인으로
        </a>
      </header>

      {/* ── Main ── */}
      <main
        style={{
          position: "relative",
          zIndex: 1,
          flex: 1,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          padding: "40px 56px",
          minHeight: 0,
        }}
      >
        <div
          style={{
            width: 460,
            padding: 36,
            borderRadius: 20,
            background: COLORS.glassBg,
            border: `1px solid ${COLORS.glassBorder}`,
            backdropFilter: "blur(24px) saturate(1.3)",
            WebkitBackdropFilter: "blur(24px) saturate(1.3)",
            boxShadow: "0 40px 80px -30px rgba(0,0,0,0.8), 0 0 0 1px rgba(255,255,255,0.03) inset",
            animation: "slide-in-up 700ms ease both",
          }}
        >
          <div style={{ marginBottom: 22 }}>
            <div
              style={{
                fontSize: 24,
                fontWeight: 800,
                color: COLORS.ink,
                letterSpacing: "-0.025em",
                marginBottom: 6,
              }}
            >
              운영자 계정 등록
            </div>
            <div style={{ fontSize: 13, color: COLORS.inkSoft }}>
              관제 시스템 접속을 위한 계정을 생성합니다.
            </div>
          </div>

          {done ? (
            <div
              style={{
                padding: "20px 8px 8px",
                textAlign: "center",
                animation: "slide-in-up 360ms ease both",
              }}
            >
              <div
                style={{
                  width: 56,
                  height: 56,
                  borderRadius: "50%",
                  background: "linear-gradient(135deg, #f59e0b, #f97316)",
                  margin: "0 auto 16px",
                  display: "grid",
                  placeItems: "center",
                  boxShadow: "0 12px 30px -8px rgba(245,158,11,0.55)",
                }}
              >
                <Icons.clock size={28} color="#fff" />
              </div>
              <div style={{ fontSize: 18, fontWeight: 700, marginBottom: 8 }}>
                가입 신청이 접수되었습니다
              </div>
              <div style={{ fontSize: 13, color: COLORS.inkSoft, lineHeight: 1.6, marginBottom: 18 }}>
                관리자 승인 후 로그인하실 수 있습니다.<br />
                승인 완료 여부는 관리자에게 문의해 주세요.
              </div>
              <div
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: 6,
                  padding: "6px 12px",
                  borderRadius: 999,
                  background: "rgba(245,158,11,0.12)",
                  border: "1px solid rgba(245,158,11,0.32)",
                  color: "#fbbf24",
                  fontSize: 11,
                  fontWeight: 700,
                  letterSpacing: "0.02em",
                  marginBottom: 22,
                }}
              >
                <span style={{ width: 5, height: 5, borderRadius: "50%", background: "#fbbf24" }} />
                승인 대기 중
              </div>
              <div style={{ fontSize: 12, color: COLORS.inkMuted }}>
                로그인 화면으로 이동합니다…
              </div>
            </div>
          ) : (
            <form onSubmit={submit}>
              <Field
                icon={<Icons.user size={16} />}
                label="운영자 ID"
                hint="2~20자"
                error={errors.id}
                focused={focus === "id"}
              >
                <input
                  value={form.id}
                  onChange={update("id")}
                  onFocus={() => setFocus("id")}
                  onBlur={() => setFocus(null)}
                  style={inputStyle}
                  placeholder="예: kim.operator"
                  autoComplete="username"
                />
              </Field>

              <Field
                icon={<Icons.id_card size={16} />}
                label="이름"
                error={errors.name}
                focused={focus === "name"}
              >
                <input
                  value={form.name}
                  onChange={update("name")}
                  onFocus={() => setFocus("name")}
                  onBlur={() => setFocus(null)}
                  style={inputStyle}
                  placeholder="홍길동"
                  autoComplete="name"
                />
              </Field>

              <Field
                icon={<Icons.briefcase size={16} />}
                label="역할"
                error={errors.role}
                focused={focus === "role"}
              >
                <select
                  value={form.role}
                  onChange={update("role")}
                  onFocus={() => setFocus("role")}
                  onBlur={() => setFocus(null)}
                  style={selectStyle}
                >
                  {Object.entries(ROLE_LABEL).map(([v, label]) => (
                    <option key={v} value={v} style={{ background: COLORS.bg1, color: COLORS.ink }}>
                      {label}
                    </option>
                  ))}
                </select>
                <span style={{ color: COLORS.inkMuted, marginLeft: 4 }}>
                  <Icons.chevron_right size={14} style={{ transform: "rotate(90deg)" }} />
                </span>
              </Field>

              <Field
                icon={<Icons.lock size={16} />}
                label="비밀번호"
                hint="4자 이상"
                error={errors.pw}
                focused={focus === "pw"}
              >
                <input
                  type="password"
                  value={form.pw}
                  onChange={update("pw")}
                  onFocus={() => setFocus("pw")}
                  onBlur={() => setFocus(null)}
                  style={inputStyle}
                  placeholder="새 비밀번호 입력"
                  autoComplete="new-password"
                />
              </Field>

              <Field
                icon={<Icons.lock size={16} />}
                label="비밀번호 확인"
                error={errors.pw2}
                focused={focus === "pw2"}
              >
                <input
                  type="password"
                  value={form.pw2}
                  onChange={update("pw2")}
                  onFocus={() => setFocus("pw2")}
                  onBlur={() => setFocus(null)}
                  style={inputStyle}
                  placeholder="한 번 더 입력"
                  autoComplete="new-password"
                />
              </Field>

              {globalError && (
                <div
                  style={{
                    fontSize: 12,
                    color: COLORS.err,
                    marginBottom: 12,
                    padding: "8px 12px",
                    borderRadius: 8,
                    background: "rgba(251,113,133,0.08)",
                    border: "1px solid rgba(251,113,133,0.22)",
                  }}
                >
                  {globalError}
                </div>
              )}

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
                  marginTop: 6,
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
                    등록 중...
                  </>
                ) : (
                  <>
                    계정 등록
                    <Icons.chevron_right size={16} />
                  </>
                )}
              </button>

              <div
                style={{
                  marginTop: 18,
                  paddingTop: 18,
                  borderTop: `1px solid ${COLORS.inkFaint}`,
                  textAlign: "center",
                  fontSize: 13,
                  color: COLORS.inkSoft,
                }}
              >
                이미 계정이 있으신가요?{" "}
                <a
                  onClick={(e) => { e.preventDefault(); onBackToLogin && onBackToLogin(); }}
                  href="#"
                  style={{ color: COLORS.brand, fontWeight: 600, cursor: "pointer" }}
                >
                  로그인
                </a>
              </div>
            </form>
          )}
        </div>
      </main>

      {/* ── Footer ── */}
      <footer
        style={{
          position: "relative",
          zIndex: 1,
          padding: "0 56px 24px",
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
