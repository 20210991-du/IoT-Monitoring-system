import { useState, useEffect } from "react";
import { Icons } from "./Icons.jsx";
import {
  updateProfile,
  changePassword,
  findUser,
  ROLE_LABEL,
  STATUS_LABEL,
} from "../lib/authMock.js";

/* ── 사용자 본인 모달: 내 정보 (프로필 / 비밀번호 탭) ──────
 *  Header 드롭다운에서 두 가지 진입:
 *    - "내 정보"        → defaultTab="profile"
 *    - "비밀번호 변경"  → defaultTab="password"
 *  내부적으론 같은 모달, 탭만 다르게 시작.
 */

const ROLE_GRAD = {
  admin:    "linear-gradient(135deg, #8b5cf6, #6d28d9)",
  operator: "linear-gradient(135deg, #38bdf8, #0284c7)",
  viewer:   "linear-gradient(135deg, #94a3b8, #64748b)",
  guest:    "linear-gradient(135deg, #fbbf24, #d97706)",
};

function fmtDate(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")} ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// 내 정보 모달 (프로필 + 비밀번호 통합)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export function ProfileModal({ user, onClose, onUpdate, defaultTab = "profile" }) {
  const [tab, setTab] = useState(defaultTab); // "profile" | "password"

  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  const fullRecord = user ? findUser(user.id) : null;
  const initial = (user?.name?.trim()[0] || "U").toUpperCase();
  const grad = ROLE_GRAD[user?.role] || ROLE_GRAD.operator;

  return (
    <Backdrop onClose={onClose}>
      <div style={modalCard(520)}>
        <ModalHeader title="내 정보" onClose={onClose} />

        {/* 프로필 헤더 */}
        <div style={{
          display: "flex", alignItems: "center", gap: 14,
          padding: "16px 20px",
          borderBottom: "1px solid var(--line)",
          background: "var(--bg-sunk)",
        }}>
          <div style={{
            width: 52, height: 52, borderRadius: "50%",
            background: grad,
            display: "grid", placeItems: "center",
            color: "#fff", fontWeight: 800, fontSize: 18,
            boxShadow: "0 6px 16px -4px rgba(0,0,0,0.18)",
            flexShrink: 0,
          }}>{initial}</div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: 14, fontWeight: 700, color: "var(--ink)" }}>
              {user?.id}
            </div>
            <div style={{ fontSize: 12, color: "var(--ink-3)", marginTop: 2 }}>
              <span style={{
                display: "inline-block", padding: "2px 8px",
                borderRadius: 999, marginRight: 8,
                background: grad, color: "#fff",
                fontSize: 10, fontWeight: 700, letterSpacing: "0.02em",
              }}>
                {ROLE_LABEL[user?.role] || user?.role}
              </span>
              {STATUS_LABEL[fullRecord?.status] || "—"}
            </div>
          </div>
        </div>

        {/* 탭 */}
        <div style={{
          display: "flex", padding: "0 20px",
          borderBottom: "1px solid var(--line)",
        }}>
          <TabBtn k="profile"  cur={tab} set={setTab} icon={<Icons.user size={13} />} label="프로필" />
          <TabBtn k="password" cur={tab} set={setTab} icon={<Icons.lock size={13} />} label="비밀번호" />
        </div>

        {/* 컨텐츠 */}
        {tab === "profile" ? (
          <ProfileTab
            user={user}
            fullRecord={fullRecord}
            onClose={onClose}
            onUpdate={onUpdate}
          />
        ) : (
          <PasswordTab
            user={user}
            onClose={onClose}
          />
        )}
      </div>
    </Backdrop>
  );
}

function TabBtn({ k, cur, set, icon, label }) {
  const active = cur === k;
  return (
    <button
      onClick={() => set(k)}
      style={{
        position: "relative",
        display: "flex", alignItems: "center", gap: 6,
        padding: "12px 16px",
        background: "transparent", border: "none",
        color: active ? "var(--brand)" : "var(--ink-3)",
        fontSize: 13, fontWeight: 700, cursor: "pointer",
      }}
    >
      {icon}{label}
      {active && (
        <span style={{
          position: "absolute", left: 0, right: 0, bottom: -1, height: 2,
          background: "var(--brand)", borderRadius: 2,
        }} />
      )}
    </button>
  );
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// 프로필 탭
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function ProfileTab({ user, fullRecord, onClose, onUpdate }) {
  const [name, setName] = useState(user?.name || "");
  const [errors, setErrors] = useState({});
  const [globalError, setGlobalError] = useState("");
  const [saving, setSaving] = useState(false);
  const [savedTick, setSavedTick] = useState(false);

  const dirty = (name !== user?.name);

  const submit = (e) => {
    e && e.preventDefault();
    setErrors({}); setGlobalError("");
    if (!dirty) { onClose(); return; }
    setSaving(true);
    setTimeout(() => {
      const res = updateProfile(user, { name });
      setSaving(false);
      if (!res.ok) {
        if (res.field) setErrors({ [res.field]: res.error });
        else setGlobalError(res.error);
        return;
      }
      onUpdate && onUpdate(res.user);
      setSavedTick(true);
      setTimeout(() => onClose(), 700);
    }, 300);
  };

  return (
    <form onSubmit={submit} style={{ padding: "18px 20px 6px" }}>
      <FormField
        label="이름"
        error={errors.name}
        icon={<Icons.id_card size={14} />}
      >
        <input
          value={name}
          onChange={(e) => { setName(e.target.value); if (errors.name) setErrors((s) => ({ ...s, name: undefined })); }}
          placeholder="홍길동"
          style={fieldInput}
        />
      </FormField>

      <FormField label="ID" icon={<Icons.user size={14} />}>
        <input value={user?.id || ""} readOnly style={{ ...fieldInput, color: "var(--ink-3)" }} />
        <span style={{ fontSize: 10, color: "var(--ink-4)", paddingRight: 4, whiteSpace: "nowrap" }}>변경 불가</span>
      </FormField>

      <div style={{
        marginTop: 12, padding: "10px 12px",
        borderRadius: 10, background: "var(--bg-sunk)",
        border: "1px solid var(--line)",
        display: "grid", gap: 6, fontSize: 11,
      }}>
        <MetaRow label="가입일"     value={fmtDate(fullRecord?.createdAt)} />
        <MetaRow label="승인일"     value={fmtDate(fullRecord?.approvedAt)} />
        <MetaRow
          label="마지막 접속"
          value={fullRecord?.previousLoginAt ? fmtDate(fullRecord.previousLoginAt) : "이번이 처음"}
        />
        {fullRecord?.passwordChangedAt && (
          <MetaRow label="비번 변경" value={fmtDate(fullRecord.passwordChangedAt)} />
        )}
      </div>

      {globalError && <ErrorBox text={globalError} />}

      <div style={{ display: "flex", gap: 8, justifyContent: "flex-end", marginTop: 18, marginBottom: 4 }}>
        <SecondaryBtn onClick={onClose}>취소</SecondaryBtn>
        <PrimaryBtn type="submit" disabled={saving || !dirty} loading={saving} done={savedTick}>
          {savedTick ? "저장됨" : "저장"}
        </PrimaryBtn>
      </div>
    </form>
  );
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// 비밀번호 탭
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function PasswordTab({ user, onClose }) {
  const [currentPw, setCurrentPw] = useState("");
  const [newPw, setNewPw]         = useState("");
  const [newPw2, setNewPw2]       = useState("");
  const [show, setShow]           = useState(false);
  const [errors, setErrors] = useState({});
  const [globalError, setGlobalError] = useState("");
  const [saving, setSaving] = useState(false);
  const [done, setDone]     = useState(false);

  const submit = (e) => {
    e && e.preventDefault();
    setErrors({}); setGlobalError("");
    setSaving(true);
    setTimeout(() => {
      const res = changePassword(user, { currentPw, newPw, newPw2 });
      setSaving(false);
      if (!res.ok) {
        if (res.field) setErrors({ [res.field]: res.error });
        else setGlobalError(res.error);
        return;
      }
      setDone(true);
      setTimeout(() => onClose(), 1100);
    }, 350);
  };

  if (done) {
    return (
      <div style={{ padding: "32px 20px 28px", textAlign: "center" }}>
        <div style={{
          width: 52, height: 52, borderRadius: "50%",
          background: "linear-gradient(135deg, #34d399, #10b981)",
          margin: "0 auto 14px", display: "grid", placeItems: "center",
          boxShadow: "0 12px 30px -8px rgba(16,185,129,0.55)",
        }}>
          <Icons.check size={26} color="#fff" />
        </div>
        <div style={{ fontSize: 16, fontWeight: 700, color: "var(--ink)", marginBottom: 4 }}>
          비밀번호가 변경되었습니다
        </div>
        <div style={{ fontSize: 12, color: "var(--ink-3)" }}>
          다음 로그인부터 새 비밀번호를 사용해 주세요.
        </div>
      </div>
    );
  }

  return (
    <form onSubmit={submit} style={{ padding: "18px 20px 6px" }}>
      <FormField
        label="현재 비밀번호"
        error={errors.currentPw}
        icon={<Icons.lock size={14} />}
      >
        <input
          type={show ? "text" : "password"}
          value={currentPw}
          onChange={(e) => { setCurrentPw(e.target.value); if (errors.currentPw) setErrors((s) => ({ ...s, currentPw: undefined })); }}
          placeholder="현재 비밀번호 입력"
          autoComplete="current-password"
          style={fieldInput}
          autoFocus
        />
      </FormField>

      <FormField
        label="새 비밀번호"
        hint="4자 이상"
        error={errors.newPw}
        icon={<Icons.lock size={14} />}
      >
        <input
          type={show ? "text" : "password"}
          value={newPw}
          onChange={(e) => { setNewPw(e.target.value); if (errors.newPw) setErrors((s) => ({ ...s, newPw: undefined })); }}
          placeholder="새 비밀번호 입력"
          autoComplete="new-password"
          style={fieldInput}
        />
      </FormField>

      <FormField
        label="새 비밀번호 확인"
        error={errors.newPw2}
        icon={<Icons.lock size={14} />}
      >
        <input
          type={show ? "text" : "password"}
          value={newPw2}
          onChange={(e) => { setNewPw2(e.target.value); if (errors.newPw2) setErrors((s) => ({ ...s, newPw2: undefined })); }}
          placeholder="한 번 더 입력"
          autoComplete="new-password"
          style={fieldInput}
        />
      </FormField>

      <label style={{
        display: "flex", alignItems: "center", gap: 8,
        fontSize: 12, color: "var(--ink-3)", cursor: "pointer",
        userSelect: "none", marginTop: 4, marginBottom: 6,
      }}>
        <input
          type="checkbox"
          checked={show}
          onChange={(e) => setShow(e.target.checked)}
          style={{ accentColor: "var(--brand)" }}
        />
        비밀번호 표시
      </label>

      {globalError && <ErrorBox text={globalError} />}

      <div style={{ display: "flex", gap: 8, justifyContent: "flex-end", marginTop: 14, marginBottom: 4 }}>
        <SecondaryBtn onClick={onClose}>취소</SecondaryBtn>
        <PrimaryBtn type="submit" disabled={saving || !currentPw || !newPw || !newPw2} loading={saving}>
          변경
        </PrimaryBtn>
      </div>
    </form>
  );
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// 공용 헬퍼
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function Backdrop({ children, onClose }) {
  return (
    <div
      style={{
        position: "fixed", inset: 0, zIndex: 95,
        background: "rgba(10,15,30,0.45)", backdropFilter: "blur(4px)",
        display: "grid", placeItems: "center",
        animation: "slide-in-up 160ms ease both",
      }}
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      {children}
    </div>
  );
}

function modalCard(width) {
  return {
    width,
    background: "var(--bg-elev)",
    border: "1px solid var(--line)",
    borderRadius: 16,
    boxShadow: "0 30px 80px -20px rgba(0,0,0,0.4)",
    overflow: "hidden",
  };
}

function ModalHeader({ title, onClose }) {
  return (
    <div style={{
      display: "flex", alignItems: "center", justifyContent: "space-between",
      padding: "16px 20px",
      borderBottom: "1px solid var(--line)",
    }}>
      <div style={{ fontSize: 16, fontWeight: 800, color: "var(--ink)", letterSpacing: "-0.01em" }}>
        {title}
      </div>
      <button
        onClick={onClose}
        title="닫기"
        style={{
          width: 28, height: 28, borderRadius: 8,
          background: "transparent", border: "none",
          color: "var(--ink-3)", cursor: "pointer",
          display: "grid", placeItems: "center",
        }}
      >
        <Icons.close size={16} />
      </button>
    </div>
  );
}

function FormField({ label, hint, error, icon, children }) {
  const showError = !!error;
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: "var(--ink-3)", letterSpacing: "0.02em" }}>
          {label}
        </div>
        {hint && !showError && <div style={{ fontSize: 10, color: "var(--ink-4)" }}>{hint}</div>}
      </div>
      <div style={{
        display: "flex", alignItems: "center", gap: 8,
        padding: "0 12px", height: 40,
        background: "var(--bg-sunk)",
        border: `1px solid ${showError ? "rgba(239,68,68,0.5)" : "var(--line)"}`,
        borderRadius: 10,
        boxShadow: showError ? "0 0 0 3px rgba(239,68,68,0.10)" : "none",
        transition: "border-color 140ms ease, box-shadow 140ms ease",
      }}>
        {icon && <span style={{ color: "var(--ink-3)", display: "grid", placeItems: "center" }}>{icon}</span>}
        {children}
      </div>
      {showError && (
        <div style={{ fontSize: 10, color: "#dc2626", marginTop: 4 }}>{error}</div>
      )}
    </div>
  );
}

const fieldInput = {
  flex: 1, width: "100%",
  background: "transparent",
  border: "none", outline: "none",
  color: "var(--ink)",
  fontSize: 13, fontWeight: 500,
  padding: "10px 0",
  fontFamily: "inherit",
};

function MetaRow({ label, value }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", color: "var(--ink-3)" }}>
      <span>{label}</span>
      <span style={{ color: "var(--ink-2)", fontWeight: 600, fontFamily: "ui-monospace, Menlo, monospace" }}>
        {value}
      </span>
    </div>
  );
}

function ErrorBox({ text }) {
  return (
    <div style={{
      fontSize: 12, color: "#dc2626",
      padding: "8px 12px", borderRadius: 8,
      background: "rgba(239,68,68,0.08)",
      border: "1px solid rgba(239,68,68,0.22)",
      marginTop: 4, marginBottom: 4,
    }}>
      {text}
    </div>
  );
}

function PrimaryBtn({ children, type = "button", disabled, loading, done, onClick }) {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      style={{
        padding: "9px 18px", borderRadius: 9,
        fontSize: 13, fontWeight: 700,
        background: disabled
          ? "var(--bg-sunk)"
          : done
            ? "linear-gradient(135deg, #10b981, #059669)"
            : "linear-gradient(135deg, #4f46e5, #8b83ff)",
        color: disabled ? "var(--ink-4)" : "#fff",
        border: "none",
        boxShadow: disabled ? "none" : "0 8px 18px -6px rgba(79,70,229,0.45)",
        cursor: disabled ? "not-allowed" : "pointer",
        display: "flex", alignItems: "center", gap: 6,
        transition: "all 140ms ease",
      }}
    >
      {loading && (
        <span style={{
          width: 12, height: 12, borderRadius: "50%",
          border: "2px solid rgba(255,255,255,0.35)",
          borderTopColor: "#fff",
          animation: "spin 0.7s linear infinite",
        }} />
      )}
      {done && <Icons.check size={13} color="#fff" />}
      {children}
    </button>
  );
}

function SecondaryBtn({ children, onClick }) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        padding: "9px 18px", borderRadius: 9,
        fontSize: 13, fontWeight: 600,
        background: "transparent",
        color: "var(--ink-2)",
        border: "1px solid var(--line)",
        cursor: "pointer",
      }}
    >
      {children}
    </button>
  );
}
