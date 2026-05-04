import { useState, useEffect, useMemo, useCallback } from "react";
import { Icons } from "../components/Icons.jsx";
import {
  listAllUsers,
  approveUser,
  rejectUser,
  reactivateUser,
  adminResetPassword,
  ROLE_LABEL,
  STATUS_LABEL,
} from "../lib/authMock.js";

/* ── 관리자 페이지 (admin 전용 통합 대시보드) ─────────────────
 *  sub-tabs:
 *    1) 개요         — 사용자 KPI · pending 큐 · 시스템 상태
 *    2) 운영자       — 가입 신청 승인/반려, 활성/반려 이력
 *    3) 시스템 설정  — 폴링 주기·임계·색약·데이터 (placeholder)
 *
 *  2026-05-04 신규 — UserManagement.jsx 의 후속.
 */

const STATUS_TONE = {
  pending:  { bg: "rgba(245,158,11,0.14)", bd: "rgba(245,158,11,0.30)", fg: "#b45309", dot: "#f59e0b", label: "승인 대기" },
  active:   { bg: "rgba(16,185,129,0.14)", bd: "rgba(16,185,129,0.30)", fg: "#047857", dot: "#10b981", label: "활성" },
  rejected: { bg: "rgba(100,116,139,0.14)", bd: "rgba(100,116,139,0.30)", fg: "#475569", dot: "#64748b", label: "반려" },
};

const ROLE_TONE = {
  admin:    { fg: "#7c3aed", bg: "rgba(124,58,237,0.10)", bd: "rgba(124,58,237,0.28)" },
  operator: { fg: "#0369a1", bg: "rgba(14,165,233,0.10)", bd: "rgba(14,165,233,0.28)" },
  viewer:   { fg: "#475569", bg: "rgba(100,116,139,0.10)", bd: "rgba(100,116,139,0.28)" },
  guest:    { fg: "#b45309", bg: "rgba(245,158,11,0.10)", bd: "rgba(245,158,11,0.28)" },
};

function fmtDate(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")} ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

// ── 메인 ─────────────────────────────────────────────────
export function Admin({ user, equipment, anomalies, watch, apiStatus }) {
  const [section, setSection] = useState("overview");
  const [users, setUsers] = useState([]);
  const [toast, setToast] = useState(null);

  const reload = useCallback(() => {
    const res = listAllUsers(user);
    if (res.ok) setUsers(res.users);
  }, [user]);

  useEffect(() => { reload(); }, [reload]);

  // 자동 새로고침: 5초 폴링 + 탭 포커스 복귀 시
  useEffect(() => {
    const id = setInterval(reload, 5000);
    const onFocus = () => reload();
    window.addEventListener("focus", onFocus);
    document.addEventListener("visibilitychange", onFocus);
    return () => {
      clearInterval(id);
      window.removeEventListener("focus", onFocus);
      document.removeEventListener("visibilitychange", onFocus);
    };
  }, [reload]);

  useEffect(() => {
    if (!toast) return;
    const id = setTimeout(() => setToast(null), 2400);
    return () => clearTimeout(id);
  }, [toast]);

  if (!user || user.role !== "admin") {
    return (
      <div style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center", color: "var(--ink-3)" }}>
        관리자 권한이 필요합니다.
      </div>
    );
  }

  const counts = {
    pending:  users.filter((u) => u.status === "pending").length,
    active:   users.filter((u) => u.status === "active").length,
    rejected: users.filter((u) => u.status === "rejected").length,
    all:      users.length,
  };

  return (
    <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", overflow: "hidden" }}>
      {/* ── 헤더 + sub-tabs ── */}
      <div style={{ padding: "20px 32px 0", flexShrink: 0 }}>
        <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", marginBottom: 14 }}>
          <div>
            <div style={{ fontSize: 20, fontWeight: 800, color: "var(--ink)", letterSpacing: "-0.02em" }}>
              관리자 페이지
            </div>
            <div style={{ fontSize: 12, color: "var(--ink-3)", marginTop: 4 }}>
              운영자 승인 · 시스템 운영 설정 · 사용 통계
            </div>
          </div>
          <AdminBadge user={user} />
        </div>

        <div style={{ display: "flex", gap: 4, borderBottom: "1px solid var(--line)" }}>
          <SubTabBtn k="overview"  cur={section} set={setSection} label="요약" />
          <SubTabBtn k="operators" cur={section} set={setSection} label="운영자 관리" badge={counts.pending} />
          <SubTabBtn k="settings"  cur={section} set={setSection} label="시스템 설정" />
        </div>
      </div>

      {/* ── 섹션 컨텐츠 ── */}
      <div style={{ flex: 1, overflow: "auto", padding: "20px 32px 32px" }}>
        {section === "overview" && (
          <OverviewSection
            users={users} counts={counts}
            equipment={equipment} anomalies={anomalies} watch={watch}
            apiStatus={apiStatus}
            onJump={(s) => setSection(s)}
          />
        )}
        {section === "operators" && (
          <OperatorsSection
            user={user} users={users} counts={counts}
            reload={reload} setToast={setToast}
          />
        )}
        {section === "settings" && (
          <SettingsSection apiStatus={apiStatus} setToast={setToast} />
        )}
      </div>

      {toast && <Toast toast={toast} />}
    </div>
  );
}

// ── sub-tabs UI ────────────────────────────────────────
function SubTabBtn({ k, cur, set, label, badge }) {
  const active = cur === k;
  return (
    <button
      onClick={() => set(k)}
      style={{
        position: "relative", padding: "10px 16px",
        background: "transparent", border: "none",
        color: active ? "var(--brand)" : "var(--ink-3)",
        fontSize: 13, fontWeight: 700, cursor: "pointer",
        display: "flex", alignItems: "center", gap: 6,
      }}
    >
      {label}
      {!!badge && badge > 0 && (
        <span style={{
          display: "inline-flex", alignItems: "center", justifyContent: "center",
          minWidth: 18, height: 18, padding: "0 5px", borderRadius: 999,
          background: "linear-gradient(135deg, #f59e0b, #f97316)",
          color: "#fff", fontSize: 10, fontWeight: 800,
        }}>{badge}</span>
      )}
      {active && (
        <span style={{
          position: "absolute", left: 0, right: 0, bottom: -1, height: 2,
          background: "var(--brand)", borderRadius: 2,
        }} />
      )}
    </button>
  );
}

function AdminBadge({ user }) {
  return (
    <div style={{
      display: "inline-flex", alignItems: "center", gap: 6,
      padding: "4px 10px", borderRadius: 999,
      background: "rgba(139,92,246,0.10)",
      border: "1px solid rgba(139,92,246,0.28)",
      fontSize: 11, fontWeight: 700, color: "#7c3aed",
    }}>
      <span style={{ width: 5, height: 5, borderRadius: "50%", background: "#8b5cf6" }} />
      관리자 · {user.id}
    </div>
  );
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// 1) 개요 섹션
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function OverviewSection({ users, counts, equipment, anomalies, watch, apiStatus, onJump }) {
  const recentPending = users
    .filter((u) => u.status === "pending")
    .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt))
    .slice(0, 5);

  const lastActiveJoin = users
    .filter((u) => u.status === "active" && u.approvedAt)
    .sort((a, b) => new Date(b.approvedAt) - new Date(a.approvedAt))[0];

  const apiText = ({
    mock:    "목업 데이터로 동작 중",
    loading: "백엔드 연결 시도 중",
    ok:      "AI 백엔드 연동됨",
    error:   "백엔드 오프라인 — 목업 fallback",
  })[apiStatus] || "—";
  const apiColor = ({
    mock: "var(--ink-3)", loading: "var(--warn)", ok: "var(--ok)", error: "var(--err)",
  })[apiStatus] || "var(--ink-3)";

  return (
    <div style={{ display: "grid", gap: 16 }}>
      {/* KPI 카드 4개 */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
        <KpiCard label="전체 사용자"     value={counts.all}      hint="등록된 모든 운영자" tone="brand" />
        <KpiCard label="승인 대기"       value={counts.pending}  hint="처리 필요" tone="warn"
                 onClick={counts.pending > 0 ? () => onJump("operators") : undefined} />
        <KpiCard label="활성 계정"       value={counts.active}   hint="현재 로그인 가능" tone="ok" />
        <KpiCard label="반려 이력"       value={counts.rejected} hint="가입 거부 누적" tone="muted" />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 380px", gap: 16 }}>
        {/* 좌: 최근 가입 신청 */}
        <Panel title="최근 가입 신청" right={
          counts.pending > 0 && (
            <button
              onClick={() => onJump("operators")}
              style={{
                fontSize: 11, fontWeight: 700, color: "var(--brand)",
                background: "transparent", border: "none", cursor: "pointer",
              }}
            >
              전체 보기 →
            </button>
          )
        }>
          {recentPending.length === 0 ? (
            <Empty text="처리 대기 중인 가입 신청이 없습니다." />
          ) : (
            <div style={{ display: "flex", flexDirection: "column" }}>
              {recentPending.map((u, i) => (
                <div key={u.id} style={{
                  display: "flex", alignItems: "center", gap: 12,
                  padding: "10px 4px",
                  borderTop: i === 0 ? "none" : "1px solid var(--line)",
                }}>
                  <div style={{
                    width: 32, height: 32, borderRadius: "50%",
                    background: ROLE_TONE[u.role]?.bg || "var(--bg-sunk)",
                    color: ROLE_TONE[u.role]?.fg || "var(--ink-2)",
                    display: "grid", placeItems: "center",
                    fontSize: 12, fontWeight: 800,
                  }}>
                    {u.name?.[0]?.toUpperCase() || "U"}
                  </div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: 13, fontWeight: 700, color: "var(--ink)" }}>
                      {u.name} <span style={{ fontWeight: 500, color: "var(--ink-3)" }}>· {u.id}</span>
                    </div>
                    <div style={{ fontSize: 11, color: "var(--ink-3)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {ROLE_LABEL[u.role]}
                    </div>
                  </div>
                  <div style={{ fontSize: 11, color: "var(--ink-3)", whiteSpace: "nowrap" }}>
                    {fmtDate(u.createdAt)}
                  </div>
                </div>
              ))}
            </div>
          )}
        </Panel>

        {/* 우: 시스템 상태 */}
        <Panel title="시스템 상태">
          <div style={{ display: "grid", gap: 12 }}>
            <StatRow
              label="백엔드 연결"
              value={apiText}
              color={apiColor}
            />
            <StatRow
              label="감시 노드"
              value={`${equipment?.length || 0}개`}
            />
            <StatRow
              label="이상 탐지"
              value={`${anomalies?.length || 0}건`}
              color={(anomalies?.length || 0) > 0 ? "var(--err)" : "var(--ink-2)"}
            />
            <StatRow
              label="관찰 필요"
              value={`${watch?.length || 0}건`}
              color={(watch?.length || 0) > 0 ? "var(--warn)" : "var(--ink-2)"}
            />
            {lastActiveJoin && (
              <StatRow
                label="최근 승인"
                value={`${lastActiveJoin.name} · ${fmtDate(lastActiveJoin.approvedAt)}`}
              />
            )}
          </div>
        </Panel>
      </div>
    </div>
  );
}

function KpiCard({ label, value, hint, tone, onClick }) {
  const toneMap = {
    brand: "var(--brand)",
    warn:  "#f59e0b",
    ok:    "#10b981",
    muted: "var(--ink-3)",
  };
  const c = toneMap[tone] || "var(--ink)";
  return (
    <div
      onClick={onClick}
      style={{
        padding: 16, borderRadius: 14,
        background: "var(--bg-elev)",
        border: "1px solid var(--line)",
        cursor: onClick ? "pointer" : "default",
        transition: "transform 120ms ease, box-shadow 140ms ease",
      }}
    >
      <div style={{ fontSize: 11, fontWeight: 700, color: "var(--ink-3)", letterSpacing: "0.04em", textTransform: "uppercase", marginBottom: 6 }}>
        {label}
      </div>
      <div style={{ fontSize: 28, fontWeight: 800, color: c, letterSpacing: "-0.03em", lineHeight: 1 }}>
        {value}
      </div>
      <div style={{ fontSize: 11, color: "var(--ink-3)", marginTop: 6 }}>{hint}</div>
    </div>
  );
}

function Panel({ title, right, children }) {
  return (
    <div style={{
      padding: "14px 16px 16px",
      borderRadius: 14,
      background: "var(--bg-elev)",
      border: "1px solid var(--line)",
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
        <div style={{ fontSize: 13, fontWeight: 700, color: "var(--ink)" }}>{title}</div>
        {right}
      </div>
      {children}
    </div>
  );
}

function StatRow({ label, value, color }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 12 }}>
      <span style={{ color: "var(--ink-3)" }}>{label}</span>
      <span style={{ color: color || "var(--ink)", fontWeight: 700 }}>{value}</span>
    </div>
  );
}

function Empty({ text }) {
  return (
    <div style={{ padding: "28px 0", textAlign: "center", fontSize: 12, color: "var(--ink-3)" }}>
      {text}
    </div>
  );
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// 2) 운영자 관리 섹션 (구 UserManagement)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
const FILTERS = [
  { k: "pending",  ko: "승인 대기" },
  { k: "active",   ko: "활성" },
  { k: "rejected", ko: "반려" },
  { k: "all",      ko: "전체" },
];

function OperatorsSection({ user, users, counts, reload, setToast }) {
  const [filter, setFilter] = useState("pending");
  const [search, setSearch] = useState("");
  const [rejectTarget, setRejectTarget] = useState(null);
  const [rejectReason, setRejectReason] = useState("");
  const [resetTarget, setResetTarget] = useState(null);
  const [resetPw, setResetPw]         = useState("");
  const [resetError, setResetError]   = useState("");
  const [resetDone, setResetDone]     = useState(null); // {userId, newPw} 또는 null

  const filtered = useMemo(() => {
    let list = filter === "all" ? users : users.filter((u) => u.status === filter);
    const q = search.trim().toLowerCase();
    if (q) {
      list = list.filter((u) =>
        [u.id, u.name].some((v) => (v || "").toLowerCase().includes(q))
      );
    }
    return list.slice().sort((a, b) => {
      const order = { pending: 0, active: 1, rejected: 2 };
      const o = (order[a.status] || 9) - (order[b.status] || 9);
      if (o !== 0) return o;
      return new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime();
    });
  }, [users, filter, search]);

  const handleApprove = (target) => {
    const res = approveUser(user, target.id);
    if (res.ok) { setToast({ kind: "ok", text: `${target.name} 님의 가입을 승인했습니다.` }); reload(); }
    else        { setToast({ kind: "err", text: res.error }); }
  };
  const openReject = (t) => { setRejectTarget(t); setRejectReason(""); };
  const confirmReject = () => {
    if (!rejectTarget) return;
    const res = rejectUser(user, rejectTarget.id, rejectReason);
    if (res.ok) {
      setToast({ kind: "ok", text: `${rejectTarget.name} 님의 가입을 반려했습니다.` });
      setRejectTarget(null); setRejectReason("");
      reload();
    } else {
      setToast({ kind: "err", text: res.error });
    }
  };
  const handleReactivate = (t) => {
    const res = reactivateUser(user, t.id);
    if (res.ok) { setToast({ kind: "ok", text: `${t.name} 님을 승인 대기로 되돌렸습니다.` }); reload(); }
    else        { setToast({ kind: "err", text: res.error }); }
  };
  const openReset = (t) => {
    setResetTarget(t);
    setResetPw("");
    setResetError("");
    setResetDone(null);
  };
  const closeReset = () => { setResetTarget(null); setResetPw(""); setResetError(""); setResetDone(null); };
  const confirmReset = () => {
    if (!resetTarget) return;
    const res = adminResetPassword(user, resetTarget.id, resetPw);
    if (!res.ok) { setResetError(res.error); return; }
    setResetDone({ userId: res.userId, newPw: res.newPw });
    setToast({ kind: "ok", text: `${resetTarget.name} 님의 비밀번호를 재설정했습니다.` });
    reload();
  };
  const genRandomPw = () => {
    // 데모용 랜덤 8자 (영숫자)
    const chars = "abcdefghjkmnpqrstuvwxyzABCDEFGHJKMNPQRSTUVWXYZ23456789";
    let s = "";
    for (let i = 0; i < 8; i++) s += chars[Math.floor(Math.random() * chars.length)];
    setResetPw(s);
    setResetError("");
  };

  return (
    <>
      {/* 필터 + 검색 */}
      <div style={{ display: "flex", gap: 10, marginBottom: 14, alignItems: "center" }}>
        <div style={{ display: "flex", gap: 6 }}>
          {FILTERS.map((f) => {
            const active = filter === f.k;
            const cnt = counts[f.k];
            return (
              <button
                key={f.k}
                onClick={() => setFilter(f.k)}
                style={{
                  padding: "8px 14px", borderRadius: 999,
                  fontSize: 13, fontWeight: 700,
                  background: active ? "var(--brand)" : "var(--bg-elev)",
                  color: active ? "#fff" : "var(--ink-2)",
                  border: `1px solid ${active ? "var(--brand)" : "var(--line)"}`,
                  display: "flex", alignItems: "center", gap: 6,
                  cursor: "pointer",
                }}
              >
                {f.ko}
                <span style={{
                  fontSize: 11, fontWeight: 700,
                  padding: "1px 7px", borderRadius: 999,
                  background: active ? "rgba(255,255,255,0.22)" : "var(--bg)",
                  color: active ? "#fff" : "var(--ink-3)",
                }}>{cnt}</span>
              </button>
            );
          })}
        </div>
        <div style={{ flex: 1 }} />
        <button
          onClick={() => { reload(); setToast({ kind: "ok", text: "최신 정보로 갱신했습니다." }); }}
          title="새로고침"
          style={{
            display: "flex", alignItems: "center", gap: 6,
            padding: "0 12px", height: 36, borderRadius: 10,
            background: "var(--bg-elev)",
            border: "1px solid var(--line)",
            color: "var(--ink-2)", fontSize: 12, fontWeight: 600,
            cursor: "pointer",
          }}
        >
          <Icons.refresh size={13} />새로고침
        </button>
        <div style={{
          display: "flex", alignItems: "center", gap: 8,
          padding: "0 12px", height: 36, width: 280,
          background: "var(--bg-elev)", border: "1px solid var(--line)", borderRadius: 10,
        }}>
          <Icons.search size={14} color="var(--ink-3)" />
          <input
            value={search} onChange={(e) => setSearch(e.target.value)}
            placeholder="ID · 이름 검색"
            style={{
              flex: 1, background: "transparent", border: "none", outline: "none",
              color: "var(--ink)", fontSize: 13, fontFamily: "inherit",
            }}
          />
        </div>
      </div>

      {/* 테이블 */}
      <div style={{
        background: "var(--bg-elev)", border: "1px solid var(--line)",
        borderRadius: 14, overflow: "hidden",
      }}>
        <table style={{ width: "100%", borderCollapse: "collapse", tableLayout: "fixed" }}>
          <colgroup>
            <col style={{ width: "22%" }} />
            <col style={{ width: "18%" }} />
            <col style={{ width: "11%" }} />
            <col style={{ width: "12%" }} />
            <col style={{ width: "15%" }} />
            <col style={{ width: "22%" }} />
          </colgroup>
          <thead>
            <tr style={{ background: "var(--bg)", borderBottom: "1px solid var(--line)" }}>
              {["ID", "이름", "역할", "상태", "신청일", "처리"].map((h) => (
                <th key={h} style={{
                  padding: "11px 14px", textAlign: "left",
                  fontSize: 11, fontWeight: 700, letterSpacing: "0.04em",
                  color: "var(--ink-3)", textTransform: "uppercase",
                }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 && (
              <tr><td colSpan={6} style={{ padding: "48px 14px", textAlign: "center", color: "var(--ink-3)", fontSize: 13 }}>
                표시할 사용자가 없습니다.
              </td></tr>
            )}
            {filtered.map((u, i) => {
              const st = STATUS_TONE[u.status] || STATUS_TONE.pending;
              const rt = ROLE_TONE[u.role] || ROLE_TONE.operator;
              return (
                <tr key={u.id} style={{
                  borderBottom: i === filtered.length - 1 ? "none" : "1px solid var(--line)",
                  background: i % 2 === 1 ? "rgba(0,0,0,0.015)" : "transparent",
                  height: 56, verticalAlign: "middle",
                }}>
                  <td style={{ padding: "10px 14px", fontSize: 13, color: "var(--ink)", fontWeight: 600 }}>
                    {u.id}
                    {u.id === user.id && <span style={{ fontSize: 10, fontWeight: 700, color: "var(--brand)", marginLeft: 6 }}>(나)</span>}
                  </td>
                  <td style={{ padding: "10px 14px", fontSize: 13, color: "var(--ink-2)" }}>{u.name}</td>
                  <td style={{ padding: "10px 14px" }}>
                    <span style={{
                      display: "inline-flex", alignItems: "center",
                      padding: "3px 9px", borderRadius: 999,
                      fontSize: 11, fontWeight: 700,
                      color: rt.fg, background: rt.bg, border: `1px solid ${rt.bd}`,
                    }}>{ROLE_LABEL[u.role] || u.role}</span>
                  </td>
                  <td style={{ padding: "10px 14px" }}>
                    <span style={{
                      display: "inline-flex", alignItems: "center", gap: 5,
                      padding: "3px 9px", borderRadius: 999,
                      fontSize: 11, fontWeight: 700,
                      color: st.fg, background: st.bg, border: `1px solid ${st.bd}`,
                    }}>
                      <span style={{ width: 5, height: 5, borderRadius: "50%", background: st.dot }} />
                      {STATUS_LABEL[u.status] || u.status}
                    </span>
                    {u.status === "rejected" && u.rejectedReason && (
                      <div style={{
                        fontSize: 10, color: "var(--ink-3)", marginTop: 4, fontStyle: "italic",
                        overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                      }} title={u.rejectedReason}>
                        “{u.rejectedReason}”
                      </div>
                    )}
                  </td>
                  <td style={{ padding: "10px 14px", fontSize: 12, color: "var(--ink-3)" }}>
                    {fmtDate(u.createdAt)}
                  </td>
                  <td style={{ padding: "10px 14px" }}>
                    {u.status === "pending" && (
                      <div style={{ display: "flex", gap: 6 }}>
                        <ActionButton tone="ok"  icon={<Icons.check size={13} />} label="승인" onClick={() => handleApprove(u)} />
                        <ActionButton tone="err" icon={<Icons.close size={13} />} label="반려" onClick={() => openReject(u)} />
                      </div>
                    )}
                    {u.status === "active" && u.id !== user.id && (
                      <div style={{ display: "flex", gap: 6 }}>
                        <ActionButton tone="brand" icon={<Icons.lock size={13} />} label="비번 재설정" onClick={() => openReset(u)} />
                        <ActionButton tone="muted" icon={<Icons.close size={13} />} label="반려" onClick={() => openReject(u)} />
                      </div>
                    )}
                    {u.status === "active" && u.id === user.id && (
                      <span style={{ fontSize: 11, color: "var(--ink-3)" }}>—</span>
                    )}
                    {u.status === "rejected" && (
                      <ActionButton tone="brand" icon={<Icons.refresh size={13} />} label="재심사" onClick={() => handleReactivate(u)} />
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* 반려 모달 */}
      {rejectTarget && (
        <RejectModal
          target={rejectTarget}
          reason={rejectReason}
          setReason={setRejectReason}
          onConfirm={confirmReject}
          onCancel={() => setRejectTarget(null)}
        />
      )}

      {/* 비밀번호 재설정 모달 */}
      {resetTarget && (
        <ResetPasswordModal
          target={resetTarget}
          pw={resetPw}
          setPw={(v) => { setResetPw(v); setResetError(""); }}
          error={resetError}
          done={resetDone}
          onGenerate={genRandomPw}
          onConfirm={confirmReset}
          onCancel={closeReset}
        />
      )}
    </>
  );
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// 3) 시스템 설정 섹션 (옵션 B 자리)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
const POLLING_OPTIONS = [
  { k: 10,  ko: "10초"  },
  { k: 30,  ko: "30초"  },
  { k: 60,  ko: "1분 (기본)" },
  { k: 300, ko: "5분"  },
];

function SettingsSection({ apiStatus, setToast }) {
  const [polling, setPolling] = useState(() => {
    const v = parseInt(localStorage.getItem("siwon.settings.polling"), 10);
    return Number.isFinite(v) ? v : 60;
  });

  const savePolling = (v) => {
    setPolling(v);
    localStorage.setItem("siwon.settings.polling", String(v));
    setToast({ kind: "ok", text: `폴링 주기를 ${v}초로 저장했습니다. (적용은 다음 새로고침)` });
  };

  const resetMock = () => {
    if (!window.confirm("모든 mock 데이터(사용자·세션·코드 등) 를 초기화합니다. 계속하시겠습니까?")) return;
    Object.keys(localStorage)
      .filter((k) => k.startsWith("siwon.auth.") || k.startsWith("siwon.settings.") || k.startsWith("siwon.prefs."))
      .forEach((k) => localStorage.removeItem(k));
    setToast({ kind: "ok", text: "초기화 완료. 페이지를 새로고침하면 시드 admin 으로 복귀됩니다." });
  };

  return (
    <div style={{ display: "grid", gap: 16, maxWidth: 880 }}>
      {/* 데이터 갱신 */}
      <SettingPanel title="데이터 갱신" desc="대시보드가 백엔드에서 데이터를 가져오는 주기">
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {POLLING_OPTIONS.map((o) => {
            const active = polling === o.k;
            return (
              <button
                key={o.k}
                onClick={() => savePolling(o.k)}
                style={{
                  padding: "8px 14px", borderRadius: 999,
                  fontSize: 13, fontWeight: 700,
                  background: active ? "var(--brand)" : "transparent",
                  color: active ? "#fff" : "var(--ink-2)",
                  border: `1px solid ${active ? "var(--brand)" : "var(--line)"}`,
                  cursor: "pointer",
                }}
              >
                {o.ko}
              </button>
            );
          })}
        </div>
        <div style={{ marginTop: 10, fontSize: 11, color: "var(--ink-3)" }}>
          현재 백엔드 상태:{" "}
          <strong style={{ color: apiStatus === "ok" ? "var(--ok)" : "var(--ink-2)" }}>
            {apiStatus === "ok" ? "연결됨" : apiStatus === "error" ? "오프라인" : apiStatus === "loading" ? "연결 중" : "MOCK"}
          </strong>
        </div>
      </SettingPanel>

      {/* 알림 임계값 (read-only — 모델/백엔드 영역) */}
      <SettingPanel title="알림 임계값" desc="이상 탐지 MSE 임계 (모델 학습 시 결정)">
        <div style={{ display: "grid", gap: 6 }}>
          <ReadonlyRow label="이상 임계 (절대)"   value="MSE > 0.0085"  hint="ai/config/model_config.json" />
          <ReadonlyRow label="관찰 임계 (이상 대비)" value="0.7 ~ 1.0배" hint="3단계 분류 기준" />
          <ReadonlyRow label="통신장애 판정"        value="60초 무응답"   hint="백엔드 설정" />
        </div>
        <div style={{ marginTop: 10, fontSize: 11, color: "var(--ink-3)" }}>
          임계값 조정은 모델 재학습 또는 백엔드 설정 파일 수정 필요. 5/11 1차 자문 보고서 후 이 화면에서 직접 조정 가능하도록 확장 예정.
        </div>
      </SettingPanel>

      {/* 데이터 내보내기 (placeholder) */}
      <SettingPanel title="데이터 내보내기" desc="감사 로그 · 운영자 활동 · 이상 이력">
        <div style={{ display: "flex", gap: 8 }}>
          <DisabledBtn label="운영자 목록 (CSV)" />
          <DisabledBtn label="이상 이력 (CSV)"   />
          <DisabledBtn label="감사 로그 (JSON)"  />
        </div>
        <div style={{ marginTop: 10, fontSize: 11, color: "var(--ink-3)" }}>
          백엔드 감사 로그 도입 후 활성화. 현재 mock 환경.
        </div>
      </SettingPanel>

      {/* 개발자 도구 */}
      <SettingPanel title="개발자 도구" desc="mock 데이터 초기화 등 — 시연 후 정리용">
        <button
          onClick={resetMock}
          style={{
            padding: "10px 14px", borderRadius: 10,
            background: "transparent",
            border: "1px solid rgba(239,68,68,0.4)",
            color: "#dc2626", fontSize: 13, fontWeight: 700,
            cursor: "pointer",
          }}
        >
          🔄  Mock 데이터 전체 초기화
        </button>
      </SettingPanel>
    </div>
  );
}

function SettingPanel({ title, desc, children }) {
  return (
    <div style={{
      padding: "16px 18px",
      borderRadius: 14,
      background: "var(--bg-elev)",
      border: "1px solid var(--line)",
    }}>
      <div style={{ marginBottom: 12 }}>
        <div style={{ fontSize: 14, fontWeight: 700, color: "var(--ink)" }}>{title}</div>
        {desc && <div style={{ fontSize: 11, color: "var(--ink-3)", marginTop: 3 }}>{desc}</div>}
      </div>
      {children}
    </div>
  );
}

function ReadonlyRow({ label, value, hint }) {
  return (
    <div style={{
      display: "flex", justifyContent: "space-between", alignItems: "center",
      padding: "8px 12px", borderRadius: 8,
      background: "var(--bg-sunk)", border: "1px solid var(--line)",
    }}>
      <div>
        <div style={{ fontSize: 12, fontWeight: 700, color: "var(--ink-2)" }}>{label}</div>
        {hint && <div style={{ fontSize: 10, color: "var(--ink-4)", marginTop: 1 }}>{hint}</div>}
      </div>
      <div style={{ fontSize: 12, fontWeight: 700, color: "var(--ink)", fontFamily: "ui-monospace, Menlo, monospace" }}>
        {value}
      </div>
    </div>
  );
}

function DisabledBtn({ label }) {
  return (
    <button
      disabled
      style={{
        padding: "9px 14px", borderRadius: 9,
        background: "transparent",
        border: "1px solid var(--line)",
        color: "var(--ink-4)", fontSize: 12, fontWeight: 600,
        cursor: "not-allowed",
      }}
    >
      {label}
    </button>
  );
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// 공용
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function ActionButton({ tone, icon, label, onClick }) {
  const tones = {
    ok:    { bg: "linear-gradient(135deg, #10b981, #059669)", color: "#fff", shadow: "0 6px 14px -4px rgba(16,185,129,0.45)" },
    err:   { bg: "transparent", color: "#dc2626", border: "1px solid rgba(239,68,68,0.35)" },
    muted: { bg: "transparent", color: "var(--ink-3)", border: "1px solid var(--line)" },
    brand: { bg: "transparent", color: "var(--brand)", border: "1px solid var(--brand)" },
  };
  const t = tones[tone] || tones.muted;
  return (
    <button
      onClick={onClick}
      style={{
        display: "inline-flex", alignItems: "center", gap: 5,
        padding: "6px 11px", borderRadius: 8,
        fontSize: 12, fontWeight: 700,
        background: t.bg, color: t.color,
        border: t.border || "none",
        boxShadow: t.shadow || "none",
        cursor: "pointer",
      }}
    >
      {icon}{label}
    </button>
  );
}

function RejectModal({ target, reason, setReason, onConfirm, onCancel }) {
  return (
    <div
      style={{
        position: "fixed", inset: 0, zIndex: 95,
        background: "rgba(10,15,30,0.45)", backdropFilter: "blur(4px)",
        display: "grid", placeItems: "center",
        animation: "slide-in-up 160ms ease both",
      }}
      onClick={(e) => { if (e.target === e.currentTarget) onCancel(); }}
    >
      <div style={{
        width: 460, padding: 28, borderRadius: 16,
        background: "var(--bg-elev)", border: "1px solid var(--line)",
        boxShadow: "0 30px 80px -20px rgba(0,0,0,0.4)",
      }}>
        <div style={{ fontSize: 17, fontWeight: 800, marginBottom: 6, color: "var(--ink)" }}>
          가입 신청 반려
        </div>
        <div style={{ fontSize: 13, color: "var(--ink-3)", marginBottom: 16, lineHeight: 1.6 }}>
          <strong style={{ color: "var(--ink-2)" }}>{target.name}</strong> ({target.id}) 님의
          가입을 반려합니다. 반려 사유는 본인에게 표시됩니다 (선택).
        </div>
        <textarea
          value={reason} onChange={(e) => setReason(e.target.value)}
          placeholder="예: 등록되지 않은 외부 인원입니다."
          rows={3} maxLength={200}
          style={{
            width: "100%", padding: "10px 12px",
            fontSize: 13, color: "var(--ink)",
            background: "var(--bg)", border: "1px solid var(--line)",
            borderRadius: 10, outline: "none", resize: "vertical",
            fontFamily: "inherit", marginBottom: 16, boxSizing: "border-box",
          }}
          autoFocus
        />
        <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
          <button
            onClick={onCancel}
            style={{
              padding: "9px 16px", borderRadius: 9, fontSize: 13, fontWeight: 600,
              background: "transparent", color: "var(--ink-2)",
              border: "1px solid var(--line)", cursor: "pointer",
            }}
          >취소</button>
          <button
            onClick={onConfirm}
            style={{
              padding: "9px 16px", borderRadius: 9, fontSize: 13, fontWeight: 700,
              background: "linear-gradient(135deg, #ef4444, #dc2626)",
              color: "#fff", border: "none", cursor: "pointer",
              boxShadow: "0 8px 18px -6px rgba(239,68,68,0.5)",
            }}
          >반려 확정</button>
        </div>
      </div>
    </div>
  );
}

function ResetPasswordModal({ target, pw, setPw, error, done, onGenerate, onConfirm, onCancel }) {
  return (
    <div
      style={{
        position: "fixed", inset: 0, zIndex: 95,
        background: "rgba(10,15,30,0.45)", backdropFilter: "blur(4px)",
        display: "grid", placeItems: "center",
        animation: "slide-in-up 160ms ease both",
      }}
      onClick={(e) => { if (e.target === e.currentTarget) onCancel(); }}
    >
      <div style={{
        width: 460, padding: 28, borderRadius: 16,
        background: "var(--bg-elev)", border: "1px solid var(--line)",
        boxShadow: "0 30px 80px -20px rgba(0,0,0,0.4)",
      }}>
        <div style={{ fontSize: 17, fontWeight: 800, marginBottom: 6, color: "var(--ink)" }}>
          비밀번호 재설정
        </div>
        <div style={{ fontSize: 13, color: "var(--ink-3)", marginBottom: 16, lineHeight: 1.6 }}>
          <strong style={{ color: "var(--ink-2)" }}>{target.name}</strong> ({target.id}) 님의
          비밀번호를 새로 설정합니다. 변경 후 본인에게 안전한 경로로 전달해 주세요.
        </div>

        {done ? (
          <>
            <div style={{
              padding: "14px 16px", borderRadius: 10,
              background: "rgba(16,185,129,0.10)",
              border: "1px solid rgba(16,185,129,0.30)",
              marginBottom: 12,
            }}>
              <div style={{ fontSize: 11, fontWeight: 700, color: "#047857", marginBottom: 4, letterSpacing: "0.02em" }}>
                ✓ 변경 완료 — 새 비밀번호
              </div>
              <div style={{
                fontSize: 22, fontWeight: 800, color: "var(--ink)",
                fontFamily: "ui-monospace, Menlo, monospace",
                letterSpacing: "0.08em", textAlign: "center", padding: "10px 0",
              }}>
                {done.newPw}
              </div>
              <div style={{ fontSize: 11, color: "var(--ink-3)", textAlign: "center" }}>
                사용자에게 직접 전달 후 변경 권장.
              </div>
            </div>
            <div style={{ display: "flex", justifyContent: "flex-end" }}>
              <button
                onClick={onCancel}
                style={{
                  padding: "9px 18px", borderRadius: 9, fontSize: 13, fontWeight: 700,
                  background: "linear-gradient(135deg, #4f46e5, #8b83ff)",
                  color: "#fff", border: "none", cursor: "pointer",
                  boxShadow: "0 8px 18px -6px rgba(79,70,229,0.45)",
                }}
              >확인</button>
            </div>
          </>
        ) : (
          <>
            <div style={{
              display: "flex", alignItems: "center", gap: 8,
              padding: "0 12px", height: 44,
              background: "var(--bg-sunk)",
              border: `1px solid ${error ? "rgba(239,68,68,0.5)" : "var(--line)"}`,
              borderRadius: 10, marginBottom: 8,
            }}>
              <span style={{ color: "var(--ink-3)" }}><Icons.lock size={14} /></span>
              <input
                value={pw}
                onChange={(e) => setPw(e.target.value)}
                placeholder="새 비밀번호 (4자 이상)"
                style={{
                  flex: 1, background: "transparent", border: "none", outline: "none",
                  color: "var(--ink)", fontSize: 14, fontWeight: 600,
                  fontFamily: "ui-monospace, Menlo, monospace",
                  letterSpacing: "0.06em",
                }}
                autoFocus
              />
              <button
                type="button"
                onClick={onGenerate}
                title="자동 생성"
                style={{
                  padding: "5px 10px", borderRadius: 7,
                  fontSize: 11, fontWeight: 700,
                  background: "transparent",
                  border: "1px solid var(--line)",
                  color: "var(--ink-2)", cursor: "pointer",
                  display: "flex", alignItems: "center", gap: 4,
                }}
              >
                <Icons.refresh size={11} />자동 생성
              </button>
            </div>
            {error && <div style={{ fontSize: 11, color: "#dc2626", marginBottom: 12 }}>{error}</div>}
            <div style={{ fontSize: 11, color: "var(--ink-3)", marginBottom: 18, lineHeight: 1.5 }}>
              관리자가 직접 발급하는 임시 비밀번호. 사용자가 다음 로그인 후{" "}
              <strong style={{ color: "var(--ink-2)" }}>내 정보 → 비밀번호 변경</strong> 으로 직접 변경하도록 안내해 주세요.
            </div>

            <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
              <button
                onClick={onCancel}
                style={{
                  padding: "9px 16px", borderRadius: 9, fontSize: 13, fontWeight: 600,
                  background: "transparent", color: "var(--ink-2)",
                  border: "1px solid var(--line)", cursor: "pointer",
                }}
              >취소</button>
              <button
                onClick={onConfirm}
                disabled={!pw}
                style={{
                  padding: "9px 16px", borderRadius: 9, fontSize: 13, fontWeight: 700,
                  background: pw ? "linear-gradient(135deg, #4f46e5, #8b83ff)" : "var(--bg-sunk)",
                  color: pw ? "#fff" : "var(--ink-4)",
                  border: "none",
                  cursor: pw ? "pointer" : "not-allowed",
                  boxShadow: pw ? "0 8px 18px -6px rgba(79,70,229,0.45)" : "none",
                }}
              >재설정</button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function Toast({ toast }) {
  return (
    <div style={{
      position: "fixed", left: "50%", bottom: 32, transform: "translateX(-50%)",
      padding: "10px 16px", borderRadius: 10,
      background: toast.kind === "ok" ? "rgba(16,185,129,0.95)" : "rgba(239,68,68,0.95)",
      color: "#fff", fontSize: 13, fontWeight: 600,
      boxShadow: "0 12px 30px -8px rgba(0,0,0,0.4)",
      display: "flex", alignItems: "center", gap: 8,
      animation: "slide-in-up 220ms ease both",
      zIndex: 100,
    }}>
      {toast.kind === "ok" ? <Icons.check size={14} color="#fff" /> : <Icons.alert size={14} color="#fff" />}
      {toast.text}
    </div>
  );
}
