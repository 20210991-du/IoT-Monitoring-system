import { useState, useEffect, useMemo, useCallback, useRef } from "react";
import { Icons } from "../components/Icons.jsx";
import {
  listAllUsers,
  adminResetPassword,
  adminCreateUser,
  adminDeleteUser,
  adminSetMemo,
  adminUpdateUser,
  getAnnouncement,
  saveAnnouncement,
  getLoginBg,
  saveLoginBg,
  getChatModels,
  saveChatModels,
  ROLE_LABEL,
  ROLE_AVATAR,
  STATUS_LABEL,
} from "../lib/authMock.js";
import { getBootBar, setBootBar } from "../lib/userPrefs.js";

/* ── 관리자 페이지 (admin 전용 통합 대시보드) ─────────────────
 *  sub-tabs:
 *    1) 개요         — 사용자 KPI · 시스템 상태
 *    2) 운영자       — 사용자 등록 · 비밀번호 재설정
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
  superadmin: { fg: "#e11d48", bg: "rgba(225,29,72,0.10)", bd: "rgba(225,29,72,0.28)" },
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
// 초 단위까지 (마지막 로그인 시각용)
function fmtDateSec(iso) {
  if (!iso) return "—";
  const d = new Date(iso); const p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

// 관리자 sub-tab 정의 (드래그로 순서 변경 — 순서는 localStorage 저장)
const TAB_DEFS = [
  { k: "operators", label: "사용자 관리" },
  { k: "chatbot",   label: "챗봇 통계" },
  { k: "inq_admin", label: "상담원 문의함" },
  { k: "personas",  label: "챗봇 설정" },
  { k: "aimodels",  label: "AI 모델" },
  { k: "settings",  label: "시스템 설정" },
];
const TAB_ORDER_KEY = "siwon.admin.tabOrder";

// ── 메인 ─────────────────────────────────────────────────
export function Admin({ user, equipment, anomalies, watch, commOutage = [], apiStatus }) {
  const [section, setSection] = useState("operators");
  const [userTab, setUserTab] = useState("list");   // 사용자 관리 내부 서브탭: list(사용자 목록) | log(로그인 기록)
  const [settingsTab, setSettingsTab] = useState("settings");   // 시스템 설정 내부 서브탭: settings(설정) | notice(공지사항)
  const [chatbotTab, setChatbotTab] = useState("personas");   // 챗봇 설정 내부 서브탭: personas(봇 페르소나) | tokens(토큰 사용량)
  // 탭 순서 (드래그 변경 + localStorage 유지). 저장에 없는 새 탭은 뒤에 붙이고, 사라진 탭은 제거.
  const [tabOrder, setTabOrder] = useState(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(TAB_ORDER_KEY) || "[]");
      const valid = (Array.isArray(saved) ? saved : []).filter((k) => TAB_DEFS.some((t) => t.k === k));
      const missing = TAB_DEFS.filter((t) => !valid.includes(t.k)).map((t) => t.k);
      return [...valid, ...missing];
    } catch { return TAB_DEFS.map((t) => t.k); }
  });
  const dragKey = useRef(null);
  const [draggingKey, setDraggingKey] = useState(null);   // 드래그 중인 탭 (반투명)
  const [dragOverKey, setDragOverKey] = useState(null);   // 드롭 대상 탭 (앞에 표시선)
  const reorderTabs = (fromKey, toKey) => {
    if (!fromKey || fromKey === toKey) return;
    setTabOrder((order) => {
      const arr = order.slice();
      const from = arr.indexOf(fromKey), to = arr.indexOf(toKey);
      if (from < 0 || to < 0) return order;
      arr.splice(from, 1);
      arr.splice(to, 0, fromKey);
      try { localStorage.setItem(TAB_ORDER_KEY, JSON.stringify(arr)); } catch {}
      return arr;
    });
  };
  const [users, setUsers] = useState([]);
  const [toast, setToast] = useState(null);

  const reload = useCallback(async () => {
    const res = await listAllUsers(user);
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

  if (!user || (user.role !== "admin" && user.role !== "superadmin" && user.role !== "viewer" && user.role !== "guest")) {
    return (
      <div style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center", color: "var(--ink-3)" }}>
        관리자 권한이 필요합니다.
      </div>
    );
  }

  const readOnly = user.role === "viewer" || user.role === "guest";   // 뷰어·게스트 = 읽기 전용 관람 (편집 컨트롤 비활성 + 백엔드도 쓰기 차단, 이중 안전)
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
              사용자 관리 · AI 사용량 통계 · 시스템 설정
            </div>
          </div>
        </div>

        <div style={{ display: "flex", gap: 4, borderBottom: "1px solid var(--line)" }}>
          {tabOrder.map((k) => {
            const def = TAB_DEFS.find((t) => t.k === k);
            if (!def) return null;
            return (
              <SubTabBtn
                key={k} k={k} cur={section} set={setSection} label={def.label}
                draggable
                dragging={draggingKey === k}
                dropTarget={dragOverKey === k && !!draggingKey && draggingKey !== k}
                onDragStart={(e) => { dragKey.current = k; setDraggingKey(k); e.dataTransfer.effectAllowed = "move"; try { e.dataTransfer.setData("text/plain", k); } catch {} }}
                onDragOver={(e) => { e.preventDefault(); e.dataTransfer.dropEffect = "move"; if (dragOverKey !== k) setDragOverKey(k); }}
                onDrop={(e) => { e.preventDefault(); reorderTabs(dragKey.current, k); dragKey.current = null; setDraggingKey(null); setDragOverKey(null); }}
                onDragEnd={() => { dragKey.current = null; setDraggingKey(null); setDragOverKey(null); }}
              />
            );
          })}
        </div>
      </div>

      {/* ── 섹션 컨텐츠 ── */}
      <div style={{ flex: 1, overflow: "auto", padding: "20px 32px 32px" }}>
        {readOnly && (
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 16, padding: "10px 14px", borderRadius: 10, background: "rgba(100,116,139,0.12)", border: "1px solid rgba(100,116,139,0.28)", color: "var(--ink-2)", fontSize: 12.5, fontWeight: 600 }}>
            👁️ 읽기 전용(뷰어) — 모든 내용을 볼 수 있지만 편집·삭제·전송은 할 수 없습니다.
          </div>
        )}
        {/* 사용자 관리 + 로그인 기록 통합 서브탭 — fieldset 밖(읽기전용에도 탭 전환 가능). loginlog 옛 링크는 '로그인 기록'으로 */}
        {(section === "operators" || section === "loginlog") && (
          <div style={{ display: "flex", gap: 6, marginBottom: 14 }}>
            {[["list", "사용자 목록"], ["log", "로그인 기록"]].map(([k, lbl]) => {
              const active = (section === "loginlog" ? "log" : userTab) === k;
              return (
                <button key={k} onClick={() => { if (section === "loginlog") setSection("operators"); setUserTab(k); }} style={{
                  padding: "6px 14px", borderRadius: 999, fontSize: 13, fontWeight: 700, cursor: "pointer",
                  border: "1px solid " + (active ? "var(--brand)" : "var(--line)"),
                  background: active ? "var(--brand)" : "var(--bg-elev)", color: active ? "#fff" : "var(--ink-3)",
                }}>{lbl}</button>
              );
            })}
          </div>
        )}
        {/* 시스템 설정 + 공지사항 통합 서브탭 — fieldset 밖(읽기전용에도 탭 전환 가능). notice 옛 링크는 '공지사항'으로 */}
        {(section === "settings" || section === "notice") && (
          <div style={{ display: "flex", gap: 6, marginBottom: 14 }}>
            {[["settings", "시스템 설정"], ["notice", "공지사항"]].map(([k, lbl]) => {
              const active = (section === "notice" ? "notice" : settingsTab) === k;
              return (
                <button key={k} onClick={() => { if (section === "notice") setSection("settings"); setSettingsTab(k); }} style={{
                  padding: "6px 14px", borderRadius: 999, fontSize: 13, fontWeight: 700, cursor: "pointer",
                  border: "1px solid " + (active ? "var(--brand)" : "var(--line)"),
                  background: active ? "var(--brand)" : "var(--bg-elev)", color: active ? "#fff" : "var(--ink-3)",
                }}>{lbl}</button>
              );
            })}
          </div>
        )}
        {/* 챗봇 설정 = 봇 페르소나 + 토큰 사용량 통합 서브탭 — fieldset 밖(읽기전용에도 탭 전환 가능). tokens 옛 링크는 '토큰 사용량'으로 */}
        {(section === "personas" || section === "tokens") && (
          <div style={{ display: "flex", gap: 6, marginBottom: 14 }}>
            {[["personas", "봇 페르소나"], ["tokens", "토큰 사용량"]].map(([k, lbl]) => {
              const active = (section === "tokens" ? "tokens" : chatbotTab) === k;
              return (
                <button key={k} onClick={() => { if (section === "tokens") setSection("personas"); setChatbotTab(k); }} style={{
                  padding: "6px 14px", borderRadius: 999, fontSize: 13, fontWeight: 700, cursor: "pointer",
                  border: "1px solid " + (active ? "var(--brand)" : "var(--line)"),
                  background: active ? "var(--brand)" : "var(--bg-elev)", color: active ? "#fff" : "var(--ink-3)",
                }}>{lbl}</button>
              );
            })}
          </div>
        )}
        {/* 뷰어면 fieldset disabled 로 내부 모든 폼 컨트롤(버튼·입력·선택) 비활성. 백엔드도 쓰기 차단(이중 안전). 탭 네비는 헤더에 있어 영향 없음. */}
        <fieldset disabled={readOnly} style={{ border: "none", margin: 0, padding: 0, minInlineSize: 0 }}>
        {(section === "operators" || section === "loginlog") && (
          (section === "loginlog" ? "log" : userTab) === "list"
            ? <OperatorsSection user={user} users={users} counts={counts} reload={reload} setToast={setToast} />
            : <LoginLogSection />
        )}
        {section === "chatbot" && (
          <ChatbotStatsSection setToast={setToast} />
        )}
        {section === "inq_admin" && (
          <InquiriesSection channel="admin" setToast={setToast} />
        )}
        {(section === "personas" || section === "tokens") && (
          (section === "tokens" ? "tokens" : chatbotTab) === "tokens"
            ? <TokenUsageSection />
            : <BotPersonasSection setToast={setToast} />
        )}
        {(section === "settings" || section === "notice") && (
          (section === "notice" ? "notice" : settingsTab) === "notice"
            ? <NoticeSection setToast={setToast} />
            : <SettingsSection apiStatus={apiStatus} setToast={setToast} />
        )}
        {section === "aimodels" && (
          <AiModelsSection user={user} setToast={setToast} />
        )}
        </fieldset>
      </div>

      {toast && <Toast toast={toast} />}
    </div>
  );
}

// ── 봇 페르소나 설정 (on/off·키워드·모델·프롬프트 편집 — 재시작 없이 즉시 반영) ──
const PERSONA_MODEL_OPTS = [
  { v: "", label: "기본 (로컬 Qwen3.5:9b · 무료)" },
  { v: "qwen3.5:9b", label: "로컬 Qwen3.5:9b (무료)" },
  { v: "gpt-4o-mini", label: "GPT-4o mini (외부 · 소액 비용)" },
  { v: "gpt-4o", label: "GPT-4o (외부 · 고비용)" },
];
const PERSONA_LOUNGE_KEYS = ["park", "lee_jaeheon", "lee_duhyeon"];
// ── AI 모델 레지스트리 — 버전 목록 + 활성화 핫스왑 (활성화는 총괄 관리자만) ──
function AiModelsSection({ user, setToast }) {
  const [data, setData] = useState({ active: null, versions: [] });
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState("");   // 활성화 중인 version
  const [showUpload, setShowUpload] = useState(false);
  const [form, setForm] = useState({ version: "", label: "", note: "" });
  const [files, setFiles] = useState({ keras: null, scalers: null, thresholds: null, config: null });
  const [uploading, setUploading] = useState(false);
  const [rb, setRb] = useState({ running: false, done: 0, total: 55 });   // 재백필 상태
  const [editing, setEditing] = useState(null);                            // 메타 수정 중인 version
  const [editForm, setEditForm] = useState({ version: "", label: "", note: "", trained_at: "" });
  const [editFiles, setEditFiles] = useState({ keras: null, scalers: null, thresholds: null, config: null });
  const canEdit = user?.role === "admin" || user?.role === "superadmin";   // 편집 권한: 시원팀(admin) + 총괄(superadmin)

  const load = useCallback(async () => {
    try {
      const r = await fetch("/api/ai/models", { credentials: "same-origin" }).then((x) => x.json());
      if (r.ok) setData({ active: r.active, versions: r.versions || [] });
    } catch { /* ignore */ }
    setLoading(false);
  }, []);
  useEffect(() => { load(); }, [load]);

  // 재백필 상태 폴링 (4초)
  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try { const r = await fetch("/api/ai/models/rebackfill/status", { credentials: "same-origin" }).then((x) => x.json()); if (alive && r.ok) setRb(r); } catch { /* ignore */ }
    };
    poll();
    const id = setInterval(poll, 4000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  const startRebackfill = async () => {
    if (!canEdit || rb.running) return;
    if (!window.confirm("활성 모델로 과거 1년 AI 이상도 그래프를 재생성할까요?\n~25분 소요 · 8코어 사용(라이브 사이트가 잠시 느려질 수 있음).")) return;
    try {
      const r = await fetch("/api/ai/models/rebackfill", { method: "POST", credentials: "same-origin" }).then((x) => x.json());
      if (r.ok) { setRb(r); setToast && setToast({ kind: "ok", text: "과거 재생성 시작 — 약 25분" }); }
      else setToast && setToast({ kind: "err", text: r.error || "재생성 시작 실패" });
    } catch { setToast && setToast({ kind: "err", text: "재생성 오류" }); }
  };

  const activate = async (ver) => {
    if (!canEdit || busy) return;
    if (!window.confirm(`'${ver}' 모델을 활성화할까요?\n다음 예측 주기부터 이 모델이 적용됩니다. (과거 그래프 재생성은 별도 버튼)`)) return;
    setBusy(ver);
    try {
      const r = await fetch(`/api/ai/models/${encodeURIComponent(ver)}/activate`, { method: "POST", credentials: "same-origin" }).then((x) => x.json());
      if (r.ok) { setData({ active: r.active, versions: r.versions || [] }); setToast && setToast({ kind: "ok", text: `활성 모델 → ${ver}` }); }
      else setToast && setToast({ kind: "err", text: r.error || "활성화 실패" });
    } catch { setToast && setToast({ kind: "err", text: "활성화 오류" }); }
    setBusy("");
  };

  const del = async (ver) => {
    if (!canEdit || !window.confirm(`'${ver}' 버전을 삭제할까요? (되돌릴 수 없음)`)) return;
    try {
      const r = await fetch(`/api/ai/models/${encodeURIComponent(ver)}`, { method: "DELETE", credentials: "same-origin" }).then((x) => x.json());
      if (r.ok) { setData({ active: r.active, versions: r.versions || [] }); setToast && setToast({ kind: "ok", text: `삭제됨 — ${ver}` }); }
      else setToast && setToast({ kind: "err", text: r.error || "삭제 실패" });
    } catch { setToast && setToast({ kind: "err", text: "삭제 오류" }); }
  };
  const startEdit = (v) => {
    setEditing(v.version);
    setEditForm({ version: v.version, label: v.label || "", note: v.note || "", trained_at: v.trained_at || "" });
    setEditFiles({ keras: null, scalers: null, thresholds: null, config: null });
  };
  const saveEdit = async () => {
    const ver = editing;
    try {
      const body = { label: editForm.label, note: editForm.note, trained_at: editForm.trained_at };
      if (editForm.version && editForm.version.trim() !== ver) body.newVersion = editForm.version.trim();
      if (editFiles.keras)      body.keras_b64   = await readB64(editFiles.keras);
      if (editFiles.scalers)    body.scalers_b64 = await readB64(editFiles.scalers);
      if (editFiles.thresholds) body.thresholds  = await readText(editFiles.thresholds);
      if (editFiles.config)     body.config      = await readText(editFiles.config);
      const r = await fetch(`/api/ai/models/${encodeURIComponent(ver)}`, {
        method: "PATCH", credentials: "same-origin", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }).then((x) => x.json());
      if (r.ok) { setData({ active: r.active, versions: r.versions || [] }); setEditing(null); setToast && setToast({ kind: "ok", text: `수정됨 — ${r.version || ver}` }); }
      else setToast && setToast({ kind: "err", text: r.error || "수정 실패" });
    } catch (e) { setToast && setToast({ kind: "err", text: "수정 오류: " + (e.message || "") }); }
  };
  const readB64 = (file) => new Promise((res, rej) => { const r = new FileReader(); r.onload = () => res(String(r.result).split(",")[1] || ""); r.onerror = rej; r.readAsDataURL(file); });
  const readText = (file) => new Promise((res, rej) => { const r = new FileReader(); r.onload = () => res(String(r.result)); r.onerror = rej; r.readAsText(file); });
  // 폴더 통째 선택 → 이름으로 4개 아티팩트 자동 매칭 (이두현 common_model_artifacts/ 그대로)
  const onPickFolder = (e) => {
    const fl = Array.from(e.target.files || []);
    const find = (name) => fl.find((f) => f.name === name) || null;
    const next = {
      keras: find("common_lstm_autoencoder.keras"),
      scalers: find("group_scalers.pkl"),
      thresholds: find("device_thresholds.json"),
      config: find("model_config.json"),
    };
    setFiles(next);
    const miss = Object.entries(next).filter(([, v]) => !v).map(([k]) => k);
    if (miss.length) setToast && setToast({ kind: "err", text: `폴더에서 못 찾음: ${miss.join(", ")}` });
    else setToast && setToast({ kind: "ok", text: "폴더에서 4개 아티팩트 자동 매칭됨 ✓" });
  };
  const submitUpload = async () => {
    if (uploading) return;
    if (!/^[A-Za-z0-9._-]{3,40}$/.test(form.version.trim())) { setToast && setToast({ kind: "err", text: "버전명: 영문/숫자/._- 3~40자" }); return; }
    if (!files.keras || !files.scalers || !files.thresholds || !files.config) { setToast && setToast({ kind: "err", text: "4개 파일 모두 선택하세요" }); return; }
    setUploading(true);
    try {
      const [keras_b64, scalers_b64, thresholds, config] = await Promise.all([
        readB64(files.keras), readB64(files.scalers), readText(files.thresholds), readText(files.config),
      ]);
      const r = await fetch("/api/ai/models/upload", {
        method: "POST", credentials: "same-origin", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ version: form.version.trim(), label: form.label.trim(), note: form.note.trim(), keras_b64, scalers_b64, thresholds, config }),
      }).then((x) => x.json());
      if (r.ok) {
        setData({ active: r.active, versions: r.versions || [] });
        setForm({ version: "", label: "", note: "" }); setFiles({ keras: null, scalers: null, thresholds: null, config: null }); setShowUpload(false);
        setToast && setToast({ kind: "ok", text: `업로드됨 — ${r.version}` });
      } else setToast && setToast({ kind: "err", text: r.error || "업로드 실패" });
    } catch (e) { setToast && setToast({ kind: "err", text: "업로드 오류: " + (e.message || "") }); }
    setUploading(false);
  };

  const fmtBytes = (n) => (n ? (n / 1048576).toFixed(1) + "MB" : "-");
  const lblS = { display: "flex", flexDirection: "column", gap: 2, fontSize: 11.5, fontWeight: 600, color: "var(--ink-3)" };
  const inpS = { marginTop: 3, padding: "7px 10px", borderRadius: 8, border: "1px solid var(--line-soft)", background: "var(--bg)", color: "var(--ink)", fontSize: 13, fontFamily: "inherit", outline: "none" };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <div>
        <div style={{ fontSize: 14, fontWeight: 800, color: "var(--ink)" }}>AI 모델 레지스트리</div>
        <div style={{ fontSize: 12, color: "var(--ink-3)", marginTop: 3 }}>
          LSTM-AE 모델을 버전별로 관리하고 활성 모델을 교체합니다. 활성화는 다음 예측 주기부터 적용 · {canEdit ? "편집 가능(시원팀 관리자)" : "보기 전용 — 편집은 시원팀(관리자)만"}.
        </div>
      </div>
      {canEdit && (
        <div style={{ display: "flex", alignItems: "center", gap: 12, padding: "10px 14px", border: "1px solid var(--line-soft)", borderRadius: 10, background: "var(--bg-sunk)", flexWrap: "wrap" }}>
          <div style={{ flex: 1, minWidth: 180 }}>
            <div style={{ fontSize: 12.5, fontWeight: 700, color: "var(--ink-2)" }}>과거 AI 이상도 그래프 재생성</div>
            <div style={{ fontSize: 11, color: "var(--ink-4)", marginTop: 2 }}>
              {rb.running
                ? `진행 중… ${rb.done}/${rb.total} 단말 (8코어 병렬, ~25분)`
                : "활성 모델 기준으로 과거 1년 예측을 다시 계산합니다. (모델 교체 후 권장)"}
            </div>
            {rb.running && (
              <div style={{ marginTop: 6, height: 5, borderRadius: 999, background: "var(--line-soft)", overflow: "hidden" }}>
                <div style={{ height: "100%", width: `${Math.min(100, Math.round((rb.done / Math.max(1, rb.total)) * 100))}%`, background: "var(--brand)", transition: "width 600ms ease" }} />
              </div>
            )}
          </div>
          <button type="button" disabled={rb.running} onClick={startRebackfill} style={{
            padding: "9px 16px", borderRadius: 9, border: "1px solid var(--brand)",
            background: rb.running ? "var(--bg-elev)" : "transparent", color: rb.running ? "var(--ink-3)" : "var(--brand)",
            fontSize: 12.5, fontWeight: 700, cursor: rb.running ? "default" : "pointer", whiteSpace: "nowrap",
          }}>{rb.running ? "재생성 중…" : "과거 재생성"}</button>
        </div>
      )}
      {loading ? (
        <div style={{ fontSize: 13, color: "var(--ink-3)", padding: 16 }}>불러오는 중…</div>
      ) : data.versions.length === 0 ? (
        <div style={{ fontSize: 13, color: "var(--ink-3)", padding: 16 }}>등록된 모델 버전이 없습니다.</div>
      ) : (
        <div style={{ display: "grid", gap: 10 }}>
          {data.versions.map((v) => (
            <div key={v.version} style={{
              border: `1px solid ${v.active ? "var(--brand)" : "var(--line-soft)"}`,
              background: v.active ? "var(--brand-wash)" : "var(--bg-sunk)",
              borderRadius: 12, padding: "14px 16px",
              display: "flex", alignItems: "center", justifyContent: "space-between", gap: 14, flexWrap: "wrap",
            }}>
              <div style={{ minWidth: 0, flex: 1 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                  <span className="mono" style={{ fontSize: 14, fontWeight: 800, color: "var(--ink)" }}>{v.version}</span>
                  {v.active && <span style={{ fontSize: 10, fontWeight: 800, color: "#fff", background: "var(--brand)", padding: "2px 8px", borderRadius: 999 }}>활성</span>}
                  {!v.complete && <span style={{ fontSize: 10, fontWeight: 700, color: "var(--err)", background: "rgba(239,68,68,0.12)", padding: "2px 8px", borderRadius: 999 }}>아티팩트 누락</span>}
                  {v.label && <span style={{ fontSize: 11.5, color: "var(--ink-2)" }}>{v.label}</span>}
                </div>
                {editing === v.version ? (
                  <div style={{ display: "flex", flexDirection: "column", gap: 6, marginTop: 8, maxWidth: 460 }}>
                    <label style={lblS}>버전명<input value={editForm.version} onChange={(e) => setEditForm((f) => ({ ...f, version: e.target.value }))} placeholder="영문/숫자/._- 3~40자" style={inpS} /></label>
                    <label style={lblS}>라벨<input value={editForm.label} onChange={(e) => setEditForm((f) => ({ ...f, label: e.target.value }))} placeholder="설명 라벨" style={inpS} /></label>
                    <label style={lblS}>학습일<input value={editForm.trained_at} onChange={(e) => setEditForm((f) => ({ ...f, trained_at: e.target.value }))} placeholder="2026-05-27 14:48" style={inpS} /></label>
                    <label style={lblS}>메모<input value={editForm.note} onChange={(e) => setEditForm((f) => ({ ...f, note: e.target.value }))} placeholder="변경점 등" style={inpS} /></label>
                    <div style={{ fontSize: 10.5, color: "var(--ink-4)", marginTop: 2 }}>아티팩트 교체 (비워두면 기존 유지):</div>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
                      {[["keras", ".keras", ".keras"], ["scalers", "scalers.pkl", ".pkl"], ["thresholds", "thresholds.json", ".json"], ["config", "config.json", ".json"]].map(([key, label, accept]) => (
                        <label key={key} style={{ ...lblS, fontSize: 10.5 }}>{label}{editFiles[key] && <span style={{ color: "var(--ok)" }}> ✓</span>}
                          <input type="file" accept={accept} onChange={(e) => setEditFiles((f) => ({ ...f, [key]: e.target.files?.[0] || null }))} style={{ fontSize: 11, marginTop: 2 }} />
                        </label>
                      ))}
                    </div>
                    <div style={{ fontSize: 10, color: "var(--ink-4)", lineHeight: 1.4 }}>⚠️ 활성 버전의 아티팩트를 바꾸면 다음 예측부터 즉시 반영 · pkl 은 신뢰된 파일만.</div>
                    <div style={{ display: "flex", gap: 6, marginTop: 2 }}>
                      <button type="button" onClick={saveEdit} style={{ padding: "6px 14px", borderRadius: 8, border: "none", background: "var(--brand)", color: "#fff", fontSize: 12, fontWeight: 700, cursor: "pointer" }}>저장</button>
                      <button type="button" onClick={() => setEditing(null)} style={{ padding: "6px 12px", borderRadius: 8, border: "1px solid var(--line)", background: "transparent", color: "var(--ink-3)", fontSize: 12, fontWeight: 600, cursor: "pointer" }}>취소</button>
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="mono" style={{ fontSize: 11, color: "var(--ink-3)", marginTop: 6, display: "flex", gap: 12, flexWrap: "wrap" }}>
                      <span>학습 {v.trained_at || "-"}</span>
                      <span>단말 {v.device_count ?? "-"}</span>
                      <span>time_steps {v.time_steps ?? "-"}</span>
                      <span>피처 {v.feature_count ?? "-"}</span>
                      <span>임계평균 {v.mean_threshold != null ? Number(v.mean_threshold).toExponential(2) : "-"}</span>
                      <span>{fmtBytes(v.keras_bytes)}</span>
                    </div>
                    {v.note && <div style={{ fontSize: 11, color: "var(--ink-4)", marginTop: 4 }}>{v.note}</div>}
                  </>
                )}
              </div>
              <div style={{ flexShrink: 0, display: "flex", alignItems: "center", gap: 8 }}>
                {v.active && <span style={{ fontSize: 12, fontWeight: 700, color: "var(--brand)" }}>● 사용 중</span>}
                {canEdit && editing !== v.version && (
                  <>
                    {!v.active && v.complete && (
                      <button type="button" disabled={!!busy} onClick={() => activate(v.version)} style={{
                        padding: "8px 16px", borderRadius: 9, border: "1px solid var(--brand)",
                        background: busy === v.version ? "var(--bg-elev)" : "var(--brand)", color: busy === v.version ? "var(--ink-3)" : "#fff",
                        fontSize: 12.5, fontWeight: 700, cursor: busy ? "default" : "pointer", opacity: busy && busy !== v.version ? 0.5 : 1,
                      }}>{busy === v.version ? "활성화 중…" : "활성화"}</button>
                    )}
                    <button type="button" onClick={() => startEdit(v)} title="수정 — 버전명·아티팩트·메타 전체" style={{
                      padding: "8px 12px", borderRadius: 9, border: "1px solid var(--line)", background: "transparent",
                      color: "var(--ink-2)", fontSize: 12.5, fontWeight: 700, cursor: "pointer",
                    }}>수정</button>
                    {!v.active && (
                      <button type="button" onClick={() => del(v.version)} title="이 버전 삭제" style={{
                        padding: "8px 12px", borderRadius: 9, border: "1px solid var(--line)", background: "transparent",
                        color: "var(--ink-3)", fontSize: 12.5, fontWeight: 700, cursor: "pointer",
                      }}>삭제</button>
                    )}
                  </>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* 새 모델 버전 업로드 (총괄 관리자) */}
      {canEdit && (
        <div style={{ border: "1px dashed var(--line)", borderRadius: 12, padding: "12px 16px", background: "var(--bg-sunk)" }}>
          <button type="button" onClick={() => setShowUpload((s) => !s)} style={{
            border: "none", background: "transparent", color: "var(--ink-2)", fontSize: 13, fontWeight: 700, cursor: "pointer", padding: 0,
          }}>＋ 새 모델 버전 업로드 {showUpload ? "▲" : "▼"}</button>
          {showUpload && (
            <div style={{ display: "flex", flexDirection: "column", gap: 10, marginTop: 12 }}>
              <label style={{ ...lblS, padding: "9px 11px", border: "1px dashed var(--brand)", borderRadius: 8, background: "var(--brand-wash)", color: "var(--ink-2)" }}>
                📁 이두현 산출물 폴더 통째 선택 → 4개 자동 매칭 (권장)
                <input type="file" webkitdirectory="" directory="" multiple onChange={onPickFolder} style={{ fontSize: 11, marginTop: 5 }} />
              </label>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                <label style={lblS}>버전명 (영문/숫자/._-)
                  <input value={form.version} onChange={(e) => setForm((f) => ({ ...f, version: e.target.value }))} placeholder="예: v20260701-unified" style={inpS} />
                </label>
                <label style={lblS}>라벨 (설명)
                  <input value={form.label} onChange={(e) => setForm((f) => ({ ...f, label: e.target.value }))} placeholder="예: 통합 모델 v2" style={inpS} />
                </label>
              </div>
              <label style={lblS}>메모 (선택)
                <input value={form.note} onChange={(e) => setForm((f) => ({ ...f, note: e.target.value }))} placeholder="변경점 등" style={inpS} />
              </label>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                {[["keras", "모델 (.keras)", ".keras"], ["scalers", "스케일러 (group_scalers.pkl)", ".pkl"], ["thresholds", "임계값 (device_thresholds.json)", ".json"], ["config", "설정 (model_config.json)", ".json"]].map(([key, label, accept]) => (
                  <label key={key} style={lblS}>{label}
                    <input type="file" accept={accept} onChange={(e) => setFiles((f) => ({ ...f, [key]: e.target.files?.[0] || null }))} style={{ fontSize: 12, marginTop: 4 }} />
                    {files[key] && <span style={{ fontSize: 10.5, color: "var(--ok)", marginTop: 2 }}>✓ {files[key].name}</span>}
                  </label>
                ))}
              </div>
              <div style={{ fontSize: 10.5, color: "var(--ink-4)", lineHeight: 1.5 }}>
                ⚠️ <b>group_scalers.pkl</b> 은 예측 시 pickle 로 로드됩니다 — <b>본인이 학습시킨 신뢰할 수 있는 파일만</b> 올리세요. 업로드는 활성화와 별개(등록 후 "활성화" 필요).
              </div>
              <div style={{ display: "flex", gap: 8 }}>
                <button type="button" disabled={uploading} onClick={submitUpload} style={{
                  padding: "9px 18px", borderRadius: 9, border: "none", background: uploading ? "var(--bg-elev)" : "var(--brand)",
                  color: uploading ? "var(--ink-3)" : "#fff", fontSize: 13, fontWeight: 700, cursor: uploading ? "default" : "pointer",
                }}>{uploading ? "업로드 중…" : "버전 등록"}</button>
                <button type="button" onClick={() => { setShowUpload(false); setForm({ version: "", label: "", note: "" }); setFiles({ keras: null, scalers: null, thresholds: null, config: null }); }} style={{
                  padding: "9px 14px", borderRadius: 9, border: "1px solid var(--line)", background: "transparent", color: "var(--ink-3)", fontSize: 13, fontWeight: 600, cursor: "pointer",
                }}>취소</button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function BotPersonasSection({ setToast }) {
  const [items, setItems] = useState([]);
  const [drafts, setDrafts] = useState({});
  const [loading, setLoading] = useState(true);
  const [savingKey, setSavingKey] = useState(null);

  const load = useCallback(async () => {
    try { const r = await fetch("/api/admin/bot-personas").then((x) => x.json()); if (r.ok) setItems(r.personas || []); }
    catch { /* ignore */ }
    setLoading(false);
  }, []);
  useEffect(() => { load(); }, [load]);

  const val = (p, f) => (drafts[p.persona_key]?.[f] !== undefined ? drafts[p.persona_key][f] : (p[f] ?? ""));
  const setField = (key, f, v) => setDrafts((d) => ({ ...d, [key]: { ...d[key], [f]: v } }));
  const dirty = (key) => !!drafts[key] && Object.keys(drafts[key]).length > 0;
  const clearDraft = (key) => setDrafts((d) => { const n = { ...d }; delete n[key]; return n; });

  const patch = async (key, body) => {
    const r = await fetch(`/api/admin/bot-personas/${key}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then((x) => x.json());
    if (!r.ok) throw new Error(r.error || "저장 실패");
    return r.persona;
  };
  const toggleEnabled = async (p) => {
    try {
      const next = p.enabled ? 0 : 1;
      const updated = await patch(p.persona_key, { enabled: next });
      setItems((arr) => arr.map((x) => (x.persona_key === p.persona_key ? (updated || { ...x, enabled: next }) : x)));
      setToast && setToast({ kind: "ok", text: `${p.name} ${next ? "켜짐" : "꺼짐"}` });
    } catch (e) { setToast && setToast({ kind: "err", text: e.message }); }
  };
  const save = async (p) => {
    const key = p.persona_key;
    if (!dirty(key)) return;
    setSavingKey(key);
    try {
      const updated = await patch(key, drafts[key]);
      setItems((arr) => arr.map((x) => (x.persona_key === key ? (updated || { ...x, ...drafts[key] }) : x)));
      clearDraft(key);
      setToast && setToast({ kind: "ok", text: `${p.name} 저장됨` });
    } catch (e) { setToast && setToast({ kind: "err", text: e.message }); }
    finally { setSavingKey(null); }
  };

  const inp = { width: "100%", padding: "7px 9px", borderRadius: 8, border: "1px solid var(--line)", background: "var(--bg-elev)", color: "var(--ink)", fontSize: 12.5, outline: "none", fontFamily: "inherit", boxSizing: "border-box" };
  const lbl = { fontSize: 11, fontWeight: 700, color: "var(--ink-3)", marginBottom: 4, display: "block" };

  if (loading) return <div style={{ color: "var(--ink-4)", fontSize: 13, padding: "24px 0" }}>불러오는 중…</div>;

  return (
    <div style={{ maxWidth: 920 }}>
      <div style={{ fontSize: 13, color: "var(--ink-3)", marginBottom: 4, lineHeight: 1.6 }}>
        AI 봇 페르소나를 켜고/끄거나 <b>키워드(자동 라우팅)·모델·시스템 프롬프트</b>를 편집합니다. 저장하면 <b>재시작 없이 즉시</b> 반영돼요.
      </div>
      <div style={{ fontSize: 11.5, color: "var(--ink-4)", marginBottom: 16, lineHeight: 1.6 }}>
        · 공통 안전수칙(비밀번호·접속정보·개인정보 노출 금지, 근거 밖 추측 금지)은 프롬프트와 별개로 <b>항상 적용</b>됩니다.<br />
        · GPT 모델 선택 시 공개 LLM 호출로 <b>외부 비용</b>이 발생합니다(기본 로컬은 무료).
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        {items.map((p) => {
          const isLounge = PERSONA_LOUNGE_KEYS.includes(p.persona_key);
          const on = !!p.enabled;
          const d = dirty(p.persona_key);
          return (
            <div key={p.persona_key} style={{ border: "1px solid var(--line)", borderRadius: 14, padding: 16, background: "var(--bg-elev)", opacity: on ? 1 : 0.72 }}>
              {/* 헤더 — 아바타 · 이름 · key · on/off */}
              <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 14 }}>
                {p.avatar
                  ? <img src={p.avatar} alt="" style={{ width: 38, height: 38, borderRadius: "50%", objectFit: "cover", flexShrink: 0, border: "1px solid var(--line)" }} />
                  : <div style={{ width: 38, height: 38, borderRadius: "50%", background: "var(--bg-sunk)", flexShrink: 0 }} />}
                <div style={{ flex: 1, minWidth: 0 }}>
                  <input value={val(p, "name")} onChange={(e) => setField(p.persona_key, "name", e.target.value)}
                    style={{ ...inp, fontWeight: 800, fontSize: 14, padding: "4px 8px" }} />
                  <div style={{ fontSize: 10.5, color: "var(--ink-4)", marginTop: 3, fontFamily: "JetBrains Mono, ui-monospace, monospace" }}>
                    {p.persona_key}{isLounge && " · 공개문의 페르소나"}
                  </div>
                </div>
                <button onClick={() => toggleEnabled(p)} title={on ? "끄기" : "켜기"} style={{
                  flexShrink: 0, padding: "6px 14px", borderRadius: 999, border: "none", cursor: "pointer",
                  fontSize: 12, fontWeight: 800, color: "#fff", background: on ? "var(--ok)" : "var(--ink-4)",
                }}>{on ? "ON" : "OFF"}</button>
              </div>

              {/* 필드 그리드 */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 12 }}>
                <div style={{ gridColumn: "1 / -1" }}>
                  <label style={lbl}>담당 분야 (lane)</label>
                  <input value={val(p, "lane")} onChange={(e) => setField(p.persona_key, "lane", e.target.value)} style={inp} />
                </div>
                <div>
                  <label style={lbl}>모델</label>
                  <select value={val(p, "model")} onChange={(e) => setField(p.persona_key, "model", e.target.value)} style={{ ...inp, cursor: "pointer" }}>
                    {PERSONA_MODEL_OPTS.map((o) => <option key={o.v} value={o.v}>{o.label}</option>)}
                  </select>
                </div>
                <div style={{ display: "flex", gap: 12 }}>
                  <div style={{ width: 90 }}>
                    <label style={lbl}>정렬 순서</label>
                    <input type="number" value={val(p, "sort_order")} onChange={(e) => setField(p.persona_key, "sort_order", e.target.value)} style={inp} />
                  </div>
                  <div style={{ flex: 1 }}>
                    <label style={lbl}>대표 답변자{isLounge ? "" : " (공개문의용)"}</label>
                    <label style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12.5, color: "var(--ink)", cursor: "pointer", padding: "6px 0" }}>
                      <input type="checkbox" checked={!!val(p, "is_fallback")} onChange={(e) => setField(p.persona_key, "is_fallback", e.target.checked ? 1 : 0)} />
                      애매한 질문 fallback
                    </label>
                  </div>
                </div>
                <div style={{ gridColumn: "1 / -1" }}>
                  <label style={lbl}>키워드 (쉼표로 구분 — 자동 라우팅 점수에 사용)</label>
                  <textarea value={val(p, "keywords")} onChange={(e) => setField(p.persona_key, "keywords", e.target.value)} rows={2}
                    style={{ ...inp, resize: "vertical", lineHeight: 1.5 }} />
                </div>
                <div style={{ gridColumn: "1 / -1" }}>
                  <label style={lbl}>시스템 프롬프트 (페르소나 지시문 — 공통 안전수칙은 별도로 항상 적용됨)</label>
                  <textarea value={val(p, "system_prompt")} onChange={(e) => setField(p.persona_key, "system_prompt", e.target.value)} rows={6}
                    style={{ ...inp, resize: "vertical", lineHeight: 1.55, fontSize: 12 }} />
                </div>
                <div style={{ gridColumn: "1 / -1" }}>
                  <label style={lbl}>연락 이메일</label>
                  <input value={val(p, "contact_email")} onChange={(e) => setField(p.persona_key, "contact_email", e.target.value)} style={inp} placeholder="(없음)" />
                </div>
              </div>

              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <button onClick={() => save(p)} disabled={!d || savingKey === p.persona_key} style={{
                  padding: "8px 18px", borderRadius: 9, border: "none", cursor: d ? "pointer" : "not-allowed",
                  fontSize: 12.5, fontWeight: 800, color: "#fff",
                  background: d ? "var(--brand)" : "var(--line)", opacity: savingKey === p.persona_key ? 0.6 : 1,
                }}>{savingKey === p.persona_key ? "저장 중…" : "저장"}</button>
                {d && <span style={{ fontSize: 11.5, color: "var(--ink-4)" }}>변경됨 — 저장 시 즉시 반영</span>}
                {d && <button onClick={() => clearDraft(p.persona_key)} style={{ border: "none", background: "transparent", color: "var(--ink-4)", cursor: "pointer", fontSize: 11.5, marginLeft: "auto" }}>되돌리기</button>}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── sub-tabs UI ────────────────────────────────────────
// ── 문의함 (사용자 문의/버그 신고 — admin) ──
function InquiriesSection({ setToast, channel = "admin" }) {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("needs");   // needs(미답변) | open(미처리) | done | all
  const [query, setQuery] = useState("");          // 검색어 (내용·사용자)
  const [drafts, setDrafts] = useState({});        // id -> 답변 초안
  const load = useCallback(async () => {
    try { const r = await fetch("/api/inquiries").then((x) => x.json()); if (r.ok) setItems(r.inquiries || []); } catch { /* ignore */ }
    setLoading(false);
  }, []);
  useEffect(() => { load(); const id = setInterval(load, 8000); return () => clearInterval(id); }, [load]);
  const patch = async (id, body, okMsg) => {
    try {
      await fetch(`/api/inquiries/${id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      setItems((arr) => arr.map((q) => (q.id === id ? { ...q, ...body } : q)));
      setToast && setToast({ kind: "ok", text: okMsg });
    } catch { setToast && setToast({ kind: "err", text: "변경 실패" }); }
  };
  // 문의 삭제 (소프트 삭제 — 백엔드 deleted_at, 복구 가능). 확인 후 목록에서 제거.
  const del = async (id) => {
    if (!window.confirm("이 문의를 삭제할까요? (목록에서 숨겨지며 DB에는 보존됩니다)")) return;
    try {
      const r = await fetch(`/api/inquiries/${id}`, { method: "DELETE" }).then((x) => x.json());
      if (!r.ok) throw new Error(r.error || "");
      setItems((arr) => arr.filter((q) => q.id !== id));
      setToast && setToast({ kind: "ok", text: "문의 삭제됨" });
    } catch { setToast && setToast({ kind: "err", text: "삭제 실패" }); }
  };
  // 답변 저장 (alsoDone=true 면 완료 처리까지 한 번에)
  const saveReply = (id, alsoDone) => {
    const text = (drafts[id] || "").trim();
    if (!text) return;
    patch(id, alsoDone ? { adminReply: text, status: "done" } : { adminReply: text }, alsoDone ? "답변 후 완료 처리" : "답변 저장됨");
    setDrafts((d) => { const n = { ...d }; delete n[id]; return n; });
  };
  // 처리 상태 — needs(미답변) / answered(답변했으나 미완료) / done(완료)
  const statusOf = (q) => (q.status === "done" ? "done" : (q.adminReply ? "answered" : "needs"));
  const chanItems = items.filter((q) => (q.target || "admin") === channel);   // 이 탭의 채널만
  const counts = {
    needs: chanItems.filter((q) => statusOf(q) === "needs").length,
    open:  chanItems.filter((q) => q.status === "open").length,
    done:  chanItems.filter((q) => q.status === "done").length,
    all:   chanItems.length,
  };
  const ql = query.trim().toLowerCase();
  const view = chanItems
    .filter((q) => (filter === "all" ? true : filter === "done" ? q.status === "done" : filter === "open" ? q.status === "open" : statusOf(q) === "needs"))
    .filter((q) => !ql || `${q.message} ${q.displayName || ""} ${q.loginId || ""} ${q.adminReply || ""}`.toLowerCase().includes(ql));
  const fmt = (d) => { try { return new Date(d).toLocaleString("ko-KR", { timeZone: "Asia/Seoul", month: "numeric", day: "numeric", hour: "numeric", minute: "2-digit" }); } catch { return ""; } };
  const STAT = {
    needs:    { label: "미답변", fg: "#ef4444", bg: "rgba(239,68,68,0.12)" },
    answered: { label: "답변함", fg: "#b45309", bg: "rgba(245,158,11,0.16)" },
    done:     { label: "완료",   fg: "#047857", bg: "rgba(16,185,129,0.12)" },
  };
  const hasDraft = (id) => !!(drafts[id] || "").trim();
  const pill = (kbl, cur, set) => kbl.map(([k, label]) => (
    <button key={k} onClick={() => set(k)} style={{
      padding: "5px 12px", borderRadius: 8, fontSize: 12, fontWeight: 700, cursor: "pointer",
      border: "1px solid " + (cur === k ? "var(--brand)" : "var(--line)"),
      background: cur === k ? "var(--brand)" : "var(--bg-elev)",
      color: cur === k ? "#fff" : "var(--ink-3)",
    }}>{label}</button>
  ));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14, maxWidth: 920 }}>
      {/* 요약 KPI */}
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
        {[
          { label: "미답변", value: counts.needs, fg: "#ef4444" },
          { label: "미처리", value: counts.open,  fg: "var(--brand)" },
          { label: "완료",   value: counts.done,  fg: "var(--ok)" },
          { label: "전체",   value: counts.all,   fg: "var(--ink-3)" },
        ].map((k) => (
          <div key={k.label} style={{ flex: 1, minWidth: 88, padding: "9px 14px", borderRadius: 12, background: "var(--bg-elev)", border: "1px solid var(--line)" }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: "var(--ink-3)" }}>{k.label}</div>
            <div style={{ fontSize: 23, fontWeight: 800, color: k.fg, lineHeight: 1.15 }}>{k.value}</div>
          </div>
        ))}
      </div>
      {/* 필터 + 검색 */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
        {pill([["needs", `미답변 ${counts.needs}`], ["open", `미처리 ${counts.open}`], ["done", `완료 ${counts.done}`]], filter, setFilter)}
        <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="검색 (내용·사용자)"
          style={{ marginLeft: "auto", minWidth: 150, padding: "6px 10px", borderRadius: 8, border: "1px solid var(--line)", background: "var(--bg-sunk)", color: "var(--ink)", fontSize: 12.5, outline: "none" }} />
      </div>
      {loading && <div style={{ color: "var(--ink-3)", fontSize: 13 }}>불러오는 중…</div>}
      {!loading && view.length === 0 && <div style={{ color: "var(--ink-4)", fontSize: 13, padding: "24px 0", textAlign: "center" }}>해당 문의가 없습니다.</div>}
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        {view.map((q) => {
          const isDev = (q.target || "admin") === "developer";
          const st = statusOf(q); const s = STAT[st];
          return (
          <div key={q.id} style={{
            border: "1px solid var(--line)", borderLeft: `4px solid ${s.fg}`, borderRadius: 12, padding: "16px 18px",
            background: "var(--bg-elev)", opacity: st === "done" ? 0.72 : 1,
            boxShadow: st === "done" ? "none" : "0 1px 3px rgba(0,0,0,0.04)",
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 7, flexWrap: "wrap" }}>
              <span style={{
                fontSize: 11, fontWeight: 800, padding: "2px 8px", borderRadius: 999,
                background: isDev ? "rgba(249,115,22,0.12)" : "rgba(100,116,139,0.12)",
                color: isDev ? "var(--accent)" : "var(--ink-3)",
                display: "inline-flex", alignItems: "center", gap: 5,
              }}>
                <img src={isDev ? "/avatars/developer.png" : "/avatars/agent.png"} alt="" style={{ width: 16, height: 16, borderRadius: "50%", objectFit: "cover" }} />
                {isDev ? "개발자" : "상담원"}
              </span>
              {!isDev && q.kind === "bug" && (
                <span style={{ fontSize: 11, fontWeight: 800, padding: "2px 8px", borderRadius: 999, background: "rgba(239,68,68,0.12)", color: "var(--err)" }}>버그</span>
              )}
              <span style={{ fontSize: 11, fontWeight: 800, padding: "2px 8px", borderRadius: 999, background: s.bg, color: s.fg }}>{s.label}</span>
              <span style={{ fontSize: 12.5, fontWeight: 700, color: "var(--ink)" }}>{q.displayName || q.loginId || "사용자"}</span>
              {q.loginId && <span style={{ fontSize: 11, color: "var(--ink-4)" }}>@{q.loginId}</span>}
              <span style={{ marginLeft: "auto", fontSize: 11, color: "var(--ink-4)" }}>{fmt(q.createdAt)}</span>
              <button onClick={() => patch(q.id, { status: q.status === "done" ? "open" : "done" }, q.status === "done" ? "미처리로 되돌림" : "완료 처리")} style={{
                padding: "4px 11px", borderRadius: 8, fontSize: 11.5, fontWeight: 700, cursor: "pointer",
                border: "1px solid " + (q.status === "done" ? "var(--line)" : "var(--ok)"),
                background: "transparent",
                color: q.status === "done" ? "var(--ink-3)" : "#047857",
              }}>{q.status === "done" ? "↩ 되돌리기" : "✓ 완료 처리"}</button>
              <button onClick={() => del(q.id)} title="문의 삭제" style={{
                padding: "4px 9px", borderRadius: 8, fontSize: 11.5, fontWeight: 700, cursor: "pointer",
                border: "1px solid rgba(239,68,68,0.4)", background: "transparent", color: "var(--err)",
              }}>삭제</button>
            </div>
            <div style={{ fontSize: 10.5, fontWeight: 700, color: "var(--ink-4)", letterSpacing: "0.04em", marginBottom: 3 }}>문의 내용</div>
            <div style={{ fontSize: 14.5, fontWeight: 500, color: "var(--ink)", whiteSpace: "pre-wrap", lineHeight: 1.55 }}>{q.message}</div>
            {Array.isArray(q.images) && q.images.length > 0 && (
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginTop: 8 }}>
                {q.images.map((src, i) => (
                  <a key={i} href={src} target="_blank" rel="noreferrer" style={{ display: "inline-block" }}>
                    <img src={src} alt="첨부 이미지" style={{ maxWidth: 200, maxHeight: 180, borderRadius: 8, border: "1px solid var(--line)", display: "block" }} />
                  </a>
                ))}
              </div>
            )}
            {q.botReply && (
              <div style={{ marginTop: 10, padding: "9px 12px", borderRadius: 8, background: "var(--bg-sunk)", borderLeft: "3px solid var(--brand)", fontSize: 12.5, color: "var(--ink-2)", whiteSpace: "pre-wrap", lineHeight: 1.55 }}>
                <div style={{ fontWeight: 700, color: "var(--brand)", fontSize: 11, marginBottom: 3 }}>{isDev ? "AI 설명" : "봇 답변"}</div>
                {q.botReply}
              </div>
            )}
            {q.adminReply && (
              <div style={{ marginTop: 8, padding: "9px 12px", borderRadius: 8, background: "rgba(16,185,129,0.08)", borderLeft: "3px solid var(--ok)", fontSize: 12.5, color: "var(--ink)", whiteSpace: "pre-wrap", lineHeight: 1.55 }}>
                <div style={{ fontWeight: 700, color: "#047857", fontSize: 11, marginBottom: 3 }}>{isDev ? "개발자 답변" : "관리자 답변"}</div>
                {q.adminReply}
              </div>
            )}
            {/* 직접 답변 작성 — 답변+완료(원클릭) / 저장(미완료 유지) */}
            <div style={{ marginTop: 12, paddingTop: 12, borderTop: "1px solid var(--line)", display: "flex", gap: 6, alignItems: "stretch" }}>
              <textarea
                value={drafts[q.id] ?? ""}
                onChange={(e) => setDrafts((d) => ({ ...d, [q.id]: e.target.value }))}
                placeholder={q.adminReply ? "답변 수정…" : (isDev ? "개발자 답변 작성…" : "답변 작성…")}
                rows={2}
                style={{
                  flex: 1, resize: "vertical", padding: "7px 9px", borderRadius: 8,
                  border: "1px solid var(--line)", background: "var(--bg-sunk)", color: "var(--ink)",
                  fontSize: 12.5, fontFamily: "inherit", outline: "none", lineHeight: 1.5,
                }}
              />
              <div style={{ display: "flex", flexDirection: "column", gap: 5, justifyContent: "center" }}>
                <button onClick={() => saveReply(q.id, true)} disabled={!hasDraft(q.id)} style={{
                  padding: "0 12px", height: 30, borderRadius: 8, fontSize: 11.5, fontWeight: 700, whiteSpace: "nowrap",
                  cursor: hasDraft(q.id) ? "pointer" : "not-allowed", border: "none",
                  background: hasDraft(q.id) ? "var(--ok)" : "var(--bg-sunk)", color: hasDraft(q.id) ? "#fff" : "var(--ink-4)",
                }}>답변+완료</button>
                <button onClick={() => saveReply(q.id, false)} disabled={!hasDraft(q.id)} style={{
                  padding: "0 12px", height: 30, borderRadius: 8, fontSize: 11.5, fontWeight: 700, whiteSpace: "nowrap",
                  cursor: hasDraft(q.id) ? "pointer" : "not-allowed", border: "1px solid var(--line)",
                  background: "var(--bg-elev)", color: hasDraft(q.id) ? "var(--ink-2)" : "var(--ink-4)",
                }}>{q.adminReply ? "수정 저장" : "저장"}</button>
              </div>
            </div>
          </div>
          );
        })}
      </div>
    </div>
  );
}

function SubTabBtn({ k, cur, set, label, badge, draggable, dragging, dropTarget, onDragStart, onDragOver, onDrop, onDragEnd }) {
  const active = cur === k;
  return (
    <button
      onClick={() => set(k)}
      draggable={draggable}
      onDragStart={onDragStart}
      onDragOver={onDragOver}
      onDrop={onDrop}
      onDragEnd={onDragEnd}
      title={draggable ? "드래그해서 순서 변경" : undefined}
      style={{
        position: "relative", padding: "10px 16px",
        background: "transparent", border: "none",
        color: active ? "var(--brand)" : "var(--ink-3)",
        fontSize: 13, fontWeight: 700, cursor: "pointer",
        display: "flex", alignItems: "center", gap: 6,
        opacity: dragging ? 0.4 : 1, transition: "opacity 120ms",
      }}
    >
      {dropTarget && <span style={{ position: "absolute", left: -2, top: 6, bottom: 6, width: 3, background: "var(--brand)", borderRadius: 2 }} />}
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
      {ROLE_LABEL[user.role] || user.role} · {user.id}
    </div>
  );
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// 1) 개요 섹션
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function OverviewSection({ users, counts, equipment, anomalies, watch, commOutage = [], apiStatus, onJump }) {
  const lastActiveJoin = users
    .filter((u) => u.status === "active")
    .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt))[0];

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
      {/* KPI 카드 */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 12 }}>
        <KpiCard label="전체 사용자"     value={counts.all}      hint="등록된 모든 운영자" tone="brand" />
        <KpiCard label="활성 계정"       value={counts.active}   hint="현재 로그인 가능" tone="ok" />
      </div>

      <div style={{ display: "grid", gap: 16 }}>
        {/* 시스템 상태 */}
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
            <StatRow
              label="통신 장애"
              value={`${commOutage?.length || 0}건`}
              color={(commOutage?.length || 0) > 0 ? "var(--ink-3)" : "var(--ink-2)"}
            />
            {lastActiveJoin && (
              <StatRow
                label="최근 등록"
                value={`${lastActiveJoin.name} · ${fmtDate(lastActiveJoin.createdAt)}`}
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
  { k: "all",      ko: "전체" },
];

function OperatorsSection({ user, users, counts, reload, setToast }) {
  const [filter, setFilter] = useState("all");
  const [search, setSearch] = useState("");
  const [resetTarget, setResetTarget] = useState(null);
  const [resetPw, setResetPw]         = useState("");
  const [resetError, setResetError]   = useState("");
  const [resetDone, setResetDone]     = useState(null); // {userId, newPw} 또는 null
  const [deleteTarget, setDeleteTarget] = useState(null);  // 삭제 대상 사용자
  const [deleteErr, setDeleteErr]       = useState("");
  const [memoTarget, setMemoTarget]   = useState(null);    // 메모 편집 대상
  const [memoText, setMemoText]       = useState("");
  const [memoErr, setMemoErr]         = useState("");
  const [editTarget, setEditTarget]   = useState(null);    // 수정 대상 사용자
  const [eName, setEName] = useState("");
  const [eRole, setERole] = useState("viewer");
  const [eMemo, setEMemo] = useState("");
  const [ePw, setEPw]     = useState("");
  const [eErr, setEErr]   = useState("");
  const [eDone, setEDone] = useState(null);                // 저장 후 새 비밀번호 표시용
  // 사용자 등록 폼 (관리자 직접 생성 — 공개 회원가입 대체)
  const [showCreate, setShowCreate] = useState(false);
  const [cId, setCId]     = useState("");
  const [cName, setCName] = useState("");
  const [cRole, setCRole] = useState("viewer");
  const [cPw, setCPw]     = useState("");
  const [cMemo, setCMemo] = useState("");
  const [cErr, setCErr]   = useState("");
  const [createdCred, setCreatedCred] = useState(null); // {id, pw, name} — 등록 직후 1회 표시

  const filtered = useMemo(() => {
    // "전체"는 게스트(1회용) 제외 — 누적은 보존하되 목록이 게스트로 길어지지 않게. 게스트는 '게스트' 탭에서만.
    let list = filter === "all" ? users.filter((u) => u.role !== "guest") : users.filter((u) => u.role === filter);   // 역할별 필터
    const q = search.trim().toLowerCase();
    if (q) {
      list = list.filter((u) =>
        [u.id, u.name].some((v) => (v || "").toLowerCase().includes(q))
      );
    }
    const roleOrder = { superadmin: 0, admin: 1, viewer: 2, guest: 3 };
    return list.slice().sort((a, b) => {
      const o = (roleOrder[a.role] ?? 9) - (roleOrder[b.role] ?? 9);   // 역할별 정렬: 총괄 관리자 → 관리자 → 뷰어 → 게스트
      if (o !== 0) return o;
      return new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime();   // 같은 역할 내 등록일 순
    });
  }, [users, filter, search]);

  // 역할별 필터 태그 (전체 + 존재하는 역할만, 각 카운트 포함)
  const filterTags = useMemo(() => {
    const byRole = users.reduce((a, u) => { a[u.role] = (a[u.role] || 0) + 1; return a; }, {});
    return [
      { k: "all", ko: "전체", cnt: users.length - (byRole.guest || 0) },   // 전체 = 게스트 제외(비게스트)
      ...["superadmin", "admin", "viewer", "guest"].filter((r) => byRole[r]).map((r) => ({ k: r, ko: ROLE_LABEL[r] || r, cnt: byRole[r] })),
    ];
  }, [users]);

  const openReset = (t) => {
    setResetTarget(t);
    setResetPw("");
    setResetError("");
    setResetDone(null);
  };
  const closeReset = () => { setResetTarget(null); setResetPw(""); setResetError(""); setResetDone(null); };
  const confirmReset = async () => {
    if (!resetTarget) return;
    const res = await adminResetPassword(user, resetTarget.id, resetPw);
    if (!res.ok) { setResetError(res.error); return; }
    setResetDone({ userId: res.userId, newPw: res.newPw });
    setToast({ kind: "ok", text: `${resetTarget.name} 님의 비밀번호를 재설정했습니다.` });
    reload();
  };
  const closeDelete = () => { setDeleteTarget(null); setDeleteErr(""); };
  const handleDelete = async () => {
    if (!deleteTarget) return;
    const res = await adminDeleteUser(user, deleteTarget.id);
    if (!res.ok) { setDeleteErr(res.error); return; }
    setToast({ kind: "ok", text: `${deleteTarget.name} (${deleteTarget.id}) 계정을 삭제했습니다.` });
    setDeleteTarget(null); setDeleteErr("");
    reload();
  };
  const openMemo = (u) => { setMemoTarget(u); setMemoText(u.memo || ""); setMemoErr(""); };
  const closeMemo = () => { setMemoTarget(null); setMemoText(""); setMemoErr(""); };
  const confirmMemo = async () => {
    if (!memoTarget) return;
    const res = await adminSetMemo(user, memoTarget.id, memoText);
    if (!res.ok) { setMemoErr(res.error); return; }
    setToast({ kind: "ok", text: `${memoTarget.name} 님의 메모를 저장했습니다.` });
    setMemoTarget(null); setMemoText(""); setMemoErr("");
    reload();
  };
  const openEdit = (u) => { setEditTarget(u); setEName(u.name || ""); setERole(u.role || "viewer"); setEMemo(u.memo || ""); setEPw(""); setEErr(""); setEDone(null); };
  const closeEdit = () => { setEditTarget(null); setEName(""); setEMemo(""); setEPw(""); setEErr(""); setEDone(null); };
  const genEditPw = () => {
    const chars = "abcdefghjkmnpqrstuvwxyzABCDEFGHJKMNPQRSTUVWXYZ23456789";
    let s = ""; for (let i = 0; i < 10; i++) s += chars[Math.floor(Math.random() * chars.length)];
    setEPw(s); setEErr("");
  };
  const confirmEdit = async () => {
    if (!editTarget) return;
    const res = await adminUpdateUser(user, editTarget.id, { name: eName, role: eRole, memo: eMemo, newPw: ePw });
    if (!res.ok) { setEErr(res.error); return; }
    setToast({ kind: "ok", text: `${eName} (${editTarget.id}) 정보를 수정했습니다.` });
    reload();
    if (res.newPw) setEDone(res.newPw);   // 새 비번 설정 시 모달 유지하며 표시
    else closeEdit();
  };
  const genRandomPw = () => {
    // 데모용 랜덤 8자 (영숫자)
    const chars = "abcdefghjkmnpqrstuvwxyzABCDEFGHJKMNPQRSTUVWXYZ23456789";
    let s = "";
    for (let i = 0; i < 8; i++) s += chars[Math.floor(Math.random() * chars.length)];
    setResetPw(s);
    setResetError("");
  };
  const genCreatePw = () => {
    const chars = "abcdefghjkmnpqrstuvwxyzABCDEFGHJKMNPQRSTUVWXYZ23456789";
    let s = ""; for (let i = 0; i < 8; i++) s += chars[Math.floor(Math.random() * chars.length)];
    setCPw(s); setCErr("");
  };
  const handleCreate = async () => {
    const res = await adminCreateUser(user, { id: cId.trim(), pw: cPw, name: cName.trim(), role: cRole, memo: cMemo.trim() });
    if (!res.ok) { setCErr(res.error); return; }
    setCreatedCred({ id: cId.trim(), pw: cPw, name: cName.trim() });
    setToast({ kind: "ok", text: `${cName.trim()} (${cId.trim()}) 계정을 등록했습니다.` });
    setCId(""); setCName(""); setCPw(""); setCMemo(""); setCErr("");
    reload();
  };

  return (
    <>
      {/* 사용자 요약 — 전체·활성·대기 + 역할별 (예전 개요 스타일) */}
      {(() => {
        const byRole = users.reduce((a, u) => { a[u.role] = (a[u.role] || 0) + 1; return a; }, {});
        const cards = [
          { label: "전체", value: counts.all, fg: "var(--brand)" },
          { label: "활성", value: counts.active, fg: "var(--ok)" },
          { label: "대기", value: counts.pending, fg: "var(--warn)" },
          { label: "총괄 관리자", value: byRole.superadmin || 0, fg: "var(--ink-2)" },
          { label: "시원팀", value: byRole.admin || 0, fg: "var(--ink-2)" },
          { label: "뷰어", value: byRole.viewer || 0, fg: "var(--ink-2)" },
          { label: "게스트", value: byRole.guest || 0, fg: "var(--ink-2)" },
        ];
        return (
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 14 }}>
            {cards.map((k) => (
              <div key={k.label} style={{ flex: 1, minWidth: 78, padding: "8px 12px", borderRadius: 10, background: "var(--bg-elev)", border: "1px solid var(--line)" }}>
                <div style={{ fontSize: 10.5, fontWeight: 700, color: "var(--ink-3)" }}>{k.label}</div>
                <div style={{ fontSize: 20, fontWeight: 800, color: k.fg, lineHeight: 1.15 }}>{k.value}</div>
              </div>
            ))}
          </div>
        );
      })()}
      {/* 사용자 등록 — 관리자 직접 생성 (즉시 활성) */}
      <div style={{ marginBottom: 10, padding: "8px 12px", borderRadius: 12, background: "var(--bg-elev)", border: "1px solid var(--line)" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10, marginBottom: showCreate ? 12 : 0 }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: "var(--ink)", minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            사용자 등록 {showCreate && <span style={{ fontSize: 11, fontWeight: 500, color: "var(--ink-3)" }}>— 관리자가 직접 계정 생성 · 즉시 활성 · 역할 지정 (공개 가입은 뷰어로 생성)</span>}
          </div>
          <button type="button" onClick={() => { setShowCreate((v) => !v); setCErr(""); setCreatedCred(null); }}
            style={{ padding: "6px 12px", borderRadius: 8, fontSize: 12, fontWeight: 700,
              background: showCreate ? "var(--bg-sunk)" : "var(--brand)", color: showCreate ? "var(--ink-2)" : "#fff",
              border: `1px solid ${showCreate ? "var(--line)" : "var(--brand)"}`, cursor: "pointer" }}>
            {showCreate ? "닫기" : "+ 새 사용자"}
          </button>
        </div>
        {showCreate && (
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "flex-end" }}>
            {[["ID", cId, setCId, "영문/숫자 2~20"], ["이름", cName, setCName, "표시 이름"], ["비밀번호", cPw, setCPw, "4자 이상"], ["메모", cMemo, setCMemo, "선택 · 관리자 메모"]].map(([lab, val, set, ph]) => (
              <label key={lab} style={{ display: "flex", flexDirection: "column", gap: 4, fontSize: 11, color: "var(--ink-3)", fontWeight: 600 }}>
                {lab}
                <input value={val} onChange={(e) => { set(e.target.value); setCErr(""); }} placeholder={ph}
                  style={{ height: 34, width: 150, padding: "0 10px", borderRadius: 8, border: "1px solid var(--line)", background: "var(--bg)", color: "var(--ink)", fontSize: 13 }} />
              </label>
            ))}
            <label style={{ display: "flex", flexDirection: "column", gap: 4, fontSize: 11, color: "var(--ink-3)", fontWeight: 600 }}>
              역할
              <select value={cRole} onChange={(e) => setCRole(e.target.value)}
                style={{ height: 34, width: 120, padding: "0 8px", borderRadius: 8, border: "1px solid var(--line)", background: "var(--bg)", color: "var(--ink)", fontSize: 13 }}>
                {Object.entries(ROLE_LABEL).filter(([k]) => ["superadmin", "admin", "viewer", "guest"].includes(k)).map(([k, v]) => <option key={k} value={k} disabled={k === "guest"}>{v}</option>)}
              </select>
            </label>
            <button type="button" onClick={genCreatePw}
              style={{ height: 34, padding: "0 12px", borderRadius: 8, fontSize: 12, fontWeight: 600, background: "var(--bg-sunk)", color: "var(--ink-2)", border: "1px solid var(--line)", cursor: "pointer" }}>
              랜덤 비번
            </button>
            <button type="button" onClick={handleCreate}
              style={{ height: 34, padding: "0 18px", borderRadius: 8, fontSize: 13, fontWeight: 700, background: "var(--brand)", color: "#fff", border: "1px solid var(--brand)", cursor: "pointer" }}>
              등록
            </button>
          </div>
        )}
        {cErr && <div style={{ marginTop: 8, fontSize: 12, color: "var(--err, #dc2626)", fontWeight: 600 }}>{cErr}</div>}
        {createdCred && (
          <div style={{ marginTop: 10, padding: "10px 12px", borderRadius: 8, background: "rgba(16,185,129,0.10)", border: "1px solid rgba(16,185,129,0.32)", fontSize: 12.5, color: "var(--ink)", lineHeight: 1.6 }}>
            ✅ <strong>{createdCred.name}</strong> 계정 등록 완료 — 아래 정보를 사용자에게 전달하세요.
            <span style={{ color: "var(--ink-3)" }}> (이 화면을 벗어나면 비밀번호는 다시 확인할 수 없습니다.)</span>
            <div style={{ marginTop: 6, fontFamily: "ui-monospace, monospace", fontSize: 13 }}>
              ID <strong>{createdCred.id}</strong>　·　비밀번호 <strong>{createdCred.pw}</strong>
            </div>
          </div>
        )}
      </div>

      {/* 필터 + 검색 */}
      <div style={{ display: "flex", gap: 10, marginBottom: 14, alignItems: "center" }}>
        <div style={{ display: "flex", gap: 6 }}>
          {filterTags.map((f) => {
            const active = filter === f.k;
            const cnt = f.cnt;
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
            <col style={{ width: "15%" }} />
            <col style={{ width: "14%" }} />
            <col style={{ width: "10%" }} />
            <col style={{ width: "15%" }} />
            <col style={{ width: "15%" }} />
            <col style={{ width: "17%" }} />
            <col style={{ width: "14%" }} />
          </colgroup>
          <thead>
            <tr style={{ background: "var(--bg)", borderBottom: "1px solid var(--line)" }}>
              {["ID", "이름", "역할", "메모", "등록일", "마지막 로그인", "처리"].map((h) => (
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
              <tr><td colSpan={7} style={{ padding: "48px 14px", textAlign: "center", color: "var(--ink-3)", fontSize: 13 }}>
                표시할 사용자가 없습니다.
              </td></tr>
            )}
            {filtered.map((u, i) => {
              return (
                <tr key={u.id} style={{
                  borderBottom: i === filtered.length - 1 ? "none" : "1px solid var(--line)",
                  background: i % 2 === 1 ? "rgba(0,0,0,0.015)" : "transparent",
                  height: 56, verticalAlign: "middle",
                }}>
                  <td style={{ padding: "10px 14px", fontSize: 13, color: "var(--ink)", fontWeight: 600 }}>
                    <span style={{ display: "inline-flex", alignItems: "center", gap: 8, minWidth: 0 }}>
                      <img src={u.avatar || ROLE_AVATAR[u.role] || "/avatars/guest.png"} alt="" style={{ width: 26, height: 26, borderRadius: "50%", objectFit: "cover", flexShrink: 0, border: "1px solid var(--line)" }} />
                      <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                        {u.id}
                        {u.id === user.id && <span style={{ fontSize: 10, fontWeight: 700, color: "var(--brand)", marginLeft: 6 }}>(나)</span>}
                      </span>
                    </span>
                  </td>
                  <td style={{ padding: "10px 14px", fontSize: 13, color: "var(--ink-2)" }}>{u.name}</td>
                  <td style={{ padding: "10px 14px", fontSize: 13, color: "var(--ink-2)" }}>
                    {ROLE_LABEL[u.role] || u.role}
                  </td>
                  <td style={{ padding: "10px 14px", fontSize: 12 }}>
                    <span style={{ display: "block", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", color: u.memo ? "var(--ink-2)" : "var(--ink-3)" }} title={u.memo || ""}>
                      {u.memo || "—"}
                    </span>
                  </td>
                  <td style={{ padding: "10px 14px", fontSize: 12, color: "var(--ink-3)" }}>
                    {fmtDate(u.createdAt)}
                  </td>
                  <td style={{ padding: "10px 14px", fontSize: 11.5, color: u.lastLoginAt ? "var(--ink-2)" : "var(--ink-4)", fontFamily: "JetBrains Mono, ui-monospace, monospace" }}>
                    {fmtDateSec(u.lastLoginAt)}
                  </td>
                  <td style={{ padding: "10px 14px" }}>
                    {u.role === "guest" ? (
                      <span style={{ fontSize: 11, color: "var(--ink-4)" }}>—</span>   /* 게스트 계정은 처리(수정/삭제) 불가 — 버튼 제거 */
                    ) : (
                      <div style={{ display: "flex", gap: 6 }}>
                        <ActionButton tone="brand" icon={<Icons.pencil size={13} />} label="수정" onClick={() => openEdit(u)} />
                      </div>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

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

      {/* 계정 삭제 확인 모달 */}
      {deleteTarget && (
        <DeleteUserModal
          target={deleteTarget}
          error={deleteErr}
          onConfirm={handleDelete}
          onCancel={closeDelete}
        />
      )}

      {/* 사용자 메모 모달 */}
      {memoTarget && (
        <MemoModal
          target={memoTarget}
          value={memoText}
          setValue={(v) => { setMemoText(v); setMemoErr(""); }}
          error={memoErr}
          onConfirm={confirmMemo}
          onCancel={closeMemo}
        />
      )}

      {/* 사용자 수정 모달 */}
      {editTarget && (
        <EditUserModal
          target={editTarget}
          name={eName} setName={(v) => { setEName(v); setEErr(""); }}
          role={eRole} setRole={(v) => { setERole(v); setEErr(""); }}
          memo={eMemo} setMemo={setEMemo}
          pw={ePw} setPw={(v) => { setEPw(v); setEErr(""); }}
          error={eErr} done={eDone}
          onGenPw={genEditPw}
          onConfirm={confirmEdit}
          onCancel={closeEdit}
          canDelete={editTarget.id !== user.id}
          onDelete={() => { const t = editTarget; closeEdit(); setDeleteTarget(t); setDeleteErr(""); }}
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

const LOGIN_BG_OPTIONS = [
  { k: "light",  ko: "빛퍼짐", desc: "기본 · 빛이 퍼지는 영상" },
  { k: "flower", ko: "꽃",     desc: "이전 배경 영상" },
];

// 챗봇 모델 잠금 카탈로그 — Dashboard CHAT_MODELS 와 동일(로컬 3 + GPT 3). value 가 백엔드 화이트리스트와 일치해야 함.
const CHAT_MODEL_CATALOG = [
  { group: "로컬 (Mac Studio · 무료)", items: [
    { value: "qwen3.5:9b",  label: "Qwen 빠름",   hint: "속도 빠름" },
    { value: "qwen3:14b",   label: "Qwen 균형",   hint: "속도 중간" },
    { value: "qwen3.5:27b", label: "Qwen 고품질", hint: "매우 느림" },
  ]},
  { group: "GPT (외부 · OpenAI · 유료)", items: [
    { value: "gpt-4o-mini", label: "GPT 빠름",     hint: "외부 · 빠름" },
    { value: "gpt-5",       label: "GPT 고품질",   hint: "외부 · 고품질" },
    { value: "gpt-5.5",     label: "GPT 최고품질", hint: "외부 · 최고품질" },
  ]},
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

  // 로그인 배경화면 (서버 저장 — 전체 사용자 공통)
  const [loginBg, setLoginBg] = useState("light");
  useEffect(() => { getLoginBg().then((r) => { if (r.ok && r.bg) setLoginBg(r.bg); }); }, []);
  const applyLoginBg = async (v) => {
    setLoginBg(v);
    const r = await saveLoginBg(v);
    setToast(r.ok
      ? { kind: "ok", text: `로그인 배경을 '${v === "flower" ? "꽃" : "빛퍼짐"}'(으)로 저장했습니다.` }
      : { kind: "err", text: r.error || "배경 저장 실패" });
  };

  // 부팅 로딩바 (BootScreen on/off — 이 브라우저 localStorage, 다음 접속부터 적용)
  const [bootBar, setBootBarState] = useState(() => getBootBar());
  const saveBootBar = (v) => {
    setBootBarState(v);
    setBootBar(v);
    setToast({ kind: "ok", text: `부팅 로딩바를 ${v ? "켜짐" : "꺼짐"}으로 저장했습니다. (다음 접속 시 적용)` });
  };

  // 챗봇 모델 잠금 (서버 저장 — 전체 사용자 공통, 즉시 적용). null=로딩중.
  const [enabledModels, setEnabledModels] = useState(null);
  useEffect(() => { getChatModels().then((r) => { if (r.ok && Array.isArray(r.enabled)) setEnabledModels(r.enabled); }); }, []);
  const toggleModel = async (value) => {
    if (!enabledModels) return;
    const has = enabledModels.includes(value);
    const next = has ? enabledModels.filter((v) => v !== value) : [...enabledModels, value];
    if (!next.length) { setToast({ kind: "err", text: "최소 1개 모델은 허용해야 합니다." }); return; }
    const prev = enabledModels;
    setEnabledModels(next);   // 낙관적 반영
    const r = await saveChatModels(next);
    if (r.ok) {
      setEnabledModels(r.enabled);
      setToast({ kind: "ok", text: `모델 잠금을 저장했습니다. (허용 ${r.enabled.length}개 · 즉시 적용)` });
    } else {
      setEnabledModels(prev);   // 롤백
      setToast({ kind: "err", text: r.error || "모델 잠금 저장 실패" });
    }
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

      {/* 로그인 배경화면 — 서버 저장(전체 사용자 공통) */}
      <SettingPanel title="로그인 배경화면" desc="로그인 페이지 배경 영상 — 저장 시 모든 사용자에게 적용">
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {LOGIN_BG_OPTIONS.map((o) => {
            const active = loginBg === o.k;
            return (
              <button
                key={o.k}
                onClick={() => applyLoginBg(o.k)}
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
          {LOGIN_BG_OPTIONS.find((o) => o.k === loginBg)?.desc} · 저장 즉시 반영(로그인 페이지 새로고침 시)
        </div>
      </SettingPanel>

      {/* 챗봇 모델 잠금 — 서버 저장(전체 사용자 공통 · 즉시 적용) */}
      <SettingPanel title="AI 챗봇 모델 잠금" desc="관제 도우미·라운지에서 사용 가능한 모델을 선택. 끄면 모든 사용자(관리자 포함)의 피커에서 숨겨지고 백엔드에서도 차단됩니다.">
        {enabledModels == null ? (
          <div style={{ fontSize: 12, color: "var(--ink-3)" }}>불러오는 중…</div>
        ) : (
          <div style={{ display: "grid", gap: 14 }}>
            {CHAT_MODEL_CATALOG.map((grp) => (
              <div key={grp.group}>
                <div style={{ fontSize: 11, fontWeight: 800, color: "var(--ink-3)", marginBottom: 6, letterSpacing: 0.2 }}>{grp.group}</div>
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                  {grp.items.map((m) => {
                    const on = enabledModels.includes(m.value);
                    return (
                      <button
                        key={m.value}
                        onClick={() => toggleModel(m.value)}
                        title={`${m.value} · ${m.hint}`}
                        style={{
                          display: "inline-flex", alignItems: "center", gap: 6,
                          padding: "8px 13px", borderRadius: 999, fontSize: 13, fontWeight: 700,
                          background: on ? "var(--brand)" : "transparent",
                          color: on ? "#fff" : "var(--ink-3)",
                          border: `1px solid ${on ? "var(--brand)" : "var(--line)"}`,
                          cursor: "pointer", transition: "all 140ms ease",
                        }}
                      >
                        <span style={{ fontSize: 11 }}>{on ? "✓" : "🔒"}</span>{m.label}
                      </button>
                    );
                  })}
                </div>
              </div>
            ))}
            <div style={{ fontSize: 11, color: "var(--ink-3)" }}>
              허용 {enabledModels.length}개 · 클릭 시 즉시 저장·적용(최소 1개 유지). GPT는 외부 전송·유료입니다.
            </div>
          </div>
        )}
      </SettingPanel>

      {/* 부팅 로딩바 — 이 브라우저 localStorage(다음 접속 시 적용) */}
      <SettingPanel title="부팅 로딩바" desc="앱 시작 시 로그인 전 로딩 진행바(BootScreen) 표시 — 이 브라우저에만 적용 · 다음 접속부터">
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {[{ k: true, ko: "켜짐" }, { k: false, ko: "꺼짐" }].map((o) => {
            const active = bootBar === o.k;
            return (
              <button
                key={String(o.k)}
                onClick={() => saveBootBar(o.k)}
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
          {bootBar ? "다음 접속 시 부팅 시퀀스(진행바·로그) 표시" : "꺼짐 · 로그인 화면 바로 표시"}
        </div>
      </SettingPanel>

      {/* 알림 임계값 (read-only — 모델/백엔드 영역) */}
      <SettingPanel title="알림 임계값" desc="이상 탐지 MSE 임계 (모델 학습 시 결정)">
        <div style={{ display: "grid", gap: 6 }}>
          <ReadonlyRow label="이상 임계"          value="단말별 threshold 초과" hint="ai_predictions.threshold" />
          <ReadonlyRow label="관찰 임계"          value="threshold 0.7 ~ 1.0배" hint="3단계 분류 기준" />
          <ReadonlyRow label="통신장애 판정"       value="24시간 이상 무측정"    hint="백엔드 mapStatus" />
        </div>
        <div style={{ marginTop: 10, fontSize: 11, color: "var(--ink-3)" }}>
          임계값 조정은 모델 재학습 또는 백엔드 설정 파일 수정 필요. 5/11 1차 자문 보고서 후 이 화면에서 직접 조정 가능하도록 확장 예정.
        </div>
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

function EditUserModal({ target, name, setName, role, setRole, memo, setMemo, pw, setPw, error, done, onGenPw, onConfirm, onCancel, onDelete, canDelete }) {
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
        width: 480, padding: 28, borderRadius: 16,
        background: "var(--bg-elev)", border: "1px solid var(--line)",
        boxShadow: "0 30px 80px -20px rgba(0,0,0,0.4)",
      }}>
        <div style={{ fontSize: 17, fontWeight: 800, marginBottom: 6, color: "var(--ink)" }}>
          사용자 수정
        </div>
        <div style={{ fontSize: 13, color: "var(--ink-3)", marginBottom: 16, lineHeight: 1.6 }}>
          <strong style={{ color: "var(--ink-2)" }}>{target.id}</strong> 계정 정보를 수정합니다.
        </div>
        {done ? (
          <div style={{ padding: "14px 16px", borderRadius: 10, background: "rgba(16,185,129,0.10)", border: "1px solid rgba(16,185,129,0.32)", fontSize: 13, color: "var(--ink)", lineHeight: 1.7 }}>
            저장되었습니다. <strong>새 비밀번호</strong>를 사용자에게 전달하세요:
            <div style={{ marginTop: 6, fontFamily: "ui-monospace, monospace", fontSize: 15, fontWeight: 700 }}>{done}</div>
            <div style={{ marginTop: 4, fontSize: 11, color: "var(--ink-3)" }}>이 화면을 닫으면 다시 확인할 수 없습니다.</div>
            <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 12 }}>
              <button onClick={onCancel} style={{ padding: "9px 16px", borderRadius: 9, fontSize: 13, fontWeight: 700, background: "var(--brand)", color: "#fff", border: "none", cursor: "pointer" }}>닫기</button>
            </div>
          </div>
        ) : (
          <>
            <div style={{ display: "grid", gap: 12 }}>
              <label style={{ display: "flex", flexDirection: "column", gap: 5, fontSize: 11, fontWeight: 600, color: "var(--ink-3)" }}>
                이름
                <input value={name} onChange={(e) => setName(e.target.value)}
                  style={{ height: 36, padding: "0 10px", borderRadius: 8, border: "1px solid var(--line)", background: "var(--bg)", color: "var(--ink)", fontSize: 13 }} />
              </label>
              <label style={{ display: "flex", flexDirection: "column", gap: 5, fontSize: 11, fontWeight: 600, color: "var(--ink-3)" }}>
                역할
                <select value={role} onChange={(e) => setRole(e.target.value)}
                  style={{ height: 36, padding: "0 8px", borderRadius: 8, border: "1px solid var(--line)", background: "var(--bg)", color: "var(--ink)", fontSize: 13 }}>
                  {Object.entries(ROLE_LABEL).filter(([k]) => ["superadmin", "admin", "viewer", "guest"].includes(k)).map(([k, v]) => <option key={k} value={k} disabled={k === "guest"}>{v}</option>)}
                </select>
              </label>
              <label style={{ display: "flex", flexDirection: "column", gap: 5, fontSize: 11, fontWeight: 600, color: "var(--ink-3)" }}>
                메모
                <textarea value={memo} onChange={(e) => setMemo(e.target.value)} maxLength={500} rows={3}
                  placeholder="선택 · 관리자 메모"
                  style={{ padding: "8px 10px", borderRadius: 8, border: "1px solid var(--line)", background: "var(--bg)", color: "var(--ink)", fontSize: 13, resize: "vertical", fontFamily: "inherit", boxSizing: "border-box" }} />
              </label>
              <label style={{ display: "flex", flexDirection: "column", gap: 5, fontSize: 11, fontWeight: 600, color: "var(--ink-3)" }}>
                새 비밀번호 <span style={{ fontWeight: 500 }}>(비우면 변경 안 함)</span>
                <div style={{ display: "flex", gap: 6 }}>
                  <input value={pw} onChange={(e) => setPw(e.target.value)} placeholder="4자 이상"
                    style={{ flex: 1, height: 36, padding: "0 10px", borderRadius: 8, border: "1px solid var(--line)", background: "var(--bg)", color: "var(--ink)", fontSize: 13 }} />
                  <button type="button" onClick={onGenPw}
                    style={{ height: 36, padding: "0 12px", borderRadius: 8, fontSize: 12, fontWeight: 600, background: "var(--bg-sunk)", color: "var(--ink-2)", border: "1px solid var(--line)", cursor: "pointer", whiteSpace: "nowrap" }}>랜덤</button>
                </div>
              </label>
            </div>
            {error && <div style={{ marginTop: 12, fontSize: 12, color: "var(--err, #dc2626)", fontWeight: 600 }}>{error}</div>}
            <div style={{ display: "flex", gap: 8, justifyContent: "flex-end", alignItems: "center", marginTop: 18 }}>
              {canDelete && (
                <button onClick={onDelete} style={{ marginRight: "auto", padding: "9px 14px", borderRadius: 9, fontSize: 13, fontWeight: 700, background: "transparent", color: "var(--err)", border: "1px solid var(--err)", cursor: "pointer" }}>계정 삭제</button>
              )}
              <button onClick={onCancel} style={{ padding: "9px 16px", borderRadius: 9, fontSize: 13, fontWeight: 600, background: "transparent", color: "var(--ink-2)", border: "1px solid var(--line)", cursor: "pointer" }}>취소</button>
              <button onClick={onConfirm} style={{ padding: "9px 16px", borderRadius: 9, fontSize: 13, fontWeight: 700, background: "var(--brand)", color: "#fff", border: "none", cursor: "pointer" }}>저장</button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function MemoModal({ target, value, setValue, error, onConfirm, onCancel }) {
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
          사용자 메모
        </div>
        <div style={{ fontSize: 13, color: "var(--ink-3)", marginBottom: 14, lineHeight: 1.6 }}>
          <strong style={{ color: "var(--ink-2)" }}>{target.name}</strong> ({target.id}) 님에 대한 관리자 메모입니다.
        </div>
        <textarea
          value={value} onChange={(e) => setValue(e.target.value)}
          maxLength={500} rows={4} autoFocus
          placeholder="예: 현장 담당자 · 외부 협력사 · 임시 계정 등"
          style={{
            width: "100%", padding: "10px 12px", fontSize: 13, color: "var(--ink)",
            background: "var(--bg)", border: "1px solid var(--line)", borderRadius: 10,
            outline: "none", resize: "vertical", fontFamily: "inherit",
            marginBottom: 6, boxSizing: "border-box",
          }}
        />
        <div style={{ fontSize: 11, color: "var(--ink-3)", textAlign: "right", marginBottom: 14 }}>{value.length}/500</div>
        {error && <div style={{ marginBottom: 12, fontSize: 12, color: "var(--err, #dc2626)", fontWeight: 600 }}>{error}</div>}
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
              background: "var(--brand)", color: "#fff", border: "none", cursor: "pointer",
            }}
          >저장</button>
        </div>
      </div>
    </div>
  );
}

function DeleteUserModal({ target, error, onConfirm, onCancel }) {
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
        width: 440, padding: 28, borderRadius: 16,
        background: "var(--bg-elev)", border: "1px solid var(--line)",
        boxShadow: "0 30px 80px -20px rgba(0,0,0,0.4)",
      }}>
        <div style={{ fontSize: 17, fontWeight: 800, marginBottom: 6, color: "var(--ink)" }}>
          계정 삭제
        </div>
        <div style={{ fontSize: 13, color: "var(--ink-3)", marginBottom: 16, lineHeight: 1.6 }}>
          <strong style={{ color: "var(--ink-2)" }}>{target.name}</strong> ({target.id}) 님의 계정을 삭제합니다.
          이 작업은 <strong style={{ color: "var(--err, #dc2626)" }}>되돌릴 수 없습니다.</strong>
        </div>
        {error && <div style={{ marginBottom: 12, fontSize: 12, color: "var(--err, #dc2626)", fontWeight: 600 }}>{error}</div>}
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
          >삭제</button>
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

// ─────────────────────────────────────────────────────
// 4) 챗봇 통계 — /api/admin/tool-stats + /api/chat/sessions 시각화
// ─────────────────────────────────────────────────────
function TokenUsageSection() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(null);

  useEffect(() => {
    let alive = true;
    setLoading(true); setErr(null);
    fetch("/api/admin/token-usage", { credentials: "same-origin" })
      .then((r) => r.json())
      .then((d) => { if (!alive) return; if (d.ok) setData(d); else setErr(d.error || "조회 실패"); })
      .catch((e) => { if (alive) setErr(e.message); })
      .finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
  }, []);

  const fmt = (n) => Number(n || 0).toLocaleString("en-US");

  if (loading) return <Empty text="토큰 사용량 불러오는 중..." />;
  if (err)     return <Empty text={`오류: ${err}`} />;
  if (!data)   return <Empty text="데이터 없음" />;

  const maxDay = Math.max(1, ...(data.daily || []).map((d) => d.tokens));
  const PROV_COLOR = { "OpenAI": "#10b981", "Ollama(로컬)": "var(--brand)", "기타": "var(--ink-3)" };

  return (
    <div style={{ display: "grid", gap: 16 }}>
      {/* KPI */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
        <KpiCard label="총 토큰" value={fmt(data.totals.tokens)} hint={`프롬프트 ${fmt(data.totals.prompt)} · 응답 ${fmt(data.totals.completion)}`} tone="brand" />
        <KpiCard label="AI 응답" value={fmt(data.totals.messages)} hint="누적 생성 횟수" tone="ok" />
        <KpiCard label="사용 모델" value={data.totals.models} hint="고유 모델 수" tone="muted" />
        <KpiCard label="OpenAI 예상 비용"
          value={`$${data.cost.usd < 1 ? data.cost.usd.toFixed(4) : data.cost.usd.toFixed(2)} / ${fmt(data.cost.krw)}원`}
          hint={`실시간 환율 ₩${fmt(data.cost.fx)}/$${data.cost.hasEstimated ? " · 일부 추정" : ""}${data.cost.hasUnpriced ? " · 일부 미설정" : ""}`}
          tone="warn" />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "300px 1fr", gap: 16 }}>
        {/* 제공자별 */}
        <Panel title="제공자별">
          {data.byProvider.length === 0 ? <Empty text="사용 기록 없음" /> : (
            <div style={{ display: "grid", gap: 12 }}>
              {data.byProvider.map((p) => (
                <StatRow key={p.provider} label={p.provider} color={PROV_COLOR[p.provider]}
                  value={`${fmt(p.tokens)} · ${fmt(p.messages)}건`} />
              ))}
            </div>
          )}
        </Panel>

        {/* 모델별 표 */}
        <Panel title="모델별 토큰 사용량">
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
            <thead>
              <tr style={{ color: "var(--ink-3)", fontSize: 11, fontWeight: 700, textTransform: "uppercase", borderBottom: "1px solid var(--line)" }}>
                {[["모델", "left"], ["제공자", "left"], ["응답", "right"], ["프롬프트", "right"], ["응답토큰", "right"], ["총 토큰", "right"], ["예상 비용", "right"], ["마지막", "right"]].map(([h, al]) => (
                  <th key={h} style={{ textAlign: al, padding: "8px 10px" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.byModel.length === 0 && (
                <tr><td colSpan={8} style={{ padding: 24, textAlign: "center", color: "var(--ink-3)" }}>사용 기록 없음</td></tr>
              )}
              {data.byModel.map((m) => (
                <tr key={m.model} style={{ borderBottom: "1px solid var(--line)" }}>
                  <td style={{ padding: "8px 10px", fontWeight: 700, color: "var(--ink)" }}>{m.model}</td>
                  <td style={{ padding: "8px 10px", color: PROV_COLOR[m.provider] || "var(--ink-3)", fontWeight: 600 }}>{m.provider}</td>
                  <td style={{ padding: "8px 10px", textAlign: "right" }}>{fmt(m.messages)}</td>
                  <td style={{ padding: "8px 10px", textAlign: "right", color: "var(--ink-3)" }}>{fmt(m.prompt)}</td>
                  <td style={{ padding: "8px 10px", textAlign: "right", color: "var(--ink-3)" }}>{fmt(m.completion)}</td>
                  <td style={{ padding: "8px 10px", textAlign: "right", fontWeight: 800, color: "var(--ink)" }}>{fmt(m.total)}</td>
                  <td style={{ padding: "8px 10px", textAlign: "right", fontWeight: 700, whiteSpace: "nowrap",
                    color: m.costUsd == null ? "#f59e0b" : (m.provider === "OpenAI" ? "var(--ink)" : "var(--ink-3)") }}>
                    {m.provider !== "OpenAI" ? "로컬·무료" : (m.costUsd == null ? "요율 미설정" : `${m.rate && m.rate.est ? "~" : ""}$${m.costUsd.toFixed(4)}`)}
                  </td>
                  <td style={{ padding: "8px 10px", textAlign: "right", color: "var(--ink-3)", fontSize: 11, whiteSpace: "nowrap" }}>{fmtDate(m.lastUsed)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{ marginTop: 10, fontSize: 11, color: "var(--ink-3)", lineHeight: 1.6 }}>
            예상 비용 = 입력·출력 토큰 × 요율. <strong>gpt-4o-mini · gpt-4o</strong>는 공개 요율, <strong>gpt-5 계열은 추정 요율</strong>(<code>~</code> 표시). 로컬(Ollama)은 무료.
            실제 단가는 <code>server.js</code>의 <code>PRICING</code>에서 조정하면 즉시 반영됩니다.
          </div>
        </Panel>
      </div>

      {/* 최근 14일 추이 */}
      <Panel title="최근 14일 일별 토큰">
        {(!data.daily || data.daily.length === 0) ? <Empty text="최근 14일 사용 없음" /> : (
          <div style={{ display: "flex", alignItems: "flex-end", gap: 6, height: 130, paddingTop: 10 }}>
            {data.daily.map((d) => (
              <div key={d.day} title={`${d.day} · ${fmt(d.tokens)} 토큰`}
                style={{ flex: 1, height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "flex-end", gap: 5 }}>
                <div style={{ width: "100%", maxWidth: 30, height: `${Math.max(2, (d.tokens / maxDay) * 100)}%`, background: "var(--brand)", opacity: 0.85, borderRadius: "4px 4px 0 0" }} />
                <div style={{ fontSize: 9, color: "var(--ink-3)", whiteSpace: "nowrap" }}>{String(d.day).slice(5)}</div>
              </div>
            ))}
          </div>
        )}
      </Panel>
    </div>
  );
}

// ── 로그인 로그 섹션 — audit_log 의 로그인 성공/실패 + 침입 시도 가시화 ──
function LoginLogSection() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(null);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    let alive = true;
    setLoading(true); setErr(null);
    fetch("/api/admin/login-log?limit=200", { credentials: "same-origin" })
      .then((r) => r.json())
      .then((d) => { if (!alive) return; if (d.ok) setData(d); else setErr(d.error || "조회 실패"); })
      .catch((e) => { if (alive) setErr(e.message); })
      .finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
  }, [tick]);

  if (loading) return <Empty text="로그인 로그 불러오는 중..." />;
  if (err)     return <Empty text={`오류: ${err}`} />;
  if (!data)   return <Empty text="데이터 없음" />;

  const events  = data.events  || [];
  const summary = data.summary || [];
  const totSuccess = summary.reduce((a, s) => a + Number(s.success || 0), 0);
  const totFail    = summary.reduce((a, s) => a + Number(s.fail || 0), 0);
  // 성공 0 · 실패만 있는 ID = 미등록 ID 추정(무차별 대입/probe 가능성)
  const probes = summary.filter((s) => Number(s.success || 0) === 0 && Number(s.fail || 0) > 0);

  const refreshBtn = (
    <button onClick={() => setTick((t) => t + 1)}
      style={{ fontSize: 12, padding: "4px 10px", border: "1px solid var(--line)", borderRadius: 6, background: "var(--bg)", color: "var(--ink-2)", cursor: "pointer" }}>
      새로고침
    </button>
  );

  return (
    <div style={{ display: "grid", gap: 16 }}>
      {/* KPI */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
        <KpiCard label="로그인 성공" value={totSuccess} hint="누적 성공 횟수" tone="ok" />
        <KpiCard label="로그인 실패" value={totFail} hint="비번 불일치·미등록 등" tone={totFail > 0 ? "warn" : "muted"} />
        <KpiCard label="시도된 ID" value={summary.length} hint="로그인 시도된 ID 종류" tone="brand" />
        <KpiCard label="의심 ID" value={probes.length} hint="성공 0 · 실패만 (미등록 추정)" tone={probes.length > 0 ? "warn" : "muted"} />
      </div>

      {/* 의심 시도 강조 */}
      {probes.length > 0 && (
        <Panel title="⚠ 의심 시도 (성공 0회 · 실패만)">
          <div style={{ display: "grid", gap: 8 }}>
            {probes.map((s) => (
              <StatRow key={s.account} label={s.account} color="#ef4444"
                value={`실패 ${s.fail}회 · IP ${s.ips}개 · ${fmtDate(s.lastAttempt)}`} />
            ))}
          </div>
          <div style={{ marginTop: 10, fontSize: 11, color: "var(--ink-3)", lineHeight: 1.6 }}>
            등록되지 않은 ID 로 로그인을 시도한 흔적입니다(무차별 대입 가능성). 인증(bcrypt + 시도 제한)이 막고 있으나, <strong>admin 비밀번호는 강하게</strong> 유지하세요.
          </div>
        </Panel>
      )}

      {/* 계정별 집계 */}
      <Panel title="계정별 로그인 집계">
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
          <thead>
            <tr style={{ color: "var(--ink-3)", fontSize: 11, fontWeight: 700, textTransform: "uppercase", borderBottom: "1px solid var(--line)" }}>
              {[["계정", "left"], ["성공", "right"], ["실패", "right"], ["IP 수", "right"], ["마지막 시도", "right"]].map(([h, al]) => (
                <th key={h} style={{ textAlign: al, padding: "8px 10px" }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {summary.length === 0 && (
              <tr><td colSpan={5} style={{ padding: 24, textAlign: "center", color: "var(--ink-3)" }}>기록 없음</td></tr>
            )}
            {summary.map((s) => {
              const isProbe = Number(s.success || 0) === 0 && Number(s.fail || 0) > 0;
              return (
                <tr key={s.account} style={{ borderBottom: "1px solid var(--line)" }}>
                  <td style={{ padding: "8px 10px", fontWeight: 700, color: isProbe ? "#ef4444" : "var(--ink)" }}>{s.account}{isProbe ? " ⚠" : ""}</td>
                  <td style={{ padding: "8px 10px", textAlign: "right", color: "#10b981", fontWeight: 700 }}>{s.success}</td>
                  <td style={{ padding: "8px 10px", textAlign: "right", color: Number(s.fail) > 0 ? "#ef4444" : "var(--ink-3)", fontWeight: 700 }}>{s.fail}</td>
                  <td style={{ padding: "8px 10px", textAlign: "right", color: "var(--ink-3)" }}>{s.ips}</td>
                  <td style={{ padding: "8px 10px", textAlign: "right", color: "var(--ink-3)", fontSize: 11, whiteSpace: "nowrap" }}>{fmtDate(s.lastAttempt)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </Panel>

      {/* 최근 이벤트 */}
      <Panel title={`최근 로그인 이벤트 (${events.length})`} right={refreshBtn}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
          <thead>
            <tr style={{ color: "var(--ink-3)", fontSize: 11, fontWeight: 700, textTransform: "uppercase", borderBottom: "1px solid var(--line)" }}>
              {[["시각", "left"], ["결과", "left"], ["계정", "left"], ["IP", "left"], ["사유", "left"]].map(([h, al]) => (
                <th key={h} style={{ textAlign: al, padding: "8px 10px" }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {events.length === 0 && (
              <tr><td colSpan={5} style={{ padding: 24, textAlign: "center", color: "var(--ink-3)" }}>기록 없음</td></tr>
            )}
            {events.map((e, i) => {
              const ok = e.action === "login";
              return (
                <tr key={i} style={{ borderBottom: "1px solid var(--line)" }}>
                  <td style={{ padding: "8px 10px", color: "var(--ink-3)", fontSize: 11, whiteSpace: "nowrap" }}>{fmtDate(e.ts)}</td>
                  <td style={{ padding: "8px 10px" }}>
                    <span style={{ fontSize: 11, fontWeight: 700, padding: "2px 8px", borderRadius: 99, background: ok ? "rgba(16,185,129,0.12)" : "rgba(239,68,68,0.12)", color: ok ? "#10b981" : "#ef4444" }}>
                      {ok ? "성공" : "실패"}
                    </span>
                  </td>
                  <td style={{ padding: "8px 10px", fontWeight: 600, color: "var(--ink)" }}>
                    {e.account}{e.name ? <span style={{ color: "var(--ink-3)", fontWeight: 400 }}> · {e.name}</span> : null}
                  </td>
                  <td style={{ padding: "8px 10px", color: "var(--ink-2)", fontFamily: "ui-monospace, SFMono-Regular, monospace", fontSize: 12 }}>{e.ip || "-"}</td>
                  <td style={{ padding: "8px 10px", color: "var(--ink-3)", fontSize: 12 }}>{e.reason || (ok ? "-" : "")}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </Panel>
    </div>
  );
}

function ChatbotStatsSection({ setToast }) {
  const [stats, setStats] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [days, setDays] = useState(7);
  const [loading, setLoading] = useState(false);

  const load = async (d = days) => {
    setLoading(true);
    try {
      const [a, b] = await Promise.all([
        fetch(`/api/admin/tool-stats?days=${d}`).then((r) => r.json()),
        fetch(`/api/chat/sessions?scope=all`).then((r) => r.json()),   // 관리자 통계 = 전역(계정 스코프 우회)
      ]);
      if (a.ok) setStats(a);
      if (b.ok) setSessions(b.sessions || []);
    } catch (e) {
      setToast && setToast({ kind: "error", text: `로드 실패: ${e.message}` });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(days); /* eslint-disable-line */ }, [days]);

  const tools = stats?.tools || [];
  const totals = stats?.totals || { calls: 0, ok: 0, cached: 0 };
  const maxCalls = Math.max(1, ...tools.map((t) => t.calls));
  const successRate = totals.calls > 0 ? Math.round((totals.ok / totals.calls) * 100) : 0;
  const cacheRate   = totals.calls > 0 ? Math.round((totals.cached / totals.calls) * 100) : 0;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* 상단 KPI + 기간 토글 */}
      <div style={{ display: "flex", gap: 12, alignItems: "stretch", justifyContent: "space-between" }}>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, flex: 1 }}>
          <KpiCard label="총 도구 호출"   value={totals.calls.toLocaleString()} hint={`최근 ${days}일`} tone="brand" />
          <KpiCard label="성공률"        value={`${successRate}%`}    hint={`${totals.ok}/${totals.calls}`} tone="ok" />
          <KpiCard label="캐시 hit"     value={`${cacheRate}%`}      hint={`${totals.cached}회`} tone="warn" />
          <KpiCard label="대화 세션"    value={sessions.length}     hint="저장된 챗봇 대화" tone="anomaly" />
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 6, alignItems: "flex-end" }}>
          <div style={{ fontSize: 10, color: "var(--ink-4)" }}>기간</div>
          <div style={{ display: "flex", gap: 4 }}>
            {[1, 7, 30].map((d) => (
              <button
                key={d}
                onClick={() => setDays(d)}
                style={{
                  padding: "4px 12px", fontSize: 11, fontWeight: 600,
                  borderRadius: 6,
                  border: "1px solid var(--line)",
                  background: days === d ? "var(--brand)" : "transparent",
                  color: days === d ? "#fff" : "var(--ink-3)",
                  cursor: "pointer",
                }}
              >{d}일</button>
            ))}
            <button
              onClick={() => load(days)}
              title="새로고침"
              style={{
                padding: "4px 10px", fontSize: 11,
                borderRadius: 6, border: "1px solid var(--line)",
                background: "transparent", color: "var(--ink-3)", cursor: "pointer",
              }}
            ><Icons.refresh size={11} /></button>
          </div>
        </div>
      </div>

      {/* 도구별 호출 막대 + 통계 */}
      <div style={{ background: "var(--bg-elev)", border: "1px solid var(--line)", borderRadius: 12, padding: 20 }}>
        <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 14, display: "flex", alignItems: "center", gap: 8 }}>
          <Icons.activity size={14} color="var(--brand)" />
          도구별 호출 통계 ({tools.length} 도구)
        </div>
        {loading && <div style={{ color: "var(--ink-4)", fontSize: 12 }}>로딩 중...</div>}
        {!loading && tools.length === 0 && (
          <div style={{ color: "var(--ink-4)", fontSize: 12 }}>해당 기간 도구 호출 없음</div>
        )}
        {!loading && tools.length > 0 && (
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {tools.map((t) => {
              const pct = (t.calls / maxCalls) * 100;
              const fail = t.calls - t.ok;
              return (
                <div key={t.tool} style={{ display: "flex", alignItems: "center", gap: 12 }}>
                  <div style={{
                    width: 200, fontSize: 11, fontWeight: 600,
                    fontFamily: "JetBrains Mono, ui-monospace, monospace",
                    color: "var(--ink-2)",
                    overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                  }}>{t.tool}</div>
                  <div style={{ flex: 1, position: "relative", height: 22, background: "var(--bg)", borderRadius: 4, border: "1px solid var(--line)", overflow: "hidden" }}>
                    <div style={{
                      width: `${pct}%`, height: "100%",
                      background: fail > 0
                        ? "linear-gradient(90deg, rgba(239,68,68,0.4) 0%, var(--brand) 100%)"
                        : "linear-gradient(90deg, rgba(79,70,229,0.5) 0%, var(--brand) 100%)",
                      transition: "width 220ms",
                    }} />
                    <div style={{
                      position: "absolute", top: 0, left: 8, height: "100%",
                      display: "flex", alignItems: "center",
                      fontSize: 10, fontWeight: 700, color: pct > 30 ? "#fff" : "var(--ink-2)",
                    }}>{t.calls.toLocaleString()}</div>
                  </div>
                  <div style={{ width: 120, display: "flex", gap: 8, fontSize: 10, color: "var(--ink-4)" }}>
                    <span title="평균 응답시간">⏱ {t.avgMs ?? "-"}ms</span>
                    {t.cached > 0 && <span title="캐시 hit">💾 {t.cached}</span>}
                    {fail > 0 && <span style={{ color: "#dc2626" }} title="실패">⚠ {fail}</span>}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* 최근 대화 세션 */}
      <div style={{ background: "var(--bg-elev)", border: "1px solid var(--line)", borderRadius: 12, padding: 20 }}>
        <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 14, display: "flex", alignItems: "center", gap: 8 }}>
          <Icons.sparkle size={14} color="var(--brand)" />
          최근 챗봇 대화 (상위 {Math.min(sessions.length, 15)})
        </div>
        {sessions.length === 0 && (
          <div style={{ color: "var(--ink-4)", fontSize: 12 }}>저장된 세션 없음</div>
        )}
        {sessions.length > 0 && (
          <table style={{ width: "100%", fontSize: 11, borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ borderBottom: "1px solid var(--line)", color: "var(--ink-4)", fontSize: 10, textAlign: "left" }}>
                <th style={{ padding: "8px 6px", width: 40 }}>#</th>
                <th style={{ padding: "8px 6px" }}>제목 (첫 질문)</th>
                <th style={{ padding: "8px 6px", width: 70, textAlign: "right" }}>메시지</th>
                <th style={{ padding: "8px 6px", width: 150 }}>최종 갱신</th>
              </tr>
            </thead>
            <tbody>
              {sessions.slice(0, 15).map((s) => {
                const dt = s.updated_at ? new Date(s.updated_at) : null;
                return (
                  <tr key={s.id} style={{ borderBottom: "1px solid var(--line)" }}>
                    <td style={{ padding: "8px 6px", fontFamily: "JetBrains Mono, monospace", color: "var(--ink-4)" }}>{s.id}</td>
                    <td style={{ padding: "8px 6px", color: "var(--ink)" }}>
                      {(s.title || "(제목 없음)").slice(0, 60)}
                    </td>
                    <td style={{ padding: "8px 6px", textAlign: "right", fontWeight: 600, color: "var(--ink-2)" }}>{s.messageCount || 0}</td>
                    <td style={{ padding: "8px 6px", color: "var(--ink-4)", fontSize: 10 }}>
                      {dt ? `${dt.toLocaleDateString("ko-KR")} ${dt.toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit" })}` : "-"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

// ── 공지사항(대시보드 배너) 편집 ──────────────────────────
function NoticeSection({ setToast }) {
  const [message, setMessage] = useState("");
  const [saving,  setSaving]  = useState(false);
  const [meta,    setMeta]    = useState(null);   // {updatedBy, updatedAt}

  const load = useCallback(async () => {
    const res = await getAnnouncement();
    if (res.ok && res.announcement) {
      const a = res.announcement;
      setMessage(a.message || "");
      setMeta({ updatedBy: a.updatedBy, updatedAt: a.updatedAt });
    }
  }, []);
  useEffect(() => { load(); }, [load]);

  const handleSave = async () => {
    const text = message.trim();
    setSaving(true);
    const res = await saveAnnouncement({ message: text, level: "critical" });
    setSaving(false);
    if (res.ok) {
      setToast && setToast({ kind: "ok", text: text ? "공지를 게시했습니다." : "공지를 내렸습니다." });
      load();
    } else {
      setToast && setToast({ kind: "error", text: res.error || "공지 저장에 실패했습니다." });
    }
  };

  const live = !!message.trim();
  const STRIPES = "repeating-linear-gradient(45deg, transparent, transparent 20px, rgba(255,255,255,0.06) 20px, rgba(255,255,255,0.06) 40px)";

  return (
    <div style={{ display: "grid", gap: 16, maxWidth: 720 }}>
      <Panel title="대시보드 공지 배너" right={meta?.updatedAt ? <span style={{ fontSize: 11, color: "var(--ink-3)" }}>최근: {meta.updatedBy || "관리자"} · {fmtDate(meta.updatedAt)}</span> : null}>
        <div style={{ fontSize: 12, color: "var(--ink-3)", marginBottom: 12, lineHeight: 1.5 }}>
          대시보드 상단에 표시할 공지입니다. 내용을 쓰고 <b>저장</b>하면 전 사용자에게 <b>빨강 긴급 배너</b>로 노출되고, <b>내용을 비우고 저장</b>하면 "운영 중 · 등록된 공지 없음" 중립 바로 내려갑니다.
        </div>
        <label style={{ display: "block", fontSize: 11, fontWeight: 600, color: "var(--ink-3)", marginBottom: 4 }}>
          공지 내용 <span style={{ color: "var(--ink-4)" }}>({message.length}/500)</span>
        </label>
        <textarea
          value={message} onChange={(e) => setMessage(e.target.value)} maxLength={500} rows={2}
          placeholder="예) 6/1 정기 점검 09:00~10:00 · 일시 접속 지연 가능"
          style={{ width: "100%", padding: "8px 10px", borderRadius: 8, border: "1px solid var(--line)", background: "var(--bg)", color: "var(--ink)", fontSize: 13, resize: "vertical", fontFamily: "inherit" }}
        />
      </Panel>

      <Panel title="미리보기">
        <div style={{
          position: "relative", height: 40, borderRadius: 8, padding: "0 16px", overflow: "hidden",
          display: "flex", alignItems: "center", justifyContent: "space-between",
          background: live ? "linear-gradient(90deg, #dc2626 0%, #991b1b 100%)" : "var(--bg-sunk)",
          border: live ? "none" : "1px solid var(--line)",
          boxShadow: live ? "0 4px 14px -4px rgba(220,38,38,0.55)" : "none",
        }}>
          {live ? (
            <>
              <div style={{ position: "absolute", inset: 0, background: STRIPES, pointerEvents: "none" }} />
              <div style={{ display: "flex", alignItems: "center", gap: 10, minWidth: 0, zIndex: 1 }}>
                <div style={{ width: 18, height: 18, flexShrink: 0, color: "#fff" }}><Icons.alert size={18} /></div>
                <span style={{ fontSize: 13, fontWeight: 700, color: "#fff", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{message}</span>
              </div>
              <span className="mono" style={{ flexShrink: 0, fontSize: 11, color: "rgba(255,255,255,0.9)", zIndex: 1 }}>관리자 · 지금</span>
            </>
          ) : (
            <>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <span style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--ok)" }} />
                <span style={{ fontSize: 13, fontWeight: 600, color: "var(--ink-2)" }}>운영 중</span>
                <span style={{ fontSize: 12, color: "var(--ink-3)" }}>· 등록된 공지 없음</span>
              </div>
              <span className="mono" style={{ fontSize: 11, color: "var(--ink-3)" }}>지금 기준</span>
            </>
          )}
        </div>
        <div style={{ fontSize: 11, color: "var(--ink-4)", marginTop: 8 }}>
          {live ? "저장 시 대시보드 상단에 위와 같이(빨강 긴급) 노출됩니다." : "내용이 비어 있어 중립 바로 표시됩니다. (게시된 공지 없음)"}
        </div>
      </Panel>

      <div style={{ display: "flex", justifyContent: "flex-end" }}>
        <button type="button" disabled={saving} onClick={handleSave}
          style={{ height: 36, padding: "0 24px", borderRadius: 8, fontSize: 13, fontWeight: 700, background: "var(--brand)", color: "#fff", border: "1px solid var(--brand)", cursor: saving ? "default" : "pointer", opacity: saving ? 0.6 : 1 }}>
          {saving ? "저장 중…" : (live ? "저장 (게시)" : "저장 (공지 내림)")}
        </button>
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
