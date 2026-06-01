import { useState, useEffect, useCallback, useRef } from "react";

// 단톡방(공개 실시간 채팅) — WebSocket 실시간 + MySQL 저장. 화면엔 "단톡방"으로 표기.
// 메시지는 챗봇의 ChatMessage 컴포넌트를 그대로 재사용 → 디자인 100% 일치.
//  · 내 메시지 = 오른쪽 보라(role:"user")   · 남의 메시지 = 왼쪽 회색 + 아바타·이름(role:"ai")
//  · 내 식별: 내가 보낸 메시지 id 추적(sentIds) + 이름 매칭 보조. XSS 안전(순수 텍스트).

const ROLE_AVATAR = {
  superadmin: "/avatars/developer.png",
  admin:      "/avatars/admin.png",
  operator:   "/avatars/operator.png",
  guest:      "/avatars/guest.png",
};
const ROLE_LABEL = { superadmin: "총관리자", admin: "관리자", operator: "운영자", guest: "회원" };

function fmtTime(ts) {
  try { return new Date(ts).toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit" }); }
  catch { return ""; }
}

// ── 상태 + WebSocket 훅 — active(단톡방 열림)일 때만 연결/로드. send()는 공유 컴포저가 호출. ──
export function useGuestbook(active) {
  const [messages, setMessages] = useState([]);
  const [online, setOnline] = useState(0);
  const [connected, setConnected] = useState(false);
  const [guestName, setGuestName] = useState(() => { try { return localStorage.getItem("siwon.gb.name") || ""; } catch { return ""; } });
  const guestNameRef = useRef(guestName);
  guestNameRef.current = guestName;
  const sentIdsRef = useRef(new Set());          // 내가 이 세션에서 보낸 메시지 id → 무조건 '내 것'(오른쪽)

  const dedupAppend = (prev, msg) => (prev.some((x) => x.id === msg.id) ? prev : [...prev, msg]);
  const load = useCallback(async () => {
    try { const r = await fetch("/api/guestbook?limit=80", { credentials: "include" }); const j = await r.json(); if (j.ok) setMessages(j.messages || []); } catch {}
  }, []);

  useEffect(() => {
    if (!active) return;          // 단톡방 열렸을 때만 연결
    let ws, retry = 0, alive = true, timer = null;
    const connect = () => {
      if (!alive) return;
      const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
      try { ws = new WebSocket(`${proto}//${window.location.host}/ws/guestbook`); }
      catch { timer = setTimeout(connect, 3000); return; }
      ws.onopen = () => { retry = 0; setConnected(true); load(); };
      ws.onmessage = (e) => {
        let m; try { m = JSON.parse(e.data); } catch { return; }
        if (m.type === "gb:msg" && m.message) setMessages((p) => dedupAppend(p, m.message));
        else if (m.type === "gb:del") setMessages((p) => p.filter((x) => x.id !== m.id));
        else if (m.type === "gb:presence" || m.type === "gb:hello") setOnline(m.online || 0);
      };
      ws.onclose = () => { setConnected(false); if (alive) { const d = Math.min(1000 * 2 ** retry++, 15000); timer = setTimeout(connect, d); } };
      ws.onerror = () => { try { ws.close(); } catch {} };
    };
    connect();
    return () => { alive = false; if (timer) clearTimeout(timer); try { ws && ws.close(); } catch {} };
  }, [active, load]);

  const send = useCallback(async (body, loggedIn) => {
    const text = String(body || "").trim();
    if (!text) return false;
    const gn = guestNameRef.current.trim();
    if (!loggedIn) { try { localStorage.setItem("siwon.gb.name", gn); } catch {} }
    try {
      const payload = loggedIn ? { body: text } : { body: text, name: gn || "게스트" };
      const r = await fetch("/api/guestbook", { method: "POST", headers: { "Content-Type": "application/json" }, credentials: "include", body: JSON.stringify(payload) });
      const j = await r.json();
      if (j.ok && j.message) { sentIdsRef.current.add(j.message.id); setMessages((p) => dedupAppend(p, j.message)); return true; }
      window.alert(j.error || "전송에 실패했습니다."); return false;
    } catch { window.alert("전송에 실패했습니다."); return false; }
  }, []);

  const del = useCallback(async (id) => {
    setMessages((p) => p.filter((x) => x.id !== id));   // 낙관적 제거 (WS gb:del 도 동일)
    try { await fetch(`/api/guestbook/${id}`, { method: "DELETE", credentials: "include" }); } catch {}
  }, []);

  return { messages, online, connected, guestName, setGuestName, send, del, sentIds: sentIdsRef.current };
}

// ── 메시지 리스트 — 챗봇 ChatMessage 재사용으로 디자인 동일. 입력칸은 공유 컴포저(밖). ──
export function GuestbookList({ gb, isGuest = false, isAdmin = false, me = null, ChatMessage }) {
  const listRef = useRef(null);
  const stickRef = useRef(true);
  useEffect(() => { const el = listRef.current; if (el && stickRef.current) el.scrollTop = el.scrollHeight; }, [gb.messages]);
  const onScroll = (e) => { const c = e.currentTarget; stickRef.current = c.scrollHeight - c.scrollTop - c.clientHeight < 60; };

  const myGuest = (gb.guestName || "").trim() || "게스트";
  const isMine = (m) =>
    gb.sentIds.has(m.id) ||
    (me ? (m.userId != null && m.name === me.name) : (m.userId == null && m.name === myGuest));

  return (
    <div ref={listRef} className="scroll" onScroll={onScroll}
      style={{ flex: 1, overflowY: "auto", padding: "10px 12px clamp(150px, 30vh, 380px)", minHeight: 0, background: "var(--bg-sunk)" }}>
      {/* 헤더 — 접속자 / (게스트) 닉네임 */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12, flexWrap: "wrap" }}>
        <span style={{ fontSize: 12, fontWeight: 800, color: "var(--ink)" }}>단톡방</span>
        <span style={{ fontSize: 11, color: "var(--ink-3)" }}>공개 채팅 · 누구나 자유롭게</span>
        <span style={{ marginLeft: "auto", display: "inline-flex", alignItems: "center", gap: 5, fontSize: 11, color: "var(--ink-3)" }}>
          <span style={{ width: 7, height: 7, borderRadius: "50%", background: gb.connected ? "var(--ok)" : "var(--ink-4)" }} />
          {gb.connected ? `접속 ${gb.online}` : "연결 중…"}
        </span>
      </div>
      {isGuest && (
        <input value={gb.guestName} onChange={(e) => gb.setGuestName(e.target.value.slice(0, 40))}
          placeholder="닉네임 (게스트) — 아래 채팅칸에 작성" maxLength={40}
          style={{ width: "100%", marginBottom: 12, padding: "6px 10px", borderRadius: 8, border: "1px solid var(--line)", background: "var(--bg-elev)", color: "var(--ink)", fontSize: 12, outline: "none" }} />
      )}

      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {gb.messages.length === 0 && (
          <div style={{ margin: "24px auto", textAlign: "center", color: "var(--ink-4)", fontSize: 12.5, lineHeight: 1.8 }}>
            아직 대화가 없어요.<br />첫 메시지를 보내보세요 💬
          </div>
        )}
        {gb.messages.map((m) => {
          const mine = isMine(m);
          const label = m.role ? `${m.name} · ${ROLE_LABEL[m.role] || ""}` : m.name;
          return (
            <div key={m.id}>
              {ChatMessage
                ? <ChatMessage message={{ role: mine ? "user" : "ai", text: m.body }} botAvatar={ROLE_AVATAR[m.role] || "/avatars/guest.png"} botLabel={label} hideTime />
                : <div style={{ fontSize: 12.5 }}>{m.body}</div>}
              <div style={{ display: "flex", justifyContent: mine ? "flex-end" : "flex-start", alignItems: "center", gap: 6, padding: "2px 2px 0", marginLeft: mine ? 0 : 42 }}>
                <span style={{ fontSize: 10, color: "var(--ink-4)" }}>{fmtTime(m.createdAt)}</span>
                {isAdmin && (
                  <button onClick={() => gb.del(m.id)} title="삭제(모더레이션)"
                    style={{ border: "none", background: "transparent", color: "var(--ink-4)", cursor: "pointer", fontSize: 10.5, padding: 0, lineHeight: 1 }}>삭제</button>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
