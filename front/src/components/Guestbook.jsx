import { useState, useEffect, useCallback, useRef } from "react";

// 시원팀 공개문의(단톡방) — 공개 실시간. 질문하면 AI 페르소나(박지훈·이재헌·이두현)가 답.
// 메시지는 챗봇 ChatMessage 재사용. 봇 메시지(botKey)는 페르소나 아바타 + 이름으로 렌더, 'AI 답변 중…' 인디케이터 표시.

const ROLE_AVATAR = { superadmin: "/avatars/developer.png", admin: "/avatars/admin.png", operator: "/avatars/operator.png", guest: "/avatars/guest.png" };
const ROLE_LABEL = { superadmin: "총괄 관리자", admin: "시원팀", operator: "관제사", viewer: "뷰어", guest: "게스트" };
const LOUNGE_KEYS = ["park", "lee_jaeheon", "lee_duhyeon"];
const MAX_BODY = 500, MAX_NAME = 40;

function fmtTime(ts) {
  try { return new Date(ts).toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit" }); }
  catch { return ""; }
}
// 날짜 구분선용 dateKey (YYYY-MM-DD, 로컬). DayDivider 가 new Date(key+"T00:00:00") 로 파싱.
function gbDateKey(iso) {
  const d = new Date(iso);
  return isNaN(d.getTime()) ? "" : `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}
function isQuestionClient(t) {
  const s = String(t || "");
  return /[?？]/.test(s) || /@\S/.test(s) || /(뭐|무엇|어떻게|어떤|왜|언제|어디|누구|누가|얼마|몇|있나요|있어요|인가요|까요|나요|되나요|될까|할까|을까|설명|알려|궁금|차이|이유|무슨)/.test(s);
}

export function useGuestbook(active) {
  const [messages, setMessages] = useState([]);
  const [online, setOnline] = useState(0);
  const [connected, setConnected] = useState(false);
  const [personas, setPersonas] = useState([]);
  const [pendingBot, setPendingBot] = useState(false);
  const [typingBot, setTypingBot] = useState(null);   // 답변 작성 중인 담당 페르소나 {botKey,name,avatar} — 타이핑 인디케이터용
  const [humanTyping, setHumanTyping] = useState([]);          // 입력 중인 다른 사용자 [{connId,name}]
  const typingMapRef = useRef({});                             // connId -> {name, expiresAt}
  const myConnId = useRef(Math.random().toString(36).slice(2) + Date.now().toString(36));   // 내 WS 식별자(타이핑 신호 자기 제외용)
  const wsRef = useRef(null);                                  // 현재 살아있는 WS (타이핑 신호 송신용)
  const typingThrottleRef = useRef(0);
  const typingStopRef = useRef(null);
  const syncTyping = () => setHumanTyping(Object.entries(typingMapRef.current).map(([connId, v]) => ({ connId, name: v.name })));
  const [guestName, setGuestName] = useState(() => { try { return localStorage.getItem("siwon.gb.name") || ""; } catch { return ""; } });
  const guestNameRef = useRef(guestName);
  guestNameRef.current = guestName;
  const pendingTimer = useRef(null);
  const sentIdsRef = useRef(new Set());   // 내가 이 세션에서 보낸 메시지 id → 무조건 내 것(오른쪽)
  // 공개 프로필 캐시(uid → {name,role,title,bio,github,avatar}) — 라운지 아바타·프로필 카드용. 한 번 받으면 재사용.
  const [profiles, setProfiles] = useState({});
  const profilesRef = useRef(profiles); profilesRef.current = profiles;
  const fetchingRef = useRef(new Set());
  const fetchProfile = useCallback(async (uid) => {
    if (uid == null) return;
    if (profilesRef.current[uid] || fetchingRef.current.has(uid)) return;
    fetchingRef.current.add(uid);
    try {
      const r = await fetch(`/api/profile/${uid}`, { credentials: "include" });
      const j = await r.json();
      if (j.ok && j.profile) setProfiles((p) => ({ ...p, [uid]: j.profile }));
    } catch {} finally { fetchingRef.current.delete(uid); }
  }, []);

  const dedupAppend = (prev, msg) => (prev.some((x) => x.id === msg.id) ? prev : [...prev, msg]);
  const load = useCallback(async () => {
    try { const r = await fetch("/api/guestbook?limit=80", { credentials: "include" }); const j = await r.json(); if (j.ok) setMessages(j.messages || []); } catch {}
  }, []);

  useEffect(() => {
    if (!active) return;
    fetch("/api/personas", { credentials: "include" }).then((r) => r.json()).then((j) => { if (j.ok) setPersonas(j.personas || []); }).catch(() => {});
  }, [active]);

  useEffect(() => {
    if (!active) return;
    let ws, retry = 0, alive = true, timer = null;
    const connect = () => {
      if (!alive) return;
      const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
      try { ws = new WebSocket(`${proto}//${window.location.host}/ws/guestbook`); }
      catch { timer = setTimeout(connect, 3000); return; }
      ws.onopen = () => { retry = 0; setConnected(true); wsRef.current = ws; load(); };
      ws.onmessage = (e) => {
        let m; try { m = JSON.parse(e.data); } catch { return; }
        if (m.type === "gb:msg" && m.message) {
          setMessages((p) => dedupAppend(p, m.message));
          // 실제 페르소나 답변(시원=핸드오프/인사 제외)이 오면 타이핑·대기 해제
          if (m.message.botKey && m.message.botKey !== "siwon") { setPendingBot(false); setTypingBot(null); if (pendingTimer.current) clearTimeout(pendingTimer.current); }
        } else if (m.type === "gb:typing") {
          // 담당 페르소나가 답변 작성 시작 → 타이핑 인디케이터를 그 페르소나(아바타·이름)로 표시
          setTypingBot({ botKey: m.botKey, name: m.name, avatar: m.avatar });
          setPendingBot(true);
          if (pendingTimer.current) clearTimeout(pendingTimer.current);
          pendingTimer.current = setTimeout(() => { setPendingBot(false); setTypingBot(null); }, 70000);
        } else if (m.type === "gb:usertyping") {
          // 다른 사용자 입력 중 — 내 connId 는 무시. 4.5s 후 만료(stop 누락/끊김 대비).
          if (m.connId && m.connId !== myConnId.current) {
            if (m.typing) typingMapRef.current[m.connId] = { name: m.name || "사용자", expiresAt: Date.now() + 4500 };
            else delete typingMapRef.current[m.connId];
            syncTyping();
          }
        } else if (m.type === "gb:del") setMessages((p) => p.filter((x) => x.id !== m.id));
        else if (m.type === "gb:presence" || m.type === "gb:hello") setOnline(m.online || 0);
      };
      ws.onclose = () => { setConnected(false); wsRef.current = null; typingMapRef.current = {}; syncTyping(); if (alive) { const d = Math.min(1000 * 2 ** retry++, 15000); timer = setTimeout(connect, d); } };
      ws.onerror = () => { try { ws.close(); } catch {} };
    };
    connect();
    return () => { alive = false; if (timer) clearTimeout(timer); if (pendingTimer.current) clearTimeout(pendingTimer.current); try { ws && ws.close(); } catch {} };
  }, [active, load]);

  useEffect(() => {
    const el = null; // 스크롤은 리스트에서 처리
  }, [messages]);

  // 입력 중 표시 만료 정리 (stop 누락/끊김 대비)
  useEffect(() => {
    if (!active) return;
    const id = setInterval(() => {
      const now = Date.now(); let changed = false;
      for (const k in typingMapRef.current) if (typingMapRef.current[k].expiresAt <= now) { delete typingMapRef.current[k]; changed = true; }
      if (changed) setHumanTyping(Object.entries(typingMapRef.current).map(([connId, v]) => ({ connId, name: v.name })));
    }, 1200);
    return () => clearInterval(id);
  }, [active]);

  // 내가 입력 중 신호 — 입력칸 onChange 에서 호출(내부 스로틀 1.8s) + 3s 무입력 시 자동 stop
  const sendTyping = useCallback((name) => {
    const ws = wsRef.current; if (!ws || ws.readyState !== 1) return;
    const now = Date.now();
    if (now - typingThrottleRef.current > 1800) {
      typingThrottleRef.current = now;
      try { ws.send(JSON.stringify({ type: "typing", connId: myConnId.current, name: String(name || "").slice(0, 40), typing: true })); } catch {}
    }
    if (typingStopRef.current) clearTimeout(typingStopRef.current);
    typingStopRef.current = setTimeout(() => {
      const w = wsRef.current; if (w && w.readyState === 1) { try { w.send(JSON.stringify({ type: "typing", connId: myConnId.current, typing: false })); } catch {} }
    }, 3000);
  }, []);
  const stopTyping = useCallback(() => {
    if (typingStopRef.current) clearTimeout(typingStopRef.current);
    typingThrottleRef.current = 0;
    const ws = wsRef.current; if (ws && ws.readyState === 1) { try { ws.send(JSON.stringify({ type: "typing", connId: myConnId.current, typing: false })); } catch {} }
  }, []);

  const send = useCallback(async (body, loggedIn, image = null, model = null) => {
    const text = String(body || "").trim();
    if (!text && !image) return false;
    const gn = guestNameRef.current.trim();
    if (!loggedIn) { try { localStorage.setItem("siwon.gb.name", gn); } catch {} }
    try {
      const payload = loggedIn ? { body: text } : { body: text, name: gn || "게스트" };
      if (image) payload.image = image;
      if (model) payload.model = model;   // 봇 답변 모델(라운지 모델 선택기) — 백엔드 허용목록 검증
      const r = await fetch("/api/guestbook", { method: "POST", headers: { "Content-Type": "application/json" }, credentials: "include", body: JSON.stringify(payload) });
      const j = await r.json();
      if (j.ok && j.message) {
        sentIdsRef.current.add(j.message.id);
        setMessages((p) => dedupAppend(p, j.message));
        if (isQuestionClient(text)) {   // 질문이면 'AI 답변 중…' 표시 (봇 메시지 도착 or 70s 타임아웃 시 해제)
          setPendingBot(true);
          if (pendingTimer.current) clearTimeout(pendingTimer.current);
          pendingTimer.current = setTimeout(() => { setPendingBot(false); setTypingBot(null); }, 70000);
        }
        return true;
      }
      window.alert(j.error || "전송에 실패했습니다."); return false;
    } catch { window.alert("전송에 실패했습니다."); return false; }
  }, []);

  const del = useCallback(async (id) => {
    setMessages((p) => p.filter((x) => x.id !== id));
    try { await fetch(`/api/guestbook/${id}`, { method: "DELETE", credentials: "include" }); } catch {}
  }, []);

  return { messages, online, connected, personas, pendingBot, typingBot, humanTyping, sendTyping, stopTyping, guestName, setGuestName, send, del, sentIds: sentIdsRef.current, profiles, fetchProfile };
}

function Avatar({ src, name, size = 34 }) {
  if (src) return <div style={{ width: size, height: size, borderRadius: "50%", flexShrink: 0, marginTop: 1, overflow: "hidden", background: "linear-gradient(135deg, #4f46e5, #8b83ff)" }}><img src={src} alt="" style={{ width: "100%", height: "100%", objectFit: "cover" }} /></div>;
  let h = 0; const s = String(name || "?"); for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) % 360;
  return <div style={{ width: size, height: size, borderRadius: "50%", flexShrink: 0, marginTop: 1, background: `hsl(${h},52%,54%)`, color: "#fff", display: "grid", placeItems: "center", fontSize: size * 0.42, fontWeight: 700 }}>{s.trim().charAt(0).toUpperCase() || "?"}</div>;
}

export function GuestbookList({ gb, isGuest = false, isSuper = false, me = null, ChatMessage, DayDivider = null, bottomSpace = 120 }) {
  const listRef = useRef(null);
  const stickRef = useRef(true);
  // 핀(맨아래)일 때만 자동 스크롤 — 새 메시지/봇응답/여백변화 시. rAF 로 레이아웃 정착 후 (관제 도우미와 동일 방식)
  useEffect(() => {
    if (!stickRef.current) return;
    const el = listRef.current; if (!el) return;
    requestAnimationFrame(() => { el.scrollTop = el.scrollHeight; });
  }, [gb.messages, gb.pendingBot, bottomSpace]);
  const onScroll = (e) => { const c = e.currentTarget; stickRef.current = c.scrollHeight - c.scrollTop - c.clientHeight < 48; };

  const [cardUid, setCardUid] = useState(null);   // 열려있는 프로필 카드의 uid (로그인 작성자)
  const fetchProfile = gb.fetchProfile;
  // 보이는 메시지의 로그인 작성자 공개 프로필을 미리 받아 캐시(아바타·카드용)
  useEffect(() => {
    const seen = new Set();
    (gb.messages || []).forEach((m) => { if (!m.botKey && m.userId != null && !seen.has(m.userId)) { seen.add(m.userId); fetchProfile && fetchProfile(m.userId); } });
  }, [gb.messages, fetchProfile]);

  const personaMap = {}; (gb.personas || []).forEach((p) => { personaMap[p.key] = p; });
  const team = (gb.personas || []).filter((p) => LOUNGE_KEYS.includes(p.key));
  const myGuest = (gb.guestName || "").trim() || "게스트";
  const isMine = (m) => !m.botKey && (m.mine || gb.sentIds.has(m.id) || (me ? (m.userId != null && m.name === me.name) : (m.userId == null && m.name === myGuest)));

  return (
    <div ref={listRef} className="scroll" onScroll={onScroll}
      style={{ flex: 1, overflowY: "auto", padding: "3px 12px 0", paddingBottom: bottomSpace, minHeight: 0, background: "var(--bg-sunk)" }}>
      {/* 헤더(시원팀 공개문의 · 접속수) 제거 — 탭 이름과 중복, 사용자 요청 */}
      {team.length > 0 && (
        <div style={{ position: "sticky", top: 0, zIndex: 3, display: "flex", alignItems: "center", gap: 8, padding: "8px 10px", marginBottom: 12, borderRadius: 12, background: "var(--brand-wash-solid)", border: "1px solid var(--line)", boxShadow: "0 6px 16px -8px rgba(15,23,42,0.22)" }}>
          <div style={{ display: "flex" }}>
            {team.map((p, i) => <div key={p.key} style={{ marginLeft: i ? -8 : 0, border: "2px solid var(--bg-sunk)", borderRadius: "50%" }}><Avatar src={p.avatar} name={p.name} size={26} /></div>)}
          </div>
          <span style={{ fontSize: 11.5, color: "var(--ink-2)", lineHeight: 1.4 }}>
            <b>질문을 남기면 시원팀 AI가 답해요</b>
          </span>
        </div>
      )}
      {isGuest && (
        <input value={gb.guestName} onChange={(e) => gb.setGuestName(e.target.value.slice(0, MAX_NAME))} placeholder="닉네임 (게스트) — 아래 채팅칸에 작성" maxLength={MAX_NAME}
          style={{ width: "100%", marginBottom: 12, padding: "6px 10px", borderRadius: 8, border: "1px solid var(--line)", background: "var(--bg-elev)", color: "var(--ink)", fontSize: 12, outline: "none" }} />
      )}

      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {gb.messages.length === 0 && (
          <div style={{ margin: "16px auto", textAlign: "center", color: "var(--ink-4)", fontSize: 12.5, lineHeight: 1.8 }}>
            아직 대화가 없어요.<br />프로젝트에 대해 무엇이든 물어보세요 — AI가 답해드려요 💬
          </div>
        )}
        {gb.messages.map((m, i) => {
          const bot = !!m.botKey;
          const p = bot ? (personaMap[m.botKey] || {}) : null;
          const mine = isMine(m);
          const prof = (!bot && m.userId != null) ? gb.profiles[m.userId] : null;   // 로그인 작성자 공개 프로필(캐시)
          const avatar = bot ? (m.avatar || p.avatar || "/avatars/chatbot.png") : (prof?.avatar || (m.role ? ROLE_AVATAR[m.role] : null));
          const label = bot ? m.name : (m.role ? `${m.name} · ${ROLE_LABEL[m.role] || ""}` : m.name);
          const canCard = !bot && m.userId != null;   // 로그인 사용자만 프로필 카드 열람
          const _dk = gbDateKey(m.createdAt);
          const _showDiv = _dk && (i === 0 || _dk !== gbDateKey(gb.messages[i - 1]?.createdAt));   // 날짜 구분선(날짜 바뀔 때)
          return (
            <div key={m.id}>
              {DayDivider && _showDiv && <DayDivider dateKey={_dk} />}
              {ChatMessage
                ? <ChatMessage message={{ role: mine ? "user" : "ai", text: m.body, image: m.image }} botAvatar={avatar || "/avatars/guest.png"} botLabel={label} hideTime
                    onAuthorClick={canCard ? () => { setCardUid(m.userId); fetchProfile && fetchProfile(m.userId); } : undefined} />
                : <div style={{ fontSize: 12.5 }}><b>{label}</b>: {m.body}</div>}
              <div style={{ display: "flex", justifyContent: mine ? "flex-end" : "flex-start", alignItems: "center", gap: 6, padding: "2px 2px 0", marginLeft: mine ? 0 : 42 }}>
                {bot && <span style={{ fontSize: 9, fontWeight: 800, padding: "1px 5px", borderRadius: 999, color: "#fff", background: "var(--brand)" }}>AI</span>}
                <span style={{ fontSize: 10, color: "var(--ink-4)" }}>{fmtTime(m.createdAt)}</span>
                {isSuper && <button onClick={() => gb.del(m.id)} title="삭제(모더레이션)" style={{ border: "none", background: "transparent", color: "var(--ink-4)", cursor: "pointer", fontSize: 10.5, padding: 0 }}>삭제</button>}
              </div>
            </div>
          );
        })}
        {/* AI 답변 중 인디케이터 */}
        {gb.pendingBot && (() => {
          // 답변 작성 중 표시 — 담당 페르소나(gb.typingBot)가 정해지면 그 아바타·이름으로, 아니면 일반 AI
          const t = gb.typingBot;
          const av = (t && t.avatar) || "/avatars/chatbot.png";
          const nm = (t && t.name) || "AI";
          return (
            <div style={{ display: "flex", gap: 9, alignItems: "flex-start" }}>
              <Avatar src={av} name={nm} size={34} />
              <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                <div style={{ fontSize: 11, fontWeight: 600, color: "var(--ink-3)", paddingLeft: 2 }}>{nm}</div>
                <div style={{ display: "inline-flex", alignItems: "center", gap: 4, padding: "10px 14px", width: "fit-content", borderRadius: "18px 18px 18px 4px", background: "var(--chat-ai-bg)" }}>
                  <span style={{ width: 5, height: 5, borderRadius: "50%", background: "var(--ink-3)", animation: "pulse-dot 1.2s 0s infinite" }} />
                  <span style={{ width: 5, height: 5, borderRadius: "50%", background: "var(--ink-3)", animation: "pulse-dot 1.2s 0.2s infinite" }} />
                  <span style={{ width: 5, height: 5, borderRadius: "50%", background: "var(--ink-3)", animation: "pulse-dot 1.2s 0.4s infinite" }} />
                </div>
              </div>
            </div>
          );
        })()}
        {/* 다른 사용자 입력 중 — 단톡방 느낌 */}
        {gb.humanTyping && gb.humanTyping.length > 0 && (() => {
          const ht = gb.humanTyping;
          const label = ht.length === 1 ? `${ht[0].name}님이 입력 중`
            : ht.length === 2 ? `${ht[0].name}, ${ht[1].name}님이 입력 중`
            : `${ht.length}명이 입력 중`;
          return (
            <div style={{ display: "inline-flex", alignItems: "center", gap: 7, marginLeft: 2, padding: "5px 12px", width: "fit-content", borderRadius: 999, background: "var(--bg-sunk)", fontSize: 11.5, color: "var(--ink-3)" }}>
              <span style={{ display: "inline-flex", gap: 3 }}>
                <span style={{ width: 4, height: 4, borderRadius: "50%", background: "var(--ink-4)", animation: "pulse-dot 1.2s 0s infinite" }} />
                <span style={{ width: 4, height: 4, borderRadius: "50%", background: "var(--ink-4)", animation: "pulse-dot 1.2s 0.2s infinite" }} />
                <span style={{ width: 4, height: 4, borderRadius: "50%", background: "var(--ink-4)", animation: "pulse-dot 1.2s 0.4s infinite" }} />
              </span>
              {label}
            </div>
          );
        })()}
      </div>
      {cardUid != null && (
        <ProfileCard profile={gb.profiles[cardUid]} onClose={() => setCardUid(null)} />
      )}
    </div>
  );
}

// 프로필 카드 — 라운지에서 작성자 이름/아바타 클릭 시. 공개 필드만(이름·역할·직무·소개·링크·아바타).
function ProfileCard({ profile, onClose }) {
  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);
  const roleLabel = profile ? (ROLE_LABEL[profile.role] || profile.role || "") : "";
  return (
    <div onClick={onClose} style={{ position: "fixed", inset: 0, zIndex: 200, background: "rgba(15,23,42,0.45)", display: "grid", placeItems: "center", padding: 20 }}>
      <div onClick={(e) => e.stopPropagation()} style={{ width: "min(340px, 92vw)", background: "var(--bg-elev)", border: "1px solid var(--line)", borderRadius: 16, boxShadow: "0 20px 50px -12px rgba(15,23,42,0.4)", overflow: "hidden" }}>
        {!profile ? (
          <div style={{ padding: 28, textAlign: "center", color: "var(--ink-3)", fontSize: 12.5 }}>불러오는 중…</div>
        ) : (
          <>
            <div style={{ display: "flex", alignItems: "center", gap: 14, padding: "18px 20px", borderBottom: "1px solid var(--line)", background: "var(--bg-sunk)" }}>
              <Avatar src={profile.avatar || (profile.role ? ROLE_AVATAR[profile.role] : null)} name={profile.name} size={56} />
              <div style={{ minWidth: 0 }}>
                <div style={{ fontSize: 15, fontWeight: 800, color: "var(--ink)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{profile.name}</div>
                <div style={{ marginTop: 3, display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
                  {roleLabel && <span style={{ fontSize: 10, fontWeight: 700, padding: "2px 8px", borderRadius: 999, background: "var(--brand)", color: "#fff" }}>{roleLabel}</span>}
                  {profile.title && <span style={{ fontSize: 11.5, color: "var(--ink-3)" }}>{profile.title}</span>}
                </div>
              </div>
            </div>
            <div style={{ padding: "16px 20px", display: "flex", flexDirection: "column", gap: 12 }}>
              {profile.bio
                ? <div style={{ fontSize: 12.5, lineHeight: 1.6, color: "var(--ink-2)", whiteSpace: "pre-wrap", wordBreak: "break-word" }}>{profile.bio}</div>
                : <div style={{ fontSize: 12, color: "var(--ink-4)" }}>소개가 아직 없어요.</div>}
              {profile.github && (
                <a href={profile.github} target="_blank" rel="noopener noreferrer"
                   style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12, fontWeight: 700, color: "var(--brand)", textDecoration: "none", wordBreak: "break-all" }}>
                  🔗 {profile.github.replace(/^https?:\/\//, "")}
                </a>
              )}
            </div>
            <div style={{ padding: "0 20px 16px", textAlign: "right" }}>
              <button onClick={onClose} style={{ padding: "7px 16px", borderRadius: 8, border: "1px solid var(--line)", background: "var(--bg-elev)", color: "var(--ink-2)", fontSize: 12.5, fontWeight: 700, cursor: "pointer" }}>닫기</button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
