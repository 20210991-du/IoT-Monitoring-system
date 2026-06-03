/**
 * authMock.js — 백엔드 인증 클라이언트 (이전 localStorage 목업 대체).
 *
 *   서버: /api/auth/*  (MySQL `users` + bcrypt + JWT httpOnly 쿠키)
 *   세션 토큰은 httpOnly 쿠키(JS 접근 불가) → XSS 토큰 탈취 완화.
 *   여기선 "표시용 user 캐시"만 localStorage 에 둔다 (UI 게이팅용, 권한의 근거 아님).
 *   실제 권한 검증은 매 요청마다 서버가 수행한다.
 *
 *   동기 API:  currentSession(), findUser(self)   — 캐시 읽기 (즉시 렌더용)
 *   비동기 API: signIn/signOut/validateSession/listAllUsers/adminCreateUser/
 *              adminResetPassword/updateProfile/changePassword  — fetch
 *
 *   ※ 파일명은 호환을 위해 유지(authMock). 더 이상 목업 아님.
 */

const SESSION_KEY = "siwon.auth.session"; // 표시용 캐시 {id,name,role,status,createdAt,lastLoginAt,...}

// ── 검증 규칙 / 라벨 (프론트 표시·입력검증용) ────────────
export const RULES = {
  ID_RE:    /^[A-Za-z0-9._-]{2,20}$/,
  PW_RE:    /^.{4,}$/,
  NAME_RE:  /^.{1,20}$/,
  ROLES:    ["superadmin", "admin", "operator", "guest"], // viewer 제외 (superadmin=총괄 관리자, 임명은 총괄 관리자만 — 백엔드 강제)
  STATUSES: ["active", "disabled"],
};

export const ROLE_LABEL = {
  superadmin: "총괄 관리자",
  admin:    "시원팀",
  operator: "관제사",
  viewer:   "뷰어",      // 레거시 계정 표시용 폴백 (신규 생성 불가)
  guest:    "게스트",
};

// 역할별 아바타 이미지 (public/avatars/). viewer 는 신규 불가 → 사용처에서 guest 로 폴백.
export const ROLE_AVATAR = {
  superadmin: "/avatars/developer.png",
  admin:    "/avatars/admin.png",
  operator: "/avatars/operator.png",
  guest:    "/avatars/guest.png",
};

export const STATUS_LABEL = {
  active:   "활성",
  disabled: "비활성",
  pending:  "대기",      // 레거시 폴백
};

// ── 표시용 세션 캐시 ──────────────────────────────────────
function cacheSession(user) {
  try {
    if (user) localStorage.setItem(SESSION_KEY, JSON.stringify(user));
    else      localStorage.removeItem(SESSION_KEY);
  } catch { /* localStorage 비활성 환경 무시 */ }
}

/** 현재 세션(표시용 캐시) — 동기. UI 게이팅/초기 렌더용. */
export function currentSession() {
  try {
    const raw = localStorage.getItem(SESSION_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

/** 본인 레코드(동기) — UserModals fullRecord 용. 본인 id 일 때만 캐시 반환. */
export function findUser(id) {
  const s = currentSession();
  return s && s.id === id ? s : null;
}

// ── fetch 헬퍼 (same-origin → 쿠키 자동 동봉) ─────────────
async function jfetch(url, method, body) {
  const opt = { method, credentials: "same-origin" };
  if (body !== undefined) {
    opt.headers = { "Content-Type": "application/json" };
    opt.body = JSON.stringify(body || {});
  }
  let http = 0, data = {};
  try {
    const r = await fetch(url, opt);
    http = r.status;
    try { data = await r.json(); } catch { data = {}; }
  } catch (e) {
    return { ok: false, error: "서버에 연결할 수 없습니다.", http: 0 };
  }
  return { http, ...data };
}
const jget   = (u)    => jfetch(u, "GET");
const jpost  = (u, b) => jfetch(u, "POST", b ?? {});
const jpatch = (u, b) => jfetch(u, "PATCH", b ?? {});

// ── 인증 ──────────────────────────────────────────────────

/** 로그인. @returns {{ok:true,user}}|{{ok:false,error,status?}} */
export async function signIn({ id, pw }) {
  const res = await jpost("/api/auth/login", { id, pw });
  if (res.ok && res.user) {
    cacheSession(res.user);
    return { ok: true, user: res.user };
  }
  return { ok: false, error: res.error || "로그인에 실패했습니다.", status: res.status };
}

/** 1회용(에페메럴) 로그인 — DB 계정 없이 viewer 둘러보기 세션. @returns {{ok:true,user}}|{{ok:false,error}} */
export async function guestLogin() {
  const res = await jpost("/api/auth/guest", {});
  if (res.ok && res.user) {
    cacheSession(res.user);
    return { ok: true, user: res.user };
  }
  return { ok: false, error: res.error || "1회용 로그인에 실패했습니다." };
}

/** 로그아웃 — 서버 쿠키 무효화 + 캐시 제거. */
export async function signOut() {
  cacheSession(null);                       // 로컬 캐시 먼저 제거 (UI 즉시 반영)
  try { await jpost("/api/auth/logout", {}); } catch { /* best-effort */ }
}

/** 부팅/새로고침 시 서버 세션 검증 → 캐시 갱신. @returns {{ok:true,user}}|{{ok:false}} */
export async function validateSession() {
  const res = await jget("/api/auth/me");
  if (res.ok && res.user) {
    cacheSession(res.user);
    return { ok: true, user: res.user };
  }
  // 서버가 명확히 "세션 없음"(HTTP 200 + ok:false)일 때만 캐시 비움.
  // 네트워크/5xx 는 일시 오류로 보고 캐시 유지 (blip 으로 로그아웃 방지).
  if (res.http === 200) { cacheSession(null); return { ok: false, authed: false }; }
  return { ok: false, transient: true };
}

/** 로그인 ID 실시간 존재 확인 (로그인 폼 체크표시용). 형식 불량이면 서버 조회 없이 false. */
export async function checkUserId(id) {
  const v = String(id || "").trim();
  if (!RULES.ID_RE.test(v)) return false;
  const res = await jget(`/api/auth/exists?id=${encodeURIComponent(v)}`);
  return !!(res.ok && res.exists);
}

// ── 관리자 ────────────────────────────────────────────────

export async function listAllUsers() {
  const res = await jget("/api/auth/users");
  return { ok: !!res.ok, users: res.users || [], error: res.error };
}

// ── 공지사항(배너) ─────────────────────────────────────────
/** 현재 공지 조회(관리자 폼 프리필용). @returns {{ok, announcement}} */
export async function getAnnouncement() {
  const res = await jget("/api/announcement");
  return { ok: !!res.ok, announcement: res.announcement || null };
}
/** 공지 저장(관리자 전용). @param {{message, level, active}} */
export async function saveAnnouncement({ message, level, active }) {
  const res = await jpost("/api/announcement", { message, level, active });
  return res.ok ? { ok: true } : { ok: false, error: res.error || "공지 저장에 실패했습니다." };
}

export async function adminCreateUser(_actor, { id, pw, name, role, memo }) {
  const res = await jpost("/api/auth/users", { id, pw, name, role, memo });
  return res.ok
    ? { ok: true, user: res.user }
    : { ok: false, error: res.error || "등록 실패", field: res.field };
}

export async function adminResetPassword(_actor, id, newPw) {
  const res = await jpost(`/api/auth/users/${encodeURIComponent(id)}/reset-password`, newPw ? { newPw } : {});
  return res.ok
    ? { ok: true, userId: res.userId, newPw: res.newPw }
    : { ok: false, error: res.error || "재설정 실패" };
}

export async function adminDeleteUser(_actor, id) {
  const res = await jfetch(`/api/auth/users/${encodeURIComponent(id)}`, "DELETE");
  return res.ok ? { ok: true, userId: res.userId } : { ok: false, error: res.error || "삭제 실패" };
}

export async function adminSetMemo(_actor, id, memo) {
  const res = await jpatch(`/api/auth/users/${encodeURIComponent(id)}/memo`, { memo });
  return res.ok ? { ok: true, memo: res.memo } : { ok: false, error: res.error || "메모 저장 실패" };
}

// 사용자 정보 통합 수정 (이름/역할/메모/새 비밀번호). 변경할 필드만 전달.
export async function adminUpdateUser(_actor, id, { name, role, memo, newPw }) {
  const body = {};
  if (name !== undefined) body.name = name;
  if (role !== undefined) body.role = role;
  if (memo !== undefined) body.memo = memo;
  if (newPw) body.newPw = newPw;
  const res = await jpatch(`/api/auth/users/${encodeURIComponent(id)}`, body);
  return res.ok
    ? { ok: true, user: res.user, newPw: res.newPw }
    : { ok: false, error: res.error || "수정 실패", field: res.field };
}

// ── 본인 계정 ─────────────────────────────────────────────

export async function updateProfile(_actor, { name, bio, title, github, avatar }) {
  // avatar: data URL(새 사진) | null(제거) | undefined(변경 안 함). bio/title/github 는 문자열.
  const body = { name };
  if (bio !== undefined) body.bio = bio;
  if (title !== undefined) body.title = title;
  if (github !== undefined) body.github = github;
  if (avatar !== undefined) body.avatar = avatar;
  const res = await jpatch("/api/auth/profile", body);
  if (res.ok && res.user) {
    cacheSession(res.user);
    return { ok: true, user: res.user };
  }
  return { ok: false, error: res.error || "수정 실패", field: res.field };
}

export async function changePassword(_actor, { currentPw, newPw, newPw2 }) {
  if (newPw2 !== undefined && newPw !== newPw2)
    return { ok: false, error: "새 비밀번호 확인이 일치하지 않습니다.", field: "newPw2" };
  const res = await jpost("/api/auth/change-password", { currentPw, newPw });
  return res.ok ? { ok: true } : { ok: false, error: res.error || "변경 실패", field: res.field };
}

// ── 공개 회원가입 — 역할 guest 고정 (백엔드가 강제). 승급은 관리자. ──
export async function signUp(form) {
  const res = await jpost("/api/auth/signup", { id: form?.id, pw: form?.pw, name: form?.name });
  return res.ok ? { ok: true, user: res.user } : { ok: false, error: res.error || "가입 실패", field: res.field };
}
