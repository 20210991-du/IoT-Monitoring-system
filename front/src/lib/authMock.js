/**
 * authMock.js — localStorage 기반 mock 인증 (캡스톤 데모용)
 *
 * ⚠️ 데모 전용. 실제 운영은 백엔드 /api/auth/* 로 교체 예정.
 *    교체 인터페이스 정의: vault `02. 기획/04. 기획결정사항/ADR-010-인증-백엔드인터페이스.md`
 *
 * 저장 위치: localStorage["siwon.auth.users"] (JSON 배열)
 *           localStorage["siwon.auth.session"] (현재 세션, JSON)
 *
 * 비밀번호는 해시 X (mock). 실제 백엔드는 bcrypt 등 사용 가정.
 */

const USERS_KEY   = "siwon.auth.users";
const SESSION_KEY = "siwon.auth.session";
const MIGRATION_KEY = "siwon.auth.migration"; // 시드 admin 동기화 버전
const SEED_VERSION = "2026-05-04-admin-id";   // 시드 변경 시 이 값 갱신 → 다음 로드 1회 동기화

// ── 시드 계정 (최초 부팅 시 1회) ─────────────────────────
//
//  status: "active" — 정상 사용
//          "pending" — 가입 신청, 관리자 승인 대기
//          "rejected" — 관리자 반려
//
const SEED_USERS = [
  {
    id: "admin",
    pw: "11111111",
    name: "관리자",
    role: "admin",
    status: "active",
    createdAt: "2026-04-20T00:00:00.000Z",
    approvedAt: "2026-04-20T00:00:00.000Z",
    approvedBy: "system",
  },
];

function load() {
  try {
    const raw = localStorage.getItem(USERS_KEY);
    if (!raw) {
      localStorage.setItem(USERS_KEY, JSON.stringify(SEED_USERS));
      localStorage.setItem(MIGRATION_KEY, SEED_VERSION);
      return [...SEED_USERS];
    }
    let arr = JSON.parse(raw);
    if (!Array.isArray(arr)) arr = [];

    // 시드 동기화: SEED_VERSION 이 바뀌었을 때만 1회 적용.
    // (시드 admin 의 pw/role/status 를 SEED_USERS 와 강제 동기화 — 데모 편의)
    if (localStorage.getItem(MIGRATION_KEY) !== SEED_VERSION) {
      // 구 admin ID(admin.siwon) → 신 ID(admin) 마이그레이션 (있을 때만)
      const oldAdminIdx = arr.findIndex((u) => u.id === "admin.siwon");
      if (oldAdminIdx >= 0) {
        // 같은 id="admin" 신규 사용자가 이미 있으면 구계정 제거, 아니면 rename
        const newAdminExists = arr.some((u, i) => u.id === "admin" && i !== oldAdminIdx);
        if (newAdminExists) {
          arr.splice(oldAdminIdx, 1);
        } else {
          arr[oldAdminIdx] = { ...arr[oldAdminIdx], id: "admin" };
        }
        // 활성 세션이 admin.siwon 이었으면 무효화 (재로그인 강제)
        try {
          const sess = JSON.parse(localStorage.getItem(SESSION_KEY) || "null");
          if (sess && sess.id === "admin.siwon") localStorage.removeItem(SESSION_KEY);
        } catch { /* ignore */ }
      }

      let changed = true;  // 위에서 이미 손댔거나, 시드 sync 로 변경 가능성
      SEED_USERS.forEach((seed) => {
        const idx = arr.findIndex((u) => u.id === seed.id);
        if (idx < 0) {
          arr.push({ ...seed });
        } else {
          arr[idx] = {
            ...arr[idx],
            pw:     seed.pw,
            role:   seed.role,
            status: seed.status,
            name:   arr[idx].name || seed.name,
          };
        }
      });
      if (changed) localStorage.setItem(USERS_KEY, JSON.stringify(arr));
      localStorage.setItem(MIGRATION_KEY, SEED_VERSION);
    }
    return arr;
  } catch {
    return [...SEED_USERS];
  }
}

function save(users) {
  localStorage.setItem(USERS_KEY, JSON.stringify(users));
}

// ── 검증 규칙 ────────────────────────────────────────────
// 데모용 느슨한 검증 규칙 (실 운영 전환 시 강화 — ADR-010 참조)
export const RULES = {
  ID_RE:    /^[A-Za-z0-9._-]{2,20}$/,            // 2~20자, 영숫·._-
  PW_RE:    /^.{4,}$/,                            // 4자 이상, 종류 무관
  NAME_RE:  /^.{1,20}$/,                          // 1~20자, 종류 무관
  ROLES:    ["admin", "operator", "viewer", "guest"],
  STATUSES: ["pending", "active", "rejected"],
};

export const ROLE_LABEL = {
  admin:    "관리자",
  operator: "관제사",
  viewer:   "뷰어",
  guest:    "게스트",
};

export const STATUS_LABEL = {
  pending:  "승인 대기",
  active:   "활성",
  rejected: "반려",
};

// ── API ──────────────────────────────────────────────────

export function listUsers() {
  return load();
}

export function findUser(id) {
  return load().find((u) => u.id === id) || null;
}

/**
 * 회원가입.
 * @returns {{ok:true, user}} | {ok:false, error:string, field?:string}
 */
export function signUp({ id, pw, pw2, name, role }) {
  // 입력 검증 (데모 — 느슨함)
  if (!RULES.ID_RE.test(id))       return { ok: false, error: "ID 는 2~20자 (영문/숫자/._-).", field: "id" };
  if (!RULES.PW_RE.test(pw))       return { ok: false, error: "비밀번호는 4자 이상.", field: "pw" };
  if (pw !== pw2)                  return { ok: false, error: "비밀번호 확인이 일치하지 않습니다.", field: "pw2" };
  if (!RULES.NAME_RE.test(name))   return { ok: false, error: "이름은 1~20자.", field: "name" };
  if (!RULES.ROLES.includes(role)) return { ok: false, error: "역할을 선택해 주세요.", field: "role" };

  const users = load();
  if (users.some((u) => u.id === id))
    return { ok: false, error: "이미 사용 중인 ID 입니다.", field: "id" };

  const user = {
    id, pw, name, role,
    status: "pending",                 // 신규 가입은 무조건 승인 대기
    createdAt: new Date().toISOString(),
    approvedAt: null,
    approvedBy: null,
  };
  users.push(user);
  save(users);
  return { ok: true, user: { ...user, pw: undefined }, pending: true };
}

/**
 * 로그인.
 * @returns {{ok:true, user}} | {ok:false, error, status?}
 */
export function signIn({ id, pw, remember = true }) {
  const u = findUser(id);
  if (!u)            return { ok: false, error: "존재하지 않는 ID 입니다." };
  if (u.pw !== pw)   return { ok: false, error: "비밀번호가 일치하지 않습니다." };

  // 상태 확인 (pending/rejected 차단)
  if (u.status === "pending") {
    return {
      ok: false,
      status: "pending",
      error: "관리자 승인 대기 중인 계정입니다. 승인 완료 후 로그인할 수 있습니다.",
    };
  }
  if (u.status === "rejected") {
    return {
      ok: false,
      status: "rejected",
      error: "가입이 반려된 계정입니다. 관리자에게 문의해 주세요.",
    };
  }

  // 마지막 접속 기록 (이전값을 lastLoginAt 에 보관 → 모달에서 "직전 접속" 표시 가능)
  const nowIso = new Date().toISOString();
  const users = load();
  const idx   = users.findIndex((x) => x.id === u.id);
  if (idx >= 0) {
    users[idx] = {
      ...users[idx],
      previousLoginAt: users[idx].lastLoginAt || null,
      lastLoginAt:     nowIso,
    };
    save(users);
  }

  const session = {
    id: u.id, name: u.name, role: u.role,
    status: u.status,
    loggedInAt: nowIso,
  };
  if (remember) localStorage.setItem(SESSION_KEY, JSON.stringify(session));
  return { ok: true, user: session };
}

export function signOut() {
  localStorage.removeItem(SESSION_KEY);
}

export function currentSession() {
  try {
    const raw = localStorage.getItem(SESSION_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

// ── 본인 계정: 프로필 수정 / 비밀번호 변경 ───────────────

/**
 * 내 정보 수정 (이름).
 * @returns {{ok:true, user}} | {ok:false, error, field?}
 */
export function updateProfile(actor, { name }) {
  if (!actor) return { ok: false, error: "로그인이 필요합니다." };
  const users = load();
  const idx = users.findIndex((u) => u.id === actor.id);
  if (idx < 0) return { ok: false, error: "사용자를 찾을 수 없습니다." };

  const trimmedName = (name || "").trim();
  if (!RULES.NAME_RE.test(trimmedName)) return { ok: false, error: "이름은 1~20자.", field: "name" };

  users[idx] = {
    ...users[idx],
    name:      trimmedName,
    updatedAt: new Date().toISOString(),
  };
  save(users);

  // 세션도 갱신 (Header 즉시 반영)
  const session = currentSession();
  if (session && session.id === actor.id) {
    const newSession = { ...session, name: trimmedName };
    localStorage.setItem(SESSION_KEY, JSON.stringify(newSession));
    return { ok: true, user: newSession };
  }
  return { ok: true, user: { ...users[idx], pw: undefined } };
}

/**
 * 비밀번호 변경 (현재 비번 검증 후).
 * @returns {{ok:true}} | {ok:false, error, field?}
 */
export function changePassword(actor, { currentPw, newPw, newPw2 }) {
  if (!actor) return { ok: false, error: "로그인이 필요합니다." };
  const users = load();
  const idx = users.findIndex((u) => u.id === actor.id);
  if (idx < 0) return { ok: false, error: "사용자를 찾을 수 없습니다." };

  if (users[idx].pw !== currentPw) return { ok: false, error: "현재 비밀번호가 일치하지 않습니다.", field: "currentPw" };
  if (!RULES.PW_RE.test(newPw))    return { ok: false, error: "비밀번호는 4자 이상.", field: "newPw" };
  if (newPw !== newPw2)            return { ok: false, error: "새 비밀번호 확인이 일치하지 않습니다.", field: "newPw2" };
  if (newPw === currentPw)         return { ok: false, error: "새 비밀번호가 현재와 동일합니다.", field: "newPw" };

  users[idx] = {
    ...users[idx],
    pw: newPw,
    passwordChangedAt: new Date().toISOString(),
  };
  save(users);
  return { ok: true };
}

// ── 관리자 전용: 사용자 승인/반려 ─────────────────────────
//
//  실제 백엔드는 JWT 의 role=admin 검증으로 보호. mock 은 호출자가
//  admin role 인지를 actor 로 받아 클라이언트 측에서 가드.

function requireAdmin(actor) {
  if (!actor || actor.role !== "admin") {
    return { ok: false, error: "관리자 권한이 필요합니다." };
  }
  return null;
}

/**
 * 모든 사용자 조회 (admin).
 * status 필터 가능: "pending" | "active" | "rejected" | undefined(전체)
 */
export function listAllUsers(actor, status) {
  const guard = requireAdmin(actor);
  if (guard) return guard;
  const users = load().map((u) => ({ ...u, pw: undefined }));
  return {
    ok: true,
    users: status ? users.filter((u) => u.status === status) : users,
  };
}

export function listPending(actor) {
  return listAllUsers(actor, "pending");
}

/**
 * 가입 신청 승인 (status: pending → active).
 */
export function approveUser(actor, userId) {
  const guard = requireAdmin(actor);
  if (guard) return guard;
  const users = load();
  const idx = users.findIndex((u) => u.id === userId);
  if (idx < 0)                       return { ok: false, error: "존재하지 않는 사용자입니다." };
  if (users[idx].status === "active") return { ok: false, error: "이미 활성화된 계정입니다." };

  users[idx] = {
    ...users[idx],
    status: "active",
    approvedAt: new Date().toISOString(),
    approvedBy: actor.id,
    rejectedReason: undefined,
  };
  save(users);
  return { ok: true, user: { ...users[idx], pw: undefined } };
}

/**
 * 가입 신청 반려 (status: pending → rejected).
 */
export function rejectUser(actor, userId, reason) {
  const guard = requireAdmin(actor);
  if (guard) return guard;
  const users = load();
  const idx = users.findIndex((u) => u.id === userId);
  if (idx < 0) return { ok: false, error: "존재하지 않는 사용자입니다." };

  // admin 본인을 반려 못 하게
  if (users[idx].id === actor.id) return { ok: false, error: "본인 계정은 반려할 수 없습니다." };

  users[idx] = {
    ...users[idx],
    status: "rejected",
    rejectedAt: new Date().toISOString(),
    rejectedBy: actor.id,
    rejectedReason: (reason || "").trim() || null,
  };
  save(users);
  return { ok: true, user: { ...users[idx], pw: undefined } };
}

/**
 * 반려된 사용자를 다시 승인 대기로 되돌리기 (선택).
 */
export function reactivateUser(actor, userId) {
  const guard = requireAdmin(actor);
  if (guard) return guard;
  const users = load();
  const idx = users.findIndex((u) => u.id === userId);
  if (idx < 0) return { ok: false, error: "존재하지 않는 사용자입니다." };

  users[idx] = {
    ...users[idx],
    status: "pending",
    rejectedReason: undefined,
  };
  save(users);
  return { ok: true, user: { ...users[idx], pw: undefined } };
}

// ── 관리자 전용: 사용자 비밀번호 재설정 ───────────────────
//
//  운영 정책: 비밀번호 분실 시 본인 셀프 재설정 X.
//  관리자가 운영자 관리 페이지에서 직접 새 비번 발급 → 본인에게 전달.
//
/**
 * @returns {{ok:true, userId, newPw}} | {ok:false, error}
 */
export function adminResetPassword(actor, userId, newPw) {
  const guard = requireAdmin(actor);
  if (guard) return guard;
  const users = load();
  const idx = users.findIndex((u) => u.id === userId);
  if (idx < 0)                   return { ok: false, error: "존재하지 않는 사용자입니다." };
  if (!RULES.PW_RE.test(newPw))  return { ok: false, error: "비밀번호는 4자 이상.", field: "newPw" };

  users[idx] = {
    ...users[idx],
    pw: newPw,
    passwordChangedAt: new Date().toISOString(),
    passwordChangedBy: actor.id,
  };
  save(users);
  return { ok: true, userId, newPw };
}

// 초기화(개발 편의)
export function _resetAll() {
  localStorage.removeItem(USERS_KEY);
  localStorage.removeItem(SESSION_KEY);
  localStorage.removeItem(MIGRATION_KEY);
}
