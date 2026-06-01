/**
 * 통합 서버 (Express)
 *  - 정적 파일 (dist/) 서빙
 *  - POST /api/chat → Ollama 프록시 (시스템 프롬프트 + 컨텍스트 주입)
 *  - GET  /api/devices, /api/alarms, /api/summary 등 → siwon MySQL 쿼리
 *
 * 실행: node server.js
 *   PORT          (default 5050)
 *   OLLAMA_URL    (default http://localhost:11434)
 *   OLLAMA_MODEL  (default qwen3.5:9b)
 *   SIWON_DB_*    (siwon-db.env 에서 source — 없으면 DB API 비활성)
 *
 * 같은 origin 으로 정적 파일과 API 동시 서비스 → CORS 불필요.
 */

import express from "express";
import helmet from "helmet";
import rateLimit from "express-rate-limit";
import path from "path";
import { readFileSync, existsSync, createWriteStream } from "fs";
import { fileURLToPath } from "url";
import mysql from "mysql2/promise";
import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
import { randomBytes, randomInt } from "crypto";
import { WebSocketServer } from "ws";

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

// ── AI 설정 로드 (이두현 학습 결과) ────────────────
// ai/config/device_thresholds.json  — 단말별 99 percentile MSE
// ai/config/model_config.json       — 시퀀스 길이·피처·희생전류 단말 등
// ai/config/eval_metrics.json       — 학습 시점 평가 통계
let DEVICE_THRESHOLDS = {};   // { "TB24-250401": 0.00106..., ... }
let MODEL_CONFIG = null;
let EVAL_METRICS = null;
{
  const aiDir = path.join(__dirname, "..", "ai", "config");
  const load = (file, fallback) => {
    try {
      const fp = path.join(aiDir, file);
      if (existsSync(fp)) return JSON.parse(readFileSync(fp, "utf8"));
    } catch (e) { console.warn(`[AI config ${file}]`, e.message); }
    return fallback;
  };
  DEVICE_THRESHOLDS = load("device_thresholds.json", {});
  MODEL_CONFIG      = load("model_config.json", null);
  EVAL_METRICS      = load("eval_metrics.json", null);
  console.log(`▶ AI cfg  thresholds=${Object.keys(DEVICE_THRESHOLDS).length}대 · model_config=${MODEL_CONFIG ? "OK" : "X"} · eval_metrics=${EVAL_METRICS ? "OK" : "X"}`);
}

// 단말 위험도 판정 (이두현 명세 — threshold 의 70%/100% 분기)
//   정상  : mse < threshold × 0.70
//   관찰  : threshold × 0.70 ≤ mse ≤ threshold × 1.00
//   이상  : mse > threshold × 1.00
function classifyMse(deviceId, mse) {
  const th = DEVICE_THRESHOLDS[deviceId];
  if (th == null || !Number.isFinite(Number(mse))) return null;
  const ratio = Number(mse) / th;
  let level = "정상";
  if (ratio > 1.0) level = "이상";
  else if (ratio >= 0.7) level = "관찰";
  return {
    deviceId,
    threshold: th,
    threshold70: Number((th * 0.7).toFixed(6)),
    threshold100: Number(th.toFixed(6)),
    mse: Number(Number(mse).toFixed(6)),
    ratio: Number(ratio.toFixed(3)),
    ratioPercent: Number((ratio * 100).toFixed(1)),
    level,
  };
}

const PORT         = process.env.PORT          || 5050;
const OLLAMA_URL   = process.env.OLLAMA_URL    || "http://localhost:11434";
const OLLAMA_MODEL = process.env.OLLAMA_MODEL  || "qwen3.5:9b";
// UI 에서 선택 가능한 챗봇 모델 화이트리스트 (그 외 값은 기본 모델로 폴백 — 안전)
const SELECTABLE_MODELS = ["qwen3.5:9b", "qwen3:14b", "qwen3.5:27b", "gpt-4o-mini", "gpt-5", "gpt-5.5"];
const pickModel = (m) => (SELECTABLE_MODELS.includes(m) ? m : OLLAMA_MODEL);
// OpenAI(GPT) 프로바이더 — 외부 전송. 키는 secrets/local/openai.env → process.env.OPENAI_API_KEY.
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || "";
const OPENAI_URL = "https://api.openai.com/v1/chat/completions";
const isOpenAI = (m) => typeof m === "string" && m.startsWith("gpt-");

const SIWON_DB_HOST = process.env.SIWON_DB_HOST || "127.0.0.1";
const SIWON_DB_PORT = parseInt(process.env.SIWON_DB_PORT || "3306", 10);
const SIWON_DB_USER = process.env.SIWON_DB_USER || "siwon_app";
const SIWON_DB_PASS = process.env.SIWON_DB_PASS || "";
const SIWON_DB_NAME = process.env.SIWON_DB_NAME || "siwon";
const SITE_ID       = parseInt(process.env.SITE_ID || "2", 10);  // 군산도시가스

// ─────────────────────────────────────────────────────
// 데모 모드 — 발표/시연용 가상 장비 10대 (메모리 only, DB INSERT 안 함)
//   토글 OFF 시 흔적 제로 / ON 시 KPI·지도·표·챗봇 모두에 추가 표시.
//   클라이언트: GET ?demo=1 또는 POST body { demo: true }
// ─────────────────────────────────────────────────────
function isDemoMode(req) {
  if (!req) return false;
  if (req.query && String(req.query.demo) === "1") return true;
  if (req.body && req.body.demo === true) return true;
  return false;
}

// 시간 마커 — lastMeasured 가 호출 시점 기준 N시간 전 timestamp 가 되도록
function nowMinusH(h) { return new Date(Date.now() - h * 3600_000).toISOString().slice(0, 19).replace("T", " "); }

// 가상 장비 10대 (DB 충돌 회피용 음수 TRANSMITTER_ID).
//   이상 3 + 관찰 4 + 통신장애 3.
//   좌표·시설번호·POSITION 모두 군산 자연스러운 위치로.
function getDemoDevices() {
  return [
    // ── 이상 3대 (critical: 최근 알람 발생) ──
    { deviceId: "DEMO-001", txid: -1, facility: "1-DEMO01", location: "시청 사거리 (데모)",
      lat: 35.9676, lng: 126.7369, hoursSilent: 2, recentAlarms: 3, status: "critical",
      sensors: { volt: -540, sacrificial: 0.6, ac: 312, battery: 3580, temp: 28.4, hum: 71, shock: 0, commDbm: -88 },
      lastMeasured: nowMinusH(2),
      mse: 0.0042, threshold: 0.0011, riskLevel: "이상", aiReliability: "신뢰" },
    { deviceId: "DEMO-002", txid: -2, facility: "2-DEMO02", location: "대야 사거리 (데모)",
      lat: 35.962, lng: 126.745, hoursSilent: 1, recentAlarms: 5, status: "critical",
      sensors: { volt: -480, sacrificial: 0.3, ac: 540, battery: 3520, temp: 30.1, hum: 68, shock: 1, commDbm: -82 },
      lastMeasured: nowMinusH(1),
      mse: 0.0061, threshold: 0.0013, riskLevel: "이상", aiReliability: "신뢰" },
    { deviceId: "DEMO-003", txid: -3, facility: "8-DEMO03", location: "소룡동 공단 입구 (데모)",
      lat: 35.985, lng: 126.700, hoursSilent: 1, recentAlarms: 2, status: "critical",
      sensors: { volt: -620, sacrificial: 0.8, ac: 280, battery: 3610, temp: 27.8, hum: 70, shock: 0, commDbm: -85 },
      lastMeasured: nowMinusH(1),
      mse: 0.0038, threshold: 0.0011, riskLevel: "이상", aiReliability: "신뢰" },
    // ── 관찰 4대 (warn) ──
    { deviceId: "DEMO-101", txid: -101, facility: "4-DEMO11", location: "미룡동 교차로 (데모)",
      lat: 35.937, lng: 126.696, hoursSilent: 1, recentAlarms: 0, status: "warn",
      sensors: { volt: -780, sacrificial: 0.9, ac: 195, battery: 3700, temp: 26.5, hum: 65, shock: 0, commDbm: -78 },
      lastMeasured: nowMinusH(1),
      mse: 0.0019, threshold: 0.0011, riskLevel: "관찰", aiReliability: "주의" },
    { deviceId: "DEMO-102", txid: -102, facility: "9-DEMO12", location: "해망동 박물관 앞 (데모)",
      lat: 35.990, lng: 126.704, hoursSilent: 1, recentAlarms: 0, status: "warn",
      sensors: { volt: -810, sacrificial: 1.1, ac: 175, battery: 3680, temp: 27.0, hum: 67, shock: 0, commDbm: -76 },
      lastMeasured: nowMinusH(1),
      mse: 0.0016, threshold: 0.0011, riskLevel: "관찰", aiReliability: "주의" },
    { deviceId: "DEMO-103", txid: -103, facility: "3-DEMO13", location: "소룡동 현대자동차 옆 (데모)",
      lat: 35.983, lng: 126.682, hoursSilent: 2, recentAlarms: 0, status: "warn",
      sensors: { volt: -795, sacrificial: 1.0, ac: 188, battery: 3690, temp: 27.5, hum: 66, shock: 0, commDbm: -79 },
      lastMeasured: nowMinusH(2),
      mse: 0.0015, threshold: 0.0011, riskLevel: "관찰", aiReliability: "주의" },
    { deviceId: "DEMO-104", txid: -104, facility: "8-DEMO14", location: "대야 버스터미널 옆 (데모)",
      lat: 35.946, lng: 126.810, hoursSilent: 1, recentAlarms: 0, status: "warn",
      sensors: { volt: -805, sacrificial: 1.2, ac: 180, battery: 3675, temp: 27.2, hum: 68, shock: 0, commDbm: -77 },
      lastMeasured: nowMinusH(1),
      mse: 0.0014, threshold: 0.0011, riskLevel: "관찰", aiReliability: "주의" },
    // ── 통신장애 3대 (offline) ──
    { deviceId: "DEMO-201", txid: -201, facility: "1-DEMO21", location: "새만금방조제 5공구 (데모)",
      lat: 35.819, lng: 126.477, hoursSilent: 72, recentAlarms: 0, status: "offline",
      sensors: { volt: -1850, sacrificial: 0.8, ac: 95, battery: 3450, temp: 26.0, hum: 72, shock: 0, commDbm: -118 },
      lastMeasured: nowMinusH(72),
      mse: null, threshold: 0.0011, riskLevel: "관찰", aiReliability: "신뢰불가" },
    { deviceId: "DEMO-202", txid: -202, facility: "8-DEMO22", location: "조촌동 (데모)",
      lat: 35.975, lng: 126.741, hoursSilent: 120, recentAlarms: 0, status: "offline",
      sensors: { volt: -1920, sacrificial: 0.9, ac: 102, battery: 3480, temp: 26.5, hum: 70, shock: 0, commDbm: -120 },
      lastMeasured: nowMinusH(120),
      mse: null, threshold: 0.0011, riskLevel: "관찰", aiReliability: "신뢰불가" },
    { deviceId: "DEMO-203", txid: -203, facility: "9-DEMO23", location: "해망동 굴 입구 (데모)",
      lat: 35.991, lng: 126.703, hoursSilent: 200, recentAlarms: 0, status: "offline",
      sensors: { volt: -1880, sacrificial: 0.7, ac: 88, battery: 3420, temp: 26.2, hum: 73, shock: 0, commDbm: -119 },
      lastMeasured: nowMinusH(200),
      mse: null, threshold: 0.0011, riskLevel: "관찰", aiReliability: "신뢰불가" },
  ];
}

// 데모 알람 (위험 3대 + 충격 1건)
function getDemoAlarms() {
  return [
    { occurredAt: nowMinusH(2),  grade: "위험", gradeId: 1, deviceId: "DEMO-001", facility: "1-DEMO01", value: 312, contents: "AC 유입 임계 초과 (200mV)" },
    { occurredAt: nowMinusH(1),  grade: "위험", gradeId: 1, deviceId: "DEMO-002", facility: "2-DEMO02", value: 540, contents: "AC 유입 임계 초과 (500mV 즉각 점검)" },
    { occurredAt: nowMinusH(3),  grade: "경고", gradeId: 2, deviceId: "DEMO-002", facility: "2-DEMO02", value: 1,   contents: "충격 센서 감지" },
    { occurredAt: nowMinusH(1),  grade: "위험", gradeId: 1, deviceId: "DEMO-003", facility: "8-DEMO03", value: -620, contents: "방식전위 -850 mV 미달 (부식 진행 가능)" },
  ];
}

// 데모 시계열 — deviceId + kind 의 해시 기반 시드된 가짜 데이터 (재현 가능)
function generateDemoHistory(deviceId, kind, hours) {
  // deviceId+kind 해시 → mulberry32 시드
  let seed = 0;
  for (const ch of deviceId + kind) seed = (seed * 31 + ch.charCodeAt(0)) >>> 0;
  let s = seed >>> 0;
  const rng = () => {
    s |= 0; s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  // 도구 종류별 기본값·변동폭
  const base = { volt: -650, sacrificial: 0.6, ac: 350, battery: 3550, temp: 28, hum: 70, commDbm: -85 }[kind] || 0;
  const swing = { volt: 80, sacrificial: 0.2, ac: 100, battery: 50, temp: 2, hum: 5, commDbm: 5 }[kind] || 1;
  const points = [];
  const now = Date.now();
  const stepMs = (hours * 3600_000) / Math.min(hours, 100);
  const count = Math.min(hours, 100);
  for (let i = 0; i < count; i++) {
    const t = new Date(now - (count - i) * stepMs).toISOString().slice(0, 19).replace("T", " ");
    const v = base + (rng() - 0.5) * 2 * swing;
    points.push({ t, v: Number(v.toFixed(2)) });
  }
  return points;
}

// 데모 단말 ID 빠른 룩업
function findDemoDevice(deviceId) {
  return getDemoDevices().find((d) => d.deviceId === deviceId) || null;
}
function findDemoDeviceByTxid(txid) {
  return getDemoDevices().find((d) => d.txid === txid) || null;
}

const app = express();
app.use(express.json({ limit: "8mb" }));   // base64 이미지 첨부(문의·최대 5장) 대비 상향

// ── 보안: trust proxy(CF 터널 1홉) · helmet 보안헤더 · rate-limit(방문자별) ──
app.set("trust proxy", 1);
app.use(helmet({ contentSecurityPolicy: false }));  // CSP off — 인라인 스타일 SPA·leaflet 타일 안 깨지게
app.use(rateLimit({
  windowMs: 60_000,
  max: 300,                                          // 방문자당 분당 300 (대시보드 폴링 ~20-30/min 대비 넉넉)
  keyGenerator: (req) => req.headers["cf-connecting-ip"] || req.ip,
  skip: (req) => req.path.startsWith("/assets"),     // 정적 에셋 제외
  standardHeaders: true,
  legacyHeaders: false,
  validate: { trustProxy: false },                   // keyGenerator 가 CF-Connecting-IP 사용 → 프록시 검증 스킵
  message: { ok: false, error: "요청이 너무 많습니다. 잠시 후 다시 시도하세요." },
}));

// ── 접속 IP 로깅 — CF-Connecting-IP(실제 방문자, 터널 통과해도 보존). 정적 에셋 제외. ──
//   기록: ~/PJHwork/infra/logs/access.log  (탭 구분: ISO시각 · IP · METHOD · 경로)
//   안전 가드: 스트림/쓰기 실패해도 요청 처리에 영향 없음.
const ACCESS_LOG_PATH = path.join(process.env.HOME || __dirname, "PJHwork/infra/logs/access.log");
let accessLogStream = null;
try {
  accessLogStream = createWriteStream(ACCESS_LOG_PATH, { flags: "a" });
  accessLogStream.on("error", () => { accessLogStream = null; });
} catch { accessLogStream = null; }
app.use((req, _res, next) => {
  try {
    if (accessLogStream && !req.path.startsWith("/assets")) {
      const ip = req.headers["cf-connecting-ip"] || req.headers["x-forwarded-for"] || req.socket?.remoteAddress || "-";
      accessLogStream.write(`${new Date().toISOString()}\t${ip}\t${req.method}\t${req.path}\n`);
    }
  } catch {}
  next();
});

// ── MySQL pool ────────────────────────────────────────
let pool = null;
if (SIWON_DB_PASS) {
  pool = mysql.createPool({
    host: SIWON_DB_HOST, port: SIWON_DB_PORT,
    user: SIWON_DB_USER, password: SIWON_DB_PASS,
    database: SIWON_DB_NAME,
    waitForConnections: true, connectionLimit: 8, queueLimit: 0,
    charset: "utf8mb4_unicode_ci",
    dateStrings: false,
  });
  console.log(`▶ MySQL  ${SIWON_DB_USER}@${SIWON_DB_HOST}:${SIWON_DB_PORT}/${SIWON_DB_NAME}`);
} else {
  console.log("▶ MySQL  비활성 (SIWON_DB_PASS 환경변수 없음)");
}

// 방명록 테이블 보장 (멱등) — 스키마를 코드에 둠. 서버 시작 시 1회.
async function ensureGuestbookSchema() {
  if (!pool) return;
  try {
    await pool.query(`CREATE TABLE IF NOT EXISTS guestbook_messages (
      id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
      user_id INT NULL,
      display_name VARCHAR(40) NOT NULL,
      role VARCHAR(20) NULL,
      body VARCHAR(500) NOT NULL,
      ip VARCHAR(45) NULL,
      ua VARCHAR(255) NULL,
      created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
      deleted_at DATETIME NULL,
      INDEX idx_gb_created (created_at),
      INDEX idx_gb_active (deleted_at, id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci`);
    console.log("▶ 방명록  guestbook_messages 테이블 준비됨");
  } catch (e) { console.error("✗ 방명록 스키마 생성 실패:", e.message); }
}
ensureGuestbookSchema();

function dbRequired(_req, res, next) {
  if (!pool) return res.status(503).json({ ok: false, error: "DB pool 비활성 (서버 환경변수 SIWON_DB_PASS 누락)" });
  next();
}

// ══════════════════════════════════════════════════════════
// 인증 (백엔드) — users 테이블 + bcrypt + JWT httpOnly 쿠키
//   localStorage 목업(authMock.js) 대체. stateless JWT(쿠키 siwon_auth).
//   프론트 user.id = users.login_id, user.name = display_name.
// ══════════════════════════════════════════════════════════
const AUTH_JWT_SECRET = process.env.AUTH_JWT_SECRET || randomBytes(32).toString("hex");
if (!process.env.AUTH_JWT_SECRET) console.warn("⚠ AUTH_JWT_SECRET 미설정 — 임시 시크릿 사용(재시작 시 세션 만료).");
const AUTH_COOKIE  = "siwon_auth";
const AUTH_TTL_SEC = 12 * 3600;
const CREATABLE_ROLES = ["superadmin", "admin", "operator", "guest"];   // viewer 제외
const ROLE_SET = new Set(CREATABLE_ROLES);
const ADMIN_TIER = (role) => role === "admin" || role === "superadmin";  // 관리자급 — 생성·삭제·역할변경은 총관리자만
const LOGIN_ID_RE = /^[A-Za-z0-9._-]{2,20}$/;
const PW_MIN = 4;

const authLimiter = rateLimit({
  windowMs: 60_000, max: 20,                               // 로그인 무차별 대입 완화
  keyGenerator: (req) => req.headers["cf-connecting-ip"] || req.ip,
  standardHeaders: true, legacyHeaders: false, validate: { trustProxy: false },
  message: { ok: false, error: "로그인 시도가 너무 많습니다. 잠시 후 다시 시도하세요." },
});

function parseCookies(req) {
  const out = {}; const raw = req.headers.cookie; if (!raw) return out;
  for (const part of raw.split(";")) { const i = part.indexOf("="); if (i < 0) continue; out[part.slice(0, i).trim()] = decodeURIComponent(part.slice(i + 1).trim()); }
  return out;
}
function signAuthToken(u) { return jwt.sign({ uid: u.id, lid: u.login_id, role: u.role, name: u.display_name }, AUTH_JWT_SECRET, { expiresIn: AUTH_TTL_SEC }); }
function setAuthCookie(req, res, token) {
  const secure = (req.headers["x-forwarded-proto"] || "").includes("https") || !!req.secure;
  res.cookie(AUTH_COOKIE, token, { httpOnly: true, sameSite: "lax", secure, path: "/", maxAge: AUTH_TTL_SEC * 1000 });
}
function authClaims(req) { const tok = parseCookies(req)[AUTH_COOKIE]; if (!tok) return null; try { return jwt.verify(tok, AUTH_JWT_SECRET); } catch { return null; } }
function requireAuth(req, res, next)  { const c = authClaims(req); if (!c) return res.status(401).json({ ok: false, error: "로그인이 필요합니다." }); req.auth = c; next(); }
function requireAdmin(req, res, next) { const c = authClaims(req); if (!c) return res.status(401).json({ ok: false, error: "로그인이 필요합니다." }); if (!ADMIN_TIER(c.role)) return res.status(403).json({ ok: false, error: "관리자 권한이 필요합니다." }); req.auth = c; next(); }
function requireSuperAdmin(req, res, next) { const c = authClaims(req); if (!c) return res.status(401).json({ ok: false, error: "로그인이 필요합니다." }); if (c.role !== "superadmin") return res.status(403).json({ ok: false, error: "총관리자 권한이 필요합니다." }); req.auth = c; next(); }
// 챗봇 계정 스코프 — 로그인 개인 계정만 계정-스코프(uid 반환), 공유 게스트(siwon)·비로그인은 null(브라우저-로컬).
//   이 함수가 세션 소유권 + WS room 키를 동시에 결정 → 둘이 항상 일치.
const SHARED_ACCOUNTS = new Set(["siwon"]);
function chatOwner(req) { const c = authClaims(req); if (!c) return null; if (SHARED_ACCOUNTS.has(c.lid)) return null; return c.uid; }
// 챗봇 실시간 동기화(WebSocket) — uid 별 room. 같은 계정의 다른 화면에 새 메시지 push.
const chatRooms = new Map();   // ownerUid -> Set<ws>
function broadcastToOwner(ownerUid, payload, exceptConnId = null) {
  if (ownerUid == null) return;
  const set = chatRooms.get(ownerUid);
  if (!set) return;
  const data = JSON.stringify(payload);
  for (const ws of set) {
    if (ws.readyState === 1 /* OPEN */ && ws._connId !== exceptConnId) {
      try { ws.send(data); } catch {}
    }
  }
}
// ── 방명록(공개 단톡방) — 단일 전체 방. 로그인/게스트/비로그인 누구나 연결·수신. 작성은 REST(/api/guestbook)로 검증·레이트리밋 후 broadcast. ──
const guestbookRoom = new Set();   // 모든 방명록 WS 연결
function broadcastGuestbook(payload) {
  const data = JSON.stringify(payload);
  for (const ws of guestbookRoom) {
    if (ws.readyState === 1 /* OPEN */) { try { ws.send(data); } catch {} }
  }
}
function mapUser(r) { return { id: r.login_id, name: r.display_name, role: r.role, status: r.status, memo: r.memo ?? "", createdAt: r.created_at, lastLoginAt: r.last_login_at, previousLoginAt: null, approvedAt: r.approved_at }; }
async function findUserRow(loginId) { const [rows] = await pool.query("SELECT * FROM users WHERE login_id = ? LIMIT 1", [loginId]); return rows[0] || null; }
function genPw(n = 10) { const cs = "abcdefghjkmnpqrstuvwxyzABCDEFGHJKMNPQRSTUVWXYZ23456789"; let s = ""; for (let i = 0; i < n; i++) s += cs[randomInt(cs.length)]; return s; }
async function auditAuth(req, ok, loginId, reason, name, role) {
  if (!pool) return;
  try {
    const ip = (req.headers["cf-connecting-ip"] || req.headers["x-forwarded-for"] || req.socket?.remoteAddress || "").toString().split(",")[0].slice(0, 45);
    await pool.query(
      "INSERT INTO audit_log (action, target_type, target_id, ip, user_agent, metadata_json) VALUES (?, ?, ?, ?, ?, ?)",
      [ok ? "login" : "login_fail", "auth", String(loginId || "(unknown)").slice(0, 60), ip, String(req.headers["user-agent"] || "").slice(0, 255), JSON.stringify({ ok, name: name || "", role: role || "", reason: reason || "" })],
    );
  } catch { /* swallow — 감사 실패가 로그인 흐름을 막지 않음 */ }
}

// POST /api/auth/login {id, pw}
app.post("/api/auth/login", authLimiter, dbRequired, async (req, res) => {
  const { id, pw } = req.body || {};
  if (!id || !pw) return res.status(400).json({ ok: false, error: "ID 와 비밀번호를 입력하세요." });
  try {
    const u = await findUserRow(String(id).trim());
    if (!u) { await auditAuth(req, false, id, "존재하지 않는 ID"); return res.status(401).json({ ok: false, error: "존재하지 않는 ID 입니다." }); }
    if (!(await bcrypt.compare(String(pw), u.password_hash))) { await auditAuth(req, false, u.login_id, "비밀번호 불일치", u.display_name, u.role); return res.status(401).json({ ok: false, error: "비밀번호가 일치하지 않습니다." }); }
    if (u.status !== "active") {
      await auditAuth(req, false, u.login_id, "status=" + u.status, u.display_name, u.role);
      return res.status(403).json({ ok: false, error: u.status === "disabled" ? "비활성화된 계정입니다. 관리자에게 문의하세요." : "아직 활성화되지 않은 계정입니다.", status: u.status });
    }
    await pool.query("UPDATE users SET last_login_at = NOW() WHERE id = ?", [u.id]);
    setAuthCookie(req, res, signAuthToken(u));
    await auditAuth(req, true, u.login_id, null, u.display_name, u.role);
    res.json({ ok: true, user: mapUser(await findUserRow(u.login_id)) });
  } catch (e) { console.error("[/api/auth/login]", e.message); res.status(500).json({ ok: false, error: "로그인 처리 중 오류" }); }
});

// POST /api/auth/logout
app.post("/api/auth/logout", (req, res) => { res.clearCookie(AUTH_COOKIE, { path: "/" }); res.json({ ok: true }); });

// POST /api/auth/signup — 공개 회원가입. 역할은 guest 고정 · 즉시 활성. 더 높은 권한은 관리자가 부여.
app.post("/api/auth/signup", authLimiter, dbRequired, async (req, res) => {
  const { id, pw, name } = req.body || {};
  const lid = String(id || "").trim();
  if (!LOGIN_ID_RE.test(lid))            return res.status(400).json({ ok: false, error: "ID 는 2~20자 (영문/숫자/._-).", field: "id" });
  if (!pw || String(pw).length < PW_MIN) return res.status(400).json({ ok: false, error: `비밀번호는 ${PW_MIN}자 이상.`, field: "pw" });
  if (!name || !String(name).trim())     return res.status(400).json({ ok: false, error: "이름을 입력하세요.", field: "name" });
  try {
    if (await findUserRow(lid)) return res.status(409).json({ ok: false, error: "이미 사용 중인 ID 입니다.", field: "id" });
    const hash = await bcrypt.hash(String(pw), 10);
    await pool.query("INSERT INTO users (login_id, password_hash, display_name, role, status) VALUES (?, ?, ?, 'guest', 'active')", [lid, hash, String(name).trim().slice(0, 60)]);
    res.json({ ok: true, user: mapUser(await findUserRow(lid)) });
  } catch (e) { console.error("[/api/auth/signup]", e.message); res.status(500).json({ ok: false, error: "가입 처리 중 오류" }); }
});

// GET /api/auth/me — 부팅/새로고침 시 세션 검증
app.get("/api/auth/me", dbRequired, async (req, res) => {
  const c = authClaims(req); if (!c) return res.json({ ok: false });
  try {
    const u = await findUserRow(c.lid);
    if (!u || u.status !== "active") { res.clearCookie(AUTH_COOKIE, { path: "/" }); return res.json({ ok: false }); }
    res.json({ ok: true, user: mapUser(u) });
  } catch { res.status(500).json({ ok: false }); }
});

// GET /api/auth/exists?id= — 로그인 ID 실시간 존재 확인 (로그인 폼 체크표시용).
//   ⚠️ ID enumeration 노출 — 사용자 요청으로 추가. 스크래핑 완화용 별도 rate-limit(분당 100).
//   활성(active) 계정만 true (로그인 가능 여부와 일치). 형식 불량 ID 는 DB 조회 없이 false.
const existsLimiter = rateLimit({
  windowMs: 60_000, max: 100,
  keyGenerator: (req) => req.headers["cf-connecting-ip"] || req.ip,
  standardHeaders: true, legacyHeaders: false, validate: { trustProxy: false },
  message: { ok: false, error: "요청이 너무 많습니다." },
});
app.get("/api/auth/exists", existsLimiter, dbRequired, async (req, res) => {
  const lid = String(req.query.id || "").trim();
  if (!LOGIN_ID_RE.test(lid)) return res.json({ ok: true, exists: false });
  try {
    const [rows] = await pool.query("SELECT 1 FROM users WHERE login_id = ? AND status = 'active' LIMIT 1", [lid]);
    res.json({ ok: true, exists: rows.length > 0 });
  } catch { res.json({ ok: true, exists: false }); }
});

// GET /api/auth/users (admin)
app.get("/api/auth/users", dbRequired, requireAdmin, async (_req, res) => {
  try { const [rows] = await pool.query("SELECT * FROM users ORDER BY created_at ASC"); res.json({ ok: true, users: rows.map(mapUser) }); }
  catch (e) { console.error("[/api/auth/users GET]", e.message); res.status(500).json({ ok: false, error: "목록 조회 오류" }); }
});

// POST /api/auth/users (admin) {id, pw, name, role}
app.post("/api/auth/users", dbRequired, requireAdmin, async (req, res) => {
  const { id, pw, name, role } = req.body || {};
  const lid = String(id || "").trim();
  if (!LOGIN_ID_RE.test(lid))            return res.status(400).json({ ok: false, error: "ID 는 2~20자 (영문/숫자/._-).", field: "id" });
  if (!pw || String(pw).length < PW_MIN) return res.status(400).json({ ok: false, error: `비밀번호는 ${PW_MIN}자 이상.`, field: "pw" });
  if (!name || !String(name).trim())     return res.status(400).json({ ok: false, error: "이름을 입력하세요.", field: "name" });
  if (!ROLE_SET.has(role))               return res.status(400).json({ ok: false, error: "역할을 선택하세요.", field: "role" });
  if (ADMIN_TIER(role) && req.auth.role !== "superadmin") return res.status(403).json({ ok: false, error: "관리자·총관리자 계정은 총관리자만 생성할 수 있습니다.", field: "role" });
  try {
    if (await findUserRow(lid)) return res.status(409).json({ ok: false, error: "이미 사용 중인 ID 입니다.", field: "id" });
    const hash = await bcrypt.hash(String(pw), 10);
    const memo = String((req.body && req.body.memo) || "").slice(0, 500) || null;
    await pool.query("INSERT INTO users (login_id, password_hash, display_name, role, status, approved_at, approved_by, memo) VALUES (?, ?, ?, ?, 'active', NOW(), ?, ?)", [lid, hash, String(name).trim().slice(0, 60), role, req.auth.uid, memo]);
    res.json({ ok: true, user: mapUser(await findUserRow(lid)) });
  } catch (e) { console.error("[/api/auth/users POST]", e.message); res.status(500).json({ ok: false, error: "등록 처리 중 오류" }); }
});

// POST /api/auth/users/:id/reset-password (admin) {newPw?}
app.post("/api/auth/users/:id/reset-password", dbRequired, requireAdmin, async (req, res) => {
  const lid = String(req.params.id || "").trim();
  let newPw = (req.body && req.body.newPw) ? String(req.body.newPw) : "";
  if (newPw && newPw.length < PW_MIN) return res.status(400).json({ ok: false, error: `비밀번호는 ${PW_MIN}자 이상.` });
  if (!newPw) newPw = genPw(10);
  try {
    const u = await findUserRow(lid);
    if (!u) return res.status(404).json({ ok: false, error: "사용자를 찾을 수 없습니다." });
    if (ADMIN_TIER(u.role) && req.auth.role !== "superadmin") return res.status(403).json({ ok: false, error: "관리자·총관리자 비밀번호는 총관리자만 재설정할 수 있습니다." });
    await pool.query("UPDATE users SET password_hash = ?, updated_at = NOW() WHERE id = ?", [await bcrypt.hash(newPw, 10), u.id]);
    res.json({ ok: true, userId: lid, newPw });
  } catch (e) { console.error("[/api/auth/reset-pw]", e.message); res.status(500).json({ ok: false, error: "재설정 처리 중 오류" }); }
});

// DELETE /api/auth/users/:id (admin) — 계정 삭제. 본인·마지막 활성 관리자 삭제 차단.
app.delete("/api/auth/users/:id", dbRequired, requireAdmin, async (req, res) => {
  const lid = String(req.params.id || "").trim();
  try {
    const u = await findUserRow(lid);
    if (!u) return res.status(404).json({ ok: false, error: "사용자를 찾을 수 없습니다." });
    if (u.login_id === req.auth.lid) return res.status(400).json({ ok: false, error: "본인 계정은 삭제할 수 없습니다." });
    if (ADMIN_TIER(u.role) && req.auth.role !== "superadmin") return res.status(403).json({ ok: false, error: "관리자·총관리자 계정은 총관리자만 삭제할 수 있습니다." });
    if (u.role === "admin") {
      const [[{ n }]] = await pool.query("SELECT COUNT(*) n FROM users WHERE role = 'admin' AND status = 'active'");
      if (n <= 1) return res.status(400).json({ ok: false, error: "마지막 관리자 계정은 삭제할 수 없습니다." });
    }
    if (u.role === "superadmin") {
      const [[{ n }]] = await pool.query("SELECT COUNT(*) n FROM users WHERE role = 'superadmin' AND status = 'active'");
      if (n <= 1) return res.status(400).json({ ok: false, error: "마지막 총관리자 계정은 삭제할 수 없습니다." });
    }
    await pool.query("DELETE FROM users WHERE id = ?", [u.id]);
    res.json({ ok: true, userId: lid });
  } catch (e) { console.error("[/api/auth/users DELETE]", e.message); res.status(500).json({ ok: false, error: "삭제 처리 중 오류" }); }
});

// PATCH /api/auth/users/:id/memo (admin) {memo} — 관리자 메모 저장/수정
app.patch("/api/auth/users/:id/memo", dbRequired, requireAdmin, async (req, res) => {
  const lid = String(req.params.id || "").trim();
  const memo = String((req.body && req.body.memo) || "").slice(0, 500);
  try {
    const u = await findUserRow(lid);
    if (!u) return res.status(404).json({ ok: false, error: "사용자를 찾을 수 없습니다." });
    await pool.query("UPDATE users SET memo = ?, updated_at = NOW() WHERE id = ?", [memo || null, u.id]);
    res.json({ ok: true, userId: lid, memo });
  } catch (e) { console.error("[/api/auth/memo]", e.message); res.status(500).json({ ok: false, error: "메모 저장 중 오류" }); }
});

// PATCH /api/auth/users/:id (admin) {name?, role?, memo?, newPw?} — 사용자 정보 통합 수정.
//   변경할 필드만 전달. newPw 가 비어있지 않으면 비밀번호도 변경(+응답에 echo).
//   마지막 활성 관리자의 역할 강등 차단. 라우트 순서상 /:id/memo 보다 뒤에 둔다.
app.patch("/api/auth/users/:id", dbRequired, requireAdmin, async (req, res) => {
  const lid = String(req.params.id || "").trim();
  const b = req.body || {};
  try {
    const u = await findUserRow(lid);
    if (!u) return res.status(404).json({ ok: false, error: "사용자를 찾을 수 없습니다." });
    const sets = [], vals = [];
    if (b.name !== undefined) {
      const nm = String(b.name).trim();
      if (!nm) return res.status(400).json({ ok: false, error: "이름을 입력하세요.", field: "name" });
      sets.push("display_name = ?"); vals.push(nm.slice(0, 60));
    }
    if (b.role !== undefined) {
      if (!ROLE_SET.has(b.role)) return res.status(400).json({ ok: false, error: "역할을 선택하세요.", field: "role" });
      if ((ADMIN_TIER(b.role) || ADMIN_TIER(u.role)) && req.auth.role !== "superadmin")
        return res.status(403).json({ ok: false, error: "관리자·총관리자 권한 변경은 총관리자만 할 수 있습니다.", field: "role" });
      if (u.role === "admin" && b.role !== "admin") {
        const [[{ n }]] = await pool.query("SELECT COUNT(*) n FROM users WHERE role = 'admin' AND status = 'active'");
        if (n <= 1) return res.status(400).json({ ok: false, error: "마지막 관리자의 역할은 변경할 수 없습니다.", field: "role" });
      }
      sets.push("role = ?"); vals.push(b.role);
    }
    if (b.memo !== undefined) { sets.push("memo = ?"); vals.push(String(b.memo).slice(0, 500) || null); }
    let newPw = null;
    if (b.newPw !== undefined && String(b.newPw) !== "") {
      if (String(b.newPw).length < PW_MIN) return res.status(400).json({ ok: false, error: `비밀번호는 ${PW_MIN}자 이상.`, field: "newPw" });
      newPw = String(b.newPw);
      sets.push("password_hash = ?"); vals.push(await bcrypt.hash(newPw, 10));
    }
    if (!sets.length) return res.json({ ok: true, user: mapUser(u) });
    sets.push("updated_at = NOW()");
    vals.push(u.id);
    await pool.query(`UPDATE users SET ${sets.join(", ")} WHERE id = ?`, vals);
    res.json({ ok: true, user: mapUser(await findUserRow(lid)), newPw: newPw || undefined });
  } catch (e) { console.error("[/api/auth/users PATCH]", e.message); res.status(500).json({ ok: false, error: "수정 처리 중 오류" }); }
});

// PATCH /api/auth/profile (auth) {name}
app.patch("/api/auth/profile", dbRequired, requireAuth, async (req, res) => {
  const name = String((req.body && req.body.name) || "").trim();
  if (!name) return res.status(400).json({ ok: false, error: "이름을 입력하세요.", field: "name" });
  try { await pool.query("UPDATE users SET display_name = ?, updated_at = NOW() WHERE id = ?", [name.slice(0, 60), req.auth.uid]); res.json({ ok: true, user: mapUser(await findUserRow(req.auth.lid)) }); }
  catch (e) { res.status(500).json({ ok: false, error: "프로필 수정 오류" }); }
});

// POST /api/auth/change-password (auth) {currentPw, newPw}
app.post("/api/auth/change-password", dbRequired, requireAuth, async (req, res) => {
  const { currentPw, newPw } = req.body || {};
  if (!newPw || String(newPw).length < PW_MIN) return res.status(400).json({ ok: false, error: `새 비밀번호는 ${PW_MIN}자 이상.`, field: "newPw" });
  try {
    const u = await findUserRow(req.auth.lid);
    if (!u) return res.status(404).json({ ok: false, error: "사용자를 찾을 수 없습니다." });
    if (!(await bcrypt.compare(String(currentPw || ""), u.password_hash))) return res.status(403).json({ ok: false, error: "현재 비밀번호가 일치하지 않습니다.", field: "currentPw" });
    await pool.query("UPDATE users SET password_hash = ?, updated_at = NOW() WHERE id = ?", [await bcrypt.hash(String(newPw), 10), u.id]);
    res.json({ ok: true });
  } catch (e) { res.status(500).json({ ok: false, error: "비밀번호 변경 오류" }); }
});

// ── 공지사항(배너) — 단일 행(id=1). GET 공개, 저장은 관리자 전용 ──
//   배너 텍스트는 프론트에서 텍스트로만 렌더(React 기본 이스케이프) → 저장 XSS 차단.
app.get("/api/announcement", dbRequired, async (_req, res) => {
  try {
    const [rows] = await pool.query("SELECT message, level, active, updated_by, updated_at FROM announcements WHERE id = 1 LIMIT 1");
    const r = rows[0];
    res.json({ ok: true, announcement: r ? {
      message: r.message || "", level: r.level || "info",
      active: !!r.active, updatedBy: r.updated_by || "", updatedAt: r.updated_at,
    } : null });
  } catch (e) { console.error("[/api/announcement GET]", e.message); res.status(500).json({ ok: false, error: "공지 조회 오류" }); }
});

app.post("/api/announcement", dbRequired, requireAdmin, async (req, res) => {
  try {
    const message = String(req.body?.message ?? "").trim().slice(0, 500);
    const level   = ["info", "warn", "critical"].includes(req.body?.level) ? req.body.level : "info";
    const active  = message ? 1 : 0;   // 게시 개념 제거 — 내용 있으면 노출, 비우면 내림
    const by      = String(req.auth?.name || req.auth?.lid || "관리자").slice(0, 60);
    await pool.query(
      `INSERT INTO announcements (id, message, level, active, updated_by) VALUES (1, ?, ?, ?, ?)
       ON DUPLICATE KEY UPDATE message = VALUES(message), level = VALUES(level), active = VALUES(active), updated_by = VALUES(updated_by)`,
      [message, level, active, by]
    );
    res.json({ ok: true });
  } catch (e) { console.error("[/api/announcement POST]", e.message); res.status(500).json({ ok: false, error: "공지 저장 오류" }); }
});

// ── /api/health ──────────────────────────────────────
app.get("/api/health", async (_req, res) => {
  try {
    const r = await fetch(`${OLLAMA_URL}/api/version`);
    const v = r.ok ? await r.json() : null;
    let db = null;
    if (pool) {
      try {
        const [rows] = await pool.query("SELECT COUNT(*) AS rows_ FROM kscg_sensor_data");
        db = { ok: true, sensor_data_rows: rows[0].rows_ };
      } catch (e) {
        db = { ok: false, error: e.message };
      }
    }
    res.json({ ok: true, ollama: v, model: OLLAMA_MODEL, db });
  } catch (err) {
    res.json({ ok: false, error: err.message, model: OLLAMA_MODEL });
  }
});

// ─────────────────────────────────────────────────────
// Tools (Function Calling) — LLM 이 필요시 DB 직접 조회
// ─────────────────────────────────────────────────────

// ── 메모리 캐시 (10초 TTL, LRU 200) ────────────────────
// 자주 호출되는 도구만 캐시. 시계열·점검이력처럼 매번 신선해야 하는 건 제외.
const TOOL_CACHE = new Map();
const TOOL_CACHE_TTL_MS = 10_000;
const TOOL_CACHE_MAX = 200;
const CACHEABLE_TOOLS = new Set([
  "list_devices", "get_device_detail", "get_summary",
  "get_aggregate", "get_zone_summary", "compare_devices",
  "find_devices_by_value", "get_predictions",
  "search_devices_by_location", "geocode_location", "find_devices_near",
  "get_ai_model_info",
  "describe_table",    // 스키마는 잘 안 바뀜 → 캐시 OK
  // execute_safe_sql 은 캐시 X — 매번 동적 SQL 이라 hit 율 낮음 + audit 위해
]);

// SQL 안전장치 정규식 — execute_safe_sql 용
const SAFE_SQL_BLOCKED = /\b(INSERT|UPDATE|DELETE|REPLACE|MERGE|DROP|TRUNCATE|ALTER|CREATE|RENAME|GRANT|REVOKE|SHUTDOWN|FLUSH|RESET|KILL|SET\s+PASSWORD|SOURCE|LOAD\s+DATA|HANDLER|CALL|LOCK\s+TABLES|UNLOCK\s+TABLES|INTO\s+OUTFILE|INTO\s+DUMPFILE|XA\s+|SAVEPOINT|RELEASE\s+SAVEPOINT|COMMIT|ROLLBACK|START\s+TRANSACTION|BEGIN)\b/i;
const SAFE_TABLE_NAME = /^[A-Za-z_][A-Za-z0-9_]*$/;

// Haversine 거리 (km). 두 좌표 사이 대권 거리.
function haversineKm(lat1, lng1, lat2, lng2) {
  const R = 6371;
  const toRad = (d) => (d * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1);
  const dLng = toRad(lng2 - lng1);
  const a = Math.sin(dLat / 2) ** 2
          + Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLng / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}
function toolCacheGet(key) {
  const hit = TOOL_CACHE.get(key);
  if (!hit) return null;
  if (hit.expires < Date.now()) { TOOL_CACHE.delete(key); return null; }
  // LRU: 재접근 시 끝으로 이동 (Map insertion order)
  TOOL_CACHE.delete(key);
  TOOL_CACHE.set(key, hit);
  return hit.value;
}
function toolCacheSet(key, value) {
  if (TOOL_CACHE.size >= TOOL_CACHE_MAX) {
    const oldestKey = TOOL_CACHE.keys().next().value;
    TOOL_CACHE.delete(oldestKey);
  }
  TOOL_CACHE.set(key, { value, expires: Date.now() + TOOL_CACHE_TTL_MS });
}

// ── audit_log INSERT (best-effort, 실패 무시) ──────────
async function logToolCall(name, args, ok, durationMs, _cached) {
  if (!pool) return;
  try {
    await pool.query(
      `INSERT INTO audit_log (action, target_type, target_id, metadata_json) VALUES (?, ?, ?, ?)`,
      ["tool_call", "chat", name, JSON.stringify({ args, ok, durationMs, cached: _cached })],
    );
  } catch (_) { /* swallow — audit 실패해도 진행 */ }
}

// ── 챗봇 영구 대화 저장 (chat_sessions + chat_messages) ──
// 클라이언트가 sessionId 보내면 그것 사용, 없으면 새로 만들어 응답에 동봉.
async function ensureChatSession(sessionId, firstUserMsg, ownerUid = null) {
  if (!pool) return null;
  if (sessionId) {
    try {
      // 숨김(soft-deleted) 세션은 재사용하지 않음 → 새 세션을 만들어 보이게 (안 그러면 메시지가 숨겨진 세션에 쌓여 안 보임)
      const [rows] = await pool.query(`SELECT id, user_id FROM chat_sessions WHERE id = ? AND deleted_at IS NULL`, [sessionId]);
      if (rows.length) {
        // 익명으로 시작한 세션을 로그인 개인 계정이 처음 이어서 쓰면 소유권 승계(익명→로그인 전이)
        if (ownerUid && rows[0].user_id == null) {
          try { await pool.query(`UPDATE chat_sessions SET user_id = ? WHERE id = ? AND user_id IS NULL`, [ownerUid, sessionId]); } catch {}
        }
        return Number(sessionId);
      }
    } catch (_) {}
  }
  try {
    const title = String(firstUserMsg || "").slice(0, 30).trim() || "(제목 없음)";
    const [r] = await pool.query(`INSERT INTO chat_sessions (title, user_id) VALUES (?, ?)`, [title, ownerUid]);
    return r.insertId;
  } catch (e) {
    console.warn("[ensureChatSession]", e.message);
    return null;
  }
}

// 요청자 실제 IP (Cloudflare 터널 통과 시 CF-Connecting-IP). 챗봇 메시지 작성자 기록용.
function reqIp(req) {
  return (req && (req.headers["cf-connecting-ip"] || req.headers["x-forwarded-for"] || req.socket?.remoteAddress)) || null;
}

async function persistMessage(sessionId, role, text, contextJson, tokens, model, ip = null) {
  if (!pool || !sessionId || !text) return;
  try {
    await pool.query(
      `INSERT INTO chat_messages (session_id, role, text, context_json, tokens_prompt, tokens_completion, model, ip) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
      [
        sessionId, role, text,
        contextJson ? JSON.stringify(contextJson) : null,
        tokens?.prompt ?? null,
        tokens?.completion ?? null,
        model || null,
        ip || null,
      ],
    );
    // 세션 updated_at 자동 갱신 (ON UPDATE CURRENT_TIMESTAMP 가 INSERT 만으론 안 트리거됨)
    await pool.query(`UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?`, [sessionId]);
  } catch (e) {
    console.warn("[persistMessage]", e.message);
  }
}

const TOOLS = [
  {
    type: "function",
    function: {
      name: "list_devices",
      description: "군산도시가스 단말기 목록 조회. status/zone 필터 가능. 단말 ID 모를 때 후보 좁히기에 사용.",
      parameters: {
        type: "object",
        properties: {
          status: { type: "string", enum: ["normal","critical","warn","offline","all"], description: "단말 상태" },
          zone:   { type: "string", description: "예: '제1구역', '제8구역'" },
          limit:  { type: "integer", default: 20 }
        }
      }
    }
  },
  {
    type: "function",
    function: {
      name: "get_device_detail",
      description: "특정 단말의 메타(시설번호, 위경도, 설치일) + 8 센서(방식전위/희생전류/AC유입/배터리/온도/습도/충격/RSSI) 최신값 + 통신 상태",
      parameters: {
        type: "object",
        properties: {
          deviceId: { type: "string", description: "예: 'TB24-250401'" }
        },
        required: ["deviceId"]
      }
    }
  },
  {
    type: "function",
    function: {
      name: "get_device_history",
      description: "단말 시계열 추이 (특정 센서, 시간 범위). 추세·변화 분석용.",
      parameters: {
        type: "object",
        properties: {
          deviceId: { type: "string" },
          kind:     { type: "string", enum: ["volt","sacrificial","ac","battery","temp","hum","shock","commDbm"], description: "센서 종류" },
          range:    { type: "string", enum: ["1h","24h","7d","30d"], default: "24h" }
        },
        required: ["deviceId","kind"]
      }
    }
  },
  {
    type: "function",
    function: {
      name: "get_alarms",
      description: "최근 알람 이력. 등급(1=위험,2=경고,3=주의) 필터 가능.",
      parameters: {
        type: "object",
        properties: {
          days:    { type: "integer", default: 7,  description: "최근 N일" },
          gradeId: { type: "integer", description: "1=위험, 2=경고, 3=주의" },
          limit:   { type: "integer", default: 20 }
        }
      }
    }
  },
  {
    type: "function",
    function: {
      name: "get_summary",
      description: "전체 KPI 카운트 (정상/관찰/이상/통신장애 단말 수)",
      parameters: { type: "object", properties: {} }
    }
  },
  {
    type: "function",
    function: {
      name: "get_aggregate",
      description: "전체 단말의 센서값 집계 (평균/최대/최소). 예: 전체 평균 방식전위, 최저 RSSI 등.",
      parameters: {
        type: "object",
        properties: {
          metric: { type: "string", enum: ["volt","sacrificial","ac","temp","hum","battery","commDbm"], description: "측정 종류" },
          op:     { type: "string", enum: ["avg","max","min"], default: "avg" }
        },
        required: ["metric"]
      }
    }
  },
  {
    type: "function",
    function: {
      name: "find_devices_by_value",
      description: "전체 단말 중 특정 센서값이 임계 조건을 만족하는 단말 목록. 예: '방식전위 -800 이상 단말', 'RSSI -80 이하', '온도 30 이상' 등 조건 기반 단말 검색.",
      parameters: {
        type: "object",
        properties: {
          metric:    { type: "string", enum: ["volt","sacrificial","ac","battery","temp","hum","commDbm"] },
          op:        { type: "string", enum: ["gte","lte","eq","gt","lt"], default: "gte", description: "비교 연산자" },
          threshold: { type: "number", description: "임계값" },
          limit:     { type: "integer", default: 20 }
        },
        required: ["metric","threshold"]
      }
    }
  },
  {
    type: "function",
    function: {
      name: "get_zone_summary",
      description: "특정 구역(시설번호 prefix)의 통계: 단말 수, 정상/관찰/이상/통신두절 카운트, 평균 방식전위 + 평균 RSSI.",
      parameters: {
        type: "object",
        properties: {
          zone: { type: "string", description: "예: '제1구역', '제8구역'. 숫자만(1,8)도 가능." }
        },
        required: ["zone"]
      }
    }
  },
  {
    type: "function",
    function: {
      name: "compare_devices",
      description: "여러 단말의 8 센서 최신값을 한꺼번에 조회 (단말 간 비교용). 2~5개 권장.",
      parameters: {
        type: "object",
        properties: {
          deviceIds: { type: "array", items: { type: "string" }, description: "단말 ID 배열. 예: ['TB24-250401','TB24-250402']" }
        },
        required: ["deviceIds"]
      }
    }
  },
  {
    type: "function",
    function: {
      name: "get_recent_changes",
      description: "단말의 최근 N 시간 센서값 변화 (시작/끝/변화량/최저/최고/평균/표준편차/방향). '추세', '변화량', '얼마나 변했나' 질문에 사용.",
      parameters: {
        type: "object",
        properties: {
          deviceId: { type: "string" },
          kind:     { type: "string", enum: ["volt","sacrificial","ac","battery","temp","hum","commDbm"] },
          hours:    { type: "integer", default: 24, description: "최근 N시간 (1~720)" }
        },
        required: ["deviceId","kind"]
      }
    }
  },
  {
    type: "function",
    function: {
      name: "get_maintenance_log",
      description: "현장 점검·정비 이력. 특정 단말 또는 전체. 최근 N일.",
      parameters: {
        type: "object",
        properties: {
          deviceId: { type: "string", description: "단말 ID. 없으면 전체 이력" },
          days:     { type: "integer", default: 30 },
          limit:    { type: "integer", default: 10 }
        }
      }
    }
  },
  {
    type: "function",
    function: {
      name: "get_predictions",
      description: "AI LSTM-AutoEncoder 최신 예측 (MSE, threshold, 위험도, 통신상태, 신뢰도). 단말 ID 없으면 전체 최근 예측.",
      parameters: {
        type: "object",
        properties: {
          deviceId: { type: "string", description: "단말 ID. 없으면 전체 최근 예측" },
          limit:    { type: "integer", default: 10 }
        }
      }
    }
  },
  {
    type: "function",
    function: {
      name: "search_devices_by_location",
      description: "지명/장소/랜드마크 키워드로 단말 검색. POSITION + 시설번호 LIKE 매칭 (한국어 OK). 예: '미룡동', '시청', '버스터미널', '해망동 DM기술'. 공백으로 여러 키워드 AND.",
      parameters: {
        type: "object",
        properties: {
          query: { type: "string", description: "검색 키워드 (공백 구분 다중 가능)" },
          limit: { type: "integer", default: 20 }
        },
        required: ["query"]
      }
    }
  },
  {
    type: "function",
    function: {
      name: "geocode_location",
      description: "지명·랜드마크·주소 → 좌표(lat/lng) 변환. OpenStreetMap Nominatim 사용. 일반 지명(은파호수공원, 군산시청, 군산교도소)에 적합. 군산 우선 검색. 결과 받아서 find_devices_near 로 인근 단말 조회 가능.",
      parameters: {
        type: "object",
        properties: {
          query: { type: "string", description: "지명/랜드마크. 예: '은파호수공원', '군산시청'" }
        },
        required: ["query"]
      }
    }
  },
  {
    type: "function",
    function: {
      name: "find_devices_near",
      description: "특정 좌표 반경 N km 내 단말 검색 (Haversine 거리). 결과는 가까운 순. 군산 범위: lat 35.8~36.0, lng 126.4~126.9. 좌표 모르면 geocode_location 먼저.",
      parameters: {
        type: "object",
        properties: {
          lat:      { type: "number", description: "위도" },
          lng:      { type: "number", description: "경도" },
          radiusKm: { type: "number", default: 2.0, description: "반경 (km). 기본 2km, 좁으면 0.5, 넓으면 5" },
          limit:    { type: "integer", default: 20 }
        },
        required: ["lat", "lng"]
      }
    }
  },
  {
    type: "function",
    function: {
      name: "get_ai_model_info",
      description: "AI 모델 메타 (LSTM AutoEncoder 학습 정보). deviceId 주면 그 단말의 threshold + 분류 기준 반환. 없으면 전체 모델 config + 평가 통계. '이 단말 정상 한계는?', 'AI 모델 어떻게 학습됐어?' 등.",
      parameters: {
        type: "object",
        properties: {
          deviceId: { type: "string", description: "단말 ID. 없으면 전체 모델 메타" }
        }
      }
    }
  },
  {
    type: "function",
    function: {
      name: "execute_safe_sql",
      description: "**자가확장 챗봇 핵심 도구.** 위의 16개 전용 도구로 답할 수 없는 복잡한 분석 질문에 자유 MySQL SELECT 작성. 안전장치: SELECT/WITH 만 허용 (INSERT/UPDATE/DELETE/DDL 차단), LIMIT 자동 1000 강제, 5초 timeout, 다중 statement 차단. siwon DB 모든 테이블 접근 가능 (kscg_* 미러 + 자체 7). 스키마 모르면 describe_table 먼저 호출.",
      parameters: {
        type: "object",
        properties: {
          sql:   { type: "string", description: "MySQL SELECT 또는 WITH 문. 한 statement 만." },
          limit: { type: "integer", default: 100, description: "결과 row 최대 (max 1000). SQL 안에 LIMIT 있으면 그 값 우선." }
        },
        required: ["sql"]
      }
    }
  },
  {
    type: "function",
    function: {
      name: "describe_table",
      description: "테이블 스키마 (컬럼·타입·인덱스) 조회. execute_safe_sql 전 자가 탐색용. siwon DB 모든 테이블 가능. 예: 'kscg_sensor_data', 'audit_log', 'chat_messages', 'kscg_alarm_log'. INFORMATION_SCHEMA 도 가능.",
      parameters: {
        type: "object",
        properties: {
          tableName: { type: "string", description: "테이블 이름 (스키마 접두 X)" }
        },
        required: ["tableName"]
      }
    }
  },
  {
    type: "function",
    function: {
      name: "get_weather_forecast",
      description: "군산(SITE_ID=2 대상 지역) 일별 날씨 예보 — 오늘 포함 최대 7일. Open-Meteo 실시간 예보 API. '일주일 날씨', '주말 날씨', '내일 비 와?' 같은 다일·예보 질문에 사용. 현재 단일 시점 날씨는 이미 시스템 컨텍스트에 있으니 예보가 필요할 때만 호출.",
      parameters: {
        type: "object",
        properties: {
          days: { type: "integer", description: "예보 일수 (1~7, 기본 7). 오늘부터 포함." }
        }
      }
    }
  },
  {
    type: "function",
    function: {
      name: "get_weather_history",
      description: "군산 과거 날씨 조회 — 최대 1년(아카이브는 어제까지 가능). Open-Meteo 과거기상 API(ERA5). '작년 이맘때 날씨', '지난 1년 강수량', '어느 달에 비 많았나' 같은 과거·추세 질문에 사용. 범위가 길면(>45일) 자동으로 월별 집계 + 전체 통계로 반환, 짧으면 일별.",
      parameters: {
        type: "object",
        properties: {
          start_date: { type: "string", description: "시작일 YYYY-MM-DD (생략 시 종료일로부터 1년 전)" },
          end_date: { type: "string", description: "종료일 YYYY-MM-DD (생략 시 어제). 최대 어제까지." }
        }
      }
    }
  },
  {
    type: "function",
    function: {
      name: "web_search",
      description: "웹 검색(DuckDuckGo). 시스템 DB·날씨 도구로 답할 수 없는 일반 지식·개념·외부 정보가 필요할 때만 사용. 사전/개념/엔티티 질의에 강함. 단말/센서/날씨 데이터는 전용 도구를 쓸 것.",
      parameters: {
        type: "object",
        properties: { query: { type: "string", description: "검색어" } },
        required: ["query"]
      }
    }
  }
];

// 웹검색 도구(web_search)는 토글 ON 일 때만 LLM 에 노출 (기본 OFF)
const TOOLS_BASE = TOOLS.filter((t) => t.function?.name !== "web_search");
const toolsFor = (web) => (web ? TOOLS : TOOLS_BASE);

// 단말 seq → SENSOR_ID 찾는 헬퍼
async function findSensorId(transmitterId, seq) {
  const [rows] = await pool.query(`
    SELECT SENSOR_ID, ROW_NUMBER() OVER (PARTITION BY TRANSMITTER_ID ORDER BY SENSOR_ID) AS sq
    FROM kscg_sensor_info WHERE TRANSMITTER_ID = ?
  `, [transmitterId]);
  return rows.find((r) => Number(r.sq) === seq)?.SENSOR_ID || null;
}

// 단말 ID(NAME) → TRANSMITTER_ID
async function getTransmitterIdByName(deviceId) {
  const [rows] = await pool.query(`SELECT TRANSMITTER_ID FROM kscg_transmitter_info WHERE NAME = ?`, [deviceId]);
  return rows[0]?.TRANSMITTER_ID || null;
}

// ── tool dispatchers ────────────────────────────────
// execTool: wrapper — 캐시 hit / 실행 / 캐시 store / audit_log INSERT.
//   캐시 가능한 도구만 캐시 (CACHEABLE_TOOLS).
//   audit_log 는 모든 도구 (best-effort).
//   demoMode: 가상 장비 포함 여부. cache key 에 포함되어 demo/real 응답 분리.
// 보안: 자유 SQL/스키마 노출 도구 비활성화 — URL 아는 누구나 DB 전체 SELECT 가능하던 노출 차단.
//   재활성화: 환경변수 ENABLE_SELF_EXPAND_SQL=1 (기본 OFF).
const DISABLED_TOOLS = process.env.ENABLE_SELF_EXPAND_SQL === "1" ? [] : ["execute_safe_sql", "describe_table"];
for (const _n of DISABLED_TOOLS) {
  const _i = TOOLS.findIndex((t) => t?.function?.name === _n);
  if (_i >= 0) TOOLS.splice(_i, 1);
}

async function execTool(name, args, demoMode = false) {
  args = args || {};
  if (DISABLED_TOOLS.includes(name)) return { error: "이 도구는 보안상 비활성화되어 있습니다." };
  const cacheable = CACHEABLE_TOOLS.has(name);
  const key = cacheable ? `${name}:${demoMode ? "D" : "R"}:${JSON.stringify(args)}` : null;
  if (cacheable) {
    const cached = toolCacheGet(key);
    if (cached) {
      logToolCall(name, args, !cached?.error, 0, true);
      return { ...cached, _cached: true };
    }
  }
  const t0 = Date.now();
  const result = await execToolInternal(name, args, demoMode);
  const dt = Date.now() - t0;
  if (cacheable && !result?.error) toolCacheSet(key, result);
  logToolCall(name, args, !result?.error, dt, false);
  return result;
}

// execToolInternal: 실제 도구 실행. switch dispatcher.
async function execToolInternal(name, args, demoMode = false) {
  if (!pool) return { error: "DB pool 비활성" };
  try {
    switch (name) {
      // 단말 목록
      //   주의: status 필터는 JS 후처리 (DB 컬럼이 아니라 lastSeen + 알람 카운트로 계산).
      //         그래서 SQL 단에서 LIMIT 걸면 안 됨 → 전체 가져온 뒤 filter → slice.
      case "list_devices": {
        const limit = Math.min(Number(args.limit) || 20, 60);
        // 데모 모드 단말 — 실제와 동일 shape 로 prepend (위험·warn·offline 우선 노출 위해)
        const demoRows = demoMode ? getDemoDevices().map((d) => ({
          deviceId: d.deviceId, facility: d.facility, location: d.location,
          deviceStatus: 1, lastSeen: d.lastMeasured, recentAlarms: d.recentAlarms,
          hoursSilent: d.hoursSilent, status: d.status, demo: true,
        })) : [];
        let sql = `
          SELECT t.TRANSMITTER_ID AS txid, t.NAME AS deviceId, f.NUMBER AS facility, f.POSITION AS location,
                 t.DEVICE_STATUS AS deviceStatus,
                 (SELECT MAX(r.DATE)
                  FROM kscg_sensor_info si JOIN kscg_recent_data r ON r.SENSOR_ID = si.SENSOR_ID
                  WHERE si.TRANSMITTER_ID = t.TRANSMITTER_ID) AS lastSeen,
                 (SELECT COUNT(*)
                  FROM kscg_alarm_log a JOIN kscg_sensor_info si ON si.SENSOR_ID = a.SENSOR_ID
                  WHERE si.TRANSMITTER_ID = t.TRANSMITTER_ID
                    AND a.GEN_DATE > DATE_SUB(NOW(), INTERVAL 7 DAY)) AS recentAlarms
          FROM kscg_transmitter_info t
          JOIN kscg_site_mydevice m ON m.TRANSMITTER_ID = t.TRANSMITTER_ID AND m.SITE_ID = ?
          LEFT JOIN kscg_facility_info f ON f.TRANSMITTER_ID = t.TRANSMITTER_ID
        `;
        const where = [];
        const params = [SITE_ID];
        if (args.zone) {
          // zone "제1구역" → facility number "1-XXX" prefix
          const m = String(args.zone).match(/(\d+)/);
          if (m) { where.push(`f.NUMBER LIKE ?`); params.push(`${m[1]}-%`); }
        }
        if (where.length) sql += ` WHERE ${where.join(" AND ")}`;
        sql += ` ORDER BY t.TRANSMITTER_ID`;     // ★ LIMIT 제거 (필터 전에 자르면 X)
        const [rows] = await pool.query(sql, params);
        const aiMap = await loadLatestAi();
        const now = Date.now();
        const annotated = rows.map((r) => {
          const hoursSilent = r.lastSeen ? Math.floor((now - new Date(r.lastSeen).getTime()) / 3600000) : null;
          const recentAlarms = Number(r.recentAlarms) || 0;
          const ai = aiMap.get(r.txid);
          // 통합 mapStatus (24h두절→offline / 알람 OR AI이상≥5×→critical / AI이상·관찰→warn)
          const status = mapStatus(r.deviceStatus, hoursSilent, recentAlarms, ai);
          return { ...r, hoursSilent, recentAlarms, status,
                   aiRisk: ai ? ai.risk : null, aiRatio: aiRatioOf(ai) };
        });
        // 데모 단말은 status 가 이미 fixed → annotated 와 합쳐서 필터
        const combined = [...demoRows, ...annotated];
        const filtered = combined.filter((r) => !args.status || args.status === "all" || r.status === args.status);
        return {
          totalScanned: combined.length,
          count: Math.min(filtered.length, limit),
          devices: filtered.slice(0, limit),
          demoMode,
        };
      }

      // 단말 상세 (8 센서 최신값)
      case "get_device_detail": {
        const deviceId = args.deviceId;
        if (!deviceId) return { error: "deviceId 필수" };
        // 데모 단말 우선 룩업
        if (demoMode && deviceId.startsWith("DEMO-")) {
          const d = findDemoDevice(deviceId);
          if (!d) return { error: `데모 단말 없음: ${deviceId}` };
          return {
            deviceId: d.deviceId, serial: `DEMO-SER-${Math.abs(d.txid)}`, installDate: null,
            deviceStatus: 1, periodSec: 3600,
            facility: d.facility, location: d.location, lat: d.lat, lng: d.lng,
            zone: zoneFromFacility(d.facility),
            sensors: d.sensors,
            sensorJudgement: buildSensorJudgement(d.sensors, MODEL_CONFIG?.sacrificial_devices?.includes(d.deviceId) || false),
            lastMeasured: d.lastMeasured,
            hoursSilent: d.hoursSilent,
            status: d.status,
            mse: d.mse, threshold: d.threshold, riskLevel: d.riskLevel, aiReliability: d.aiReliability,
            aiJudgement: classifyAiPrediction({ mse: d.mse, threshold: d.threshold }),
            demo: true,
          };
        }
        const txid = await getTransmitterIdByName(deviceId);
        if (!txid) return { error: `단말 없음: ${deviceId}` };
        const [meta] = await pool.query(`
          SELECT t.NAME AS deviceId, t.SERIAL_NUM AS serial, t.INSTALL_DATE AS installDate,
                 t.DEVICE_STATUS AS deviceStatus, t.PERIOD_SEC AS periodSec,
                 f.NUMBER AS facility, f.POSITION AS location, f.LATITUDE AS lat, f.LONGITUDE AS lng
          FROM kscg_transmitter_info t
          LEFT JOIN kscg_facility_info f ON f.TRANSMITTER_ID = t.TRANSMITTER_ID
          WHERE t.TRANSMITTER_ID = ?
        `, [txid]);
        const [sensors] = await pool.query(`
          SELECT si.UNIT, r.DATE AS measuredAt, r.VALUE AS value,
                 ROW_NUMBER() OVER (PARTITION BY si.TRANSMITTER_ID ORDER BY si.SENSOR_ID) AS seq
          FROM kscg_sensor_info si
          LEFT JOIN kscg_recent_data r ON r.SENSOR_ID = si.SENSOR_ID
          WHERE si.TRANSMITTER_ID = ?
          ORDER BY si.SENSOR_ID
        `, [txid]);
        const sensorVals = {};
        let lastMeasured = null;
        for (const s of sensors) {
          const kind = SENSOR_SEQ_KIND[s.seq - 1];
          if (kind) sensorVals[kind] = s.value;
          if (s.measuredAt && (!lastMeasured || s.measuredAt > lastMeasured)) lastMeasured = s.measuredAt;
        }
        const hoursSilent = lastMeasured ? Math.floor((Date.now() - new Date(lastMeasured).getTime()) / 3600000) : null;
        // 최근 7일 알람 수 (통합 mapStatus 용)
        const [[almCnt]] = await pool.query(`
          SELECT COUNT(*) AS cnt
          FROM kscg_alarm_log a
          JOIN kscg_sensor_info si ON si.SENSOR_ID = a.SENSOR_ID
          WHERE si.TRANSMITTER_ID = ? AND a.GEN_DATE > DATE_SUB(NOW(), INTERVAL 7 DAY)
        `, [txid]);
        const recentAlarms = Number(almCnt?.cnt) || 0;
        // 최신 AI 예측 (ai_predictions) — 통합 mapStatus + 상세 필드
        const aiRow = (await loadLatestAi()).get(txid) || null;
        const status = mapStatus(meta[0]?.deviceStatus, hoursSilent, recentAlarms, aiRow);
        // 학습 threshold (정적, 참고용) — 실측 AI 예측과 별개
        const aiTh = DEVICE_THRESHOLDS[deviceId];
        const ai = aiTh != null ? {
          threshold: aiTh,
          threshold70: Number((aiTh * 0.7).toFixed(6)),
          threshold100: Number(aiTh.toFixed(6)),
          isSacrificial: MODEL_CONFIG?.sacrificial_devices?.includes(deviceId) || false,
        } : null;
        return {
          ...meta[0],
          zone: zoneFromFacility(meta[0]?.facility),
          sensors: sensorVals,
          lastMeasured,
          hoursSilent,
          recentAlarms,
          status,
          sensorJudgement: buildSensorJudgement(sensorVals, MODEL_CONFIG?.sacrificial_devices?.includes(deviceId) || false),
          ai,
          // 최신 LSTM 예측 (ai_predictions 기반)
          aiRisk:        aiRow ? aiRow.risk : null,
          aiMse:         aiRow && aiRow.mse != null ? Number(aiRow.mse) : null,
          aiThreshold:   aiRow && aiRow.threshold != null ? Number(aiRow.threshold) : null,
          aiRatio:       aiRatioOf(aiRow),
          aiJudgement:   classifyAiPrediction(aiRow),
          aiReliability: aiRow ? aiRow.aiReliability : null,
          commStatus:    aiRow ? aiRow.commStatus : null,
          featureContributions: aiRow ? contribFromFeatures(aiRow.featureContributions) : [],
          predictedAt:   aiRow ? aiRow.predictedAt : null,
        };
      }

      // 단말 시계열
      case "get_device_history": {
        const deviceId = args.deviceId;
        const kind     = args.kind || "volt";
        const range    = args.range || "24h";
        const seq      = SENSOR_SEQ_KIND.indexOf(kind) + 1;
        if (seq < 1) return { error: `unknown kind: ${kind}` };
        // 데모 단말 — seeded mock 시계열
        if (demoMode && deviceId.startsWith("DEMO-")) {
          const hours = range === "1h" ? 1 : range === "7d" ? 168 : range === "30d" ? 720 : 24;
          const points = generateDemoHistory(deviceId, kind, hours);
          const latest = points[points.length - 1]?.v;
          return {
            deviceId, kind, range, count: points.length, sampled: points.length, points,
            latestValue: latest ?? null,
            latestJudgement: sensorJudgementForKind(kind, latest, false),
            demo: true,
          };
        }
        const txid = await getTransmitterIdByName(deviceId);
        if (!txid) return { error: `단말 없음: ${deviceId}` };
        const sensorId = await findSensorId(txid, seq);
        if (!sensorId) return { error: "센서 매핑 없음" };
        const hours = range === "1h" ? 1 : range === "7d" ? 168 : range === "30d" ? 720 : 24;
        const [rows] = await pool.query(`
          SELECT WRITE_DATE AS t, VALUE AS v
          FROM kscg_sensor_data
          WHERE SENSOR_ID = ? AND WRITE_DATE > DATE_SUB(NOW(), INTERVAL ? HOUR)
          ORDER BY WRITE_DATE
        `, [sensorId, hours]);
        // 30d 처럼 큰 범위는 샘플링 (집계)
        let points = rows;
        if (rows.length > 100) {
          const step = Math.ceil(rows.length / 100);
          points = rows.filter((_, i) => i % step === 0);
        }
        const latest = rows[rows.length - 1]?.v;
        return {
          deviceId, kind, range, count: rows.length, sampled: points.length, points,
          latestValue: latest ?? null,
          latestJudgement: sensorJudgementForKind(kind, latest, MODEL_CONFIG?.sacrificial_devices?.includes(deviceId) || false),
        };
      }

      // 알람
      case "get_alarms": {
        const days = Number(args.days) || 7;
        const limit = Math.min(Number(args.limit) || 20, 50);
        const where = ["a.GEN_DATE > DATE_SUB(NOW(), INTERVAL ? DAY)"];
        const params = [days];
        if (args.gradeId) { where.push("a.GRADE_ID = ?"); params.push(Number(args.gradeId)); }
        const [rows] = await pool.query(`
          SELECT a.GEN_DATE AS occurredAt, g.GRADE_TEXT AS grade,
                 t.NAME AS deviceId, f.NUMBER AS facility,
                 a.VALUE AS value, a.CONTENTS AS contents
          FROM kscg_alarm_log a
          LEFT JOIN kscg_alarm_grade_info g ON g.GRADE_ID = a.GRADE_ID
          LEFT JOIN kscg_sensor_info si ON si.SENSOR_ID = a.SENSOR_ID
          LEFT JOIN kscg_transmitter_info t ON t.TRANSMITTER_ID = si.TRANSMITTER_ID
          LEFT JOIN kscg_facility_info f ON f.TRANSMITTER_ID = t.TRANSMITTER_ID
          WHERE ${where.join(" AND ")}
          ORDER BY a.GEN_DATE DESC LIMIT ${limit}
        `, params);
        // 데모 알람 prepend (최근순)
        const demoAlarms = demoMode
          ? getDemoAlarms().filter((a) => !args.gradeId || a.gradeId === Number(args.gradeId)).slice(0, limit)
          : [];
        const merged = [...demoAlarms, ...rows].slice(0, limit);
        return { count: merged.length, alarms: merged, demoMode };
      }

      // KPI 카운트
      case "get_summary": {
        // /api/summary 와 동일 기준 — 단말별 (통신두절·7일알람·최신AI) → mapStatus 집계
        const [silentRows] = await pool.query(`
          SELECT t.TRANSMITTER_ID AS txid,
                 TIMESTAMPDIFF(HOUR, MAX(r.DATE), NOW()) AS hoursSilent
          FROM kscg_transmitter_info t
          JOIN kscg_site_mydevice m ON m.TRANSMITTER_ID = t.TRANSMITTER_ID AND m.SITE_ID = ?
          LEFT JOIN kscg_sensor_info si ON si.TRANSMITTER_ID = t.TRANSMITTER_ID
          LEFT JOIN kscg_recent_data r  ON r.SENSOR_ID = si.SENSOR_ID
          GROUP BY t.TRANSMITTER_ID
        `, [SITE_ID]);
        const [almRows] = await pool.query(`
          SELECT si.TRANSMITTER_ID AS txid, COUNT(*) AS cnt
          FROM kscg_alarm_log a
          JOIN kscg_sensor_info si ON si.SENSOR_ID = a.SENSOR_ID
          JOIN kscg_site_mydevice m ON m.TRANSMITTER_ID = si.TRANSMITTER_ID AND m.SITE_ID = ?
          WHERE a.GEN_DATE > DATE_SUB(NOW(), INTERVAL 7 DAY)
          GROUP BY si.TRANSMITTER_ID
        `, [SITE_ID]);
        const almMap = new Map(almRows.map((r) => [r.txid, Number(r.cnt)]));
        const aiMap  = await loadLatestAi();
        let all = silentRows.length, offline = 0, critical = 0, warn = 0, normal = 0;
        for (const dRow of silentRows) {
          const hs = dRow.hoursSilent == null ? null : Number(dRow.hoursSilent);
          const st = mapStatus(null, hs, almMap.get(dRow.txid) || 0, aiMap.get(dRow.txid));
          if (st === "offline") offline++;
          else if (st === "critical") critical++;
          else if (st === "warn") warn++;
          else normal++;
        }
        if (demoMode) {
          const d = getDemoDevices();
          const dC = d.filter((x) => x.status === "critical").length;
          const dW = d.filter((x) => x.status === "warn").length;
          const dO = d.filter((x) => x.status === "offline").length;
          return {
            total: all + d.length,
            normal, critical: critical + dC, warn: warn + dW, offline: offline + dO,
            demoMode: true,
          };
        }
        return { total: all, normal, critical, warn, offline };
      }

      // 집계 (평균·최대·최소)
      case "get_aggregate": {
        const metric = args.metric;
        const op = args.op || "avg";
        const seq = SENSOR_SEQ_KIND.indexOf(metric) + 1;
        if (seq < 1) return { error: `unknown metric: ${metric}` };
        const opSql = { avg: "AVG", max: "MAX", min: "MIN" }[op] || "AVG";
        // RECENT_DATA 의 모든 군산 단말 seq 번째 센서값 집계
        const [rows] = await pool.query(`
          SELECT ${opSql}(r.VALUE) AS result
          FROM kscg_recent_data r
          JOIN kscg_sensor_info si ON si.SENSOR_ID = r.SENSOR_ID
          JOIN kscg_site_mydevice m ON m.TRANSMITTER_ID = si.TRANSMITTER_ID AND m.SITE_ID = ?
          WHERE (
            SELECT COUNT(*) FROM kscg_sensor_info si2
            WHERE si2.TRANSMITTER_ID = si.TRANSMITTER_ID AND si2.SENSOR_ID <= si.SENSOR_ID
          ) = ?
        `, [SITE_ID, seq]);
        const v = rows[0].result;
        // 데모: DEMO_DEVICES sensors[metric] 합쳐서 재계산
        if (demoMode) {
          const demoVals = getDemoDevices().map((d) => d.sensors[metric]).filter((x) => x != null && isFinite(x));
          if (demoVals.length > 0) {
            const allVals = v != null ? [Number(v)].concat(demoVals) : demoVals;
            // op 별 재계산 (실 v 는 이미 op 적용된 값이지만, DEMO 합쳐서 단순 op)
            const merged = { avg: allVals.reduce((a, b) => a + b, 0) / allVals.length,
                             max: Math.max(...allVals), min: Math.min(...allVals) }[op];
            return { metric, op, result: Number(merged.toFixed(2)), demoMode: true };
          }
        }
        return { metric, op, result: v != null ? Number(v.toFixed(2)) : null };
      }

      // 조건 만족 단말 검색 ("방식전위 -800 이상" 류)
      case "find_devices_by_value": {
        const metric = args.metric;
        const op = args.op || "gte";
        const threshold = Number(args.threshold);
        const limit = Math.min(Number(args.limit) || 20, 60);
        const seq = SENSOR_SEQ_KIND.indexOf(metric) + 1;
        if (seq < 1) return { error: `unknown metric: ${metric}` };
        if (!isFinite(threshold)) return { error: "threshold 가 숫자가 아닙니다" };
        const opSql = ({ gte:">=", lte:"<=", eq:"=", gt:">", lt:"<" })[op] || ">=";
        const orderAsc = (op === "lte" || op === "lt") ? "ASC" : "DESC";
        const [rows] = await pool.query(`
          SELECT t.NAME AS deviceId, f.NUMBER AS facility, f.POSITION AS location,
                 r.VALUE AS value, r.DATE AS measuredAt
          FROM kscg_recent_data r
          JOIN kscg_sensor_info si ON si.SENSOR_ID = r.SENSOR_ID
          JOIN kscg_transmitter_info t ON t.TRANSMITTER_ID = si.TRANSMITTER_ID
          JOIN kscg_site_mydevice m ON m.TRANSMITTER_ID = si.TRANSMITTER_ID AND m.SITE_ID = ?
          LEFT JOIN kscg_facility_info f ON f.TRANSMITTER_ID = t.TRANSMITTER_ID
          WHERE (
            SELECT COUNT(*) FROM kscg_sensor_info si2
            WHERE si2.TRANSMITTER_ID = si.TRANSMITTER_ID AND si2.SENSOR_ID <= si.SENSOR_ID
          ) = ?
            AND r.VALUE ${opSql} ?
          ORDER BY r.VALUE ${orderAsc}
          LIMIT ?
        `, [SITE_ID, seq, threshold, limit]);
        // 데모 단말도 조건 매칭
        const cmp = { gte: (a,b)=>a>=b, lte: (a,b)=>a<=b, eq: (a,b)=>a===b, gt: (a,b)=>a>b, lt: (a,b)=>a<b }[op] || ((a,b)=>a>=b);
        const demoMatches = demoMode
          ? getDemoDevices()
              .filter((d) => d.sensors[metric] != null && cmp(d.sensors[metric], threshold))
              .map((d) => ({ deviceId: d.deviceId, facility: d.facility, zone: zoneFromFacility(d.facility),
                location: d.location, value: Number(Number(d.sensors[metric]).toFixed(2)), measuredAt: d.lastMeasured, demo: true }))
          : [];
        const realMatches = rows.map((r) => ({
          deviceId: r.deviceId, facility: r.facility, zone: zoneFromFacility(r.facility),
          location: r.location, value: r.value != null ? Number(Number(r.value).toFixed(2)) : null,
          measuredAt: r.measuredAt,
        }));
        const all = [...demoMatches, ...realMatches]
          .sort((a, b) => (op === "lte" || op === "lt") ? a.value - b.value : b.value - a.value)
          .slice(0, limit);
        return { metric, op, threshold, count: all.length, devices: all, demoMode };
      }

      // 구역 요약
      case "get_zone_summary": {
        const zoneInput = String(args.zone || "");
        const m = zoneInput.match(/(\d+)/);
        if (!m) return { error: `구역 인식 실패: ${zoneInput}` };
        const zoneNum = m[1];
        const [rows] = await pool.query(`
          SELECT t.TRANSMITTER_ID, t.NAME AS deviceId, f.NUMBER AS facility, f.POSITION AS location,
                 (SELECT MAX(r.DATE) FROM kscg_sensor_info si JOIN kscg_recent_data r ON r.SENSOR_ID = si.SENSOR_ID WHERE si.TRANSMITTER_ID = t.TRANSMITTER_ID) AS lastSeen,
                 (SELECT COUNT(*) FROM kscg_alarm_log a JOIN kscg_sensor_info si ON si.SENSOR_ID = a.SENSOR_ID WHERE si.TRANSMITTER_ID = t.TRANSMITTER_ID AND a.GEN_DATE > DATE_SUB(NOW(), INTERVAL 7 DAY)) AS recentAlarms
          FROM kscg_transmitter_info t
          JOIN kscg_site_mydevice mm ON mm.TRANSMITTER_ID = t.TRANSMITTER_ID AND mm.SITE_ID = ?
          LEFT JOIN kscg_facility_info f ON f.TRANSMITTER_ID = t.TRANSMITTER_ID
          WHERE f.NUMBER LIKE ?
        `, [SITE_ID, `${zoneNum}-%`]);

        if (rows.length === 0) {
          return { zone: `제${zoneNum}구역`, count: 0, message: "해당 구역 단말 없음" };
        }

        const now = Date.now();
        let normal = 0, critical = 0, offline = 0;
        for (const r of rows) {
          const hours = r.lastSeen ? Math.floor((now - new Date(r.lastSeen).getTime()) / 3600000) : null;
          if (hours != null && hours >= 24) offline++;
          else if (Number(r.recentAlarms) > 0) critical++;
          else normal++;
        }

        const txids = rows.map((r) => r.TRANSMITTER_ID);
        const [[volt]] = await pool.query(`
          SELECT AVG(r.VALUE) AS avg
          FROM kscg_recent_data r
          JOIN kscg_sensor_info si ON si.SENSOR_ID = r.SENSOR_ID
          WHERE si.TRANSMITTER_ID IN (?)
            AND (SELECT COUNT(*) FROM kscg_sensor_info si2 WHERE si2.TRANSMITTER_ID = si.TRANSMITTER_ID AND si2.SENSOR_ID <= si.SENSOR_ID) = 1
        `, [txids]);
        const [[rssi]] = await pool.query(`
          SELECT AVG(r.VALUE) AS avg
          FROM kscg_recent_data r
          JOIN kscg_sensor_info si ON si.SENSOR_ID = r.SENSOR_ID
          WHERE si.TRANSMITTER_ID IN (?)
            AND (SELECT COUNT(*) FROM kscg_sensor_info si2 WHERE si2.TRANSMITTER_ID = si.TRANSMITTER_ID AND si2.SENSOR_ID <= si.SENSOR_ID) = 8
        `, [txids]);

        // 데모 단말 — facility prefix 의 첫 숫자가 zoneNum 와 일치하면 추가
        let dNormal = 0, dCrit = 0, dOff = 0, dVolts = [], dRssi = [];
        if (demoMode) {
          for (const d of getDemoDevices()) {
            const m2 = String(d.facility).match(/^(\d+)/);
            if (!m2 || m2[1] !== zoneNum) continue;
            if (d.status === "offline") dOff++;
            else if (d.status === "critical") dCrit++;
            else dNormal++;
            if (d.sensors.volt    != null) dVolts.push(d.sensors.volt);
            if (d.sensors.commDbm != null) dRssi.push(d.sensors.commDbm);
          }
        }
        const realVolt = volt.avg != null ? Number(volt.avg) : null;
        const realRssi = rssi.avg != null ? Number(rssi.avg) : null;
        const avgVolt = dVolts.length || realVolt != null
          ? Number(((realVolt != null ? realVolt * rows.length : 0) + dVolts.reduce((a,b)=>a+b,0)) / (rows.length + dVolts.length || 1)).toFixed(2)
          : null;
        const avgRssi = dRssi.length || realRssi != null
          ? Number(((realRssi != null ? realRssi * rows.length : 0) + dRssi.reduce((a,b)=>a+b,0)) / (rows.length + dRssi.length || 1)).toFixed(2)
          : null;
        return {
          zone: `제${zoneNum}구역`,
          count: rows.length + dNormal + dCrit + dOff,
          normal: normal + dNormal, critical: critical + dCrit, offline: offline + dOff,
          avgVolt: avgVolt != null ? Number(avgVolt) : null,
          avgRssi: avgRssi != null ? Number(avgRssi) : null,
          devices: rows.slice(0, 10).map((r) => r.deviceId),    // 미리보기 10대만
          demoMode,
        };
      }

      // 다중 단말 비교
      case "compare_devices": {
        const ids = Array.isArray(args.deviceIds) ? args.deviceIds.slice(0, 5) : [];
        if (ids.length === 0) return { error: "deviceIds (배열) 필수" };
        const results = [];
        for (const id of ids) {
          // 데모 단말 우선 룩업
          if (demoMode && id.startsWith("DEMO-")) {
            const d = findDemoDevice(id);
            if (d) { results.push({ deviceId: id, sensors: d.sensors, lastMeasured: d.lastMeasured, demo: true }); continue; }
          }
          const txid = await getTransmitterIdByName(id);
          if (!txid) { results.push({ deviceId: id, error: "단말 없음" }); continue; }
          const [sensors] = await pool.query(`
            SELECT r.DATE AS measuredAt, r.VALUE AS value,
                   ROW_NUMBER() OVER (PARTITION BY si.TRANSMITTER_ID ORDER BY si.SENSOR_ID) AS seq
            FROM kscg_sensor_info si
            LEFT JOIN kscg_recent_data r ON r.SENSOR_ID = si.SENSOR_ID
            WHERE si.TRANSMITTER_ID = ?
            ORDER BY si.SENSOR_ID
          `, [txid]);
          const sensorVals = {};
          let lastMeasured = null;
          for (const s of sensors) {
            const kind = SENSOR_SEQ_KIND[s.seq - 1];
            if (kind) sensorVals[kind] = s.value;
            if (s.measuredAt && (!lastMeasured || s.measuredAt > lastMeasured)) lastMeasured = s.measuredAt;
          }
          results.push({ deviceId: id, sensors: sensorVals, lastMeasured });
        }
        return { count: results.length, devices: results, demoMode };
      }

      // 최근 N시간 변화 통계
      case "get_recent_changes": {
        const deviceId = args.deviceId;
        const kind = args.kind;
        const hours = Math.min(Math.max(Number(args.hours) || 24, 1), 720);
        const seq = SENSOR_SEQ_KIND.indexOf(kind) + 1;
        if (seq < 1) return { error: `unknown kind: ${kind}` };
        // 데모 단말 — seeded mock 시계열 통계
        if (demoMode && deviceId.startsWith("DEMO-")) {
          const points = generateDemoHistory(deviceId, kind, hours);
          const vals = points.map((p) => p.v);
          const first = vals[0], last = vals[vals.length - 1];
          const mn = Math.min(...vals), mx = Math.max(...vals);
          const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
          const std = Math.sqrt(vals.reduce((s, v) => s + (v - mean) ** 2, 0) / vals.length);
          const delta = last - first, pct = first !== 0 ? (delta / Math.abs(first)) * 100 : 0;
          return {
            deviceId, kind, hours, count: vals.length,
            start: Number(first.toFixed(2)), end: Number(last.toFixed(2)),
            delta: Number(delta.toFixed(2)), percentChange: Number(pct.toFixed(1)),
            min: Number(mn.toFixed(2)), max: Number(mx.toFixed(2)),
            mean: Number(mean.toFixed(2)), std: Number(std.toFixed(2)),
            direction: delta > 0.01 ? "상승" : delta < -0.01 ? "하락" : "평탄",
            startJudgement: sensorJudgementForKind(kind, first, false),
            endJudgement: sensorJudgementForKind(kind, last, false),
            demo: true,
          };
        }
        const txid = await getTransmitterIdByName(deviceId);
        if (!txid) return { error: `단말 없음: ${deviceId}` };
        const sensorId = await findSensorId(txid, seq);
        if (!sensorId) return { error: "센서 매핑 없음" };
        const [rows] = await pool.query(`
          SELECT VALUE AS v
          FROM kscg_sensor_data
          WHERE SENSOR_ID = ? AND WRITE_DATE > DATE_SUB(NOW(), INTERVAL ? HOUR)
          ORDER BY WRITE_DATE
        `, [sensorId, hours]);
        if (rows.length === 0) return { deviceId, kind, hours, count: 0, message: "해당 기간 데이터 없음" };
        const values = rows.map((r) => r.v).filter((v) => v != null && isFinite(v));
        if (values.length === 0) return { deviceId, kind, hours, count: 0, message: "유효 데이터 없음" };
        const first = values[0];
        const last  = values[values.length - 1];
        const min   = Math.min(...values);
        const max   = Math.max(...values);
        const mean  = values.reduce((a, b) => a + b, 0) / values.length;
        const std   = Math.sqrt(values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length);
        const delta = last - first;
        const pct   = first !== 0 ? (delta / Math.abs(first)) * 100 : 0;
        return {
          deviceId, kind, hours,
          count: values.length,
          start: Number(first.toFixed(2)),
          end:   Number(last.toFixed(2)),
          delta: Number(delta.toFixed(2)),
          percentChange: Number(pct.toFixed(1)),
          min:   Number(min.toFixed(2)),
          max:   Number(max.toFixed(2)),
          mean:  Number(mean.toFixed(2)),
          std:   Number(std.toFixed(2)),
          direction: delta > 0.01 ? "상승" : delta < -0.01 ? "하락" : "평탄",
          startJudgement: sensorJudgementForKind(kind, first, MODEL_CONFIG?.sacrificial_devices?.includes(deviceId) || false),
          endJudgement: sensorJudgementForKind(kind, last, MODEL_CONFIG?.sacrificial_devices?.includes(deviceId) || false),
        };
      }

      // 점검·정비 이력
      case "get_maintenance_log": {
        const days = Math.min(Math.max(Number(args.days) || 30, 1), 365);
        const limit = Math.min(Number(args.limit) || 10, 50);
        const where = ["ml.DATE > DATE_SUB(NOW(), INTERVAL ? DAY)"];
        const params = [days];
        if (args.deviceId) {
          const txid = await getTransmitterIdByName(args.deviceId);
          if (!txid) return { error: `단말 없음: ${args.deviceId}` };
          where.push("ml.TRANSMITTER_ID = ?");
          params.push(txid);
        }
        const [rows] = await pool.query(`
          SELECT ml.DATE AS occurredAt, t.NAME AS deviceId, mt.TYPE_TEXT AS type,
                 ml.DESCRIPTION AS description, ml.USER_ID AS userId
          FROM kscg_maintenance_log ml
          LEFT JOIN kscg_transmitter_info t ON t.TRANSMITTER_ID = ml.TRANSMITTER_ID
          LEFT JOIN kscg_maintenance_type_info mt ON mt.TYPE_ID = ml.TYPE
          WHERE ${where.join(" AND ")}
          ORDER BY ml.DATE DESC LIMIT ${limit}
        `, params);
        return { count: rows.length, logs: rows };
      }

      // AI 예측 (LSTM AutoEncoder 결과)
      case "get_predictions": {
        const limit = Math.min(Number(args.limit) || 10, 50);
        let sql = `
          SELECT p.transmitter_id AS txid, t.NAME AS deviceId,
                 p.predicted_at AS predictedAt, p.mse, p.threshold,
                 p.risk_level AS riskLevel, p.comm_status AS commStatus,
                 p.ai_reliability AS aiReliability,
                 p.is_sacrificial_device AS isSacrificial
          FROM ai_predictions p
          LEFT JOIN kscg_transmitter_info t ON t.TRANSMITTER_ID = p.transmitter_id
        `;
        const params = [];
        if (args.deviceId) {
          const txid = await getTransmitterIdByName(args.deviceId);
          if (!txid) return { error: `단말 없음: ${args.deviceId}` };
          sql += " WHERE p.transmitter_id = ?";
          params.push(txid);
        }
        sql += ` ORDER BY p.predicted_at DESC LIMIT ${limit}`;
        const [rows] = await pool.query(sql, params);
        if (rows.length === 0) {
          // ai_predictions 비어있을 때 device_thresholds.json 정보로 fallback
          const deviceId = args.deviceId;
          if (deviceId && DEVICE_THRESHOLDS[deviceId] != null) {
            const th = DEVICE_THRESHOLDS[deviceId];
            return {
              count: 0,
              stub: true,
              deviceId,
              threshold: th,
              threshold70: Number((th * 0.7).toFixed(6)),
              threshold100: Number(th.toFixed(6)),
              message: `LSTM 실시간 예측 MSE 는 아직 INSERT 안 됨. 그러나 학습 시점 threshold 는 알려진 값: ${th.toExponential(3)} (정상 한계). 실측 MSE 가 ${(th*0.7).toExponential(2)} 미만이면 정상, ${(th*0.7).toExponential(2)}~${th.toExponential(2)} 이면 관찰, 초과면 이상.`,
            };
          }
          return {
            count: 0,
            stub: true,
            message: "AI 예측 데이터 없음 (이두현 LSTM 백엔드 INSERT 대기). 단말별 threshold 는 get_ai_model_info(deviceId) 로 조회 가능.",
            thresholdsAvailable: Object.keys(DEVICE_THRESHOLDS).length,
          };
        }
        // 실데이터 있을 때 — classifyMse 자동 적용해서 level 추가
        const enriched = rows.map((r) => ({
          ...r,
          classification: r.mse != null && r.deviceId ? classifyMse(r.deviceId, r.mse) : null,
          aiJudgement: classifyAiPrediction(r),
        }));
        // 데모 단말 예측 합치기
        if (demoMode) {
          const demoPreds = getDemoDevices()
            .filter((d) => !args.deviceId || d.deviceId === args.deviceId)
            .map((d) => ({
              txid: d.txid, deviceId: d.deviceId, predictedAt: d.lastMeasured,
              mse: d.mse, threshold: d.threshold,
              riskLevel: d.riskLevel, commStatus: d.status === "offline" ? "통신고장" : "정상통신",
              aiReliability: d.aiReliability, isSacrificial: 0,
              classification: { level: d.riskLevel, ratio: d.mse != null ? Number((d.mse / d.threshold).toFixed(3)) : null },
              aiJudgement: classifyAiPrediction({ mse: d.mse, threshold: d.threshold }),
              demo: true,
            }));
          return { count: enriched.length + demoPreds.length, predictions: [...demoPreds, ...enriched], demoMode };
        }
        return { count: enriched.length, predictions: enriched };
      }

      // 위치/지명 키워드 검색 (POSITION + facility number LIKE)
      case "search_devices_by_location": {
        const query = String(args.query || "").trim();
        if (!query) return { error: "query 필수" };
        const limit = Math.min(Number(args.limit) || 20, 60);
        // 공백으로 분리한 토큰 각각 AND 매칭 (POSITION OR facility number)
        const tokens = query.split(/\s+/).filter(Boolean).slice(0, 5);
        const conds = tokens.map(() => "(f.POSITION LIKE ? OR f.NUMBER LIKE ?)").join(" AND ");
        const params = [SITE_ID];
        for (const t of tokens) { params.push(`%${t}%`, `%${t}%`); }
        const sql = `
          SELECT t.NAME AS deviceId, f.NUMBER AS facility, f.POSITION AS location,
                 f.LATITUDE AS lat, f.LONGITUDE AS lng,
                 (SELECT MAX(r.DATE) FROM kscg_sensor_info si JOIN kscg_recent_data r ON r.SENSOR_ID = si.SENSOR_ID WHERE si.TRANSMITTER_ID = t.TRANSMITTER_ID) AS lastSeen
          FROM kscg_transmitter_info t
          JOIN kscg_site_mydevice mm ON mm.TRANSMITTER_ID = t.TRANSMITTER_ID AND mm.SITE_ID = ?
          LEFT JOIN kscg_facility_info f ON f.TRANSMITTER_ID = t.TRANSMITTER_ID
          ${conds ? `WHERE ${conds}` : ""}
          ORDER BY t.TRANSMITTER_ID
          LIMIT ${limit}
        `;
        const [rows] = await pool.query(sql, params);
        const realDevs = rows.map((r) => ({
          deviceId: r.deviceId, facility: r.facility, zone: zoneFromFacility(r.facility),
          location: r.location, lat: r.lat, lng: r.lng, lastSeen: r.lastSeen,
        }));
        // 데모: location/facility/deviceId LIKE 매칭
        const demoMatches = demoMode
          ? getDemoDevices()
              .filter((d) => tokens.every((tk) =>
                String(d.location).includes(tk) || String(d.facility).includes(tk) || d.deviceId.includes(tk)))
              .map((d) => ({
                deviceId: d.deviceId, facility: d.facility, zone: zoneFromFacility(d.facility),
                location: d.location, lat: d.lat, lng: d.lng, lastSeen: d.lastMeasured, demo: true,
              }))
          : [];
        const out = [...demoMatches, ...realDevs].slice(0, limit);
        return { query, count: out.length, devices: out, demoMode };
      }

      // 지명 → 좌표 (Nominatim, free OpenStreetMap)
      case "geocode_location": {
        const query = String(args.query || "").trim();
        if (!query) return { error: "query 필수" };
        // 군산 우선 검색 (도시명 자동 추가). 한국어 우선 헤더.
        // 군산 viewbox: lng_min=126.3, lat_min=35.7 — lng_max=127.1, lat_max=36.1
        const qWithCity = /군산|군산시|gunsan/i.test(query) ? query : `${query} 군산`;
        const url = `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(qWithCity)}&format=json&addressdetails=0&limit=3&accept-language=ko&viewbox=126.3,36.1,127.1,35.7&bounded=0&countrycodes=kr`;
        try {
          const res = await fetch(url, {
            headers: { "User-Agent": "siwon-IoT-monitoring/1.0 capstone-project" },
            signal: AbortSignal.timeout(8000),
          });
          if (!res.ok) return { error: `Nominatim HTTP ${res.status}` };
          const data = await res.json();
          if (!Array.isArray(data) || data.length === 0) {
            return { query, count: 0, message: "지명 못 찾음 — 다른 키워드 시도 또는 search_devices_by_location 사용 권장" };
          }
          return {
            query,
            count: data.length,
            results: data.map((r) => ({
              displayName: r.display_name,
              lat: Number(r.lat),
              lng: Number(r.lon),
              type: r.type,
              importance: r.importance,
            })),
          };
        } catch (e) {
          return { error: `geocode 실패: ${e.message}` };
        }
      }

      // 군산 일별 날씨 예보 (Open-Meteo, 최대 7일)
      case "get_weather_forecast": {
        const days = Math.min(7, Math.max(1, parseInt(args.days, 10) || 7));
        const url = `https://api.open-meteo.com/v1/forecast?latitude=35.9678&longitude=126.7369&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max&forecast_days=${days}&timezone=Asia%2FSeoul`;
        const WMO = { 0:"맑음",1:"대체로 맑음",2:"부분 흐림",3:"흐림",45:"안개",48:"안개",51:"약한 이슬비",53:"이슬비",55:"강한 이슬비",61:"약한 비",63:"비",65:"강한 비",71:"약한 눈",73:"눈",75:"강한 눈",77:"눈알갱이",80:"소나기",81:"소나기",82:"강한 소나기",85:"눈 소나기",86:"눈 소나기",95:"뇌우",96:"뇌우(우박)",99:"강한 뇌우" };
        try {
          const res = await fetch(url, { signal: AbortSignal.timeout(8000) });
          if (!res.ok) return { error: `Open-Meteo HTTP ${res.status}` };
          const j = await res.json();
          const d = j.daily;
          if (!d || !Array.isArray(d.time)) return { error: "예보 데이터 없음" };
          const forecast = d.time.map((date, i) => ({
            date,
            sky: WMO[d.weather_code[i]] ?? `code ${d.weather_code[i]}`,
            tempMin: d.temperature_2m_min[i],
            tempMax: d.temperature_2m_max[i],
            precipMm: d.precipitation_sum[i],
            precipProb: d.precipitation_probability_max?.[i] ?? null,
          }));
          return {
            location: "군산", source: "Open-Meteo 실시간 예보 API",
            unit: { temp: "°C", precip: "mm", precipProb: "%" },
            days: forecast.length, forecast,
          };
        } catch (e) {
          return { error: `예보 조회 실패: ${e.message}` };
        }
      }

      // 군산 과거 날씨 (Open-Meteo 아카이브, 최대 1년 — 길면 월별 집계)
      case "get_weather_history": {
        const DAY = 86400000;
        const iso = (ms) => new Date(ms).toISOString().slice(0, 10);
        const valid = (s) => typeof s === "string" && /^\d{4}-\d{2}-\d{2}$/.test(s);
        const archiveMax = iso(Date.now() - DAY);              // 아카이브는 어제까지
        let end = valid(args.end_date) ? args.end_date : archiveMax;
        if (end > archiveMax) end = archiveMax;                // 미래·오늘 → 어제로 클램프
        let start = valid(args.start_date) ? args.start_date : iso(new Date(end).getTime() - 364 * DAY);
        if (start > end) start = end;
        if ((new Date(end) - new Date(start)) / DAY > 366) start = iso(new Date(end).getTime() - 366 * DAY);  // 최대 366일
        const WMO = { 0:"맑음",1:"대체로 맑음",2:"부분 흐림",3:"흐림",45:"안개",48:"안개",51:"약한 이슬비",53:"이슬비",55:"강한 이슬비",61:"약한 비",63:"비",65:"강한 비",71:"약한 눈",73:"눈",75:"강한 눈",77:"눈알갱이",80:"소나기",81:"소나기",82:"강한 소나기",85:"눈 소나기",86:"눈 소나기",95:"뇌우",96:"뇌우(우박)",99:"강한 뇌우" };
        const url = `https://archive-api.open-meteo.com/v1/archive?latitude=35.9678&longitude=126.7369&start_date=${start}&end_date=${end}&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=Asia%2FSeoul`;
        try {
          const res = await fetch(url, { signal: AbortSignal.timeout(12000) });
          if (!res.ok) {
            let reason = ""; try { reason = (await res.json()).reason || ""; } catch {}
            return { error: `Open-Meteo 아카이브 HTTP ${res.status}${reason ? ` — ${reason}` : ""}` };
          }
          const d = (await res.json()).daily;
          if (!d || !Array.isArray(d.time) || d.time.length === 0) return { error: "과거 데이터 없음" };
          const n = d.time.length;
          const nums = (a) => a.filter((v) => v != null);
          const avg = (a) => nums(a).length ? +(nums(a).reduce((x, y) => x + y, 0) / nums(a).length).toFixed(1) : null;
          const stats = {
            tempMaxAvg: avg(d.temperature_2m_max),
            tempMinAvg: avg(d.temperature_2m_min),
            tempPeak: nums(d.temperature_2m_max).length ? Math.max(...nums(d.temperature_2m_max)) : null,
            tempLow:  nums(d.temperature_2m_min).length ? Math.min(...nums(d.temperature_2m_min)) : null,
            precipTotalMm: +nums(d.precipitation_sum).reduce((x, y) => x + y, 0).toFixed(1),
            rainyDays: d.precipitation_sum.filter((v) => v != null && v > 0).length,
          };
          const base = { location: "군산", source: "Open-Meteo 과거기상 API(ERA5)", period: { start, end, days: n }, unit: { temp: "°C", precip: "mm" }, stats };
          if (n > 45) {   // 월별 집계
            const m = {};
            d.time.forEach((t, i) => {
              const mm = (m[t.slice(0, 7)] ||= { tMax: [], tMin: [], precip: 0, rainy: 0 });
              if (d.temperature_2m_max[i] != null) mm.tMax.push(d.temperature_2m_max[i]);
              if (d.temperature_2m_min[i] != null) mm.tMin.push(d.temperature_2m_min[i]);
              if (d.precipitation_sum[i] != null) { mm.precip += d.precipitation_sum[i]; if (d.precipitation_sum[i] > 0) mm.rainy++; }
            });
            const monthly = Object.entries(m).map(([month, v]) => ({
              month,
              tempMaxAvg: v.tMax.length ? +(v.tMax.reduce((x, y) => x + y, 0) / v.tMax.length).toFixed(1) : null,
              tempMinAvg: v.tMin.length ? +(v.tMin.reduce((x, y) => x + y, 0) / v.tMin.length).toFixed(1) : null,
              precipTotalMm: +v.precip.toFixed(1),
              rainyDays: v.rainy,
            }));
            return { ...base, granularity: "monthly", monthly };
          }
          const daily = d.time.map((date, i) => ({
            date, sky: WMO[d.weather_code[i]] ?? `code ${d.weather_code[i]}`,
            tempMin: d.temperature_2m_min[i], tempMax: d.temperature_2m_max[i], precipMm: d.precipitation_sum[i],
          }));
          return { ...base, granularity: "daily", daily };
        } catch (e) {
          return { error: `과거 날씨 조회 실패: ${e.message}` };
        }
      }

      // 웹 검색 (DuckDuckGo Instant Answer — 무료·키X. 개념/엔티티 질의에 강함)
      case "web_search": {
        const q = String(args.query || "").trim();
        if (!q) return { error: "query 필수" };
        const url = `https://api.duckduckgo.com/?q=${encodeURIComponent(q)}&format=json&no_html=1&no_redirect=1&t=siwon`;
        try {
          const res = await fetch(url, { headers: { "User-Agent": "siwon-IoT-monitoring/1.0 capstone" }, signal: AbortSignal.timeout(10000) });
          if (!res.ok) return { error: `DuckDuckGo HTTP ${res.status}` };
          const d = await res.json();
          const results = [];
          if (d.AbstractText) results.push({ title: d.Heading || q, snippet: d.AbstractText, url: d.AbstractURL || "" });
          const flat = [];
          for (const t of (d.RelatedTopics || [])) {
            if (t?.Text) flat.push(t);
            else if (t?.Topics) for (const s of t.Topics) if (s?.Text) flat.push(s);
          }
          for (const t of flat.slice(0, 6)) results.push({ title: String(t.Text).split(" - ")[0].slice(0, 80), snippet: t.Text, url: t.FirstURL || "" });
          if (results.length === 0) {
            return { query: q, source: "DuckDuckGo", count: 0, note: "즉답 결과 없음 — DuckDuckGo 무료 API는 사전·개념·엔티티 질의에 강하고 일반 웹 결과목록은 제공하지 않음. 가진 지식으로 답하거나 키워드를 바꿔 재시도." };
          }
          return { query: q, source: "DuckDuckGo Instant Answer", count: results.length, results };
        } catch (e) {
          return { error: `웹검색 실패: ${e.message}` };
        }
      }

      // AI 모델 메타 + 단말 threshold
      case "get_ai_model_info": {
        if (!MODEL_CONFIG && !Object.keys(DEVICE_THRESHOLDS).length) {
          return { error: "AI 설정 파일을 로드하지 못했습니다 (ai/config/*.json)" };
        }
        const deviceId = args.deviceId;
        if (deviceId) {
          const th = DEVICE_THRESHOLDS[deviceId];
          if (th == null) return { deviceId, error: `해당 단말 threshold 없음 (학습 대상 아님)` };
          const isSacrificial = MODEL_CONFIG?.sacrificial_devices?.includes(deviceId) || false;
          return {
            deviceId,
            threshold: th,
            threshold70: Number((th * 0.7).toFixed(6)),
            threshold100: Number(th.toFixed(6)),
            isSacrificial,
            note: `정상=MSE<${(th*0.7).toExponential(2)}, 관찰=${(th*0.7).toExponential(2)}~${th.toExponential(2)}, 이상=>${th.toExponential(2)}. MSE 는 LSTM AE 복원 오차.`,
            modelConfig: MODEL_CONFIG ? {
              timeSteps: MODEL_CONFIG.time_steps,
              baseFeatures: MODEL_CONFIG.base_features,
              commQualityThresholdDbm: MODEL_CONFIG.comm_quality_threshold_dbm,
            } : null,
          };
        }
        // 전체 메타
        const allThresholds = Object.values(DEVICE_THRESHOLDS);
        const avgTh = allThresholds.length ? allThresholds.reduce((a, b) => a + b, 0) / allThresholds.length : null;
        const maxTh = allThresholds.length ? Math.max(...allThresholds) : null;
        const minTh = allThresholds.length ? Math.min(...allThresholds) : null;
        return {
          model: "LSTM AutoEncoder",
          deviceCount: Object.keys(DEVICE_THRESHOLDS).length,
          thresholdStats: {
            avg: avgTh != null ? Number(avgTh.toExponential(3)) : null,
            max: maxTh != null ? Number(maxTh.toExponential(3)) : null,
            min: minTh != null ? Number(minTh.toExponential(3)) : null,
          },
          modelConfig: MODEL_CONFIG,
          evalMetrics: EVAL_METRICS,
          classification: {
            정상: "MSE < threshold × 0.70",
            관찰: "threshold × 0.70 ≤ MSE ≤ threshold × 1.00",
            이상: "MSE > threshold × 1.00",
          },
          training: {
            time_steps: MODEL_CONFIG?.time_steps,
            epochs: 50,
            thresholdPercentile: 99,
            note: "정상 데이터만 학습, AutoEncoder 복원 오차로 이상 탐지",
          },
        };
      }

      // 좌표 + 반경 검색 (Haversine 거리)
      case "find_devices_near": {
        const lat = Number(args.lat);
        const lng = Number(args.lng);
        const radiusKm = Math.min(Math.max(Number(args.radiusKm) || 2.0, 0.1), 30);
        const limit = Math.min(Number(args.limit) || 20, 60);
        if (!isFinite(lat) || !isFinite(lng)) return { error: "lat/lng 필수 (숫자)" };
        const [rows] = await pool.query(`
          SELECT t.NAME AS deviceId, f.NUMBER AS facility, f.POSITION AS location,
                 f.LATITUDE AS lat, f.LONGITUDE AS lng,
                 (SELECT MAX(r.DATE) FROM kscg_sensor_info si JOIN kscg_recent_data r ON r.SENSOR_ID = si.SENSOR_ID WHERE si.TRANSMITTER_ID = t.TRANSMITTER_ID) AS lastSeen
          FROM kscg_transmitter_info t
          JOIN kscg_site_mydevice mm ON mm.TRANSMITTER_ID = t.TRANSMITTER_ID AND mm.SITE_ID = ?
          LEFT JOIN kscg_facility_info f ON f.TRANSMITTER_ID = t.TRANSMITTER_ID
          WHERE f.LATITUDE IS NOT NULL AND f.LONGITUDE IS NOT NULL
        `, [SITE_ID]);
        const realDist = rows.map((r) => ({
          deviceId: r.deviceId, facility: r.facility, zone: zoneFromFacility(r.facility),
          location: r.location, lat: r.lat, lng: r.lng, lastSeen: r.lastSeen,
          distanceKm: Number(haversineKm(lat, lng, r.lat, r.lng).toFixed(3)),
        }));
        const demoDist = demoMode
          ? getDemoDevices().map((d) => ({
              deviceId: d.deviceId, facility: d.facility, zone: zoneFromFacility(d.facility),
              location: d.location, lat: d.lat, lng: d.lng, lastSeen: d.lastMeasured,
              distanceKm: Number(haversineKm(lat, lng, d.lat, d.lng).toFixed(3)),
              demo: true,
            }))
          : [];
        const withDist = [...realDist, ...demoDist]
          .filter((r) => r.distanceKm <= radiusKm)
          .sort((a, b) => a.distanceKm - b.distanceKm)
          .slice(0, limit);
        return { center: { lat, lng }, radiusKm, count: withDist.length, devices: withDist, demoMode };
      }

      // ─── 자가확장 챗봇 (Self-Extending Chatbot) ───
      // execute_safe_sql: 미리 정의된 16 도구로 답 못 하는 복잡 분석을 위한 자유 SQL.
      //   5단 안전장치:
      //     1) 첫 keyword 가 SELECT/WITH 만 허용
      //     2) DML/DDL/admin keyword 정규식 차단 (SAFE_SQL_BLOCKED)
      //     3) 다중 statement 차단 (세미콜론 후 비어있지 않으면 reject)
      //     4) LIMIT 강제 (없으면 자동 추가, max 1000)
      //     5) 5초 timeout (mysql2 timeout 옵션)
      case "execute_safe_sql": {
        let sql = String(args.sql || "").trim().replace(/;+\s*$/g, "");  // 끝 세미콜론 제거
        if (!sql) return { error: "sql 필수" };
        const limit = Math.min(Math.max(Number(args.limit) || 100, 1), 1000);

        // [1] 첫 keyword 검사
        const firstWord = (sql.match(/^\s*(\w+)/)?.[1] || "").toUpperCase();
        if (!["SELECT", "WITH"].includes(firstWord)) {
          return { error: `차단됨: SELECT/WITH 만 허용. 받은 첫 keyword: '${firstWord}'` };
        }

        // [2] 위험 keyword 정규식 차단 (큰따옴표·작은따옴표 안 문자열은 사전 제거 후 검사)
        const stripped = sql
          .replace(/'(?:\\.|[^'\\])*'/g, "''")
          .replace(/"(?:\\.|[^"\\])*"/g, '""')
          .replace(/`(?:[^`])*`/g, "``")
          .replace(/--[^\n]*/g, "")        // 주석
          .replace(/\/\*[\s\S]*?\*\//g, ""); // 블록 주석
        if (SAFE_SQL_BLOCKED.test(stripped)) {
          return { error: "차단됨: DML/DDL/관리 keyword 포함 (INSERT/UPDATE/DELETE/DDL 류). SELECT 만 가능" };
        }

        // [3] 다중 statement (세미콜론 뒤 비어있지 않음) — 위에서 끝 세미콜론 제거했으므로 중간 ; 만 검사
        if (stripped.includes(";")) {
          return { error: "차단됨: 다중 statement 금지 (한 SELECT 문만)" };
        }

        // [4] LIMIT 강제 추가 (없으면)
        const hasLimit = /\blimit\s+\d+/i.test(sql);
        const finalSql = hasLimit ? sql : `${sql} LIMIT ${limit}`;

        // [5] 5초 timeout 실행
        try {
          const [rows] = await pool.query({ sql: finalSql, timeout: 5000 });
          const arr = Array.isArray(rows) ? rows : [];
          const truncated = arr.length > 1000;
          return {
            sql: finalSql,
            rowCount: arr.length,
            truncated,
            rows: arr.slice(0, 1000),
          };
        } catch (e) {
          return { error: `SQL 실행 오류: ${e.message}`, sql: finalSql };
        }
      }

      // describe_table: 스키마 자가 탐색
      case "describe_table": {
        const tableName = String(args.tableName || "").trim();
        if (!SAFE_TABLE_NAME.test(tableName)) {
          return { error: "잘못된 테이블 이름 (영문/숫자/_만)" };
        }
        try {
          const [cols] = await pool.query(`SHOW COLUMNS FROM \`${tableName}\``);
          const [idx]  = await pool.query(`SHOW INDEX FROM \`${tableName}\``);
          // 인덱스 그룹화
          const indexMap = {};
          for (const i of idx) {
            const k = i.Key_name;
            if (!indexMap[k]) indexMap[k] = { name: k, columns: [], unique: i.Non_unique === 0 };
            indexMap[k].columns.push(i.Column_name);
          }
          // 대략적인 row 수 (INFORMATION_SCHEMA, 빠른 추정)
          let rowEstimate = null;
          try {
            const [[r]] = await pool.query(
              `SELECT TABLE_ROWS FROM information_schema.TABLES WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = ?`,
              [tableName],
            );
            rowEstimate = r?.TABLE_ROWS ?? null;
          } catch (_) {}
          return {
            tableName,
            rowEstimate,
            columns: cols.map((c) => ({
              name:    c.Field,
              type:    c.Type,
              null:    c.Null,
              key:     c.Key,
              default: c.Default,
              extra:   c.Extra,
            })),
            indexes: Object.values(indexMap),
          };
        } catch (e) {
          return { error: e.message };
        }
      }

      default:
        return { error: `unknown tool: ${name}` };
    }
  } catch (err) {
    console.error(`[tool ${name}]`, err);
    return { error: err.message };
  }
}

// Ollama tool calling 라운드 루프 (최대 5회)
//   - tool_calls 가 있으면 execTool 실행 후 messages 에 append → 다음 라운드
//   - tool_calls 가 없으면 최종 응답 (content) 반환
//   - toolTrace 로 어떤 도구가 호출됐는지 추적 (디버깅용)
// OpenAI(GPT) 라운드 루프 — 비스트리밍. 도구호출 시 send('tool') + 결과 append 후 다음 라운드.
//   tool 메시지는 OpenAI 형식(tool_call_id) 사용. TOOLS/execTool/시스템프롬프트는 Ollama 와 공유.
async function runOpenAIRounds({ send, working, model, demoMode, signal, maxRounds, toolTrace, webSearch = false }) {
  if (!OPENAI_API_KEY) throw new Error("OPENAI_API_KEY 미설정 (secrets/local/openai.env)");
  let usage = {};
  for (let round = 0; round < maxRounds; round++) {
    const res = await fetch(OPENAI_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${OPENAI_API_KEY}` },
      body: JSON.stringify({ model, messages: working, tools: toolsFor(webSearch), tool_choice: "auto" }),  // temperature 생략 — gpt-5.x 는 기본값(1)만 허용
      signal,
    });
    if (!res.ok) {
      const t = await res.text().catch(() => "");
      throw new Error(`OpenAI HTTP ${res.status}: ${t.slice(0, 200)}`);
    }
    const data = await res.json();
    usage = data.usage || usage;
    const msg = (data.choices && data.choices[0] && data.choices[0].message) || {};
    const toolCalls = msg.tool_calls || [];
    const tokens = { prompt: usage.prompt_tokens || 0, completion: usage.completion_tokens || 0 };
    if (!toolCalls.length) {
      return { reply: (msg.content || "(빈 응답)").trim(), rounds: round + 1, tokens };
    }
    working.push({ role: "assistant", content: msg.content || "", tool_calls: toolCalls });
    for (const tc of toolCalls) {
      const name = tc.function && tc.function.name;
      let args = {};
      try { args = JSON.parse((tc.function && tc.function.arguments) || "{}"); } catch { args = {}; }
      send("tool", { round: round + 1, name, args });
      const result = await execTool(name, args, demoMode);
      toolTrace.push({ round: round + 1, name, args, ok: !(result && result.error) });
      working.push({ role: "tool", tool_call_id: tc.id, content: JSON.stringify(result) });
    }
  }
  return { reply: "(도구 호출 한도 초과 — 정보가 충분하지 않아 답변을 마무리하지 못했습니다.)", rounds: maxRounds,
           tokens: { prompt: usage.prompt_tokens || 0, completion: usage.completion_tokens || 0 } };
}

async function runChatWithTools(messages, signal, demoMode = false, model = OLLAMA_MODEL, webSearch = false) {
  const MAX_ROUNDS = 5;
  const working = [...messages];
  const toolTrace = [];
  let lastTokens = {};
  for (let round = 0; round < MAX_ROUNDS; round++) {
    const res = await fetch(`${OLLAMA_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        messages: working,
        tools: toolsFor(webSearch),
        stream: false,
        think: false,
        options: { temperature: 0.3, num_predict: 1000 },
      }),
      signal,
    });
    if (!res.ok) throw new Error(`Ollama HTTP ${res.status}: ${await res.text().catch(() => "")}`);
    const data = await res.json();
    const msg = data.message || {};
    lastTokens = { prompt: data.prompt_eval_count, completion: data.eval_count };

    // tool 호출이 없으면 최종 응답
    const toolCalls = msg.tool_calls || [];
    if (toolCalls.length === 0) {
      return { content: msg.content || "", rounds: round + 1, toolTrace, tokens: lastTokens };
    }

    // tool 호출 실행 → 결과 messages 에 append → 다음 라운드
    working.push({ role: "assistant", content: msg.content || "", tool_calls: toolCalls });
    for (const tc of toolCalls) {
      const name = tc.function?.name;
      const args = tc.function?.arguments || {};
      console.log(`[tool${demoMode ? " DEMO" : ""}] round ${round + 1} → ${name}(${JSON.stringify(args)})`);
      const result = await execTool(name, args, demoMode);
      const ok = !result?.error;
      toolTrace.push({ round: round + 1, name, args, ok });
      working.push({
        role: "tool",
        content: JSON.stringify(result),
        tool_name: name,
      });
    }
  }
  return {
    content: "(도구 호출 한도 초과 — 정보를 더 얻지 못해 답변을 마무리하지 못했습니다.)",
    rounds: MAX_ROUNDS,
    toolTrace,
    tokens: lastTokens,
  };
}

// ── /api/chat ────────────────────────────────────────
// Function Calling 사용:
//   - LLM 에 TOOLS 정의 동봉 → 필요시 list_devices / get_device_detail 등 호출
//   - 서버가 execTool() 로 MySQL 조회 → 결과를 messages 에 append → 다시 LLM 호출
//   - 최대 5 라운드 (runChatWithTools 내부 MAX_ROUNDS)
app.post("/api/chat", async (req, res) => {
  const { message, context = {}, history = [], sessionId, model, webSearch = false } = req.body || {};
  const useModel = pickModel(model);
  if (!message || typeof message !== "string") {
    return res.status(400).json({ ok: false, error: "message 필드가 비어있습니다." });
  }

  const systemPrompt = buildSystemPrompt(context);

  // 최근 6턴까지만 히스토리에 포함 (토큰 절약)
  const recent = history.slice(-6).map((h) => ({
    role: h.role === "ai" ? "assistant" : "user",
    content: h.text,
  }));

  const messages = [
    { role: "system", content: systemPrompt },
    ...recent,
    { role: "user", content: message },
  ];

  // 세션 영구화 (best-effort). 로그인 개인 계정만 user_id 귀속(게스트/익명은 null=브라우저-로컬).
  const ownerUid = chatOwner(req);
  const clientId = req.body?.clientId || null;
  const sid = await ensureChatSession(sessionId, message, ownerUid);
  const demoMode = isDemoMode(req);

  try {
    const ctrl = new AbortController();
    const timeout = setTimeout(() => ctrl.abort(), 120_000); // 120s (최대 5 tool 라운드 여유)
    const result = await runChatWithTools(messages, ctrl.signal, demoMode, useModel, webSearch);
    clearTimeout(timeout);

    const reply = (result.content || "(빈 응답)").trim();

    // chat_messages 영구화 (background, best-effort)
    if (sid) {
      persistMessage(sid, "user", message, context, null, null, reqIp(req))
        .then(() => persistMessage(sid, "ai", reply, { rounds: result.rounds, toolCalls: result.toolTrace }, result.tokens, useModel, reqIp(req)));
    }

    return res.json({
      ok:        true,
      sessionId: sid,
      reply,
      model:     useModel,
      rounds:    result.rounds,
      toolCalls: result.toolTrace || [],
      tokens:    result.tokens || {},
    });
  } catch (err) {
    console.error("[chat] error:", err.message);
    return res.status(500).json({ ok: false, error: err.message });
  }
});

// ── /api/chat/stream (SSE + Function Calling) ───────
// 진짜 스트리밍 + tools 동시 지원.
//   1 라운드: Ollama stream:true 로 호출 → delta 들 SSE 로 forward
//   라운드 끝(done:true) 메시지에 tool_calls 있으면 → 실행 → result append → 다음 라운드
//   tool_calls 없으면 최종 → `done` 이벤트 보내고 종료. 최대 MAX_ROUNDS 라운드.
//
// SSE events:
//   - delta : { text }              모델이 토큰 생성한 조각
//   - tool  : { round, name, args } 도구 호출 발생 알림 (UI 가 "조회 중..." 표시 가능)
//   - done  : { reply, tokens, rounds, toolCalls }   최종 완성 답변
//   - error : { message }
app.post("/api/chat/stream", async (req, res) => {
  const { message, context = {}, history = [], sessionId, model, webSearch = false } = req.body || {};
  const useModel = pickModel(model);
  if (!message || typeof message !== "string") {
    return res.status(400).json({ ok: false, error: "message 필드가 비어있습니다." });
  }

  // SSE 헤더
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Accel-Buffering", "no");
  res.flushHeaders && res.flushHeaders();

  // 클라이언트가 끊겨도 생성은 계속 — 끊긴 뒤 SSE write 는 조용히 무시(서버는 끝까지 만들어 DB 저장)
  let clientGone = false;
  const send = (event, data) => {
    if (clientGone) return;
    try {
      res.write(`event: ${event}\n`);
      res.write(`data: ${JSON.stringify(data)}\n\n`);
    } catch { clientGone = true; }
  };

  const systemPrompt = buildSystemPrompt(context);
  const recent = history.slice(-6).map((h) => ({
    role: h.role === "ai" ? "assistant" : "user",
    content: h.text,
  }));

  const working = [
    { role: "system", content: systemPrompt },
    ...recent,
    { role: "user", content: message },
  ];

  // 세션 영구화 (best-effort). 응답에 session 이벤트로 동봉. 로그인 개인 계정만 user_id 귀속.
  const ownerUid = chatOwner(req);
  const clientId = req.body?.clientId || null;
  const sid = await ensureChatSession(sessionId, message, ownerUid);
  if (sid) send("session", { sessionId: sid });
  // 다른 화면에 사용자 메시지 즉시 반영 (발신 화면 제외)
  if (sid && ownerUid != null) broadcastToOwner(ownerUid, { type: "chat:user", sessionId: sid, text: message, ts: Date.now() }, clientId);
  const demoMode = isDemoMode(req);

  const MAX_ROUNDS = 5;
  const toolTrace = [];
  let finalAccum = "";
  let lastTokens = { prompt: 0, completion: 0 };

  const ctrl = new AbortController();
  const STREAM_TIMEOUT_MS = useModel === "qwen3.5:27b" ? 1_200_000 : isOpenAI(useModel) ? 240_000 : 180_000;   // 27b 20분 · GPT 4분 · 로컬 3분
  const timeout = setTimeout(() => { console.log(`[chat/stream] timeout ${STREAM_TIMEOUT_MS}ms`); ctrl.abort(); }, STREAM_TIMEOUT_MS);
  // client disconnect → 진행 중인 Ollama fetch abort.
  // res.writableFinished 는 res.end() 호출 후 true 가 됨. 우리가 아직 응답 안 끝낸 상태라면
  // 'close' 는 진짜 클라이언트 단절. (express body parser 가 emit 하는 spurious close 는
  //  대부분 응답 헤더 전이라서 첫 write 이후엔 안전.)
  res.on("close", () => {
    if (res.writableFinished) return;   // 우리가 정상 종료한 경우
    // 실제 채팅처럼 — 클라이언트가 끊겨도 LLM 생성을 중단하지 않는다.
    // 서버가 끝까지 생성해 chat_messages 에 저장 → 사용자가 다시 열면 그 답변을 보게 됨.
    // (STREAM_TIMEOUT_MS 타임아웃이 폭주 방지 안전망)
    if (clientGone) return;
    clientGone = true;
    console.log("[chat/stream] client disconnected → 생성 계속(중단 안 함), 완료 후 저장");
  });

  // ── OpenAI(GPT) 분기 — 비스트리밍 라운드 루프(도구호출 + 최종 done). 외부 전송. ──
  if (isOpenAI(useModel)) {
    try {
      const r = await runOpenAIRounds({ send, working, model: useModel, demoMode, signal: ctrl.signal, maxRounds: MAX_ROUNDS, toolTrace, webSearch });
      send("done", { reply: r.reply, sessionId: sid, model: useModel, rounds: r.rounds, toolCalls: toolTrace, tokens: r.tokens });
      if (sid) {
        persistMessage(sid, "user", message, context, null, null, reqIp(req))
          .then(() => persistMessage(sid, "ai", r.reply, { rounds: r.rounds, toolCalls: toolTrace }, r.tokens, useModel, reqIp(req)));
      }
      if (sid && ownerUid != null) broadcastToOwner(ownerUid, { type: "chat:ai", sessionId: sid, text: r.reply, model: useModel, ts: Date.now() }, clientId);
    } catch (err) {
      console.error("[chat/stream openai]", err.message);
      send("error", { message: err.message });
    }
    clearTimeout(timeout);
    return res.end();
  }

  try {
    for (let round = 0; round < MAX_ROUNDS; round++) {
      const ollamaRes = await fetch(`${OLLAMA_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: useModel,
          messages: working,
          tools: toolsFor(webSearch),
          stream: true,
          think: false,
          options: { temperature: 0.3, num_predict: 1000 },
        }),
        signal: ctrl.signal,
      });
      if (!ollamaRes.ok) {
        const txt = await ollamaRes.text().catch(() => "");
        send("error", { message: `Ollama HTTP ${ollamaRes.status}: ${txt}` });
        clearTimeout(timeout);
        return res.end();
      }

      // ndjson 파싱: 이번 라운드의 content 누적 + 마지막 chunk 에서 tool_calls 파악
      const reader = ollamaRes.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      let roundContent = "";
      let toolCalls = [];
      let roundDone = false;

      while (!roundDone) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let nl;
        while ((nl = buf.indexOf("\n")) !== -1) {
          const line = buf.slice(0, nl).trim();
          buf = buf.slice(nl + 1);
          if (!line) continue;
          let obj;
          try { obj = JSON.parse(line); } catch { continue; }
          const msg = obj.message || {};
          // ★ Ollama 는 tool_calls 를 done:false chunk 에 넣음 (그 후 빈 done:true 가 옴).
          //   chunk 어디서 오든 잡아 누적.
          if (Array.isArray(msg.tool_calls) && msg.tool_calls.length) {
            toolCalls.push(...msg.tool_calls);
          }
          const piece = msg.content;
          if (piece) {
            roundContent += piece;
            send("delta", { text: piece });
          }
          if (obj.done) {
            roundDone = true;
            lastTokens = {
              prompt:    obj.prompt_eval_count    || lastTokens.prompt,
              completion:obj.eval_count           || lastTokens.completion,
            };
          }
        }
      }

      // tool_calls 가 없으면 최종 답변
      if (toolCalls.length === 0) {
        finalAccum += roundContent;
        const finalReply = finalAccum.trim();
        send("done", {
          reply:     finalReply,
          sessionId: sid,
          model:     useModel,
          rounds:    round + 1,
          toolCalls: toolTrace,
          tokens:    lastTokens,
        });
        // chat_messages 영구화 (background)
        if (sid) {
          persistMessage(sid, "user", message, context, null, null, reqIp(req))
            .then(() => persistMessage(sid, "ai", finalReply, { rounds: round + 1, toolCalls: toolTrace }, lastTokens, useModel, reqIp(req)));
        }
        if (sid && ownerUid != null) broadcastToOwner(ownerUid, { type: "chat:ai", sessionId: sid, text: finalReply, model: useModel, ts: Date.now() }, clientId);
        clearTimeout(timeout);
        return res.end();
      }

      // tool_calls 가 있으면 실행 → result append → 다음 라운드
      working.push({ role: "assistant", content: roundContent, tool_calls: toolCalls });
      for (const tc of toolCalls) {
        const name = tc.function?.name;
        const args = tc.function?.arguments || {};
        console.log(`[stream tool${demoMode ? " DEMO" : ""}] round ${round + 1} → ${name}(${JSON.stringify(args)})`);
        send("tool", { round: round + 1, name, args });
        const result = await execTool(name, args, demoMode);
        const ok = !result?.error;
        toolTrace.push({ round: round + 1, name, args, ok });
        working.push({
          role: "tool",
          content: JSON.stringify(result),
          tool_name: name,
        });
      }
    }

    // MAX_ROUNDS 초과
    send("done", {
      reply: "(도구 호출 한도 초과 — 정보가 충분하지 않아 답변을 마무리하지 못했습니다.)",
      model: useModel,
      rounds: MAX_ROUNDS,
      toolCalls: toolTrace,
      tokens: lastTokens,
    });
    clearTimeout(timeout);
    res.end();
  } catch (err) {
    console.error("[chat/stream] error:", err.message);
    send("error", { message: err.message });
    clearTimeout(timeout);
    res.end();
  }
});

// ─────────────────────────────────────────────────────
// MySQL 기반 데이터 API
//   - 우리 siwon DB (Mac Studio MySQL) 에서 조회.
//   - 옴니 KSCG 미러 + 자체 테이블 둘 다 활용.
//   - 대시보드용 도메인 매핑 (8 센서 → 6 표시 + 배터리/RSSI) 적용.
// ─────────────────────────────────────────────────────

// 단말당 SENSOR_ID 순서 → 측정 종류 매핑 (각 단말 8 센서, 고정 시퀀스)
// TB_SENSOR_INFO.TYPE 이 모두 1 로 잘못 입력돼 있어 SENSOR_ID 순서로 추정.
//   seq 1=방식전위(mV)  2=희생전류(mA)  3=AC유입(mV)  4=배터리(mV)
//   seq 5=온도(℃)       6=습도(%)      7=충격/가스   8=RSSI
const SENSOR_SEQ_KIND = ["volt", "sacrificial", "ac", "battery", "temp", "hum", "shock", "commDbm"];

// 시설번호 "1-178" 형식의 첫 자리 → 구역 라벨
function zoneFromFacility(num) {
  if (!num) return "-";
  const m = String(num).match(/^(\d+)/);
  return m ? `제${m[1]}구역` : "-";
}

// (낡음) 옛 승격 휴리스틱 잔재. 현재 status 는 mapStatus 가 이두현 risk_level 그대로 매핑 → 이 상수는 status 판정에 미사용.
const AI_CRITICAL_RATIO = 5;

// 단말별 최신 ai_predictions 1건씩 → Map<transmitter_id, {risk, mse, threshold}>
async function loadLatestAi() {
  const map = new Map();
  try {
    const [rows] = await pool.query(`
      SELECT p.transmitter_id AS txid, p.risk_level AS risk, p.mse, p.threshold,
             p.comm_status AS commStatus, p.ai_reliability AS aiReliability,
             p.feature_contributions AS featureContributions, p.predicted_at AS predictedAt
      FROM ai_predictions p
      JOIN (SELECT transmitter_id, MAX(predicted_at) AS mx
            FROM ai_predictions GROUP BY transmitter_id) l
        ON l.transmitter_id = p.transmitter_id AND l.mx = p.predicted_at
    `);
    for (const r of rows) map.set(r.txid, r);
  } catch (e) { console.error("[loadLatestAi]", e.message); }
  return map;
}

// AI mse/threshold → 배수 (없으면 null)
function aiRatioOf(ai) {
  return ai && ai.mse != null && Number(ai.threshold) > 0
    ? Number((Number(ai.mse) / Number(ai.threshold)).toFixed(2)) : null;
}

// feature_contributions(JSON dict) → [{sensor, pct}] 상위 5 (정규화 %)
// 엔지니어드 피처명 → 사람이 읽는 라벨. (프론트 featureLabel 과 동일 규칙 + AC유입 공백)
//   원시 컬럼명(_dev24, _diff1)이 LLM 답변/anomalies 라벨에 그대로 노출되지 않게 함.
function featureLabelKo(name) {
  return String(name || "")
    .replace(/_dev24$/u, " 편차")
    .replace(/_diff1$/u, " 변화")
    .replace(/AC유입/u, "AC 유입")
    .replace(/_/gu, " ")
    .trim();
}

function contribFromFeatures(fc) {
  let obj = fc;
  if (typeof fc === "string") { try { obj = JSON.parse(fc); } catch { obj = null; } }
  if (!obj || typeof obj !== "object") return [];
  const ent = Object.entries(obj).map(([k, v]) => [k, Math.abs(Number(v) || 0)]);
  const sum = ent.reduce((s, [, v]) => s + v, 0) || 1;
  return ent.sort((a, b) => b[1] - a[1]).slice(0, 5)
            .map(([sensor, v]) => ({ sensor: featureLabelKo(sensor), pct: Number((v / sum * 100).toFixed(1)) }));
}

function classifyAcInput(ac) {
  const value = Number(ac);
  if (!Number.isFinite(value)) return null;
  const cautionThreshold = 200;
  const criticalThreshold = 500;
  if (value >= criticalThreshold) {
    return {
      metric: "ac",
      value,
      unit: "mV",
      level: "위험",
      comparison: "critical_threshold_exceeded",
      cautionThreshold,
      criticalThreshold,
      overCriticalBy: Number((value - criticalThreshold).toFixed(1)),
      ratioToCritical: Number((value / criticalThreshold).toFixed(3)),
      wording: `${value} mV 는 500 mV 즉각 점검 기준을 초과했습니다. '근접'이 아니라 '초과'로 설명하세요.`,
    };
  }
  if (value >= cautionThreshold) {
    return {
      metric: "ac",
      value,
      unit: "mV",
      level: "주의",
      comparison: value >= criticalThreshold * 0.8 ? "near_critical_threshold" : "caution_threshold_exceeded",
      cautionThreshold,
      criticalThreshold,
      underCriticalBy: Number((criticalThreshold - value).toFixed(1)),
      ratioToCritical: Number((value / criticalThreshold).toFixed(3)),
      wording: value >= criticalThreshold * 0.8
        ? `${value} mV 는 500 mV 즉각 점검 기준에 근접했습니다.`
        : `${value} mV 는 200 mV 주의 기준을 초과했지만 500 mV 즉각 점검 기준에는 아직 못 미칩니다.`,
    };
  }
  return {
    metric: "ac",
    value,
    unit: "mV",
    level: "정상",
    comparison: "below_caution_threshold",
    cautionThreshold,
    criticalThreshold,
    ratioToCritical: Number((value / criticalThreshold).toFixed(3)),
    wording: `${value} mV 는 200 mV 주의 기준 미만입니다.`,
  };
}

// 주요 센서 도메인 판정 — LLM 이 "근접 vs 초과", 정상/주의/위험을 정확히 답하도록 도구 결과에 동봉.
//   기준값은 시스템 프롬프트 도메인 지식 및 차트 밴드와 일치. (상태 판정/mapStatus 와 무관 — 설명용)
function mkJudge(metric, value, unit, level, wording) {
  return { metric, value, unit, level, wording };
}

// 방식전위(P/S Potential) — -850mV 이하(더 음수)가 양호. 차트 밴드: normal -850 / warn -700.
function classifyCpVolt(volt) {
  const v = Number(volt);
  if (!Number.isFinite(v)) return null;
  if (v <= -850) return mkJudge("volt", v, "mV", "정상", `${v}mV 는 -850mV 방호 기준을 충족(이하)합니다.`);
  if (v <= -700) return mkJudge("volt", v, "mV", "주의", `${v}mV 는 -850mV 방호 기준을 초과(미달)했습니다. 방식전위 기준초과 확인이 필요합니다.`);
  return mkJudge("volt", v, "mV", "위험", `${v}mV 는 -700mV 보다 높아 방식 보호가 미흡합니다. 정류기 출력 점검이 필요합니다.`);
}

// 희생전류 — 희생양극 단말 기준 1mA 이하 교체 검토. 비희생양극 단말은 판정 대상 아님.
function classifySacrificial(mA, isSacrificial) {
  const v = Number(mA);
  if (!Number.isFinite(v)) return null;
  if (!isSacrificial) return mkJudge("sacrificial", v, "mA", "해당없음", "희생양극 단말이 아니므로 희생전류 판정 대상이 아닙니다.");
  if (v <= 1) return mkJudge("sacrificial", v, "mA", "주의", `${v}mA 로 1mA 이하입니다. 양극 소모/접속부 점검·교체 검토가 필요합니다.`);
  if (v <= 2) return mkJudge("sacrificial", v, "mA", "관찰", `${v}mA 로 보호 전류가 낮은 편입니다. 추이 관찰을 권장합니다.`);
  return mkJudge("sacrificial", v, "mA", "정상", `${v}mA 로 보호 전류가 유지되고 있습니다.`);
}

// 통신 품질(RSSI dBm) — -65 이상 양호, -75 이하 주의, -85 이하 두절 임박, -115 이하 두절.
function classifyCommDbm(dbm) {
  const v = Number(dbm);
  if (!Number.isFinite(v)) return null;
  if (v <= -115) return mkJudge("commDbm", v, "dBm", "위험", `${v}dBm 로 통신 두절 수준입니다. 안테나·전원·맨홀 확인이 필요합니다.`);
  if (v <= -85)  return mkJudge("commDbm", v, "dBm", "위험", `${v}dBm 로 두절 임박 수준입니다. 안테나·전원·맨홀 확인이 필요합니다.`);
  if (v <= -75)  return mkJudge("commDbm", v, "dBm", "주의", `${v}dBm 로 신호가 약합니다(-75dBm 이하).`);
  return mkJudge("commDbm", v, "dBm", "정상", `${v}dBm 로 통신 신호가 양호합니다.`);
}

// 배터리 — 차트 밴드: normal 3500 / warn 3200 mV.
function classifyBattery(mV) {
  const v = Number(mV);
  if (!Number.isFinite(v)) return null;
  if (v >= 3500) return mkJudge("battery", v, "mV", "정상", `${v}mV 로 배터리 전압이 충분합니다.`);
  if (v >= 3200) return mkJudge("battery", v, "mV", "주의", `${v}mV 로 배터리 전압이 낮아지고 있습니다. 교체 일정 검토가 필요합니다.`);
  return mkJudge("battery", v, "mV", "위험", `${v}mV 로 배터리 전압이 부족합니다. 교체가 필요합니다.`);
}

// 주요 센서 종합 판정 (get_device_detail 등 도구 결과 동봉용)
function buildSensorJudgement(s, isSacrificial = false) {
  if (!s || typeof s !== "object") return null;
  return {
    ac:          classifyAcInput(s.ac),
    volt:        classifyCpVolt(s.volt),
    sacrificial: classifySacrificial(s.sacrificial, isSacrificial),
    commDbm:     classifyCommDbm(s.commDbm),
    battery:     classifyBattery(s.battery),
  };
}

function sensorJudgementForKind(kind, value, isSacrificial = false) {
  switch (kind) {
    case "volt":        return classifyCpVolt(value);
    case "sacrificial": return classifySacrificial(value, isSacrificial);
    case "ac":          return classifyAcInput(value);
    case "battery":     return classifyBattery(value);
    case "commDbm":     return classifyCommDbm(value);
    default:            return null;
  }
}

function classifyAiPrediction(ai) {
  if (!ai || ai.mse == null || !(Number(ai.threshold) > 0)) return null;
  const mse = Number(ai.mse);
  const threshold = Number(ai.threshold);
  const ratio = mse / threshold;
  const ratioText = ratio >= 10 ? `x${Math.round(ratio)}` : `x${Number(ratio.toFixed(2))}`;
  const percentText = `${Number((ratio * 100).toFixed(1))}%`;
  let level = "정상";
  if (ratio > 1) level = "이상";
  else if (ratio >= 0.7) level = "관찰";
  return {
    level,
    mse,
    threshold,
    ratio: Number(ratio.toFixed(3)),
    ratioPercent: Number((ratio * 100).toFixed(1)),
    ratioText,
    wording: ratio >= 10
      ? `현재 MSE 는 AI 기준 대비 ${ratioText} 수준입니다. 큰 퍼센트(${percentText})보다 배수로 설명하세요.`
      : `현재 MSE 는 AI 임계값의 ${percentText} 수준입니다(AI 기준 대비 ${ratioText}).`,
    trendCaution: "단일 최신 예측값만으로 상승/하락 추세를 단정하지 마세요.",
  };
}

// 단말 status 판정 — 이두현 LSTM 모델 출력(risk_level + comm_status) 그대로 매핑 (캡스톤: AI 작성자 기준 충실).
//   - offline(통신장애) : comm_status '통신고장' (이두현 연속3회 단절 = AI 신뢰불가). + AI 데이터 자체 없음(24h) fallback
//   - critical(화면 '이상') : risk_level '이상' (MSE ≥ threshold, 100% 초과)
//   - warn(화면 '관찰')     : risk_level '관찰' (threshold 70~100%)
//   - normal           : risk_level '정상' (<70%)
//   ※ KSCG 알람·우리 500% 휴리스틱은 status 에 미반영 (알람은 별도 알림/로그로만).
function mapStatus(_deviceStatus, hoursSilent, activeAlarmCount, ai) {
  if (ai && ai.commStatus === "통신고장") return "offline";               // 이두현 통신 판정
  if (!ai && hoursSilent != null && hoursSilent >= 24) return "offline";  // AI 판정할 데이터 없음
  if (ai && ai.risk === "이상") return "critical";                        // 모델 '이상' → 화면 '이상'
  if (ai && ai.risk === "관찰") return "warn";                            // 모델 '관찰' → 화면 '관찰'
  return "normal";
}

// 시간(h) → "N일 M시간" 표기 (Gemini 5/26 피드백 — 687h 같은 큰 숫자 직관성 부족)
function fmtHours(h) {
  if (h == null || !isFinite(h)) return "확인 불가";
  const n = Math.floor(Number(h));
  if (n < 24) return `${n}시간`;
  const days = Math.floor(n / 24);
  const hours = n % 24;
  return hours === 0 ? `${days}일` : `${days}일 ${hours}시간`;
}

// ── GET /api/summary — KPI 카운트 ─────────────────────
app.get("/api/summary", dbRequired, async (req, res) => {
  try {
    const demoMode = isDemoMode(req);
    // 단말별 (통신두절시간 · 7일 알람수 · 최신 AI) → mapStatus 로 /api/devices 와 동일 집계
    // (이전엔 offline/critical 을 독립 SQL 로 세어 중복카운트 + warn=0 버그가 있었음)
    const [silentRows] = await pool.query(`
      SELECT t.TRANSMITTER_ID AS txid,
             TIMESTAMPDIFF(HOUR, MAX(r.DATE), NOW()) AS hoursSilent
      FROM kscg_transmitter_info t
      JOIN kscg_site_mydevice m ON m.TRANSMITTER_ID = t.TRANSMITTER_ID AND m.SITE_ID = ?
      LEFT JOIN kscg_sensor_info si ON si.TRANSMITTER_ID = t.TRANSMITTER_ID
      LEFT JOIN kscg_recent_data r  ON r.SENSOR_ID = si.SENSOR_ID
      GROUP BY t.TRANSMITTER_ID
    `, [SITE_ID]);
    const [almRows] = await pool.query(`
      SELECT si.TRANSMITTER_ID AS txid, COUNT(*) AS cnt
      FROM kscg_alarm_log a
      JOIN kscg_sensor_info si ON si.SENSOR_ID = a.SENSOR_ID
      JOIN kscg_site_mydevice m ON m.TRANSMITTER_ID = si.TRANSMITTER_ID AND m.SITE_ID = ?
      WHERE a.GEN_DATE > DATE_SUB(NOW(), INTERVAL 7 DAY)
      GROUP BY si.TRANSMITTER_ID
    `, [SITE_ID]);
    const almMap = new Map(almRows.map(r => [r.txid, Number(r.cnt)]));
    const aiMap  = await loadLatestAi();

    let total = silentRows.length, offline = 0, critical = 0, warn = 0, normal = 0;
    for (const dRow of silentRows) {
      const hs = dRow.hoursSilent == null ? null : Number(dRow.hoursSilent);
      const st = mapStatus(null, hs, almMap.get(dRow.txid) || 0, aiMap.get(dRow.txid));
      if (st === "offline") offline++;
      else if (st === "critical") critical++;
      else if (st === "warn") warn++;
      else normal++;
    }

    // 데모 모드: 가상 장비 카운트 추가
    let allCount = total;
    if (demoMode) {
      const demo = getDemoDevices();
      const dCrit = demo.filter((d) => d.status === "critical").length;
      const dWarn = demo.filter((d) => d.status === "warn").length;
      const dOff  = demo.filter((d) => d.status === "offline").length;
      allCount += demo.length;
      return res.json({
        ok: true, demoMode: true,
        site_id: SITE_ID,
        counts: {
          all: allCount, normal, critical: critical + dCrit, warn: warn + dWarn, offline: offline + dOff,
        },
      });
    }

    res.json({
      ok: true,
      site_id: SITE_ID,
      counts: { all: total, normal, critical, warn, offline },
    });
  } catch (err) {
    console.error("[/api/summary]", err);
    res.status(500).json({ ok: false, error: err.message });
  }
});

// ── GET /api/devices — 단말 55대 + 시설 + 최신 8 센서 ──
app.get("/api/devices", dbRequired, async (req, res) => {
  try {
    const demoMode = isDemoMode(req);
    // 1. 단말 + 시설 메타
    const [devices] = await pool.query(`
      SELECT
        t.TRANSMITTER_ID AS id,
        t.NAME           AS deviceId,
        t.SERIAL_NUM     AS serial,
        t.DEVICE_STATUS  AS deviceStatus,
        t.INSTALL_DATE   AS installDate,
        t.PERIOD_SEC     AS periodSec,
        f.FACILITY_ID    AS facilityId,
        f.NUMBER         AS facilityNum,
        f.POSITION       AS location,
        f.LATITUDE       AS lat,
        f.LONGITUDE      AS lng
      FROM kscg_transmitter_info t
      JOIN kscg_site_mydevice m ON m.TRANSMITTER_ID = t.TRANSMITTER_ID AND m.SITE_ID = ?
      LEFT JOIN kscg_facility_info f ON f.TRANSMITTER_ID = t.TRANSMITTER_ID
      ORDER BY t.TRANSMITTER_ID
    `, [SITE_ID]);

    // 2. 단말별 8 센서 최신값 (ROW_NUMBER 로 단말당 seq 부여)
    const [sensorRows] = await pool.query(`
      SELECT
        si.TRANSMITTER_ID,
        si.SENSOR_ID,
        si.UNIT,
        r.DATE  AS measuredAt,
        r.VALUE AS value,
        ROW_NUMBER() OVER (PARTITION BY si.TRANSMITTER_ID ORDER BY si.SENSOR_ID) AS seq
      FROM kscg_sensor_info si
      JOIN kscg_site_mydevice m ON m.TRANSMITTER_ID = si.TRANSMITTER_ID AND m.SITE_ID = ?
      LEFT JOIN kscg_recent_data r ON r.SENSOR_ID = si.SENSOR_ID
      ORDER BY si.TRANSMITTER_ID, si.SENSOR_ID
    `, [SITE_ID]);

    // 단말별 sensors 매핑 + 최신 측정시각·통신 두절 시간
    const byDev = {};
    for (const s of sensorRows) {
      const tid  = s.TRANSMITTER_ID;
      const kind = SENSOR_SEQ_KIND[s.seq - 1] || `sensor${s.seq}`;
      if (!byDev[tid]) byDev[tid] = { sensors: {}, lastMeasured: null };
      byDev[tid].sensors[kind] = s.value;
      if (s.measuredAt && (!byDev[tid].lastMeasured || s.measuredAt > byDev[tid].lastMeasured)) {
        byDev[tid].lastMeasured = s.measuredAt;
      }
    }

    // 3. 단말별 최근 7일 활성 알람 개수
    const [alarmRows] = await pool.query(`
      SELECT si.TRANSMITTER_ID, COUNT(*) AS cnt, MAX(a.GEN_DATE) AS latest
      FROM kscg_alarm_log a
      JOIN kscg_sensor_info si ON si.SENSOR_ID = a.SENSOR_ID
      JOIN kscg_site_mydevice m ON m.TRANSMITTER_ID = si.TRANSMITTER_ID AND m.SITE_ID = ?
      WHERE a.GEN_DATE > DATE_SUB(NOW(), INTERVAL 7 DAY)
      GROUP BY si.TRANSMITTER_ID
    `, [SITE_ID]);
    const alarmsByDev = Object.fromEntries(alarmRows.map(r => [r.TRANSMITTER_ID, r]));

    // 4. 단말별 최신 AI 예측 (ai_predictions) — status 판정에 반영
    const aiByDev = await loadLatestAi();

    const now = new Date();
    const out = devices.map((d) => {
      const slot   = byDev[d.id] || { sensors: {}, lastMeasured: null };
      const alm    = alarmsByDev[d.id];
      const ai     = aiByDev.get(d.id);
      const hoursSilent = slot.lastMeasured
        ? Math.floor((now - new Date(slot.lastMeasured)) / 3600000)
        : null;
      const status = mapStatus(d.deviceStatus, hoursSilent, alm ? Number(alm.cnt) : 0, ai);
      return {
        id:          d.id,
        deviceId:    d.deviceId,
        facilityId:  d.facilityNum,
        zone:        zoneFromFacility(d.facilityNum),
        location:    d.location,
        lat:         d.lat,
        lng:         d.lng,
        installDate: d.installDate,
        periodSec:   d.periodSec,
        deviceStatus: d.deviceStatus,
        status,
        // 6+2 센서 최신값 (프론트 mock shape 와 매칭)
        volt:        slot.sensors.volt        ?? null,
        sacrificial: slot.sensors.sacrificial ?? null,
        ac:          slot.sensors.ac          ?? null,
        battery:     slot.sensors.battery     ?? null,
        temp:        slot.sensors.temp        ?? null,
        hum:         slot.sensors.hum         ?? null,
        shock:       slot.sensors.shock       ?? null,
        commDbm:     slot.sensors.commDbm     ?? null,
        commOk:      slot.sensors.commDbm != null && slot.sensors.commDbm > -115,
        updatedAt:   slot.lastMeasured,
        hoursSilent,
        recentAlarms: alm ? Number(alm.cnt) : 0,
        aiRisk:      ai ? ai.risk : null,
        aiMse:       ai && ai.mse != null ? Number(ai.mse) : null,
        aiThreshold: ai && ai.threshold != null ? Number(ai.threshold) : null,
        aiRatio:     ai && ai.mse != null && Number(ai.threshold) > 0 ? Number((Number(ai.mse) / Number(ai.threshold)).toFixed(2)) : null,
      };
    });

    // 데모 모드: 가상 장비 10대 append
    let finalOut = out;
    if (demoMode) {
      const demo = getDemoDevices().map((d) => ({
        id:           d.txid,
        deviceId:     d.deviceId,
        facilityId:   d.facility,
        zone:         zoneFromFacility(d.facility),
        location:     d.location,
        lat:          d.lat,
        lng:          d.lng,
        installDate:  null,
        periodSec:    3600,
        deviceStatus: 1,
        status:       d.status,
        volt:         d.sensors.volt,
        sacrificial:  d.sensors.sacrificial,
        ac:           d.sensors.ac,
        battery:      d.sensors.battery,
        temp:         d.sensors.temp,
        hum:          d.sensors.hum,
        shock:        d.sensors.shock,
        commDbm:      d.sensors.commDbm,
        commOk:       d.sensors.commDbm != null && d.sensors.commDbm > -115,
        updatedAt:    d.lastMeasured,
        hoursSilent:  d.hoursSilent,
        recentAlarms: d.recentAlarms,
        demo:         true,
      }));
      finalOut = [...out, ...demo];
    }

    res.json({ ok: true, demoMode, site_id: SITE_ID, count: finalOut.length, devices: finalOut });
  } catch (err) {
    console.error("[/api/devices]", err);
    res.status(500).json({ ok: false, error: err.message });
  }
});

// ── GET /api/devices/:id/history — 단말 시계열 추이 ───
//   query: range=1h|24h|7d  (default 24h)  kind=volt|ac|temp|hum|... (default volt)
app.get("/api/devices/:id/history", dbRequired, async (req, res) => {
  try {
    const demoMode = isDemoMode(req);
    const idRaw   = req.params.id;
    const id      = parseInt(idRaw, 10);
    const range   = req.query.range || "24h";
    const kind    = req.query.kind  || "volt";
    const seq     = SENSOR_SEQ_KIND.indexOf(kind) + 1;
    if (seq < 1) return res.status(400).json({ ok: false, error: `unknown kind: ${kind}` });

    const hours   = range === "1h" ? 1 : range === "7d" ? 168 : 24;

    // 데모 단말 (id 가 음수 또는 DEMO-* deviceId 면)
    if (demoMode && (id < 0 || String(idRaw).startsWith("DEMO-"))) {
      const demoDev = id < 0 ? findDemoDeviceByTxid(id) : findDemoDevice(idRaw);
      if (!demoDev) return res.status(404).json({ ok: false, error: "데모 단말 없음" });
      const points = generateDemoHistory(demoDev.deviceId, kind, hours);
      return res.json({
        ok: true, demo: true, device_id: id, deviceId: demoDev.deviceId,
        kind, unit: "", range, count: points.length, points,
      });
    }

    // 단말의 해당 seq SENSOR_ID 찾기 (단말당 SENSOR_ID 정렬 후 seq 번째)
    const [sensorRows] = await pool.query(`
      SELECT SENSOR_ID, UNIT,
             ROW_NUMBER() OVER (PARTITION BY TRANSMITTER_ID ORDER BY SENSOR_ID) AS seq
      FROM kscg_sensor_info WHERE TRANSMITTER_ID = ?
    `, [id]);
    const sensor = sensorRows.find((s) => Number(s.seq) === seq);
    if (!sensor) return res.status(404).json({ ok: false, error: "센서 없음" });

    const [rows] = await pool.query(`
      SELECT WRITE_DATE AS t, VALUE AS v
      FROM kscg_sensor_data
      WHERE SENSOR_ID = ?
        AND WRITE_DATE > DATE_SUB(NOW(), INTERVAL ? HOUR)
      ORDER BY WRITE_DATE
    `, [sensor.SENSOR_ID, hours]);

    res.json({
      ok: true,
      device_id: id, kind, unit: sensor.UNIT, range,
      count: rows.length,
      points: rows,
    });
  } catch (err) {
    console.error("[/api/devices/:id/history]", err);
    res.status(500).json({ ok: false, error: err.message });
  }
});

// ── GET /api/alarms — 최근 알람 ───────────────────────
//   query: limit=20 (default), days=7 (default)
app.get("/api/alarms", dbRequired, async (req, res) => {
  try {
    const demoMode = isDemoMode(req);
    const limit = Math.min(parseInt(req.query.limit || "20", 10), 100);
    const days  = parseInt(req.query.days  || "7", 10);
    const [rows] = await pool.query(`
      SELECT
        a.ALARM_ID    AS id,
        a.GEN_DATE    AS occurredAt,
        a.SENDED_DATE AS sentAt,
        a.GRADE_ID    AS gradeId,
        g.GRADE_TEXT  AS gradeText,
        a.SENSOR_ID   AS sensorId,
        t.NAME        AS deviceId,
        f.NUMBER      AS facilityNum,
        a.VALUE       AS value,
        a.CONTENTS    AS contents,
        a.STATUS      AS status,
        a.TYPE        AS notifyType
      FROM kscg_alarm_log a
      LEFT JOIN kscg_alarm_grade_info g ON g.GRADE_ID = a.GRADE_ID
      LEFT JOIN kscg_sensor_info si ON si.SENSOR_ID = a.SENSOR_ID
      LEFT JOIN kscg_transmitter_info t ON t.TRANSMITTER_ID = si.TRANSMITTER_ID
      LEFT JOIN kscg_facility_info f ON f.TRANSMITTER_ID = t.TRANSMITTER_ID
      LEFT JOIN kscg_site_mydevice m ON m.TRANSMITTER_ID = si.TRANSMITTER_ID
      WHERE a.GEN_DATE > DATE_SUB(NOW(), INTERVAL ? DAY)
      ORDER BY a.GEN_DATE DESC
      LIMIT ?
    `, [days, limit]);
    // 데모 알람 prepend
    let merged = rows;
    if (demoMode) {
      const demoAl = getDemoAlarms().map((a, i) => ({
        id: -100 - i, occurredAt: a.occurredAt, sentAt: a.occurredAt,
        gradeId: a.gradeId, gradeText: a.grade, sensorId: null,
        deviceId: a.deviceId, facilityNum: a.facility,
        value: a.value, contents: a.contents, status: 0, notifyType: "DEMO",
        demo: true,
      }));
      merged = [...demoAl, ...rows].slice(0, limit);
    }
    res.json({ ok: true, count: merged.length, alarms: merged, demoMode });
  } catch (err) {
    console.error("[/api/alarms]", err);
    res.status(500).json({ ok: false, error: err.message });
  }
});

// ── GET /api/anomalies — AI 탐지 (이상/관찰) + 통신두절 분리 ──
//   anomalies = 심각 AI 이상, watch = 경계 이상/관찰, commOutage = 24h+ 무측정
app.get("/api/anomalies", dbRequired, async (req, res) => {
  try {
    const demoMode = isDemoMode(req);
    // AI 예측(ai_predictions) 최신 1건/단말 + 단말명/시설 → 이상/관찰 목록
    const [aiRows] = await pool.query(`
      SELECT t.NAME AS node, f.NUMBER AS facility,
             p.risk_level AS riskLevel, p.mse, p.threshold,
             p.comm_status AS commStatus, p.ai_reliability AS aiReliability,
             p.feature_contributions AS fc, p.predicted_at AS predictedAt
      FROM ai_predictions p
      JOIN (SELECT transmitter_id, MAX(predicted_at) AS mx
            FROM ai_predictions GROUP BY transmitter_id) l
        ON l.transmitter_id = p.transmitter_id AND l.mx = p.predicted_at
      JOIN kscg_transmitter_info t ON t.TRANSMITTER_ID = p.transmitter_id
      JOIN kscg_site_mydevice m ON m.TRANSMITTER_ID = p.transmitter_id AND m.SITE_ID = ?
      LEFT JOIN kscg_facility_info f ON f.TRANSMITTER_ID = p.transmitter_id
    `, [SITE_ID]);

    // ai_predictions row → 프론트 shape (mse/threshold 는 실제 AI 값, label 은 "통신 두절" 로 시작하지 않음)
    const mkAi = (r) => {
      const contribution = contribFromFeatures(r.fc);
      const top = contribution[0]?.sensor;
      const ratio = Number(r.threshold) > 0 ? Number((Number(r.mse) / Number(r.threshold)).toFixed(2)) : null;
      return {
        node: r.node,
        zone: zoneFromFacility(r.facility),
        label: `${top ? top + " " : ""}${r.riskLevel === "이상" ? "이상" : "관찰"}`,
        mse: Number(r.mse),
        threshold: Number(r.threshold),
        riskLevel: r.riskLevel,
        aiReliability: r.aiReliability,
        aiRatio: ratio,
        commStatus: r.commStatus,
        predictedAt: r.predictedAt,
        contribution,
        ts: r.predictedAt,
      };
    };
    // 이두현 모델 등급 그대로: '이상'(≥100%) → anomalies, '관찰'(70~100%) → watch
    const aiAnoms = aiRows.filter((r) => r.riskLevel === "이상").map(mkAi)
                          .sort((a, b) => (b.aiRatio || 0) - (a.aiRatio || 0));
    const aiWatch = aiRows.filter((r) => r.riskLevel === "관찰").map(mkAi)
                          .sort((a, b) => (b.aiRatio || 0) - (a.aiRatio || 0));

    // 통신 24h 두절 — 별도 배열(commOutage). AI mse/threshold 와 섞지 않음.
    const [outageRows] = await pool.query(`
      SELECT t.NAME AS node, f.NUMBER AS facility,
             MAX(r.DATE) AS lastSeen,
             TIMESTAMPDIFF(HOUR, MAX(r.DATE), NOW()) AS hoursSilent
      FROM kscg_transmitter_info t
      JOIN kscg_site_mydevice m ON m.TRANSMITTER_ID = t.TRANSMITTER_ID AND m.SITE_ID = ?
      LEFT JOIN kscg_facility_info f ON f.TRANSMITTER_ID = t.TRANSMITTER_ID
      JOIN kscg_sensor_info si ON si.TRANSMITTER_ID = t.TRANSMITTER_ID
      LEFT JOIN kscg_recent_data r ON r.SENSOR_ID = si.SENSOR_ID
      GROUP BY t.TRANSMITTER_ID, t.NAME, f.NUMBER
      HAVING hoursSilent >= 24
      ORDER BY hoursSilent DESC
      LIMIT 20
    `, [SITE_ID]);
    const commOutage = outageRows.map((r) => ({
      node: r.node, zone: zoneFromFacility(r.facility),
      hoursSilent: Number(r.hoursSilent), lastSeen: r.lastSeen,
      label: `통신 두절 ${fmtHours(r.hoursSilent)}`,
    }));

    // 데모 가산 (critical→anomalies, warn→watch — AI 값 사용)
    let demoAnomalies = [], demoWatch = [], demoOutage = [];
    if (demoMode) {
      const dd = getDemoDevices();
      const mkDemo = (d, rl) => ({
        node: d.deviceId, zone: zoneFromFacility(d.facility),
        label: rl, mse: d.mse, threshold: d.threshold, riskLevel: rl,
        aiReliability: d.aiReliability,
        aiRatio: d.threshold ? Number((d.mse / d.threshold).toFixed(2)) : null,
        contribution: [], ts: d.lastMeasured, demo: true,
      });
      demoAnomalies = dd.filter((d) => d.status === "critical").map((d) => mkDemo(d, "이상"));
      demoWatch     = dd.filter((d) => d.status === "warn").map((d) => mkDemo(d, "관찰"));
      demoOutage    = dd.filter((d) => d.status === "offline").map((d) => ({
        node: d.deviceId,
        zone: zoneFromFacility(d.facility),
        hoursSilent: d.hoursSilent,
        lastSeen: d.lastMeasured,
        label: `통신 두절 ${fmtHours(d.hoursSilent)}`,
        demo: true,
      }));
    }

    res.json({
      ok: true,
      demoMode,
      anomalies: [...demoAnomalies, ...aiAnoms],
      watch: [...demoWatch, ...aiWatch],
      commOutage: [...demoOutage, ...commOutage],
    });
  } catch (err) {
    console.error("[/api/anomalies]", err);
    res.status(500).json({ ok: false, error: err.message });
  }
});

// ── GET /api/insights — AI 조치 권고 (stub, ai_predictions 연동 전) ──
app.get("/api/insights", (_req, res) => {
  res.json({ ok: true, insights: [] });
});

// ── POST /api/audit/login — 로그인 감사 기록 (시스템 로그 표시용) ──
// 로그인은 프론트(authMock, localStorage) 처리라 서버에 흔적이 없음 →
// 성공/실패 시 클라이언트가 이 엔드포인트로 알려 audit_log 에 남긴다.
// ⚠️ 비밀번호는 절대 받지도/저장하지도 않는다. id·name·role·결과만.
app.post("/api/audit/login", async (req, res) => {
  if (!pool) return res.json({ ok: false, error: "DB 비활성" });
  try {
    const b = req.body || {};
    const ok = b.ok === true;
    const action = ok ? "login" : "login_fail";
    const userId = String(b.id || "").slice(0, 60);
    const name   = String(b.name || "").slice(0, 60);
    const role   = String(b.role || "").slice(0, 30);
    const reason = String(b.reason || "").slice(0, 120);  // 실패 사유(비번 외)
    const ip = (req.headers["x-forwarded-for"] || req.socket?.remoteAddress || "").toString().split(",")[0].slice(0, 45);
    const ua = String(req.headers["user-agent"] || "").slice(0, 255);
    await pool.query(
      `INSERT INTO audit_log (action, target_type, target_id, ip, user_agent, metadata_json) VALUES (?, ?, ?, ?, ?, ?)`,
      [action, "auth", userId || "(unknown)", ip, ua, JSON.stringify({ name, role, ok, reason })],
    );
    res.json({ ok: true });
  } catch (err) {
    console.error("[/api/audit/login]", err.message);
    res.json({ ok: false, error: err.message });  // 감사 실패가 로그인 흐름을 막지 않도록 200
  }
});

// ── GET /api/log-events — 시스템 로그 (영구 + 30초 polling) ──
// audit_log (tool_call) + kscg_alarm_log 통합. 시간 역순 LIMIT.
// query:
//   after (ISO datetime): 그 이후 이벤트만 (polling 증분)
//   q (string): text 부분 검색
//   limit (default 100, max 300)
app.get("/api/log-events", dbRequired, async (req, res) => {
  try {
    const after = req.query.after;
    const q     = (req.query.q || "").trim().toLowerCase();
    const limit = Math.min(parseInt(req.query.limit || "100", 10), 300);

    // ── 소스별 "보장 쿼리" ───────────────────────────────────────
    // 문제: audit_log(매분 누적) 가 alarm(드물지만 중요) 을 시간순 정렬에서 가림.
    // 해결: 각 소스를 자기 몫만큼 따로 뽑아 합쳐 한 종류가 다른 종류를 묻지 않게 함.
    //   alarm  — DB 원본 알람 (위험/경고/주의). 항상 최근 N건 확보.
    //   ai     — ai_predictions 분석 갱신 (DB 에 새로 들어온 분석 결과).
    //   sync   — sync_state 동기화 이벤트 (옴니 → siwon 데이터 반영).
    //   auth   — audit_log 로그인/실패 (운영 감사).
    //   tool   — audit_log AI 도구호출. 비중 제한(가장 흔하므로).
    const PER = {
      alarm: Math.max(40, Math.ceil(limit * 0.5)),
      ai:    Math.max(30, Math.ceil(limit * 0.4)),
      sync:  20,
      auth:  20,
      tool:  Math.max(40, Math.ceil(limit * 0.6)),
    };

    // alarm — 최근 30일 (after 증분)
    const [alarms] = await pool.query(`
      SELECT a.ALARM_ID AS id, a.GEN_DATE AS ts, a.CONTENTS, a.VALUE,
             g.GRADE_TEXT AS grade,
             t.NAME AS deviceId
      FROM kscg_alarm_log a
      LEFT JOIN kscg_alarm_grade_info g ON g.GRADE_ID = a.GRADE_ID
      LEFT JOIN kscg_sensor_info si ON si.SENSOR_ID = a.SENSOR_ID
      LEFT JOIN kscg_transmitter_info t ON t.TRANSMITTER_ID = si.TRANSMITTER_ID
      WHERE a.GEN_DATE > DATE_SUB(NOW(), INTERVAL 30 DAY)
        ${after ? "AND a.GEN_DATE > ?" : ""}
      ORDER BY a.GEN_DATE DESC LIMIT ${PER.alarm}
    `, after ? [after] : []);

    // ai — ai_predictions 분석 갱신 (transmitter 이름 조인)
    const [preds] = await pool.query(`
      SELECT p.id, p.created_at AS ts, p.mse, p.threshold, p.risk_level AS risk,
             t.NAME AS deviceId
      FROM ai_predictions p
      LEFT JOIN kscg_transmitter_info t ON t.TRANSMITTER_ID = p.transmitter_id
      WHERE p.created_at > DATE_SUB(NOW(), INTERVAL 7 DAY)
        ${after ? "AND p.created_at > ?" : ""}
      ORDER BY p.created_at DESC LIMIT ${PER.ai}
    `, after ? [after] : []);

    // sync — 옴니→Mac Studio 미러링으로 실제 들어온 "센서 측정값" 이벤트.
    //   sync_log 의 단순 건수 대신, 측정 시각(WRITE_DATE)별 대표 장비 1대의 6종 센서
    //   스냅샷을 보여줘 "어느 장비에서 무슨 값이 들어왔는지" 가 드러나게 함.
    //   각 WRITE_DATE 시각마다 transmitter 하나를 골라(그 시각 데이터가 가장 많은 단말)
    //   센서별 값을 JSON 집계.
    const [syncs] = await pool.query(`
      SELECT g.ts, g.transmitter_id, t.NAME AS deviceId, g.device_cnt,
             JSON_OBJECTAGG(COALESCE(si.NAME, 'sensor'), ROUND(g2.VALUE, 1)) AS readings
      FROM (
        SELECT sd.WRITE_DATE AS ts, si.TRANSMITTER_ID AS transmitter_id,
               COUNT(*) AS device_cnt
        FROM kscg_sensor_data sd
        JOIN kscg_sensor_info si ON si.SENSOR_ID = sd.SENSOR_ID
        ${after ? "WHERE sd.WRITE_DATE > ?" : ""}
        GROUP BY sd.WRITE_DATE, si.TRANSMITTER_ID
        ORDER BY sd.WRITE_DATE DESC
        LIMIT ${PER.sync}
      ) g
      JOIN kscg_sensor_data g2 ON g2.WRITE_DATE = g.ts
      JOIN kscg_sensor_info si ON si.SENSOR_ID = g2.SENSOR_ID AND si.TRANSMITTER_ID = g.transmitter_id
      LEFT JOIN kscg_transmitter_info t ON t.TRANSMITTER_ID = g.transmitter_id
      GROUP BY g.ts, g.transmitter_id, t.NAME, g.device_cnt
      ORDER BY g.ts DESC
      LIMIT ${PER.sync}
    `, after ? [after] : []);

    // tool — audit_log AI 도구호출 (최근 7일, after 증분). action='tool_call' 한정.
    const [audits] = await pool.query(`
      SELECT id, created_at, action, target_id, metadata_json
      FROM audit_log
      WHERE action = 'tool_call'
        AND created_at > DATE_SUB(NOW(), INTERVAL 7 DAY)
        ${after ? "AND created_at > ?" : ""}
      ORDER BY created_at DESC LIMIT ${PER.tool}
    `, after ? [after] : []);

    // auth — audit_log 로그인/실패 (최근 30일, after 증분)
    const [auths] = await pool.query(`
      SELECT id, created_at, action, target_id, metadata_json
      FROM audit_log
      WHERE action IN ('login', 'login_fail')
        AND created_at > DATE_SUB(NOW(), INTERVAL 30 DAY)
        ${after ? "AND created_at > ?" : ""}
      ORDER BY created_at DESC LIMIT ${PER.auth}
    `, after ? [after] : []);

    const fmtTime = (d) => {
      const dt = new Date(d);
      const hh = String(dt.getHours()).padStart(2, "0");
      const mm = String(dt.getMinutes()).padStart(2, "0");
      const ss = String(dt.getSeconds()).padStart(2, "0");
      return `${hh}:${mm}:${ss}`;
    };
    // cursor 용 — DB 로컬시각(KST) 그대로의 'YYYY-MM-DD HH:MM:SS' 문자열.
    // 프론트는 이 값을 파싱 없이 그대로 after= 로 되돌려줘 타임존 변환 오차를 차단.
    const fmtCursor = (d) => {
      const dt = new Date(d);
      const p = (n) => String(n).padStart(2, "0");
      return `${dt.getFullYear()}-${p(dt.getMonth() + 1)}-${p(dt.getDate())} `
           + `${p(dt.getHours())}:${p(dt.getMinutes())}:${p(dt.getSeconds())}`;
    };

    // alarm → 이벤트 (위험=alert, 그 외=warn)
    const alarmEvents = alarms.map((a) => ({
      id:   `alm-${a.id}`,
      ts:   a.ts,
      cur:  fmtCursor(a.ts),
      time: fmtTime(a.ts),
      kind: a.grade === "위험" ? "alert" : "warn",
      text: `${a.grade || "ALARM"}: ${a.deviceId || "(unknown)"} · ${a.CONTENTS || ""} · 값 ${a.VALUE != null ? Number(a.VALUE).toFixed(2) : "-"}`,
      source: "alarm",
    }));

    // ai → 이벤트 (이상=alert, 관찰=warn, 그 외=ai). 임계 대비 배수 표기.
    const aiEvents = preds.map((p) => {
      const ratio = (p.threshold > 0 && p.mse != null) ? (p.mse / p.threshold) : null;
      const ratioTxt = ratio != null ? ` · 임계 ×${ratio.toFixed(2)}` : "";
      const kind = p.risk === "이상" ? "alert" : p.risk === "관찰" ? "warn" : "ai";
      return {
        id:   `ai-${p.id}`,
        ts:   p.ts,
        cur:  fmtCursor(p.ts),
        time: fmtTime(p.ts),
        kind,
        text: `AI 분석: ${p.deviceId || `#${p.id}`} · ${p.risk || "정상"}${ratioTxt}`,
        source: "ai",
      };
    });

    // sync → 이벤트 (옴니 KSCG → Mac Studio MySQL 미러링: 실제 측정값 스냅샷)
    //   "미러링: TB24-XXX 측정 수신 · 방식전위 -71mV · AC유입 2309mV · 온도 27.4℃ …"
    //   센서종류별 단위 매핑(없는 종류는 mV 가정). 주요 6종만 우선 노출.
    const SENSOR_UNIT = { 방식전위:"mV", 방식전류:"mA", AC유입:"mV", 온도:"℃", 습도:"%", 배터리:"mV", 수신감도:"dBm", 가스누출:"%LEL", 수위:"", 충격:"" };
    const SENSOR_ORDER = ["방식전위", "AC유입", "온도", "습도", "방식전류", "배터리"];
    const syncEvents = syncs.map((s) => {
      let readings = {};
      try { readings = typeof s.readings === "string" ? JSON.parse(s.readings) : (s.readings || {}); } catch {}
      // 주요 센서 우선 정렬 후 나머지, 최대 5개
      const allKeys = Object.keys(readings).sort((a, b) => {
        const ia = SENSOR_ORDER.indexOf(a), ib = SENSOR_ORDER.indexOf(b);
        return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib);
      });
      const keys = allKeys.slice(0, 4);   // 주요 4종 노출
      const parts = keys.map((k) => {
        const u = SENSOR_UNIT[k] ?? "";
        return `${k} ${readings[k]}${u}`;
      });
      const dev = s.deviceId || `#${s.transmitter_id}`;
      const more = allKeys.length > keys.length ? ` 외 ${allKeys.length - keys.length}종` : "";
      return {
        id:   `syn-${s.transmitter_id}-${fmtCursor(s.ts)}`,
        ts:   s.ts,
        cur:  fmtCursor(s.ts),
        time: fmtTime(s.ts),
        kind: "data",
        text: parts.length
          ? `미러링: ${dev} 측정 수신 · ${parts.join(" · ")}${more}`
          : `미러링: ${dev} 측정 수신`,
        source: "sync",
      };
    });

    // tool → 이벤트 (도구 호출)
    const toolEvents = audits.map((a) => {
      let meta = {};
      try { meta = a.metadata_json ? (typeof a.metadata_json === "string" ? JSON.parse(a.metadata_json) : a.metadata_json) : {}; } catch {}
      const argsTxt = meta.args ? Object.entries(meta.args).slice(0, 2).map(([k, v]) => `${k}:${String(v).slice(0, 16)}`).join(",") : "";
      const dur = meta.durationMs != null ? ` · ${meta.durationMs}ms` : "";
      const ok  = meta.ok === false ? " · ✗ 실패" : "";
      return {
        id:   `aud-${a.id}`,
        ts:   a.created_at,
        cur:  fmtCursor(a.created_at),
        time: fmtTime(a.created_at),
        kind: meta.ok === false ? "warn" : "ai",
        text: `도구: ${a.target_id}(${argsTxt})${dur}${ok}`,
        source: "tool",
      };
    });

    // auth → 이벤트 (로그인 성공/실패)
    const authEvents = auths.map((a) => {
      let meta = {};
      try { meta = a.metadata_json ? (typeof a.metadata_json === "string" ? JSON.parse(a.metadata_json) : a.metadata_json) : {}; } catch {}
      const who = meta.name ? `${meta.name}${meta.role ? `(${meta.role})` : ""}` : (a.target_id || "(unknown)");
      const isFail = a.action === "login_fail";
      return {
        id:   `auth-${a.id}`,
        ts:   a.created_at,
        cur:  fmtCursor(a.created_at),
        time: fmtTime(a.created_at),
        kind: isFail ? "warn" : "auth",
        text: isFail
          ? `⚠ 로그인 실패: ${who}${meta.reason ? ` · ${meta.reason}` : ""}`
          : `로그인: ${who}`,
        source: "auth",
      };
    });

    // nextCursor — 조회된 모든 소스의 최대 cur (KST DATETIME 문자열).
    // 반환 events 가 아닌 "조회 풀 전체"의 max 를 써야 limit 밖으로 밀린 행이
    // 다음 폴링에서 누락되지 않음. 프론트는 이 값을 파싱 없이 그대로 after= 로 전달.
    const allEvents = [...alarmEvents, ...aiEvents, ...syncEvents, ...authEvents, ...toolEvents];
    const nextCursor = allEvents.reduce((mx, e) => (e.cur && (!mx || e.cur > mx) ? e.cur : mx), after || null);

    // 검색어가 있으면 단순 통합·필터 (보장 불필요 — 검색은 전체 대상)
    const byTs = (x, y) => new Date(y.ts).getTime() - new Date(x.ts).getTime();
    if (q) {
      let merged = allEvents
        .filter((e) => e.text.toLowerCase().includes(q))
        .sort(byTs)
        .slice(0, limit);
      return res.json({ ok: true, count: merged.length, events: merged, nextCursor });
    }

    // 보장 병합: 단순 시간순 cut 은 자주 발생하는 tool 이 드문 alarm 을 밀어냄.
    // → 소스별 "최소 확보분"을 먼저 떼고, 남는 자리를 나머지에서 시간순으로 채움.
    const GUARANTEE = { alarm: 12, auth: 10, sync: 8, ai: 12, tool: 0 };  // 각 소스 최소 노출 건수
    const pools = {
      alarm: alarmEvents.slice().sort(byTs),
      ai:    aiEvents.slice().sort(byTs),
      sync:  syncEvents.slice().sort(byTs),
      auth:  authEvents.slice().sort(byTs),
      tool:  toolEvents.slice().sort(byTs),
    };
    const picked = [];
    const usedIds = new Set();
    // 1단계 — 소스별 최소분 확보
    for (const [s, g] of Object.entries(GUARANTEE)) {
      for (const e of pools[s].slice(0, g)) {
        if (!usedIds.has(e.id)) { usedIds.add(e.id); picked.push(e); }
      }
    }
    // 2단계 — 남는 자리를 전체에서 시간순으로 채움
    const rest = allEvents
      .filter((e) => !usedIds.has(e.id))
      .sort(byTs);
    for (const e of rest) {
      if (picked.length >= limit) break;
      usedIds.add(e.id); picked.push(e);
    }
    // 최종 표시 순서는 시간 역순
    const events = picked.sort(byTs).slice(0, limit);

    res.json({ ok: true, count: events.length, events, nextCursor });
  } catch (err) {
    console.error("[/api/log-events]", err);
    res.status(500).json({ ok: false, error: err.message });
  }
});

// ── GET /api/admin/tool-stats — 도구 호출 통계 (audit_log 집계) ──
app.get("/api/admin/tool-stats", dbRequired, async (req, res) => {
  try {
    const days = Math.min(Math.max(parseInt(req.query.days || "7", 10), 1), 90);
    const [rows] = await pool.query(`
      SELECT target_id AS tool,
             COUNT(*) AS calls,
             SUM(JSON_EXTRACT(metadata_json, '$.ok') = true) AS ok,
             SUM(JSON_EXTRACT(metadata_json, '$.cached') = true) AS cached,
             AVG(JSON_EXTRACT(metadata_json, '$.durationMs')) AS avgMs,
             MAX(JSON_EXTRACT(metadata_json, '$.durationMs')) AS maxMs
      FROM audit_log
      WHERE action = 'tool_call'
        AND created_at > DATE_SUB(NOW(), INTERVAL ? DAY)
      GROUP BY target_id
      ORDER BY calls DESC
    `, [days]);
    const totals = rows.reduce((a, r) => {
      a.calls  += Number(r.calls)  || 0;
      a.ok     += Number(r.ok)     || 0;
      a.cached += Number(r.cached) || 0;
      return a;
    }, { calls: 0, ok: 0, cached: 0 });
    res.json({
      ok: true,
      days,
      totals,
      tools: rows.map((r) => ({
        tool:   r.tool,
        calls:  Number(r.calls)  || 0,
        ok:     Number(r.ok)     || 0,
        cached: Number(r.cached) || 0,
        avgMs:  r.avgMs != null ? Math.round(Number(r.avgMs)) : null,
        maxMs:  r.maxMs != null ? Math.round(Number(r.maxMs)) : null,
      })),
    });
  } catch (err) {
    console.error("[/api/admin/tool-stats]", err);
    res.status(500).json({ ok: false, error: err.message });
  }
});

// USD→KRW 실시간 환율 (6시간 캐시 + 실패 시 마지막값/기본값 폴백). 외부 무료 API.
let _fxCache = { rate: 1380, at: 0 };
async function getUsdKrw() {
  if (Date.now() - _fxCache.at < 6 * 3600 * 1000) return _fxCache.rate;
  try {
    const r = await fetch("https://open.er-api.com/v6/latest/USD", { signal: AbortSignal.timeout(4000) });
    const d = await r.json();
    const krw = Number(d?.rates?.KRW);
    if (krw > 500 && krw < 3000) { _fxCache = { rate: Math.round(krw), at: Date.now() }; }
  } catch { /* 네트워크 실패 → 마지막값 유지 */ }
  return _fxCache.rate;
}

// ── GET /api/admin/token-usage — AI 모델별 토큰 사용량 (chat_messages 집계, admin 전용) ──
app.get("/api/admin/token-usage", dbRequired, requireAdmin, async (_req, res) => {
  try {
    const [rows] = await pool.query(`
      SELECT COALESCE(NULLIF(model, ''), '(미지정)') AS model,
             COUNT(*) AS messages,
             COALESCE(SUM(tokens_prompt), 0)                      AS prompt_tok,
             COALESCE(SUM(tokens_completion), 0)                  AS compl_tok,
             COALESCE(SUM(tokens_prompt + tokens_completion), 0)  AS total_tok,
             MAX(created_at) AS last_used
      FROM chat_messages
      WHERE role = 'ai'
      GROUP BY model
      ORDER BY total_tok DESC
    `);
    const provOf = (m) =>
      /^gpt/i.test(m) ? "OpenAI"
      : /(qwen|llama|mistral|gemma|phi|deepseek)/i.test(m) ? "Ollama(로컬)"
      : "기타";
    // OpenAI 요율 (USD / 1M 토큰, in=입력 out=출력). est:true = 추정 요율(공개가 미확정).
    //   실제 단가가 다르면 이 값만 수정하면 비용에 즉시 반영됨.
    const PRICING = {
      "gpt-4o-mini": { in: 0.15, out: 0.60 },
      "gpt-4o":      { in: 2.50, out: 10.00 },
      "gpt-5":       { in: 1.25, out: 10.00, est: true },
      "gpt-5.5":     { in: 1.50, out: 12.00, est: true },
      "gpt-5-mini":  { in: 0.25, out: 2.00,  est: true },
      "gpt-5-nano":  { in: 0.05, out: 0.40,  est: true },
    };
    const byModel = rows.map((r) => {
      const provider = provOf(r.model);
      const prompt = Number(r.prompt_tok) || 0;
      const completion = Number(r.compl_tok) || 0;
      const pr = PRICING[r.model];
      let rate = null, costUsd = null;
      if (provider === "OpenAI" && pr) {
        rate = { in: pr.in, out: pr.out, est: !!pr.est };
        costUsd = (prompt / 1e6) * pr.in + (completion / 1e6) * pr.out;
      } else if (provider !== "OpenAI") {
        costUsd = 0; // 로컬(Ollama) = 무료
      }
      return {
        model: r.model, provider,
        messages: Number(r.messages) || 0,
        prompt, completion,
        total: Number(r.total_tok) || 0,
        lastUsed: r.last_used,
        rate, costUsd,
      };
    });
    const totals = byModel.reduce((a, m) => {
      a.tokens += m.total; a.prompt += m.prompt; a.completion += m.completion; a.messages += m.messages; return a;
    }, { tokens: 0, prompt: 0, completion: 0, messages: 0, models: byModel.length });
    const provMap = {};
    for (const m of byModel) {
      const p = provMap[m.provider] || (provMap[m.provider] = { provider: m.provider, tokens: 0, messages: 0 });
      p.tokens += m.total; p.messages += m.messages;
    }
    const byProvider = Object.values(provMap).sort((a, b) => b.tokens - a.tokens);
    // 모델 목록을 제공자별로 묶어 내림차순 정렬 (제공자=토큰합 큰 순 → 그 안에서 모델 토큰 큰 순)
    const provRank = new Map(byProvider.map((p, i) => [p.provider, i]));
    byModel.sort((a, b) => (provRank.get(a.provider) - provRank.get(b.provider)) || (b.total - a.total));
    const [daily] = await pool.query(`
      SELECT DATE_FORMAT(created_at, '%Y-%m-%d') AS day,
             COALESCE(SUM(tokens_prompt + tokens_completion), 0) AS tokens
      FROM chat_messages
      WHERE role = 'ai' AND created_at > DATE_SUB(NOW(), INTERVAL 14 DAY)
      GROUP BY day ORDER BY day ASC
    `);
    const USD_KRW = await getUsdKrw(); // 실시간 환율(6h 캐시·폴백 1380)
    const openaiModels = byModel.filter((m) => m.provider === "OpenAI");
    const costUsd = openaiModels.reduce((a, m) => a + (m.costUsd || 0), 0);
    const cost = {
      usd: costUsd,
      krw: Math.round(costUsd * USD_KRW),
      fx: USD_KRW,
      hasUnpriced: openaiModels.some((m) => m.costUsd == null),
      hasEstimated: openaiModels.some((m) => m.rate && m.rate.est),
    };
    res.json({
      ok: true, totals, byProvider, byModel, cost,
      daily: daily.map((d) => ({ day: d.day, tokens: Number(d.tokens) || 0 })),
    });
  } catch (err) {
    console.error("[/api/admin/token-usage]", err);
    res.status(500).json({ ok: false, error: err.message });
  }
});

// ── GET /api/admin/login-log — 로그인 감사 로그 (admin 전용) ──
//   audit_log 의 login/login_fail 이벤트 + 계정별 집계. 침입 시도 가시화용.
app.get("/api/admin/login-log", dbRequired, requireAdmin, async (req, res) => {
  try {
    const limit = Math.min(Math.max(parseInt(req.query.limit, 10) || 100, 1), 500);
    const [events] = await pool.query(
      `SELECT created_at AS ts, action, target_id AS account, ip,
              JSON_UNQUOTE(JSON_EXTRACT(metadata_json,'$.name'))   AS name,
              JSON_UNQUOTE(JSON_EXTRACT(metadata_json,'$.role'))   AS role,
              JSON_UNQUOTE(JSON_EXTRACT(metadata_json,'$.reason')) AS reason,
              user_agent AS ua
         FROM audit_log
        WHERE action IN ('login','login_fail') AND target_type = 'auth'
        ORDER BY created_at DESC
        LIMIT ${limit}`,
    );
    const [summary] = await pool.query(
      `SELECT target_id AS account,
              CAST(SUM(action = 'login')      AS UNSIGNED) AS success,
              CAST(SUM(action = 'login_fail') AS UNSIGNED) AS fail,
              COUNT(DISTINCT ip) AS ips,
              MAX(created_at)    AS lastAttempt
         FROM audit_log
        WHERE action IN ('login','login_fail') AND target_type = 'auth'
        GROUP BY target_id
        ORDER BY fail DESC, lastAttempt DESC`,
    );
    res.json({ ok: true, events, summary });
  } catch (err) {
    console.error("[/api/admin/login-log]", err.message);
    res.status(500).json({ ok: false, error: err.message });
  }
});

// ── 챗봇 세션 관리 ───────────────────────────────────
// GET    /api/chat/sessions          — 세션 목록 (최근 30)
// GET    /api/chat/sessions/:id      — 세션 + 메시지
// POST   /api/chat/sessions          — 새 세션
// DELETE /api/chat/sessions/:id      — 세션 삭제

app.get("/api/chat/sessions", dbRequired, async (req, res) => {
  try {
    const c = authClaims(req);
    const owner = chatOwner(req);
    const adminAll = c?.role === "admin" && req.query.scope === "all";   // 관리자 통계 = 전역
    if (!adminAll && owner == null) return res.json({ ok: true, count: 0, sessions: [] });  // 익명/공유는 서버 목록 없음(로컬)
    const ownerWhere = adminAll ? "" : "AND s.user_id = ?";
    const params = adminAll ? [] : [owner];
    const [rows] = await pool.query(`
      SELECT s.id, s.title, s.pinned, s.created_at, s.updated_at,
             (SELECT COUNT(*) FROM chat_messages WHERE session_id = s.id) AS messageCount
      FROM chat_sessions s
      WHERE s.deleted_at IS NULL
        AND EXISTS (SELECT 1 FROM chat_messages WHERE session_id = s.id)
        ${ownerWhere}
      ORDER BY s.pinned DESC, s.updated_at DESC LIMIT 30
    `, params);
    res.json({ ok: true, count: rows.length, sessions: rows });
  } catch (err) {
    console.error("[/api/chat/sessions]", err);
    res.status(500).json({ ok: false, error: err.message });
  }
});

// GET /api/chat/search?q=... — 세션 제목 + 메시지 본문 통합 검색.
//   본문이 매칭되면 가장 최근 매칭 메시지에서 스니펫(LEFT 1000자) 동봉. titleMatch=1/0.
app.get("/api/chat/search", dbRequired, async (req, res) => {
  try {
    const c = authClaims(req);
    const owner = chatOwner(req);
    const adminAll = c?.role === "admin" && req.query.scope === "all";
    if (!adminAll && owner == null) return res.json({ ok: true, count: 0, sessions: [] });
    const q = String(req.query.q || "").trim().slice(0, 100);
    if (!q) return res.json({ ok: true, count: 0, sessions: [] });
    const esc = q.replace(/[\\%_]/g, (ch) => "\\" + ch);   // LIKE 와일드카드 이스케이프(기본 escape '\')
    const like = `%${esc}%`;
    const ownerWhere = adminAll ? "" : "AND s.user_id = ?";
    const params = adminAll ? [like, like, like, like] : [like, like, like, like, owner];
    const [rows] = await pool.query(`
      SELECT s.id, s.title, s.pinned, s.created_at, s.updated_at,
             (SELECT COUNT(*) FROM chat_messages WHERE session_id = s.id) AS messageCount,
             (s.title LIKE ?) AS titleMatch,
             (SELECT LEFT(m.text, 1000) FROM chat_messages m
                WHERE m.session_id = s.id AND m.text LIKE ?
                ORDER BY m.created_at DESC, m.id DESC LIMIT 1) AS matchSnippet
      FROM chat_sessions s
      WHERE s.deleted_at IS NULL
        AND (s.title LIKE ?
         OR EXISTS (SELECT 1 FROM chat_messages m WHERE m.session_id = s.id AND m.text LIKE ?))
        ${ownerWhere}
      ORDER BY s.pinned DESC, s.updated_at DESC
      LIMIT 50
    `, params);
    res.json({ ok: true, count: rows.length, sessions: rows });
  } catch (err) {
    console.error("[/api/chat/search]", err);
    res.status(500).json({ ok: false, error: err.message });
  }
});

// GET /api/chat/sessions/current — 로그인 개인 계정의 현재(최근) 세션 + 메시지. 계정 마운트 로드용.
//   (반드시 /:id 보다 먼저 등록 — 안 그러면 "current"가 :id 로 파싱됨)
app.get("/api/chat/sessions/current", dbRequired, async (req, res) => {
  try {
    const owner = chatOwner(req);
    if (owner == null) return res.json({ ok: true, session: null, messages: [] });   // 익명/공유 → 프론트가 localStorage 폴백
    const [sess] = await pool.query(
      `SELECT * FROM chat_sessions WHERE user_id = ? AND deleted_at IS NULL ORDER BY updated_at DESC LIMIT 1`, [owner]);
    if (sess.length === 0) return res.json({ ok: true, session: null, messages: [] });
    const [msgs] = await pool.query(`
      SELECT role, text, tokens_prompt AS tokensPrompt, tokens_completion AS tokensCompletion,
             model, created_at AS createdAt
      FROM chat_messages WHERE session_id = ?
      ORDER BY created_at, (role = 'ai'), id LIMIT 200
    `, [sess[0].id]);
    res.json({ ok: true, session: sess[0], messages: msgs });
  } catch (err) {
    console.error("[/api/chat/sessions/current]", err);
    res.status(500).json({ ok: false, error: err.message });
  }
});

app.get("/api/chat/sessions/:id", dbRequired, async (req, res) => {
  try {
    const id = parseInt(req.params.id, 10);
    if (!Number.isFinite(id)) return res.status(400).json({ ok: false, error: "id 숫자" });
    const [sess] = await pool.query(`SELECT * FROM chat_sessions WHERE id = ? AND deleted_at IS NULL`, [id]);
    if (sess.length === 0) return res.status(404).json({ ok: false, error: "session not found" });
    // 소유권 가드 — user_id 있는(개인 계정) 세션은 소유자/admin 만. 익명(NULL) 세션은 기존대로 id 로 접근 가능.
    const cClaim = authClaims(req);
    if (sess[0].user_id != null && sess[0].user_id !== cClaim?.uid && cClaim?.role !== "admin") {
      return res.status(404).json({ ok: false, error: "session not found" });
    }
    const [msgs] = await pool.query(`
      SELECT role, text, tokens_prompt AS tokensPrompt, tokens_completion AS tokensCompletion,
             model, created_at AS createdAt
      FROM chat_messages WHERE session_id = ?
      ORDER BY created_at, (role = 'ai'), id LIMIT 200
    `, [id]);   // created_at 초단위 동률이면 user 를 ai 보다 먼저 (질문→답변 순)
    res.json({ ok: true, session: sess[0], messages: msgs });
  } catch (err) {
    console.error("[/api/chat/sessions/:id]", err);
    res.status(500).json({ ok: false, error: err.message });
  }
});

app.post("/api/chat/sessions", dbRequired, async (req, res) => {
  try {
    const title = String(req.body?.title || "").slice(0, 200) || "(제목 없음)";
    const [r] = await pool.query(`INSERT INTO chat_sessions (title) VALUES (?)`, [title]);
    res.json({ ok: true, sessionId: r.insertId, title });
  } catch (err) {
    console.error("[POST /api/chat/sessions]", err);
    res.status(500).json({ ok: false, error: err.message });
  }
});

app.delete("/api/chat/sessions/:id", dbRequired, async (req, res) => {
  try {
    const id = parseInt(req.params.id, 10);
    if (!Number.isFinite(id)) return res.status(400).json({ ok: false, error: "id 숫자" });
    // 소유권 가드 — 개인 계정 세션은 소유자/admin 만 삭제 가능
    const cClaim = authClaims(req);
    const [own] = await pool.query(`SELECT user_id FROM chat_sessions WHERE id = ? AND deleted_at IS NULL`, [id]);
    if (own.length && own[0].user_id != null && own[0].user_id !== cClaim?.uid && cClaim?.role !== "admin") {
      return res.status(404).json({ ok: false, error: "session not found" });
    }
    // 소프트 삭제 — 행/메시지는 보존하고 deleted_at 만 찍어 목록·검색에서 숨김 (DB 영구 보존, 복구 가능)
    const [r] = await pool.query(`UPDATE chat_sessions SET deleted_at = NOW() WHERE id = ? AND deleted_at IS NULL`, [id]);
    res.json({ ok: true, deleted: r.affectedRows });
  } catch (err) {
    console.error("[DELETE /api/chat/sessions/:id]", err);
    res.status(500).json({ ok: false, error: err.message });
  }
});

// PATCH /api/chat/sessions/:id — 세션 제목 변경 (rename)
app.patch("/api/chat/sessions/:id", dbRequired, async (req, res) => {
  try {
    const id = parseInt(req.params.id, 10);
    if (!Number.isFinite(id)) return res.status(400).json({ ok: false, error: "id 숫자" });
    // 소유권 가드 — 개인 계정 세션은 소유자/admin 만 변경 가능
    const cClaim = authClaims(req);
    const [own] = await pool.query(`SELECT user_id FROM chat_sessions WHERE id = ? AND deleted_at IS NULL`, [id]);
    if (own.length && own[0].user_id != null && own[0].user_id !== cClaim?.uid && cClaim?.role !== "admin") {
      return res.status(404).json({ ok: false, error: "session not found" });
    }
    // title / pinned 둘 중 보낸 것만 부분 업데이트
    const sets = [];
    const params = [];
    if (req.body?.title != null) {
      const title = String(req.body.title).trim().slice(0, 60);
      if (!title) return res.status(400).json({ ok: false, error: "title 필요" });
      sets.push("title = ?"); params.push(title);
    }
    if (req.body?.pinned != null) {
      sets.push("pinned = ?"); params.push(req.body.pinned ? 1 : 0);
    }
    if (sets.length === 0) return res.status(400).json({ ok: false, error: "title 또는 pinned 필요" });
    params.push(id);
    const [r] = await pool.query(`UPDATE chat_sessions SET ${sets.join(", ")} WHERE id = ?`, params);
    if (r.affectedRows === 0) return res.status(404).json({ ok: false, error: "세션 없음" });
    res.json({ ok: true, id });
  } catch (err) {
    console.error("[PATCH /api/chat/sessions/:id]", err);
    res.status(500).json({ ok: false, error: err.message });
  }
});

// ── 문의 채널 (inquiries) — 관리자 문의 + 개발자 문의(포폴) ───────────────
// 관리자 문의: 고객지원 봇(문의/버그). 개발자 문의: 프로젝트 지식베이스로 기능 설명 봇.
const SUPPORT_SYSTEM = `당신은 '군산도시가스 매설배관 AI 통합관제 시스템'의 고객지원 봇입니다. 사용자가 보낸 문의 또는 버그 신고에 한국어로 친절하고 간결하게(3~5문장) 답하세요.
- 먼저 접수되어 관리자에게 전달되었음을 알리세요.
- 사용법·기능 질문이면 아는 선에서 도와주세요(대시보드/지도/장비목록/AI 챗봇 등).
- 버그 신고면 어떤 화면·동작에서 생겼는지 정중히 한 번 확인 요청하고, 확인 후 조치하겠다고 안내.
- 센서·단말 데이터 분석 요청이면 메인 AI 챗봇(일반 대화)에서 도와준다고 안내.
- 모르면 모른다고 하고 관리자 확인이 필요하다고 안내. 추측·과장 금지.`;

// 개발자 문의(포폴) — 프로젝트 지식베이스(옵시디언 노트 정리본)를 주입
let PROJECT_KNOWLEDGE = "";
try { PROJECT_KNOWLEDGE = readFileSync(path.join(__dirname, "project-knowledge.md"), "utf8"); }
catch (e) { console.warn("[project-knowledge] 로드 실패:", e.message); }
const DEV_SYSTEM = `당신은 '매설배관 AI 통합관제 시스템'(호서대 캡스톤, 시원팀) 프로젝트를 소개하는 개발자 어시스턴트입니다.
아래 [프로젝트 지식]에 근거해, 이 프로젝트가 무엇이고 어떻게 동작하는지 한국어로 친절하고 구체적으로 설명하세요(포트폴리오 소개 톤).
- 기능·아키텍처·기술스택·AI(이상탐지)·챗봇·도메인 용어 질문에 자세히 답하세요.
- 지식에 없는 내용은 지어내지 말고, "그 부분은 문서에 없어 개발자(박지훈)가 직접 확인·답변할 수 있다"고 안내하세요.
- 비밀번호·서버 접속·내부 IP·DB 자격증명 등 보안 정보는 절대 공개하지 마세요.

[프로젝트 지식]
${PROJECT_KNOWLEDGE}`;

// 개발자 문의(포폴 Q&A) 답변 모델 — 품질 우선으로 GPT 사용. 키 없거나 실패 시 로컬 폴백.
const INQUIRY_DEV_MODEL = process.env.INQUIRY_DEV_MODEL || "gpt-4o-mini";

async function runInquiryReply(target, kind, message) {
  const isDev = target === "developer";
  const system = isDev ? DEV_SYSTEM : SUPPORT_SYSTEM;
  const userContent = isDev ? message : `[${kind === "bug" ? "버그 신고" : "문의"}]\n${message}`;
  const useGpt = isDev && isOpenAI(INQUIRY_DEV_MODEL) && OPENAI_API_KEY;
  try {
    // 1) 개발자 문의 = GPT (OpenAI) 우선
    if (useGpt) {
      const res = await fetch(OPENAI_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${OPENAI_API_KEY}` },
        body: JSON.stringify({
          model: INQUIRY_DEV_MODEL,
          messages: [{ role: "system", content: system }, { role: "user", content: userContent }],
        }),  // temperature·max_tokens 생략 — gpt-4o-mini/5.x 호환
        signal: AbortSignal.timeout(40_000),
      });
      if (res.ok) {
        const data = await res.json();
        const out = (data.choices?.[0]?.message?.content || "").trim();
        if (out) return out;
      } else {
        console.warn("[runInquiryReply] OpenAI HTTP", res.status, "→ 로컬 폴백");
      }
      // 실패 시 아래 로컬로 폴백
    }
    // 2) 로컬 Ollama (상담원 문의 + 개발자 폴백)
    const res = await fetch(`${OLLAMA_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: OLLAMA_MODEL,
        messages: [{ role: "system", content: system }, { role: "user", content: userContent }],
        stream: false, think: false,
        options: { temperature: isDev ? 0.5 : 0.4, num_predict: isDev ? 700 : 400 },
      }),
      signal: AbortSignal.timeout(30_000),   // 30초 초과 시 접수 확인 폴백
    });
    if (!res.ok) return null;
    const data = await res.json();
    return (data.message?.content || "").trim() || null;
  } catch (e) {
    console.warn("[runInquiryReply]", e.message);
    return null;
  }
}

// inquiries.image(JSON 배열 또는 레거시 단일 data URL) → 배열로 파싱
function parseInquiryImages(v) {
  if (!v) return [];
  if (typeof v === "string" && v[0] === "[") { try { const a = JSON.parse(v); return Array.isArray(a) ? a : []; } catch { return []; } }
  return [v];
}

// POST /api/inquiries (로그인) — 문의 접수 + AI 지원 답변
app.post("/api/inquiries", requireAuth, dbRequired, async (req, res) => {
  try {
    const target = req.body?.target === "developer" ? "developer" : "admin";
    const kind = req.body?.kind === "bug" ? "bug" : "question";
    const message = String(req.body?.message || "").trim().slice(0, 4000);
    // 첨부 이미지 배열(data URL, png/jpeg/webp만, 최대 5장). 형식 불량 제외 + 총 용량 가드.
    let images = Array.isArray(req.body?.images) ? req.body.images : (req.body?.image ? [req.body.image] : []);
    images = images.filter((x) => typeof x === "string" && /^data:image\/(png|jpe?g|webp);base64,/.test(x)).slice(0, 5);
    while (images.length && images.reduce((a, x) => a + x.length, 0) > 12_000_000) images.pop();
    const imageJson = images.length ? JSON.stringify(images) : null;
    if (!message && !images.length) return res.status(400).json({ ok: false, error: "내용을 입력하세요." });
    const c = req.auth;
    const replyQuote = String(req.body?.replyQuote || "").trim().slice(0, 500) || null;
    const [r] = await pool.query(
      `INSERT INTO inquiries (target, user_id, login_id, display_name, kind, message, reply_quote, image, status, ip) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)`,
      [target, c.uid, c.lid, c.name || null, kind, message, replyQuote, imageJson, reqIp(req)],
    );
    const reply = await runInquiryReply(target, kind, message);
    if (reply) await pool.query(`UPDATE inquiries SET bot_reply = ? WHERE id = ?`, [reply, r.insertId]);
    // 같은 계정 다른 화면에 실시간 반영 (발신 화면 제외)
    broadcastToOwner(c.uid, { type: "inquiry:new", target, kind, message, reply: reply || null, ts: Date.now() }, req.body?.clientId || null);
    const fallback = target === "developer"
      ? "질문이 접수되었습니다. 개발자가 직접 확인 후 답변해 드릴게요. 🙇"
      : "문의가 접수되었습니다. 관리자가 확인 후 반영하겠습니다. 🙇";
    res.json({ ok: true, id: r.insertId, reply: reply || fallback });
  } catch (err) {
    console.error("[POST /api/inquiries]", err);
    res.status(500).json({ ok: false, error: err.message });
  }
});

// GET /api/inquiries/mine (로그인) — 본인 문의 내역 (문의방 렌더용)
app.get("/api/inquiries/mine", requireAuth, dbRequired, async (req, res) => {
  try {
    const target = (req.query.target === "developer" || req.query.target === "admin") ? req.query.target : null;
    const [rows] = await pool.query(
      `SELECT id, target, kind, message, reply_quote AS replyQuote, image, bot_reply AS botReply, status, admin_reply AS adminReply, created_at AS createdAt
       FROM inquiries WHERE user_id = ? ${target ? "AND target = ?" : ""} ORDER BY created_at, id LIMIT 200`,
      target ? [req.auth.uid, target] : [req.auth.uid]);
    res.json({ ok: true, count: rows.length, inquiries: rows.map(({ image, ...rest }) => ({ ...rest, images: parseInquiryImages(image) })) });
  } catch (err) {
    console.error("[GET /api/inquiries/mine]", err);
    res.status(500).json({ ok: false, error: err.message });
  }
});

// GET /api/inquiries (admin) — 전체 문의 목록
app.get("/api/inquiries", requireAdmin, dbRequired, async (req, res) => {
  try {
    const status = (req.query.status === "open" || req.query.status === "done") ? req.query.status : null;
    const target = (req.query.target === "developer" || req.query.target === "admin") ? req.query.target : null;
    const where = [], params = [];
    if (status) { where.push("status = ?"); params.push(status); }
    if (target) { where.push("target = ?"); params.push(target); }
    const [rows] = await pool.query(
      `SELECT id, target, user_id AS userId, login_id AS loginId, display_name AS displayName,
              kind, message, image, bot_reply AS botReply, status, admin_reply AS adminReply,
              created_at AS createdAt, updated_at AS updatedAt
       FROM inquiries ${where.length ? "WHERE " + where.join(" AND ") : ""} ORDER BY (status='done'), created_at DESC LIMIT 500`,
      params);
    res.json({ ok: true, count: rows.length, inquiries: rows.map(({ image, ...rest }) => ({ ...rest, images: parseInquiryImages(image) })) });
  } catch (err) {
    console.error("[GET /api/inquiries]", err);
    res.status(500).json({ ok: false, error: err.message });
  }
});

// PATCH /api/inquiries/:id (admin) — 상태/답변
app.patch("/api/inquiries/:id", requireAdmin, dbRequired, async (req, res) => {
  try {
    const id = parseInt(req.params.id, 10);
    if (!Number.isFinite(id)) return res.status(400).json({ ok: false, error: "id 숫자" });
    const sets = [], params = [];
    if (req.body?.status === "open" || req.body?.status === "done") { sets.push("status = ?"); params.push(req.body.status); }
    if (req.body?.adminReply != null) { sets.push("admin_reply = ?"); params.push(String(req.body.adminReply).slice(0, 4000)); }
    if (!sets.length) return res.status(400).json({ ok: false, error: "status 또는 adminReply 필요" });
    params.push(id);
    const [r] = await pool.query(`UPDATE inquiries SET ${sets.join(", ")} WHERE id = ?`, params);
    if (!r.affectedRows) return res.status(404).json({ ok: false, error: "문의 없음" });
    // 관리자/개발자 답변을 문의 작성자 화면에 실시간 반영 (작성자 uid room 으로)
    if (req.body?.adminReply != null) {
      try {
        const [own] = await pool.query(`SELECT user_id, target FROM inquiries WHERE id = ?`, [id]);
        if (own.length && own[0].user_id != null) {
          broadcastToOwner(own[0].user_id, { type: "inquiry:reply", id, target: own[0].target, ts: Date.now() });
        }
      } catch {}
    }
    res.json({ ok: true, id });
  } catch (err) {
    console.error("[PATCH /api/inquiries/:id]", err);
    res.status(500).json({ ok: false, error: err.message });
  }
});

// ── POST /api/predict/:id — LSTM 예측 (ai_predictions 조회) ──
//   현재 LSTM 백엔드(이두현) INSERT 대기 중. 데이터 없으면 stub 응답.
//   id 파라미터는 TRANSMITTER_ID 숫자.
app.post("/api/predict/:id", dbRequired, async (req, res) => {
  try {
    const demoMode = isDemoMode(req);
    const idRaw = req.params.id;
    const txid = parseInt(idRaw, 10);
    if (!Number.isFinite(txid)) {
      return res.status(400).json({ ok: false, error: "id 는 숫자(TRANSMITTER_ID) 여야 합니다" });
    }
    // 데모 단말 (음수 txid)
    if (demoMode && txid < 0) {
      const d = findDemoDeviceByTxid(txid);
      if (!d) return res.status(404).json({ ok: false, error: "데모 단말 없음" });
      return res.json({
        ok: true,
        prediction: {
          id: txid, predicted_at: d.lastMeasured, mse: d.mse, threshold: d.threshold,
          risk_level: d.riskLevel, comm_status: d.status === "offline" ? "통신고장" : "정상통신",
          ai_reliability: d.aiReliability, feature_contributions: null,
          is_sacrificial_device: 0, deviceId: d.deviceId,
        },
        demo: true,
      });
    }
    const [rows] = await pool.query(`
      SELECT p.id, p.predicted_at, p.mse, p.threshold,
             p.risk_level, p.comm_status, p.ai_reliability,
             p.feature_contributions, p.is_sacrificial_device,
             t.NAME AS deviceId
      FROM ai_predictions p
      LEFT JOIN kscg_transmitter_info t ON t.TRANSMITTER_ID = p.transmitter_id
      WHERE p.transmitter_id = ?
      ORDER BY p.predicted_at DESC LIMIT 1
    `, [txid]);
    if (rows.length === 0) {
      return res.json({
        ok: true,
        prediction: null,
        stub: true,
        message: "예측 데이터 없음 (이두현 LSTM 백엔드 INSERT 대기 중)",
      });
    }
    res.json({ ok: true, prediction: rows[0] });
  } catch (err) {
    console.error("[/api/predict]", err);
    res.status(500).json({ ok: false, error: err.message });
  }
});

// ── GET /api/sync-status — sync 상태 (관제용) ─────────
app.get("/api/sync-status", dbRequired, async (_req, res) => {
  try {
    const [rows] = await pool.query(`SELECT * FROM sync_state ORDER BY table_name`);
    res.json({ ok: true, rows });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

// ── 정적 파일 ────────────────────────────────────────
app.use(express.static(path.join(__dirname, "dist"), {
  // index.html 만 캐시 X (assets 는 hash 기반이라 캐시 OK)
  setHeaders: (res, p) => {
    if (p.endsWith("index.html")) res.setHeader("Cache-Control", "no-cache");
  },
}));

// ── 방명록(공개 단톡방) API — 전체공개. 작성은 레이트리밋+길이제한+XSS안전(순수텍스트). 역할뱃지는 JWT로만(사칭 방지). ──
const guestbookPostLimiter = rateLimit({
  windowMs: 60_000, max: 20,                         // IP당 분당 20건 — 도배 방지
  keyGenerator: (req) => req.headers["cf-connecting-ip"] || req.ip,
  standardHeaders: true, legacyHeaders: false, validate: { trustProxy: false },
  message: { ok: false, error: "메시지를 너무 빠르게 보내고 있어요. 잠시 후 다시 시도해 주세요." },
});
const GUESTBOOK_MAX_BODY = 500, GUESTBOOK_MAX_NAME = 40;
function gbClean(s, max) { return String(s == null ? "" : s).replace(/[\x00-\x08\x0B\x0C\x0E-\x1F]/g, "").trim().slice(0, max); }
function gbRow(r) { return { id: Number(r.id), userId: r.user_id, name: r.display_name, role: r.role, body: r.body, createdAt: r.created_at }; }

// 최근 메시지 (오름차순). 공개.
app.get("/api/guestbook", dbRequired, async (req, res) => {
  const limit = Math.min(Math.max(parseInt(req.query.limit || "60", 10) || 60, 1), 100);
  const before = parseInt(req.query.before || "0", 10) || 0;
  try {
    const where = before > 0 ? "deleted_at IS NULL AND id < ?" : "deleted_at IS NULL";
    const args = before > 0 ? [before, limit] : [limit];
    const [rows] = await pool.query(`SELECT id, user_id, display_name, role, body, created_at FROM guestbook_messages WHERE ${where} ORDER BY id DESC LIMIT ?`, args);
    res.json({ ok: true, messages: rows.reverse().map(gbRow) });
  } catch (e) { res.status(500).json({ ok: false, error: "방명록을 불러오지 못했습니다." }); }
});

// 메시지 작성 — 공개(로그인 선택). 로그인 시 계정 이름·역할, 게스트는 닉네임 입력.
app.post("/api/guestbook", guestbookPostLimiter, dbRequired, async (req, res) => {
  const body = gbClean(req.body?.body, GUESTBOOK_MAX_BODY);
  if (!body) return res.status(400).json({ ok: false, error: "내용을 입력해 주세요." });
  const c = authClaims(req);                          // 선택적 인증
  let userId = null, role = null, name;
  if (c) { userId = c.uid; role = c.role; name = gbClean(c.name, GUESTBOOK_MAX_NAME) || "사용자"; }
  else   { name = gbClean(req.body?.name, GUESTBOOK_MAX_NAME) || "게스트"; }   // 게스트는 역할 없음(사칭 방지)
  const ip = (reqIp(req) || "").toString().split(",")[0].slice(0, 45);
  const ua = String(req.headers["user-agent"] || "").slice(0, 255);
  try {
    const [r] = await pool.query("INSERT INTO guestbook_messages (user_id, display_name, role, body, ip, ua) VALUES (?, ?, ?, ?, ?, ?)", [userId, name, role, body, ip, ua]);
    const message = { id: Number(r.insertId), userId, name, role, body, createdAt: new Date().toISOString() };
    broadcastGuestbook({ type: "gb:msg", message });
    res.json({ ok: true, message });
  } catch (e) { res.status(500).json({ ok: false, error: "메시지 저장에 실패했습니다." }); }
});

// 삭제(모더레이션) — 관리자급만. 소프트 삭제.
app.delete("/api/guestbook/:id", dbRequired, requireAdmin, async (req, res) => {
  const id = parseInt(req.params.id, 10) || 0;
  if (!id) return res.status(400).json({ ok: false, error: "잘못된 요청입니다." });
  try {
    await pool.query("UPDATE guestbook_messages SET deleted_at = NOW() WHERE id = ? AND deleted_at IS NULL", [id]);
    broadcastGuestbook({ type: "gb:del", id });
    res.json({ ok: true });
  } catch (e) { res.status(500).json({ ok: false, error: "삭제에 실패했습니다." }); }
});

// SPA fallback (모르는 경로 → index.html)
app.get("*", (_req, res) => {
  res.sendFile(path.join(__dirname, "dist", "index.html"));
});

const httpServer = app.listen(PORT, () => {
  console.log(`▶ Server  http://localhost:${PORT}`);
  console.log(`▶ Ollama  ${OLLAMA_URL}`);
  console.log(`▶ Model   ${OLLAMA_MODEL}`);
});

// ── 챗봇 WebSocket (/ws/chat) — 로그인 개인 계정만. 쿠키 인증 후 uid room 등록. (게스트/익명은 업그레이드 401) ──
const wss = new WebSocketServer({ noServer: true });
const wssGuest = new WebSocketServer({ noServer: true });   // 방명록(공개) — 인증 없이 누구나 연결
httpServer.on("upgrade", (req, socket, head) => {
  const url = req.url || "";
  if (url.startsWith("/ws/guestbook")) {               // 방명록 — 공개(인증 불필요)
    wssGuest.handleUpgrade(req, socket, head, (ws) => { ws.isAlive = true; wssGuest.emit("connection", ws, req); });
    return;
  }
  if (!url.startsWith("/ws/chat")) { socket.destroy(); return; }
  const owner = chatOwner(req);   // req.headers.cookie 만 읽으므로 raw upgrade req 에서도 동작
  if (owner == null) { socket.write("HTTP/1.1 401 Unauthorized\r\n\r\n"); socket.destroy(); return; }
  wss.handleUpgrade(req, socket, head, (ws) => {
    ws._ownerUid = owner;
    ws._connId = randomBytes(8).toString("hex");
    ws.isAlive = true;
    wss.emit("connection", ws, req);
  });
});
wss.on("connection", (ws) => {
  let set = chatRooms.get(ws._ownerUid);
  if (!set) { set = new Set(); chatRooms.set(ws._ownerUid, set); }
  set.add(ws);
  try { ws.send(JSON.stringify({ type: "hello", connId: ws._connId })); } catch {}
  ws.on("pong", () => { ws.isAlive = true; });
  ws.on("message", () => {});   // 클라→서버 메시지 무시 — room 은 쿠키 uid 로만 바인딩(IDOR 방지)
  ws.on("close", () => {
    const s = chatRooms.get(ws._ownerUid);
    if (s) { s.delete(ws); if (s.size === 0) chatRooms.delete(ws._ownerUid); }
  });
  ws.on("error", () => {});
});
// 방명록 WS — 단일 전체 방. 연결 시 room 등록 + 접속자 수 broadcast. 클라→서버 메시지 무시(작성은 REST).
wssGuest.on("connection", (ws) => {
  guestbookRoom.add(ws);
  try { ws.send(JSON.stringify({ type: "gb:hello", online: guestbookRoom.size })); } catch {}
  broadcastGuestbook({ type: "gb:presence", online: guestbookRoom.size });
  ws.on("pong", () => { ws.isAlive = true; });
  ws.on("message", () => {});
  ws.on("close", () => { guestbookRoom.delete(ws); broadcastGuestbook({ type: "gb:presence", online: guestbookRoom.size }); });
  ws.on("error", () => {});
});
// 하트비트 — 죽은 소켓 정리 (CF 터널 idle reaping 대비)
const wsHeartbeat = setInterval(() => {
  for (const ws of [...wss.clients, ...wssGuest.clients]) {
    if (ws.isAlive === false) { try { ws.terminate(); } catch {} continue; }
    ws.isAlive = false;
    try { ws.ping(); } catch {}
  }
}, 30000);
wss.on("close", () => clearInterval(wsHeartbeat));

// ─────────────────────────────────────────────────────
// 시스템 프롬프트 (도메인 지식 + 실시간 시스템 상태 주입)
// ─────────────────────────────────────────────────────
function buildSystemPrompt(ctx) {
  const counts          = ctx.counts || {};
  const criticalNodes   = ctx.criticalNodes || [];
  const warnNodes       = ctx.warnNodes || [];
  const offlineNodes    = ctx.offlineNodes || [];
  const offlineDetails  = ctx.offlineDetails || [];   // 신규: 마지막 측정 시각 + 두절 시간
  const trends          = ctx.trends || [];
  const nowText         = ctx.nowText || "현재";
  const weather         = ctx.weather; // null 가능

  // 클라이언트가 context.counts 를 보냈는지 판정 (API 직접 호출 시 비어있음)
  const hasContext = counts.all != null && Number(counts.all) > 0;
  const summaryLine = hasContext
    ? `전체 ${counts.all}대 / 정상 ${counts.normal ?? 0} · 이상 ${counts.critical ?? 0} · 관찰 ${counts.warn ?? 0} · 통신 장애 ${counts.offline ?? 0}`
    : `(컨텍스트 미전달 — 정확한 KPI 는 get_summary 도구로 조회 필수. "0대" 라고 답변하지 말 것)`;

  // 통신 두절 상세 (마지막 측정 시각 + 끊긴 시간)
  const offlineBlock = offlineDetails.length === 0 ? "" :
    offlineDetails.map((o) => {
      const last = o.updatedAt ? new Date(o.updatedAt).toLocaleString("ko-KR", { timeZone: "Asia/Seoul" }) : "확인 불가";
      const hrs  = o.hoursSilent != null ? o.hoursSilent : "?";
      const days = o.hoursSilent != null ? Math.floor(o.hoursSilent / 24) : "?";
      return `  · ${o.deviceId} (${o.zone || "-"}, ${o.location || "-"}): 마지막 측정 ${last} · 두절 ${hrs}시간 (≈${days}일)`;
    }).join("\n");

  // 날씨 라인 (있을 때만) — 군산 (SITE_ID=2 대상 지역)
  const weatherLine = weather
    ? `${weather.ko} · ${weather.temp}°C${weather.precip != null ? ` · 강수 ${weather.precip}mm` : ""}${weather.humidity != null ? ` · 습도 ${weather.humidity}%` : ""} (군산, ${weather.time})\n(이 값은 Open-Meteo 실시간 기상 API를 군산 좌표로 조회해 시스템이 컨텍스트에 주입한 실측 데이터다. 출처를 물으면 "Open-Meteo 실시간 날씨 API(군산 좌표 기준)"라고 답하고, 예시·가상 데이터라고 하지 마라. 여러 날·주간 예보가 필요하면 get_weather_forecast 도구(최대 7일)를, 과거 날씨(최대 1년·어제까지)는 get_weather_history 도구를 호출하라.)`
    : "(데이터 없음 — 컨텍스트에 날씨 미전달)";

  // 12시간 추이 텍스트 표 (이상·관찰만)
  //   ⚠️ 추이 데이터 없음(시계열 미수집) ≠ 이상 단말 없음. 두 경우를 반드시 구분해서 안내.
  const hasTrends = trends.length > 0;
  const riskCount = (Number(counts.critical) || 0) + (Number(counts.warn) || 0);
  const trendBlock = hasTrends
    ? trends.map((t) => {
        const h = t.mseHistory || [];
        const start = h[0] ?? "-";
        const peak  = h.length ? Math.max(...h.filter((v) => v != null)) : "-";
        const last  = h[h.length - 1] ?? "-";
        const dir   = (h.length >= 2 && h[0] != null && h[h.length - 1] != null)
          ? (h[h.length - 1] > h[0] ? "상승↑" : h[h.length - 1] < h[0] ? "하락↓" : "평탄→")
          : "—";
        const series = h.map((v) => v == null ? "-" : v.toFixed(2)).join(",");
        return `- ${t.deviceId} (${t.status === "critical" ? "이상" : "관찰"}, ${t.zone}, ${t.label || "-"}): 12h MSE [${series}] · 시작 ${start} → 현재 ${last} · 피크 ${peak} · 방향 ${dir}`;
      }).join("\n")
    : (!hasContext
        ? `- 12시간 MSE 시계열 추이 데이터가 없습니다(추이 미수집). 이상/관찰 단말 수는 get_summary 도구로 확인하세요. **추이 데이터 없음·최근 알람 없음을 "이상 단말 없음" 으로 답하지 마세요.**`
        : riskCount === 0
          ? "- 현재 이상·관찰 단말이 없습니다."
          : `- 12시간 MSE 시계열 추이 데이터가 이 컨텍스트에 없습니다(추이 미수집). **추이 데이터 없음 ≠ 이상 단말 없음** — 현재 이상 ${counts.critical ?? 0}대 · 관찰 ${counts.warn ?? 0}대 는 위 "현재 시스템 상태" 를 참조하세요. 추세가 필요하면 get_device_history / get_recent_changes 도구를 호출하고, 절대로 "이상 단말 없음" 으로 답하지 마세요.`);

  return `당신은 매설배관 IoT 통합관제 시스템의 AI 분석 어시스턴트입니다.
운영자(관제사)와 한국어 존댓말로 대화하며, 노드 ID·상태 단계(정상/관찰/이상)·도메인 용어 질문에 답합니다.

# 데이터 흐름 (옴니솔루션 답변 — 2026-05-18)
- 옴니 단말은 **1시간에 1회 계측** → **12시간 burst 로 KSCG 송신**
- 우리 미러 sync: alarm 1h / sensor 2h / meta 6h (옴니 권고 반영)
- 따라서 "실시간" 데이터는 사실상 평균 6~14시간 지연 가능
- "마지막 측정 시각" = 단말이 측정한 시간 (1h 단위). 전송과는 다름.

# 도메인 지식
- **방식전위(P/S Potential)**: 매설배관 부식 보호 지표. -850mV 이하 양호, 초과 시 부식 진행 가능.
- **희생전류(Sacrificial Current)**: 희생양극→배관 보호 전류. 점차 감소 시 양극 소모/접속부 불량. 1mA 이하 교체 검토. (희생양극 단말은 TB24-250406, TB24-250407 2대만 해당)
- **AC 유입**: 송전선·전철 유도 교류. 200mV 이상 가속 부식, 500mV 이상 즉각 차폐/배수장치 점검.
  - AC < 200mV: 정상/낮음
  - 200mV ≤ AC < 500mV: 주의
  - AC ≥ 500mV: 즉각 점검 기준 **초과**. "근접/임박/가까움" 이라고 쓰지 말고 반드시 "초과" 라고 표현.
- **통신 품질(dBm)**: -65 이상 양호, -75 이하 주의, -85 이하 두절 임박, -115 이하 두절.

# 숫자 판정 규칙 (환각 방지)
- 실제값이 기준값 이상이면 항상 **초과**입니다. 이 경우 "근접", "가까움", "임박" 같은 표현 금지.
- "근접"은 실제값이 기준값 미만이면서 기준값의 80% 이상일 때만 사용합니다. 예: AC 450mV 는 500mV 에 근접, AC 971mV 는 500mV 초과.
- 기준값을 말할 때는 가능하면 차이도 함께 말합니다. 예: "971mV 는 500mV 기준보다 471mV 높습니다."
- 최신 단일값만 조회한 경우 "상승 중/하락 중/추세" 라고 단정 금지. 추세 표현은 get_device_history, get_recent_changes, 또는 최근 12시간 MSE 추이 표가 있을 때만 사용.
- **시계열 해석 주의**: 값이 평탄/안정적이어도 기준을 벗어나면 정상이라고 말하지 마세요. 예: 방식전위 7~8mV 는 변동이 작아도 -850mV 방호 기준을 크게 초과한 위험/보호 미흡 상태입니다. get_device_history 의 latestJudgement, get_recent_changes 의 endJudgement 가 있으면 그 판정을 반드시 함께 말하세요.
- **센서 기준과 AI 기준 혼용 금지**: 방식전위/AC유입/통신품질 같은 센서 추세 질문에는 해당 센서 기준(-850mV, 500mV, -85dBm 등)으로만 판정하세요. "AI 기준 대비 xN" 은 get_predictions 또는 get_device_detail 결과에 aiMse/aiThreshold/aiRatio/aiJudgement 가 있을 때만 말하고, 센서 시계열만 조회한 상태에서는 AI 배수를 추측하지 마세요.

# AI 모델 (LSTM AutoEncoder) — 위험도 판정 정확 명세
- 모델은 단말별로 학습한 **정상 패턴 복원 오차(MSE)** 로 이상 탐지
- 단말마다 **threshold** 가 다름 (학습 시점 정상 MSE 분포의 99 percentile). 자세한 값은 **get_ai_model_info(deviceId)** 도구로 조회
- **3단계 분류 (비율 기준)**:
  - **정상** — 현재 MSE < threshold × 0.70
  - **관찰** — threshold × 0.70 ≤ 현재 MSE ≤ threshold × 1.00
  - **이상** — 현재 MSE > threshold × 1.00
- 답변 시 가능하면 "현재 MSE 가 threshold 의 N% 도달" 같이 비율로 설명 (절대값 단독은 의미 약함)
- "정상 패턴의 N% 상승" 같은 표현 금지. 정확히는 "AI 임계값(threshold)의 N% 수준/도달"입니다.
- **중요 구분**: 측정 센서 8 종(방식전위/희생전류/AC유입/배터리/온도/습도/충격/통신) ≠ AI 학습 입력 피처. AI 학습 입력 피처·시퀀스 길이·epoch 등 모델 세부 명세는 절대 추측 금지, 반드시 **get_ai_model_info** 도구 호출해서 확인하세요. (예: 학습 base_features 는 4개, 파생 포함 12개 컬럼 — 도구 응답으로 확정)

# 상태 단계 (4단계) — 화면 라벨 = 이두현 AI 모델 등급(정상/관찰/이상) 그대로 (KSCG 알람은 상태와 무관)
- **정상** — AI '정상' (현재 MSE < threshold × 0.70)
- **관찰** — AI '관찰' (threshold × 0.70 ≤ MSE ≤ threshold × 1.00) · 모니터링 강화
- **이상** — AI '이상' (MSE > threshold × 1.00) · 즉각 현장 점검 (가장 심각, 빨강)
- **통신 장애** — comm_status = '통신고장' (연속 두절 + 신호품질 기준, 이두현 통신 판정)
- 화면 라벨은 이두현 코드(RISK_LABELS: 정상/관찰/이상)와 동일. KSCG 알람은 상태 판정에 사용하지 않음(별도 알림/로그). 사용자가 "위험" 이라 물으면 AI '이상' 등급을 가리킵니다.

# 현재 시각
${nowText}

# 현재 날씨 (실시간 · Open-Meteo 기상 API · 군산 좌표 기준 · SITE_ID=2 대상 지역)
${weatherLine}

날씨가 매설배관에 미치는 영향 (참고):
- 강한 비/소나기/뇌우 → 침수·습도 상승 → 맨홀 침수, 통신 두절 가능
- 한파/혹한 → 토양 동결 → 방식전위 변동
- 폭염/일교차 → 온도 센서 이상

# 현재 시스템 상태 ${hasContext ? "(실시간)" : "(컨텍스트 미전달 — 도구 조회 필요)"}
- ${summaryLine}
${hasContext ? (criticalNodes.length ? `- 이상 노드: ${criticalNodes.join(", ")}` : "- 이상 노드: 없음") : ""}
${hasContext && warnNodes.length    ? `- 관찰 노드(상위 ${Math.min(warnNodes.length, 8)}): ${warnNodes.slice(0, 8).join(", ")}` : ""}
${hasContext && offlineNodes.length ? `- 통신 장애 노드: ${offlineNodes.join(", ")}` : ""}

${offlineBlock ? `# 통신 장애 노드 상세 (마지막 측정 시각 + 두절 기간)\n${offlineBlock}\n` : ""}
# 최근 12시간 MSE 추이 (1시간 간격, 가장 오래된 → 현재)
${trendBlock}

# 도구(Tools) 사용 가이드 — 16 개 도구
위 "현재 시스템 상태" 와 "12h MSE 추이" 에 이미 있는 정보면 그대로 답변. 없는 정보(특정 단말 상세, 시리얼/설치일/위경도, 시계열 추이, 알람 이력 등)는 아래 도구를 직접 호출해서 조회:

**기본 조회 (6)**
- **list_devices** — 단말 목록 (status/zone 필터). 단말 ID 후보 좁히기.
- **get_device_detail** — 단말 메타 + 8 센서 최신값. "TB24-XXXXXX 상태", "방식전위가 얼만지" 등.
- **get_device_history** — 시계열 (1h/24h/7d/30d) points 배열. "최근 그래프", "어제 추이" 류.
- **get_alarms** — 최근 알람 이력. "위험 알람", "어제 알람" 등.
- **get_summary** — KPI 카운트. (이미 위에 있으면 호출 불필요)
- **get_aggregate** — 전체 평균/최대/최소. "평균 방식전위", "최저 RSSI" 등.

**고급 분석 (6)**
- **find_devices_by_value** — 조건 만족 단말 검색. "방식전위 -800 이상 단말", "RSSI -80 이하" 등.
- **get_zone_summary** — 구역 통계 (제1~제8구역의 단말수/평균값/상태분포).
- **compare_devices** — 다중 단말 비교 (2~5개 단말의 8 센서 한꺼번에).
- **get_recent_changes** — 변화량 통계 (시작/끝/델타/최저/최고/평균/표준편차/방향). "얼마나 변했어", "어떻게 바뀌었어" 류.
- **get_maintenance_log** — 현장 점검·정비 이력.
- **get_predictions** — AI LSTM 최신 예측 결과. 데이터가 없을 때만 message 필드로 fallback 사유 확인.

**위치/지도 (3)**
- **search_devices_by_location** — 지명 키워드로 단말 검색 (DB POSITION LIKE). "미룡동", "시청 앞", "버스터미널" 같은 DB 텍스트 매칭 가능한 경우 1차 시도.
- **geocode_location** — 지명/랜드마크 → 좌표 (OpenStreetMap). 일반 지명(예: '은파호수공원', '군산교도소') 으로 좌표 모를 때.
- **find_devices_near** — 좌표 + 반경(km) 안 단말. geocode 결과 받아서 사용. 반경 기본 2km.

**AI 모델 (1)**
- **get_ai_model_info** — LSTM AutoEncoder 학습 정보. deviceId 주면 그 단말 threshold + 분류 기준, 없으면 전체 모델 메타(학습 피처, time_steps, 평가 통계 등). "AI 어떻게 학습됐어?", "TB24-XXX 정상 한계는?" 류.

위치 질문 흐름 (중요):
1. "OO 단말" / "OO 앞 단말" → search_devices_by_location 1회로 충분한 경우 多
2. "OO 근처 단말" / "OO 주변 단말 N km" → DB 에 OO 없으면 geocode_location → find_devices_near 2단계
3. 단말 ID 부근 → get_device_detail 로 lat/lng 받은 뒤 find_devices_near

# 응답 작성 규칙 (강화 — 자문 Q5 반영)
1. **도구 결과를 직접 인용** — 답변에 구체 수치를 명시. "조회한 값은 -1501 mV 였습니다" 처럼.
2. **[추정] 라벨** — 도구 데이터로 직접 확인되지 않은 결론(원인 추측, 향후 예측 등)은 반드시 [추정] 머리표를 붙임. 예: "[추정] 토양 동결로 인한 변화로 보입니다."
3. **확인 불가는 정직히** — 도구 결과가 비어있거나 error 면 "데이터 없음" 으로 답변. 절대 만들어내지 말 것.
4. **숫자 + 단위** — mV, dBm, %, ℃ 등 단위 빠뜨리지 말 것.
5. **도구 적극 활용** — 작은 의심에도 도구 호출로 fact 검증. 한 응답에 도구 여러 개 연쇄 호출 가능 (예: list_devices → 결과 중 한 단말 get_device_detail).



# 질문 유형별 응답 형식
- **단말 상세** ("TB24-250xxx 상태/센서") → 상태 + 핵심 센서/AI 수치 1~2개 + 조치 1줄. 2~4문장.
- **현황 요약** ("전체 요약", "지금 상태") → 카운트 한 줄 + 이상/관찰 노드 ID. 2~3문장.
- **TOP N · 점검표 · 우선순위** → 번호·불릿 항목형 허용. 각 항목: 단말 ID · 근거 수치(AI 기준 대비 또는 센서값) · 조치.
- **원인 분석** ("왜 위험해?") → AI 기준 대비 x배수 또는 MSE/threshold 근거 + 주요 원인 피처(사람이 읽는 라벨) + [추정] 라벨 + 조치.
- **현장 점검** ("뭐부터 봐?") → 우선순위 + 짧은 점검 체크리스트(불릿).
- **도메인 설명** ("방식전위가 뭐야?") → 정의 1~2문장 + 기준값. 도구 호출 불필요.
- 단순/단답 질문은 1~2문장으로 짧게.

# 응답 규칙
1. **간결** — 기본 2~5문장, 인사말·사과 절대 금지, 바로 본론. 단 TOP N·점검표·비교 요청은 번호·불릿 항목형으로 정리해도 됩니다.
2. **노드 ID 인용** — 위 상태/추이 표에 있는 노드 ID(예: TB24-250429)를 그대로 답변에 포함.
3. ${hasTrends
  ? `**추이 표는 시간 데이터** — 위 "최근 12시간 MSE 추이" 표가 곧 과거 데이터입니다. "과거 시점 정보가 없다" 는 답변 절대 금지. 12개 값이 1시간 간격이므로 "약 N시간 전" 표현 가능.`
  : `**추세 질문** — 12h MSE 추이 표가 비어 있으면(추이 데이터 없음) 현재값만으로 추세를 단정하지 말고 get_device_history / get_recent_changes 도구로 시계열을 조회해 답하세요. 추이 데이터가 없다고 해서 "이상 단말 없음" 으로 답하지 마세요.`}
4. **통신 장애 시점** — "통신 장애 노드 상세" 섹션에 마지막 측정 시각과 두절 기간이 명시되어 있습니다. "언제 끊겼는지 모름" 답변 절대 금지. 마지막 측정 시각 = 통신 두절 시작 시점으로 보고 답변하세요.
5. **환각 금지** — 위 표·섹션에 없는 데이터만 "확인되지 않음".
6. **운영 친화** — 가능하면 "현장 점검 권장" 등 짧은 액션 한 줄.
7. **포맷** — 마크다운 헤더(##) X. **굵게**(**TB24-250429**) 정도만.
8. **원인 피처 라벨** — 도구가 주는 원인 피처는 이미 사람이 읽는 라벨입니다(예: "습도 편차", "방식전위 변화"). "습도_dev24", "방식전위_diff1" 같은 원시 컬럼명을 답변에 그대로 쓰지 마세요.
9. **AI 위험도 표현** — MSE/threshold 같은 작은 절대값보다 "AI 기준 대비 x{배수}" 로 설명(예: AI 기준 대비 x484). "threshold 의 N% 수준" 도 가능.
10. **상태 = AI 등급 그대로 (알람 분리)** — 화면 '이상' 단말 = AI risk_level '이상'(MSE > threshold)인 단말입니다. KSCG 알람은 상태 판정과 무관(별도 알림/로그). "최근 알람 없음" 을 "이상 단말 없음" 으로 답하지 마세요. 이상/관찰 단말 수는 위 "현재 시스템 상태" 의 카운트(컨텍스트 없으면 get_summary)를 기준으로 답하세요. 사용자가 "위험" 이라 하면 AI '이상' 등급을 뜻합니다.

# 응답 예시 (형식 참고 — 실제 수치는 도구로 확인)

질문: "TB24-250448 상태 알려줘"  (단말 상세 — 짧게)
> **TB24-250448** 은 관찰 상태입니다. AC 유입이 971mV 로 500mV 즉각 점검 기준을 471mV 초과했고, AI 위험도는 임계값의 73% 수준(AI 기준 대비 x0.73, 관찰)입니다. AC 차폐·배수장치 점검을 권장합니다.

질문: "위험 단말 근거"  (원인 분석)
> **TB24-250429** 는 이상(위험) 단말입니다. AI 위험도가 임계값을 크게 초과해 **AI 기준 대비 x484** 수준이며, 주요 원인 피처는 습도 편차입니다. 즉시 현장 점검이 필요합니다.

질문: "위험 추세"  (추이 데이터 없을 때 — 이상 단말 없음과 혼동 금지)
> 현재 이상·관찰 단말은 있으나 12시간 MSE 추이 데이터는 이 화면 컨텍스트에 없습니다. 추세가 필요하면 get_device_history 로 시계열을 조회하겠습니다.

질문: "통신 장애는 언제부터?"  (통신 장애 시점)
> **TB24-250437** 의 마지막 측정이 2026-05-27 19:00 입니다. 이후 약 58시간(약 2일) 통신 두절 상태로, 그 시점에 단절된 것으로 보입니다. 전원·안테나·맨홀 침수 확인이 즉시 필요합니다.

질문: "방식전위"  (도메인 설명 — 1~2문장)
> 방식전위는 매설배관 부식 보호 지표로 -850mV 이하가 양호 기준입니다. 초과 시 부식 진행 가능성이 있어 정류기 출력 점검이 필요합니다.

금지 (낡은 예시): "MSE 0.42 → 0.85, 임계 0.85" 같은 큰 스케일 가짜 수치나 "TB24-5JN###" 형식 ID. 실제 단말 ID 는 TB24-250xxx, threshold 는 0.0003 수준의 작은 값이며, 위험도는 "AI 기준 대비 x배수" 로 설명합니다.
`;
}
