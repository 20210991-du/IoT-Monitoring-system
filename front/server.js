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
import path from "path";
import { readFileSync, existsSync } from "fs";
import { fileURLToPath } from "url";
import mysql from "mysql2/promise";

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
//   위험 3 + 이상의심 4 + 통신장애 3.
//   좌표·시설번호·POSITION 모두 군산 자연스러운 위치로.
function getDemoDevices() {
  return [
    // ── 위험 3대 (critical: 최근 알람 발생) ──
    { deviceId: "DEMO-001", txid: -1, facility: "1-DEMO01", location: "시청 사거리 (데모)",
      lat: 35.9676, lng: 126.7369, hoursSilent: 2, recentAlarms: 3, status: "critical",
      sensors: { volt: -540, sacrificial: 0.6, ac: 312, battery: 3580, temp: 28.4, hum: 71, shock: 0, commDbm: -88 },
      lastMeasured: nowMinusH(2),
      mse: 0.0042, threshold: 0.0011, riskLevel: "위험", aiReliability: "신뢰" },
    { deviceId: "DEMO-002", txid: -2, facility: "2-DEMO02", location: "대야 사거리 (데모)",
      lat: 35.962, lng: 126.745, hoursSilent: 1, recentAlarms: 5, status: "critical",
      sensors: { volt: -480, sacrificial: 0.3, ac: 540, battery: 3520, temp: 30.1, hum: 68, shock: 1, commDbm: -82 },
      lastMeasured: nowMinusH(1),
      mse: 0.0061, threshold: 0.0013, riskLevel: "위험", aiReliability: "신뢰" },
    { deviceId: "DEMO-003", txid: -3, facility: "8-DEMO03", location: "소룡동 공단 입구 (데모)",
      lat: 35.985, lng: 126.700, hoursSilent: 1, recentAlarms: 2, status: "critical",
      sensors: { volt: -620, sacrificial: 0.8, ac: 280, battery: 3610, temp: 27.8, hum: 70, shock: 0, commDbm: -85 },
      lastMeasured: nowMinusH(1),
      mse: 0.0038, threshold: 0.0011, riskLevel: "위험", aiReliability: "신뢰" },
    // ── 이상의심 4대 (warn) ──
    { deviceId: "DEMO-101", txid: -101, facility: "4-DEMO11", location: "미룡동 교차로 (데모)",
      lat: 35.937, lng: 126.696, hoursSilent: 1, recentAlarms: 0, status: "warn",
      sensors: { volt: -780, sacrificial: 0.9, ac: 195, battery: 3700, temp: 26.5, hum: 65, shock: 0, commDbm: -78 },
      lastMeasured: nowMinusH(1),
      mse: 0.0019, threshold: 0.0011, riskLevel: "이상", aiReliability: "주의" },
    { deviceId: "DEMO-102", txid: -102, facility: "9-DEMO12", location: "해망동 박물관 앞 (데모)",
      lat: 35.990, lng: 126.704, hoursSilent: 1, recentAlarms: 0, status: "warn",
      sensors: { volt: -810, sacrificial: 1.1, ac: 175, battery: 3680, temp: 27.0, hum: 67, shock: 0, commDbm: -76 },
      lastMeasured: nowMinusH(1),
      mse: 0.0016, threshold: 0.0011, riskLevel: "이상", aiReliability: "주의" },
    { deviceId: "DEMO-103", txid: -103, facility: "3-DEMO13", location: "소룡동 현대자동차 옆 (데모)",
      lat: 35.983, lng: 126.682, hoursSilent: 2, recentAlarms: 0, status: "warn",
      sensors: { volt: -795, sacrificial: 1.0, ac: 188, battery: 3690, temp: 27.5, hum: 66, shock: 0, commDbm: -79 },
      lastMeasured: nowMinusH(2),
      mse: 0.0015, threshold: 0.0011, riskLevel: "이상", aiReliability: "주의" },
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
app.use(express.json({ limit: "1mb" }));

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

function dbRequired(_req, res, next) {
  if (!pool) return res.status(503).json({ ok: false, error: "DB pool 비활성 (서버 환경변수 SIWON_DB_PASS 누락)" });
  next();
}

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
async function ensureChatSession(sessionId, firstUserMsg) {
  if (!pool) return null;
  if (sessionId) {
    try {
      const [rows] = await pool.query(`SELECT id FROM chat_sessions WHERE id = ?`, [sessionId]);
      if (rows.length) return Number(sessionId);
    } catch (_) {}
  }
  try {
    const title = String(firstUserMsg || "").slice(0, 30).trim() || "(제목 없음)";
    const [r] = await pool.query(`INSERT INTO chat_sessions (title) VALUES (?)`, [title]);
    return r.insertId;
  } catch (e) {
    console.warn("[ensureChatSession]", e.message);
    return null;
  }
}

async function persistMessage(sessionId, role, text, contextJson, tokens, model) {
  if (!pool || !sessionId || !text) return;
  try {
    await pool.query(
      `INSERT INTO chat_messages (session_id, role, text, context_json, tokens_prompt, tokens_completion, model) VALUES (?, ?, ?, ?, ?, ?, ?)`,
      [
        sessionId, role, text,
        contextJson ? JSON.stringify(contextJson) : null,
        tokens?.prompt ?? null,
        tokens?.completion ?? null,
        model || null,
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
      description: "전체 KPI 카운트 (정상/위험/이상의심/통신장애 단말 수)",
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
      description: "특정 구역(시설번호 prefix)의 통계: 단말 수, 정상/위험/통신두절 카운트, 평균 방식전위 + 평균 RSSI.",
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
      description: "AI LSTM-AutoEncoder 예측 (MSE, 위험도, 통신상태, 신뢰도). 단말 ID 없으면 전체 최근 예측. 이두현 백엔드 연동 전이므로 결과가 비어있을 수 있음.",
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
  }
];

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
async function execTool(name, args, demoMode = false) {
  args = args || {};
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
          SELECT t.NAME AS deviceId, f.NUMBER AS facility, f.POSITION AS location,
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
        const now = Date.now();
        const annotated = rows.map((r) => {
          const hoursSilent = r.lastSeen ? Math.floor((now - new Date(r.lastSeen).getTime()) / 3600000) : null;
          const recentAlarms = Number(r.recentAlarms) || 0;
          // /api/devices mapStatus 와 일치: 24h 두절 → offline, 최근 7일 알람 → critical, else normal
          const status = hoursSilent != null && hoursSilent >= 24 ? "offline"
                       : recentAlarms > 0 ? "critical"
                       : "normal";
          return { ...r, hoursSilent, recentAlarms, status };
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
            lastMeasured: d.lastMeasured,
            hoursSilent: d.hoursSilent,
            status: d.status,
            mse: d.mse, threshold: d.threshold, riskLevel: d.riskLevel, aiReliability: d.aiReliability,
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
        // AI threshold (있을 때만)
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
          status: hoursSilent != null && hoursSilent >= 24 ? "offline" : "normal",
          ai,
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
          return { deviceId, kind, range, count: points.length, sampled: points.length, points, demo: true };
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
        return { deviceId, kind, range, count: rows.length, sampled: points.length, points };
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
        const [[total]] = await pool.query(`
          SELECT COUNT(*) AS all_ FROM kscg_transmitter_info t
          JOIN kscg_site_mydevice m ON m.TRANSMITTER_ID = t.TRANSMITTER_ID AND m.SITE_ID = ?
        `, [SITE_ID]);
        const [[silent]] = await pool.query(`
          SELECT COUNT(*) AS offline FROM (
            SELECT t.TRANSMITTER_ID, TIMESTAMPDIFF(HOUR, MAX(r.DATE), NOW()) AS h
            FROM kscg_transmitter_info t
            JOIN kscg_site_mydevice m ON m.TRANSMITTER_ID = t.TRANSMITTER_ID AND m.SITE_ID = ?
            JOIN kscg_sensor_info si ON si.TRANSMITTER_ID = t.TRANSMITTER_ID
            JOIN kscg_recent_data r ON r.SENSOR_ID = si.SENSOR_ID
            GROUP BY t.TRANSMITTER_ID HAVING h >= 24
          ) x
        `, [SITE_ID]);
        const [[alm]] = await pool.query(`
          SELECT COUNT(DISTINCT si.TRANSMITTER_ID) AS critical
          FROM kscg_alarm_log a
          JOIN kscg_sensor_info si ON si.SENSOR_ID = a.SENSOR_ID
          JOIN kscg_site_mydevice m ON m.TRANSMITTER_ID = si.TRANSMITTER_ID AND m.SITE_ID = ?
          WHERE a.GEN_DATE > DATE_SUB(NOW(), INTERVAL 7 DAY)
        `, [SITE_ID]);
        const all = Number(total.all_) || 0;
        const offline = Number(silent.offline) || 0;
        const critical = Number(alm.critical) || 0;
        // 데모 가산
        if (demoMode) {
          const d = getDemoDevices();
          const dC = d.filter((x) => x.status === "critical").length;
          const dW = d.filter((x) => x.status === "warn").length;
          const dO = d.filter((x) => x.status === "offline").length;
          return {
            total: all + d.length,
            normal: all - offline - critical,
            critical: critical + dC, warn: dW, offline: offline + dO,
            demoMode: true,
          };
        }
        return { total: all, normal: all - offline - critical, critical, warn: 0, offline };
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
async function runChatWithTools(messages, signal, demoMode = false) {
  const MAX_ROUNDS = 5;
  const working = [...messages];
  const toolTrace = [];
  let lastTokens = {};
  for (let round = 0; round < MAX_ROUNDS; round++) {
    const res = await fetch(`${OLLAMA_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: OLLAMA_MODEL,
        messages: working,
        tools: TOOLS,
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
  const { message, context = {}, history = [], sessionId } = req.body || {};
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

  // 세션 영구화 (best-effort)
  const sid = await ensureChatSession(sessionId, message);
  const demoMode = isDemoMode(req);

  try {
    const ctrl = new AbortController();
    const timeout = setTimeout(() => ctrl.abort(), 120_000); // 120s (최대 5 tool 라운드 여유)
    const result = await runChatWithTools(messages, ctrl.signal, demoMode);
    clearTimeout(timeout);

    const reply = (result.content || "(빈 응답)").trim();

    // chat_messages 영구화 (background, best-effort)
    if (sid) {
      persistMessage(sid, "user", message, context, null, null);
      persistMessage(sid, "ai", reply, { rounds: result.rounds, toolCalls: result.toolTrace }, result.tokens, OLLAMA_MODEL);
    }

    return res.json({
      ok:        true,
      sessionId: sid,
      reply,
      model:     OLLAMA_MODEL,
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
  const { message, context = {}, history = [], sessionId } = req.body || {};
  if (!message || typeof message !== "string") {
    return res.status(400).json({ ok: false, error: "message 필드가 비어있습니다." });
  }

  // SSE 헤더
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Accel-Buffering", "no");
  res.flushHeaders && res.flushHeaders();

  const send = (event, data) => {
    res.write(`event: ${event}\n`);
    res.write(`data: ${JSON.stringify(data)}\n\n`);
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

  // 세션 영구화 (best-effort). 응답에 session 이벤트로 동봉.
  const sid = await ensureChatSession(sessionId, message);
  if (sid) send("session", { sessionId: sid });
  const demoMode = isDemoMode(req);

  const MAX_ROUNDS = 5;
  const toolTrace = [];
  let finalAccum = "";
  let lastTokens = { prompt: 0, completion: 0 };

  const ctrl = new AbortController();
  const timeout = setTimeout(() => { console.log("[chat/stream] timeout 180s"); ctrl.abort(); }, 180_000);   // 180s (라운드 여유)
  // client disconnect → 진행 중인 Ollama fetch abort.
  // res.writableFinished 는 res.end() 호출 후 true 가 됨. 우리가 아직 응답 안 끝낸 상태라면
  // 'close' 는 진짜 클라이언트 단절. (express body parser 가 emit 하는 spurious close 는
  //  대부분 응답 헤더 전이라서 첫 write 이후엔 안전.)
  let aborted = false;
  res.on("close", () => {
    if (res.writableFinished) return;   // 우리가 정상 종료한 경우
    if (aborted) return;
    aborted = true;
    console.log("[chat/stream] response closed early → abort");
    try { ctrl.abort(); } catch {}
  });

  try {
    for (let round = 0; round < MAX_ROUNDS; round++) {
      const ollamaRes = await fetch(`${OLLAMA_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: OLLAMA_MODEL,
          messages: working,
          tools: TOOLS,
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
          model:     OLLAMA_MODEL,
          rounds:    round + 1,
          toolCalls: toolTrace,
          tokens:    lastTokens,
        });
        // chat_messages 영구화 (background)
        if (sid) {
          persistMessage(sid, "user", message, context, null, null);
          persistMessage(sid, "ai", finalReply, { rounds: round + 1, toolCalls: toolTrace }, lastTokens, OLLAMA_MODEL);
        }
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
      model: OLLAMA_MODEL,
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

// 단말 status 판정
//   - offline  : 최근 측정 24h+ 없음 (진짜 통신 두절)
//   - critical : 최근 7일 활성 알람 발생
//   - warn     : (LSTM 예측 연동 후 확장 예정)
//   - normal   : 그 외
//
// 주의: KSCG 의 DEVICE_STATUS 컬럼은 의미가 명확하지 않음 (시범 5대=1, 확대 50대=0).
//       데이터 자체는 둘 다 정상 흐름이라 status 판정에 사용하지 않음.
function mapStatus(_deviceStatus, hoursSilent, activeAlarmCount) {
  if (hoursSilent != null && hoursSilent >= 24) return "offline";
  if (activeAlarmCount > 0) return "critical";
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
    const [[counts]] = await pool.query(`
      SELECT
        COUNT(*) AS total,
        SUM(CASE WHEN t.DEVICE_STATUS = 1 THEN 1 ELSE 0 END) AS active,
        SUM(CASE WHEN t.DEVICE_STATUS = 0 THEN 1 ELSE 0 END) AS inactive
      FROM kscg_transmitter_info t
      JOIN kscg_site_mydevice m ON m.TRANSMITTER_ID = t.TRANSMITTER_ID AND m.SITE_ID = ?
    `, [SITE_ID]);

    // 최근 1시간 안 들어온 단말 = 통신 두절 추정
    const [[silent]] = await pool.query(`
      SELECT COUNT(*) AS offline
      FROM (
        SELECT t.TRANSMITTER_ID,
               TIMESTAMPDIFF(HOUR, MAX(r.DATE), NOW()) AS hours_silent
        FROM kscg_transmitter_info t
        JOIN kscg_site_mydevice m ON m.TRANSMITTER_ID = t.TRANSMITTER_ID AND m.SITE_ID = ?
        JOIN kscg_sensor_info si  ON si.TRANSMITTER_ID = t.TRANSMITTER_ID
        JOIN kscg_recent_data r   ON r.SENSOR_ID = si.SENSOR_ID
        GROUP BY t.TRANSMITTER_ID
        HAVING hours_silent >= 24
      ) x
    `, [SITE_ID]);

    // 활성 알람 (status=0 등 운영 중) → critical 추정
    const [[alarmsRecent]] = await pool.query(`
      SELECT COUNT(DISTINCT si.TRANSMITTER_ID) AS critical
      FROM kscg_alarm_log a
      JOIN kscg_sensor_info si ON si.SENSOR_ID = a.SENSOR_ID
      JOIN kscg_site_mydevice m ON m.TRANSMITTER_ID = si.TRANSMITTER_ID AND m.SITE_ID = ?
      WHERE a.GEN_DATE > DATE_SUB(NOW(), INTERVAL 7 DAY)
    `, [SITE_ID]);

    const total    = Number(counts.total) || 0;
    const offline  = Number(silent.offline) || 0;
    const critical = Number(alarmsRecent.critical) || 0;
    let   warn     = 0;  // TODO: LSTM 예측 → ai_predictions 연계 시 채우기
    let   normal   = total - offline - critical - warn;

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

    const now = new Date();
    const out = devices.map((d) => {
      const slot   = byDev[d.id] || { sensors: {}, lastMeasured: null };
      const alm    = alarmsByDev[d.id];
      const hoursSilent = slot.lastMeasured
        ? Math.floor((now - new Date(slot.lastMeasured)) / 3600000)
        : null;
      const status = mapStatus(d.deviceStatus, hoursSilent, alm ? Number(alm.cnt) : 0);
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

// ── GET /api/anomalies — AI 탐지 (이상 의심 + 관찰) ──
//   현재 LSTM 예측(ai_predictions) 미연동 → KSCG 알람 + 통신 두절로 임시 매핑.
//   anomalies = 최근 7일 알람 발생 단말, watch = 24h 통신 두절 단말
app.get("/api/anomalies", dbRequired, async (req, res) => {
  try {
    const demoMode = isDemoMode(req);
    // anomalies: 최근 7일 알람 → 위험·이상으로 매핑
    const [anomalyRows] = await pool.query(`
      SELECT
        t.NAME AS node,
        f.NUMBER AS facility,
        a.GEN_DATE AS ts,
        a.GRADE_ID AS gradeId,
        g.GRADE_TEXT AS gradeText,
        a.VALUE AS mse,
        a.CONTENTS AS label
      FROM kscg_alarm_log a
      JOIN kscg_sensor_info si ON si.SENSOR_ID = a.SENSOR_ID
      JOIN kscg_transmitter_info t ON t.TRANSMITTER_ID = si.TRANSMITTER_ID
      LEFT JOIN kscg_facility_info f ON f.TRANSMITTER_ID = t.TRANSMITTER_ID
      LEFT JOIN kscg_alarm_grade_info g ON g.GRADE_ID = a.GRADE_ID
      JOIN kscg_site_mydevice m ON m.TRANSMITTER_ID = si.TRANSMITTER_ID AND m.SITE_ID = ?
      WHERE a.GEN_DATE > DATE_SUB(NOW(), INTERVAL 30 DAY)
      ORDER BY a.GEN_DATE DESC
      LIMIT 20
    `, [SITE_ID]);

    // watch: 통신 24h 두절 단말
    const [watchRows] = await pool.query(`
      SELECT
        t.NAME AS node, f.NUMBER AS facility,
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

    // 데모 anomalies (위험 3 + warn 4 = 7) / watch (offline 3)
    let demoAnomalies = [], demoWatch = [];
    if (demoMode) {
      const dd = getDemoDevices();
      demoAnomalies = dd.filter((d) => d.status === "critical" || d.status === "warn").map((d) => ({
        node: d.deviceId, zone: zoneFromFacility(d.facility),
        label: d.riskLevel, mse: d.mse, threshold: d.threshold,
        contribution: [], ts: d.lastMeasured, demo: true,
      }));
      demoWatch = dd.filter((d) => d.status === "offline").map((d) => ({
        node: d.deviceId, zone: zoneFromFacility(d.facility),
        label: `통신 두절 ${fmtHours(d.hoursSilent)}`, mse: d.hoursSilent, threshold: 24,
        contribution: [], demo: true,
      }));
    }

    res.json({
      ok: true,
      demoMode,
      anomalies: [...demoAnomalies, ...anomalyRows.map((r) => ({
        node:  r.node,
        zone:  zoneFromFacility(r.facility),
        label: r.label,
        mse:   r.mse,
        threshold: -850,
        contribution: [],
        ts: r.ts,
      }))],
      watch: [...demoWatch, ...watchRows.map((r) => ({
        node:  r.node,
        zone:  zoneFromFacility(r.facility),
        label: `통신 두절 ${fmtHours(r.hoursSilent)}`,
        mse:   r.hoursSilent,
        threshold: 24,
        contribution: [],
      }))],
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
    // 검색용 풀은 limit 보다 넉넉히 — 적은 limit 으로도 검색이 의미 있도록
    const sqlLimit = Math.max(limit, 500);

    // audit_log (도구 호출, AI 동작 등) — 최근 7일
    const [audits] = await pool.query(`
      SELECT id, created_at, action, target_id, metadata_json
      FROM audit_log
      WHERE created_at > DATE_SUB(NOW(), INTERVAL 7 DAY)
        ${after ? "AND created_at > ?" : ""}
      ORDER BY created_at DESC LIMIT ${sqlLimit}
    `, after ? [after] : []);

    // kscg_alarm_log (옴니 원본 알람) — 최근 30일
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
      ORDER BY a.GEN_DATE DESC LIMIT ${sqlLimit}
    `, after ? [after] : []);

    const fmtTime = (d) => {
      const dt = new Date(d);
      const hh = String(dt.getHours()).padStart(2, "0");
      const mm = String(dt.getMinutes()).padStart(2, "0");
      const ss = String(dt.getSeconds()).padStart(2, "0");
      return `${hh}:${mm}:${ss}`;
    };

    // audit → 이벤트 변환 (도구 호출 표시)
    const auditEvents = audits.map((a) => {
      let meta = {};
      try { meta = a.metadata_json ? (typeof a.metadata_json === "string" ? JSON.parse(a.metadata_json) : a.metadata_json) : {}; } catch {}
      const argsTxt = meta.args ? Object.entries(meta.args).slice(0, 2).map(([k, v]) => `${k}:${String(v).slice(0, 16)}`).join(",") : "";
      const dur = meta.durationMs != null ? ` · ${meta.durationMs}ms` : "";
      const ok  = meta.ok === false ? " · ✗ 실패" : "";
      return {
        id:   `aud-${a.id}`,
        ts:   a.created_at,
        time: fmtTime(a.created_at),
        kind: meta.ok === false ? "warn" : "ai",
        text: `AI: ${a.target_id}(${argsTxt})${dur}${ok}`,
        source: "audit",
      };
    });

    // alarm → 이벤트 변환 (위험/경고/주의)
    const alarmEvents = alarms.map((a) => ({
      id:   `alm-${a.id}`,
      ts:   a.ts,
      time: fmtTime(a.ts),
      kind: a.grade === "위험" ? "alert" : "warn",
      text: `${a.grade || "ALARM"}: ${a.deviceId || "(unknown)"} · ${a.CONTENTS || ""} · 값 ${a.VALUE != null ? Number(a.VALUE).toFixed(2) : "-"}`,
      source: "alarm",
    }));

    // 통합 + 시간순 정렬 + 검색 필터 + LIMIT
    let merged = [...auditEvents, ...alarmEvents]
      .sort((x, y) => new Date(y.ts).getTime() - new Date(x.ts).getTime());
    if (q) merged = merged.filter((e) => e.text.toLowerCase().includes(q));
    merged = merged.slice(0, limit);

    res.json({ ok: true, count: merged.length, events: merged });
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

// ─────────────────────────────────────────────────────
// 📅 매일 자정 빈 일일 세션 강제 (cron) — launchd 가 00:01 에 호출
//   사용자 5/26 결정 — "새 세션이 날짜별로 새 세션이 되도록".
//   매일 00:01 에 "📅 YYYY-MM-DD 대화" 세션 ensure (이미 있으면 skip).
//   운영자가 그날 처음 챗봇 진입 시 자연스럽게 그 세션이 이미 만들어져 있음.
//   드롭다운 그룹 "오늘" 에 항상 1개 이상 세션 존재.
// ─────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────
// 📚 어제 일반 세션 → 1개로 merge (매일 자정 cron)
//   사용자 5/26 결정 — "다음날에 볼때는 어제 모든 세션이 합친 1개로 표시".
//   자동분석 세션 (🤖) 은 제외, 어제 일반 세션만 대상.
//   target = 어제 첫 일반 세션 (id 최소, 보통 빈 일일 세션 📅).
//   메시지 모두 target 으로 UPDATE → 빈 세션 DELETE → title "📚 YYYY-MM-DD 대화 통합".
// ─────────────────────────────────────────────────────
app.post("/api/admin/merge-yesterday-sessions", dbRequired, async (_req, res) => {
  try {
    const y = new Date();
    y.setDate(y.getDate() - 1);
    const yKey = `${y.getFullYear()}-${String(y.getMonth() + 1).padStart(2, "0")}-${String(y.getDate()).padStart(2, "0")}`;

    // 어제의 첫 일반 세션 찾기
    const [tRows] = await pool.query(`
      SELECT MIN(s.id) AS target FROM chat_sessions s
      JOIN chat_messages m ON m.session_id = s.id
      WHERE s.title NOT LIKE '🤖%' AND DATE(m.created_at) = ?
    `, [yKey]);
    const target = tRows[0]?.target;
    if (!target) {
      return res.json({ ok: true, dateKey: yKey, merged: 0, message: "어제 일반 세션 없음" });
    }

    // 다른 일반 세션 메시지 → target 으로 UPDATE
    await pool.query(`
      UPDATE chat_messages SET session_id = ?
      WHERE DATE(created_at) = ? AND session_id != ?
        AND session_id IN (SELECT * FROM (
          SELECT DISTINCT s.id FROM chat_sessions s
          JOIN chat_messages m ON m.session_id = s.id
          WHERE s.title NOT LIKE '🤖%' AND DATE(m.created_at) = ?
        ) x)
    `, [target, yKey, target, yKey]);

    // 빈 일반 세션 DELETE (자동분석 🤖 제외)
    const [delResult] = await pool.query(`
      DELETE FROM chat_sessions
      WHERE title NOT LIKE '🤖%'
        AND id != ?
        AND id NOT IN (SELECT DISTINCT session_id FROM chat_messages)
    `, [target]);

    // target title 갱신 + updated_at 도 메시지 마지막 시각으로 강제
    //   (그냥 UPDATE title 만 하면 ON UPDATE CURRENT_TIMESTAMP 가 발동되어
    //    어제 세션 updated_at 이 자정 시각으로 갱신 → 그룹 분류 '오늘' 로 잘못 떨어짐)
    await pool.query(
      `UPDATE chat_sessions SET title = ?,
         updated_at = COALESCE(
           (SELECT MAX(m.created_at) FROM chat_messages m WHERE m.session_id = ?),
           updated_at
         )
       WHERE id = ?`,
      [`📚 ${yKey} 대화 통합`, target, target],
    );

    console.log(`[merge-yesterday] dateKey=${yKey} target=${target} deleted=${delResult.affectedRows}`);
    res.json({ ok: true, dateKey: yKey, target, deletedSessions: delResult.affectedRows });
  } catch (err) {
    console.error("[merge-yesterday]", err.message);
    res.status(500).json({ ok: false, error: err.message });
  }
});

app.post("/api/admin/ensure-daily-session", dbRequired, async (_req, res) => {
  try {
    const now = new Date();
    const dateKey = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-${String(now.getDate()).padStart(2, "0")}`;
    const title = `📅 ${dateKey} 대화`;
    // 같은 title 이미 있으면 skip (idempotent)
    const [existing] = await pool.query(
      `SELECT id FROM chat_sessions WHERE title = ? ORDER BY created_at DESC LIMIT 1`,
      [title],
    );
    if (existing.length) {
      return res.json({ ok: true, sessionId: Number(existing[0].id), created: false, dateKey });
    }
    const [r] = await pool.query(`INSERT INTO chat_sessions (title) VALUES (?)`, [title]);
    console.log(`[daily-session] created session ${r.insertId} : ${title}`);
    res.json({ ok: true, sessionId: r.insertId, created: true, dateKey, title });
  } catch (err) {
    console.error("[ensure-daily-session]", err.message);
    res.status(500).json({ ok: false, error: err.message });
  }
});

// ─────────────────────────────────────────────────────
// 🤖 AI 자동 분석 (cron) — 매시 정각 launchd 가 호출
//   1. 군산 날씨 fetch (Open-Meteo)
//   2. 도구 5개 호출로 시스템 현황 수집
//   3. LLM 한 번 호출 → C 톤 풍부 메시지 생성
//   4. "AI 자동 분석 — YYYY-MM-DD" 일일 세션에 system role 로 INSERT
// ─────────────────────────────────────────────────────

// 서버측 군산 날씨 fetch (Open-Meteo)
const WMO_KO = {
  0:"맑음",1:"대체로 맑음",2:"부분 흐림",3:"흐림",45:"안개",48:"안개",
  51:"약한 이슬비",53:"이슬비",55:"강한 이슬비",
  61:"약한 비",63:"비",65:"강한 비",
  71:"약한 눈",73:"눈",75:"강한 눈",77:"눈알갱이",
  80:"소나기",81:"소나기",82:"강한 소나기",
  85:"눈 소나기",86:"눈 소나기",
  95:"뇌우",96:"뇌우(우박)",99:"강한 뇌우",
};
async function fetchGunsanWeather() {
  try {
    const url = `https://api.open-meteo.com/v1/forecast?latitude=35.9678&longitude=126.7369&current=temperature_2m,weather_code,precipitation,relative_humidity_2m&timezone=Asia%2FSeoul`;
    const r = await fetch(url, { signal: AbortSignal.timeout(8000) });
    if (!r.ok) return null;
    const j = await r.json();
    if (!j.current) return null;
    const code = j.current.weather_code;
    return {
      temp:     Math.round(j.current.temperature_2m),
      ko:       WMO_KO[code] || "-",
      code,
      precip:   j.current.precipitation,
      humidity: j.current.relative_humidity_2m != null ? Math.round(j.current.relative_humidity_2m) : null,
      time:     j.current.time,
    };
  } catch (e) {
    console.warn("[fetchGunsanWeather]", e.message);
    return null;
  }
}

// "AI 자동 분석 — YYYY-MM-DD" 세션 ensure (오늘 세션 없으면 생성)
async function ensureAutoInsightSession(dateKey) {
  if (!pool) return null;
  const title = `🤖 AI 자동 분석 — ${dateKey}`;
  try {
    const [rows] = await pool.query(
      `SELECT id FROM chat_sessions WHERE title = ? ORDER BY created_at DESC LIMIT 1`,
      [title],
    );
    if (rows.length) return Number(rows[0].id);
    const [r] = await pool.query(`INSERT INTO chat_sessions (title) VALUES (?)`, [title]);
    return r.insertId;
  } catch (e) {
    console.warn("[ensureAutoInsightSession]", e.message);
    return null;
  }
}

// 자동 분석 prompt 빌더 (C 톤 풍부 형식 강제)
//   시각 기준 = 정시 (매시 00분 floor). 운영자가 14:23 에 눌러도 메시지는 "14:00" 기준.
//   다음 분석은 항상 다음 정시 (HH+1):00.
function buildAutoInsightPrompt(weather, dataBundle) {
  const { summary, offlineList, criticalList, lowVolt, recentAlarms } = dataBundle;
  // 정시 floor (Asia/Seoul) — 분/초 = 00 으로 잘라냄
  const nowDate = new Date();
  const fmtHour = (d) =>
    `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")} ${String(d.getHours()).padStart(2, "0")}:00`;
  const hourFloor = new Date(nowDate.getFullYear(), nowDate.getMonth(), nowDate.getDate(), nowDate.getHours(), 0, 0);
  const nextHour  = new Date(hourFloor.getTime() + 60 * 60 * 1000);
  const currentSlot = fmtHour(hourFloor);   // "2026-05-26 14:00"
  const nextSlot    = fmtHour(nextHour);    // "2026-05-26 15:00"
  const weatherLine = weather
    ? `${weather.ko} · ${weather.temp}°C${weather.precip != null ? ` · 강수 ${weather.precip}mm` : ""}${weather.humidity != null ? ` · 습도 ${weather.humidity}%` : ""}`
    : "(데이터 없음)";
  return `당신은 매설배관 IoT 통합관제 시스템의 AI 분석 엔진입니다.
운영자가 챗봇에 접속할 때마다 자동으로 push 되는 시간별 분석 메시지를 작성합니다.
한국어 존댓말. 답변 외 메타·설명 X. 메시지 본문만 출력.

# 분석 기준 시각 (정시 기준 — 매시 00분 floor)
- 현재 분석 시각: ${currentSlot}
- 다음 분석 시각: ${nextSlot}
- 메시지 본문에 이 두 시각을 정확히 인용 (다른 분/초 표현 금지)

# 현재 군산 날씨 (Open-Meteo)
${weatherLine}

# 시스템 KPI
${JSON.stringify(summary)}

# 통신 두절 단말 목록 (상위 5)
${JSON.stringify(offlineList)}

# 위험 단말 목록 (최근 7일 알람 발생, 상위 5)
${JSON.stringify(criticalList)}

# 방식전위 -800 mV 이상 단말 (부식 진행 가능)
${JSON.stringify(lowVolt)}

# 최근 24시간 알람
${JSON.stringify(recentAlarms)}

# 출력 형식 (반드시 이대로 · 시각은 위 "분석 기준 시각" 정확히 인용)

🤖 자동 분석 · ${currentSlot} ── 분석 엔진 가동중

🌦 군산 현재 — {날씨 한 줄}

🔔 즉각 조치
{즉시 조치가 필요한 단말 1~2건. 날씨 영향 고려. 데이터 근거 명시.
 없으면 "현재 즉각 조치 대상 없음. 모니터링 지속."}

📈 트렌드 예측
{향후 N시간 변화 예측 + [추정] 확률. 날씨가 향후에 미칠 영향 1줄.}

✓ 우선 점검 권고
1. {단말 또는 구역} ── {긴급도 (즉시/24시간/1주)}
2. ...
3. ...

다음 분석 ${nextSlot}

# 규칙
- 수치는 도구 데이터 그대로 인용. 추측 결론은 [추정] 라벨
- 데이터 없는 항목은 "없음" 명시 (만들지 말 것)
- 마크다운 **굵게** 만 사용 (헤더 ## X)
- 5문장 이상 길게 쓰지 말 것 (각 섹션 2~4문장)
- 시각 표기는 절대 "오후 2:01" 같은 분/초 X. 무조건 "${currentSlot}" / "${nextSlot}" 형태`;
}

app.post("/api/admin/run-auto-insight", dbRequired, async (_req, res) => {
  const t0 = Date.now();
  try {
    // 1. 날씨
    const weather = await fetchGunsanWeather();

    // 2. 데이터 수집 (도구 직접 호출 — 캐시 hit 도 OK)
    const [summary, offlineList, criticalList, lowVolt, recentAlarms] = await Promise.all([
      execTool("get_summary", {}),
      execTool("list_devices", { status: "offline", limit: 5 }),
      execTool("list_devices", { status: "critical", limit: 5 }),
      execTool("find_devices_by_value", { metric: "volt", op: "gte", threshold: -800, limit: 5 }),
      execTool("get_alarms", { days: 1, limit: 5 }),
    ]);

    // 3. LLM 호출 (도구 X — 데이터는 이미 prompt 에 포함)
    const systemPrompt = buildAutoInsightPrompt(weather, { summary, offlineList, criticalList, lowVolt, recentAlarms });
    const ollamaRes = await fetch(`${OLLAMA_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: OLLAMA_MODEL,
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user",   content: "지금 분석 메시지를 작성하세요." },
        ],
        stream: false,
        think: false,
        options: { temperature: 0.4, num_predict: 800 },
      }),
      signal: AbortSignal.timeout(60_000),
    });
    if (!ollamaRes.ok) throw new Error(`Ollama HTTP ${ollamaRes.status}`);
    const data = await ollamaRes.json();
    const text = String(data.message?.content || "").trim();
    if (!text) throw new Error("LLM empty response");

    // 4. 자동 분석 세션 ensure + INSERT
    const now = new Date();
    const dateKey = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-${String(now.getDate()).padStart(2, "0")}`;
    const sid = await ensureAutoInsightSession(dateKey);
    if (!sid) throw new Error("session ensure failed");
    await persistMessage(
      sid, "system", text,
      { auto: true, weather, summary, hourlyTrigger: now.toISOString() },
      { prompt: data.prompt_eval_count, completion: data.eval_count },
      OLLAMA_MODEL,
    );

    const elapsedMs = Date.now() - t0;
    console.log(`[ai-insight] session=${sid} · ${elapsedMs}ms · ${text.length} chars`);
    res.json({ ok: true, sessionId: sid, dateKey, text, weather, elapsedMs });
  } catch (err) {
    console.error("[run-auto-insight]", err.message);
    res.status(500).json({ ok: false, error: err.message });
  }
});

// ── 챗봇 세션 관리 ───────────────────────────────────
// GET    /api/chat/sessions          — 세션 목록 (최근 30)
// GET    /api/chat/sessions/:id      — 세션 + 메시지
// POST   /api/chat/sessions          — 새 세션
// DELETE /api/chat/sessions/:id      — 세션 삭제

app.get("/api/chat/sessions", dbRequired, async (_req, res) => {
  try {
    const [rows] = await pool.query(`
      SELECT s.id, s.title, s.created_at, s.updated_at,
             (SELECT COUNT(*) FROM chat_messages WHERE session_id = s.id) AS messageCount
      FROM chat_sessions s
      ORDER BY s.updated_at DESC LIMIT 30
    `);
    res.json({ ok: true, count: rows.length, sessions: rows });
  } catch (err) {
    console.error("[/api/chat/sessions]", err);
    res.status(500).json({ ok: false, error: err.message });
  }
});

app.get("/api/chat/sessions/:id", dbRequired, async (req, res) => {
  try {
    const id = parseInt(req.params.id, 10);
    if (!Number.isFinite(id)) return res.status(400).json({ ok: false, error: "id 숫자" });
    const [sess] = await pool.query(`SELECT * FROM chat_sessions WHERE id = ?`, [id]);
    if (sess.length === 0) return res.status(404).json({ ok: false, error: "session not found" });
    const [msgs] = await pool.query(`
      SELECT role, text, tokens_prompt AS tokensPrompt, tokens_completion AS tokensCompletion,
             model, created_at AS createdAt
      FROM chat_messages WHERE session_id = ?
      ORDER BY created_at, id LIMIT 200
    `, [id]);
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
    await pool.query(`DELETE FROM chat_messages WHERE session_id = ?`, [id]);
    const [r] = await pool.query(`DELETE FROM chat_sessions WHERE id = ?`, [id]);
    res.json({ ok: true, deleted: r.affectedRows });
  } catch (err) {
    console.error("[DELETE /api/chat/sessions/:id]", err);
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

// SPA fallback (모르는 경로 → index.html)
app.get("*", (_req, res) => {
  res.sendFile(path.join(__dirname, "dist", "index.html"));
});

app.listen(PORT, () => {
  console.log(`▶ Server  http://localhost:${PORT}`);
  console.log(`▶ Ollama  ${OLLAMA_URL}`);
  console.log(`▶ Model   ${OLLAMA_MODEL}`);
});

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
    ? `전체 ${counts.all}대 / 정상 ${counts.normal ?? 0} · 위험 ${counts.critical ?? 0} · 이상 의심 ${counts.warn ?? 0} · 통신 장애 ${counts.offline ?? 0}`
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
    ? `${weather.ko} · ${weather.temp}°C${weather.precip != null ? ` · 강수 ${weather.precip}mm` : ""}${weather.humidity != null ? ` · 습도 ${weather.humidity}%` : ""} (군산, ${weather.time})`
    : "(데이터 없음)";

  // 12시간 추이 텍스트 표 (위험·이상 의심만)
  const trendBlock = trends.length === 0 ? "- (위험·이상 의심 노드 없음)" :
    trends.map((t) => {
      const h = t.mseHistory || [];
      const start = h[0] ?? "-";
      const peak  = h.length ? Math.max(...h.filter((v) => v != null)) : "-";
      const last  = h[h.length - 1] ?? "-";
      const dir   = (h.length >= 2 && h[0] != null && h[h.length - 1] != null)
        ? (h[h.length - 1] > h[0] ? "상승↑" : h[h.length - 1] < h[0] ? "하락↓" : "평탄→")
        : "—";
      const series = h.map((v) => v == null ? "-" : v.toFixed(2)).join(",");
      return `- ${t.deviceId} (${t.status === "critical" ? "위험" : "이상의심"}, ${t.zone}, ${t.label || "-"}): 12h MSE [${series}] · 시작 ${start} → 현재 ${last} · 피크 ${peak} · 방향 ${dir}`;
    }).join("\n");

  return `당신은 매설배관 IoT 통합관제 시스템의 AI 분석 어시스턴트입니다.
운영자(관제사)와 한국어 존댓말로 대화하며, 노드 ID·위험 단계·도메인 용어 질문에 답합니다.

# 데이터 흐름 (옴니솔루션 답변 — 2026-05-18)
- 옴니 단말은 **1시간에 1회 계측** → **12시간 burst 로 KSCG 송신**
- 우리 미러 sync: alarm 1h / sensor 2h / meta 6h (옴니 권고 반영)
- 따라서 "실시간" 데이터는 사실상 평균 6~14시간 지연 가능
- "마지막 측정 시각" = 단말이 측정한 시간 (1h 단위). 전송과는 다름.

# 도메인 지식
- **방식전위(P/S Potential)**: 매설배관 부식 보호 지표. -850mV 이하 양호, 초과 시 부식 진행 가능.
- **희생전류(Sacrificial Current)**: 희생양극→배관 보호 전류. 점차 감소 시 양극 소모/접속부 불량. 1mA 이하 교체 검토. (희생양극 단말은 TB24-250406, TB24-250407 2대만 해당)
- **AC 유입**: 송전선·전철 유도 교류. 200mV 이상 가속 부식, 500mV 이상 즉각 차폐/배수장치 점검.
- **통신 품질(dBm)**: -65 이상 양호, -75 이하 주의, -85 이하 두절 임박, -115 이하 두절.

# AI 모델 (LSTM AutoEncoder) — 위험도 판정 정확 명세
- 모델은 단말별로 학습한 **정상 패턴 복원 오차(MSE)** 로 이상 탐지
- 단말마다 **threshold** 가 다름 (학습 시점 정상 MSE 분포의 99 percentile). 자세한 값은 **get_ai_model_info(deviceId)** 도구로 조회
- **3단계 분류 (비율 기준)**:
  - **정상** — 현재 MSE < threshold × 0.70
  - **관찰** — threshold × 0.70 ≤ 현재 MSE ≤ threshold × 1.00
  - **이상** — 현재 MSE > threshold × 1.00
- 답변 시 가능하면 "현재 MSE 가 threshold 의 N% 도달" 같이 비율로 설명 (절대값 단독은 의미 약함)
- **중요 구분**: 측정 센서 8 종(방식전위/희생전류/AC유입/배터리/온도/습도/충격/통신) ≠ AI 학습 입력 피처. AI 학습 입력 피처·시퀀스 길이·epoch 등 모델 세부 명세는 절대 추측 금지, 반드시 **get_ai_model_info** 도구 호출해서 확인하세요. (예: 학습 base_features 는 4개, 파생 포함 12개 컬럼 — 도구 응답으로 확정)

# 위험 단계 (5단계)
- 정상 / 위험(즉각 현장 점검) / 이상 의심 / 통신 장애

# 현재 시각
${nowText}

# 현재 날씨 (군산 — SITE_ID=2 대상 지역)
${weatherLine}

날씨가 매설배관에 미치는 영향 (참고):
- 강한 비/소나기/뇌우 → 침수·습도 상승 → 맨홀 침수, 통신 두절 가능
- 한파/혹한 → 토양 동결 → 방식전위 변동
- 폭염/일교차 → 온도 센서 이상

# 현재 시스템 상태 ${hasContext ? "(실시간)" : "(컨텍스트 미전달 — 도구 조회 필요)"}
- ${summaryLine}
${hasContext ? (criticalNodes.length ? `- 위험 노드: ${criticalNodes.join(", ")}` : "- 위험 노드: 없음") : ""}
${hasContext && warnNodes.length    ? `- 이상 의심 노드(상위 ${Math.min(warnNodes.length, 8)}): ${warnNodes.slice(0, 8).join(", ")}` : ""}
${hasContext && offlineNodes.length ? `- 통신 장애 노드: ${offlineNodes.join(", ")}` : ""}

${offlineBlock ? `# 통신 장애 노드 상세 (마지막 측정 시각 + 두절 기간)\n${offlineBlock}\n` : ""}
# 최근 12시간 MSE 추이 (1시간 간격, 가장 오래된 → 현재)
${trendBlock}

# 도구(Tools) 사용 가이드 — 18 개 도구 (자가확장 모드 ON)
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
- **get_predictions** — AI LSTM 예측 결과 (현재 데이터 비어있을 수 있음 — message 필드 확인).

**위치/지도 (3)**
- **search_devices_by_location** — 지명 키워드로 단말 검색 (DB POSITION LIKE). "미룡동", "시청 앞", "버스터미널" 같은 DB 텍스트 매칭 가능한 경우 1차 시도.
- **geocode_location** — 지명/랜드마크 → 좌표 (OpenStreetMap). 일반 지명(예: '은파호수공원', '군산교도소') 으로 좌표 모를 때.
- **find_devices_near** — 좌표 + 반경(km) 안 단말. geocode 결과 받아서 사용. 반경 기본 2km.

**AI 모델 (1)**
- **get_ai_model_info** — LSTM AutoEncoder 학습 정보. deviceId 주면 그 단말 threshold + 분류 기준, 없으면 전체 모델 메타(학습 피처, time_steps, 평가 통계 등). "AI 어떻게 학습됐어?", "TB24-XXX 정상 한계는?" 류.

**자가확장 (2) — 위 17개로 답 안 되는 모든 분석 질문**
- **describe_table** — siwon DB 임의 테이블 스키마 (컬럼·타입·인덱스·row 추정). 자유 SQL 작성 전 자가 탐색용. 예: 'kscg_sensor_data', 'audit_log', 'chat_messages'.
- **execute_safe_sql** — 자유 MySQL SELECT/WITH. siwon DB 모든 테이블 접근. 5초 timeout / 1000 row cap / DML/DDL 차단 / 다중 statement 차단. **위 17개 도구로 안 되는 어떤 분석이든 SQL 로 직접 해결**:
  - "월별 평균 방식전위" → GROUP BY DATE_FORMAT(WRITE_DATE, '%Y-%m')
  - "TB24-250425 의 최근 1주일 일별 통계" → SELECT DATE, AVG, MIN, MAX
  - "audit_log 에서 가장 자주 호출된 도구 TOP 5" → 자기성찰 메타 질의
  - "센서값이 가장 많이 변동한 단말" → STDDEV 정렬
  - "특정 시간대 (예: 새벽 2~4시) 측정 빈도" → HOUR() 필터
  - 등등 모든 자유 분석. 모르는 컬럼/테이블은 describe_table 먼저.

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



# 응답 규칙
1. **간결** — 2~5문장. 인사말·사과 절대 금지. 바로 본론.
2. **노드 ID 인용** — 위 상태/추이 표에 있는 노드 ID 를 그대로 답변에 포함.
3. **추이 표는 시간 데이터** — 위 "최근 12시간 MSE 추이" 표가 곧 과거 데이터입니다. "과거 시점 정보가 없다" 는 답변 절대 금지. 표의 12개 값이 1시간 간격이므로 "약 N시간 전" 표현 가능.
4. **통신 장애 시점** — "통신 장애 노드 상세" 섹션에 마지막 측정 시각과 두절 기간이 명시되어 있습니다. "언제 끊겼는지 모름" 답변 절대 금지. 마지막 측정 시각 = 통신 두절 시작 시점으로 보고 답변하세요.
5. **환각 금지** — 위 표·섹션에 없는 데이터만 "확인되지 않음".
6. **운영 친화** — 가능하면 "현장 점검 권장" 등 짧은 액션 한 줄.
7. **포맷** — 마크다운 헤더(##) X. **굵게**(**TB24-5JN011**) 정도만.

# 응답 예시 (이대로 따라할 것)

질문: "위험이 언제 발생했어?"
좋은 답변:
> **TB24-5JN011** 은 12시간 전 0.42 에서 시작해 약 8시간 전부터 임계 0.85 를 초과했습니다. **TB24-5JN012** 도 12시간 전 0.38 에서 시작해 약 9시간 전부터 임계를 넘었습니다. 두 노드 모두 현재까지 상승 추세이므로 즉시 현장 점검이 필요합니다.

나쁜 답변 (금지):
> 죄송하지만 과거 시점 정보는 확인되지 않습니다. (X — 위 추이 표가 있음)

질문: "TB24-5JN042 추세는?"
좋은 답변:
> **TB24-5JN042** 의 MSE 는 12시간 전 0.41 → 현재 0.84 로 지속 상승 중입니다. 임계 0.85 직전이라 즉각 점검을 권장합니다.

질문: "통신 장애는 언제부터 발생함?"
좋은 답변:
> **TB24-250429** 의 마지막 측정이 2026-04-28 01:00 입니다. 그 이후 480시간(약 20일) 통신 두절 상태로, 4/28 새벽에 단절된 것으로 보입니다. 현장 점검 (전원·안테나·맨홀 침수 확인) 즉시 필요합니다.

질문: "방식전위"
좋은 답변:
> 방식전위는 매설배관 부식 보호 지표로 -850 mV 이하가 양호 기준입니다. 초과 시 부식 진행 가능성이 있어 정류기 출력 조정이 필요합니다.
`;
}
