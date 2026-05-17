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
import { fileURLToPath } from "url";
import mysql from "mysql2/promise";

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

const PORT         = process.env.PORT          || 5050;
const OLLAMA_URL   = process.env.OLLAMA_URL    || "http://localhost:11434";
const OLLAMA_MODEL = process.env.OLLAMA_MODEL  || "qwen3.5:9b";

const SIWON_DB_HOST = process.env.SIWON_DB_HOST || "127.0.0.1";
const SIWON_DB_PORT = parseInt(process.env.SIWON_DB_PORT || "3306", 10);
const SIWON_DB_USER = process.env.SIWON_DB_USER || "siwon_app";
const SIWON_DB_PASS = process.env.SIWON_DB_PASS || "";
const SIWON_DB_NAME = process.env.SIWON_DB_NAME || "siwon";
const SITE_ID       = parseInt(process.env.SITE_ID || "2", 10);  // 군산도시가스

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
          metric: { type: "string", enum: ["volt","ac","temp","hum","battery","commDbm"], description: "측정 종류" },
          op:     { type: "string", enum: ["avg","max","min"], default: "avg" }
        },
        required: ["metric"]
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
async function execTool(name, args) {
  if (!pool) return { error: "DB pool 비활성" };
  args = args || {};
  try {
    switch (name) {
      // 단말 목록
      //   주의: status 필터는 JS 후처리 (DB 컬럼이 아니라 lastSeen + 알람 카운트로 계산).
      //         그래서 SQL 단에서 LIMIT 걸면 안 됨 → 전체 가져온 뒤 filter → slice.
      case "list_devices": {
        const limit = Math.min(Number(args.limit) || 20, 60);
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
        const filtered = annotated.filter((r) => !args.status || args.status === "all" || r.status === args.status);
        return {
          totalScanned: annotated.length,
          count: Math.min(filtered.length, limit),
          devices: filtered.slice(0, limit),
        };
      }

      // 단말 상세 (8 센서 최신값)
      case "get_device_detail": {
        const deviceId = args.deviceId;
        if (!deviceId) return { error: "deviceId 필수" };
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
        return {
          ...meta[0],
          zone: zoneFromFacility(meta[0]?.facility),
          sensors: sensorVals,
          lastMeasured,
          hoursSilent,
          status: hoursSilent != null && hoursSilent >= 24 ? "offline" : "normal",
        };
      }

      // 단말 시계열
      case "get_device_history": {
        const deviceId = args.deviceId;
        const kind     = args.kind || "volt";
        const range    = args.range || "24h";
        const seq      = SENSOR_SEQ_KIND.indexOf(kind) + 1;
        if (seq < 1) return { error: `unknown kind: ${kind}` };
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
        return { count: rows.length, alarms: rows };
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
        return { metric, op, result: Number(rows[0].result?.toFixed(2)) };
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
async function runChatWithTools(messages, signal) {
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
      console.log(`[tool] round ${round + 1} → ${name}(${JSON.stringify(args)})`);
      const result = await execTool(name, args);
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
  const { message, context = {}, history = [] } = req.body || {};
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

  try {
    const ctrl = new AbortController();
    const timeout = setTimeout(() => ctrl.abort(), 120_000); // 120s (최대 5 tool 라운드 여유)
    const result = await runChatWithTools(messages, ctrl.signal);
    clearTimeout(timeout);

    return res.json({
      ok:        true,
      reply:     (result.content || "(빈 응답)").trim(),
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
  const { message, context = {}, history = [] } = req.body || {};
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
        send("done", {
          reply:     finalAccum.trim(),
          model:     OLLAMA_MODEL,
          rounds:    round + 1,
          toolCalls: toolTrace,
          tokens:    lastTokens,
        });
        clearTimeout(timeout);
        return res.end();
      }

      // tool_calls 가 있으면 실행 → result append → 다음 라운드
      working.push({ role: "assistant", content: roundContent, tool_calls: toolCalls });
      for (const tc of toolCalls) {
        const name = tc.function?.name;
        const args = tc.function?.arguments || {};
        console.log(`[stream tool] round ${round + 1} → ${name}(${JSON.stringify(args)})`);
        send("tool", { round: round + 1, name, args });
        const result = await execTool(name, args);
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

// ── GET /api/summary — KPI 카운트 ─────────────────────
app.get("/api/summary", dbRequired, async (_req, res) => {
  try {
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
    const warn     = 0;  // TODO: LSTM 예측 → ai_predictions 연계 시 채우기
    const normal   = total - offline - critical - warn;

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
app.get("/api/devices", dbRequired, async (_req, res) => {
  try {
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

    res.json({ ok: true, site_id: SITE_ID, count: out.length, devices: out });
  } catch (err) {
    console.error("[/api/devices]", err);
    res.status(500).json({ ok: false, error: err.message });
  }
});

// ── GET /api/devices/:id/history — 단말 시계열 추이 ───
//   query: range=1h|24h|7d  (default 24h)  kind=volt|ac|temp|hum|... (default volt)
app.get("/api/devices/:id/history", dbRequired, async (req, res) => {
  try {
    const id      = parseInt(req.params.id, 10);
    const range   = req.query.range || "24h";
    const kind    = req.query.kind  || "volt";
    const seq     = SENSOR_SEQ_KIND.indexOf(kind) + 1;
    if (seq < 1) return res.status(400).json({ ok: false, error: `unknown kind: ${kind}` });

    const hours   = range === "1h" ? 1 : range === "7d" ? 168 : 24;

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
    res.json({ ok: true, count: rows.length, alarms: rows });
  } catch (err) {
    console.error("[/api/alarms]", err);
    res.status(500).json({ ok: false, error: err.message });
  }
});

// ── GET /api/anomalies — AI 탐지 (이상 의심 + 관찰) ──
//   현재 LSTM 예측(ai_predictions) 미연동 → KSCG 알람 + 통신 두절로 임시 매핑.
//   anomalies = 최근 7일 알람 발생 단말, watch = 24h 통신 두절 단말
app.get("/api/anomalies", dbRequired, async (_req, res) => {
  try {
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

    res.json({
      ok: true,
      anomalies: anomalyRows.map((r) => ({
        node:  r.node,
        zone:  zoneFromFacility(r.facility),
        label: r.label,
        mse:   r.mse,
        threshold: -850,
        contribution: [],
        ts: r.ts,
      })),
      watch: watchRows.map((r) => ({
        node:  r.node,
        zone:  zoneFromFacility(r.facility),
        label: `통신 두절 ${r.hoursSilent}h`,
        mse:   r.hoursSilent,
        threshold: 24,
        contribution: [],
      })),
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

  const summaryLine =
    `전체 ${counts.all ?? 0}대 / 정상 ${counts.normal ?? 0} · 위험 ${counts.critical ?? 0} · 이상 의심 ${counts.warn ?? 0} · 통신 장애 ${counts.offline ?? 0}`;

  // 통신 두절 상세 (마지막 측정 시각 + 끊긴 시간)
  const offlineBlock = offlineDetails.length === 0 ? "" :
    offlineDetails.map((o) => {
      const last = o.updatedAt ? new Date(o.updatedAt).toLocaleString("ko-KR", { timeZone: "Asia/Seoul" }) : "확인 불가";
      const hrs  = o.hoursSilent != null ? o.hoursSilent : "?";
      const days = o.hoursSilent != null ? Math.floor(o.hoursSilent / 24) : "?";
      return `  · ${o.deviceId} (${o.zone || "-"}, ${o.location || "-"}): 마지막 측정 ${last} · 두절 ${hrs}시간 (≈${days}일)`;
    }).join("\n");

  // 날씨 라인 (있을 때만)
  const weatherLine = weather
    ? `${weather.ko} · ${weather.temp}°C (서울, ${weather.time})`
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

# 도메인 지식
- **방식전위(P/S Potential)**: 매설배관 부식 보호 지표. -850mV 이하 양호, 초과 시 부식 진행 가능.
- **희생전류(Sacrificial Current)**: 희생양극→배관 보호 전류. 점차 감소 시 양극 소모/접속부 불량. 1mA 이하 교체 검토.
- **AC 유입**: 송전선·전철 유도 교류. 200mV 이상 가속 부식, 500mV 이상 즉각 차폐/배수장치 점검.
- **통신 품질(dBm)**: -65 이상 양호, -75 이하 주의, -85 이하 두절 임박.
- **MSE 임계**: 0.85 이상 = 위험(즉각 조치), 0.28~0.85 = 이상 의심(추세 모니터링).

# 위험 단계 (5단계)
- 정상 / 위험(즉각 현장 점검) / 이상 의심 / 통신 장애

# 현재 시각
${nowText}

# 현재 날씨 (서울)
${weatherLine}

날씨가 매설배관에 미치는 영향 (참고):
- 강한 비/소나기/뇌우 → 침수·습도 상승 → 맨홀 침수, 통신 두절 가능
- 한파/혹한 → 토양 동결 → 방식전위 변동
- 폭염/일교차 → 온도 센서 이상

# 현재 시스템 상태 (실시간)
- ${summaryLine}
${criticalNodes.length ? `- 위험 노드: ${criticalNodes.join(", ")}` : "- 위험 노드: 없음"}
${warnNodes.length    ? `- 이상 의심 노드(상위 ${Math.min(warnNodes.length, 8)}): ${warnNodes.slice(0, 8).join(", ")}` : "- 이상 의심 노드: 없음"}
${offlineNodes.length ? `- 통신 장애 노드: ${offlineNodes.join(", ")}` : ""}

${offlineBlock ? `# 통신 장애 노드 상세 (마지막 측정 시각 + 두절 기간)\n${offlineBlock}\n` : ""}
# 최근 12시간 MSE 추이 (1시간 간격, 가장 오래된 → 현재)
${trendBlock}

# 도구(Tools) 사용 가이드
위 "현재 시스템 상태" 와 "12h MSE 추이" 에 이미 있는 정보면 그대로 답변. 없는 정보(특정 단말 상세, 시리얼/설치일/위경도, 시계열 추이, 알람 이력 등)는 아래 도구를 직접 호출해서 조회:
- **list_devices** — 단말 목록 (status/zone 필터). 단말 ID 후보 좁히기.
- **get_device_detail** — 단말 메타 + 8 센서 최신값. "TB24-XXXXXX 상태", "방식전위가 얼만지" 등.
- **get_device_history** — 시계열 (1h/24h/7d/30d). "추이", "어제부터", "변화" 류 질문.
- **get_alarms** — 최근 알람 이력. "위험 알람", "어제 알람" 등.
- **get_summary** — KPI 카운트. (이미 위에 있으면 호출 불필요)
- **get_aggregate** — 전체 평균/최대/최소. "평균 방식전위", "최저 RSSI" 등.

도구를 부른 후엔 받은 JSON 의 실제 값을 근거로 답하고, **추측 금지**. 호출 결과가 비어있거나 error 면 "데이터 없음" 으로 정직하게 답변.

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
