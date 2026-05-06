/**
 * 통합 서버 (Express)
 *  - 정적 파일 (dist/) 서빙
 *  - POST /api/chat → Ollama 프록시 (시스템 프롬프트 + 컨텍스트 주입)
 *
 * 실행: node server.js
 *   PORT (default 5050)
 *   OLLAMA_URL (default http://localhost:11434)
 *   OLLAMA_MODEL (default gemma4:e4b)
 *
 * 같은 origin 으로 정적 파일과 API 동시 서비스 → CORS 불필요.
 * 프론트는 fetch("/api/chat") 으로 호출.
 */

import express from "express";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

const PORT         = process.env.PORT          || 5050;
const OLLAMA_URL   = process.env.OLLAMA_URL    || "http://localhost:11434";
const OLLAMA_MODEL = process.env.OLLAMA_MODEL  || "gemma4:e4b";

const app = express();
app.use(express.json({ limit: "1mb" }));

// ── /api/health ──────────────────────────────────────
app.get("/api/health", async (_req, res) => {
  try {
    const r = await fetch(`${OLLAMA_URL}/api/version`);
    const v = r.ok ? await r.json() : null;
    res.json({ ok: true, ollama: v, model: OLLAMA_MODEL });
  } catch (err) {
    res.json({ ok: false, error: err.message, model: OLLAMA_MODEL });
  }
});

// ── /api/chat ────────────────────────────────────────
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

  const ollamaPayload = {
    model: OLLAMA_MODEL,
    messages: [
      { role: "system", content: systemPrompt },
      ...recent,
      { role: "user", content: message },
    ],
    stream: false,
    options: { temperature: 0.3, num_predict: 700 },
  };

  try {
    const ctrl = new AbortController();
    const timeout = setTimeout(() => ctrl.abort(), 60_000); // 60s
    const ollamaRes = await fetch(`${OLLAMA_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(ollamaPayload),
      signal: ctrl.signal,
    });
    clearTimeout(timeout);

    if (!ollamaRes.ok) {
      const txt = await ollamaRes.text().catch(() => "");
      throw new Error(`Ollama HTTP ${ollamaRes.status}: ${txt}`);
    }
    const data = await ollamaRes.json();
    const reply = (data.message && data.message.content) || "(빈 응답)";
    return res.json({
      ok:    true,
      reply: reply.trim(),
      model: OLLAMA_MODEL,
      tokens: { prompt: data.prompt_eval_count, completion: data.eval_count },
    });
  } catch (err) {
    console.error("[chat] error:", err.message);
    return res.status(500).json({ ok: false, error: err.message });
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
  const counts        = ctx.counts || {};
  const criticalNodes = ctx.criticalNodes || [];
  const warnNodes     = ctx.warnNodes || [];
  const offlineNodes  = ctx.offlineNodes || [];
  const trends        = ctx.trends || [];
  const nowText       = ctx.nowText || "현재";

  const summaryLine =
    `전체 ${counts.all ?? 0}대 / 정상 ${counts.normal ?? 0} · 위험 ${counts.critical ?? 0} · 이상 의심 ${counts.warn ?? 0} · 통신 장애 ${counts.offline ?? 0}`;

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

# 현재 시스템 상태 (실시간)
- ${summaryLine}
${criticalNodes.length ? `- 위험 노드: ${criticalNodes.join(", ")}` : "- 위험 노드: 없음"}
${warnNodes.length    ? `- 이상 의심 노드(상위 ${Math.min(warnNodes.length, 8)}): ${warnNodes.slice(0, 8).join(", ")}` : "- 이상 의심 노드: 없음"}
${offlineNodes.length ? `- 통신 장애 노드: ${offlineNodes.join(", ")}` : ""}

# 최근 12시간 MSE 추이 (1시간 간격, 가장 오래된 → 현재)
${trendBlock}

# 응답 규칙
1. **간결** — 2~5문장. 인사말·사과 절대 금지. 바로 본론.
2. **노드 ID 인용** — 위 상태/추이 표에 있는 노드 ID 를 그대로 답변에 포함.
3. **추이 표는 시간 데이터** — 위 "최근 12시간 MSE 추이" 표가 곧 과거 데이터입니다. "과거 시점 정보가 없다" 는 답변 절대 금지. 표의 12개 값이 1시간 간격이므로 "약 N시간 전" 표현 가능.
4. **환각 금지** — 표에 없는 데이터(예: 12시간 이전, 다른 센서 시계열)만 "확인되지 않음".
5. **운영 친화** — 가능하면 "현장 점검 권장" 등 짧은 액션 한 줄.
6. **포맷** — 마크다운 헤더(##) X. **굵게**(**TB24-5JN011**) 정도만.

# 응답 예시 (이대로 따라할 것)

질문: "위험이 언제 발생했어?"
좋은 답변:
> **TB24-5JN011** 은 12시간 전 0.42 에서 시작해 약 8시간 전부터 임계 0.85 를 초과했습니다. **TB24-5JN012** 도 12시간 전 0.38 에서 시작해 약 9시간 전부터 임계를 넘었습니다. 두 노드 모두 현재까지 상승 추세이므로 즉시 현장 점검이 필요합니다.

나쁜 답변 (금지):
> 죄송하지만 과거 시점 정보는 확인되지 않습니다. (X — 위 추이 표가 있음)

질문: "TB24-5JN042 추세는?"
좋은 답변:
> **TB24-5JN042** 의 MSE 는 12시간 전 0.41 → 현재 0.84 로 지속 상승 중입니다. 임계 0.85 직전이라 즉각 점검을 권장합니다.

질문: "방식전위"
좋은 답변:
> 방식전위는 매설배관 부식 보호 지표로 -850 mV 이하가 양호 기준입니다. 초과 시 부식 진행 가능성이 있어 정류기 출력 조정이 필요합니다.
`;
}
