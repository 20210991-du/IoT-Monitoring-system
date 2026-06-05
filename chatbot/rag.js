// chatbot/rag.js — 로컬 RAG (project-knowledge·kb·persona-knowledge 청킹·임베딩 → kb_chunks, 코사인 top-k 검색).
// server.js 에서 createRag(deps) 로 생성. deps 로 외부 의존성(pool·Ollama·경로) 주입 — server.js 내부에 결합 안 됨.
import { readFileSync, readdirSync, existsSync } from "fs";
import path from "path";
import { createHash } from "crypto";

export function createRag({ pool, ollamaUrl, keepAlive, chatbotDir, embedModel = process.env.EMBED_MODEL || "nomic-embed-text" }) {
  // project-knowledge.md (공통 지식) 로드
  let PROJECT_KNOWLEDGE = "";
  try { PROJECT_KNOWLEDGE = readFileSync(path.join(chatbotDir, "project-knowledge.md"), "utf8"); }
  catch (e) { console.warn("[project-knowledge] 로드 실패:", e.message); }

  let KB_CHUNKS = [];   // [{ id, domain, section, text, vec:[...] }]

  async function embedText(text, kind = "document") {
    // nomic-embed-text는 task prefix 필요(검색 품질 핵심): 문서=search_document, 질문=search_query
    const prefixed = (kind === "query" ? "search_query: " : "search_document: ") + String(text || "").slice(0, 6000);
    try {
      const res = await fetch(`${ollamaUrl}/api/embed`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: embedModel, input: prefixed, keep_alive: keepAlive }),
        signal: AbortSignal.timeout(20_000),
      });
      if (!res.ok) return null;
      const j = await res.json();
      const v = (j.embeddings && j.embeddings[0]) || j.embedding || null;
      return Array.isArray(v) ? v : null;
    } catch { return null; }
  }

  // 섹션(## ) 단위 청킹 + 도메인 태그(헤딩 키워드)
  function chunkKnowledge(md) {
    const RULES = [
      [/이상탐지|LSTM|모델|threshold|MSE|학습|피처|AI 이상/i, "ai"],
      [/\bDB\b|데이터베이스|테이블|스키마|동기화|MySQL|미러/i, "db"],
      [/챗봇|대시보드|프론트|UI|화면|지도|시각화|기능/i, "dashboard"],
    ];
    return md.split(/\n(?=## )/).map((p) => p.trim()).filter((t) => t.length >= 20 && !/응대\s*지침/.test(t.slice(0, 40))).map((text) => {
      const heading = (text.match(/^#{1,3}\s+(.+)/) || [, "intro"])[1].trim();
      let domain = "general";
      for (const [re, d] of RULES) if (re.test(heading)) { domain = d; break; }
      return { domain, section: heading, text };
    });
  }

  // project-knowledge.md(공통, 헤딩 키워드로 도메인) + kb/*.md + persona-knowledge/*.md(멤버 자필, 도메인 고정) → 청크 소스
  function gatherKbSources() {
    const out = chunkKnowledge(PROJECT_KNOWLEDGE || "");
    // 추가 지식: chatbot/kb/*.md (옵시디언에서 안전 큐레이션·스크럽한 ADR·자문·요구사항) — 공통(헤딩 키워드로 도메인 분류)
    try {
      const kbDir = path.join(chatbotDir, "kb");
      for (const fn of readdirSync(kbDir)) {
        if (!fn.endsWith(".md")) continue;
        let md = ""; try { md = readFileSync(path.join(kbDir, fn), "utf8"); } catch { continue; }
        md = md.replace(/<!--[\s\S]*?-->/g, "").replace(/^---[\s\S]*?---\s*/, "");   // 주석·frontmatter 제거
        for (const c of chunkKnowledge(md)) out.push({ domain: c.domain, section: `kb · ${c.section}`, text: c.text });
      }
    } catch { /* kb 폴더 없음 — 무시 */ }
    const PF = [["lee_duhyeon", "ai"], ["lee_jaeheon", "db"], ["park", "dashboard"]];
    const dir = path.join(chatbotDir, "persona-knowledge");
    for (const [key, domain] of PF) {
      const fp = path.join(dir, `${key}.md`);
      if (!existsSync(fp)) continue;
      let md = ""; try { md = readFileSync(fp, "utf8"); } catch { continue; }
      md = md.replace(/<!--[\s\S]*?-->/g, "");   // 주석 제거
      for (const seg of md.split(/\n(?=## )/).map((s) => s.trim()).filter((s) => s.length >= 30)) {
        const heading = (seg.match(/^#{1,3}\s+(.+)/) || [, "intro"])[1].trim().slice(0, 80);
        out.push({ domain, section: `${key} · ${heading}`, text: seg });
      }
    }
    return out;
  }

  async function loadKbChunks() {
    if (!pool) return;
    try {
      const [rows] = await pool.query("SELECT id, domain, section, text, embedding FROM kb_chunks");
      KB_CHUNKS = rows.map((r) => ({ id: r.id, domain: r.domain, section: r.section, text: r.text, vec: JSON.parse(r.embedding) }));
      console.log(`▶ RAG  kb_chunks ${KB_CHUNKS.length}개 메모리 로드`);
    } catch (e) { console.error("✗ kb_chunks 로드 실패:", e.message); }
  }

  async function ensureKbChunks() {
    if (!pool) return;
    try {
      await pool.query(`CREATE TABLE IF NOT EXISTS kb_chunks (
        id INT AUTO_INCREMENT PRIMARY KEY,
        domain VARCHAR(20) NOT NULL,
        section VARCHAR(200) NULL,
        text MEDIUMTEXT NOT NULL,
        embedding MEDIUMTEXT NOT NULL,
        doc_hash VARCHAR(40) NOT NULL,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
      ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci`);
      const sources = gatherKbSources();
      const hash = createHash("sha1").update("v4-persona:" + sources.map((s) => s.domain + "|" + s.text).join("")).digest("hex");
      const [[{ n }]] = await pool.query("SELECT COUNT(*) AS n FROM kb_chunks WHERE doc_hash = ?", [hash]);
      if (n === 0 && sources.length) {
        const rows = [];
        for (const c of sources) {
          const vec = await embedText(c.text, "document");
          if (vec) rows.push([c.domain, c.section, c.text, JSON.stringify(vec), hash]);
          else console.warn("[kb] 임베딩 실패:", c.section);
        }
        if (rows.length) {
          await pool.query("DELETE FROM kb_chunks");
          await pool.query("INSERT INTO kb_chunks (domain, section, text, embedding, doc_hash) VALUES ?", [rows]);
          console.log(`▶ RAG  kb_chunks ${rows.length}개 인제스트 (hash ${hash.slice(0, 8)})`);
        }
      }
      await loadKbChunks();
    } catch (e) { console.error("✗ kb_chunks 인제스트 실패:", e.message); }
  }

  function cosineSim(a, b) {
    let dot = 0, na = 0, nb = 0;
    const len = Math.min(a.length, b.length);
    for (let i = 0; i < len; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
  }

  // 페르소나 → 허용 청크 도메인. park(fallback/메인)=전체, 나머지=자기+general
  const PERSONA_RAG_DOMAIN = { lee_duhyeon: "ai", lee_jaeheon: "db", control_assistant: "dashboard", agent: "general", park: "dashboard" };

  async function retrieveChunks(query, personaKey, k = 6) {
    if (!KB_CHUNKS.length) return [];
    const qv = await embedText(query, "query");
    if (!qv) return [];
    const dom = PERSONA_RAG_DOMAIN[personaKey] || "*";
    const qWords = String(query).toLowerCase().match(/[가-힣a-z0-9]{2,}/g) || [];
    // 하이브리드: 코사인 + 도메인 부스트 + 키워드(헤딩>본문 가중) + 타 페르소나 자필 누출 패널티
    const PFX = ["park · ", "lee_jaeheon · ", "lee_duhyeon · "];
    return KB_CHUNKS
      .map((c) => {
        let score = cosineSim(qv, c.vec);
        if (dom !== "*" && c.domain === dom) score += 0.08;
        const sec = c.section || "", secLo = sec.toLowerCase(), txtLo = String(c.text).toLowerCase();
        let secKw = 0, txtKw = 0;
        for (const w of qWords) { if (secLo.includes(w)) secKw++; else if (txtLo.includes(w)) txtKw++; }
        score += Math.min(secKw, 4) * 0.05 + Math.min(txtKw, 6) * 0.02;
        if (PFX.some((p) => sec.startsWith(p)) && !sec.startsWith(personaKey + " · ")) score -= 0.15;
        return { c, score };
      })
      .sort((a, b) => b.score - a.score)
      .slice(0, k)
      .map((x) => ({ section: x.c.section, text: x.c.text, score: +x.score.toFixed(3) }));
  }

  return { embedText, retrieveChunks, ensureKbChunks, kbCount: () => KB_CHUNKS.length, PERSONA_RAG_DOMAIN };
}
