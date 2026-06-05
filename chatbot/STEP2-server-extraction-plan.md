# Step 2 — 챗봇 서버 로직 추출 계획 (server.js → chatbot/ 모듈)

> **상태:** 예약됨. **server.js 동시편집이 멈춘 깨끗한 창**에서 집중 작업으로 실행할 것.
> Step 1(데이터 → `chatbot/kb`·`persona-knowledge`·`project-knowledge.md`)은 완료(커밋 `f5a2fde`).
> ⚠️ server.js 는 ~4,950줄에 auth·dashboard·guestbook·sync 등과 공유됨. 챗봇 코드만 ~1,500~2,500줄 흩어져 있음.

## 1. server.js 내 챗봇 코드 블록 (추출 대상 — 라인은 작업 시점에 재확인)
| 블록 | 대략 위치 | 비고 |
|---|---|---|
| 페르소나 시드/로드 | `BOT_PERSONAS`·`BOT_PERSONA_SEED` (~269–322) | DB `bot_personas` 테이블 |
| function-calling 툴 정의 | `TOOLS` 배열 (~1085–1389) + `TOOLS_BASE`/`toolsFor()` | 15개 (get_device_detail·get_predictions·find_devices_near·get_ai_model_info·get_weather_* 등) |
| 툴 실행기(핸들러) | chat 루프 내부 (~2300–2640) | `pool` SQL 조회 + 모델 데이터 함수 호출 |
| 채팅 엔드포인트 | `/api/chat/models`(931)·`/api/chat`(2445)·`/api/chat/stream`(2521) | LLM 라우팅 포함 |
| 세션 엔드포인트 | `/api/chat/sessions`·`/search`·`/current`·`/:id`·POST (3827–3933) | `chatOwner` 스코프(계정별/공유) |
| 관리 통계 | `/api/admin/tool-stats`(3644)·`/api/admin/token-usage`(3699) | |
| RAG | `embedText`(4013)·`chunkKnowledge`(4029)·`gatherKbSources`(4043)·검색·`KB_CHUNKS`·`EMBED_MODEL` | Ollama 임베딩 + `kb_chunks` 테이블 |
| 페르소나 프롬프트 | `PERSONA_BASE`(4142)·`PERSONA_RAG_DOMAIN`·빌더 | |
| WebSocket 채팅 | `WebSocketServer`(26)·`wss`·`/ws/chat`·rooms·`broadcastToOwner` | 계정별 실시간(최근 추가) |

## 2. 의존성 (모듈이 받아야 할 것 — `registerChatbot(app, server, deps)`)
- `pool` (MySQL), `authClaims`/`chatOwner`/`requireAdminView` (인증)
- 모델 데이터 함수: `classifyMse`·`DEVICE_THRESHOLDS`·`MODEL_CONFIG`·`aiRatioOf`·`loadLatestAi`·단말/이력/알람 조회 헬퍼
- LLM 설정: `OLLAMA_URL`·`KEEP_ALIVE`·OpenAI 키/클라이언트·`EMBED_MODEL`
- 세션: `ensureChatSession`·`persistMessage`
- 경로: `chatbot/kb`·`chatbot/persona-knowledge`·`chatbot/project-knowledge.md` (Step 1에서 이미 분리됨)

## 3. 목표 구조
```
chatbot/
├── kb/ · persona-knowledge/ · project-knowledge.md   (Step 1 완료)
├── rag.js        embedText·chunkKnowledge·gatherKbSources·ingest·retrieve
├── personas.js   BOT_PERSONA_SEED·PERSONA_BASE·프롬프트 빌더
├── tools.js      TOOLS 정의 + 핸들러(executeTool)
├── routes.js     /api/chat*·/sessions*·LLM 루프
├── ws.js         /ws/chat 서버
└── index.js      registerChatbot(app, server, deps) — 위 전부 mount
```
`front/server.js`:
```js
import { registerChatbot } from "../chatbot/index.js";
registerChatbot(app, server, { pool, authClaims, chatOwner, classifyMse, getThresholds: () => DEVICE_THRESHOLDS, ... });
```

## 4. 추출 순서 (위험 낮은 것부터 · 각 단계 후 검증)
1. **rag.js** — 가장 자기완결적. (검증: 질문 임베딩 검색 동작, kb_chunks 재인제스트)
2. **personas.js** — 프롬프트/시드. (검증: 페르소나별 응답 톤)
3. **tools.js** — 툴 정의+핸들러. (검증: "TB24-XXX 상태" → get_device_detail 호출)
4. **routes.js** — 엔드포인트+LLM 루프. (검증: /api/chat 왕복, 스트리밍)
5. **ws.js** — 실시간. (검증: 창 2개 동기화)
6. server.js 에서 옮긴 코드 제거 + import.

## 5. 검증 체크리스트 (각 단계)
- [ ] `node --check server.js` + 부팅 OK + `▶ RAG kb_chunks N개`
- [ ] 챗봇 일반 질문 1왕복 (관제 도우미)
- [ ] 툴 질문 1건 (`get_predictions`/`get_device_detail`)
- [ ] 페르소나(park/lee_*) 응답
- [ ] WS 실시간(로그인 2창)
- [ ] 라이브 사이트 무중단 (`curl 127.0.0.1:5050` 200)

## 6. 주의
- **동시편집**: 시작 전 server.js 를 만지는 다른 세션이 없는지 확인(파일 mtime). 작업 중엔 단독 점유.
- **라이브 챗봇**: siwon.msbuger.com 운영 중 → 단계마다 빌드·재시작·검증, 깨지면 즉시 직전 커밋 복구.
- 큰 블록 이동은 한 번에 한 모듈씩 + 커밋 체크포인트.
