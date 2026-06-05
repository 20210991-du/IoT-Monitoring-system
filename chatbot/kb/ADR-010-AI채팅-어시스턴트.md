---
tags: [ADR, 기술, AI, LLM, UI, 챗봇, ollama, SSE, streaming]
date: 2026-05-04
status: 4차 폴리시 완료 (스트리밍 SSE + 마크다운 + 히스토리 보존)
---

# ADR-010 · AI 분석 어시스턴트 (AI 조치 권고 → 채팅 UI)

## 맥락

기존 대시보드 우하단 영역은 **AI 조치 권고** 패널 — 정적 인사이트 카드 (지금 조치 / 48시간 내 / 장기 개선) 3건을 고정 표시. 7차 회의(2026-05-04)에서:
- 정적 인사이트 카드는 시연 임팩트 약함 (3개로 고정, 상호작용 X)
- 운영자가 직접 질문할 수 있는 **AI 어시스턴트** 가 더 의미 있음
- 도메인 지식(방식전위·희생전류·AC유입 등)을 프롬프트에 사전 주입 → 채팅으로 답변
- **시스템 로그(좌하단)는 그대로 유지** — 운영 가시성 + 발표 시연 효과

> **이재헌:** "프롬포트 만들어 가지고 채팅할 때마다 프롬프트를 읽게 해서 답변하도록 만들면 좋겠다."
> **이재헌:** "어차피 시스템 로그 같은 거 보면은 그치 그래도 보일 텐데."

## 결정

**AI 조치 권고 영역을 AI 채팅 패널로 교체**. 시스템 로그(좌하단)는 그대로 유지. 1차 구현은 **mock 응답 (키워드 매칭)** 으로 완성, 실 LLM 연동은 자문 회신/모델 정합 후 추후 작업.

### 1차 (이번 주, 박지훈) — 완료

`front/src/pages/Dashboard.jsx` 의 `AIAdvicePanel` (우하단) → `ChatPanel` 교체.
좌하단 `LogPanel` 은 유지.

- 메시지 말풍선 UI (AI 좌측 / 사용자 우측 그라디언트)
- 입력창 + 전송 버튼 (입력 빈/전송 중 disable)
- typing indicator (3 dots 펄스)
- 자동 스크롤 (최신 메시지로)
- 헤더 우측 "mock 모드" 배지

**`mockAIResponse(input, ctx)` — 키워드/패턴 매칭 분기:**

| 패턴 | 응답 |
|---|---|
| `TB24-5JN\d+` | equipment 에서 조회 → 상태/MSE/구역/라벨/기여도/갱신시각 |
| `위험` / `critical` | critical 노드 목록 |
| `이상` / `의심` / `anomaly` | anomaly 노드 목록 |
| `관찰` / `watch` / `warn` | warn 노드 목록 |
| `장애` / `offline` / `통신` | offline 노드 목록 |
| `요약` / `현황` / `상태` | 전체 카운트 통계 |
| `방식전위` | 도메인 설명 (P/S Potential, -850mV 기준) |
| `희생전류` / `희생양극` | 도메인 설명 (양극 소모, 1mA 임계) |
| `AC유입` / `ac` | 도메인 설명 (200/500mV 단계) |
| `통신품질` / `dBm` / `RSSI` | 도메인 설명 (신호 세기 단계) |
| `임계` / `threshold` / `MSE` | 임계값 설명 |
| `도움` / `help` / `?` | 사용 예시 |
| (그 외) | "노드 ID 또는 도메인 키워드로 질문해 주세요" |

### 2차 (실 LLM 연동) — 2026-05-06 완료

**아키텍처:** Mac Studio Express 통합 서버 (옵션 A 선택)

```
Browser  ──cloudflared 터널──>  Mac Studio (PORT 5050)
                                  │
                                  ├── express.static(dist/)   ← 정적 SPA
                                  └── POST /api/chat
                                        │
                                        ▼ HTTP
                                   Ollama (localhost:[비공개포트])
                                        │
                                        ▼
                                   gemma4:e4b (9.6GB)
```

**구현 파일:**
- `front/server.js` — Express 서버 (정적 + /api/chat)
- `front/package.json` — express 의존성, `npm start` 스크립트
- `front/src/pages/Dashboard.jsx` `ChatPanel` — fetch /api/chat + mock fallback
- `front/src/lib/weather.js` — Open-Meteo 훅 (서울 실시간 기상)
- `front/src/data/mockData.js` — 각 device 에 `mseHistory[12]` 신설

**핵심 동작:**
1. ChatPanel 이 fetch `/api/chat` 으로 메시지+컨텍스트 전송
2. server.js 가 시스템 프롬프트(도메인 + 시스템 상태 + 추이 표 + 날씨) 추가
3. Ollama qwen3.5:9b 호출 (60초 타임아웃, `think:false` 로 reasoning 토큰 차단)
4. 응답 텍스트 분석:
   - 노드 ID 추출 → 지도 flyTo / fitBounds 자동 트리거
   - 단일 status 키워드 감지 → KPI 자동 필터 활성 (30초 후 자동 복귀)
5. LLM 실패 시 자동 mock fallback (헤더 배지: 대기/LLM 연결됨/mock fallback)

**Ollama 호출 옵션 (3차):**
- model: `qwen3.5:9b` (gemma4:e4b → 변경, 한국어 ↑, instruction following ↑)
- temperature: 0.3 (일관된 답변)
- num_predict: 700 (응답 잘림 방지)
- think: false (qwen3 thinking 모드 차단 — content 비는 이슈 회피)
- 히스토리: 최근 6턴까지 (토큰 절약)

**시스템 프롬프트 구성 (3차 — 풀 컨텍스트):**
- 도메인 지식 (방식전위/희생전류/AC유입/통신품질/MSE 임계)
- 위험 단계 5단계 정의
- **현재 시각 + 현재 시스템 상태** (counts + 위험·이상 의심·통신장애 노드 ID)
- **🆕 최근 12시간 MSE 추이** — 위험·이상 의심 노드의 1h 간격 시계열 (시작/피크/현재/방향)
- **🆕 현재 날씨 (서울)** — Open-Meteo 데이터 + 배관 영향 가이드
- 응답 규칙 + few-shot 예시 (좋은/나쁜)

**🆕 KPI 자동 필터 (Aurora UX)**

응답 텍스트 분석으로 운영자 시선을 자동으로 필터:
- 단일 status (위험/이상의심/통신장애/정상) 명확 → 그 KPI 자동 활성
- 여러 status 동시 / 키워드 없음 → 총 장비(전체) 유지
- **30초 카운트다운 칩** 표시: "✦ AI 자동 보기 [25s] · 해제"
- 사용자가 직접 KPI 클릭 시 즉시 타이머 취소 (사용자 의도 우선)
- 30초 경과 시 자동으로 전체 보기 복귀 (운영자 평상시 관제 모드)

**🆕 12시간 MSE 추이 컨텍스트**
- 각 device 에 `mseHistory[12]` (1h 간격, 가장 오래된→현재)
- status 별 추세 패턴 (critical=급상승, warn=완만, normal=평탄)
- LLM 이 "약 N시간 전부터 임계 초과" 같은 시간 기반 답변 가능
- 마지막 값은 현재 mse 와 정확히 매칭

**🆕 날씨 컨텍스트**
- Open-Meteo (무료, 키 X, CORS OK) 서울 데이터
- WMO weather code → 한국어/이모지 매핑 24종
- 1h TTL localStorage 캐시
- LLM 이 "강우 시 침수·통신 두절 위험" 같은 환경적 추론 가능

**환경변수:**
- `OLLAMA_URL` (default `http://localhost:[비공개포트]`)
- `OLLAMA_MODEL` (default `qwen3.5:9b`)
- `PORT` (default `5050`)

### 4차 (UX 폴리시) — 2026-05-11 완료

**🆕 응답 스트리밍 (Server-Sent Events)**
- `server.js` 에 `POST /api/chat/stream` 신규
- Ollama `stream:true` 의 ndjson 응답을 SSE event 로 변환 (`event:delta`, `event:done`, `event:error`)
- `ChatPanel.callLLMStream()` — fetch + ReadableStream 으로 SSE 파싱
- 스트리밍 중 메시지에 깜빡이는 커서 (`@keyframes blink`)
- 클라이언트 disconnect 시 AbortController 로 Ollama 호출 취소
- 비스트리밍 `/api/chat` 도 fallback 유지 (구버전·디버깅용)

**🆕 마크다운 inline 렌더**
- 간단 파서 `renderInlineMD()` — `**굵게**`, `` `코드` `` 처리
- 외부 의존성 X (react-markdown 안 씀)
- LLM 응답의 `**TB24-5JN011**` 노드 ID 강조가 실제로 굵게 보임

**🆕 채팅 히스토리 localStorage 보존**
- 키 `siwon.chat.history`, 최근 60개 메시지 보관
- 새로고침해도 대화 유지
- 헤더에 ↻ 초기화 버튼 추가

**제약 / 한계 (4차 기준):**
- 토큰 한도 800 — 매우 긴 답변 잘릴 수 있음 (`num_predict` 조정 가능)
- 영구 채팅 이력은 브라우저 localStorage 만 (서버 미저장)
- 첫 호출 5~10초 (모델 로드), 이후 1~3초
- Mac Studio 가동 시에만 작동, 자동 시작 X (launchd 미설정)

## 백엔드 인터페이스 명세

### `POST /api/chat`

**Request:**
```json
{
  "message": "TB24-5JN042 왜 이상이야?",
  "context": {
    "equipment_summary": { "critical": 2, "anomaly": 4, "warn": 4 },
    "current_user": "operator.kim"
  },
  "history": [
    { "role": "user", "text": "..." },
    { "role": "ai",   "text": "..." }
  ]
}
```

**Response 200:**
```json
{
  "reply": "TB24-5JN042 의 MSE 가 0.842 로 임계값 0.409 를 2배 초과했습니다. ...",
  "tokens_used": 142,
  "model": "claude-sonnet-4-5"
}
```

### 시스템 프롬프트 (사전 주입)

```
당신은 매설배관 IoT 통합 관제 시스템의 AI 분석 어시스턴트입니다.
운영자가 노드 ID, 위험 단계, 도메인 용어를 묻습니다. 다음 규칙을 따르세요:

1. 노드 ID(TB24-5JN###) 가 있으면 해당 장비의 최신 상태를 조회해 답변
2. 도메인 용어 (방식전위/희생전류/AC유입/통신품질) 는 산업 표준에 맞게 설명
3. 운영자가 액션을 결정할 수 있도록 구체적 임계값·기준 포함
4. 한국어, 존댓말, 간결 (5문장 이내)
5. 불확실한 사실은 "추정" 명시
6. 환각 금지 — 시스템에 없는 노드/데이터는 추측하지 않음
```

## LLM 선택 기준

- **gpt-4o-mini / claude-haiku-4** — 비용/속도 (실시간 채팅)
- **claude-sonnet-4-5** — 복잡한 도메인 추론 (특정 분석 요청)
- **로컬 LLM (ollama)** — 오프라인/사설망 (Mac Studio 에 ollama 가동 중)

→ 1차 도입은 `claude-haiku` 또는 `gpt-4o-mini` 권장 (빠르고 저렴).

## 보안·운영 고려

- API 키는 백엔드 `.env` 만 보관, 프론트 노출 X
- rate-limit (사용자당 분당 N 회) — 비용 폭주 방지
- 응답 캐싱 (자주 묻는 질문 redis/메모리) — 자문 Q1 (LLM 가이드라인) 답변 받은 뒤 정책 확정

## 후속

- [ ] 자문 회신 (Q1 LLM 요약 가이드라인) 반영
- [ ] 실 LLM 연동 (현 mock → fetch /api/chat)
- [ ] 시스템 프롬프트 정밀 튜닝
- [ ] 사용 통계·token 비용 모니터링
- [ ] 채팅 이력 저장·조회 (audit log 와 연동)

## 출처

- [[2026-05-04-7차회의]] — 결정 회의
- 박지훈 작업: `mockAIResponse` + `ChatPanel` 1차 구현
