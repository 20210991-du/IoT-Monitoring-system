---
tags: [ADR, 기술, AI, LLM, function-calling, tools, ollama, SSE, streaming, MySQL]
date: 2026-05-18
status: 완료 (Mac Studio 가동 중, 스트리밍 + tools 검증 완료)
---

# ADR-012 · LLM Function Calling 도입 — 챗봇이 필요 시 MySQL 직접 조회

## 맥락

[[ADR-010-AI채팅-어시스턴트]] 의 4차 폴리시(스트리밍 SSE + 마크다운 + 히스토리)까지 완료된 상태였음. 하지만 구조적 한계 존재:

- 시스템 프롬프트에 **현재 화면 컨텍스트만** 미리 끼워넣음 (counts / criticalNodes / 12h MSE 추이 / 통신두절 상세)
- 화면에 없는 정보는 답변 불가:
  - "TB24-250455 의 시리얼번호?"
  - "전체 단말 평균 RSSI?"
  - "최근 7일 위험 등급 알람?"
- 모든 정보를 미리 컨텍스트에 끼워넣으면 토큰 폭증 + 매번 풀 페이로드 전송

**5/17 사용자 질문:** *"이런 비슷한 질문을 던졌을때 풀리게 하는법 ... LLM 이 필요할 때 직접 DB 쿼리 호출 하는 패턴 이걸로 가야지"*

## 결정

**Ollama Function Calling (Tools) 도입.** LLM 에게 도구 목록을 알리고, 필요시 LLM 이 직접 도구를 호출 → 서버가 MySQL 조회 → 결과를 LLM 에 반환 → LLM 이 최종 답변 생성.

### 6개 도구 정의

| 도구 | 용도 | 인자 |
|---|---|---|
| `list_devices` | 단말 목록 (필터) | status / zone / limit |
| `get_device_detail` | 단말 메타 + 8 센서 최신값 | deviceId |
| `get_device_history` | 단말 시계열 (1h/24h/7d/30d) | deviceId / kind / range |
| `get_alarms` | 최근 알람 (등급 필터) | days / gradeId / limit |
| `get_summary` | 전체 KPI 카운트 | (없음) |
| `get_aggregate` | 전체 평균/최대/최소 | metric / op |

### 라운드 루프 (최대 5회)

```
1) Ollama 호출 (tools 동봉)
   ↓
2) LLM 응답에 tool_calls 있나?
     - 없음 → 최종 답변, 종료
     - 있음 → execTool() → 결과 messages 에 append → 1) 으로
3) MAX_ROUNDS=5 초과 시 안전 종료
```

### 두 엔드포인트 모두 지원

- `POST /api/chat` — 비스트리밍 (단순, 디버깅/Fallback 용)
- `POST /api/chat/stream` — SSE 스트리밍 + tools (사이트 기본)

**SSE 이벤트:**
- `delta` { text } — 토큰 단위 글자 흐름
- `tool` { round, name, args } — 도구 호출 발생 알림 (UI 가 "🔧 조회 중" 표시 가능)
- `done` { reply, model, rounds, toolCalls, tokens } — 최종 완성 답변
- `error` { message }

### 시스템 프롬프트 가이드 섹션 추가

```
# 도구(Tools) 사용 가이드
위 "현재 시스템 상태" 와 "12h MSE 추이" 에 있는 정보면 그대로 답변.
없는 정보(특정 단말 상세, 시리얼/설치일/위경도, 시계열 추이, 알람 이력 등)는 도구를 직접 호출.
도구 결과를 근거로 답하고, 추측 금지. error 면 "데이터 없음" 으로 정직하게 답변.
```

## 구현 위치

`front/server.js` 한 파일에 다 들어감:
- `TOOLS[]` — 도구 6개 스키마
- `execTool(name, args)` — 6 case dispatcher (MySQL 쿼리)
- `runChatWithTools(messages, signal)` — 비스트리밍 라운드 루프
- `/api/chat/stream` 의 인라인 라운드 루프 — 스트리밍 라운드 루프
- `findSensorId` / `getTransmitterIdByName` — NAME ↔ ID 매핑 헬퍼

## 검증 (5/18)

샘플 질문 5건 → 모두 적절한 도구 자동 선택 + 1라운드 호출 → 2라운드에 최종 답변:

| 질문 | 호출된 tool | 응답 |
|---|---|---|
| TB24-250401 의 방식전위/배터리/마지막측정 | `get_device_detail` | -2071 mV / 3665 mV / 2026-05-17 06:00 |
| 전체 평균 RSSI | `get_aggregate(commDbm, avg)` | -94.96 dBm, "통신 두절 임박" |
| 최근 7일 위험 등급 알람 | `get_alarms(days:7, gradeId:1)` | 0건 |
| 통신 두절 단말 5개 | `list_devices(status:offline, limit:5)` | TB24-250429 1대 (총 55대 중) |
| TB24-250402 24h 방식전위 추이 | `get_device_history` | -1607~-1628, 양호 |

스트리밍 + tools 동시 작동 확인. event:tool 먼저 emit → event:delta 토큰 단위 흘림.

## 발견된 버그 (모두 수정)

| 버그 | 증상 | 원인 | 수정 |
|---|---|---|---|
| `list_devices` LIMIT | offline 단말 0개로 응답 | SQL LIMIT 을 status 필터 *전에* 적용 | LIMIT 제거 → JS filter → slice |
| `/api/chat/stream` 즉시 abort | "This operation was aborted" 에러 | `req.on('close')` 가 body parser 후 spurious emit | `res.on('close')` + writableFinished 가드 로 교체 |
| tool_calls 미감지 | content 비고 toolCalls 빈 배열 | Ollama 가 tool_calls 를 done:false chunk 에 넣음 | 모든 chunk 에서 tool_calls 찾도록 변경 |

## 성능 (5/18 측정)

- prompt 토큰: 2400~3200 (시스템 프롬프트 + 도구 정의 + 컨텍스트)
- completion 토큰: 25~120 (간결 답변)
- 라운드: 보통 2회 (1회 tool 호출 + 1회 최종 응답)
- 응답 시간: 1~3초 (도구 호출 1회 기준)
- timeout: 비스트리밍 120s, 스트리밍 180s

## 트레이드오프

**장점:**
- 화면 밖 임의 정보 답변 가능
- 시스템 프롬프트 폭증 회피 (도구 정의는 한 번)
- LLM 이 스스로 어떤 도구가 필요한지 판단

**단점:**
- 라운드 늘면 응답 지연 (최대 5라운드)
- 도구 정의 잘못되면 LLM 이 헷갈림 (description / enum 신중)
- tool_call schema 가 Ollama API 의존 (OpenAI 호환이지만)

## 후속

- [ ] ChatPanel 에 `event:tool` listener 추가 → "🔧 list_devices 조회 중..." UX
- [ ] 도구 호출 audit log 저장 (사용 빈도 측정)
- [ ] LLM 학습 예측 (ai_predictions 테이블) 연동 시 `get_predictions` 도구 추가
- [ ] 자문 Q5 답변 반영: 답변에 도구 호출 흔적 명시 ("DB 조회 결과")

## 출처

- [[ADR-010-AI채팅-어시스턴트]] — 챗봇 1~4차 변천
- [[2026-05-13-1차자문-현대모비스-장승철]] — Q5 챗봇 강화 멘토 답변
- 박지훈 5/18 작업
- Ollama Tools API: https://ollama.com/blog/tool-support
