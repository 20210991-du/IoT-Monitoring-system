---
tags: [ADR, 기술, AI, LLM, function-calling, tools, caching, audit, persistence]
date: 2026-05-18
status: 완료 (Mac Studio 가동 + 회귀 테스트 13/13 PASS)
---

# ADR-013 · 챗봇 운영 강화 — 도구 12 확장 + 영구 대화 + 캐시 + audit

## 맥락

[[ADR-012-Function-Calling-LLM이-DB-직접조회]] 에서 6개 도구로 Function Calling 도입 완료. 이번 자율 작업 라운드에서 운영 신뢰성과 분석 깊이를 한꺼번에 끌어올림:

- 도구 6개로는 "방식전위 -800 이상 단말", "제1구역 통계", "단말 비교" 같은 복합 질문 미흡
- 매 호출마다 같은 단순 KPI 를 풀스캔 — 캐시 없음
- 도구 호출 흔적이 휘발 — 어떤 도구가 얼마나 쓰이는지 측정 불가
- 챗봇 대화는 localStorage 만 — 다른 기기에서 못 봄, 백업 X
- /api/predict 가 client.js 에서 호출되지만 서버 미구현

## 결정

### 1) 도구 6 → 12 확장

신규 6 개:
- **find_devices_by_value** — 조건 만족 단말 검색 ("방식전위 -800 이상")
- **get_zone_summary** — 구역 통계 (단말수/평균값/상태분포)
- **compare_devices** — 다중 단말 비교 (2~5개 한꺼번에)
- **get_recent_changes** — 변화량 통계 (start/end/delta/min/max/mean/std)
- **get_maintenance_log** — 점검·정비 이력
- **get_predictions** — AI LSTM 예측 결과 (현재 stub, ai_predictions 비어있으면 message)

### 2) 메모리 캐시 (LRU + TTL)

```js
TOOL_CACHE = new Map();  // LRU, 200 개 한도
TTL = 10초
CACHEABLE_TOOLS = list_devices, get_device_detail, get_summary,
                  get_aggregate, get_zone_summary, compare_devices,
                  find_devices_by_value, get_predictions
```

- 시계열·점검이력은 캐시 X (매번 fresh)
- 응답에 `_cached: true` 동봉 (디버깅용)

### 3) audit_log INSERT (best-effort)

모든 도구 호출 → audit_log 자동 기록:
```sql
INSERT INTO audit_log (action, target_type, target_id, metadata_json)
VALUES ('tool_call', 'chat', <도구명>, {args, ok, durationMs, cached})
```

신규 endpoint: `GET /api/admin/tool-stats?days=7` — 도구별 호출수/성공률/평균응답시간 집계.

### 4) 챗봇 영구 대화 저장 (chat_sessions + chat_messages)

- 클라이언트가 sessionId 보내면 그것 사용. 없으면 백엔드가 새로 만들어 응답에 동봉.
- 세션 title = 첫 사용자 메시지 30 자
- 매 user 메시지 + ai 답변 INSERT
- /api/chat/stream 은 SSE `session` 이벤트로 sessionId 즉시 통보

신규 endpoints (4):
- GET    /api/chat/sessions          — 최근 30 세션
- GET    /api/chat/sessions/:id      — 세션 + 메시지 200
- POST   /api/chat/sessions          — 새 세션
- DELETE /api/chat/sessions/:id      — 세션 + 메시지 삭제

프론트 ChatPanel:
- localStorage `siwon.chat.session_id` 저장
- "대화 초기화" 누르면 sessionId 도 null → 새 세션 시작

### 5) 시스템 프롬프트 강화 (자문 Q5 반영)

응답 규칙 5 개:
1. 도구 결과 직접 인용 (수치 명시)
2. 도구 미확인 결론은 [추정] 라벨
3. 확인 불가는 정직히 "데이터 없음"
4. 숫자 + 단위 명시
5. 도구 적극 활용 (의심에 호출, 연쇄 호출 권장)

### 6) /api/predict/:id stub

ai_predictions 조회 → 없으면 `{ stub: true, message: "LSTM 백엔드 대기" }`.

### 7) 챗봇 UI 메타 표시

AI 메시지 하단에 dim 으로:
```
14:23  ·  2.3s · 2R · 55tok
```
elapsedMs + rounds + completion tokens (디버그 가시성).

### 8) 회귀 테스트 자동화

`front/test/test-chat-tools.sh` — bash 스크립트, 13 케이스 (12 도구 + 도메인 질문):
- 각 케이스: 자연어 질문 → LLM 이 적절한 도구 자동 선택했는지 + 응답 길이 검증
- ssh 로 Mac Studio 직접 실행 가능
- 5/18 검증: 13/13 PASS

## 발견된 버그 (모두 수정)

| 버그 | 증상 | 수정 |
|---|---|---|
| `\`[추정]\`` template literal 충돌 | 서버 crash, ReferenceError: 추정 | buildSystemPrompt 안 backtick 제거 |

## 검증 결과 (5/18)

회귀 테스트 13/13 PASS. 각 도구 정확한 인자로 자동 선택 + 응답 의미 정확:
- list_devices(offline) → TB24-250429 (1대, 482시간 두절)
- get_aggregate(commDbm, avg) → -94.96 dBm "통신 두절 임박"
- find_devices_by_value(volt, gte, -800) → 0대 (모두 양호)
- get_zone_summary(제1구역) → 8대, 모두 정상, avgVolt -1753.13 mV
- compare_devices([A, B]) → 8 센서 비교
- get_recent_changes(24h) → 시작/끝/델타/방향 모두 산출
- get_predictions → stub 메시지 "LSTM 백엔드 대기"

/api/admin/tool-stats 36 호출 모두 success, 평균 응답 2~10ms (캐시 hit 0 — 회귀 테스트가 다른 인자로 호출).

세션 영구화: 회귀 테스트 시 14 세션 자동 생성, 각 user+ai 2 메시지.

## 성능

- 도구 평균 응답: 2~10 ms (캐시 X, MySQL pool 매우 빠름)
- LLM 라운드: 1 (도구 X) 또는 2 (도구 1회)
- 종단 응답: 1~3초 (스트리밍 시작)
- 토큰: prompt 2400~3600, completion 25~120

## 트레이드오프

**장점:**
- 12 도구로 거의 모든 도메인 질문 답변 가능
- 캐시 hit 시 응답 빠름 (특히 list_devices/get_summary)
- audit_log 로 도구 사용 패턴 가시화 → 향후 개선 지표
- 영구 대화 → 다른 기기 / 운영자 공유 가능

**단점:**
- TOOL_CACHE 가 단일 서버 instance 메모리 — 다중 인스턴스 시 hit 율 떨어짐 (현재는 1대라 문제 X)
- audit_log 가 매 호출 1 INSERT — 대량 호출 시 부담 (현재 일 수십 회라 문제 X)
- chat_messages context_json 이 5KB 정도 — 100K 메시지면 500MB. 향후 archive 정책 필요

## 후속

- [ ] ai_predictions 가 채워지면 get_predictions 가 진짜 데이터 반환 → 챗봇이 위험도 자동 알림 가능
- [ ] 챗봇 헤더에 "세션 목록" 드롭다운 UI (현재는 API 만 있음)
- [ ] 사이트에 /admin/tool-stats 시각화 페이지 (현재는 raw JSON)
- [ ] chat_sessions 가 100 개 넘으면 archive 정책
- [ ] 회귀 테스트 GitHub Actions 자동화 (현재는 수동 ssh)

## 관련

- [[ADR-012-Function-Calling-LLM이-DB-직접조회]] — 도입 결정 (이 ADR 의 모체)
- [[ADR-010-AI채팅-어시스턴트]] — 챗봇 1~5차 변천 (이 작업은 6차)
- [[2026-05-18-챗봇-운영강화-자율작업]] — 이번 자율 작업 기록
- 커밋: `eab14cc` `2b60033` `04674ce` `3f9c3fb`
