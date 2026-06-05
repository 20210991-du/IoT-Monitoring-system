---
tags: [기술, 백엔드, 로직, API, 인증]
date: 2026-06-01
---
# 백엔드 로직 — API · 인증 · 챗봇

> Node/Express `front/server.js`(~3500줄). 전반 [[백엔드]], DB [[데이터베이스]], 보안 [[보안]].

## 인증 / 권한
- JWT(httpOnly 쿠키 `siwon_auth`) + bcrypt. `authClaims`/`requireAuth`/`requireAdmin`/`requireSuperAdmin`.
- RBAC: superadmin > admin > operator > guest. 관리자급 생성·삭제·역할변경·비번재설정 = **총관리자만**. last-admin/superadmin 삭제 차단. `audit_log` 기록.

## 주요 API
- `/api/auth/login({id,pw})|me|logout|users|signup|exists`
- `/api/devices`, `/api/chat(/stream)`(LLM + Function Calling), `/api/chat/sessions...`
- `/api/admin/login-log`
- `/api/guestbook` GET/POST/DELETE — **단톡방**(공개 쓰기, 레이트리밋·모더레이션). → [[데이터베이스]] · [[보안]]
- **WebSocket**: `/ws/chat`(로그인 계정 세션 동기화) · `/ws/guestbook`(공개 단톡방 broadcast).

## 챗봇 도구 16
기본6(list_devices·get_device_detail·history·alarms·summary·aggregate) + 고급6 + 위치3(geocode·find_near) + AI1(get_ai_model_info) + (자가확장 자유SQL 도구는 운영 환경 기본 비활성). → [[ADR-012-Function-Calling-LLM이-DB-직접조회]] · [[ADR-013-챗봇-운영강화-도구확장-영구저장-캐시-audit]] · [[ADR-014-위치-지도-기반-단말검색]] · [[ADR-015-AI파트-연결-챗봇과-LSTM-Threshold]]

## 동기화 / AI 연결
- launchd 3 job(alarm 1h/sensor 2h/meta 6h) KSCG→MySQL 미러.
- `classifyMse(deviceId, mse)` → 정상/관찰/이상 (단말별 threshold). config 3 json 부팅 로드.

## 관련
[[백엔드]] · [[데이터베이스]] · [[보안]] · [[AI모델]] · [[기술개요]]
