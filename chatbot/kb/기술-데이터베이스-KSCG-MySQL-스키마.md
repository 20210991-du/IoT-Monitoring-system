---
tags: [기술, 백엔드, 데이터베이스, DB, KSCG, MySQL]
date: 2026-06-01
---
# 데이터베이스 — KSCG(원본) · MySQL(미러)

> 상세 스키마 탐색 [[2026-05-18-KSCG-스키마-탐색]]. 접속정보는 [접속정보 비공개]/[접속정보 비공개](크리덴셜은 vault 별도, 여기 미기재).

## 구조
- **원본**: 옴니솔루션 KSCG **MS SQL Server 2019**(read_only, KSCG DB만). 27테이블, oneM2M(KIoT) 표준.
- **미러**: 팀 **MySQL**(siwon, Mac Studio). ~210만 행, 약 11개월(2025-06~). 동기화 alarm 1h/sensor 2h/meta 6h(옴니가 1h 계측·12h burst라 분단위 무의미).

## 주요 테이블 (KSCG)
- `TB_SENSOR_DATA`(시계열 메인) · `TB_RECENT_DATA`(최신값 캐시) · `TB_TRANSMITTER_INFO`(단말, PERIOD_SEC=3600) · `TB_FACILITY_INFO`(시설번호 예 1-178) · `TB_SENSOR_INFO`/`TB_SENSOR_TYPE_INFO`(센서) · `TB_ALARM_LOG` · `TB_MAINTENANCE_LOG` · `TB_SITE_INFO`(SITE_ID=2 군산도시가스, 55대).
- 그룹핑 = **정류기 단위**(시설번호 앞자리 아님). → [[ADR-004-정류기단위-그룹핑]]

## 앱 테이블 (MySQL)
- `users`(RBAC) · `audit_log` · `chat_sessions`/`chat_messages` · `ai_predictions` · access 로그.
- **`guestbook_messages`**(단톡방, 6/1 신규): id·user_id·display_name·role·body(500)·ip·ua·created_at·deleted_at(소프트삭제).

## 센서 타입 10종
방식전위(mV)·방식전류(≈희생전류, mA)·가스누출(%LEL)·수위·AC유입(mV)·배터리·온도(℃)·습도(%)·충격·수신감도(≈통신품질). 단말당 평균 8종.

## 관련
[[2026-05-18-KSCG-스키마-탐색]] · [[백엔드로직]] · [[AI모델]] · [[하드웨어]] · [[기술개요]]
