# 📄 Changelog

## v2026.05.25

### Added
- 자문 내용 확인

### Changed
- 로그인 페이지 개선 (2차)
  - 다크/라이트 테마 자동 연동 (MutationObserver + CSS 변수 기반, 대시보드와 색감 일치)
  - 실시간 시스템 로그 더미 제거 → `/api/health` · `/api/devices` · `/api/anomalies` 실제 폴링으로 대체
  - 로그 패널 레이아웃 점프 수정 (고정 높이 컨테이너, 5줄 고정, opacity-only 전환)
  - 부팅 메시지 단순화: 누적 표시 → 한 줄씩 교체 방식
  - 다크 모드 배경 색감 조정: `#060c18` 심화 + 멀티 블롭 (인디고·퍼플·틸)

👉 **결과**
- 테마 전환 시 로그인 ↔ 대시보드 간 시각적 이질감 해소
- 실제 시스템 데이터 기반 로그 표시로 현실감 향상

## v2026.05.18

### Added
- 로그인 페이지 전면 리디자인 (1차)
  - Canvas 파티클 네트워크 (마우스 반응 반발력 + 스파클 트레일)
  - SVG 데이터 흐름선 애니메이션 (stroke-dashoffset)
  - 부팅 시퀀스 (터미널 스타일 초기화 메시지 + 프로그레스 바)
  - 로그인 카드 3D 틸트 효과 (마우스 위치 기반 원근감 변환)
  - 글리치 텍스트 효과 (타이틀 간헐적 사이버펑크 연출)
  - 타이핑 애니메이션 · easeOutExpo 카운팅 애니메이션 (감시 노드 55개, 센서 6종)
  - 입력 완료 시 체크 애니메이션 + 버튼 Ripple 효과
  - dot grid 배경 오버레이, scanline 애니메이션, 실시간 상태 뱃지

👉 **결과**
- 로그인 화면이 AI 관제 시스템 분위기에 맞는 인터랙티브 UI로 완성

## v2026.05.11

### Fixed
- 백엔드 503 오류 수정
  - 팀 AI 스크립트 파일명에 괄호 포함(`2026.05.03`) → Python import 불가 문제
  - `ai/scripts/gas_common_model_train.py` · `gas_common_model_predict.py` clean copy 추가
- `uvicorn` 미인식 오류 수정 (Windows PATH 문제) → `python -m uvicorn` 방식으로 변경

### Changed
- 팀원 변경사항 동기화
  - `front/src/api/client.js` BASE URL 환경변수화 (`VITE_API_BASE_URL`), 데모 모드 플래그 추가
  - `front/server.js` 추가 (Express + MySQL, 포트 5050) — 로컬 개발은 Python FastAPI(8001) 유지
  - 지도 좌표계 군산 지역 기준으로 갱신 (lat 35.80~36.00, lng 126.45~126.83)

👉 **결과**
- 백엔드 정상 기동 및 API 응답 복구
- 팀 통합 환경과 로컬 개발 환경 병행 운용 구조 정리

## v2026.05.04

### Added
- Frontend, Backend, AI 통합하여 웹 서비스 형태로 구현
- 전체 시스템 연동 기반 홈페이지 구축

### Changed
- 대시보드 UI 구조 개선 및 수정
- AI 모델 로직 일부 개선 (이상 탐지 처리 간소화 및 안정성 보완)

👉 **결과**
- 단일 기능 수준에서 통합 서비스 형태로 발전
- 실제 동작 가능한 시스템 구조 완성
  
## v2026.04.27

### Changed
- 중간고사 기간으로 프로젝트 개발 일시 중단

👉 **결과**
- 개발 일정 일시 조정

## v2026.04.20
### Added
- gas_common_model_predict.py 신규 분리
- run_batch_prediction(): 전체 장비 일괄 이상 탐지 및 결과 CSV 저장
- classify_risk_level(): 이상/관찰/정상 3단계 위험 등급 분류 함수 추가
- OBSERVATION_RATIO 파라미터로 관찰 구간 기준 조정 가능
- EarlyStopping 콜백 추가 (patience=5, restore_best_weights=True)
- 희생전류 분리 처리
- SACRIFICIAL_DEVICES, SACRIFICIAL_FEATURES 상수 추가
- get_sacrificial_device_data(): TB24-250406, 407 전용 데이터 분리 함수 추가
- 통신품질 룰 기반 필터 추가
- apply_comm_quality_filter(): -115 dBm 이하 단절 판정, 연속 3회 이상 고장 판정
- 통신단절_플래그 / 통신고장_플래그 컬럼 자동 생성

### Changed
- 단일 파일(gas_common_model_v3.py) → 학습(train) / 예측(predict) 두 파일로 분리
- 데이터 분할 방식 개선
- 기존: 전체 데이터로 scaler fit (데이터 leakage 발생) -> 변경: train 70% / val 15% / test 15% 시간 순 분리, train 구간으로만 scaler fit
- Threshold 계산 기준 변경
- 기존: 학습 데이터(prepared_df) 기반 -> 변경: test 구간(미관측 데이터) 기반 → 실제 운영 환경에 가까운 기준값 산출
- LSTM activation 수정: relu → 제거(기본값 tanh) — relu 사용 시 gradient 폭발/소실 위험
- 학습 validation 방식 변경: validation_split=0.1 → validation_data=(X_val, X_val) 직접 전달
- 이상 판정 단계 확장: is_anomaly (True/False 2단계) → risk_level (이상/관찰/정상 3단계)
- BASE_FEATURES에서 희생전류, 통신품질 제외 (각각 전용 처리로 분리)
- min_points_per_device 기본값 상향: 72 → 200, 부족 시 자동 상향 로직 추가
- 학습 epochs 상향: 1 → 50

👉 **결과**
- 모델 신뢰성 및 유지보수성 향상
- 이상 판정이 2단계 → 3단계로 세분화되어 운영 대응 기준 구체화

## v2026.04.13

### Added
- 대시보드 UI 전체 구현 완료
- 센서별 데이터 시각화 그래프 구현
  - 방식전위, 전류, 전압 등 주요 센서 데이터 시계열 그래프 구성
- 센서 상태 카드 UI 구현 (정상 / 경고 / 위험)

### Changed
- UI 구조 개선 및 컴포넌트 분리
- 데이터 표시 방식 최적화 (테이블 + 그래프 연동)

👉 **핵심 내용**
- 센서 데이터를 직관적으로 확인할 수 있는 통합 대시보드 구성
- 개별 센서 단위 분석이 가능한 UI 구조 구현

👉 **결과**
- 실제 서비스 수준의 대시보드 UI 완성
- AI 결과 연동이 가능한 기반 구축

## v2026.04.06

### Added
- 대시보드 UI 초기 설계 진행
- Figma 기반 UI 구조 기획

### Changed
- 시스템 전체 흐름 재정리 (DB → AI → Dashboard)

👉 **결과**
- UI 설계 방향 확정
- 이후 개발 단계(프론트 구현) 기반 마련

## v2026.03.30

### Added
- 프로젝트 일정 문서(SCHEDULE.md) 추가
- 주차별 개발 기록 정리
- README에 일정 문서 링크 연결

### Changed
- 프로젝트 문서 구조 개선 (README / SCHEDULE / CHANGELOG 분리)

👉 **결과**
- 프로젝트 진행 과정 체계화
- 개발 이력 및 일정 관리 구조 확립

## v2026.03.22

### Added
- 대시보드 UI 기본 구조 구현

### Changed
- README 구조 개선
  - 프로젝트 개요, 일정, 실행 방법 등을 체계적으로 정리

👉 **결과**
- 전체 시스템 UI 방향 확정

## v2026.03.20

### Added
- AI 모델 학습 코드 구현 (LSTM AutoEncoder)
  - 시계열 데이터 기반 이상 탐지 구조 설계
- 데이터 전처리 기능 구현
  - 결측치 보간, 정규화, 피처 엔지니어링 적용

👉 **핵심 내용**
- 변화량(diff), 이동평균(ma), 편차(dev) 기반 특징 생성
- 그룹별 정규화를 통한 장비별 특성 반영

👉 **결과**
- 정상 패턴 학습 기반 이상 탐지 가능
- AI 모델 기반 시스템 구현 준비 완료

## v2026.03.18

### Init
- 프로젝트 초기 환경 설정
  - GitHub Repository 생성
  - 기본 폴더 구조 구성 (frontend / backend / ai / docs)

👉 **결과**
- 팀 협업 환경 구축
- 프로젝트 개발 기반 마련
