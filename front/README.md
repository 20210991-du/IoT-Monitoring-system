# 매설배관 AI 통합관제 시스템 (front)

졸업 프로젝트 — AI 기반 매설배관 방식전위 원격감시 대시보드.
1920×1080 관제실 대형 모니터용 고정 캔버스, 내부 스케일 letterbox 적용.

## 스택

- React 18
- Vite 5
- Pure CSS (CSS Variables 기반 light/dark 테마)
- 55 감시 노드 × 6 센서 mock data (`src/data/mockData.js`)

## 실행

```bash
npm install
npm run dev       # http://localhost:5173
npm run build     # dist/ 로 정적 번들 빌드
npm run preview   # 빌드 결과 미리보기 (http://localhost:4173)
```

## 주요 화면

- **로그인** (`src/pages/Login.jsx`) — SVG 배경(그리드 + 궤도 링 + 펄스 노드), 글래스모피즘 카드
- **대시보드** (`src/pages/Dashboard.jsx`) — KPI 5개, GIS 지도(3가지 스타일), AI 이상/관찰/조언, 실시간 로그 스트림
- **전체 장비 현황** (`src/pages/Equipment.jsx`) — 55행 정렬/필터/검색/페이지네이션 테이블

## 상호작용

- 로그인 → 대시보드 전환 (localStorage 에 `screen` 유지)
- 대시보드 ↔ 전체 장비 현황 탭 전환 (localStorage 에 `tab` 유지)
- KPI 카드 클릭 → 하단 요약 테이블 필터
- 지도 마커 클릭 → 장비 상세 팝업, 이상 마커 펄스 링 애니메이션
- AI 이상/관찰 카드 클릭 → `AnalysisModal` 열림 (기여도 차트 + 권장 조치)
- 테이블 행 클릭 → `EquipmentDrawer` 슬라이드 인
- 경고 배너 dismiss
- 우측 하단 Tweaks 패널: 테마(light/dark), 지도 스타일(light/dark/satellite), 실시간 로그 ON/OFF

## 디자인 출처

`claude.ai/design` 핸드오프 번들 (`졸업 프로젝트_대시보드-handoff.zip`) 기반.
원본 프로토타입은 CDN React + 브라우저 Babel 구조였고, 이 저장소는 이를 Vite + ES 모듈로 재배선한 버전입니다. 시각적 결과(색상, 레이아웃, 애니메이션, 타이포) 는 프로토타입과 1:1 로 유지됩니다.

## 백엔드 연동

현재는 `src/data/mockData.js` 의 결정적 mock 데이터만 사용합니다.
향후 API 연동 시:

1. `.env.local` 에 `VITE_API_BASE_URL` 설정
2. `src/data/mockData.js` 를 `src/api/*.js` 로 분리, fetch 기반 로더 구현
3. 페이지 레벨에서 `useEffect` + 로딩 상태 추가

## 구조

```
src/
├── main.jsx               # React 루트, CSS 로드, 1920 캔버스 스케일러
├── App.jsx                # 스크린/탭 상태, 테마 dispatch, 배너·모달·드로어
├── styles/global.css      # CSS 변수(light/dark), 키프레임, 스크롤바
├── data/mockData.js       # EQUIPMENT(55), MAP_MARKERS, AI_* 리스트
├── components/
│   ├── Icons.jsx          # ic() 팩토리 + 30여 개 SVG 아이콘
│   ├── chrome.jsx         # Header, SubNav, EmergencyBanner, FooterBar, useClock
│   ├── MapPanel.jsx       # SVG 지형 3가지 스타일, 마커, 스캔라인
│   └── TweaksPanel.jsx    # 디자인 모드 토글 패널
└── pages/
    ├── Login.jsx
    ├── Dashboard.jsx      # KPIRow, AIPanels, AIAdvicePanel, MapPanelWrap, TableSummary, LogPanel, AnalysisModal
    └── Equipment.jsx      # 전체 장비 테이블
```
