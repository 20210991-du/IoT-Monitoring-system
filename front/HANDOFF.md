# 매설배관 AI 통합관제 시스템 프론트엔드 — 이관 문서

**위치:** `/Users/pjh/PJHwork/projects/capstone/front`
**상태:** Vite + React ES 모듈 구조로 마이그레이션 완료, 뷰포트 기반 fluid 레이아웃. mock data 로 동작. 백엔드 미연동.

---

## 1. 프로젝트 정체성

- **도메인:** 매설배관 방식전위(Cathodic Protection) 원격감시 대시보드
- **화면 규모:** 55 감시 노드 × 6 센서 (방식전위 mV / AC 유입 mV / 희생전류 mA / 온도 °C / 습도 % / 통신품질 dBm)
- **AI 모델:** LSTM-Autoencoder v3 (임계치 0.00409, MSE 기반 이상탐지)
- **타겟 뷰포트:** 1920×1080 관제실 대형 모니터 기준 디자인. 최소 1280×720 보장.

---

## 2. 스택

| 항목 | 버전/선택 |
|---|---|
| 런타임 | React 18.3.1 |
| 번들러 | Vite 5.4.x + `@vitejs/plugin-react` 4 |
| 언어 | **JSX (Plain JS)** — TS 아님 |
| 스타일 | **CSS Variables + inline style objects**. CSS-in-JS 라이브러리 없음. 전역 CSS 1개 파일 |
| 라우팅 | 없음. `screen` state 로 login↔app, `tab` state 로 dashboard↔equipment (localStorage 영속) |
| 상태관리 | React `useState`/`useEffect` 만. Redux/Zustand 등 없음 |
| 테스트 | 없음 |
| 린트 | 없음 |

---

## 3. 디렉터리 구조

```
front/
├── index.html                # Vite 엔트리 (Google Fonts, #stage>#canvas>#root, tweaks-defaults JSON)
├── package.json
├── vite.config.js            # 기본 설정, port 5173
├── .env.example              # VITE_API_BASE_URL 플레이스홀더
├── .gitignore
├── README.md                 # 사용자용 실행 문서
├── HANDOFF.md                # 이 문서
└── src/
    ├── main.jsx              # createRoot + StrictMode + global.css import
    ├── App.jsx               # 최상위 라우팅, 테마 dispatch, postMessage 에디트 프로토콜
    ├── styles/global.css     # CSS 변수(light/dark), 키프레임 7종, 스크롤바
    ├── data/mockData.js      # EQUIPMENT(55), MAP_MARKERS, AI_ANOMALIES, AI_WATCH, AI_INSIGHTS, LOG_TEMPLATES
    ├── components/
    │   ├── Icons.jsx         # ic() 팩토리 + 아이콘 30여 개
    │   ├── chrome.jsx        # Header, SubNav, EmergencyBanner, FooterBar, useClock, fmtClock
    │   ├── MapPanel.jsx      # SVG 지형 3가지 스타일(light/dark/satellite), 마커, 스캔라인
    │   └── TweaksPanel.jsx   # 디자인 모드 토글 패널
    └── pages/
        ├── Login.jsx         # SVG 배경(그리드 + 궤도 링 + 펄스 노드), 글래스 카드, 1초 인증 지연
        ├── Dashboard.jsx     # KPIRow / AIPanels / AIAdvicePanel / MapPanelWrap / TableSummary /
        │                     #   LogPanel / AnalysisModal / DashboardEquipmentDrawer / useLogStream
        └── Equipment.jsx     # 전체 장비 테이블 (정렬/필터/검색/14행 페이지네이션)
```

---

## 4. 주요 설계 결정

### 4.1 뷰포트 기반 Fluid 레이아웃
- 원본 디자인 프로토타입은 1920×1080 고정 캔버스 + `transform:scale()` letterbox 였으나, 실제 다양한 해상도 대응을 위해 fluid 로 전환.
- `#canvas`: `100% × 100vh`, `min-width:1280px; min-height:720px`. 미만에서는 스크롤.
- Chrome(Header 56 + SubNav 48 + Banner 40 + Footer 32) 은 픽셀 고정, 그 사이 content 영역이 `position:absolute` + flex/grid 로 늘어남.
- 일부 컴포넌트는 디자인 의도상 여전히 고정 폭 유지: AI 사이드바 440px, 드로어 520px, 분석 모달 760px.

### 4.2 CSS 변수 기반 테마
- `:root { --bg, --bg-elev, --ink, --brand, --ok, --warn, --err, ... }` 패턴.
- 다크 모드는 `html[data-theme="dark"]` selector 로 변수만 override.
- `App.jsx` 에서 `document.documentElement.setAttribute("data-theme", ...)` 로 스위치.

### 4.3 Mock 데이터 결정론성
- `src/data/mockData.js` 는 **mulberry32(42) seeded RNG** 로 매 로드마다 동일한 55개 장비 생성.
- 개발/검증 과정에서 값이 뒤집히지 않음. 새로고침해도 `TB24-5JN042` 는 항상 같은 MSE 를 가짐.
- API 연동 시 이 파일을 대체하되, 동일한 data shape 유지 필요 (아래 4.4 참조).

### 4.4 Data Shape (API 설계용 계약)

```ts
type Equipment = {
  id: number;
  facilityId: string;        // "TB-2023-0001"
  deviceId: string;          // "TB24-5JN001"
  location: string;          // 한글 주소 자유 텍스트
  status: "normal" | "anomaly" | "warn" | "offline";
  label: string | null;      // 이상 라벨 ("위상차 급변" 등)
  volt: number;              // 방식전위 mV (음수)
  ac: number;                // AC 유입 mV
  sacrificial: number;       // 희생전류 mA
  temp: number;              // 온도 °C
  hum: number;               // 습도 %
  commDbm: number;           // 통신품질 dBm
  commOk: boolean;
  mse: number | null;        // LSTM-AE 재구성 오차
  contribution: { sensor: string; pct: number }[];  // 센서별 기여도 (top 3)
  threshold: number;         // 고정 0.409
  updatedAt: string;         // "14:32"
  nextCollectMin: number;
  zone: "제1구역" | "제2구역" | "제3구역" | "제4구역";
  lat: number; lng: number;
};

type Anomaly = {
  node: string; label: string; mse: number; threshold: number; zone: string;
  contribution: { sensor: string; pct: number }[];
  summary: string;   // AI 분석 요약 (한글 자연어)
  action: string;    // 권장 조치 (한글)
};
```

### 4.5 Design Edit Protocol
- `index.html` 의 `<script id="tweaks-defaults">` JSON 블록은 `claude.ai/design` 호환용. 주석 `/*EDITMODE-BEGIN*/ /*EDITMODE-END*/` 제거 금지.
- `App.jsx` 의 `postMessage` 리스너는 iframe 부모(디자인 툴)가 없으면 조용히 무시. 실사용 시 무해.

---

## 5. 실행

```bash
cd /Users/pjh/PJHwork/projects/capstone/front
npm install
npm run dev       # http://localhost:5173
npm run build     # dist/ 로 정적 번들 (최근 빌드: JS 230KB gzip 68KB)
npm run preview   # 빌드 결과 미리보기 http://localhost:4173
```

---

## 6. 구현된 상호작용

- 로그인 → 대시보드 전환 (1초 fake auth delay, localStorage 저장)
- 탭 전환 (대시보드 ↔ 전체 장비 현황)
- 헤더 시계 실시간 (1초 간격 setInterval, 시드 시각 2026-03-26 14:42:05)
- KPI 카드 클릭 → 하단 요약 테이블 필터
- 지도 마커 클릭 → 상세 팝업 (이상 마커엔 pulse-ring 애니메이션)
- AI 이상/관찰 카드 클릭 → AnalysisModal (3-band 임계치 차트 + 기여도 바 + 요약 + 권장 조치)
- 전체 장비 테이블: 컬럼 헤더 클릭 정렬, 상태/구역 필터, 검색, 페이지네이션
- 테이블 행 클릭 → EquipmentDrawer 슬라이드 인 (센서값 6개 + 트렌드 스파크라인)
- 경고 배너 dismiss
- Tweaks 패널: 테마 light/dark, 지도 스타일 light/dark/satellite, 실시간 로그 ON/OFF
- 실시간 시스템 로그 스트림 (1.8초 간격 신규 로그, autoPlay OFF 시 정지)

---

## 7. 다음 단계 (이관 후 작업)

### 7.1 백엔드 연동 (가장 먼저 해야 할 작업)
1. `src/api/` 디렉터리 생성
2. `VITE_API_BASE_URL` 환경변수 도입 (`.env.local`)
3. `mockData.js` 를 `src/api/equipment.js`, `src/api/anomalies.js` 등으로 분리. 각 함수는 `async` fetch.
4. Dashboard/Equipment 에서 `useEffect` + `useState` 로 로딩/에러 핸들링 추가.
5. 실시간 업데이트 전략 결정: polling(30초) vs SSE vs WebSocket. 현재 UI 는 "REALTIME STREAM" 을 강조하므로 SSE/WS 권장.

### 7.2 권장 리팩터
- **TypeScript 전환** — data shape 계약이 명확하므로 TS 이전 수월. `tsconfig.json` + 파일 확장자 변경 + Equipment 타입부터.
- **Inline style → CSS Modules 또는 vanilla-extract** — 지금은 CSS-in-JS 라이브러리가 없어 런타임 오버헤드는 없지만 재사용성이 떨어짐. `Panel`, `PanelHeader` 정도는 CSS class 로 추출.
- **`Dashboard.jsx` 파일 분할** — 800+줄. `Kpi`, `AnomalyCard`, `AnalysisModal`, `MapPanelWrap`, `LogPanel`, `TableSummary` 각각 별도 파일로.
- **상태관리 도입 고려** — 현재는 `App.jsx` 에서 props drilling. 기능 확장 시 `zustand` 또는 Context 도입.
- **테스트** — 최소 `mockData.js` 의 결정론성 (Vitest), Equipment 테이블 정렬/필터 로직 (React Testing Library).

### 7.3 디자인 부채
- 일부 문자열 하드코딩 (예: FooterBar 의 "SERVER: ASIA-KR-01 · 128.4.17.2"). 환경변수/백엔드 응답으로 이동 필요.
- Login 의 `admin.siwon` pre-populated 은 데모용. 실제 로그인 구현 시 제거.
- `useLogStream` 은 `Math.random()` 으로 가짜 로그 생성. 실제 로그 SSE 로 대체.

---

## 8. 핵심 파일 (신규 기여자가 먼저 읽어야 할 순서)

1. `README.md` — 실행/구조 개요
2. `src/App.jsx` — 최상위 라우팅, 상태 흐름, 테마
3. `src/data/mockData.js` — 데이터 shape 이해
4. `src/pages/Dashboard.jsx` — 가장 큰 화면, 대부분 컴포넌트가 여기 정의됨
5. `src/styles/global.css` — 테마 변수, 애니메이션
6. `src/components/chrome.jsx` — Header/SubNav/Banner/Footer
7. `src/pages/Equipment.jsx` — 테이블 정렬/필터 패턴 참고
8. `src/components/MapPanel.jsx` — 3가지 스타일 오브젝트 구조

---

## 9. 디자인 원본

- **Claude Design 핸드오프 번들:** `~/Downloads/졸업 프로젝트_대시보드-handoff.zip`
- **추출된 원본 프로토타입:** `/tmp/design-bundle/untitled/project/` (CDN React + 브라우저 Babel, 10개 파일)
- **스크린샷 레퍼런스:** `/tmp/design-bundle/untitled/project/screenshots/` (check.jpg / current.jpg / eq.jpg / v2.jpg)
- 디자인 원본 대비 시각 결과(색/레이아웃/애니메이션/타이포)는 1:1 로 유지됨. 단 letterbox → fluid 전환으로 인해 매우 넓은 모니터에서 여백 발생 가능.

---

## 10. 알려진 이슈 / 주의사항

- `dashboard.jsx` 와 `App.jsx` 에 **이름이 같은 `EquipmentDrawer` 컴포넌트가 각각 정의**되어 있음 (`DashboardEquipmentDrawer` vs `EquipmentDrawer`). 지도→테이블 행 클릭은 Dashboard 내부 것, Equipment 탭 행 클릭은 App 레벨 것. 통합 시 주의.
- `useLogStream` 의 `Math.random()` 은 의도적 비결정론적. mockData 의 결정론과 구분됨.
- 로그인 버튼의 "로그인 상태 유지" 체크박스는 UI 만 존재, 실제 localStorage 연동 없음.
- `App.jsx` 의 `window.parent.postMessage` 는 Claude Design iframe 용. 프로덕션에서도 iframe 아니면 try/catch 로 무시됨 — 제거 가능.
- 다크 모드 전환 시 지도 마커 팝업의 일부 색상은 흰색 고정 (Marker popup 내부에 `color: "#0f172a"` 하드코딩). 추후 수정 필요.
