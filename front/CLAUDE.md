# capstone/front — Claude Code 세션 컨텍스트

**Claude 에게:** 이 파일은 새 Claude Code 창에서 이 프로젝트를 열었을 때 자동으로 읽히는 컨텍스트입니다. 상세 문서는 [`HANDOFF.md`](./HANDOFF.md), 사용자용 README 는 [`README.md`](./README.md) 참고.

---

## 이 프로젝트가 뭐야

**매설배관 AI 통합관제 시스템** — 졸업 프로젝트 프론트엔드. 55개 감시 노드의 6종 센서(방식전위/AC유입/희생전류/온도/습도/통신품질) 데이터를 LSTM-Autoencoder 로 이상탐지하는 관제 대시보드.

**원본:** `claude.ai/design` 에서 제공된 HTML/JSX 프로토타입(`~/Downloads/졸업 프로젝트_대시보드-handoff.zip`)을 Vite + React ES 모듈로 이관한 상태.

## 스택

- React 18.3.1 + Vite 5, **Plain JSX (TS 아님)**
- CSS Variables + inline style objects (CSS-in-JS 라이브러리 없음)
- 라우팅/상태관리 라이브러리 없음 — `App.jsx` 에서 `useState` + localStorage
- 백엔드 연동 없음. `src/data/mockData.js` mock data (mulberry32 seeded, 결정론적)

## 디렉터리

```
src/
├── main.jsx                  # createRoot + global.css
├── App.jsx                   # screen/tab 상태, 테마, postMessage 에디트 프로토콜
├── styles/global.css         # CSS 변수, 키프레임 7종
├── data/mockData.js          # EQUIPMENT(55), MAP_MARKERS, AI_ANOMALIES, AI_WATCH, AI_INSIGHTS, LOG_TEMPLATES
├── components/
│   ├── Icons.jsx             # ic() 팩토리 + 30여 아이콘
│   ├── chrome.jsx            # Header, SubNav, EmergencyBanner, FooterBar, useClock
│   ├── MapPanel.jsx          # SVG 지형 3스타일(light/dark/satellite)
│   └── TweaksPanel.jsx
└── pages/
    ├── Login.jsx
    ├── Dashboard.jsx         # 800+줄, 여러 서브 컴포넌트 + AnalysisModal
    └── Equipment.jsx         # 테이블 정렬/필터/검색/페이지네이션
```

## 현재 레이아웃 전략

**뷰포트 기반 fluid** (letterbox 1920 고정에서 최근 전환됨):
- `#canvas` = `100% × 100vh`, `min-width: 1280px`, `min-height: 720px`
- Chrome(Header 56 + SubNav 48 + Banner 40 + Footer 32) 픽셀 고정
- 그 사이 content 영역은 `position:absolute inset:0` + 내부 flex/grid
- 1280 미만 뷰포트는 스크롤 허용

## 실행

```bash
npm install
npm run dev      # http://localhost:5173
npm run build    # dist/ (최근 빌드: 230KB, gzip 68KB)
```

## 지켜야 할 것 (DO)

- **시각 결과는 원본 프로토타입과 1:1 유지.** 색상/간격/폰트/애니메이션 건드리지 말 것. 사용자가 "디자인 바꾸자" 고 명시할 때만 변경.
- **CSS 변수(`--bg`, `--ink`, `--brand` 등)만 사용.** 하드코딩 색상 추가 금지 (다크모드에서 깨짐). 단 기존에 이미 하드코딩된 몇 곳(`#0f172a` 등 MarkerPopup 내부)은 이슈로 알려진 상태.
- **mock data shape 변경 금지** — 백엔드 연동 시에도 동일 shape 유지. 타입 계약은 `HANDOFF.md` §4.4.
- **`index.html` 의 `<script id="tweaks-defaults">` JSON 과 `/*EDITMODE-BEGIN/END*/` 주석 유지** — Claude Design 에디트 프로토콜용.
- **한글 UI 카피** — 번역하지 말 것. 도메인 용어(방식전위, 희생전류, 이상 의심 등) 그대로.

## 하지 말 것 (DON'T)

- 라우팅 라이브러리(react-router 등), 상태 라이브러리(redux/zustand), UI 라이브러리(MUI/Chakra 등) 사용자 요청 없이 **추가 금지**.
- `npm run dev` 를 foreground 로 띄우지 말 것 — 무한 실행됨. 반드시 `run_in_background: true`.
- 프로토타입 JSX 의 `window.*` 전역 패턴 되돌리지 말 것. 이미 ES 모듈로 이관됨.

## 알려진 이슈

1. `App.jsx` 와 `Dashboard.jsx` 에 `EquipmentDrawer` / `DashboardEquipmentDrawer` 가 **중복 정의**. 거의 동일 컴포넌트지만 각 탭 별 drawer 역할. 통합 리팩터 시 주의.
2. `MapPanel` 마커 팝업 내부에 하드코딩 색상(`#0f172a`, `#64748b` 등) 존재 → 다크 모드에서 팝업 배경/텍스트 가독성 저하.
3. 울트라와이드 모니터에서 우측 여백 발생 — 대시보드 우측 컬럼 440px 고정 때문. 필요 시 fluid 로 변경.

## 다음 작업 후보 (사용자가 명시적으로 요청할 때만)

- **백엔드 연동** — `src/api/` 신설, `VITE_API_BASE_URL`, fetch 기반. 실시간은 SSE/WS 권장. (상세: `HANDOFF.md` §7.1)
- **TypeScript 전환** — data shape 이 명확하므로 수월.
- **Dashboard.jsx 파일 분할** — 800+줄. 서브 컴포넌트별로 분리.
- **CSS Modules/vanilla-extract 도입** — inline style → reusable class.
- **테스트** — Vitest + React Testing Library.

## 세션 복원 힌트

사용자가 "어디까지 했어?" 물어보면:
1. 원본 프로토타입(CDN React + Babel) 을 Vite + ES 모듈로 마이그레이션 완료
2. 뷰포트 기반 fluid 레이아웃으로 전환 완료 (letterbox 제거)
3. `HANDOFF.md` 작성 완료
4. 백엔드 미연동, 커밋 없음 (`git init` 만 됨, staged 상태)

참고로 상위 `../CLAUDE.md` 와 `../../CLAUDE.md` 에 워크스페이스 전반 규칙(secrets/ 제외, `.env.example` 필수 등)이 있음.
