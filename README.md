# AI 기반 IoT 통합 관제 시스템

## 📌 Version
Current Version: v2026.04.19

👉 자세한 변경 내역은 [CHANGELOG.md](./CHANGELOG.md) 참고

## 📅 프로젝트 일정

프로젝트 진행 일정은 아래 문서에서 확인할 수 있습니다.

👉 [📄 일정 문서 보기](./docs/SCHEDULE.md)

## 📌 프로젝트 개요
본 프로젝트는 산업용 IoT 센서 데이터를 기반으로  
AI를 활용한 이상 탐지 및 통합 관제 대시보드를 구축하는 것을 목표로 한다.

기존 시스템이 단순 모니터링 중심이었다면,  
본 프로젝트는 **AI 기반 예지보전 및 통합 대시보드 제공**을 목표로 한다.

## 🎨 UI 설계 (Figma)

👉 [Figma 디자인 보기](https://www.figma.com/design/r1jlXaXkawhPn3qwDIJylV/%ED%86%B5%ED%99%A5%EA%B4%80%EC%A0%9C-%EB%8C%80%EC%8B%9C%EB%B3%B4%EB%93%9C?node-id=3-2&p=f)

---
## 🎯 주요 기능

### 1. 통합 대시보드
- 전체 센서 상태 요약 (정상 / 경고 / 위험)
- 실시간 데이터 시각화
- 최근 알람 및 이상 탐지 결과 표시

### 2. 센서 모니터링
- 센서별 시계열 데이터 그래프
- 실시간 값 확인
- 이상 데이터 강조 표시

### 3. AI 기반 이상 탐지
- LSTM AutoEncoder 기반 시계열 분석
- 정상 패턴 학습 후 이상 데이터 탐지
- 이상 점수 기반 위험도 분류

### 4. 지도 기반 관리
- 센서 위치 시각화
- 시설 및 장비 정보 확인

---

## 🧠 AI 모델 설명
- 모델: LSTM AutoEncoder
- 입력: 센서 시계열 데이터
- 출력: Reconstruction Error (MSE)
- 이상 판단: Threshold 초과 여부

---

## 🗂️ 프로젝트 구조
```text
iot-integrated-dashboard/
├── frontend/   # React 기반 대시보드
├── backend/    # API 서버 (DB 연동)
├── ai/         # AI 모델 및 분석 코드
├── data/       # 샘플 데이터
└── docs/       # 프로젝트 문서
```

## ⚙️ 기술 스택

### Frontend
- React (Vite)
- Recharts (데이터 시각화)
- Leaflet (지도)

### Backend
- Node.js / Express (예정)
- REST API

### AI
- Python
- TensorFlow / Keras
- LSTM AutoEncoder

### Database
- MySQL / MongoDB

---

## 📊 기대 효과

- 관제 업무 자동화
- 이상 탐지 정확도 향상
- 운영자 의존도 감소
- 데이터 기반 의사결정 지원

---

## 👥 역할 분담

- Frontend: 대시보드 UI 및 시각화
- Backend: API 및 DB 연동
- AI: 이상 탐지 모델 개발

---

## 🚀 실행 방법

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### Backend (예정)
```bash
cd backend
npm install
node app.js
```

### AI 모델 실행
```bash
cd ai/scripts
python gas_common_model_v2.py
```

---

## 📌 향후 계획
- 실시간 데이터 연동
- AI 모델 성능 개선
- 알림 시스템 (SMS / Web Push)
- 지도 기반 분석 강화
