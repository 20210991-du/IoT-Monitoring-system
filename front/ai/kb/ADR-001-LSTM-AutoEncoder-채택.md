---
tags: [ADR, AI, 모델]
date: 2026-04-19
status: 채택
supersedes: "초기 Dense AutoEncoder 안"
---

# ADR-001 · LSTM AutoEncoder (공통 모델 1개 + 장비별 Threshold) 채택

## 맥락

- 시간 순차성 있는 1시간 단위 센서 스트림
- 장비 수 55개지만 하나씩 별도 학습은 데이터량 부족
- **정상/관찰/이상** 3단계 분류가 필요

## 결정

**LSTM AutoEncoder 공통 모델 1개 + 장비별 Threshold 분리** 구조.

```
룰 기반 1차 필터
 → LSTM AutoEncoder (공통 1개)
   → 후처리 분류 (정상 / 관찰 / 이상)
```

### 모델 아키텍처

```
LSTM(128) → LSTM(64) → RepeatVector → LSTM(64) → LSTM(128) → Dense
```

### 분류 로직

```
정상   → MSE < threshold * 0.7
관찰   → threshold * 0.7 ≤ MSE < threshold
이상   → MSE ≥ threshold
```

### Threshold 산정

장비별 Threshold = **해당 장비 정상 데이터 MSE 의 99th percentile**

### 학습 주기

**1일 1회 배치 학습**.

## 근거

- **왜 LSTM?** — 1시간 단위 스트림의 시간 의존성(24시간 주기, 갑작스런 점프) 을 잡기 위함. Dense AE 대비 더 적은 false positive 기대.
- **왜 공통 모델 1개?** — 장비별 데이터량 부족. 공통 정상 패턴을 한 번 학습 후 장비별 Threshold 로만 개별화하는 것이 캡스톤 규모에 현실적.
- **왜 Threshold 는 장비별?** — 장비 설치 환경/연식이 달라 절대 MSE 값의 기준이 다르기 때문.

## 파생된 다른 결정

- 희생전류는 공통 모델 입력에서 빼고 2개 기기 전용 변수로 → [[ADR-002-희생전류-분리처리]]
- 통신품질은 모델 입력에서 빼고 룰 필터로 → [[ADR-003-통신품질-룰기반필터]]
- 그룹핑 기준은 시설번호가 아니라 정류기 단위 → [[ADR-004-정류기단위-그룹핑]]

## 입력 피처 구성

- **기본 센서 4개** (방식전위/AC유입/온도/습도)
- **파생 피처** (센서별 3종: 원본 / `diff1` / `dev24`)
- 합계: **4 × 3 = 12 채널**
- 희생전류는 별도 모델 또는 별도 입력 세트로 분리

## 저장 아티팩트

```
common_lstm_autoencoder.keras   ← 공통 모델
group_scalers.pkl               ← 그룹별 정규화 스케일러
device_thresholds.json          ← 장비별 임계치
model_config.json               ← 설정값
```

## 이상 원인 태그

센서별 MSE 기여도 기반 TOP 3 추출 → `"방식전위 이상 의심 83%"` 형식.

```python
feature_mse_array = np.mean(np.power(last_seq - pred, 2), axis=1)[0]
feature_contributions = {feat_name: float(feat_mse) for ...}
```

→ 메인 대시보드에 바로 표시 ([[프론트엔드#AnomalyCard]]).

## 현재 구현

| 파일 | 상태 |
|---|---|
| 학습: `gas_common_model_v3.py` | ✅ 완성 (v2 → v3 개선) |
| 판단 서버: `test_api.py` (FastAPI) | ✅ 거의 완성 |
| 로컬 테스트: `dashboard.py` (Streamlit) | ✅ 완성 |

## 남은 이슈

- ⚠️ `epochs=1` 하드코딩 → **100 으로 수정** 필요
- ⚠️ `test_api.py` `anomaly_scenario` 변수 버그
- ⚠️ 롤링 윈도우 feature engineering 추론 시 불안정

→ 추적: [[프로젝트/_템플릿/README]]

## 관련

- [[정보/서버 정보/맥스튜디오/서비스/AI모델]]
- [[회사_QA로그]]
