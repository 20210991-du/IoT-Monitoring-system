---
tags: [분야, AI, 모델]
date: 2026-04-19
owner: 이두현
---

# 06 · AI 모델

## 구조 한 줄

**룰 기반 1차 필터 → 공통 LSTM AutoEncoder (1개) → 장비별 Threshold 후처리 → 3단계 분류**

결정 배경: [[ADR-001-LSTM-AutoEncoder-채택]]

### 룰 기반 1차 필터 (AI 투입 전)

AI 학습/추론 전에 명백한 장애 케이스를 걸러내는 규칙 기반 단계:

1. **통신 두절** — 일정 시간 이상 데이터 없음
2. **배터리 임계치 이하**
3. **센서값 결측** 또는 **물리적으로 불가능한 값**
4. **동일값 반복** — 센서 고착 의심 (고장 가능성)

이 단계에서 걸러진 장비는 LSTM AE 입력에서 제외하고 별도 이벤트로 처리.

## 스택

- Python 3.10+
- TensorFlow / Keras 2.15+
- scikit-learn (MinMaxScaler)
- pandas / numpy

## 입력 피처

### 기본 센서 4개 (공통 모델 입력)

- 방식전위 (mV)
- AC유입 (mV)
- 온도 (℃)
- 습도 (%)

제외:
- **희생전류** — 2개 기기 전용 ([[ADR-002-희생전류-분리처리]])
- **통신품질** — 룰 필터로 이동 ([[ADR-003-통신품질-룰기반필터]])

### 파생 피처 (센서별 3종)

- 원본값
- `diff1` — 직전 값 대비 변화량
- `dev24` — 24시간 이동평균 대비 편차

합계: **4 센서 × 3 파생 = 12 채널**

## 모델 아키텍처

```
Input (seq_len, 12)
  → LSTM(128, return_sequences=True)
  → LSTM(64)
  → RepeatVector(seq_len)
  → LSTM(64, return_sequences=True)
  → LSTM(128, return_sequences=True)
  → TimeDistributed(Dense(12))
```

- `seq_len`: 보통 24~48 (1시간 간격 × 24/48시간)
- Loss: `mse`
- Optimizer: `adam`
- 학습 배치: **1일 1회**, **epochs 100** *(현재 v3 에 1 하드코딩 — 수정 필요 [[프로젝트/_템플릿/README]])*

## 이상 판단 로직

```
정상   → MSE < threshold * 0.7
관찰   → threshold * 0.7 ≤ MSE < threshold
이상   → MSE ≥ threshold
```

장비별 Threshold = 정상 데이터 MSE **99th percentile**.

## 이상 원인 태그 (센서 기여도)

이상 판정이 났을 때 **어느 센서가 얼마나 기여했는지** 를 재구성 오차의 채널별 분포로 계산.

```python
# 1. 마지막 시퀀스의 예측 vs 실제 차이의 제곱 → 시간축 평균
feature_mse_array = np.mean(np.power(last_seq - pred, 2), axis=1)[0]

# 2. 피처명과 짝지어 dict 로 변환
feature_contributions = {
    feat_name: float(feat_mse)
    for feat_name, feat_mse in zip(feature_cols, feature_mse_array)
}

# 3. 상위 3개 추출 → UI 태그로 표시
top3 = sorted(feature_contributions.items(), key=lambda x: -x[1])[:3]
# 예: [("방식전위_dev24", 0.012), ("AC유입_diff1", 0.003), ("온도", 0.001)]
```

### UI 표시 예시

```
방식전위 이상 의심 83%
AC유입  이상 의심 12%
희생전류 이상 의심  5%
```

파생 피처는 UI 에 보일 때 base 이름만 남기는 방식(예: `방식전위_dev24` → `방식전위`). 대시보드 AnomalyCard 에 바로 노출 → [[프론트엔드#AnomalyCard]].

## 저장 아티팩트

```
ai/models/
├── common_lstm_autoencoder.keras   ← 공통 모델
├── group_scalers.pkl               ← 정류기 그룹별 MinMaxScaler
├── device_thresholds.json          ← 장비별 threshold
└── model_config.json               ← 설정
```

`group_scalers` 의 그룹 단위 = 정류기 그룹 ([[ADR-004-정류기단위-그룹핑]]).

## 파일 구성 (2026-04-20 pull 후 최신)

경로: `[경로 비공개]`

| 역할 | 파일 | 상태 |
|---|---|---|
| **학습 (최신)** | `ai/scripts/gas_common_model_train(2026.04.14).py` | ✅ v2026.04.19 대대적 개선 |
| **예측 (최신)** | `ai/scripts/gas_common_model_predict(2026.04.14).py` | ✅ 신규 분리 |
| 학습 v3 (보존) | `ai/scripts/gas_common_model_v3(2026.04.06).py` | 이전 단일 파일 |
| 학습 v2 (보존) | `ai/scripts/gas_common_model_v2.py` | 초기 버전 |

> ⚠️ AI 파트는 **이두현 담당 영역**. 다른 팀원이 폴더 구조 / 코드 임의 변경 X.
>
> ⚠️ **2026-04-19 에 test_api.py + dashboard.py 삭제됨** — FastAPI 추론 서버는 향후 `backend/` 에 통합 예정.

### 아티팩트 저장 위치

```
ai/
├── config/
│   ├── device_thresholds.json     # 장비별 임계치
│   ├── model_config.json          # time_steps, feature 목록 등
│   └── eval_metrics.json          # 🆕 평가 지표 (v2026.04.19 추가)
├── models/
│   ├── common_lstm_autoencoder.keras   # 공통 모델
│   └── group_scalers.pkl              # 그룹별 정규화 스케일러
└── results/
    ├── anomaly_plot_TB24-250401.png
    ├── anomaly_plot_TB24-250402.png
    ├── anomaly_plot_TB24-250403.png
    └── training_history.png
```

### 학습/예측 분리 구조 (v2026.04.19 기준)

```
┌─ 학습 (1일 1회 배치) ──────────────────┐
│ gas_common_model_train(2026.04.14).py │
│ ┌────────────────────────────────────┐ │
│ │ 데이터 분할 (시간 순):             │ │
│ │   train 70% / val 15% / test 15%   │ │
│ │ Scaler fit: train 구간만           │ │
│ │ LSTM AE 학습 (epochs=50)           │ │
│ │   - activation: tanh (기본값)      │ │
│ │   - validation_data=(X_val, X_val) │ │
│ │   - EarlyStopping(patience=5)      │ │
│ │ Threshold: test 구간에서 99 pct    │ │
│ │ 아티팩트 저장:                     │ │
│ │   models/*.keras, *.pkl            │ │
│ │   config/*.json                    │ │
│ │ 평가: eval_metrics.json            │ │
│ └────────────────────────────────────┘ │
└────────────────────────────────────────┘

┌─ 예측 (운영) ──────────────────────────┐
│ gas_common_model_predict(2026.04.14).py│
│ ┌────────────────────────────────────┐ │
│ │ run_batch_prediction()             │ │
│ │   전체 장비 일괄 이상 탐지         │ │
│ │   → CSV 저장                       │ │
│ │ classify_risk_level(mse, th, ratio)│ │
│ │   → 이상 / 관찰 / 정상             │ │
│ │ OBSERVATION_RATIO 파라미터 조정    │ │
│ │ get_sacrificial_device_data()      │ │
│ │   → TB24-250406, 407 전용 처리     │ │
│ │ apply_comm_quality_filter()        │ │
│ │   -115 dBm 이하 = 단절             │ │
│ │   연속 3회 이상 = 고장             │ │
│ │   → 통신단절_플래그 / 통신고장_플래그 │ │
│ └────────────────────────────────────┘ │
└────────────────────────────────────────┘
```

### 주요 상수 / 함수

```python
# 희생전류 분리 처리
SACRIFICIAL_DEVICES = ["TB24-250406", "TB24-250407"]
SACRIFICIAL_FEATURES = [...]  # 희생전류 및 파생 피처

def get_sacrificial_device_data(df):
    """55개 장비 중 희생양극 방식 2개만 별도 반환"""
    ...

# 통신품질 룰 기반 필터
def apply_comm_quality_filter(df):
    """
    -115 dBm 이하 → 통신단절_플래그
    연속 3회 이상 단절 → 통신고장_플래그
    (AI 학습 feature 에서 통신품질 제외 전제)
    """
    ...

# 위험도 분류
def classify_risk_level(mse, threshold, observation_ratio=OBSERVATION_RATIO):
    """
    mse >= threshold          → '이상'
    threshold*ratio <= mse    → '관찰'
    else                      → '정상'
    """
    ...
```

### 저장된 아티팩트

| 파일 | 용도 |
|---|---|
| `ai/models/common_lstm_autoencoder.keras` | 공통 모델 |
| `ai/models/group_scalers.pkl` | 그룹별 MinMaxScaler |
| `ai/config/device_thresholds.json` | 장비별 임계치 |
| `ai/config/model_config.json` | 설정 (⚠️ 6센서 유지 중) |

### 시각화 산출물

- `ai/results/anomaly_plot_TB24-250401.png`
- `ai/results/anomaly_plot_TB24-250402.png`
- `ai/results/anomaly_plot_TB24-250403.png`
- `ai/results/training_history.png`

### ⚠️ 코드와 결정의 싱크 상태

`ai/config/model_config.json` → `base_features` 가 현재 **6센서 전체**:

```json
"base_features": ["방식전위", "AC유입", "희생전류", "온도", "습도", "통신품질"]
"feature_columns": [... 6 × 3 = 18채널 ...]
```

결정 사항 반영 시 변경될 예정:
- 희생전류 제외 ([[ADR-002-희생전류-분리처리]])
- 통신품질 제외 ([[ADR-003-통신품질-룰기반필터]])
- → **4 센서 × 3 = 12 채널** 로 축소

→ 추적: [[프로젝트/_템플릿/README#D3]]

## 알려진 코드 이슈 (2026-04-20 업데이트)

v2026.04.19 개선으로 **대부분 해결됨**.

| 우선순위 | 파일 | 문제 | 상태 |
|---|---|---|---|
| 🔴 | `train.py` (구 v3) | `epochs=1` | ✅ **50 으로 수정** |
| 🔴 | `test_api.py` | `anomaly_scenario` 변수 | ✅ 파일 삭제로 해소 |
| 🟡 | 학습 스크립트 | 희생전류 전체 보간 | ✅ **`SACRIFICIAL_DEVICES` 분리 구현** |
| 🟡 | 학습 스크립트 | 통신품질 학습 feature 포함 | ✅ **룰 필터로 분리** (`apply_comm_quality_filter`) |
| 🟡 | `test_api.py` | `predict_device_window` 롤링 불안정 | ✅ 파일 삭제로 해소 |
| 🟡 | `model_config.json` | `base_features` 6센서 | ✅ **4센서로 축소** |

남은 항목:
- FastAPI 추론 서버가 삭제되었으므로, **`backend/`** 에서 모델 로딩 + 예측 엔드포인트 새로 구현 필요

→ 통합 추적: [[프로젝트/_템플릿/README]]

## 실 데이터에서 확인된 이상 사례

- **11/18~20**: 방식전위 상승 + AC유입 하락 동시 발생 → 정류기 단위 재검토 필요
- **TB24-250436**: 통신품질 -115 dBm 통신 불량
- **TB24-250411**: 방식전위 -801mV (기준 미달)
- **TB24-250418, 419**: AC유입 3,703~4,218mV 급등

→ [[요구사항#확인된-특이사항]]

## ★ 모델 평가용 Ground Truth (회사 라벨링)

OmniSolution 의 2026-03-18 [[2026-04-20-OneDrive자료-요약#★-핵심-신규-정보--회사가-라벨링한-이상-장비-목록|예비 분석 보고서]] 에서 **9대 이상 장비** 가 명시됨. 우리 LSTM-AE + 룰 필터의 **검증 정답지**.

| 카테고리 | 장비 수 | 모델 평가 기준 |
|---|---|---|
| 결선이상 (방식전위 패턴 이상) | 8 | LSTM-AE 가 "이상 / 관찰" 으로 분류해야 함 |
| 통신점검 (통신 불가) | 1 | 룰 필터 ([[ADR-003-통신품질-룰기반필터]]) 가 잡아야 함 |
| **총** | **9 / 55 = 16%** | 정상/이상 비율의 실데이터 그대로 |

상세 장비 목록: [[데이터#★-회사-라벨링-이상-장비-ground-truth-활용-가능]]

### 평가 메트릭 (제안)

```
Precision = (모델이 이상이라 한 것 중 실제 이상 9대) / (모델이 이상이라 한 총 개수)
Recall    = (모델이 이상이라 한 것 중 9대 ↔ 실제 9대 중 잡은 수) / 9
F1        = 2·Precision·Recall / (Precision+Recall)
```

LSTM-AE 출력 + 룰 필터 출력 합산 후, 위 9대 명단과 대조.

### 도메인 태그 어휘

회사가 쓰는 분류 용어 (`결선이상` / `전극이상` / `통신점검`) 를 [[#이상-원인-태그|AI 출력 태그]] 의 한국어 별칭으로 채택 검토. 운영자가 익숙한 어휘로 보면 신뢰도 ↑.

## 챗봇 연동 (front / 2026-05-18 추가)

`front/server.js` 가 부팅 시 `ai/config/*.json` 3개를 메모리 로드해서 챗봇 도구로 노출:

| 출처 | 챗봇에서의 활용 |
|---|---|
| `device_thresholds.json` | 단말별 threshold 값 (55대) → 챗봇이 "TB24-XXX 정상 한계는?" 답변 |
| `model_config.json` | base_features 4, time_steps 24, sacrificial_devices 2 등 → "AI 모델 어떻게 학습?" 답변 |
| `eval_metrics.json` | test_mse_p99, mean_threshold 등 → 전체 모델 통계 답변 |

신규 도구: **`get_ai_model_info(deviceId?)`** — deviceId 있으면 단말 threshold + 70%/100% 한계, 없으면 전체 모델 메타. (서버 부팅 로그: `▶ AI cfg  thresholds=55대 · model_config=OK · eval_metrics=OK`)

`get_device_detail` 응답에도 `ai: { threshold, threshold70, threshold100, isSacrificial }` 동봉. 챗봇이 단말 상세 조회 시 위험도 분류 기준을 함께 답변.

분류 기준은 본 문서의 "이상 판단 로직" 과 정확히 일치:
- 정상: MSE < threshold × 0.70
- 관찰: threshold × 0.70 ≤ MSE < threshold
- 이상: MSE ≥ threshold

→ 결정 기록: [[ADR-015-AI파트-연결-챗봇과-LSTM-Threshold]]
→ 작업 흐름: [[2026-05-18-AI파트-연결]]

### 후속 — LSTM 실시간 INSERT 대기

현재 `ai_predictions` 테이블 비어있음. 이두현 LSTM 백엔드가 실측 시계열 → MSE 계산 → ai_predictions INSERT 하면 챗봇의 `get_predictions` 도구가 실시간 위험도 답변 가능. 그때까지는 stub + threshold 안내로 대체.

## 관련

- [[데이터]]
- [[백엔드]]
- [[ADR-001-LSTM-AutoEncoder-채택]]
- [[ADR-015-AI파트-연결-챗봇과-LSTM-Threshold]] — 챗봇 통합 (2026-05-18)
- [[2026-05-18-이두현-AI모델-쉬운설명]] — 비전문가용 (발표 자료 인용용)
