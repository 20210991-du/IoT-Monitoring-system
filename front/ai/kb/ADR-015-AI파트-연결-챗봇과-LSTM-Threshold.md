---
tags: [ADR, 기술, AI, LSTM, AutoEncoder, threshold, function-calling, 통합]
date: 2026-05-18
status: 완료 (16 도구 / AI 통합 10 + 종합 30 케이스 40/40 PASS)
---

# ADR-015 · AI 파트 연결 — 챗봇이 LSTM Threshold 와 분류 기준을 직접 다룸

## 맥락

[[ADR-014-위치-지도-기반-단말검색]] 까지 15 도구. AI 영역은 `get_predictions` 가 ai_predictions 빈 stub 만 반환. 챗봇이 "TB24-XXX 정상 한계는?" 류 질문에 답 못함. 또 시스템 프롬프트의 위험도 임계 (MSE 0.85, 0.28~0.85) 가 옛 mock 데이터 기준이라 부정확.

이두현 5/18 설명 ([[2026-05-18-이두현-AI모델-쉬운설명]]):
- 모델: **LSTM AutoEncoder** (정상 패턴 복원 오차로 이상 탐지)
- 학습: 시퀀스 24, epoch 50, base_features 4 (방식전위/AC유입/온도/습도) + 파생 8 = 12 컬럼
- threshold = 학습 데이터 MSE 의 **99 percentile** (단말별로 다름)
- 분류: 정상 < 70% < 관찰 < 100% < 이상 (threshold 비율)

자원 (ai/config/):
- `device_thresholds.json` — 55 단말 threshold (0.0001 ~ 0.047, 평균 0.00194)
- `model_config.json` — time_steps, base_features, sacrificial_devices 등
- `eval_metrics.json` — 학습 시점 평가 통계

## 결정

**ai/config/*.json 3 개를 server.js 부팅 시 메모리 로드 + 도구·시스템 프롬프트에 통합.**

### 1) AI config 로드 (server.js 부팅 시 1회)

```js
import { readFileSync, existsSync } from "fs";
DEVICE_THRESHOLDS = load("device_thresholds.json", {});
MODEL_CONFIG      = load("model_config.json", null);
EVAL_METRICS      = load("eval_metrics.json", null);
console.log(`▶ AI cfg  thresholds=${count}대 · model_config=OK · eval_metrics=OK`);
```

graceful fallback: 파일 없어도 서버 부팅 성공.

### 2) classifyMse 헬퍼

```js
function classifyMse(deviceId, mse) {
  const th = DEVICE_THRESHOLDS[deviceId];
  const ratio = mse / th;
  const level = ratio > 1.0 ? "이상"
              : ratio >= 0.7 ? "관찰"
              : "정상";
  return { deviceId, threshold: th, threshold70: th*0.7, mse, ratio, ratioPercent, level };
}
```

### 3) 신규 도구 — get_ai_model_info(deviceId?)

- deviceId 있으면: 그 단말 threshold + threshold70 + threshold100 + isSacrificial + note + modelConfig 요약
- 없으면: 전체 모델 메타 (model_config + eval_metrics + threshold 통계 + classification 기준 + training 요약)

CACHEABLE_TOOLS 에 포함.

### 4) 기존 도구 강화

- **get_device_detail** — 응답에 `ai: { threshold, threshold70, threshold100, isSacrificial }` 동봉 (학습 단말만)
- **get_predictions** — ai_predictions 비어도 device_thresholds 로 fallback. deviceId 주면 threshold + 분류 기준 안내.
- 실데이터 있을 때 자동 classifyMse() → level 포함

### 5) 시스템 프롬프트 정확화

기존 (옛 mock 기준, 부정확):
```
- **MSE 임계**: 0.85 이상 = 위험, 0.28~0.85 = 이상 의심
```

신규 (이두현 명세):
```
# AI 모델 (LSTM AutoEncoder) — 위험도 판정 정확 명세
- threshold = 단말별, 학습 99 percentile MSE
- 정상: MSE < threshold × 0.70
- 관찰: 0.70 ≤ MSE/threshold ≤ 1.00
- 이상: MSE > threshold × 1.00
- 답변 시 "현재 MSE 가 threshold 의 N% 도달" 비율 답변 권장

* 중요 구분: 측정 센서 8 종 ≠ AI 학습 입력 피처.
  학습 피처/시퀀스/epoch 모르면 절대 추측 금지, get_ai_model_info 호출 필수.
```

## 검증 — 10 AI 케이스 + 30 종합 = **40/40 PASS** (5/18)

### AI 통합 (10/10 PASS) — test-chat-ai-integration.sh

| 케이스 | 호출 도구 | 검증 포인트 |
|---|---|---|
| A01 AI 모델 학습 방식 | get_ai_model_info | LSTM AE / 24 step / 50 epoch / MSE 정확 |
| A02 TB24-250401 threshold | get_ai_model_info | 0.001066 / 70% = 0.000746 |
| A03 TB24-250455 threshold | get_ai_model_info | 0.000259 / 70% = 0.000181 |
| A04 희생전류 단말 | list_devices | TB24-250406, TB24-250407 |
| A05 AI 학습 피처 | get_ai_model_info | base_features 4 (방식전위/AC유입/온도/습도) — 환각 방지 |
| A06 위험도 분류 기준 | get_ai_model_info | 70%/100% 비율 명시 |
| A07 단말 AI 정상 한계 + 현재 | get_ai_model_info + get_device_detail | threshold + 70% + 현재 상태 |
| A08 LSTM 예측 결과 | get_predictions | stub fallback + threshold 안내 |
| A09 평균 threshold | get_ai_model_info | 0.001949 (55대 평균) |
| A10 비율 답변 | get_ai_model_info | 0.046685 → 70% = 0.03268 |

### 종합 30 케이스 (30/30 PASS) — 회귀 안 깨짐 확인

## 발견된 버그 (자동 수정 1건)

**A05 환각** — LLM 이 도구 호출 없이 "8 센서를 AI 입력 피처" 라고 답변.
- 원인: 시스템 프롬프트 도메인 지식 섹션에 8 센서 명시 → LLM 자체 추론
- 수정: AI 모델 섹션에 "측정 센서 8 종 ≠ AI 학습 입력 피처" 명시 + "추측 금지, 도구 호출 필수"
- 효과: 재테스트 A05 PASS — `base_features 4 (방식전위, AC유입, 온도, 습도)` 정확 답변

## 트레이드오프

**장점:**
- 챗봇이 "TB24-250401 의 정상 한계는 MSE 0.001066" 같이 정확한 단말 분석 답변 가능
- LSTM 백엔드(이두현) INSERT 대기 중에도 threshold 정보로 충분히 답변
- 옛 mock 임계 (MSE 0.85) 의 false 정보 제거
- ai/config 변경 시 서버 재시작만으로 자동 반영

**단점:**
- threshold 는 학습 시점 스냅샷 — 모델 재학습마다 갱신 필요 (서버 재부팅 또는 reload endpoint)
- 실시간 MSE 는 여전히 ai_predictions INSERT 대기 → 챗봇이 "현재 N% 도달" 답변하려면 LSTM 백엔드 필요

## 후속

- [ ] 이두현 LSTM 백엔드 → ai_predictions 자동 INSERT (실시간 MSE)
- [ ] /api/admin/reload-ai-config endpoint — 재학습 후 서버 재시작 없이 갱신
- [ ] device_thresholds.json 변경 감지 (fs.watch) → 자동 reload
- [ ] 챗봇 응답에 도구 호출 흔적 가시화 — 이미 toolCalls 칩 있음, AI 도구는 더 강조 가능

## 관련

- [[ADR-014-위치-지도-기반-단말검색]] — 15 도구 (이 ADR 의 직전)
- [[ADR-001-LSTM-AutoEncoder-채택]] — 모델 선택
- [[ADR-002-희생전류-분리처리]] — 희생전류 단말 2 대 분리
- [[2026-05-18-이두현-AI모델-쉬운설명]] — 본 ADR 의 출처
- [[정보/서버 정보/맥스튜디오/서비스/AI모델]] — 분야 living doc
- 커밋: `225ab7d` `e317ac3`
- 코드: `front/server.js` (DEVICE_THRESHOLDS, MODEL_CONFIG, classifyMse, get_ai_model_info)
- 테스트: `front/test/test-chat-ai-integration.sh`
