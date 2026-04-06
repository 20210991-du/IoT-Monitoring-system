import os
import json
import joblib
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from tensorflow.keras.models import load_model

# 전처리 함수 임포트 (gas_common_model_v3.py 파일이 같은 폴더에 있어야 함)
from gas_common_model_v3 import (
    BASE_FEATURES,
    add_engineered_features,
    create_sequences
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = None
scalers = {}
thresholds = {}
config = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scalers, thresholds, config
    print("모델 및 설정 파일을 로드합니다...")

    model_path = os.path.join(BASE_DIR, 'common_lstm_autoencoder.keras')
    scaler_path = os.path.join(BASE_DIR, 'group_scalers.pkl')
    threshold_path = os.path.join(BASE_DIR, 'device_thresholds.json')
    config_path = os.path.join(BASE_DIR, 'model_config.json')

    model = load_model(model_path)
    scalers = joblib.load(scaler_path)

    with open(threshold_path, 'r', encoding='utf-8') as f:
        thresholds = json.load(f)

    # config 파일이 없을 경우를 대비한 기본값 처리
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        # config가 없으면 기본 생성
        config = {
            'time_steps': 24,
            'feature_columns': [f for col in BASE_FEATURES for f in (col, f'{col}_diff1', f'{col}_dev24')]
        }

    print("로딩 완료! 서버가 준비되었습니다.")
    yield


app = FastAPI(title="IoT 가스 모니터링 테스트 웹", lifespan=lifespan)


# --- 스키마 정의 ---
class SensorPoint(BaseModel):
    측정시각: str
    방식전위: float
    AC유입: float
    희생전류: float
    온도: float
    습도: float
    통신품질: float


class WindowSensorData(BaseModel):
    device_id: str
    group_name: str
    window_data: List[SensorPoint]


# ==========================================================
# 1. 테스트 데이터 생성 API (이상 데이터를 연속으로 주입하여 잘림 방지)
# ==========================================================
@app.get("/generate-test-data")
async def generate_test_data():
    device_id = "TB24-250402"
    # scalers가 정의되지 않았을 경우를 대비한 안전한 코드
    group_name = list(scalers.keys())[0] if 'scalers' in globals() and scalers else "ALL"

    # 시퀀스 변환 시 데이터 잘림을 방지하기 위해 넉넉하게 30시간 생성
    base_time = datetime.now() - timedelta(hours=30)
    window_data = []

    dice = random.random()
    if dice < 0.4:
        scenario = '정상_데이터'
    elif dice < 0.7:
        scenario = '미세_변화(관찰)'
    else:
        scenario = random.choice(['방식기준_미달(이상)', 'AC_간섭심화(이상)', '이상_발열(이상)'])

    # 실제 센서처럼 부드러운 시계열을 위해 초기 기준값을 고정
    base_voltage = -2050.0
    base_ac = 950.0
    base_temp = 10.0
    base_hum = 31.7
    base_comm = -86.0

    for i in range(30):
        current_time = base_time + timedelta(hours=i)

        voltage = base_voltage + random.uniform(-5.0, 5.0)
        ac_in = base_ac + random.uniform(-10.0, 10.0)
        current = 0.0
        temp = base_temp + random.uniform(-0.05, 0.05)
        hum = base_hum + random.uniform(-0.05, 0.05)
        comm = base_comm + random.uniform(-0.5, 0.5)

        # 🌟 핵심: 찰나가 아닌 마지막 5시간(25번째~29번째) 동안 연속으로 이상치 주입!
        if i >= 25:
            if scenario == '미세_변화(관찰)':
                # 정상 노이즈보다 아주 살짝만 벗어나도록 수치 조정 (관찰 단계 유도)
                voltage = base_voltage + 1030.0  # 약 -2010.0 (살짝 튐)
                temp = base_temp + 4.35  # 약 13.0 (살짝 오름)
                ac_in = base_ac + 1000.0  # 약 1030.0 (살짝 증가)
            elif scenario == '방식기준_미달(이상)':
                voltage = -400.0  # 확 튀게 더 올림
            elif scenario == 'AC_간섭심화(이상)':
                ac_in = 8000.0  # 확 튀게 더 올림
            elif scenario == '이상_발열(이상)':
                temp = 60.0  # 확 튀게 더 올림

        window_data.append({
            "측정시각": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "방식전위": voltage,
            "AC유입": ac_in,
            "희생전류": current,
            "온도": temp,
            "습도": hum,
            "통신품질": comm
        })

    return {
        "device_id": device_id,
        "group_name": group_name,
        "window_data": window_data,
        "scenario": scenario
    }

    return {
        "device_id": device_id,
        "group_name": group_name,
        "window_data": window_data,
        "scenario": anomaly_scenario  # UI에서 어떤 상황이 연출됐는지 확인용 (선택)
    }


# ==========================================================
# 2. 실제 LSTM 예측 및 원인 도출 API
# ==========================================================
@app.post("/predict")
async def predict_status(data: WindowSensorData):
    try:
        time_steps = config.get('time_steps', 24)
        feature_cols = config.get('feature_columns', [])

        records = [point.model_dump() for point in data.window_data]
        df = pd.DataFrame(records)
        df['장비번호'] = data.device_id
        df['정규화그룹'] = data.group_name
        df['측정시각'] = pd.to_datetime(df['측정시각'])

        df = add_engineered_features(df, BASE_FEATURES)
        df = df.dropna().reset_index(drop=True)

        if len(df) < time_steps:
            raise ValueError("데이터가 부족합니다.")

        scaler = scalers[data.group_name]
        scaled_values = scaler.transform(df[feature_cols].astype(float).values)

        X = create_sequences(scaled_values, time_steps)
        last_seq = X[-1:]

        pred = model.predict(last_seq, verbose=0)

        mse = float(np.mean(np.power(last_seq - pred, 2)))
        feature_mse_array = np.mean(np.power(last_seq - pred, 2), axis=1)[0]

        feature_contributions = {feat: float(err) for feat, err in zip(feature_cols, feature_mse_array)}

        critical_threshold = float(thresholds.get(data.device_id, 0.01))
        warning_threshold = critical_threshold * 0.7

        if mse >= critical_threshold:
            status = "이상"
        elif mse >= warning_threshold:
            status = "관찰"
        else:
            status = "정상"

        main_cause_feature = max(feature_contributions, key=feature_contributions.get)
        base_cause_feature = main_cause_feature.split('_')[0]

        return {
            "device_id": data.device_id,
            "status": status,
            "reconstruction_error": round(mse, 6),
            "thresholds": {"warning": round(warning_threshold, 6), "critical": round(critical_threshold, 6)},
            "main_cause": base_cause_feature,
            "details": feature_contributions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================================
# 3. 웹 UI 페이지 제공 (HTML + JS)
# ==========================================================
@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <title>가스 모니터링 AI 테스트</title>
        <style>
            body { font-family: 'Malgun Gothic', sans-serif; padding: 20px; line-height: 1.6; }
            button { padding: 10px 20px; font-size: 16px; cursor: pointer; background-color: #007bff; color: white; border: none; border-radius: 5px; }
            button:disabled { background-color: #cccccc; cursor: not-allowed; }
            .status-정상 { color: green; font-weight: bold; }
            .status-관찰 { color: orange; font-weight: bold; }
            .status-이상 { color: red; font-weight: bold; }
            #logArea { background-color: #f8f9fa; padding: 15px; border: 1px solid #ddd; border-radius: 5px; white-space: pre-wrap; font-family: monospace; font-size: 14px; min-height: 100px; }
            .result-box { border: 1px solid #ddd; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        </style>
    </head>
    <body>
        <h2>IoT 가스 모니터링 AI 모델 테스트</h2>

        <button id="testBtn" onclick="runTest()">데이터 생성 및 탐지 실행</button>

        <div class="result-box" style="margin-top: 20px;">
            <h3>📊 분석 결과 요약</h3>
            <ul>
                <li><strong>대상 장비:</strong> <span id="resDevice">-</span></li>
                <li><strong>판단 상태:</strong> <span id="resStatus">-</span></li>
                <li><strong>주요 원인:</strong> <span id="resCause">-</span></li>
                <li><strong>현재 오차(MSE):</strong> <span id="resMse">-</span></li>
                <li><strong>이상 임계치(Critical):</strong> <span id="resThresh">-</span></li>
            </ul>
        </div>

        <h3>📝 전체 실행 로그 & 상세 결과</h3>
        <pre id="logArea">결과가 여기에 표시됩니다...</pre>

        <script>
            async function runTest() {
                const btn = document.getElementById('testBtn');
                const logArea = document.getElementById('logArea');

                // 실행 중 버튼 비활성화 및 초기 텍스트 셋팅
                btn.innerText = "분석 중...";
                btn.disabled = true;
                logArea.innerText = "데이터를 생성하고 있습니다...";

                try {
                    // 1. 데이터 생성 호출 (캐싱 방지)
                    const dataRes = await fetch('/generate-test-data?t=' + new Date().getTime());
                    const payload = await dataRes.json();

                    // 2. 생성된 시나리오 화면에 우선 표시
                    logArea.innerText = `[데이터 생성 시나리오: ${payload.scenario}]\\n...AI 모델 분석 진행 중...`;

                    // 3. 모델에 예측 요청
                    const predictRes = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                    const result = await predictRes.json();

                    // 4. 요약 결과 UI 업데이트
                    document.getElementById('resDevice').innerText = result.device_id;

                    const statusSpan = document.getElementById('resStatus');
                    statusSpan.innerText = result.status;
                    statusSpan.className = 'status-' + result.status;

                    document.getElementById('resCause').innerText = result.status === '정상' ? '원인 없음(정상)' : result.main_cause;
                    document.getElementById('resMse').innerText = result.reconstruction_error;
                    document.getElementById('resThresh').innerText = result.thresholds.critical;

                    // 5. 로그 창 덮어쓰기 방지: 기존 시나리오 텍스트 뒤에 JSON 결과를 붙여줌!
                    logArea.innerText = `[데이터 생성 시나리오: ${payload.scenario}]\\n\\n[상세 JSON 결과]\\n` + JSON.stringify(result, null, 2);

                } catch (error) {
                    logArea.innerText = '에러가 발생했습니다:\\n' + error.message;
                    alert('에러: ' + error.message);
                } finally {
                    // 버튼 원상복구
                    btn.innerText = "데이터 생성 및 탐지 실행";
                    btn.disabled = false;
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)