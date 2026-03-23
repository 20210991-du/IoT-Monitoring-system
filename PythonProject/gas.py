import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from sklearn.preprocessing import MinMaxScaler


# 1. 데이터 로드 및 Pandas 전처리 (Pandas 활용) [cite: 32, 38]
def preprocess_sensor_data(csv_path):
    # 1. 데이터 로드
    df = pd.read_csv(csv_path)

    # 2. 시간 데이터 변환 및 정렬
    # '최근 갱신일시' 컬럼을 기준으로 정렬합니다.
    df['최근 갱신일시'] = pd.to_datetime(df['최근 갱신일시'])
    df = df.sort_values('최근 갱신일시')

    # 3. 단위 제거 및 숫자형 변환 (정밀 분석 대상)
    cols_to_fix = ['방식전위', '방식전류', 'AC유입', '배터리', '온도', '습도']

    for col in cols_to_fix:
        # 문자열 내 ','와 단위(mV, mA, ℃, %) 제거 후 float 변환
        df[col] = df[col].astype(str).str.replace(r'[^0-9.-]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4. 결측치 선형 보간 (질문하신 핵심 구간)
    # 숫자로 변환된 후 수행해야 올바르게 직선 보간이 됩니다.
    df[cols_to_fix] = df[cols_to_fix].interpolate(method='linear').ffill().bfill()

    # 5. 주요 분석 대상 컬럼 선택
    features = ['방식전위', '방식전류', 'AC유입', '배터리', '온도', '습도']
    data = df[features].values

    # 6. 데이터 정규화
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    print("전처리 및 보간 완료 데이터 샘플 (첫 3행):\n", scaled_data[:3])
    return scaled_data, scaler, df['최근 갱신일시'].values  # Sliding Window 시작점 이후 시간




# 2. LSTM 입력을 위한 Sliding Window 생성 (안정성 강화)
def create_sequences(data, time_steps=100):  # 데이터가 적을 수 있으므로 기본값을 30으로 조정
    if len(data) <= time_steps:
        raise ValueError(f"데이터 총 길이({len(data)})가 설정된 time_steps({time_steps})보다 짧습니다. 데이터를 늘리거나 단계를 줄여주세요.")

    xs = []
    for i in range(len(data) - time_steps):
        xs.append(data[i:(i + time_steps)])
    return np.array(xs)


# 3. LSTM AutoEncoder 모델 구축 (유연한 구조)
def build_model(n_timesteps, n_features):
    model = Sequential([
        # Encoder: 정보를 단계적으로 압축
        LSTM(128, activation='relu', input_shape=(n_timesteps, n_features), return_sequences=True),
        LSTM(64, activation='relu', return_sequences=False),
        RepeatVector(n_timesteps),
        # Decoder: 압축된 정보를 다시 복원
        LSTM(64, activation='relu', return_sequences=True),
        LSTM(128, activation='relu', return_sequences=True),
        TimeDistributed(Dense(n_features))
    ])
    # 학습률을 낮춰서(0.001 -> 0.0005) 더 세밀하게 최적화
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='mse')
    return model


# --- 메인 실행 루틴 (수정본) ---

# 1. 파일 경로 설정 및 데이터 전처리 호출
csv_file = 'sensor_logs_sample.csv'  # 실제 생성한 파일명

# 함수를 호출하여 리턴값을 변수들에 할당합니다.
# 여기서 scaled_data가 정의됩니다.
scaled_data, scaler, timestamps = preprocess_sensor_data(csv_file)

# 2. 전처리된 데이터를 기반으로 변수 설정
n_features = scaled_data.shape[1]
time_steps = 20  # 샘플 데이터 양에 맞춰 조절

# ★ 추가된 핵심 코드: X_train 데이터가 잘려나간 만큼(time_steps), 시간 데이터도 앞부분을 잘라내어 길이를 똑같이 맞춥니다!
timestamps = timestamps[time_steps:]

try:
    # 3. Sliding Window 데이터 생성
    X_train = create_sequences(scaled_data, time_steps)
    print(f"학습 데이터 형태: {X_train.shape}")

    # 4. 모델 생성 및 학습
    model = build_model(X_train.shape[1], X_train.shape[2])

    history = model.fit(
        X_train, X_train,
        epochs=100,
        batch_size=8,
        validation_split=0.2 if len(X_train) > 20 else 0.0,
        verbose=1
    )


    # 5. 이상 탐지 및 결과 분석
    predictions = model.predict(X_train)
    mse = np.mean(np.power(X_train - predictions, 2), axis=(1, 2))

    threshold = np.mean(mse) + (np.std(mse) * 3)
    anomalies = mse > threshold

    print(f"\n[분석 결과] 임계값: {threshold:.6f}")
    print(f"탐지된 이상 구간 수: {np.sum(anomalies)}개")

except Exception as e:
    print(f"오류 발생: {e}")

# --- 5. 지능형 알림 및 상세 분석 로직 ---
print("\n" + "=" * 50)
print("🚨 [실시간 이상 탐지 모니터링 시스템 시작] 🚨")
print("=" * 50)

# 전처리 과정에서 썼던 컬럼명 순서 그대로 가져오기
feature_names = ['방식전위', '방식전류', 'AC유입', '배터리', '온도', '습도']

# 피처별 세부 오차 계산 (어떤 센서가 원인인지 찾기 위함)
# X_train 차원: (샘플수, 타임스텝, 피처수) -> axis=1(타임스텝)을 기준으로 평균을 내면 각 샘플의 '피처별 오차'가 나옴
feature_mse = np.mean(np.power(X_train - predictions, 2), axis=1)

anomaly_count = 0

for i, is_anomaly in enumerate(anomalies):
    if is_anomaly:
        anomaly_count += 1

        # 1. 이상 발생 시간 추출
        # timestamps는 전처리 함수(preprocess_sensor_data)에서 반환받은 시간 배열입니다.
        fault_time = timestamps[i]

        # 2. 가장 큰 오차를 발생시킨 주원인 센서 찾기
        max_error_idx = np.argmax(feature_mse[i])
        cause_sensor = feature_names[max_error_idx]

        # 3. 알림 출력 (실제로는 여기서 Slack, SMS, Email API를 호출합니다)
        print(f"[경보 {anomaly_count}호] ⚠️ 발생 시각: {fault_time}")
        print(f"  ▶ 종합 위험도: 임계치({threshold:.6f}) 초과 -> 현재 수치({mse[i]:.6f})")
        print(f"  ▶ 🎯 주요 의심 원인: '{cause_sensor}' 데이터 급변 (현장 점검 요망)")
        print("-" * 50)

        # (예시) 슬랙 알림 발송 함수 호출부
        # send_slack_alert(time=fault_time, sensor=cause_sensor, severity="HIGH")

if anomaly_count == 0:
    print("✅ 모든 구간 정상. 시스템이 안정적으로 운영 중입니다.")
else:
    print(f"\n총 {anomaly_count}건의 이상 징후가 감지되었습니다. 위 내역을 확인하세요.")

