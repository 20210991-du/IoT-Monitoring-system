"""
gas_common_model_predict2.py
===========================
[실행용] 학습된 LSTM AutoEncoder 모델로 이상 탐지를 수행하는 스크립트

사전 조건:
    gas_unified_model_train2.py 실행 후 common_model_artifacts/ 폴더가 생성되어 있어야 합니다.

실행 방법:
    python gas_common_model_predict2.py

주요 함수:
    - load_artifacts()              : 저장된 모델/스케일러/threshold 로드
    - predict_device_window()       : 단일 장비 윈도우 데이터로 이상 탐지 수행
    - run_batch_prediction()        : 전체 장비 일괄 이상 탐지 및 결과 CSV 저장
    - classify_comm_fault_level()   : 통신 상태 후처리 분류 (정상/일시장애/고장)

[변경 이력]
    - 통신품질: AI 입력에서 제외 → 룰 기반 필터(-115 dBm)로 통신단절/고장 플래그 생성
               이상 탐지 결과에 통신 상태 컨텍스트를 후처리로 추가
"""

import os
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model

from db_connector import read_sql, SENSOR_NAME_MAP

# ============================================================
# 0. 설정  (train 파일과 동일하게 유지)
# ============================================================

# ── 공통 AI 모델 입력 변수 ──────────────────────────────────
BASE_FEATURES = ['방식전위', 'AC유입', '온도', '습도']

# ── 통신품질 룰 기반 필터 설정 ──────────────────────────────
COMM_QUALITY_COL           = '통신품질'
COMM_QUALITY_THRESHOLD_DBM = -115   # dBm 이하 → 통신 불가
COMM_OUTAGE_CONSECUTIVE    = 3      # 연속 N회 이상 단절 → 고장 (미만 = 일시 장애)

# ── DB 예측 데이터 조회 윈도우 ──────────────────────────────
# 현재 시각 기준 최근 N시간 데이터를 DB에서 로드 (time_steps=24의 3배 이상 권장)
PREDICTION_WINDOW_HOURS = 72

# ── 위험 단계 구분 기준 ──────────────────────────────────────
# MSE 값이 threshold 대비 몇 % 이상이면 '관찰'로 분류할지 결정
OBSERVATION_RATIO = 0.7

# ── 위험 단계 레이블 및 출력용 이모지 ──────────────────────
RISK_LABELS = {
    '이상': '🔴 이상',
    '관찰': '🟡 관찰',
    '정상': '🟢 정상',
}

# ── 통신 상태 레이블 ────────────────────────────────────────
COMM_STATUS_LABELS = {
    '정상통신' : '🟢 정상통신',
    '일시장애' : '🟡 일시장애',
    '통신고장' : '🔴 통신고장',
}


# ============================================================
# 1. 통신품질 룰 기반 필터  (train 파일과 동일한 로직 유지)
# ============================================================

def apply_comm_quality_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    통신품질(dBm) 열을 룰 기반으로 처리하여 두 가지 플래그 컬럼을 추가합니다.
    통신품질 열 자체는 AI 모델 입력(BASE_FEATURES)에 포함되지 않습니다.

    생성 컬럼:
        통신단절_플래그 (int):
            측정값 <= COMM_QUALITY_THRESHOLD_DBM(-115 dBm) 이면 1, 아니면 0.

        통신고장_플래그 (int):
            장비별로 연속 COMM_OUTAGE_CONSECUTIVE(3)회 이상 단절이면 1.
            연속 횟수가 기준 미만이면 0 (일시 장애).

    후처리 활용 가이드:
        통신단절_플래그=1, 통신고장_플래그=0 → 일시 장애 (AI 결과 참고 수준)
        통신단절_플래그=1, 통신고장_플래그=1 → 통신 고장 (AI 신뢰도 낮음, 별도 알림)
        통신단절_플래그=0                    → 정상 통신 (AI 입력 신뢰 가능)
    """
    df = df.copy()

    if COMM_QUALITY_COL not in df.columns:
        df['통신단절_플래그'] = 0
        df['통신고장_플래그'] = 0
        return df

    # 1. 기본 단절 플래그
    comm_vals = pd.to_numeric(df[COMM_QUALITY_COL], errors='coerce')
    df['통신단절_플래그'] = (
        comm_vals.le(COMM_QUALITY_THRESHOLD_DBM) | comm_vals.isna()
    ).astype(int)

    # 2. 장비별 연속 단절 → 고장 판정
    def _device_fault_flag(sub: pd.DataFrame) -> pd.Series:
        sub_sorted = sub.sort_values('측정시각')
        flag       = sub_sorted['통신단절_플래그']
        changed    = (flag != flag.shift(fill_value=0)).cumsum()
        consec_len = flag.groupby(changed).transform('count')
        return ((flag == 1) & (consec_len >= COMM_OUTAGE_CONSECUTIVE)).astype(int)

    _fault = (
        df.groupby('장비번호', group_keys=False)
        .apply(_device_fault_flag)
    )

    # pandas 버전에 따라 DataFrame으로 반환될 수 있으므로 명시적으로 Series로 변환
    if isinstance(_fault, pd.DataFrame):
        _fault = _fault.iloc[:, 0]

    df['통신고장_플래그'] = _fault
    return df


def classify_comm_fault_level(df: pd.DataFrame) -> pd.Series:
    """
    통신 상태를 3단계로 분류합니다.

    판정 기준:
        통신고장_플래그=1              → '통신고장'  (연속 단절, AI 신뢰도 낮음)
        통신단절_플래그=1, 고장=0      → '일시장애'  (단발성 단절, AI 결과 참고 수준)
        통신단절_플래그=0              → '정상통신'

    Returns:
        pd.Series: '정상통신' | '일시장애' | '통신고장'
    """
    if '통신고장_플래그' not in df.columns or '통신단절_플래그' not in df.columns:
        return pd.Series('정상통신', index=df.index)

    status = pd.Series('정상통신', index=df.index)
    status[df['통신단절_플래그'] == 1] = '일시장애'
    status[df['통신고장_플래그'] == 1] = '통신고장'
    return status


# ============================================================
# 2. DB 예측 데이터 로드
# ============================================================

def load_prediction_data_from_db(
    window_hours: int = PREDICTION_WINDOW_HOURS,
    device_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    DB에서 최근 window_hours 시간의 센서 데이터를 로드하여 예측용 DataFrame 반환.

    반환 컬럼: 측정시각, 장비번호, 방식전위, AC유입, 온도, 습도, 통신품질,
               형식, 시설번호, 통신단절_플래그, 통신고장_플래그
      - DB 수신감도 → 통신품질 (SENSOR_NAME_MAP 적용)
    """
    end_dt    = datetime.now()
    start_dt  = end_dt - timedelta(hours=window_hours)
    start_str = start_dt.strftime('%Y-%m-%d %H:%M:%S')

    conditions = ["t.SERIAL_NUM IS NOT NULL", "t.SERIAL_NUM != ''", "sd.WRITE_DATE >= %s"]
    params: List = [start_str]
    if device_ids:
        placeholders = ', '.join(['%s'] * len(device_ids))
        conditions.append(f"t.SERIAL_NUM IN ({placeholders})")
        params.extend(device_ids)

    where = ' AND '.join(conditions)
    sql_sensor = f"""
    SELECT
        sd.WRITE_DATE  AS 측정시각,
        t.SERIAL_NUM   AS 장비번호,
        si.NAME        AS 센서명,
        sd.VALUE       AS 값
    FROM kscg_sensor_data sd
    JOIN kscg_sensor_info      si ON sd.SENSOR_ID      = si.SENSOR_ID
    JOIN kscg_transmitter_info t  ON si.TRANSMITTER_ID = t.TRANSMITTER_ID
    WHERE {where}
    ORDER BY t.SERIAL_NUM, sd.WRITE_DATE
    """
    long_df = read_sql(sql_sensor, params=params)
    long_df['센서명'] = long_df['센서명'].map(SENSOR_NAME_MAP)
    long_df = long_df.dropna(subset=['센서명'])

    wide_df = long_df.pivot_table(
        index=['측정시각', '장비번호'],
        columns='센서명',
        values='값',
        aggfunc='first',
    ).reset_index()
    wide_df.columns.name = None
    wide_df['측정시각'] = pd.to_datetime(wide_df['측정시각'])

    sql_info = """
    SELECT
        t.SERIAL_NUM  AS 장비번호,
        t.TYPE        AS 형식,
        f.NUMBER      AS 시설번호
    FROM kscg_transmitter_info t
    LEFT JOIN kscg_facility_info f ON t.TRANSMITTER_ID = f.TRANSMITTER_ID
    WHERE t.SERIAL_NUM IS NOT NULL AND t.SERIAL_NUM != ''
    """
    info_df = read_sql(sql_info)
    info_df['장비번호'] = info_df['장비번호'].astype(str).str.strip()
    info_df = (info_df[info_df['장비번호'].ne('')]
               .sort_values('시설번호', na_position='last')
               .drop_duplicates(subset=['장비번호']))

    df = wide_df.merge(info_df, on='장비번호', how='left')
    df = apply_comm_quality_filter(df)
    return df.sort_values(['장비번호', '측정시각']).reset_index(drop=True)


# ============================================================
# 3. 전처리 / 피처 엔지니어링  (train 파일과 동일한 로직 유지)
# ============================================================

def assign_normalization_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    그룹별 정규화 기준:
    1순위 형식
    2순위 시설번호 앞자리
    3순위 ALL
    """
    df = df.copy()
    if '형식' in df.columns:
        group = df['형식'].astype(str).str.strip()
        group = group.replace({'': np.nan, 'nan': np.nan, 'None': np.nan})
        df['정규화그룹'] = group
    else:
        df['정규화그룹'] = np.nan

    if '시설번호' in df.columns:
        fallback = df['시설번호'].astype(str).str.extract(r'^([^-]+)')[0]
        df['정규화그룹'] = df['정규화그룹'].fillna('FAC_' + fallback.fillna('UNK'))
    else:
        df['정규화그룹'] = df['정규화그룹'].fillna('ALL')

    df['정규화그룹'] = df['정규화그룹'].fillna('ALL')
    return df


def add_engineered_features(
    df: pd.DataFrame,
    base_features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    base_features(기본값: BASE_FEATURES)에 대해 결측 보간 및 파생 피처를 생성합니다.
    통신품질은 BASE_FEATURES에서 제외되어 이 함수의 처리 대상이 아닙니다.
    """
    if base_features is None:
        base_features = BASE_FEATURES

    df = df.copy().sort_values(['장비번호', '측정시각']).reset_index(drop=True)

    for col in base_features:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = (
            df.groupby('장비번호')[col]
            .transform(lambda s: s.interpolate(method='linear', limit_direction='both').ffill().bfill())
        )

    for col in base_features:
        df[f'{col}_diff1'] = df.groupby('장비번호')[col].diff().fillna(0)
        df[f'{col}_ma24']  = df.groupby('장비번호')[col].transform(
            lambda s: s.rolling(window=24, min_periods=1).mean()
        )
        df[f'{col}_dev24'] = df[col] - df[f'{col}_ma24']

    return df


def feature_columns(base_features: Optional[List[str]] = None) -> List[str]:
    if base_features is None:
        base_features = BASE_FEATURES
    cols = []
    for col in base_features:
        cols.extend([col, f'{col}_diff1', f'{col}_dev24'])
    return cols


# ============================================================
# 4. 시퀀스 데이터  (train 파일과 동일한 로직 유지)
# ============================================================

def create_sequences(values: np.ndarray, time_steps: int) -> np.ndarray:
    if len(values) <= time_steps:
        raise ValueError(f'데이터 길이({len(values)})가 time_steps({time_steps})보다 짧습니다.')
    
    seqs = []
    for i in range(len(values) - time_steps):
        seq = values[i:i + time_steps]
        # NaN이 포함된 시퀀스는 예측에서 배제 (불연속 구간/통신장애 처리)
        if not np.isnan(seq).any():
            seqs.append(seq)
            
    if not seqs:
        raise ValueError('유효한 시퀀스가 없습니다. (통신 장애 또는 결측 데이터)')
        
    return np.array(seqs, dtype=np.float32)


# ============================================================
# 5. 위험 단계 분류
# ============================================================

def classify_risk_level(
    mse: float,
    threshold: float,
    observation_ratio: float = OBSERVATION_RATIO,
) -> str:
    """
    MSE와 threshold를 비교해 3단계 위험 등급을 반환합니다.

    단계 기준 (observation_ratio = 0.7 기본값 기준):
        이상  : mse >  threshold                       (threshold 100% 초과)
        관찰  : threshold * 0.7 <= mse <= threshold    (threshold 70%~100% 구간)
        정상  : mse <  threshold * 0.7                 (threshold 70% 미만)
    """
    if mse > threshold:
        return '이상'
    elif mse >= threshold * observation_ratio:
        return '관찰'
    else:
        return '정상'


# ============================================================
# 6. 아티팩트 로드
# ============================================================

def load_artifacts(save_dir: str):
    """
    학습 시 저장된 모델, 스케일러, threshold, 설정을 로드합니다.

    Returns:
        model       : 학습된 LSTM AutoEncoder
        scaler_map  : 그룹별 MinMaxScaler 딕셔너리
        thresholds  : 장비별 MSE threshold 딕셔너리
        config      : 학습 시 사용된 설정 (time_steps, feature_columns, 통신품질 필터 설정 등)
    """
    model_path = os.path.join(save_dir, 'common_lstm_autoencoder.keras')
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f'모델 파일을 찾지 못했습니다: {model_path}\n'
            'gas_unified_model_train.py를 먼저 실행하세요.'
        )

    model = load_model(model_path)

    with open(os.path.join(save_dir, 'group_scalers.pkl'), 'rb') as f:
        scaler_map = pickle.load(f)

    with open(os.path.join(save_dir, 'device_thresholds.json'), 'r', encoding='utf-8') as f:
        thresholds = json.load(f)

    with open(os.path.join(save_dir, 'model_config.json'), 'r', encoding='utf-8') as f:
        config = json.load(f)

    return model, scaler_map, thresholds, config


# ============================================================
# 7. 단일 장비 이상 탐지 (실시간 API 연동용)
# ============================================================

def predict_device_window(
    device_window_df: pd.DataFrame,
    model: Sequential,
    scaler_map: Dict[str, MinMaxScaler],
    thresholds: Dict[str, float],
    feature_cols: List[str],
    time_steps: int,
    config: Dict,
    observation_ratio: float = OBSERVATION_RATIO,
) -> Dict[str, object]:
    """
    실시간 API 연결용 단일 장비 이상 탐지 함수.
    """
    if device_window_df.empty:
        raise ValueError('입력 데이터가 비어 있습니다.')

    ddf = device_window_df.copy()
    if '장비번호' not in ddf.columns or '측정시각' not in ddf.columns:
        raise ValueError("입력 데이터에는 '장비번호', '측정시각' 컬럼이 필요합니다.")

    # 통신품질 필터 적용
    if COMM_QUALITY_COL in ddf.columns:
        ddf = apply_comm_quality_filter(ddf)

    ddf = assign_normalization_group(ddf)
    ddf = add_engineered_features(ddf, BASE_FEATURES)
    ddf = ddf.sort_values('측정시각')

    device_id  = str(ddf['장비번호'].iloc[0])
    group_name = str(ddf['정규화그룹'].iloc[0])

    if group_name not in scaler_map:
        raise ValueError(f'정규화 그룹 "{group_name}" 의 scaler가 없습니다.')

    scaler   = scaler_map[group_name]
    scaled   = scaler.transform(ddf[feature_cols].astype(float).values)
    
    try:
        X = create_sequences(scaled, time_steps)
    except ValueError as e:
        # 시퀀스 생성 실패 시 (통신 장애 등)
        comm_status = classify_comm_fault_level(ddf.iloc[[-1]]).iloc[0] if COMM_QUALITY_COL in ddf.columns else '데이터공백'
        return {
            'device_id'            : device_id,
            'risk_level'           : '정상',
            'comm_status'          : comm_status,
            'ai_reliability'       : '신뢰불가',
            'mse'                  : 0.0,
            'threshold'            : 0.0,
            'error_msg'            : str(e)
        }

    last_seq = X[-1:]
    pred     = model.predict(last_seq, verbose=0)
    mse      = float(np.mean(np.power(last_seq - pred, 2)))

    # ── 임계값 결정 로직 ──────────────────────────────────────
    # 1순위: 튜닝된 최적 글로벌 임계값, 2순위: 장비별 임계값
    best_tuned_thr = config.get('best_tuned_threshold')
    device_thr     = thresholds.get(device_id)
    threshold      = float(best_tuned_thr if best_tuned_thr else (device_thr if device_thr else np.mean(list(thresholds.values()))))

    risk_level = classify_risk_level(mse, threshold, observation_ratio)

    # ── 통신 상태 및 신뢰도 후처리 ────────────────────────────
    comm_status = classify_comm_fault_level(ddf.iloc[[-1]]).iloc[0] if '통신단절_플래그' in ddf.columns else '정상통신'
    ai_reliability = '신뢰' if comm_status == '정상통신' else ('주의' if comm_status == '일시장애' else '신뢰불가')

    return {
        'device_id'      : device_id,
        'group'          : group_name,
        'mse'            : mse,
        'threshold'      : threshold,
        'risk_level'     : risk_level,
        'comm_status'    : comm_status,
        'ai_reliability' : ai_reliability,
    }

def run_batch_prediction(
    df: pd.DataFrame,
    model: Sequential,
    scaler_map: Dict[str, MinMaxScaler],
    thresholds: Dict[str, float],
    feature_cols: List[str],
    time_steps: int,
    config: Dict,
    observation_ratio: float = OBSERVATION_RATIO,
) -> pd.DataFrame:
    """
    전체 장비에 대해 이상 탐지를 일괄 수행합니다.
    """
    results    = []
    device_ids = df['장비번호'].unique()

    for device_id in device_ids:
        ddf = df[df['장비번호'] == device_id].copy()
        try:
            res = predict_device_window(
                ddf, model, scaler_map, thresholds, 
                feature_cols, time_steps, config, observation_ratio
            )
            results.append(res)
        except Exception as e:
            print(f'  [ERROR] {device_id}: {e}')
            continue

    return pd.DataFrame(results)


# ============================================================
# 9. 메인 실행 (예측)
# ============================================================

def main():
    base_dir      = Path(__file__).resolve().parent
    artifacts_dir = base_dir / 'common_model_artifacts'
    output_dir    = base_dir / 'prediction_results'

    # --- 아티팩트 로드 ---
    print('[1] 학습 아티팩트 로드 중...')
    model, scaler_map, thresholds, config = load_artifacts(str(artifacts_dir))
    time_steps   = config['time_steps']
    feature_cols = config['feature_columns']
    print(f'  - time_steps  : {time_steps}')
    print(f'  - 피처 수     : {len(feature_cols)}개  (BASE_FEATURES {len(BASE_FEATURES)}개 × 3)')
    print(f'  - 등록 장비 수: {len(thresholds)}')
    print(f'  - 통신품질 필터: <= {config.get("comm_quality_threshold_dbm", COMM_QUALITY_THRESHOLD_DBM)} dBm, '
          f'연속 {config.get("comm_outage_consecutive_threshold", COMM_OUTAGE_CONSECUTIVE)}회 → 고장')

    # --- 예측 대상 데이터 로드 (DB 최근 데이터) ---
    print(f'\n[2] DB에서 최근 {PREDICTION_WINDOW_HOURS}시간 데이터 로드 중...')
    target_df = load_prediction_data_from_db()
    print(f'  - 전체 행 수: {len(target_df):,}')
    print(f'  - 장비 수   : {target_df["장비번호"].nunique():,}')

    # 통신 상태 사전 요약
    if '통신단절_플래그' in target_df.columns:
        outage_cnt = int(target_df['통신단절_플래그'].sum())
        fault_cnt  = int(target_df['통신고장_플래그'].sum())
        total      = len(target_df)
        print(f'  - 통신단절 이벤트: {outage_cnt:,}건 ({outage_cnt/total:.1%}) '
              f'[고장 {fault_cnt:,}건 / 일시장애 {outage_cnt-fault_cnt:,}건]')

    # --- 일괄 이상 탐지 ---
    print('\n[3] 전체 장비 이상 탐지 실행 중...')
    result_df = run_batch_prediction(
        df=target_df,
        model=model,
        scaler_map=scaler_map,
        thresholds=thresholds,
        feature_cols=feature_cols,
        time_steps=time_steps,
        config=config,
    )

    # --- 결과 저장 ---
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, 'anomaly_results.csv')
    result_df.to_csv(result_path, index=False, encoding='utf-8-sig')
    print(f'\n[4] 결과 저장 완료: {result_path}')

    # --- 위험 단계별 요약 출력 ---
    print('\n===== 위험 단계별 요약 =====')
    if not result_df.empty:
        for level in ['이상', '관찰', '정상']:
            subset = result_df[result_df['risk_level'] == level]
            label  = RISK_LABELS[level]
            print(f'\n{label} ({len(subset)}개)')
            if not subset.empty:
                display_cols = ['device_id', 'group', 'mse', 'threshold',
                                'comm_status', 'ai_reliability']
                print(subset[display_cols].to_string(index=False))

    # --- 통신 상태별 요약 출력 ---
    print('\n===== 통신 상태별 요약 =====')
    if not result_df.empty:
        for status in ['통신고장', '일시장애', '정상통신']:
            subset = result_df[result_df['comm_status'] == status]
            label  = COMM_STATUS_LABELS[status]
            print(f'\n{label} ({len(subset)}개)')
            if not subset.empty and status != '정상통신':
                print(subset[['device_id', 'risk_level', 'ai_reliability']].to_string(index=False))

    # --- 단일 장비 실시간 예측 예시 ---
    print('\n[예시] 단일 장비 실시간 예측 (predict_device_window)')
    sample_device = list(thresholds.keys())[0]
    sample_df     = target_df[target_df['장비번호'] == sample_device].copy()

    if len(sample_df) >= time_steps + 1:
        result = predict_device_window(
            device_window_df=sample_df,
            model=model,
            scaler_map=scaler_map,
            thresholds=thresholds,
            feature_cols=feature_cols,
            time_steps=time_steps,
            config=config,
        )
        print(f'  장비번호          : {result["device_id"]}')
        print(f'  MSE               : {result["mse"]:.6f}')
        print(f'  Threshold         : {result["threshold"]:.6f}')
        print(f'  위험 단계 (AI)    : {RISK_LABELS[result["risk_level"]]}')
        print(f'  통신 상태         : {COMM_STATUS_LABELS[result["comm_status"]]}')
        print(f'  AI 결과 신뢰도    : {result["ai_reliability"]}')
        print(f'  (관찰 기준: threshold × {OBSERVATION_RATIO} = '
              f'{result["threshold"] * OBSERVATION_RATIO:.6f} 이상)')


if __name__ == '__main__':
    main()
