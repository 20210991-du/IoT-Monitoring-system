"""
gas_common_model_predict.py
===========================
[실행용] 학습된 LSTM AutoEncoder 모델로 이상 탐지를 수행하는 스크립트

사전 조건:
    gas_common_model_train.py 실행 후 common_model_artifacts/ 폴더가 생성되어 있어야 합니다.

실행 방법:
    python gas_common_model_predict.py

주요 함수:
    - load_artifacts()              : 저장된 모델/스케일러/threshold 로드
    - predict_device_window()       : 단일 장비 윈도우 데이터로 이상 탐지 수행
    - run_batch_prediction()        : 전체 장비 일괄 이상 탐지 및 결과 CSV 저장
    - classify_comm_fault_level()   : 통신 상태 후처리 분류 (정상/일시장애/고장)

[변경 이력]
    - 희생전류: BASE_FEATURES에서 제외 → SACRIFICIAL_DEVICES(TB24-250406, 407) 전용 분리 처리
    - 통신품질: AI 입력에서 제외 → 룰 기반 필터(-115 dBm)로 통신단절/고장 플래그 생성
               이상 탐지 결과에 통신 상태 컨텍스트를 후처리로 추가
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model

# ============================================================
# 0. 설정  (train 파일과 동일하게 유지)
# ============================================================

# ── 공통 AI 모델 입력 변수 ──────────────────────────────────
BASE_FEATURES = ['방식전위', 'AC유입', '온도', '습도']

# ── 희생양극 방식 기기 전용 설정 ────────────────────────────
SACRIFICIAL_DEVICES  = ['TB24-250406', 'TB24-250407']
SACRIFICIAL_FEATURES = ['희생전류']

# ── 통신품질 룰 기반 필터 설정 ──────────────────────────────
COMM_QUALITY_COL           = '통신품질'
COMM_QUALITY_THRESHOLD_DBM = -115   # dBm 이하 → 통신 불가
COMM_OUTAGE_CONSECUTIVE    = 3      # 연속 N회 이상 단절 → 고장 (미만 = 일시 장애)

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
# 2. 희생양극 기기 전용 데이터 분리  (train 파일과 동일)
# ============================================================

def get_sacrificial_device_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    희생양극 방식 기기(SACRIFICIAL_DEVICES) 전용 희생전류 데이터를 반환합니다.
    공통 LSTM 모델(BASE_FEATURES 기반)과 별도로 분석에 활용합니다.
    """
    sacr_df = df[df['장비번호'].isin(SACRIFICIAL_DEVICES)].copy()
    if sacr_df.empty:
        return pd.DataFrame(columns=['측정시각', '장비번호'] + SACRIFICIAL_FEATURES)

    available_cols = ['측정시각', '장비번호'] + [
        f for f in SACRIFICIAL_FEATURES if f in sacr_df.columns
    ]
    return (
        sacr_df[available_cols]
        .sort_values(['장비번호', '측정시각'])
        .reset_index(drop=True)
    )


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
    희생전류·통신품질은 BASE_FEATURES에서 제외되어 이 함수의 처리 대상이 아닙니다.
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
    return np.array(
        [values[i:i + time_steps] for i in range(len(values) - time_steps)],
        dtype=np.float32,
    )


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
            'gas_common_model_train.py를 먼저 실행하세요.'
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
    observation_ratio: float = OBSERVATION_RATIO,
) -> Dict[str, object]:
    """
    실시간 API 연결용 단일 장비 이상 탐지 함수.

    Args:
        device_window_df : 최근 N개 시점의 단일 장비 데이터프레임
                           필수 컬럼: '장비번호', '측정시각', BASE_FEATURES
                           선택 컬럼: '통신품질' (있으면 통신 상태 후처리 자동 적용)
        model            : 학습된 LSTM AutoEncoder
        scaler_map       : 그룹별 MinMaxScaler
        thresholds       : 장비별 MSE threshold
        feature_cols     : 사용할 피처 컬럼 리스트
        time_steps       : 시퀀스 길이 (학습 시 설정값과 동일해야 함)
        observation_ratio: 관찰 구간 시작 비율 (기본값: OBSERVATION_RATIO)

    Returns:
        dict:
            device_id             : 장비번호
            group                 : 정규화 그룹
            mse                   : 전체 재구성 오차 (MSE)
            threshold             : 해당 장비의 이상 판단 기준값
            risk_level            : AI 위험 단계 ('이상' | '관찰' | '정상')
            comm_status           : 통신 상태 ('정상통신' | '일시장애' | '통신고장')
            ai_reliability        : AI 결과 신뢰 여부 ('신뢰' | '주의' | '신뢰불가')
            feature_contributions : 피처별 MSE 기여도 딕셔너리
            is_sacrificial_device : 희생양극 방식 기기 여부 (bool)
    """
    if device_window_df.empty:
        raise ValueError('입력 데이터가 비어 있습니다.')

    ddf = device_window_df.copy()
    if '장비번호' not in ddf.columns or '측정시각' not in ddf.columns:
        raise ValueError("입력 데이터에는 '장비번호', '측정시각' 컬럼이 필요합니다.")

    # 통신품질 필터 적용 (통신품질 열이 있는 경우)
    if COMM_QUALITY_COL in ddf.columns:
        ddf = apply_comm_quality_filter(ddf)

    ddf = assign_normalization_group(ddf)
    ddf = add_engineered_features(ddf, BASE_FEATURES)
    ddf = ddf.sort_values('측정시각')

    if len(ddf) < time_steps + 1:
        raise ValueError(f'최소 {time_steps + 1}개 시점이 필요합니다. 현재: {len(ddf)}개')

    device_id  = str(ddf['장비번호'].iloc[0])
    group_name = str(ddf['정규화그룹'].iloc[0])

    if group_name not in scaler_map:
        raise ValueError(f'정규화 그룹 "{group_name}" 의 scaler가 없습니다.')

    scaler   = scaler_map[group_name]
    scaled   = scaler.transform(ddf[feature_cols].astype(float).values)
    X        = create_sequences(scaled, time_steps)
    last_seq = X[-1:]
    pred     = model.predict(last_seq, verbose=0)

    mse = float(np.mean(np.power(last_seq - pred, 2)))

    feature_mse_array    = np.mean(np.power(last_seq - pred, 2), axis=1)[0]
    feature_contributions = {
        feat_name: float(feat_mse)
        for feat_name, feat_mse in zip(feature_cols, feature_mse_array)
    }

    threshold  = float(thresholds.get(device_id, np.mean(list(thresholds.values()))))
    risk_level = classify_risk_level(mse, threshold, observation_ratio)

    # ── 통신 상태 후처리 ──────────────────────────────────────
    # 마지막 측정 시점의 통신 상태를 판정
    if '통신단절_플래그' in ddf.columns:
        last_row    = ddf.iloc[-1]
        comm_status = classify_comm_fault_level(ddf.iloc[[-1]]).iloc[0]
    else:
        comm_status = '정상통신'

    # AI 결과 신뢰도: 통신 상태에 따라 조정
    ai_reliability_map = {
        '정상통신': '신뢰',
        '일시장애': '주의',    # AI 입력 데이터가 불완전할 수 있음
        '통신고장': '신뢰불가', # 연속 단절로 AI 판단 근거 부족
    }
    ai_reliability = ai_reliability_map.get(comm_status, '신뢰')

    return {
        'device_id'            : device_id,
        'group'                : group_name,
        'mse'                  : mse,
        'threshold'            : threshold,
        'risk_level'           : risk_level,
        'comm_status'          : comm_status,
        'ai_reliability'       : ai_reliability,
        'feature_contributions': feature_contributions,
        'is_sacrificial_device': device_id in SACRIFICIAL_DEVICES,
    }


# ============================================================
# 8. 전체 장비 일괄 이상 탐지
# ============================================================

def run_batch_prediction(
    df: pd.DataFrame,
    model: Sequential,
    scaler_map: Dict[str, MinMaxScaler],
    thresholds: Dict[str, float],
    feature_cols: List[str],
    time_steps: int,
    observation_ratio: float = OBSERVATION_RATIO,
) -> pd.DataFrame:
    """
    전체 장비에 대해 이상 탐지를 일괄 수행합니다.

    Args:
        df               : build_master_dataset() 으로 만든 전체 데이터프레임
                           통신품질 열 포함 시 apply_comm_quality_filter() 자동 적용
        observation_ratio: 관찰 구간 시작 비율 (기본값: OBSERVATION_RATIO)
        (나머지 인자는 predict_device_window 와 동일)

    Returns:
        pd.DataFrame: 장비별 이상 탐지 결과 요약
            컬럼: device_id, group, mse, threshold, risk_level,
                  comm_status, ai_reliability, is_sacrificial_device,
                  + 피처별 기여도 (feature_contributions 언팩)
    """
    # 통신품질 필터 적용 (열이 있는 경우)
    if COMM_QUALITY_COL in df.columns:
        df = apply_comm_quality_filter(df)

    df = assign_normalization_group(df)
    df = add_engineered_features(df, BASE_FEATURES)

    results    = []
    device_ids = df['장비번호'].unique()

    print(f'  총 {len(device_ids)}개 장비 이상 탐지 시작...')
    print(f'  ※ AI 입력 피처: {BASE_FEATURES} (희생전류·통신품질 제외)')

    for device_id in device_ids:
        ddf = df[df['장비번호'] == device_id].copy()

        group_name = str(ddf['정규화그룹'].iloc[0])
        if group_name not in scaler_map:
            print(f'  [SKIP] {device_id}: scaler 없음 (그룹: {group_name})')
            continue

        scaler = scaler_map[group_name]
        try:
            scaled = scaler.transform(
                ddf.sort_values('측정시각')[feature_cols].astype(float).values
            )
            X = create_sequences(scaled, time_steps)
        except ValueError as e:
            print(f'  [SKIP] {device_id}: {e}')
            continue

        last_seq = X[-1:]
        pred     = model.predict(last_seq, verbose=0)

        mse               = float(np.mean(np.power(last_seq - pred, 2)))
        feature_mse_array = np.mean(np.power(last_seq - pred, 2), axis=1)[0]
        feature_contributions = {
            feat_name: float(feat_mse)
            for feat_name, feat_mse in zip(feature_cols, feature_mse_array)
        }

        threshold  = float(thresholds.get(str(device_id), np.mean(list(thresholds.values()))))
        risk_level = classify_risk_level(mse, threshold, observation_ratio)

        # ── 통신 상태 후처리 (장비의 마지막 시점 기준) ─────
        last_row_df = ddf.sort_values('측정시각').iloc[[-1]]
        if '통신단절_플래그' in ddf.columns:
            comm_status = classify_comm_fault_level(last_row_df).iloc[0]
        else:
            comm_status = '정상통신'

        ai_reliability_map = {
            '정상통신': '신뢰',
            '일시장애': '주의',
            '통신고장': '신뢰불가',
        }
        ai_reliability = ai_reliability_map.get(comm_status, '신뢰')

        row = {
            'device_id'           : device_id,
            'group'               : group_name,
            'mse'                 : mse,
            'threshold'           : threshold,
            'risk_level'          : risk_level,
            'comm_status'         : comm_status,
            'ai_reliability'      : ai_reliability,
            'is_sacrificial_device': str(device_id) in SACRIFICIAL_DEVICES,
        }
        row.update(feature_contributions)
        results.append(row)

    result_df = pd.DataFrame(results)

    if not result_df.empty:
        # AI 이상 탐지 결과 요약
        risk_counts = result_df['risk_level'].value_counts()
        print(f'  탐지 완료: 이상 {risk_counts.get("이상", 0)}개 | '
              f'관찰 {risk_counts.get("관찰", 0)}개 | '
              f'정상 {risk_counts.get("정상", 0)}개')

        # 통신 상태 요약
        comm_counts = result_df['comm_status'].value_counts()
        print(f'  통신 상태: 정상통신 {comm_counts.get("정상통신", 0)}개 | '
              f'일시장애 {comm_counts.get("일시장애", 0)}개 | '
              f'통신고장 {comm_counts.get("통신고장", 0)}개')

        # 신뢰도 저하 장비 경고
        low_rel = result_df[result_df['ai_reliability'] == '신뢰불가']
        if not low_rel.empty:
            print(f'  ⚠️  AI 신뢰불가 장비(통신고장): {low_rel["device_id"].tolist()}')

        # 희생전류 기기 별도 안내
        sacr = result_df[result_df['is_sacrificial_device'] == True]
        if not sacr.empty:
            print(f'  ℹ️  희생양극 기기 {sacr["device_id"].tolist()}: '
              f'희생전류는 AI 모델 외 별도 분석 필요')

    return result_df


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
    print(f'  - 희생전류 기기: {config.get("sacrificial_devices", SACRIFICIAL_DEVICES)}')
    print(f'  - 통신품질 필터: <= {config.get("comm_quality_threshold_dbm", COMM_QUALITY_THRESHOLD_DBM)} dBm, '
          f'연속 {config.get("comm_outage_consecutive_threshold", COMM_OUTAGE_CONSECUTIVE)}회 → 고장')

    # --- 예측 대상 데이터 로드 (train에서 저장한 미관측 test 구간) ---
    test_data_path = artifacts_dir / 'test_data.parquet'
    if not test_data_path.exists():
        raise FileNotFoundError(
            f'test_data.parquet 파일을 찾지 못했습니다: {test_data_path}\n'
            'gas_common_model_train.py를 먼저 실행하세요.'
        )

    print('\n[2] test 구간 데이터 로드 중 (미관측 15% 구간)...')
    target_df = pd.read_parquet(test_data_path)
    print(f'  - 전체 행 수: {len(target_df):,}')
    print(f'  - 장비 수   : {target_df["장비번호"].nunique():,}')
    print(f'  ※ 학습에 사용되지 않은 test 구간만 사용 (leakage 없음)')

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
        )
        print(f'  장비번호          : {result["device_id"]}')
        print(f'  MSE               : {result["mse"]:.6f}')
        print(f'  Threshold         : {result["threshold"]:.6f}')
        print(f'  위험 단계 (AI)    : {RISK_LABELS[result["risk_level"]]}')
        print(f'  통신 상태         : {COMM_STATUS_LABELS[result["comm_status"]]}')
        print(f'  AI 결과 신뢰도    : {result["ai_reliability"]}')
        print(f'  희생양극 기기 여부: {result["is_sacrificial_device"]}')
        print(f'  (관찰 기준: threshold × {OBSERVATION_RATIO} = '
              f'{result["threshold"] * OBSERVATION_RATIO:.6f} 이상)')
        print('  피처별 기여도:')
        for feat, val in result['feature_contributions'].items():
            print(f'    {feat}: {val:.6f}')


if __name__ == '__main__':
    main()