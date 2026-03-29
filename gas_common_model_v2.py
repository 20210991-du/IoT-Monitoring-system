
import os
import re
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed

# ============================================================
# 0. 설정
# ============================================================

BASE_FEATURES = ['방식전위', 'AC유입', '희생전류', '온도', '습도', '통신품질']
RAW_SHEET = '수집원본데이터'
INFO_SHEET = '설치 및 분석정보'

FEATURE_NAME_MAP = {
    '방식전위(mV)': '방식전위',
    'AC유입(mV)': 'AC유입',
    '희생전류(mA)': '희생전류',
    '온도(℃)': '온도',
    '습도(%)': '습도',
    '통신품질(dBm)': '통신품질',
}


# ============================================================
# 1. 엑셀 로드
# ============================================================

def normalize_text(x) -> str:
    if pd.isna(x):
        return ''
    return str(x).replace('\n', ' ').strip()


def parse_device_header(text: str) -> Tuple[str, Optional[str]]:
    """
    예: 'TB24-250401 (1-178)' -> ('TB24-250401', '1-178')
    """
    text = normalize_text(text)
    m = re.match(r'^(TB[\w-]+)\s*\(([^)]+)\)$', text)
    if m:
        return m.group(1), m.group(2).strip()
    return text, None


def load_install_info(excel_path: str) -> pd.DataFrame:
    """
    '설치 및 분석정보' 시트는 머리글이 2줄이므로 4~5행을 합쳐 컬럼명 생성.
    """
    raw = pd.read_excel(excel_path, sheet_name=INFO_SHEET, header=None)
    header_top = raw.iloc[3].tolist()
    header_bottom = raw.iloc[4].tolist()

    columns = []
    for a, b in zip(header_top, header_bottom):
        a = normalize_text(a)
        b = normalize_text(b)
        if a and b:
            columns.append(f'{a}_{b}')
        elif a:
            columns.append(a)
        elif b:
            columns.append(b)
        else:
            columns.append('')

    df = raw.iloc[5:].copy()
    df.columns = columns
    df = df.loc[:, [c for c in df.columns if c]]
    df = df.dropna(how='all').copy()

    # 필요한 컬럼만 정리
    rename_map = {}
    for c in df.columns:
        if '장비번호' in c:
            rename_map[c] = '장비번호'
        elif '시설번호' in c:
            rename_map[c] = '시설번호'
        elif 'CTN' in c:
            rename_map[c] = 'CTN'
        elif c == '형식' or '형식' in c:
            rename_map[c] = '형식'
        elif '주소' in c:
            rename_map[c] = '주소'
        elif '위도' in c:
            rename_map[c] = '위도'
        elif '경도' in c:
            rename_map[c] = '경도'

    df = df.rename(columns=rename_map)
    if '장비번호' not in df.columns:
        raise ValueError("'설치 및 분석정보' 시트에서 장비번호 컬럼을 찾지 못했습니다.")

    keep_cols = [c for c in ['장비번호', '시설번호', 'CTN', '형식', '주소', '위도', '경도'] if c in df.columns]
    df = df[keep_cols].copy()
    df['장비번호'] = df['장비번호'].astype(str).str.strip()
    df = df[df['장비번호'].ne('')].drop_duplicates(subset=['장비번호'])
    return df


def load_raw_collection_data(excel_path: str) -> pd.DataFrame:
    """
    '수집원본데이터' 시트 파싱.
    구조:
      2행: 장비 블록 헤더 (예: TB24-250401 (1-178))
      3행: 센서 항목 헤더 (방식전위, AC유입, ...)
      4행~: 값
    """
    raw = pd.read_excel(excel_path, sheet_name=RAW_SHEET, header=None)

    device_row = raw.iloc[1].tolist()
    metric_row = raw.iloc[2].tolist()
    data = raw.iloc[3:].copy().reset_index(drop=True)

    # 첫 컬럼 = 일자
    time_col_name = normalize_text(device_row[0]) or '일자'
    data = data.rename(columns={0: time_col_name})
    data[time_col_name] = pd.to_datetime(data[time_col_name], errors='coerce')
    data = data.dropna(subset=[time_col_name]).copy()

    long_frames = []
    col_idx = 1
    n_cols = raw.shape[1]

    while col_idx < n_cols:
        header_text = normalize_text(device_row[col_idx])
        if not header_text:
            col_idx += 1
            continue

        device_id, facility_no_from_header = parse_device_header(header_text)

        block_cols = list(range(col_idx, min(col_idx + 6, n_cols)))
        feature_map = {}
        for c in block_cols:
            metric_name = normalize_text(metric_row[c])
            canonical = FEATURE_NAME_MAP.get(metric_name)
            if canonical:
                feature_map[c] = canonical

        if feature_map:
            subset = data[[time_col_name] + list(feature_map.keys())].copy()
            subset = subset.rename(columns={time_col_name: '측정시각', **feature_map})
            subset['장비번호'] = device_id
            subset['시설번호_raw'] = facility_no_from_header

            for f in BASE_FEATURES:
                if f not in subset.columns:
                    subset[f] = np.nan
                subset[f] = pd.to_numeric(subset[f], errors='coerce')

            subset = subset[['측정시각', '장비번호', '시설번호_raw'] + BASE_FEATURES]
            long_frames.append(subset)

        col_idx += 6

    if not long_frames:
        raise ValueError("'수집원본데이터' 시트에서 장비 블록을 찾지 못했습니다.")

    merged = pd.concat(long_frames, ignore_index=True)
    merged = merged.sort_values(['장비번호', '측정시각']).reset_index(drop=True)
    return merged


def build_master_dataset(excel_path: str) -> pd.DataFrame:
    raw_df = load_raw_collection_data(excel_path)
    info_df = load_install_info(excel_path)

    df = raw_df.merge(info_df, on='장비번호', how='left')
    if '시설번호' not in df.columns:
        df['시설번호'] = df['시설번호_raw']
    else:
        df['시설번호'] = df['시설번호'].fillna(df['시설번호_raw'])

    df = df.drop(columns=['시설번호_raw'], errors='ignore')
    return df


# ============================================================
# 2. 전처리 / 피처 엔지니어링
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


def add_engineered_features(df: pd.DataFrame, base_features: Optional[List[str]] = None) -> pd.DataFrame:
    if base_features is None:
        base_features = BASE_FEATURES

    df = df.copy().sort_values(['장비번호', '측정시각']).reset_index(drop=True)

    # 장비별 결측 보간
    for col in base_features:
        df[col] = (
            df.groupby('장비번호')[col]
            .transform(lambda s: s.interpolate(method='linear', limit_direction='both').ffill().bfill())
        )

    # 변화량 / 최근 24포인트 평균 / 편차
    for col in base_features:
        df[f'{col}_diff1'] = df.groupby('장비번호')[col].diff().fillna(0)
        df[f'{col}_ma24'] = df.groupby('장비번호')[col].transform(lambda s: s.rolling(window=24, min_periods=1).mean())
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
# 3. 시퀀스 데이터
# ============================================================

def create_sequences(values: np.ndarray, time_steps: int) -> np.ndarray:
    if len(values) <= time_steps:
        raise ValueError(f'데이터 길이({len(values)})가 time_steps({time_steps})보다 짧습니다.')
    return np.array([values[i:i + time_steps] for i in range(len(values) - time_steps)], dtype=np.float32)


def prepare_training_data(
    df: pd.DataFrame,
    time_steps: int = 24,
    min_points_per_device: int = 72,
    base_features: Optional[List[str]] = None,
) -> Tuple[np.ndarray, Dict[str, MinMaxScaler], pd.DataFrame, List[str]]:
    """
    공통 모델 학습용 데이터 생성:
    - 그룹별 scaler 학습
    - 장비별 시퀀스 생성
    - 모든 장비 시퀀스 합치기
    """
    if base_features is None:
        base_features = BASE_FEATURES

    df = assign_normalization_group(df)
    df = add_engineered_features(df, base_features)
    feats = feature_columns(base_features)

    # 그룹별 스케일러 학습
    scaler_map: Dict[str, MinMaxScaler] = {}
    for group_name, gdf in df.groupby('정규화그룹'):
        scaler = MinMaxScaler()
        scaler.fit(gdf[feats].astype(float).values)
        scaler_map[str(group_name)] = scaler

    sequence_list = []
    prepared_parts = []

    for device_id, ddf in df.groupby('장비번호'):
        ddf = ddf.sort_values('측정시각').copy()
        if len(ddf) < min_points_per_device:
            continue

        group_name = str(ddf['정규화그룹'].iloc[0])
        scaler = scaler_map[group_name]
        scaled = scaler.transform(ddf[feats].astype(float).values)

        try:
            seqs = create_sequences(scaled, time_steps)
        except ValueError:
            continue

        ddf.loc[:, feats] = scaled
        prepared_parts.append(ddf)
        sequence_list.append(seqs)

    if not sequence_list:
        raise ValueError('학습 가능한 장비 시퀀스를 만들지 못했습니다. time_steps 또는 min_points_per_device를 낮춰보세요.')

    X_train = np.concatenate(sequence_list, axis=0)
    prepared_df = pd.concat(prepared_parts, ignore_index=True)
    return X_train, scaler_map, prepared_df, feats


# ============================================================
# 4. 모델
# ============================================================

def build_common_model(n_timesteps: int, n_features: int) -> Sequential:
    model = Sequential([
        LSTM(128, activation='relu', input_shape=(n_timesteps, n_features), return_sequences=True),
        LSTM(64, activation='relu', return_sequences=False),
        RepeatVector(n_timesteps),
        LSTM(64, activation='relu', return_sequences=True),
        LSTM(128, activation='relu', return_sequences=True),
        TimeDistributed(Dense(n_features))
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mse')
    return model


def compute_device_thresholds(
    model: Sequential,
    prepared_df: pd.DataFrame,
    feats: List[str],
    time_steps: int = 24,
    percentile: float = 99.0,
) -> Dict[str, float]:
    """
    공통 모델은 1개, threshold는 장비별로 따로 계산.
    """
    thresholds: Dict[str, float] = {}

    for device_id, ddf in prepared_df.groupby('장비번호'):
        values = ddf.sort_values('측정시각')[feats].astype(float).values
        if len(values) <= time_steps:
            continue

        X = create_sequences(values, time_steps)
        pred = model.predict(X, verbose=0)
        mse = np.mean(np.power(X - pred, 2), axis=(1, 2))
        thresholds[str(device_id)] = float(np.percentile(mse, percentile))

    return thresholds


# ============================================================
# 5. 저장 / 예측
# ============================================================

def save_artifacts(
    save_dir: str,
    model: Sequential,
    scaler_map: Dict[str, MinMaxScaler],
    thresholds: Dict[str, float],
    feats: List[str],
    time_steps: int,
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    model.save(os.path.join(save_dir, 'common_lstm_autoencoder.keras'))

    with open(os.path.join(save_dir, 'group_scalers.pkl'), 'wb') as f:
        pickle.dump(scaler_map, f)

    with open(os.path.join(save_dir, 'device_thresholds.json'), 'w', encoding='utf-8') as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2)

    config = {
        'time_steps': time_steps,
        'base_features': BASE_FEATURES,
        'feature_columns': feats,
        'raw_sheet': RAW_SHEET,
        'info_sheet': INFO_SHEET,
    }
    with open(os.path.join(save_dir, 'model_config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def load_artifacts(save_dir: str):
    model = load_model(os.path.join(save_dir, 'common_lstm_autoencoder.keras'))
    with open(os.path.join(save_dir, 'group_scalers.pkl'), 'rb') as f:
        scaler_map = pickle.load(f)
    with open(os.path.join(save_dir, 'device_thresholds.json'), 'r', encoding='utf-8') as f:
        thresholds = json.load(f)
    with open(os.path.join(save_dir, 'model_config.json'), 'r', encoding='utf-8') as f:
        config = json.load(f)
    return model, scaler_map, thresholds, config


def predict_device_window(
    device_window_df: pd.DataFrame,
    model: Sequential,
    scaler_map: Dict[str, MinMaxScaler],
    thresholds: Dict[str, float],
    feature_cols: List[str],
    time_steps: int,
) -> Dict[str, object]:
    """
    실시간 API 연결용 보조 함수.
    입력: 최근 N개 시점의 단일 장비 데이터프레임
    """
    if device_window_df.empty:
        raise ValueError('입력 데이터가 비어 있습니다.')

    ddf = device_window_df.copy()
    if '장비번호' not in ddf.columns or '측정시각' not in ddf.columns:
        raise ValueError("입력 데이터에는 '장비번호', '측정시각' 컬럼이 필요합니다.")

    ddf = assign_normalization_group(ddf)
    ddf = add_engineered_features(ddf, BASE_FEATURES)
    ddf = ddf.sort_values('측정시각')

    if len(ddf) < time_steps + 1:
        raise ValueError(f'최소 {time_steps + 1}개 시점이 필요합니다.')

    device_id = str(ddf['장비번호'].iloc[0])
    group_name = str(ddf['정규화그룹'].iloc[0])

    if group_name not in scaler_map:
        raise ValueError(f'정규화 그룹 {group_name} 의 scaler가 없습니다.')

    scaler = scaler_map[group_name]
    scaled = scaler.transform(ddf[feature_cols].astype(float).values)
    X = create_sequences(scaled, time_steps)
    last_seq = X[-1:]
    pred = model.predict(last_seq, verbose=0)
    mse = float(np.mean(np.power(last_seq - pred, 2)))

    threshold = float(thresholds.get(device_id, np.mean(list(thresholds.values()))))
    is_anomaly = mse > threshold

    return {
        'device_id': device_id,
        'group': group_name,
        'mse': mse,
        'threshold': threshold,
        'is_anomaly': bool(is_anomaly),
    }


# ============================================================
# 6. 메인 실행
# ============================================================

def main():
    base_dir = Path(__file__).resolve().parent
    excel_path = base_dir / '01. 시설물 50개 샘플 데이터.xlsx'
    save_dir = base_dir / 'common_model_artifacts'

    if not excel_path.exists():
        raise FileNotFoundError(
            f'엑셀 파일을 찾지 못했습니다: {excel_path}\n'
            '파이썬 파일과 같은 폴더에 "01. 시설물 50개 샘플 데이터.xlsx"를 두세요.'
        )

    print('[1] 원본 시트 로드 중...')
    master_df = build_master_dataset(str(excel_path))
    print(f'  - 전체 행 수: {len(master_df):,}')
    print(f'  - 장비 수: {master_df["장비번호"].nunique():,}')

    print('[2] 공통 학습 데이터 준비 중...')
    time_steps = 24
    X_train, scaler_map, prepared_df, feats = prepare_training_data(
        master_df,
        time_steps=time_steps,
        min_points_per_device=72,
        base_features=BASE_FEATURES,
    )
    print(f'  - 학습 시퀀스 shape: {X_train.shape}')

    print('[3] 공통 모델 학습 중...')
    model = build_common_model(X_train.shape[1], X_train.shape[2])
    model.fit(
        X_train, X_train,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
        verbose=1,
    )

    print('[4] 장비별 threshold 계산 중...')
    thresholds = compute_device_thresholds(
        model=model,
        prepared_df=prepared_df,
        feats=feats,
        time_steps=time_steps,
        percentile=99.0,
    )
    print(f'  - threshold 생성 장비 수: {len(thresholds):,}')

    print('[5] 아티팩트 저장 중...')
    save_artifacts(
        save_dir=str(save_dir),
        model=model,
        scaler_map=scaler_map,
        thresholds=thresholds,
        feats=feats,
        time_steps=time_steps,
    )

    print('\n완료')
    print(f'- 저장 폴더: {save_dir}')
    print('- 생성 파일:')
    print('  * common_lstm_autoencoder.keras')
    print('  * group_scalers.pkl')
    print('  * device_thresholds.json')
    print('  * model_config.json')


if __name__ == '__main__':
    main()
