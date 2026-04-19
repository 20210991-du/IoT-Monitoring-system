"""
gas_common_model_train.py
=========================
[학습용] LSTM AutoEncoder 공통 모델 학습 스크립트

실행 방법:
    python gas_common_model_train.py

출력물 (common_model_artifacts/ 폴더):
    - common_lstm_autoencoder.keras  : 학습된 모델
    - group_scalers.pkl              : 그룹별 MinMaxScaler
    - device_thresholds.json         : 장비별 이상 탐지 threshold
    - model_config.json              : 피처/설정 메타데이터
    - plots/                         : 장비별 시계열 시각화 이미지

[변경 이력]
    - 희생전류: 전체 공통 입력 변수에서 제외 → SACRIFICIAL_DEVICES(TB24-250406, 407) 전용 분리 처리
    - 통신품질: AI 모델 입력에서 제외 → 룰 기반 필터(-115 dBm)로 통신단절/고장 플래그 생성
"""

import os
import re
import json
import pickle
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# ============================================================
# 0. 설정
# ============================================================

# ── 공통 AI 모델 입력 변수 ──────────────────────────────────
# 희생전류: 외부전원 방식 기기에는 해당 없음 → 제외 (SACRIFICIAL_DEVICES 전용 처리)
# 통신품질: 룰 기반 필터로 처리 → AI 입력에서 제외 (통신단절/고장 플래그로 대체)
BASE_FEATURES = ['방식전위', 'AC유입', '온도', '습도']

# ── 희생양극 방식 기기 전용 설정 ────────────────────────────
# 이 두 기기만 희생양극 방식을 사용하므로 희생전류 데이터가 존재함
SACRIFICIAL_DEVICES  = ['TB24-250406', 'TB24-250407']
SACRIFICIAL_FEATURES = ['희생전류']

# ── 통신품질 룰 기반 필터 설정 ──────────────────────────────
COMM_QUALITY_COL             = '통신품질'
COMM_QUALITY_THRESHOLD_DBM   = -115   # dBm 이하 → 통신 불가 판정
COMM_OUTAGE_CONSECUTIVE      = 3      # 연속 N회 이상 단절 → 고장으로 판정 (미만 = 일시 장애)

# ── 원본 데이터 로딩 시 수집 대상 전체 피처 (6개 유지) ─────
# 모델 입력은 BASE_FEATURES(4개)로 제한하되, 원시 데이터는 그대로 보존
ALL_RAW_FEATURES = BASE_FEATURES + SACRIFICIAL_FEATURES + [COMM_QUALITY_COL]

RAW_SHEET  = '수집원본데이터'
INFO_SHEET = '설치 및 분석정보'

FEATURE_NAME_MAP = {
    '방식전위(mV)': '방식전위',
    'AC유입(mV)'  : 'AC유입',
    '희생전류(mA)': '희생전류',
    '온도(℃)'    : '온도',
    '습도(%)'    : '습도',
    '통신품질(dBm)': '통신품질',
}

# 장비별 시간 순 데이터 분할 비율 (합산 = 1.0)
TRAIN_RATIO = 0.70   # 학습  70%
VAL_RATIO   = 0.15   # 검증  15%
# TEST_RATIO  = 0.15  # 테스트 15% (= 1 - TRAIN_RATIO - VAL_RATIO)


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
    header_top    = raw.iloc[3].tolist()
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

    NOTE:
      희생전류·통신품질을 포함한 ALL_RAW_FEATURES(6개) 전체를 로드합니다.
      모델 입력 제한은 이 함수가 아닌 이후 전처리 단계에서 적용됩니다.
    """
    raw = pd.read_excel(excel_path, sheet_name=RAW_SHEET, header=None)

    device_row = raw.iloc[1].tolist()
    metric_row = raw.iloc[2].tolist()
    data = raw.iloc[3:].copy().reset_index(drop=True)

    time_col_name = normalize_text(device_row[0]) or '일자'
    data = data.rename(columns={0: time_col_name})
    data[time_col_name] = pd.to_datetime(data[time_col_name], errors='coerce')
    data = data.dropna(subset=[time_col_name]).copy()

    long_frames = []
    col_idx = 1
    n_cols  = raw.shape[1]

    while col_idx < n_cols:
        header_text = normalize_text(device_row[col_idx])
        if not header_text:
            col_idx += 1
            continue

        device_id, facility_no_from_header = parse_device_header(header_text)

        # ALL_RAW_FEATURES 개수(6)만큼 블록 열 확보
        block_cols = list(range(col_idx, min(col_idx + len(ALL_RAW_FEATURES), n_cols)))
        feature_map = {}
        for c in block_cols:
            metric_name = normalize_text(metric_row[c])
            canonical   = FEATURE_NAME_MAP.get(metric_name)
            if canonical:
                feature_map[c] = canonical

        if feature_map:
            subset = data[[time_col_name] + list(feature_map.keys())].copy()
            subset = subset.rename(columns={time_col_name: '측정시각', **feature_map})
            subset['장비번호']    = device_id
            subset['시설번호_raw'] = facility_no_from_header

            # ALL_RAW_FEATURES 기준으로 누락 열 채우기 및 숫자 변환
            for f in ALL_RAW_FEATURES:
                if f not in subset.columns:
                    subset[f] = np.nan
                subset[f] = pd.to_numeric(subset[f], errors='coerce')

            subset = subset[['측정시각', '장비번호', '시설번호_raw'] + ALL_RAW_FEATURES]
            long_frames.append(subset)

        col_idx += len(ALL_RAW_FEATURES)

    if not long_frames:
        raise ValueError("'수집원본데이터' 시트에서 장비 블록을 찾지 못했습니다.")

    merged = pd.concat(long_frames, ignore_index=True)
    merged = merged.sort_values(['장비번호', '측정시각']).reset_index(drop=True)
    return merged


def build_master_dataset(excel_path: str) -> pd.DataFrame:
    """
    원본 수집 데이터 + 설치 정보 병합 후 통신품질 필터까지 적용한
    마스터 데이터프레임을 반환합니다.

    통신품질은 AI 입력에서 제외되고, 대신 아래 두 플래그 컬럼이 추가됩니다.
      - 통신단절_플래그 : 측정값 <= COMM_QUALITY_THRESHOLD_DBM → 1
      - 통신고장_플래그 : 연속 COMM_OUTAGE_CONSECUTIVE 회 이상 단절 → 1
                         (1이지만 고장=0 인 경우 → 일시 장애로 분류)
    """
    raw_df  = load_raw_collection_data(excel_path)
    info_df = load_install_info(excel_path)

    df = raw_df.merge(info_df, on='장비번호', how='left')
    if '시설번호' not in df.columns:
        df['시설번호'] = df['시설번호_raw']
    else:
        df['시설번호'] = df['시설번호'].fillna(df['시설번호_raw'])

    df = df.drop(columns=['시설번호_raw'], errors='ignore')

    # 통신품질 룰 기반 필터 적용 (플래그 생성)
    df = apply_comm_quality_filter(df)

    return df


# ============================================================
# 2. 통신품질 룰 기반 필터
# ============================================================

def apply_comm_quality_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    통신품질(dBm) 열을 룰 기반으로 처리하여 두 가지 플래그 컬럼을 추가합니다.
    통신품질 열 자체는 AI 모델 입력(BASE_FEATURES)에 포함되지 않습니다.

    생성 컬럼:
        통신단절_플래그 (int):
            측정값 <= COMM_QUALITY_THRESHOLD_DBM(-115 dBm) 이면 1, 아니면 0.
            NaN 측정값도 단절(1)로 처리합니다.

        통신고장_플래그 (int):
            장비별 시간 순으로 연속 COMM_OUTAGE_CONSECUTIVE(3)회 이상 단절이면 1.
            연속 횟수가 기준 미만이면 0 (일시 장애로 분류).

    후처리 활용 가이드:
        - 통신단절_플래그=1, 통신고장_플래그=0 → 일시 장애 (AI 결과 참고 수준)
        - 통신단절_플래그=1, 통신고장_플래그=1 → 통신 고장 (AI 판단 신뢰도 낮음, 별도 알림)
        - 통신단절_플래그=0                    → 정상 통신 (AI 입력 신뢰 가능)
    """
    df = df.copy()

    if COMM_QUALITY_COL not in df.columns:
        df['통신단절_플래그'] = 0
        df['통신고장_플래그'] = 0
        return df

    # ── 1. 기본 단절 플래그 ──────────────────────────────────
    comm_vals = pd.to_numeric(df[COMM_QUALITY_COL], errors='coerce')
    # -115 dBm 이하 또는 NaN → 통신 불가
    df['통신단절_플래그'] = (
        comm_vals.le(COMM_QUALITY_THRESHOLD_DBM) | comm_vals.isna()
    ).astype(int)

    # ── 2. 장비별 연속 단절 횟수로 고장 판정 ────────────────
    def _device_fault_flag(sub: pd.DataFrame) -> pd.Series:
        """장비 단위 시계열에서 연속 단절이 임계 횟수 이상이면 고장으로 판정."""
        sub_sorted  = sub.sort_values('측정시각')
        flag        = sub_sorted['통신단절_플래그']
        # 값이 바뀔 때마다 새 연속 그룹 번호 부여
        changed     = (flag != flag.shift(fill_value=0)).cumsum()
        consec_len  = flag.groupby(changed).transform('count')
        fault       = ((flag == 1) & (consec_len >= COMM_OUTAGE_CONSECUTIVE)).astype(int)
        return fault

    fault_series = (
        df.groupby('장비번호', group_keys=False)
          .apply(_device_fault_flag)
    )
    df['통신고장_플래그'] = fault_series

    return df


# ============================================================
# 3. 희생양극 기기 전용 데이터 분리
# ============================================================

def get_sacrificial_device_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    희생양극 방식 기기(SACRIFICIAL_DEVICES) 전용 희생전류 데이터를 분리합니다.

    공통 LSTM 모델(BASE_FEATURES 기반)과 별도로 관리되며,
    향후 희생전류 전용 모델 또는 룰 기반 분석에 활용합니다.

    Returns:
        pd.DataFrame:
            컬럼: 측정시각, 장비번호, 희생전류
            (SACRIFICIAL_FEATURES에 정의된 항목 모두 포함)
            해당 기기가 없거나 희생전류 열이 없으면 빈 DataFrame 반환.

    활용 예시:
        sacrificial_df = get_sacrificial_device_data(master_df)
        for device_id, grp in sacrificial_df.groupby('장비번호'):
            # 희생전류 추세 분석, 별도 임계값 적용 등
            ...
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
# 4. 전처리 / 피처 엔지니어링  (predict 파일과 공유)
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

    희생전류·통신품질은 BASE_FEATURES에서 제외되었으므로 이 함수의 처리 대상이 아닙니다.
    희생전류는 get_sacrificial_device_data()로, 통신품질은 apply_comm_quality_filter()로
    별도 처리됩니다.
    """
    if base_features is None:
        base_features = BASE_FEATURES

    df = df.copy().sort_values(['장비번호', '측정시각']).reset_index(drop=True)

    # 결측 보간 (선형 보간 → ffill → bfill)
    for col in base_features:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = (
            df.groupby('장비번호')[col]
            .transform(lambda s: s.interpolate(method='linear', limit_direction='both').ffill().bfill())
        )

    # 파생 피처: 1차 차분, 24시간 이동 평균, 이동 평균 편차
    for col in base_features:
        df[f'{col}_diff1'] = df.groupby('장비번호')[col].diff().fillna(0)
        df[f'{col}_ma24']  = df.groupby('장비번호')[col].transform(
            lambda s: s.rolling(window=24, min_periods=1).mean()
        )
        df[f'{col}_dev24'] = df[col] - df[f'{col}_ma24']

    return df


def feature_columns(base_features: Optional[List[str]] = None) -> List[str]:
    """
    모델 입력 피처 컬럼 목록을 반환합니다.
    각 BASE_FEATURES 항목에 대해 [원본, _diff1, _dev24] 3개씩 생성 → 총 12개.
    (희생전류·통신품질은 BASE_FEATURES에서 제외되어 포함되지 않음)
    """
    if base_features is None:
        base_features = BASE_FEATURES
    cols = []
    for col in base_features:
        cols.extend([col, f'{col}_diff1', f'{col}_dev24'])
    return cols


# ============================================================
# 5. 시퀀스 데이터  (predict 파일과 공유)
# ============================================================

def create_sequences(values: np.ndarray, time_steps: int) -> np.ndarray:
    if len(values) <= time_steps:
        raise ValueError(f'데이터 길이({len(values)})가 time_steps({time_steps})보다 짧습니다.')
    return np.array(
        [values[i:i + time_steps] for i in range(len(values) - time_steps)],
        dtype=np.float32,
    )


# ============================================================
# 6. 학습 데이터 준비
# ============================================================

def prepare_training_data(
    df: pd.DataFrame,
    time_steps: int = 24,
    min_points_per_device: int = 200,
    base_features: Optional[List[str]] = None,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, MinMaxScaler], pd.DataFrame, List[str]]:
    """
    공통 모델 학습용 데이터를 시간 순서 기반으로 train / val / test 분리합니다.

    분리 기준 (장비별 시간 순 정렬 후):
        train : 앞 train_ratio       (기본 70%)
        val   : 중간 val_ratio        (기본 15%)
        test  : 나머지 1-train-val    (기본 15%)

    개선 사항:
        - 스케일러를 train 구간 데이터만으로 fit → leakage 방지
        - val / test 시퀀스를 별도로 구성 → 검증/테스트 신뢰성 확보
        - threshold는 test 구간(미관측 데이터) 기반으로 산출
        - 입력 피처: BASE_FEATURES(4개) 기반 파생 12개 컬럼
          (희생전류·통신품질 제외)

    Returns:
        X_train    : 학습 시퀀스 배열  (N_train, time_steps, n_features)
        X_val      : 검증 시퀀스 배열  (N_val,   time_steps, n_features)
        X_test     : 테스트 시퀀스 배열 (N_test,  time_steps, n_features)
        scaler_map : train 구간만으로 학습된 그룹별 MinMaxScaler
        test_df    : 스케일링된 test 구간 DataFrame (threshold 계산용)
        feats      : 사용된 피처 컬럼 리스트
    """
    if base_features is None:
        base_features = BASE_FEATURES

    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError('train_ratio + val_ratio 합이 1.0 이상입니다.')

    # 각 구간이 최소 1개 시퀀스를 만들 수 있는 최소 데이터 포인트 계산
    min_per_split = time_steps + 1
    min_ratio     = min(train_ratio, val_ratio, test_ratio)
    min_required  = int(np.ceil(min_per_split / min_ratio))
    if min_points_per_device < min_required:
        print(f'  [경고] min_points_per_device({min_points_per_device}) → '
              f'권장값 {min_required}으로 자동 상향합니다.')
        min_points_per_device = min_required

    df   = assign_normalization_group(df)
    df   = add_engineered_features(df, base_features)
    feats = feature_columns(base_features)

    # ── 1단계: 장비별 구간 분리 + train 원본값 수집 (scaler fit 전용) ──
    group_train_data: Dict[str, List[np.ndarray]] = {}
    device_splits:    Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}

    for device_id, ddf in df.groupby('장비번호'):
        ddf = ddf.sort_values('측정시각').reset_index(drop=True)
        if len(ddf) < min_points_per_device:
            continue

        n         = len(ddf)
        train_end = int(n * train_ratio)
        val_end   = int(n * (train_ratio + val_ratio))

        splits_ok = (
            train_end > time_steps and
            (val_end - train_end) > time_steps and
            (n - val_end) > time_steps
        )
        if not splits_ok:
            continue

        group_name  = str(ddf['정규화그룹'].iloc[0])
        train_values = ddf.iloc[:train_end][feats].astype(float).values
        group_train_data.setdefault(group_name, []).append(train_values)

        device_splits[str(device_id)] = (
            ddf.iloc[:train_end].copy(),
            ddf.iloc[train_end:val_end].copy(),
            ddf.iloc[val_end:].copy(),
        )

    if not device_splits:
        raise ValueError(
            f'유효한 장비가 없습니다. '
            f'min_points_per_device({min_points_per_device})나 time_steps({time_steps})를 조정하세요.'
        )

    # ── 2단계: train 구간 데이터만으로 scaler 학습 (leakage 방지) ──
    scaler_map: Dict[str, MinMaxScaler] = {}
    for group_name, arrays in group_train_data.items():
        scaler = MinMaxScaler()
        scaler.fit(np.vstack(arrays))
        scaler_map[group_name] = scaler

    # ── 3단계: 각 구간 스케일링 후 시퀀스 생성 ──
    train_seqs, val_seqs, test_seqs = [], [], []
    test_parts: List[pd.DataFrame] = []

    for device_id, (train_ddf, val_ddf, test_ddf) in device_splits.items():
        group_name = str(train_ddf['정규화그룹'].iloc[0])
        scaler     = scaler_map[group_name]

        for split_ddf, seq_list in [
            (train_ddf, train_seqs),
            (val_ddf,   val_seqs),
            (test_ddf,  test_seqs),
        ]:
            scaled = scaler.transform(split_ddf[feats].astype(float).values)
            try:
                seqs = create_sequences(scaled, time_steps)
                seq_list.append(seqs)
            except ValueError:
                continue

        # test 구간: 스케일링된 값을 DataFrame에 보관 (threshold 계산용)
        test_scaled          = scaler.transform(test_ddf[feats].astype(float).values)
        test_stored          = test_ddf.copy()
        test_stored.loc[:, feats] = test_scaled
        test_parts.append(test_stored)

    for name, seqs in [('train', train_seqs), ('val', val_seqs), ('test', test_seqs)]:
        if not seqs:
            raise ValueError(
                f'{name} 시퀀스를 만들지 못했습니다. '
                f'min_points_per_device를 높이거나 {name}_ratio를 조정하세요.'
            )

    X_train = np.concatenate(train_seqs, axis=0)
    X_val   = np.concatenate(val_seqs,   axis=0)
    X_test  = np.concatenate(test_seqs,  axis=0)
    test_df = pd.concat(test_parts, ignore_index=True)

    print(f'  - 학습 장비 수  : {len(device_splits)}개')
    print(f'  - X_train shape : {X_train.shape}  ({train_ratio*100:.0f}%)')
    print(f'  - X_val   shape : {X_val.shape}  ({val_ratio*100:.0f}%)')
    print(f'  - X_test  shape : {X_test.shape}  ({(1-train_ratio-val_ratio)*100:.0f}%)')
    print(f'  - 모델 입력 피처 수: {len(feats)}개 (BASE_FEATURES {len(base_features)}개 × 3)')
    print(f'  ※ 제외 피처: 희생전류(SACRIFICIAL_DEVICES 전용), 통신품질(룰 기반 처리)')

    return X_train, X_val, X_test, scaler_map, test_df, feats


# ============================================================
# 7. 모델 빌드
# ============================================================

def build_common_model(n_timesteps: int, n_features: int) -> Sequential:
    # LSTM 기본 activation은 tanh로, relu 사용 시 gradient 폭발/소실 위험이 있어 제거
    model = Sequential([
        LSTM(128, input_shape=(n_timesteps, n_features), return_sequences=True),
        LSTM(64,  return_sequences=False),
        RepeatVector(n_timesteps),
        LSTM(64,  return_sequences=True),
        LSTM(128, return_sequences=True),
        TimeDistributed(Dense(n_features))
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mse')
    return model


# ============================================================
# 8. Threshold 계산 및 아티팩트 저장
# ============================================================

def compute_device_thresholds(
    model: Sequential,
    test_df: pd.DataFrame,
    feats: List[str],
    time_steps: int = 24,
    percentile: float = 99.0,
) -> Dict[str, float]:
    """
    장비별 threshold를 test 구간(미관측 데이터)으로 산출합니다.
    """
    thresholds: Dict[str, float] = {}

    for device_id, ddf in test_df.groupby('장비번호'):
        values = ddf.sort_values('측정시각')[feats].astype(float).values
        if len(values) <= time_steps:
            continue

        X    = create_sequences(values, time_steps)
        pred = model.predict(X, verbose=0)
        mse  = np.mean(np.power(X - pred, 2), axis=(1, 2))
        thresholds[str(device_id)] = float(np.percentile(mse, percentile))

    return thresholds


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
        'time_steps'              : time_steps,
        'base_features'           : BASE_FEATURES,
        'feature_columns'         : feats,
        'raw_sheet'               : RAW_SHEET,
        'info_sheet'              : INFO_SHEET,
        # ── 희생전류 분리 처리 정보 ───────────────────────
        'sacrificial_devices'     : SACRIFICIAL_DEVICES,
        'sacrificial_features'    : SACRIFICIAL_FEATURES,
        # ── 통신품질 룰 기반 필터 정보 ───────────────────
        'comm_quality_col'                 : COMM_QUALITY_COL,
        'comm_quality_threshold_dbm'       : COMM_QUALITY_THRESHOLD_DBM,
        'comm_outage_consecutive_threshold': COMM_OUTAGE_CONSECUTIVE,
    }
    with open(os.path.join(save_dir, 'model_config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


# ============================================================
# 9. 테스트 평가
# ============================================================

def evaluate_model(
    model: Sequential,
    X_test: np.ndarray,
    thresholds: Dict[str, float],
) -> Dict[str, float]:
    """
    테스트 셋 전체의 재구성 오차(MSE) 분포를 계산하여 모델 품질을 평가합니다.

    비지도 학습 특성상 정답 레이블이 없으므로 정밀도/재현율 대신
    MSE 분포 통계와 threshold 대비 이상 비율로 간접 평가합니다.
    """
    pred         = model.predict(X_test, verbose=0)
    mse_per_seq  = np.mean(np.power(X_test - pred, 2), axis=(1, 2))

    metrics: Dict[str, float] = {
        'test_mse_mean': float(np.mean(mse_per_seq)),
        'test_mse_std' : float(np.std(mse_per_seq)),
        'test_mse_p50' : float(np.percentile(mse_per_seq, 50)),
        'test_mse_p95' : float(np.percentile(mse_per_seq, 95)),
        'test_mse_p99' : float(np.percentile(mse_per_seq, 99)),
    }

    if thresholds:
        mean_threshold = float(np.mean(list(thresholds.values())))
        metrics['mean_threshold']                    = mean_threshold
        metrics['anomaly_rate_vs_mean_threshold']    = float(np.mean(mse_per_seq > mean_threshold))

    return metrics


# ============================================================
# 10. 시각화
# ============================================================

def set_korean_font():
    """운영체제에 맞는 한글 폰트를 설정합니다."""
    system_name = platform.system()
    if system_name == 'Windows':
        plt.rc('font', family='Malgun Gothic')
    elif system_name == 'Darwin':
        plt.rc('font', family='AppleGothic')
    elif system_name == 'Linux':
        plt.rc('font', family='NanumGothic')
    plt.rcParams['axes.unicode_minus'] = False


def plot_device_features(
    df: pd.DataFrame,
    device_id: str,
    features: List[str],
    save_path: Optional[str] = None,
    show_plot: bool = True,
):
    """
    특정 장비의 피처들을 시계열 그래프로 시각화합니다.
    희생양극 방식 기기(SACRIFICIAL_DEVICES)는 별도 서브플롯에 희생전류를 추가합니다.
    """
    set_korean_font()

    device_df = df[df['장비번호'] == str(device_id)].copy()
    device_df = device_df.sort_values('측정시각')

    if device_df.empty:
        print(f'장비번호 {device_id}에 대한 데이터가 존재하지 않아 시각화할 수 없습니다.')
        return

    # 희생양극 기기는 희생전류 서브플롯 추가
    plot_features = list(features)
    if str(device_id) in SACRIFICIAL_DEVICES and '희생전류' in device_df.columns:
        plot_features = plot_features + ['희생전류']

    num_features = len(plot_features)
    fig, axes    = plt.subplots(num_features, 1, figsize=(14, 3 * num_features), sharex=True)

    if num_features == 1:
        axes = [axes]

    for ax, feature in zip(axes, plot_features):
        color = 'darkorange' if feature == '희생전류' else None
        label = f'{feature} [희생전류 전용]' if feature == '희생전류' else feature
        ax.plot(device_df['측정시각'], device_df[feature], label=label,
                linewidth=1.5, color=color)
        ax.set_ylabel(feature, fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.6)

    # 통신 단절 구간 음영 표시 (통신단절_플래그 존재 시)
    if '통신단절_플래그' in device_df.columns:
        outage_times = device_df[device_df['통신단절_플래그'] == 1]['측정시각']
        for ax in axes:
            for t in outage_times:
                ax.axvline(x=t, color='red', alpha=0.15, linewidth=0.5)

    axes[-1].set_xlabel('측정시각', fontsize=12)

    title_suffix = ' [희생양극 방식]' if str(device_id) in SACRIFICIAL_DEVICES else ''
    fig.suptitle(
        f'[{device_id}]{title_suffix} 장비 센서 데이터 시계열 추이',
        fontsize=16, fontweight='bold', y=0.98,
    )
    plt.tight_layout()

    if save_path:
        save_dir_path = os.path.dirname(save_path)
        if save_dir_path:
            os.makedirs(save_dir_path, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'  -> 그래프 이미지가 저장되었습니다: {save_path}')

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# 11. 메인 실행 (학습)
# ============================================================

def main():
    base_dir   = Path(__file__).resolve().parent
    excel_path = base_dir / '01. 시설물 50개 샘플 데이터.xlsx'
    save_dir   = base_dir / 'common_model_artifacts'

    if not excel_path.exists():
        raise FileNotFoundError(
            f'엑셀 파일을 찾지 못했습니다: {excel_path}\n'
            '파이썬 파일과 같은 폴더에 "01. 시설물 50개 샘플 데이터.xlsx"를 두세요.'
        )

    # ── [1] 데이터 로드 ──
    print('[1] 원본 시트 로드 중...')
    master_df = build_master_dataset(str(excel_path))
    print(f'  - 전체 행 수: {len(master_df):,}')
    print(f'  - 장비 수   : {master_df["장비번호"].nunique():,}')

    # 통신 상태 요약 출력
    if '통신단절_플래그' in master_df.columns:
        outage_cnt = master_df['통신단절_플래그'].sum()
        fault_cnt  = master_df['통신고장_플래그'].sum()
        total      = len(master_df)
        print(f'  - 통신단절 이벤트 : {outage_cnt:,}건 ({outage_cnt/total:.1%}) '
              f'[고장 {fault_cnt:,}건 / 일시장애 {outage_cnt-fault_cnt:,}건]')

    # 희생전류 기기 요약 출력
    sacr_df = get_sacrificial_device_data(master_df)
    if not sacr_df.empty:
        print(f'  - 희생전류 데이터 : {SACRIFICIAL_DEVICES} '
              f'→ {len(sacr_df):,}행 (공통 모델 입력 제외, 별도 보존)')

    # ── [2] Train / Val / Test 분리 + Scaler 학습 ──
    print('\n[2] 데이터 분리 및 전처리 중...')
    print(f'  - 분할 비율: train {TRAIN_RATIO*100:.0f}% / val {VAL_RATIO*100:.0f}% '
          f'/ test {(1-TRAIN_RATIO-VAL_RATIO)*100:.0f}%')

    time_steps = 24
    X_train, X_val, X_test, scaler_map, test_df, feats = prepare_training_data(
        master_df,
        time_steps=time_steps,
        min_points_per_device=200,
        base_features=BASE_FEATURES,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
    )

    # ── [3] 모델 학습 ──
    print('\n[3] 공통 모델 학습 중...')
    np.random.seed(42)
    tf.random.set_seed(42)

    model = build_common_model(X_train.shape[1], X_train.shape[2])

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    model.fit(
        X_train, X_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, X_val),
        callbacks=[early_stop],
        shuffle=True,
        verbose=1,
    )

    # ── [4] Threshold 계산 (테스트 구간 기반) ──
    print('\n[4] 장비별 threshold 계산 중 (테스트 구간 사용)...')
    thresholds = compute_device_thresholds(
        model=model,
        test_df=test_df,
        feats=feats,
        time_steps=time_steps,
        percentile=99.0,
    )
    print(f'  - threshold 생성 장비 수: {len(thresholds):,}')

    # ── [5] 테스트 평가 ──
    print('\n[5] 테스트 셋 성능 평가 중...')
    metrics = evaluate_model(model, X_test, thresholds)
    print('  ┌─────────────────────────────────────────┐')
    print(f'  │  테스트 MSE 평균          : {metrics["test_mse_mean"]:.6f}')
    print(f'  │  테스트 MSE 표준편차      : {metrics["test_mse_std"]:.6f}')
    print(f'  │  테스트 MSE 중앙값 (p50)  : {metrics["test_mse_p50"]:.6f}')
    print(f'  │  테스트 MSE p95           : {metrics["test_mse_p95"]:.6f}')
    print(f'  │  테스트 MSE p99           : {metrics["test_mse_p99"]:.6f}')
    if 'mean_threshold' in metrics:
        print(f'  │  평균 threshold           : {metrics["mean_threshold"]:.6f}')
        print(f'  │  이상 의심 비율           : {metrics["anomaly_rate_vs_mean_threshold"]:.2%}')
    print('  └─────────────────────────────────────────┘')

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'eval_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # ── [6] 아티팩트 저장 ──
    print('\n[6] 아티팩트 저장 중...')
    save_artifacts(
        save_dir=str(save_dir),
        model=model,
        scaler_map=scaler_map,
        thresholds=thresholds,
        feats=feats,
        time_steps=time_steps,
    )

    # ── [7] 시각화 ──
    print('\n[7] 모든 장비 데이터 시각화 및 저장 중...')
    if thresholds:
        plot_dir = os.path.join(save_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)

        for device_id in thresholds.keys():
            plot_file_path = os.path.join(plot_dir, f'{device_id}_features_plot.png')
            # 공통 BASE_FEATURES 기준으로 시각화 (희생양극 기기는 내부에서 희생전류 추가)
            plot_device_features(
                df=master_df,
                device_id=device_id,
                features=BASE_FEATURES,
                save_path=plot_file_path,
                show_plot=False,
            )
        print(f'  총 {len(thresholds)}개 장비의 그래프 저장 완료 (경로: {plot_dir})')
        print(f'  ※ {SACRIFICIAL_DEVICES} 그래프에는 희생전류 서브플롯이 포함됩니다.')
    else:
        print('  시각화할 장비 데이터가 없습니다.')

    print('\n===== 학습 완료 =====')
    print(f'저장 폴더: {save_dir}')
    print('생성 파일:')
    print('  * common_lstm_autoencoder.keras')
    print('  * group_scalers.pkl')
    print('  * device_thresholds.json')
    print('  * model_config.json  (희생전류·통신품질 설정 포함)')
    print('  * eval_metrics.json')
    if thresholds:
        print('  * plots/ 폴더 내 장비별 시각화 이미지')
    print('\n[설계 정책 요약]')
    print(f'  - BASE_FEATURES (AI 모델 입력): {BASE_FEATURES}')
    print(f'  - 희생전류: {SACRIFICIAL_DEVICES} 전용 → 공통 모델 제외, 별도 분리 보존')
    print(f'  - 통신품질: 룰 기반 처리 (-115 dBm 이하 = 단절, 연속 {COMM_OUTAGE_CONSECUTIVE}회 = 고장)')


if __name__ == '__main__':
    main()
