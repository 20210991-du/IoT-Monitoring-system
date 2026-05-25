"""
gas_unified_model_train.py
==========================
[통합 학습용] 공통 모델 + 희생양극 전용 모델을 한 번에 학습합니다.

  데이터 로드  ──▶  공통 LSTM 학습  ──▶  희생양극 LSTM 학습  ──▶  완료
  (1회)              (BASE_FEATURES)      (SACR_FEATURES)

기존 분리 학습 파일 2개를 대체합니다.

실행 방법:
    python gas_unified_model_train.py

출력물:
    common_model_artifacts/
        common_lstm_autoencoder.keras
        group_scalers.pkl
        device_thresholds.json
        model_config.json
        eval_metrics.json
        test_data.parquet

    sacrificial_model_artifacts/
        sacrificial_lstm_autoencoder.keras
        sacrificial_device_scalers.pkl
        sacrificial_thresholds.json
        sacrificial_model_config.json
        eval_metrics.json
        test_data_sacrificial.parquet
"""

import os
import re
import json
import pickle
import argparse
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from db_connector import get_connection, read_sql, SENSOR_NAME_MAP
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# 0. 학습 설정
# ============================================================

@dataclass(frozen=True)
class TrainingConfig:
    common_artifact_dir: str = 'common_model_artifacts'
    sacrificial_artifact_dir: str = 'sacrificial_model_artifacts'

    base_features: Tuple[str, ...] = ('방식전위', 'AC유입', '온도', '습도')
    sacrificial_devices: Tuple[str, ...] = ('TB24-250406', 'TB24-250407')
    sacrificial_features: Tuple[str, ...] = ('희생전류',)

    comm_quality_col: str = '통신품질'
    comm_quality_threshold_dbm: int = -115
    comm_outage_consecutive: int = 3

    # DB 데이터 조회 날짜 범위 (None이면 전체 기간)
    data_start_date: Optional[str] = None
    data_end_date: Optional[str] = None

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    time_steps: int = 24
    min_points_per_device: int = 200
    sacrificial_min_points: int = 50

    common_epochs: int = 50
    common_batch_size: int = 32
    common_early_stopping_patience: int = 5
    sacrificial_epochs: int = 80
    sacrificial_batch_size: int = 16
    sacrificial_early_stopping_patience: int = 10
    learning_rate: float = 0.0005
    threshold_percentile: float = 99.0
    random_seed: int = 42

    @property
    def sacr_features(self) -> List[str]:
        return list(self.base_features + self.sacrificial_features)

    @property
    def all_raw_features(self) -> List[str]:
        return self.sacr_features + [self.comm_quality_col]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> 'TrainingConfig':
        valid_fields = {f.name for f in fields(cls)}
        unknown_fields = sorted(set(values) - valid_fields)
        if unknown_fields:
            raise ValueError(f'알 수 없는 TrainingConfig 설정: {unknown_fields}')

        tuple_fields = {'base_features', 'sacrificial_devices', 'sacrificial_features'}
        normalized = {}
        for key, value in values.items():
            if key in tuple_fields and value is not None:
                normalized[key] = tuple(value)
            else:
                normalized[key] = value
        return cls(**normalized)

    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        config_path = Path(path)
        suffix = config_path.suffix.lower()
        with open(config_path, 'r', encoding='utf-8') as f:
            if suffix == '.json':
                values = json.load(f)
            elif suffix in {'.yaml', '.yml'}:
                try:
                    import yaml
                except ModuleNotFoundError as exc:
                    raise ModuleNotFoundError(
                        'YAML 설정 파일을 사용하려면 PyYAML을 설치하세요. '
                        '또는 JSON 설정 파일을 사용하세요.'
                    ) from exc
                values = yaml.safe_load(f) or {}
            else:
                raise ValueError('설정 파일은 .json, .yaml, .yml 중 하나여야 합니다.')

        if not isinstance(values, dict):
            raise ValueError('TrainingConfig 설정 파일의 최상위 값은 객체여야 합니다.')
        return cls.from_dict(values)

    def save_json(self, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


def load_training_config(path: Optional[str] = None) -> TrainingConfig:
    if path is None:
        return DEFAULT_CONFIG
    return TrainingConfig.load(path)


DEFAULT_CONFIG = TrainingConfig()

# 기존 함수 호환용 alias. 다음 리팩토링 단계에서 함수 인자로 밀어 넣어 제거한다.
BASE_FEATURES        = list(DEFAULT_CONFIG.base_features)
SACRIFICIAL_DEVICES  = list(DEFAULT_CONFIG.sacrificial_devices)
SACRIFICIAL_FEATURES = list(DEFAULT_CONFIG.sacrificial_features)
SACR_FEATURES        = DEFAULT_CONFIG.sacr_features

COMM_QUALITY_COL           = DEFAULT_CONFIG.comm_quality_col
COMM_QUALITY_THRESHOLD_DBM = DEFAULT_CONFIG.comm_quality_threshold_dbm
COMM_OUTAGE_CONSECUTIVE    = DEFAULT_CONFIG.comm_outage_consecutive

ALL_RAW_FEATURES = DEFAULT_CONFIG.all_raw_features

TRAIN_RATIO = DEFAULT_CONFIG.train_ratio
VAL_RATIO   = DEFAULT_CONFIG.val_ratio


# ============================================================
# 1. DB 로드
# ============================================================

def load_device_info_from_db() -> pd.DataFrame:
    """
    DB에서 장비 메타 정보(시설번호, 형식, 위치 등) 로드.

    반환 컬럼: 장비번호, 형식, 시설번호, 주소, 위도, 경도
    """
    sql = """
    SELECT
        t.SERIAL_NUM  AS 장비번호,
        t.TYPE        AS 형식,
        f.NUMBER      AS 시설번호,
        f.POSITION    AS 주소,
        f.LATITUDE    AS 위도,
        f.LONGITUDE   AS 경도
    FROM kscg_transmitter_info t
    LEFT JOIN kscg_facility_info f ON t.TRANSMITTER_ID = f.TRANSMITTER_ID
    WHERE t.SERIAL_NUM IS NOT NULL AND t.SERIAL_NUM != ''
    """
    df = read_sql(sql)
    df['장비번호'] = df['장비번호'].astype(str).str.strip()
    df = df[df['장비번호'].ne('')]
    # 시설번호가 있는 행을 우선으로 중복 제거
    df = df.sort_values('시설번호', na_position='last').drop_duplicates(subset=['장비번호'])
    return df.reset_index(drop=True)


def load_sensor_data_from_db(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    device_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    DB에서 센서 데이터를 wide format으로 로드.

    long format (행: 측정1건) → wide format (행: 측정시각×장비, 열: 센서명)
    반환 컬럼: 측정시각, 장비번호, 방식전위, 희생전류, AC유입, 온도, 습도, 통신품질
      - DB 방식전류  → 희생전류 (SENSOR_NAME_MAP 적용)
      - DB 수신감도  → 통신품질 (SENSOR_NAME_MAP 적용)
      - 배터리, 충격  → 매핑 없음, 제외
    """
    conditions = ["t.SERIAL_NUM IS NOT NULL", "t.SERIAL_NUM != ''"]
    params: List = []

    if start_date:
        conditions.append("sd.WRITE_DATE >= %s")
        params.append(start_date)
    if end_date:
        conditions.append("sd.WRITE_DATE <= %s")
        params.append(end_date)
    if device_ids:
        placeholders = ', '.join(['%s'] * len(device_ids))
        conditions.append(f"t.SERIAL_NUM IN ({placeholders})")
        params.extend(device_ids)

    where = ' AND '.join(conditions)
    sql = f"""
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
    long_df = read_sql(sql, params=params if params else None)

    # DB 센서명 → 코드 컬럼명 변환 (배터리·충격 등 미사용 센서는 제외)
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

    return wide_df.sort_values(['장비번호', '측정시각']).reset_index(drop=True)


def build_master_dataset_from_db(
    config: TrainingConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    DB에서 센서 데이터와 장비 메타 정보를 로드·병합하여 마스터 데이터셋 반환.

    반환 DataFrame:
        측정시각, 장비번호, 방식전위, 희생전류, AC유입, 온도, 습도, 통신품질,
        형식, 시설번호, 주소, 위도, 경도, 통신단절_플래그, 통신고장_플래그
    """
    print('  [DB] 센서 데이터 로드 중...')
    sensor_df = load_sensor_data_from_db(
        start_date=config.data_start_date,
        end_date=config.data_end_date,
    )
    print(f'  [DB] 완료: {len(sensor_df):,}행, {sensor_df["장비번호"].nunique()}개 장비')

    print('  [DB] 장비 메타 정보 로드 중...')
    info_df = load_device_info_from_db()

    df = sensor_df.merge(info_df, on='장비번호', how='left')
    df = apply_comm_quality_filter(df, config)
    return df


# ============================================================
# 2. 통신품질 룰 기반 필터
# ============================================================

def apply_comm_quality_filter(df: pd.DataFrame, config: TrainingConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    df = df.copy()
    if config.comm_quality_col not in df.columns:
        df['통신단절_플래그'] = 0
        df['통신고장_플래그'] = 0
        return df

    comm_vals              = pd.to_numeric(df[config.comm_quality_col], errors='coerce')
    df['통신단절_플래그'] = (comm_vals.le(config.comm_quality_threshold_dbm) | comm_vals.isna()).astype(int)

    def _fault_flag(sub: pd.DataFrame) -> pd.Series:
        sub_sorted = sub.sort_values('측정시각')
        flag       = sub_sorted['통신단절_플래그']
        changed    = (flag != flag.shift(fill_value=0)).cumsum()
        consec_len = flag.groupby(changed).transform('count')
        return ((flag == 1) & (consec_len >= config.comm_outage_consecutive)).astype(int)

    df['통신고장_플래그'] = (
        df.groupby('장비번호', group_keys=False).apply(_fault_flag)
    )
    return df


def get_sacrificial_device_data(df: pd.DataFrame, config: TrainingConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    sacrificial_devices = list(config.sacrificial_devices)
    sacrificial_features = list(config.sacrificial_features)
    sacr_df = df[df['장비번호'].isin(sacrificial_devices)].copy()
    if sacr_df.empty:
        return pd.DataFrame(columns=['측정시각', '장비번호'] + sacrificial_features)
    available = ['측정시각', '장비번호'] + [f for f in sacrificial_features if f in sacr_df.columns]
    return sacr_df[available].sort_values(['장비번호', '측정시각']).reset_index(drop=True)


# ============================================================
# 3. 공유 유틸리티
# ============================================================

def create_sequences(values: np.ndarray, time_steps: int) -> np.ndarray:
    if len(values) <= time_steps:
        raise ValueError(f'데이터 길이({len(values)})가 time_steps({time_steps})보다 짧습니다.')
    
    seqs = []
    for i in range(len(values) - time_steps):
        seq = values[i:i + time_steps]
        # NaN이 포함된 시퀀스는 학습에서 배제 (불연속 구간/이상치 제거)
        if not np.isnan(seq).any():
            seqs.append(seq)
            
    if not seqs:
        raise ValueError('유효한 시퀀스가 없습니다.')
        
    return np.array(seqs, dtype=np.float32)

def cleanse_data(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    통신 불량 및 통계적 이상치(IQR 기반)를 NaN으로 마스킹하여 
    시퀀스 생성 시 배제되도록 합니다.
    """
    df = df.copy()
    
    # 1. 통신 장애 구간 마스킹 (단절 및 고장 모두 배제)
    if '통신고장_플래그' in df.columns and '통신단절_플래그' in df.columns:
        mask_comm = (df['통신고장_플래그'] == 1) | (df['통신단절_플래그'] == 1)
        for col in features:
            if col in df.columns:
                df.loc[mask_comm, col] = np.nan
                
    # 2. 통계적 이상치 (IQR 기반) 마스킹
    for col in features:
        if col not in df.columns:
            continue
            
        def _mask_outliers(group):
            s = group.copy()
            Q1 = s.quantile(0.25)
            Q3 = s.quantile(0.75)
            IQR = Q3 - Q1
            # 1.5 * IQR 기준 (약 2.7 시그마)
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            s[(s < lower) | (s > upper)] = np.nan
            return s
            
        df[col] = df.groupby('장비번호')[col].transform(_mask_outliers)
        
    return df


# ============================================================
# 4. [공통 모델] 피처 엔지니어링
# ============================================================

def assign_normalization_group(df: pd.DataFrame) -> pd.DataFrame:
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
        base_features = list(DEFAULT_CONFIG.base_features)
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
        base_features = list(DEFAULT_CONFIG.base_features)
    cols = []
    for col in base_features:
        cols.extend([col, f'{col}_diff1', f'{col}_dev24'])
    return cols


# ============================================================
# 5. [공통 모델] 학습 데이터 준비
# ============================================================

def prepare_training_data(
    df: pd.DataFrame,
    time_steps: int = 24,
    min_points_per_device: int = 200,
    base_features: Optional[List[str]] = None,
    train_ratio: float = DEFAULT_CONFIG.train_ratio,
    val_ratio: float = DEFAULT_CONFIG.val_ratio,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, MinMaxScaler], pd.DataFrame, List[str]]:

    if base_features is None:
        base_features = list(DEFAULT_CONFIG.base_features)

    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError('train_ratio + val_ratio 합이 1.0 이상입니다.')

    min_per_split = time_steps + 1
    min_ratio     = min(train_ratio, val_ratio, test_ratio)
    min_required  = int(np.ceil(min_per_split / min_ratio))
    if min_points_per_device < min_required:
        print(f'  [경고] min_points_per_device({min_points_per_device}) → {min_required}으로 자동 상향')
        min_points_per_device = min_required

    df    = assign_normalization_group(df)
    df    = add_engineered_features(df, base_features)
    df    = cleanse_data(df, base_features)
    feats = feature_columns(base_features)

    group_train_data: Dict[str, List[np.ndarray]] = {}
    device_splits:    Dict[str, Tuple] = {}

    for device_id, ddf in df.groupby('장비번호'):
        ddf = ddf.sort_values('측정시각').reset_index(drop=True)
        if len(ddf) < min_points_per_device:
            continue
        n       = len(ddf)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        train_df = ddf.iloc[:n_train]
        val_df   = ddf.iloc[n_train:n_train + n_val]
        test_df  = ddf.iloc[n_train + n_val:]
        group    = str(ddf['정규화그룹'].iloc[0])

        group_train_data.setdefault(group, []).append(
            train_df[feats].astype(float).values
        )
        device_splits[str(device_id)] = (train_df, val_df, test_df, group)

    # 그룹별 스케일러 fit
    scaler_map: Dict[str, MinMaxScaler] = {}
    for group, arrays in group_train_data.items():
        combined = np.vstack(arrays)
        combined_valid = combined[~np.isnan(combined).any(axis=1)]
        scaler   = MinMaxScaler()
        if len(combined_valid) > 0:
            scaler.fit(combined_valid)
        else:
            scaler.fit(combined) # Fallback
        scaler_map[group] = scaler

    train_seqs, val_seqs, test_seqs, test_frames = [], [], [], []

    for device_id, (tr_df, va_df, te_df, group) in device_splits.items():
        if group not in scaler_map:
            continue
        sc = scaler_map[group]

        tr_scaled = sc.transform(tr_df[feats].astype(float).values)
        va_scaled = sc.transform(va_df[feats].astype(float).values)
        te_scaled = sc.transform(te_df[feats].astype(float).values) if not te_df.empty else None

        try:
            train_seqs.append(create_sequences(tr_scaled, time_steps))
            val_seqs.append(create_sequences(va_scaled, time_steps))
            if te_scaled is not None:
                test_seqs.append(create_sequences(te_scaled, time_steps))
                test_frames.append(te_df)
        except ValueError:
            continue

    if not train_seqs:
        raise ValueError('학습 데이터를 만들 수 없습니다.')

    X_train  = np.concatenate(train_seqs, axis=0)
    X_val    = np.concatenate(val_seqs,   axis=0) if val_seqs  else X_train[:1]
    X_test   = np.concatenate(test_seqs,  axis=0) if test_seqs else X_val
    test_df_all = pd.concat(test_frames, ignore_index=True) if test_frames else pd.DataFrame()

    return X_train, X_val, X_test, scaler_map, test_df_all, feats


# ============================================================
# 6. [공통 모델] 모델 아키텍처
# ============================================================

def build_common_model(
    n_timesteps: int,
    n_features: int,
    learning_rate: float = DEFAULT_CONFIG.learning_rate,
) -> Sequential:
    model = Sequential([
        LSTM(128, input_shape=(n_timesteps, n_features), return_sequences=True),
        LSTM(64,  return_sequences=False),
        RepeatVector(n_timesteps),
        LSTM(64,  return_sequences=True),
        LSTM(128, return_sequences=True),
        TimeDistributed(Dense(n_features)),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model


# ============================================================
# 7. [공통 모델] Threshold / 평가 / 저장
# ============================================================

def inject_faults(X: np.ndarray, fault_ratio: float = 0.05, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    정상 데이터 시퀀스(X)에 의도적으로 이상 패턴(Spike, Drop, Drift)을 주입합니다.
    """
    np.random.seed(seed)
    X_fault = X.copy()
    y_true = np.zeros(len(X))
    
    n_faults = int(len(X) * fault_ratio)
    if n_faults == 0:
        return X_fault, y_true
        
    indices = np.random.choice(len(X), n_faults, replace=False)
    
    for idx in indices:
        y_true[idx] = 1
        # 임의의 피처 하나에 이상 주입
        feat_idx = np.random.randint(0, X.shape[2])
        fault_type = np.random.choice(['spike', 'drop', 'drift'])
        
        if fault_type == 'spike':
            X_fault[idx, -1, feat_idx] += np.random.uniform(0.5, 0.8)
        elif fault_type == 'drop':
            X_fault[idx, -1, feat_idx] -= np.random.uniform(0.5, 0.8)
        elif fault_type == 'drift':
            drift = np.linspace(0, np.random.uniform(0.3, 0.5), X.shape[1])
            X_fault[idx, :, feat_idx] += drift
                
    return X_fault, y_true

def tune_threshold_f1(model, X_test: np.ndarray, fault_ratio: float = 0.1) -> Tuple[float, Dict[str, float]]:
    """
    의도적 오류 주입을 통해 F1-Score를 극대화하는 최적의 Threshold를 탐색합니다.
    """
    # 1. 오류 주입 데이터 생성
    X_fault, y_fault_true = inject_faults(X_test, fault_ratio=fault_ratio)
    
    # 2. 전체 평가 셋 구성 (원본 Normal + Faulty)
    X_eval = np.concatenate([X_test, X_fault], axis=0)
    y_eval = np.concatenate([np.zeros(len(X_test)), y_fault_true], axis=0)
    
    # 3. MSE 계산
    pred = model.predict(X_eval, verbose=0)
    mse = np.mean(np.power(X_eval - pred, 2), axis=(1, 2))
    
    best_f1 = -1
    best_thr = 0.0
    best_metrics = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # MSE 분포 기반 탐색 (p90 ~ p99.9 범위 탐색)
    search_thrs = np.percentile(mse, np.linspace(90, 99.9, 50))
    
    for thr in search_thrs:
        y_pred = (mse > thr).astype(int)
        
        tp = np.sum((y_pred == 1) & (y_eval == 1))
        fp = np.sum((y_pred == 1) & (y_eval == 0))
        fn = np.sum((y_pred == 0) & (y_eval == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 >= best_f1:
            best_f1 = f1
            best_thr = thr
            best_metrics = {'precision': precision, 'recall': recall, 'f1': f1}
            
    return float(best_thr), best_metrics

def compute_device_thresholds(
    model, test_df: pd.DataFrame, feats: List[str],
    scaler_map: Dict, time_steps: int = 24, percentile: float = 99.0,
) -> Dict[str, float]:
    thresholds: Dict[str, float] = {}
    for device_id, ddf in test_df.groupby('장비번호'):
        group_name = str(ddf['정규화그룹'].iloc[0]) if '정규화그룹' in ddf.columns else 'ALL'
        if group_name not in scaler_map:
            continue
        sc = scaler_map[group_name]
        values = ddf.sort_values('측정시각')[feats].astype(float).values
        scaled = sc.transform(values)
        if len(scaled) <= time_steps:
            continue
        try:
            X = create_sequences(scaled, time_steps)
        except ValueError:
            continue
        pred = model.predict(X, verbose=0)
        mse  = np.mean(np.power(X - pred, 2), axis=(1, 2))
        thresholds[str(device_id)] = float(np.percentile(mse, percentile))
    return thresholds


def save_common_artifacts(
    save_dir: str, model, scaler_map: Dict, thresholds: Dict,
    feats: List[str], time_steps: int, config: TrainingConfig = DEFAULT_CONFIG,
    metrics: Optional[Dict] = None,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, 'common_lstm_autoencoder.keras'))
    with open(os.path.join(save_dir, 'group_scalers.pkl'), 'wb') as f:
        pickle.dump(scaler_map, f)
    with open(os.path.join(save_dir, 'device_thresholds.json'), 'w', encoding='utf-8') as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2)
    artifact_config = {
        'time_steps'            : time_steps,
        'base_features'         : list(config.base_features),
        'feature_columns'       : feats,
        'sacrificial_devices'   : list(config.sacrificial_devices),
        'sacrificial_features'  : list(config.sacrificial_features),
        'comm_quality_col'                  : config.comm_quality_col,
        'comm_quality_threshold_dbm'        : config.comm_quality_threshold_dbm,
        'comm_outage_consecutive_threshold' : config.comm_outage_consecutive,
        'best_tuned_threshold'              : metrics.get('best_tuned_threshold') if metrics else None,
    }
    with open(os.path.join(save_dir, 'model_config.json'), 'w', encoding='utf-8') as f:
        json.dump(artifact_config, f, ensure_ascii=False, indent=2)
    print(f'  아티팩트 저장 완료 → {save_dir}')


def evaluate_model(model, X_test: np.ndarray, thresholds: Dict) -> Dict[str, Any]:
    """
    기존 MSE 통계와 더불어 Fault Injection 기반의 F1-Score 평가를 수행합니다.
    """
    pred        = model.predict(X_test, verbose=0)
    mse_per_seq = np.mean(np.power(X_test - pred, 2), axis=(1, 2))
    metrics     = {
        'test_mse_mean': float(np.mean(mse_per_seq)),
        'test_mse_std' : float(np.std(mse_per_seq)),
        'test_mse_p50' : float(np.percentile(mse_per_seq, 50)),
        'test_mse_p95' : float(np.percentile(mse_per_seq, 95)),
        'test_mse_p99' : float(np.percentile(mse_per_seq, 99)),
    }
    
    # 1. Fault Injection 기반 F1-Score 평가 및 최적 Threshold 탐색
    print('\n  [평가] 의도적 오류 주입 기반 F1-Score 측정 중...')
    best_thr, f1_metrics = tune_threshold_f1(model, X_test)
    metrics['best_tuned_threshold'] = best_thr
    metrics.update({f'tuned_{k}': v for k, v in f1_metrics.items()})
    
    if thresholds:
        mean_thr = float(np.mean(list(thresholds.values())))
        metrics['mean_device_threshold']          = mean_thr
        metrics['anomaly_rate_vs_mean_threshold'] = float(np.mean(mse_per_seq > mean_thr))
    return metrics


def _print_metrics(metrics: Dict[str, Any]) -> None:
    print('  ┌─────────────────────────────────────────┐')
    print(f'  │  테스트 MSE 평균     : {metrics["test_mse_mean"]:.6f}')
    print(f'  │  테스트 MSE p99      : {metrics["test_mse_p99"]:.6f}')
    print('  ├─────────────────────────────────────────┤')
    if 'tuned_f1' in metrics:
        print(f'  │  최적 tuned threshold: {metrics["best_tuned_threshold"]:.6f}')
        print(f'  │  Precision (정밀도)  : {metrics["tuned_precision"]:.2%}')
        print(f'  │  Recall (재현율)     : {metrics["tuned_recall"]:.2%}')
        print(f'  │  F1-Score (종합)     : {metrics["tuned_f1"]:.2%}')
    print('  ├─────────────────────────────────────────┤')
    if 'mean_device_threshold' in metrics:
        print(f'  │  장비별 평균 thr     : {metrics["mean_device_threshold"]:.6f}')
        print(f'  │  이상 의심 비율      : {metrics["anomaly_rate_vs_mean_threshold"]:.2%}')
    print('  └─────────────────────────────────────────┘')


# ============================================================
# 8. [희생양극 모델] 피처 엔지니어링
# ============================================================

def add_engineered_features_sacrificial(
    df: pd.DataFrame,
    config: TrainingConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    df = df.copy().sort_values(['장비번호', '측정시각']).reset_index(drop=True)
    sacr_features = config.sacr_features

    for col in sacr_features:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = (
            df.groupby('장비번호')[col]
            .transform(lambda s: s.interpolate(method='linear', limit_direction='both').ffill().bfill())
        )

    for col in sacr_features:
        df[f'{col}_diff1'] = df.groupby('장비번호')[col].diff().fillna(0)
        df[f'{col}_ma24']  = df.groupby('장비번호')[col].transform(
            lambda s: s.rolling(window=24, min_periods=1).mean()
        )
        df[f'{col}_dev24'] = df[col] - df[f'{col}_ma24']

    return df


def feature_columns_sacrificial(config: TrainingConfig = DEFAULT_CONFIG) -> List[str]:
    """SACR_FEATURES 5개 × 3 파생 = 15개 컬럼"""
    cols = []
    for col in config.sacr_features:
        cols.extend([col, f'{col}_diff1', f'{col}_dev24'])
    return cols


# ============================================================
# 9. [희생양극 모델] 학습 데이터 준비
# ============================================================

def prepare_sacrificial_training_data(
    master_df: pd.DataFrame,
    time_steps: int = 24,
    min_points: int = 50,
    config: TrainingConfig = DEFAULT_CONFIG,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, MinMaxScaler], pd.DataFrame, List[str]]:

    sacrificial_devices = list(config.sacrificial_devices)
    sacr_df = master_df[master_df['장비번호'].isin(sacrificial_devices)].copy()
    if sacr_df.empty:
        raise ValueError(f'희생양극 기기 데이터 없음: {sacrificial_devices}')

    sacr_df = add_engineered_features_sacrificial(sacr_df, config)
    sacr_df = cleanse_data(sacr_df, config.sacr_features)
    feats   = feature_columns_sacrificial(config)

    train_seqs, val_seqs, test_seqs_list, test_frames = [], [], [], []
    scaler_map: Dict[str, MinMaxScaler] = {}

    for device_id in sacrificial_devices:
        ddf = sacr_df[sacr_df['장비번호'] == device_id].sort_values('측정시각').copy()
        if '희생전류' in ddf.columns:
            ddf = ddf.dropna(subset=['희생전류'])

        if len(ddf) < min_points:
            print(f'  [SKIP] {device_id}: 데이터 부족 ({len(ddf)}행, 최소 {min_points}행)')
            continue

        n        = len(ddf)
        n_train  = int(n * config.train_ratio)
        n_val    = int(n * config.val_ratio)
        train_df = ddf.iloc[:n_train].copy()
        val_df   = ddf.iloc[n_train:n_train + n_val].copy()
        test_df  = ddf.iloc[n_train + n_val:].copy()

        scaler       = MinMaxScaler()
        train_vals   = train_df[feats].astype(float).values
        train_valid  = train_vals[~np.isnan(train_vals).any(axis=1)]
        if len(train_valid) > 0:
            scaler.fit(train_valid)
        else:
            scaler.fit(train_vals) # Fallback
            
        train_scaled = scaler.transform(train_vals)
        val_scaled   = scaler.transform(val_df[feats].astype(float).values)
        test_scaled  = scaler.transform(test_df[feats].astype(float).values)
        scaler_map[device_id] = scaler

        try:
            train_seqs.append(create_sequences(train_scaled, time_steps))
            val_seqs.append(create_sequences(val_scaled,     time_steps))
            test_seqs_list.append(create_sequences(test_scaled, time_steps))
            test_frames.append(test_df)
        except ValueError as e:
            print(f'  [SKIP] {device_id} 시퀀스 오류: {e}')
            continue

        print(
            f'  - {device_id}: train {len(train_df)}행 / '
            f'val {len(val_df)}행 / test {len(test_df)}행'
        )

    if not train_seqs:
        raise ValueError('희생양극 학습 데이터 없음. 데이터를 확인하세요.')

    X_train  = np.concatenate(train_seqs,     axis=0)
    X_val    = np.concatenate(val_seqs,       axis=0) if val_seqs       else X_train[:1]
    X_test   = np.concatenate(test_seqs_list, axis=0) if test_seqs_list else X_val
    test_all = pd.concat(test_frames, ignore_index=True) if test_frames else pd.DataFrame()

    return X_train, X_val, X_test, scaler_map, test_all, feats


# ============================================================
# 10. [희생양극 모델] 모델 아키텍처
# ============================================================

def build_sacrificial_model(
    n_timesteps: int,
    n_features: int,
    learning_rate: float = DEFAULT_CONFIG.learning_rate,
) -> Sequential:
    """
    공통 모델보다 작은 구조 (데이터 규모: 2개 장비).
    인코더 LSTM(64→32) / 디코더 LSTM(32→64).
    """
    model = Sequential([
        LSTM(64, input_shape=(n_timesteps, n_features), return_sequences=True),
        LSTM(32, return_sequences=False),
        RepeatVector(n_timesteps),
        LSTM(32, return_sequences=True),
        LSTM(64, return_sequences=True),
        TimeDistributed(Dense(n_features)),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model


# ============================================================
# 11. [희생양극 모델] Threshold / 평가 / 저장
# ============================================================

def compute_sacrificial_thresholds(
    model, test_df: pd.DataFrame, feats: List[str],
    scaler_map: Dict, time_steps: int, percentile: float = 99.0,
    config: TrainingConfig = DEFAULT_CONFIG,
) -> Dict[str, float]:
    thresholds: Dict[str, float] = {}
    for device_id in config.sacrificial_devices:
        if device_id not in scaler_map:
            continue
        ddf = test_df[test_df['장비번호'] == device_id].sort_values('측정시각').copy()
        if len(ddf) <= time_steps:
            print(f'  [SKIP] {device_id}: test 데이터 부족')
            continue
        scaled = scaler_map[device_id].transform(ddf[feats].astype(float).values)
        try:
            X = create_sequences(scaled, time_steps)
        except ValueError as e:
            print(f'  [SKIP] {device_id}: {e}')
            continue
        pred  = model.predict(X, verbose=0)
        mse   = np.mean(np.power(X - pred, 2), axis=(1, 2))
        thresholds[device_id] = float(np.percentile(mse, percentile))
        print(f'  - {device_id}: threshold={thresholds[device_id]:.6f} (p{percentile:.0f})')
    return thresholds


def save_sacrificial_artifacts(
    save_dir: str, model, scaler_map: Dict, thresholds: Dict,
    feats: List[str], time_steps: int, config: TrainingConfig = DEFAULT_CONFIG,
    metrics: Optional[Dict] = None,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, 'sacrificial_lstm_autoencoder.keras'))
    with open(os.path.join(save_dir, 'sacrificial_device_scalers.pkl'), 'wb') as f:
        pickle.dump(scaler_map, f)
    with open(os.path.join(save_dir, 'sacrificial_thresholds.json'), 'w', encoding='utf-8') as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2)
    artifact_config = {
        'time_steps'           : time_steps,
        'feature_columns'      : feats,
        'base_features'        : list(config.base_features),
        'sacrificial_features' : list(config.sacrificial_features),
        'all_features'         : config.sacr_features,
        'sacrificial_devices'  : list(config.sacrificial_devices),
        'model_type'           : 'sacrificial_lstm_autoencoder',
        'scaler_type'          : 'per_device',
        'best_tuned_threshold' : metrics.get('best_tuned_threshold') if metrics else None,
    }
    with open(os.path.join(save_dir, 'sacrificial_model_config.json'), 'w', encoding='utf-8') as f:
        json.dump(artifact_config, f, ensure_ascii=False, indent=2)
    print(f'  아티팩트 저장 완료 → {save_dir}')


# ============================================================
# 12. 통합 main()
# ============================================================

def main(config: TrainingConfig = DEFAULT_CONFIG):
    base_dir        = Path(__file__).resolve().parent
    common_save_dir = base_dir / config.common_artifact_dir
    sacr_save_dir   = base_dir / config.sacrificial_artifact_dir

    np.random.seed(config.random_seed)
    tf.random.set_seed(config.random_seed)

    # ══════════════════════════════════════════════════════════
    # STEP 1 ─ DB에서 데이터 로드 (공통 모델 + 희생양극 모델이 공유)
    # ══════════════════════════════════════════════════════════
    print('=' * 55)
    print(' STEP 1 : DB 데이터 로드')
    print('=' * 55)
    master_df = build_master_dataset_from_db(config)
    print(f'  전체 행 수: {len(master_df):,}  |  장비 수: {master_df["장비번호"].nunique()}')

    if '통신단절_플래그' in master_df.columns:
        oc  = master_df['통신단절_플래그'].sum()
        fc  = master_df['통신고장_플래그'].sum()
        tot = len(master_df)
        print(f'  통신단절: {oc:,}건({oc/tot:.1%}) | 고장: {fc:,}건 | 일시장애: {oc-fc:,}건')

    sacr_df = get_sacrificial_device_data(master_df, config)
    if not sacr_df.empty:
        print(f'  희생양극 데이터: {list(config.sacrificial_devices)} → {len(sacr_df):,}행')

    # ══════════════════════════════════════════════════════════
    # STEP 2-6 ─ 공통 LSTM AutoEncoder 학습
    # ══════════════════════════════════════════════════════════
    print('\n' + '=' * 55)
    print(' STEP 2-6 : 공통 모델 학습 (BASE_FEATURES 4개)')
    print('=' * 55)

    # [2] 데이터 준비
    print('\n[2] Train/Val/Test 분리 + Scaler 학습 중...')
    time_steps = config.time_steps
    X_train, X_val, X_test, c_scaler_map, c_test_df, c_feats = prepare_training_data(
        master_df,
        time_steps=time_steps,
        min_points_per_device=config.min_points_per_device,
        base_features=list(config.base_features),
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
    )
    print(f'  X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}')

    # [3] 모델 학습
    print('\n[3] 공통 LSTM 학습 중...')
    c_model = build_common_model(X_train.shape[1], X_train.shape[2], learning_rate=config.learning_rate)
    c_model.fit(
        X_train, X_train,
        epochs=config.common_epochs, batch_size=config.common_batch_size,
        validation_data=(X_val, X_val),
        callbacks=[EarlyStopping(
            monitor='val_loss',
            patience=config.common_early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        )],
        shuffle=True, verbose=1,
    )

    # [4] Threshold
    print('\n[4] 장비별 threshold 계산 중...')
    c_thresholds = compute_device_thresholds(
        c_model,
        c_test_df,
        c_feats,
        c_scaler_map,
        time_steps,
        percentile=config.threshold_percentile,
    )
    print(f'  threshold 생성 장비: {len(c_thresholds)}개')

    # [5] 평가
    print('\n[5] 공통 모델 평가 중...')
    c_metrics = evaluate_model(c_model, X_test, c_thresholds)
    _print_metrics(c_metrics)

    # [6] 저장
    print('\n[6] 공통 모델 아티팩트 저장 중...')
    save_common_artifacts(str(common_save_dir), c_model, c_scaler_map, c_thresholds, c_feats, time_steps, config, metrics=c_metrics)
    c_test_df.to_parquet(str(common_save_dir / 'test_data.parquet'), index=False)
    os.makedirs(str(common_save_dir), exist_ok=True)
    with open(str(common_save_dir / 'eval_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(c_metrics, f, ensure_ascii=False, indent=2)

    # ══════════════════════════════════════════════════════════
    # STEP 7-11 ─ 희생양극 전용 LSTM AutoEncoder 학습
    # ══════════════════════════════════════════════════════════
    print('\n' + '=' * 55)
    print(' STEP 7-11 : 희생양극 전용 모델 학습 (SACR_FEATURES 5개)')
    print('=' * 55)

    # [7] 데이터 준비
    print('\n[7] 희생양극 전용 Train/Val/Test 분리 중...')
    s_X_train, s_X_val, s_X_test, s_scaler_map, s_test_df, s_feats = prepare_sacrificial_training_data(
        master_df,
        time_steps=time_steps,
        min_points=config.sacrificial_min_points,
        config=config,
    )
    print(f'  X_train: {s_X_train.shape} | X_val: {s_X_val.shape} | X_test: {s_X_test.shape}')
    print(f'  피처: {len(s_feats)}개 ({config.sacr_features} × 3)')

    # [8] 모델 학습
    print('\n[8] 희생양극 LSTM 학습 중...')
    s_model = build_sacrificial_model(s_X_train.shape[1], s_X_train.shape[2], learning_rate=config.learning_rate)
    s_model.fit(
        s_X_train, s_X_train,
        epochs=config.sacrificial_epochs, batch_size=config.sacrificial_batch_size,
        validation_data=(s_X_val, s_X_val),
        callbacks=[EarlyStopping(
            monitor='val_loss',
            patience=config.sacrificial_early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        )],
        shuffle=True, verbose=1,
    )

    # [9] Threshold
    print('\n[9] 희생양극 장비별 threshold 계산 중...')
    s_thresholds = compute_sacrificial_thresholds(
        s_model,
        s_test_df,
        s_feats,
        s_scaler_map,
        time_steps,
        percentile=config.threshold_percentile,
        config=config,
    )

    # [10] 평가
    print('\n[10] 희생양극 모델 평가 중...')
    s_metrics = evaluate_model(s_model, s_X_test, s_thresholds)
    _print_metrics(s_metrics)

    # [11] 저장
    print('\n[11] 희생양극 모델 아티팩트 저장 중...')
    save_sacrificial_artifacts(str(sacr_save_dir), s_model, s_scaler_map, s_thresholds, s_feats, time_steps, config, metrics=s_metrics)
    s_test_df.to_parquet(str(sacr_save_dir / 'test_data_sacrificial.parquet'), index=False)
    with open(str(sacr_save_dir / 'eval_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(s_metrics, f, ensure_ascii=False, indent=2)

    # ══════════════════════════════════════════════════════════
    # 최종 요약
    # ══════════════════════════════════════════════════════════
    print('\n' + '=' * 55)
    print(' 학습 완료 요약')
    print('=' * 55)
    print(f'  [공통 모델]    threshold 생성 장비: {len(c_thresholds)}개')
    print(f'  [희생양극 모델] threshold 생성 장비: {len(s_thresholds)}개')
    print()
    print(f'  {config.common_artifact_dir}/')
    print(f'    common_lstm_autoencoder.keras  (LSTM 128-64)')
    print(f'    group_scalers.pkl  (그룹별 스케일러)')
    print(f'    device_thresholds.json  ({len(c_thresholds)}개 장비)')
    print()
    print(f'  {config.sacrificial_artifact_dir}/')
    print(f'    sacrificial_lstm_autoencoder.keras  (LSTM 64-32)')
    print(f'    sacrificial_device_scalers.pkl  (장비별 스케일러)')
    print(f'    sacrificial_thresholds.json  ({len(s_thresholds)}개 장비)')
    print()
    print(f'  [설계 정책]')
    print(f'    공통 모델  : {list(config.base_features)} → {len(c_feats)}개 피처')
    print(f'    희생양극  : {config.sacr_features} → {len(s_feats)}개 피처')
    print(f'    희생양극 스케일러: 장비별 개별 (공통 모델은 그룹 기반)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='가스 시설물 이상 탐지 통합 학습')
    parser.add_argument(
        '--config',
        help='TrainingConfig JSON/YAML 파일 경로',
    )
    args = parser.parse_args()
    main(load_training_config(args.config))
