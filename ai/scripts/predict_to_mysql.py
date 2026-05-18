"""
predict_to_mysql.py
─────────────────────────────────────────────────────────────────────
박지훈 wrapper — siwon MySQL → LSTM AE 추론 → ai_predictions INSERT.

기존 gas_common_model_predict.py 의 predict_device_window() 를 그대로 활용.
artifacts 폴더 통합 (ai/models + ai/config) → 임시 작업 폴더로 복사 후 load.

사용:
  export SIWON_DB_HOST=127.0.0.1 SIWON_DB_PORT=3306 \\
         SIWON_DB_USER=siwon_app SIWON_DB_PASS=... SIWON_DB_NAME=siwon
  python3 predict_to_mysql.py [--hours 48] [--site-id 2] [--dry-run]

옵션:
  --hours    예측 입력 시계열 시점 수 (LSTM time_steps 보다 큼). 기본 48.
  --site-id  KSCG SITE_ID. 기본 2 (군산).
  --dry-run  INSERT 안 하고 결과만 print
"""
import os
import sys
import json
import argparse
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pymysql

# 이 스크립트의 폴더 = ai/scripts/. 다른 모듈 (predict.py) 도 같은 폴더.
SCRIPT_DIR = Path(__file__).resolve().parent

# gas_common_model_predict 모듈 동적 import — 파일명에 공백/괄호 있어서 importlib 필요
import importlib.util
PREDICT_PY = SCRIPT_DIR / 'gas_common_model_predict(2026.05.03).py'
spec = importlib.util.spec_from_file_location('predict_mod', PREDICT_PY)
predict_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predict_mod)

load_artifacts        = predict_mod.load_artifacts
predict_device_window = predict_mod.predict_device_window
BASE_FEATURES         = predict_mod.BASE_FEATURES

AI_ROOT     = SCRIPT_DIR.parent
MODELS_DIR  = AI_ROOT / 'models'
CONFIG_DIR  = AI_ROOT / 'config'

# seq → kind (server.js 와 동일 매핑)
SENSOR_SEQ_KIND = ['volt', 'sacrificial', 'ac', 'battery', 'temp', 'hum', 'shock', 'commDbm']
# AI 모델 input — 한국어 컬럼명으로 변환
KIND_TO_KR = {
    'volt':    '방식전위',
    'ac':      'AC유입',
    'temp':    '온도',
    'hum':     '습도',
    'commDbm': '통신품질',
}


def merge_artifacts_to_temp():
    """ai/models 와 ai/config 의 4 파일을 임시 폴더에 모아 load_artifacts() 호출 가능하게."""
    tmp = Path(tempfile.mkdtemp(prefix='siwon_lstm_'))
    shutil.copy(MODELS_DIR / 'common_lstm_autoencoder.keras', tmp / 'common_lstm_autoencoder.keras')
    shutil.copy(MODELS_DIR / 'group_scalers.pkl',              tmp / 'group_scalers.pkl')
    shutil.copy(CONFIG_DIR / 'device_thresholds.json',         tmp / 'device_thresholds.json')
    shutil.copy(CONFIG_DIR / 'model_config.json',              tmp / 'model_config.json')
    return tmp


def connect_db():
    return pymysql.connect(
        host=os.environ.get('SIWON_DB_HOST', '127.0.0.1'),
        port=int(os.environ.get('SIWON_DB_PORT', '3306')),
        user=os.environ.get('SIWON_DB_USER', 'siwon_app'),
        password=os.environ['SIWON_DB_PASS'],
        database=os.environ.get('SIWON_DB_NAME', 'siwon'),
        autocommit=False,
    )


def load_device_windows(conn, site_id: int, hours: int):
    """단말별 최근 N시간 시계열 → {device_id: DataFrame}.

    DataFrame columns: 장비번호, 측정시각, 방식전위, AC유입, 온도, 습도, 통신품질
    """
    cur = conn.cursor(pymysql.cursors.DictCursor)
    # 단말 목록 + TRANSMITTER_ID
    cur.execute("""
        SELECT t.NAME AS device_id, t.TRANSMITTER_ID AS txid
        FROM kscg_transmitter_info t
        JOIN kscg_site_mydevice m ON m.TRANSMITTER_ID = t.TRANSMITTER_ID AND m.SITE_ID = %s
        ORDER BY t.TRANSMITTER_ID
    """, (site_id,))
    devices = cur.fetchall()
    print(f'[load] 단말 {len(devices)}대 발견')

    # 단말당 SENSOR_ID seq → kind 매핑
    cur.execute("""
        SELECT TRANSMITTER_ID AS txid, SENSOR_ID AS sid,
               ROW_NUMBER() OVER (PARTITION BY TRANSMITTER_ID ORDER BY SENSOR_ID) AS seq
        FROM kscg_sensor_info
    """)
    sensor_map = {}   # txid -> {kind: sid}
    for r in cur.fetchall():
        txid = r['txid']
        kind = SENSOR_SEQ_KIND[r['seq'] - 1] if r['seq'] - 1 < len(SENSOR_SEQ_KIND) else None
        if not kind:
            continue
        sensor_map.setdefault(txid, {})[kind] = r['sid']

    # 단말별 시계열 로드 (한 번에 N 단말씩 — 가벼움)
    out = {}
    for d in devices:
        device_id = d['device_id']
        txid      = d['txid']
        sids_by_kind = sensor_map.get(txid, {})
        # 4 + 1 (통신품질) 센서 데이터를 한번에 LONG → WIDE pivot
        needed_kinds = ['volt', 'ac', 'temp', 'hum', 'commDbm']
        sid_to_kind  = {sids_by_kind[k]: k for k in needed_kinds if k in sids_by_kind}
        if len(sid_to_kind) < 4:
            print(f'  {device_id}: 센서 매핑 부족 ({len(sid_to_kind)}/4 필수) — skip')
            continue

        placeholders = ','.join(['%s'] * len(sid_to_kind))
        cur.execute(f"""
            SELECT WRITE_DATE AS t, SENSOR_ID AS sid, VALUE AS v
            FROM kscg_sensor_data
            WHERE SENSOR_ID IN ({placeholders})
              AND WRITE_DATE > DATE_SUB(NOW(), INTERVAL %s HOUR)
            ORDER BY WRITE_DATE
        """, list(sid_to_kind.keys()) + [hours])
        rows = cur.fetchall()
        if not rows:
            continue

        # long → wide
        long_df = pd.DataFrame(rows)
        long_df['kind'] = long_df['sid'].map(sid_to_kind)
        long_df['v']    = pd.to_numeric(long_df['v'], errors='coerce')
        wide = long_df.pivot_table(index='t', columns='kind', values='v', aggfunc='first').reset_index()
        wide = wide.rename(columns={'t': '측정시각', **KIND_TO_KR})
        wide.insert(0, '장비번호', device_id)

        # BASE_FEATURES 모두 있어야 함
        if any(f not in wide.columns for f in BASE_FEATURES):
            continue

        wide = wide.sort_values('측정시각').reset_index(drop=True)
        out[device_id] = wide

    return out, devices


def insert_predictions(conn, results: list, dry_run: bool):
    if dry_run:
        print(f'[dry-run] INSERT 안 함. {len(results)}건 결과만 출력.')
        return
    cur = conn.cursor()
    # 같은 단말 재 INSERT 가능 — predicted_at 으로 구분
    sql = """
        INSERT INTO ai_predictions
          (transmitter_id, predicted_at, mse, threshold, risk_level,
           comm_status, ai_reliability, feature_contributions, is_sacrificial_device)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    inserted = 0
    for r in results:
        try:
            cur.execute(sql, (
                r['txid'],
                r['predicted_at'],
                r['mse'],
                r['threshold'],
                r['risk_level'],
                r['comm_status'],
                r['ai_reliability'],
                json.dumps(r.get('feature_contributions') or {}),
                1 if r.get('is_sacrificial_device') else 0,
            ))
            inserted += 1
        except Exception as e:
            print(f"  [insert err] {r.get('device_id')}: {e}")
    conn.commit()
    print(f'[insert] ai_predictions {inserted}건 INSERT 완료')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hours', type=int, default=48, help='시계열 시점 수 (LSTM time_steps + 여유)')
    ap.add_argument('--site-id', type=int, default=2)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    # 1) artifacts 통합 + load
    print('[1] artifacts 통합 + 모델 로드')
    tmp_dir = merge_artifacts_to_temp()
    try:
        model, scaler_map, thresholds, config = load_artifacts(str(tmp_dir))
        time_steps   = config['time_steps']
        feature_cols = config['feature_columns']
        print(f'  - time_steps={time_steps}, 피처 수={len(feature_cols)}, 등록 장비={len(thresholds)}')

        # 2) DB 시계열 로드
        print('[2] siwon DB 시계열 로드')
        conn = connect_db()
        try:
            windows, devices = load_device_windows(conn, args.site_id, args.hours)
            print(f'  - 입력 가능 단말: {len(windows)}대')

            # txid 매핑
            txid_by_name = {d['device_id']: d['txid'] for d in devices}

            # 3) 단말별 predict
            print('[3] 단말별 LSTM 추론')
            now_pred = datetime.now()
            results = []
            for device_id, ddf in windows.items():
                if len(ddf) < time_steps + 1:
                    print(f'  {device_id}: 시점 부족 {len(ddf)}/{time_steps+1} — skip')
                    continue
                try:
                    res = predict_device_window(ddf, model, scaler_map, thresholds, feature_cols, time_steps)
                    res['device_id']    = device_id
                    res['txid']         = txid_by_name[device_id]
                    res['predicted_at'] = now_pred
                    results.append(res)
                except Exception as e:
                    print(f'  {device_id}: 추론 실패 — {e}')

            # 4) 요약
            print('[4] 결과 요약')
            from collections import Counter
            risk_counts = Counter(r['risk_level'] for r in results)
            comm_counts = Counter(r['comm_status'] for r in results)
            print(f'  - 추론 성공: {len(results)} / {len(windows)} 단말')
            print(f'  - 위험도: {dict(risk_counts)}')
            print(f'  - 통신상태: {dict(comm_counts)}')

            # 위험·이상 단말 list
            interesting = [r for r in results if r['risk_level'] in ('이상', '관찰')]
            if interesting:
                print(f"  - 흥미로운 단말 {len(interesting)}대:")
                for r in interesting[:10]:
                    print(f"    {r['device_id']:18} risk={r['risk_level']:4} mse={r['mse']:.6f} th={r['threshold']:.6f} comm={r['comm_status']}")

            # 5) INSERT
            print('[5] ai_predictions INSERT')
            insert_predictions(conn, results, args.dry_run)
        finally:
            conn.close()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
