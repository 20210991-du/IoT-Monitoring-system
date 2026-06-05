#!/usr/bin/env python3
"""
backfill_predictions.py — 과거 시점 LSTM-AE 추론을 ai_predictions 에 백필 (source='backfill').

predict_to_mysql.py 와 동일 모델/추론(predict_device_window) 사용 — 모델 재학습 없음.
단말별 전체 이력을 한 번 로드 후 window-hours 윈도우를 시간순으로 슬라이드하며 'as-of T' 예측.
predict_device_window 는 마지막 시퀀스(끝점) 1개의 mse 를 반환하므로, T 에서 끝나는 윈도우 = 그 시점 값.

재실행 안전: 같은 (단말, 기간)의 source='backfill' 행을 먼저 DELETE 후 재삽입.
→ 이상탐지 알고리즘/기준점이 바뀌면 이 스크립트만 다시 돌리면 전체 history 가 새 모델로 재정렬됨.
  source='live'(실시간 감사 기록)은 절대 건드리지 않음.

사용:
  export SIWON_DB_HOST=... SIWON_DB_PORT=... SIWON_DB_USER=... SIWON_DB_PASS=... SIWON_DB_NAME=...
  python3 backfill_predictions.py --device TB24-250401     # 1단말 파일럿
  python3 backfill_predictions.py                          # 전체 단말 (from=데이터시작 ~ to=라이브시작)
옵션:
  --site-id 2  --step-hours 1  --window-hours 48  --from YYYY-MM-DD  --to YYYY-MM-DD  --device NAME  --dry-run
"""
import os
import sys
import json
import time
import argparse
import shutil
import tempfile
import importlib.util
from datetime import datetime
from pathlib import Path

import pandas as pd
import pymysql

SCRIPT_DIR = Path(__file__).resolve().parent          # models/scripts
REPO       = SCRIPT_DIR.parent.parent                 # 리포 루트

# gas_common_model_predict 동적 import (ai/scripts/ 이재헌 코드 — 건드리지 않고 import만)
PREDICT_PY = REPO / 'ai' / 'scripts' / 'gas_common_model_predict.py'
spec = importlib.util.spec_from_file_location('predict_mod', PREDICT_PY)
predict_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predict_mod)
load_artifacts        = predict_mod.load_artifacts
predict_device_window = predict_mod.predict_device_window
BASE_FEATURES         = predict_mod.BASE_FEATURES

# 활성 모델 = models/active (레지스트리 활성 버전 사본). 이두현 ai/ 는 안 건드림.
ACTIVE_DIR = REPO / 'models' / 'active'
MODELS_DIR = ACTIVE_DIR
CONFIG_DIR = ACTIVE_DIR

SENSOR_SEQ_KIND = ['volt', 'sacrificial', 'ac', 'battery', 'temp', 'hum', 'shock', 'commDbm']
KIND_TO_KR = {'volt': '방식전위', 'ac': 'AC유입', 'temp': '온도', 'hum': '습도', 'commDbm': '통신품질'}
NEEDED_KINDS = ['volt', 'ac', 'temp', 'hum', 'commDbm']


def merge_artifacts_to_temp():
    tmp = Path(tempfile.mkdtemp(prefix='siwon_bf_'))
    shutil.copy(MODELS_DIR / 'common_lstm_autoencoder.keras', tmp / 'common_lstm_autoencoder.keras')
    shutil.copy(MODELS_DIR / 'group_scalers.pkl',              tmp / 'group_scalers.pkl')
    shutil.copy(CONFIG_DIR / 'device_thresholds.json',         tmp / 'device_thresholds.json')
    shutil.copy(CONFIG_DIR / 'model_config.json',              tmp / 'model_config.json')
    return tmp


def model_version():
    mt = os.path.getmtime(MODELS_DIR / 'common_lstm_autoencoder.keras')
    return 'common_lstm_ae@' + datetime.fromtimestamp(mt).strftime('%Y%m%d')


def connect_db():
    return pymysql.connect(
        host=os.environ.get('SIWON_DB_HOST', '127.0.0.1'),
        port=int(os.environ.get('SIWON_DB_PORT', '3306')),
        user=os.environ.get('SIWON_DB_USER', 'siwon_app'),
        password=os.environ['SIWON_DB_PASS'],
        database=os.environ.get('SIWON_DB_NAME', 'siwon'),
        charset='utf8mb4', autocommit=False,
    )


def load_sensor_map(cur):
    cur.execute("""
        SELECT TRANSMITTER_ID AS txid, SENSOR_ID AS sid,
               ROW_NUMBER() OVER (PARTITION BY TRANSMITTER_ID ORDER BY SENSOR_ID) AS seq
        FROM kscg_sensor_info
    """)
    m = {}
    for r in cur.fetchall():
        k = SENSOR_SEQ_KIND[r['seq'] - 1] if r['seq'] - 1 < len(SENSOR_SEQ_KIND) else None
        if k:
            m.setdefault(r['txid'], {})[k] = r['sid']
    return m


def load_full_wide(cur, device_id, device_type, sids_by_kind, dfrom_pad, dto):
    """단말의 [dfrom_pad, dto] 전체 이력 → WIDE DataFrame (측정시각 × 방식전위·AC유입·온도·습도·통신품질)."""
    sid_to_kind = {sids_by_kind[k]: k for k in NEEDED_KINDS if k in sids_by_kind}
    if len(sid_to_kind) < 4:
        return None
    ph = ','.join(['%s'] * len(sid_to_kind))
    cur.execute(f"""
        SELECT WRITE_DATE AS t, SENSOR_ID AS sid, VALUE AS v
        FROM kscg_sensor_data
        WHERE SENSOR_ID IN ({ph}) AND WRITE_DATE > %s AND WRITE_DATE <= %s
        ORDER BY WRITE_DATE
    """, list(sid_to_kind.keys()) + [dfrom_pad, dto])
    rows = cur.fetchall()
    if not rows:
        return None
    long_df = pd.DataFrame(rows)
    long_df['kind'] = long_df['sid'].map(sid_to_kind)
    long_df['v'] = pd.to_numeric(long_df['v'], errors='coerce')
    wide = long_df.pivot_table(index='t', columns='kind', values='v', aggfunc='first').reset_index()
    wide = wide.rename(columns={'t': '측정시각', **KIND_TO_KR})
    wide.insert(0, '장비번호', device_id)
    wide['형식'] = device_type
    if any(f not in wide.columns for f in BASE_FEATURES):
        return None
    return wide.sort_values('측정시각').reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--site-id', type=int, default=2)
    ap.add_argument('--step-hours', type=int, default=1)
    ap.add_argument('--window-hours', type=int, default=48)
    ap.add_argument('--from', dest='dfrom', default=None, help='기본 = 센서데이터 최초')
    ap.add_argument('--to', dest='dto', default=None, help='기본 = 라이브 예측 시작(겹침 방지)')
    ap.add_argument('--device', default=None, help='특정 단말만 (파일럿)')
    ap.add_argument('--shard', default=None, help='K/N — N개 샤드 중 K번째 단말만 처리 (병렬 실행용)')
    ap.add_argument('--commission-days', type=int, default=14,
                    help='각 단말 첫 데이터 후 제외할 안정화 일수 (가동 초기 센서 불안정 스파이크 제거)')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    print('[1] 모델 로드')
    tmp = merge_artifacts_to_temp()
    try:
        model, scaler_map, thresholds, config = load_artifacts(str(tmp))
        time_steps   = config['time_steps']
        feature_cols = config['feature_columns']
        mv = model_version()
        print(f'  time_steps={time_steps}, model_version={mv}')

        conn = connect_db()
        try:
            cur = conn.cursor(pymysql.cursors.DictCursor)

            if args.dfrom:
                dfrom = pd.Timestamp(args.dfrom)
            else:
                cur.execute("SELECT MIN(WRITE_DATE) m FROM kscg_sensor_data")
                dfrom = pd.Timestamp(cur.fetchone()['m'])
            if args.dto:
                dto = pd.Timestamp(args.dto)
            else:
                cur.execute("SELECT MIN(predicted_at) m FROM ai_predictions WHERE source='live'")
                r = cur.fetchone()['m']
                dto = pd.Timestamp(r) if r else pd.Timestamp(datetime.now())
            print(f'[2] 백필 범위 {dfrom} ~ {dto}  step={args.step_hours}h  window={args.window_hours}h')

            cur.execute("""
                SELECT t.NAME AS device_id, t.TRANSMITTER_ID AS txid, t.TYPE AS device_type
                FROM kscg_transmitter_info t
                JOIN kscg_site_mydevice m ON m.TRANSMITTER_ID = t.TRANSMITTER_ID AND m.SITE_ID = %s
                ORDER BY t.TRANSMITTER_ID
            """, (args.site_id,))
            devices = cur.fetchall()
            if args.device:
                devices = [d for d in devices if d['device_id'] == args.device]
            if args.shard:
                k, n = (int(x) for x in args.shard.split('/'))
                devices = [d for i, d in enumerate(devices) if i % n == k]
            print(f'  대상 단말 {len(devices)}대' + (f' (shard {args.shard})' if args.shard else ''))
            smap = load_sensor_map(cur)

            wcur = conn.cursor()
            dfrom_pad = dfrom - pd.Timedelta(hours=args.window_hours)
            step = pd.Timedelta(hours=args.step_hours)
            total = 0
            t_all = time.time()
            for d in devices:
                dev, txid = d['device_id'], d['txid']
                wide = load_full_wide(cur, dev, d['device_type'], smap.get(txid, {}), dfrom_pad, dto)
                if wide is None or len(wide) < time_steps + 1:
                    print(f'  {dev}: 데이터 부족 — skip')
                    continue
                times = pd.to_datetime(wide['측정시각'])
                # 각 단말 가동 안정화 구간(첫 데이터 후 N일) 제외 — 설치 직후 센서 불안정 스파이크 배제
                dev_start = max(dfrom, times.iloc[0] + pd.Timedelta(days=args.commission_days))
                t0 = time.time()
                res_rows = []
                last_ts = None
                for i in range(len(wide)):
                    T = times.iloc[i]
                    if T < dev_start:
                        continue
                    if last_ts is not None and (T - last_ts) < step:
                        continue
                    win = wide.iloc[max(0, i - args.window_hours + 1): i + 1]
                    if len(win) < time_steps + 1:
                        continue
                    try:
                        r = predict_device_window(win, model, scaler_map, thresholds, feature_cols, time_steps)
                    except Exception:
                        continue
                    res_rows.append((
                        txid, T.to_pydatetime(), r['mse'], r['threshold'], r['risk_level'],
                        r['comm_status'], r['ai_reliability'],
                        json.dumps(r.get('feature_contributions') or {}),
                        1 if r.get('is_sacrificial_device') else 0, mv,
                    ))
                    last_ts = T
                if args.dry_run:
                    print(f'  {dev}: {len(res_rows)}건 (dry-run, {time.time()-t0:.1f}s)')
                else:
                    wcur.execute(
                        "DELETE FROM ai_predictions WHERE transmitter_id=%s AND source='backfill' "
                        "AND predicted_at>=%s AND predicted_at<=%s",
                        (txid, dfrom.to_pydatetime(), dto.to_pydatetime()))
                    if res_rows:
                        wcur.executemany("""
                            INSERT INTO ai_predictions
                              (transmitter_id, predicted_at, mse, threshold, risk_level,
                               comm_status, ai_reliability, feature_contributions,
                               is_sacrificial_device, source, model_version)
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,'backfill',%s)
                        """, res_rows)
                    conn.commit()
                    print(f'  {dev}: {len(res_rows)}건 INSERT ({time.time()-t0:.1f}s)')
                total += len(res_rows)
            print(f'[done] 총 {total}건 / {time.time()-t_all:.1f}s')
        finally:
            conn.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
