#!/usr/bin/env python3
"""
sync_kscg.py — 옴니솔루션 KSCG (MS SQL, read_only) → siwon (MySQL) 동기화

사용법:
  python sync_kscg.py backfill                  첫 풀 백필 (모든 테이블 전체)
  python sync_kscg.py sync alarm                알람 1분 sync (TB_ALARM_LOG, TB_RECENT_DATA 일부)
  python sync_kscg.py sync sensor               시계열 5분 sync (TB_SENSOR_DATA + TB_RECENT_DATA 전체)
  python sync_kscg.py sync meta                 메타 1시간 sync (단말·시설·센서 정보·알람 조건·점검)

환경변수 (두 .env 파일):
  ~/PJHwork/secrets/local/kscg-db.env  — KSCG (소스, MSSQL)
  ~/PJHwork/secrets/local/siwon-db.env — siwon (목적, MySQL)
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime
import pymssql
import pymysql

# ───────── 설정 로드 ─────────
SECRETS = Path.home() / "PJHwork/secrets/local"

def load_env(path):
    env = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env

KSCG = load_env(Path.home() / "PJHwork/secrets/omni/kscg-db.env")
SIWON = load_env(SECRETS / "siwon-db.env")

def src():
    return pymssql.connect(
        server=KSCG["KSCG_DB_HOST"],
        port=int(KSCG["KSCG_DB_PORT"]),
        user=KSCG["KSCG_DB_USER"],
        password=KSCG["KSCG_DB_PASS"],
        database=KSCG.get("KSCG_DB_NAME", "KSCG"),
        as_dict=False,
        charset="UTF-8",
        timeout=60,
    )

def dst():
    return pymysql.connect(
        host=SIWON["SIWON_DB_HOST"],
        port=int(SIWON["SIWON_DB_PORT"]),
        user=SIWON["SIWON_DB_USER"],
        password=SIWON["SIWON_DB_PASS"],
        database=SIWON["SIWON_DB_NAME"],
        charset="utf8mb4",
        autocommit=False,
    )

# ───────── 테이블 정의 ─────────
# (kscg_table, source_table, columns)

META_TABLES = [
    ("kscg_site_info", "TB_SITE_INFO",
     "SITE_ID NAME COMPANY DIVISION POSITION NUM_TRANSMITTER_MAX KEYWORD DESCRIPTION "
     "START_DATE END_DATE LATITUDE LONGITUDE"),
    ("kscg_transmitter_info", "TB_TRANSMITTER_INFO",
     "TRANSMITTER_ID NAME LONGITUDE LATITUDE ALTITUDE TYPE INSTALL_DATE PERIOD_SEC "
     "POSITION NUM_SENSOR MANUFACTURER SERIAL_NUM DESCRIPTION DEVICE_STATUS PERIOD_CNT"),
    ("kscg_site_mydevice", "TB_SITE_MYDEVICE", "SITE_ID TRANSMITTER_ID"),
    ("kscg_facility_info", "TB_FACILITY_INFO",
     "FACILITY_ID NAME POSITION NUMBER LATITUDE LONGITUDE TRANSMITTER_ID DESCRIPTION"),
    ("kscg_sensor_type_info", "TB_SENSOR_TYPE_INFO",
     "TYPE_ID NAME UNIT MAX_VALUE MIN_VALUE VISIBLE MAX_THRESHOLD MIN_THRESHOLD DEVICE_MODEL"),
    ("kscg_sensor_info", "TB_SENSOR_INFO",
     "SENSOR_ID TRANSMITTER_ID NAME UNIT MAX_VALUE MIN_VALUE INIT_VALUE SAMPLING_RATE "
     "MANUFACTURER SERIAL_NUM TYPE INSTALL_DATE DESCRIPTION DEVICE_STATUS "
     "OFFSET_VALUE CAL_A CAL_B CAL_C"),
    ("kscg_alarm_grade_info", "TB_ALARM_GRADE_INFO", "GRADE_ID GRADE_TEXT"),
    ("kscg_alarm_condition", "TB_ALARM_CONDITION",
     "CONDITION_ID GRADE_ID SENSOR_ID CONTENTS TYPE THRESHOLD THRESHOLD_CONDITION STATUS SITE_ID"),
    ("kscg_maintenance_type_info", "TB_MAINTENANCE_TYPE_INFO", "TYPE_ID TYPE_TEXT"),
    ("kscg_maintenance_log", "TB_MAINTENANCE_LOG",
     "MAINTENANCE_ID TRANSMITTER_ID NODE_ID USER_ID DATE TYPE DESCRIPTION"),
]

ALARM_TABLES = [
    ("kscg_alarm_log", "TB_ALARM_LOG",
     "ALARM_ID GRADE_ID SENSOR_ID TYPE SENDED_DATE GEN_DATE CONTENTS STATUS VALUE CONDITION_ID SITE_ID"),
    ("kscg_recent_data", "TB_RECENT_DATA", "SENSOR_ID DATE VALUE"),
]

SENSOR_TABLES = [
    ("kscg_sensor_data", "TB_SENSOR_DATA", "DATA_ID SENSOR_ID WRITE_DATE VALUE VALUE_INDEX"),
    ("kscg_recent_data", "TB_RECENT_DATA", "SENSOR_ID DATE VALUE"),
]


def upsert_table(src_conn, dst_conn, dst_table, src_table, cols, where=None, chunk=5000, replace=True):
    """소스에서 모든 행을 가져와 (또는 where 절 적용) MySQL 에 REPLACE INTO."""
    cols_list = cols.split()
    col_csv = ",".join(cols_list)
    placeholders = ",".join(["%s"] * len(cols_list))
    sql_dst = f"REPLACE INTO {dst_table} ({col_csv}) VALUES ({placeholders})"
    sql_src = f"SELECT {col_csv} FROM {src_table}"
    if where:
        sql_src += f" WHERE {where}"

    sc = src_conn.cursor()
    sc.execute(sql_src)
    dc = dst_conn.cursor()

    total = 0
    while True:
        rows = sc.fetchmany(chunk)
        if not rows:
            break
        dc.executemany(sql_dst, rows)
        total += len(rows)
    dst_conn.commit()
    return total


def record_sync(dst_conn, table_name, rows, last_pk=None, notes=None):
    """sync_state 갱신."""
    dc = dst_conn.cursor()
    dc.execute("""
      INSERT INTO sync_state (table_name, last_synced_at, last_pk_seen, rows_inserted, notes)
      VALUES (%s, NOW(), %s, %s, %s)
      ON DUPLICATE KEY UPDATE
        last_synced_at = VALUES(last_synced_at),
        last_pk_seen   = COALESCE(VALUES(last_pk_seen), last_pk_seen),
        rows_inserted  = rows_inserted + VALUES(rows_inserted),
        notes          = COALESCE(VALUES(notes), notes)
    """, (table_name, last_pk, rows, notes))
    dst_conn.commit()


def get_last_pk(dst_conn, table_name, pk_col):
    """대상 테이블의 최대 PK 값 (incremental sync 용)."""
    dc = dst_conn.cursor()
    dc.execute(f"SELECT MAX({pk_col}) FROM {table_name}")
    r = dc.fetchone()
    return r[0] if r and r[0] is not None else None


# ───────── 명령 핸들러 ─────────

def cmd_backfill():
    """모든 테이블 풀 백필 (한 번만 실행)."""
    print(f"[{datetime.now():%H:%M:%S}] BACKFILL 시작")
    src_conn = src()
    dst_conn = dst()
    try:
        # 메타 + 알람 (작은 테이블 먼저)
        for dst_t, src_t, cols in META_TABLES + ALARM_TABLES:
            t0 = time.time()
            n = upsert_table(src_conn, dst_conn, dst_t, src_t, cols)
            record_sync(dst_conn, dst_t, n, notes=f"backfill {n} rows")
            print(f"  {dst_t:35s}  {n:>10,} rows  ({time.time()-t0:.1f}s)")

        # 시계열 (큰 테이블 — chunk 키워야 메모리 OK)
        t0 = time.time()
        n = upsert_table(src_conn, dst_conn, "kscg_sensor_data", "TB_SENSOR_DATA",
                         "DATA_ID SENSOR_ID WRITE_DATE VALUE VALUE_INDEX",
                         chunk=10000)
        last_pk = get_last_pk(dst_conn, "kscg_sensor_data", "DATA_ID")
        record_sync(dst_conn, "kscg_sensor_data", n, last_pk=last_pk, notes=f"backfill {n} rows")
        print(f"  {'kscg_sensor_data':35s}  {n:>10,} rows  ({time.time()-t0:.1f}s)  last_pk={last_pk}")
    finally:
        src_conn.close()
        dst_conn.close()
    print(f"[{datetime.now():%H:%M:%S}] BACKFILL 완료")


def cmd_sync_alarm():
    """알람 1분 sync — ALARM_LOG incremental + RECENT_DATA refresh."""
    src_conn = src()
    dst_conn = dst()
    try:
        # ALARM_LOG — PK incremental
        last_pk = get_last_pk(dst_conn, "kscg_alarm_log", "ALARM_ID") or 0
        n = upsert_table(src_conn, dst_conn, "kscg_alarm_log", "TB_ALARM_LOG",
                         "ALARM_ID GRADE_ID SENSOR_ID TYPE SENDED_DATE GEN_DATE "
                         "CONTENTS STATUS VALUE CONDITION_ID SITE_ID",
                         where=f"ALARM_ID > {last_pk}")
        new_pk = get_last_pk(dst_conn, "kscg_alarm_log", "ALARM_ID")
        record_sync(dst_conn, "kscg_alarm_log", n, last_pk=new_pk)
        if n:
            print(f"[{datetime.now():%H:%M:%S}] alarm: +{n} rows (last_pk {last_pk} → {new_pk})")
    finally:
        src_conn.close()
        dst_conn.close()


def cmd_sync_sensor():
    """시계열 5분 sync — SENSOR_DATA incremental + RECENT_DATA refresh."""
    src_conn = src()
    dst_conn = dst()
    try:
        last_pk = get_last_pk(dst_conn, "kscg_sensor_data", "DATA_ID") or 0
        n = upsert_table(src_conn, dst_conn, "kscg_sensor_data", "TB_SENSOR_DATA",
                         "DATA_ID SENSOR_ID WRITE_DATE VALUE VALUE_INDEX",
                         where=f"DATA_ID > {last_pk}", chunk=5000)
        new_pk = get_last_pk(dst_conn, "kscg_sensor_data", "DATA_ID")
        record_sync(dst_conn, "kscg_sensor_data", n, last_pk=new_pk)
        # RECENT_DATA 전체 refresh (440 row 라 부담 X)
        n2 = upsert_table(src_conn, dst_conn, "kscg_recent_data", "TB_RECENT_DATA",
                          "SENSOR_ID DATE VALUE")
        record_sync(dst_conn, "kscg_recent_data", n2, notes="full refresh")
        if n or n2:
            print(f"[{datetime.now():%H:%M:%S}] sensor: data+{n}, recent={n2}")
    finally:
        src_conn.close()
        dst_conn.close()


def cmd_sync_meta():
    """메타 1시간 sync — 단말·시설·센서·알람조건·점검."""
    src_conn = src()
    dst_conn = dst()
    try:
        total = 0
        for dst_t, src_t, cols in META_TABLES:
            n = upsert_table(src_conn, dst_conn, dst_t, src_t, cols)
            record_sync(dst_conn, dst_t, n, notes="meta refresh")
            total += n
        print(f"[{datetime.now():%H:%M:%S}] meta: {total} rows refreshed")
    finally:
        src_conn.close()
        dst_conn.close()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "backfill":
        cmd_backfill()
    elif cmd == "sync":
        sub = sys.argv[2] if len(sys.argv) > 2 else ""
        if sub == "alarm":
            cmd_sync_alarm()
        elif sub == "sensor":
            cmd_sync_sensor()
        elif sub == "meta":
            cmd_sync_meta()
        else:
            print(f"unknown sync sub: {sub}")
            sys.exit(2)
    else:
        print(f"unknown cmd: {cmd}")
        sys.exit(2)


if __name__ == "__main__":
    main()
