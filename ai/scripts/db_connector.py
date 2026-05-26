"""
db_connector.py
===============
DB 연결 및 기본 쿼리 유틸리티 모듈

사용 예:
    from db_connector import get_connection, DB_CONFIG

    with get_connection() as conn:
        df = pd.read_sql(sql, conn)
"""

import os
from contextlib import contextmanager
from typing import Optional

import pymysql
import pymysql.cursors

# ============================================================
# DB 접속 설정
# 환경 변수가 있으면 우선 사용, 없으면 기본값 사용
# ============================================================

DB_CONFIG = {
    'host'    : os.environ.get('DB_HOST',    'pjh1015.iptime.org'),
    'port'    : int(os.environ.get('DB_PORT', 53306)),
    'db'      : os.environ.get('DB_NAME',    'siwon'),
    'user'    : os.environ.get('DB_USER',    'siwon_jaehun'),
    'password': os.environ.get('DB_PASSWORD','oexmuIuI67LTyli156mb'),
    'charset' : 'utf8mb4',
}

# ============================================================
# DB 센서명 → 코드 내부 컬럼명 매핑
# ============================================================

# kscg_sensor_info.NAME 값을 기존 코드 컬럼명으로 변환
SENSOR_NAME_MAP: dict[str, str] = {
    '방식전위': '방식전위',
    'AC유입'  : 'AC유입',
    '온도'    : '온도',
    '습도'    : '습도',
    '수신감도': '통신품질',   # RSSI dBm 값 → 통신품질
    # 방식전류(희생전류)·배터리·충격은 AI 모델에서 미사용 → 매핑 제외
}

# 역방향 매핑 (코드 컬럼명 → DB 센서명)
SENSOR_NAME_REVERSE_MAP: dict[str, str] = {v: k for k, v in SENSOR_NAME_MAP.items()}


def read_sql(sql: str, params=None) -> 'pd.DataFrame':
    """
    SQL 쿼리를 실행하여 pandas DataFrame으로 반환.
    pd.read_sql() 대신 사용하여 SQLAlchemy 경고를 방지.
    """
    import pandas as pd
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=columns)


@contextmanager
def get_connection(config: Optional[dict] = None):
    """
    pymysql 연결을 컨텍스트 매니저로 제공.

    사용:
        with get_connection() as conn:
            ...
    """
    cfg = config or DB_CONFIG
    conn = pymysql.connect(**cfg)
    try:
        yield conn
    finally:
        conn.close()


def test_connection() -> bool:
    """DB 연결 가능 여부를 반환. 성공 시 True, 실패 시 오류 출력 후 False."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('SELECT 1')
        print(f"[DB] 연결 성공: {DB_CONFIG['host']}:{DB_CONFIG['port']} / {DB_CONFIG['db']}")
        return True
    except Exception as e:
        print(f"[DB] 연결 실패: {e}")
        return False


if __name__ == '__main__':
    test_connection()
