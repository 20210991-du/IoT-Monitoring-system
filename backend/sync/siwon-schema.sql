-- =============================================================================
-- 시원팀 자체 운영 DB 스키마 (MySQL 9.6, utf8mb4)
-- =============================================================================
-- 두 영역:
--   1) KSCG 미러 (kscg_* 접두) — 옴니솔루션 KSCG read-only 에서 sync
--   2) 우리 자체 (snake_case)   — 사용자·챗봇·이벤트 등
--
-- KSCG 원본 컬럼명 (대문자 + 한글) 그대로 보존 → sync 단순화.
-- =============================================================================

-- ───────── KSCG 미러 ─────────

CREATE TABLE IF NOT EXISTS kscg_site_info (
  SITE_ID              INT PRIMARY KEY,
  NAME                 VARCHAR(50),
  COMPANY              VARCHAR(50),
  DIVISION             VARCHAR(50),
  POSITION             VARCHAR(50),
  NUM_TRANSMITTER_MAX  INT,
  KEYWORD              VARCHAR(100),
  DESCRIPTION          VARCHAR(200),
  START_DATE           DATETIME,
  END_DATE             DATETIME,
  LATITUDE             DOUBLE,
  LONGITUDE            DOUBLE,
  SYNCED_AT            DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS kscg_transmitter_info (
  TRANSMITTER_ID       INT PRIMARY KEY,
  NAME                 VARCHAR(50),
  LONGITUDE            DOUBLE,
  LATITUDE             DOUBLE,
  ALTITUDE             DOUBLE,
  TYPE                 VARCHAR(50),
  INSTALL_DATE         DATETIME,
  PERIOD_SEC           DOUBLE,
  POSITION             VARCHAR(100),
  NUM_SENSOR           INT,
  MANUFACTURER         VARCHAR(50),
  SERIAL_NUM           VARCHAR(50),
  DESCRIPTION          VARCHAR(100),
  DEVICE_STATUS        INT,
  PERIOD_CNT           INT,
  SYNCED_AT            DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_xmit_serial (SERIAL_NUM)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS kscg_site_mydevice (
  SITE_ID              INT NOT NULL,
  TRANSMITTER_ID       INT NOT NULL,
  PRIMARY KEY (SITE_ID, TRANSMITTER_ID),
  INDEX idx_smd_site (SITE_ID),
  INDEX idx_smd_xmit (TRANSMITTER_ID)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS kscg_facility_info (
  FACILITY_ID          INT PRIMARY KEY,
  NAME                 VARCHAR(100),
  POSITION             VARCHAR(200),
  NUMBER               VARCHAR(50),
  LATITUDE             DOUBLE,
  LONGITUDE            DOUBLE,
  TRANSMITTER_ID       INT,
  DESCRIPTION          VARCHAR(200),
  SYNCED_AT            DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_fac_xmit (TRANSMITTER_ID),
  INDEX idx_fac_number (NUMBER)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS kscg_sensor_type_info (
  TYPE_ID              INT PRIMARY KEY,
  NAME                 VARCHAR(50),
  UNIT                 VARCHAR(50),
  MAX_VALUE            DOUBLE,
  MIN_VALUE            DOUBLE,
  VISIBLE              INT,
  MAX_THRESHOLD        DOUBLE,
  MIN_THRESHOLD        DOUBLE,
  DEVICE_MODEL         VARCHAR(200),
  SYNCED_AT            DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS kscg_sensor_info (
  SENSOR_ID            INT PRIMARY KEY,
  TRANSMITTER_ID       INT,
  NAME                 VARCHAR(50),
  UNIT                 VARCHAR(50),
  MAX_VALUE            DOUBLE,
  MIN_VALUE            DOUBLE,
  INIT_VALUE           DOUBLE,
  SAMPLING_RATE        DOUBLE,
  MANUFACTURER         VARCHAR(50),
  SERIAL_NUM           VARCHAR(50),
  TYPE                 INT,
  INSTALL_DATE         DATETIME,
  DESCRIPTION          VARCHAR(100),
  DEVICE_STATUS        INT,
  OFFSET_VALUE         DOUBLE,
  CAL_A                DOUBLE,
  CAL_B                DOUBLE,
  CAL_C                DOUBLE,
  SYNCED_AT            DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_si_xmit (TRANSMITTER_ID),
  INDEX idx_si_type (TYPE)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 시계열 — 가장 큰 테이블 (현재 ~210만 row, 발표까지 ~250만)
CREATE TABLE IF NOT EXISTS kscg_sensor_data (
  DATA_ID              BIGINT PRIMARY KEY,
  SENSOR_ID            INT NOT NULL,
  WRITE_DATE           DATETIME NOT NULL,
  VALUE                DOUBLE,
  VALUE_INDEX          INT,
  INDEX idx_sd_sensor_date (SENSOR_ID, WRITE_DATE),
  INDEX idx_sd_date (WRITE_DATE)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS kscg_recent_data (
  SENSOR_ID            INT PRIMARY KEY,
  DATE                 DATETIME,
  VALUE                DOUBLE,
  SYNCED_AT            DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS kscg_alarm_grade_info (
  GRADE_ID             INT PRIMARY KEY,
  GRADE_TEXT           VARCHAR(50)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS kscg_alarm_condition (
  CONDITION_ID         INT PRIMARY KEY,
  GRADE_ID             INT,
  SENSOR_ID            INT,
  CONTENTS             VARCHAR(200),
  TYPE                 VARCHAR(50),
  THRESHOLD            DOUBLE,
  THRESHOLD_CONDITION  INT,
  STATUS               INT,
  SITE_ID              INT,
  SYNCED_AT            DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_ac_sensor (SENSOR_ID)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 알람 — 1분 sync (즉시성)
CREATE TABLE IF NOT EXISTS kscg_alarm_log (
  ALARM_ID             INT PRIMARY KEY,
  GRADE_ID             INT,
  SENSOR_ID            INT,
  TYPE                 VARCHAR(50),
  SENDED_DATE          DATETIME,
  GEN_DATE             DATETIME,
  CONTENTS             VARCHAR(200),
  STATUS               INT,
  VALUE                DOUBLE,
  CONDITION_ID         INT,
  SITE_ID              INT,
  SYNCED_AT            DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_al_gen (GEN_DATE),
  INDEX idx_al_sensor (SENSOR_ID),
  INDEX idx_al_site (SITE_ID)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS kscg_maintenance_type_info (
  TYPE_ID              INT PRIMARY KEY,
  TYPE_TEXT            VARCHAR(50)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS kscg_maintenance_log (
  MAINTENANCE_ID       INT PRIMARY KEY,
  TRANSMITTER_ID       INT,
  NODE_ID              INT,
  USER_ID              VARCHAR(100),
  DATE                 DATETIME,
  TYPE                 INT,
  DESCRIPTION          VARCHAR(500),
  SYNCED_AT            DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_mlog_xmit (TRANSMITTER_ID),
  INDEX idx_mlog_date (DATE)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- sync 메타 (각 테이블 마지막 동기화 시각 기록)
CREATE TABLE IF NOT EXISTS sync_state (
  table_name           VARCHAR(64) PRIMARY KEY,
  last_synced_at       DATETIME NOT NULL,
  last_pk_seen         BIGINT,
  rows_inserted        BIGINT DEFAULT 0,
  rows_updated         BIGINT DEFAULT 0,
  notes                VARCHAR(500)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ───────── 우리 자체 (snake_case) ─────────

-- 사용자 (인증 정식화 — 기존 frontend authMock 대체)
CREATE TABLE IF NOT EXISTS users (
  id                   BIGINT AUTO_INCREMENT PRIMARY KEY,
  login_id             VARCHAR(64) NOT NULL UNIQUE,
  password_hash        VARCHAR(255) NOT NULL,
  display_name         VARCHAR(100),
  role                 ENUM('admin','operator','viewer','guest') NOT NULL DEFAULT 'operator',
  status               ENUM('pending','active','disabled') NOT NULL DEFAULT 'pending',
  preferences_json     JSON,
  created_at           DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at           DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  approved_at          DATETIME,
  approved_by          BIGINT,
  last_login_at        DATETIME,
  INDEX idx_users_role (role)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 챗봇 대화 (현재 localStorage 만 → DB 정식화)
CREATE TABLE IF NOT EXISTS chat_sessions (
  id                   BIGINT AUTO_INCREMENT PRIMARY KEY,
  user_id              BIGINT,
  title                VARCHAR(200),
  created_at           DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at           DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX idx_chat_session_user (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS chat_messages (
  id                   BIGINT AUTO_INCREMENT PRIMARY KEY,
  session_id           BIGINT NOT NULL,
  role                 ENUM('user','ai','system') NOT NULL,
  text                 MEDIUMTEXT NOT NULL,
  context_json         JSON,
  tokens_prompt        INT,
  tokens_completion    INT,
  model                VARCHAR(64),
  created_at           DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_chat_msg_session (session_id, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- AI 예측 캐시 (이두헌 LSTM 출력 — 추후 backend 가 INSERT)
CREATE TABLE IF NOT EXISTS ai_predictions (
  id                   BIGINT AUTO_INCREMENT PRIMARY KEY,
  transmitter_id       INT NOT NULL,
  predicted_at         DATETIME NOT NULL,
  mse                  DOUBLE,
  threshold            DOUBLE,
  risk_level           ENUM('정상','관찰','이상','경고','위험') NOT NULL,
  comm_status          ENUM('정상통신','일시장애','통신고장'),
  ai_reliability       ENUM('신뢰','주의','신뢰불가'),
  feature_contributions JSON,
  is_sacrificial_device TINYINT(1) DEFAULT 0,
  created_at           DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_pred_xmit (transmitter_id, predicted_at),
  INDEX idx_pred_risk (risk_level)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 단말기 위경도 보완 (KSCG TRANSMITTER 메타 NULL → 우리가 채움)
CREATE TABLE IF NOT EXISTS device_geo_overrides (
  transmitter_id       INT PRIMARY KEY,
  latitude             DOUBLE,
  longitude            DOUBLE,
  zone                 VARCHAR(50),
  source               VARCHAR(50),
  note                 VARCHAR(500),
  updated_at           DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 자문 Q3 권장 — 시스템 이벤트 마커 (밸브 열림 / 작업자 출입 등)
CREATE TABLE IF NOT EXISTS events (
  id                   BIGINT AUTO_INCREMENT PRIMARY KEY,
  transmitter_id       INT,
  event_type           VARCHAR(50),
  occurred_at          DATETIME NOT NULL,
  description          VARCHAR(500),
  metadata_json        JSON,
  created_at           DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_event_time (occurred_at),
  INDEX idx_event_xmit (transmitter_id, occurred_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 시스템 감사 로그 (audit)
CREATE TABLE IF NOT EXISTS audit_log (
  id                   BIGINT AUTO_INCREMENT PRIMARY KEY,
  user_id              BIGINT,
  action               VARCHAR(100) NOT NULL,
  target_type          VARCHAR(50),
  target_id            VARCHAR(100),
  ip                   VARCHAR(45),
  user_agent           VARCHAR(255),
  metadata_json        JSON,
  created_at           DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_audit_user (user_id, created_at),
  INDEX idx_audit_action (action)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
