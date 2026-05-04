// Mock data for the AI 통합관제 시스템 — 55 devices, 6 sensors:
// 방식전위(mV) · AC유입(mV) · 희생전류(mA) · 온도(℃) · 습도(%) · 통신품질(dBm)

export const FACILITIES = [
  "서울 제1구역 가스배관", "서울 제2구역 가스배관", "서울 제3구역 가스배관",
  "인천 항만 배관망", "부천 중부 집결관", "일산 신도시 라인",
  "수원 남부 배관망", "성남 판교 라인", "안양 동서축",
  "의정부 북부 라인", "광명 지하배관", "하남 미사 라인",
];

const LOCATIONS = [
  "OK마트 삼거리 앞", "광영동 옥곡신금 대우스틸 앞",
  "서면 행정복지센터 앞 녹지 정압기 내부", "순천IC 입구 오르막길 우측 도로",
  "신대6차 608동 상가 버스정류장 앞 차도", "금호동 온아트홀 미온다 화단 내부",
  "강변북로 김포 방향 진출로 우측", "삼성역 6번 출구 인도 매립구",
  "잠실대교 남단 교각 하부", "양재IC 진입 분기점 중앙분리대",
  "신도림역 환승통로 아래", "구로디지털단지역 2번 출구",
  "학동사거리 정압기 지상부", "압구정로데오역 3번 출구 인도",
  "논현역 7번 출구 우측 20m", "판교IC 분기 진출 우측 배수로",
  "분당서울대병원 정문 우측 화단", "미금역 환승 센터 B2",
  "야탑역 1번 출구 인도 집수정", "수내역 북측 근린공원 내부",
  "정자역 3번 출구 편의점 좌측", "광화문 D광장 지하 배관실",
  "여의도 IFC몰 지하 3층", "목동 오거리 배수로 우측",
];

// Deterministic RNG — 같은 seed(42)로 매 로드마다 동일한 55개 장비가 생성되어
// 개발/검증 과정에서 값이 뒤집히지 않도록 고정합니다.
function mulberry32(s) {
  return function () {
    let t = (s += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = mulberry32(42);
function pick(a) { return a[Math.floor(rand() * a.length)]; }
function randInt(lo, hi) { return Math.floor(rand() * (hi - lo + 1)) + lo; }

// Sensor contribution helper — returns [{sensor, pct}] top 3
function makeContribution(status, dominant) {
  const sensors = ["방식전위", "AC유입", "희생전류", "온도", "습도", "통신품질"];
  if (status === "normal" || status === "offline") return [];
  const pool = sensors.filter((s) => s !== dominant);
  const second = pick(pool);
  const third = pick(pool.filter((s) => s !== second));
  const base = status === "anomaly"
    ? [randInt(72, 86), randInt(9, 17), randInt(3, 8)]
    : [randInt(55, 70), randInt(16, 24), randInt(8, 13)];
  return [
    { sensor: dominant, pct: base[0] },
    { sensor: second, pct: base[1] },
    { sensor: third, pct: base[2] },
  ];
}

const DOMINANT_BY_LABEL = {
  "위상차 급변": "방식전위",
  "유효전류 상승": "희생전류",
  "방식전위 이탈": "방식전위",
  "AC 유입 과다": "AC유입",
  "희생전류 저하": "희생전류",
  "통신 품질 저하": "통신품질",
  "온도 급변": "온도",
  "습도 상승": "습도",
  "성능 저하 트렌드": "방식전위",
  "간헐적 통신 지연": "통신품질",
};

const ANOMALY_LABELS = ["위상차 급변", "방식전위 이탈", "AC 유입 과다", "희생전류 저하", "통신 품질 저하"];
const WARN_LABELS = ["성능 저하 트렌드", "간헐적 통신 지연", "습도 상승", "온도 급변"];

// 55 devices: 42 normal, 6 anomaly, 4 warn, 3 offline
export const EQUIPMENT = (() => {
  const out = [];
  const statuses = [
    ...Array(42).fill("normal"),
    ...Array(6).fill("anomaly"),
    ...Array(4).fill("warn"),
    ...Array(3).fill("offline"),
  ];
  for (let i = statuses.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [statuses[i], statuses[j]] = [statuses[j], statuses[i]];
  }

  for (let i = 0; i < 55; i++) {
    const facilityIdx = randInt(2021, 2025);
    const seq = String(i + 1).padStart(4, "0");
    const dev = `TB24-5JN${String(i + 1).padStart(3, "0")}`;
    const status = statuses[i];
    let volt, sacrificial, ac, temp, hum, commDbm, commOk, mse;
    let label, dominant;

    if (status === "anomaly") {
      label = pick(ANOMALY_LABELS);
      dominant = DOMINANT_BY_LABEL[label];
      volt = -400 - Math.floor(rand() * 400);
      sacrificial = +(rand() * 2).toFixed(2);
      ac = Math.floor(rand() * 300) + 420;
      temp = +(22 + rand() * 8).toFixed(1);
      hum = +(60 + rand() * 25).toFixed(1);
      commDbm = -70 - Math.floor(rand() * 15);
      commOk = true;
      mse = +(0.42 + rand() * 0.55).toFixed(3);
    } else if (status === "warn") {
      label = pick(WARN_LABELS);
      dominant = DOMINANT_BY_LABEL[label];
      volt = -850 - Math.floor(rand() * 200);
      sacrificial = +(3 + rand() * 4).toFixed(2);
      ac = 180 + Math.floor(rand() * 180);
      temp = +(20 + rand() * 6).toFixed(1);
      hum = +(40 + rand() * 20).toFixed(1);
      commDbm = -65 - Math.floor(rand() * 10);
      commOk = true;
      mse = +(0.28 + rand() * 0.11).toFixed(3);
    } else if (status === "offline") {
      label = "통신 두절";
      dominant = "통신품질";
      volt = 0; sacrificial = 0; ac = 0; temp = 0; hum = 0;
      commDbm = -99; commOk = false; mse = null;
    } else {
      label = null; dominant = null;
      volt = -1000 - Math.floor(rand() * 250);
      sacrificial = +(5 + rand() * 8).toFixed(2);
      ac = Math.floor(rand() * 120);
      temp = +(17 + rand() * 8).toFixed(1);
      hum = +(35 + rand() * 20).toFixed(1);
      commDbm = -52 - Math.floor(rand() * 8);
      commOk = true;
      mse = +(rand() * 0.2).toFixed(3);
    }

    const minAgo = randInt(1, 59);
    const mm = (60 - (minAgo % 60)) % 60;
    const updatedAt = `14:${String(mm).padStart(2, "0")}`;

    out.push({
      id: i,
      facilityId: `TB-${facilityIdx}-${seq.slice(0, 4)}`,
      deviceId: dev,
      location: LOCATIONS[i % LOCATIONS.length],
      status,
      label,
      volt,
      ac,
      sacrificial,
      temp,
      hum,
      commDbm,
      commOk,
      mse,
      contribution: status === "normal" || status === "offline" ? [] : makeContribution(status, dominant),
      threshold: 0.409,
      updatedAt,
      nextCollectMin: 60 - minAgo,
      zone: pick(["제1구역", "제2구역", "제3구역", "제4구역"]),
      lng: 127 + (rand() - 0.5) * 0.6,
      lat: 37.5 + (rand() - 0.5) * 0.35,
    });
  }
  return out;
})();

// Map markers (0-1 normalized coords)
export const MAP_MARKERS = [
  { id: "m1", kind: "cluster", count: 3, x: 0.17, y: 0.22, status: "normal" },
  { id: "m2", kind: "cluster", count: 4, x: 0.22, y: 0.50, status: "normal" },
  { id: "m3", kind: "cluster", count: 6, x: 0.58, y: 0.58, status: "normal" },
  { id: "m4", kind: "cluster", count: 12, x: 0.70, y: 0.36, status: "normal" },
  { id: "a1", kind: "single", x: 0.30, y: 0.31, status: "anomaly", node: "TB24-5JN042",
    label: "위상차 급변", mse: 0.842, zone: "제2구역",
    contribution: [{ sensor: "방식전위", pct: 83 }, { sensor: "AC유입", pct: 12 }, { sensor: "희생전류", pct: 5 }] },
  { id: "a2", kind: "single", x: 0.40, y: 0.38, status: "anomaly", node: "TB24-5JN089",
    label: "희생전류 저하", mse: 0.791, zone: "제3구역",
    contribution: [{ sensor: "희생전류", pct: 78 }, { sensor: "방식전위", pct: 14 }, { sensor: "온도", pct: 8 }] },
  { id: "a3", kind: "single", x: 0.44, y: 0.47, status: "anomaly", node: "TB24-5JN117",
    label: "방식전위 이탈", mse: 0.712, zone: "제2구역",
    contribution: [{ sensor: "방식전위", pct: 75 }, { sensor: "AC유입", pct: 17 }, { sensor: "습도", pct: 8 }] },
  { id: "a4", kind: "single", x: 0.52, y: 0.42, status: "anomaly", node: "TB24-5JN063",
    label: "AC 유입 과다", mse: 0.684, zone: "제3구역",
    contribution: [{ sensor: "AC유입", pct: 81 }, { sensor: "방식전위", pct: 13 }, { sensor: "희생전류", pct: 6 }] },
  { id: "a5", kind: "single", x: 0.35, y: 0.52, status: "anomaly", node: "TB24-5JN024",
    label: "통신 품질 저하", mse: 0.651, zone: "제1구역",
    contribution: [{ sensor: "통신품질", pct: 74 }, { sensor: "AC유입", pct: 16 }, { sensor: "희생전류", pct: 10 }] },
  { id: "w1", kind: "single", x: 0.48, y: 0.63, status: "warn", node: "TB24-5JN115",
    label: "성능 저하 트렌드", mse: 0.372, zone: "제4구역",
    contribution: [{ sensor: "방식전위", pct: 62 }, { sensor: "온도", pct: 22 }, { sensor: "습도", pct: 10 }] },
];

// AI anomaly list (6 items matching EQUIPMENT.anomalies)
export const AI_ANOMALIES = [
  { node: "TB24-5JN042", label: "위상차 급변", mse: 0.842, threshold: 0.409, zone: "제2구역",
    contribution: [{ sensor: "방식전위", pct: 83 }, { sensor: "AC유입", pct: 12 }, { sensor: "희생전류", pct: 5 }],
    summary: "방식전위가 최근 3시간 동안 ±420mV 진폭으로 반복 변동. AC 유입 급증과의 상관관계(0.87) 관측됨. 외부 송전선 간섭으로 추정.",
    action: "즉시 현장 점검 · 외부 AC 간섭원 차단 · 접지 상태 재측정" },
  { node: "TB24-5JN008", label: "희생전류 저하", mse: 0.791, threshold: 0.409, zone: "제3구역",
    contribution: [{ sensor: "희생전류", pct: 78 }, { sensor: "방식전위", pct: 14 }, { sensor: "온도", pct: 8 }],
    summary: "희생양극 방식 전류가 0.4mA로 사실상 기능 상실. 양극 소모 완료 가능성 높음.",
    action: "희생양극 교체 점검 · 접속부 저항 측정 · 교체 주기 재산정" },
  { node: "TB24-5JN047", label: "방식전위 이탈", mse: 0.712, threshold: 0.409, zone: "제2구역",
    contribution: [{ sensor: "방식전위", pct: 75 }, { sensor: "AC유입", pct: 17 }, { sensor: "습도", pct: 8 }],
    summary: "-850mV 방식 기준선을 24시간 중 5회 이탈. 방식 효율 저하 구간으로 판단.",
    action: "심정 조사 · 접지 저항 측정 · 정류기 출력 재조정" },
  { node: "TB24-5JN035", label: "AC 유입 과다", mse: 0.684, threshold: 0.409, zone: "제3구역",
    contribution: [{ sensor: "AC유입", pct: 81 }, { sensor: "방식전위", pct: 13 }, { sensor: "희생전류", pct: 6 }],
    summary: "인접 송전선로 가동 패턴과 유도 전압 동기화. 피크시간대 620mV까지 상승.",
    action: "배수장치 상태 점검 · 유도 차폐 보강 · 절연 저항 측정" },
  { node: "TB24-5JN024", label: "통신 품질 저하", mse: 0.651, threshold: 0.409, zone: "제1구역",
    contribution: [{ sensor: "통신품질", pct: 74 }, { sensor: "AC유입", pct: 16 }, { sensor: "희생전류", pct: 10 }],
    summary: "통신 신호세기 -82dBm까지 저하. 전송 재시도율 12%로 상승.",
    action: "RF 환경 조사 · 안테나 방위 재조정 · 게이트웨이 점검" },
  { node: "TB24-5JN031", label: "방식전위 이탈", mse: 0.598, threshold: 0.409, zone: "제4구역",
    contribution: [{ sensor: "방식전위", pct: 70 }, { sensor: "습도", pct: 19 }, { sensor: "AC유입", pct: 11 }],
    summary: "야간 시간대 방식전위 상승 경향. 습도 증가와 상관관계 관측.",
    action: "맨홀 침수 여부 점검 · 밀봉 상태 확인" },
];

export const AI_WATCH = [
  { node: "TB24-5JN015", label: "성능 저하 트렌드", mse: 0.372, threshold: 0.409, zone: "제4구역",
    contribution: [{ sensor: "방식전위", pct: 62 }, { sensor: "온도", pct: 22 }, { sensor: "습도", pct: 10 }],
    summary: "지난 7일간 MSE가 0.21 → 0.37로 완만히 상승. 아직 임계치 이내.",
    action: "RF 환경 조사 · 향후 48시간 집중 모니터링" },
  { node: "TB24-5JN040", label: "간헐적 통신 지연", mse: 0.351, threshold: 0.409, zone: "제2구역",
    contribution: [{ sensor: "통신품질", pct: 58 }, { sensor: "AC유입", pct: 24 }, { sensor: "희생전류", pct: 12 }],
    summary: "평균 레이턴시 220ms. 피크 시간대 580ms 관측.",
    action: "게이트웨이 버퍼 증설 검토 · 재측정" },
  { node: "TB24-5JN023", label: "습도 상승", mse: 0.342, threshold: 0.409, zone: "제1구역",
    contribution: [{ sensor: "습도", pct: 55 }, { sensor: "방식전위", pct: 25 }, { sensor: "온도", pct: 15 }],
    summary: "맨홀 내 습도 72%까지 상승. 우천 이후 배수 지연 추정.",
    action: "배수 상태 확인 · 밀봉 점검" },
  { node: "TB24-5JN052", label: "온도 급변", mse: 0.318, threshold: 0.409, zone: "제3구역",
    contribution: [{ sensor: "온도", pct: 61 }, { sensor: "습도", pct: 20 }, { sensor: "AC유입", pct: 14 }],
    summary: "설치 환경 온도 일교차 28℃ 관측. 평소 15℃ 대비 증가.",
    action: "보온재 상태 점검" },
];

export const AI_INSIGHTS = [
  { kind: "now", icon: "alert", title: "지금 조치", timeLabel: "긴급",
    body: "TB24-5JN042에서 AC 유입 전압이 지속 상승 중. 절연 파손 가능성 있음. 현장 점검 필요." },
  { kind: "soon", icon: "trend", title: "48시간 내", timeLabel: "예측",
    body: "제2구역 방식전위가 임계값에 근접. 48시간 내 경고 수준 도달 확률 78%." },
  { kind: "long", icon: "sparkle", title: "장기 개선", timeLabel: "제안",
    body: "TB24-5JN115의 MSE 기여도 중 방식전위가 62%로 우세. 수집 주기를 1시간 → 10분으로 단축 권장." },
];

export const LOG_TEMPLATES = [
  { kind: "ok",    t: (n) => `CMD: PING TB24-5JN${n} ...`, tail: "OK" },
  { kind: "data",  t: (n) => `DATA: TB24-5JN${n} RECV ${(Math.random() * 3 + 0.4).toFixed(1)}KB` },
  { kind: "alert", t: (n) => `ALERT: MSE>TH @ TB24-5JN${n}` },
  { kind: "ok",    t: () => `SYS: DB BACKUP COMPLETED` },
  { kind: "ai",    t: () => `AI: LSTM INFER BATCH 64 · 12ms` },
  { kind: "auth",  t: () => `AUTH: OPERATOR_1 LOGIN` },
  { kind: "warn",  t: (n) => `WARN: RETRY_FAIL @ TB24-5JN${n}` },
  { kind: "ai",    t: () => `AI: CONTRIBUTION SCORING...` },
  { kind: "ok",    t: () => `SYS: HEARTBEAT SENT` },
  { kind: "data",  t: (n) => `DATA: TB24-5JN${n} MSE=${(Math.random() * 0.9).toFixed(3)}` },
  { kind: "ok",    t: (n) => `MODEL: LSTM-AE v3 INFER ${n} (${((Math.random() * 20 + 8) | 0)}ms)` },
  { kind: "warn",  t: () => `WARN: GW-02 BUFFER 86%` },
];
