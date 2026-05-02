"""
backend/main.py
===============
FastAPI 서버 — 프론트엔드 ↔ AI 모델 연동 브릿지

실행 방법:
    pip install fastapi uvicorn pandas numpy scikit-learn tensorflow openpyxl
    uvicorn main:app --reload --port 8000

엔드포인트:
    GET  /api/health              — 서버/모델 상태 확인
    GET  /api/devices             — 전체 장비 목록 + 최신 센서값
    GET  /api/anomalies           — AI 이상 탐지 결과 (이상/관찰)
    POST /api/predict/{device_id} — 단일 장비 실시간 이상 탐지
    GET  /api/summary             — KPI 요약 (정상/이상/관찰/통신장애 카운트)
    POST /api/upload              — 엑셀 파일 업로드 후 전체 재분석
"""

import os
import sys
import json
import pickle
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# Windows에서 이모지·한글 출력 시 cp949 인코딩 오류 방지
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── AI 스크립트 경로 추가 ────────────────────────────────────
AI_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "ai" / "scripts"
sys.path.insert(0, str(AI_SCRIPTS_DIR))

app = FastAPI(
    title="매설배관 AI 통합관제 API",
    description="LSTM AutoEncoder 기반 이상 탐지 REST API",
    version="1.0.0",
)

# ── CORS 설정 (React 개발 서버 허용) ────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 전역 상태
# ============================================================

class AppState:
    model = None
    scaler_map: Dict = {}
    thresholds: Dict = {}
    config: Dict = {}
    master_df: Optional[pd.DataFrame] = None
    last_results: Optional[pd.DataFrame] = None
    last_updated: Optional[str] = None
    model_loaded: bool = False
    excel_path: Optional[Path] = None

state = AppState()

# ── 아티팩트 경로 (ai/models, ai/config, data 폴더 기준) ───
_REPO_ROOT    = Path(__file__).resolve().parent.parent
MODELS_DIR    = _REPO_ROOT / "ai" / "models"
CONFIG_DIR    = _REPO_ROOT / "ai" / "config"
DEFAULT_EXCEL = _REPO_ROOT / "data" / "01. 시설물 50개 샘플 데이터.xlsx"

# ── 이상 라벨 한글 매핑 ──────────────────────────────────
FEATURE_LABEL_MAP = {
    "방식전위":      "방식전위 이탈",
    "방식전위_diff1": "위상차 급변",
    "방식전위_dev24": "방식전위 편차 이상",
    "AC유입":        "AC 유입 과다",
    "AC유입_diff1":  "AC 유입 급변",
    "온도":          "온도 급변",
    "습도":          "습도 상승",
    "희생전류":       "희생전류 저하",
    "통신품질":       "통신 품질 저하",
}

RISK_LABELS = {"이상": "🔴 이상", "관찰": "🟡 관찰", "정상": "🟢 정상"}

ZONE_MAP = ["제1구역", "제2구역", "제3구역", "제4구역"]


# ============================================================
# 시작 시 모델 & 데이터 로드
# ============================================================

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 아티팩트 로드 + 기본 엑셀 분석."""
    try:
        _load_artifacts()
        print("[AI] 아티팩트 로드 완료")
    except Exception as e:
        print(f"[AI] 아티팩트 로드 실패 (학습 먼저 실행 필요): {e}")

    if DEFAULT_EXCEL.exists():
        try:
            await _run_analysis(DEFAULT_EXCEL)
            print(f"[AI] 기본 데이터 분석 완료: {DEFAULT_EXCEL.name}")
        except Exception as e:
            print(f"[AI] 기본 데이터 분석 실패: {e}")
    else:
        print(f"[AI] 기본 엑셀 없음 ({DEFAULT_EXCEL}). /api/upload 로 업로드하세요.")


def _load_artifacts():
    """학습된 모델/스케일러/threshold 로드."""
    from tensorflow.keras.models import load_model

    model_path = MODELS_DIR / "common_lstm_autoencoder.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일 없음: {model_path}")

    state.model = load_model(str(model_path))

    with open(MODELS_DIR / "group_scalers.pkl", "rb") as f:
        state.scaler_map = pickle.load(f)

    with open(CONFIG_DIR / "device_thresholds.json", "r", encoding="utf-8") as f:
        state.thresholds = json.load(f)

    with open(CONFIG_DIR / "model_config.json", "r", encoding="utf-8") as f:
        state.config = json.load(f)

    state.model_loaded = True


async def _run_analysis(excel_path: Path):
    """백그라운드에서 전체 장비 이상 탐지 실행."""
    from gas_common_model_train import build_master_dataset
    from gas_common_model_predict import run_batch_prediction

    loop = asyncio.get_event_loop()

    def _sync():
        master_df = build_master_dataset(str(excel_path))
        state.master_df = master_df
        state.excel_path = excel_path

        if not state.model_loaded:
            return

        feature_cols = state.config.get("feature_columns", [])
        time_steps   = state.config.get("time_steps", 24)

        result_df = run_batch_prediction(
            df=master_df.copy(),
            model=state.model,
            scaler_map=state.scaler_map,
            thresholds=state.thresholds,
            feature_cols=feature_cols,
            time_steps=time_steps,
        )
        state.last_results = result_df
        state.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    await loop.run_in_executor(None, _sync)


# ============================================================
# 헬퍼 함수
# ============================================================

def _status_from_risk(risk: str, comm: str) -> str:
    if comm == "통신고장":
        return "offline"
    if risk == "이상":
        return "anomaly"
    if risk == "관찰":
        return "warn"
    return "normal"


def _dominant_feature(contributions: Dict[str, float]) -> str:
    """기여도 딕셔너리에서 가장 높은 피처명 반환."""
    if not contributions:
        return "방식전위"
    best = max(contributions, key=contributions.get)
    # _diff1, _dev24 등 suffix 제거
    base = best.split("_")[0]
    return FEATURE_LABEL_MAP.get(base, base)


def _make_contribution_list(contributions: Dict[str, float]) -> List[Dict]:
    """feature_contributions dict → [{sensor, pct}] top 3."""
    if not contributions:
        return []
    total = sum(contributions.values()) or 1
    sorted_items = sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:3]
    result = []
    for feat, val in sorted_items:
        base = feat.split("_")[0]
        pct  = round(val / total * 100)
        result.append({"sensor": base, "pct": pct})
    return result


def _device_row_from_master(device_id: str) -> Dict:
    """master_df에서 장비 최신 센서값 추출."""
    if state.master_df is None:
        return {}
    ddf = state.master_df[state.master_df["장비번호"] == device_id]
    if ddf.empty:
        return {}
    latest = ddf.sort_values("측정시각").iloc[-1]

    def _v(col, default=0):
        v = latest.get(col, default)
        return float(v) if pd.notna(v) else default

    comm_dbm = _v("통신품질", -99)
    comm_ok  = bool(comm_dbm > -115 and latest.get("통신단절_플래그", 0) == 0)

    return {
        "volt":       round(_v("방식전위", -850)),
        "ac":         round(_v("AC유입", 0)),
        "sacrificial": round(_v("희생전류", 0), 2),
        "temp":        round(_v("온도", 20), 1),
        "hum":         round(_v("습도", 50), 1),
        "commDbm":    round(comm_dbm),
        "commOk":     comm_ok,
        "updatedAt":  str(latest.get("측정시각", ""))[:16],
    }


def _result_row_to_anomaly(row: Dict, idx: int) -> Dict:
    """run_batch_prediction 결과 행 → 프론트 AI_ANOMALIES 형식."""
    contributions_raw = {
        k: v for k, v in row.items()
        if k not in {"device_id", "group", "mse", "threshold",
                     "risk_level", "comm_status", "ai_reliability", "is_sacrificial_device"}
        and isinstance(v, float)
    }
    contrib_list = _make_contribution_list(contributions_raw)
    dominant     = _dominant_feature(contributions_raw)

    return {
        "node":         row["device_id"],
        "label":        dominant,
        "mse":          round(float(row["mse"]), 3),
        "threshold":    round(float(row["threshold"]), 3),
        "zone":         ZONE_MAP[hash(row["device_id"]) % 4],
        "contribution": contrib_list,
        "summary":      f'{row["device_id"]} 장비에서 {dominant} 이상 패턴 감지. MSE {row["mse"]:.4f} (임계값 {row["threshold"]:.4f} 초과).',
        "action":       "현장 점검 · 센서 상태 확인 · 필요 시 정류기 출력 재조정",
        "comm_status":  row.get("comm_status", "정상통신"),
        "ai_reliability": row.get("ai_reliability", "신뢰"),
    }


# ============================================================
# API 엔드포인트
# ============================================================

@app.get("/api/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": state.model_loaded,
        "last_updated": state.last_updated,
        "device_count": len(state.thresholds),
        "excel_path":   str(state.excel_path) if state.excel_path else None,
    }


@app.get("/api/summary")
def summary():
    """KPI 카드용 요약 데이터."""
    if state.last_results is None:
        raise HTTPException(status_code=503, detail="분석 결과 없음. /api/upload 로 데이터를 업로드하세요.")

    df = state.last_results
    counts = df["risk_level"].value_counts().to_dict()
    comm   = df["comm_status"].value_counts().to_dict()

    return {
        "total":   len(df),
        "normal":  counts.get("정상", 0),
        "anomaly": counts.get("이상", 0),
        "warn":    counts.get("관찰", 0),
        "offline": comm.get("통신고장", 0),
        "last_updated": state.last_updated,
    }


@app.get("/api/devices")
def get_devices():
    """전체 장비 목록 (프론트 EQUIPMENT 배열 형식)."""
    if state.last_results is None:
        raise HTTPException(status_code=503, detail="분석 결과 없음.")

    result_df = state.last_results
    devices   = []

    for i, (_, row) in enumerate(result_df.iterrows()):
        device_id  = str(row["device_id"])
        sensor_row = _device_row_from_master(device_id)
        risk       = str(row.get("risk_level", "정상"))
        comm       = str(row.get("comm_status", "정상통신"))
        status     = _status_from_risk(risk, comm)

        contributions_raw = {
            k: v for k, v in row.items()
            if k not in {"device_id", "group", "mse", "threshold",
                         "risk_level", "comm_status", "ai_reliability", "is_sacrificial_device"}
            and isinstance(v, (int, float))
        }
        contrib_list = _make_contribution_list(contributions_raw)
        label        = _dominant_feature(contributions_raw) if status in ("anomaly", "warn") else None
        lat_offset   = (hash(device_id) % 100 - 50) / 100 * 0.35
        lng_offset   = (hash(device_id + "x") % 100 - 50) / 100 * 0.6

        mse_val = row.get("mse")
        devices.append({
            "id":             i,
            "facilityId":     f"TB-{2021 + (i % 5)}-{str(i+1).zfill(4)}",
            "deviceId":       device_id,
            "location":       f"{device_id} 설치 위치",
            "status":         status,
            "label":          label,
            "volt":           sensor_row.get("volt", -850),
            "ac":             sensor_row.get("ac", 0),
            "sacrificial":    sensor_row.get("sacrificial", 0),
            "temp":           sensor_row.get("temp", 20),
            "hum":            sensor_row.get("hum", 50),
            "commDbm":        sensor_row.get("commDbm", -60),
            "commOk":         sensor_row.get("commOk", True),
            "mse":            round(float(mse_val), 3) if mse_val is not None and not np.isnan(float(mse_val)) else None,
            "contribution":   contrib_list,
            "threshold":      round(float(row.get("threshold", 0.409)), 3),
            "updatedAt":      sensor_row.get("updatedAt", "")[-5:] or "00:00",
            "nextCollectMin": 60 - (i % 60),
            "zone":           ZONE_MAP[hash(device_id) % 4],
            "lat":            37.5 + lat_offset,
            "lng":            127.0 + lng_offset,
            "comm_status":    comm,
            "ai_reliability": str(row.get("ai_reliability", "신뢰")),
            "is_sacrificial": bool(row.get("is_sacrificial_device", False)),
        })

    return {"devices": devices, "count": len(devices), "last_updated": state.last_updated}


@app.get("/api/anomalies")
def get_anomalies():
    """AI 이상/관찰 장비 목록 (프론트 AI_ANOMALIES / AI_WATCH 형식)."""
    if state.last_results is None:
        raise HTTPException(status_code=503, detail="분석 결과 없음.")

    df = state.last_results

    anomalies = []
    watch     = []

    for _, row in df.sort_values("mse", ascending=False).iterrows():
        risk = str(row.get("risk_level", "정상"))
        if risk == "이상":
            anomalies.append(_result_row_to_anomaly(row.to_dict(), len(anomalies)))
        elif risk == "관찰":
            watch.append(_result_row_to_anomaly(row.to_dict(), len(watch)))

    return {
        "anomalies":   anomalies,
        "watch":       watch,
        "last_updated": state.last_updated,
    }


class PredictRequest(BaseModel):
    device_id: str


@app.post("/api/predict/{device_id}")
async def predict_single(device_id: str):
    """단일 장비 실시간 이상 탐지."""
    if not state.model_loaded:
        raise HTTPException(status_code=503, detail="모델 미로드")
    if state.master_df is None:
        raise HTTPException(status_code=503, detail="데이터 없음")

    from gas_common_model_predict import predict_device_window

    device_df = state.master_df[state.master_df["장비번호"] == device_id].copy()
    if device_df.empty:
        raise HTTPException(status_code=404, detail=f"장비 {device_id} 데이터 없음")

    feature_cols = state.config.get("feature_columns", [])
    time_steps   = state.config.get("time_steps", 24)

    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: predict_device_window(
            device_window_df=device_df,
            model=state.model,
            scaler_map=state.scaler_map,
            thresholds=state.thresholds,
            feature_cols=feature_cols,
            time_steps=time_steps,
        )
    )

    contrib_list = _make_contribution_list(result.get("feature_contributions", {}))
    return {
        "device_id":      result["device_id"],
        "mse":            round(result["mse"], 6),
        "threshold":      round(result["threshold"], 6),
        "risk_level":     result["risk_level"],
        "comm_status":    result["comm_status"],
        "ai_reliability": result["ai_reliability"],
        "contribution":   contrib_list,
        "is_sacrificial": result["is_sacrificial_device"],
    }


@app.post("/api/upload")
async def upload_excel(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """엑셀 파일 업로드 후 백그라운드 재분석."""
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="xlsx 파일만 허용됩니다.")

    upload_path = AI_SCRIPTS_DIR / file.filename
    content     = await file.read()
    with open(upload_path, "wb") as f:
        f.write(content)

    background_tasks.add_task(_run_analysis, upload_path)
    return {"message": f"{file.filename} 업로드 완료. 백그라운드 분석 시작됨.", "path": str(upload_path)}


@app.get("/api/insights")
def get_insights():
    """AI 조언 패널용 인사이트 생성."""
    if state.last_results is None:
        return {"insights": []}

    df  = state.last_results
    ins = []

    anomaly_df = df[df["risk_level"] == "이상"]
    if not anomaly_df.empty:
        worst = anomaly_df.sort_values("mse", ascending=False).iloc[0]
        ins.append({
            "kind":      "now",
            "timeLabel": "긴급",
            "body":      f'{worst["device_id"]}에서 이상 스코어 {worst["mse"]:.4f} 감지. 즉각 현장 점검이 필요합니다.',
        })

    watch_df = df[df["risk_level"] == "관찰"]
    if not watch_df.empty:
        ins.append({
            "kind":      "soon",
            "timeLabel": "예측",
            "body":      f'관찰 대상 {len(watch_df)}개 장비의 MSE가 임계값 근접 중. 48시간 내 집중 모니터링 권장.',
        })

    comm_fault = df[df["comm_status"] == "통신고장"]
    if not comm_fault.empty:
        ins.append({
            "kind":      "long",
            "timeLabel": "제안",
            "body":      f'통신 고장 {len(comm_fault)}개 장비: {", ".join(comm_fault["device_id"].tolist()[:3])}. 게이트웨이 점검 및 안테나 재조정 권장.',
        })

    if not ins:
        ins.append({
            "kind":      "long",
            "timeLabel": "정상",
            "body":      f'전체 {len(df)}개 장비 모두 정상 범위. 다음 수집 주기에 재분석 예정.',
        })

    return {"insights": ins, "last_updated": state.last_updated}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
