#!/bin/bash
# AI 예측 배치 실행 래퍼 — siwon DB → LSTM AE → ai_predictions INSERT.
# launchd(com.siwon.ai.predict)에서 매시 호출. 수동 실행도 가능: bash run-predict.sh
# DB 자격증명은 ~/PJHwork/secrets/local/siwon-db.env (600) 에서만 읽음 — plist 에 노출 X.
set -euo pipefail
set -a
. "$HOME/PJHwork/secrets/local/siwon-db.env"
set +a
REPO="$HOME/PJHwork/projects/team/IoT-Monitoring-system"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ai.predict 시작"
exec "$REPO/ai/venv/bin/python" "$REPO/ai/scripts/predict_to_mysql.py" "$@"
