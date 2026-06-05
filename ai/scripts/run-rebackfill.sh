#!/bin/bash
# run-rebackfill.sh — 활성 모델로 과거 AI 예측(source='backfill') 8샤드 병렬 재생성.
# server.js 가 spawn (SIWON_DB_* env 는 부모 node 프로세스에서 상속). backfill_predictions.py 가
# 단말별 source='backfill' 행을 DELETE 후 재삽입하므로, 현재 활성 모델 기준으로 전체 1년이 일관 재정렬됨.
cd "$(dirname "$0")" || exit 1
LOGD="$HOME/PJHwork/infra/logs"
mkdir -p "$LOGD"
rm -f "$LOGD"/rebackfill-shard-*.log
N=8
for K in $(seq 0 $((N - 1))); do
  TF_CPP_MIN_LOG_LEVEL=3 ../venv/bin/python3 -u backfill_predictions.py --shard "$K/$N" \
    > "$LOGD/rebackfill-shard-$K.log" 2>&1 &
done
wait
echo "[rebackfill done] $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOGD/rebackfill.log"
