#!/bin/bash
# 캡스톤 통합관제 웹 서버(Express, :5050) 실행 래퍼.
# DB 자격증명은 ~/PJHwork/secrets/local/siwon-db.env (600) 에서만 읽음 — plist 에 노출 X.
# server.js 는 __dirname 기준이라 cwd 무관(정적 dist/ + ai/config 자동 해석).
set -euo pipefail
set -a
. "$HOME/PJHwork/secrets/local/siwon-db.env"
if [ -f "$HOME/PJHwork/secrets/local/openai.env" ]; then . "$HOME/PJHwork/secrets/local/openai.env"; fi
set +a
export PORT="${PORT:-5050}"
exec /Users/pjh/local/node/bin/node /Users/pjh/PJHwork/projects/team/IoT-Monitoring-system/front/server.js
