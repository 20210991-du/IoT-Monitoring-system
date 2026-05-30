#!/bin/bash
# siwon.msbuger.com Cloudflare Named Tunnel 실행 래퍼.
# 토큰은 ~/PJHwork/secrets/local/cloudflare.env (600) 에서만 읽음 — plist 에 노출 X.
# 원격 관리(config_src=cloudflare) 터널이라 cert.pem/config.yml 불필요, --token 만으로 동작.
set -euo pipefail
set -a
. "$HOME/PJHwork/secrets/local/cloudflare.env"
set +a
exec /Users/pjh/local/node/bin/cloudflared tunnel run --token "$CF_TUNNEL_TOKEN"
