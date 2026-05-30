#!/bin/bash
# Mac Studio cloudflared tunnel URL → README.md 자동 갱신.
# 호출: launchd 주기 실행 (5분) 또는 cloudflared 재시작 직후.
# 변경 없으면 무동작. 변경 시 commit + push.

set -uo pipefail

REPO="$HOME/work/IoT-Monitoring-system"
TUNNEL_LOG="$HOME/tunnel.log"
SCRIPT_LOG="$HOME/work/sync/log/update-readme-url.log"
README="$REPO/README.md"

mkdir -p "$(dirname "$SCRIPT_LOG")"

log() { echo "[$(date '+%F %T')] $*" >> "$SCRIPT_LOG"; }

# 1) URL 추출 — tunnel.log 의 가장 최근 'INF' 라인에 있는 trycloudflare URL
if [ ! -f "$TUNNEL_LOG" ]; then
  log "tunnel.log not found ($TUNNEL_LOG)"
  exit 1
fi

URL=$(grep "trycloudflare.com" "$TUNNEL_LOG" \
      | grep -oE 'https://[a-zA-Z0-9.-]+\.trycloudflare\.com' \
      | tail -1)

if [ -z "$URL" ]; then
  log "URL 추출 실패"
  exit 2
fi

# 2) 살아있는 URL 인지 health check
if ! curl -sf --max-time 10 "$URL/api/health" > /dev/null 2>&1; then
  log "URL 응답 실패: $URL — skip (오래된 URL 일 가능성)"
  exit 3
fi

# 3) README 마커 사이 교체 (Python — sed 보다 안전)
NOW=$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M KST')
/opt/homebrew/bin/python3 - "$README" "$URL" "$NOW" <<'PY'
import sys, re
path, url, now = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path, encoding="utf-8") as f: s = f.read()
new = f"""<!-- TUNNEL_URL_START -->
👉 **<{url}>**

- 헬스 체크: [`/api/health`]({url}/api/health)
- 마지막 확인: {now} (자동 갱신)
<!-- TUNNEL_URL_END -->"""
pat = re.compile(r"<!-- TUNNEL_URL_START -->.*?<!-- TUNNEL_URL_END -->", re.DOTALL)
if not pat.search(s):
    print("MARKERS_MISSING", file=sys.stderr); sys.exit(10)
s2 = pat.sub(new, s)
if s == s2:
    print("NO_CHANGE")
else:
    with open(path, "w", encoding="utf-8") as f: f.write(s2)
    print("UPDATED")
PY
PY_EXIT=$?

if [ $PY_EXIT -eq 10 ]; then
  log "README 마커 누락 — 수동 복구 필요"
  exit 4
fi

# 4) diff 있는 경우만 commit + push
cd "$REPO"
if git diff --quiet README.md; then
  log "URL=$URL · README 변경 없음 (이미 동기화)"
  exit 0
fi

# pull --rebase (충돌 회피)
if ! git pull --rebase --autostash origin main >> "$SCRIPT_LOG" 2>&1; then
  log "pull rebase 실패 — abort"
  git rebase --abort 2>/dev/null
  git stash pop 2>/dev/null
  exit 5
fi

git add README.md
git -c user.name="Mac Studio Tunnel Bot" \
    -c user.email="pjh-tunnel@macstudio.local" \
    commit -m "docs(readme): tunnel URL 자동 갱신 → $(basename "${URL%/}")" >> "$SCRIPT_LOG" 2>&1

if git push origin main >> "$SCRIPT_LOG" 2>&1; then
  log "✅ pushed — $URL"
else
  log "❌ push 실패 — 수동 확인 필요"
  exit 6
fi
