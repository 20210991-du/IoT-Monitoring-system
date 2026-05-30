#!/bin/bash
# 챗봇 의미 회귀 — LLM 답변 "내용"이 현재 단말 체계/표현 규칙과 맞는지 검증.
#   - 실제 단말 ID 체계 (TB24-250xxx) 인용, 낡은 TB24-5JN 금지
#   - AC 500mV 초과는 "초과" 로 표현
#   - 원시 피처명(_dev24, _diff1) 답변 노출 금지
#   - 추이 데이터 없음 ≠ 위험 단말 없음 (잘못된 "위험 단말 없음" 금지)
#   - 날씨 습도/강수 컨텍스트 반영
# 사용:
#   ssh pjh@macstudio 'BASE=http://localhost:5050 bash -s' < front/test/test-chat-semantics.sh
#   (LLM 응답은 약간의 변동성이 있으므로 1~2개 변동 가능 — 0 FAIL 목표)

set -uo pipefail

BASE="${BASE:-http://localhost:5050}"
PASS=0
FAIL=0
FAILED_CASES=()

bold() { printf "\033[1m%s\033[0m" "$1"; }
green() { printf "\033[32m%s\033[0m" "$1"; }
red()   { printf "\033[31m%s\033[0m" "$1"; }
gray()  { printf "\033[90m%s\033[0m" "$1"; }

# ask <question> [context_json] → reply 텍스트를 stdout 으로
ask() {
  local q="$1" ctx="${2:-}"
  [[ -z "$ctx" ]] && ctx='{}'
  local payload
  payload=$(python3 -c "import json,sys; print(json.dumps({'message':sys.argv[1],'context':json.loads(sys.argv[2]),'history':[]}))" "$q" "$ctx") || return 1
  curl -s --max-time 120 -X POST "$BASE/api/chat" \
    -H "Content-Type: application/json" -d "$payload" \
    | python3 -c "import json,sys; print(json.load(sys.stdin).get('reply',''))" 2>/dev/null
}

# check <title> <reply> [has PAT | lacks PAT]...
check() {
  local title="$1"; shift
  local reply="$1"; shift
  printf "\n%s %s\n" "$(bold "▸")" "$(bold "$title")"
  local prev
  prev=$(printf '%s' "$reply" | python3 -c 'import sys; print(sys.stdin.read().replace(chr(10)," ")[:170], end="")')
  printf "  %s %s...\n" "$(gray "A:")" "$prev"
  local okall=1
  if [[ -z "$reply" ]]; then
    printf "    %s 빈 응답\n" "$(red NO)"; okall=0
  fi
  while [[ $# -gt 1 ]]; do
    local mode="$1" pat="$2"; shift 2
    if [[ "$mode" == "has" ]]; then
      if [[ "$reply" == *"$pat"* ]]; then printf "    %s has '%s'\n" "$(green ok)" "$pat"
      else printf "    %s MISSING '%s'\n" "$(red NO)" "$pat"; okall=0; fi
    else # lacks
      if [[ "$reply" != *"$pat"* ]]; then printf "    %s lacks '%s'\n" "$(green ok)" "$pat"
      else printf "    %s FORBIDDEN '%s'\n" "$(red NO)" "$pat"; okall=0; fi
    fi
  done
  if [[ $okall -eq 1 ]]; then PASS=$((PASS+1)); else FAIL=$((FAIL+1)); FAILED_CASES+=("$title"); fi
}

echo "============================================================"
echo " 챗봇 의미 회귀 — BASE=$BASE"
echo "============================================================"

# S01 — 단말 상세 + AC 500mV 초과 표현 + 원시 피처명 금지
R=$(ask "TB24-250448 의 현재 상태와 AC 유입 판정을 짧게 알려줘")
check "S01 TB24-250448 상세 (AC 초과 표현)" "$R" \
  has "TB24-250448" has "초과" lacks "_dev24" lacks "_diff1" lacks "TB24-5JN"

# S02 — 위험 TOP5: 실제 단말 ID 체계 + 낡은 ID/원시 피처명 금지
R=$(ask "지금 위험 단말 TOP 5와 각 단말의 근거 수치를 정리해줘")
check "S02 위험 TOP5 (실제 ID 체계)" "$R" \
  has "TB24-250" lacks "TB24-5JN" lacks "_dev24" lacks "_diff1"

# S03 — AC 500mV 초과 판정 (971mV)
R=$(ask "TB24-250448 의 AC 유입이 500mV 즉각 점검 기준을 넘었는지 판정해줘")
check "S03 AC 500mV 초과 판정" "$R" \
  has "초과" has "500"

# S04 — 추이 데이터 없음 ≠ 위험 단말 없음 (프론트처럼 counts 컨텍스트 전달, trends 는 없음)
S04CTX='{"counts":{"all":55,"normal":30,"critical":2,"warn":22,"offline":1},"criticalNodes":["TB24-250429","TB24-250431"],"warnNodes":["TB24-250442"],"nowText":"2026-05-30 15:00"}'
R=$(ask "현재 위험 단말의 최근 추세를 알려줘" "$S04CTX")
check "S04 추이 데이터 없음 (위험단말 없음과 혼동 금지)" "$R" \
  lacks "위험 단말이 없" lacks "위험 단말은 없" lacks "위험 노드가 없" lacks "위험 단말은 현재 없" lacks "_dev24"

# S05 — 원인 분석: 사람이 읽는 라벨 (원시 피처명 금지)
R=$(ask "TB24-250429 가 왜 위험한지 주요 원인 피처로 설명해줘")
check "S05 원인 분석 (원시 피처명 금지)" "$R" \
  has "TB24-250429" has "AI 기준 대비" lacks "_dev24" lacks "_diff1"

# S06 — 날씨 습도/강수 컨텍스트 반영
WCTX='{"weather":{"temp":18,"ko":"비","code":63,"time":"2026-05-30T14:00","precip":3.2,"humidity":88}}'
R=$(ask "지금 군산 날씨와 습도가 매설배관에 어떤 영향을 줄 수 있어?" "$WCTX")
check "S06 날씨 습도/강수 컨텍스트" "$R" \
  has "습도"

# S07 — 평탄한 시계열이어도 기준 이탈이면 정상이라고 답하지 않기
R=$(ask "TB24-250429 방식전위 최근 24시간 추세와 기준 판정을 알려줘")
check "S07 평탄 추세와 기준 판정 분리" "$R" \
  has "TB24-250429" has "위험" lacks "정상적인 센서 값" lacks "정상적인 센서값" lacks "AI 기준 대비"

echo
echo "============================================================"
echo " 결과: $(green "$PASS PASS") · $(red "$FAIL FAIL")"
if [[ $FAIL -gt 0 ]]; then
  echo " 실패:"
  for c in "${FAILED_CASES[@]}"; do echo "   - $c"; done
fi
echo "============================================================"
exit $FAIL
