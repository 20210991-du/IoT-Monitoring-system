#!/bin/bash
# 데모 모드 (DEMO=1) 회귀 — 가상 장비 10대 (위험 3 + 이상의심 4 + 통신장애 3) 통합 검증.
#
# 사용:
#   ssh pjh@macstudio 'BASE=http://localhost:5050 bash -s' < front/test/test-chat-demo-mode.sh
#
# 모든 케이스 body 에 demo:true 동봉.

set -uo pipefail

BASE="${BASE:-http://localhost:5050}"
PASS=0
FAIL=0
FAILED_CASES=()

bold() { printf "\033[1m%s\033[0m" "$1"; }
green() { printf "\033[32m%s\033[0m" "$1"; }
red()   { printf "\033[31m%s\033[0m" "$1"; }
gray()  { printf "\033[90m%s\033[0m" "$1"; }

test_case() {
  local title="$1"
  local question="$2"
  local expect_in_reply="$3"        # 응답에 포함돼야 할 문자열 (정규식)

  local payload
  payload=$(cat <<JSON
{"message":"${question}","context":{},"history":[],"demo":true}
JSON
)

  printf "\n%s %s\n" "$(bold "▸")" "$(bold "$title")"
  printf "  %s %s\n" "$(gray "Q:")" "$question"

  local resp
  resp=$(curl -s --max-time 120 -X POST "${BASE}/api/chat" \
    -H "Content-Type: application/json" -d "$payload")

  if [[ -z "$resp" ]]; then
    printf "  %s 빈 응답\n" "$(red FAIL)"
    FAIL=$((FAIL + 1)); FAILED_CASES+=("$title"); return
  fi

  local ok reply tools
  ok=$(echo "$resp" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('ok',False))" 2>/dev/null || echo "False")
  reply=$(echo "$resp" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('reply',''))" 2>/dev/null || echo "")
  tools=$(echo "$resp" | python3 -c "import json,sys; d=json.load(sys.stdin); print(','.join([t['name'] for t in d.get('toolCalls',[])]))" 2>/dev/null || echo "")

  if [[ "$ok" != "True" ]]; then
    printf "  %s ok=false\n" "$(red FAIL)"
    FAIL=$((FAIL + 1)); FAILED_CASES+=("$title"); return
  fi

  if [[ -n "$expect_in_reply" ]] && [[ ! "$reply" =~ $expect_in_reply ]]; then
    printf "  %s 응답에 [%s] 없음\n" "$(red FAIL)" "$expect_in_reply"
    printf "  %s %s\n" "$(gray "A:")" "$(echo "$reply" | head -c 150)"
    FAIL=$((FAIL + 1)); FAILED_CASES+=("$title"); return
  fi

  printf "  %s tools=[%s]\n" "$(green PASS)" "$tools"
  printf "  %s %s\n" "$(gray "A:")" "$(echo "$reply" | head -c 130 | tr '\n' ' ')"
  PASS=$((PASS + 1))
}

echo "============================================================"
echo " 데모 모드 (DEMO=1) 회귀 — BASE=$BASE"
echo "============================================================"

# 직접 endpoint 검증 (LLM 우회)
echo "-- 직접 endpoint 검증 --"
SUM=$(curl -s "${BASE}/api/summary?demo=1")
ALL=$(echo "$SUM" | python3 -c "import json,sys; print(json.load(sys.stdin)['counts']['all'])" 2>/dev/null)
CRIT=$(echo "$SUM" | python3 -c "import json,sys; print(json.load(sys.stdin)['counts']['critical'])" 2>/dev/null)
WARN=$(echo "$SUM" | python3 -c "import json,sys; print(json.load(sys.stdin)['counts']['warn'])" 2>/dev/null)
OFFL=$(echo "$SUM" | python3 -c "import json,sys; print(json.load(sys.stdin)['counts']['offline'])" 2>/dev/null)
if [[ "$ALL" == "65" && "$CRIT" == "3" && "$WARN" == "4" && "$OFFL" == "4" ]]; then
  printf "  %s /api/summary?demo=1 → all=%s critical=%s warn=%s offline=%s (기대 65/3/4/4)\n" "$(green PASS)" "$ALL" "$CRIT" "$WARN" "$OFFL"
  PASS=$((PASS + 1))
else
  printf "  %s /api/summary?demo=1 → all=%s critical=%s warn=%s offline=%s (기대 65/3/4/4)\n" "$(red FAIL)" "$ALL" "$CRIT" "$WARN" "$OFFL"
  FAIL=$((FAIL + 1)); FAILED_CASES+=("/api/summary count")
fi

# 챗봇 케이스 — 데모 단말 노출 확인
test_case "01 위험 단말 → DEMO-001~003 노출"     "현재 위험 단말 알려줘"               "DEMO-00[123]"
test_case "02 이상의심 단말 → DEMO-101~104"      "이상의심 단계 단말 알려줘"           "DEMO-10[1-4]"
test_case "03 통신두절 단말 → DEMO-201~203"      "통신 두절 단말 알려줘"                "DEMO-20[1-3]"
test_case "04 KPI 카운트 → 65/3/4/4"             "전체 KPI 카운트 알려줘"               "(65|위험\\s*3|이상의심\\s*4|통신\\s*장애\\s*4|통신두절\\s*4)"
test_case "05 DEMO-001 상세 → 방식전위·AC"      "DEMO-001 상태 알려줘"                "(방식전위|AC|-540|312)"
test_case "06 DEMO-002 24h 추이"                 "DEMO-002 의 최근 24시간 방식전위 추이" "DEMO-002"
test_case "07 시청 부근 → DEMO-001 포함"         "시청 근처 단말 알려줘"                "(시청|DEMO-001|TB24-250422)"
test_case "08 미룡동 → DEMO-101 포함"            "미룡동 단말 알려줘"                   "(미룡동|DEMO-101|TB24-250405)"
test_case "09 새만금 → DEMO-201 포함"            "새만금방조제 단말 알려줘"             "(새만금|DEMO-201|TB24-250401)"
test_case "10 최근 알람 → DEMO 알람 포함"        "최근 7일 알람 보여줘"                 "(DEMO-00[12]|AC 유입|충격)"
test_case "11 AI 예측 → DEMO 위험 단말"          "AI 예측 결과 보여줘"                  "(위험|DEMO-00[123])"
test_case "12 평균 RSSI → 데모 반영"             "전체 단말 평균 RSSI 알려줘"           "dBm"
test_case "13 비교 → DEMO-001 vs DEMO-002"       "DEMO-001 과 DEMO-002 비교"            "(DEMO-001|DEMO-002)"
test_case "14 변화량 → DEMO-002 mock"            "DEMO-002 의 24시간 방식전위 변화량"   "(DEMO-002|변화|상승|하락|평탄)"
test_case "15 군산시청 근처 + DEMO"              "군산시청 근처 위험 단말 알려줘"       "(DEMO|시청|위험)"

echo
echo "============================================================"
echo " 결과: $(green "$PASS PASS") · $(red "$FAIL FAIL")"
if [[ $FAIL -gt 0 ]]; then
  echo " 실패:"
  for c in "${FAILED_CASES[@]}"; do echo "   - $c"; done
fi
echo "============================================================"

exit $FAIL
