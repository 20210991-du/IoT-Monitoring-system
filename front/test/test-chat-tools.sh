#!/bin/bash
# Function Calling 도구 12개 회귀 테스트
# 사용법:
#   BASE=http://localhost:5050 ./test/test-chat-tools.sh
#   ssh pjh@macstudio 'BASE=http://localhost:5050 bash -s' < front/test/test-chat-tools.sh
#
# 각 케이스: 자연어 질문 → LLM 이 적절한 도구를 자동 선택했는지 + 응답 검증.

set -uo pipefail

BASE="${BASE:-http://localhost:5050}"
PASS=0
FAIL=0
FAILED_CASES=()

bold() { printf "\033[1m%s\033[0m" "$1"; }
green() { printf "\033[32m%s\033[0m" "$1"; }
red()   { printf "\033[31m%s\033[0m" "$1"; }
gray()  { printf "\033[90m%s\033[0m" "$1"; }
preview() {
  local n="${1:-120}"
  python3 -c "import sys; print(sys.stdin.read().replace('\n', ' ')[:$n], end='')"
}

# 한 케이스 실행: 질문 + 기대 도구 이름 (정규식)
test_case() {
  local title="$1"
  local question="$2"
  local expected_tool="$3"           # 정규식, 빈 문자열이면 도구 호출 안 해도 OK

  local payload
  payload=$(cat <<JSON
{"message":"${question}","context":{},"history":[]}
JSON
)

  printf "\n%s %s\n" "$(bold "▸")" "$(bold "$title")"
  printf "  %s %s\n" "$(gray "질문:")" "$question"

  local resp
  resp=$(curl -s --max-time 90 -X POST "${BASE}/api/chat" \
    -H "Content-Type: application/json" -d "$payload")

  if [[ -z "$resp" ]]; then
    printf "  %s 응답 없음\n" "$(red FAIL)"
    FAIL=$((FAIL + 1))
    FAILED_CASES+=("$title")
    return
  fi

  local ok reply rounds tools
  ok=$(echo "$resp" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('ok',False))" 2>/dev/null || echo "False")
  reply=$(echo "$resp" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('reply',''))" 2>/dev/null || echo "")
  rounds=$(echo "$resp" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('rounds',0))" 2>/dev/null || echo "0")
  tools=$(echo "$resp" | python3 -c "import json,sys; d=json.load(sys.stdin); print(','.join([t['name'] for t in d.get('toolCalls',[])]))" 2>/dev/null || echo "")

  if [[ "$ok" != "True" ]]; then
    printf "  %s ok=false\n" "$(red FAIL)"
    FAIL=$((FAIL + 1))
    FAILED_CASES+=("$title")
    return
  fi

  if [[ -n "$expected_tool" ]]; then
    if [[ ! "$tools" =~ $expected_tool ]]; then
      printf "  %s 도구 [%s] 호출 안 됨. 실제: [%s]\n" "$(red FAIL)" "$expected_tool" "$tools"
      printf "  %s %s\n" "$(gray "응답:")" "$(printf "%s" "$reply" | preview 150)"
      FAIL=$((FAIL + 1))
      FAILED_CASES+=("$title")
      return
    fi
  fi

  if [[ ${#reply} -lt 10 ]]; then
    printf "  %s 응답이 너무 짧음 (%s 자)\n" "$(red FAIL)" "${#reply}"
    FAIL=$((FAIL + 1))
    FAILED_CASES+=("$title")
    return
  fi

  printf "  %s rounds=%s · tools=[%s]\n" "$(green PASS)" "$rounds" "$tools"
  printf "  %s %s%s\n" "$(gray "응답:")" "$(printf "%s" "$reply" | preview 120)" "$([ ${#reply} -gt 120 ] && echo "...")"
  PASS=$((PASS + 1))
}

echo "============================================================"
echo " Function Calling 도구 회귀 테스트"
echo " BASE = $BASE"
echo "============================================================"

# 헬스 체크
echo "--- 헬스 체크 ---"
health=$(curl -s --max-time 10 "${BASE}/api/health")
if [[ -z "$health" ]]; then
  echo "$(red 서버 응답 없음). 종료."
  exit 1
fi
echo "$health" | python3 -m json.tool | head -10

# 12 도구 × 한 케이스씩
test_case "01 list_devices(offline)"      "현재 통신 두절 단말 알려줘"                            "list_devices"
test_case "02 get_device_detail"          "TB24-250401 의 상태를 알려줘"                          "get_device_detail"
test_case "03 get_device_history"         "TB24-250402 의 최근 24시간 방식전위 추이"             "get_device_history"
test_case "04 get_alarms"                 "최근 30일 동안 발생한 알람 보여줘"                     "get_alarms"
test_case "05 get_summary"                "전체 시스템 KPI 카운트 알려줘"                         "get_summary"
test_case "06 get_aggregate(avg RSSI)"    "전체 단말 평균 RSSI 가 얼만지"                         "get_aggregate"

test_case "07 find_devices_by_value"      "방식전위가 -800 mV 이상인 단말 알려줘"                "find_devices_by_value"
test_case "08 get_zone_summary"           "제1구역 통계 알려줘"                                  "get_zone_summary"
test_case "09 compare_devices"            "TB24-250401 과 TB24-250402 의 센서값 비교해줘"        "compare_devices"
test_case "10 get_recent_changes"         "TB24-250402 의 최근 24시간 방식전위 변화량 통계"      "get_recent_changes"
test_case "11 get_maintenance_log"        "최근 30일 점검 이력 보여줘"                           "get_maintenance_log"
test_case "12 get_predictions"            "AI 예측 결과 알려줘"                                  "get_predictions"

# 도구 사용 안 해도 답할 수 있는 케이스 (도메인 질문)
test_case "13 domain (no tool needed)"    "방식전위가 뭐야?"                                     ""

echo
echo "============================================================"
echo " 결과: $(green "$PASS PASS") · $(red "$FAIL FAIL")"
if [[ $FAIL -gt 0 ]]; then
  echo " 실패 케이스:"
  for c in "${FAILED_CASES[@]}"; do echo "   - $c"; done
fi
echo "============================================================"

exit $FAIL
