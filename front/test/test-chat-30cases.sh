#!/bin/bash
# 챗봇 종합 회귀 — 30 케이스 (위치 10 + 기본 도구 10 + 복합 10)
# 사용:
#   ssh pjh@macstudio 'BASE=http://localhost:5050 bash -s' < front/test/test-chat-30cases.sh
#
# 검증: ok=true + reply 길이 >= 15자 + (있으면) expected_tool 정규식 매칭.
# expected_tool 은 "선호" — 미매칭이어도 PASS (단 응답 짧으면 FAIL).

set -uo pipefail

BASE="${BASE:-http://localhost:5050}"
PASS=0
SOFT=0          # expected_tool 미매칭 (소프트 경고)
FAIL=0
FAILED_CASES=()
SOFT_CASES=()

bold() { printf "\033[1m%s\033[0m" "$1"; }
green() { printf "\033[32m%s\033[0m" "$1"; }
yellow() { printf "\033[33m%s\033[0m" "$1"; }
red()   { printf "\033[31m%s\033[0m" "$1"; }
gray()  { printf "\033[90m%s\033[0m" "$1"; }

test_case() {
  local title="$1"
  local question="$2"
  local expected_tool="$3"

  local payload
  payload=$(cat <<JSON
{"message":"${question}","context":{},"history":[]}
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

  local ok reply rounds tools
  ok=$(echo "$resp" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('ok',False))" 2>/dev/null || echo "False")
  if [[ "$ok" != "True" ]]; then
    err=$(echo "$resp" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('error','?'))" 2>/dev/null || echo "?")
    printf "  %s ok=false (%s)\n" "$(red FAIL)" "$err"
    FAIL=$((FAIL + 1)); FAILED_CASES+=("$title"); return
  fi

  reply=$(echo "$resp" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('reply',''))" 2>/dev/null || echo "")
  rounds=$(echo "$resp" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('rounds',0))" 2>/dev/null || echo "0")
  tools=$(echo "$resp" | python3 -c "import json,sys; d=json.load(sys.stdin); print(','.join([t['name'] for t in d.get('toolCalls',[])]))" 2>/dev/null || echo "")

  if [[ ${#reply} -lt 15 ]]; then
    printf "  %s 응답 너무 짧음 (%s 자)\n" "$(red FAIL)" "${#reply}"
    FAIL=$((FAIL + 1)); FAILED_CASES+=("$title"); return
  fi

  # expected_tool 정규식 검증 (있으면)
  local tool_ok=1
  if [[ -n "$expected_tool" ]]; then
    if [[ ! "$tools" =~ $expected_tool ]]; then
      tool_ok=0
    fi
  fi

  if [[ $tool_ok -eq 1 ]]; then
    printf "  %s rounds=%s tools=[%s]\n" "$(green PASS)" "$rounds" "$tools"
    PASS=$((PASS + 1))
  else
    printf "  %s 예상 도구 [%s] 미사용, 실제: [%s] (응답은 정상)\n" "$(yellow "SOFT")" "$expected_tool" "$tools"
    SOFT=$((SOFT + 1)); SOFT_CASES+=("$title")
    PASS=$((PASS + 1))   # SOFT 도 일단 PASS 카운트
  fi

  printf "  %s %s%s\n" "$(gray "A:")" "$(echo "$reply" | head -c 110 | tr '\n' ' ')" "$([ ${#reply} -gt 110 ] && echo "...")"
}

echo "============================================================"
echo " 챗봇 30 케이스 종합 회귀 — BASE=$BASE"
echo "============================================================"

# 헬스 체크
health=$(curl -s --max-time 10 "${BASE}/api/health")
if [[ -z "$health" ]]; then echo "$(red 서버 응답 X)"; exit 1; fi

#  ─────── 위치 / 지도 (10) ───────
test_case "L01 미룡동 단말"              "미룡동 단말 알려줘"                       "search_devices_by_location|list_devices"
test_case "L02 시청 앞 단말"             "시청 앞 단말 알려줘"                      "search_devices_by_location"
test_case "L03 버스터미널 근처"          "버스터미널 근처 단말 있어?"                "search_devices_by_location|find_devices_near"
test_case "L04 은파호수공원 근처"        "은파호수공원 근처 장비 알려줘"             "geocode_location|find_devices_near"
test_case "L05 군산교도소 부근"          "군산교도소 부근 단말 알려줘"              "geocode_location|search_devices_by_location"
test_case "L06 새만금방조제"             "새만금방조제 단말 보여줘"                  "search_devices_by_location"
test_case "L07 군산시립박물관"           "군산시립박물관 인근 단말"                  "search_devices_by_location|geocode_location"
test_case "L08 해망동 DM기술 앞"         "해망동 DM기술 앞 단말"                    "search_devices_by_location"
test_case "L09 TB24-250401 주변"        "TB24-250401 주변 1km 안 단말 알려줘"      "find_devices_near|get_device_detail"
test_case "L10 군산시청 좌표"            "군산시청 좌표 알려줘"                      "geocode_location"

#  ─────── 기본 도구 (10) ───────
test_case "B01 통신 두절"                "통신 두절 단말 알려줘"                      "list_devices"
test_case "B02 단말 상태"                "TB24-250410 상태 알려줘"                   "get_device_detail"
test_case "B03 24h 추이"                 "TB24-250420 어제 방식전위 추이"            "get_device_history"
test_case "B04 최근 알람"                "최근 7일 알람 보여줘"                       "get_alarms"
test_case "B05 전체 KPI"                 "전체 KPI 카운트 알려줘"                    "get_summary"
test_case "B06 평균 RSSI"                "전체 단말 평균 RSSI"                       "get_aggregate"
test_case "B07 방식전위 임계"            "방식전위 -1500 이상 단말 알려줘"           "find_devices_by_value"
test_case "B08 제8구역 통계"             "제8구역 통계 알려줘"                       "get_zone_summary"
test_case "B09 두 단말 비교"             "TB24-250410 과 TB24-250420 비교"          "compare_devices"
test_case "B10 변화량"                   "TB24-250420 의 최근 24시간 방식전위 변화량 통계" "get_recent_changes"

#  ─────── 복합 / 어려운 케이스 (10) ───────
test_case "X01 통신두절 단말 위치"        "통신 두절된 단말 어디에 있어?"            "list_devices|search_devices_by_location|get_device_detail"
test_case "X02 시청에서 가까운 5대"      "군산시청에서 가장 가까운 단말 5대 알려줘"   "geocode_location|find_devices_near"
test_case "X03 제8구역 RSSI 낮은"        "제8구역에서 RSSI 가 가장 낮은 단말은?"    "get_zone_summary|find_devices_by_value|list_devices"
test_case "X04 마지막 측정 언제"          "TB24-250429 마지막 측정 언제야?"           "get_device_detail|list_devices"
test_case "X05 평균 온도"                "전체 단말 평균 온도 알려줘"                 "get_aggregate"
test_case "X06 점검 이력"                "최근 30일 점검 이력 보여줘"                 "get_maintenance_log"
test_case "X07 AI 예측"                  "AI 예측 결과 알려줘"                       "get_predictions"
test_case "X08 방식전위 뭐야"            "방식전위가 뭐야?"                          ""
test_case "X09 군산 전체 평균 방식전위"  "군산 전체 평균 방식전위는?"                "get_aggregate"
test_case "X10 위험 단말 위치"            "위험 단계 단말이 있다면 어디야?"           "get_summary|list_devices|search_devices_by_location"

echo
echo "============================================================"
echo " 결과: $(green "$PASS PASS") · $(yellow "$SOFT SOFT") · $(red "$FAIL FAIL")"
if [[ $FAIL -gt 0 ]]; then
  echo " 실패:"
  for c in "${FAILED_CASES[@]}"; do echo "   - $c"; done
fi
if [[ $SOFT -gt 0 ]]; then
  echo " 예상 도구 미사용 (응답은 정상):"
  for c in "${SOFT_CASES[@]}"; do echo "   - $c"; done
fi
echo "============================================================"

exit $FAIL
