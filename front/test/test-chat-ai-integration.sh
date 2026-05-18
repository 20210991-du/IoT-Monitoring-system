#!/bin/bash
# AI 파트 연결 회귀 — get_ai_model_info + threshold 동봉 + classifyMse
# 사용:
#   ssh pjh@macstudio 'BASE=http://localhost:5050 bash -s' < front/test/test-chat-ai-integration.sh

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
  local title="$1" question="$2" expected_tool="$3"
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
    printf "  %s 빈 응답\n" "$(red FAIL)"; FAIL=$((FAIL+1)); FAILED_CASES+=("$title"); return
  fi

  local ok reply tools
  ok=$(echo "$resp" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('ok',False))" 2>/dev/null || echo "False")
  reply=$(echo "$resp" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('reply',''))" 2>/dev/null || echo "")
  tools=$(echo "$resp" | python3 -c "import json,sys; d=json.load(sys.stdin); print(','.join([t['name'] for t in d.get('toolCalls',[])]))" 2>/dev/null || echo "")

  if [[ "$ok" != "True" ]]; then
    printf "  %s ok=false\n" "$(red FAIL)"; FAIL=$((FAIL+1)); FAILED_CASES+=("$title"); return
  fi
  if [[ ${#reply} -lt 15 ]]; then
    printf "  %s 응답 짧음 (%s자)\n" "$(red FAIL)" "${#reply}"; FAIL=$((FAIL+1)); FAILED_CASES+=("$title"); return
  fi
  if [[ -n "$expected_tool" && ! "$tools" =~ $expected_tool ]]; then
    printf "  %s 예상 도구 [%s] 미사용, 실제: [%s]\n" "$(red FAIL)" "$expected_tool" "$tools"
    FAIL=$((FAIL+1)); FAILED_CASES+=("$title"); return
  fi
  printf "  %s tools=[%s]\n" "$(green PASS)" "$tools"
  printf "  %s %s%s\n" "$(gray "A:")" "$(echo "$reply" | head -c 150 | tr '\n' ' ')" "$([ ${#reply} -gt 150 ] && echo "...")"
  PASS=$((PASS+1))
}

echo "============================================================"
echo " AI 파트 연결 회귀 — BASE=$BASE"
echo "============================================================"

test_case "A01 AI 모델 학습 방식"          "AI 모델 어떻게 학습됐어?"                     "get_ai_model_info"
test_case "A02 단말 threshold (특정)"       "TB24-250401 의 정상 한계 MSE 는?"            "get_ai_model_info"
test_case "A03 다른 단말 threshold"         "TB24-250455 의 threshold 알려줘"             "get_ai_model_info"
test_case "A04 희생전류 단말 목록"          "희생전류 단말은 어느 거야?"                   "get_ai_model_info|get_device_detail|list_devices"
test_case "A05 학습 피처"                   "AI 모델이 사용하는 입력 피처는 뭐야?"          "get_ai_model_info"
test_case "A06 위험도 분류 기준"            "위험도 단계 어떻게 정해?"                      ""
test_case "A07 get_device_detail 의 ai"     "TB24-250410 의 AI 정상 한계와 현재 상태"      "get_device_detail|get_ai_model_info"
test_case "A08 predictions stub"            "TB24-250401 LSTM 예측 결과"                  "get_predictions|get_ai_model_info"
test_case "A09 전체 threshold 평균"          "단말들 평균 threshold 는 얼만지?"              "get_ai_model_info"
test_case "A10 비율 답변 확인"              "TB24-250403 의 threshold 와 70% 한계 알려줘" "get_ai_model_info"

echo
echo "============================================================"
echo " 결과: $(green "$PASS PASS") · $(red "$FAIL FAIL")"
if [[ $FAIL -gt 0 ]]; then
  echo " 실패:"
  for c in "${FAILED_CASES[@]}"; do echo "   - $c"; done
fi
echo "============================================================"
exit $FAIL
