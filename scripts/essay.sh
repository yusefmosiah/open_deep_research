#!/usr/bin/env bash
set -euo pipefail

# Simple CLI to call the LangGraph API with a prompt and write outputs to files.
# - Accepts a prompt via positional arg, --file, or stdin (pipe)
# - Defaults to the "Essay Writer" graph
# - Saves final_report to --out if provided (or prints to stdout)
# - Optionally saves full JSON (non-stream) or NDJSON events (stream) via --json-out
#
# Requirements: curl, jq
# Env overrides:
#   BASE_URL (default http://127.0.0.1:2024)
#   THREAD_ID (default: essay-<epoch seconds>)

usage() {
  cat <<'USAGE'
Usage:
  scripts/essay.sh [options] [PROMPT]

Options:
  -g, --graph NAME       Graph name (default: "Essay Writer")
  -u, --base-url URL     API base URL (default: $BASE_URL or http://127.0.0.1:2024)
  -o, --out FILE         Write final report markdown to FILE (default: stdout)
  -j, --json-out FILE    Save raw JSON (invoke) or NDJSON (stream) to FILE
  -t, --thread ID        Thread/session id (default: $THREAD_ID or essay-<epoch>)
  -f, --file PATH        Read prompt from text file
  -s, --stream           Use streaming endpoint and extract final report from events
  -h, --help             Show help

Input precedence:
  PROMPT arg > --file PATH > stdin (pipe)

Examples:
  scripts/essay.sh "Write a 1k-word essay on AI in education"
  scripts/essay.sh -f prompt.txt -o out/essay.md
  echo "Market dynamics of residential solar + storage" | scripts/essay.sh -o out/essay.md
  scripts/essay.sh -s -j out/events.ndjson "Nuclear energy: risks and benefits"
USAGE
}

# Defaults
BASE_URL="${BASE_URL:-http://127.0.0.1:2024}"
GRAPH="Essay Writer"
OUT=""
JSON_OUT=""
THREAD_ID="${THREAD_ID:-}"
STREAM=0
INPUT_FILE=""
PROMPT=""
API_BASE_SEG=""

is_uuid() {
  local s="$1"
  [[ "$s" =~ ^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$ ]]
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -g|--graph) GRAPH="$2"; shift 2;;
    -u|--base-url) BASE_URL="$2"; shift 2;;
    -o|--out) OUT="$2"; shift 2;;
    -j|--json-out) JSON_OUT="$2"; shift 2;;
    -t|--thread) THREAD_ID="$2"; shift 2;;
    -f|--file) INPUT_FILE="$2"; shift 2;;
    -s|--stream) STREAM=1; shift;;
    -h|--help) usage; exit 0;;
    --) shift; break;;
    -*) echo "Unknown option: $1" >&2; usage; exit 1;;
    *) PROMPT="$*"; break;;
  esac
done

# Validate tools
command -v curl >/dev/null 2>&1 || { echo "Error: curl not found" >&2; exit 1; }
command -v jq >/dev/null 2>&1 || { echo "Error: jq not found" >&2; exit 1; }

# Read prompt from precedence: PROMPT arg > --file > stdin
if [[ -n "$PROMPT" ]]; then
  : # use as-is
elif [[ -n "${INPUT_FILE}" ]]; then
  if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: file not found: $INPUT_FILE" >&2; exit 1
  fi
  PROMPT=$(cat "$INPUT_FILE")
else
  # if stdin is piped
  if [ ! -t 0 ]; then
    PROMPT=$(cat)
  fi
fi

if [[ -z "$PROMPT" ]]; then
  echo "Error: no prompt provided (arg, --file, or stdin)" >&2
  usage
  exit 1
fi

# Encode graph name for URL
GRAPH_ENC=$(jq -rn --arg x "$GRAPH" '$x|@uri')

# Build input payload safely via jq (threadless uses only `input`)
TMP_DIR=$(mktemp -d)
INPUT_TEMPLATE="$TMP_DIR/input.json"
INPUT_FILLED="$TMP_DIR/input_filled.json"

cat > "$INPUT_TEMPLATE" <<'JSON'
{ "messages": [ { "role": "human", "content": "" } ] }
JSON

jq --arg msg "$PROMPT" \
  '.messages[0].content=$msg' \
  "$INPUT_TEMPLATE" > "$INPUT_FILLED"

set +e
if [[ "$STREAM" -eq 1 ]]; then
  # STREAMING: use runs/stream (threadless) or threads/{id}/runs/stream (threaded)
  if [[ -n "$THREAD_ID" ]]; then
    if ! is_uuid "$THREAD_ID"; then
      echo "Invalid thread ID" >&2; STATUS=22; set -e; rm -rf "$TMP_DIR"; exit 22;
    fi
    URL="${BASE_URL%/}/threads/${THREAD_ID}/runs/stream"
    RUN_PAYLOAD="$TMP_DIR/run_payload.json"
    jq -n --arg assistant "$GRAPH" --slurpfile input "$INPUT_FILLED" '{assistant_id:$assistant, input:$input[0]}' > "$RUN_PAYLOAD"
  else
    URL="${BASE_URL%/}/runs/stream"
    RUN_PAYLOAD="$TMP_DIR/run_payload.json"
    jq -n --arg assistant "$GRAPH" --slurpfile input "$INPUT_FILLED" '{assistant_id:$assistant, input:$input[0], stream_mode:"messages-tuple"}' > "$RUN_PAYLOAD"
  fi
  if [[ -n "$JSON_OUT" ]]; then
    if [[ -n "$OUT" ]]; then
      curl -sN -X POST "$URL" -H "Content-Type: application/json" --data @"$RUN_PAYLOAD" \
        | tee "$JSON_OUT" \
        | jq -r 'select(.event=="on_chain_end") | .data.output.final_report' > "$OUT"
      STATUS=${PIPESTATUS[2]:-0}
    else
      curl -sN -X POST "$URL" -H "Content-Type: application/json" --data @"$RUN_PAYLOAD" \
        | tee "$JSON_OUT" \
        | jq -r 'select(.event=="on_chain_end") | .data.output.final_report'
      STATUS=${PIPESTATUS[2]:-0}
    fi
  else
    if [[ -n "$OUT" ]]; then
      curl -sN -X POST "$URL" -H "Content-Type: application/json" --data @"$RUN_PAYLOAD" \
        | jq -r 'select(.event=="on_chain_end") | .data.output.final_report' > "$OUT"
      STATUS=$?
    else
      curl -sN -X POST "$URL" -H "Content-Type: application/json" --data @"$RUN_PAYLOAD" \
        | jq -r 'select(.event=="on_chain_end") | .data.output.final_report'
      STATUS=$?
    fi
  fi
else
  # NON-STREAM: create run then wait (threadless or threaded)
  if [[ -n "$THREAD_ID" ]]; then
    if ! is_uuid "$THREAD_ID"; then
      echo "Invalid thread ID" >&2; STATUS=22; set -e; rm -rf "$TMP_DIR"; exit 22;
    fi
    CREATE_URL="${BASE_URL%/}/threads/${THREAD_ID}/runs"
    WAIT_URL="${BASE_URL%/}/threads/${THREAD_ID}/runs/wait"
  else
    CREATE_URL="${BASE_URL%/}/runs"
    WAIT_URL="${BASE_URL%/}/runs/wait"
  fi
  RUN_CREATE_PAYLOAD="$TMP_DIR/run_create.json"
  jq -n --arg assistant "$GRAPH" --slurpfile input "$INPUT_FILLED" '{assistant_id:$assistant, input:$input[0]}' > "$RUN_CREATE_PAYLOAD"
  CREATE_RESP="$TMP_DIR/create_resp.json"
  HTTP_CODE=$(curl -sS -w '\n%{http_code}' -X POST "$CREATE_URL" -H "Content-Type: application/json" --data @"$RUN_CREATE_PAYLOAD" -o "$CREATE_RESP")
  CODE=$(printf "%s" "$HTTP_CODE" | sed '$!d')
  if [[ "$CODE" != "200" && "$CODE" != "201" ]]; then
    cat "$CREATE_RESP" >&2
    STATUS=22
  else
    RUN_ID=$(jq -r '.id // .run_id // empty' "$CREATE_RESP")
    if [[ -z "$RUN_ID" ]]; then
      echo "Could not parse run_id" >&2
      STATUS=22
    else
      WAIT_PAYLOAD="$TMP_DIR/wait_payload.json"
      if [[ -n "$THREAD_ID" ]]; then
        jq -n --arg rid "$RUN_ID" '{run_id:$rid}' > "$WAIT_PAYLOAD"
      else
        jq -n --arg rid "$RUN_ID" --arg assistant "$GRAPH" '{assistant_id:$assistant, run_id:$rid}' > "$WAIT_PAYLOAD"
      fi
      if [[ -n "$JSON_OUT" ]]; then
        HTTP_CODE=$(curl -sS -w '\n%{http_code}' -X POST "$WAIT_URL" -H "Content-Type: application/json" --data @"$WAIT_PAYLOAD" -o "$JSON_OUT")
        CODE=$(printf "%s" "$HTTP_CODE" | sed '$!d')
        if [[ "$CODE" != "200" ]]; then
          cat "$JSON_OUT" >&2
          STATUS=22
        else
          if [[ -n "$OUT" ]]; then
            jq -r '.output.final_report // .final_report' "$JSON_OUT" > "$OUT"
          else
            jq -r '.output.final_report // .final_report' "$JSON_OUT"
          fi
          STATUS=$?
        fi
      else
        RESP_JSON="$TMP_DIR/resp.json"
        HTTP_CODE=$(curl -sS -w '\n%{http_code}' -X POST "$WAIT_URL" -H "Content-Type: application/json" --data @"$WAIT_PAYLOAD" -o "$RESP_JSON")
        CODE=$(printf "%s" "$HTTP_CODE" | sed '$!d')
        if [[ "$CODE" != "200" ]]; then
          cat "$RESP_JSON" >&2
          STATUS=22
        else
          if [[ -n "$OUT" ]]; then
            jq -r '.output.final_report // .final_report' "$RESP_JSON" > "$OUT"
          else
            jq -r '.output.final_report // .final_report' "$RESP_JSON"
          fi
          STATUS=$?
        fi
      fi
    fi
  fi
fi
set -e

# Cleanup
rm -rf "$TMP_DIR"

if [[ ${STATUS:-0} -ne 0 ]]; then
  echo "Request failed (exit $STATUS). If jq said 'parse error', the response may not be JSON. Try saving with --json-out and inspect." >&2
  exit $STATUS
fi
