#!/usr/bin/env bash
# PR evidence steps for action.yml.
#
#   pr-evidence.sh prepare  writes the evidence packet and cognitive debt files
#   pr-evidence.sh check    writes review.json and review.md, posts one PR
#                           comment, and exits with the check result
#
# Inputs come from environment variables only. Values are passed to
# agent-strace as single arguments and are never evaluated by the shell.
#
#   AGENT_TRACE_DIR          trace directory (default .agent-traces)
#   PR_EVIDENCE_DIR          output directory (default $RUNNER_TEMP/agent-trace-pr-evidence)
#   REVIEW_FAIL_ON           --fail-on value, empty uses config
#   REVIEW_MIN_CONFIDENCE    --min-confidence value, empty uses config
#   REVIEW_COMMITS           --commits range for external trailers (check)
#   REVIEW_OTEL_LOGS         --otel-logs Claude Code OTel export (check)
#   COGNITIVE_DEBT_MODE      false, report, or check
#   ARTIFACT_URL             https URL of the uploaded evidence artifact (check)
#   REVIEW_POST_COMMENT      true to post or update the PR comment (check)
#   PR_HEAD_REPOSITORY       head repository of the pull request (check)
#   PR_BASE_REPOSITORY       repository running the workflow (check)

set -uo pipefail

mode="${1:-}"
trace_dir="${AGENT_TRACE_DIR:-.agent-traces}"
base_tmp="${RUNNER_TEMP:-${TMPDIR:-/tmp}}"
dir="${PR_EVIDENCE_DIR:-$base_tmp/agent-trace-pr-evidence}"
output_file="${GITHUB_OUTPUT:-/dev/null}"
cognitive_mode="${COGNITIVE_DEBT_MODE:-report}"

packet="$dir/evidence-packet.html"
debt_json="$dir/cognitive-debt.json"
debt_badge="$dir/cognitive-debt-badge.json"
review_json="$dir/review.json"
review_md="$dir/review.md"

emit() {
  case "$2" in
    *$'\n'* | *$'\r'*)
      echo "::warning::Output $1 has a line break and was not set."
      return
      ;;
  esac
  printf '%s=%s\n' "$1" "$2" >> "$output_file"
}

has_sessions() {
  find "$trace_dir" -mindepth 2 -name meta.json -print -quit 2>/dev/null | grep -q .
}

check_cognitive_mode() {
  case "$cognitive_mode" in
    false | report | check) ;;
    *)
      echo "::error::cognitive-debt must be false, report, or check."
      exit 2
      ;;
  esac
}

# Print the first linked session ID from review JSON on stdin, if it is safe.
first_session() {
  python3 -c '
import json, re, sys
try:
    sid = json.load(sys.stdin)["sessions"][0]["session_id"]
except Exception:
    sys.exit(0)
if isinstance(sid, str) and re.fullmatch(r"[A-Za-z0-9._-]{1,128}", sid):
    print(sid)
'
}

# Print "status result" from the cognitive debt block of review JSON.
cognitive_fields() {
  python3 -c '
import json, sys
try:
    block = json.load(open(sys.argv[1], encoding="utf-8"))["artifacts"]["cognitive_debt"]
except Exception:
    block = {}
status = block.get("status") if isinstance(block.get("status"), str) else ""
result = block.get("result") if block.get("result") in ("pass", "fail", "unavailable") else "unavailable"
print(status or "-", result)
' "$1"
}

prepare() {
  check_cognitive_mode
  mkdir -p "$dir" || exit 2
  rm -f "$packet" "$debt_json" "$debt_badge" "$review_json" "$review_md"
  emit artifacts-dir "$dir"

  if ! has_sessions; then
    echo "::notice::No agent-trace sessions found; no evidence packet or cognitive debt report."
    emit session-id ""
    return 0
  fi

  sid="$(agent-strace --trace-dir "$trace_dir" review --format json --fail-on none 2>/dev/null | first_session)"
  emit session-id "$sid"
  if [ -z "$sid" ]; then
    echo "::notice::No linked session; no evidence packet or cognitive debt report."
    return 0
  fi

  if agent-strace --trace-dir "$trace_dir" share "$sid" --output "$packet" > /dev/null && [ -s "$packet" ]; then
    emit evidence-packet-path "$packet"
  else
    rm -f "$packet"
    echo "::notice::Evidence packet was not generated."
  fi

  if [ "$cognitive_mode" != "false" ]; then
    if agent-strace --trace-dir "$trace_dir" cognitive-debt --session "$sid" --format json --output "$debt_json" > /dev/null; then
      emit cognitive-debt-json-path "$debt_json"
    else
      rm -f "$debt_json"
      echo "::notice::Cognitive debt report was not generated."
    fi
    if agent-strace --trace-dir "$trace_dir" cognitive-debt --session "$sid" --format badge --output "$debt_badge" > /dev/null; then
      emit cognitive-debt-badge-path "$debt_badge"
    else
      rm -f "$debt_badge"
    fi
  fi
  return 0
}

check() {
  check_cognitive_mode
  mkdir -p "$dir" || exit 2

  args=()
  if [ -n "${REVIEW_FAIL_ON:-}" ]; then
    args+=(--fail-on "$REVIEW_FAIL_ON")
  fi
  if [ -n "${REVIEW_MIN_CONFIDENCE:-}" ]; then
    args+=(--min-confidence "$REVIEW_MIN_CONFIDENCE")
  fi
  if [ -n "${REVIEW_COMMITS:-}" ]; then
    args+=(--commits "$REVIEW_COMMITS")
  fi
  if [ -n "${REVIEW_OTEL_LOGS:-}" ]; then
    args+=(--otel-logs "$REVIEW_OTEL_LOGS")
  fi
  args+=(--evidence-packet "$packet")
  if [ "$cognitive_mode" != "false" ]; then
    args+=(--cognitive-debt "$debt_json" --cognitive-debt-mode "$cognitive_mode")
  fi
  url="${ARTIFACT_URL:-}"
  case "$url" in
    https://*) args+=(--artifact-url "$url") ;;
    "") ;;
    *) echo "::warning::Ignoring an artifact URL that is not https." ;;
  esac

  status=0
  agent-strace --trace-dir "$trace_dir" review --format json "${args[@]}" > "$review_json" || status=$?
  agent-strace --trace-dir "$trace_dir" review --format markdown "${args[@]}" > "$review_md" || true

  case "$status" in
    0) result=pass ;;
    1) result=fail ;;
    *) result=error ;;
  esac
  emit review-result "$result"
  emit artifacts-dir "$dir"
  if [ -s "$review_json" ]; then
    emit review-json-path "$review_json"
  fi
  if [ -s "$review_md" ]; then
    emit review-markdown-path "$review_md"
    if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
      cat "$review_md" >> "$GITHUB_STEP_SUMMARY"
    fi
  fi
  if [ "$cognitive_mode" != "false" ]; then
    read -r debt_status debt_result < <(cognitive_fields "$review_json")
    [ "$debt_status" = "-" ] && debt_status=""
    emit cognitive-debt-status "$debt_status"
    emit cognitive-debt-result "$debt_result"
  fi

  if [ "${REVIEW_POST_COMMENT:-false}" = "true" ]; then
    head_repo="${PR_HEAD_REPOSITORY:-}"
    if [ -n "$head_repo" ] && [ "$head_repo" != "${PR_BASE_REPOSITORY:-}" ]; then
      echo "::notice::Skipping PR comment for a fork because pull_request tokens are read-only."
    elif ! has_sessions; then
      echo "::notice::No agent-trace sessions found; skipping PR comment."
    elif [ "$status" -le 1 ]; then
      post_status=0
      agent-strace --trace-dir "$trace_dir" pr-comment --review "${args[@]}" || post_status=$?
      if [ "$post_status" -ne 0 ] && [ "$post_status" -ne "$status" ]; then
        echo "::error::The PR comment was not posted."
        status=$post_status
      fi
    fi
  fi

  if [ "$status" -eq 0 ]; then
    emit passed true
  else
    emit passed false
  fi
  exit "$status"
}

case "$mode" in
  prepare) prepare ;;
  check) check ;;
  *)
    echo "usage: pr-evidence.sh prepare|check" >&2
    exit 2
    ;;
esac
