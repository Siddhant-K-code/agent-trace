# External evidence fixtures

Synthetic inputs for `tests/test_external_evidence.py`. No real sessions.

- `claude_otel_safety.json`: an OTLP/JSON logs export from Claude Code 2.1.310 with
  `safety_event` records, a rejected `tool_decision` that has no safety event, and a
  prompt string that must never appear in output.
- `claude_otel_no_safety.jsonl`: flat JSON Lines records, one bad line, a reject, and
  no `safety_event`.
- `claude_otel_old_version.jsonl`: Claude Code 2.1.200, before `safety_event`.
- `other_service_otel.json`: another tool's OTel export, so no Claude Code events.
- `copilot_squash_message.txt`: a squash-merge message with `Agent-Logs-Url` lines in
  the body.
- `entire_body_trailer_message.txt`: an `Entire-Checkpoint` line outside the final
  trailer block.
