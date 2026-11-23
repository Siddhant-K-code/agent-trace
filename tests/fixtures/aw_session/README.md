# aw_session fixtures

Inputs for `tests/test_aw_session_import.py`: gh-aw unified agent session files
(`usage/aw_session.jsonl`, file-format version 1). No real runs.

- `claude_engine.jsonl`: synthetic Claude-engine run. Tool calls that succeed, fail
  (`success: false`), report no outcome, never complete, and complete with no call;
  firewall usage with a repeated `requestId`; an untimed message; an `otel.span`; an
  unknown record type; one `session.result` snapshot.
- `codex_engine.jsonl`: synthetic Codex-engine run. OpenAI usage whose input includes
  cached tokens, a failed request with no usage, detection-phase usage, two
  `session.result` snapshots, and a nonzero agent exit.
- `pi_interleaved.jsonl`: copied from github/gh-aw (MIT, see `LICENSE-gh-aw`). Two
  interleaved Pi sub-agents with their requests and proxy usage.
