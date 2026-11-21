# Representative evidence fixture

`src/agent_trace/corpus/v1/fixture-review-001/` is a synthetic, versioned session for review-oriented regression tests. It is also the recovered and redacted case in the public [Agent Evidence Corpus](agent-evidence-corpus.md). It contains no customer data, real secrets, or private repository content.

The scenario intentionally includes a session boundary, redacted prompt, file read/write, a failing test command, an error, a retry, successful recovery, a human correction annotation, and one orphaned tool result that models a known capture gap.

## Refresh procedure

1. Keep the session ID and event IDs deterministic unless a schema change requires replacing them.
2. Edit `events.ndjson` using fields accepted by `TraceEvent.from_json()` and `annotations.jsonl` using fields accepted by `Annotation.from_json()`. Use the data keys that capture writes, so the fixture renders like a real session.
3. Update `expected.json` whenever event count, failure/retry/recovery IDs, annotation IDs, the intentional gap, the facts each event shows (`expected_review_facts`), or the health result (`expected_health`) changes. Increase `fixture_version` in `expected.json` and in the `session_start` event when event content changes.
4. Run `agent-strace validate src/agent_trace/corpus/v1/fixture-review-001 --format json`.
5. Run `python -m unittest tests.test_representative_fixture tests.test_review_surface tests.test_trace_validation -v`. These cover the fixture itself; its json, ndjson, csv, anonymized, EU AI Act, OTLP and OTLP GenAI exports; its health result; and CLI replay showing the same ordered events and facts as the review surface.
6. Run `npm test` in `vscode-extension/`. It checks that the extension's review model shows the same ordered facts.
7. Consumers may add assertions against this fixture, but generated expectations should be changed together with their source event change rather than hand-edited independently.

## Walkthrough

This walkthrough uses `fixture_version` 2. Its output comes from the agent-strace source tree at version 0.119.0 after `fixture_version` 2 was added, where `agent-strace --version` prints `0.119.0`. Released packages up to and including 0.119.0 ship `fixture_version` 1, which used fixture-only data keys such as `content`, `tool`, `output` and `message`, so their replay output lacks most of the details below.

Load the fixture into a trace directory, then replay it and check its health:

```bash
mkdir -p traces/fixture-review-001
cp src/agent_trace/corpus/v1/fixture-review-001/{events.ndjson,meta.json,annotations.jsonl} traces/fixture-review-001/
agent-strace --trace-dir traces replay fixture-review-001
agent-strace --trace-dir traces health fixture-review-001
```

After a session summary, replay prints one line per event, in recorded order:

```text
+  0.00s ▶ session_start
+  1.00s 👤 user_prompt
                "[REDACTED synthetic request] fix the failing example test"
+  2.00s 📖 file_read src/example.py
+  3.00s → tool_call bash
                $ python -m pytest -q tests/test_example.py
+  4.10s ✗ error bash
                example test failed
+  5.00s 📝 file_write src/example.py
+  6.00s → tool_call bash
                $ python -m pytest -q tests/test_example.py
+  7.00s ← tool_result bash  (510ms)
                1 passed
+  8.00s ◆ decision keep the minimal fix
                reason: the retry passed
+  9.00s ← tool_result bash
                deliberate orphan used to test capture-gap handling
+ 10.00s 🤖 assistant_response
                "Updated the example and the focused test now passes."
+ 11.00s ■ session_end exit=?
```

The review surface (`agent-strace server --dashboard --storage traces` and the VS Code extension) lists the same events in the same order with the same facts:

| Event | Categories | Summary |
|---|---|---|
| `ev-start` | | session start |
| `ev-prompt` | privacy | [REDACTED synthetic request] fix the failing example test |
| `ev-read` | files | src/example.py |
| `tool-test-1` | commands, tests, tools | bash: python -m pytest -q tests/test_example.py |
| `ev-error` | failures | example test failed |
| `ev-write` | files | src/example.py |
| `tool-test-2` | commands, retry, tests, tools | bash: python -m pytest -q tests/test_example.py |
| `result-test-2` | recovery, tools | 1 passed |
| `ev-decision` | annotations, corrections, decisions | keep the minimal fix (reason: the retry passed) |
| `ev-orphan` | gaps, tools | deliberate orphan used to test capture-gap handling |
| `ev-response` | | Updated the example and the focused test now passes. |
| `ev-end` | | session end |

The `ev-decision` row carries the annotation "Human correction: the retry demonstrates recovery, not proof of task correctness." Replay does not show annotations.

Health reports the session as partial and exits 0:

```text
Evidence health: partial
Session:   fixture-review-001
Provider:  unknown
Finalized: yes
Events:    12
Types:     session_start 1, session_end 1, tool_call 2, tool_result 2, file_read 1, file_write 1, decision 1, error 1, user_prompt 1, assistant_response 1
Redaction: not recorded (this trace does not show whether redaction was on)
Redacted:  1 of 12 events

Observed (1):
  - orphan_tool_result: tool result 'ev-orphan' references unknown parent 'missing-tool-call'

Provider limitations (0):
  not assessed

Capture matrix: no entry applied
Note: Provider not identified from hook-written evidence; no capture-matrix entry was applied, so provider limitations were not assessed.
Matrix version: sha256:53d548b7753eb7ca741e651b30d533bd4535d9264841d4f21448deac168ffb19
```

### Observed facts

These facts are recorded in the events:

- The prompt, marked as redacted.
- A read and a write of `src/example.py`.
- Two `bash` calls that run `python -m pytest -q tests/test_example.py`.
- The error "example test failed" on the first call.
- The result "1 passed" with `exit_code` 0 on the second call.
- The decision "keep the minimal fix" with the reason "the retry passed".
- A tool result whose parent is `missing-tool-call`.
- The final response and the session start and end.

The human correction is an annotation on `ev-decision`, not an event.

### Derived facts

AgentTrace computes these facts. They are not recorded:

- `tests` comes from the command text. `commands` comes from the command text and the tool name.
- `retry` comes from `retry_of` on `tool-test-2`.
- `recovery` comes from a result with `exit_code` 0 under a call that has `retry_of`.
- `linked` and `orphaned` come from checking each `parent_id` against the events in the trace.
- The health status `partial` comes from the orphaned result.
- The validator labels, such as `recovered` and `orphan_parent`, come from the same events.

### Missing facts

- The parent of `ev-orphan` is not in the trace. Replay, review and health keep it missing and do not invent it.
- The first call has no tool result. Its outcome is the error event.
- The session end records no exit code, so replay prints `exit=?`.
- The trace has no token usage, so the validator adds `usage_missing`.
- The trace does not record whether redaction was on, so health prints "Redaction: not recorded".

### Provider-specific facts

- The `session_start` event names provider `synthetic` and capture method `fixture`. The review header shows these recorded values.
- Health identifies a provider only from hook-written evidence. It reports `Provider: unknown` and applies no capture-matrix entry. This is intended, because the fixture is not a real capture.
- Event data uses the keys that capture writes: `prompt`, `tool_name`, `arguments.command`, `result`, `error`, `text` and `path`. `bash` is the tool name that GitHub Copilot CLI hooks record for shell commands. Claude Code and Codex hooks record `Bash`.
- Decision events come from the SDK `log_decision` call, which records `choice` and `reason`.
- Hooks do not write `retry_of` or `exit_code`, so live hook sessions do not show the `retry` and `recovery` categories.

## Trust boundary

The fixture is evidence of AgentTrace's data model and review semantics, not evidence that any real provider exposes every event shown here. Provider-specific coverage belongs in the capture matrix, and the orphan event exists specifically so UIs and health checks must preserve uncertainty instead of inventing a missing parent.
