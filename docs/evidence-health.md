# Evidence health

Evidence health answers a narrow review question: **is this captured session structurally usable as evidence, and what limitations should the reviewer know about?**

It does not score agent quality, infer misconduct, or claim that a provider exposed everything that happened.

## Versioned result

`agent_trace.evidence_health.assess_evidence_health()` returns a versioned result with one of four statuses:

- `healthy` — the observed stream has the expected boundaries and relationships and no declared capture limitation was supplied.
- `partial` — the stream is reviewable but has an observed gap or a declared provider limitation.
- `unknown` — there is not enough completed evidence to judge yet, for example an active session without an end marker.
- `invalid` — the stream is internally contradictory, for example duplicate boundaries, duplicate event IDs, mixed session IDs, or a non-finite timestamp.

Every non-healthy result includes machine-readable reason codes and human explanations. Reasons also distinguish observed defects from provider limitations.

## Observed checks

The first version checks:

- session start/end boundaries, allowing [continuation starts](#session-boundaries);
- events before start or after end;
- duplicate event IDs;
- mixed session IDs;
- non-finite timestamps and timestamp regressions;
- unpaired tool calls/results once the session is finalized;
- unpaired LLM requests/responses once the session is finalized;
- `ERROR` events with a missing or unknown parent request (retaining the error event ID); and
- export/drop failures explicitly reported by the caller.

The calculation is deterministic and local. It performs no network calls.

For an active session (`session_finalized=False`), a tool call or LLM request
without a terminal outcome is treated as in flight rather than as missing
evidence. Orphan, duplicate, invalid, and out-of-order outcomes are still
reported immediately.

## Session boundaries

Boundaries are checked in event order. The first `session_start` opens the
session and a `session_end` closes it.

A later `session_start` continues the same session when its `source` is
`resume` or `compact` (`CONTINUATION_START_SOURCES`). Claude Code and Codex keep
the session ID when a session is resumed or compacted, and Gemini CLI and GitHub
Copilot CLI keep it on resume, so the hooks record these starts in the existing
trace. A continuation start:

- is not a `duplicate_session_start`;
- re-opens a session that a `session_end` closed, so the next `session_end`
  is not a `duplicate_session_end` and the events after the continuation start
  are not `events_after_session_end`.

These stay `invalid`:

- a later start with any other source is `duplicate_session_start`. This
  includes `startup`, `clear`, `fork`, `new`, a hook event name such as
  `SessionStart`, and a missing source. Only `resume` and `compact` are known
  to keep the session ID, so any other repeated start contradicts the trace;
- a `session_end` with no continuation start since the previous end is
  `duplicate_session_end`;
- any other event after a `session_end` with no continuation start is
  `events_after_session_end`, reported once per closed span on the first such
  event.

A start and an end that each appear twice, with every other event doubled, usually
means the same hooks are registered twice, for example by the GitHub Copilot CLI
plugin and by `agent-strace setup --cli copilot`. Keep one registration; see
[GitHub Copilot CLI hooks](setup.md#github-copilot-cli-hooks).

Source matching is exact and case-sensitive. When a continued session has no
end after its last continuation start, it reports `session_active` or
`missing_session_end` like any other open session. `observed_end` still reports
whether any end marker was captured.

Some providers send events before their session start hook. A caller can
declare those event types with `allowed_pre_start_event_types`. When every
event before the first `session_start` has a declared type, the result reports
one `declared_events_before_session_start` provider limitation instead of the
invalid `events_before_session_start`. Any undeclared event type before the
start is still invalid, and a session with no start is still
`missing_session_start`. `agent-strace health`, `agent-strace validate` and the
`agent.strace.evidence.*` attributes of `--format otlp-genai` take the declared
types from the capture matrix's `events_before_start`.

Some providers record results with no request event. A caller can declare
those result types with `allowed_unrequested_result_types`. Result events of a
declared type with no `parent_id` are then reported once as the provider
limitation `declared_results_without_requests`, instead of one
`orphan_<type>` each. A result that names a request missing from the trace is
still an orphan. `agent-strace health`, `agent-strace validate` and the
`agent.strace.evidence.*` attributes of `--format otlp-genai` take the declared
types from the capture matrix's `results_without_requests`; Codex declares
`llm_response`, because AgentTrace records its model requests from
`PostModelRequest` alone.

## Provider limitations

A missing event alone is not enough to conclude that capture failed: some providers never expose certain categories of activity. Callers should therefore pass known blind spots from a versioned provider/capture matrix:

```python
from agent_trace.evidence_health import assess_evidence_health

health = assess_evidence_health(
    events,
    provider="codex",
    capture_method="hooks",
    capture_matrix_version="<actual matrix version or stable reference>",
    provider_blind_spots=[
        "example limitation supplied by the capture matrix",
    ],
    session_finalized=True,
)
```

A declared blind spot makes the result `partial` even when the observed event stream is structurally clean. Include the applicable `capture_matrix_version` in the assessment so downstream consumers can interpret provider limitations against the version of the matrix used to declare them. An empty version means the adapter did not supply a reference; it must not be interpreted as verified completeness. This prevents `healthy` from becoming a false claim about activity a provider cannot expose.

## External evidence

Other tools record evidence that agent-strace did not capture. `health`,
`review` and `pr-comment --review` can read three kinds of it with
`--commits RANGE` and `--otel-logs FILE`. Each one is reported as an external
signal with a source, a status and a confidence. Signals are reported, not
graded: they never change the health status, a review finding's status, the
threshold result or the exit code. A trailer alone never shows that the
evidence behind it is complete.

| Status | Meaning |
|---|---|
| `present` | The tool's evidence exists where it says it is |
| `unpushed` | The trailer is present but the checkpoint has not reached the remote |
| `missing` | The input was read and has no evidence from this tool |
| `inconclusive` | The input cannot say either way |

Nothing is fetched. Remote state is read only from local remote-tracking refs,
so a stale fetch can make pushed evidence read as unpushed. With
`actions/checkout`, set `fetch-depth: 0` so the range is in the clone.

### Entire checkpoints

Entire links a commit to a checkpoint with an `Entire-Checkpoint: <id>`
trailer. Since Entire CLI 0.13.1, only lines in the commit message's last
paragraph link; a line pasted into the body does not. The ID is 12 lowercase
hex characters or a 26 character ULID. Entire stores checkpoints either as
refs under `refs/entire/checkpoints/` or on the `entire/checkpoints/v1`
branch, and since 0.13.1 a clone can keep them local until they are sent.

| Signal | Status | Confidence |
|---|---|---|
| Ref exists and is listed in `entire-checkpoint-push-queue.jsonl` | `unpushed` | high |
| Ref exists, not queued, `push_sessions` is `false` in `.entire/settings.json` or `settings.local.json` | `unpushed` | medium |
| Ref exists, not queued | `present` | medium. The remote is not checked. |
| On `entire/checkpoints/v1` and on a `refs/remotes/*/entire/checkpoints/v1` tree | `present` | medium |
| On `entire/checkpoints/v1`, a remote-tracking branch exists without it | `unpushed` | medium |
| On `entire/checkpoints/v1`, no remote-tracking branch | `unpushed` | low |
| Trailer present, checkpoint in neither store | `inconclusive` | low. It may be on the remote. |
| `Entire-Checkpoint:` line outside the last paragraph, or an invalid ID | `inconclusive` | medium |
| No trailer in the range | `missing` | high, or medium when more than 500 commits were in the range |

The push policy in `.git/entire/checkpoint-push-policy` (`automatic`, the
default, or `manual`) is named in the detail. A policy that cannot be read is
named as `unreadable`, matching Entire, which then uploads nothing.

### Copilot `Agent-Logs-Url`

Commits from the Copilot cloud agent carry an `Agent-Logs-Url` trailer that
links to the agent session log, such as
`https://github.com/OWNER/REPO/sessions/<id>`. The line is also found inside
squash-merge bodies, where it is often indented or in a bullet.

| Signal | Status | Confidence |
|---|---|---|
| A GitHub session URL | `present` | low. The log is not read, and it may be private or expired. |
| Any other value | `inconclusive` | medium |
| No trailer in the range | `missing` | high |

### Claude Code OpenTelemetry

`--otel-logs` reads a Claude Code OpenTelemetry logs export: an OTLP JSON
`resourceLogs` document, a JSON list of records, or JSON Lines of either.
Records count as Claude Code when their body starts with `claude_code.`,
their `service.name` is `claude-code`, or they have no `service.name` and their
`event.name` is a Claude Code event name. Counts, decisions,
categories and versions are reported. Prompt text and commands are never
printed.

| Signal | Status | Confidence |
|---|---|---|
| One or more `safety_event` records | `present` | medium. The count is a floor. |
| No `safety_event`, every recorded `service.version` before 2.1.307 | `inconclusive` | medium |
| No `safety_event`, otherwise | `inconclusive` | medium |
| No Claude Code events | `missing` | high, or medium when some lines could not be parsed |
| File cannot be read | `inconclusive` | low |

Some refusals log only a `tool_decision` with `decision: reject` and no safety
event. Those rejects are counted, and the ones whose `tool_use_id` has no
matching safety event are reported as `rejects_without_safety_event`.

### Sources

- Entire: [Checkpoint storage](https://docs.entire.com/guides/checkpoints/checkpoint-storage),
  [Keep checkpoints local](https://docs.entire.com/guides/checkpoints/keep-checkpoints-local),
  [Git hooks](https://docs.entire.com/guides/configuration/git-hooks), and the
  [entireio/cli changelog](https://github.com/entireio/cli/blob/main/CHANGELOG.md)
  for 0.13.1 (2026-11-06): checkpoint push policy (entireio/cli#2921) and last
  paragraph trailers (entireio/cli#2923).
- Copilot: [Trace any Copilot coding agent commit to its session logs](https://github.blog/changelog/2026-03-20-trace-any-copilot-coding-agent-commit-to-its-session-logs/)
  and [Tracking GitHub Copilot's sessions](https://docs.github.com/copilot/how-tos/use-copilot-agents/coding-agent/track-copilot-sessions).
- Claude Code: [Monitoring](https://code.claude.com/docs/en/monitoring-usage),
  sections Safety event, Tool decision event and Calls without a safety event.

## Integration boundary

The core assessment module intentionally does not read configuration files, provider matrices, session metadata, or UI state. Those surfaces should adapt their existing data into the same function so CLI, JSON/API, replay, and the local dashboard can expose one consistent result.

[`agent-strace health`](commands.md#health) is the CLI adapter. It reads session metadata, hook active-session markers, and the [capture matrix](capture-matrix.md) packaged with agent-strace, then passes them to this function.

This keeps the health rules independently testable and gives later integration work one stable source of reason codes rather than several slightly different implementations.

[`agent-strace review`](review.md) reuses this adapter as the evidence health finding in the PR reviewability check.
