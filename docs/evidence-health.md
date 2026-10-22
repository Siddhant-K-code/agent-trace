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
`missing_session_start`. `agent-strace health` takes the declared types from
the capture matrix's `events_before_start`.

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

## Integration boundary

The core assessment module intentionally does not read configuration files, provider matrices, session metadata, or UI state. Those surfaces should adapt their existing data into the same function so CLI, JSON/API, replay, and the local dashboard can expose one consistent result.

[`agent-strace health`](commands.md#health) is the CLI adapter. It reads session metadata, hook active-session markers, and the [capture matrix](capture-matrix.md) packaged with agent-strace, then passes them to this function.

This keeps the health rules independently testable and gives later integration work one stable source of reason codes rather than several slightly different implementations.
