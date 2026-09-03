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

- session start/end boundaries;
- events before start or after end;
- duplicate event IDs;
- mixed session IDs;
- non-finite timestamps and timestamp regressions;
- unpaired tool calls/results;
- unpaired LLM requests/responses; and
- export/drop failures explicitly reported by the caller.

The calculation is deterministic and local. It performs no network calls.

## Provider limitations

A missing event alone is not enough to conclude that capture failed: some providers never expose certain categories of activity. Callers should therefore pass known blind spots from a versioned provider/capture matrix:

```python
from agent_trace.evidence_health import assess_evidence_health

health = assess_evidence_health(
    events,
    provider="codex",
    capture_method="hooks",
    provider_blind_spots=[
        "example limitation supplied by the capture matrix",
    ],
    session_finalized=True,
)
```

A declared blind spot makes the result `partial` even when the observed event stream is structurally clean. This prevents `healthy` from becoming a false claim about activity a provider cannot expose.

## Integration boundary

The core assessment module intentionally does not read configuration files, provider matrices, session metadata, or UI state. Those surfaces should adapt their existing data into the same function so CLI, JSON/API, replay, and the local dashboard can expose one consistent result.

This keeps the health rules independently testable and gives later integration work one stable source of reason codes rather than several slightly different implementations.
