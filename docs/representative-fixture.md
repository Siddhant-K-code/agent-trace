# Representative evidence fixture

`tests/fixtures/representative_session/` is a synthetic, versioned session for review-oriented regression tests. It contains no customer data, real secrets, or private repository content.

The scenario intentionally includes a session boundary, redacted prompt, file read/write, a failing test command, an error, a retry, successful recovery, a human correction annotation, and one orphaned tool result that models a known capture gap.

## Refresh procedure

1. Keep the session ID and event IDs deterministic unless a schema change requires replacing them.
2. Edit `events.ndjson` using fields accepted by `TraceEvent.from_json()` and `annotations.jsonl` using fields accepted by `Annotation.from_json()`.
3. Update `expected.json` whenever event count, failure/retry/recovery IDs, annotation IDs, or the intentional gap changes.
4. Run `python -m unittest tests.test_representative_fixture -v`.
5. Consumers may add assertions against this fixture, but generated expectations should be changed together with their source event change rather than hand-edited independently.

## Trust boundary

The fixture is evidence of AgentTrace's data model and review semantics, not evidence that any real provider exposes every event shown here. Provider-specific coverage belongs in the capture matrix, and the orphan event exists specifically so UIs and health checks must preserve uncertainty instead of inventing a missing parent.
