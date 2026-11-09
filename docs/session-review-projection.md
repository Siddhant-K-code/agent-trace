# Session evidence review

`agent_trace.review` supplies the conservative event projection. `agent_trace.review_surface` composes that projection with session identity, evidence health, annotations, privacy state, and local actions for user interfaces.

The local dashboard and VS Code extension consume schema version 1. The pending CLI `review` command can use the same module after its branch is rebased. It should not create a second projection.

## Evidence order and relationships

Items are ordered by timestamp with recorded source order as the tie breaker. Each item retains its event ID, event type, timestamp, duration, source index, redaction flag, and raw provider fields.

Parent links are accepted only when they identify one compatible event. Missing parents, duplicate parent IDs, wrong parent types, and requests without a terminal result remain visible as gaps. Linked items expose parent and child event IDs. The review does not create synthetic events or repair ambiguous relationships.

## Review filters

The review surfaces expose filters for:

- files
- commands
- tools
- tests
- failures
- retries and recovery
- annotations and human corrections
- privacy transformations
- capture gaps

Test, retry, and recovery categories require recorded event fields. The projection does not infer them from prose.

Annotations remain sidecar overlays. Notes attached by event ID appear on that event. Offset annotations attach to the nearest recorded event. An annotation that cannot be attached remains unresolved.

## Header and evidence health

The review header shows session ID, provider, capture method, agent identity, start, end, duration, redaction mode, and content mode. Evidence health comes from the packaged capture matrix and the same assessment used by `agent-strace health`.

Capture gaps include observed structural reasons and declared provider limitations. A clean relationship graph is not proof that the provider exposed every action.

For Claude Code and Codex sessions, the local payload also reports the recorded-evidence result from live capture verification. The VS Code panel can run the complete `setup --verify --live` check for the selected session, including static hook setup and recorded evidence.

## Local actions

The browser dashboard and extension keep actions on the local machine:

- **Compare** runs the existing deterministic session comparison.
- **Save minimized share** writes HTML that omits session identity, absolute timestamps, raw fields, prompts, responses, commands, paths, results, and annotation text.
- **Export OTLP JSON** writes the existing OTLP/HTTP JSON representation without an endpoint. It does not send the payload.

The ordinary `agent-strace share` command still includes recorded content. Use the review surface's minimized action when content must be omitted.

## Navigation

The browser accepts `?event=<event-id>` and updates the URL as the reviewer moves. Both surfaces support J/K and arrow navigation, Enter to open raw fields, `/` to focus search, and Escape to clear search.

## Representative fixture

`tests/fixtures/representative_session/` covers a redacted prompt, file access, a failed test, retry, recovery, correction annotation, and an orphaned tool result. Python tests validate the shared payload and browser API. Extension tests validate the matching TypeScript projection and minimized share.
