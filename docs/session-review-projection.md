# Session review projection

`agent_trace.review` provides a small deterministic layer between raw trace events and a review UI.

It is intentionally conservative: the projection classifies only facts that are present in the recorded event stream. It does not infer a missing tool call, retry, test, or provider event from surrounding prose.

## Ordering and relationships

Items are ordered by event timestamp with recorded source order as the tie breaker. A `parent_id` is marked `linked` only when it uniquely identifies an event of the expected type (a tool call for tool results, an LLM request for LLM responses, or either request type for errors). Missing parents, duplicate parent IDs, and wrong-type parents are marked `orphaned`, placed in `gaps`, and distinguished by `relationship_issue` (`missing_parent`, `ambiguous_parent`, `invalid_parent_type`). The original parent ID is preserved.

This makes an incomplete trace visibly incomplete instead of manufacturing a clean execution tree.

## Filters

The first schema version exposes reviewer-oriented categories for files, tools, commands, tests, failures, retries, recovery, decisions, privacy transformations, and gaps.

Core event types provide categories such as files, tools, failures, and decisions. Recorded hook tool calls with `data["arguments"]["command"]` (or equivalent recorded command fields) are classified as commands, and well-known test-runner commands such as `pytest` are also classified as tests. `retry_of` and explicit retry metadata identify retries. A successful result of a recorded retry can be classified as recovery using the linked call and explicit `exit_code == 0` or `success == True`. The projector does not infer intent from natural-language output or manufacture missing events.

## Raw evidence

Each projected item retains the source event ID, event type, timestamp, parent ID, redaction flag, source index, and a copy of the event's provider-specific `data`. A UI can therefore expose the underlying recorded fields rather than presenting derived labels as ground truth.

## Scope

This is a foundation for issue #242. It does not yet replace the local dashboard, calculate evidence health, render annotations, or add URL/keyboard navigation. Those surfaces can consume the projection without duplicating event-ordering and relationship rules.
