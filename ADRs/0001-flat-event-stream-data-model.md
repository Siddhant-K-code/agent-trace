# ADR-0001: Flat Event Stream as the Core Data Model

**Status:** Accepted  
**Date:** 2025-03  
**Deciders:** Siddhant Khare

## Context

An agent session produces many kinds of observable events: tool calls, tool results, LLM requests, file reads, errors, and conversation turns. These events have natural parent-child relationships (a `tool_result` belongs to a `tool_call`; an `llm_response` belongs to an `llm_request`). The data model must represent these relationships while remaining simple to store, stream, and consume.

## Decision

The trace is a flat, ordered list of `TraceEvent` objects. Hierarchical relationships are expressed via a `parent_id` field on child events rather than nesting. Each event is self-contained and carries its own `event_id`, `session_id`, `timestamp`, `event_type`, and `data` payload.

Twelve event types cover the full agent lifecycle:

| Category | Types |
|---|---|
| Session | `session_start`, `session_end` |
| Tool I/O | `tool_call`, `tool_result` |
| LLM I/O | `llm_request`, `llm_response` |
| File I/O | `file_read`, `file_write` |
| Conversation | `user_prompt`, `assistant_response` |
| Other | `decision`, `error` |

## Consequences

- **Storage maps directly to NDJSON** — one event per line, no nesting to serialize.
- **Streaming is natural** — consumers can process events as they arrive without buffering the full session.
- **Relationship reconstruction requires a scan** — consumers must scan for matching `parent_id` values to correlate tool calls with results. This is O(n) but acceptable for typical session sizes.
- **`EventType` inherits from `str`** so enum values serialize to their string form in JSON without a custom encoder (`EventType.TOOL_CALL == 'tool_call'`).
- **`to_json()` drops `None` and empty-string fields** using `separators=(',', ':')` to keep NDJSON lines compact.
