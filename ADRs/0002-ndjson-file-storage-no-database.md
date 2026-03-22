# ADR-0002: NDJSON File Storage — No Database

**Status:** Accepted  
**Date:** 2025-03  
**Deciders:** Siddhant Khare

## Context

Trace data must be persisted across the lifetime of an agent session and survive process restarts. The storage mechanism must be readable by any tool without requiring agent-strace to be installed, and must not introduce runtime dependencies.

## Decision

Events are stored as newline-delimited JSON (NDJSON) in append-only files. Each session gets its own directory:

```
.agent-traces/
  <session-id>/
    meta.json        # session metadata (counters, timestamps, agent name)
    events.ndjson    # one TraceEvent per line, append-only
  .active-session    # plain-text file containing the current session ID
  .pending-calls.json  # in-flight tool call state for cross-process latency tracking
```

The default trace directory is `.agent-traces/` relative to the working directory, making traces project-local.

## Consequences

- **Zero dependencies** — NDJSON is readable with `cat`, `grep`, `jq`, or any JSON parser in any language.
- **Append-safe** — file appends are atomic at the OS level for writes smaller than the filesystem block size (~4KB). A single NDJSON line is always well under this limit.
- **No locking required** — Claude Code fires hooks sequentially, so concurrent writes from the same session are not expected.
- **No indexing** — session listing is O(n) directory iteration. Prefix matching for session IDs is O(n) over sessions. Acceptable for tens to hundreds of sessions.
- **Session listing is sorted by directory name** (reverse-alphabetical), not by `started_at` timestamp. UUID hex strings are not time-ordered, so the "latest" session is determined by sort order, not creation time. This is a known simplification.
- **No compaction or rotation** — traces grow indefinitely. Cleanup is manual.
