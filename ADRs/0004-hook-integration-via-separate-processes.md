# ADR-0004: Hook Integration via Separate OS Processes

**Status:** Accepted  
**Date:** 2025-03  
**Deciders:** Siddhant Khare

## Context

Claude Code exposes a hooks system where external commands are invoked at lifecycle points: `PreToolUse`, `PostToolUse`, `UserPromptSubmit`, `Stop`, `SessionStart`, `SessionEnd`. Each hook is a separate OS process invocation. There is no persistent process or shared memory between hook calls.

## Decision

Each hook event spawns a new `agent-strace hook <event>` process. Session state is shared via two files in `.agent-traces/`:

- `.active-session` — plain text file containing the current session ID. Written by `session-start`, read by all other hooks.
- `.pending-calls.json` — JSON dict of in-flight tool calls keyed by the `tool_call` event ID (`{tool_name, tool_use_id, session_id, …}`). Written by `pre-tool`, read and cleared by `post-tool` and `post-tool-failure`. When the agent supplies a session ID, the file is scoped per provider and session (`.pending-calls.v2.<provider>.<session>.<hash>.json`). The file exists only while calls are in flight: it is deleted when the map becomes empty, and `session-end` removes that session's state.

Hook errors are caught and logged to stderr, then the process exits with code 0. Claude Code may block or abort the agent loop on non-zero hook exit codes, so a tracing tool must never interfere with the agent it observes.

Redaction is controlled by the `AGENT_TRACE_REDACT=1` environment variable, baked into the hook command by `agent-strace setup --redact`. Since each hook is a separate process, a flag cannot be passed through process state.

## Consequences

- **~50ms overhead per hook** from process startup. This is acceptable for a debugging tool but would be unacceptable in production.
- **Concurrent hook firing** — agents can run tool hooks in parallel, and several agents or sessions can share one store. Each read-modify-write of a pending-calls file holds the store's `.store.lock`, so parallel hooks cannot lose each other's entries or race an unlink. The lock is not re-entrant, so code holding it never constructs a store or appends events or metadata.
- **Unmatched calls at session end are dropped from the pending state** — the `tool_call` event is already in `events.ndjson`, which is what unpaired-call reports read. A result that arrives after `session-end` is recorded without a link to its call.
- **Claude Code session ID reuse** — when Claude Code provides a `session_id` in the `SessionStart` payload, agent-strace uses the first 16 characters as its own session ID, enabling direct correlation between Claude Code's internal tracking and agent-strace trace files.
- **Tool output truncated at 1000 characters** — Claude Code tool outputs can be arbitrarily large. 1000 characters captures enough context for debugging without bloating the trace.
