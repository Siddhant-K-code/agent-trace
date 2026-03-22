# ADR-0004: Hook Integration via Separate OS Processes

**Status:** Accepted  
**Date:** 2025-03  
**Deciders:** Siddhant Khare

## Context

Claude Code exposes a hooks system where external commands are invoked at lifecycle points: `PreToolUse`, `PostToolUse`, `UserPromptSubmit`, `Stop`, `SessionStart`, `SessionEnd`. Each hook is a separate OS process invocation. There is no persistent process or shared memory between hook calls.

## Decision

Each hook event spawns a new `agent-strace hook <event>` process. Session state is shared via two files in `.agent-traces/`:

- `.active-session` — plain text file containing the current session ID. Written by `session-start`, read by all other hooks.
- `.pending-calls.json` — JSON dict mapping `tool_name → {event_id, timestamp}`. Written by `pre-tool`, read and cleared by `post-tool` and `post-tool-failure`.

Hook errors are caught and logged to stderr, then the process exits with code 0. Claude Code may block or abort the agent loop on non-zero hook exit codes, so a tracing tool must never interfere with the agent it observes.

Redaction is controlled by the `AGENT_TRACE_REDACT=1` environment variable, baked into the hook command by `agent-strace setup --redact`. Since each hook is a separate process, a flag cannot be passed through process state.

## Consequences

- **~50ms overhead per hook** from process startup. This is acceptable for a debugging tool but would be unacceptable in production.
- **Sequential hook firing** — Claude Code fires hooks one at a time, so concurrent writes to `.pending-calls.json` are not expected. No file locking is needed.
- **`tool_name` as pending call key** — if two calls to the same tool are in flight simultaneously, only the last one is tracked. Claude Code's sequential execution makes this safe in practice, but it is a known limitation.
- **Claude Code session ID reuse** — when Claude Code provides a `session_id` in the `SessionStart` payload, agent-strace uses the first 16 characters as its own session ID, enabling direct correlation between Claude Code's internal tracking and agent-strace trace files.
- **Tool output truncated at 1000 characters** — Claude Code tool outputs can be arbitrarily large. 1000 characters captures enough context for debugging without bloating the trace.
