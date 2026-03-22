# ADR-0009: Claude Code JSONL Session Import

**Status:** Accepted  
**Date:** 2025-03  
**Deciders:** Siddhant Khare

## Context

Claude Code stores session logs as JSONL files in `~/.claude/projects/<encoded-project-path>/<session-id>.jsonl`. Users who did not set up agent-strace hooks before a session still want to analyze those sessions. Importing existing JSONL files enables replay, export, and stats without re-running the session.

## Decision

`import_jsonl()` reads a Claude Code JSONL file and converts it to agent-strace's `TraceEvent`/`SessionMeta` format. The import is a single-pass read into memory followed by a conversion loop:

**Entry type mapping:**

| Claude Code entry type | agent-strace events produced |
|---|---|
| `user` with text content | `USER_PROMPT` |
| `user` with `tool_result` content blocks | `TOOL_RESULT` |
| `user` with `toolUseResult` field | `TOOL_RESULT` (only if no content-block results) |
| `assistant` with `tool_use` content blocks | `TOOL_CALL` (one per block) |
| `assistant` with text content | `ASSISTANT_RESPONSE` |
| `system` with `subtype: turn_duration` | contributes to `total_duration_ms` |
| `queue-operation` | skipped |

**Session ID:** The full Claude Code session UUID is preserved as the agent-strace session ID (not truncated), enabling cross-referencing with Claude Code's own logs.

**Project path decoding:** Claude Code encodes project directory names by replacing `/` with `-` and prepending `-`. The decoder uses a simple `str.replace('-', '/')`. This is ambiguous for project names containing hyphens (e.g. `/home/user/my-project` → `-home-user-my-project` → `/home/user/my/project`), but is correct for display purposes in `--discover`.

**Discovery:** `agent-strace import --discover` lists available sessions. Claude Code names session files `<session-uuid>.jsonl`, so the session ID is the filename stem — no file open is needed.

## Consequences

- **Retroactive analysis** — existing Claude Code sessions can be analyzed without re-running.
- **No hooks required** — the import path is independent of the hook integration.
- **`ASSISTANT_RESPONSE` is emitted even when tool calls are present** in the same message — Claude Code commonly emits reasoning text alongside a `tool_use` block.
- **`toolUseResult` is only used if no content-block `tool_result` entries exist** — prevents duplicate `TOOL_RESULT` events for the same tool call.
- **Path decoding is display-only** — the decoded project path is shown in `--discover` output but is not used for any file operations.
- **Large sessions are loaded fully into memory** — a 315-line JSONL file with 38 tool calls is typical. Very large sessions (thousands of entries) may use significant memory.
