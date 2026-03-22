# ADR-0010: Session Explanation via Prompt-Boundary Phase Detection

**Status:** Accepted  
**Date:** 2025-03  
**Deciders:** Siddhant Khare

## Context

A raw event stream of 50–200 events is difficult to read. Developers want a human-readable summary of what the agent did, organized by intent rather than by individual event. The summary must also identify wasted work (retries, failed phases) to help developers improve their prompts and agent configurations.

## Decision

`explain_session()` splits the event stream into phases at `USER_PROMPT` boundaries. Each phase is annotated with:

- **Files read/written** — collected from `Read`, `Write`, and `Edit` tool calls.
- **Commands run** — collected from `Bash` tool calls.
- **Retry count** — number of duplicate Bash commands within the phase (exact string match via `collections.Counter`).
- **Failed flag** — set if any `ERROR` event appears in the phase.

**Phase label derivation priority:**
1. First `USER_PROMPT` text (truncated to 50 chars)
2. Most frequent tool name in the phase (if no prompt)
3. `Phase N` fallback

**Wasted time** is the sum of durations of failed phases. **Total retries** is the sum of retry counts across all phases.

Sessions with no `USER_PROMPT` events (e.g., MCP proxy mode, or imported JSONL with only tool calls) produce a single phase containing all events.

## Consequences

- **Prompt boundaries are the natural unit of agent work** — this matches how developers think about sessions.
- **Single-phase sessions** are correct for MCP proxy mode and tool-only sessions.
- **Retry detection is exact-match only** — semantically similar commands (e.g., `pytest` vs `python -m pytest`) are not detected as retries. This avoids false positives at the cost of some false negatives.
- **Only Bash commands are checked for retries** — repeated file reads are not considered retries even if the same file is read multiple times. This reflects the intent: retrying a failing command is waste; re-reading a file for context is not.
- **`explain` is the foundation for `cost`, `diff`, and `why`** (issues #4, #5) — downstream features depend on the `Phase` and `ExplainResult` data structures.
