# Import GitHub Agentic Workflows runs

[GitHub Agentic Workflows](https://github.github.com/gh-aw/) (gh-aw) writes each
run as one `aw_session.jsonl` file in the run's `usage` artifact. The file
follows the
[Unified Agent Session specification](https://github.github.com/gh-aw/specs/unified-agent-session-specification/)
and covers the Claude, Copilot, Codex, Gemini, Pi and custom engines.
`agent-strace import` reads file-format version 1 (specification draft 2.12.0)
and stores the run as an agent-strace session, so `replay`, `explain`, `cost`,
`health`, evidence packets and `export --format otlp|otlp-genai` work on it.

## Import a run

```bash
gh run download <run-id> -n usage -D usage
agent-strace import usage/aw_session.jsonl
agent-strace cost claude-run-0001
agent-strace export claude-run-0001 --format otlp-genai --output run.json
```

`gh run download` needs read access to the repository's Actions artifacts and
fails once the artifact has expired. agent-strace does not download artifacts
itself.

```
Imported session claude-run-0001
  4 tool calls, 2 LLM requests, 1820 tokens
  14 events
  1 event has no recorded time (timing: inferred)
  Skipped 3 records not imported: engine.heartbeat 1, firewall.token_usage (repeated requestId) 1, otel.span 1
  Replay with: agent-strace replay claude-run-0001
```

The session ID is the main agent's `session.init` or `session.start`
`sessionId` when it is usable (see [`import`](commands.md#import)), else `aw-`
and the first 16 hex digits of the file's SHA-256. Importing the same file
again writes nothing and prints `Session <id> is already imported; nothing
written.`

## The header

The first record must be `session.format` with a whole-number `data.version`.
A file with that header is never read as a Claude Code transcript. These print
one line, exit 1 and write nothing:

- `Cannot import <path>: no session.format header; an aw_session.jsonl file
  starts with one`: the first record is shaped like a unified session event
  (a dotted `type` and a `provenance.component`) but is not the header.
- `Cannot import <path>: invalid session.format header: data.version must be a
  whole number of at least 1`
- `Cannot import <path>: unsupported aw_session format version 2; this
  agent-strace reads version 1`

## What is kept

| Record | Stored as |
|---|---|
| `session.init`, `session.start`, `workflow.info`, `session.collection` | Session metadata: `session_start` holds `mode: aw-session-import`, `format_version`, `engine`, `model`, `cwd`, `workflow`, `repository`, `trigger_type`, `gh_aw_version`, `awf_version`, `agent_version` and `run_id` when recorded |
| `user.message` | `user_prompt` |
| `assistant.message`, `assistant.refusal` | `assistant_response` text (a refusal keeps `stop_reason`, `policy_category`, `explanation`; a partial refusal is skipped) |
| `tool.execution_start` | `tool_call` with `tool_name`, `arguments`, `tool_use_id` (the `toolCallId`) and `mcp_server_name` |
| `tool.execution_complete` | `tool_result`, or `error` when the call failed, linked to its call |
| `firewall.token_usage`, `usage.report` | `assistant_response` with the recorded token usage |
| `session.error` | `error` (warnings and lower severities are skipped) |
| `subagent.started`, `subagent.completed`, `subagent.failed` | `subagent_start`, `subagent_stop` |
| `session.result` | `session_end` fields, as a snapshot |
| `agent.execution` | `session_end` `exit_code`, `error_categories`, `error_codes`, `error_types` |

Every stored event keeps its record `type` as `source_type`, its source
`provenance` (`component`, `phase`, `path`, `index`, `timestampMs`,
`attemptId`, `startupAttemptId` and a short `native`), and a `context_stream`:
`main` for the agent phase, the phase name (such as `detection`) for other
phases, or `sidechain:<agentId>` for a subagent's events. Text is redacted
before it is stored and cut at 2,000 characters, as in other imports. An
object kept as text, such as structured tool output, is redacted by key
(`password`, `token`, `authorization` and the like) before it becomes text.

### Tool calls

A completion pairs with the start that has the same `toolCallId` in the same
phase, artifact path and subagent. A paired result has `parent_id` set to its
call and `duration_ms` from the recorded `durationMs`, else from the two
timestamps when both are recorded. A completion with no matching start is
kept unpaired; a start with no completion stays a call with no result. Both
show as unpaired tool [evidence gaps](evidence-health.md#observed-checks).

`success: false`, `isError: true`, a non-zero `exitCode` or a failed `status`
(such as `error`, `cancelled` or `timeout`) makes the completion an `error`
event, which `explain`, `health` and the exporters treat as a failed tool.
`success: true` is stored as `success: true`. When the completion records no
outcome, the `tool_result` has no `success` field: the outcome is unknown, not
assumed.

### Tokens and cost

Each `firewall.token_usage` record is one model request. Its usage is stored
on an `assistant_response` with `model`, `provider` and the `requestId` as
`message_id`; a record that repeats a `requestId` is skipped. A `usage.report`
without a `requestId` is skipped when the file has firewall records, because
it would count the same request twice. `cost` prices these recorded tokens,
and `export --format otlp|otlp-genai` turns each into a `chat {model}` span
with its token counts and `gen_ai.response.id`. The span's start is inferred
(`agent.strace.timing` `inferred`), as for
[imported transcripts](production.md#model-responses-imported-from-a-transcript).

Cache reads and writes stay separate. Whether `inputTokens` already includes
them follows the record's `inputTokensIncludeCache`. When the record does not
say, gh-aw's rule applies: Anthropic, OpenAI, Azure OpenAI and GitHub Copilot
counts include cache reads but not writes, and other providers include
neither. No token is counted twice.

A request that failed (HTTP status 400 or more) is kept with `status: failed`
and `http_status`. Without usage it gets no `chat` span. Usage records with
`purpose: subagent` but no `agentName` stay on the main stream; agent-strace
does not guess which subagent made them.

`session.result` is a snapshot, not a sum. The main agent's last result
supplies `status`, `num_turns`, `reported_cost_usd` (the engine's
`totalCostUsd`), `duration_ms` and `result_usage`; each result's errors are
kept in order, up to 20. These do not add to `cost`, which counts only the
per-request usage. A run with no request usage is priced from event sizes,
like any session without recorded usage.

### Times

An event's time is its ISO 8601 `timestamp` with a time zone, a numeric
timestamp in the unit
`session.collection` gives for its source, or `provenance.timestampMs`. An
event with none of these is stored at its source's latest earlier time (or the
file's, when the source has none yet) with `timing: inferred`, and the import
counts it. When nothing in the file has a time, events are stored at `0.0`
with `timing: untimed`. File order is kept, as the specification requires, so
an event recorded earlier than the one before it shows as a
`timestamp_regression` [evidence gap](evidence-health.md#observed-checks). The session's start
and end come from recorded times only.

### Subagents

`subagent.started` and `subagent.completed`/`subagent.failed` become
`subagent_start`/`subagent_stop` in the parent session. Events with an
`agentId` are marked as that subagent's sidechain (`is_sidechain`,
`agent_id`, `agent_type`), as the Claude Code import marks sidechains.
`tree` lists the subagents, and the exporters keep these fields in each span's
provenance.
`subagent.request` records are skipped: they repeat requests the firewall
already counts.

## What is dropped

Records with no stored form are counted in the import output and in
`session_end` `skipped_records`; no placeholder events are written. They
include `assistant.reasoning`, `engine.heartbeat`, `firewall.model_routing`,
`otel.span` (exporting them would duplicate agent-strace's own spans),
MCP gateway, guardrail and safe-output records, `session.result` from a
subagent or another phase, and a `session.format` record after the header.
Lines that are not JSON objects are counted as `(unreadable line)`.

Not supported: writing `aw_session.jsonl`, the compact profile (specification
section 3.8), and modelling MCP gateway, firewall or safe-output payloads
beyond their provenance.
