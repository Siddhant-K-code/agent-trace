# agent-strace OTLP to Telemetry.dev: observed results (#241)

This records the real send. [MAPPING.md](MAPPING.md) holds the doc-based predictions it is compared with, and [CHECKLIST.md](CHECKLIST.md) holds the row-by-row checks.

## Run record

| | |
| --- | --- |
| Sent | 2026-11-22T17:16:23Z, run ID `20261122T171623Z` |
| How | `python3 experiments/telemetry-dev/send.py` at commit `fb29a2d`, with no `--environment`. The rebase onto 0.121.0 left `experiments/telemetry-dev/` byte-identical. |
| agent-strace | 0.120.0. The exporter did not change in 0.121.0: regenerating the payloads on 0.121.0 changes only `service.version` and the scope version. |
| Telemetry.dev | It exposes no version. `GET https://api.telemetry.dev/health` returned `200 {"ok":true}` at 2026-11-22T17:20:51Z. |
| HTTP status | `payload-otlp.json` 200, `payload-otlp-genai.json` 200, `payload-otlp-genai-tree.json` 200 |
| Bodies sent (sha256) | otlp `a79fab069b693326f10b47e43a07579c3f267b536b52fcc3563fd6557e265e08`, genai `1ff9e1ba58aabdfc211567b286f577500b4191ff44e11afc280e95dfe918e0d8`, genai-tree `6d7ac5d7db5b9105872f5a9c2aabb4f33a26fdd8a852036456f2887e1cfe744e`. The bodies are the committed payloads with new IDs and times (see [README.md](README.md)). They stay in the gitignored `sent/20261122T171623Z/`, which holds no key. |
| Observed with | Telemetry.dev's read-only CLI `@telemetry-dev/cli` 0.1.8 (`npx @telemetry-dev/cli@latest`), signed in as the owner: `traces`, `trace <id> --json` and `payload`. The raw output is not committed, because it holds dashboard URLs with workspace and project IDs. |
| UI cross-check | The owner's Telemetry.dev projects page showed the test project with 27 spans, 10 LLM spans, 32.4K tokens, $0.10 LLM cost, 3 errors and p95 latency 11s. These agree with the CLI: 27 spans across the 5 traces, 32,400 tokens (13,500 + 2,700 + 16,200), $0.0987 in total, and 3 error traces. "10 LLM spans" is Telemetry.dev's own count; only 6 spans carry a model in the CLI view. |
| Not observable | Neither the CLI 0.1.8 nor the Telemetry.dev MCP server exposes raw span attributes, and no one opened a span's **Attributes** tab in the web UI for this run. Rows that need it are marked "Not observed (UI-only)". They are not passed, and neither the decision nor the backend guide relies on them. |

Traces received:

| Payload | Trace | Name | Status | Session | Spans | Tokens in / out | Cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| otlp, parent | `a9d636957996c3ae3ffe7a43c32fd412` | `agent-session (claude-code)` | error | `synthetic-conversation-td241-otlp` | 5 | none | none |
| otlp, child | `7b5e7bdbcaa23b4e95fc62b9707081b5` | `agent-session (claude-code)` | success | none | 2 | none | none |
| genai, parent | `251f78ee54d0195975a3c292b7552eb7` | `invoke_agent claude-code` | error, type `overloaded_error` | `synthetic-conversation-td241-genai` | 7 | 12,000 / 1,500 | $0.03765 |
| genai, child | `dda4b88cc5f053d4d07ff4c82b6dacb9` | `invoke_agent claude-code` | success | `td241-genai-child` | 3 | 2,400 / 300 | $0.0117 |
| genai-tree | `a54d33c58c951c037303fb2619a349c9` | `invoke_agent claude-code` | error, type `overloaded_error` | `synthetic-conversation-td241-tree` | 10 | 14,400 / 1,800 | $0.04935 |

Every trace has Environment `default`, Service `agent-trace` and Framework `otel`, and the CLI reports each as complete.

## Decision: PARTIAL, bounded

This is the predicted decision. All three payloads were accepted, and `--format otlp-genai` maps:
- the hierarchy within a trace
- provider, model, agent and tool
- tokens, errors and status
- session and environment

These are lost or bounded:

1. **Cost is Telemetry.dev's own.** Telemetry.dev prices the tokens itself and ignores AgentTrace's `llm.cost.total` and `agent.strace.cost.total_usd`. On the child, AgentTrace sent the sentinel 0.01234 and Telemetry.dev shows $0.0117.
2. **No content.** Input is empty on the root, on `chat` and on the tool spans, and the trace list's Input column is empty for every trace. A `chat` span's Output is only `[{"role":"assistant","finish_reasons":["tool_use"]}]`. Tool Arguments and Result are empty. [#552](https://github.com/Siddhant-K-code/agent-trace/issues/552)
3. **The CLI export splits a subagent into its own trace.** Only the library call with a parent span gives one tree. [#552](https://github.com/Siddhant-K-code/agent-trace/issues/552)
4. **Retries are not linked.** Telemetry.dev groups spans with the same name as "Repeated", which is not a retry link. [#552](https://github.com/Siddhant-K-code/agent-trace/issues/552)
5. **AgentTrace's evidence is not first-class.** Evidence health, provenance, redaction flags and gaps are not Telemetry.dev fields. Whether they survive as attributes is not observed.
6. **Classic `--format otlp` is weaker.** It has no model calls, tokens or cost, and its child trace has no session. Its failed tool span `bash` has no operation and no tool name: Telemetry.dev shows neither for it. Here otlp.py writes only `tool.name`, `tool.input.*` and `error.message` (the tool-error branch of `session_to_otlp`, about line 856), with no `gen_ai.operation.name`, `gen_ai.tool.name` or `gen_ai.tool.call.id`. MAPPING.md predicted this (risk 4). It is recorded here as a limit, and the fix is handed to the #552 work. `src/` is unchanged.

## Differences from the predictions

| # | Topic | Predicted (MAPPING.md) | Observed |
| --- | --- | --- | --- |
| 1 | Logs | Only the `exception` event becomes a log record, at level `error` | **Every span event** becomes a log record linked to its span. `exception` is an `error` record. Every other event is an `info` record, and the CLI shows the event name as its message, not its attributes. Log counts: genai parent 3, genai child 2, tree 5, otlp parent 8, otlp child 4. Each trace with a failed tool has exactly one `error` record, `example test failed`. |
| 2 | Tree session | Unknown: `synthetic-conversation-td241-tree` (parent spans) or `td241-tree-child` (child spans) | `synthetic-conversation-td241-tree`, the root's session |
| 3 | Repeated spans | The retry is only a second span with the same name | Still true, and Telemetry.dev also groups spans by name as "Repeated". genai parent: `execute_tool bash` ×3, `chat claude-sonnet-4-5` ×2. Tree: `invoke_agent claude-code` ×2, `execute_tool bash` ×3, `chat claude-sonnet-4-5` ×3. otlp parent: `tool/bash` ×2. The groups count the orphan gap span and the child agent too, so they are not retry links. |
| 4 | Trace error type | Not predicted | The genai parent and tree traces carry error type `overloaded_error`. That is the error type of the failed model call, which started before the failed tool call (`_OTHER`). One run does not show how Telemetry.dev picks it. The `invoke_agent` root span itself has no error type. The otlp parent trace has none. |

Everything else that the CLI shows matches MAPPING.md:
- Hierarchy and the error spans and their types and messages
- Finish reasons, tokens and cost, down to each cost component
- Agent, provider, model and tool names
- Environment, service and framework
- The empty content fields

Two rows that may look like differences are not:
- The classic `tool/*` spans have operation `tool.call` and Tool name from `gen_ai.tool.name`, as MAPPING.md's `tool/<name>` table predicted.
- The classic failed `bash` span has no operation or tool, as risk 4 predicted.

## Observed detail

### Hierarchy

- **genai-tree:** One tree, as predicted:
  `invoke_agent claude-code` > `execute_tool Task` > `invoke_agent claude-code` (success) > `chat claude-sonnet-4-5` and `execute_tool read_file`.
  The tree has 10 spans, and every parent resolves.
- **genai and otlp:** The parent and the child are two separate traces, as predicted. Within a trace, every span sits directly under the root.
- **Not observed (UI-only):** the child roots' `agent.strace.parent_session_id`, `agent.strace.parent_event_id`, `agent.strace.depth` and `agent.depth`.

### Model, provider, tool and agent

- **`invoke_agent` roots:** Agent `claude-code`, Provider `anthropic`, no Model. The tree's child `invoke_agent` is the same.
- **`chat` spans:** Model `claude-sonnet-4-5`, Provider `anthropic`. Finish reason is `tool_use`, or `error` on the failed call.
- **`execute_tool` spans:** Tool `bash`, `Task` or `read_file`.
- **Classic:** The root has Provider `anthropic` and operation `agent.session`. `tool/*` spans have operation `tool.call` and their Tool name. The failed `bash` span has neither an operation nor a tool name.

### Tokens and cost

| Span | Tokens in / out | Cost | Input | Cache read | Cache write | Output |
| --- | --- | --- | --- | --- | --- | --- |
| Parent `chat` (retry) | 12,000 / 1,500 | $0.03765 | $0.009 | $0.0024 | $0.00375 | $0.0225 |
| Child `chat` | 2,400 / 300 | $0.0117 | $0.0072 | $0 | $0 | $0.0045 |
| Failed `chat` | none | none | | | | |

The cache components price 8,000 cache-read tokens at $0.30/M and 1,000 cache-write tokens at $3.75/M, so Telemetry.dev read both cache counts. The token hover itself was not checked. The trace totals are the sums of their model calls. Classic traces have no tokens and no cost.

### Errors and status

| Span | Status | Error type | Error |
| --- | --- | --- | --- |
| `invoke_agent` root, genai parent and tree | error | none | none |
| Failed `chat` | error | `overloaded_error` | `{"message":"synthetic overload, retrying","type":"overloaded_error"}` |
| Failed `execute_tool bash` | error | `_OTHER` | `{"type":"Error","message":"example test failed","status_message":"example test failed"}` |
| Classic root, otlp parent | error | none | none |
| Classic failed `bash` | error | none | `{"message":"example test failed","status_message":"example test failed"}` |

There are 3 error spans each in the genai parent and the tree, 2 in the otlp parent, and 0 in either child. The roots are errors even though both retries passed, as predicted.

### Content and redaction

The content and redaction results are those in limit 2. Telemetry.dev redacted nothing that the CLI shows: both error messages arrive as sent.

**Not observed (UI-only):**
- the event bodies, including `gen_ai.prompt` with `[REDACTED synthetic request]`
- `gen_ai.tool.input.auth_header` = `[REDACTED:bearer]`
- any key or value Telemetry.dev dropped or redacted
- the `*.redacted` flags

### Evidence health, provenance and the capture gap

The capture gap arrives as a successful 1 ms `execute_tool bash` span with empty Arguments and Result. Nothing in the CLI output marks it as a gap, and it is counted among the "Repeated" `execute_tool bash` ×3.

**Not observed (UI-only):**
- every `agent.strace.evidence.*` key, including `status`, `gap_count`, `gap_codes` and `gaps`
- `agent.strace.provenance` and the hash chain
- `agent.strace.response.parent_event_id` = `missing-tool-call` on the gap span
- `service.version`
- the `metadata` attribute, and whether the trace's **Metadata** stays empty

The CLI's trace and span records have no evidence, provenance or redaction field.

## Left for the owner

- In the UI, open the genai-tree trace and fill in the "Not observed" rows of [CHECKLIST.md](CHECKLIST.md) from a span's **Attributes** tab. This shows whether the attribute-level rows hold: key drops, redaction, opaque JSON strings, integer types and evidence keys.
- After #552 lands, rerun `send.py` and compare content, tree export and retry links.
- Directory outreach waits until after #552.
