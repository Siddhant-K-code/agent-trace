# agent-strace OTLP to Telemetry.dev: observed results (#241)

This records the real send. [MAPPING.md](MAPPING.md) holds the doc-based predictions it is compared with, and [CHECKLIST.md](CHECKLIST.md) holds the row-by-row checks.

## Run record

| | |
| --- | --- |
| Sent | 2026-11-22T17:16:23Z, run ID `20261122T171623Z` |
| How | `python3 experiments/telemetry-dev/send.py` at commit `fb29a2d`, with no `--environment`. The rebase onto 0.121.0 left the payloads byte-identical. |
| agent-strace | 0.120.0. The exporter did not change in 0.121.0: regenerating the payloads on 0.121.0 changes only `service.version` and the scope version. |
| Telemetry.dev | It exposes no version. `GET https://api.telemetry.dev/health` returned `200 {"ok":true}` at 2026-11-22T17:20:51Z. |
| Privacy settings | Prompt capture on (the default), Issue checks on, retention 14 days (free plan). The Privacy settings say "Common secrets are redacted before storage". The custom patterns box was empty. |
| HTTP status | `payload-otlp.json` 200, `payload-otlp-genai.json` 200, `payload-otlp-genai-tree.json` 200 |
| Bodies sent (sha256) | otlp `a79fab069b693326f10b47e43a07579c3f267b536b52fcc3563fd6557e265e08`, genai `1ff9e1ba58aabdfc211567b286f577500b4191ff44e11afc280e95dfe918e0d8`, genai-tree `6d7ac5d7db5b9105872f5a9c2aabb4f33a26fdd8a852036456f2887e1cfe744e`. The bodies are the committed payloads with new IDs and times (see [README.md](README.md)). They stay in the gitignored `sent/20261122T171623Z/`, which holds no key. |
| Observed with | Telemetry.dev's read-only CLI `@telemetry-dev/cli` 0.1.8 (`npx @telemetry-dev/cli@latest`), signed in as the owner: `traces`, `trace <id> --json` and `payload`. Then the web UI, signed in as the owner, at about 17:30Z: the projects page, and the span **Attributes**, **Events** and **Metadata** panels, read only. No raw output is committed, because it holds dashboard URLs with workspace and project IDs. |
| UI cross-check | The projects page showed the test project with 27 spans, 10 LLM spans, 32.4K tokens, $0.10 LLM cost, 3 errors and p95 latency 11s. These agree with the CLI: 27 spans across the 5 traces, 32,400 tokens (13,500 + 2,700 + 16,200), $0.0987 in total, and 3 error traces. "10 LLM spans" is Telemetry.dev's own count; only 6 spans carry a model in the CLI view. |
| Not checked | The genai-tree trace's attributes. It comes from the same exporter as the genai payload. Whether attributes can be searched or filtered. |

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

This is the predicted decision. All three payloads were accepted. With `--format otlp-genai`, these map:
- the hierarchy within a trace
- provider, model, agent and tool
- tokens, errors and status
- session and environment

On every span checked, every attribute key AgentTrace sent is kept, except one that is renamed. This includes:
- evidence health, provenance and AgentTrace's redaction markers, as plain attributes
- prompts and completions, as span event attributes
- tool input and output, as span attributes

These are lost or bounded:

1. **Cost is Telemetry.dev's own.** Telemetry.dev prices the tokens itself. On the child, AgentTrace sent the sentinel 0.01234 and Telemetry.dev shows $0.0117. AgentTrace's `llm.cost.total` and `agent.strace.cost.total_usd` are kept as plain attributes.
2. **Content is not in the content panels.** Telemetry.dev's Input, Output, Arguments and Result panels and the trace list's Input column stay empty. The one exception: a `chat` span's Output shows only `[{"role":"assistant","finish_reasons":["tool_use"]}]`. The content is still visible:
   - prompts and completions in a span's **Events** list
   - tool input and output in the span **Attributes**

   [#552](https://github.com/Siddhant-K-code/agent-trace/issues/552)
3. **Telemetry.dev's redaction rewrites ordinary IDs.** Telemetry.dev redacts "common secrets" before storage, and its `sk-` match fires inside a word. In all five values checked, everything from `sk-` on was replaced, so `td241-genai-task-1` and `td241-genai-task-1-result` both became `td241-genai-ta[REDACTED]`. The stored copy then differs from AgentTrace's record, and the event-ID links through the `Task` call break. AgentTrace's local session stays the record to verify against. See [difference 7](#differences-from-the-predictions).
4. **The CLI export splits a subagent into its own trace.** Only the library call with a parent span gives one tree. [#552](https://github.com/Siddhant-K-code/agent-trace/issues/552)
5. **Retries are not linked.** Telemetry.dev groups spans with the same name as "Repeated", which is not a retry link. [#552](https://github.com/Siddhant-K-code/agent-trace/issues/552)
6. **AgentTrace's evidence is not first-class.** Evidence health, provenance, redaction flags and gaps are plain attributes, not Telemetry.dev fields, and get no badge.
   - Nothing marks the capture gap. Telemetry.dev even shows "Called by: chat claude-sonnet-4-5" on the orphan tool span.
   - Whether these attributes can be searched or filtered was not checked.
7. **A recovered session still shows as an error.** The root is ERROR even when the retry succeeded.
8. **Classic `--format otlp` is weaker.** It has no model calls, tokens or cost, and its child trace has no session. Its failed tool span `bash` has exactly three attributes, `error.message`, `tool.input.command` and `tool.name`, so Telemetry.dev shows no operation and no tool for it.
   - MAPPING.md predicted this (risk 4).
   - Here otlp.py writes no `gen_ai.operation.name`, `gen_ai.tool.name` or `gen_ai.tool.call.id` (the tool-error branch of `session_to_otlp`, about line 856).
   - It is recorded as a limit, and the fix is handed to the #552 work. `src/` is unchanged.

## Differences from the predictions

| # | Topic | Predicted (MAPPING.md) | Observed |
| --- | --- | --- | --- |
| 1 | Logs | Only the `exception` event becomes a log record, at level `error` | **Every span event** becomes a log record linked to its span. `exception` is an `error` record; every other event is an `info` record. The CLI shows the event name as its message, not its attributes. Log counts: genai parent 3, genai child 2, tree 5, otlp parent 8, otlp child 4. Each trace with a failed tool has exactly one `error` record, `example test failed`. |
| 2 | Tree session | Unknown: `synthetic-conversation-td241-tree` (parent spans) or `td241-tree-child` (child spans) | `synthetic-conversation-td241-tree`, the root's session |
| 3 | Repeated spans | The retry is only a second span with the same name | Still true, and Telemetry.dev also groups spans by name as "Repeated". genai parent: `execute_tool bash` ×3, `chat claude-sonnet-4-5` ×2. Tree: `invoke_agent claude-code` ×2, `execute_tool bash` ×3, `chat claude-sonnet-4-5` ×3. otlp parent: `tool/bash` ×2. The groups count the orphan gap span and the child agent too, so they are not retry links. |
| 4 | Trace error type | Not predicted | The genai parent and tree traces carry error type `overloaded_error`. That is the failed model call's error type, and that call started before the failed tool call (`_OTHER`). One run does not show how Telemetry.dev picks it. The `invoke_agent` root span has no error type, and the otlp parent trace has none. |
| 5 | JSON-string attributes | Kept as opaque strings | **Parsed.** Each is shown as one expandable object or array, not flattened into separate keys: `agent.strace.provenance` (12 keys), `agent.strace.evidence.gaps`, `agent.strace.evidence.gap_codes`, `metadata`, `input.value`, `llm.invocation_parameters`, `agent.strace.request.provenance` and `agent.strace.response.provenance`. |
| 6 | `gen_ai.response.finish_reasons` | Read into Finish reason and a synthetic Output, and kept as an attribute | Read into Finish reason and the Output, but **not listed** under Attributes. It appears there as `gen_ai.output.messages` = `[{"role":"assistant","finish_reasons":[…]}]`. This is the only key that did not arrive under its own name. |
| 7 | Value redaction | None: the `sk-` pattern was assumed to need a word boundary, so `validate.py` predicted no rewrite | Telemetry.dev matched `sk-` inside `task-`. The stored value became `td241-genai-ta[REDACTED]` in five places: `gen_ai.tool.call.id` on `execute_tool Task`; Task's `agent.strace.request.event_id`, `agent.strace.response.event_id` and `agent.strace.response.parent_event_id`; and the child root's `agent.strace.parent_event_id`. IDs without `sk-`, such as `td241-genai-c-tool-1`, are unchanged. **Not checked:** the same substring in the two `metadata` JSON strings and in the classic `tool/Task` call ID `td241-otlp-task-1`. `validate.py` now matches `sk-` inside a word, so it flags all of these. |
| 8 | Tool caller | The order of model and tool calls shows only in timing and `agent.strace.response.parent_event_id` | Each `execute_tool` span shows "Called by: chat claude-sonnet-4-5", including the orphan gap span, which has no tool call. |
| 9 | `service.version` | Kept on the resource | Not visible in any panel that was checked |

Everything else matches MAPPING.md:
- Hierarchy and the error spans and their types and messages
- Finish reasons, tokens and cost, down to each cost component
- Agent, provider, model and tool names
- Environment, service and framework, and an empty Metadata list
- The empty content panels, with content kept in attributes and events
- Every attribute key kept, with numbers shown as numbers
- Session from `gen_ai.conversation.id`, else `session.id`

Two rows that may look like differences are not:
- The classic `tool/*` spans have operation `tool.call` and Tool name from `gen_ai.tool.name`, as MAPPING.md's `tool/<name>` table predicted.
- The classic failed `bash` span has no operation or tool, as risk 4 predicted.

## Observed detail

### Attribute keys

On each span checked, every key sent arrived, and the count in the UI equals the payload's:

| Span | Attributes |
| --- | --- |
| genai root `invoke_agent claude-code` | 21 |
| Failed `chat` | 24 |
| Successful `chat` (the retry) | 36 |
| Failed `execute_tool bash` | 19 |
| Orphan `execute_tool bash` (the gap) | 15 |
| `execute_tool Task` | 23 |
| Child root `invoke_agent claude-code` | 23 |
| Child `execute_tool read_file` | 23 |
| Classic failed `bash` | 3 |
| Classic child root | 9 |

`gen_ai.response.finish_reasons` is counted under its new name, `gen_ai.output.messages` (difference 6). Token counts, `gen_ai.request.max_tokens`, `agent.strace.evidence.gap_count` and `agent.strace.depth` are shown as numbers.

### Hierarchy

- **genai-tree:** One tree, as predicted:
  `invoke_agent claude-code` > `execute_tool Task` > `invoke_agent claude-code` (success) > `chat claude-sonnet-4-5` and `execute_tool read_file`.
  The tree has 10 spans, and every parent resolves.
- **genai and otlp:** The parent and the child are two separate traces, as predicted. Within a trace, every span sits directly under the root.
- **The link between them:** The genai child root keeps `agent.strace.parent_session_id` and `agent.strace.depth` = 1. Its `agent.strace.parent_event_id` is rewritten (difference 7). The classic child root keeps `agent.depth` = 1.

### Session and metadata

- The trace's Session is `gen_ai.conversation.id` when the root has one (`synthetic-conversation-td241-genai`), and `session.id` is then only an attribute. The genai child has no conversation ID, so it uses `session.id` (`td241-genai-child`).
- The **Metadata** panel's Source shows Agent, Service `agent-trace`, Environment `default`, Framework `otel`, Session and User. It lists no metadata entries, as predicted: the `metadata` attribute is not read as metadata.

### Model, provider, tool and agent

- **`invoke_agent` roots:** Agent `claude-code`, Provider `anthropic`, no Model. The tree's child `invoke_agent` is the same.
- **`chat` spans:** Model `claude-sonnet-4-5`, Provider `anthropic`. Finish reason is `tool_use`, or `error` on the failed call.
- **`execute_tool` spans:** Tool `bash`, `Task` or `read_file`. Each is "Called by: chat claude-sonnet-4-5".
- **Classic:** The root has Provider `anthropic` and operation `agent.session`. `tool/*` spans have operation `tool.call` and their Tool name. The failed `bash` span has neither.

### Tokens and cost

| Span | Tokens in / out | Cost | Input | Cache read | Cache write | Output |
| --- | --- | --- | --- | --- | --- | --- |
| Parent `chat` (retry) | 12,000 / 1,500 | $0.03765 | $0.009 | $0.0024 | $0.00375 | $0.0225 |
| Child `chat` | 2,400 / 300 | $0.0117 | $0.0072 | $0 | $0 | $0.0045 |
| Failed `chat` | none | none | | | | |

- The cache components price 8,000 cache-read tokens at $0.30/M and 1,000 cache-write tokens at $3.75/M, so Telemetry.dev read both cache counts. The token hover itself was not checked.
- The trace totals are the sums of their model calls.
- AgentTrace's own cost is kept as plain attributes (`agent.strace.cost.total_usd` and `llm.cost.total` = 0.03765 on the parent's successful `chat`). The cost shown is Telemetry.dev's.
- Classic traces have no tokens and no cost. Their span events keep the token counts as attributes (for example on `gen_ai.assistant.message`), but nothing counts them.

### Errors and status

| Span | Status | Error type | Error |
| --- | --- | --- | --- |
| `invoke_agent` root, genai parent and tree | error | none | none |
| Failed `chat` | error | `overloaded_error` | `{"message":"synthetic overload, retrying","type":"overloaded_error"}` |
| Failed `execute_tool bash` | error | `_OTHER` | `{"type":"Error","message":"example test failed","status_message":"example test failed"}` |
| Classic root, otlp parent | error | none | none |
| Classic failed `bash` | error | none | `{"message":"example test failed","status_message":"example test failed"}` |

There are 3 error spans each in the genai parent and the tree, 2 in the otlp parent, and 0 in either child. The roots are errors even though both retries passed, as predicted.

### Content

- The Input, Output, Arguments and Result panels are empty, apart from the `chat` Output in limit 2.
- A span's **Events** list shows each event's attributes in full. Examples on the root:
  - `gen_ai.prompt` = "[REDACTED synthetic request] fix the failing example test"
  - `gen_ai.completion` = "Updated the example and the focused test now passes."
- Tool input and output are span attributes: `gen_ai.tool.input.*`, `gen_ai.tool.output`, `input.value` and `output.value`. Arguments stays empty even when `input.value` holds JSON.

### Redaction

- **AgentTrace's markers arrive as sent:**
  - the prompt's `[REDACTED synthetic request]`
  - the child `read_file`'s `gen_ai.tool.input.auth_header` = `[REDACTED:bearer]`
  - provenance `git_commit` = `[REDACTED:hex-token]`
- **The flags:** `agent.strace.redacted` = true is a plain attribute, with no badge.
- **Telemetry.dev's own redaction:** this changed values that were not secrets (difference 7).

### Evidence health, provenance and the capture gap

- **Parent root:** `agent.strace.evidence.status` = `partial`, `agent.strace.evidence.gap_count` = 1 and `agent.strace.evidence.gap_codes` = `["orphan_tool_result"]`. `agent.strace.evidence.gaps` has one item, with code, kind `observed`, event ID and message.
- **Child root:** status `healthy`, gap count 0, gaps `[]`.
- **`agent.strace.provenance`:** 12 keys, including `synthetic-user`, `synthetic-host` and `git_commit` `[REDACTED:hex-token]`.
- **The capture gap:** a successful 1 ms `execute_tool bash` span with empty Arguments and Result. It keeps `agent.strace.response.parent_event_id` = `missing-tool-call`, but nothing marks it as a gap. It is "Called by" a `chat` span and counted among the "Repeated" `execute_tool bash` ×3.

All of these are plain attributes. None is a Telemetry.dev field, filter or badge.

## Left for the owner

- After #552 lands, rerun `send.py` and compare content, tree export and retry links. `validate.py` now lists the values Telemetry.dev's `sk-` match will rewrite.
- Optionally check the tree trace's attributes, the `metadata` strings and attribute search, which this run did not check.
- Directory outreach waits until after #552.
