# #241 checklist: what was checked in Telemetry.dev

The Predicted column comes from [MAPPING.md](MAPPING.md) and is doc-based. The Observed column records the send of 2026-11-22. It was observed through Telemetry.dev's read-only CLI 0.1.8, signed in as the owner. Then, at about 17:30Z, the owner signed in to the web UI, and the projects page and the span **Attributes**, **Events** and **Metadata** panels were read, read only. [RESULTS.md](RESULTS.md) lists every difference from the predictions. #241 says a 2xx alone does not show compatibility, so every row matters.

**Not checked** marks a row that nobody looked at in this run: the genai-tree trace's attributes, which come from the same exporter as the genai payload, and attribute search and filters. These rows are not passed.

## Run record

| | |
| --- | --- |
| Date and time (UTC) | 2026-11-22T17:16:23Z (`sent_at` in `sent/20261122T171623Z/manifest.json`) |
| agent-strace | 0.120.0, from `send.py` at commit `fb29a2d`. The 0.121.0 exporter differs only in the version string. |
| Telemetry.dev | No version is exposed. `GET https://api.telemetry.dev/health` returned `200 {"ok":true}` at 2026-11-22T17:20:51Z. Plan: free (14-day retention). |
| Observed with | `@telemetry-dev/cli` 0.1.8: `traces`, `trace <id> --json`, `payload`. The web UI at about 17:30Z: the projects page, and the span **Attributes**, **Events** and **Metadata** panels. |
| Run ID | `20261122T171623Z` |
| Environment the key sends to | `default`. No `--environment` was passed. |
| Privacy settings | Prompt capture on (the default). Issue checks on. Retention 14 days. "Common secrets are redacted before storage", and no custom patterns. |
| HTTP status per payload | otlp: 200 genai: 200 genai-tree: 200 |
| Trace IDs | otlp: `a9d636957996c3ae3ffe7a43c32fd412` (parent), `7b5e7bdbcaa23b4e95fc62b9707081b5` (child). genai: `251f78ee54d0195975a3c292b7552eb7` (parent), `dda4b88cc5f053d4d07ff4c82b6dacb9` (child). genai-tree: `a54d33c58c951c037303fb2619a349c9` |
| UI cross-check | The projects page showed the test project with 27 spans, 10 LLM spans, 32.4K tokens, $0.10 LLM cost, 3 errors and p95 latency 11s. The CLI data agrees: 27 spans in 5 traces, 32,400 tokens, $0.0987 in total, 3 error traces. "10 LLM spans" is Telemetry.dev's own count; 6 spans carry a model in the CLI view. |

Do not paste the key anywhere. `sent/<run id>/` holds the exact bytes sent and their sha256, and no key, so you can compare sent and received without secrets. [RESULTS.md](RESULTS.md) records the sha256 of each body sent.

## Before you send

- Use a key with **Send telemetry** access. A key limited to one environment sends there. Use `--environment NAME` only for a key that is not limited, or the send fails with `403 environment_mismatch`.
- If **Issue checks** is on, Telemetry.dev sends stored content to its AI sub-processor. The payloads are synthetic, so this is harmless here.
- Run it once. A second run gets new IDs and makes a second copy of every trace.

## Checks by acceptance criterion

### 1. A real export reaches a Telemetry.dev project

| Check | Predicted | Observed |
| --- | --- | --- |
| **Traces** list, Date range Last 7 days: new traces | 5 (2 otlp, 2 genai, 1 tree) | 5 (2 otlp, 2 genai, 1 tree). As predicted. |
| Turn on the **Environment** column (**Columns**; it is hidden by default) | The key's environment, else `default` | `default` on all 5 (CLI) |
| The trace names | `agent-session (claude-code)` for otlp, `invoke_agent claude-code` for genai and tree | As predicted |

### 2. Versions and date recorded

The run record above is filled in. Service is `agent-trace` and Framework is `otel` on all 5 traces. The `service.version` resource attribute is not visible in any panel that was checked.

### 3. Sent and received can be compared without secrets

| Check | Predicted | Observed |
| --- | --- | --- |
| Open a GenAI `chat` span, **Attributes** | Every key in MAPPING.md's `chat` table is listed with the value from `sent/<run id>/payload-otlp-genai.json` | Every key arrived, with the count equal to the payload's: 24 on the failed `chat`, 36 on the successful one. One rename: `gen_ai.response.finish_reasons` is listed as `gen_ai.output.messages`. |
| Any key missing from **Attributes** | none (`validate.py` predicts no secret-key drops) | None on any span checked. The UI counts equal the payload's on the genai root (21), failed `execute_tool bash` (19), orphan bash (15), `Task` (23), child root (23), child `read_file` (23), classic failed `bash` (3) and classic child root (9). |
| Any value shown as redacted by Telemetry.dev | none. AgentTrace's own `[REDACTED…]` markers show as sent. | **Differs.** AgentTrace's markers show as sent, but Telemetry.dev's `sk-` match fired inside `task-`. `td241-genai-task-1` and `td241-genai-task-1-result` became `td241-genai-ta[REDACTED]` in `gen_ai.tool.call.id` on `Task`, in Task's `agent.strace.request.event_id`, `agent.strace.response.event_id` and `agent.strace.response.parent_event_id`, and in the child root's `agent.strace.parent_event_id`. IDs without `sk-` are unchanged. The `metadata` strings and the classic `tool/Task` call ID: not checked. |

### 4. Dropped, renamed or flattened attributes

Check these against MAPPING.md and note every difference.

| Check | Predicted | Observed |
| --- | --- | --- |
| `chat` span **Model call** section | Requested model `claude-sonnet-4-5`, Provider `anthropic`, Finish reason | Model `claude-sonnet-4-5`, Provider `anthropic`, Finish reason `tool_use`, or `error` on the failed call (CLI) |
| `execute_tool` span **Tool call** section | Tool name `bash`, `Task` or `read_file` | As predicted. Each also shows "Called by: chat claude-sonnet-4-5", which was not predicted. |
| Span list **Tool / Agent** column | `claude-code` on both `invoke_agent` roots | Agent `claude-code` on both roots and on the tree's child `invoke_agent` (CLI) |
| `gen_ai.tool.input.*`, `gen_ai.tool.output`, `input.value`, `output.value` | Kept in **Attributes**, not in Arguments or Result | As predicted. Kept in **Attributes**. Arguments and Result are empty, even when `input.value` holds JSON. |
| `agent.strace.provenance`, `agent.strace.evidence.gaps` | One opaque JSON string each, not expanded into separate keys | **Differs.** Not flattened into separate keys, but parsed: each is shown as one expandable object or array. The same goes for `agent.strace.evidence.gap_codes`, `metadata`, `input.value`, `llm.invocation_parameters` and the `*.provenance` keys. |
| Integer attributes such as `gen_ai.usage.input_tokens` | Shown as numbers | As predicted: token counts, `gen_ai.request.max_tokens`, `agent.strace.evidence.gap_count` and `agent.strace.depth` are numbers. |
| The trace's **Metadata** under **Source** | Empty (the `metadata` attribute is not read as metadata) | As predicted: Source shows Agent, Service, Environment, Framework, Session and User, and there are no metadata entries. |
| Classic `tool/*` spans | Operation `tool.call`, Tool name from `gen_ai.tool.name` | As predicted (CLI) |
| Classic failed tool span `bash` | No operation, no Tool name (risk 4) | As predicted: neither is set. It has exactly 3 attributes: `error.message`, `tool.input.command` and `tool.name`. Recorded as a limit. |

### 5. Hierarchy, errors, tokens, cost, redaction, provenance and gaps

**Hierarchy**

| Check | Predicted | Observed |
| --- | --- | --- |
| genai-tree trace, **Trace** tab, **Expand all** | One tree: `invoke_agent claude-code` > `execute_tool Task` > child `invoke_agent claude-code` > `chat` and `execute_tool read_file` | As predicted. 10 spans, and every parent resolves (CLI). |
| genai parent and child | Two separate traces, not linked | As predicted |
| genai child root **Attributes** | `agent.strace.parent_session_id`, `agent.strace.parent_event_id`, `agent.strace.depth` = 1 | All kept, with `agent.strace.depth` = 1 as a number. `agent.strace.parent_event_id` is rewritten to `td241-genai-ta[REDACTED]`, so it no longer matches the parent's Task call. |
| otlp child | A separate trace, linked only by `agent.depth` = 1 | As predicted. A separate trace with no session, and the root keeps `agent.depth` = 1. |
| genai-tree **Session** | Unknown: `synthetic-conversation-td241-tree` or `td241-tree-child` | **`synthetic-conversation-td241-tree`**, the root's session |

**Errors and status**

| Check | Predicted | Observed |
| --- | --- | --- |
| **Spans** column, genai parent and tree | 3 errors: the root, `chat` with `overloaded_error`, `execute_tool bash` | As predicted. Also new: the trace's error type is `overloaded_error`, the failed model call's. |
| **Spans** column, otlp parent | 2 errors: the root and `bash` | As predicted |
| Children | 0 errors | As predicted |
| Failed `execute_tool bash`, **Error** section | Error type `_OTHER` (from `error.type`). The details add exception type `Error` and the message `example test failed`. | As predicted: `_OTHER`, `{"type":"Error","message":"example test failed","status_message":"example test failed"}` |
| Failed `chat`, **Error** section | Error type `overloaded_error`, status message `synthetic overload, retrying` | As predicted: `overloaded_error`, `{"message":"synthetic overload, retrying","type":"overloaded_error"}` |
| **Logs** (level filter starts at `error`) | 1 error record for each parent and tree trace, from the `exception` event, linked to its trace | **Differs.** As predicted, there is 1 `error` record, `example test failed`, per trace with a failed tool. But every other span event is also an `info` record linked to its span, under the event name, for 3, 2, 5, 8 and 4 logs in total. |
| Status filter `error` | The parent roots are listed even though both retries passed (an AgentTrace design choice) | Trace status `error` on the genai parent, the tree and the otlp parent (CLI). The filter itself was not used. |
| Retry link | None. The retry is only a second span with the same name. | None. Telemetry.dev groups spans with the same name as "Repeated" (`execute_tool bash` ×3, `chat` ×2), counting the gap span too. |

**Tokens and cost**

| Check | Predicted | Observed |
| --- | --- | --- |
| genai parent **Tokens** | 12,000 in / 1,500 out; hover: cache read 8,000 | 12,000 / 1,500. The hover was not checked, but the cost has a cache-read part ($0.0024) and a cache-write part ($0.00375). |
| genai child **Tokens** | 2,400 in / 300 out | As predicted |
| genai-tree **Tokens** | 14,400 in / 1,800 out | As predicted |
| genai parent **Cost** | $0.03765 | $0.03765 = input $0.009 + cache read $0.0024 + cache write $0.00375 + output $0.0225 |
| **genai child Cost (the key test)** | **$0.0117**, computed by Telemetry.dev. If it shows $0.01234, Telemetry.dev reads AgentTrace's cost after all. | **$0.0117** = input $0.0072 + output $0.0045. AgentTrace's 0.01234 is not used. |
| genai-tree **Cost** | $0.04935 | As predicted |
| Failed `chat` span **Cost** | none | As predicted |
| otlp traces **Tokens** and **Cost** | none. Model calls exist only as events on the root. | As predicted |
| Model **Models** column | `claude-sonnet-4-5` for genai, empty for otlp | `chat` spans carry `claude-sonnet-4-5`. The `invoke_agent` roots and every otlp span carry none (CLI). The trace-list column was not checked. |

**Content and redaction**

| Check | Predicted | Observed |
| --- | --- | --- |
| Trace list **Input** column | Empty for every trace | As predicted |
| Trace list **Output** column | Empty, or a synthetic reply from the finish reason | A successful `chat` span's Output is only `[{"role":"assistant","finish_reasons":["tool_use"]}]`. The root's Input is empty (CLI `payload`). |
| `execute_tool` **Arguments** and **Result** | Empty | As predicted |
| Root **Events**: `gen_ai.user.message` | `gen_ai.prompt` starts with `[REDACTED synthetic request]` | As predicted: `gen_ai.prompt` = "[REDACTED synthetic request] fix the failing example test". The **Events** list shows every event's attributes, including `gen_ai.completion`. Each event is also an `info` log. |
| Child `execute_tool read_file` **Attributes** | `gen_ai.tool.input.auth_header` = `[REDACTED:bearer]`, kept | As predicted |
| A field or badge showing AgentTrace redaction | None. `agent.strace.redacted` and the `*.redacted` keys are plain attributes. | As predicted: `agent.strace.redacted` = true is a plain attribute, with no badge. |

**Provenance and evidence health**

| Check | Predicted | Observed |
| --- | --- | --- |
| Parent root **Attributes** | `agent.strace.evidence.status` = `partial`, `agent.strace.evidence.gap_count` = 1 | As predicted. `agent.strace.evidence.gap_codes` = `["orphan_tool_result"]`, and `agent.strace.evidence.gaps` has one item with code, kind `observed`, event ID and message. |
| Child root | `agent.strace.evidence.status` = `healthy` | As predicted, with gap count 0 and gaps `[]` |
| `agent.strace.provenance` | Present, an opaque JSON string, with `synthetic-user`, `synthetic-host` and `[REDACTED:hex-token]` | Present, with all three values, but parsed into an expandable object of 12 keys |
| Searchable or filterable evidence | No | No field, filter or badge. Whether the attributes can be searched: not checked. |

**The capture gap**

| Check | Predicted | Observed |
| --- | --- | --- |
| genai parent: the orphan `execute_tool bash` span | Has `gen_ai.tool.output` but no `gen_ai.tool.input.*`, and `agent.strace.response.parent_event_id` = `missing-tool-call` | As predicted. A successful 1 ms span with empty Arguments and Result and all 15 attributes. |
| Anything in Telemetry.dev marking it as a gap | No. Only the root's `agent.strace.evidence.gap_codes` names `orphan_tool_result`. | As predicted, nothing. Telemetry.dev even shows it as "Called by: chat claude-sonnet-4-5", and counts it in the "Repeated" `execute_tool bash` ×3. |

### 6. Decision

| | |
| --- | --- |
| Predicted | **PARTIAL**: accepted, with model, provider, tools, tokens, errors and status mapped; cost, content, cross-session hierarchy, retry and evidence lost or opaque |
| Observed | **PARTIAL, bounded.** It does not rest on the "Not checked" rows. |
| Rows where Observed differs from Predicted | Logs (every span event becomes a log record). The genai-tree Session (the root's). "Repeated" grouping. The trace error type. JSON-string attributes are parsed. `gen_ai.response.finish_reasons` is renamed. Telemetry.dev rewrites values containing `sk-`. Tool spans show a "Called by" model call. `service.version` is not visible. See [RESULTS.md](RESULTS.md#differences-from-the-predictions). |
| Bounded limits to document if PARTIAL | Cost is recomputed from tokens, and AgentTrace's cost is kept only as attributes. Content is in the span attributes and events, not in the Input, Output, Arguments or Result panels. Telemetry.dev's redaction rewrites ordinary IDs containing `sk-` (inside `task-`), so AgentTrace's local session stays the record. The CLI export splits subagents into separate traces. Retries are not linked. Evidence, provenance, redaction and gaps are plain attributes, not first-class. A recovered session's root is an error. Classic `otlp`'s failed tool span has no operation or tool. Content, trees and retries: [#552](https://github.com/Siddhant-K-code/agent-trace/issues/552). |

### 7. Documentation and outreach

The result is a bounded PARTIAL, so this branch adds the backend guide [docs/telemetry-dev.md](../../docs/telemetry-dev.md) with the limits up front. Directory outreach is left to the owner until after #552.
