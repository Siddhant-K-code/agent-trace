# agent-strace OTLP to Telemetry.dev: predicted mapping (#241)

| | |
| --- | --- |
| agent-strace | 0.120.0, the payloads were generated from `src/` at commit `af39be2` |
| Date | 2026-11-22 |
| Docs read | 2026-11-22: [llms-full.txt](https://telemetry.dev/docs/llms-full.txt), [OTLP ingest](https://telemetry.dev/docs/reference/otlp), [Span attributes](https://telemetry.dev/docs/reference/span-attributes), [Cost](https://telemetry.dev/docs/reference/cost), [How it works](https://telemetry.dev/docs/how-it-works), [Privacy](https://telemetry.dev/docs/platform/privacy), Traces UI |
| Backend | `GET https://api.telemetry.dev/health` returned `200 {"ok":true}` with `x-telemetry-dev: telemetry.dev` at 2026-11-22T16:53:54Z. The endpoint does not expose a version. |
| Status | **Every prediction here is doc-based, not observed.** This file was written before anything was sent to `/v1/traces` and is kept unchanged as the record of the predictions. The real send of 2026-11-22T17:16:23Z is recorded in [RESULTS.md](RESULTS.md), with every difference from these predictions. |

## Predicted decision: PARTIAL

The payloads should be accepted. They pass every structural rule the OTLP reference lists (see [Will Telemetry.dev accept them?](#will-telemetrydev-accept-them)). Telemetry.dev should read the model, provider, tool name, tokens, errors and status from `--format otlp-genai`. It does not keep what a review needs: hierarchy across sessions, content, AgentTrace's cost, the retry link, or the evidence fields.

Top 5 risks:

1. **AgentTrace's cost is ignored.** AgentTrace sends cost in `llm.cost.total` and `agent.strace.cost.total_usd`. Telemetry.dev reads a client cost only from `gen_ai.usage.cost`, `ai.usage.cost`, `td.cost_usd` and `telemetry.dev.cost_usd`, so it computes its own cost from the tokens.
   - The child chat carries a sentinel cost of 0.01234. Telemetry.dev should show $0.0117 for it.
   - A model missing from Telemetry.dev's snapshot gets a null cost, even when AgentTrace has an estimate.
   - Classic `--format otlp` sends no cost.
2. **The CLI loses the subagent hierarchy.** `agent-strace export` writes one trace per session, so the parent and the child become two unrelated traces in both formats. One tree needs the library call that `payload-otlp-genai-tree.json` uses (`parent_span_id`). That trace then carries two session IDs.
3. **Input, Output, Arguments and Result are empty.** Telemetry.dev reads content only from span attributes with content keys.
   - AgentTrace puts `gen_ai.prompt` and `gen_ai.completion` on span events.
   - It puts tool arguments and results in `gen_ai.tool.input.*`, `gen_ai.tool.output`, `input.value` and `output.value`. None of these is a content key.
   - Chat spans get only a synthetic output built from the finish reason.
4. **Classic `--format otlp` maps poorly.**
   - The operations are `agent.session` and `tool.call`. Neither is a known operation, so no content is read.
   - Model calls exist only as events on the root span, so there are no model-call spans, tokens, cost or Models value.
   - The failed tool span has no operation, no tool name field and no `error.type`.
5. **AgentTrace's evidence is opaque or missing.**
   - Evidence health, gaps, provenance, redaction flags and the hash chain arrive as plain attributes. Gaps and provenance are JSON strings. None becomes a field or metadata.
   - `retry_of` is not exported by either format.
   - The root span is ERROR even though both retries passed.
   - With no `deployment.environment.name`, the data lands in the key's environment, or in `default`.

**What would move this to PASS:** the trace arrives, the hierarchy, tokens, errors and status match the predictions below, and the limits above are documented prominently. #241 allows a "clearly bounded partial". The AgentTrace changes that would close the gaps are listed under [Follow-ups for agent-strace](#follow-ups-for-agent-strace). They are not made here.

## Payloads

All three payloads are synthetic. `generate.py` builds them deterministically from the corpus fixture `fixture-review-001` plus a parent and child session.

| File | Produced by | Traces | Spans | Bytes as committed |
| --- | --- | --- | --- | --- |
| `payload-otlp.json` | `agent-strace export <sid> --format otlp --output …` (CLI), parent plus child | 2 | 7 | 21,390 |
| `payload-otlp-genai.json` | `agent-strace export <sid> --format otlp-genai --output …` (CLI), parent plus child | 2 | 10 | 67,914 |
| `payload-otlp-genai-tree.json` | `session_to_otlp_genai(..., parent_trace_id=..., parent_span_id=...)` (library) | 1 | 10 | 67,876 |

How the issue's checklist is covered:

| #241 item | Where it is |
| --- | --- |
| Parent and child spans | A `Task` tool call in the parent starts the child subagent session (`agent_id` synthetic-agent-1, type code-reviewer). In the tree payload, the child's root span sits under the parent's `execute_tool Task` span. |
| Provider and model | `anthropic` / `claude-sonnet-4-5` on every model call |
| Tool calls and results | `bash` (failed, then retried), `Task`, and `read_file` in the child |
| Failure and retry | A model call fails with `overloaded_error` and is retried. A `bash` test run fails and is retried. Both retries pass. |
| Tokens and estimated cost | Parent: 12,000 input (8,000 cache read, 1,000 cache write) and 1,500 output, $0.03765. Child: 2,400 input and 300 output, with a sentinel cost of 0.01234. |
| Redacted content | The prompt reads `[REDACTED synthetic request] …`. The child tool argument `auth_header` is `[REDACTED:bearer]`. The provenance `git_commit` is `[REDACTED:hex-token]`. |
| Known capture gap | `orphan_tool_result`: a tool result whose parent call `missing-tool-call` was never recorded |

## Will Telemetry.dev accept them?

`validate.py` checks the rules below offline. All three payloads pass.

| Rule (OTLP ingest reference) | Rejection | Result |
| --- | --- | --- |
| JSON field names are lowerCamelCase; no snake_case or duplicate keys | `400 invalid_payload` | Pass |
| `kind` and `status.code` are integers | `400 invalid_payload` | Pass |
| `traceId` is 32 hex characters and `spanId` is 16; neither is all zeros | `400 invalid_id` | Pass |
| Timestamps are decimal nanoseconds | `400 invalid_payload` | Pass |
| The batch has at least one span | `400 empty_payload` | Pass (7, 10 and 10 spans) |
| The body is at most 16 MiB | `413 body_too_large` | Pass (about 68 KB at most) |
| `Content-Type: application/json` | `415` | The exporter sets it |
| `Authorization: Bearer td_live_…` | `401 invalid_api_key` | Needs the owner's key |
| The resource environment matches the key's environment | `403 environment_mismatch` | No resource environment is sent, so it cannot mismatch. Telemetry.dev uses the key's environment if the key is limited to one, else `default`. |
| An identical body within 1 hour is a no-op `200` | (not stored again) | `send.py` gives every run new trace and span IDs |
| Data older than retention is hidden (Free plan: 14 days) | (hidden) | `send.py` moves the timestamps to the send time |

Also checked offline:

- Every `parentSpanId` resolves within its trace.
- No trace ID or span ID is shared between payloads.
- No local path, user name or host name appears.
- The secret-key rule (see [Redaction](#redaction)) drops no key.
- The value patterns redact no string.

The classic library function `tree_to_otlp` was also checked in memory. It gives 7 spans in 1 trace, with the child root under `tool/Task`.

## Legend

The **Prediction** column says what Telemetry.dev does with each key. All of it is doc-based, not observed.

- **field**: read into a Telemetry.dev field, such as operation, model, provider, token counts, tool, agent, session or error. Fields drive the trace list, the filters and the span panel sections. Every attribute is also kept in the stored attributes, unless content extraction moves it.
- **attr**: kept as sent in the span's stored attributes, with no field. The docs call this "attribute only".
- **shadowed**: a recognized alias that loses to a higher-priority attribute on the same span. It is kept as **attr**.
- **event**: an attribute on a span event. It is shown under **Events** in the span panel and not read into any field. The one exception is the `exception` event.
- **resource**: a resource attribute, kept with the resource.

Content extraction moves a content attribute into Input or Output. This payload set has no content keys, so nothing is moved.

## Resource and scope (all payloads)

| Key | Value | Prediction |
| --- | --- | --- |
| `service.name` | `agent-trace` | field: Service |
| `service.version` | `0.120.0` | resource |
| `agent.session_id` | session ID | resource. It is not a recognized session key. |
| (absent) `deployment.environment.name` | | Environment is the key's environment, else `default` |
| scope name | `agent-trace` | Framework `otel`: the scope matches no framework rule |

## Classic `--format otlp`

Spans: the root `agent-session (claude-code)`, one `tool/<name>` span per completed tool call, and a separate `bash` span for the tool error. Every span is a direct child of the root. The child session is a second trace, linked to the parent only by `agent.depth`.

### Root `agent-session (claude-code)` (INTERNAL)

| Key | Value | Prediction |
| --- | --- | --- |
| `gen_ai.operation.name` | `agent.session` | field: operation `agent.session`. It is not a known value, so no content is read. |
| `gen_ai.system` | `anthropic` | field: Provider |
| `gen_ai.conversation.id` | `synthetic-conversation-td241-otlp` (parent only) | field: Session. The classic child has no session. |
| `agent.name` | `claude-code` | attr. It is not `gen_ai.agent.name`, so there is no Agent. |
| `agent.command`, `agent.session_id`, `agent.tool_calls`, `agent.llm_requests`, `agent.errors`, `agent.depth` | counts and IDs | attr |
| status | ERROR on the parent | field: error status, set because the session recorded any error |

Root events. All are **event**: nothing in them becomes a model call, a token count or a cost.

| Event | Attributes |
| --- | --- |
| `user_prompt` | `prompt` |
| `gen_ai.system.message` | `gen_ai.operation.name` = `chat`, `gen_ai.request.model` |
| `gen_ai.assistant.message` | `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.usage.cache_read.input_tokens`, `gen_ai.usage.cache_write.input_tokens`, `gen_ai.response.finish_reasons`; on the failed call also `error.type` = `overloaded_error` and `error.message` |
| `decision` | `choice`, `reason` |
| `assistant_response` | `text` |

The failed model call exists only as an event, so it does not count as an error span.

### `tool/<name>` (CLIENT)

| Key | Prediction |
| --- | --- |
| `gen_ai.tool.name` | field: Tool name |
| `gen_ai.operation.name` = `tool.call` | field: operation `tool.call`. It is not `execute_tool`, so Arguments and Result are not read. |
| `gen_ai.tool.call.id` | attr |
| `tool.input.command`, `tool.input.description`, `tool.input.prompt`, `tool.input.subagent_type`, `tool.input.path`, `tool.input.auth_header` | attr |
| `tool.output` | attr |

### Tool error span `bash` (INTERNAL)

| Key | Prediction |
| --- | --- |
| `tool.name` | attr. There is no Tool field; the Tool filter falls back to the span name `bash`. |
| `error.message` | attr. The error text comes from the status message and the event below. |
| `tool.input.command` | attr |
| (no `gen_ai.*`, INTERNAL) | No operation: no derived rule matches |
| status ERROR, message `example test failed` | field: error status |
| event `exception` with `exception.message` | Error details (a message but no type), and an error log record |

## GenAI `--format otlp-genai` (and the tree payload)

Spans: the root `invoke_agent claude-code`, one `chat <model>` span per model call, and one `execute_tool <name>` span per tool call. All are direct children of the root.

### Root `invoke_agent claude-code` (INTERNAL)

| Key | Value | Prediction |
| --- | --- | --- |
| `gen_ai.operation.name` | `invoke_agent` | field: operation. It reads model content attributes, but the root carries none. |
| `gen_ai.agent.name` | `claude-code` (both parent and child) | field: Agent. The child's type `code-reviewer` appears only in the event provenance. |
| `gen_ai.provider.name` | `anthropic` | field: Provider |
| `gen_ai.system` | `anthropic` | shadowed |
| `gen_ai.conversation.id` | `synthetic-conversation-…` (parent only) | field: Session on the parent |
| `session.id` | session ID | Shadowed on the parent. It is the Session field on the child. |
| `openinference.span.kind` | `AGENT` | attr |
| `agent.session_id`, `agent.tool_calls`, `agent.llm_requests`, `agent.errors` | | attr |
| `agent.strace.provider` | `claude-code` | attr |
| `agent.strace.redacted`, `agent.strace.redacted_event_count` | `true`, `1` | attr |
| `agent.strace.evidence.schema_version`, `agent.strace.evidence.status`, `agent.strace.evidence.gap_count` | `1`, `partial` (child `healthy`), `1` | attr |
| `agent.strace.evidence.gap_codes`, `agent.strace.evidence.gaps` | JSON strings | attr, an opaque string |
| `agent.strace.provenance` | JSON string | attr, an opaque string |
| `agent.strace.parent_session_id`, `agent.strace.parent_event_id`, `agent.strace.depth` | child only | attr. It is the only link between the two CLI traces. |
| `metadata` | JSON string | attr. Metadata is read only from `td.metadata.*`, `ai.telemetry.metadata.*` and `ai.settings.context.*`, so the trace's Metadata is empty. |
| status | ERROR on the parent | field: error status |

Root events. All are **event**: content is read from span attributes, not from events.

| Event | Attributes |
| --- | --- |
| `gen_ai.user.message` | `gen_ai.prompt`, `agent.strace.event.event_id`, `agent.strace.event.previous_hash`, `agent.strace.event.redacted`, `agent.strace.event.provenance` (child), `metadata` |
| `gen_ai.assistant.message` | `gen_ai.completion` and the same `agent.strace.event.*` and `metadata` keys |

The trace list's Input and Output columns should therefore be empty for every GenAI trace.

### `chat claude-sonnet-4-5` (CLIENT)

| Key | Prediction |
| --- | --- |
| `gen_ai.operation.name` = `chat` | field: operation |
| `gen_ai.request.model` | field: Model, used for the cost lookup |
| `llm.model_name` | shadowed |
| `gen_ai.provider.name` | field: Provider |
| `gen_ai.system` | shadowed |
| `llm.provider`, `agent.strace.provider` | attr. Neither is a recognized provider alias. |
| `gen_ai.usage.input_tokens` | field: Input tokens. The value is a total that includes the cache, as the docs expect. |
| `gen_ai.usage.output_tokens` | field: Output tokens |
| `gen_ai.usage.cache_read.input_tokens` | field: Cache read |
| `gen_ai.usage.cache_write.input_tokens` | field: Cache creation (the second alias) |
| `llm.token_count.prompt`, `llm.token_count.completion` | shadowed |
| `llm.token_count.total`, `llm.token_count.prompt_details.cache_read`, `llm.token_count.prompt_details.cache_write` | attr. Not recognized. No `unrecognized_usage_attribute` diagnostic is expected, because the matching fields are already filled. |
| `gen_ai.request.max_tokens` | attr (a request parameter) |
| `llm.invocation_parameters` | attr (a JSON string) |
| `gen_ai.response.finish_reasons` | field: Finish reason (`error`, then `tool_use`). With no output attribute, it also becomes a synthetic assistant Output. |
| `agent.strace.finish_reason`, `llm.finish_reason` | attr |
| `llm.cost.total`, `agent.strace.cost.total_usd` | attr. **Not a client cost**; see [Cost](#cost). |
| `error.type` = `overloaded_error` (the failed call) | field: error status and type |
| `agent.strace.request.event_id`, `agent.strace.request.previous_hash`, `agent.strace.request.provenance`, `agent.strace.request.redacted` | attr |
| `agent.strace.response.event_id`, `agent.strace.response.parent_event_id`, `agent.strace.response.previous_hash`, `agent.strace.response.provenance`, `agent.strace.response.redacted` | attr |
| `session.id`, `gen_ai.conversation.id`, `openinference.span.kind`, `metadata` | As on the root |

### `execute_tool <name>` (INTERNAL)

| Key | Prediction |
| --- | --- |
| `gen_ai.operation.name` = `execute_tool` | field: operation. Tool content would be read from `gen_ai.tool.call.arguments` and `gen_ai.tool.call.result`, but neither is sent. |
| `gen_ai.tool.name` | field: Tool name |
| `tool.name` | attr |
| `gen_ai.tool.call.id` | attr (documented as attribute-only) |
| `gen_ai.tool.input.command`, `gen_ai.tool.input.description`, `gen_ai.tool.input.prompt`, `gen_ai.tool.input.subagent_type`, `gen_ai.tool.input.path`, `gen_ai.tool.input.auth_header` | attr. **Arguments is empty.** |
| `gen_ai.tool.output` | attr. **Result is empty.** |
| `input.value`, `input.mime_type`, `output.value`, `output.mime_type` | attr. The docs do not list these OpenInference keys. |
| `error.type` = `_OTHER` (the failed `bash`) | field: error status and type |
| status ERROR, message `example test failed` | field: error status |
| event `exception`: `exception.type`, `exception.message`, `exception.escaped`, plus `agent.strace.response.*` | Fills the error details (type `Error` and the message) and is stored as an error log record. `exception.escaped` and the `agent.strace.*` keys stay **event**. |
| `agent.strace.request.*`, `agent.strace.response.*`, `session.id`, `gen_ai.conversation.id`, `openinference.span.kind`, `metadata` | As on `chat` |

## Results by topic

### Hierarchy

| Payload | Traces | Parent and child |
| --- | --- | --- |
| `payload-otlp.json` | 2 | Lost. The child trace's root has no `parentSpanId`. Only `agent.depth` = 1 is left. |
| `payload-otlp-genai.json` | 2 | Lost as a tree. The child root keeps `agent.strace.parent_session_id` and `agent.strace.parent_event_id` as attributes, so the link can be searched for but is not drawn. |
| `payload-otlp-genai-tree.json` | 1 | Kept. The child's `invoke_agent` sits under `execute_tool Task`. The parent agent span's Metrics should include the child's model call. |

Within a session, every span is a direct child of the root. The order of model calls and tool calls shows only in the timing and in `agent.strace.response.parent_event_id`.

In the tree trace, the parent spans report the session `synthetic-conversation-td241-tree` and the child spans report `td241-tree-child`. The docs say any span can supply the trace's session but do not say which span wins. Treat this as unknown until it is observed.

### Model, provider, tool and agent

- **GenAI:** Model `claude-sonnet-4-5` and Provider `anthropic` on every chat span. Tool names `bash`, `Task` and `read_file`. Agent `claude-code` on both roots: the child shows the same agent name as its parent.
- **Classic:** Provider only, from `gen_ai.system` on the root. No Model. The Tool names come from the `tool/*` spans; the `bash` error span falls back to its span name.

### Tokens

| Trace | Input | Cache read | Cache write | Output |
| --- | --- | --- | --- | --- |
| GenAI parent | 12,000 | 8,000 | 1,000 | 1,500 |
| GenAI child | 2,400 | 0 | 0 | 300 |
| GenAI tree | 14,400 | 8,000 | 1,000 | 1,800 |
| Classic parent and child | none (tokens sit on events) | | | |

### Cost

Telemetry.dev's cost reference uses exactly the parent call as its worked example: `claude-sonnet-4-5` with 3,000 uncached, 8,000 cache-read and 1,000 cache-write input tokens and 1,500 output gives $0.03765.

| Span | AgentTrace sent | Predicted Telemetry.dev cost |
| --- | --- | --- |
| Parent `chat` (failed) | nothing, no tokens | null |
| Parent `chat` (retry) | 0.03765 | $0.03765, server-computed; it happens to equal AgentTrace's value |
| Child `chat` | **0.01234 (sentinel)** | **$0.0117** = 2,400 × $3/M + 300 × $15/M, server-computed |
| `invoke_agent` roots | nothing | The span has no cost of its own. Metrics shows the sum of the model calls under it. |
| Classic spans | nothing | null; the trace shows no cost, not `$0` |

Trace totals: GenAI parent $0.03765, child $0.0117, tree $0.04935.

If the child shows 0.01234 instead, Telemetry.dev reads a key the docs do not list.

### Errors, status and retry

| Trace | Error spans | Error logs |
| --- | --- | --- |
| GenAI parent (and tree) | 3: the root, the failed `chat` (`overloaded_error`), the failed `execute_tool bash` (`_OTHER`) | 1, from the `exception` event |
| Classic parent | 2: the root and `bash` | 1, from the `exception` event |
| Both children | 0 | 0 |

- AgentTrace marks a root ERROR whenever the session recorded an error (`otlp.py`, root status). So a session that recovered through retries shows as failed in the Status filter.
- `retry_of` is not exported. The retry shows only as a second span with the same name after the failed one. Telemetry.dev has no retry field either.

### Redaction

AgentTrace already redacted the synthetic content before export: the prompt marker, `[REDACTED:bearer]` and `[REDACTED:hex-token]`.

- **Keys:** Telemetry.dev drops a key containing one of these words: `authorization`, `cookie`, `credential(s)`, `password`, `secret` or `token`. It also drops a key with `api` followed by `key`, or a key that is exactly `apikey`. Usage keys are exempt.
  - `tool.input.auth_header` and `gen_ai.tool.input.auth_header` split into the words `auth` and `header`, so they are **kept**.
  - `validate.py` predicts that no key is dropped.
- **Values:** The Bearer, JWT, `sk-`, email, phone and Luhn-card patterns match none of the strings, so no value is redacted. AgentTrace's markers stay as sent.
- **Flags:** `agent.strace.redacted`, `agent.strace.redacted_event_count`, `agent.strace.event.redacted`, `agent.strace.request.redacted` and `agent.strace.response.redacted` stay **attr** or **event**. No field shows that content was redacted.
- **Capture off:** This is not the default. With capture off for the environment, Telemetry.dev would also do the following:
  - drop the classic `prompt` event attribute, because it is a whole-key word
  - probably drop `gen_ai.prompt` and `gen_ai.completion`, which it matches "at any depth"
  - replace every span name with `[REDACTED]`

### Evidence health, provenance and gaps

- **GenAI:** Every AgentTrace evidence key is **attr**, with no field, no metadata entry and no badge.
  - The scalars `agent.strace.evidence.status`, `agent.strace.evidence.gap_count` and `agent.strace.evidence.schema_version` keep their values.
  - `agent.strace.evidence.gap_codes`, `agent.strace.evidence.gaps`, `agent.strace.provenance` and the `*.provenance` keys are JSON strings, kept opaque.
  - The hash chain (`*.previous_hash`, `*.event_id`) is kept but not checked.
- **The capture gap:** It shows as an `execute_tool bash` span with a Result attribute but no input, and `agent.strace.response.parent_event_id` = `missing-tool-call`. Nothing marks it as a gap except the root's attributes.
- **Classic:** There are no evidence, provenance or redaction-flag keys at all. The gap shows only as a `tool/bash` span with `tool.output` and no input.

### `otlp` vs `otlp-genai`

Use `--format otlp-genai` for Telemetry.dev. Classic `otlp` gives no model calls, tokens or cost, no content, and no evidence keys. It adds nothing that GenAI lacks.

## Exporter check

**No exporter bug was found.** The OTLP JSON is valid, every `parentSpanId` matches a span, and the IDs are well-formed and unique. `src/` is unchanged.

These are fidelity gaps rather than bugs, and none is fixed here:

- `retry_of` is not exported.
- The root is ERROR even when the session recovered.
- Cost goes under non-standard keys.
- Prompts and completions are on events, not on span attributes.
- Tool arguments and results are not sent as `gen_ai.tool.call.arguments` and `gen_ai.tool.call.result`.
- The child agent's type is not used as its agent name.

The fixture's `ev-error` has no `duration_ms`, so the exporter puts its `exception` event before its span's start. Every live recorder sets `duration_ms`. `generate.py` sets 1,100 ms on its copy, and the fixture is unchanged.

## Follow-ups for agent-strace

Make these only after a real send confirms the predictions, as #241 requires. They are not part of this branch.

- Add `gen_ai.usage.cost` to `chat` spans.
- Send `gen_ai.input.messages` and `gen_ai.output.messages` as span attributes on `chat` and on the `invoke_agent` root. Send `gen_ai.tool.call.arguments` and `gen_ai.tool.call.result` on `execute_tool`.
- Export `retry_of`.
- Add a CLI option that exports a parent session and its children as one trace.
- Optionally send evidence status and gap codes as `td.metadata.*` so they become searchable metadata. That is Telemetry.dev-specific, so it should be opt-in.
