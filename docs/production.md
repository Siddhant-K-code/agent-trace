# Production tracing (OTLP export)

## Quick demo (no signup required)

Spin up Jaeger locally with one Docker command and a browser trace UI:

```bash
# Start Jaeger (OTLP on 4318, UI on 16686)
docker run --rm -d --name jaeger \
  -p 16686:16686 -p 4318:4318 \
  jaegertracing/all-in-one:latest

# Export your session
agent-strace export --format otlp-genai --endpoint http://localhost:4318

# Open the UI
open http://localhost:16686
```

Select service `agent-trace` in the Jaeger UI to see the full trace.

Export sessions as OpenTelemetry spans to your existing observability stack. Sessions become traces. Tool calls become spans with duration and inputs. Errors get exception events. No new dependencies.

---

## OTel GenAI semantic conventions

Use `--format otlp-genai` to export with the [OTel GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai). This produces AI-native spans that populate token usage charts, cost views, and LLM dashboards automatically.

```bash
agent-strace export <session-id> --format otlp-genai \
  --endpoint http://localhost:4318
```

| Aspect | `--format otlp` | `--format otlp-genai` |
|---|---|---|
| LLM calls | Events on root span; a model response imported from a Claude Code transcript is also a `chat {model}` span | `chat {model}` child spans |
| Tool calls | `tool/<name>` spans | `execute_tool {name}` spans |
| Root span | `agent.name` attribute | `invoke_agent {agent}` span with `gen_ai.agent.name` |
| Errors | `tool/<name>` span with ERROR status and an `exception` event; see [Classic tool spans](#classic-tool-spans) | OTel `exception` event format and `error.type` |
| Failed or cancelled model request | `error.type` on the `gen_ai.assistant.message` event | ERROR status and `error.type` on the `chat` span |
| `gen_ai.conversation.id` | Root span | Root, LLM and tool spans |
| Token usage | `gen_ai.usage.*` on the LLM events, and on the `chat` spans of imported responses | `gen_ai.usage.*` and `llm.token_count.*` on the LLM spans; see [Token usage and prompt caching](#token-usage-and-prompt-caching) |
| Prompts and replies | Events on root span | Events on root span, and `gen_ai.input.messages` and `gen_ai.output.messages` on it |
| Tool input and output | `tool.input.*` and `tool.output` | `input.value` and `output.value`, and `gen_ai.tool.call.arguments` and `gen_ai.tool.call.result` |
| Retries (`retry_of`) | Span links on tool spans | Span links on tool and `chat` spans |
| Subagents | One trace with `--include-subagents`; see [Subagents in one trace](#subagents-in-one-trace) | The same |

`--format otlp` keeps its span and attribute names for backwards compatibility.
Two of its values change in this release, as in `otlp-genai`: see
[Changed values](#changed-values). See [ADR-0011](../ADRs/0011-otlp-genai-semantic-conventions.md) for design rationale.

### Spans and attributes

`--format otlp-genai`, Temporal, `watch --stream-otlp`, the SDK
auto-instrumentation and the `@trace_*` decorators all build these spans. They
follow the GenAI conventions in
[open-telemetry/semantic-conventions-genai](https://github.com/open-telemetry/semantic-conventions-genai)
release [`v0.1.0`](https://github.com/open-telemetry/semantic-conventions-genai/releases/tag/v0.1.0)
(2026-11-21). The schema URL that release names,
`https://opentelemetry.io/schemas/semconv-genai-dev/0.1.0-dev`, does not
resolve yet, so the payload sets no `schemaUrl`. The instrumentation scope
is `agent-trace` with the agent-strace version, and the `service.version`
resource attribute is the agent-strace version. See
[Conformance check](#conformance-check) for what OpenTelemetry's own check
reports.

| Span | Name | Kind | `gen_ai.operation.name` | Spec |
|---|---|---|---|---|
| Session root | `invoke_agent {gen_ai.agent.name}`, or `invoke_agent` when the session has no agent name | `INTERNAL` (`1`) | `invoke_agent` | [Invoke agent internal span](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-agent-spans.md) |
| Model request | `chat {gen_ai.request.model}`, or `chat` when no model was recorded | `CLIENT` (`3`) | `chat` | [Inference](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/client-inference.md) |
| Tool call, including a failed one | `execute_tool {gen_ai.tool.name}` | `INTERNAL` (`1`) | `execute_tool` | [Execute tool](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/execute-tool.md) |

- The root carries `gen_ai.agent.name` only when the session has an agent
  name. It carries no `gen_ai.agent.id`: the conventions reserve that for the
  stable ID of a hosted agent resource, and the agent-strace session ID is
  neither. The session ID is on `session.id` and `agent.session_id`, and the
  agent's own session ID on `gen_ai.conversation.id`.
- A failed tool call carries `error.type`: the recorded `error_type` or
  `exception_type`, else `_OTHER`. Its span is named after the error's
  `tool_name`, else its call's, else `error`. The `@trace_tool` decorator, for
  one, records the name only on the call.
- The content attributes of the
  [GenAI spans](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md)
  carry text the export already sends, with the same redaction and
  truncation. The events and keys that carried this text before are kept.
  See [Content capture](#content-capture) for when content is exported.
  - Each is a JSON string. The conventions say these attributes MUST be
    recorded as an [AnyValue](https://opentelemetry.io/docs/specs/otel/common/#anyvalue),
    a structured array or map, and describe it as the data model, not an
    encoding. agent-strace sends a JSON string, which backends that do not
    read structured values can still show and search. OpenTelemetry's
    conformance check accepts either form: it parses a string and checks
    the result against the message schema, and agent-strace's messages pass.
  - The root has `gen_ai.input.messages`: the user prompts and assistant
    replies up to and including the last user prompt, each a message with
    one `text` part. Its `gen_ai.output.messages` is one `assistant` message
    with a `text` part per later reply. With no user prompt, every reply is
    output. Each text is at most 1000 characters, and a key with no message is
    left out.
  - A tool span whose call has input has `gen_ai.tool.call.arguments`: the
    `input.value` JSON, or, when that is cut, a JSON object of the
    `gen_ai.tool.input.*` values, each at most 200 characters. Input that is
    not an object, such as a list, is cut the way a result is.
  - A tool span with a result has `gen_ai.tool.call.result`, at most 500
    characters, as `output.value` is. A result that is JSON, either a
    recorded object or list or text that parses as JSON, stays valid JSON
    when it is cut: members are kept in order while they fit, the first that
    does not fit is cut the same way, string values are shortened, and the
    rest are left out. Text that starts with `{` or `[` but does not parse
    becomes a JSON string of as much of it as fits. That includes JSON a
    recorder already cut, such as a hook result over 1000 characters, and
    the Python `repr` that hooks in 0.121.1 and earlier stored when an agent
    reported a tool result as an object, such as `{'type': 'text'}`. Hooks
    now store that object as JSON, so it is cut with its structure kept.
    Other text, such as `1 passed`, is the plain text, as before.
    `output.value` keeps the plain cut text.
  - A `chat` span has no content: recorders store no prompt or reply per
    model request.
- A `chat` span carries `gen_ai.response.id`, `gen_ai.response.model`,
  `server.address` and `server.port` only when the capture path recorded
  them. Nothing is derived from the request model or the provider.
  - The OpenAI and Anthropic SDK integrations (`instrument_openai`,
    `instrument_anthropic`) record the response ID and model from the
    response, or from the first stream chunk or `message_start` that has
    them, and the server host and port from the client's base URL. The port
    is the one in the URL, else 443 for `https` and 80 for `http`. A failed
    request keeps the server but has no response ID or model.
  - `import` of a Claude Code transcript records the response model, and the
    response ID when the stored `message_id` is there, but no server address
    or port. The `chat` span of each imported model response carries what was
    recorded as `gen_ai.response.model` and `gen_ai.response.id`, in
    `--format otlp` too; see
    [Model responses imported from a transcript](#model-responses-imported-from-a-transcript).
  - The hooks, `record`, `record-http`, the LiteLLM, LangChain, CrewAI,
    Strands and OpenAI Agents integrations, and the `@trace_*` decorators
    record none of them, so their `chat` spans have none.
  - `--format otlp` exports other model requests as events on the root span,
    without these keys.
- A tool or `chat` span whose event names another event in `retry_of` has a
  [span link](https://opentelemetry.io/docs/concepts/signals/traces/#span-links)
  to that event's span, with `agent.strace.link.type` `retry_of`. A link to
  the span itself, or to an event that is not in the session, is left out.
  agent-strace's hooks and imports do not write `retry_of`.
- `gen_ai.provider.name` names the provider on the root and model request
  spans, using the conventions' well-known values: `anthropic`, `openai`,
  `aws.bedrock`, `azure.ai.openai`, `azure.ai.inference`, `gcp.gen_ai`,
  `gcp.vertex_ai`, `gcp.gemini`, `mistral`, `deepseek`, `x_ai`, `groq`,
  `ollama`, `cohere`, `perplexity`, `ibm.watsonx.ai` and `moonshot_ai`. A
  provider it cannot name is `unknown`. Gemini and Google models are `gcp.gen_ai` unless
  the recorded provider or model says Vertex AI (`gcp.vertex_ai`) or the
  Gemini API (`gcp.gemini`), for example a LiteLLM `vertex_ai/` or `gemini/`
  model prefix. The raw recorded provider stays on `agent.strace.provider`.
- Codex, GitHub Copilot and Cursor can run models from more than one
  provider, so their product name names none. The Codex hook records `codex`
  as the provider of each model request; for such a request, and for the
  product names `openai-codex`, `copilot`, `github-copilot`, `cursor` and
  `cursor-agent`, the model decides: `gpt-5.3-codex` is `openai`, and a model
  that names no provider is `unknown`. The root span has no model, so the root
  of a Codex, Copilot or Cursor session is `unknown`.
- `gen_ai.response.finish_reasons` is a string array (an OTLP `arrayValue`),
  one entry per generation, such as `["end_turn"]`. A recorded list keeps its
  order. The recorded value stays on `agent.strace.finish_reason`.
- `gen_ai.system` repeats `gen_ai.provider.name` for backends that still read
  it. It is deprecated, since the conventions renamed it to
  `gen_ai.provider.name` in semantic conventions v1.37.0, and will be removed
  in a later release. Read `gen_ai.provider.name` instead.

#### Content capture

The conventions make content opt-in: instrumentation reads
`OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT`, whose default `NO_CONTENT`
records no prompts, replies or tool arguments and results. agent-strace does
not read that variable. Its exports send the content a session recorded, in
the content attributes above and in the events and keys that carried it
before. Secrets are removed when a session is recorded, unless it was recorded
with `--no-redact`; see [Secret redaction](security.md#secret-redaction).
Export has no option that leaves the content out, so do not send a session
whose prompts or tool output must not leave the machine.

#### Keys outside the GenAI registry

The export also sends keys that the GenAI registry does not define.
OpenTelemetry's conformance check lists each as `missing_attribute`. They are
kept for backends and queries that read them:

- `agent.strace.*`: agent-strace's own values, such as the recorded provider,
  cost, evidence and provenance.
- `agent.session_id` on the resource and the root, `agent.errors`,
  `agent.llm_requests` and `agent.tool_calls` on the root, `tenant_id` on the
  resource when the session has one, and `session.id`.
- The [OpenInference](#openinference-compatibility) keys: `openinference.span.kind`,
  `input.value`, `input.mime_type`, `output.value`, `output.mime_type`,
  `llm.*` and `metadata`.
- `gen_ai.system`, the deprecated copy of `gen_ai.provider.name`.
- `gen_ai.prompt` and `gen_ai.completion` on the prompt and reply events.
- `gen_ai.tool.input.*` and `gen_ai.tool.output` on tool spans, and
  `tool.name`.
- `exception.escaped` on `exception` events.
- `service.version` on the resource. It is a core OpenTelemetry attribute,
  outside the GenAI registry the check reads.

#### Renamed in this release

Dashboards, alerts and queries on the earlier `otlp-genai` names need
updating. `--format otlp` keeps its names, but two of its values change; see
[Changed values](#changed-values).

| Earlier | Now |
|---|---|
| `gen_ai.agent.session ({agent})` root span, `gen_ai.operation.name` `agent.session` | `invoke_agent {agent}` (or `invoke_agent`), `invoke_agent` |
| `gen_ai.client.operation` model request span | `chat {model}` (or `chat`) |
| `gen_ai.tool.call/{tool}` tool span, `gen_ai.operation.name` `execute`, kind `CLIENT` (`3`) | `execute_tool {tool}`, `execute_tool`, kind `INTERNAL` (`1`) |
| `gen_ai.agent.id` on the root (the agent-strace session ID) | not emitted; use `session.id` |
| `gen_ai.agent.name` `agent` when the session had no agent name | not emitted |
| `gen_ai.system` `google` | `gcp.gen_ai` (or `gcp.vertex_ai`, `gcp.gemini`) |
| `gen_ai.system` only | `gen_ai.provider.name`, with `gen_ai.system` kept as a deprecated copy |
| Scope version `genai-semconv-1.27` | the agent-strace version |
| `gen_ai.provider.name` `unknown` on a Codex `chat` span | from the model, such as `openai` for `gpt-5.3-codex`; `unknown` when the model names no provider |
| `gen_ai.provider.name` `openai` on the root of an `openai-codex` session recorded without hook attribution | `unknown`, as for a hook-recorded Codex session |

#### Changed values

Both `--format otlp` and `--format otlp-genai` change these values in this
release:

| Attribute | Earlier | Now |
|---|---|---|
| `gen_ai.response.finish_reasons`, on the `chat` span (`otlp-genai`) or the `gen_ai.assistant.message` event (`otlp`) | a string, such as `"end_turn"` or `"error"` | a string array, such as `["end_turn"]` or `["error"]`, as the conventions type it (`string[]`) |
| `service.version` resource attribute | `agent-trace` | the agent-strace version |

#### Classic tool spans

In `--format otlp`, a failed tool call and a call with no result have the
same identity as a call that returned one, so a backend shows their operation
and tool. A failed call's name comes from the error's `tool_name`, else its
call's, else `error`.

| Span | 0.121.1 and earlier | Now |
|---|---|---|
| Failed tool call | named `{tool}`, kind `INTERNAL` (`1`), with `tool.name` but no `gen_ai.*` keys | named `tool/{tool}`, kind `CLIENT` (`3`), adding `gen_ai.tool.name`, `gen_ai.operation.name` `tool.call` and `gen_ai.tool.call.id` |
| Tool call with no result | named `{tool}`, kind `INTERNAL` (`1`), with `tool.name` but no `gen_ai.*` keys | named `tool/{tool}`, kind `CLIENT` (`3`), adding the same three keys |

The other attributes, the status and the events are unchanged.

An `error` that ends a model request is not a tool call. It keeps its
earlier span: named after its `tool_name`, else `error`, kind `INTERNAL`,
with only `tool.name`.

### Subagents in one trace

By default each session is its own trace. With `--include-subagents`, `export`
sends a session and the subagent sessions under it as one trace, in one OTLP
payload, for both `--format otlp` and `--format otlp-genai`:

```bash
agent-strace export <session-id> --format otlp-genai --include-subagents \
  --endpoint http://localhost:4318
```

- The trace ID is the root session's. Each subagent's root span has the
  parent span of the call that started it (its `parent_event_id`). When that
  is not a span in the parent's payload, or is not a non-empty string, the
  subagent's root is under the parent session's root.
- Each subagent's root span gets its status, `error.type` and outcome
  attributes from its own session's events (see
  [Session root status](#session-root-status)). The parent's root reads only
  the parent's events, such as the result of the call that started the
  subagent.
- The subagents are those `agent-strace tree` shows, up to the same depth. A
  session in the tree twice is exported once.
- Codex subagents run inside their parent's session, so they are already in
  its trace.

### Joining with native agent telemetry

Both formats emit `gen_ai.conversation.id` when the agent supplied its own
session ID: hook sessions whose payload carries `session_id` (Claude Code,
GitHub Copilot CLI, Codex, Gemini CLI, Cursor), and `import` of Claude Code
JSONL transcripts. The value is the agent's full
session ID, not the agent-strace session ID. Sessions without one (`record`,
proxy, SDK integrations, and sessions recorded before this field existed)
omit the attribute.

| Native source | Join on |
|---|---|
| GitHub Copilot CLI OTel (`gen_ai.conversation.id`) | `gen_ai.conversation.id` = `gen_ai.conversation.id` |
| Claude Code OTel (`session.id`) | `gen_ai.conversation.id` = `session.id` |

Do not join on `session.id` from agent-strace: it is the local agent-strace
session ID, which is truncated to 16 characters for hook sessions.

#### Version skew

Sessions with a native ID store it as `conversation_id` in `meta.json`.
Readers and collectors on v0.106.0 or older reject metadata that contains
this key: `list` and other readers on a shared trace store fail to load the
session, and the collector returns 400 on `POST /sessions`. Upgrade the
collector and every reader that shares the trace store before the clients
that write it. Sessions without a native ID do not write the key and stay
readable by older versions.

From this release, readers and the collector ignore metadata and event keys
they do not know and keep them when they rewrite the record, so later
additive fields do not need this ordering.

### OpenInference compatibility

The same `otlp-genai` payload includes additive OpenInference attributes for
Phoenix and other OpenInference consumers. The OTLP span kind stays unchanged.
`openinference.span.kind` identifies the AI operation.

| Span | OTLP kind | OpenInference kind | Key attributes |
|---|---:|---|---|
| Session root | `INTERNAL` (`1`) | `AGENT` | `session.id`, `metadata` |
| LLM call | `CLIENT` (`3`) | `LLM` | `llm.provider`, `llm.model_name`, `llm.finish_reason`, token counts, recorded cost |
| Tool call | `INTERNAL` (`1`) | `TOOL` | `tool.name`, `input.value`, `output.value` |

`agent.strace.*` attributes preserve redaction state, persisted provenance,
event IDs, previous hashes, recorded cost, and observed structural evidence
gaps. `session.id` is copied to every span for backend filtering.

`llm.finish_reason` is the OpenInference finish reason, added in OpenInference
semantic conventions 0.1.29. It appears only when the model request recorded a
finish reason, and is the first entry of `gen_ai.response.finish_reasons`,
such as `"end_turn"`.

The mapping is not fully lossless:

- Multi-turn user and assistant messages remain root span events, and the
  root's `gen_ai.input.messages` and `gen_ai.output.messages`. They have no
  OpenInference message attributes.
- OpenInference types `llm.finish_reason` as one string, so when a request
  recorded several finish reasons it carries only the first, as OpenInference's
  own instrumentations and GenAI converter do. `gen_ai.response.finish_reasons`
  keeps them all. A failed or cancelled request with no recorded reason has
  `gen_ai.response.finish_reasons` `["error"]` and no `llm.finish_reason`.
- Tool input is capped at 1000 characters and tool output at 500 characters.
  `gen_ai.tool.call.arguments` and `gen_ai.tool.call.result` keep the same
  limits, and a value that is JSON is cut so that it stays valid JSON; see
  [Spans and attributes](#spans-and-attributes).
- Only recorded cost is exported. Missing cost is not estimated.
- Redaction markers and flags survive, but removed secret values do not.
- Observed structural gaps survive. Provider capture-matrix blind spots remain
  in `agent-strace health`.
- Complete hash-chain verification still requires the source NDJSON.

The machine-readable contract is
[`examples/otlp/openinference-mapping.json`](../examples/otlp/openinference-mapping.json).

### Token usage and prompt caching

Each model request exports its input, output and prompt cache token counts in
`--format otlp`, `--format otlp-genai` and the direct Langfuse export. They are
read with the same provider normalization as [`cost`](commands.md#cost), so
the input is the whole input, cache reads and writes included, for every
provider. Anthropic `input_tokens` and Bedrock `inputTokens` leave cache
tokens out, so they are added. OpenAI, Codex and Gemini counts already include
them. Gemini thinking tokens count as output and tool-use prompt tokens as
input.

| Count | `--format otlp-genai` | `--format otlp` | Langfuse `usageDetails` |
|---|---|---|---|
| Whole input | `gen_ai.usage.input_tokens`, `llm.token_count.prompt` | `gen_ai.usage.input_tokens` | sum of the `input*` keys |
| Uncached input | not exported | not exported | `input` |
| Cache reads | `gen_ai.usage.cache_read.input_tokens`, `llm.token_count.prompt_details.cache_read` | `gen_ai.usage.cache_read.input_tokens` | `input_cached_tokens` |
| Cache writes | `gen_ai.usage.cache_write.input_tokens`, `llm.token_count.prompt_details.cache_write` | `gen_ai.usage.cache_write.input_tokens` | `input_cache_creation`, and `input_cache_creation_1h` for Anthropic and Bedrock 1-hour writes |
| Output | `gen_ai.usage.output_tokens`, `llm.token_count.completion` | `gen_ai.usage.output_tokens` | `output` |
| Reasoning output | `gen_ai.usage.reasoning.output_tokens` | not exported | not exported |
| Total | `llm.token_count.total` | not exported | `total` |

- The cache read and write attributes appear only when they are not zero, as
  the [GenAI inference span](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/client-inference.md)
  conventions recommend. They are part of `gen_ai.usage.input_tokens`, not
  added to it. The OpenInference `prompt_details` counts are part of
  `llm.token_count.prompt` in the same way.
- `gen_ai.usage.reasoning.output_tokens` appears only when the side that
  supplied the output recorded a reasoning count that is not zero: Codex
  `reasoning_output_tokens`, OpenAI `output_tokens_details.reasoning_tokens`
  or `completion_tokens_details.reasoning_tokens`, or Gemini
  `thoughtsTokenCount`. It is part of `gen_ai.usage.output_tokens`, as the
  conventions require, not added to it, and never exceeds it. Anthropic
  reports no separate thinking count, so its thinking stays in the output
  only.
- The input and its cache counts come from the request when it recorded an
  input count, otherwise from the response. The output comes from the
  response, otherwise from the request. In `--format otlp` they go on the
  request's `gen_ai.system.message` event when the request recorded them, and
  on the response's `gen_ai.assistant.message` event otherwise, as for the SDK
  integrations, which record usage on the response.
- Langfuse generations always carry `usage` with `input` (the whole input),
  `output` and `total`. A request that read or wrote a cache also carries
  `usageDetails`. Langfuse prices each usage key with the model price of the
  same key and expects the keys to be exclusive, so `usageDetails.input` is
  the uncached input. Langfuse lays `usageDetails` over `usage`. See
  [Langfuse token and cost tracking](https://langfuse.com/docs/observability/features/token-and-cost-tracking).
  OpenAI 12-hour cache writes are sent as `input_cache_creation`, because
  Langfuse has no 12-hour key. Long writes of a model whose provider cannot be
  inferred from its name are sent the same way. No cost is sent: Langfuse
  computes it from its model prices.
- Langfuse's OTLP endpoint reads the same `gen_ai.usage.*` attributes and
  subtracts the cache reads and writes from the input itself.
- Usage that a Claude Code import records on `assistant_response` events is
  exported on the `chat {model}` span of each imported model response, in
  both formats, with the same keys; see
  [Model responses imported from a transcript](#model-responses-imported-from-a-transcript).
  Temporal, built on `otlp-genai`, has the same spans. The direct Langfuse
  export does not export it yet. `cost` counts it.

### Conformance check

[`experiments/semconv-conformance/`](../experiments/semconv-conformance/README.md)
runs OpenTelemetry's
[GenAI conformance runner](https://github.com/open-telemetry/semantic-conventions-conformance),
which checks a payload with Weaver live-check against the GenAI registry. Its
README has the commands to run it again. It exports three scenarios: the
synthetic parent and subagent sessions of
[`experiments/telemetry-dev`](../experiments/telemetry-dev/), once as two
traces and once as one trace with `--include-subagents`, and a session
recorded by the OpenAI and Anthropic SDK integrations against the runner's
mock model server. The runs against `v0.1.0` and `e6658f1` report the same
findings. [RESULTS.md](../experiments/semconv-conformance/RESULTS.md) lists
them. No ERROR span lacks `error.type`: every ERROR root has one (see
[Session root status](#session-root-status)). What remains:

- The scope has no schema URL (`instrumentation_scope_schema_url_missing`),
  because the release's schema URL does not resolve.
- A `chat` span lacks `gen_ai.response.id`, `gen_ai.response.model` or
  `server.address` when its capture path did not record them, and a failed
  request has no response or usage (`genai_expected_attribute_missing`).
- The keys listed in [Keys outside the GenAI registry](#keys-outside-the-genai-registry)
  are `missing_attribute`, and attributes still in development are
  `not_stable`.
- Two information-level findings: `error.type` values such as
  `BadRequestError` on failed `chat` spans and roots, which the conventions
  allow, and `gen_ai.provider.name`
  `unknown` on the root of a session that used more than one provider.

The check also confirms that `gen_ai.input.messages` and
`gen_ai.output.messages` match the message schema, and that span links reach
the collector.

---

## Per-backend setup

Export POSTs OTLP/HTTP JSON by default. `--protocol http/protobuf` sends the
same payload as OTLP protobuf instead, for collectors that take only protobuf,
such as Phoenix. Without the flag, export reads
`OTEL_EXPORTER_OTLP_TRACES_PROTOCOL`, then `OTEL_EXPORTER_OTLP_PROTOCOL`.
agent-strace encodes protobuf with the standard library and has no gRPC
exporter. A collector that answers 415 Unsupported Media Type gets a hint that
names the other `--protocol`. See [`export`](commands.md#export).

### Phoenix / OpenInference

Phoenix's `/v1/traces` takes only OTLP protobuf, so pass
`--protocol http/protobuf`. Without it Phoenix answers 415 and export exits 1.

```bash
agent-strace export <session-id> --format otlp-genai \
  --endpoint http://localhost:6006 --protocol http/protobuf

# Or set it once for every OTLP send
export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf
agent-strace export <session-id> --format otlp-genai --include-subagents \
  --endpoint http://localhost:6006

# Stream while the session runs
agent-strace watch --stream-to http://localhost:6006 --stream-otlp \
  --protocol http/protobuf
```

The exporter appends `/v1/traces`. Phoenix serves OTLP/HTTP on its UI port
`6006`. OpenInference span kinds and compatibility attributes are included in
the payload.

Tested locally with Phoenix 22.0.1 and synthetic sessions; see
[experiments/phoenix/RESULTS.md](../experiments/phoenix/RESULTS.md). Phoenix
stored every span with its parent, name, status and times. It took each span's
kind from `openinference.span.kind`, and kept the token counts and the
`agent.strace.*` redaction, provenance and evidence attributes. In that test:

- Phoenix shows a cost it works out from the token counts, not
  `llm.cost.total`, which stays on the span. For one request the two
  differed.
- Phoenix did not store span links, so retry links are lost.
- Exception events kept only their `exception.*` attributes.
- Each session is a trace and a Phoenix session of its own. Use
  `--include-subagents` to send a parent and its subagents as one trace.
- `--format otlp` spans have no OpenInference kind, so Phoenix shows them as
  `UNKNOWN`, with no token counts or cost.
- With `watch --stream-otlp`, Phoenix kept the first copy of the root span
  and ignored the later ones.

### Langfuse

Langfuse accepts OTLP/HTTP JSON at `/api/public/otel/v1/traces`. Pass the base
OTLP endpoint because the exporter appends `/v1/traces`.

```bash
AUTH_STRING="$(
  printf '%s' "$LANGFUSE_PUBLIC_KEY:$LANGFUSE_SECRET_KEY" |
  base64 | tr -d '\n'
)"

agent-strace export <session-id> --format otlp-genai \
  --endpoint https://cloud.langfuse.com/api/public/otel \
  --header "Authorization: Basic $AUTH_STRING" \
  --header "x-langfuse-ingestion-version: 4"
```

The OTLP recipe exports traces. Eval scores still use the existing direct
exporter:

```bash
agent-strace export --scores --backend langfuse
```

It posts to `--langfuse-host`, else `LANGFUSE_HOST`, else
`https://cloud.langfuse.com`. The POST does not follow redirects, so pass the
final URL.

### Honeycomb

```bash
agent-strace export <session-id> --format otlp-genai \
  --endpoint https://api.honeycomb.io \
  --header "x-honeycomb-team: $HONEYCOMB_API_KEY" \
  --service-name my-agent
```

Use `https://api.eu1.honeycomb.io` for the EU instance.

### Datadog

Enable the Datadog Agent OTLP HTTP receiver, then send to port `4318`:

```bash
agent-strace export <session-id> --format otlp-genai \
  --endpoint http://localhost:4318
```

This recipe does not send OTLP JSON to the Datadog logs intake.

### Grafana Tempo / Jaeger

```bash
# Local collector
agent-strace export <session-id> --format otlp-genai \
  --endpoint http://localhost:4318
```

### New Relic

```bash
agent-strace export <session-id> --format otlp \
  --endpoint https://otlp.nr-data.net \
  --header "api-key: $NEW_RELIC_LICENSE_KEY"
```

### Splunk

```bash
agent-strace export <session-id> --format otlp \
  --endpoint https://ingest.<realm>.signalfx.com \
  --header "X-SF-Token: $SPLUNK_ACCESS_TOKEN"
```

### Telemetry.dev

```bash
agent-strace export <session-id> --format otlp-genai \
  --endpoint https://api.telemetry.dev \
  --header "Authorization: Bearer $TELEMETRY_DEV_API_KEY"
```

Tested once on 0.120.0, with a partial result. In that test, content was not
in Telemetry.dev's Input, Output, Arguments or Result panels, a subagent was a
separate trace, and retries were not linked. The export now sets the GenAI
content attributes, links retries, and sends subagents in one trace with
`--include-subagents`. How Telemetry.dev shows these is predicted, not tested.
Telemetry.dev's own redaction can rewrite ordinary IDs. Read the limits in
[telemetry-dev.md](telemetry-dev.md) before relying on it.

### Offline fixture validation

The fixtures contain no credentials and require no network:

```bash
python3 -m json.tool examples/otlp/genai-conformance.json >/dev/null
PYTHONPATH=src python3 -m unittest tests.test_otlp_conformance -v
```

Backend metadata and limitations are in
[`examples/otlp/backend-recipes.json`](../examples/otlp/backend-recipes.json).

---

## Live OTLP streaming

Stream spans to an OTLP/HTTP receiver while the session runs, rather than exporting after it ends. Pass the receiver's base URL to `watch --stream-to` and add `--stream-otlp`:

```bash
agent-strace watch --stream-to http://localhost:4318 --stream-otlp
```

Without `--stream-otlp`, `--stream-to` POSTs the raw events as NDJSON to the URL exactly as given. That suits the [agent-strace collector](server.md#live-streaming-from-watch), not an OTLP receiver.

Watch sends OTLP JSON unless `--protocol http/protobuf`, or one of the `OTEL_EXPORTER_OTLP_*PROTOCOL` variables, says otherwise, as [export](#per-backend-setup) does. Phoenix needs protobuf:

```bash
agent-strace watch --stream-to http://localhost:6006 --stream-otlp --protocol http/protobuf
```

With `--stream-otlp`, watch drops trailing slashes from the URL path, then appends `/v1/traces` unless the path already ends with it. The query string is kept:

| `--stream-to` | Spans are POSTed to |
|---|---|
| `http://localhost:4318` | `http://localhost:4318/v1/traces` |
| `http://localhost:4318/v1/traces` | `http://localhost:4318/v1/traces` |
| `https://otel.example.com/api/public/otel/` | `https://otel.example.com/api/public/otel/v1/traces` |
| `https://otel.example.com/ingest?team=a` | `https://otel.example.com/ingest/v1/traces?team=a` |

Watch sends events in batches: when `--stream-batch-size` events are waiting, when `--stream-flush-interval` seconds have passed since the last POST, and when watch stops. Each POST carries only the spans built from its own batch, so the backend receives the session in pieces:

- Every POST includes the session root span again, with the same trace ID and span ID. A backend may show it more than once or keep only one copy. Phoenix keeps the first copy.
- The root span's status is always UNSET, with no `error.type` and no outcome attributes, because a batch does not show how the session ended. See [Session root status](#session-root-status).
- A tool call whose call and result land in different batches arrives as two spans, one from each batch.

For one complete set of spans per session, export it after it ends with `agent-strace export --format otlp-genai` (see [Per-backend setup](#per-backend-setup)).

Watch sends no authentication headers and has no flag to add them, so it cannot stream straight to a hosted backend that needs an API key, such as Honeycomb. A local receiver that needs no key, such as Jaeger or the Datadog Agent's OTLP receiver, works directly. For a hosted backend, run a local [OpenTelemetry Collector](https://opentelemetry.io/docs/collector/) that adds the key and forwards the spans:

```yaml
# otel-collector.yaml
receivers:
  otlp:
    protocols:
      http:
        endpoint: 127.0.0.1:4318
exporters:
  otlphttp:
    endpoint: https://api.honeycomb.io
    headers:
      x-honeycomb-team: ${env:HONEYCOMB_API_KEY}
service:
  pipelines:
    traces:
      receivers: [otlp]
      exporters: [otlphttp]
```

```bash
agent-strace watch --stream-to http://127.0.0.1:4318 --stream-otlp
```

| Flag | Default | Description |
|---|---|---|
| `--stream-to URL` | none | `http://` or `https://` URL to stream to |
| `--stream-otlp` | off | POST OTLP/HTTP JSON spans to the `/v1/traces` path under `--stream-to` instead of NDJSON to the URL itself |
| `--stream-batch-size N` | `10` | Send once N events are waiting |
| `--stream-flush-interval S` | `2` | Send waiting events once S seconds have passed since the last POST, even if the batch is not full. `0` sends each event at once |

Watch checks `--stream-to` and `--stream-flush-interval` before it looks up the session or writes anything. It refuses a URL that does not start with `http://` or `https://`, has no host, has a bad port, contains spaces or control characters, or carries a user name or password. It also refuses a flush interval that is negative, infinite or NaN. Each refusal is one line, such as `[watch] invalid --stream-to value: it must not contain a user name or password`, and exit status 1.

A failed POST never stops watch. It prints one line to stderr, such as `[stream] OTLP POST to https://otel.example.com failed: HTTP Error 500: Internal Server Error`, and goes on with the next batch. The failed batch is not sent again. These lines, and the `[watch] Streaming events to` line at startup, name only the URL's scheme, host and port, never its path, query or user info. Watch does not follow redirects: a 3xx answer is a failed POST whose line says so, so pass the final URL. Each POST times out after 10 seconds.

When watch stops, it waits up to 15 seconds for the last POST to finish. If that takes more than a second, it prints `[stream] Waiting up to 14s more for the last events to be sent; press Ctrl-C to stop waiting.` If the wait runs out or you press Ctrl-C, it prints one line such as `[stream] Stopped waiting; 3 events were not confirmed sent.` and exits as it would have otherwise.

---

## Baseline anomaly detection

Build a statistical baseline from completed sessions that started recently and
alert when a new session deviates:

```bash
# Build a baseline from completed sessions that started in the last 30 days
agent-strace baseline update --since 30

# Check a session against the baseline
agent-strace baseline check <session-id>

# Show baseline statistics
agent-strace baseline show
```

Metrics tracked: cost, tool call count, duration, error rate, and LLM request
count. Each metric gets a mean and sample standard deviation. A session is
flagged when any comparable metric exceeds `--sigma` standard deviations from
the mean (default: `2.0`). At least three baseline samples are required before
a metric is compared.

Cost is the same total that `agent-strace cost` reports with its default
model, including recorded token usage where the agent provides it. A session
whose events cannot be priced adds no cost sample and is not compared on cost.
Older versions saved every cost as 0, so run `agent-strace baseline update`
again after upgrading.

```bash
agent-strace baseline check <session-id> --sigma 2.5
```

Use the same custom profile path when updating, checking, and displaying a
baseline:

```bash
agent-strace baseline update --since 14 --output .agent-traces/ci-baseline.json
agent-strace baseline check <session-id> --baseline .agent-traces/ci-baseline.json
agent-strace baseline show --baseline .agent-traces/ci-baseline.json
```

Use `baseline check` as a CI gate — it exits 1 when the session is anomalous:

```bash
agent-strace baseline check "$SESSION_ID"
```

It also exits 1 when the session cannot be measured (`[UNSCORED]`). A session
with no end time, such as one the watchdog killed, is scored on the events it
recorded; see [commands.md](commands.md#baseline).

## EU AI Act Audit Packages

`agent-strace export --format eu-ai-act` creates a local JSON package for
Article 12 logging evidence and Article 13 transparency documentation:

```bash
agent-strace export <session-id> --format eu-ai-act --output compliance-report.json

agent-strace export --all --since 2026-01-01 --until 2026-03-31 \
  --format eu-ai-act --output Q1-2026-audit.json
```

The export includes event summaries, data categories processed, tools and
models observed, human oversight points, and hash-chain integrity metadata.
Verify the exported chain links with:

```bash
agent-strace verify --from-export compliance-report.json
```

This proves the package is complete and consistent with itself, not that it
matches the source store. For that, compare each session's `root_hash` and
`event_count` with `root_hash` and `total_events` from
`agent-strace verify <session-id> --format json` on the source store. See
[EU AI Act export package](commands.md#eu-ai-act-export-package).

Before exporting a store for review, run:

```bash
agent-strace audit-readiness
```

---

## Behavioral metrics

Export per-session behavioral metrics as OTLP gauge metrics:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=https://api.honeycomb.io

agent-strace export --metrics --backend otlp --since 30d
```

The metrics are POSTed to `--otlp-endpoint`, else `OTEL_EXPORTER_OTLP_ENDPOINT`,
with `/v1/metrics` appended. The POST does not follow redirects, so pass the
final URL.

| Metric | Description |
|---|---|
| `agent_strace.session.cost_usd` | Session cost, as `agent-strace cost` reports it with its default model |
| `agent_strace.session.error_rate` | Errors, failed model requests included, / tool calls (see [Session error counts](#session-error-counts)) |
| `agent_strace.session.retry_rate` | Consecutive same-tool retries / tool calls |
| `agent_strace.session.blast_radius` | Distinct files written |
| `agent_strace.session.duration_s` | Wall-clock session duration |
| `agent_strace.eval.score` | Judge score per session |

`agent_strace.session.cost_usd` prices recorded token usage, cache reads and
writes included, at the model's rates. Earlier versions priced each session's
token count at a flat $3 per million. A session whose events cannot be read, or
that `agent-strace cost` refuses, has no `cost_usd` data point; its other
metrics are still exported.

---

## How sessions map to OpenTelemetry

| agent-trace | OpenTelemetry |
|---|---|
| session | trace |
| session with subagents, with `--include-subagents` | one trace; each subagent's root span under the span that started it |
| tool_call + tool_result | span (with duration) |
| error | span with error status + exception event |
| failed or cancelled model request | error status and `error.type`; see [Failed and cancelled model requests](#failed-and-cancelled-model-requests) |
| user_prompt | event on root span, and in `gen_ai.input.messages` (`otlp-genai`) |
| assistant_response | event on root span, and in `gen_ai.input.messages` or `gen_ai.output.messages` (`otlp-genai`); one imported with recorded usage is also a `chat {model}` span (see [Model responses imported from a transcript](#model-responses-imported-from-a-transcript)) |
| session_id | trace ID |
| event_id | span ID |
| parent_id | parent span ID |
| retry_of | span link |

### Span status

The `otlp` and `otlp-genai` exports, Temporal, `watch --stream-otlp` and
`a2a-tree --format json` set span status as
[Recording errors](https://github.com/open-telemetry/semantic-conventions/blob/v1.45.0/docs/general/recording-errors.md)
specifies for instrumentation. The GenAI span conventions defer to that
document for span status.

| Span | Status |
|---|---|
| A tool call with a result, a completed model request, a successful A2A call | UNSET (`{"code": 0}`) |
| A request or tool call still pending at the end | UNSET (`{"code": 0}`) |
| A session root span that did not fail, as below | UNSET (`{"code": 0}`) |
| An `error`, a failed or cancelled model request, a failed A2A call | ERROR (`{"code": 2}`); error and model request spans carry the recorded error text, at most 200 characters, as `message` |
| A session root span that failed, as below | ERROR (`{"code": 2}`) with `error.type` |

#### Session root status

A session's root span, and each subagent's root span with
`--include-subagents`, follows how the session ended, not whether anything
failed along the way. It is ERROR when any of these holds:

| Condition | `error.type` |
|---|---|
| The recorded session status is `timeout` or `terminated` | `timeout` or `terminated` |
| The session's last model request failed | That request's error type (see [Failed and cancelled model requests](#failed-and-cancelled-model-requests)), else `_OTHER` |
| A failed tool operation's last outcome failed | The `error_type` or `exception_type` of the last unresolved failure, else `_OTHER` |

When more than one holds, the first row in the table names `error.type`. Every
ERROR root has `error.type`, as
[Recording errors](https://github.com/open-telemetry/semantic-conventions/blob/v1.45.0/docs/general/recording-errors.md)
requires when an operation fails; OTel's GenAI conformance check reports a
missing one as `error_type_missing_on_error`. Any other root is UNSET, never
OK, and has no `error.type`: the export does not claim the session succeeded.

- A session that failed, retried and recovered has an UNSET root. Its failed
  spans below the root keep status ERROR.
- An operation that recovered and then failed again is unresolved: the root
  is ERROR, and only the failures after its last success count as unresolved.
- A failed operation retried with different arguments is a new operation, so
  the failed one stays unresolved and the root is ERROR.
- An earlier failed model request followed by one that completed leaves the
  root UNSET. A cancelled last request does not fail the root.
- An `error` that names no tool call and no model request, such as a proxy
  failure, is no operation. It keeps its ERROR span and counts in
  `agent.errors`, but it alone does not fail the root.

The root also has these attributes, in both exports. In `otlp-genai` the
OpenInference `metadata` JSON repeats them in its `agent_strace` object,
without the `agent.strace.` prefix.

| Attribute | Value |
|---|---|
| `agent.strace.behavior.failed_operation_count` | Tool calls whose result failed or that an `error` ended |
| `agent.strace.behavior.retry_count` | Tool calls that repeat an earlier call with the same tool name, arguments, command and path |
| `agent.strace.behavior.recovered_operation_count` | Failed operations whose last outcome was a successful retry |
| `agent.strace.behavior.unresolved_failed_operation_count` | Failed operations whose last outcome failed |
| `agent.strace.session_status` | `completed`, `timeout`, `terminated`, or `not_recorded` when there is no `session_end` or a later `session_start` resumed the session |
| `agent.strace.last_model_request_failed` | Whether the session's last model request failed |

A tool result failed when it has `success` or `ok` false, a nonzero
`exit_code`, or a `status` of `error`, `failed`, `failure`, `cancelled` or
`timeout`. The session status comes from the last `session_end`, unless a
later `session_start` (a resume) reopened the session, which is
`not_recorded`. It reads the `session_end`'s `status`
(`timed_out` and `timeout` are `timeout`; `killed`, `cancelled`,
`interrupted`, `failed`, `aborted` and similar are `terminated`), else a
`reason`, `stop_reason` or `termination_reason` that mentions a timeout, a
kill, a cancel, an interrupt, a termination, an abort or an error (Copilot's
`error` and `abort` reasons are `terminated`; `complete` and `user_exit` are
`completed`), else an `exit_code` of 124
(`timeout`) or any other nonzero code (`terminated`). `completed` is the
recorded process status, not task success. The evidence packet and the
assignment `session_status` scorer ([hiring.md](hiring.md)) use the same
counts and status. The `eval` gate's `session_status` scorer uses the same
status rules, with `killed` for `terminated`.
[commands.md](commands.md#eval) lists where it reads a session differently.

The root no longer reads the session's stored `errors`. Earlier versions set
the root ERROR whenever `errors` was above 0, so a recovered session exported
as a failed trace, and the ERROR root had no `error.type`. A query, alert or
dashboard that counts ERROR roots now counts fewer of them; to keep counting
sessions with any error, filter on `agent.errors` above 0.

`watch --stream-otlp` cannot apply this rule: each POST carries one batch, not
the session's end state, so its root span is always UNSET, with no
`error.type` and none of the attributes above. Export the session after it ends
for its root status.

`session_to_otlp` and `session_to_otlp_genai` take `root_outcome=False` to
build the root that way.

No span has status OK. The OpenTelemetry trace API reserves OK for application
developers and operators who have validated an operation. Earlier versions set
OK (`{"code": 1}`) on every successful span. A query, alert or dashboard that
filters on status OK (for example TraceQL `status = ok` or
`otel.status_code = "OK"`) now matches no agent-strace spans; filter on status
not ERROR, or on UNSET, instead.

### How requests and responses are paired

The OTLP, OTLP GenAI, Temporal and Langfuse exports and `cost` pair events with
the same rules, defined in `src/agent_trace/pairing.py`:

1. A response pairs with the pending request its `parent_id` names.
2. If there is no such request, a `tool_result` or `error` stays unpaired. An
   `llm_response` pairs with the most recent pending `llm_request` that names
   the same model. A request and a response that both name no model count as
   the same model. Only when no pending request names the same model does the
   response pair with the most recent pending request where one side names no
   model. Requests and responses that name different models never pair. This
   covers SDK integrations that record responses without a `parent_id`. The
   most recent request wins because a failed call leaves its request without
   a response.

   A matching model is the only evidence this fallback has, so it always
   beats a missing one. Without that order, a request with no model (for
   example a CrewAI task) would take the response of an older `gpt-4o`
   request, and a CrewAI task result would take a failed `gpt-4o` request.
   The match where one side names no model stays as a last resort because
   some integrations record the model on only one side of a call.
3. A response whose `parent_id` names a request that already paired stays
   unpaired. It never takes another request's place.

The exports also end an `llm_request` at an `error` whose `parent_id` names
it, and export the pair as one failed model call (see
[Failed and cancelled model requests](#failed-and-cancelled-model-requests)).
Only rule 1 applies: such an `error` never pairs with a request by model, and
any other `error` stays on the tool channel. `cost` does not read `error`
events.

Requests still pending at the end are exported as unfinished spans in OTLP and
are not exported to Langfuse.

Recorders set `parent_id` while they write the trace, with rules from the same
module. `record` and `record-http` link a `tool_result` or `error` to the
pending `tool_call` with the same JSON-RPC id, and keep at most 10,000 calls
pending per proxy. Hooks link a result to the pending call with its
`tool_use_id`, or, when the provider sends none, to the oldest pending call of
the same tool name, preferring calls from the same subagent.

### Responses recorded without a request

OpenAI Codex records each model request only as an `llm_response` event, from
its `PostModelRequest` hook. No `llm_request` is recorded, by design: the
Codex [capture matrix](capture-matrix.md#openai-codex) entry declares
`results_without_requests: ["llm_response"]`. Each response carries its own
`model` (when Codex reports one), `status` and token usage. `input_tokens`
already includes cached input and `output_tokens` already includes reasoning.
A failed or cancelled request records no usage.

A response that pairs with no request is treated as the whole model request:

- `--format otlp-genai` and Temporal: the response is its own `chat {model}` span with
  its model and token counts. The root span's `agent.strace.evidence.*`
  attributes read the same capture matrix declaration as `validate`, so
  declared Codex responses are one `declared_results_without_requests`
  provider limitation instead of one `orphan_llm_response` gap each. Responses
  without a request from any other source are still orphans.
- `--format otlp`: the `gen_ai.assistant.message` event carries the
  response's own `gen_ai.request.model` and its token counts. A paired
  response carries the input counts only when its request event recorded
  none (see [Token usage and prompt caching](#token-usage-and-prompt-caching)).
- Langfuse: the generation takes its model and input count from the response.
  A paired response takes its model from its request, and its input count from
  its request when the request recorded one.
- `cost`, `tree`, `timeline`, `replay`, `watch` and the EU AI Act compliance
  export: once a session has such a response with token counts, every
  `llm_response` counts the tokens it recorded, or none if it recorded none
  (as for a failed or cancelled request), and no other event is counted,
  because its content is already part of a request's input. Nothing is also
  estimated from event size. Sessions without such a response keep the size
  estimate. `watch` counts events as they arrive, so events before the first
  recorded usage keep their size estimate.
- `cost --breakdown provider`: a response that names no model still reports
  its tokens, under the model `(not recorded)` (empty in CSV) and the provider
  `Unknown`, with no estimated price.

Cost estimates price each response's `cached_input_tokens` at the model's
cached input rate and its `cache_write_input_tokens` at the cache write rate,
and the rest of `input_tokens` at the input rate. A model without a published
cache rate, such as the `haiku`, `gpt4` and `gpt4o` choices of `--model`,
bills cached input at its input rate. See [`cost`](commands.md#cost)
for the usage shapes read from each provider.

### Model responses imported from a transcript

`agent-strace import` stores each line of a Claude Code transcript's assistant
messages as an `assistant_response` event with the recorded `model`, the
transcript `message.id` as `message_id`, and `stop_reason` when the line has
one. Claude Code writes one line per content block of a message, and the
import keeps the message's `usage` on its last line only.

In `--format otlp`, `--format otlp-genai` and Temporal, each
`assistant_response` with recorded token counts, other than a `<synthetic>`
message (see below), is also a `chat {model}` span of kind `CLIENT`, a child
of the root span, as a model request is. Its root span event is kept. The span
carries:

- `gen_ai.request.model` and `gen_ai.response.model`: the recorded model.
- `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens` and the cache read
  and write attributes, read as in
  [Token usage and prompt caching](#token-usage-and-prompt-caching), and in
  `otlp-genai` the `llm.token_count.*` attributes.
- `gen_ai.response.finish_reasons`: the recorded `stop_reason`, when there is
  one.
- `gen_ai.response.id`: the stored `message_id`, when there is one. It is
  never taken from any other field.
- `agent.strace.timing`: `inferred`.

A transcript line has one timestamp, so when the model request started was
not recorded. The span ends at the response's timestamp and starts at the
latest earlier event, in trace order, with the same `context_stream` (the main
conversation, or one sidechain) that is a `user_prompt`, a `tool_result` or
another of these responses. With no such event the span has zero length at the
response. `agent.strace.timing=inferred` marks the start as inferred: the
span's duration is the time since the model's last input, not a measured
latency.

A message recorded on more than one event with the same `message_id` is one
span, and its usage counts once.

Claude Code writes an API error, such as "Not logged in", as an assistant
message of its own with the model `<synthetic>` and zero usage. No model
request was sent, so such a message gets no span and moves no span's start; it
stays an event on the root span. The import still counts it in the session's
`llm_requests`, so the root span's `agent.llm_requests` can be higher than the
number of `chat` spans.

`cost`, `tree`, `timeline`, `replay`, `watch` and the EU AI Act compliance
export price such a session from the recorded usage, as in
[Responses recorded without a request](#responses-recorded-without-a-request):
each of these responses counts the tokens it recorded, once per
`message_id`, and no other event is also estimated from size. Sessions
recorded by hooks, whose `assistant_response` events carry no usage, keep the
size estimate.

Sessions imported by earlier versions have no `message_id` or `stop_reason`,
so their spans have no `gen_ai.response.id` or
`gen_ai.response.finish_reasons`. Sessions imported before v0.108.0 also
repeat a message's usage on each of its lines, and each line counts. To
correct either, delete the session's directory and import the log again.

### Failed and cancelled model requests

A model request failed or was cancelled when either:

- its `llm_response` has `status` `failed` or `cancelled` (also `canceled`,
  in any case), or carries a non-empty `error`. Codex records failed and
  cancelled requests this way. A response with any other `status`, or none,
  completed.
- an `error` event's `parent_id` names its `llm_request`. The SDK
  integrations and `@trace_llm_call` record a model call that raised this way.

The exports report such a request as an error, following the
[OTel GenAI spans](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md)
conventions and
[Recording errors](https://opentelemetry.io/docs/specs/semconv/general/recording-errors/):

| Export | Failed or cancelled request |
|---|---|
| `--format otlp-genai`, Temporal, `watch --stream-otlp` | The `chat` span has status ERROR with the recorded error text (at most 200 characters) as its description, and `error.type`. A request ended by an `error` is one span with the request's span ID and an `exception` event, with no separate error span. |
| `--format otlp` | There is no span per request, so the `gen_ai.assistant.message` event carries `error.type` and, when recorded, `error.message`. The `error` that ended the request stays an `INTERNAL` error span named after its `tool_name`, else `error`, with no tool call identity. |
| `export --scores --backend langfuse` | The generation has level `ERROR`, `statusMessage` set to the error text (`failed` or `cancelled` when none was recorded), and `error.type` in its metadata. A request ended by an `error` is exported as a generation too. |

`error.type` is the recorded `error_type` or `exception_type`, else the `type`
or `code` of an `error` object, else `cancelled` for a cancelled request and
`_OTHER`, the conventions' fallback value, for a failed one. When no finish
reason was recorded, `gen_ai.response.finish_reasons` is `["error"]`, which the
conventions specify for a generation that failed or was cancelled.

The GenAI conventions name no span status for a cancelled request, but they
treat cancellation as an abnormal end, so the exports report it as an error
with `error.type` `cancelled`.

What does not change:

- A completed request, or a response with no `status`, exports exactly as
  before, except that its span status is UNSET instead of OK; see
  [Span status](#span-status).
- An `error` with no `parent_id`, as older versions recorded a failed SDK
  call, still exports as an unfinished request span and a separate error
  span.

### Session error counts

A session's error count is its `error` events plus its failed model requests,
with the definition of a failed request above. The exports and every session
error count share that definition, in `src/agent_trace/failures.py`, so they
agree. It applies to:

- `errors` in `meta.json`, as the hooks write it while they capture a session
  and as `backfill-meta` derives it. The export root span's `agent.errors`
  attribute reads it. The root span's status does not; see
  [Session root status](#session-root-status).
- `agent_strace.session.error_rate`, and the error rate in `baseline`,
  `drift`, `sample` and `dashboard --trend`.
- `timeline`, `stats`, `dashboard`, `pr-comment`, the `eval` `error_count`
  scorer and `has-errors` dataset filter, the subagent error counts in
  `stats`, and the summary check in `validate`.

Each failed request counts once. A request with both an `error` event linked
to it and a failed `llm_response` is one error. Each attempt is its own
request: Codex fires `PostModelRequest` for every attempt, so a request that
failed twice before it succeeded on retry counts as two errors.

A cancelled request is not a session error and does not count toward
`error_rate`. Codex records `cancelled` when the user interrupts the turn or
the turn is aborted, with no error text and no token usage, and the SDK
integrations record it when the calling task is cancelled. Neither means the
agent failed, so a session the user interrupted is not counted as failed.
`timeline` marks a cancelled request `⊘` (`cancelled` in JSON) rather than ok,
and the exports still report it as an error with `error.type` `cancelled`.

`error_rate` is still errors divided by tool calls (at least 1), so a session
with more failed requests than tool calls has a rate above 1.

#### Sessions recorded by earlier versions

Earlier versions left failed model requests out of the stored `errors`, and
`backfill-meta` never rewrites an existing `meta.json`. No re-import is
needed:

- Commands that read the events count failed requests in old sessions too:
  `timeline`, `stats`, `baseline`, `drift`, `sample`, `eval`, and the Langfuse
  `error_rate` metric. `dashboard` and `pr-comment` show the larger of the
  stored count and the count from the events.
- Commands that read only `meta.json` keep the stored count for an old
  session: `list`, `diff`, `replay`, `share`, `standup`, the subagent `tree`,
  the web dashboard, the MCP server's session details, the export root span's
  `agent.errors`, and the Langfuse trace's `errors` metadata.
- `validate` reports a stored count that left out failed requests as the
  `info` finding `legacy_error_summary`, not as a contradiction. A count of 0
  is not written to `meta.json`, so `validate` has nothing to compare for a
  session whose only errors are failed requests.
- Profiles and fingerprints saved by earlier versions (`baseline update`,
  `drift --save-baseline`, `fingerprint --output`, `eval --save-baseline`)
  left failed requests out. Save them again to compare like with like.

---

## Inspect the OTLP payload

Without `--endpoint`, export writes the payload to a file and prints nothing
on stdout:

```bash
agent-strace export <session-id> --format otlp --output trace.json
jq . trace.json
```
