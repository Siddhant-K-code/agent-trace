# Conformance results

Run on 2026-11-23 with the pinned runner, Weaver `v0.28.0`, and the GenAI
registry at release `v0.1.0` and at commit `e6658f1`. See
[README.md](README.md) for the pins and the commands. "Before" is the export
without #563's changes, "after" is with them. Both include #567 and #569.

## Counts

Each count is the number of times Weaver gave that advice, summed over the
spans, the resource and the scope of one scenario's payload
([`summarize.py`](summarize.py)). The registry at `e6658f1` and at `v0.1.0`
gives the same counts, before and after, so one column each is shown.

| Scenario | Finding | Level | Before | After |
|---|---|---|---:|---:|
| sdk | `genai_expected_attribute_missing` | violation | 20 | 4 |
| sdk | `missing_attribute` | violation | 148 | 148 |
| sdk | `instrumentation_scope_schema_url_missing` | violation | 1 | 1 |
| sdk | `not_stable` | improvement | 47 | 57 |
| sdk | `undefined_enum_variant` | information | 3 | 3 |
| session | `genai_expected_attribute_missing` | violation | 11 | 11 |
| session | `missing_attribute` | violation | 230 | 230 |
| session | `instrumentation_scope_schema_url_missing` | violation | 2 | 2 |
| session | `not_stable` | improvement | 61 | 61 |
| session | `undefined_enum_variant` | information | 1 | 1 |
| tree | `genai_expected_attribute_missing` | violation | 11 | 11 |
| tree | `missing_attribute` | violation | 228 | 228 |
| tree | `instrumentation_scope_schema_url_missing` | violation | 1 | 1 |
| tree | `not_stable` | improvement | 61 | 61 |
| tree | `undefined_enum_variant` | information | 1 | 1 |

Violations in `sdk` go from 169 to 153. In `session` and `tree` they do not
change: the synthetic sessions record no response ID, response model or
server, and their payloads are byte for byte the same before and after. The
10 more `not_stable` in `sdk` are the new `gen_ai.response.id` and
`gen_ai.response.model` on the five successful requests, which the registry
marks as in development. `server.address` and `server.port` are stable.

Every scenario finished with `status: WARN` (violations, reported with
`--report-only`), and none crashed or missed what it declared.

Before #567, each payload also had one `error_type_missing_on_error`: a root
`invoke_agent` span with ERROR status and no `error.type`. #567 gives every
ERROR root an `error.type`, so the finding is in neither column. The `sdk`
root is now ERROR with the failed last request's `error.type`, and the
`session` and `tree` roots, whose failures recovered, are UNSET. #567 also
added six keys to each root that the registry does not define, so
`missing_attribute` is 6 per root higher than in the run before #567.

## What was fixed

`genai_expected_attribute_missing` in `sdk`, before:

| Span | Missing |
|---|---|
| `chat gpt-4o-mini`, 3 successful requests | `gen_ai.response.id`, `gen_ai.response.model`, `server.address`, each ×3 |
| `chat claude-sonnet-4-5`, 2 successful requests | `gen_ai.response.id`, `gen_ai.response.model`, `server.address`, each ×2 |
| `chat gpt-4o-mini`, the failed request | `gen_ai.response.id`, `gen_ai.response.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `server.address` |

After, only the failed request's four response and usage values are missing.
The OpenAI and Anthropic SDK integrations now record the response ID and
model and the server host and port, and the `chat` span carries them. The
server is recorded on the failed request too, so `server.address` is no
longer missing there.

## What remains

- `instrumentation_scope_schema_url_missing`, once per scope: `v0.1.0` names
  the schema URL `https://opentelemetry.io/schemas/semconv-genai-dev/0.1.0-dev`,
  which returns 404, so the payload sets no `schemaUrl`.
- `genai_expected_attribute_missing` in `sdk`: the failed request has no
  response, so no response ID, model or usage. There is nothing to record.
- `genai_expected_attribute_missing` in `session` and `tree`, 11 each, all on
  `chat claude-sonnet-4-5`: the failed request lacks `gen_ai.response.id`,
  `gen_ai.response.model`, both usage counts and `server.address`, and the two
  successful requests lack `gen_ai.response.id`, `gen_ai.response.model` and
  `server.address`. The synthetic sessions model the hooks, which record none
  of these, and agent-strace does not make them up.
- `missing_attribute`: the 68 keys below, which the GenAI registry does not
  define.
- `not_stable`: attributes the registry marks as in development. That is
  most of the GenAI attributes, so it is not something to fix.
- `undefined_enum_variant`, information level: `error.type` values such as
  `BadRequestError` on failed `chat` spans and on the `sdk` root, which the
  conventions allow, and `gen_ai.provider.name` `unknown` on the `sdk` root, a
  session that used both OpenAI and Anthropic.

## Keys outside the registry

The same 68 keys before and after, against either registry:

- `agent.errors`, `agent.llm_requests`, `agent.session_id`, `agent.tool_calls`
- 34 `agent.strace.*` keys:
  `behavior.{failed_operation_count,recovered_operation_count,retry_count,unresolved_failed_operation_count}`,
  `cost.total_usd`, `depth`,
  `event.{event_id,previous_hash,provenance,redacted}`,
  `evidence.{gap_codes,gap_count,gaps,schema_version,status}`,
  `finish_reason`, `last_model_request_failed`, `link.type`,
  `parent_event_id`, `parent_session_id`, `provenance`, `provider`,
  `redacted`, `redacted_event_count`,
  `request.{event_id,previous_hash,provenance,redacted}`,
  `response.{event_id,parent_event_id,previous_hash,provenance,redacted}`,
  `session_status`
- `exception.escaped`
- `gen_ai.completion`, `gen_ai.prompt`, `gen_ai.system`
- `gen_ai.tool.input.{auth_header,command,description,path,prompt,subagent_type}`,
  `gen_ai.tool.output`
- OpenInference: `input.mime_type`, `input.value`, `output.mime_type`,
  `output.value`, `llm.cost.total`, `llm.finish_reason`,
  `llm.invocation_parameters`, `llm.model_name`, `llm.provider`,
  `llm.token_count.{completion,prompt,prompt_details.cache_read,prompt_details.cache_write,total}`,
  `metadata`, `openinference.span.kind`
- `service.version`, `session.id`, `tool.name`

They are kept for the backends and queries that read them.
[docs/production.md](../../docs/production.md#keys-outside-the-genai-registry)
says why each is there.

## Content

- `gen_ai.input.messages` and `gen_ai.output.messages` are JSON strings. The
  `v0.1.0` registry says content MUST be recorded as an AnyValue. The
  conformance check parses a string value and checks it against the message
  schema, and agent-strace's messages pass in every scenario.
- The negative check proves the schema check runs. With each root's
  `gen_ai.input.messages` replaced by a message with no `parts`, Weaver
  reports `genai_content_schema` ×2, once on each `invoke_agent claude-code`
  root, with `"parts" is a required property`. All the other counts match
  the `session` scenario.
- The check does not read `gen_ai.tool.call.arguments` or
  `gen_ai.tool.call.result`, so it could not catch the next two findings.
  They came from reading the code, and the unit tests in
  `tests/test_otlp_genai.py` cover them.
  - `gen_ai.tool.call.result` was `output.value` cut at 500 characters, so a
    longer JSON result was invalid JSON. It is now cut with its structure
    kept, in the same 500 characters.
  - The hooks store a tool result that an agent reports as an object as its
    Python `repr`, such as `{'type': 'text'}`, which is not JSON. It now
    becomes a JSON string in `gen_ai.tool.call.result`. The stored event is
    unchanged.

## Span links

The `retry_of` links survive the OTLP/gRPC shim: Weaver received 2 links in
`session` (the model request retry and the tool retry), 2 in `tree`, and 2 in
the negative check. `sdk` has no retries, so no links.
