# Phoenix OTLP/HTTP protobuf results (#560)

Observed on 2026-11-23 against a local Phoenix 22.0.1 with a fresh working directory, using [run.py](run.py) as described in [README.md](README.md). Every number below is in [report.json](report.json) unless the line says how else it was checked. Phoenix was read through its REST span API and its GraphQL API; its UI was not opened.

**Result:** with `--protocol http/protobuf`, export and `watch --stream-otlp` reach Phoenix and every span is stored with its parent, name, status and times. Without it, Phoenix answers 415 and the new hint names the fix.

## Transport

| Command | Result |
| --- | --- |
| `export px560-genai-parent --format otlp-genai --endpoint …` (default JSON) | Exit 1, nothing stored. stderr: `OTLP export to http://127.0.0.1:6016 failed: HTTP Error 415: Unsupported Media Type`, then `  Unsupported content type: application/json`, then `  This collector does not accept OTLP JSON; send protobuf with --protocol http/protobuf` |
| The same with `--protocol http/protobuf`, parent and child | Exit 0: `Exported 18 events … (HTTP 200)` and `Exported 8 events … (HTTP 200)` |
| `export px560-otlp-{parent,child} --format otlp --endpoint …` with `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf` | Exit 0, HTTP 200 for both |
| `export … --protocol grpc` | Exit 1, nothing sent: `--protocol grpc is not supported: agent-strace sends OTLP over HTTP only; use http/protobuf or http/json` |
| `export px560-tree-parent --format otlp-genai --include-subagents --endpoint … --protocol http/protobuf` | Exit 0: `Exported 26 events … (HTTP 200)` |
| `watch --stream-otlp --protocol http/protobuf` | Exit 0. 3 POSTs to `/v1/traces`, `Content-Type: application/x-protobuf`, all HTTP 200 |
| `watch --stream-otlp` (default JSON) | Exit 0. 3 POSTs, each answered 415 and each reported on stderr; the hint `[stream] This collector does not accept OTLP JSON; send protobuf with --protocol http/protobuf` printed once, after the first. Nothing stored |

## Spans

| Group | Spans sent | Spans stored | Traces |
| --- | ---: | ---: | ---: |
| `otlp-genai` (parent and child exported separately) | 10 | 10 | 2 |
| `otlp` (parent and child exported separately) | 7 | 7 | 2 |
| `otlp-genai --include-subagents` | 10 | 10 | 1 |
| `watch --stream-otlp` protobuf | 9 in 3 POSTs (7 distinct) | 7 | 1 |

For every exported span, Phoenix's name, trace ID, parent span ID, start and end (to the microsecond: Phoenix rounds the nanoseconds), status code and status message equal what was sent. In the `--include-subagents` trace the child's root is under the parent's `execute_tool Task` span.

### Span kinds

- `otlp-genai` and the tree: Phoenix's span kind is the `openinference.span.kind` that was sent, for all 10 spans: `AGENT` 2, `LLM` 3, `TOOL` 5.
- `otlp`: all 7 spans are `UNKNOWN`. Classic spans carry no `openinference.span.kind`.

### Attributes

- `otlp-genai`: 71 distinct attribute keys were sent. Phoenix stored every one with the value sent, except two it rewrites: it removes `openinference.span.kind` (it becomes the span kind) and splits the JSON string `metadata` into `metadata.*` attributes. The kept keys include `agent.strace.redacted`, `agent.strace.redacted_event_count`, `agent.strace.provenance`, `agent.strace.{request,response}.{provenance,redacted,previous_hash}`, `agent.strace.evidence.{status,gap_count,gap_codes,gaps,schema_version}`, `agent.strace.cost.total_usd`, `llm.cost.total`, `gen_ai.conversation.id`, `gen_ai.input.messages` and `gen_ai.output.messages`.
- Phoenix added attributes that were not sent: `llm.system` (5 spans), `llm.provider` (2), `tool.id` (5), `llm.finish_reason` (1), `tool.parameters.*` on 4 tool spans, parsed from `gen_ai.tool.call.arguments`, and `llm.input_messages.0.message.{role,content}` on the 2 `AGENT` roots, from `gen_ai.input.messages`. A redacted argument stays redacted: `tool.parameters.auth_header` is `[REDACTED:bearer]`.
- `otlp`: all 21 sent keys were stored unchanged. Phoenix added `llm.system` and `llm.provider` (2 spans each) and `session.id` on the parent root, from `gen_ai.conversation.id`.

### Events

- The failed `bash` tool span's `exception` event kept `exception.type`, `exception.message` and `exception.escaped`, and lost the 4 other attributes sent on it: `agent.strace.response.event_id`, `.parent_event_id`, `.previous_hash` and `.redacted`. The same attributes stay on the span itself.
- The root spans' `gen_ai.user.message` and `gen_ai.assistant.message` events kept every attribute.

### Links

Retry links were sent: 2 in `otlp-genai`, 1 in `otlp`, 2 in the tree, each with `agent.strace.link.type`. Phoenix accepted them (HTTP 200) but did not store them: its REST span API and its GraphQL `Span` type have no links field (`graphql_span_has_links_field: false`), and, checked by hand with `sqlite3 "$PHOENIX_DIR/phoenix.db"`, its `spans` table has no links column and no table holds links.

### Tokens and cost

| Span | Prompt | Completion | Total | Sent `llm.cost.total` | Phoenix cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Parent `chat claude-sonnet-4-5` | 12000 | 1500 | 13500 | 0.03765 | 0.03765 |
| Child `chat claude-sonnet-4-5` | 2400 | 300 | 2700 | 0.01234 | 0.0117 |

Phoenix stored the token counts as sent, including the parent request's 8000 cache-read and 1000 cache-write tokens. Its cost is not read from `llm.cost.total`: for the child request it is 0.0117, not the recorded 0.01234. Both Phoenix costs equal Claude Sonnet 4.5 list prices applied to the stored tokens: $3 per million uncached input, $0.30 cache read, $3.75 cache write and $15 output. The parent works out to 3000 × 3 + 8000 × 0.30 + 1000 × 3.75 + 1500 × 15 = 37650 per million, i.e. 0.03765, which is also what the synthetic session recorded. `--format otlp` spans have no token counts or cost in Phoenix.

### Input and output

Phoenix shows no input or output for the `AGENT` and `LLM` spans, which carry no `input.value` or `output.value`. Of the 5 `TOOL` spans, 3 have both, 1 only input (the call with no result) and 1 only output (the orphan result), as sent.

### Sessions

Phoenix's sessions follow the root span of each trace:

- Exported separately, `px560-genai-parent` (cost 0.03765, 13500 tokens) and `px560-genai-child` (0.0117, 2700) are two sessions.
- With `--include-subagents`, Phoenix lists one session, `px560-tree-parent` (0.04935, 16200), holding both. The child's spans keep `session.id` `px560-tree-child`, but Phoenix lists no session by that name.
- `px560-watch-parent`: 0.03765, 13500.
- `otlp`: one session, `synthetic-conversation-px560-otlp`, with no cost and 0 tokens. The child root has no `session.id`, and its trace is in no session.

## watch --stream-otlp

- The 3 protobuf POSTs carried 1, 5 and 3 spans. Each carried the session root again, with the same span ID. Phoenix stored one root: the copy from the first POST, matched by its times and attributes. It ignored the later copies.
- The streamed root is `invoke_agent agent-trace`, with `agent.tool_calls`, `agent.llm_requests` and `agent.errors` 0 in every POST. watch builds it from a minimal session record (`_flush_otlp` in `watch.py`), as before #560, which changes only the encoding.
- Stored: 7 spans in 1 trace. Kinds `AGENT` 1, `LLM` 2, `TOOL` 4. Names `invoke_agent agent-trace` 1, `chat claude-sonnet-4-5` 2, `execute_tool bash` 3, `execute_tool Task` 1. Status `ERROR` 2, `UNSET` 5.

## Not checked

- Phoenix's UI.
- `OTEL_EXPORTER_OTLP_PROTOCOL` against Phoenix; its precedence is covered by unit tests.
- `--format temporal` with protobuf against Phoenix; unit tests cover it.
- Phoenix projects: every span went to the `default` project.
- Phoenix with authentication on, so `--header` was not sent to it.
- Where Phoenix gets its prices; only the arithmetic above was checked.
- Phoenix versions other than 22.0.1, and gRPC (out of scope).
