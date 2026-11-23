# Azure Monitor exporter results (#586)

Observed on 2026-11-23 with otelcol-contrib 0.165.0, using [run.py](run.py) as described in [README.md](README.md). Every number below is in [report.json](report.json) unless the line says how else it was checked. What the exporter does with a span was also read in its source at v0.165.0 (`trace_to_envelope.go`, `azuremonitor_exporter.go`, `transmitter.go`) and in `microsoft/ApplicationInsights-Go` v0.4.4 (`contracts/*.go`).

**Neither Application Insights nor Foundry was tested.** These are the envelopes the exporter posts to its `IngestionEndpoint`, here a local capture server. Whether Application Insights stores them as shown, and whether Foundry then lists the traces under a registered external agent, was not checked.

**Result:** with `--agent-id` (or `AGENT_STRACE_AGENT_ID`), `gen_ai.agent.id` reaches the exporter's output as a property on every span's envelope: the root, each tool and model span, and the subagent's spans in an `--include-subagents` tree, in both `otlp-genai` and classic `otlp`, and from `watch --stream-otlp`. Without it, no envelope has the property. Every span becomes a dependency, never a request. Array attributes are dropped, and a property longer than 8192 bytes is cut, which leaves `gen_ai.input.messages` of a long session as invalid JSON.

## Transport

| Item | Observed |
| --- | --- |
| URL | `POST {IngestionEndpoint}/v2.1/track` |
| Body | newline-delimited JSON envelopes, `Content-Type: application/x-json-stream`, `Content-Encoding: gzip` |
| Posts | 14 across both exporters; the largest was 4910 bytes gzipped, 67168 unzipped |
| Envelopes, default exporter | 42: 38 `RemoteDependencyData`, 4 `ExceptionData` |
| Envelopes, `spaneventsenabled: true` | 83: the same 42 plus 41 `MessageData` |
| Envelope tags | `ai.operation.id` (the trace ID), `ai.operation.parentId` (the parent span ID, empty on a root), `ai.cloud.role` (`service.name`, here `agent-trace`), `ai.application.ver` (`service.version`), `ai.internal.sdkVersion` |
| CLI | every `export` exited 0 with `Exported N events to <otlp-receiver> (HTTP 200)`; `watch` exited 0 |

The collector logged two warnings, both from the long session: `RemoteDependencyData.Properties has value with length exceeding max of 8192: gen_ai.input.messages`, once per exporter. With the older exporter name `azuremonitor` it also warns that the alias is deprecated in favour of `azure_monitor`; `collector.yaml` uses the new name.

## Spans

Every span sent has exactly one envelope, matched by trace ID and span ID, and each is `RemoteDependencyData` (`Microsoft.ApplicationInsights.RemoteDependency`):

| Export | Span | Kind | Count | Dependency `type` |
| --- | --- | --- | ---: | --- |
| `otlp-genai` tree | `invoke_agent` | INTERNAL | 2 | `InProc` |
| | `execute_tool` | INTERNAL | 5 | `InProc` |
| | `chat` | CLIENT | 3 | empty |
| `otlp` tree | `agent-session` | INTERNAL | 2 | `InProc` |
| | `tool/*` | CLIENT | 5 | empty |
| `otlp-genai` long session | `invoke_agent`, `execute_tool` | INTERNAL | 2 | `InProc` |

The exporter makes a request only from a SERVER or CONSUMER span, and agent-strace sends neither, so no root becomes a request. A CLIENT or PRODUCER span becomes a dependency typed from its `http.*`, `rpc.*`, `db.*` or `messaging.*` attributes; agent-strace spans have none, so the type is empty and `target` and `data` are empty too. Version 0.165.0 has no handling for `gen_ai.*` attributes.

Name, duration, `ai.operation.id` and `ai.operation.parentId` follow the span, so the tree is kept. The failed `chat` and `execute_tool bash` spans have `success: false` and `resultCode` `"2"` (the OTLP status code); others have `success: true` and `"0"`.

## Attributes

String, boolean, integer and double attributes become `properties` with their value as a string: `"true"`, `"4096"`. Arrays are dropped, with no warning: `gen_ai.response.finish_reasons` on the 3 `chat` spans of the tree. Nothing else was dropped. The kept keys include:

- `gen_ai.agent.id`, `gen_ai.agent.name`, `gen_ai.conversation.id`, `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.request.max_tokens`;
- `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.usage.cache_read.input_tokens`, `gen_ai.usage.cache_write.input_tokens`;
- `gen_ai.input.messages`, `gen_ai.output.messages`, `gen_ai.tool.name`, `gen_ai.tool.call.id`, `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result`;
- every `agent.strace.*` key sent, including `agent.strace.redacted`, `agent.strace.evidence.*` and `agent.strace.cost.total_usd`, and the OpenInference keys (`openinference.span.kind`, `input.value`, `llm.token_count.*`).

The exporter adds properties that were not span attributes: `otel.status_code` (`STATUS_CODE_ERROR`, `STATUS_CODE_UNSET`) and `otel.status_description` (the status message), the resource attributes (`service.name`, `service.version`, `agent.session_id`), `instrumentationlibrary.name` and `.version`, and `_MS.links` (below).

A resource attribute overwrites a span attribute with the same key. In an `--include-subagents` tree the resource is the parent's, so the child's root, which has its own `agent.session_id` span attribute, is stored with the parent's: `am586-genai-child` became `am586-genai-parent`. The child's `gen_ai.agent.name`, `session.id` and `agent.strace.parent_session_id` were kept.

## gen_ai.agent.id

| Export | Span envelopes with `gen_ai.agent.id` | Value |
| --- | --- | --- |
| `otlp-genai --include-subagents --agent-id demo-agent-v1` | 10 of 10 (parent 7, child 3) | `demo-agent-v1` |
| `otlp --include-subagents` with `AGENT_STRACE_AGENT_ID=demo-agent-v1` | 7 of 7 | `demo-agent-v1` |
| `otlp-genai --agent-id demo-agent-v1`, long session | 2 of 2 | `demo-agent-v1` |
| `watch --stream-otlp --agent-id demo-agent-v1` | 9 of 9 | `demo-agent-v1` |
| `otlp-genai --include-subagents`, no ID | 0 of 10 | |

The envelopes made from span events don't have it. An exception or message envelope gets the event's own attributes and the resource and scope properties, not the span's, so none of the 4 `ExceptionData` and 41 `MessageData` envelopes has `gen_ai.agent.id` or `gen_ai.conversation.id`. Each has `ai.operation.parentId` set to its span, which does have them.

## Links

Each retry link becomes the `_MS.links` property, a JSON array `[{"operation_Id": "<trace ID>", "id": "<span ID>"}]`. All 5 link targets (the `otlp-genai` model and `bash` retries, twice, and the `otlp` `bash` retry) matched what was sent. The link's attributes, here `agent.strace.link.type`, are not carried.

## Span events

| Event | Default | `spaneventsenabled: true` |
| --- | --- | --- |
| `exception` | `ExceptionData`: `typeName` from `exception.type`, `message` from `exception.message`, `severityLevel` 3, empty stack | the same |
| `gen_ai.user.message`, `gen_ai.assistant.message`, `gen_ai.system.message` | dropped | `MessageData` with the event name as `message` and the content in properties (for example `gen_ai.prompt`) |
| classic `user_prompt`, `assistant_response`, `decision` | dropped | `MessageData`, the same way |

So by default the conversation reaches Application Insights only through the `gen_ai.input.messages` and `gen_ai.output.messages` span attributes of `otlp-genai`.

## Content size limits

The exporter cuts each property value to 8192 bytes and each key to 150 bytes before sending (`Sanitize` in ApplicationInsights-Go). Other limits from the same source, also in bytes: a dependency name 1024, its `data` 8192, a message 32768.

agent-strace already limits what it puts in a span: 1000 characters of each message text, 1000 of tool arguments and 500 of a tool result. So only the message lists, which hold every turn, pass 8192 bytes. In the long session the root's `gen_ai.input.messages` was 69691 bytes holding 19 messages (each `é` is escaped as the six bytes `\u00e9`). The exporter stored its first 8192 bytes: a prefix of what was sent, holding the start of 3 messages, and not valid JSON. The 5000-character `Bash` command arrived already cut by agent-strace and was stored unchanged. The JSON is ASCII, so the byte cut can't split a character.

## watch --stream-otlp

`watch` sends the root span again with each batch. The exporter forwarded every copy: 9 dependency envelopes for 7 distinct spans, 3 of them the root with the same span ID. All 9 had `gen_ai.agent.id`. Whether Application Insights keeps one row or three for the root was not checked.

## Not checked

- Application Insights itself: what it stores, the `customDimensions` a query sees, ingestion limits beyond the exporter's own, sampling, and duplicate handling.
- Foundry: registering an external agent, its trace view, and trace-based or conversation evaluation. Microsoft's page says Foundry matches traces by `gen_ai.agent.id` and that conversation evaluation needs a stable `gen_ai.conversation.id` and message attributes on `invoke_agent` spans; that was read, not tested.
- The Azure Monitor OpenTelemetry distros, which map spans in their own code, not this exporter's.
- OTLP/HTTP JSON into the collector; only `--protocol http/protobuf` was sent.
- Other collector versions or platforms.
