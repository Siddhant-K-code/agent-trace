# Send traces to Telemetry.dev

[Telemetry.dev](https://telemetry.dev) accepts OTLP/HTTP JSON traces, so `agent-strace export --format otlp-genai` can send a session there. This page records what one tested send showed. Telemetry.dev is not a default endpoint, and agent-strace has no Telemetry.dev dependency. Nothing is sent there unless you pass its endpoint.

## Limits as of 0.121.0

The result was **partial**. Read these before you rely on it.

- **Content is not in Telemetry.dev's content panels.** Prompts and completions appear only in a span's **Events** list, and tool input and output only in its **Attributes**. The Input, Output, Arguments and Result panels are empty; a model call's Output shows only its finish reason. [#552](https://github.com/Siddhant-K-code/agent-trace/issues/552)
- **Telemetry.dev's redaction can change ordinary values.** Telemetry.dev redacts common secrets before it stores a trace, and a match can occur inside an ordinary word.
  - In the test, the `sk-` in `task-` matched: an event ID such as `...-task-1` was stored as `...-ta[REDACTED]`.
  - This broke the ID links between a `Task` call and the subagent it started.
  - The stored copy can differ from what AgentTrace recorded. AgentTrace's local session stays the record to verify against.
- **A subagent is a separate trace.** `agent-strace export` sends each session as its own trace, so a subagent does not appear under the `Task` call that started it. [#552](https://github.com/Siddhant-K-code/agent-trace/issues/552)
- **Retries are not linked.** A retried model call or tool call is a second span with the same name. Telemetry.dev groups spans with the same name as "Repeated", which is not a retry link. [#552](https://github.com/Siddhant-K-code/agent-trace/issues/552)
- **Cost is Telemetry.dev's own.** Telemetry.dev prices the tokens itself. AgentTrace's estimate is kept only as the attribute `agent.strace.cost.total_usd`. A model Telemetry.dev cannot price has no cost.
- **A recovered session still shows as an error.** If any call failed, the session's root span is an error, even when a retry succeeded.
- **AgentTrace's evidence is not first-class.** Evidence health, provenance, redaction flags and capture gaps are plain span attributes, with no Telemetry.dev field, filter or badge. A capture gap shows as an ordinary tool span, which Telemetry.dev even shows as called by a model call. Whether these attributes can be searched was not checked.
- **Use `otlp-genai`, not `otlp`.** Classic `--format otlp` sends no model calls, tokens or cost, and its span for a failed tool call has no operation or tool name in Telemetry.dev.

## Send a session

Create a Telemetry.dev API key with **Send telemetry** access, then:

```bash
agent-strace export <session-id> --format otlp-genai \
  --endpoint https://api.telemetry.dev \
  --header "Authorization: Bearer $TELEMETRY_DEV_API_KEY"
```

- The exporter appends `/v1/traces` to `--endpoint`.
- `--header` takes `NAME: VALUE` and can be repeated.
- The key is on the command line, where `ps` and shell history can see it. Read it with `read -rs TELEMETRY_DEV_API_KEY` and `export` it, and `unset` it afterwards.
- The export sets no `deployment.environment.name`, so the traces go to the key's environment, or to `default`.
- `--service-name` sets the Service column (default `agent-trace`).
- Check your project's Privacy settings before you send real sessions. The test ran with Telemetry.dev's defaults on the free plan: prompt capture on, Issue checks on, 14-day retention.

To look at the payload first, leave out `--endpoint` and `--header`, and write it to a file:

```bash
agent-strace export <session-id> --format otlp-genai --output payload.json
```

## What maps

The test session was synthetic. It had:
- a subagent
- a failed and retried model call, and a failed and retried tool call
- tokens with prompt caching
- redacted content and a capture gap

Telemetry.dev showed it like this:

| Telemetry.dev | From the `otlp-genai` export |
| --- | --- |
| Trace | One trace per session, with every span under the session's `invoke_agent` root |
| Agent | The agent on the `invoke_agent` root, for example `claude-code` |
| Provider | The provider, on the root and on each `chat` span |
| Model | The model on each `chat` span. The root has none. |
| Tool | The tool name on each `execute_tool` span |
| Tokens | Input and output tokens per model call, summed for the trace. Cache reads and writes are priced separately. |
| Errors | Error status, error type and message on a failed model call and a failed tool call. In the test, the trace's error type was the failed model call's. |
| Session | The session's conversation ID, else its session ID |
| Service and Framework | `agent-trace` (or `--service-name`), and `otel` |
| Attributes | Every attribute AgentTrace sends, on each span checked. `gen_ai.response.finish_reasons` appears as `gen_ai.output.messages`. JSON values such as `agent.strace.provenance` are shown as expandable objects, and counts as numbers. |
| Events and Logs | Each span event with its attributes, including prompts and completions. Each event is also a log record linked to its span. A failed tool call's `exception` event is an `error` record. |

The test ran on 2026-11-22 with agent-strace 0.120.0. The exporter is unchanged in 0.121.0. It was observed through Telemetry.dev's CLI and web UI. The subagent-tree trace's attributes and attribute search were not checked. The payloads, the predictions and the results are in [experiments/telemetry-dev](../experiments/telemetry-dev/RESULTS.md).

See [production.md](production.md#per-backend-setup) for other backends.
