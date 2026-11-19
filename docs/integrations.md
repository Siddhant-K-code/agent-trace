# Auto-instrumentation

Instrument any supported agent framework without modifying application code.

---

## Quick start

```bash
# Instrument a specific framework
agent-strace auto --framework langchain -- python my_agent.py

# Auto-detect all installed frameworks
agent-strace auto --detect -- python my_agent.py

# Modules, console scripts, and shebang scripts work the same way
agent-strace auto --detect -- python -m my_package
agent-strace auto --detect -- my-agent-cli --some-flag
```

[How `agent-strace auto` works](#how-agent-strace-auto-works) describes what it changes in the command's environment and its limits.

Without the CLI wrapper, set `AGENT_STRACE_AUTO_INSTRUMENT` and import `agent_trace.auto` in your program before it imports any framework. The variable is read only when that module is imported, so setting it on its own instruments nothing.

```python
# my_agent.py: the first import, before any framework import
import agent_trace.auto
```

```bash
AGENT_STRACE_AUTO_INSTRUMENT=langchain,litellm python my_agent.py
```

`AGENT_STRACE_AUTO_INSTRUMENT` takes a comma-separated list of `openai-agents`, `langchain` (or `langgraph`), `litellm`, `anthropic`, `openai`, `strands`, and `crewai`, or `detect` to instrument every installed framework.

Or call an instrument function directly in code:

```python
from agent_trace.integrations import instrument_langchain
instrument_langchain()
```

---

## Supported frameworks

| Framework | Install | What's traced |
|---|---|---|
| OpenAI Agents SDK | `pip install agent-strace[openai-agents]` | `Runner.run_sync`, function tool calls (from tracing spans) |
| LangChain / LangGraph | `pip install agent-strace[langchain]` | `BaseTool.run` and `BaseTool.arun`, chat model `_generate` and `_agenerate`, node runs of LangGraph graphs compiled after instrumenting |
| CrewAI | `pip install agent-strace[crewai]` | `Crew.kickoff`, `Agent.execute_task`, `Task.execute_sync` |
| LiteLLM | `pip install agent-strace[litellm]` | `litellm.completion`, `litellm.acompletion` |
| Anthropic SDK | `pip install anthropic` | `messages.create` (sync and async clients, streams) |
| OpenAI SDK | `pip install openai` | `chat.completions.create`, `responses.create` (sync and async clients, streams) |
| AWS Strands | `pip install agent-strace[strands]` | `Agent.__call__` and `invoke_async`, tool calls (from tool hooks) |

Install all integrations at once:

```bash
pip install agent-strace[all-integrations]
```

Each integration is an optional extra. The core package stays dependency-free. See [ADR-0003](../ADRs/0003-zero-runtime-dependencies.md).

When a model call raises, the OpenAI, Anthropic, LiteLLM, LangChain chat model and CrewAI `Agent.execute_task` integrations record an `error` event whose `parent_id` names the call's `llm_request`, so the exports report that call as failed. See [Failed and cancelled model requests](production.md#failed-and-cancelled-model-requests).

### Sessions

The CrewAI, OpenAI Agents SDK and Strands integrations record each run as its own session: each `Crew.kickoff`, `Runner.run_sync`, or Strands `Agent.__call__` or `invoke_async`. The session starts with `session_start` and ends with `session_end`, which is also recorded when the run raises, after an `error` event for the exception. When the run ends, the session's `ended_at`, total duration and its `tool_calls`, `llm_requests` and `errors` counts are set, so `agent-strace list` shows them and retention treats the session as finished. The counts are taken from the session's stored events, so they include events other processes recorded into it; with `AGENT_STRACE_ENDPOINT` set, only the run's own process's events are counted. A run started inside another run of the same integration, in the same thread or task, records into the outer run's session.

Events from a thread or task that is not inside a run, such as a thread the program started itself, go to the session named by the `.active-session.<name>` marker in the trace directory. Runs that overlap, in one process or in several processes sharing the trace directory, are kept in a locked registry (`.integration-runs.<name>`), and the marker names the newest run still in progress. When a run ends, the marker moves back to the newest run still going, so runs may finish in any order. A run is dropped from the registry, and the marker no longer names it, once its process has exited without ending it; each run records its process's start time when the platform reports it (Linux, macOS and Windows do), so a new process that reuses the PID does not keep it alive. A new run is written to the registry before the marker names it. When no run is left, the marker and registry are removed, and the next event outside a run starts a new session. Such events can only be told apart by run when the thread or task is inside the run, so with overlapping runs they go to the newest one. `session_end` is always the last event of a run: the end is recorded and the session closed under a lock (`.integration-seal.lock`) that every integration event append also takes, and an event for a session that has already ended, such as one a process looked up before the run finished, goes to the agent's current session instead. This holds with `AGENT_STRACE_ENDPOINT` set too. A moved event keeps the session it was recorded for in `data.original_session_id`; a moved `tool_result` or `error` does not name its `tool_call` in `parent_id`, which would point into the ended session, but in `data.original_parent_id`.

Events recorded outside a run, such as a LangChain tool call or a model call through the OpenAI, Anthropic or LiteLLM integration, go to one session per integration, which stays open while the program records into it.

Each tool `tool_result`, and each `error` a tool raises, names its `tool_call` in its `parent_id`, so the exports pair them.

### Model calls

The OpenAI, Anthropic, LiteLLM and LangChain integrations record each model call as an `llm_request`, then an `llm_response`, or an `error` if the call raises. The response or error names the request in its `parent_id`, so concurrent calls pair correctly.

- Sync and async calls are both recorded. The OpenAI and Anthropic integrations patch the SDK resource classes, so clients created before instrumenting are covered too.
- With `stream=True`, the OpenAI and Anthropic integrations record the response when the caller reaches the end of the stream. The stream the caller gets is unchanged. Token counts come from the last chunk of a Chat Completions stream when the request sets `stream_options={"include_usage": True}` (without it, none are recorded), from the final `Response` of a Responses API stream, and from the `message_start` and `message_delta` events of an Anthropic stream. A stream closed or dropped before its end records only its `llm_request`. Cancelling the task that reads an async stream, or awaits an async call, records a response with `"status": "cancelled"`. Any other `BaseException` that ends a model call or a stream read, such as `KeyboardInterrupt` or `SystemExit`, also records a cancelled response, so the `llm_request` is closed even if the program catches it and goes on.
- Token counts include cache reads and writes when the SDK reports them, in shapes [`agent-strace cost`](commands.md#cost) reads. The Anthropic SDK integration records Anthropic's `cache_read_input_tokens`, `cache_creation_input_tokens` and `cache_creation` fields. The OpenAI, LiteLLM and LangChain integrations record `cached_input_tokens` and `cache_write_input_tokens` beside an `input_tokens` that includes them, as those libraries count it. The 1-hour and 12-hour cache write split is not kept in that shape, so those writes are priced at the shorter lifetime's rate.
- A Responses API response also records its `status`, and the `error` of a `failed` response, because the API returns those instead of raising.
- A `with_raw_response` or `with_streaming_response` call to the OpenAI or Anthropic SDK records its response when the caller parses the body with `parse()`. A body that is never parsed, or is read as bytes, records only the `llm_request`.
- A call that passes through two instrumented layers, such as LangChain or LiteLLM calling the OpenAI SDK, is recorded once, by the outer layer.
- A failure to record a call never reaches the caller.

Not recorded: OpenAI `chat.completions.parse` and `responses.parse`; Anthropic `messages.stream`, `messages.parse` and `client.beta.messages`; LangChain `stream` and `astream` output, `streaming=True` models, and responses served from the LangChain LLM cache; token counts of LiteLLM streams. When LangChain streams through an instrumented OpenAI or Anthropic SDK, that SDK call is still recorded.

---

## How `agent-strace auto` works

`agent-strace auto [--framework NAME | --detect] -- <command>` runs `<command>` with two changes to its environment:

- `AGENT_STRACE_AUTO_INSTRUMENT` is set to the `--framework` value (one name or a comma-separated list), or to `detect` when `--detect` is given or no framework is named.
- A private temporary directory holding a `sitecustomize.py` is put first on `PYTHONPATH`. Existing `PYTHONPATH` entries are kept after it. The directory is deleted when the command exits, including when agent-strace is stopped by SIGTERM, SIGINT or SIGHUP on Linux and macOS: it passes the signal on to the command's process group and deletes the directory once every process in that group has exited ([commands.md](commands.md#auto)). SIGKILL cannot be caught, so it leaves the directory behind.

Python imports `sitecustomize` while it starts, before any user code. So every Python interpreter the command starts is instrumented: `python my_agent.py`, `python -m my_package`, console scripts and shebang scripts that run Python, and Python subprocesses those start while the command runs. In each interpreter the hook:

1. Runs the `sitecustomize` module it shadows, if there is one (for example a venv's or a Linux distribution's), so existing start-up code still runs.
2. Imports `agent_trace.auto`, which instruments the frameworks named in `AGENT_STRACE_AUTO_INSTRUMENT`. If the interpreter has agent-strace installed, that copy is used. Otherwise the package is loaded from the agent-strace installation that ran `auto`, without adding that directory to `sys.path`, so the interpreter's own packages are never shadowed. A venv Python therefore needs no extra install. agent-strace supports Python 3.10 and newer; an older interpreter may fail to load it.
3. If `agent_trace` cannot be imported, prints one `[agent-strace] auto-instrumentation disabled ...` line on stderr and lets the program run uninstrumented.

`agent-strace auto` exits with the command's exit code, or 128 plus the signal number when a signal killed the command or stopped agent-strace ([commands.md](commands.md#auto)). Events go to `$AGENT_TRACE_DIR`, or to `.agent-traces` in the program's working directory. `agent-strace --trace-dir DIR auto -- <command>` sets `AGENT_TRACE_DIR` for the command.

Limits:

- Interpreters started with `-E`, `-I`, or `-S`, and embedded interpreters that ignore `PYTHONPATH`, are not instrumented.
- Frameworks are instrumented at start-up, so they must be importable then. A framework the program only makes importable later, for example by editing `sys.path`, is not instrumented; call its `instrument_*` function in code instead.
- A non-Python command is not instrumented itself, but the Python processes it starts are.

---

## Agent CLI hooks

Use setup-generated hooks when the agent CLI has its own lifecycle hook system.

| CLI | Setup | What's traced |
|---|---|---|
| Claude Code | [Native plugin](claude-code-plugin.md) or `agent-strace setup --cli claude` | Session start/end, user prompts, assistant responses, tool calls/results |
| OpenAI Codex | `agent-strace setup --cli codex` | Session start/end, user prompts, assistant responses, `PreToolUse`/`PostToolUse` tools, subagent starts/stops recorded in the parent session, with the subagent's tool calls and final message attributed to it, and per-request token usage from `PostModelRequest` (Codex 0.164.0 and later). Re-run setup after upgrading and trust the new hooks in `/hooks`. See the [capture matrix](capture-matrix.md#openai-codex) |
| Gemini CLI | `agent-strace setup --cli gemini` | Session start/end, prompts, assistant responses, `BeforeTool`/`AfterTool` tools |
| Cursor | `agent-strace setup --cli cursor` | Session start/end, prompts, shell execution, file edits, assistant responses when emitted by Cursor hooks |
| GitHub Copilot CLI | `agent-strace setup --cli copilot` | Session start/end, prompts, hook-visible tool calls/results and failures, stop markers (no response text), and `task` subagent starts/stops, with subagent sessions linked to their parent. Re-run setup after upgrading to register new hooks. See the [capture matrix](capture-matrix.md#github-copilot-cli) |
| GitHub Copilot Desktop | Wrap each local MCP server with `agent-strace record -- ...` | MCP initialize/tool call/tool result traffic for wrapped servers only; full chat transcripts and non-MCP activity are not exposed |

All paths write the same event stream under `.agent-traces/`, so replay, timeline, explain, why, watch, export, and audit commands work the same way after capture.

For Copilot Desktop, use a shared absolute trace directory such as `/Users/alice/.agent-strace/traces`, configure `agent-strace --trace-dir <dir> mcp` as an optional trace-reader MCP server, and wrap each tool-providing MCP server with `agent-strace --trace-dir <dir> record --name <label> -- <original-command> <args>`. See [setup.md](setup.md#github-copilot-desktop) for the full Local server field values and validation commands.

### Remote MCP servers over HTTPS

`agent-strace record-http --url https://...` proxies a remote HTTP/SSE MCP server. The proxy verifies the server certificate and host name against the system CA store. To trust a private CA, pass `--ca-file /path/to/ca.pem`; the bundle replaces the system CA store for the upstream connection. Alternatively, set `SSL_CERT_FILE=/path/to/ca.pem` for the proxy process; this replaces OpenSSL's default CA file for the whole process. There is no flag to turn verification off. A failed TLS connection returns 502 to the agent and records an `error` event. Ctrl-C also stops a TLS handshake that the server never completes. See [`record-http`](commands.md#record-http).

---

## OpenAI Agents SDK

```python
from agent_trace.integrations import instrument_openai_agents
instrument_openai_agents()

# Now use the SDK normally
from agents import Agent, Runner
agent = Agent(name="my-agent", instructions="...")
result = Runner.run_sync(agent, "Do the task")
```

Traces: `Runner.run_sync` (one session per run), and function tool calls. The SDK runs a function tool by awaiting its `on_invoke_tool` inside a function span, so the integration adds a trace processor with `agents.tracing.add_trace_processor` and records each function span as a `tool_call` and its `tool_result`, or an `error` when the span has one. Tool calls are therefore recorded only while the SDK's tracing is enabled (not with `set_tracing_disabled(True)`, `RunConfig(tracing_disabled=True)` or `OPENAI_AGENTS_DISABLE_TRACING=1`), and their arguments only when the run includes sensitive data in traces (the default). The SDK sets a function span's input only after the span starts, so the `tool_call` is written when the tool starts, with empty `arguments`, and the arguments are on its `tool_result` or `error`. A tool that never returns still leaves its `tool_call`. `Runner.run` and `Runner.run_streamed` do not start a session, but their tool calls are recorded; see [Sessions](#sessions).

---

## LangChain / LangGraph

```python
from agent_trace.integrations import instrument_langchain
instrument_langchain()

# Now use LangChain normally
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
```

Traces: `BaseTool.run` and `BaseTool.arun`, which `invoke`, `ainvoke` and every tool class (including `StructuredTool`) go through, and the `_generate` and `_agenerate` methods of each chat model class, so `invoke`, `ainvoke`, `batch` and `abatch` are recorded with the token counts in the message's `usage_metadata`. A model class is patched the first time one of its models is called. `stream` and `astream` are not recorded; see [Model calls](#model-calls).

LangGraph nodes are traced in graphs compiled after `instrument_langchain()`. A graph compiled before it is not traced. Each node run records a `decision` naming the node, a `tool_call` named `langgraph.node.<name>`, then a `tool_result` with the run's `duration_ms`, or an `error` if the node raises. `interrupt()`, `NodeInterrupt` and `ParentCommand` stop a node on purpose, so they record a `tool_result` with `control_flow` set to the exception's class name, such as `GraphInterrupt`, not an `error`. A retried node records each attempt, and an interrupted node records again when it resumes. Node input and output are not recorded. The graph runs, streams and draws as it does untraced.

A node is traced only in LangGraph's `PregelNode` shape: it has a `copy` method and its `bound` is a LangChain `Runnable`. A node of another shape is left untraced, and so is a node whose runnable is a LangChain sequence, such as `prompt | model`. The functional API (`@entrypoint`, `@task`) is not traced. This was checked with LangGraph 1.2.23 and langchain-core 1.8.3 on Python 3.14, and with LangGraph 0.2.76 and langchain-core 0.3.86 on Python 3.10.

---

## CrewAI

```python
from agent_trace.integrations import instrument_crewai
instrument_crewai()

# Now use CrewAI normally
from crewai import Crew, Agent, Task
crew = Crew(agents=[...], tasks=[...])
result = crew.kickoff()
```

Traces: `Crew.kickoff` (one session per kickoff), `Agent.execute_task` (LLM request/response), `Task.execute_sync` (tool call/result).

---

## LiteLLM

```python
from agent_trace.integrations import instrument_litellm
instrument_litellm()

import litellm
response = litellm.completion(model="gpt-4o", messages=[...])
```

Traces: `litellm.completion`, `litellm.acompletion`. Code that ran `from litellm import completion` before instrumenting keeps the unpatched function. A stream is recorded when the call returns it, without token counts.

---

## Anthropic SDK

```python
from agent_trace.integrations import instrument_anthropic
instrument_anthropic()

import anthropic
client = anthropic.Anthropic()
message = client.messages.create(model="claude-3-5-sonnet-20241022", ...)
```

Traces: `messages.create` on `Anthropic` and `AsyncAnthropic` clients, including `stream=True`. `messages.stream` is not recorded; see [Model calls](#model-calls).

---

## OpenAI SDK

```python
from agent_trace.integrations import instrument_openai
instrument_openai()

from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(model="gpt-4o", messages=[...])
```

Traces: `chat.completions.create` and `responses.create` on `OpenAI` and `AsyncOpenAI` clients, including `stream=True` and the `chat.completions.stream` and `responses.stream` helpers, which call `create`. `parse` is not recorded; see [Model calls](#model-calls).

---

## AWS Strands

```python
from agent_trace.integrations import instrument_strands
instrument_strands()

from strands import Agent
agent = Agent(tools=[...])
result = agent("Do the task")
```

Traces: `Agent.__call__` and `Agent.invoke_async` (one session per call), and tool calls. The SDK's tool executor invokes the agent's `BeforeToolCallEvent` and `AfterToolCallEvent` hooks around each tool's `stream`, for `@tool` functions, `PythonAgentTool` and MCP tools alike, so the integration adds callbacks for them to each agent's hook registry, when the agent is created or on its first call. Each tool records a `tool_call` and its `tool_result`, or an `error` when the tool raised or returned an error result. Calling `stream_async` directly does not start a session.

---

## Cross-agent trace correlation (W3C traceparent)

When one agent calls another over HTTP, inject a `traceparent` header so both sessions share the same W3C trace ID. This links spans across agents in any OTLP backend (Jaeger, Tempo, Datadog, Honeycomb).

**Injecting (outbound call):**

```python
from agent_trace.propagation import inject_traceparent

headers = inject_traceparent({}, session_id="abc123", event_id="evt456")
# headers now contains:
#   "traceparent": "00-<trace-id>-<span-id>-01"
#   "tracestate":  "agent-trace=abc123"

response = requests.post("https://other-agent/run", headers=headers, json={...})
```

**Extracting (inbound call):**

```python
from agent_trace.propagation import extract_traceparent

ctx = extract_traceparent(request.headers)
if ctx:
    # ctx["version"]        2-hex traceparent version, such as "00"
    # ctx["trace_id"]       32-hex W3C trace ID from upstream
    # ctx["parent_id"]      16-hex parent span ID from the caller
    # ctx["flags"]          2-hex W3C trace flags, such as "01"
    # ctx["at_session_id"]  the caller's agent-trace session ID from
    #                       tracestate, or "" if there is none or it is
    #                       not a valid session ID
    pass
```

`extract_traceparent` returns `None` when there is no valid `traceparent` header.

Pass `trace_id` from the extracted context into `inject_traceparent` on any further outbound calls to propagate the same trace ID through the full call chain.

The W3C sampled flag is the lowest bit of `flags`, so `int(ctx["flags"], 16) & 1` is `1` when the caller marked the trace as sampled. agent-strace records a session whether or not this bit is set. `inject_traceparent` always sends `01`, which marks the trace as sampled.

The implementation follows [W3C Trace Context Level 1](https://www.w3.org/TR/trace-context/). No third-party dependencies required.

---

## OTLP GenAI interoperability

`agent-strace export --format otlp-genai` uses one OTLP/HTTP JSON payload for
Phoenix/OpenInference, Langfuse, Honeycomb, Datadog, and standard OTel
collectors. It does not install an OpenTelemetry SDK or add a gRPC path.

The payload carries both OpenTelemetry GenAI semantic convention attributes
(see [Spans and attributes](production.md#spans-and-attributes)) and additive
OpenInference attributes. Root, LLM, and tool spans map to `AGENT`, `LLM`, and
`TOOL`. Parent span IDs, exceptions, token counts (prompt cache reads and
writes included), recorded cost, redaction
state, persisted provenance, and observed structural evidence gaps remain
queryable where the backend indexes custom attributes. See
[Token usage and prompt caching](production.md#token-usage-and-prompt-caching).

Use the local fixtures before configuring credentials:

```bash
python3 -m json.tool examples/otlp/genai-conformance.json >/dev/null
python3 -m json.tool examples/otlp/openinference-mapping.json >/dev/null
```

Message arrays, provider blind spots, removed redacted values, and complete
hash-chain verification are not losslessly represented. See
[production tracing](production.md#openinference-compatibility) for the mapping
and backend recipes.

---

## Temporal workflows

Agent spans can be attached beneath an existing Temporal activity span without
adding the Temporal SDK to agent-strace. Expose the activity's W3C `traceparent`
to the agent process as `AGENT_STRACE_TEMPORAL_TRACE_PARENT` (preferred) or
`TEMPORAL_TRACE_PARENT`, then start the normal watcher:

```bash
export AGENT_STRACE_TEMPORAL_TRACE_PARENT="00-<trace-id>-<activity-span-id>-01"
agent-strace watch "$AGENT_SESSION_ID"
```

The watcher validates the value and stores the upstream trace ID, parent span
ID, and trace flags in additive session metadata. This makes later exports work
after the activity process exits:

```bash
# Write inspectable OTLP/JSON
agent-strace export "$AGENT_SESSION_ID" --format temporal --output temporal-trace.json

# Or send it to Jaeger, Tempo, Datadog, or another OTLP/HTTP collector
agent-strace export "$AGENT_SESSION_ID" --format temporal \
  --endpoint http://localhost:4318 \
  --header "Authorization: Bearer $OTLP_TOKEN"
```

The session root span reuses Temporal's trace ID and sets the Temporal activity
span as its parent; LLM and tool spans remain children of the session root. The
setup belongs in the activity worker or agent launcher, so Temporal workflow
definitions do not need to change. Export remains HTTP/JSON only and uses no
Temporal, OpenTelemetry, or gRPC runtime dependency.
