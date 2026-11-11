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
| OpenAI Agents SDK | `pip install agent-strace[openai-agents]` | `Runner.run`, `FunctionTool` calls |
| LangChain / LangGraph | `pip install agent-strace[langchain]` | `BaseTool._run`, chat model `_generate` and `_agenerate`, LangGraph nodes |
| CrewAI | `pip install agent-strace[crewai]` | `Crew.kickoff`, `Agent.execute_task`, `Task.execute_sync` |
| LiteLLM | `pip install agent-strace[litellm]` | `litellm.completion`, `litellm.acompletion` |
| Anthropic SDK | `pip install anthropic` | `messages.create` (sync and async clients, streams) |
| OpenAI SDK | `pip install openai` | `chat.completions.create`, `responses.create` (sync and async clients, streams) |
| AWS Strands | `pip install agent-strace[strands]` | `Agent.__call__`, `BaseTool.invoke` |

Install all integrations at once:

```bash
pip install agent-strace[all-integrations]
```

Each integration is an optional extra. The core package stays dependency-free. See [ADR-0003](../ADRs/0003-zero-runtime-dependencies.md).

When a model call raises, the OpenAI, Anthropic, LiteLLM, LangChain chat model and CrewAI `Agent.execute_task` integrations record an `error` event whose `parent_id` names the call's `llm_request`, so the exports report that call as failed. See [Failed and cancelled model requests](production.md#failed-and-cancelled-model-requests).

### Model calls

The OpenAI, Anthropic, LiteLLM and LangChain integrations record each model call as an `llm_request`, then an `llm_response`, or an `error` if the call raises.

- Sync and async calls are both recorded. The OpenAI and Anthropic integrations patch the SDK resource classes, so clients created before instrumenting are covered too.
- With `stream=True`, the OpenAI and Anthropic integrations record the response when the caller reaches the end of the stream. The stream the caller gets is unchanged. Token counts come from the last chunk of a Chat Completions stream when the request sets `stream_options={"include_usage": True}` (without it, none are recorded), from the final `Response` of a Responses API stream, and from the `message_start` and `message_delta` events of an Anthropic stream. A stream closed or dropped before its end records only its `llm_request`. Cancelling the task that reads an async stream, or awaits an async call, records a response with `"status": "cancelled"`.
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
- A private temporary directory holding a `sitecustomize.py` is put first on `PYTHONPATH`. Existing `PYTHONPATH` entries are kept after it. The directory is deleted when the command exits.

Python imports `sitecustomize` while it starts, before any user code. So every Python interpreter the command starts is instrumented: `python my_agent.py`, `python -m my_package`, console scripts and shebang scripts that run Python, and Python subprocesses those start while the command runs. In each interpreter the hook:

1. Runs the `sitecustomize` module it shadows, if there is one (for example a venv's or a Linux distribution's), so existing start-up code still runs.
2. Imports `agent_trace.auto`, which instruments the frameworks named in `AGENT_STRACE_AUTO_INSTRUMENT`. If the interpreter has agent-strace installed, that copy is used. Otherwise the package is loaded from the agent-strace installation that ran `auto`, without adding that directory to `sys.path`, so the interpreter's own packages are never shadowed. A venv Python therefore needs no extra install. agent-strace supports Python 3.10 and newer; an older interpreter may fail to load it.
3. If `agent_trace` cannot be imported, prints one `[agent-strace] auto-instrumentation disabled ...` line on stderr and lets the program run uninstrumented.

`agent-strace auto` exits with the command's exit code. Events go to `$AGENT_TRACE_DIR`, or to `.agent-traces` in the program's working directory. `agent-strace --trace-dir DIR auto -- <command>` sets `AGENT_TRACE_DIR` for the command.

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

`agent-strace record-http --url https://...` proxies a remote HTTP/SSE MCP server. The proxy verifies the server certificate and host name against the system CA store. To trust a private CA, set `SSL_CERT_FILE=/path/to/ca.pem` for the proxy process; this replaces OpenSSL's default CA file. There is no flag to turn verification off. A failed TLS connection returns 502 to the agent and records an `error` event. Ctrl-C also stops a TLS handshake that the server never completes. See [`record-http`](commands.md#record-http).

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

Traces: `Runner.run`, `Runner.run_sync`, `Runner.run_streamed`, all `FunctionTool` calls.

---

## LangChain / LangGraph

```python
from agent_trace.integrations import instrument_langchain
instrument_langchain()

# Now use LangChain normally
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
```

Traces: `BaseTool._run`, and the `_generate` and `_agenerate` methods of each chat model class, so `invoke`, `ainvoke`, `batch` and `abatch` are recorded with the token counts in the message's `usage_metadata`. A model class is patched the first time one of its models is called. `stream` and `astream` are not recorded; see [Model calls](#model-calls).

LangGraph node-level tracing is included — each node execution appears as a separate `tool_call` span with the node name and input/output.

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

Traces: `Crew.kickoff` (session start/end), `Agent.execute_task` (LLM request/response), `Task.execute_sync` (tool call/result).

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

Traces: `Agent.__call__`, `BaseTool.invoke`.

---

## Cross-agent trace correlation (W3C traceparent)

When one agent calls another over HTTP, inject a `traceparent` header so both sessions share the same W3C trace ID. This links spans across agents in any OTLP backend (Jaeger, Tempo, Datadog, Honeycomb).

**Injecting (outbound call):**

```python
from agent_trace.propagation import inject_traceparent

headers = inject_traceparent({}, session_id="abc123", event_id="evt456")
# headers now contains: {"traceparent": "00-<trace-id>-<span-id>-01"}

response = requests.post("https://other-agent/run", headers=headers, json={...})
```

**Extracting (inbound call):**

```python
from agent_trace.propagation import extract_traceparent

ctx = extract_traceparent(request.headers)
if ctx:
    # ctx["trace_id"]   — 32-hex W3C trace ID from upstream
    # ctx["parent_id"]  — span ID of the calling agent
    # ctx["sampled"]    — sampling flag
    pass
```

Pass `trace_id` from the extracted context into `inject_traceparent` on any further outbound calls to propagate the same trace ID through the full call chain.

The implementation follows [W3C Trace Context Level 1](https://www.w3.org/TR/trace-context/). No third-party dependencies required.

---

## OTLP GenAI interoperability

`agent-strace export --format otlp-genai` uses one OTLP/HTTP JSON payload for
Phoenix/OpenInference, Langfuse, Honeycomb, Datadog, and standard OTel
collectors. It does not install an OpenTelemetry SDK or add a gRPC path.

The payload carries both the project's `gen_ai.*` attributes and additive
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
