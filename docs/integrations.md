# Auto-instrumentation

Instrument any supported agent framework without modifying application code.

---

## Quick start

```bash
# Instrument a specific framework
agent-strace auto --framework langchain -- python my_agent.py

# Auto-detect all installed frameworks
agent-strace auto --detect -- python my_agent.py

# Via environment variable (no CLI wrapper needed)
AGENT_STRACE_AUTO_INSTRUMENT=langchain,litellm python my_agent.py
```

Or in code:

```python
from agent_trace.integrations import instrument_langchain
instrument_langchain()
```

---

## Supported frameworks

| Framework | Install | What's traced |
|---|---|---|
| OpenAI Agents SDK | `pip install agent-strace[openai-agents]` | `Runner.run`, `FunctionTool` calls |
| LangChain / LangGraph | `pip install agent-strace[langchain]` | `BaseTool._run`, `BaseChatModel._generate` |
| CrewAI | `pip install agent-strace[crewai]` | `Crew.kickoff`, `Agent.execute_task`, `Task.execute_sync` |
| LiteLLM | `pip install agent-strace[litellm]` | `litellm.completion` |
| Anthropic SDK | `pip install anthropic` | `messages.create` |
| OpenAI SDK | `pip install openai` | `chat.completions.create` |
| AWS Strands | `pip install agent-strace[strands]` | `Agent.__call__`, `BaseTool.invoke` |

Install all integrations at once:

```bash
pip install agent-strace[all-integrations]
```

Each integration is an optional extra — the core package stays dependency-free. See [ADR-0003](../ADRs/0003-zero-runtime-dependencies.md).

---

## Agent CLI hooks

Use setup-generated hooks when the agent CLI has its own lifecycle hook system.

| CLI | Setup | What's traced |
|---|---|---|
| Claude Code | `agent-strace setup --cli claude` | Session start/end, user prompts, assistant responses, tool calls/results |
| OpenAI Codex | `agent-strace setup --cli codex` | Session start, user prompts, assistant responses, `PreToolUse`/`PostToolUse` tools |
| Gemini CLI | `agent-strace setup --cli gemini` | Session start/end, prompts, assistant responses, `BeforeTool`/`AfterTool` tools |
| Cursor | `agent-strace setup --cli cursor` | Session start/end, prompts, shell execution, file edits, assistant responses when emitted by Cursor hooks |

All paths write the same event stream under `.agent-traces/`, so replay, timeline, explain, why, watch, export, and audit commands work the same way after capture.

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

Traces: `BaseTool._run`, `BaseChatModel._generate`, `BaseChatModel._stream`.

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

Traces: `litellm.completion`, `litellm.acompletion`.

---

## Anthropic SDK

```python
from agent_trace.integrations import instrument_anthropic
instrument_anthropic()

import anthropic
client = anthropic.Anthropic()
message = client.messages.create(model="claude-3-5-sonnet-20241022", ...)
```

Traces: `messages.create`, `messages.stream`.

---

## OpenAI SDK

```python
from agent_trace.integrations import instrument_openai
instrument_openai()

from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(model="gpt-4o", messages=[...])
```

Traces: `chat.completions.create`, `chat.completions.stream`.

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
