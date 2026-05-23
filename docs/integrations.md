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
