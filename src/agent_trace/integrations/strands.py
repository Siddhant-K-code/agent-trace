"""Auto-instrumentation for AWS Strands Agents.

Patches:
  - strands.Agent.__call__ and invoke_async  → one session per call:
                                               session_start / session_end
  - strands.Agent.__init__ adds hook callbacks:
      BeforeToolCallEvent                    → tool_call
      AfterToolCallEvent                     → tool_result, or error when
                                               the tool failed

Install:
    pip install agent-strace[strands]
    # or: pip install strands-agents
"""

from __future__ import annotations

import functools
import threading
import time
from typing import Any

from ._base import Patches

_PATCHED = False
_PATCHES = Patches()
# (store, agent name) while instrumented. Agents created while instrumented
# keep their hook callbacks, which record nothing when this is None.
_STATE: Any = None
# Open tool calls by (agent, toolUseId): (store, session ID, tool_call, start).
_CALLS: dict[tuple[int, str], tuple[Any, str, Any, float]] = {}
_CALLS_LOCK = threading.Lock()
# The SDK's before and after tool call hook event classes.
_HOOK_EVENTS: Any = None


def _tool_events() -> tuple[Any, Any]:
    """The SDK's before and after tool call hook events."""
    from strands import hooks

    before = getattr(hooks, "BeforeToolCallEvent", None) or hooks.BeforeToolInvocationEvent
    after = getattr(hooks, "AfterToolCallEvent", None) or hooks.AfterToolInvocationEvent
    return before, after


def _call_key(event: Any) -> tuple[int, str]:
    return (id(getattr(event, "agent", None)), str(event.tool_use.get("toolUseId", "")))


def _result_text(result: Any) -> str:
    content = result.get("content") if isinstance(result, dict) else None
    if isinstance(content, list):
        parts = [str(block.get("text", block)) if isinstance(block, dict) else str(block)
                 for block in content]
        return " ".join(parts)
    return str(result)


def _before_tool_call(event: Any) -> None:
    from ._base import _get_or_create_session, emit
    from ..models import EventType

    state = _STATE
    if state is None:
        return
    store, agent_name = state
    tool_use = event.tool_use
    arguments = tool_use.get("input")
    sid = _get_or_create_session(store, agent_name)
    call = emit(EventType.TOOL_CALL, sid, store, tool_name=tool_use.get("name", ""),
                arguments=arguments if isinstance(arguments, dict) else {"input": arguments})
    with _CALLS_LOCK:
        _CALLS[_call_key(event)] = (store, sid, call, time.time())


def _after_tool_call(event: Any) -> None:
    from ._base import emit
    from ..models import EventType

    with _CALLS_LOCK:
        entry = _CALLS.pop(_call_key(event), None)
    if entry is None:
        return
    store, sid, call, t0 = entry
    tool_name = event.tool_use.get("name", "")
    duration_ms = (time.time() - t0) * 1000
    result = getattr(event, "result", None)
    exception = getattr(event, "exception", None)
    status = result.get("status") if isinstance(result, dict) else None
    if exception is not None or status == "error":
        fields = {"error_type": type(exception).__name__} if exception is not None else {}
        emit(EventType.ERROR, sid, store, parent=call, tool_name=tool_name,
             error=str(exception) if exception is not None else _result_text(result)[:500],
             duration_ms=duration_ms, **fields)
    else:
        emit(EventType.TOOL_RESULT, sid, store, parent=call, tool_name=tool_name,
             result=_result_text(result)[:500], duration_ms=duration_ms)


def _quietly(handler: Any) -> Any:
    def callback(event: Any) -> None:
        try:
            handler(event)
        except Exception:
            pass
    return callback


def _add_hooks(agent: Any) -> None:
    """Add the tool call callbacks to an agent's hook registry, once."""
    events = _HOOK_EVENTS
    hooks = getattr(agent, "hooks", None)
    if events is None or hooks is None or vars(agent).get("_agent_strace_hooks") is hooks:
        return
    before, after = events
    hooks.add_callback(before, _quietly(_before_tool_call))
    hooks.add_callback(after, _quietly(_after_tool_call))
    agent._agent_strace_hooks = hooks


def instrument_strands(agent_name: str = "strands") -> None:
    """Patch AWS Strands Agents to emit agent-trace events. Idempotent."""
    global _PATCHED, _STATE, _HOOK_EVENTS
    if _PATCHED:
        return

    try:
        import strands  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "AWS Strands Agents is not installed. "
            "Install it with: pip install strands-agents"
        ) from exc

    from ._base import _get_store, run_session

    store = _get_store()
    _STATE = (store, agent_name)

    try:
        from strands import Agent
    except ImportError:
        _PATCHED = True
        return

    # --- Tool calls through the agent's hook registry ---
    # The SDK's tool executor invokes these hooks around tool.stream() for
    # every tool type (PythonAgentTool, @tool functions, MCP tools). Agents
    # get the callbacks when created, or on their first call if they were
    # created before instrumenting.
    try:
        _HOOK_EVENTS = _tool_events()
    except (ImportError, AttributeError):
        _HOOK_EVENTS = None

    def add_hooks(agent: Any) -> None:
        try:
            _add_hooks(agent)
        except Exception:
            pass

    def wrap_init(original: Any) -> Any:
        @functools.wraps(original)
        def _patched_init(self, *args, **kwargs):
            original(self, *args, **kwargs)
            add_hooks(self)

        return _patched_init

    # --- One session per Agent.__call__ / invoke_async ---
    def start_data(agent: Any) -> dict[str, Any]:
        return {"agent_name": getattr(agent, "name", None) or agent_name}

    def wrap_agent_call(original: Any) -> Any:
        @functools.wraps(original)
        def _patched_agent_call(self, *args, **kwargs):
            add_hooks(self)
            with run_session(store, agent_name, **start_data(self)):
                return original(self, *args, **kwargs)

        return _patched_agent_call

    def wrap_invoke_async(original: Any) -> Any:
        @functools.wraps(original)
        async def _patched_invoke_async(self, *args, **kwargs):
            add_hooks(self)
            with run_session(store, agent_name, **start_data(self)):
                return await original(self, *args, **kwargs)

        return _patched_invoke_async

    for attr, wrap in (("__init__", wrap_init), ("__call__", wrap_agent_call),
                       ("invoke_async", wrap_invoke_async)):
        try:
            _PATCHES.wrap(Agent, attr, wrap)
        except AttributeError:
            pass

    _PATCHED = True


def uninstrument_strands() -> None:
    """Remove patches and stop recording tool calls (for testing)."""
    global _PATCHED, _STATE
    _PATCHES.restore()
    _STATE = None
    with _CALLS_LOCK:
        _CALLS.clear()
    _PATCHED = False
