"""Auto-instrumentation for the OpenAI Agents SDK.

Patches:
  - agents.Runner.run_sync               → one session per run:
                                           session_start / session_end
  - function tool spans (a processor added with
    agents.tracing.add_trace_processor)  → tool_call / tool_result, or
                                           error when the span has one

Tool calls are recorded only while the SDK's tracing is enabled.

Install:
    pip install agent-strace[openai-agents]
    # or: pip install openai-agents
"""

from __future__ import annotations

import functools
import json
import threading
import time
from typing import Any

from ._base import Patches, is_wrapped

_PATCHED = False
_PATCHES = Patches()
# (store, agent name) while instrumented. The trace processor cannot be
# removed from the SDK, so it records nothing when this is None.
_STATE: Any = None
# The add_trace_processor the processor was registered with.
_REGISTERED_WITH: Any = None
# Open tool calls by span ID: (store, session ID, tool_call event, start).
_CALLS: dict[str, tuple[Any, str, Any, float]] = {}
_CALLS_LOCK = threading.Lock()


def _tool_arguments(raw: Any) -> Any:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {"input": str(raw)}
    return parsed if isinstance(parsed, dict) else {"input": parsed}


def _on_span_start(span: Any) -> None:
    from ._base import _get_or_create_session, emit
    from ..models import EventType

    state = _STATE
    data = getattr(span, "span_data", None)
    if state is None or getattr(data, "type", None) != "function":
        return
    store, agent_name = state
    sid = _get_or_create_session(store, agent_name)
    call = emit(EventType.TOOL_CALL, sid, store, tool_name=data.name,
                arguments=_tool_arguments(getattr(data, "input", None)))
    with _CALLS_LOCK:
        _CALLS[span.span_id] = (store, sid, call, time.time())


def _on_span_end(span: Any) -> None:
    from ._base import emit
    from ..models import EventType

    with _CALLS_LOCK:
        entry = _CALLS.pop(getattr(span, "span_id", None), None)
    if entry is None:
        return
    store, sid, call, t0 = entry
    data = span.span_data
    error = getattr(span, "error", None)
    fields: dict[str, Any] = {"duration_ms": (time.time() - t0) * 1000}
    # The SDK sets the span's input after the span starts.
    if getattr(data, "input", None):
        fields["arguments"] = _tool_arguments(data.input)
    if error:
        message = error
        if isinstance(error, dict):
            message = error.get("message") or "Error running tool"
            detail = (error.get("data") or {}).get("error") if isinstance(
                error.get("data"), dict) else None
            if detail:
                message = f"{message}: {detail}"
        emit(EventType.ERROR, sid, store, parent=call, tool_name=data.name,
             error=str(message), **fields)
    else:
        emit(EventType.TOOL_RESULT, sid, store, parent=call, tool_name=data.name,
             result=str(getattr(data, "output", ""))[:500], **fields)


def _tool_span_processor() -> Any:
    """A trace processor that records function tool spans as tool events."""
    try:
        from agents.tracing import TracingProcessor
        bases: tuple = (TracingProcessor,)
    except (ImportError, AttributeError):
        bases = (object,)

    def quietly(handler: Any) -> Any:
        def method(self: Any, span: Any) -> None:
            try:
                handler(span)
            except Exception:
                pass
        return method

    def nothing(self: Any, *args: Any, **kwargs: Any) -> None:
        return None

    namespace = {
        "on_trace_start": nothing,
        "on_trace_end": nothing,
        "on_span_start": quietly(_on_span_start),
        "on_span_end": quietly(_on_span_end),
        "shutdown": nothing,
        "force_flush": nothing,
    }
    return type("AgentStraceToolSpans", bases, namespace)()


def instrument_openai_agents(agent_name: str = "openai-agents") -> None:
    """Patch the OpenAI Agents SDK to emit agent-trace events.

    Idempotent — calling this more than once has no effect.
    """
    global _PATCHED, _STATE, _REGISTERED_WITH
    if _PATCHED:
        return

    try:
        import agents  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "OpenAI Agents SDK is not installed. "
            "Install it with: pip install openai-agents"
        ) from exc

    from ._base import _get_store, run_session

    store = _get_store()

    # --- Patch Runner.run_sync ---
    try:
        from agents import Runner

        def wrap_run_sync(original: Any) -> Any:
            function = getattr(original, "__func__", original)

            @functools.wraps(function)
            def _patched_run_sync(cls, *args, **kwargs):
                with run_session(store, agent_name, agent_name=agent_name):
                    return function(cls, *args, **kwargs)

            return classmethod(_patched_run_sync)

        # Runner.run_sync is a classmethod, so the attribute read from the
        # class is a bound method that Patches cannot recognise as a wrapper.
        if not is_wrapped(vars(Runner).get("run_sync")):
            _PATCHES.wrap(Runner, "run_sync", wrap_run_sync)
    except (ImportError, AttributeError):
        pass

    # --- Record function tools from the SDK's tracing ---
    # The SDK calls a tool through its FunctionTool.on_invoke_tool instance
    # attribute, inside a function span. A trace processor sees every span,
    # for Runner.run, run_sync and run_streamed alike.
    _STATE = (store, agent_name)
    try:
        from agents.tracing import add_trace_processor

        if _REGISTERED_WITH is not add_trace_processor:
            add_trace_processor(_tool_span_processor())
            _REGISTERED_WITH = add_trace_processor
    except (ImportError, AttributeError):
        pass

    _PATCHED = True


def uninstrument_openai_agents() -> None:
    """Remove patches and stop recording tool spans (for testing)."""
    global _PATCHED, _STATE
    _PATCHES.restore()
    _STATE = None
    with _CALLS_LOCK:
        _CALLS.clear()
    _PATCHED = False
