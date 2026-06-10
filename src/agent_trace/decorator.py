"""Python decorator for tracing agent tool calls.

For agents that don't use MCP, wrap your tool functions:

    from agent_trace import trace_tool

    @trace_tool
    def search_web(query: str) -> str:
        return requests.get(f"https://api.search.com?q={query}").text

    @trace_tool(name="custom_name")
    def my_tool(x: int) -> int:
        return x * 2

Every call to a traced function is logged to the active session.
"""

from __future__ import annotations

import functools
import inspect
import threading
import time
from typing import Any, Callable

from .models import EventType, SessionMeta, TraceEvent
from .redact import redact_data
from .store import TraceStore

# Thread-local session state so concurrent agents in the same process
# don't overwrite each other's active session.
_local = threading.local()


def _get_active_store() -> TraceStore | None:
    return getattr(_local, "store", None)


def _get_active_session() -> SessionMeta | None:
    return getattr(_local, "session", None)


def _get_active_redact() -> bool:
    return getattr(_local, "redact", False)


def start_session(
    name: str = "",
    trace_dir: str = ".agent-traces",
    redact: bool | None = None,
) -> str:
    """Start a new trace session. Returns the session ID."""
    _local.store = TraceStore(trace_dir, redact=redact)
    _local.session = SessionMeta(agent_name=name)
    _local.redact = _local.store.redact
    _local.store.create_session(_local.session)

    event = TraceEvent(
        event_type=EventType.SESSION_START,
        session_id=_local.session.session_id,
        data={"agent_name": name},
    )
    _local.store.append_event(_local.session.session_id, event)

    return _local.session.session_id


def end_session() -> SessionMeta | None:
    """End the active trace session. Returns session metadata."""
    store = _get_active_store()
    session = _get_active_session()

    if not store or not session:
        return None

    session.ended_at = time.time()
    session.total_duration_ms = (
        session.ended_at - session.started_at
    ) * 1000

    event = TraceEvent(
        event_type=EventType.SESSION_END,
        session_id=session.session_id,
        data={
            "duration_ms": session.total_duration_ms,
            "tool_calls": session.tool_calls,
            "errors": session.errors,
        },
    )
    store.append_event(session.session_id, event)
    store.update_meta(session)

    meta = session
    _local.store = None
    _local.session = None
    _local.redact = False
    return meta


def _emit_event(event: TraceEvent) -> None:
    store = _get_active_store()
    session = _get_active_session()
    if store and session:
        event.session_id = session.session_id
        if _get_active_redact():
            event.data = redact_data(event.data)
        store.append_event(session.session_id, event)

        if event.event_type == EventType.TOOL_CALL:
            session.tool_calls += 1
        elif event.event_type == EventType.ERROR:
            session.errors += 1


def trace_tool(_func: Callable | None = None, *, name: str = ""):
    """Decorator to trace a tool function.

    Usage:
        @trace_tool
        def my_tool(x): ...

        @trace_tool(name="custom")
        def my_tool(x): ...
    """

    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # build argument map
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            arg_data = {k: _safe_repr(v) for k, v in bound.arguments.items()}

            call_event = TraceEvent(
                event_type=EventType.TOOL_CALL,
                data={
                    "tool_name": tool_name,
                    "arguments": arg_data,
                },
            )
            _emit_event(call_event)

            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start) * 1000

                result_event = TraceEvent(
                    event_type=EventType.TOOL_RESULT,
                    parent_id=call_event.event_id,
                    duration_ms=duration_ms,
                    data={
                        "content_preview": _safe_repr(result)[:200],
                    },
                )
                _emit_event(result_event)
                return result

            except Exception as e:
                duration_ms = (time.time() - start) * 1000
                error_event = TraceEvent(
                    event_type=EventType.ERROR,
                    parent_id=call_event.event_id,
                    duration_ms=duration_ms,
                    data={
                        "message": str(e),
                        "exception_type": type(e).__name__,
                    },
                )
                _emit_event(error_event)
                raise

        return wrapper

    if _func is not None:
        return decorator(_func)
    return decorator


def trace_llm_call(_func: Callable | None = None, *, name: str = ""):
    """Decorator to trace an LLM call function.

    Usage:
        @trace_llm_call
        def call_openai(messages): ...
    """

    def decorator(func: Callable) -> Callable:
        llm_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # try to extract message count
            messages = bound.arguments.get("messages", [])
            message_count = len(messages) if isinstance(messages, list) else 0

            req_event = TraceEvent(
                event_type=EventType.LLM_REQUEST,
                data={
                    "method": llm_name,
                    "message_count": message_count,
                    "model": bound.arguments.get("model", ""),
                },
            )
            _emit_event(req_event)

            session = _get_active_session()
            if session:
                session.llm_requests += 1

            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start) * 1000

                resp_event = TraceEvent(
                    event_type=EventType.LLM_RESPONSE,
                    parent_id=req_event.event_id,
                    duration_ms=duration_ms,
                    data={
                        "content_preview": _safe_repr(result)[:200],
                    },
                )
                _emit_event(resp_event)
                return result

            except Exception as e:
                duration_ms = (time.time() - start) * 1000
                error_event = TraceEvent(
                    event_type=EventType.ERROR,
                    parent_id=req_event.event_id,
                    duration_ms=duration_ms,
                    data={
                        "message": str(e),
                        "exception_type": type(e).__name__,
                    },
                )
                _emit_event(error_event)
                raise

        return wrapper

    if _func is not None:
        return decorator(_func)
    return decorator


def log_decision(choice: str, reason: str = "", alternatives: list[str] | None = None) -> None:
    """Log an agent decision point."""
    _emit_event(
        TraceEvent(
            event_type=EventType.DECISION,
            data={
                "choice": choice,
                "reason": reason,
                "alternatives": alternatives or [],
            },
        )
    )


def _safe_repr(obj: Any) -> str:
    """Safe string representation, truncated."""
    try:
        s = repr(obj)
        if len(s) > 500:
            return s[:500] + "..."
        return s
    except Exception:
        return "<unrepresentable>"
