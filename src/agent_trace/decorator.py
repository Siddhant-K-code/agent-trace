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

A failure to record never reaches the caller. The call still runs, and the
caller gets its result or its own exception. Nothing more is recorded for a
call whose call event was not written.
"""

from __future__ import annotations

import functools
import inspect
import json
import threading
import time
from typing import Any, Callable, NamedTuple

from .models import EventType, SessionMeta, TraceEvent
from .redact import redact_data
from .store import TraceStore, resolve_trace_dir

# Thread-local session state so concurrent agents in the same process
# don't overwrite each other's active session.
_local = threading.local()

_STR_FAILED = "<exception str() failed>"


class _Call(NamedTuple):
    """A call event that was written, and the session that holds it."""

    event: TraceEvent
    session: SessionMeta


def _get_active_store() -> TraceStore | None:
    return getattr(_local, "store", None)


def _get_active_session() -> SessionMeta | None:
    return getattr(_local, "session", None)


def _get_active_redact() -> bool:
    return getattr(_local, "redact", False)


def start_session(
    name: str = "",
    trace_dir: str | None = None,
    redact: bool | None = None,
) -> str:
    """Start a new trace session. Returns the session ID.

    Without ``trace_dir`` the session goes to ``$AGENT_TRACE_DIR`` when it is
    set and non-empty, else ``.agent-traces``. An explicit ``trace_dir`` wins.

    Sessions are per thread and do not nest. If this thread already has an
    active session, it is ended first (as ``end_session()`` would) so it is
    not left open with no end time. If ending it fails, the new session
    still starts.

    The new session becomes active only after its directory and its
    ``session_start`` event are written. If that fails, the error is raised
    and the thread has no active session.
    """
    try:
        end_session()
    except Exception:
        # end_session() has cleared the old session, so the new one can start.
        pass
    store = TraceStore(resolve_trace_dir(trace_dir), redact=redact)
    session = SessionMeta(agent_name=name)
    store.create_session(session)
    store.append_event(
        session.session_id,
        TraceEvent(
            event_type=EventType.SESSION_START,
            session_id=session.session_id,
            data={"agent_name": name},
        ),
    )
    _local.store = store
    _local.session = session
    _local.redact = store.redact
    return session.session_id


def end_session() -> SessionMeta | None:
    """End the active trace session. Returns session metadata.

    Writes a ``session_end`` event, then ``meta.json``. Afterwards the thread
    has no active session, even when a write fails. A failed write raises.
    """
    store = _get_active_store()
    session = _get_active_session()

    if not store or not session:
        return None

    try:
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
    finally:
        _local.store = None
        _local.session = None
        _local.redact = False
    return session


def _quietly(action: Callable, *args: Any) -> Any:
    """Run one recording step. An Exception it raises is dropped."""
    try:
        return action(*args)
    except Exception:
        return None


def _emit_event(event: TraceEvent) -> SessionMeta | None:
    """Write an event to the active session, and return that session.

    Returns None when no session is active. A counter goes up only after
    its event is written.
    """
    store = _get_active_store()
    session = _get_active_session()
    if not store or not session:
        return None
    event.session_id = session.session_id
    if _get_active_redact():
        event.data = redact_data(event.data)
    store.append_event(session.session_id, event)

    if event.event_type == EventType.TOOL_CALL:
        session.tool_calls += 1
    elif event.event_type == EventType.LLM_REQUEST:
        session.llm_requests += 1
    elif event.event_type == EventType.ERROR:
        session.errors += 1
    return session


def _record_call(event: TraceEvent) -> _Call | None:
    session = _emit_event(event)
    if session is None:
        return None
    return _Call(event, session)


def _bind(func: Callable, args: Any, kwargs: Any) -> inspect.BoundArguments | None:
    """Bind a call to the function's signature, or return None.

    A builtin can have no signature, and a wrapper can take arguments that
    the signature of its ``__wrapped__`` function does not.
    """
    try:
        bound = inspect.signature(func).bind(*args, **kwargs)
    except Exception:
        return None
    bound.apply_defaults()
    return bound


def _repr_values(mapping: dict[str, Any]) -> dict[str, str]:
    return {key: _safe_repr(value) for key, value in mapping.items()}


def _tool_arguments(func: Callable, args: Any, kwargs: Any) -> dict[str, Any]:
    bound = _bind(func, args, kwargs)
    if bound is None:
        return {"args": _safe_repr(args), "kwargs": _repr_values(kwargs)}
    parameters = bound.signature.parameters
    arguments: dict[str, Any] = {}
    for name, value in bound.arguments.items():
        if parameters[name].kind is inspect.Parameter.VAR_KEYWORD:
            # An object, so key-named redaction sees each keyword.
            arguments[name] = _repr_values(value)
        else:
            arguments[name] = _safe_repr(value)
    return arguments


def _begin_tool_call(func: Callable, tool_name: str, args: Any, kwargs: Any) -> _Call | None:
    return _record_call(
        TraceEvent(
            event_type=EventType.TOOL_CALL,
            data={
                "tool_name": tool_name,
                "arguments": _tool_arguments(func, args, kwargs),
            },
        )
    )


def _llm_arguments(func: Callable, args: Any, kwargs: Any) -> dict[str, Any]:
    """Return the arguments that ``messages`` and ``model`` are read from.

    A named parameter wins over a keyword collected by ``**kwargs``. A
    ``*args`` or ``**kwargs`` parameter is never read by its own name. A call
    that cannot be bound is read from its keyword arguments only.
    """
    bound = _bind(func, args, kwargs)
    if bound is None:
        return kwargs
    named: dict[str, Any] = {}
    collected: dict[str, Any] = {}
    for name, value in bound.arguments.items():
        kind = bound.signature.parameters[name].kind
        if kind is inspect.Parameter.VAR_KEYWORD:
            collected = value
        elif kind is not inspect.Parameter.VAR_POSITIONAL:
            named[name] = value
    return {**collected, **named}


def _begin_llm_call(func: Callable, llm_name: str, args: Any, kwargs: Any) -> _Call | None:
    arguments = _llm_arguments(func, args, kwargs)

    # try to extract message count
    messages = arguments.get("messages", [])
    message_count = len(messages) if isinstance(messages, list) else 0

    return _record_call(
        TraceEvent(
            event_type=EventType.LLM_REQUEST,
            data={
                "method": llm_name,
                "message_count": message_count,
                "model": _json_value(arguments.get("model", "")),
            },
        )
    )


def _emit_success(
    call: _Call | None, event_type: EventType, start: float, result: Any
) -> None:
    # A result belongs to the session that holds its call event.
    if call is None or _get_active_session() is not call.session:
        return
    _emit_event(
        TraceEvent(
            event_type=event_type,
            parent_id=call.event.event_id,
            duration_ms=(time.time() - start) * 1000,
            data={
                "content_preview": _safe_repr(result)[:200],
            },
        )
    )


def _emit_failure(call: _Call | None, start: float, exc: Exception) -> None:
    # An error belongs to the session that holds its call event.
    if call is None or _get_active_session() is not call.session:
        return
    _emit_event(
        TraceEvent(
            event_type=EventType.ERROR,
            parent_id=call.event.event_id,
            duration_ms=(time.time() - start) * 1000,
            data={
                "message": _exception_message(exc),
                "exception_type": type(exc).__name__,
            },
        )
    )


def _exception_message(exc: BaseException) -> str:
    try:
        return str(exc)
    except Exception:
        return _STR_FAILED


def _wrap(
    func: Callable,
    begin: Callable[[Callable, str, Any, Any], _Call | None],
    label: str,
    success_type: EventType,
) -> Callable:
    # An async def function returns a coroutine at call time and runs only
    # when awaited, so the result, error and duration must be taken after
    # the await, not when the coroutine object is created.
    #
    # The function is called outside any except block, so a recording error
    # that _quietly dropped is never in the __context__ of its exception.
    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            call = _quietly(begin, func, label, args, kwargs)
            start = time.time()
            try:
                result = await func(*args, **kwargs)
            except Exception as e:
                _quietly(_emit_failure, call, start, e)
                raise
            _quietly(_emit_success, call, success_type, start, result)
            return result

        return async_wrapper

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        call = _quietly(begin, func, label, args, kwargs)
        start = time.time()
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            _quietly(_emit_failure, call, start, e)
            raise
        _quietly(_emit_success, call, success_type, start, result)
        return result

    return wrapper


def trace_tool(_func: Callable | None = None, *, name: str = ""):
    """Decorator to trace a tool function, sync or ``async def``.

    Usage:
        @trace_tool
        def my_tool(x): ...

        @trace_tool(name="custom")
        async def my_async_tool(x): ...
    """

    def decorator(func: Callable) -> Callable:
        return _wrap(func, _begin_tool_call, name or func.__name__, EventType.TOOL_RESULT)

    if _func is not None:
        return decorator(_func)
    return decorator


def trace_llm_call(_func: Callable | None = None, *, name: str = ""):
    """Decorator to trace an LLM call function, sync or ``async def``.

    It records the length of a ``messages`` list and the ``model``, each
    from the parameter of that name or else from a keyword collected by
    ``**kwargs``, and from the keyword arguments only when the call does not
    fit the signature.

    Usage:
        @trace_llm_call
        def call_openai(messages): ...
    """

    def decorator(func: Callable) -> Callable:
        return _wrap(func, _begin_llm_call, name or func.__name__, EventType.LLM_RESPONSE)

    if _func is not None:
        return decorator(_func)
    return decorator


def log_decision(choice: str, reason: str = "", alternatives: list[str] | None = None) -> None:
    """Log an agent decision point.

    A failure to record it is dropped. A value that is not JSON is stored
    as its repr.
    """
    _quietly(_record_decision, choice, reason, alternatives)


def _record_decision(choice: Any, reason: Any, alternatives: Any) -> None:
    if isinstance(alternatives, (list, tuple)):
        stored = [_json_value(item) for item in alternatives]
    else:
        stored = _json_value(alternatives or [])
    _emit_event(
        TraceEvent(
            event_type=EventType.DECISION,
            data={
                "choice": _json_value(choice),
                "reason": _json_value(reason),
                "alternatives": stored,
            },
        )
    )


def _json_value(value: Any) -> Any:
    """Return the value if it is JSON, else its repr."""
    try:
        json.dumps(value)
    except Exception:
        return _safe_repr(value)
    return value


def _safe_repr(obj: Any) -> str:
    """Safe string representation, truncated."""
    try:
        s = repr(obj)
        if len(s) > 500:
            return s[:500] + "..."
        return s
    except Exception:
        return "<unrepresentable>"
