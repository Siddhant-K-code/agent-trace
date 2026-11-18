"""Shared utilities for all auto-instrumentation integrations."""

from __future__ import annotations

import contextlib
import contextvars
import functools
import inspect
import os
import threading
import time
from typing import Any, Callable, Iterator

from ..fileutil import file_lock
from ..models import EventType, SessionMeta, TraceEvent
from ..pairing import link_response
from ..store import TraceStore, default_trace_dir


def _get_store() -> TraceStore:
    return TraceStore(default_trace_dir())


@contextlib.contextmanager
def _session_marker(store: TraceStore, name: str) -> Iterator[Any]:
    """Hold the per-name lock and yield the path of the agent's session marker.

    The marker is scoped by agent name so concurrent agents sharing the same
    AGENT_TRACE_DIR don't overwrite each other's active session.
    """
    safe_name = name.replace("/", "_").replace(" ", "_") or "default"
    # The lock file name does not start with ".active-session" because hook
    # and store cleanup code treats every such file as a session marker.
    store.base_dir.mkdir(parents=True, exist_ok=True)
    with file_lock(store.base_dir / f".integration-session.{safe_name}"):
        yield store.base_dir / f".active-session.{safe_name}"


def _read_marker(path: Any) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return ""


def _new_session(store: TraceStore, name: str, active_path: Any) -> str:
    meta = SessionMeta(agent_name=name)
    store.create_session(meta)
    active_path.write_text(meta.session_id, encoding="utf-8", newline="\n")
    return meta.session_id


# The run (see run_session) active in this context: ((trace dir, agent
# name), session ID).
_RUN: contextvars.ContextVar[tuple[tuple[str, str], str] | None] = contextvars.ContextVar(
    "agent_strace_integration_run", default=None,
)
# Event counts of the runs in progress in this process, by session ID.
_RUN_COUNTS: dict[str, dict[str, int]] = {}
_RUN_COUNTS_LOCK = threading.Lock()
_COUNTED = {
    EventType.TOOL_CALL: "tool_calls",
    EventType.LLM_REQUEST: "llm_requests",
    EventType.ERROR: "errors",
}


def _run_key(store: TraceStore, name: str) -> tuple[str, str]:
    return (str(store.base_dir), name)


def _get_or_create_session(store: TraceStore, name: str) -> str:
    """Return the active session ID for this agent, creating one if needed.

    Inside a :func:`run_session` for this agent, that run's session is
    returned. Otherwise the session named by the agent's marker file is
    returned, so threads that do not inherit the run's context still find
    it. The check and the create run under a per-name file lock, so
    concurrent first calls from threads or processes share one session.
    """
    run = _RUN.get()
    if run is not None and run[0] == _run_key(store, name):
        return run[1]
    with _session_marker(store, name) as active_path:
        sid = _read_marker(active_path)
        if sid and store.session_exists(sid):
            return sid
        return _new_session(store, name, active_path)


def _count_run_event(session_id: str, event_type: EventType) -> None:
    field_name = _COUNTED.get(event_type)
    if field_name is None:
        return
    with _RUN_COUNTS_LOCK:
        counts = _RUN_COUNTS.get(session_id)
        if counts is not None:
            counts[field_name] += 1


def _start_run(store: TraceStore, name: str) -> tuple[str, str]:
    """Create the run's session and point the agent's marker at it.

    Returns the session ID and the session the marker named before.
    """
    with _session_marker(store, name) as active_path:
        previous = _read_marker(active_path)
        sid = _new_session(store, name, active_path)
    with _RUN_COUNTS_LOCK:
        _RUN_COUNTS[sid] = dict.fromkeys(_COUNTED.values(), 0)
    return sid, previous


def _end_run(store: TraceStore, name: str, sid: str, previous: str, started: float) -> None:
    """Record ``session_end``, close the session's metadata and clear its marker."""
    try:
        emit(EventType.SESSION_END, sid, store,
             duration_ms=(time.time() - started) * 1000)
    finally:
        with _RUN_COUNTS_LOCK:
            counts = _RUN_COUNTS.pop(sid, {})
            previous_running = previous in _RUN_COUNTS

        def close(meta: SessionMeta) -> None:
            meta.ended_at = time.time()
            meta.total_duration_ms = (meta.ended_at - meta.started_at) * 1000
            for field_name, value in counts.items():
                setattr(meta, field_name, value)

        try:
            store.mutate_meta(sid, close)
        finally:
            with _session_marker(store, name) as active_path:
                if _read_marker(active_path) == sid:
                    if previous_running:
                        active_path.write_text(previous, encoding="utf-8", newline="\n")
                    else:
                        active_path.unlink(missing_ok=True)


@contextlib.contextmanager
def run_session(store: TraceStore, name: str, **start: Any) -> Iterator[str]:
    """Record one framework run, such as a crew kickoff, as its own session.

    Emits ``session_start`` with *start* as its data, yields the session ID,
    and when the run ends emits ``session_end`` (also after an exception,
    which is first recorded as an ``error``). The session's ``ended_at``,
    ``total_duration_ms``, ``tool_calls``, ``llm_requests`` and ``errors``
    are then set and the agent's marker is cleared. While the run is in
    progress, :func:`_get_or_create_session` returns its session for this
    agent. A run started inside another run of the same agent, in the same
    context, records into the outer run's session. A failure to record the
    end of the run never reaches the caller.
    """
    key = _run_key(store, name)
    current = _RUN.get()
    if current is not None and current[0] == key:
        yield current[1]
        return
    sid, previous = _start_run(store, name)
    started = time.time()
    token = _RUN.set((key, sid))
    try:
        try:
            emit(EventType.SESSION_START, sid, store, **start)
        except Exception:
            pass
        yield sid
    except BaseException as exc:
        try:
            emit(EventType.ERROR, sid, store,
                 error=str(exc), error_type=type(exc).__name__)
        except Exception:
            pass
        raise
    finally:
        try:
            _RUN.reset(token)
        except ValueError:
            # Exited in another context than the one it was entered in.
            pass
        try:
            _end_run(store, name, sid, previous, started)
        except Exception:
            pass


def emit(
    event_type: EventType,
    session_id: str,
    store: TraceStore,
    *,
    parent: TraceEvent | None = None,
    **data: Any,
) -> TraceEvent:
    """Write a single event to the store (or remote endpoint if configured).

    With ``parent``, the event records that it answers that request: its
    ``parent_id`` and ``duration_ms`` are set (see pairing.link_response).
    Returns the event.
    """
    ev = TraceEvent(event_type=event_type, session_id=session_id, data=dict(data))
    if parent is not None:
        link_response(ev, parent.event_id, parent.timestamp)
    endpoint = os.environ.get("AGENT_STRACE_ENDPOINT", "").rstrip("/")
    if endpoint:
        from ..server import send_event_to_endpoint
        send_event_to_endpoint(ev, endpoint)
    else:
        store.append_event(session_id, ev)
    _count_run_event(session_id, event_type)
    return ev


def count(value: Any) -> int | None:
    """Return *value* if it is a token count (an int, not a bool), else None."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def message_count(messages: Any) -> int:
    """Number of messages in a request, or 0 if they cannot be counted.

    The SDKs accept any iterable of messages; a generator is not counted,
    because counting it would consume it.
    """
    if not messages:
        return 0
    try:
        return len(messages)
    except TypeError:
        return 0


def given(value: Any, default: Any = None) -> Any:
    """*value*, or *default* if it is an SDK "not given" placeholder.

    The OpenAI and Anthropic SDKs mark an argument the caller left out with
    ``NOT_GIVEN`` or ``omit``, and their ``.stream()`` helpers pass those on
    to ``create``.
    """
    return default if type(value).__name__ in ("NotGiven", "Omit") else value


# ---------------------------------------------------------------------------
# Model calls
# ---------------------------------------------------------------------------

_IN_MODEL_CALL: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "agent_strace_in_model_call", default=False,
)


def in_model_call() -> bool:
    """True while an instrumented model call runs in this context.

    One call can pass through several instrumented layers: LiteLLM and
    LangChain call the OpenAI SDK, for example. Only the outermost layer
    records it.
    """
    return _IN_MODEL_CALL.get()


@contextlib.contextmanager
def _model_call_scope() -> Iterator[None]:
    token = _IN_MODEL_CALL.set(True)
    try:
        yield
    finally:
        _IN_MODEL_CALL.reset(token)


class ModelCall:
    """A recorded model call: its ``llm_request`` and how the call ended."""

    def __init__(self, store: TraceStore, agent_name: str, model: str, **request: Any) -> None:
        self.store = store
        self.session_id = _get_or_create_session(store, agent_name)
        self.model = model
        self.started = time.time()
        self.request = emit(
            EventType.LLM_REQUEST, self.session_id, store, model=model, **request,
        )

    def respond(self, fields: dict[str, Any]) -> None:
        """Record the ``llm_response``, linked to the request.

        ``fields`` may override ``model``.
        """
        data = {
            "model": self.model,
            **fields,
            "duration_ms": (time.time() - self.started) * 1000,
        }
        emit(EventType.LLM_RESPONSE, self.session_id, self.store,
             parent=self.request, **data)

    def fail(self, exc: BaseException) -> None:
        emit(EventType.ERROR, self.session_id, self.store, parent=self.request,
             error=str(exc), error_type=type(exc).__name__)

    def cancel(self) -> None:
        self.respond({"status": "cancelled"})


def wrap_model_call(
    original: Callable[..., Any],
    start: Callable[[tuple, dict], ModelCall],
    finish: Callable[[ModelCall, dict, Any], None],
    *,
    is_async: bool,
) -> Callable[..., Any]:
    """Wrap a function that calls a model so that the call is recorded.

    ``start(args, kwargs)`` records the request and returns its
    :class:`ModelCall`; ``finish(call, kwargs, result)`` records the
    response. An exception from the call is recorded as an error linked to
    the request. A call ended by any other ``BaseException``, such as a
    cancelled async call, ``KeyboardInterrupt`` or ``SystemExit``, is
    recorded as a cancelled response, so its request is always closed. A call
    made inside another recorded call is passed through unrecorded. A
    failure to record never reaches the caller.
    """
    if is_async:
        @functools.wraps(original)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            call = None if in_model_call() else _begin(start, args, kwargs)
            if call is None:
                return await original(*args, **kwargs)
            try:
                with _model_call_scope():
                    result = await original(*args, **kwargs)
            except Exception as exc:
                _quietly(call.fail, exc)
                raise
            except BaseException:
                _quietly(call.cancel)
                raise
            _quietly(finish, call, kwargs, result)
            return result

        return async_wrapper

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        call = None if in_model_call() else _begin(start, args, kwargs)
        if call is None:
            return original(*args, **kwargs)
        try:
            with _model_call_scope():
                result = original(*args, **kwargs)
        except Exception as exc:
            _quietly(call.fail, exc)
            raise
        except BaseException:
            _quietly(call.cancel)
            raise
        _quietly(finish, call, kwargs, result)
        return result

    return wrapper


def _begin(start: Callable[[tuple, dict], ModelCall], args: tuple,
           kwargs: dict) -> ModelCall | None:
    """Record the request; None if that fails, so the call goes on unrecorded."""
    try:
        return start(args, kwargs)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Streams
# ---------------------------------------------------------------------------

def record_stream(stream: Any, call: ModelCall, accumulator: Any) -> bool:
    """Record the response of a model stream when the caller reaches its end.

    The OpenAI and Anthropic SDK streams read their items from a generator
    in their ``_iterator`` attribute. It is replaced, on the same object, by
    one that yields the same items, so the caller gets the stream it would
    have got. Each item is passed to ``accumulator.add(item)``; at the end
    of the stream ``accumulator.fields()`` is recorded as the
    ``llm_response``. An exception raised by the stream is recorded as an
    error linked to the request. Any other ``BaseException`` raised while
    reading the next item, such as cancelling the task that reads an async
    stream or a ``KeyboardInterrupt``, is recorded as a cancelled response.
    Nothing is recorded for a stream dropped before its end (its generator
    gets ``GeneratorExit``), because it is closed by the garbage collector,
    where writing to the trace store is not safe. A failure to record never
    reaches the caller.

    Returns False, leaving the stream unchanged, if it has no such generator.
    """
    try:
        inner = stream._iterator
    except Exception:
        return False
    if inspect.isgenerator(inner):
        wrapped: Any = _recorded_items(inner, call, accumulator)
    elif inspect.isasyncgen(inner):
        wrapped = _recorded_async_items(inner, call, accumulator)
    else:
        return False
    try:
        stream._iterator = wrapped
    except Exception:
        return False
    return True


def record_when_parsed(raw: Any, record: Callable[[Any], None]) -> bool:
    """Record a ``with_raw_response`` or ``with_streaming_response`` result
    when the caller parses it.

    Such a result wraps the HTTP response, and its ``parse()`` method
    returns the response body: the model's response, or its stream. The
    OpenAI and Anthropic SDKs pass that body through the request's
    ``post_parser`` option, which is chained here so that the first body
    with ``usage``, or a stream, is passed to ``record(body)`` and then
    returned unchanged. Nothing is recorded if the caller never parses it.
    A failure to record never reaches the caller.

    Returns False, leaving ``raw`` unchanged, if it is not such a result.
    """
    try:
        if not callable(getattr(raw, "parse", None)) or not hasattr(raw, "http_response"):
            return False
        options = raw._options
        previous = options.post_parser
    except Exception:
        return False
    chained = previous if callable(previous) else None
    recorded = False

    def post_parser(body: Any) -> Any:
        nonlocal recorded
        if chained is not None:
            body = chained(body)
        if not recorded and (hasattr(body, "usage") or hasattr(body, "_iterator")):
            recorded = True
            _quietly(record, body)
        return body

    try:
        options.post_parser = post_parser
    except Exception:
        return False
    return True


def _quietly(action: Callable[..., Any], *args: Any) -> None:
    try:
        action(*args)
    except Exception:
        pass


def _respond_from(call: ModelCall, accumulator: Any) -> None:
    call.respond(accumulator.fields())


def _recorded_items(inner: Any, call: ModelCall, accumulator: Any) -> Any:
    try:
        for item in inner:
            _quietly(accumulator.add, item)
            yield item
    except GeneratorExit:
        raise
    except Exception as exc:
        _quietly(call.fail, exc)
        raise
    except BaseException:
        _quietly(call.cancel)
        raise
    _quietly(_respond_from, call, accumulator)


async def _recorded_async_items(inner: Any, call: ModelCall, accumulator: Any) -> Any:
    try:
        async for item in inner:
            _quietly(accumulator.add, item)
            yield item
    except GeneratorExit:
        raise
    except Exception as exc:
        _quietly(call.fail, exc)
        raise
    except BaseException:
        _quietly(call.cancel)
        raise
    _quietly(_respond_from, call, accumulator)


# ---------------------------------------------------------------------------
# Patching
# ---------------------------------------------------------------------------

_ORIGINAL_ATTR = "__agent_strace_original__"
_ABSENT = object()


def is_wrapped(function: Any) -> bool:
    """True if *function* is a wrapper installed by :class:`Patches`."""
    try:
        return _ORIGINAL_ATTR in vars(function)
    except TypeError:
        return False


class Patches:
    """Attributes replaced by an integration, so that they can be restored."""

    def __init__(self) -> None:
        self._applied: list[tuple[Any, str, Any, Any]] = []
        self._lock = threading.Lock()

    def wrap(self, owner: Any, name: str, make_wrapper: Callable[[Any], Any]) -> bool:
        """Replace ``owner.<name>`` with ``make_wrapper(original)``.

        Does nothing, and returns False, if the attribute is missing or is
        already a wrapper, so patching twice never wraps twice.
        """
        with self._lock:
            try:
                original = getattr(owner, name)
            except AttributeError:
                return False
            if is_wrapped(original):
                return False
            wrapper = make_wrapper(original)
            setattr(wrapper, _ORIGINAL_ATTR, original)
            own = vars(owner).get(name, _ABSENT)
            setattr(owner, name, wrapper)
            self._applied.append((owner, name, own, wrapper))
            return True

    def restore(self) -> None:
        """Put back every attribute this object replaced that is still a wrapper."""
        with self._lock:
            while self._applied:
                owner, name, own, wrapper = self._applied.pop()
                if vars(owner).get(name) is not wrapper:
                    continue
                if own is _ABSENT:
                    delattr(owner, name)
                else:
                    setattr(owner, name, own)
