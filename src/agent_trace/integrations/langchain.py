"""Auto-instrumentation for LangChain / LangGraph.

Patches:
  - langchain_core.tools.BaseTool.run and .arun    → tool_call / tool_result
  - chat models' ``_generate`` and ``_agenerate``  → llm_request / llm_response
  - langgraph.graph.StateGraph.compile             → records each node run as
                                                     decision / tool_call / tool_result

Chat model classes override ``_generate``, so the class that defines a
model's ``_generate`` and ``_agenerate`` is patched the first time the model
is invoked. A response served from the LangChain LLM cache, and output read
with ``stream`` or ``astream`` (or with ``streaming=True``), is not recorded.

Install:
    pip install agent-strace[langchain]
    # or: pip install langchain-core
    # LangGraph node tracing also requires: pip install langgraph
    # It traces graphs compiled after instrument_langchain() only.
"""

from __future__ import annotations

import asyncio
import functools
import json
import threading
import time
from typing import Any

from ._base import Patches, count as _count, message_count

_PATCHED = False
_orig_compile = None
_TOOL_PATCHES = Patches()
_MODEL_PATCHES = Patches()
_MODEL_CLASSES: set[type] = set()
_MODEL_CLASSES_LOCK = threading.Lock()


def _detail(details: dict, name: str) -> int | None:
    """``details[name]``, or its service-tier variant such as ``priority_<name>``."""
    value = _count(details.get(name))
    if value is not None:
        return value
    for key, item in details.items():
        if isinstance(key, str) and key.endswith("_" + name):
            return _count(item)
    return None


def _usage_fields(result: Any) -> dict[str, Any]:
    """Token counts of a ``ChatResult``, from its message's ``usage_metadata``.

    LangChain's ``input_tokens`` is the sum of all input tokens, cache reads
    and writes included, and ``input_token_details`` holds their counts.
    LangChain Anthropic reports cache writes split by TTL, with a
    ``cache_creation`` of 0, when Anthropic splits them.
    """
    try:
        message = getattr(result.generations[0], "message", None)
    except (AttributeError, IndexError, KeyError, TypeError):
        return {}
    usage = getattr(message, "usage_metadata", None)
    if not isinstance(usage, dict):
        return {}
    fields: dict[str, Any] = {
        "input_tokens": _count(usage.get("input_tokens")) or 0,
        "output_tokens": _count(usage.get("output_tokens")) or 0,
    }
    details = usage.get("input_token_details")
    if not isinstance(details, dict):
        return fields
    read = _detail(details, "cache_read")
    write = _detail(details, "cache_creation")
    if not write:
        split = [_count(details.get(key))
                 for key in ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens")]
        if any(value is not None for value in split):
            write = sum(value or 0 for value in split)
    if read is not None:
        fields["cached_input_tokens"] = read
    if write is not None:
        fields["cache_write_input_tokens"] = write
    return fields


def instrument_langchain(agent_name: str = "langchain") -> None:
    """Patch LangChain (and LangGraph if installed) to emit agent-trace events. Idempotent."""
    global _PATCHED, _orig_compile
    if _PATCHED:
        return

    try:
        import langchain_core  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "LangChain Core is not installed. "
            "Install it with: pip install langchain-core"
        ) from exc

    from ._base import _get_store, _get_or_create_session, emit
    from ..models import EventType

    store = _get_store()

    # --- Patch BaseTool.run / BaseTool.arun ---
    # Tool classes such as StructuredTool override _run, but not run and
    # arun, which invoke and ainvoke call.
    try:
        from langchain_core.tools import BaseTool
    except (ImportError, AttributeError):
        BaseTool = None
    if BaseTool is not None:
        _instrument_tools(BaseTool, store, agent_name)

    # --- Patch chat models ---
    try:
        from langchain_core.language_models import BaseChatModel
    except (ImportError, AttributeError):
        BaseChatModel = None
    if BaseChatModel is not None:
        _instrument_chat_models(BaseChatModel, store, agent_name)

    # --- Patch LangGraph StateGraph.compile (optional, best-effort) ---
    try:
        from langgraph.graph import StateGraph

        _orig_compile = StateGraph.compile

        def _patched_compile(self, *args, **kwargs):
            compiled = _orig_compile(self, *args, **kwargs)
            _wrap_langgraph_nodes(compiled, store, agent_name)
            return compiled

        StateGraph.compile = _patched_compile
    except (ImportError, AttributeError):
        pass  # LangGraph not installed — skip silently

    _PATCHED = True


def _tool_arguments(tool_input: Any) -> Any:
    """The recorded arguments of a tool's input, which must be JSON."""
    try:
        json.dumps(tool_input)
    except (TypeError, ValueError):
        return {"input": str(tool_input)[:200]}
    return tool_input if isinstance(tool_input, dict) else {"input": tool_input}


def _instrument_tools(base: type, store: Any, agent_name: str) -> None:
    """Record tool calls made through ``base.run`` and ``base.arun``."""
    from ._base import _get_or_create_session, emit
    from ..models import EventType

    def begin(tool: Any, tool_input: Any) -> tuple[str, Any, Any, float]:
        sid = _get_or_create_session(store, agent_name)
        tool_name = getattr(tool, "name", str(tool))
        call = emit(EventType.TOOL_CALL, sid, store,
                    tool_name=tool_name, arguments=_tool_arguments(tool_input))
        return sid, tool_name, call, time.time()

    def succeed(started: tuple[str, Any, Any, float], result: Any) -> None:
        sid, tool_name, call, t0 = started
        emit(EventType.TOOL_RESULT, sid, store, parent=call,
             tool_name=tool_name,
             result=str(result)[:500],
             duration_ms=(time.time() - t0) * 1000)

    def fail(started: tuple[str, Any, Any, float], exc: BaseException) -> None:
        sid, tool_name, call, _ = started
        emit(EventType.ERROR, sid, store, parent=call,
             tool_name=tool_name, error=str(exc), error_type=type(exc).__name__)

    def cancel(started: tuple[str, Any, Any, float]) -> None:
        # Cancelled, KeyboardInterrupt or SystemExit, as for a model call.
        sid, tool_name, call, t0 = started
        try:
            emit(EventType.TOOL_RESULT, sid, store, parent=call, tool_name=tool_name,
                 duration_ms=(time.time() - t0) * 1000, status="cancelled")
        except Exception:
            pass

    def wrap_run(original: Any) -> Any:
        @functools.wraps(original)
        def run(self: Any, tool_input: Any, *args: Any, **kwargs: Any) -> Any:
            started = begin(self, tool_input)
            try:
                result = original(self, tool_input, *args, **kwargs)
            except Exception as exc:
                fail(started, exc)
                raise
            except BaseException:
                cancel(started)
                raise
            succeed(started, result)
            return result

        return run

    def wrap_arun(original: Any) -> Any:
        @functools.wraps(original)
        async def arun(self: Any, tool_input: Any, *args: Any, **kwargs: Any) -> Any:
            started = begin(self, tool_input)
            try:
                result = await original(self, tool_input, *args, **kwargs)
            except Exception as exc:
                fail(started, exc)
                raise
            except BaseException:
                cancel(started)
                raise
            succeed(started, result)
            return result

        return arun

    _TOOL_PATCHES.wrap(base, "run", wrap_run)
    _TOOL_PATCHES.wrap(base, "arun", wrap_arun)


def _instrument_chat_models(base: type, store: Any, agent_name: str) -> None:
    """Record chat model calls made through ``base`` and its subclasses."""
    from ._base import ModelCall, wrap_model_call

    def start(args: tuple, kwargs: dict) -> ModelCall:
        model = args[0]
        messages = kwargs["messages"] if "messages" in kwargs else (
            args[1] if len(args) > 1 else None)
        name = getattr(model, "model_name", getattr(model, "model", "unknown"))
        return ModelCall(store, agent_name, str(name),
                         message_count=message_count(messages))

    def finish(call: ModelCall, kwargs: dict, result: Any) -> None:
        call.respond(_usage_fields(result))

    def recorder(is_async: bool):
        return lambda original: wrap_model_call(original, start, finish, is_async=is_async)

    def patch_model_class(cls: type) -> None:
        if cls in _MODEL_CLASSES:
            return
        with _MODEL_CLASSES_LOCK:
            if cls in _MODEL_CLASSES:
                return
            for name, is_async in (("_generate", False), ("_agenerate", True)):
                owner = next((k for k in cls.__mro__ if name in vars(k)), None)
                if owner is not None:
                    _MODEL_PATCHES.wrap(owner, name, recorder(is_async))
            _MODEL_CLASSES.add(cls)

    def hook(original: Any, is_async: bool) -> Any:
        if is_async:
            @functools.wraps(original)
            async def async_hook(self: Any, *args: Any, **kwargs: Any) -> Any:
                patch_model_class(type(self))
                return await original(self, *args, **kwargs)

            return async_hook

        @functools.wraps(original)
        def sync_hook(self: Any, *args: Any, **kwargs: Any) -> Any:
            patch_model_class(type(self))
            return original(self, *args, **kwargs)

        return sync_hook

    _MODEL_PATCHES.wrap(base, "_generate", recorder(False))
    for name, is_async in (("_generate_with_cache", False), ("_agenerate_with_cache", True)):
        _MODEL_PATCHES.wrap(base, name, lambda original, a=is_async: hook(original, a))


def _loop_is_running() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _node_recorder(runnable: type, store: Any, agent_name: str) -> type:
    """A runnable that records a LangGraph node's runs and runs its ``bound``.

    It records no LangChain callback run of its own.
    """
    from ._base import _get_or_create_session, emit
    from ..models import EventType

    try:
        from langgraph.errors import GraphBubbleUp
    except ImportError:
        GraphBubbleUp = ()
    try:
        from langgraph.pregel._call import identifier
    except ImportError:
        try:
            from langgraph.pregel.call import identifier
        except ImportError:
            identifier = None

    class NodeRecorder(runnable):
        def __init__(self, bound: Any, name: str) -> None:
            self.bound = bound
            self.node_name = name
            key = identifier(bound) if identifier is not None else None
            if key:
                # LangGraph names a node's cache entries with identifier(node.bound).
                self.__module__, _, self.__qualname__ = key.rpartition(".")

        def __getattr__(self, name: str) -> Any:
            # LangGraph reads some attributes of node.bound, such as a ToolNode's tools_by_name.
            if name.startswith("__") or "bound" not in self.__dict__:
                raise AttributeError(name)
            return getattr(self.bound, name)

        def _begin(self) -> tuple[str, str, Any, float] | None:
            try:
                sid = _get_or_create_session(store, agent_name)
                emit(EventType.DECISION, sid, store, choice=self.node_name,
                     reason=f"LangGraph node: {self.node_name}")
                tool_name = f"langgraph.node.{self.node_name}"
                call = emit(EventType.TOOL_CALL, sid, store, tool_name=tool_name)
            except Exception:
                return None
            return sid, tool_name, call, time.time()

        def _finish(self, run: tuple[str, str, Any, float] | None,
                    exc: BaseException | None = None) -> None:
            if run is None:
                return
            sid, tool_name, call, t0 = run
            try:
                if exc is None:
                    emit(EventType.TOOL_RESULT, sid, store, parent=call,
                         tool_name=tool_name, duration_ms=(time.time() - t0) * 1000)
                elif isinstance(exc, (GraphBubbleUp, GeneratorExit)):
                    # interrupt(), NodeInterrupt and ParentCommand stop a node on
                    # purpose, and so does a caller that closes the node's stream.
                    emit(EventType.TOOL_RESULT, sid, store, parent=call,
                         tool_name=tool_name, duration_ms=(time.time() - t0) * 1000,
                         control_flow=type(exc).__name__)
                elif isinstance(exc, Exception):
                    emit(EventType.ERROR, sid, store, parent=call, tool_name=tool_name,
                         error=str(exc), error_type=type(exc).__name__)
                else:
                    # Cancelled, KeyboardInterrupt or SystemExit, as for a model call.
                    emit(EventType.TOOL_RESULT, sid, store, parent=call,
                         tool_name=tool_name, duration_ms=(time.time() - t0) * 1000,
                         status="cancelled")
            except Exception:
                pass

        def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
            run = self._begin()
            try:
                output = self.bound.invoke(input, config, **kwargs)
            except BaseException as exc:
                self._finish(run, exc)
                raise
            self._finish(run)
            return output

        async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
            run = self._begin()
            try:
                output = await self.bound.ainvoke(input, config, **kwargs)
            except BaseException as exc:
                self._finish(run, exc)
                raise
            self._finish(run)
            return output

        async def astream(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
            # LangGraph streams a node's runnable under astream_events.
            run = self._begin()
            try:
                chunks = self.bound.astream(input, config, **kwargs)
                try:
                    async for chunk in chunks:
                        yield chunk
                finally:
                    if hasattr(chunks, "aclose"):
                        await chunks.aclose()
            except GeneratorExit as exc:
                # Without a running loop the garbage collector is closing the
                # stream, where writing to the trace store is not safe.
                if _loop_is_running():
                    self._finish(run, exc)
                raise
            except BaseException as exc:
                self._finish(run, exc)
                raise
            self._finish(run)

    return NodeRecorder


def _wrap_langgraph_nodes(compiled_graph, store, agent_name: str) -> None:
    """Record each node run of a compiled LangGraph graph.

    Each node keeps its LangGraph node object. Only its ``bound`` runnable is
    replaced, by a recorder that calls the original, so the graph runs, streams
    and draws as it does untraced. A node of another shape is left untraced. If
    wrapping a node raises, no node is wrapped.
    """
    nodes = getattr(compiled_graph, "nodes", None)
    if not isinstance(nodes, dict):
        return
    try:
        from langchain_core.runnables import Runnable, RunnableSequence

        recorder = _node_recorder(Runnable, store, agent_name)
        updates = {}
        for name, node in nodes.items():
            if name in ("__start__", "__end__"):
                continue
            bound = getattr(node, "bound", None)
            if not isinstance(bound, Runnable) or not callable(getattr(node, "copy", None)):
                continue
            if isinstance(bound, RunnableSequence):
                # LangGraph runs a sequence's steps as the node's own steps.
                # A recorder around it would add a LangChain run.
                continue
            new = node.copy({"bound": recorder(bound, name)})
            subgraphs = getattr(node, "subgraphs", None)
            if getattr(new, "subgraphs", None) != subgraphs:
                # LangGraph 0.2 looks for subgraphs in the new bound and finds none.
                new.subgraphs = subgraphs
            updates[name] = new
    except Exception:
        return
    nodes.update(updates)


def uninstrument_langchain() -> None:
    """Remove patches (for testing)."""
    global _PATCHED, _orig_compile
    _TOOL_PATCHES.restore()
    _MODEL_PATCHES.restore()
    with _MODEL_CLASSES_LOCK:
        _MODEL_CLASSES.clear()
    if _orig_compile is not None:
        try:
            from langgraph.graph import StateGraph
            StateGraph.compile = _orig_compile
        except ImportError:
            pass
    _PATCHED = False
    _orig_compile = None
