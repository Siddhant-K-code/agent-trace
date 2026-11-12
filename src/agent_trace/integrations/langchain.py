"""Auto-instrumentation for LangChain / LangGraph.

Patches:
  - langchain_core.tools.BaseTool._run             → tool_call / tool_result
  - chat models' ``_generate`` and ``_agenerate``  → llm_request / llm_response
  - langgraph.graph.StateGraph.compile             → wraps each node with
                                                     decision / tool_call events

Chat model classes override ``_generate``, so the class that defines a
model's ``_generate`` and ``_agenerate`` is patched the first time the model
is invoked. A response served from the LangChain LLM cache, and output read
with ``stream`` or ``astream`` (or with ``streaming=True``), is not recorded.

Install:
    pip install agent-strace[langchain]
    # or: pip install langchain-core
    # LangGraph node tracing also requires: pip install langgraph
"""

from __future__ import annotations

import functools
import threading
import time
from typing import Any

from ._base import Patches, count as _count, message_count

_PATCHED = False
_orig_tool_run = None
_orig_compile = None
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
    global _PATCHED, _orig_tool_run, _orig_compile
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

    # --- Patch BaseTool._run ---
    try:
        from langchain_core.tools import BaseTool

        _orig_tool_run = BaseTool._run

        def _patched_run(self, *args, **kwargs):
            sid = _get_or_create_session(store, agent_name)
            tool_name = getattr(self, "name", str(self))
            t0 = time.time()
            emit(EventType.TOOL_CALL, sid, store,
                 tool_name=tool_name, arguments={"args": str(args)[:200], **kwargs})
            try:
                result = _orig_tool_run(self, *args, **kwargs)
                emit(EventType.TOOL_RESULT, sid, store,
                     tool_name=tool_name,
                     result=str(result)[:500],
                     duration_ms=(time.time() - t0) * 1000)
                return result
            except Exception as exc:
                emit(EventType.ERROR, sid, store,
                     tool_name=tool_name, error=str(exc))
                raise

        BaseTool._run = _patched_run
    except (ImportError, AttributeError):
        pass

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


def _wrap_langgraph_nodes(compiled_graph, store, agent_name: str) -> None:
    """Wrap each node in a compiled LangGraph graph to emit decision events."""
    from ._base import _get_or_create_session, emit
    from ..models import EventType

    nodes = getattr(compiled_graph, "nodes", None)
    if not isinstance(nodes, dict):
        return

    for node_name, node_fn in list(nodes.items()):
        if node_name in ("__start__", "__end__"):
            continue

        def _make_wrapper(name, fn):
            def _wrapped(state, *args, **kwargs):
                sid = _get_or_create_session(store, agent_name)
                t0 = time.time()
                emit(EventType.DECISION, sid, store,
                     choice=name,
                     reason=f"LangGraph node: {name}")
                try:
                    result = fn(state, *args, **kwargs)
                    emit(EventType.TOOL_RESULT, sid, store,
                         tool_name=f"langgraph.node.{name}",
                         duration_ms=(time.time() - t0) * 1000)
                    return result
                except Exception as exc:
                    emit(EventType.ERROR, sid, store,
                         tool_name=f"langgraph.node.{name}",
                         error=str(exc))
                    raise
            return _wrapped

        nodes[node_name] = _make_wrapper(node_name, node_fn)


def uninstrument_langchain() -> None:
    """Remove patches (for testing)."""
    global _PATCHED, _orig_tool_run, _orig_compile
    if _orig_tool_run is not None:
        try:
            from langchain_core.tools import BaseTool
            BaseTool._run = _orig_tool_run
        except ImportError:
            pass
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
    _orig_tool_run = None
    _orig_compile = None
