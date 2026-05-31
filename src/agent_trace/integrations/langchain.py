"""Auto-instrumentation for LangChain / LangGraph.

Patches:
  - langchain_core.tools.BaseTool._run             → tool_call / tool_result
  - langchain_core.language_models.BaseChatModel._generate → llm_request / llm_response
  - langgraph.graph.StateGraph.compile             → wraps each node with
                                                     decision / tool_call events

Install:
    pip install agent-strace[langchain]
    # or: pip install langchain-core
    # LangGraph node tracing also requires: pip install langgraph
"""

from __future__ import annotations

import time

_PATCHED = False
_orig_tool_run = None
_orig_generate = None
_orig_compile = None


def instrument_langchain(agent_name: str = "langchain") -> None:
    """Patch LangChain (and LangGraph if installed) to emit agent-trace events. Idempotent."""
    global _PATCHED, _orig_tool_run, _orig_generate, _orig_compile
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

    # --- Patch BaseChatModel._generate ---
    try:
        from langchain_core.language_models import BaseChatModel

        _orig_generate = BaseChatModel._generate

        def _patched_generate(self, messages, *args, **kwargs):
            sid = _get_or_create_session(store, agent_name)
            model = getattr(self, "model_name", getattr(self, "model", "unknown"))
            t0 = time.time()
            emit(EventType.LLM_REQUEST, sid, store,
                 model=str(model), message_count=len(messages))
            try:
                result = _orig_generate(self, messages, *args, **kwargs)
                emit(EventType.LLM_RESPONSE, sid, store,
                     model=str(model),
                     duration_ms=(time.time() - t0) * 1000)
                return result
            except Exception as exc:
                emit(EventType.ERROR, sid, store,
                     error=str(exc), error_type=type(exc).__name__)
                raise

        BaseChatModel._generate = _patched_generate
    except (ImportError, AttributeError):
        pass

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
    global _PATCHED, _orig_tool_run, _orig_generate, _orig_compile
    if _orig_tool_run is not None:
        try:
            from langchain_core.tools import BaseTool
            BaseTool._run = _orig_tool_run
        except ImportError:
            pass
    if _orig_generate is not None:
        try:
            from langchain_core.language_models import BaseChatModel
            BaseChatModel._generate = _orig_generate
        except ImportError:
            pass
    if _orig_compile is not None:
        try:
            from langgraph.graph import StateGraph
            StateGraph.compile = _orig_compile
        except ImportError:
            pass
    _PATCHED = False
    _orig_tool_run = None
    _orig_generate = None
    _orig_compile = None
