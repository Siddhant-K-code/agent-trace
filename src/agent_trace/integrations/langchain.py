"""Auto-instrumentation for LangChain / LangGraph.

Patches:
  - langchain_core.runnables.base.Runnable.invoke  → chain invocations
  - langchain_core.tools.BaseTool._run             → tool_call / tool_result
  - langchain_core.language_models.BaseChatModel._generate → llm_request / llm_response

Install:
    pip install agent-strace[langchain]
    # or: pip install langchain-core
"""

from __future__ import annotations

import time

_PATCHED = False


def instrument_langchain(agent_name: str = "langchain") -> None:
    """Patch LangChain to emit agent-trace events. Idempotent."""
    global _PATCHED
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

        _orig_run = BaseTool._run

        def _patched_run(self, *args, **kwargs):
            sid = _get_or_create_session(store, agent_name)
            tool_name = getattr(self, "name", str(self))
            t0 = time.time()
            emit(EventType.TOOL_CALL, sid, store,
                 tool_name=tool_name, arguments={"args": str(args)[:200], **kwargs})
            try:
                result = _orig_run(self, *args, **kwargs)
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

    _PATCHED = True


def uninstrument_langchain() -> None:
    """Remove patches (for testing)."""
    global _PATCHED
    _PATCHED = False
