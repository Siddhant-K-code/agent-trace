"""Auto-instrumentation for AWS Strands Agents.

Patches:
  - strands.Agent.__call__           → session start/end
  - strands.tools.BaseTool.invoke    → tool_call / tool_result
  - strands.models.BedrockModel.*    → llm_request / llm_response

Install:
    pip install agent-strace[strands]
    # or: pip install strands-agents
"""

from __future__ import annotations

import time

_PATCHED = False


def instrument_strands(agent_name: str = "strands") -> None:
    """Patch AWS Strands Agents to emit agent-trace events. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return

    try:
        import strands  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "AWS Strands Agents is not installed. "
            "Install it with: pip install strands-agents"
        ) from exc

    from ._base import _get_store, _get_or_create_session, emit
    from ..models import EventType

    store = _get_store()

    # --- Patch Agent.__call__ ---
    try:
        from strands import Agent

        _orig_call = Agent.__call__

        def _patched_agent_call(self, *args, **kwargs):
            sid = _get_or_create_session(store, agent_name)
            name = getattr(self, "name", agent_name)
            emit(EventType.SESSION_START, sid, store, agent_name=name)
            t0 = time.time()
            try:
                result = _orig_call(self, *args, **kwargs)
                emit(EventType.SESSION_END, sid, store,
                     duration_ms=(time.time() - t0) * 1000)
                return result
            except Exception as exc:
                emit(EventType.ERROR, sid, store,
                     error=str(exc), error_type=type(exc).__name__)
                raise

        Agent.__call__ = _patched_agent_call
    except (ImportError, AttributeError):
        pass

    # --- Patch BaseTool.invoke ---
    try:
        from strands.tools import BaseTool

        _orig_invoke = BaseTool.invoke

        def _patched_invoke(self, *args, **kwargs):
            sid = _get_or_create_session(store, agent_name)
            tool_name = getattr(self, "name", str(self))
            t0 = time.time()
            emit(EventType.TOOL_CALL, sid, store,
                 tool_name=tool_name, arguments=kwargs)
            try:
                result = _orig_invoke(self, *args, **kwargs)
                emit(EventType.TOOL_RESULT, sid, store,
                     tool_name=tool_name,
                     result=str(result)[:500],
                     duration_ms=(time.time() - t0) * 1000)
                return result
            except Exception as exc:
                emit(EventType.ERROR, sid, store,
                     tool_name=tool_name, error=str(exc))
                raise

        BaseTool.invoke = _patched_invoke
    except (ImportError, AttributeError):
        pass

    _PATCHED = True


def uninstrument_strands() -> None:
    """Remove patches (for testing)."""
    global _PATCHED
    _PATCHED = False
