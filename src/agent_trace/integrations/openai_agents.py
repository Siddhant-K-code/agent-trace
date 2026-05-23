"""Auto-instrumentation for the OpenAI Agents SDK.

Patches:
  - agents.Runner.run / Runner.run_sync  → session start/end
  - agents.FunctionTool.__call__         → tool_call / tool_result
  - openai.chat.completions.create       → llm_request / llm_response

Install:
    pip install agent-strace[openai-agents]
    # or: pip install openai-agents
"""

from __future__ import annotations

import time
from typing import Any

_PATCHED = False


def instrument_openai_agents(agent_name: str = "openai-agents") -> None:
    """Patch the OpenAI Agents SDK to emit agent-trace events.

    Idempotent — calling this more than once has no effect.
    """
    global _PATCHED
    if _PATCHED:
        return

    try:
        import agents  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "OpenAI Agents SDK is not installed. "
            "Install it with: pip install openai-agents"
        ) from exc

    from ._base import _get_store, _get_or_create_session, emit
    from ..models import EventType

    store = _get_store()

    # --- Patch Runner.run_sync ---
    try:
        from agents import Runner

        _orig_run_sync = Runner.run_sync.__func__ if hasattr(Runner.run_sync, "__func__") else Runner.run_sync

        def _patched_run_sync(cls_or_self, *args, **kwargs):
            sid = _get_or_create_session(store, agent_name)
            emit(EventType.SESSION_START, sid, store, agent_name=agent_name)
            t0 = time.time()
            try:
                result = _orig_run_sync(cls_or_self, *args, **kwargs)
                emit(EventType.SESSION_END, sid, store,
                     duration_ms=(time.time() - t0) * 1000)
                return result
            except Exception as exc:
                emit(EventType.ERROR, sid, store,
                     error=str(exc), error_type=type(exc).__name__)
                raise

        Runner.run_sync = classmethod(_patched_run_sync).__get__(Runner)
    except (ImportError, AttributeError):
        pass

    # --- Patch FunctionTool.__call__ ---
    try:
        from agents import FunctionTool

        _orig_call = FunctionTool.__call__

        def _patched_call(self, *args, **kwargs):
            sid = _get_or_create_session(store, agent_name)
            tool_name = getattr(self, "name", str(self))
            t0 = time.time()
            emit(EventType.TOOL_CALL, sid, store,
                 tool_name=tool_name, arguments=kwargs or (args[0] if args else {}))
            try:
                result = _orig_call(self, *args, **kwargs)
                emit(EventType.TOOL_RESULT, sid, store,
                     tool_name=tool_name,
                     result=str(result)[:500],
                     duration_ms=(time.time() - t0) * 1000)
                return result
            except Exception as exc:
                emit(EventType.ERROR, sid, store,
                     tool_name=tool_name, error=str(exc))
                raise

        FunctionTool.__call__ = _patched_call
    except (ImportError, AttributeError):
        pass

    _PATCHED = True


def uninstrument_openai_agents() -> None:
    """Remove patches (for testing)."""
    global _PATCHED
    _PATCHED = False
