"""Auto-instrumentation for AWS Strands Agents.

Patches:
  - strands.Agent.__call__           → one session per call:
                                       session_start / session_end
  - strands.tools.BaseTool.invoke    → tool_call / tool_result

Install:
    pip install agent-strace[strands]
    # or: pip install strands-agents
"""

from __future__ import annotations

import functools
import time
from typing import Any

from ._base import Patches

_PATCHED = False
_PATCHES = Patches()


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

    from ._base import _get_store, _get_or_create_session, emit, run_session
    from ..models import EventType

    store = _get_store()

    # --- Patch Agent.__call__ ---
    try:
        from strands import Agent

        def wrap_agent_call(original: Any) -> Any:
            @functools.wraps(original)
            def _patched_agent_call(self, *args, **kwargs):
                name = getattr(self, "name", agent_name)
                with run_session(store, agent_name, agent_name=name):
                    return original(self, *args, **kwargs)

            return _patched_agent_call

        _PATCHES.wrap(Agent, "__call__", wrap_agent_call)
    except (ImportError, AttributeError):
        pass

    # --- Patch BaseTool.invoke ---
    try:
        from strands.tools import BaseTool

        def wrap_invoke(original: Any) -> Any:
            @functools.wraps(original)
            def _patched_invoke(self, *args, **kwargs):
                sid = _get_or_create_session(store, agent_name)
                tool_name = getattr(self, "name", str(self))
                t0 = time.time()
                call = emit(EventType.TOOL_CALL, sid, store,
                            tool_name=tool_name, arguments=kwargs)
                try:
                    result = original(self, *args, **kwargs)
                    emit(EventType.TOOL_RESULT, sid, store, parent=call,
                         tool_name=tool_name,
                         result=str(result)[:500],
                         duration_ms=(time.time() - t0) * 1000)
                    return result
                except Exception as exc:
                    emit(EventType.ERROR, sid, store, parent=call,
                         tool_name=tool_name, error=str(exc),
                         error_type=type(exc).__name__)
                    raise

            return _patched_invoke

        _PATCHES.wrap(BaseTool, "invoke", wrap_invoke)
    except (ImportError, AttributeError):
        pass

    _PATCHED = True


def uninstrument_strands() -> None:
    """Remove patches (for testing)."""
    global _PATCHED
    _PATCHES.restore()
    _PATCHED = False
