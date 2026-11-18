"""Auto-instrumentation for the OpenAI Agents SDK.

Patches:
  - agents.Runner.run_sync               → one session per run:
                                           session_start / session_end
  - agents.FunctionTool.__call__         → tool_call / tool_result

Install:
    pip install agent-strace[openai-agents]
    # or: pip install openai-agents
"""

from __future__ import annotations

import functools
import time
from typing import Any

from ._base import Patches, is_wrapped

_PATCHED = False
_PATCHES = Patches()


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

    from ._base import _get_store, _get_or_create_session, emit, run_session
    from ..models import EventType

    store = _get_store()

    # --- Patch Runner.run_sync ---
    try:
        from agents import Runner

        def wrap_run_sync(original: Any) -> Any:
            function = getattr(original, "__func__", original)

            @functools.wraps(function)
            def _patched_run_sync(cls, *args, **kwargs):
                with run_session(store, agent_name, agent_name=agent_name):
                    return function(cls, *args, **kwargs)

            return classmethod(_patched_run_sync)

        # Runner.run_sync is a classmethod, so the attribute read from the
        # class is a bound method that Patches cannot recognise as a wrapper.
        if not is_wrapped(vars(Runner).get("run_sync")):
            _PATCHES.wrap(Runner, "run_sync", wrap_run_sync)
    except (ImportError, AttributeError):
        pass

    # --- Patch FunctionTool.__call__ ---
    try:
        from agents import FunctionTool

        def wrap_call(original: Any) -> Any:
            @functools.wraps(original)
            def _patched_call(self, *args, **kwargs):
                sid = _get_or_create_session(store, agent_name)
                tool_name = getattr(self, "name", str(self))
                t0 = time.time()
                call = emit(EventType.TOOL_CALL, sid, store,
                            tool_name=tool_name,
                            arguments=kwargs or (args[0] if args else {}))
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

            return _patched_call

        _PATCHES.wrap(FunctionTool, "__call__", wrap_call)
    except (ImportError, AttributeError):
        pass

    _PATCHED = True


def uninstrument_openai_agents() -> None:
    """Remove patches (for testing)."""
    global _PATCHED
    _PATCHES.restore()
    _PATCHED = False
