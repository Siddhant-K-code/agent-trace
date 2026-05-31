"""Auto-instrumentation for CrewAI.

Patches:
  - crewai.Crew.kickoff          → session_start / session_end
  - crewai.Agent.execute_task    → llm_request / llm_response
  - crewai.Task.execute_sync     → tool_call / tool_result

Install:
    pip install agent-strace[crewai]
    # or: pip install crewai
"""

from __future__ import annotations

import time

_PATCHED = False
_orig_kickoff = None
_orig_execute_task = None
_orig_task_execute = None


def instrument_crewai(agent_name: str = "crewai") -> None:
    """Patch CrewAI to emit agent-trace events. Idempotent."""
    global _PATCHED, _orig_kickoff, _orig_execute_task, _orig_task_execute
    if _PATCHED:
        return

    try:
        import crewai  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "CrewAI is not installed. "
            "Install it with: pip install crewai"
        ) from exc

    from ._base import _get_store, _get_or_create_session, emit
    from ..models import EventType

    store = _get_store()

    # --- Patch Crew.kickoff ---
    try:
        from crewai import Crew

        _orig_kickoff = Crew.kickoff

        def _patched_kickoff(self, *args, **kwargs):
            sid = _get_or_create_session(store, agent_name)
            crew_name = getattr(self, "name", None) or agent_name
            emit(EventType.SESSION_START, sid, store,
                 agent_name=crew_name,
                 agent_count=len(getattr(self, "agents", [])),
                 task_count=len(getattr(self, "tasks", [])))
            t0 = time.time()
            try:
                result = _orig_kickoff(self, *args, **kwargs)
                emit(EventType.SESSION_END, sid, store,
                     duration_ms=(time.time() - t0) * 1000)
                return result
            except Exception as exc:
                emit(EventType.ERROR, sid, store,
                     error=str(exc), error_type=type(exc).__name__)
                raise

        Crew.kickoff = _patched_kickoff
    except (ImportError, AttributeError):
        pass

    # --- Patch Agent.execute_task ---
    try:
        from crewai import Agent

        _orig_execute_task = Agent.execute_task

        def _patched_execute_task(self, task, *args, **kwargs):
            sid = _get_or_create_session(store, agent_name)
            agent_role = getattr(self, "role", str(self))
            task_desc = getattr(task, "description", str(task))[:200]
            t0 = time.time()
            emit(EventType.LLM_REQUEST, sid, store,
                 agent_role=agent_role,
                 task=task_desc)
            try:
                result = _orig_execute_task(self, task, *args, **kwargs)
                emit(EventType.LLM_RESPONSE, sid, store,
                     agent_role=agent_role,
                     duration_ms=(time.time() - t0) * 1000,
                     result_preview=str(result)[:300])
                return result
            except Exception as exc:
                emit(EventType.ERROR, sid, store,
                     agent_role=agent_role, error=str(exc))
                raise

        Agent.execute_task = _patched_execute_task
    except (ImportError, AttributeError):
        pass

    # --- Patch Task.execute_sync (tool calls within a task) ---
    try:
        from crewai import Task

        _orig_task_execute = Task.execute_sync

        def _patched_task_execute(self, *args, **kwargs):
            sid = _get_or_create_session(store, agent_name)
            task_desc = getattr(self, "description", str(self))[:200]
            t0 = time.time()
            emit(EventType.TOOL_CALL, sid, store,
                 tool_name="crewai.task",
                 task=task_desc)
            try:
                result = _orig_task_execute(self, *args, **kwargs)
                emit(EventType.TOOL_RESULT, sid, store,
                     tool_name="crewai.task",
                     duration_ms=(time.time() - t0) * 1000,
                     result_preview=str(result)[:300])
                return result
            except Exception as exc:
                emit(EventType.ERROR, sid, store,
                     tool_name="crewai.task", error=str(exc))
                raise

        Task.execute_sync = _patched_task_execute
    except (ImportError, AttributeError):
        pass

    _PATCHED = True


def uninstrument_crewai() -> None:
    """Remove patches (for testing)."""
    global _PATCHED, _orig_kickoff, _orig_execute_task, _orig_task_execute
    if _orig_kickoff is not None:
        try:
            from crewai import Crew
            Crew.kickoff = _orig_kickoff
        except ImportError:
            pass
    if _orig_execute_task is not None:
        try:
            from crewai import Agent
            Agent.execute_task = _orig_execute_task
        except ImportError:
            pass
    if _orig_task_execute is not None:
        try:
            from crewai import Task
            Task.execute_sync = _orig_task_execute
        except ImportError:
            pass
    _PATCHED = False
    _orig_kickoff = None
    _orig_execute_task = None
    _orig_task_execute = None
