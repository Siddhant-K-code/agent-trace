"""Auto-instrumentation for LiteLLM.

Patches:
  - litellm.completion   → llm_request / llm_response
  - litellm.acompletion  → llm_request / llm_response (async)

Install:
    pip install agent-strace[litellm]
    # or: pip install litellm
"""

from __future__ import annotations

import time

_PATCHED = False


def instrument_litellm(agent_name: str = "litellm") -> None:
    """Patch LiteLLM to emit agent-trace events. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return

    try:
        import litellm  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "LiteLLM is not installed. "
            "Install it with: pip install litellm"
        ) from exc

    import litellm
    from ._base import _get_store, _get_or_create_session, emit
    from ..models import EventType

    store = _get_store()

    _orig_completion = litellm.completion

    def _patched_completion(model: str = "", messages=None, **kwargs):
        sid = _get_or_create_session(store, agent_name)
        t0 = time.time()
        emit(EventType.LLM_REQUEST, sid, store,
             model=model, message_count=len(messages or []))
        try:
            result = _orig_completion(model=model, messages=messages, **kwargs)
            usage = getattr(result, "usage", None)
            emit(EventType.LLM_RESPONSE, sid, store,
                 model=model,
                 input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                 output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
                 duration_ms=(time.time() - t0) * 1000)
            return result
        except Exception as exc:
            emit(EventType.ERROR, sid, store,
                 error=str(exc), error_type=type(exc).__name__)
            raise

    litellm.completion = _patched_completion

    _PATCHED = True


def uninstrument_litellm() -> None:
    """Remove patches (for testing)."""
    global _PATCHED
    _PATCHED = False
