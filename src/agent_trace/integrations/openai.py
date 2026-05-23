"""Auto-instrumentation for the OpenAI Python SDK.

Patches:
  - openai.OpenAI.chat.completions.create       → llm_request / llm_response
  - openai.AsyncOpenAI.chat.completions.create  → llm_request / llm_response

Install:
    pip install openai
"""

from __future__ import annotations

import time

_PATCHED = False


def instrument_openai(agent_name: str = "openai") -> None:
    """Patch the OpenAI SDK to emit agent-trace events. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return

    try:
        import openai  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "OpenAI SDK is not installed. "
            "Install it with: pip install openai"
        ) from exc

    import openai
    from ._base import _get_store, _get_or_create_session, emit
    from ..models import EventType

    store = _get_store()

    try:
        _orig_create = openai.OpenAI.chat.completions.create

        def _patched_create(self_completions, model: str = "", messages=None, **kwargs):
            sid = _get_or_create_session(store, agent_name)
            t0 = time.time()
            emit(EventType.LLM_REQUEST, sid, store,
                 model=model,
                 max_tokens=kwargs.get("max_tokens", 0),
                 message_count=len(messages or []))
            try:
                result = _orig_create(self_completions, model=model, messages=messages, **kwargs)
                usage = getattr(result, "usage", None)
                choices = getattr(result, "choices", [])
                finish = choices[0].finish_reason if choices else ""
                emit(EventType.LLM_RESPONSE, sid, store,
                     model=model,
                     input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                     output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
                     finish_reason=finish,
                     duration_ms=(time.time() - t0) * 1000)
                return result
            except Exception as exc:
                emit(EventType.ERROR, sid, store,
                     error=str(exc), error_type=type(exc).__name__)
                raise

        openai.OpenAI.chat.completions.create = _patched_create
    except AttributeError:
        pass

    _PATCHED = True


def uninstrument_openai() -> None:
    """Remove patches (for testing)."""
    global _PATCHED
    _PATCHED = False
