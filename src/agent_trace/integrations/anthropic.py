"""Auto-instrumentation for the Anthropic Python SDK.

Patches:
  - anthropic.Anthropic.messages.create       → llm_request / llm_response
  - anthropic.AsyncAnthropic.messages.create  → llm_request / llm_response

Install:
    pip install anthropic
"""

from __future__ import annotations

import time

_PATCHED = False


def instrument_anthropic(agent_name: str = "anthropic") -> None:
    """Patch the Anthropic SDK to emit agent-trace events. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return

    try:
        import anthropic  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Anthropic SDK is not installed. "
            "Install it with: pip install anthropic"
        ) from exc

    import anthropic
    from ._base import _get_store, _get_or_create_session, emit
    from ..models import EventType

    store = _get_store()

    # Patch sync client
    try:
        _orig_create = anthropic.Anthropic.messages.create

        def _patched_create(self_messages, model: str = "", messages=None, **kwargs):
            sid = _get_or_create_session(store, agent_name)
            t0 = time.time()
            emit(EventType.LLM_REQUEST, sid, store,
                 model=model,
                 max_tokens=kwargs.get("max_tokens", 0),
                 message_count=len(messages or []))
            try:
                result = _orig_create(self_messages, model=model, messages=messages, **kwargs)
                usage = getattr(result, "usage", None)
                emit(EventType.LLM_RESPONSE, sid, store,
                     model=model,
                     input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
                     output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
                     stop_reason=getattr(result, "stop_reason", ""),
                     duration_ms=(time.time() - t0) * 1000)
                return result
            except Exception as exc:
                emit(EventType.ERROR, sid, store,
                     error=str(exc), error_type=type(exc).__name__)
                raise

        anthropic.Anthropic.messages.create = _patched_create
    except AttributeError:
        pass

    _PATCHED = True


def uninstrument_anthropic() -> None:
    """Remove patches (for testing)."""
    global _PATCHED
    _PATCHED = False
