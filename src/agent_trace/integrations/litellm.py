"""Auto-instrumentation for LiteLLM.

Patches:
  - litellm.completion   → llm_request / llm_response
  - litellm.acompletion  → llm_request / llm_response (async)

Code that imported ``completion`` or ``acompletion`` from ``litellm`` before
instrumenting keeps the unpatched function. With ``stream=True`` the
response is recorded when the call returns the stream, without token counts,
because the model has not answered yet.

Install:
    pip install agent-strace[litellm]
    # or: pip install litellm
"""

from __future__ import annotations

from typing import Any

from ._base import Patches, count as _count, message_count

_PATCHED = False
_PATCHES = Patches()


def _usage_fields(usage: Any) -> dict[str, Any]:
    """Token counts of a LiteLLM ``usage`` object.

    LiteLLM reports usage in the OpenAI shape for every provider:
    ``prompt_tokens`` includes cache reads and writes, and
    ``prompt_tokens_details`` holds their counts. Older releases report
    some providers' cache counts only as ``cache_read_input_tokens`` and
    ``cache_creation_input_tokens``, which are used when the details lack
    them.
    """
    fields: dict[str, Any] = {
        "input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
        "output_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
    }
    if not usage:
        return fields
    details = getattr(usage, "prompt_tokens_details", None)
    for detail, fallback, key in (
        ("cached_tokens", "cache_read_input_tokens", "cached_input_tokens"),
        ("cache_write_tokens", "cache_creation_input_tokens", "cache_write_input_tokens"),
    ):
        value = _count(getattr(details, detail, None))
        if not value:
            value = _count(getattr(usage, fallback, None)) or value
        if value is not None:
            fields[key] = value
    return fields


def _argument(args: tuple, kwargs: dict, index: int, name: str, default: Any) -> Any:
    if name in kwargs:
        return kwargs[name]
    return args[index] if len(args) > index else default


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
    from ._base import ModelCall, _get_store, wrap_model_call

    store = _get_store()

    def start(args: tuple, kwargs: dict) -> ModelCall:
        return ModelCall(store, agent_name, _argument(args, kwargs, 0, "model", ""),
                         message_count=message_count(_argument(args, kwargs, 1, "messages", None)))

    def finish(call: ModelCall, kwargs: dict, result: Any) -> None:
        usage = getattr(result, "usage", None)
        # A stream is returned before the model answers, so its counts are unknown.
        fields = {} if usage is None and kwargs.get("stream") else _usage_fields(usage)
        call.respond(fields)

    for name, is_async in (("completion", False), ("acompletion", True)):
        _PATCHES.wrap(litellm, name, lambda original, a=is_async:
                      wrap_model_call(original, start, finish, is_async=a))

    _PATCHED = True


def uninstrument_litellm() -> None:
    """Remove patches (for testing)."""
    global _PATCHED
    _PATCHES.restore()
    _PATCHED = False
