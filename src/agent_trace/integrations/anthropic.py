"""Auto-instrumentation for the Anthropic Python SDK.

Patches the SDK resource classes, so every ``Anthropic`` and
``AsyncAnthropic`` client is covered, including clients created before
instrumenting:

  - ``client.messages.create``  → llm_request / llm_response

With ``stream=True`` the response is recorded when the caller reaches the
end of the stream; the stream itself is unchanged. A
``messages.with_raw_response.create`` or ``with_streaming_response`` call is
recorded when the caller parses its body. ``messages.stream``,
``messages.parse`` and ``client.beta.messages`` send their request without
``messages.create`` and are not recorded.

Install:
    pip install anthropic
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

from ._base import Patches, count as _count, given, message_count

_PATCHED = False
_PATCHES = Patches()

# (module, class, method, is_async)
_TARGETS = (
    ("anthropic.resources.messages", "Messages", "create", False),
    ("anthropic.resources.messages", "AsyncMessages", "create", True),
)

_CACHE_KEYS = ("cache_read_input_tokens", "cache_creation_input_tokens")


def _usage_fields(usage: Any) -> dict[str, Any]:
    """Token counts of an Anthropic ``usage`` object.

    ``input_tokens`` excludes cache reads and writes, which Anthropic reports
    beside it and bills at their own rates, so those are kept as well.
    """
    fields: dict[str, Any] = {
        "input_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
        "output_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
    }
    if not usage:
        return fields
    for key in _CACHE_KEYS:
        value = _count(getattr(usage, key, None))
        if value is not None:
            fields[key] = value
    split = getattr(usage, "cache_creation", None)
    ttl = {
        key: value
        for key in ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens")
        if (value := _count(getattr(split, key, None))) is not None
    }
    if ttl:
        fields["cache_creation"] = ttl
    return fields


class _MessageStream:
    """Usage and stop reason of a Messages stream, from its events.

    ``message_start`` carries the input and cache counts; ``message_delta``
    carries the final output count and the stop reason, and may carry final
    input and cache counts too.
    """

    def __init__(self) -> None:
        self.usage: dict[str, Any] = {}
        self.cache_creation: Any = None
        self.stop_reason: Any = ""

    def _take(self, usage: Any, keys: tuple[str, ...]) -> None:
        for key in keys:
            value = _count(getattr(usage, key, None))
            if value is not None:
                self.usage[key] = value

    def add(self, event: Any) -> None:
        kind = getattr(event, "type", None)
        if kind == "message_start":
            usage = getattr(getattr(event, "message", None), "usage", None)
            self._take(usage, ("input_tokens", "output_tokens", *_CACHE_KEYS))
            split = getattr(usage, "cache_creation", None)
            if split is not None:
                self.cache_creation = split
        elif kind == "message_delta":
            self._take(getattr(event, "usage", None),
                       ("input_tokens", "output_tokens", *_CACHE_KEYS))
            reason = getattr(getattr(event, "delta", None), "stop_reason", None)
            if reason:
                self.stop_reason = reason

    def fields(self) -> dict[str, Any]:
        usage = SimpleNamespace(
            input_tokens=self.usage.get("input_tokens", 0),
            output_tokens=self.usage.get("output_tokens", 0),
            cache_read_input_tokens=self.usage.get("cache_read_input_tokens"),
            cache_creation_input_tokens=self.usage.get("cache_creation_input_tokens"),
            cache_creation=self.cache_creation,
        )
        fields = _usage_fields(usage)
        fields["stop_reason"] = self.stop_reason
        return fields


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

    from ._base import (
        ModelCall, _get_store, record_stream, record_when_parsed, wrap_model_call,
    )

    store = _get_store()

    def start(args: tuple, kwargs: dict) -> ModelCall:
        return ModelCall(store, agent_name, given(kwargs.get("model", ""), ""),
                         max_tokens=given(kwargs.get("max_tokens", 0), 0),
                         message_count=message_count(kwargs.get("messages")))

    def finisher(is_async: bool):
        def finish(call: ModelCall, kwargs: dict, result: Any) -> None:
            if record_when_parsed(result, lambda body: finish(call, kwargs, body)):
                return
            if kwargs.get("stream") and record_stream(result, call, _MessageStream()):
                return
            fields = _usage_fields(getattr(result, "usage", None))
            fields["stop_reason"] = getattr(result, "stop_reason", "")
            # A sync call records an unlinked response, as it always has.
            call.respond(fields, linked=is_async)

        return finish

    for module_name, class_name, method, is_async in _TARGETS:
        try:
            owner = getattr(importlib.import_module(module_name), class_name)
        except (ImportError, AttributeError):
            continue
        finish = finisher(is_async)
        _PATCHES.wrap(owner, method, lambda original, f=finish, a=is_async:
                      wrap_model_call(original, start, f, is_async=a))

    _PATCHED = True


def uninstrument_anthropic() -> None:
    """Remove patches (for testing)."""
    global _PATCHED
    _PATCHES.restore()
    _PATCHED = False
