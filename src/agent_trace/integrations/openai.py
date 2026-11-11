"""Auto-instrumentation for the OpenAI Python SDK.

Patches the SDK resource classes, so every ``OpenAI`` and ``AsyncOpenAI``
client is covered, including clients created before instrumenting:

  - ``client.chat.completions.create``  → llm_request / llm_response
  - ``client.responses.create``         → llm_request / llm_response

The ``chat.completions.stream`` and ``responses.stream`` helpers call
``create``, so they are recorded too. With ``stream=True`` the response is
recorded when the caller reaches the end of the stream; the stream itself is
unchanged. A ``with_raw_response`` or ``with_streaming_response`` call is
recorded when the caller parses its body. ``chat.completions.parse`` and
``responses.parse`` send their request without ``create`` and are not
recorded.

Install:
    pip install openai
"""

from __future__ import annotations

import importlib
from typing import Any

from ._base import Patches, count as _count, given, message_count

_PATCHED = False
_PATCHES = Patches()

# (module, class, method, is_async, API)
_TARGETS = (
    ("openai.resources.chat.completions", "Completions", "create", False, "chat"),
    ("openai.resources.chat.completions", "AsyncCompletions", "create", True, "chat"),
    ("openai.resources.responses", "Responses", "create", False, "responses"),
    ("openai.resources.responses", "AsyncResponses", "create", True, "responses"),
)


def _cache_fields(details: Any) -> dict[str, Any]:
    """Cache read and write counts of an OpenAI input token details object."""
    fields: dict[str, Any] = {}
    for source, key in (
        ("cached_tokens", "cached_input_tokens"),
        ("cache_write_tokens", "cache_write_input_tokens"),
    ):
        value = _count(getattr(details, source, None))
        if value is not None:
            fields[key] = value
    return fields


def _usage_fields(usage: Any) -> dict[str, Any]:
    """Token counts of a Chat Completions ``usage`` object.

    ``prompt_tokens`` includes the cached prompt tokens, which are billed at
    their own rates, so the cache read and write counts are kept as well.
    """
    fields: dict[str, Any] = {
        "input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
        "output_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
    }
    details = getattr(usage, "prompt_tokens_details", None) if usage else None
    fields.update(_cache_fields(details))
    return fields


def _responses_usage_fields(usage: Any) -> dict[str, Any]:
    """Token counts of a Responses API ``usage`` object.

    Like ``prompt_tokens``, its ``input_tokens`` includes the cached tokens.
    """
    fields: dict[str, Any] = {
        "input_tokens": _count(getattr(usage, "input_tokens", None)) or 0,
        "output_tokens": _count(getattr(usage, "output_tokens", None)) or 0,
    }
    fields.update(_cache_fields(getattr(usage, "input_tokens_details", None)))
    return fields


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _chat_finish_reason(result: Any) -> Any:
    choices = getattr(result, "choices", [])
    return choices[0].finish_reason if choices else ""


def _responses_fields(response: Any, *, with_usage: bool = True) -> dict[str, Any]:
    """Response fields of a Responses API ``Response`` object.

    ``status`` is kept because a ``failed`` or ``cancelled`` response is
    returned rather than raised.
    """
    usage = getattr(response, "usage", None)
    fields: dict[str, Any] = {}
    if with_usage or usage is not None:
        fields.update(_responses_usage_fields(usage))
    status = _text(getattr(response, "status", None))
    incomplete = _text(getattr(getattr(response, "incomplete_details", None), "reason", None))
    fields["finish_reason"] = incomplete or status
    if status:
        fields["status"] = status
    error = getattr(response, "error", None)
    message = _text(getattr(error, "message", None))
    if message:
        fields["error"] = message
        code = _text(getattr(error, "code", None))
        if code:
            fields["error_type"] = code
    return fields


class _ChatStream:
    """Usage and finish reason of a Chat Completions stream, from its chunks.

    Usage arrives only with ``stream_options={"include_usage": True}``, on
    the last chunk, so without it no token counts are recorded.
    """

    def __init__(self) -> None:
        self.usage: Any = None
        self.finish_reason: Any = ""

    def add(self, chunk: Any) -> None:
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            self.usage = usage
        choices = getattr(chunk, "choices", None)
        if not isinstance(choices, list):
            return
        for choice in choices:
            reason = getattr(choice, "finish_reason", None)
            if reason and getattr(choice, "index", 0) == 0:
                self.finish_reason = reason

    def fields(self) -> dict[str, Any]:
        fields = _usage_fields(self.usage) if self.usage is not None else {}
        fields["finish_reason"] = self.finish_reason
        return fields


class _ResponsesStream:
    """The last ``Response`` object a Responses API stream carried.

    ``response.completed``, ``response.incomplete`` and ``response.failed``
    carry the final response with its usage.
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self.response: Any = None

    def add(self, event: Any) -> None:
        if _text(getattr(event, "type", None)).startswith("response."):
            response = getattr(event, "response", None)
            if response is not None:
                self.response = response

    def fields(self) -> dict[str, Any]:
        return _with_model(_responses_fields(self.response, with_usage=False),
                           self.model, self.response)


def _with_model(fields: dict[str, Any], model: str, response: Any) -> dict[str, Any]:
    """Name the response's model when the request named none.

    A Responses API request can take its model from a stored prompt.
    """
    if not model:
        reported = _text(getattr(response, "model", None))
        if reported:
            fields["model"] = reported
    return fields


def _input_count(value: Any) -> int:
    """Number of input items of a Responses API request; a string is one."""
    return 1 if isinstance(value, str) and value else message_count(value)


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

    from ._base import (
        ModelCall, _get_store, record_stream, record_when_parsed, wrap_model_call,
    )

    store = _get_store()

    def start_chat(args: tuple, kwargs: dict) -> ModelCall:
        return ModelCall(store, agent_name, given(kwargs.get("model", ""), ""),
                         max_tokens=given(kwargs.get("max_tokens", 0), 0),
                         message_count=message_count(kwargs.get("messages")))

    def start_responses(args: tuple, kwargs: dict) -> ModelCall:
        return ModelCall(store, agent_name, given(kwargs.get("model", ""), ""),
                         max_tokens=given(kwargs.get("max_output_tokens", 0), 0),
                         message_count=_input_count(kwargs.get("input")))

    def finisher(api: str, is_async: bool):
        def finish(call: ModelCall, kwargs: dict, result: Any) -> None:
            if record_when_parsed(result, lambda body: finish(call, kwargs, body)):
                return
            if kwargs.get("stream"):
                stream = _ChatStream() if api == "chat" else _ResponsesStream(call.model)
                if record_stream(result, call, stream):
                    return
            if api == "chat":
                fields = _usage_fields(getattr(result, "usage", None))
                fields["finish_reason"] = _chat_finish_reason(result)
                # A sync call records an unlinked response, as it always has.
                call.respond(fields, linked=is_async)
            else:
                call.respond(_with_model(_responses_fields(result), call.model, result))

        return finish

    for module_name, class_name, method, is_async, api in _TARGETS:
        try:
            owner = getattr(importlib.import_module(module_name), class_name)
        except (ImportError, AttributeError):
            continue
        start = start_chat if api == "chat" else start_responses
        finish = finisher(api, is_async)
        _PATCHES.wrap(owner, method, lambda original, s=start, f=finish, a=is_async:
                      wrap_model_call(original, s, f, is_async=a))

    _PATCHED = True


def uninstrument_openai() -> None:
    """Remove patches (for testing)."""
    global _PATCHED
    _PATCHES.restore()
    _PATCHED = False
