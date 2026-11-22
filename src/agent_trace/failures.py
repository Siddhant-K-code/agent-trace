"""Failed and cancelled model requests, and session error counts.

The exports (``otlp``, ``otlp-genai``, ``langfuse``) and every session error
count read the same definition from this module, so a session's error count
agrees with the error spans and observations it exports.

A model request **failed** when an ``error`` event is linked to its
``llm_request`` by ``parent_id`` (the SDK integrations and
``@trace_llm_call`` record a raised error this way), or when its
``llm_response`` has ``status`` ``failed`` or carries a non-empty ``error``
(Codex ``PostModelRequest``, OpenAI Responses API). A response whose
``status`` is ``cancelled`` or ``canceled`` is **cancelled**.

A session's error count is every ``error`` event plus every failed model
request that no ``error`` event already counts. A request counts once: a
failed ``llm_response`` adds nothing when an ``error`` linked to the same
``llm_request`` is in the trace, or when another failed response for that
request already counted. Responses are matched to requests with the LLM
channel rules in :mod:`pairing`, as the exporters do.

Cancelled requests export as errors but are not session errors. Codex
reports a request cancelled only when the user interrupted it or the turn was
aborted, and the SDK integrations only when the caller cancelled the task, so
a cancelled request does not make a session count as failed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
from typing import Any

from .event_data import event_id_of, parent_id_of
from .models import EventType, TraceEvent
from .pairing import LLM_REQUEST_TYPES, LLM_RESPONSE_TYPES, Pairer, llm_pairer


# error.type when nothing more specific was recorded: the GenAI semconv
# fallback value.
ERROR_TYPE_OTHER = "_OTHER"
# error.type for a cancelled request that recorded no error type. The GenAI
# semconv names no span status for a cancelled request, but ends its span and
# reports the finish reason "error" for it, so it exports as an error.
ERROR_TYPE_CANCELLED = "cancelled"
ERROR_MESSAGE_LIMIT = 200
FAILED_STATUSES = frozenset({"failed"})
CANCELLED_STATUSES = frozenset({"cancelled", "canceled"})
_ERROR_TYPE_KEYS = ("error_type", "exception_type")
_ERROR_CODE_KEYS = ("type", "code")


@dataclass(frozen=True)
class LLMFailure:
    """How a model request ended when it did not complete."""

    error_type: str
    """Low-cardinality value for the ``error.type`` attribute."""
    message: str = ""
    """Recorded error text, at most 200 characters. Empty when there is none."""
    cancelled: bool = False


def error_text(value: Any) -> str:
    """Return the text of a recorded ``error`` or ``message`` value."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        message = value.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str) if value else ""
    return str(value)


def _recorded_error_type(data: Mapping[str, Any], error: Any) -> str:
    for key in _ERROR_TYPE_KEYS:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:ERROR_MESSAGE_LIMIT]
    if isinstance(error, Mapping):
        for key in _ERROR_CODE_KEYS:
            value = error.get(key)
            if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value).strip():
                return str(value).strip()[:ERROR_MESSAGE_LIMIT]
    return ""


def llm_failure(event: TraceEvent) -> LLMFailure | None:
    """Return how the model request that ``event`` ends failed, or ``None``.

    ``event`` is an ``llm_response``, or an ``error`` linked to the
    ``llm_request`` it ends. An error always fails. A response is cancelled
    when its ``status`` is ``cancelled`` and failed when its ``status`` is
    ``failed`` or it carries a non-empty ``error``. Any other response,
    including one with no ``status``, completed and returns ``None``.

    ``error_type`` is the recorded ``error_type`` or ``exception_type``, else
    the ``type`` or ``code`` of an ``error`` object, else ``cancelled`` for a
    cancelled request and ``_OTHER`` for a failed one.
    """
    data = event.data if isinstance(event.data, Mapping) else {}
    error = data.get("error")
    text = error_text(error)
    cancelled = False
    if event.event_type == EventType.ERROR:
        text = text or error_text(data.get("message"))
    else:
        status = data.get("status")
        status = status.strip().lower() if isinstance(status, str) else ""
        cancelled = status in CANCELLED_STATUSES
        if not cancelled and status not in FAILED_STATUSES and not text:
            return None
    error_type = _recorded_error_type(data, error) or (
        ERROR_TYPE_CANCELLED if cancelled else ERROR_TYPE_OTHER
    )
    return LLMFailure(error_type, text[:ERROR_MESSAGE_LIMIT], cancelled)


def response_failure(event: TraceEvent) -> LLMFailure | None:
    """Return :func:`llm_failure` for an ``llm_response``; ``None`` for any other event."""
    if event.event_type not in LLM_RESPONSE_TYPES:
        return None
    return llm_failure(event)


def is_failed_response(event: TraceEvent) -> bool:
    """Return whether ``event`` is an ``llm_response`` that failed, not cancelled."""
    failure = response_failure(event)
    return failure is not None and not failure.cancelled


@dataclass(frozen=True)
class SessionErrors:
    """The errors of one session, from :func:`session_errors`."""

    error_events: int = 0
    """Every ``error`` event."""
    failed_requests: int = 0
    """Failed model requests that no ``error`` event already counts."""
    cancelled_requests: int = 0
    """Cancelled model requests. Not part of :attr:`errors`."""
    failed_responses: frozenset[int] = frozenset()
    """0-based positions of the ``llm_response`` events counted in
    :attr:`failed_requests`, one per request."""
    cancelled_responses: frozenset[int] = frozenset()
    """0-based positions of the ``llm_response`` events counted in
    :attr:`cancelled_requests`, one per request."""

    @property
    def errors(self) -> int:
        """The session error count: error events plus failed model requests."""
        return self.error_events + self.failed_requests


def session_errors(events: Iterable[TraceEvent]) -> SessionErrors:
    """Count the errors of one session's events, given in trace order."""
    requests: Pairer[int] = llm_pairer()
    request_positions: dict[str, int] = {}
    errored: set[int] = set()
    failed: dict[tuple[str, int], int] = {}
    cancelled: dict[tuple[str, int], int] = {}
    error_events = 0

    def request_named(event: TraceEvent) -> int | None:
        parent_id = parent_id_of(event)
        return request_positions.get(parent_id) if parent_id else None

    for index, event in enumerate(events):
        if event.event_type in LLM_REQUEST_TYPES:
            requests.add(event, index)
            event_id = event_id_of(event)
            if event_id:
                request_positions[event_id] = index
        elif event.event_type == EventType.ERROR:
            error_events += 1
            request = requests.match_parent(event)
            if request is None:
                request = request_named(event)
            if request is not None:
                errored.add(request)
        elif event.event_type in LLM_RESPONSE_TYPES:
            request = requests.match(event)
            if request is None:
                # A duplicate response, or one that follows a linked error,
                # names a request that no longer pends.
                request = request_named(event)
            failure = llm_failure(event)
            if failure is None:
                continue
            key = ("request", request) if request is not None else ("response", index)
            (cancelled if failure.cancelled else failed).setdefault(key, index)

    def ended_by_error(key: tuple[str, int]) -> bool:
        return key[0] == "request" and key[1] in errored

    failed_responses = frozenset(
        index for key, index in failed.items() if not ended_by_error(key)
    )
    cancelled_responses = frozenset(
        index for key, index in cancelled.items()
        if key not in failed and not ended_by_error(key)
    )
    return SessionErrors(
        error_events=error_events,
        failed_requests=len(failed_responses),
        cancelled_requests=len(cancelled_responses),
        failed_responses=failed_responses,
        cancelled_responses=cancelled_responses,
    )


def count_session_errors(events: Iterable[TraceEvent]) -> int:
    """Return the session error count of ``events``; see :func:`session_errors`."""
    return session_errors(events).errors


__all__ = [
    "CANCELLED_STATUSES",
    "ERROR_MESSAGE_LIMIT",
    "ERROR_TYPE_CANCELLED",
    "ERROR_TYPE_OTHER",
    "FAILED_STATUSES",
    "LLMFailure",
    "SessionErrors",
    "count_session_errors",
    "error_text",
    "is_failed_response",
    "llm_failure",
    "response_failure",
    "session_errors",
]
