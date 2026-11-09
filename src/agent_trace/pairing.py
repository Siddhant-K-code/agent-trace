"""Pair request events with their response events in a stored trace.

Every exporter that turns a trace into spans, observations or usage samples
must decide which response belongs to which request. This module defines
that decision once so the same trace pairs the same way in every exporter.

Rules, applied in order to each response as the trace is read:

1. **parent_id.** If the response has a ``parent_id`` that names a pending
   request, the response pairs with that request and the request stops
   being pending.
2. **Fallback.** Only when rule 1 finds nothing, and only when the response
   has no ``parent_id`` or its ``parent_id`` names no request seen so far,
   the pairer applies its fallback policy:

   - ``NO_FALLBACK``: the response stays unpaired.
   - ``NEWEST_SAME_MODEL``: the response pairs with the most recent
     pending request that names the same model as the response. Two events
     that both name no model count as the same model. Only when no pending
     request names the same model does the response pair with the most
     recent pending request where exactly one side names no model. A
     request and a response that name different models never pair. The
     model is the only evidence the fallback has, so a request whose model
     agrees with the response always beats one that names no model: a
     model-less request, such as a CrewAI task, never takes a response from
     an older request for the response's model, and a model-less response
     never takes a model-named request while a model-less one is pending.
     The model-less match remains as a last resort because some
     integrations record the model on only one side of a call. The most
     recent request wins because SDK integrations record a failed call as
     an ``error`` with no response, which leaves its request pending.
     Taking the oldest request would then pair every later response with
     the request before it.

   A response whose ``parent_id`` names a request that already paired is a
   duplicate. It stays unpaired, so it never takes another request's place.
3. Otherwise the response is unpaired.

A request with a non-empty ``event_id`` replaces a pending request with the
same ID, which keeps its place in the pending order. Requests with an empty
``event_id`` never replace each other. Requests that are still pending at
the end are returned by :meth:`Pairer.unmatched` in the order they arrived.

The standard channels are:

- Tool: ``tool_call`` requests, ``tool_result`` and ``error`` responses,
  ``NO_FALLBACK``. Tool calls often run in parallel, so a result without a
  ``parent_id`` cannot be placed safely.
- LLM: ``llm_request`` requests, ``llm_response`` responses,
  ``NEWEST_SAME_MODEL``. Several SDK integrations record responses without
  a ``parent_id``.

A caller may feed more event types to a pairer, for example cost reporting
also treats ``assistant_response`` as an LLM response because it carries
usage, but the rules above stay the same.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from .models import EventType, TraceEvent

NO_FALLBACK = "none"
NEWEST_SAME_MODEL = "newest_same_model"
FALLBACK_POLICIES = frozenset({NO_FALLBACK, NEWEST_SAME_MODEL})

TOOL_REQUEST_TYPES = frozenset({EventType.TOOL_CALL})
TOOL_RESPONSE_TYPES = frozenset({EventType.TOOL_RESULT, EventType.ERROR})
TOOL_FALLBACK = NO_FALLBACK

LLM_REQUEST_TYPES = frozenset({EventType.LLM_REQUEST})
LLM_RESPONSE_TYPES = frozenset({EventType.LLM_RESPONSE})
LLM_FALLBACK = NEWEST_SAME_MODEL

_MODEL_KEYS = ("model", "model_id", "modelId")

T = TypeVar("T")


def event_model(data: Any) -> str:
    """Return the model an event's data names, or an empty string."""
    if not isinstance(data, Mapping):
        return ""
    for key in _MODEL_KEYS:
        value = data.get(key)
        if value:
            return str(value).strip()
    return ""


class Pairer(Generic[T]):
    """Pair responses with pending requests, one event at a time.

    Feed events in trace order: :meth:`add` for each request and
    :meth:`match` for each response. Each request carries an item, which is
    the event itself unless the caller passes something else, such as its
    index. Items must not be ``None``.
    """

    def __init__(self, fallback: str = NO_FALLBACK) -> None:
        if fallback not in FALLBACK_POLICIES:
            raise ValueError(
                f"unknown fallback policy {fallback!r}; "
                f"use one of {sorted(FALLBACK_POLICIES)}"
            )
        self.fallback = fallback
        self._pending: dict[object, tuple[str, Any]] = {}
        self._seen: set[str] = set()

    def add(self, request: TraceEvent, item: Any = None) -> None:
        """Make ``request`` pending, carrying ``item`` (default: the request)."""
        if item is None:
            item = request
        key: object = request.event_id or object()
        if request.event_id:
            self._seen.add(request.event_id)
        self._pending[key] = (event_model(request.data), item)

    def match(self, response: TraceEvent) -> T | None:
        """Return the item of the request ``response`` pairs with, or ``None``."""
        parent_id = response.parent_id
        if parent_id and parent_id in self._pending:
            return self._pending.pop(parent_id)[1]
        if parent_id and parent_id in self._seen:
            return None
        if self.fallback == NEWEST_SAME_MODEL:
            model = event_model(response.data)
            partial: object = None
            for key, (request_model, _) in reversed(self._pending.items()):
                if request_model == model:
                    return self._pending.pop(key)[1]
                if partial is None and (not model or not request_model):
                    partial = key
            if partial is not None:
                return self._pending.pop(partial)[1]
        return None

    def unmatched(self) -> list[T]:
        """Return the items of the requests still pending, oldest first."""
        return [item for _, item in self._pending.values()]


def tool_pairer() -> Pairer[Any]:
    """Return a pairer with the tool channel's fallback policy."""
    return Pairer(TOOL_FALLBACK)


def llm_pairer() -> Pairer[Any]:
    """Return a pairer with the LLM channel's fallback policy."""
    return Pairer(LLM_FALLBACK)


@dataclass
class Pairing:
    """Result of :func:`pair_events`, as 0-based positions in the event list."""

    pairs: dict[int, int] = field(default_factory=dict)
    """Response index to the index of the request it paired with."""
    unpaired_responses: list[int] = field(default_factory=list)
    unmatched_requests: list[int] = field(default_factory=list)


def pair_events(
    events: Iterable[TraceEvent],
    request_types: Collection[EventType],
    response_types: Collection[EventType],
    fallback: str = NO_FALLBACK,
) -> Pairing:
    """Pair a whole trace at once and return the result by event index."""
    pairer: Pairer[int] = Pairer(fallback)
    result = Pairing()
    for index, event in enumerate(events):
        if event.event_type in request_types:
            pairer.add(event, index)
        elif event.event_type in response_types:
            request_index = pairer.match(event)
            if request_index is None:
                result.unpaired_responses.append(index)
            else:
                result.pairs[index] = request_index
    result.unmatched_requests = pairer.unmatched()
    return result


def pair_tool_events(events: Iterable[TraceEvent]) -> Pairing:
    """Pair tool calls with tool results and errors."""
    return pair_events(events, TOOL_REQUEST_TYPES, TOOL_RESPONSE_TYPES, TOOL_FALLBACK)


def pair_llm_events(events: Iterable[TraceEvent]) -> Pairing:
    """Pair LLM requests with LLM responses."""
    return pair_events(events, LLM_REQUEST_TYPES, LLM_RESPONSE_TYPES, LLM_FALLBACK)


__all__ = [
    "FALLBACK_POLICIES",
    "LLM_FALLBACK",
    "LLM_REQUEST_TYPES",
    "LLM_RESPONSE_TYPES",
    "NEWEST_SAME_MODEL",
    "NO_FALLBACK",
    "Pairer",
    "Pairing",
    "TOOL_FALLBACK",
    "TOOL_REQUEST_TYPES",
    "TOOL_RESPONSE_TYPES",
    "event_model",
    "llm_pairer",
    "pair_events",
    "pair_llm_events",
    "pair_tool_events",
]
