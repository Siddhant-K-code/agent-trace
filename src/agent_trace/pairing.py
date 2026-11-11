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
  a ``parent_id``. An ``error`` whose ``parent_id`` names a pending
  ``llm_request`` ends that request as a failed call. Exporters pass each
  ``error`` to :meth:`Pairer.match_parent` on the LLM channel before the
  tool channel; it applies rule 1 only, so an error without that link never
  takes a request through the fallback and stays on the tool channel.

A caller may feed more event types to a pairer, for example cost reporting
also treats ``assistant_response`` as an LLM response because it carries
usage, but the rules above stay the same.

Recorders pair while the trace is written, so a stored response carries
the ``parent_id`` and ``duration_ms`` that the rules above then follow:

- Proxies (:class:`KeyedPairer`): the stdio and HTTP MCP proxies see the
  JSON-RPC ``id`` on both the ``tools/call`` request and its response, so a
  ``tool_result`` or ``error`` pairs with the pending ``tool_call`` of the
  same id and nothing else. A request with the same id replaces the pending
  one. The pending map is bounded; the oldest request is dropped first.
- Hooks (:func:`match_hook_tool_call`): each tool hook runs in its own
  process, so pending calls live in a JSON file that the caller reads and
  writes under the store write lock. A result pairs with the pending call
  that has its ``tool_use_id``. Without one, it pairs with the oldest pending
  call of the same tool name, preferring calls from the same agent so a
  subagent's result does not take its parent's call. The tool name is the
  only evidence some providers send, and the hook records the pairing as a
  ``parent_id`` at once, so exporters never need a fallback for it.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Collection, Hashable, Iterable, Mapping
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

    def match_parent(self, response: TraceEvent) -> T | None:
        """Like :meth:`match`, but apply rule 1 only, never the fallback."""
        parent_id = response.parent_id
        if parent_id and parent_id in self._pending:
            return self._pending.pop(parent_id)[1]
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


def link_response(response: TraceEvent, request_id: str, request_timestamp: float) -> None:
    """Record on ``response`` that it answers the request ``request_id``."""
    response.parent_id = request_id
    response.duration_ms = (response.timestamp - request_timestamp) * 1000


class KeyedPairer(Generic[T]):
    """Pair live responses with pending requests by a correlation key.

    Used where the request and its response carry the same key, such as a
    JSON-RPC ``id``. A response pairs only with the request of its key; there
    is no fallback. A ``None`` key is never pending and never pairs. Adding a
    key that is already pending replaces that request, which becomes the
    newest. With ``max_pending`` set, the oldest requests are dropped once
    more than that many are pending, so requests that never get a response
    do not accumulate.

    Methods are safe to call from several threads. Each instance has its own
    pending map, so two proxies never pair with each other's requests.
    """

    def __init__(self, max_pending: int | None = None) -> None:
        if max_pending is not None and max_pending < 1:
            raise ValueError("max_pending must be at least 1")
        self.max_pending = max_pending
        self._pending: dict[Hashable, T] = {}
        self._lock = threading.Lock()

    def add(self, key: Hashable | None, item: T) -> None:
        """Make ``item`` pending under ``key``."""
        if key is None:
            return
        with self._lock:
            self._pending.pop(key, None)
            self._pending[key] = item
            if self.max_pending is not None:
                while len(self._pending) > self.max_pending:
                    del self._pending[next(iter(self._pending))]

    def match(self, key: Hashable | None) -> T | None:
        """Return and stop tracking the item pending under ``key``, if any."""
        if key is None:
            return None
        with self._lock:
            return self._pending.pop(key, None)

    def clear(self) -> int:
        """Drop every pending item. Returns how many were dropped."""
        with self._lock:
            count = len(self._pending)
            self._pending.clear()
        return count

    def pending(self) -> dict[Hashable, T]:
        """Return a copy of the pending items by key, oldest first."""
        with self._lock:
            return dict(self._pending)

    def __len__(self) -> int:
        with self._lock:
            return len(self._pending)


def _hook_timestamp(info: Mapping[str, Any]) -> float | None:
    value = info.get("timestamp")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def match_hook_tool_call(
    pending: Mapping[str, Any],
    tool_name: Any,
    tool_use_id: Any = "",
    agent_id: Any = "",
) -> str | None:
    """Return the ID of the pending hook tool call a tool result answers.

    ``pending`` maps a ``tool_call`` event ID to the entry the pre-tool hook
    stored for it: ``tool_name``, ``tool_use_id``, ``timestamp`` and, for
    subagents, ``agent_id``. With a ``tool_use_id``, only the entry with that
    ``tool_use_id`` matches. Without one, the oldest entry for ``tool_name``
    matches, and an entry from the same ``agent_id`` beats any entry from
    another agent. Entries that are not objects, or have no numeric
    ``timestamp``, never match. The caller removes the matched entry.
    """
    match_id: str | None = None
    match_rank: tuple[bool, float] | None = None
    for event_id, info in pending.items():
        if not event_id or not isinstance(event_id, str) or not isinstance(info, Mapping):
            continue
        timestamp = _hook_timestamp(info)
        if timestamp is None:
            continue
        if tool_use_id:
            if info.get("tool_use_id") == tool_use_id:
                return event_id
            continue
        if info.get("tool_name") != tool_name:
            continue
        rank = (info.get("agent_id", "") == agent_id, -timestamp)
        if match_rank is None or rank > match_rank:
            match_id = event_id
            match_rank = rank
    return match_id


def hook_call_timestamp(info: Any) -> float | None:
    """Return the timestamp a pending hook entry recorded, if it is valid."""
    return _hook_timestamp(info) if isinstance(info, Mapping) else None


__all__ = [
    "FALLBACK_POLICIES",
    "KeyedPairer",
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
    "hook_call_timestamp",
    "link_response",
    "llm_pairer",
    "match_hook_tool_call",
    "pair_events",
    "pair_llm_events",
    "pair_tool_events",
]
