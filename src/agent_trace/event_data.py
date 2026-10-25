"""Refuse event ``data`` that is not a JSON object.

Hook capture always records ``data`` as an object, but the collector server
and hand edits can store an array, string, number, boolean, or null, and
``TraceEvent.from_json`` keeps whatever it finds. Commands that read keys from
``data`` call :func:`require_object_event_data` before they print or write
anything, so such a session is refused with one message instead of a
traceback. The CLI turns :class:`NonObjectEventDataError` into one
``Error:`` line on stderr and exit code 1.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

from .models import EventType, TraceEvent


class NonObjectEventDataError(ValueError):
    """An event's ``data`` is not a JSON object."""


def _json_type_name(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (list, tuple)):
        return "array"
    return type(value).__name__


def _event_type_value(event: TraceEvent) -> str:
    try:
        return EventType(event.event_type).value
    except (TypeError, ValueError):
        return "unknown type"


_JSON_TYPE_PHRASES = {"null": "null", "array": "an array"}


def skipped_event_warning(command: str, event: TraceEvent) -> str:
    """Return the stderr line a live command prints when it skips an event.

    A live command cannot refuse an event that arrives after it starts, so it
    skips the checks for that event, prints this line, and keeps watching.
    """
    name = _json_type_name(event.data)
    phrase = _JSON_TYPE_PHRASES.get(name, f"a {name}")
    return (
        f"[{command}] Skipped checks for one {_event_type_value(event)} event "
        f"because its data is {phrase}, not an object.\n"
    )


def require_object_event_data(
    events: Sequence[TraceEvent],
    *,
    event_types: Collection[EventType] | None = None,
    indexes: Collection[int] | None = None,
    session_id: str | None = None,
) -> None:
    """Raise :class:`NonObjectEventDataError` if a checked event's ``data`` is not an object.

    Every event is checked by default. ``event_types`` limits the check to
    events of those types and ``indexes`` (0-based positions in ``events``)
    limits it to those events, so a command can check only the events whose
    data it reads. A session whose only non-object data sits in events the
    command never reads keeps working.

    The message names the first checked event with non-object data by 1-based
    position in ``events``, its type, and the JSON type of its data, then
    counts every other event in ``events`` with non-object data. It never
    includes data values or event IDs. ``session_id`` names the session for
    commands that read more than one.
    """
    non_object = [
        (position, event)
        for position, event in enumerate(events, start=1)
        if not isinstance(event.data, Mapping)
    ]
    index_set = None if indexes is None else set(indexes)
    checked = [
        (position, event)
        for position, event in non_object
        if (event_types is None or event.event_type in event_types)
        and (index_set is None or position - 1 in index_set)
    ]
    if not checked:
        return
    position, event = checked[0]
    event_type = _event_type_value(event)
    subject = f"session {session_id}" if session_id else "session"
    message = (
        f"{subject} event data must be an object: event {position} of {len(events)} "
        f"({event_type}) has {_json_type_name(event.data)} data"
    )
    others = len(non_object) - 1
    if others == 1:
        message += "; 1 more event has non-object data"
    elif others:
        message += f"; {others} more events have non-object data"
    raise NonObjectEventDataError(message)
