"""Refuse event ``data`` that is not a JSON object.

Hook capture always records ``data`` as an object, and the collector server
refuses an event whose ``data`` is not one (see :func:`refused_event_warning`).
An older collector stored array, string, number, and boolean ``data`` as it
arrived, and a hand edit can store any JSON value, so a session can still
contain such data. ``TraceEvent.from_json`` keeps whatever it finds. Commands
that read keys from ``data`` call :func:`require_object_event_data` before
they print or write anything, so such a session is refused with one message
instead of a traceback. The CLI turns :class:`NonObjectEventDataError` into
one ``Error:`` line on stderr and exit code 1.

A live command cannot refuse an event with such data that arrives after it
starts. ``watch`` and ``mcp-scan --watch`` skip their checks for such an event
(see :func:`skipped_event_warning`), and the OTLP stream of
``watch --stream-otlp`` leaves such an event out when its converter would read
the data (see :func:`left_out_event_warning`).

A field inside an object ``data`` can have the wrong JSON type too, such as a
number for ``tool_name`` or a list for ``arguments``. Commands read such a
field as missing and keep going. The typed readers below return the caller's
default both when the key is absent and when its value has another JSON type,
so a caller that passes the default it already used for an absent key gets
the same result for both. A JSON number is an ``int`` or ``float`` but not a
``bool``, and a numeric string such as ``"12"`` is a string, not a number.
The readers check only the value. They call ``data.get`` like the reads they
replace, so data that is not an object still fails where it did before and
:func:`require_object_event_data` alone decides which sessions are refused.

Hooks and adapters store a tool call's arguments in ``arguments`` (or
``tool_input``) as an object, as a serialized object, or as plain text such as
the patch an ``apply_patch`` call sends. :func:`tool_arguments` reads all
three, and :func:`patch_paths` reads the file paths a patch names.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Collection, Mapping, Sequence
from typing import Any

from .models import EventType, TraceEvent

MAX_ARGUMENT_CHARS = 200_000

# Lower-cased names of tools that write, edit, create or delete files.
EDIT_TOOLS = frozenset({
    "apply_patch",
    "create",
    "delete_file",
    "edit",
    "edit_file",
    "file_edit",
    "insert",
    "multiedit",
    "notebookedit",
    "patch",
    "replace",
    "search_replace",
    "str_replace_based_edit_tool",
    "str_replace_editor",
    "write",
    "write_file",
    "write_to_file",
})
# Argument keys that name a file.
PATH_KEYS = ("file_path", "path", "absolute_path", "filePath", "target_file", "notebook_path", "uri")

_PATCH_PATH_LINE = re.compile(
    r"^\*\*\* (?:Add File|Update File|Delete File|Move to): (.*)$", re.MULTILINE,
)


def str_field(data: Mapping[str, Any], key: str, default: Any = "") -> Any:
    """Return ``data[key]`` if it is a string, otherwise ``default``."""
    value = data.get(key, default)
    return value if isinstance(value, str) else default


def number_field(data: Mapping[str, Any], key: str, default: Any = 0) -> Any:
    """Return ``data[key]`` if it is an int or float but not a bool, otherwise ``default``."""
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return value


def object_field(data: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """Return ``data[key]`` if it is a JSON object, otherwise ``default``."""
    value = data.get(key, default)
    return value if isinstance(value, Mapping) else default


def list_field(data: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """Return ``data[key]`` if it is a JSON array, otherwise ``default``."""
    value = data.get(key, default)
    return value if isinstance(value, (list, tuple)) else default


def str_list_field(data: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """Return ``data[key]`` if it is an array of strings, otherwise ``default``."""
    value = list_field(data, key, None)
    if value is None or not all(isinstance(item, str) for item in value):
        return default
    return value


def parse_mapping(value: str) -> Mapping[str, Any] | None:
    """Return the object a serialized argument string holds, or None.

    Only text that starts with ``{`` and has at most MAX_ARGUMENT_CHARS
    characters is read: as JSON, then as a Python literal. No code runs.
    """
    text = value.strip()
    if not text.startswith("{") or len(text) > MAX_ARGUMENT_CHARS:
        return None
    try:
        parsed = json.loads(text)
    except (ValueError, RecursionError, MemoryError):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError, MemoryError, RecursionError, TypeError):
            return None
    return parsed if isinstance(parsed, Mapping) else None


def tool_arguments(event: TraceEvent) -> tuple[Mapping[str, Any], str | None]:
    """Return a tool call's arguments as an object, and the string stored, if any.

    Reads ``arguments``, then ``tool_input``, and stops at the first object or
    at the first string that holds a serialized object (see
    :func:`parse_mapping`). The first item is that object, or ``{}``. The
    second is the first string found, such as the raw patch of an
    ``apply_patch`` call, or None when no string was found.
    """
    data = event.data if isinstance(event.data, Mapping) else {}
    raw: str | None = None
    for key in ("arguments", "tool_input"):
        value = data.get(key)
        if isinstance(value, Mapping):
            return value, raw
        if isinstance(value, str):
            if raw is None:
                raw = value
            parsed = parse_mapping(value)
            if parsed is not None:
                return parsed, raw
    return {}, raw


def event_arguments(event: TraceEvent) -> Mapping[str, Any]:
    """Return tool arguments from ``arguments`` or ``tool_input``.

    Some adapters store arguments as a serialized mapping (JSON or a Python
    literal); those are parsed without evaluating code.
    """
    return tool_arguments(event)[0]


def patch_paths(text: str) -> list[str]:
    """Return the file paths an ``apply_patch`` text names, in order, once each.

    Reads the path on every line that starts with ``*** Add File: ``,
    ``*** Update File: ``, ``*** Delete File: `` or ``*** Move to: ``, so both
    ends of a move count. Patch content lines start with ``+``, ``-``, a space
    or ``@@`` and are never read as markers. ``\\r\\n`` line ends are allowed,
    and ``*** Begin Patch`` is not required.
    """
    paths: dict[str, None] = {}
    for match in _PATCH_PATH_LINE.finditer(text):
        path = match.group(1).strip()
        if path:
            paths[path] = None
    return list(paths)


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


def _json_type_phrase(value: object) -> str:
    name = _json_type_name(value)
    return _JSON_TYPE_PHRASES.get(name, f"a {name}")


def skipped_event_warning(command: str, event: TraceEvent) -> str:
    """Return the stderr line a live command prints when it skips an event.

    A live command cannot refuse an event that arrives after it starts, so it
    skips the checks for that event, prints this line, and keeps watching.
    """
    return (
        f"[{command}] Skipped checks for one {_event_type_value(event)} event "
        f"because its data is {_json_type_phrase(event.data)}, not an object.\n"
    )


def refused_event_warning(command: str, event: TraceEvent) -> str:
    """Return the stderr line the collector prints when it refuses an event.

    ``POST /events`` does not store an event whose ``data`` is present but not
    an object. The line names the event type and the JSON type of the data,
    never data values or event IDs.
    """
    return (
        f"[{command}] Refused one {_event_type_value(event)} event "
        f"because its data is {_json_type_phrase(event.data)}, not an object.\n"
    )


def left_out_event_warning(command: str, event: TraceEvent) -> str:
    """Return the stderr line a live OTLP stream prints when it leaves an event out.

    ``watch --stream-otlp`` cannot turn an event into spans when the converter
    would read its data and the data is not an object, so it sends the other
    events of the batch without it. The line names the event type and the
    JSON type of the data, never data values or event IDs.
    """
    return (
        f"[{command}] Left one {_event_type_value(event)} event out of the OTLP "
        f"stream because its data is {_json_type_phrase(event.data)}, not an object.\n"
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
