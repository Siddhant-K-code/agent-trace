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

Four top-level event fields can have the wrong JSON type too: ``event_id``,
``parent_id``, ``duration_ms`` and ``redacted``. The collector refuses such an
event, but a hand edit, an older store or another tool can still write one,
and ``TraceEvent.from_json`` keeps the stored value. Commands that link,
count, compare, format or convert these fields read them through
:func:`event_id_of`, :func:`parent_id_of`, :func:`duration_ms_of` and
:func:`is_redacted`, which read a value with the wrong type as missing.
Output that copies the stored event, such as ``export --format json`` and
``inspect``, still shows the stored value.

Hooks and adapters store a tool call's arguments in ``arguments`` (or
``tool_input``) as an object, as a serialized object, or as plain text such as
the patch an ``apply_patch`` call sends. :func:`tool_arguments` reads all
three, and :func:`patch_paths` reads the file paths a patch names.
:func:`event_paths` returns the files an event names, and :func:`is_edit`
says whether the event writes them.

An event's ``timestamp`` must be a Unix time in seconds. The collector refuses
one that is not (see :func:`refused_timestamp_warning`), but an older version
or a hand edit can still store one. Commands that read timestamps call
:func:`require_usable_timestamps` before they print or write anything, and the
CLI turns :class:`UnusableTimestampError` into one ``Error:`` line on stderr
and exit code 1. ``watch --stream-otlp`` leaves such an event out of the OTLP
stream (see :func:`left_out_timestamp_warning`).
"""

from __future__ import annotations

import ast
import json
import math
import re
from collections.abc import Collection, Mapping, Sequence
from typing import Any

from .models import (
    EventType,
    SessionMeta,
    TraceEvent,
    usable_number,
    usable_text,
    usable_time,
)

# Longest argument text parse_mapping reads as a Python literal. JSON text is
# read at any length.
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
# FILE_WRITE and FILE_READ events name their file under these keys.
FILE_EVENT_PATH_KEYS = ("path", "file_path", "uri")
# Editor tools that only read the file when their command is "view".
VIEW_COMMAND_TOOLS = frozenset({"str_replace_editor", "str_replace_based_edit_tool"})

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


def event_id_of(event: TraceEvent) -> str:
    """Return the event's ``event_id`` if it is a string, otherwise ""."""
    return usable_text(event.event_id)


def parent_id_of(event: TraceEvent) -> str:
    """Return the event's ``parent_id`` if it is a string, otherwise ""."""
    return usable_text(event.parent_id)


def duration_ms_of(event: TraceEvent) -> int | float | None:
    """Return the event's ``duration_ms`` if :func:`usable_number` accepts it, otherwise None.

    The stored value comes back unchanged, so an int stays an int.
    """
    value = event.duration_ms
    return value if usable_number(value) is not None else None


def is_redacted(event: TraceEvent) -> bool:
    """Return True only if the event's ``redacted`` is the JSON value ``true``."""
    return event.redacted is True


def parse_mapping(value: str) -> Mapping[str, Any] | None:
    """Return the object a serialized argument string holds, or None.

    Only text that starts with ``{`` is read: as JSON at any length, then, if
    that fails and the text has at most MAX_ARGUMENT_CHARS characters, as a
    Python literal. No code runs.
    """
    text = value.strip()
    if not text.startswith("{"):
        return None
    try:
        parsed = json.loads(text)
    except (ValueError, RecursionError, MemoryError):
        if len(text) > MAX_ARGUMENT_CHARS:
            return None
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


def _call_name(event: TraceEvent, tool_name: str | None) -> str:
    return str_field(event.data, "tool_name", "").lower() if tool_name is None else tool_name


def event_paths(event: TraceEvent, tool_name: str | None = None) -> list[str]:
    """Return the file paths an event names, as stored, in order, once each.

    A tool call names each PATH_KEYS argument and, for apply_patch with the
    patch stored as text, each file the patch adds, updates, deletes or moves
    to. A FILE_WRITE or FILE_READ event names its path, file_path or uri.
    Other events name none. *tool_name* is the lower-cased tool name to use
    instead of the event's ``tool_name`` field.
    """
    if not isinstance(event.data, Mapping):
        return []
    if event.event_type in (EventType.FILE_WRITE, EventType.FILE_READ):
        values = [event.data.get(key) for key in FILE_EVENT_PATH_KEYS]
    elif event.event_type == EventType.TOOL_CALL:
        args, raw = tool_arguments(event)
        values = [args.get(key) for key in PATH_KEYS]
        if raw is not None and _call_name(event, tool_name) == "apply_patch":
            values.extend(patch_paths(raw))
    else:
        return []
    return list(dict.fromkeys(v for v in values if isinstance(v, str) and v))


def is_editor_view(event: TraceEvent, tool_name: str | None = None) -> bool:
    """Return True if *event* is a VIEW_COMMAND_TOOLS call whose command is "view".

    *tool_name* works as in :func:`event_paths`.
    """
    return (
        event.event_type == EventType.TOOL_CALL
        and isinstance(event.data, Mapping)
        and _call_name(event, tool_name) in VIEW_COMMAND_TOOLS
        and tool_arguments(event)[0].get("command") == "view"
    )


def is_edit(event: TraceEvent, tool_name: str | None = None) -> bool:
    """Return True if *event* writes, edits, creates or deletes the files it names.

    That is a FILE_WRITE event, or a call to an EDIT_TOOLS tool or
    str_replace. An editor call whose command is "view" only reads. Shell
    commands are never edits, even when they write files. *tool_name* works
    as in :func:`event_paths`.
    """
    if event.event_type == EventType.FILE_WRITE:
        return True
    if event.event_type != EventType.TOOL_CALL or not isinstance(event.data, Mapping):
        return False
    name = _call_name(event, tool_name)
    if name in VIEW_COMMAND_TOOLS:
        return not is_editor_view(event, name)
    return name in EDIT_TOOLS or name == "str_replace"


class NonObjectEventDataError(ValueError):
    """An event's ``data`` is not a JSON object."""


class UnusableTimestampError(ValueError):
    """An event or session time is not a Unix time in seconds."""


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


def refused_timestamp_warning(command: str, event: TraceEvent) -> str:
    """Return the stderr line the collector prints when it refuses a timestamp.

    ``POST /events`` does not store an event whose ``timestamp`` is not a Unix
    time in seconds. The line names the event type only, never the timestamp,
    data values or event IDs.
    """
    return (
        f"[{command}] Refused one {_event_type_value(event)} event "
        "because its timestamp is not a Unix time in seconds.\n"
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


def left_out_timestamp_warning(
    command: str, event: TraceEvent, target: str = "OTLP stream",
) -> str:
    """Return the line a live command prints when it leaves an event out of
    *target* for its timestamp.

    ``watch --stream-otlp`` cannot turn an event into spans when the converter
    would read its timestamp and the timestamp is not a Unix time in seconds,
    so it sends the other events of the batch without it and prints the line
    to stderr. ``watch --compaction-checkpoint`` leaves such an event out of
    the compaction checkpoint and prints the line with its other checkpoint
    lines. The line names the event type only, never the timestamp, data
    values or event IDs.
    """
    return (
        f"[{command}] Left one {_event_type_value(event)} event out of the {target} "
        "because its timestamp is not a Unix time in seconds.\n"
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


def _timestamp_kind(value: object) -> str:
    # Never call float() on an int: 10**400 does not fit in a float.
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if value is None:
        return "null"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, (list, tuple)):
        return "array"
    if isinstance(value, int):
        return "negative" if value < 0 else "out-of-range"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "non-finite"
        return "negative" if value < 0 else "out-of-range"
    return "non-JSON"


_TIMESTAMP_KINDS_WITH_AN = frozenset({"object", "array", "out-of-range"})


def require_usable_timestamps(
    events: Sequence[TraceEvent],
    *,
    event_types: Collection[EventType] | None = None,
    indexes: Collection[int] | None = None,
    session_id: str | None = None,
) -> None:
    """Raise :class:`UnusableTimestampError` if a checked event's ``timestamp``
    is not a Unix time in seconds.

    A timestamp is usable when :func:`~agent_trace.models.usable_time` accepts
    it: a finite number of 0 or more that ``datetime.fromtimestamp`` accepts.
    Every event is checked by default. ``event_types`` limits the check to
    events of those types and ``indexes`` (0-based positions in ``events``)
    limits it to those events, so a command can check only the events whose
    timestamps it reads. When both are given, both apply.

    The message names the first checked event with an unusable timestamp by
    1-based position in ``events``, its type, and the kind of the timestamp
    (string, null, boolean, object, array, non-finite, negative or
    out-of-range), then counts every other event in ``events`` with an
    unusable timestamp. It never includes the timestamp, any part of it, data
    values or event IDs. ``session_id`` names the session for commands that
    read more than one.
    """
    unusable = [
        (position, event)
        for position, event in enumerate(events, start=1)
        if usable_time(event.timestamp) is None
    ]
    index_set = None if indexes is None else set(indexes)
    checked = [
        (position, event)
        for position, event in unusable
        if (event_types is None or event.event_type in event_types)
        and (index_set is None or position - 1 in index_set)
    ]
    if not checked:
        return
    position, event = checked[0]
    kind = _timestamp_kind(event.timestamp)
    article = "an" if kind in _TIMESTAMP_KINDS_WITH_AN else "a"
    subject = f"session {session_id}" if session_id else "session"
    message = (
        f"{subject} event timestamp must be a Unix time in seconds: event {position} "
        f"of {len(events)} ({_event_type_value(event)}) has {article} {kind} timestamp"
    )
    others = len(unusable) - 1
    if others == 1:
        message += "; 1 more event has an unusable timestamp"
    elif others:
        message += f"; {others} more events have unusable timestamps"
    raise UnusableTimestampError(message)


def require_usable_session_times(meta: SessionMeta, *, session_id: str | None = None) -> None:
    """Raise :class:`UnusableTimestampError` if the session's ``started_at``, or
    its ``ended_at`` when it is not None, is not a Unix time in seconds.

    OTLP export reads both for the root span. The compliance exports write
    both in their reports. ``ended_at`` is None while a session is open. The
    message never includes the value. ``session_id`` names the session for
    commands that read more than one.
    """
    subject = f"session {session_id}" if session_id else "session"
    if usable_time(meta.started_at) is None:
        raise UnusableTimestampError(f"{subject} started_at must be a Unix time in seconds")
    if meta.ended_at is not None and usable_time(meta.ended_at) is None:
        raise UnusableTimestampError(f"{subject} ended_at must be a Unix time in seconds")
