"""Value-free disclosure previews for share/export review flows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Iterable, Literal

from .models import EventType, TraceEvent

DisclosureMode = Literal["minimized", "include_content"]

_CONTENT_EVENT_TYPES = {
    EventType.USER_PROMPT,
    EventType.ASSISTANT_RESPONSE,
    EventType.LLM_REQUEST,
    EventType.LLM_RESPONSE,
    EventType.TOOL_CALL,
    EventType.TOOL_RESULT,
    EventType.FILE_READ,
    EventType.FILE_WRITE,
}
_COMMAND_KEYS = {"command", "cmd", "argv"}
_PATH_KEYS = {"path", "file", "file_path", "cwd", "workdir"}


@dataclass(frozen=True)
class DisclosurePreview:
    """Counts disclosure categories without retaining their values."""

    schema_version: int
    mode: DisclosureMode
    event_count: int
    prompt_count: int
    response_count: int
    tool_input_count: int
    tool_result_count: int
    command_count: int
    path_count: int
    annotation_count: int
    content_rich_event_count: int
    redacted_event_count: int
    unredacted_content_rich_event_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def count_redacted_events(events: Iterable[TraceEvent]) -> int:
    """Count events whose ``redacted`` flag says redaction changed them before write."""
    return sum(1 for event in events if event.redacted)


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


def require_object_event_data(events: Sequence[TraceEvent]) -> None:
    """Raise :class:`NonObjectEventDataError` unless every event's ``data`` is an object.

    Key names cannot be read from an array, string, number, boolean, or null
    payload, so such an event cannot be classified without under-counting. The
    message names the first such event by 1-based position and type only. It
    never includes data values.
    """
    offending = [
        (position, event)
        for position, event in enumerate(events, start=1)
        if not isinstance(event.data, Mapping)
    ]
    if not offending:
        return
    position, event = offending[0]
    try:
        event_type = EventType(event.event_type).value
    except (TypeError, ValueError):
        event_type = "unknown type"
    message = (
        f"session event data must be an object: event {position} of {len(events)} "
        f"({event_type}) has {_json_type_name(event.data)} data"
    )
    others = len(offending) - 1
    if others == 1:
        message += "; 1 more event has non-object data"
    elif others:
        message += f"; {others} more events have non-object data"
    raise NonObjectEventDataError(message)


def build_disclosure_preview(
    events: Iterable[TraceEvent],
    *,
    annotation_count: int = 0,
    include_content: bool = False,
) -> DisclosurePreview:
    """Summarize what a share operation could disclose, without copying values.

    This function deliberately works from event metadata and key names only. It never
    stores prompt text, tool payloads, command strings, paths, or annotation text in
    the returned preview. Every event's ``data`` must be an object; otherwise
    :class:`NonObjectEventDataError` (a ``ValueError``) is raised.
    """

    if annotation_count < 0:
        raise ValueError("annotation_count must be non-negative")

    event_list = list(events)
    require_object_event_data(event_list)
    prompt_count = 0
    response_count = 0
    tool_input_count = 0
    tool_result_count = 0
    command_count = 0
    path_count = 0
    content_rich_event_count = 0
    unredacted_content_rich_event_count = 0

    for event in event_list:
        if event.event_type in {EventType.USER_PROMPT, EventType.LLM_REQUEST}:
            prompt_count += 1
        if event.event_type in {EventType.ASSISTANT_RESPONSE, EventType.LLM_RESPONSE}:
            response_count += 1
        if event.event_type == EventType.TOOL_CALL:
            tool_input_count += 1
        if event.event_type == EventType.TOOL_RESULT:
            tool_result_count += 1

        keys = {str(key).lower() for key in event.data}
        # Hook TOOL_CALL payloads keep command/path arguments in data["arguments"].
        # Only inspect key names; never copy argument values into this preview.
        arguments = event.data.get("arguments")
        if event.event_type == EventType.TOOL_CALL and isinstance(arguments, Mapping):
            keys.update(str(key).lower() for key in arguments)
        # Like path keys, command keys count on every event type: share output embeds
        # all event data, and the MCP proxy records its launch command on session_start.
        if keys & _COMMAND_KEYS:
            command_count += 1
        if event.event_type in {EventType.FILE_READ, EventType.FILE_WRITE} or keys & _PATH_KEYS:
            path_count += 1

        if event.event_type in _CONTENT_EVENT_TYPES:
            content_rich_event_count += 1
            if not event.redacted:
                unredacted_content_rich_event_count += 1

    mode: DisclosureMode = "include_content" if include_content else "minimized"
    return DisclosurePreview(
        schema_version=1,
        mode=mode,
        event_count=len(event_list),
        prompt_count=prompt_count,
        response_count=response_count,
        tool_input_count=tool_input_count,
        tool_result_count=tool_result_count,
        command_count=command_count,
        path_count=path_count,
        annotation_count=annotation_count,
        content_rich_event_count=content_rich_event_count,
        redacted_event_count=count_redacted_events(event_list),
        unredacted_content_rich_event_count=unredacted_content_rich_event_count,
    )
