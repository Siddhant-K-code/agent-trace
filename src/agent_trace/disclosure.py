"""Value-free disclosure previews for share/export review flows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Iterable, Literal

from .event_data import (  # noqa: F401  (re-exported for existing imports)
    NonObjectEventDataError,
    require_object_event_data,
)
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


def _is_subagent_response(event: TraceEvent) -> bool:
    text = event.data.get("text")
    return event.event_type == EventType.SUBAGENT_STOP and isinstance(text, str) and bool(text)


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
        # A subagent_stop can carry the subagent's final message as ``text``.
        subagent_response = _is_subagent_response(event)
        if event.event_type in {EventType.USER_PROMPT, EventType.LLM_REQUEST}:
            prompt_count += 1
        if (
            event.event_type in {EventType.ASSISTANT_RESPONSE, EventType.LLM_RESPONSE}
            or subagent_response
        ):
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

        if event.event_type in _CONTENT_EVENT_TYPES or subagent_response:
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
