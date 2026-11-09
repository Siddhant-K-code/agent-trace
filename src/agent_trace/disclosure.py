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
    source_annotation_count: int
    omitted_annotation_count: int
    content_rich_event_count: int
    minimized_content_rich_event_count: int
    redacted_event_count: int
    unredacted_content_rich_event_count: int
    unresolved_sensitive_field_warning_count: int
    anonymization_action_count: int
    evidence_health_status: str
    known_gap_count: int
    redaction_mode: str
    anonymization_mode: str
    provider: str
    capture_method: str
    capture_matrix_version: str
    included_field_categories: tuple[str, ...]
    omitted_field_categories: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["included_field_categories"] = list(self.included_field_categories)
        payload["omitted_field_categories"] = list(self.omitted_field_categories)
        payload["warnings"] = list(self.warnings)
        return payload


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
    source_annotation_count: int | None = None,
    include_content: bool = False,
    evidence_health_status: str = "not_assessed",
    known_gap_count: int = 0,
    redaction_mode: str = "not_recorded",
    anonymization_mode: str = "not_applied",
    provider: str = "unknown",
    capture_method: str = "unknown",
    capture_matrix_version: str = "unknown",
) -> DisclosurePreview:
    """Summarize what a share operation could disclose, without copying values.

    This function deliberately works from event metadata and key names only. It never
    stores prompt text, tool payloads, command strings, paths, or annotation text in
    the returned preview. Every event's ``data`` must be an object; otherwise
    :class:`NonObjectEventDataError` (a ``ValueError``) is raised.
    """

    if annotation_count < 0:
        raise ValueError("annotation_count must be non-negative")
    if source_annotation_count is None:
        source_annotation_count = annotation_count
    if source_annotation_count < annotation_count:
        raise ValueError("source_annotation_count must be at least annotation_count")
    if known_gap_count < 0:
        raise ValueError("known_gap_count must be non-negative")

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
    if include_content:
        included_field_categories = (
            "session metadata",
            "ordered event types and timing",
            "provider and capture provenance",
            "evidence health and known gaps",
            "redaction and anonymization state",
            "prompts and responses",
            "commands and paths",
            "tool inputs and results",
            "other content-rich values",
            "all annotations",
        )
        omitted_field_categories: tuple[str, ...] = ()
        warnings = (
            "Full-content mode may include sensitive values.",
            "Heuristic redaction can miss secrets and personal data.",
        )
        warning_count = unredacted_content_rich_event_count + annotation_count
    else:
        included_field_categories = (
            "session metadata",
            "ordered event types and timing",
            "tool names and effect categories",
            "failure, retry, and recovery signals",
            "provider and capture provenance",
            "evidence health and known gaps",
            "redaction and anonymization state",
            "explicitly shareable annotations",
            "integrity digests",
        )
        omitted_field_categories = (
            "prompts and responses",
            "commands and paths",
            "tool input and result values",
            "other content-rich values",
            "annotations not explicitly marked shareable",
        )
        warnings = (
            "The packet reflects captured evidence only.",
            "Heuristic redaction can miss secrets and personal data in the source trace.",
        )
        warning_count = 0
    return DisclosurePreview(
        schema_version=2,
        mode=mode,
        event_count=len(event_list),
        prompt_count=prompt_count,
        response_count=response_count,
        tool_input_count=tool_input_count,
        tool_result_count=tool_result_count,
        command_count=command_count,
        path_count=path_count,
        annotation_count=annotation_count,
        source_annotation_count=source_annotation_count,
        omitted_annotation_count=source_annotation_count - annotation_count,
        content_rich_event_count=content_rich_event_count,
        minimized_content_rich_event_count=(
            0 if include_content else content_rich_event_count
        ),
        redacted_event_count=count_redacted_events(event_list),
        unredacted_content_rich_event_count=unredacted_content_rich_event_count,
        unresolved_sensitive_field_warning_count=warning_count,
        anonymization_action_count=0,
        evidence_health_status=evidence_health_status,
        known_gap_count=known_gap_count,
        redaction_mode=redaction_mode,
        anonymization_mode=anonymization_mode,
        provider=provider,
        capture_method=capture_method,
        capture_matrix_version=capture_matrix_version,
        included_field_categories=included_field_categories,
        omitted_field_categories=omitted_field_categories,
        warnings=warnings,
    )
