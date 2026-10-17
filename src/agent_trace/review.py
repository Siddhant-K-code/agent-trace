"""Deterministic projection of trace events into reviewer-facing evidence.

This module does not infer missing activity.  It only classifies facts present in
TraceEvent fields and marks broken relationships explicitly so UI consumers can
show gaps instead of inventing a complete story.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Iterable, Literal

from .models import EventType, TraceEvent

ReviewCategory = Literal[
    "commands",
    "decisions",
    "failures",
    "files",
    "gaps",
    "privacy",
    "recovery",
    "retry",
    "tests",
    "tools",
]
RelationshipState = Literal["none", "linked", "orphaned"]
RelationshipIssue = Literal["missing_parent", "ambiguous_parent", "invalid_parent_type"]


@dataclass(frozen=True)
class ReviewItem:
    event_id: str
    event_type: str
    timestamp: float
    parent_id: str | None
    relationship: RelationshipState
    categories: tuple[ReviewCategory, ...]
    redacted: bool
    source_index: int
    data: dict[str, object] = field(default_factory=dict)
    relationship_issue: RelationshipIssue | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ReviewProjection:
    schema_version: int
    items: tuple[ReviewItem, ...]
    orphaned_event_ids: tuple[str, ...]

    def filter(self, category: ReviewCategory) -> tuple[ReviewItem, ...]:
        return tuple(item for item in self.items if category in item.categories)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "items": [item.to_dict() for item in self.items],
            "orphaned_event_ids": list(self.orphaned_event_ids),
        }


_EXPECTED_PARENTS = {
    EventType.TOOL_RESULT: frozenset({EventType.TOOL_CALL}),
    EventType.LLM_RESPONSE: frozenset({EventType.LLM_REQUEST}),
    EventType.ERROR: frozenset({EventType.TOOL_CALL, EventType.LLM_REQUEST}),
}
_COMMAND_TOOLS = frozenset({"bash", "shell", "execute_command", "exec_command", "terminal", "run_command"})


def _tool_command(event: TraceEvent) -> str:
    if event.event_type != EventType.TOOL_CALL:
        return ""
    args = event.data.get("arguments")
    if not isinstance(args, Mapping):
        args = event.data.get("tool_input")
    if not isinstance(args, Mapping):
        args = {}
    command = args.get("command") or args.get("cmd") or event.data.get("command")
    return command if isinstance(command, str) else ""


def _is_test_command(command: str) -> bool:
    tokens = command.strip().lower().split()
    if not tokens:
        return False
    if tokens[0] in ("pytest", "py.test") or tokens[:3] == ["python", "-m", "pytest"]:
        return True
    if tokens[:2] in (["npm", "test"], ["bun", "test"], ["cargo", "test"], ["go", "test"]):
        return True
    if tokens[:3] in (["python3", "-m", "pytest"], ["uv", "run", "pytest"]):
        return True
    return False


def _explicit_categories(event: TraceEvent) -> set[ReviewCategory]:
    categories: set[ReviewCategory] = set()
    if event.event_type in (EventType.FILE_READ, EventType.FILE_WRITE):
        categories.add("files")
    if event.event_type in (EventType.TOOL_CALL, EventType.TOOL_RESULT):
        categories.add("tools")
    if event.event_type == EventType.ERROR:
        categories.add("failures")
    if event.event_type == EventType.DECISION:
        categories.add("decisions")
    if event.redacted:
        categories.add("privacy")

    data = event.data
    tool_name = str(data.get("tool_name") or data.get("tool") or "").lower()
    command = _tool_command(event)
    if data.get("is_test") is True or data.get("category") == "test" or _is_test_command(command):
        categories.add("tests")
    if data.get("is_command") is True or data.get("category") == "command" or command or (
        event.event_type == EventType.TOOL_CALL and tool_name in _COMMAND_TOOLS
    ):
        categories.add("commands")
    if data.get("retry_of") or data.get("retry") is True:
        categories.add("retry")
    if data.get("recovery_of") or data.get("recovered") is True:
        categories.add("recovery")
    if data.get("privacy_transformed") is True:
        categories.add("privacy")
    return categories


def build_review_projection(events: Iterable[TraceEvent]) -> ReviewProjection:
    """Build a stable review projection without guessing provider semantics.

    Parent links are checked only against event IDs present in the supplied event
    stream.  Missing parents are surfaced as ``orphaned`` and categorized as a
    gap; no synthetic parent is created.
    """
    indexed = list(enumerate(events))
    events_by_id: dict[str, list[TraceEvent]] = {}
    for _, event in indexed:
        if event.event_id:
            events_by_id.setdefault(event.event_id, []).append(event)
    items: list[ReviewItem] = []
    orphaned: list[str] = []

    for source_index, event in indexed:
        categories = _explicit_categories(event)
        parent_id = event.parent_id or None
        relationship_issue: RelationshipIssue | None = None
        if parent_id is None:
            relationship: RelationshipState = "none"
        else:
            parents = events_by_id.get(parent_id, [])
            valid_types = _EXPECTED_PARENTS.get(event.event_type)
            if not parents:
                relationship_issue = "missing_parent"
            elif len(parents) != 1:
                relationship_issue = "ambiguous_parent"
            elif valid_types is not None and parents[0].event_type not in valid_types:
                relationship_issue = "invalid_parent_type"
            if relationship_issue is not None:
                relationship = "orphaned"
                categories.add("gaps")
                orphaned.append(event.event_id)
            else:
                relationship = "linked"
                parent = parents[0]
                if event.event_type == EventType.TOOL_RESULT and (
                    parent.data.get("retry_of")
                    and (event.data.get("exit_code") == 0 or event.data.get("success") is True)
                ):
                    categories.add("recovery")

        items.append(
            ReviewItem(
                event_id=event.event_id,
                event_type=event.event_type.value,
                timestamp=event.timestamp,
                parent_id=parent_id,
                relationship=relationship,
                categories=tuple(sorted(categories)),
                redacted=event.redacted,
                source_index=source_index,
                data=dict(event.data),
                relationship_issue=relationship_issue,
            )
        )

    # Timestamp is the primary execution order. source_index makes ties stable
    # and preserves recorded order when providers emit identical timestamps.
    items.sort(key=lambda item: (item.timestamp, item.source_index))
    return ReviewProjection(
        schema_version=1,
        items=tuple(items),
        orphaned_event_ids=tuple(orphaned),
    )
