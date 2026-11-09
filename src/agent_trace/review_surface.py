"""Stable session-review payloads for local and editor user interfaces.

The payload composes the conservative event projection, annotations, capture
health, and disclosure metadata without changing the trace storage format.
Consumers receive raw provider fields for local review and can request a
separate privacy-minimized payload that contains no recorded content.
"""

from __future__ import annotations

import hashlib
import html
import json
from collections import Counter

from .annotate import Annotation, load_annotations
from .disclosure import build_disclosure_preview
from .health import CaptureMatrix, assess_session, load_capture_matrix
from .live_verify import LiveCaptureError, inspect_live_capture
from .models import EventType, SessionMeta, TraceEvent
from .review import build_review_projection
from .store import TraceStore

REVIEW_SURFACE_SCHEMA_VERSION = 1
MINIMIZED_REVIEW_SCHEMA_VERSION = 1

_FILTERS = (
    "files",
    "commands",
    "tools",
    "tests",
    "failures",
    "retry",
    "recovery",
    "annotations",
    "corrections",
    "privacy",
    "gaps",
)
_EXPECTED_PARENT_TYPES = {
    EventType.TOOL_RESULT,
    EventType.LLM_RESPONSE,
    EventType.ERROR,
}
_REQUEST_TYPES = {
    EventType.TOOL_CALL: frozenset({EventType.TOOL_RESULT, EventType.ERROR}),
    EventType.LLM_REQUEST: frozenset({EventType.LLM_RESPONSE, EventType.ERROR}),
}


def _recorded_capture_identity(
    meta: SessionMeta,
    events: list[TraceEvent],
    health_provider: str,
    health_capture_method: str,
) -> dict[str, object]:
    starts = [event for event in events if event.event_type == EventType.SESSION_START]
    start_data = starts[0].data if starts else {}
    provider = (
        health_provider
        or str(start_data.get("provider") or "")
        or meta.agent_name
        or "unknown"
    )
    capture_method = (
        health_capture_method
        or str(start_data.get("capture_method") or "")
        or "unknown"
    )
    return {
        "provider": provider,
        "capture_method": capture_method,
        "agent_name": meta.agent_name or None,
        "command": meta.command or None,
        "attribution": dict(meta.attribution),
    }


def _content_mode(events: list[TraceEvent], annotation_count: int) -> tuple[str, dict[str, object]]:
    disclosure = build_disclosure_preview(
        events,
        annotation_count=annotation_count,
        include_content=True,
    )
    if disclosure.content_rich_event_count == 0:
        mode = "metadata_only"
    elif disclosure.unredacted_content_rich_event_count == 0:
        mode = "redacted"
    elif (
        disclosure.unredacted_content_rich_event_count
        < disclosure.content_rich_event_count
    ):
        mode = "mixed"
    else:
        mode = "recorded"
    return mode, disclosure.to_dict()


def _live_capture_state(
    store: TraceStore,
    session_id: str,
    provider: str,
) -> dict[str, object]:
    if provider not in {"claude", "codex"}:
        return {
            "supported": False,
            "status": "not_supported",
            "provider": provider or None,
        }
    try:
        verification = inspect_live_capture(
            store.base_dir,
            provider,
            requested_session=session_id,
        )
    except LiveCaptureError as exc:
        return {
            "supported": True,
            "status": "unavailable",
            "provider": provider,
            "message": str(exc),
        }
    return {
        "supported": True,
        "status": "observed" if verification.exit_code == 0 else "needs_attention",
        "provider": provider,
        "selection": verification.selection,
        "evidence_health": verification.report.result.status.value,
    }


def _annotation_payload(annotation: Annotation) -> dict[str, object]:
    correction = (
        annotation.label.strip().lower() == "correction"
        or annotation.note.strip().lower().startswith("human correction:")
    )
    return {
        "annotation_id": annotation.annotation_id,
        "event_id": annotation.event_id or None,
        "offset_seconds": annotation.offset_seconds,
        "label": annotation.label or None,
        "note": annotation.note or None,
        "author": annotation.author or None,
        "created_at": annotation.created_at,
        "correction": correction,
    }


def _annotation_targets(
    annotations: list[Annotation],
    events: list[TraceEvent],
) -> tuple[dict[str, list[dict[str, object]]], list[dict[str, object]]]:
    by_event: dict[str, list[dict[str, object]]] = {}
    unresolved: list[dict[str, object]] = []
    event_id_counts = Counter(event.event_id for event in events)
    started_at = events[0].timestamp if events else 0.0

    for annotation in annotations:
        payload = _annotation_payload(annotation)
        target = annotation.event_id
        if target and event_id_counts[target] == 1:
            by_event.setdefault(target, []).append(payload)
            continue
        if not target and events:
            target_event = min(
                events,
                key=lambda event: abs(
                    (event.timestamp - started_at) - annotation.offset_seconds
                ),
            )
            payload["event_id"] = target_event.event_id
            if event_id_counts[target_event.event_id] == 1:
                by_event.setdefault(target_event.event_id, []).append(payload)
                continue
        unresolved.append(payload)

    return by_event, unresolved


def _summary(event: TraceEvent) -> str:
    data = event.data
    if event.event_type == EventType.TOOL_CALL:
        tool = data.get("tool_name") or data.get("tool") or "tool"
        arguments = data.get("arguments") or data.get("tool_input")
        command = data.get("command")
        detail = command if isinstance(command, str) else arguments
        return f"{tool}: {detail}" if detail else str(tool)
    if event.event_type == EventType.TOOL_RESULT:
        return str(data.get("output") or data.get("result") or "tool result")
    if event.event_type in (EventType.FILE_READ, EventType.FILE_WRITE):
        return str(data.get("path") or data.get("file_path") or event.event_type.value)
    if event.event_type == EventType.ERROR:
        return str(data.get("message") or data.get("error") or "recorded error")
    if event.event_type == EventType.USER_PROMPT:
        return str(data.get("prompt") or data.get("content") or data.get("text") or "user prompt")
    if event.event_type == EventType.ASSISTANT_RESPONSE:
        return str(
            data.get("response") or data.get("content") or data.get("text")
            or "assistant response"
        )
    if event.event_type == EventType.DECISION:
        return str(
            data.get("decision") or data.get("reasoning") or data.get("text")
            or "recorded decision"
        )
    return event.event_type.value.replace("_", " ")


def _review_items(
    events: list[TraceEvent],
    annotations: list[Annotation],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    projection = build_review_projection(events)
    annotation_map, unresolved_annotations = _annotation_targets(annotations, events)
    event_by_index = {index: event for index, event in enumerate(events)}
    children: dict[str, list[str]] = {}

    for projected in projection.items:
        if projected.relationship == "linked" and projected.parent_id:
            children.setdefault(projected.parent_id, []).append(projected.event_id)

    items: list[dict[str, object]] = []
    for sequence, projected in enumerate(projection.items, start=1):
        event = event_by_index[projected.source_index]
        categories = set(projected.categories)
        event_annotations = annotation_map.get(projected.event_id, [])
        if event_annotations:
            categories.add("annotations")
        if any(bool(annotation["correction"]) for annotation in event_annotations):
            categories.add("corrections")

        relationship = projected.relationship
        relationship_issue = projected.relationship_issue
        child_ids = children.get(projected.event_id, [])

        if event.event_type in _EXPECTED_PARENT_TYPES and not projected.parent_id:
            relationship = "orphaned"
            relationship_issue = "missing_parent"
            categories.add("gaps")

        expected_children = _REQUEST_TYPES.get(event.event_type)
        linked_terminal_children = [
            child_id
            for child_id in child_ids
            if next(
                (
                    candidate.event_type
                    for candidate in events
                    if candidate.event_id == child_id
                ),
                None,
            ) in expected_children
        ] if expected_children else []
        missing_child = bool(expected_children and not linked_terminal_children)
        if missing_child:
            categories.add("gaps")

        items.append({
            "sequence": sequence,
            "event_id": projected.event_id,
            "event_type": projected.event_type,
            "timestamp": projected.timestamp,
            "duration_ms": event.duration_ms,
            "parent_id": projected.parent_id,
            "child_ids": child_ids,
            "relationship": relationship,
            "relationship_issue": relationship_issue,
            "missing_child": missing_child,
            "categories": sorted(categories),
            "redacted": projected.redacted,
            "source_index": projected.source_index,
            "summary": _summary(event),
            "data": dict(projected.data),
            "annotations": event_annotations,
        })

    return items, unresolved_annotations


def build_session_review(
    store: TraceStore,
    session_id: str,
    *,
    matrix: CaptureMatrix | None = None,
) -> dict[str, object]:
    """Build the complete local review payload for one stored session."""
    meta = store.load_meta(session_id)
    events = store.load_events(session_id)
    annotations = load_annotations(store, session_id)
    health = assess_session(
        store,
        session_id,
        meta,
        events,
        matrix or load_capture_matrix(),
    )
    items, unresolved_annotations = _review_items(events, annotations)
    content_mode, disclosure = _content_mode(events, len(annotations))
    started_at = meta.started_at or (items[0]["timestamp"] if items else None)
    ended_events = [event.timestamp for event in events if event.event_type == EventType.SESSION_END]
    ended_at = meta.ended_at or (ended_events[-1] if ended_events else None)
    duration_ms = (
        max(0.0, (float(ended_at) - float(started_at)) * 1000)
        if started_at is not None and ended_at is not None
        else None
    )
    category_counts = Counter(
        category
        for item in items
        for category in item["categories"]
    )
    gap_reasons = [
        {
            "code": reason.code,
            "message": reason.message,
            "kind": reason.kind.value,
            "event_id": reason.event_id or None,
        }
        for reason in health.result.reasons
    ]
    for annotation in unresolved_annotations:
        gap_reasons.append({
            "code": "unresolved_annotation",
            "message": "An annotation references an event that is not present.",
            "kind": "observed",
            "event_id": annotation.get("event_id"),
        })

    return {
        "schema_version": REVIEW_SURFACE_SCHEMA_VERSION,
        "session": {
            "session_id": session_id,
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_ms": duration_ms,
            "active": ended_at is None,
            "redacted_metadata": bool(meta.redacted),
            **_recorded_capture_identity(
                meta,
                events,
                health.result.provider,
                health.result.capture_method,
            ),
        },
        "evidence_health": health.to_dict(),
        "live_capture": _live_capture_state(
            store,
            session_id,
            health.result.provider,
        ),
        "capture_gaps": gap_reasons,
        "privacy": {
            "redaction_mode": health.redaction.mode,
            "content_mode": content_mode,
            "disclosure": disclosure,
        },
        "filters": [
            {"id": filter_id, "count": category_counts.get(filter_id, 0)}
            for filter_id in _FILTERS
        ],
        "items": items,
        "unresolved_annotations": unresolved_annotations,
    }


def build_minimized_review_payload(review: dict[str, object]) -> dict[str, object]:
    """Return a content-free review payload suitable for local file sharing."""
    session = review["session"]
    assert isinstance(session, dict)
    items = review["items"]
    assert isinstance(items, list)
    sequence_by_id = {
        str(item["event_id"]): int(item["sequence"])
        for item in items
        if isinstance(item, dict)
    }
    started_at = session.get("started_at")

    minimized_items: list[dict[str, object]] = []
    for item in items:
        assert isinstance(item, dict)
        timestamp = item.get("timestamp")
        relative_ms = (
            max(0.0, (float(timestamp) - float(started_at)) * 1000)
            if timestamp is not None and started_at is not None
            else None
        )
        minimized_items.append({
            "sequence": item["sequence"],
            "event_type": item["event_type"],
            "relative_ms": relative_ms,
            "duration_ms": item.get("duration_ms"),
            "categories": list(item.get("categories", [])),
            "relationship": item.get("relationship"),
            "relationship_issue": item.get("relationship_issue"),
            "missing_child": bool(item.get("missing_child")),
            "parent_sequence": sequence_by_id.get(str(item.get("parent_id") or "")),
            "child_sequences": [
                sequence_by_id[child_id]
                for child_id in item.get("child_ids", [])
                if child_id in sequence_by_id
            ],
            "redacted": bool(item.get("redacted")),
            "annotations": [
                {
                    "label": annotation.get("label"),
                    "correction": bool(annotation.get("correction")),
                }
                for annotation in item.get("annotations", [])
            ],
        })

    evidence_health = review["evidence_health"]
    assert isinstance(evidence_health, dict)
    reasons = evidence_health.get("reasons", [])
    session_id = str(session["session_id"])
    return {
        "schema_version": MINIMIZED_REVIEW_SCHEMA_VERSION,
        "privacy": {
            "mode": "minimized",
            "omitted": [
                "session identity",
                "absolute timestamps",
                "raw provider fields",
                "prompts and responses",
                "commands and paths",
                "tool inputs and results",
                "annotation notes and authors",
            ],
        },
        "session": {
            "session_ref": hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:12],
            "provider": session.get("provider"),
            "capture_method": session.get("capture_method"),
            "duration_ms": session.get("duration_ms"),
            "event_count": len(minimized_items),
        },
        "evidence_health": {
            "status": evidence_health.get("status"),
            "reasons": [
                {
                    "code": reason.get("code"),
                    "kind": reason.get("kind"),
                }
                for reason in reasons
                if isinstance(reason, dict)
            ],
        },
        "items": minimized_items,
    }


def render_minimized_review_html(payload: dict[str, object]) -> str:
    """Render a self-contained privacy-minimized review file."""
    session = payload["session"]
    assert isinstance(session, dict)
    health = payload["evidence_health"]
    assert isinstance(health, dict)
    items = payload["items"]
    assert isinstance(items, list)
    rows = []
    for item in items:
        assert isinstance(item, dict)
        categories = ", ".join(str(value) for value in item.get("categories", [])) or "none"
        relation = str(item.get("relationship") or "none")
        issue = item.get("relationship_issue")
        if issue:
            relation += f" ({issue})"
        rows.append(
            "<tr>"
            f"<td>{int(item['sequence'])}</td>"
            f"<td>{html.escape(str(item['event_type']))}</td>"
            f"<td>{html.escape(categories)}</td>"
            f"<td>{html.escape(relation)}</td>"
            f"<td>{'yes' if item.get('redacted') else 'no'}</td>"
            "</tr>"
        )
    embedded = html.escape(json.dumps(payload, indent=2))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Privacy-minimized evidence review</title>
<style>
body{{font:15px/1.5 system-ui,sans-serif;max-width:980px;margin:40px auto;padding:0 20px;color:#172033;background:#f7f8fb}}
h1{{font-size:1.6rem}}.meta{{display:flex;gap:24px;flex-wrap:wrap;margin:20px 0;padding:16px;background:#fff;border:1px solid #ccd3df}}
table{{width:100%;border-collapse:collapse;background:#fff}}th,td{{padding:8px 10px;text-align:left;border-bottom:1px solid #dfe4ec}}
th{{color:#475569}}details{{margin-top:24px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;padding:16px;background:#111827;color:#e5e7eb}}
</style>
</head>
<body>
<h1>Privacy-minimized evidence review</h1>
<p>This local file omits recorded content, identities, paths, commands, results, and annotation text.</p>
<div class="meta">
<span>Session ref: {html.escape(str(session.get("session_ref") or ""))}</span>
<span>Provider: {html.escape(str(session.get("provider") or "unknown"))}</span>
<span>Evidence health: {html.escape(str(health.get("status") or "unknown"))}</span>
<span>Events: {len(items)}</span>
</div>
<table>
<thead><tr><th>#</th><th>Evidence</th><th>Categories</th><th>Relationship</th><th>Redacted</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
<details><summary>Minimized JSON</summary><pre>{embedded}</pre></details>
</body>
</html>
"""
