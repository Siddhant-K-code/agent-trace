"""Portable evidence packets for privacy-minimized session sharing."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import PurePath
from typing import Any, Mapping, Sequence

from . import __version__
from .annotate import Annotation, load_annotations
from .correlation import correlate_tool_events
from .disclosure import DisclosurePreview, build_disclosure_preview
from .event_data import require_object_event_data
from .health import HealthReport, assess_session, load_capture_matrix
from .models import EventType, SessionMeta, TraceEvent
from .store import TraceStore

EVIDENCE_PACKET_SCHEMA = "agent-strace.evidence-packet"
EVIDENCE_PACKET_SCHEMA_VERSION = 1

_COMMAND_KEYS = ("command", "cmd", "argv")
_PATH_KEYS = ("file_path", "path", "file", "uri", "cwd", "workdir")
_FAILURE_STATUSES = frozenset({"error", "failed", "failure", "cancelled", "timeout"})

@dataclass(frozen=True)
class BehaviorSummary:
    """Value-free failure, retry, and recovery facts."""

    failure_event_count: int
    failed_operation_count: int
    retry_count: int
    recovery_attempt_count: int
    recovered_operation_count: int
    unresolved_failed_operation_count: int
    orphan_outcome_count: int
    ambiguous_outcome_count: int
    duplicate_call_id_event_count: int
    unobserved_operation_count: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class EvidencePacket:
    """Selected events, annotations, and a disclosure manifest."""

    meta: SessionMeta
    events: tuple[TraceEvent, ...]
    annotations: tuple[Annotation, ...]
    preview: DisclosurePreview
    health: HealthReport
    behavior: BehaviorSummary
    manifest: dict[str, Any]


@dataclass(frozen=True)
class _OperationState:
    ref: str
    attempt: int
    retry: bool
    recovery_attempt: bool


@dataclass(frozen=True)
class _BehaviorAnalysis:
    summary: BehaviorSummary
    calls: dict[int, _OperationState]
    outcomes: dict[int, tuple[int, str]]


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _event_dict(event: TraceEvent) -> dict[str, Any]:
    payload = asdict(event)
    payload["event_type"] = event.event_type.value
    return payload


def _annotation_dict(annotation: Annotation) -> dict[str, Any]:
    return asdict(annotation)


def _mapping_keys(value: object) -> list[str]:
    if not isinstance(value, Mapping):
        return []
    return sorted(str(key) for key in value)


def _first_value(data: Mapping[str, Any], keys: Sequence[str]) -> object:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _path_category(value: object) -> dict[str, str]:
    if not isinstance(value, str) or not value:
        return {}
    if "://" in value:
        kind = "uri"
    elif value.startswith("~"):
        kind = "home_relative"
    elif os.path.isabs(value):
        kind = "absolute"
    else:
        kind = "relative"
    suffix = PurePath(value.split("?", 1)[0]).suffix.lower().lstrip(".")
    if not suffix or len(suffix) > 16 or not suffix.replace("_", "").isalnum():
        suffix = ""
    result = {"path_kind": kind}
    if suffix:
        result["file_type"] = suffix
    return result


def _tool_effect(tool_name: str) -> str:
    name = tool_name.strip().lower()
    if any(token in name for token in ("bash", "shell", "terminal", "exec", "command")):
        return "command"
    if any(token in name for token in ("write", "edit", "patch", "create", "delete", "move")):
        return "file_change"
    if any(token in name for token in ("read", "view", "open_file")):
        return "file_read"
    if any(token in name for token in ("search", "grep", "glob", "find", "rg")):
        return "search"
    if any(token in name for token in ("http", "curl", "fetch", "web")):
        return "network"
    if any(token in name for token in ("browser", "chrome", "playwright")):
        return "browser"
    return "tool"


def _tool_signature(event: TraceEvent) -> bytes:
    data = event.data
    arguments = data.get("arguments")
    signature = {
        "tool_name": data.get("tool_name", data.get("name", "")),
        "arguments": arguments if isinstance(arguments, Mapping) else arguments,
        "command": _first_value(data, _COMMAND_KEYS),
        "path": _first_value(data, _PATH_KEYS),
    }
    return _canonical_json_bytes(signature)


def _result_failed(event: TraceEvent) -> bool:
    data = event.data
    if data.get("success") is False or data.get("ok") is False:
        return True
    exit_code = data.get("exit_code")
    if isinstance(exit_code, (int, float)) and not isinstance(exit_code, bool):
        return exit_code != 0
    status = data.get("status")
    return isinstance(status, str) and status.strip().lower() in _FAILURE_STATUSES


def _behavior_analysis(events: Sequence[TraceEvent]) -> _BehaviorAnalysis:
    failed_calls: set[int] = set()
    successful_calls: set[int] = set()
    outcomes: dict[int, tuple[int, str]] = {}
    correlation = correlate_tool_events(events)

    for index, call_index in correlation.outcome_calls.items():
        event = events[index]
        outcome = "failed"
        if event.event_type == EventType.TOOL_RESULT and not _result_failed(event):
            outcome = "succeeded"
            successful_calls.add(call_index)
        else:
            failed_calls.add(call_index)
        outcomes[index] = (call_index, outcome)

    signatures: dict[bytes, tuple[str, list[int]]] = {}
    call_states: dict[int, _OperationState] = {}
    recovery_refs: set[str] = set()
    recovered_refs: set[str] = set()

    for index, event in enumerate(events):
        if event.event_type != EventType.TOOL_CALL:
            continue
        signature = _tool_signature(event)
        if signature not in signatures:
            signatures[signature] = (f"operation-{len(signatures) + 1:03d}", [])
        ref, previous = signatures[signature]
        attempt = len(previous) + 1
        recovery_attempt = any(call_index in failed_calls for call_index in previous)
        if recovery_attempt:
            recovery_refs.add(ref)
        call_states[index] = _OperationState(
            ref=ref,
            attempt=attempt,
            retry=attempt > 1,
            recovery_attempt=recovery_attempt,
        )
        previous.append(index)

    for call_index in successful_calls:
        state = call_states.get(call_index)
        if state is not None and state.recovery_attempt:
            recovered_refs.add(state.ref)

    failed_refs = {
        call_states[index].ref for index in failed_calls if index in call_states
    }
    summary = BehaviorSummary(
        failure_event_count=sum(event.event_type == EventType.ERROR for event in events),
        failed_operation_count=len(failed_calls),
        retry_count=sum(state.retry for state in call_states.values()),
        recovery_attempt_count=len(recovery_refs),
        recovered_operation_count=len(recovered_refs),
        unresolved_failed_operation_count=len(failed_refs - recovered_refs),
        orphan_outcome_count=correlation.quality["orphan_results"],
        ambiguous_outcome_count=sum(
            correlation.quality[key]
            for key in (
                "ambiguous_results",
                "ambiguous_tool_use_results",
                "multiple_results",
                "before_call_results",
            )
        ),
        duplicate_call_id_event_count=correlation.quality["duplicate_call_ids"],
        unobserved_operation_count=correlation.quality["missing_results"],
    )
    return _BehaviorAnalysis(summary=summary, calls=call_states, outcomes=outcomes)


def _copy_safe_fields(data: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in keys:
        value = data.get(key)
        if isinstance(value, (str, int, float, bool)) and value != "":
            result[key] = value
    return result


def _health_manifest_payload(health: HealthReport) -> dict[str, Any]:
    result = health.result
    reasons = []
    for reason in result.reasons:
        item = {
            "code": reason.code,
            "kind": reason.kind.value,
        }
        if reason.kind.value == "provider_limitation":
            item["message"] = reason.message
        reasons.append(item)
    entry = None
    if health.entry is not None:
        entry = {
            key: health.entry.get(key)
            for key in (
                "id",
                "name",
                "capture_method",
                "verification_status",
                "tested_version",
                "last_verified",
                "coverage",
            )
        }
    return {
        "schema_version": result.schema_version,
        "status": result.status.value,
        "session_finalized": health.session_finalized,
        "observed_start": result.observed_start,
        "observed_end": result.observed_end,
        "event_count": result.event_count,
        "event_types": dict(health.event_types),
        "reasons": reasons,
        "known_gaps": [spot.to_dict() for spot in health.blind_spots],
        "capture_matrix": {
            "schema_version": health.matrix.schema_version,
            "version": health.matrix.version,
            "entry": entry,
            "note": health.note or None,
        },
    }


def _minimized_data(
    event: TraceEvent,
    index: int,
    behavior: _BehaviorAnalysis,
) -> dict[str, Any]:
    data = event.data
    event_type = event.event_type

    if event_type == EventType.SESSION_START:
        result = _copy_safe_fields(
            data,
            ("provider", "source", "mode", "hook_event", "redaction"),
        )
        omitted = sorted(set(data) - set(result))
        if omitted:
            result["omitted_fields"] = omitted
        return result

    if event_type == EventType.SESSION_END:
        result = _copy_safe_fields(data, ("exit_code", "success"))
        omitted = sorted(set(data) - set(result))
        if omitted:
            result["omitted_fields"] = omitted
        return result

    if event_type == EventType.TOOL_CALL:
        tool_name = str(data.get("tool_name", data.get("name", "")))
        arguments = data.get("arguments")
        argument_map = arguments if isinstance(arguments, Mapping) else {}
        result: dict[str, Any] = {
            "tool_name": tool_name or "unknown",
            "effect": _tool_effect(tool_name),
            "input_fields": _mapping_keys(argument_map),
            "content_omitted": True,
        }
        path_value = _first_value(argument_map, _PATH_KEYS)
        if path_value is None:
            path_value = _first_value(data, _PATH_KEYS)
        result.update(_path_category(path_value))
        state = behavior.calls.get(index)
        if state is not None:
            result.update({
                "operation_ref": state.ref,
                "attempt": state.attempt,
                "retry": state.retry,
                "recovery_attempt": state.recovery_attempt,
            })
        return result

    if event_type == EventType.TOOL_RESULT:
        result = {
            "result_fields": sorted(str(key) for key in data),
            "content_omitted": True,
        }
        outcome = behavior.outcomes.get(index)
        if outcome is not None:
            call_index, status = outcome
            result["outcome"] = status
            state = behavior.calls.get(call_index)
            if state is not None:
                result["operation_ref"] = state.ref
                result["recovered"] = status == "succeeded" and state.recovery_attempt
        result.update(_copy_safe_fields(data, ("exit_code", "success", "ok")))
        return result

    if event_type == EventType.ERROR:
        result = {
            "failure": True,
            "error_fields": sorted(str(key) for key in data),
            "content_omitted": True,
        }
        outcome = behavior.outcomes.get(index)
        if outcome is not None:
            call_index, _status = outcome
            state = behavior.calls.get(call_index)
            if state is not None:
                result["operation_ref"] = state.ref
        return result

    if event_type in (EventType.FILE_READ, EventType.FILE_WRITE):
        result = {
            "effect": event_type.value,
            "content_omitted": True,
        }
        result.update(_path_category(_first_value(data, _PATH_KEYS)))
        return result

    if event_type in (EventType.LLM_REQUEST, EventType.LLM_RESPONSE):
        result = _copy_safe_fields(
            data,
            (
                "message_count",
                "input_tokens",
                "output_tokens",
                "total_tokens",
            ),
        )
        result["content_fields"] = sorted(str(key) for key in data)
        result["content_omitted"] = True
        return result

    if event_type in (EventType.SUBAGENT_START, EventType.SUBAGENT_STOP):
        result = _copy_safe_fields(data, ("agent_type",))
        result["child_session_recorded"] = bool(data.get("child_session_id"))
        result["content_fields"] = sorted(str(key) for key in data)
        result["content_omitted"] = True
        return result

    return {
        "content_fields": sorted(str(key) for key in data),
        "content_omitted": bool(data),
    }


def _minimize_events(
    events: Sequence[TraceEvent],
    behavior: _BehaviorAnalysis,
) -> tuple[tuple[TraceEvent, ...], dict[str, str]]:
    event_indices: dict[str, list[int]] = {}
    for index, event in enumerate(events):
        if event.event_id:
            event_indices.setdefault(event.event_id, []).append(index)
    packet_refs = {
        index: f"event-{index + 1:06d}"
        for index in range(len(events))
    }
    unique_event_refs = {
        event_id: packet_refs[indices[0]]
        for event_id, indices in event_indices.items()
        if len(indices) == 1
    }
    minimized: list[TraceEvent] = []
    for index, event in enumerate(events):
        selected = copy.copy(event)
        selected.data = _minimized_data(event, index, behavior)
        selected.event_id = packet_refs[index]
        selected.parent_id = unique_event_refs.get(event.parent_id, "")
        selected.prev_hash = ""
        selected.tenant_id = ""
        minimized.append(selected)
    return tuple(minimized), unique_event_refs


def _minimize_annotations(
    annotations: Sequence[Annotation],
    event_refs: Mapping[str, str],
) -> tuple[Annotation, ...]:
    selected: list[Annotation] = []
    for annotation in annotations:
        if not annotation.shareable:
            continue
        minimized = copy.copy(annotation)
        minimized.annotation_id = f"annotation-{len(selected) + 1:06d}"
        minimized.event_id = event_refs.get(annotation.event_id, "")
        minimized.author = ""
        selected.append(minimized)
    return tuple(selected)


def _limitations(include_content: bool) -> list[str]:
    limitations = [
        "The packet contains only captured evidence and cannot prove that uncaptured actions did not occur.",
        "Heuristic redaction can miss secrets and personal data.",
        "Local SHA-256 digests are integrity references, not independent signatures or timestamps.",
    ]
    if include_content:
        limitations.append(
            "Full-content mode may contain secrets, personal data, commands, paths, and proprietary material."
        )
    else:
        limitations.append(
            "Prompts, responses, commands, paths, tool inputs, tool results, and other content-rich values are omitted."
        )
    return limitations


def _build_manifest(
    *,
    meta: SessionMeta,
    source_events: Sequence[TraceEvent],
    selected_events: Sequence[TraceEvent],
    selected_annotations: Sequence[Annotation],
    preview: DisclosurePreview,
    health: HealthReport,
    behavior: BehaviorSummary,
    include_content: bool,
) -> dict[str, Any]:
    mode = "include_content" if include_content else "minimized"
    health_payload = _health_manifest_payload(health)
    manifest: dict[str, Any] = {
        "schema": EVIDENCE_PACKET_SCHEMA,
        "schema_version": EVIDENCE_PACKET_SCHEMA_VERSION,
        "tool": {"name": "agent-strace", "version": __version__},
        "mode": mode,
        "session": {
            "session_id": meta.session_id,
            "started_at": meta.started_at,
            "ended_at": meta.ended_at,
            "event_count": len(source_events),
        },
        "disclosure": preview.to_dict(),
        "provenance": {
            "provider": health.result.provider or "unknown",
            "provider_source": health.provider.source or "unknown",
            "conflicting_providers": list(health.provider.conflicting),
            "capture_method": health.result.capture_method or "unknown",
            "capture_matrix_version": health.result.capture_matrix_version or "unknown",
        },
        "evidence_health": health_payload,
        "behavior": behavior.to_dict(),
        "privacy": {
            "source_trace_unchanged": True,
            "content": "included" if include_content else "omitted",
            "annotations": (
                "all_included"
                if include_content
                else "explicitly_shareable_only"
            ),
            "redaction": health.redaction.to_dict(),
            "anonymization": {
                "mode": "not_applied",
                "reason": (
                    "explicit_full_content_mode"
                    if include_content
                    else "identity-rich content omitted by packet minimization"
                ),
                "replacement_count": 0,
            },
        },
        "integrity": {
            "algorithm": "sha256",
            "source_trace_digest_scope": (
                "canonical session metadata and decoded ordered source events"
            ),
            "source_trace_sha256": _digest({
                "meta": json.loads(meta.to_json()),
                "events": [_event_dict(event) for event in source_events],
            }),
            "packet_events_digest_scope": "canonical selected packet events",
            "packet_events_sha256": _digest(
                [_event_dict(event) for event in selected_events]
            ),
            "included_annotations_digest_scope": (
                "canonical annotations included in the selected packet mode"
            ),
            "included_annotations_sha256": _digest(
                [_annotation_dict(annotation) for annotation in selected_annotations]
            ),
            "manifest_digest_scope": "canonical manifest with manifest_sha256 omitted",
        },
        "limitations": _limitations(include_content),
    }
    manifest["integrity"]["manifest_sha256"] = _digest(manifest)
    return manifest


def build_evidence_packet(
    store: TraceStore,
    session_id: str,
    *,
    include_content: bool = False,
) -> EvidencePacket:
    """Build a packet without modifying the source trace or annotation sidecar."""
    meta = store.load_meta(session_id)
    source_events = tuple(store.load_events(session_id))
    require_object_event_data(source_events)
    source_annotations = tuple(load_annotations(store, session_id))
    behavior = _behavior_analysis(source_events)
    if include_content:
        selected_events = tuple(copy.copy(event) for event in source_events)
        selected_annotations = source_annotations
    else:
        selected_events, event_refs = _minimize_events(source_events, behavior)
        selected_annotations = _minimize_annotations(source_annotations, event_refs)
    health = assess_session(
        store,
        session_id,
        meta,
        source_events,
        load_capture_matrix(),
    )
    known_gap_count = sum(
        reason.kind.value == "provider_limitation"
        for reason in health.result.reasons
    )
    preview = build_disclosure_preview(
        source_events,
        annotation_count=len(selected_annotations),
        source_annotation_count=len(source_annotations),
        include_content=include_content,
        evidence_health_status=health.result.status.value,
        known_gap_count=known_gap_count,
        redaction_mode=health.redaction.mode,
        anonymization_mode=(
            "not_applied_explicit_content"
            if include_content
            else "not_needed_content_omitted"
        ),
        provider=health.result.provider or "unknown",
        capture_method=health.result.capture_method or "unknown",
        capture_matrix_version=health.result.capture_matrix_version or "unknown",
    )
    manifest = _build_manifest(
        meta=meta,
        source_events=source_events,
        selected_events=selected_events,
        selected_annotations=selected_annotations,
        preview=preview,
        health=health,
        behavior=behavior.summary,
        include_content=include_content,
    )
    return EvidencePacket(
        meta=meta,
        events=selected_events,
        annotations=selected_annotations,
        preview=preview,
        health=health,
        behavior=behavior.summary,
        manifest=manifest,
    )
