"""Deterministic evidence-health assessment for captured agent sessions.

The goal of evidence health is narrower than judging whether an agent behaved
correctly. It answers whether the *recorded evidence* is structurally usable for
review and which limitations a reviewer must keep in mind.

The calculation is deliberately local and dependency-free. Provider blind
spots are supplied explicitly by capture adapters/matrices instead of inferred
from missing events, so a clean event stream is never presented as proof that a
provider exposed everything that happened.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Iterable, Sequence

from .models import EventType, TraceEvent

EVIDENCE_HEALTH_SCHEMA_VERSION = 1


class EvidenceHealthStatus(str, Enum):
    """Reviewability state for one captured session."""

    HEALTHY = "healthy"
    PARTIAL = "partial"
    UNKNOWN = "unknown"
    INVALID = "invalid"


class EvidenceHealthReasonKind(str, Enum):
    """Whether a reason comes from observed evidence or declared capture limits."""

    OBSERVED = "observed"
    PROVIDER_LIMITATION = "provider_limitation"


@dataclass(frozen=True)
class EvidenceHealthReason:
    """Machine-readable reason explaining an evidence-health result."""

    code: str
    message: str
    kind: EvidenceHealthReasonKind = EvidenceHealthReasonKind.OBSERVED
    event_id: str = ""


@dataclass(frozen=True)
class EvidenceHealthResult:
    """Versioned result returned by :func:`assess_evidence_health`."""

    status: EvidenceHealthStatus
    reasons: tuple[EvidenceHealthReason, ...] = field(default_factory=tuple)
    schema_version: int = EVIDENCE_HEALTH_SCHEMA_VERSION
    provider: str = ""
    capture_method: str = ""
    observed_start: bool = False
    observed_end: bool = False
    event_count: int = 0

    def to_dict(self) -> dict:
        """Return a stable JSON-compatible representation."""
        payload = asdict(self)
        payload["status"] = self.status.value
        for reason in payload["reasons"]:
            reason["kind"] = reason["kind"].value
        return payload


@dataclass(frozen=True)
class _RelationshipSpec:
    request_type: EventType
    result_type: EventType
    missing_result_code: str
    orphan_result_code: str
    label: str


_RELATIONSHIPS = (
    _RelationshipSpec(
        request_type=EventType.TOOL_CALL,
        result_type=EventType.TOOL_RESULT,
        missing_result_code="unpaired_tool_call",
        orphan_result_code="orphan_tool_result",
        label="tool",
    ),
    _RelationshipSpec(
        request_type=EventType.LLM_REQUEST,
        result_type=EventType.LLM_RESPONSE,
        missing_result_code="unpaired_llm_request",
        orphan_result_code="orphan_llm_response",
        label="LLM",
    ),
)


def _reason(
    code: str,
    message: str,
    *,
    kind: EvidenceHealthReasonKind = EvidenceHealthReasonKind.OBSERVED,
    event: TraceEvent | None = None,
) -> EvidenceHealthReason:
    return EvidenceHealthReason(
        code=code,
        message=message,
        kind=kind,
        event_id=event.event_id if event is not None else "",
    )


def _relationship_reasons(
    events: Sequence[TraceEvent],
    spec: _RelationshipSpec,
) -> list[EvidenceHealthReason]:
    requests = {event.event_id: event for event in events if event.event_type == spec.request_type}
    results = [event for event in events if event.event_type == spec.result_type]
    paired_request_ids: set[str] = set()
    reasons: list[EvidenceHealthReason] = []

    for result in results:
        parent_id = result.parent_id.strip()
        if parent_id and parent_id in requests:
            paired_request_ids.add(parent_id)
            continue

        detail = (
            f" references unknown parent {parent_id!r}"
            if parent_id
            else " has no parent_id"
        )
        reasons.append(
            _reason(
                spec.orphan_result_code,
                f"{spec.label} result {result.event_id!r}{detail}",
                event=result,
            )
        )

    for request_id, request in requests.items():
        if request_id not in paired_request_ids:
            reasons.append(
                _reason(
                    spec.missing_result_code,
                    f"{spec.label} request {request_id!r} has no recorded result",
                    event=request,
                )
            )

    return reasons


def _structural_reasons(
    events: Sequence[TraceEvent],
    *,
    session_finalized: bool | None,
) -> tuple[list[EvidenceHealthReason], bool]:
    reasons: list[EvidenceHealthReason] = []
    invalid = False

    starts = [event for event in events if event.event_type == EventType.SESSION_START]
    ends = [event for event in events if event.event_type == EventType.SESSION_END]

    if not starts:
        reasons.append(_reason("missing_session_start", "session start marker was not captured"))
    elif len(starts) > 1:
        reasons.append(
            _reason(
                "duplicate_session_start",
                f"captured {len(starts)} session start markers",
                event=starts[1],
            )
        )
        invalid = True

    if not ends:
        if session_finalized is False:
            reasons.append(_reason("session_active", "session is still active; no end marker expected yet"))
        else:
            reasons.append(_reason("missing_session_end", "session end marker was not captured"))
    elif len(ends) > 1:
        reasons.append(
            _reason(
                "duplicate_session_end",
                f"captured {len(ends)} session end markers",
                event=ends[1],
            )
        )
        invalid = True

    if starts and events[0] is not starts[0]:
        reasons.append(
            _reason(
                "events_before_session_start",
                "events were recorded before the first session start marker",
                event=events[0],
            )
        )
        invalid = True

    if ends and events[-1] is not ends[-1]:
        reasons.append(
            _reason(
                "events_after_session_end",
                "events were recorded after the last session end marker",
                event=events[-1],
            )
        )
        invalid = True

    session_ids = {event.session_id for event in events if event.session_id}
    if len(session_ids) > 1:
        reasons.append(
            _reason(
                "mixed_session_ids",
                f"event stream contains {len(session_ids)} different session IDs",
            )
        )
        invalid = True

    previous_timestamp: float | None = None
    for event in events:
        timestamp = event.timestamp
        if not isinstance(timestamp, (int, float)) or not math.isfinite(float(timestamp)):
            reasons.append(
                _reason(
                    "invalid_timestamp",
                    f"event {event.event_id!r} has a non-finite timestamp",
                    event=event,
                )
            )
            invalid = True
            continue
        if previous_timestamp is not None and timestamp < previous_timestamp:
            reasons.append(
                _reason(
                    "timestamp_regression",
                    f"event {event.event_id!r} is earlier than the preceding event",
                    event=event,
                )
            )
        previous_timestamp = timestamp

    event_ids: set[str] = set()
    for event in events:
        if not event.event_id:
            reasons.append(_reason("missing_event_id", "captured event has no event_id", event=event))
            invalid = True
            continue
        if event.event_id in event_ids:
            reasons.append(
                _reason(
                    "duplicate_event_id",
                    f"event_id {event.event_id!r} appears more than once",
                    event=event,
                )
            )
            invalid = True
        event_ids.add(event.event_id)

    for spec in _RELATIONSHIPS:
        reasons.extend(_relationship_reasons(events, spec))

    return reasons, invalid


def assess_evidence_health(
    events: Iterable[TraceEvent],
    *,
    provider: str = "",
    capture_method: str = "",
    provider_blind_spots: Iterable[str] = (),
    session_finalized: bool | None = None,
    export_failures: Iterable[str] = (),
) -> EvidenceHealthResult:
    """Assess whether a captured session is structurally reviewable.

    ``provider_blind_spots`` must come from declared provider/capture metadata.
    This function never invents a provider limitation from an absent event.

    ``session_finalized=False`` distinguishes an active session from a completed
    session whose end marker is unexpectedly absent.
    """
    event_list = list(events)
    no_events = not event_list
    if no_events:
        reasons = [_reason("no_events", "no captured events are available for review")]
        invalid = False
    else:
        reasons, invalid = _structural_reasons(
            event_list,
            session_finalized=session_finalized,
        )

    for failure in export_failures:
        failure_text = str(failure).strip()
        if failure_text:
            reasons.append(
                _reason(
                    "export_failure",
                    f"capture/export pipeline reported a failure: {failure_text}",
                )
            )

    for limitation in provider_blind_spots:
        limitation_text = str(limitation).strip()
        if limitation_text:
            reasons.append(
                _reason(
                    "provider_blind_spot",
                    limitation_text,
                    kind=EvidenceHealthReasonKind.PROVIDER_LIMITATION,
                )
            )

    active_only = (
        session_finalized is False
        and reasons
        and all(reason.code == "session_active" for reason in reasons)
    )

    if invalid:
        status = EvidenceHealthStatus.INVALID
    elif no_events or active_only:
        status = EvidenceHealthStatus.UNKNOWN
    elif reasons:
        status = EvidenceHealthStatus.PARTIAL
    else:
        status = EvidenceHealthStatus.HEALTHY

    return EvidenceHealthResult(
        status=status,
        reasons=tuple(reasons),
        provider=provider,
        capture_method=capture_method,
        observed_start=any(event.event_type == EventType.SESSION_START for event in event_list),
        observed_end=any(event.event_type == EventType.SESSION_END for event in event_list),
        event_count=len(event_list),
    )
