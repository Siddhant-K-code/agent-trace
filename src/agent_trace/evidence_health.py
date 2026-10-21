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

# ``session_start`` events carry the provider's start ``source``. A later start
# with one of these sources continues the session already recorded under the
# same session ID instead of starting another one: Claude Code and Codex keep
# the session ID on ``resume`` and ``compact``, and Gemini CLI and GitHub
# Copilot CLI keep it on ``resume``. Sources that begin a new session ID
# (``startup``, ``clear``, ``fork``, ``new``), a missing source and hook
# event-name fallbacks such as ``SessionStart`` are deliberately absent, so a
# repeated start of that kind stays a duplicate. Matching is exact.
CONTINUATION_START_SOURCES = frozenset({"resume", "compact"})


def is_continuation_source(source: object) -> bool:
    """Return whether a session-start ``source`` continues an existing session."""
    return isinstance(source, str) and source in CONTINUATION_START_SOURCES


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
    capture_matrix_version: str = ""
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
    duplicate_outcome_code: str
    out_of_order_outcome_code: str
    label: str
    error_is_terminal: bool = False


_RELATIONSHIPS = (
    _RelationshipSpec(
        request_type=EventType.TOOL_CALL,
        result_type=EventType.TOOL_RESULT,
        missing_result_code="unpaired_tool_call",
        orphan_result_code="orphan_tool_result",
        duplicate_outcome_code="duplicate_tool_outcome",
        out_of_order_outcome_code="out_of_order_tool_outcome",
        label="tool",
        error_is_terminal=True,
    ),
    _RelationshipSpec(
        request_type=EventType.LLM_REQUEST,
        result_type=EventType.LLM_RESPONSE,
        missing_result_code="unpaired_llm_request",
        orphan_result_code="orphan_llm_response",
        duplicate_outcome_code="duplicate_llm_outcome",
        out_of_order_outcome_code="out_of_order_llm_outcome",
        label="LLM",
        error_is_terminal=True,
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
    *,
    require_terminal_outcome: bool,
) -> list[EvidenceHealthReason]:
    request_positions = {
        event.event_id: index
        for index, event in enumerate(events)
        if event.event_type == spec.request_type
    }
    requests = {
        event.event_id: event
        for event in events
        if event.event_type == spec.request_type
    }
    outcomes_by_request: dict[str, list[tuple[int, TraceEvent]]] = {}
    reasons: list[EvidenceHealthReason] = []

    for index, event in enumerate(events):
        is_result = event.event_type == spec.result_type
        is_terminal_error = (
            spec.error_is_terminal
            and event.event_type == EventType.ERROR
            and event.parent_id.strip() in requests
        )
        if not is_result and not is_terminal_error:
            continue

        parent_id = event.parent_id.strip()
        if parent_id and parent_id in requests:
            outcomes_by_request.setdefault(parent_id, []).append((index, event))
            continue

        if is_result:
            detail = (
                f" references unknown parent {parent_id!r}"
                if parent_id
                else " has no parent_id"
            )
            reasons.append(
                _reason(
                    spec.orphan_result_code,
                    f"{spec.label} result {event.event_id!r}{detail}",
                    event=event,
                )
            )

    for request_id, request in requests.items():
        outcomes = outcomes_by_request.get(request_id, [])
        if not outcomes:
            if require_terminal_outcome:
                reasons.append(
                    _reason(
                        spec.missing_result_code,
                        f"{spec.label} request {request_id!r} has no recorded terminal outcome",
                        event=request,
                    )
                )
            continue

        if len(outcomes) > 1:
            reasons.append(
                _reason(
                    spec.duplicate_outcome_code,
                    f"{spec.label} request {request_id!r} has {len(outcomes)} recorded terminal outcomes",
                    event=outcomes[1][1],
                )
            )

        request_position = request_positions[request_id]
        for outcome_position, outcome in outcomes:
            if outcome_position < request_position:
                reasons.append(
                    _reason(
                        spec.out_of_order_outcome_code,
                        f"{spec.label} outcome {outcome.event_id!r} appears before request {request_id!r}",
                        event=outcome,
                    )
                )

    return reasons


def _orphan_error_reasons(events: Sequence[TraceEvent]) -> list[EvidenceHealthReason]:
    """Report ERROR events that cannot be tied to any tool or LLM request."""
    request_ids = {
        event.event_id
        for event in events
        if event.event_type in (EventType.TOOL_CALL, EventType.LLM_REQUEST)
    }
    reasons: list[EvidenceHealthReason] = []
    for event in events:
        if event.event_type != EventType.ERROR:
            continue
        parent_id = event.parent_id.strip()
        if not parent_id:
            reasons.append(
                _reason(
                    "unlinked_error",
                    f"error event {event.event_id!r} has no parent_id",
                    event=event,
                )
            )
        elif parent_id not in request_ids:
            reasons.append(
                _reason(
                    "orphan_error",
                    f"error event {event.event_id!r} references unknown parent {parent_id!r}",
                    event=event,
                )
            )
    return reasons


def _start_source(event: TraceEvent) -> object:
    return event.data.get("source") if isinstance(event.data, dict) else None


def _boundary_reasons(
    events: Sequence[TraceEvent],
    *,
    session_finalized: bool | None,
) -> tuple[list[EvidenceHealthReason], bool]:
    """Check session start and end markers in event order.

    The first start opens the session and an end closes it. A later start whose
    source is in :data:`CONTINUATION_START_SOURCES` continues the same session
    and re-opens it if it was closed; any other later start is a duplicate.
    While the session is closed, another end is a duplicate and any other event
    was recorded after the session end.
    """
    first_start: TraceEvent | None = None
    closing_end: TraceEvent | None = None
    reopened = False
    after_end_reported = False
    duplicate_starts: list[EvidenceHealthReason] = []
    duplicate_ends: list[EvidenceHealthReason] = []
    after_end: list[EvidenceHealthReason] = []

    for event in events:
        if event.event_type == EventType.SESSION_START:
            source = _start_source(event)
            continuation = is_continuation_source(source)
            if first_start is None:
                first_start = event
            elif not continuation:
                described = "no source" if source in (None, "") else f"source {source!r}"
                allowed = " and ".join(sorted(CONTINUATION_START_SOURCES))
                duplicate_starts.append(
                    _reason(
                        "duplicate_session_start",
                        f"captured another session start with {described}; "
                        f"only {allowed} starts continue a session",
                        event=event,
                    )
                )
            if continuation and closing_end is not None:
                closing_end = None
                reopened = True
        elif event.event_type == EventType.SESSION_END:
            if closing_end is None:
                closing_end = event
                after_end_reported = False
            else:
                duplicate_ends.append(
                    _reason(
                        "duplicate_session_end",
                        "captured another session end with no continuation start "
                        f"since session end {closing_end.event_id!r}",
                        event=event,
                    )
                )
        elif closing_end is not None and not after_end_reported:
            after_end.append(
                _reason(
                    "events_after_session_end",
                    f"events were recorded after session end {closing_end.event_id!r} "
                    "with no continuation start",
                    event=event,
                )
            )
            after_end_reported = True

    reasons: list[EvidenceHealthReason] = []
    if first_start is None:
        reasons.append(_reason("missing_session_start", "session start marker was not captured"))
    reasons.extend(duplicate_starts)

    if closing_end is None:
        if session_finalized is False:
            reasons.append(_reason("session_active", "session is still active; no end marker expected yet"))
        elif reopened:
            reasons.append(
                _reason(
                    "missing_session_end",
                    "session end marker was not captured after the session was continued",
                )
            )
        else:
            reasons.append(_reason("missing_session_end", "session end marker was not captured"))
    reasons.extend(duplicate_ends)

    events_before_start = first_start is not None and events[0] is not first_start
    if events_before_start:
        reasons.append(
            _reason(
                "events_before_session_start",
                "events were recorded before the first session start marker",
                event=events[0],
            )
        )
    reasons.extend(after_end)

    invalid = bool(duplicate_starts or duplicate_ends or after_end or events_before_start)
    return reasons, invalid


def _structural_reasons(
    events: Sequence[TraceEvent],
    *,
    session_finalized: bool | None,
) -> tuple[list[EvidenceHealthReason], bool]:
    reasons, invalid = _boundary_reasons(events, session_finalized=session_finalized)

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
        reasons.extend(
            _relationship_reasons(
                events,
                spec,
                require_terminal_outcome=session_finalized is not False,
            )
        )
    reasons.extend(_orphan_error_reasons(events))

    return reasons, invalid


def assess_evidence_health(
    events: Iterable[TraceEvent],
    *,
    provider: str = "",
    capture_method: str = "",
    capture_matrix_version: str = "",
    provider_blind_spots: Iterable[str] = (),
    session_finalized: bool | None = None,
    export_failures: Iterable[str] = (),
) -> EvidenceHealthResult:
    """Assess whether a captured session is structurally reviewable.

    ``provider_blind_spots`` must come from declared provider/capture metadata.
    Pass its version or stable reference in ``capture_matrix_version`` so callers
    can interpret limitations after the capture matrix changes. This function
    never invents a provider limitation from an absent event.

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
        capture_matrix_version=capture_matrix_version,
        observed_start=any(event.event_type == EventType.SESSION_START for event in event_list),
        observed_end=any(event.event_type == EventType.SESSION_END for event in event_list),
        event_count=len(event_list),
    )
