"""How a session's tool operations and the session itself ended.

:func:`behavior_analysis` counts failed, retried and recovered tool
operations, and :func:`recorded_session_status` reads the recorded end state.
Evidence packets, assignment bundles and the OTLP root span read them from
here, so they agree on what a session did. The eval gate's ``session_status``
scorer applies the same :func:`session_end_status` rules.

This module imports only :mod:`correlation`, :mod:`models` and the standard
library, so any exporter can use it without an import cycle.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .correlation import correlate_tool_events
from .models import EventType, TraceEvent

_COMMAND_KEYS = ("command", "cmd", "argv")
_PATH_KEYS = ("file_path", "path", "file", "uri", "cwd", "workdir")
_FAILURE_STATUSES = frozenset({"error", "failed", "failure", "cancelled", "timeout"})

SESSION_END_REASON_KEYS = ("reason", "stop_reason", "termination_reason")
_COMPLETED_STATUSES = frozenset({"completed", "complete", "success", "succeeded"})
_TIMEOUT_STATUSES = frozenset({"timeout", "timed_out"})
_TERMINATED_STATUSES = frozenset({
    "killed", "cancelled", "canceled", "interrupted", "terminated",
    "failed", "failure", "aborted", "error",
})
_TIMEOUT_REASON_WORDS = ("timeout", "timed out")
_TERMINATED_REASON_WORDS = ("kill", "cancel", "interrupt", "terminate", "abort", "error")


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
class OperationState:
    ref: str
    attempt: int
    retry: bool
    recovery_attempt: bool


@dataclass(frozen=True)
class BehaviorAnalysis:
    summary: BehaviorSummary
    calls: dict[int, OperationState]
    outcomes: dict[int, tuple[int, str]]
    unresolved_failures: tuple[int, ...] = ()
    """Positions of the failed outcomes of unresolved operations after their
    last success, in order."""


def _first_value(data: Mapping[str, Any], keys: Sequence[str]) -> object:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _tool_signature(event: TraceEvent) -> bytes:
    data = event.data
    arguments = data.get("arguments")
    signature = {
        "tool_name": data.get("tool_name", data.get("name", "")),
        "arguments": arguments if isinstance(arguments, Mapping) else arguments,
        "command": _first_value(data, _COMMAND_KEYS),
        "path": _first_value(data, _PATH_KEYS),
    }
    try:
        return json.dumps(
            signature,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        # NaN, an infinity, a lone surrogate or keys that do not sort have no
        # canonical JSON. Equal values still give equal text here.
        return repr(signature).encode("utf-8", "surrogatepass")


def _result_failed(event: TraceEvent) -> bool:
    data = event.data
    if data.get("success") is False or data.get("ok") is False:
        return True
    exit_code = data.get("exit_code")
    if isinstance(exit_code, (int, float)) and not isinstance(exit_code, bool):
        return exit_code != 0
    status = data.get("status")
    return isinstance(status, str) and status.strip().lower() in _FAILURE_STATUSES


def behavior_analysis(events: Sequence[TraceEvent]) -> BehaviorAnalysis:
    """Count failed, retried and recovered tool operations.

    An operation is a tool name with its arguments, command and path. An
    outcome belongs to a call only through :func:`correlate_tool_events`.
    A call that repeats an earlier call is a retry, and a recovery attempt
    when an earlier call of that operation failed. Outcomes are read in call
    order: an operation recovered when its last outcome succeeded after an
    earlier failure. A failed operation whose last outcome failed is
    unresolved, even if it recovered before, and so is one retried only with
    different arguments.

    Event data must be objects and timestamps must be numbers.
    """
    failed_calls: set[int] = set()
    outcomes: dict[int, tuple[int, str]] = {}
    correlation = correlate_tool_events(events)

    for index, call_index in correlation.outcome_calls.items():
        event = events[index]
        outcome = "failed"
        if event.event_type == EventType.TOOL_RESULT and not _result_failed(event):
            outcome = "succeeded"
        else:
            failed_calls.add(call_index)
        outcomes[index] = (call_index, outcome)

    signatures: dict[bytes, tuple[str, list[int]]] = {}
    call_states: dict[int, OperationState] = {}
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
        call_states[index] = OperationState(
            ref=ref,
            attempt=attempt,
            retry=attempt > 1,
            recovery_attempt=recovery_attempt,
        )
        previous.append(index)

    outcome_of_call = {
        call_index: index for index, (call_index, _) in outcomes.items()
    }
    unresolved_refs: set[str] = set()
    unresolved_indices: list[int] = []
    for ref, calls in signatures.values():
        observed = [
            (outcome_of_call[call_index], call_index in failed_calls)
            for call_index in calls
            if call_index in outcome_of_call
        ]
        if not any(failed for _, failed in observed):
            continue
        if not observed[-1][1]:
            recovered_refs.add(ref)
            continue
        unresolved_refs.add(ref)
        # Only the failures after the last success are still unresolved.
        for index, failed in reversed(observed):
            if not failed:
                break
            unresolved_indices.append(index)
    unresolved_failures = tuple(sorted(unresolved_indices))
    summary = BehaviorSummary(
        failure_event_count=sum(event.event_type == EventType.ERROR for event in events),
        failed_operation_count=len(failed_calls),
        retry_count=sum(state.retry for state in call_states.values()),
        recovery_attempt_count=len(recovery_refs),
        recovered_operation_count=len(recovered_refs),
        unresolved_failed_operation_count=len(unresolved_refs),
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
    return BehaviorAnalysis(
        summary=summary,
        calls=call_states,
        outcomes=outcomes,
        unresolved_failures=unresolved_failures,
    )


def final_session_end_index(event_types: Sequence[str]) -> int | None:
    """Return the index of the ``session_end`` that ends the last segment.

    That is the last ``session_end``, unless a later ``session_start`` (a
    resume) reopened the session: then the session has not ended and this
    returns None. *event_types* holds each event's type, as an ``EventType``
    or its string value.
    """
    for index in range(len(event_types) - 1, -1, -1):
        if event_types[index] == EventType.SESSION_END:
            return index
        if event_types[index] == EventType.SESSION_START:
            return None
    return None


def final_session_end(events: Sequence[TraceEvent]) -> TraceEvent | None:
    """Return the ``session_end`` that ends the last segment, or None.

    See ``final_session_end_index``.
    """
    index = final_session_end_index([event.event_type for event in events])
    return None if index is None else events[index]


def session_end_status(
    status: str | None,
    reason: str,
    exit_code: int | float | None,
) -> str:
    """Return ``completed``, ``timeout`` or ``terminated`` for a ``session_end``.

    ``status`` is its ``status`` or ``session_status`` text, ``reason`` the
    text of its :data:`SESSION_END_REASON_KEYS` joined with spaces, and
    ``exit_code`` its exit code as a number. The status decides first, then
    the reason, then the exit code. Callers read these values from the event
    in their own way.
    """
    if status is not None:
        normalized = status.strip().lower().replace("-", "_")
        if normalized in _COMPLETED_STATUSES:
            return "completed"
        if normalized in _TIMEOUT_STATUSES:
            return "timeout"
        if normalized in _TERMINATED_STATUSES:
            return "terminated"
    reason = reason.lower()
    if any(word in reason for word in _TIMEOUT_REASON_WORDS):
        return "timeout"
    if any(word in reason for word in _TERMINATED_REASON_WORDS):
        return "terminated"
    if exit_code == 124:
        return "timeout"
    if exit_code is not None and exit_code != 0:
        return "terminated"
    return "completed"


def recorded_session_status(events: Sequence[TraceEvent]) -> str:
    """Return how the final ``session_end`` says the process ended.

    One of ``completed``, ``timeout``, ``terminated`` or ``not_recorded``
    (no ``session_end``, or a later ``session_start`` resumed the session).
    This is the recorded process status: ``completed`` does not mean the
    task succeeded.
    """
    end = final_session_end(events)
    if end is None:
        return "not_recorded"
    data = end.data if isinstance(end.data, dict) else {}
    explicit = data.get("status") or data.get("session_status")
    reason = " ".join(
        value for key in SESSION_END_REASON_KEYS
        if isinstance((value := data.get(key)), str)
    )
    exit_code = data.get("exit_code")
    # An int too large for a float is still a nonzero exit code.
    if not (
        (isinstance(exit_code, int) and not isinstance(exit_code, bool))
        or (isinstance(exit_code, float) and math.isfinite(exit_code))
    ):
        exit_code = None
    return session_end_status(
        explicit if isinstance(explicit, str) else None, reason, exit_code
    )


__all__ = [
    "BehaviorAnalysis",
    "BehaviorSummary",
    "OperationState",
    "SESSION_END_REASON_KEYS",
    "behavior_analysis",
    "final_session_end",
    "final_session_end_index",
    "recorded_session_status",
    "session_end_status",
]
