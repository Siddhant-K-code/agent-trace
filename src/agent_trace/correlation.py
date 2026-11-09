"""Fail-closed correlation for tool calls and recorded outcomes."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from .models import EventType, TraceEvent


@dataclass(frozen=True)
class ToolCorrelation:
    """Tool-call outcomes plus value-free correlation quality counts."""

    call_outcomes: dict[int, str]
    outcome_calls: dict[int, int]
    quality: Counter[str]


def correlate_tool_events(
    events: Sequence[TraceEvent],
    in_scope_indices: set[int] | None = None,
) -> ToolCorrelation:
    """Correlate outcomes only through explicit, unique, earlier identifiers.

    ``parent_id`` is preferred. ``tool_use_id`` is accepted when it is present
    on both events and identifies one earlier call. Conflicting identifiers,
    duplicate call IDs, multiple outcomes, orphan outcomes, and outcomes before
    calls stay unresolved.
    """
    call_outcomes: dict[int, str] = {}
    quality: Counter[str] = Counter()
    calls_by_id: dict[str, list[int]] = {}
    calls_by_tool_use: dict[str, list[int]] = {}
    for index, event in enumerate(events):
        if event.event_type == EventType.TOOL_CALL:
            calls_by_id.setdefault(event.event_id, []).append(index)
            tool_use = event.data.get("tool_use_id")
            if isinstance(tool_use, str) and tool_use:
                calls_by_tool_use.setdefault(tool_use, []).append(index)
            call_outcomes[index] = "result_unobserved"
    matched_results: Counter[int] = Counter()
    result_indices: dict[int, list[int]] = {}
    scope = in_scope_indices if in_scope_indices is not None else set(range(len(events)))

    def relevant(result_index: int, candidates: Iterable[int] = ()) -> bool:
        return result_index in scope or any(item in scope for item in candidates)

    for index, event in enumerate(events):
        if event.event_type not in {EventType.TOOL_RESULT, EventType.ERROR}:
            continue
        matched: int | None = None
        parent_candidates: list[int] = []
        if isinstance(event.parent_id, str) and event.parent_id:
            parent_candidates = calls_by_id.get(event.parent_id, [])
            prior_parent = [item for item in parent_candidates if item < index]
            later_parent = [item for item in parent_candidates if item >= index]
            if len(prior_parent) == 1 and not later_parent:
                matched = prior_parent[0]
            elif len(prior_parent) > 1 or (prior_parent and later_parent):
                if relevant(index, parent_candidates):
                    quality["ambiguous_results"] += 1
                continue
            elif later_parent:
                if relevant(index, later_parent):
                    quality["before_call_results"] += 1
                continue
        tool_matched: int | None = None
        if matched is None:
            tool_use = event.data.get("tool_use_id")
            if isinstance(tool_use, str) and tool_use:
                candidates = calls_by_tool_use.get(tool_use, [])
                prior = [item for item in candidates if item < index]
                later = [item for item in candidates if item >= index]
                if len(prior) == 1 and not later:
                    tool_matched = prior[0]
                elif len(prior) > 1 or (prior and later):
                    if relevant(index, candidates):
                        quality["ambiguous_tool_use_results"] += 1
                    continue
                elif later:
                    if relevant(index, later):
                        quality["before_call_results"] += 1
                    continue
            if tool_matched is not None:
                matched = tool_matched
        else:
            tool_use = event.data.get("tool_use_id")
            if isinstance(tool_use, str) and tool_use:
                candidates = [
                    item for item in calls_by_tool_use.get(tool_use, []) if item < index
                ]
                if len(candidates) != 1 or candidates[0] != matched:
                    if relevant(index, [matched, *candidates]):
                        quality["ambiguous_results"] += 1
                    continue
        if matched is None:
            if relevant(index):
                quality["orphan_results"] += 1
            continue
        if float(event.timestamp) < float(events[matched].timestamp):
            if relevant(index, [matched]):
                quality["before_call_results"] += 1
            continue
        if matched not in scope:
            if index in scope:
                quality["results_for_out_of_window_calls"] += 1
            continue
        if index not in scope:
            quality["boundary_context_results"] += 1
            if call_outcomes.get(matched) == "result_unobserved":
                call_outcomes[matched] = (
                    "boundary_context_error"
                    if event.event_type == EventType.ERROR
                    else "boundary_context_result"
                )
            continue
        matched_results[matched] += 1
        result_indices.setdefault(matched, []).append(index)
        if matched_results[matched] > 1:
            call_outcomes[matched] = "ambiguous_result"
            if matched in scope:
                quality["multiple_results"] += 1
        elif event.event_type == EventType.ERROR:
            call_outcomes[matched] = "recorded_error"
        else:
            call_outcomes[matched] = "recorded_result"
    quality["missing_results"] = sum(
        1
        for call_index, outcome in call_outcomes.items()
        if call_index in scope and outcome == "result_unobserved"
    )
    quality["duplicate_call_ids"] = sum(
        len([item for item in indices if item in scope])
        for indices in calls_by_id.values()
        if len(indices) > 1 and any(item in scope for item in indices)
    )
    outcome_calls = {
        indices[0]: call_index
        for call_index, indices in result_indices.items()
        if len(indices) == 1 and call_outcomes.get(call_index) != "ambiguous_result"
    }
    return ToolCorrelation(
        call_outcomes=call_outcomes,
        outcome_calls=outcome_calls,
        quality=quality,
    )
