"""Instruction and skill compliance evidence for recorded eval runs.

The report is an evidence inventory, not an outcome or developer score.
It preserves whether each claim is observed, inferred, missing, or
unavailable, and it keeps post-load validation facts separate from causality.
"""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .context_score import context_file_load_events
from .models import EventType, TraceEvent
from .postmortem import _detect_agents_md_violations, _load_agents_md
from .review import is_test_command
from .skills import (
    CONTEXT_FILES,
    USER_SKILL_DIRS,
    _is_edit,
    analyze_session,
    discover_skills,
    event_arguments,
)

SCHEMA_VERSION = 1
EVIDENCE_STATES = ("observed", "inferred", "missing", "unavailable")
ACTION_KINDS = ("file_read", "file_write", "tool_call", "test")

LIMITATIONS = (
    "This is evidence about recorded actions, not an outcome or developer score.",
    "An inferred instruction load means the file was available to a client that commonly injects repository instructions, but no read event was recorded.",
    "A missing action means the bounded trace did not contain the expected observable event. It does not prove that prose guidance was ignored.",
    "Post-load validation facts are associated by event order only. They are not causal.",
    "Availability is read from the benchmark workspace at analysis time.",
)


@dataclass(frozen=True)
class ActionExpectation:
    id: str
    kind: str
    path: str = ""
    tool: str = ""

    def to_dict(self) -> dict[str, str]:
        data = {"id": self.id, "kind": self.kind}
        if self.path:
            data["path"] = self.path
        if self.tool:
            data["tool"] = self.tool
        return data


def _short_skill_name(name: str) -> str:
    return name.rsplit(":", 1)[-1]


def _skill_matches(loaded_name: str, wanted: str) -> bool:
    return loaded_name == wanted or _short_skill_name(loaded_name) == _short_skill_name(wanted)


def _tool_name(event: TraceEvent) -> str:
    data = event.data if isinstance(event.data, Mapping) else {}
    return str(data.get("tool_name") or data.get("tool") or "").lower()


def _tool_command(event: TraceEvent) -> str:
    arguments = event_arguments(event)
    return str(arguments.get("command") or arguments.get("cmd") or arguments.get("script") or "")


def _event_path(event: TraceEvent) -> str:
    data = event.data if isinstance(event.data, Mapping) else {}
    if event.event_type in (EventType.FILE_READ, EventType.FILE_WRITE):
        return str(data.get("path") or data.get("file_path") or data.get("uri") or "")
    arguments = event_arguments(event)
    return str(
        arguments.get("file_path")
        or arguments.get("path")
        or arguments.get("absolute_path")
        or arguments.get("filePath")
        or arguments.get("target_file")
        or arguments.get("uri")
        or ""
    )


def _relative_path(raw: str, workspace: Path) -> str:
    if not raw:
        return ""
    candidate = Path(os.path.expanduser(raw))
    if not candidate.is_absolute():
        return candidate.as_posix().lstrip("./")
    try:
        return candidate.resolve().relative_to(workspace.resolve()).as_posix()
    except (OSError, ValueError):
        return candidate.as_posix()


def _path_matches(raw: str, pattern: str, workspace: Path) -> bool:
    relative = _relative_path(raw, workspace)
    normalized = raw.replace("\\", "/")
    return fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(normalized, pattern)


def _is_test_event(event: TraceEvent) -> bool:
    data = event.data if isinstance(event.data, Mapping) else {}
    if data.get("is_test") is True or data.get("category") == "test":
        return True
    return event.event_type == EventType.TOOL_CALL and is_test_command(_tool_command(event))


def _matches_action(event: TraceEvent, expectation: ActionExpectation, workspace: Path) -> bool:
    if not isinstance(event.data, Mapping):
        return False
    if expectation.kind == "test":
        return _is_test_event(event)
    if expectation.kind == "tool_call":
        return event.event_type == EventType.TOOL_CALL and _tool_name(event) == expectation.tool.lower()
    if expectation.kind == "file_read":
        return event.event_type == EventType.FILE_READ and _path_matches(_event_path(event), expectation.path, workspace)
    if expectation.kind == "file_write":
        return (
            event.event_type == EventType.FILE_WRITE or _is_edit(event)
        ) and _path_matches(_event_path(event), expectation.path, workspace)
    return False


def _linked_results(events: Iterable[TraceEvent]) -> dict[str, TraceEvent]:
    return {
        event.parent_id: event
        for event in events
        if event.parent_id and event.event_type in (EventType.TOOL_RESULT, EventType.ERROR)
    }


def _result_fact(event: TraceEvent | None) -> dict[str, object] | None:
    if event is None:
        return None
    data = event.data if isinstance(event.data, dict) else {}
    fact: dict[str, object] = {"event_type": event.event_type.value}
    for key in ("exit_code", "success"):
        if key in data and isinstance(data[key], (bool, int)):
            fact[key] = data[key]
    return fact


def _validation_facts(events: list[TraceEvent], after_timestamp: float | None) -> dict[str, object]:
    if after_timestamp is None:
        return {
            "tests_run": 0,
            "last_test": None,
            "errors": 0,
            "session_end_recorded": False,
        }
    after = [event for event in events if event.timestamp > after_timestamp]
    tests = [event for event in after if _is_test_event(event)]
    linked = _linked_results(after)
    last_test = None
    if tests:
        last = tests[-1]
        last_test = {
            "event_id": last.event_id,
            "result": _result_fact(linked.get(last.event_id)),
        }
    return {
        "tests_run": len(tests),
        "last_test": last_test,
        "errors": sum(1 for event in after if event.event_type == EventType.ERROR),
        "session_end_recorded": any(event.event_type == EventType.SESSION_END for event in after),
    }


def _skill_availability(
    workspace: Path,
    skill: str,
    extra_dirs: Iterable[str],
    user_dirs: Iterable[str],
) -> bool:
    catalog = discover_skills(workspace, extra_dirs, user_dirs=user_dirs)
    return any(_skill_matches(name, skill) for name in catalog.skills)


def _skill_load_evidence(
    reports: list[dict[str, Any]],
    skill: str,
    should_trigger: bool,
    available: bool,
    has_events: bool,
) -> tuple[dict[str, object], list[float]]:
    model_loads: list[dict[str, object]] = []
    user_loads = 0
    for report in reports:
        for entry in report.get("loaded", ()):
            if not _skill_matches(str(entry.get("name", "")), skill):
                continue
            for load in entry.get("loads", ()):
                if load.get("invoked_by") == "model":
                    model_loads.append(load)
                else:
                    user_loads += 1

    expected = "loaded" if should_trigger else "not_loaded"
    if not has_events or not available:
        state = "unavailable"
        actual = "unavailable"
        met = None
    elif model_loads:
        state = "observed"
        actual = "loaded"
        met = should_trigger
    elif user_loads:
        state = "observed"
        actual = "user_invoked"
        met = False if should_trigger else True
    else:
        state = "missing" if should_trigger else "observed"
        actual = "not_loaded"
        met = not should_trigger

    return ({
        "kind": "skill",
        "name": skill,
        "availability": {
            "actual": "available" if available else "unavailable",
            "evidence": "observed" if available else "missing",
        },
        "expected": expected,
        "actual": actual,
        "evidence": state,
        "expectation_met": met,
        "model_loads": len(model_loads),
        "user_loads": user_loads,
    }, [float(load["timestamp"]) for load in model_loads if isinstance(load.get("timestamp"), (int, float))])


def _skill_follow_evidence(
    reports: list[dict[str, Any]],
    skill: str,
    should_trigger: bool,
    has_events: bool,
) -> dict[str, object] | None:
    if not should_trigger:
        return None
    details = []
    for report in reports:
        for entry in report.get("loaded", ()):
            if not _skill_matches(str(entry.get("name", "")), skill):
                continue
            if not any(load.get("invoked_by") == "model" for load in entry.get("loads", ())):
                continue
            follow = entry.get("follow", {})
            details.append({
                "session": report.get("session_id", ""),
                "status": follow.get("status", "not_measurable"),
                "reason": follow.get("reason"),
                "signals": follow.get("signals", {}),
            })

    statuses = {str(detail["status"]) for detail in details}
    if "signals_observed" in statuses:
        actual = "signals_observed"
        evidence = "observed"
        met: bool | None = True
    elif "no_signal" in statuses:
        actual = "no_signal"
        evidence = "missing"
        met = False
    elif details:
        actual = "not_measurable"
        evidence = "unavailable"
        met = None
    elif has_events:
        actual = "no_skill_load"
        evidence = "unavailable"
        met = None
    else:
        actual = "unavailable"
        evidence = "unavailable"
        met = None
    return {
        "kind": "skill_follow",
        "name": skill,
        "expected": "observable_signal_after_load",
        "actual": actual,
        "evidence": evidence,
        "expectation_met": met,
        "session_evidence": details,
    }


def _instruction_load_evidence(
    events: list[TraceEvent],
    workspace: Path,
    instruction: str,
) -> tuple[dict[str, object], list[float]]:
    path = workspace / instruction
    available = os.path.isfile(path)
    loads = context_file_load_events(events, workspace, instruction)
    if loads:
        actual = "loaded"
        evidence = "observed"
        met: bool | None = True
    elif not available:
        actual = "unavailable"
        evidence = "unavailable"
        met = False
    elif not events:
        actual = "unavailable"
        evidence = "unavailable"
        met = None
    elif instruction in CONTEXT_FILES:
        actual = "possibly_loaded"
        evidence = "inferred"
        met = None
    elif events:
        actual = "not_loaded"
        evidence = "missing"
        met = False
    else:
        actual = "unavailable"
        evidence = "unavailable"
        met = None
    violations = (
        [item.replace(" \u2014 ", ": ") for item in _detect_agents_md_violations(events, _load_agents_md(path))]
        if available
        else []
    )
    return ({
        "kind": "instruction",
        "name": instruction,
        "availability": {
            "actual": "available" if available else "unavailable",
            "evidence": "observed" if available else "missing",
        },
        "expected": "loaded",
        "actual": actual,
        "evidence": evidence,
        "expectation_met": met,
        "observed_loads": len(loads),
        "violation_facts": violations,
    }, [event.timestamp for event in loads])


def _action_evidence(
    expectation: ActionExpectation,
    events: list[TraceEvent],
    workspace: Path,
    observed_anchor: float | None,
    inferred_anchor: bool,
    load_available: bool,
) -> dict[str, object]:
    matches = [event for event in events if _matches_action(event, expectation, workspace)]
    after = [event for event in matches if observed_anchor is None or event.timestamp > observed_anchor]
    if observed_anchor is not None and after:
        evidence = "observed"
        met: bool | None = True
    elif observed_anchor is None and inferred_anchor and matches:
        evidence = "inferred"
        met = None
    elif not events or not load_available:
        evidence = "unavailable"
        met = None
    else:
        evidence = "missing"
        met = False
    return {
        **expectation.to_dict(),
        "expected": "observed_after_load",
        "actual": "observed" if matches else "not_observed",
        "evidence": evidence,
        "expectation_met": met,
        "matching_events": len(after if observed_anchor is not None else matches),
    }


def evidence_counts(items: Iterable[Mapping[str, object]]) -> dict[str, int]:
    counts = {state: 0 for state in EVIDENCE_STATES}
    for item in items:
        state = item.get("evidence")
        if state in counts:
            counts[str(state)] += 1
    return counts


def analyze_compliance_run(
    session_events: list[tuple[str, list[TraceEvent]]],
    *,
    workspace: str,
    skill: str,
    should_trigger: bool,
    instructions: Iterable[str],
    expected_actions: Iterable[ActionExpectation],
    extra_dirs: Iterable[str] = (),
    user_dirs: Iterable[str] = USER_SKILL_DIRS,
) -> dict[str, object]:
    root = Path(os.path.expanduser(workspace or os.getcwd()))
    extra = list(extra_dirs)
    users = list(user_dirs)
    reports = []
    for session_id, session in session_events:
        try:
            reports.append(analyze_session(
                session,
                session_id=session_id,
                workspace=str(root),
                extra_dirs=extra,
                user_dirs=users,
            ))
        except (ValueError, TypeError, KeyError, OSError):
            continue
    events = sorted(
        [event for _, session in session_events for event in session],
        key=lambda event: (event.timestamp, event.event_id),
    )
    has_events = bool(events)

    skill_available = _skill_availability(root, skill, extra, users)
    skill_item, skill_anchors = _skill_load_evidence(
        reports, skill, should_trigger, skill_available, has_events,
    )
    load_items = [skill_item]
    instruction_anchors: list[float] = []
    for instruction in instructions:
        item, anchors = _instruction_load_evidence(events, root, instruction)
        load_items.append(item)
        instruction_anchors.extend(anchors)

    anchors = skill_anchors + instruction_anchors
    observed_anchor = min(anchors) if anchors else None
    inferred_anchor = any(item["evidence"] == "inferred" for item in load_items)
    load_available = any(item["actual"] in ("loaded", "possibly_loaded") for item in load_items)
    actions = [
        _action_evidence(
            expectation,
            events,
            root,
            observed_anchor,
            inferred_anchor,
            load_available,
        )
        for expectation in expected_actions
    ]
    skill_follow = _skill_follow_evidence(reports, skill, should_trigger, has_events)
    follow_items = ([skill_follow] if skill_follow is not None else []) + actions

    if observed_anchor is not None:
        validation_evidence = "observed"
    elif inferred_anchor and has_events:
        validation_evidence = "inferred"
    else:
        validation_evidence = "unavailable"
    validation = {
        "evidence": validation_evidence,
        "anchor": (
            "observed_load"
            if observed_anchor is not None
            else ("inferred_instruction_load" if inferred_anchor and has_events else "none")
        ),
        **_validation_facts(events, observed_anchor if observed_anchor is not None else (min(
            (event.timestamp for event in events), default=None
        ) if inferred_anchor else None)),
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "not_a_score": True,
        "loads": load_items,
        "follow": {
            "skill": skill_follow,
            "instruction_actions": actions,
        },
        "validation": validation,
        "evidence_counts": {
            "loads": evidence_counts(load_items),
            "follow": evidence_counts(follow_items),
            "validation": evidence_counts([validation]),
        },
        "limitations": list(LIMITATIONS),
    }


def summarize_compliance_runs(runs: Iterable[Mapping[str, object]]) -> dict[str, object]:
    compliance = [
        run["compliance"]
        for run in runs
        if isinstance(run.get("compliance"), Mapping)
    ]
    load_items = [
        item
        for report in compliance
        for item in report.get("loads", ())
        if isinstance(item, Mapping)
    ]
    follow_items = [
        item
        for report in compliance
        for item in (
            ([report["follow"]["skill"]] if report.get("follow", {}).get("skill") is not None else [])
            + list(report.get("follow", {}).get("instruction_actions", ()))
        )
        if isinstance(item, Mapping)
    ]
    validation_items = [
        report["validation"]
        for report in compliance
        if isinstance(report.get("validation"), Mapping)
    ]
    return {
        "runs": len(compliance),
        "loads": evidence_counts(load_items),
        "follow": evidence_counts(follow_items),
        "validation": evidence_counts(validation_items),
    }


__all__ = [
    "ACTION_KINDS",
    "EVIDENCE_STATES",
    "LIMITATIONS",
    "SCHEMA_VERSION",
    "ActionExpectation",
    "analyze_compliance_run",
    "evidence_counts",
    "summarize_compliance_runs",
]
