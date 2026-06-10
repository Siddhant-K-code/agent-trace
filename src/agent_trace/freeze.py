"""Behavioral regression fixtures for tool-call sequences."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO

from .models import EventType, TraceEvent
from .store import TraceStore


SCHEMA_VERSION = 1


@dataclass
class FrozenStep:
    seq: int
    tool: str
    input_hash: str

    @property
    def signature(self) -> tuple[str, str]:
        return (self.tool, self.input_hash)


@dataclass
class RegressionFixture:
    session: str
    task: str
    steps: list[FrozenStep]
    schema_version: int = SCHEMA_VERSION
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session": self.session,
            "task": self.task,
            "created_at": self.created_at,
            "steps": [asdict(step) for step in self.steps],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "RegressionFixture":
        raw = json.loads(text)
        return cls(
            schema_version=int(raw.get("schema_version", SCHEMA_VERSION)),
            session=str(raw.get("session", "")),
            task=str(raw.get("task", "")),
            created_at=float(raw.get("created_at", 0.0) or 0.0),
            steps=[
                FrozenStep(
                    seq=int(step.get("seq", index + 1)),
                    tool=str(step.get("tool", "")),
                    input_hash=str(step.get("input_hash", "")),
                )
                for index, step in enumerate(raw.get("steps", []))
            ],
        )


@dataclass
class RegressionChange:
    kind: str
    expected_seq: int | None = None
    actual_seq: int | None = None
    tool: str = ""
    expected_hash: str = ""
    actual_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, "")}


@dataclass
class RegressionReport:
    fixture: str
    actual: str
    expected_steps: int
    actual_steps: int
    edit_distance: int
    divergence_score: float
    threshold: float
    exceeded: bool
    changes: list[RegressionChange]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture": self.fixture,
            "actual": self.actual,
            "expected_steps": self.expected_steps,
            "actual_steps": self.actual_steps,
            "edit_distance": self.edit_distance,
            "divergence_score": self.divergence_score,
            "threshold": self.threshold,
            "exceeded": self.exceeded,
            "changes": [change.to_dict() for change in self.changes],
        }


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _input_hash(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _tool_name(event: TraceEvent) -> str:
    return str(
        event.data.get("tool_name")
        or event.data.get("name")
        or event.data.get("tool")
        or "unknown"
    )


def _tool_input(event: TraceEvent) -> Any:
    if "arguments" in event.data:
        return event.data["arguments"]
    if "input" in event.data:
        return event.data["input"]
    if "args" in event.data:
        return event.data["args"]
    return {}


def extract_steps(events: list[TraceEvent]) -> list[FrozenStep]:
    steps: list[FrozenStep] = []
    for event in events:
        if event.event_type != EventType.TOOL_CALL:
            continue
        steps.append(
            FrozenStep(
                seq=len(steps) + 1,
                tool=_tool_name(event),
                input_hash=_input_hash(_tool_input(event)),
            )
        )
    return steps


def _session_task(events: list[TraceEvent], fallback: str = "") -> str:
    for event in events:
        if event.event_type != EventType.USER_PROMPT:
            continue
        task = (
            event.data.get("prompt")
            or event.data.get("content")
            or event.data.get("text")
            or ""
        )
        if task:
            return str(task)
    return fallback


def freeze_session(
    store: TraceStore,
    session_id: str,
    task: str = "",
) -> RegressionFixture:
    meta = store.load_meta(session_id)
    events = store.load_events(session_id)
    return RegressionFixture(
        session=session_id,
        task=task or _session_task(events, meta.command or meta.agent_name),
        created_at=time.time(),
        steps=extract_steps(events),
    )


def _edit_distance(a: list[tuple[str, str]], b: list[tuple[str, str]]) -> int:
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def _signature_positions(steps: list[FrozenStep]) -> dict[tuple[str, str], int]:
    positions: dict[tuple[str, str], int] = {}
    for step in steps:
        positions.setdefault(step.signature, step.seq)
    return positions


def _find_matching_tool(step: FrozenStep, candidates: list[FrozenStep]) -> FrozenStep | None:
    for candidate in candidates:
        if candidate.tool == step.tool:
            return candidate
    return None


def compare_fixtures(
    expected: RegressionFixture,
    actual: RegressionFixture,
    threshold: float = 0.0,
) -> RegressionReport:
    expected_signatures = [step.signature for step in expected.steps]
    actual_signatures = [step.signature for step in actual.steps]
    edit_distance = _edit_distance(expected_signatures, actual_signatures)
    denominator = max(len(expected.steps), len(actual.steps), 1)
    score = round(edit_distance / denominator, 3)

    expected_positions = _signature_positions(expected.steps)
    actual_positions = _signature_positions(actual.steps)
    changes: list[RegressionChange] = []

    for step in expected.steps:
        actual_seq = actual_positions.get(step.signature)
        if actual_seq is not None and actual_seq != step.seq:
            changes.append(
                RegressionChange(
                    kind="reordered",
                    expected_seq=step.seq,
                    actual_seq=actual_seq,
                    tool=step.tool,
                    expected_hash=step.input_hash,
                    actual_hash=step.input_hash,
                )
            )

    for step in expected.steps:
        if step.signature in actual_positions:
            continue
        match = _find_matching_tool(
            step,
            [candidate for candidate in actual.steps if candidate.signature not in expected_positions],
        )
        if match:
            changes.append(
                RegressionChange(
                    kind="changed_input",
                    expected_seq=step.seq,
                    actual_seq=match.seq,
                    tool=step.tool,
                    expected_hash=step.input_hash,
                    actual_hash=match.input_hash,
                )
            )
        else:
            changes.append(
                RegressionChange(
                    kind="removed",
                    expected_seq=step.seq,
                    tool=step.tool,
                    expected_hash=step.input_hash,
                )
            )

    for step in actual.steps:
        if step.signature in expected_positions:
            continue
        match = _find_matching_tool(
            step,
            [candidate for candidate in expected.steps if candidate.signature not in actual_positions],
        )
        if match:
            continue
        changes.append(
            RegressionChange(
                kind="added",
                actual_seq=step.seq,
                tool=step.tool,
                actual_hash=step.input_hash,
            )
        )

    return RegressionReport(
        fixture=expected.session,
        actual=actual.session,
        expected_steps=len(expected.steps),
        actual_steps=len(actual.steps),
        edit_distance=edit_distance,
        divergence_score=score,
        threshold=threshold,
        exceeded=score > threshold,
        changes=changes,
    )


def print_regression_report(report: RegressionReport, out: TextIO = sys.stdout) -> None:
    w = out.write
    status = "FAIL" if report.exceeded else "PASS"
    w("\nTool-call regression report\n")
    w(f"Status:    {status}\n")
    w(f"Fixture:   {report.fixture}\n")
    w(f"Actual:    {report.actual}\n")
    w(f"Steps:     {report.expected_steps} expected, {report.actual_steps} actual\n")
    w(f"Divergence: {report.divergence_score:.3f} (threshold {report.threshold:.3f})\n")
    if not report.changes:
        w("Changes:   none\n\n")
        return
    w("Changes:\n")
    for change in report.changes:
        if change.kind == "reordered":
            w(f"  reordered: {change.tool} expected #{change.expected_seq}, actual #{change.actual_seq}\n")
        elif change.kind == "changed_input":
            w(f"  changed input: {change.tool} expected #{change.expected_seq}, actual #{change.actual_seq}\n")
        elif change.kind == "removed":
            w(f"  removed: {change.tool} expected #{change.expected_seq}\n")
        elif change.kind == "added":
            w(f"  added: {change.tool} actual #{change.actual_seq}\n")
    w("\n")


def _resolve_session(store: TraceStore, raw: str | None) -> str | None:
    if raw:
        return store.find_session(raw)
    latest = store.get_latest_session()
    return latest.session_id if latest else None


def _default_fixture_path(session_id: str) -> Path:
    return Path(".agent-strace-fixtures") / f"{session_id}.json"


def cmd_freeze(args: argparse.Namespace) -> int:
    store = TraceStore(args.trace_dir)
    session_id = _resolve_session(store, getattr(args, "session_id", None))
    if not session_id:
        sys.stderr.write(f"Session not found: {getattr(args, 'session_id', '')}\n")
        return 1

    try:
        fixture = freeze_session(store, session_id, task=getattr(args, "task", "") or "")
    except Exception as exc:
        sys.stderr.write(f"Could not freeze session {session_id}: {exc}\n")
        return 1

    fmt = getattr(args, "format", "text")
    output = getattr(args, "output", None)
    if output is None and fmt == "text":
        output = str(_default_fixture_path(session_id))

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(fixture.to_json() + "\n")
        message = f"Frozen fixture written to {out_path}\n"
        if fmt == "json":
            sys.stderr.write(message)
        else:
            sys.stdout.write(message)

    if fmt == "json":
        sys.stdout.write(fixture.to_json() + "\n")
    elif not output:
        sys.stdout.write(fixture.to_json() + "\n")

    return 0


def cmd_regression(args: argparse.Namespace) -> int:
    fixture_path = Path(args.fixture_file)
    try:
        expected = RegressionFixture.from_json(fixture_path.read_text())
    except Exception as exc:
        sys.stderr.write(f"Could not read fixture {fixture_path}: {exc}\n")
        return 1

    store = TraceStore(args.trace_dir)
    session_id = _resolve_session(store, getattr(args, "session_id", None))
    if not session_id:
        sys.stderr.write("Session not found. Provide a session ID or record a session first.\n")
        return 1

    try:
        actual = freeze_session(store, session_id)
    except Exception as exc:
        sys.stderr.write(f"Could not load session {session_id}: {exc}\n")
        return 1

    threshold = float(getattr(args, "threshold", 0.0) or 0.0)
    report = compare_fixtures(expected, actual, threshold=threshold)
    if getattr(args, "format", "text") == "json":
        sys.stdout.write(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    else:
        print_regression_report(report)
    return 1 if report.exceeded else 0
