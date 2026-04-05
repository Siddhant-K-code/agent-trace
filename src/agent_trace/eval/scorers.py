"""Built-in scorer implementations.

A scorer takes a list of TraceEvent objects and returns a score between
0.0 and 1.0, plus an optional reason string. Zero new dependencies for
built-in scorers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from ..cost import estimate_cost
from ..models import EventType, TraceEvent
from ..store import TraceStore


@dataclass
class ScoreResult:
    scorer: str
    score: float          # 0.0 – 1.0
    threshold: float      # minimum passing score
    passed: bool
    reason: str = ""

    @property
    def status(self) -> str:
        return "pass" if self.passed else "fail"


# ---------------------------------------------------------------------------
# Built-in scorers
# ---------------------------------------------------------------------------

def score_no_errors(events: list[TraceEvent], threshold: float = 1.0) -> ScoreResult:
    """1.0 if no ERROR events, 0.0 otherwise."""
    errors = [e for e in events if e.event_type == EventType.ERROR]
    score = 0.0 if errors else 1.0
    reason = f"{len(errors)} error(s) found" if errors else "no errors"
    return ScoreResult("no_errors", score, threshold, score >= threshold, reason)


def score_regex(
    events: list[TraceEvent],
    pattern: str,
    event_type: str = "assistant_response",
    threshold: float = 1.0,
) -> ScoreResult:
    """1.0 if any event of *event_type* matches *pattern*, 0.0 otherwise."""
    try:
        et = EventType(event_type)
    except ValueError:
        return ScoreResult("regex", 0.0, threshold, False, f"unknown event_type: {event_type}")

    compiled = re.compile(pattern)
    for event in events:
        if event.event_type != et:
            continue
        text = " ".join(str(v) for v in event.data.values())
        if compiled.search(text):
            return ScoreResult("regex", 1.0, threshold, True, f"pattern matched in {event_type}")

    return ScoreResult("regex", 0.0, threshold, False, f"pattern not found in any {event_type} event")


def score_cost_under(
    store: TraceStore,
    session_id: str,
    max_dollars: float,
    threshold: float = 1.0,
) -> ScoreResult:
    """1.0 if estimated cost ≤ max_dollars, else proportional score."""
    try:
        result = estimate_cost(store, session_id)
        actual = result.total_cost
    except Exception as exc:
        return ScoreResult("cost_under", 0.0, threshold, False, f"cost estimation failed: {exc}")

    if actual <= max_dollars:
        score = 1.0
        reason = f"${actual:.4f} ≤ ${max_dollars}"
    else:
        # Proportional: score = max_dollars / actual (capped at 1.0)
        score = min(1.0, max_dollars / actual) if actual > 0 else 1.0
        reason = f"${actual:.4f} actual > ${max_dollars} limit"

    return ScoreResult("cost_under", score, threshold, score >= threshold, reason)


def score_files_scoped(
    events: list[TraceEvent],
    allowed_paths: list[str],
    threshold: float = 1.0,
) -> ScoreResult:
    """1.0 if all file operations are within allowed_paths."""
    if not allowed_paths:
        return ScoreResult("files_scoped", 1.0, threshold, True, "no path restrictions")

    violations: list[str] = []
    for event in events:
        if event.event_type != EventType.TOOL_CALL:
            continue
        name = event.data.get("tool_name", "").lower()
        if name not in ("read", "write", "edit", "view", "create"):
            continue
        args = event.data.get("arguments", {}) or {}
        path = str(args.get("file_path") or args.get("path") or "")
        if not path:
            continue
        if not any(path.startswith(allowed) for allowed in allowed_paths):
            violations.append(path)

    if not violations:
        return ScoreResult("files_scoped", 1.0, threshold, True, "all files within allowed paths")

    score = max(0.0, 1.0 - len(violations) / max(1, len(violations) + 1))
    reason = f"{len(violations)} file(s) outside allowed paths: {', '.join(violations[:3])}"
    return ScoreResult("files_scoped", score, threshold, score >= threshold, reason)


def score_duration_under(
    events: list[TraceEvent],
    max_seconds: float,
    threshold: float = 1.0,
) -> ScoreResult:
    """1.0 if session duration ≤ max_seconds."""
    if len(events) < 2:
        return ScoreResult("duration_under", 1.0, threshold, True, "insufficient events to measure duration")

    duration = events[-1].timestamp - events[0].timestamp
    if duration <= max_seconds:
        return ScoreResult("duration_under", 1.0, threshold, True, f"{duration:.1f}s ≤ {max_seconds}s")

    score = min(1.0, max_seconds / duration) if duration > 0 else 1.0
    reason = f"{duration:.1f}s actual > {max_seconds}s limit"
    return ScoreResult("duration_under", score, threshold, score >= threshold, reason)


def score_custom(
    events: list[TraceEvent],
    fn: Callable[[list[TraceEvent]], float],
    name: str = "custom",
    threshold: float = 1.0,
) -> ScoreResult:
    """Run a user-supplied callable that returns a float in [0, 1]."""
    try:
        score = float(fn(events))
        score = max(0.0, min(1.0, score))
    except Exception as exc:
        return ScoreResult(name, 0.0, threshold, False, f"scorer raised: {exc}")
    return ScoreResult(name, score, threshold, score >= threshold, "custom scorer")


# ---------------------------------------------------------------------------
# Scorer registry (name → factory)
# ---------------------------------------------------------------------------

def run_scorer(
    name: str,
    config: dict,
    events: list[TraceEvent],
    store: TraceStore | None = None,
    session_id: str = "",
) -> ScoreResult:
    """Dispatch to the appropriate built-in scorer by name."""
    threshold = float(config.get("threshold", config.get("weight", 1.0)))

    if name == "no_errors":
        return score_no_errors(events, threshold=threshold)

    if name == "regex":
        return score_regex(
            events,
            pattern=config.get("pattern", ""),
            event_type=config.get("event_type", "assistant_response"),
            threshold=threshold,
        )

    if name == "cost_under":
        if store is None or not session_id:
            return ScoreResult(name, 0.0, threshold, False, "store/session_id required for cost_under")
        return score_cost_under(store, session_id, max_dollars=float(config.get("max_dollars", 10.0)), threshold=threshold)

    if name == "files_scoped":
        return score_files_scoped(
            events,
            allowed_paths=config.get("allowed_paths", []),
            threshold=threshold,
        )

    if name == "duration_under":
        return score_duration_under(
            events,
            max_seconds=float(config.get("max_seconds", 120.0)),
            threshold=threshold,
        )

    return ScoreResult(name, 0.0, threshold, False, f"unknown scorer: {name}")
