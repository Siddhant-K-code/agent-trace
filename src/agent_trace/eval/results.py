"""The eval results file that `eval run` and `eval ci` write.

Each run writes ``<session>/eval.json`` in the trace store, replacing the
file an earlier run wrote. `export --scores`, `dashboard --trend` and
`eval dataset auto --filter low-eval-score:N` read it, and
`export --format otlp-genai` turns each result into a
``gen_ai.evaluation.result`` event on the session's root span.

A run in which a scorer could not judge the session (a bad config, a failed
request, an exception) writes nothing, so the file only ever holds verdicts.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import __version__
from ..fileutil import write_atomic
from ..inert import inert

if TYPE_CHECKING:
    from ..store import TraceStore
    from .runner import EvalReport

EVAL_RESULTS_VERSION = 1


@dataclass(frozen=True)
class StoredEvaluation:
    """One result read back from eval.json."""

    scorer: str
    score: float
    passed: bool
    reason: str = ""


def eval_results_data(report: EvalReport, command: str, evaluated_at: float) -> dict[str, Any]:
    """Return the eval.json content for *report*."""
    return {
        "version": EVAL_RESULTS_VERSION,
        "session_id": report.session_id,
        "command": command,
        "evaluated_at": evaluated_at,
        "agent_strace_version": __version__,
        "passed": report.overall_passed,
        "pass_count": report.passed,
        "fail_count": report.failed,
        "weighted_score": report.weighted_score,
        "results": [
            {
                "scorer": r.scorer,
                "score": r.score,
                "threshold": r.threshold,
                "passed": r.passed,
                "reason": r.reason,
            }
            for r in report.results
        ],
    }


def _finite(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # an int too large for a float
        return False


def unsaved_reason(report: EvalReport) -> str | None:
    """Return why *report* must not be saved, or None when it can be."""
    for r in report.results:
        if getattr(r, "error", False):
            return f"scorer {inert(r.scorer)} could not judge the session: {inert(r.reason)}"
    for r in report.results:
        if not _finite(r.score) or not _finite(r.threshold):
            return f"scorer {inert(r.scorer)} has a score or threshold that is not a finite number"
    return None


def save_eval_results(store: TraceStore, report: EvalReport, command: str) -> Path | None:
    """Write *report* to the session's eval.json and return its path.

    Returns None, after saying why on stderr, when a scorer could not judge
    the session, a score or threshold is not finite, or the file cannot be
    written. An earlier eval.json is then left as it was.
    """
    problem = unsaved_reason(report)
    if problem:
        sys.stderr.write(f"Eval results not saved: {problem}\n")
        return None
    data = eval_results_data(report, command, time.time())
    try:
        path = store.eval_results_path(report.session_id)
        write_atomic(path, json.dumps(data, indent=2, allow_nan=False) + "\n")
    except OSError as exc:
        sys.stderr.write(f"Eval results not saved: {inert(exc.strerror or exc)}\n")
        return None
    except ValueError as exc:
        sys.stderr.write(f"Eval results not saved: {inert(exc)}\n")
        return None
    sys.stderr.write(f"Eval results written to {inert(path)}\n")
    return path


def load_stored_evaluations(store: TraceStore, session_id: str) -> list[StoredEvaluation]:
    """Return the results in the session's eval.json, in file order.

    Returns [] when the file is missing or is not a JSON object with a
    ``results`` list. A result is left out unless it has a non-empty string
    ``scorer``, a finite number ``score`` and a true/false ``passed``: the
    verdict comes from the file and is never worked out here. A ``reason``
    that is not a string reads as empty.
    """
    try:
        path = store.eval_results_path(session_id)
        if not path.is_file():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError, RecursionError):
        return []
    if not isinstance(data, dict):
        return []
    results = data.get("results")
    if not isinstance(results, list):
        return []
    stored: list[StoredEvaluation] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        scorer = result.get("scorer")
        score = result.get("score")
        passed = result.get("passed")
        if not isinstance(scorer, str) or not scorer:
            continue
        if not _finite(score) or not isinstance(passed, bool):
            continue
        reason = result.get("reason")
        stored.append(StoredEvaluation(
            scorer=scorer,
            score=float(score),
            passed=passed,
            reason=reason if isinstance(reason, str) else "",
        ))
    return stored
