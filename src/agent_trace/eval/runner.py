"""Eval execution engine.

Runs scorers against sessions, compares sessions, and provides CI exit codes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, TextIO

from ..store import TraceStore
from .config import EvalConfig, EvalCriterionConfig, load_config, load_gate_config
from .scorers import MetricError, MetricValue, ScoreResult, measure_metric, run_scorer


@dataclass
class EvalReport:
    session_id: str
    results: list[ScoreResult]
    config: EvalConfig

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def overall_passed(self) -> bool:
        return self.failed == 0

    @property
    def weighted_score(self) -> float:
        if not self.results:
            return 0.0
        total_weight = sum(r.threshold for r in self.results)
        if total_weight == 0:
            return 0.0
        weighted = sum(r.score * r.threshold for r in self.results)
        return weighted / total_weight


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_eval(
    store: TraceStore,
    session_id: str,
    config: EvalConfig,
) -> EvalReport:
    events = store.load_events(session_id)
    results: list[ScoreResult] = []

    for scorer_cfg in config.scorers:
        params = dict(scorer_cfg.params)
        params["threshold"] = scorer_cfg.threshold
        result = run_scorer(
            name=scorer_cfg.type,
            config=params,
            events=events,
            store=store,
            session_id=session_id,
        )
        results.append(result)

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed

    return EvalReport(
        session_id=session_id,
        results=results,
        config=config,
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _col_width(results: list[ScoreResult]) -> int:
    return max((len(r.scorer) for r in results), default=10) + 2


def format_report_table(report: EvalReport, out=sys.stdout) -> None:
    w = out.write
    w(f"\nSession: {report.session_id}\n")
    w("─" * 70 + "\n")
    col = _col_width(report.results)
    w(f"  {'Scorer':<{col}} {'Score':>7}  {'Threshold':>10}  {'Status':<8}  Reason\n")
    w("─" * 70 + "\n")
    for r in report.results:
        status = "✓ pass" if r.passed else "✗ fail"
        w(f"  {r.scorer:<{col}} {r.score:>7.2f}  {r.threshold:>10.2f}  {status:<8}  {r.reason}\n")
    w("─" * 70 + "\n")
    w(f"Overall: {report.passed}/{len(report.results)} passed\n\n")


def format_report_json(report: EvalReport, out=sys.stdout) -> None:
    data = {
        "session_id": report.session_id,
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
    out.write(json.dumps(data, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Named eval gates
# ---------------------------------------------------------------------------

@dataclass
class CriterionResult:
    name: str
    scorer: str
    score: MetricValue | None
    fail_on: str
    threshold: Any = None
    expected: Any = None
    criterion_passed: bool = False
    baseline: MetricValue | None = None
    regressed: bool = False
    reason: str = ""

    @property
    def target(self) -> Any:
        return self.expected if self.fail_on == "not_equal" else self.threshold

    @property
    def passed(self) -> bool:
        return self.criterion_passed and not self.regressed


@dataclass
class GateReport:
    session_id: str
    results: list[CriterionResult]

    @property
    def failed(self) -> int:
        return sum(1 for result in self.results if not result.passed)

    @property
    def passed(self) -> int:
        return len(self.results) - self.failed

    @property
    def regressions(self) -> int:
        return sum(1 for result in self.results if result.regressed)

    @property
    def overall_passed(self) -> bool:
        return bool(self.results) and self.failed == 0


def _numeric(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise MetricError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise MetricError(f"{label} must be numeric") from exc
    if number != number or number in (float("inf"), float("-inf")):
        raise MetricError(f"{label} must be finite")
    return number


def _criterion_failed(score: MetricValue, criterion: EvalCriterionConfig) -> bool:
    mode = criterion.fail_on
    target = criterion.expected if mode == "not_equal" else criterion.threshold
    if mode in {"above", "at_or_above", "below", "at_or_below"}:
        actual_number = _numeric(score, f"score from {criterion.scorer!r}")
        target_number = _numeric(target, f"threshold for {criterion.name!r}")
        if mode == "above":
            return actual_number > target_number
        if mode == "at_or_above":
            return actual_number >= target_number
        if mode == "below":
            return actual_number < target_number
        return actual_number <= target_number
    if mode == "equal":
        return score == target
    if mode == "not_equal":
        return score != target
    raise MetricError(f"unsupported fail_on mode: {mode}")


def _is_regression(
    current: MetricValue,
    baseline: MetricValue,
    fail_on: str,
    tolerance: float,
) -> bool:
    lower_is_better = fail_on in {"above", "at_or_above"}
    higher_is_better = fail_on in {"below", "at_or_below"}
    if lower_is_better or higher_is_better:
        current_number = _numeric(current, "current score")
        baseline_number = _numeric(baseline, "baseline score")
        # Tolerance is a fractional change.  For a zero baseline, treating the
        # fraction itself as an absolute allowance keeps zero-cost/count gates
        # usable while still catching a one-error regression at tolerance .05.
        allowance = abs(baseline_number) * tolerance
        if baseline_number == 0:
            allowance = tolerance
        if lower_is_better:
            return current_number > baseline_number + allowance
        return current_number < baseline_number - allowance
    return current != baseline


def run_eval_gate(
    store: TraceStore,
    session_id: str,
    config: EvalConfig,
    baseline: dict[str, MetricValue] | None = None,
    tolerance: float = 0.0,
) -> GateReport:
    """Measure and gate one session against named raw-value criteria."""

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    events = store.load_events(session_id)
    results: list[CriterionResult] = []
    baseline = baseline or {}

    for criterion in config.evals:
        try:
            score = measure_metric(
                criterion.scorer,
                events,
                store,
                session_id,
                criterion.params,
            )
            failed = _criterion_failed(score, criterion)
            result = CriterionResult(
                name=criterion.name,
                scorer=criterion.scorer,
                score=score,
                fail_on=criterion.fail_on,
                threshold=criterion.threshold,
                expected=criterion.expected,
                criterion_passed=not failed,
            )
            if criterion.name in baseline:
                result.baseline = baseline[criterion.name]
                result.regressed = _is_regression(
                    score,
                    result.baseline,
                    criterion.fail_on,
                    tolerance,
                )
                if result.regressed:
                    result.reason = f"regressed from baseline {result.baseline!r}"
        except Exception as exc:
            result = CriterionResult(
                name=criterion.name,
                scorer=criterion.scorer,
                score=None,
                fail_on=criterion.fail_on,
                threshold=criterion.threshold,
                expected=criterion.expected,
                criterion_passed=False,
                reason=str(exc),
            )
        results.append(result)
    return GateReport(session_id=session_id, results=results)


def _display_value(value: object, scorer: str = "") -> str:
    if value is None:
        return "—"
    if scorer == "cost_usd" and isinstance(value, (int, float)):
        return f"${value:.4f}"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def format_gate_table(report: GateReport, out: TextIO = sys.stdout) -> None:
    """Render a human-readable named-eval result table."""

    name_width = max(20, *(len(result.name) for result in report.results))
    out.write(f"Eval results — session {report.session_id}\n")
    out.write("─" * 72 + "\n")
    out.write(f"{'Eval':<{name_width}}  {'Score':<14}  {'Threshold':<14}  Result\n")
    out.write("─" * 72 + "\n")
    for result in report.results:
        score = _display_value(result.score, result.scorer)
        target = _display_value(result.target, result.scorer)
        status = "pass" if result.passed else "fail"
        suffix = f" — {result.reason}" if result.reason else ""
        out.write(f"{result.name:<{name_width}}  {score:<14}  {target:<14}  {status}{suffix}\n")
    out.write("─" * 72 + "\n")
    overall = "PASS" if report.overall_passed else "FAIL"
    out.write(f"Overall: {overall} ({report.failed}/{len(report.results)} evals failed)\n")


def gate_report_data(report: GateReport) -> dict[str, Any]:
    return {
        "session_id": report.session_id,
        "passed": report.overall_passed,
        "pass_count": report.passed,
        "fail_count": report.failed,
        "regression_count": report.regressions,
        "results": [
            {
                "name": result.name,
                "scorer": result.scorer,
                "score": result.score,
                "threshold": result.threshold,
                "expected": result.expected,
                "fail_on": result.fail_on,
                "baseline": result.baseline,
                "regressed": result.regressed,
                "criterion_passed": result.criterion_passed,
                "passed": result.passed,
                "reason": result.reason,
            }
            for result in report.results
        ],
    }


def format_gate_json(report: GateReport, out: TextIO = sys.stdout) -> None:
    out.write(json.dumps(gate_report_data(report), indent=2) + "\n")


def save_gate_baseline(path: str | Path, report: GateReport) -> None:
    """Save raw named scores in a versioned, machine-readable baseline."""

    unavailable = [result.name for result in report.results if result.score is None]
    if unavailable:
        raise ValueError(f"cannot save unavailable scores: {', '.join(unavailable)}")
    baseline_path = Path(path)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": 1,
        "session_id": report.session_id,
        "scores": {result.name: result.score for result in report.results},
    }
    baseline_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_gate_baseline(path: str | Path) -> dict[str, MetricValue]:
    """Load a named baseline, accepting both versioned and flat mappings."""

    baseline_path = Path(path)
    try:
        data = json.loads(baseline_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"baseline not found: {baseline_path}") from None
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read baseline {baseline_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"baseline {baseline_path} must be a JSON object")
    raw_scores = data.get("scores", data)
    if not isinstance(raw_scores, dict):
        raise ValueError(f"baseline {baseline_path} has invalid scores")

    scores: dict[str, MetricValue] = {}
    for name, raw_value in raw_scores.items():
        value = raw_value.get("score") if isinstance(raw_value, dict) else raw_value
        if not isinstance(value, (int, float, str, bool)) or value is None:
            raise ValueError(f"baseline score for {name!r} has an unsupported value")
        scores[str(name)] = value
    return scores


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_gate_github_summary(
    report: GateReport,
    path: str | Path | None = None,
) -> Path | None:
    """Append a Markdown result table to the GitHub Actions step summary."""

    destination = Path(path) if path else None
    if destination is None:
        env_path = os.environ.get("GITHUB_STEP_SUMMARY")
        if not env_path:
            return None
        destination = Path(env_path)

    lines = [
        "## agent-strace eval",
        "",
        "| Eval | Scorer | Score | Threshold | Baseline | Result |",
        "|---|---|---:|---:|---:|---|",
    ]
    for result in report.results:
        status = "✅ pass" if result.passed else "❌ fail"
        lines.append(
            "| "
            + " | ".join(
                _markdown_cell(value)
                for value in (
                    result.name,
                    result.scorer,
                    _display_value(result.score, result.scorer),
                    _display_value(result.target, result.scorer),
                    _display_value(result.baseline, result.scorer),
                    status,
                )
            )
            + " |"
        )
    overall = "PASS" if report.overall_passed else "FAIL"
    lines.extend(("", f"**Overall: {overall}** — {report.failed}/{len(report.results)} evals failed.", ""))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as summary:
        summary.write("\n".join(lines))
    return destination


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------

def format_compare(
    report_a: EvalReport,
    report_b: EvalReport,
    out=sys.stdout,
) -> None:
    w = out.write
    w(f"\nCompare: {report_a.session_id} vs {report_b.session_id}\n")
    w("─" * 80 + "\n")

    scorers_a = {r.scorer: r for r in report_a.results}
    scorers_b = {r.scorer: r for r in report_b.results}
    all_scorers = sorted(set(scorers_a) | set(scorers_b))

    col = max((len(s) for s in all_scorers), default=10) + 2
    w(f"  {'Scorer':<{col}} {'Session A':>10}  {'Session B':>10}  {'Delta':>8}\n")
    w("─" * 80 + "\n")

    for scorer in all_scorers:
        ra = scorers_a.get(scorer)
        rb = scorers_b.get(scorer)
        score_a = f"{ra.score:.2f}" if ra else "n/a"
        score_b = f"{rb.score:.2f}" if rb else "n/a"
        if ra and rb:
            delta = rb.score - ra.score
            delta_str = f"{delta:+.2f}"
        else:
            delta_str = "n/a"
        w(f"  {scorer:<{col}} {score_a:>10}  {score_b:>10}  {delta_str:>8}\n")

    w("─" * 80 + "\n")
    ws_a = f"{report_a.weighted_score:.2f}"
    ws_b = f"{report_b.weighted_score:.2f}"
    delta_ws = report_b.weighted_score - report_a.weighted_score
    w(f"  {'Weighted score':<{col}} {ws_a:>10}  {ws_b:>10}  {delta_ws:>+8.2f}\n\n")


# ---------------------------------------------------------------------------
# CLI handlers
# ---------------------------------------------------------------------------

def _resolve_session(store: TraceStore, session_id: str | None) -> str | None:
    if not session_id:
        return store.get_latest_session_id()
    found = store.find_session(session_id)
    return found


def cmd_eval_run(args: argparse.Namespace) -> int:
    store = TraceStore(args.trace_dir)
    config = load_config(getattr(args, "config", ".agent-evals.yaml"))

    session_id = _resolve_session(store, getattr(args, "session_id", None))
    if not session_id:
        sys.stderr.write("No sessions found.\n")
        return 1

    report = run_eval(store, session_id, config)
    fmt = getattr(args, "format", "table")
    if fmt == "json":
        format_report_json(report)
    else:
        format_report_table(report)

    return 0 if report.overall_passed else 1


def cmd_eval_compare(args: argparse.Namespace) -> int:
    store = TraceStore(args.trace_dir)
    config = load_config(getattr(args, "config", ".agent-evals.yaml"))

    sid_a = store.find_session(args.session_a)
    sid_b = store.find_session(args.session_b)

    if not sid_a:
        sys.stderr.write(f"Session not found: {args.session_a}\n")
        return 1
    if not sid_b:
        sys.stderr.write(f"Session not found: {args.session_b}\n")
        return 1

    report_a = run_eval(store, sid_a, config)
    report_b = run_eval(store, sid_b, config)
    format_compare(report_a, report_b)
    return 0


def _load_baseline(path: str) -> dict[str, float]:
    """Load a saved baseline: {scorer_name: score}."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _save_baseline(path: str, report: "EvalReport") -> None:
    """Save current scores as a baseline file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {r.scorer: r.score for r in report.results}
    p.write_text(json.dumps(data, indent=2))


def _write_github_summary(report: "EvalReport", baseline: dict[str, float], tolerance: float) -> None:
    """Write a PR-comment-ready Markdown summary to .agent-traces/eval-summary.md."""
    lines = ["## agent-strace eval\n"]
    lines.append("| Judge | Pass rate | Baseline | Delta | Status |")
    lines.append("|---|---|---|---|---|")
    for r in report.results:
        base_score = baseline.get(r.scorer)
        if base_score is not None:
            delta = r.score - base_score
            delta_str = f"{delta:+.0%}"
            regressed = delta < -tolerance
            status = "❌" if regressed else "✅"
            base_str = f"{base_score:.0%}"
        else:
            delta_str = "—"
            status = "✅" if r.passed else "❌"
            base_str = "—"
        lines.append(f"| `{r.scorer}` | {r.score:.0%} | {base_str} | {delta_str} | {status} |")

    lines.append("")
    if report.overall_passed:
        lines.append("**Result: PASS**")
    else:
        lines.append(f"**Result: FAIL** — {report.failed} scorer(s) below threshold.")

    failing = [r for r in report.results if not r.passed]
    if failing:
        lines.append("")
        lines.append("<details>")
        lines.append("<summary>Failing scorers</summary>")
        lines.append("")
        for r in failing:
            lines.append(f"- `{r.scorer}` — score {r.score:.2f} (threshold {r.threshold:.2f}): {r.reason}")
        lines.append("")
        lines.append("</details>")

    content = "\n".join(lines) + "\n"

    # Write to $GITHUB_STEP_SUMMARY when running inside GitHub Actions
    gha_summary = os.environ.get("GITHUB_STEP_SUMMARY", "")
    if gha_summary:
        with open(gha_summary, "a", encoding="utf-8") as f:
            f.write(content)
        sys.stderr.write(f"GitHub Actions step summary written to {gha_summary}\n")
    else:
        summary_path = Path(".agent-traces/eval-summary.md")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(content)
        sys.stderr.write(f"GitHub summary written to {summary_path}\n")


def cmd_eval_ci(args: argparse.Namespace) -> int:
    """Run evals and exit 1 if any scorer fails (for CI integration).

    Supports baseline comparison (--baseline), saving baselines
    (--save-baseline), regression tolerance (--tolerance), and
    GitHub Actions PR comment output (--github-summary).
    """
    store = TraceStore(args.trace_dir)
    config = load_config(getattr(args, "config", ".agent-evals.yaml"))

    session_id = _resolve_session(store, getattr(args, "session_id", None))
    if not session_id:
        sys.stderr.write("No sessions found.\n")
        return 1

    report = run_eval(store, session_id, config)
    format_report_table(report, out=sys.stderr)

    # Save baseline if requested
    save_baseline_path = getattr(args, "save_baseline", None)
    if save_baseline_path:
        _save_baseline(save_baseline_path, report)
        sys.stderr.write(f"Baseline saved to {save_baseline_path}\n")
        return 0

    # Load baseline for comparison
    baseline_path = getattr(args, "baseline", None)
    baseline: dict[str, float] = {}
    if baseline_path:
        baseline = _load_baseline(baseline_path)

    tolerance = float(getattr(args, "tolerance", 0.0) or 0.0)

    # GitHub summary
    if getattr(args, "github_summary", False):
        _write_github_summary(report, baseline, tolerance)

    # Determine pass/fail with optional baseline regression check
    failed = False
    if not report.overall_passed:
        failed = True
    elif baseline:
        for r in report.results:
            base_score = baseline.get(r.scorer)
            if base_score is not None and (r.score - base_score) < -tolerance:
                sys.stderr.write(
                    f"CI: {r.scorer} regressed {r.score:.2f} vs baseline {base_score:.2f} "
                    f"(tolerance {tolerance:.2f})\n"
                )
                failed = True

    if failed:
        sys.stderr.write(f"CI: FAIL — {report.failed} scorer(s) failed\n")
        return 1

    sys.stderr.write("CI: PASS — all scorers passed\n")
    return 0


def cmd_eval_gate(args: argparse.Namespace) -> int:
    """CLI handler for the named eval-as-a-gate workflow."""

    try:
        config = load_gate_config(getattr(args, "config", ".agent-evals.yaml"))
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"Eval config error: {exc}\n")
        return 1

    store = TraceStore(args.trace_dir)
    session_id = _resolve_session(store, getattr(args, "session_id", None))
    if not session_id:
        requested = getattr(args, "session_id", None)
        if requested:
            sys.stderr.write(f"Session not found: {requested}\n")
        else:
            sys.stderr.write("No sessions found.\n")
        return 1

    baseline: dict[str, MetricValue] = {}
    baseline_path = getattr(args, "baseline", None)
    if baseline_path:
        try:
            baseline = load_gate_baseline(baseline_path)
        except ValueError as exc:
            sys.stderr.write(f"Baseline error: {exc}\n")
            return 1

    try:
        tolerance = float(getattr(args, "tolerance", 0.0) or 0.0)
        report = run_eval_gate(store, session_id, config, baseline, tolerance)
    except (OSError, TypeError, ValueError) as exc:
        sys.stderr.write(f"Eval error: {exc}\n")
        return 1

    if getattr(args, "format", "table") == "json":
        format_gate_json(report, out=sys.stdout)
    else:
        format_gate_table(report, out=sys.stdout)

    save_path = getattr(args, "save_baseline", None)
    if save_path:
        try:
            save_gate_baseline(save_path, report)
        except (OSError, ValueError) as exc:
            sys.stderr.write(f"Baseline error: {exc}\n")
            return 1
        sys.stderr.write(f"Baseline saved to {save_path}\n")

    try:
        write_gate_github_summary(report)
    except OSError as exc:
        sys.stderr.write(f"GitHub summary error: {exc}\n")
        return 1
    if save_path:
        return 0
    return 0 if report.overall_passed else 1
