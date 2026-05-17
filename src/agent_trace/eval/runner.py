"""Eval execution engine.

Runs scorers against sessions, compares sessions, and provides CI exit codes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass, field

from ..store import TraceStore
from .config import EvalConfig, load_config
from .scorers import ScoreResult, run_scorer


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

    summary_path = Path(".agent-traces/eval-summary.md")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n")
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
