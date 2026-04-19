"""Personal cost curve: which task types are cost-efficient for you?

Analyses stored session history and classifies sessions by task type using
session titles and initial prompt text from meta.json. Compares your average
cost per task type against community sweet-spot benchmarks.

Usage:
    agent-strace curve
    agent-strace curve --min-sessions 10
    agent-strace curve --export csv
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from dataclasses import dataclass, field
from typing import TextIO

from .store import TraceStore


# ---------------------------------------------------------------------------
# Task type classification
# ---------------------------------------------------------------------------

# Keyword patterns for classifying sessions by task type.
# Each entry: (task_type_label, [keywords_in_title_or_prompt])
TASK_CLASSIFIERS: list[tuple[str, list[str]]] = [
    ("Unit test writing",   ["test", "unittest", "pytest", "spec", "coverage"]),
    ("Code refactoring",    ["refactor", "cleanup", "clean up", "reorganize", "restructure"]),
    ("Bug debugging",       ["debug", "fix", "bug", "error", "traceback", "exception", "crash"]),
    ("Architecture",        ["architect", "design", "system design", "schema", "database design", "api design"]),
    ("Boilerplate gen",     ["scaffold", "boilerplate", "template", "generate", "init", "setup", "create project"]),
    ("Documentation",       ["doc", "readme", "comment", "docstring", "changelog"]),
    ("Code review",         ["review", "pr", "pull request", "feedback", "lint"]),
    ("Feature impl",        ["implement", "feature", "add", "build", "create", "develop"]),
    ("Performance",         ["performance", "optimize", "speed", "latency", "profil"]),
    ("Security",            ["security", "auth", "vulnerability", "cve", "injection", "xss"]),
]

DEFAULT_TASK_TYPE = "General / other"

# Community sweet-spot benchmarks (dollars per task).
# These are conservative estimates; users can override via --sweet-spot.
SWEET_SPOTS: dict[str, float] = {
    "Unit test writing":  0.12,
    "Code refactoring":   0.45,
    "Bug debugging":      0.80,
    "Architecture":       1.20,
    "Boilerplate gen":    0.08,
    "Documentation":      0.15,
    "Code review":        0.20,
    "Feature impl":       0.60,
    "Performance":        0.90,
    "Security":           0.70,
    DEFAULT_TASK_TYPE:    0.40,
}


def _classify_session(agent_name: str, command: str) -> str:
    """Return the task type label for a session based on its name/command."""
    text = (agent_name + " " + command).lower()
    for task_type, keywords in TASK_CLASSIFIERS:
        if any(kw in text for kw in keywords):
            return task_type
    return DEFAULT_TASK_TYPE


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TaskTypeStat:
    task_type: str
    session_count: int
    avg_cost: float
    sweet_spot: float
    ratio: float          # avg_cost / sweet_spot
    monthly_estimate: float   # avg_cost * sessions_per_month

    @property
    def verdict(self) -> str:
        if self.ratio <= 1.1:
            return "efficient"
        if self.ratio <= 2.0:
            return "over sweet spot"
        return "do this yourself"

    @property
    def verdict_icon(self) -> str:
        if self.ratio <= 1.1:
            return "✅"
        if self.ratio <= 2.0:
            return "⚠️ "
        return "❌"


@dataclass
class CurveReport:
    session_count: int
    days_analysed: int
    stats: list[TaskTypeStat]
    potential_monthly_savings: float
    savings_breakdown: list[tuple[str, float]]   # (task_type, monthly_saving)
    insufficient_data: bool = False
    min_sessions: int = 20


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse_curve(
    store: TraceStore,
    min_sessions: int = 20,
    sessions_per_day: float = 8.0,
) -> CurveReport:
    """Build a CurveReport from all stored sessions."""
    from .cost import estimate_cost

    all_metas = store.list_sessions()
    if not all_metas:
        return CurveReport(
            session_count=0,
            days_analysed=0,
            stats=[],
            potential_monthly_savings=0.0,
            savings_breakdown=[],
            insufficient_data=True,
            min_sessions=min_sessions,
        )

    # Collect per-session data
    type_costs: dict[str, list[float]] = {}
    earliest_ts: float | None = None
    latest_ts: float | None = None

    for meta in all_metas:
        sid = meta.session_id
        task_type = _classify_session(meta.agent_name, meta.command)

        try:
            cost = estimate_cost(store, sid).total_cost
        except Exception:
            cost = 0.0

        type_costs.setdefault(task_type, []).append(cost)

        ts = meta.started_at
        if earliest_ts is None or ts < earliest_ts:
            earliest_ts = ts
        if latest_ts is None or ts > latest_ts:
            latest_ts = ts

    total_sessions = sum(len(v) for v in type_costs.values())
    if total_sessions == 0:
        return CurveReport(
            session_count=0,
            days_analysed=0,
            stats=[],
            potential_monthly_savings=0.0,
            savings_breakdown=[],
            insufficient_data=True,
            min_sessions=min_sessions,
        )
    days = max(1, int((latest_ts or 0) - (earliest_ts or 0)) // 86400) if earliest_ts else 1

    # Sessions per month per task type (based on observed frequency)
    sessions_per_month_total = sessions_per_day * 30
    sessions_per_month: dict[str, float] = {}
    for tt, costs in type_costs.items():
        fraction = len(costs) / max(total_sessions, 1)
        sessions_per_month[tt] = fraction * sessions_per_month_total

    stats: list[TaskTypeStat] = []
    savings_breakdown: list[tuple[str, float]] = []
    total_savings = 0.0

    for task_type, costs in sorted(type_costs.items(), key=lambda x: -len(x[1])):
        avg = sum(costs) / len(costs)
        sweet = SWEET_SPOTS.get(task_type, SWEET_SPOTS[DEFAULT_TASK_TYPE])
        ratio = avg / sweet if sweet > 0 else 1.0
        spm = sessions_per_month.get(task_type, 0.0)
        monthly = avg * spm

        stat = TaskTypeStat(
            task_type=task_type,
            session_count=len(costs),
            avg_cost=avg,
            sweet_spot=sweet,
            ratio=ratio,
            monthly_estimate=monthly,
        )
        stats.append(stat)

        # Savings if you stopped delegating over-sweet-spot tasks
        if ratio > 1.5:
            saving = (avg - sweet) * spm
            savings_breakdown.append((task_type, saving))
            total_savings += saving

    savings_breakdown.sort(key=lambda x: -x[1])

    return CurveReport(
        session_count=total_sessions,
        days_analysed=days,
        stats=stats,
        potential_monthly_savings=total_savings,
        savings_breakdown=savings_breakdown,
        insufficient_data=total_sessions < min_sessions,
        min_sessions=min_sessions,
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_curve(report: CurveReport, out: TextIO = sys.stdout) -> None:
    w = out.write
    sep = "─" * 75

    w(f"\nYour Agent Cost Curve ({report.session_count} sessions, {report.days_analysed} days)\n")
    w(f"{sep}\n")

    if report.insufficient_data:
        w(
            f"⚠️  Only {report.session_count} session(s) found "
            f"(minimum: {report.min_sessions} for meaningful analysis).\n"
            f"   Record more sessions and re-run.\n\n"
        )
        if not report.stats:
            return

    w(f"  {'Task type':<22}  {'Sweet spot':>10}  {'Your avg':>10}  {'Ratio':>6}  Verdict\n")
    w(f"{sep}\n")

    for stat in report.stats:
        w(
            f"  {stat.task_type:<22}  "
            f"${stat.sweet_spot:>8.2f}  "
            f"${stat.avg_cost:>8.2f}  "
            f"{stat.ratio:>5.1f}x  "
            f"{stat.verdict_icon} {stat.verdict}\n"
        )

    w(f"{sep}\n")

    if report.savings_breakdown:
        w(f"\nPotential monthly savings if you stop delegating:\n")
        for task_type, saving in report.savings_breakdown[:5]:
            w(f"  - {task_type:<22}  ${saving:.2f}/month\n")
        w(f"  {'Total:':<24}  ${report.potential_monthly_savings:.2f}/month\n")

    w("\n")


def export_curve_csv(report: CurveReport, out: TextIO = sys.stdout) -> None:
    writer = csv.writer(out)
    writer.writerow([
        "task_type", "session_count", "avg_cost_usd",
        "sweet_spot_usd", "ratio", "verdict", "monthly_estimate_usd",
    ])
    for stat in report.stats:
        writer.writerow([
            stat.task_type,
            stat.session_count,
            f"{stat.avg_cost:.4f}",
            f"{stat.sweet_spot:.4f}",
            f"{stat.ratio:.2f}",
            stat.verdict,
            f"{stat.monthly_estimate:.2f}",
        ])


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def cmd_curve(args: argparse.Namespace) -> int:
    store = TraceStore(args.trace_dir)
    min_sessions = getattr(args, "min_sessions", 20) or 20
    export = getattr(args, "export", "") or ""

    report = analyse_curve(store, min_sessions=min_sessions)

    if export == "csv":
        export_curve_csv(report)
    else:
        format_curve(report)

    return 0
