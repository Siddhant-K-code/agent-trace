"""Weekly spend digest: agent-strace budget-report.

Aggregates cost across sessions for a configurable time window and produces
a human-readable or markdown-formatted spend summary. Reads watchdog
post-mortem files to calculate budget-ceiling savings.
"""

from __future__ import annotations

import argparse
import json
import sys
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from .cost import estimate_cost, DEFAULT_MODEL
from .models import EventType, SessionMeta
from .store import TraceStore


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SessionSpend:
    session_id: str
    agent_name: str
    started_at: float
    cost: float
    tool_breakdown: dict[str, float]   # tool_name -> estimated cost share
    watchdog_terminated: bool = False
    watchdog_budget: float | None = None   # budget ceiling at time of kill
    team: str = ""                         # team name (from SessionMeta.team)


@dataclass
class BudgetReport:
    window_start: float
    window_end: float
    sessions: list[SessionSpend]
    # Prior-window data for week-over-week comparison (may be empty)
    prior_sessions: list[SessionSpend] = field(default_factory=list)

    @property
    def total_cost(self) -> float:
        return sum(s.cost for s in self.sessions)

    @property
    def prior_total_cost(self) -> float:
        return sum(s.cost for s in self.prior_sessions)

    @property
    def session_count(self) -> int:
        return len(self.sessions)

    @property
    def prior_session_count(self) -> int:
        return len(self.prior_sessions)

    @property
    def avg_cost(self) -> float:
        return self.total_cost / max(1, self.session_count)

    @property
    def watchdog_terminated_sessions(self) -> list[SessionSpend]:
        return [s for s in self.sessions if s.watchdog_terminated]

    @property
    def watchdog_savings(self) -> float:
        """Estimated savings from watchdog budget ceilings."""
        total = 0.0
        for s in self.watchdog_terminated_sessions:
            if s.watchdog_budget is not None and s.cost < s.watchdog_budget:
                total += s.watchdog_budget - s.cost
        return total

    @property
    def tool_totals(self) -> dict[str, float]:
        """Aggregate cost by tool across all sessions."""
        totals: dict[str, float] = {}
        for s in self.sessions:
            for tool, cost in s.tool_breakdown.items():
                totals[tool] = totals.get(tool, 0.0) + cost
        return dict(sorted(totals.items(), key=lambda x: x[1], reverse=True))

    @property
    def top_sessions(self) -> list[SessionSpend]:
        return sorted(self.sessions, key=lambda s: s.cost, reverse=True)[:5]


# ---------------------------------------------------------------------------
# Post-mortem reading
# ---------------------------------------------------------------------------

def _read_postmortem(store: TraceStore, session_id: str) -> dict | None:
    """Read watchdog post-mortem JSON for a session, or None if absent."""
    pm_path = store._session_dir(session_id) / "watchdog-postmortem.json"
    if not pm_path.exists():
        return None
    try:
        return json.loads(pm_path.read_text())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Tool cost breakdown
# ---------------------------------------------------------------------------

_TOOL_COST_FRACTION = 0.30  # rough: ~30% of session cost attributable to tool calls


def _tool_breakdown(store: TraceStore, session_id: str, total_cost: float) -> dict[str, float]:
    """Estimate cost share per tool based on event count."""
    try:
        events = store.load_events(session_id)
    except Exception:
        return {}

    tool_counts: dict[str, int] = {}
    for ev in events:
        if ev.event_type == EventType.TOOL_CALL:
            tool = ev.data.get("tool_name", "unknown")
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

    total_calls = sum(tool_counts.values())
    if not total_calls:
        return {}

    tool_cost_pool = total_cost * _TOOL_COST_FRACTION
    return {
        tool: (count / total_calls) * tool_cost_pool
        for tool, count in tool_counts.items()
    }


# ---------------------------------------------------------------------------
# Core report builder
# ---------------------------------------------------------------------------

def build_report(
    store: TraceStore,
    window_start: float,
    window_end: float,
    include_prior: bool = True,
) -> BudgetReport:
    """Build a BudgetReport for the given time window."""
    all_sessions = store.list_sessions()

    window_sessions: list[SessionSpend] = []
    prior_sessions: list[SessionSpend] = []

    window_size = window_end - window_start
    prior_start = window_start - window_size
    prior_end = window_start

    for meta in all_sessions:
        in_window = window_start <= meta.started_at < window_end
        in_prior = include_prior and (prior_start <= meta.started_at < prior_end)

        if not in_window and not in_prior:
            continue

        # Estimate cost
        try:
            cost_result = estimate_cost(store, meta.session_id)
            cost = cost_result.total_cost
        except Exception:
            cost = 0.0

        # Check for watchdog termination
        pm = _read_postmortem(store, meta.session_id)
        watchdog_terminated = pm is not None
        watchdog_budget: float | None = None
        if pm:
            # Extract budget from post-mortem or session meta
            watchdog_budget = pm.get("budget_at_death") or pm.get("max_cost_dollars")

        breakdown = _tool_breakdown(store, meta.session_id, cost)

        spend = SessionSpend(
            session_id=meta.session_id,
            agent_name=meta.agent_name or meta.command or "",
            started_at=meta.started_at,
            cost=cost,
            tool_breakdown=breakdown,
            watchdog_terminated=watchdog_terminated,
            watchdog_budget=watchdog_budget,
            team=getattr(meta, "team", "") or "",
        )

        if in_window:
            window_sessions.append(spend)
        elif in_prior:
            prior_sessions.append(spend)

    return BudgetReport(
        window_start=window_start,
        window_end=window_end,
        sessions=window_sessions,
        prior_sessions=prior_sessions,
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_date(ts: float) -> str:
    import datetime
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def _fmt_delta(current: float, prior: float, higher_is_worse: bool = True) -> str:
    """Format a week-over-week delta string."""
    if prior == 0:
        return ""
    pct = (current - prior) / prior * 100
    if abs(pct) < 1:
        return "(≈ same)"
    arrow = "↑" if pct > 0 else "↓"
    direction = "worse" if (pct > 0) == higher_is_worse else "better"
    return f"({arrow} {abs(pct):.0f}% vs prior period)"


def team_summary(sessions: list[SessionSpend]) -> dict[str, dict]:
    """Aggregate spend by team. Returns {team_name: {cost, sessions, agents}}."""
    teams: dict[str, dict] = {}
    for s in sessions:
        key = s.team or "(unassigned)"
        if key not in teams:
            teams[key] = {"cost": 0.0, "sessions": 0, "agents": set()}
        teams[key]["cost"] += s.cost
        teams[key]["sessions"] += 1
        teams[key]["agents"].add(s.agent_name)
    # Convert sets to counts for serialisability
    for v in teams.values():
        v["agents"] = len(v["agents"])
    return dict(sorted(teams.items(), key=lambda x: x[1]["cost"], reverse=True))


def format_report_text(report: BudgetReport, out: TextIO = sys.stdout) -> None:
    w = out.write
    start = _fmt_date(report.window_start)
    end = _fmt_date(report.window_end)
    w(f"Budget Report — {start} to {end}\n\n")

    if not report.sessions:
        w("No sessions in this period.\n")
        return

    has_prior = bool(report.prior_sessions)

    # Summary
    cost_delta = _fmt_delta(report.total_cost, report.prior_total_cost) if has_prior else ""
    count_delta = _fmt_delta(report.session_count, report.prior_session_count) if has_prior else ""

    w(f"Total spend:        ${report.total_cost:.2f}  {cost_delta}\n")
    w(f"Sessions:           {report.session_count}      {count_delta}\n")
    w(f"Avg cost/session:   ${report.avg_cost:.2f}\n")
    w("\n")

    # Top sessions
    top = report.top_sessions
    w(f"Top {len(top)} most expensive sessions:\n")
    for i, s in enumerate(top, 1):
        tag = "  ⚠ watchdog" if s.watchdog_terminated else ""
        name = (s.agent_name or s.session_id[:12])[:30]
        w(f"  {i}. {s.session_id[:12]}  ${s.cost:.2f}  {name:<30}  {_fmt_date(s.started_at)}{tag}\n")
    w("\n")

    # Cost by tool
    tool_totals = report.tool_totals
    if tool_totals:
        w("Cost by tool (estimated):\n")
        total_tool_cost = sum(tool_totals.values()) or 1e-9
        for tool, cost in list(tool_totals.items())[:8]:
            pct = cost / report.total_cost * 100
            w(f"  {tool:<20}  ${cost:.2f}  ({pct:.0f}%)\n")
        w("\n")

    # Team breakdown (only shown when at least one session has a team set)
    teams = team_summary(report.sessions)
    if any(k != "(unassigned)" for k in teams):
        w("Cost by team:\n")
        for team_name, stats in teams.items():
            pct = stats["cost"] / report.total_cost * 100 if report.total_cost else 0
            w(f"  {team_name:<24}  ${stats['cost']:.2f}  ({pct:.0f}%)  "
              f"{stats['sessions']} sessions  {stats['agents']} agents\n")
        w("\n")

    # Watchdog savings
    terminated = report.watchdog_terminated_sessions
    if terminated:
        w(f"Sessions terminated by watchdog:  {len(terminated)}")
        savings = report.watchdog_savings
        if savings > 0:
            w(f"  (${savings:.2f} saved by budget ceiling)")
        w("\n")


def format_report_markdown(report: BudgetReport, out: TextIO = sys.stdout) -> None:
    w = out.write
    start = _fmt_date(report.window_start)
    end = _fmt_date(report.window_end)
    w(f"## Budget Report — {start} to {end}\n\n")

    if not report.sessions:
        w("_No sessions in this period._\n")
        return

    has_prior = bool(report.prior_sessions)
    cost_delta = _fmt_delta(report.total_cost, report.prior_total_cost) if has_prior else ""
    count_delta = _fmt_delta(report.session_count, report.prior_session_count) if has_prior else ""

    w(f"**Total spend:** ${report.total_cost:.2f} {cost_delta}  \n")
    w(f"**Sessions:** {report.session_count} {count_delta}  \n")
    w(f"**Avg cost/session:** ${report.avg_cost:.2f}\n\n")

    # Top sessions table
    w(f"### Top {len(report.top_sessions)} sessions\n\n")
    w("| # | Session | Cost | Name | Date |\n")
    w("|---|---------|------|------|------|\n")
    for i, s in enumerate(report.top_sessions, 1):
        tag = " ⚠" if s.watchdog_terminated else ""
        name = (s.agent_name or s.session_id[:12])[:30]
        w(f"| {i} | `{s.session_id[:12]}` | ${s.cost:.2f}{tag} | {name} | {_fmt_date(s.started_at)} |\n")
    w("\n")

    # Cost by tool
    tool_totals = report.tool_totals
    if tool_totals:
        w("### Cost by tool\n\n")
        w("| Tool | Cost | % |\n")
        w("|------|------|---|\n")
        for tool, cost in list(tool_totals.items())[:8]:
            pct = cost / report.total_cost * 100
            w(f"| {tool} | ${cost:.2f} | {pct:.0f}% |\n")
        w("\n")

    # Watchdog
    terminated = report.watchdog_terminated_sessions
    if terminated:
        savings = report.watchdog_savings
        savings_str = f" (${savings:.2f} saved by budget ceiling)" if savings > 0 else ""
        w(f"**Sessions terminated by watchdog:** {len(terminated)}{savings_str}\n")


def format_report_json(report: BudgetReport) -> str:
    return json.dumps({
        "window_start": report.window_start,
        "window_end": report.window_end,
        "total_cost": report.total_cost,
        "session_count": report.session_count,
        "avg_cost": report.avg_cost,
        "prior_total_cost": report.prior_total_cost,
        "prior_session_count": report.prior_session_count,
        "watchdog_terminated": len(report.watchdog_terminated_sessions),
        "watchdog_savings": report.watchdog_savings,
        "top_sessions": [
            {
                "session_id": s.session_id,
                "agent_name": s.agent_name,
                "started_at": s.started_at,
                "cost": s.cost,
                "watchdog_terminated": s.watchdog_terminated,
            }
            for s in report.top_sessions
        ],
        "tool_totals": report.tool_totals,
    }, indent=2)


# ---------------------------------------------------------------------------
# Date parsing helpers
# ---------------------------------------------------------------------------

def _parse_date(s: str) -> float:
    """Parse ISO date string or relative duration (7d, 24h) to Unix timestamp."""
    s = s.strip()
    multipliers = {"d": 86400, "h": 3600, "m": 60, "s": 1}
    if s and s[-1] in multipliers:
        try:
            return _time.time() - float(s[:-1]) * multipliers[s[-1]]
        except ValueError:
            pass
    import datetime
    try:
        dt = datetime.datetime.fromisoformat(s)
        return dt.timestamp()
    except ValueError:
        raise ValueError(f"Cannot parse date: {s!r}. Use ISO format (2026-05-01) or duration (7d).")


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def cmd_budget_report(args: argparse.Namespace) -> int:
    store = TraceStore(args.trace_dir)
    fmt = getattr(args, "format", "text")
    team_filter = getattr(args, "team", "") or ""

    # Time window
    since_str = getattr(args, "since", None)
    until_str = getattr(args, "until", None)

    now = _time.time()
    window_end = _parse_date(until_str) if until_str else now
    window_start = _parse_date(since_str) if since_str else (window_end - 7 * 86400)

    report = build_report(store, window_start, window_end)

    # Apply team filter
    if team_filter:
        report.sessions = [s for s in report.sessions if s.team == team_filter]
        report.prior_sessions = [s for s in report.prior_sessions if s.team == team_filter]

    if fmt == "json":
        sys.stdout.write(format_report_json(report) + "\n")
    elif fmt == "markdown":
        format_report_markdown(report, sys.stdout)
    else:
        format_report_text(report, sys.stdout)

    return 0
