"""Team cost attribution report.

Aggregates session spend by git author, branch, or pull request. The report is
best-effort: when git metadata is missing or git is unavailable, it falls back
to session attribution and the local user.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time as _time
from dataclasses import dataclass, field
from io import StringIO
from typing import TextIO

from .budget_report import _parse_date
from .cost import session_cost
from .event_data import require_object_event_data
from .inert import escape_surrogates, inert
from .models import EventType, SessionMeta, TraceEvent, usable_count, usable_text
from .store import TraceStore


@dataclass
class SessionAttribution:
    session_id: str
    started_at: float
    branch: str
    author: str
    cost: float | None  # None when `agent-strace cost` refuses the session
    tool_calls: int
    files: list[str] = field(default_factory=list)
    shares: dict[str, float] = field(default_factory=dict)


@dataclass
class TeamReport:
    group_by: str
    window_start: float
    window_end: float
    rows: dict[str, dict]
    sessions: list[SessionAttribution]
    outlier_threshold: float = 2.0

    @property
    def total_cost(self) -> float | None:
        """Cost of the priced sessions, or None when sessions exist but none is priced."""
        if self.sessions and self.unpriced_sessions == len(self.sessions):
            return None
        return sum(row["cost"] or 0.0 for row in self.rows.values())

    @property
    def unpriced_sessions(self) -> int:
        return sum(1 for s in self.sessions if s.cost is None)

    @property
    def total_sessions(self) -> float:
        return sum(row["sessions"] for row in self.rows.values())

    @property
    def total_tool_calls(self) -> float:
        return sum(row["tool_calls"] for row in self.rows.values())

    @property
    def avg_cost(self) -> float | None:
        """Average cost of the priced sessions, or None when none is priced."""
        total = self.total_cost
        if total is None:
            return None
        return total / max(1.0, len(self.sessions) - self.unpriced_sessions)

    @property
    def outliers(self) -> list[SessionAttribution]:
        limit = (self.avg_cost or 0.0) * self.outlier_threshold
        if limit <= 0:
            return []
        return sorted(
            [s for s in self.sessions if s.cost is not None and s.cost > limit],
            key=lambda s: s.cost,
            reverse=True,
        )


def _git(cwd: str, *args: str) -> str:
    if not shutil.which("git"):
        return ""
    try:
        result = subprocess.run(
            ["git", "-C", cwd, *args],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _meta_attr(meta: SessionMeta) -> dict:
    attr = getattr(meta, "attribution", {}) or {}
    return attr if isinstance(attr, dict) else {}


def _working_dir(meta: SessionMeta) -> str:
    attr = _meta_attr(meta)
    wd = attr.get("working_dir") or os.getcwd()
    return str(wd)


def session_engineer_attribution(meta: SessionMeta) -> tuple[str, str]:
    """Return ``(identifier, confidence)`` from persisted attribution only."""
    attr = _meta_attr(meta)
    explicit = (
        attr.get("actor_id")
        or attr.get("engineer_id")
        or attr.get("user_id")
        or attr.get("git_author")
        or attr.get("git_author_email")
    )
    if explicit:
        return str(explicit), "explicit"
    user = str(attr.get("os_user") or "")
    host = str(attr.get("hostname") or "")
    if user and host:
        return f"{user}@{host}", "host-user"
    if user:
        return user, "user-only"
    return "(unknown)", "unavailable"


def session_engineer(meta: SessionMeta) -> str:
    """Return the stable engineer identifier persisted with a session.

    Organization-wide reports must not inspect the reporting machine's git
    configuration, especially when the session came from a remote collector.
    This helper therefore uses stored attribution only.  The legacy team
    report keeps its local-git fallback below for backward compatibility.
    """
    return session_engineer_attribution(meta)[0]


def _fallback_author(meta: SessionMeta) -> str:
    attr = _meta_attr(meta)
    cwd = _working_dir(meta)
    return (
        _git(cwd, "config", "user.email")
        or attr.get("os_user", "")
        or os.environ.get("USER", "")
        or os.environ.get("USERNAME", "")
        or "(unknown)"
    )


def _branch(meta: SessionMeta) -> str:
    attr = _meta_attr(meta)
    branch = attr.get("git_branch") or ""
    if branch:
        return str(branch)
    # Imported Claude sessions store the branch in the command string.
    match = re.search(r"branch:\s*([^,)]+)", usable_text(getattr(meta, "command", "")))
    if match:
        return match.group(1).strip()
    return "(unknown)"


def _pr_from_branch(branch: str) -> str:
    if not branch or branch == "(unknown)":
        return "(unknown)"
    match = re.search(r"(?:^|[/_-])(?:pr|pull|gh)[/_-]?(\d+)(?:$|[/_-])", branch, re.I)
    if not match:
        match = re.search(r"#(\d+)", branch)
    if match:
        return f"#{match.group(1)}"
    return branch


# _modified_files reads the data of every event of these types.
MODIFIED_FILES_DATA_EVENT_TYPES = frozenset({EventType.FILE_WRITE, EventType.TOOL_CALL})


def _modified_files(events: list[TraceEvent]) -> list[str]:
    files: list[str] = []
    seen: set[str] = set()
    write_tools = {"write", "edit", "multiedit", "write_file", "edit_file", "replace"}
    for ev in events:
        path = ""
        if ev.event_type == EventType.FILE_WRITE:
            path = str(ev.data.get("path") or ev.data.get("file_path") or "")
        elif ev.event_type == EventType.TOOL_CALL:
            tool = str(ev.data.get("tool_name", "")).lower()
            args = ev.data.get("arguments", {})
            if isinstance(args, dict) and (tool in write_tools or "write" in tool or "edit" in tool):
                path = str(args.get("file_path") or args.get("path") or "")
        if path and path not in seen:
            seen.add(path)
            files.append(path)
    return files


def _author_for_file(path: str, cwd: str) -> str:
    return _git(cwd, "log", "-1", "--format=%ae", "--", path)


def _author_shares(meta: SessionMeta, files: list[str]) -> dict[str, float]:
    fallback = _fallback_author(meta)
    if not files:
        return {fallback: 1.0}

    cwd = _working_dir(meta)
    counts: dict[str, int] = {}
    for path in files:
        author = _author_for_file(path, cwd) or fallback
        counts[author] = counts.get(author, 0) + 1
    total = sum(counts.values()) or 1
    return {author: count / total for author, count in counts.items()}


def _session_cost(events: list[TraceEvent], session_id: str) -> float | None:
    """Return the `agent-strace cost` total, or None when it refuses the session."""
    try:
        return session_cost(events, session_id=session_id)
    except ValueError:
        return None


def _session_tool_calls(events: list[TraceEvent], meta: SessionMeta) -> int:
    count = sum(1 for ev in events if ev.event_type == EventType.TOOL_CALL)
    return count or usable_count(getattr(meta, "tool_calls", 0))


def _primary_author(shares: dict[str, float]) -> str:
    if not shares:
        return "(unknown)"
    return sorted(shares.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _row_key(group_by: str, session: SessionAttribution, author: str) -> str:
    if group_by == "author":
        return author
    if group_by == "branch":
        return session.branch
    if group_by == "pr":
        return _pr_from_branch(session.branch)
    return author


def build_team_report(
    store: TraceStore,
    window_start: float,
    window_end: float,
    group_by: str = "author",
    outlier_threshold: float = 2.0,
) -> TeamReport:
    rows: dict[str, dict] = {}
    sessions: list[SessionAttribution] = []

    for meta in store.list_sessions():
        if not (window_start <= meta.started_at < window_end):
            continue
        try:
            events = store.load_events(meta.session_id)
            loaded = True
        except Exception:
            events = []
            loaded = False
        require_object_event_data(
            events,
            event_types=MODIFIED_FILES_DATA_EVENT_TYPES,
            session_id=meta.session_id,
        )
        files = _modified_files(events)
        shares = _author_shares(meta, files)
        cost = _session_cost(events, meta.session_id) if loaded else None
        tool_calls = _session_tool_calls(events, meta)
        session = SessionAttribution(
            session_id=meta.session_id,
            started_at=meta.started_at,
            branch=_branch(meta),
            author=_primary_author(shares),
            cost=cost,
            tool_calls=tool_calls,
            files=files,
            shares=shares,
        )
        sessions.append(session)

        for author, share in shares.items():
            key = _row_key(group_by, session, author)
            if key not in rows:
                rows[key] = {
                    "group": key,
                    "sessions": 0.0,
                    "tool_calls": 0.0,
                    "cost": 0.0,
                    "unpriced_sessions": 0.0,
                    "authors": set(),
                    "branches": set(),
                }
            rows[key]["sessions"] += share
            rows[key]["tool_calls"] += tool_calls * share
            if cost is None:
                rows[key]["unpriced_sessions"] += share
            else:
                rows[key]["cost"] += cost * share
            rows[key]["authors"].add(author)
            rows[key]["branches"].add(session.branch)

    for row in rows.values():
        priced = row["sessions"] - row["unpriced_sessions"]
        if priced > 1e-9:
            row["avg_session"] = row["cost"] / priced
        else:
            # No priced session in this row: its cost is unknown, not $0.
            row["cost"] = None
            row["avg_session"] = None
        row["authors"] = sorted(row["authors"])
        row["branches"] = sorted(row["branches"])

    rows = dict(sorted(rows.items(), key=lambda item: _cost_key(item[1]["cost"]), reverse=True))
    return TeamReport(
        group_by=group_by,
        window_start=window_start,
        window_end=window_end,
        rows=rows,
        sessions=sessions,
        outlier_threshold=outlier_threshold,
    )


def _cost_key(cost: float | None) -> float:
    return -1.0 if cost is None else cost


def _fmt_cost(cost: float | None, spec: str = ".2f") -> str:
    return "n/a" if cost is None else f"${cost:{spec}}"


def _fmt_date(ts: float) -> str:
    import datetime
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def _fmt_count(value: float) -> str:
    if abs(value - round(value)) < 0.01:
        return str(int(round(value)))
    return f"{value:.1f}"


def format_team_report_text(report: TeamReport, out: TextIO = sys.stdout) -> None:
    """Write the cost report grouped by author, branch or pull request.

    Groups, branches and authors come from the session meta and git, so each
    goes through inert() and an escape code or a line break in it shows as
    text.
    """
    w = out.write
    start = _fmt_date(report.window_start)
    end = _fmt_date(report.window_end)
    label = report.group_by.title()
    w(f"Team Agent Cost Report - {start} to {end}\n\n")
    if not report.rows:
        w("No sessions in this period.\n")
        return

    w(f"{label:<28}  Sessions  Tool calls  Cost      Avg/session\n")
    w("-" * 72 + "\n")
    for row in report.rows.values():
        w(
            f"{inert(row['group']):<28}  {_fmt_count(row['sessions']):>8}  "
            f"{_fmt_count(row['tool_calls']):>10}  {_fmt_cost(row['cost'], '>7.2f'):>8}  "
            f"{_fmt_cost(row['avg_session'])}\n"
        )
    w("-" * 72 + "\n")
    unpriced = f" ({report.unpriced_sessions} not priced)" if report.unpriced_sessions else ""
    w(
        f"{'Total':<28}  {_fmt_count(report.total_sessions):>8}  "
        f"{_fmt_count(report.total_tool_calls):>10}  {_fmt_cost(report.total_cost, '>7.2f'):>8}  "
        f"{_fmt_cost(report.avg_cost)}{unpriced}\n"
    )

    if report.group_by != "branch":
        top_branches: dict[str, dict] = {}
        for session in report.sessions:
            key = session.branch
            if key not in top_branches:
                top_branches[key] = {"cost": None, "sessions": 0}
            if session.cost is not None:
                top_branches[key]["cost"] = (top_branches[key]["cost"] or 0.0) + session.cost
            top_branches[key]["sessions"] += 1
        if top_branches:
            w("\nTop cost drivers by branch:\n")
            for branch, stats in sorted(top_branches.items(), key=lambda item: _cost_key(item[1]["cost"]), reverse=True)[:5]:
                if stats["cost"] is None:
                    w(f"  {inert(branch):<28}  n/a  {stats['sessions']} sessions\n")
                    continue
                pct = stats["cost"] / (report.total_cost or 1e-9) * 100
                w(f"  {inert(branch):<28}  ${stats['cost']:.2f}  ({pct:.0f}%)  {stats['sessions']} sessions\n")

    if report.outliers:
        w(f"\nEfficiency outliers (cost > {report.outlier_threshold:g}x average):\n")
        for session in report.outliers[:5]:
            w(
                f"  {session.session_id[:12]}  ${session.cost:.2f}  "
                f"{inert(session.author)}  {inert(session.branch)}  {_fmt_date(session.started_at)}\n"
            )
        w("  Run `agent-strace lint <id>` to check for waste patterns.\n")


def format_team_report_csv(report: TeamReport) -> str:
    """Return the report as CSV. Each lone surrogate is written as its escape."""
    out = StringIO()
    writer = csv.writer(out)
    writer.writerow([
        "group_by", "group", "sessions", "tool_calls", "cost", "avg_session",
        "authors", "branches", "unpriced_sessions",
    ])
    for row in report.rows.values():
        writer.writerow([escape_surrogates(cell) for cell in (
            report.group_by,
            row["group"],
            f"{row['sessions']:.3f}",
            f"{row['tool_calls']:.3f}",
            "" if row["cost"] is None else f"{row['cost']:.6f}",
            "" if row["avg_session"] is None else f"{row['avg_session']:.6f}",
            ";".join(row["authors"]),
            ";".join(row["branches"]),
            f"{row['unpriced_sessions']:.3f}",
        )])
    return out.getvalue()


def format_team_report_json(report: TeamReport) -> str:
    return json.dumps({
        "group_by": report.group_by,
        "window_start": report.window_start,
        "window_end": report.window_end,
        "total_cost": report.total_cost,
        "unpriced_sessions": report.unpriced_sessions,
        "total_sessions": report.total_sessions,
        "total_tool_calls": report.total_tool_calls,
        "rows": list(report.rows.values()),
        "outliers": [
            {
                "session_id": s.session_id,
                "started_at": s.started_at,
                "author": s.author,
                "branch": s.branch,
                "cost": s.cost,
                "tool_calls": s.tool_calls,
            }
            for s in report.outliers
        ],
    }, indent=2)


def cmd_team_report(args: argparse.Namespace) -> int:
    store = TraceStore(args.trace_dir)
    now = _time.time()
    until = _parse_date(args.until) if getattr(args, "until", None) else now
    since = _parse_date(args.since) if getattr(args, "since", None) else until - 7 * 86400
    group_by = getattr(args, "by", "author") or "author"
    outlier_threshold = float(getattr(args, "outlier_threshold", 2.0) or 2.0)

    report = build_team_report(
        store,
        since,
        until,
        group_by=group_by,
        outlier_threshold=outlier_threshold,
    )

    fmt = getattr(args, "export", "text") or "text"
    if fmt == "csv":
        sys.stdout.write(format_team_report_csv(report))
    elif fmt == "json":
        sys.stdout.write(format_team_report_json(report) + "\n")
    else:
        format_team_report_text(report, sys.stdout)
    return 0
