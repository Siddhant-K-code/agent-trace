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
from .cost import estimate_cost
from .models import EventType, SessionMeta, TraceEvent
from .store import TraceStore


@dataclass
class SessionAttribution:
    session_id: str
    started_at: float
    branch: str
    author: str
    cost: float
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
    def total_cost(self) -> float:
        return sum(row["cost"] for row in self.rows.values())

    @property
    def total_sessions(self) -> float:
        return sum(row["sessions"] for row in self.rows.values())

    @property
    def total_tool_calls(self) -> float:
        return sum(row["tool_calls"] for row in self.rows.values())

    @property
    def avg_cost(self) -> float:
        return self.total_cost / max(1.0, self.total_sessions)

    @property
    def outliers(self) -> list[SessionAttribution]:
        limit = self.avg_cost * self.outlier_threshold
        if limit <= 0:
            return []
        return sorted(
            [s for s in self.sessions if s.cost > limit],
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
    match = re.search(r"branch:\s*([^,)]+)", getattr(meta, "command", "") or "")
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


def _session_cost(store: TraceStore, session_id: str) -> float:
    try:
        return estimate_cost(store, session_id).total_cost
    except Exception:
        return 0.0


def _session_tool_calls(events: list[TraceEvent], meta: SessionMeta) -> int:
    count = sum(1 for ev in events if ev.event_type == EventType.TOOL_CALL)
    return count or getattr(meta, "tool_calls", 0) or 0


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
        except Exception:
            events = []
        files = _modified_files(events)
        shares = _author_shares(meta, files)
        cost = _session_cost(store, meta.session_id)
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
                    "authors": set(),
                    "branches": set(),
                }
            rows[key]["sessions"] += share
            rows[key]["tool_calls"] += tool_calls * share
            rows[key]["cost"] += cost * share
            rows[key]["authors"].add(author)
            rows[key]["branches"].add(session.branch)

    for row in rows.values():
        row["avg_session"] = row["cost"] / max(1e-9, row["sessions"])
        row["authors"] = sorted(row["authors"])
        row["branches"] = sorted(row["branches"])

    rows = dict(sorted(rows.items(), key=lambda item: item[1]["cost"], reverse=True))
    return TeamReport(
        group_by=group_by,
        window_start=window_start,
        window_end=window_end,
        rows=rows,
        sessions=sessions,
        outlier_threshold=outlier_threshold,
    )


def _fmt_date(ts: float) -> str:
    import datetime
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def _fmt_count(value: float) -> str:
    if abs(value - round(value)) < 0.01:
        return str(int(round(value)))
    return f"{value:.1f}"


def format_team_report_text(report: TeamReport, out: TextIO = sys.stdout) -> None:
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
            f"{row['group']:<28}  {_fmt_count(row['sessions']):>8}  "
            f"{_fmt_count(row['tool_calls']):>10}  ${row['cost']:>7.2f}  "
            f"${row['avg_session']:.2f}\n"
        )
    w("-" * 72 + "\n")
    w(
        f"{'Total':<28}  {_fmt_count(report.total_sessions):>8}  "
        f"{_fmt_count(report.total_tool_calls):>10}  ${report.total_cost:>7.2f}  "
        f"${report.avg_cost:.2f}\n"
    )

    if report.group_by != "branch":
        top_branches: dict[str, dict] = {}
        for session in report.sessions:
            key = session.branch
            if key not in top_branches:
                top_branches[key] = {"cost": 0.0, "sessions": 0}
            top_branches[key]["cost"] += session.cost
            top_branches[key]["sessions"] += 1
        if top_branches:
            w("\nTop cost drivers by branch:\n")
            for branch, stats in sorted(top_branches.items(), key=lambda item: item[1]["cost"], reverse=True)[:5]:
                pct = stats["cost"] / (report.total_cost or 1e-9) * 100
                w(f"  {branch:<28}  ${stats['cost']:.2f}  ({pct:.0f}%)  {stats['sessions']} sessions\n")

    if report.outliers:
        w(f"\nEfficiency outliers (cost > {report.outlier_threshold:g}x average):\n")
        for session in report.outliers[:5]:
            w(
                f"  {session.session_id[:12]}  ${session.cost:.2f}  "
                f"{session.author}  {session.branch}  {_fmt_date(session.started_at)}\n"
            )
        w("  Run `agent-strace lint <id>` to check for waste patterns.\n")


def format_team_report_csv(report: TeamReport) -> str:
    out = StringIO()
    writer = csv.writer(out)
    writer.writerow(["group_by", "group", "sessions", "tool_calls", "cost", "avg_session", "authors", "branches"])
    for row in report.rows.values():
        writer.writerow([
            report.group_by,
            row["group"],
            f"{row['sessions']:.3f}",
            f"{row['tool_calls']:.3f}",
            f"{row['cost']:.6f}",
            f"{row['avg_session']:.6f}",
            ";".join(row["authors"]),
            ";".join(row["branches"]),
        ])
    return out.getvalue()


def format_team_report_json(report: TeamReport) -> str:
    return json.dumps({
        "group_by": report.group_by,
        "window_start": report.window_start,
        "window_end": report.window_end,
        "total_cost": report.total_cost,
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
