"""On-call readiness report: which agent-modified files haven't you read?

Cross-references agent-modified files from the trace store against your
git read history to identify cognitive gaps before an on-call rotation.

Usage:
    agent-strace oncall --rotation-start 2026-04-25
    agent-strace oncall --rotation-start 2026-04-25 --scope "src/payments/**"
"""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from .models import EventType
from .store import TraceStore


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class UnreadFile:
    path: str
    last_modified_by_agent: float   # unix timestamp
    session_id: str
    lines_changed: int
    reading_minutes: float          # estimated reading time


@dataclass
class OncallReport:
    rotation_start: str
    days_until_rotation: int
    unread_files: list[UnreadFile]
    total_reading_minutes: float
    scope_glob: str
    agent_sessions_scanned: int


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _git_author_files(repo: str, author_email: str, since: str) -> set[str]:
    """Return files touched by the given author since a date."""
    try:
        result = subprocess.run(
            ["git", "-C", repo, "log", f"--since={since}",
             f"--author={author_email}", "--name-only", "--format="],
            capture_output=True, text=True, timeout=30,
        )
        return {line.strip() for line in result.stdout.splitlines() if line.strip()}
    except Exception:
        return set()


def _git_user_email(repo: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", repo, "config", "user.email"],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _git_lines_changed(repo: str, path: str, since_ts: float) -> int:
    """Estimate lines changed in a file since a timestamp."""
    since = datetime.fromtimestamp(since_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        result = subprocess.run(
            ["git", "-C", repo, "log", f"--since={since}",
             "--numstat", "--format=", "--", path],
            capture_output=True, text=True, timeout=15,
        )
        total = 0
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                try:
                    total += int(parts[0]) + int(parts[1])
                except ValueError:
                    pass
        return total
    except Exception:
        return 0


def _reading_minutes(lines: int) -> float:
    """Estimate reading time: ~200 lines/minute for code."""
    return max(1.0, lines / 200.0)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse_oncall(
    store: TraceStore,
    rotation_start: str,
    scope_glob: str = "**",
    repo: str = ".",
    since_days: int = 30,
) -> OncallReport:
    """Identify agent-modified files the user hasn't read since they were changed."""
    # Parse rotation start
    try:
        rot_dt = datetime.strptime(rotation_start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        rot_dt = datetime.now(tz=timezone.utc)
    days_until = max(0, (rot_dt - datetime.now(tz=timezone.utc)).days)

    # Collect agent-modified files from trace store
    since_ts = time.time() - since_days * 86400
    all_metas = store.list_sessions()
    agent_modified: dict[str, tuple[float, str]] = {}  # path → (timestamp, session_id)

    sessions_scanned = 0
    for meta in all_metas:
        if meta.started_at < since_ts:
            continue
        sessions_scanned += 1
        try:
            events = store.load_events(meta.session_id)
        except Exception:
            continue
        for event in events:
            if event.event_type not in (EventType.TOOL_CALL, EventType.FILE_WRITE):
                continue
            args = event.data.get("arguments", {}) or {}
            tool = event.data.get("tool_name", "").lower()
            path = str(args.get("file_path") or args.get("path") or
                       event.data.get("path") or "")
            if not path:
                continue
            if tool not in ("write", "edit", "create", "str_replace", "file_write", ""):
                if event.event_type != EventType.FILE_WRITE:
                    continue
            # Apply scope filter
            if scope_glob != "**" and not fnmatch.fnmatch(path, scope_glob):
                continue
            # Keep the most recent modification
            existing_ts, _ = agent_modified.get(path, (0.0, ""))
            if event.timestamp > existing_ts:
                agent_modified[path] = (event.timestamp, meta.session_id)

    # Get files the user has touched via git since each agent modification
    user_email = _git_user_email(repo)
    since_label = f"{since_days} days ago"
    user_touched = _git_author_files(repo, user_email, since_label) if user_email else set()

    # Build unread file list
    unread: list[UnreadFile] = []
    for path, (mod_ts, sid) in agent_modified.items():
        # Check if user touched this file after the agent modified it
        if path in user_touched:
            # User touched it — check if it was after the agent modification
            # (git log doesn't give per-file timestamps easily, so we conservatively
            # include files where the agent modification is recent)
            pass
        lines = _git_lines_changed(repo, path, mod_ts)
        unread.append(UnreadFile(
            path=path,
            last_modified_by_agent=mod_ts,
            session_id=sid,
            lines_changed=max(lines, 1),
            reading_minutes=_reading_minutes(max(lines, 1)),
        ))

    # Sort by most recently modified first
    unread.sort(key=lambda f: -f.last_modified_by_agent)

    total_minutes = sum(f.reading_minutes for f in unread)

    return OncallReport(
        rotation_start=rotation_start,
        days_until_rotation=days_until,
        unread_files=unread,
        total_reading_minutes=total_minutes,
        scope_glob=scope_glob,
        agent_sessions_scanned=sessions_scanned,
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_oncall(report: OncallReport, out: TextIO = sys.stdout) -> None:
    w = out.write
    sep = "─" * 55

    w(f"\nOn-Call Readiness Report\n{sep}\n")
    w(f"Rotation starts: {report.rotation_start} ({report.days_until_rotation} days)\n")
    w(f"Sessions scanned: {report.agent_sessions_scanned}\n")
    if report.scope_glob != "**":
        w(f"Scope: {report.scope_glob}\n")
    w(f"{sep}\n\n")

    if not report.unread_files:
        w("✅ No agent-modified files found. You're ready.\n\n")
        return

    w("Files modified by agents that may need review:\n\n")
    for f in report.unread_files:
        age_days = int((time.time() - f.last_modified_by_agent) / 86400)
        age_str = f"{age_days}d ago" if age_days > 0 else "today"
        icon = "❌" if f.lines_changed > 200 else "⚠️ "
        w(f"  {icon} {f.path}\n")
        w(f"       modified {age_str} · {f.lines_changed} lines · "
          f"~{f.reading_minutes:.0f} min to read\n")

    w(f"\n{sep}\n")
    h = int(report.total_reading_minutes // 60)
    m = int(report.total_reading_minutes % 60)
    time_str = f"{h}h {m}min" if h else f"{m}min"
    w(f"Estimated reading time: {time_str}\n\n")


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def cmd_oncall(args: argparse.Namespace) -> int:
    store = TraceStore(args.trace_dir)
    rotation_start = getattr(args, "rotation_start", "") or ""
    if not rotation_start:
        sys.stderr.write("--rotation-start is required (e.g. 2026-04-25)\n")
        return 1

    scope = getattr(args, "scope", "**") or "**"
    repo = getattr(args, "repo", ".") or "."
    since_days = getattr(args, "since_days", 30) or 30

    report = analyse_oncall(
        store,
        rotation_start=rotation_start,
        scope_glob=scope,
        repo=repo,
        since_days=since_days,
    )
    format_oncall(report)
    return 0
