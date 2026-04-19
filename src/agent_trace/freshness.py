"""Context freshness check: flag stale context before a session starts.

Compares the current codebase state against what the agent last saw,
using git diff between the last session timestamp and HEAD.

Usage:
    agent-strace freshness
    agent-strace freshness --since 2026-04-01
    agent-strace freshness --scope "src/**"
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from .store import TraceStore


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StaleFile:
    path: str
    lines_changed: int
    change_type: str    # "modified" | "added" | "deleted" | "renamed"
    in_scope: bool      # True if in CLAUDE.md / AGENTS.md scope


@dataclass
class FreshnessReport:
    last_session_ts: float | None
    last_session_id: str
    files_changed_total: int
    files_in_scope: int
    stale_files: list[StaleFile]
    freshness_score: int        # 0–100 (100 = fully fresh)
    reading_minutes: float
    scope_source: str           # "CLAUDE.md" | "AGENTS.md" | "--scope flag" | "all files"
    scope_glob: str


# ---------------------------------------------------------------------------
# Scope detection
# ---------------------------------------------------------------------------

def _parse_scope_from_agents_md(path: str = "CLAUDE.md") -> list[str]:
    """Extract file globs from CLAUDE.md / AGENTS.md scope sections."""
    globs: list[str] = []
    for fname in ("CLAUDE.md", "AGENTS.md", path):
        p = Path(fname)
        if not p.exists():
            continue
        text = p.read_text(errors="replace")
        # Look for lines that look like file globs after scope/files headers
        in_scope = False
        for line in text.splitlines():
            stripped = line.strip()
            if re.search(r"(scope|files|include|watch)", stripped, re.I) and stripped.endswith(":"):
                in_scope = True
                continue
            if in_scope:
                if stripped.startswith("-") or stripped.startswith("*"):
                    glob = stripped.lstrip("- ").strip("`")
                    if glob:
                        globs.append(glob)
                elif stripped and not stripped.startswith("#"):
                    in_scope = False
        if globs:
            return globs
    return []


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _git_diff_since(repo: str, since_ts: float) -> list[tuple[str, int, str]]:
    """Return list of (path, lines_changed, change_type) since a timestamp."""
    since = datetime.fromtimestamp(since_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        # Get the commit hash at that time
        result = subprocess.run(
            ["git", "-C", repo, "rev-list", "-1", f"--before={since}", "HEAD"],
            capture_output=True, text=True, timeout=15,
        )
        base_commit = result.stdout.strip()
        if not base_commit:
            return []

        # Diff from that commit to HEAD
        diff_result = subprocess.run(
            ["git", "-C", repo, "diff", "--numstat", base_commit, "HEAD"],
            capture_output=True, text=True, timeout=30,
        )
        files = []
        for line in diff_result.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            added_str, removed_str, path = parts[0], parts[1], parts[2]
            try:
                lines = int(added_str) + int(removed_str)
            except ValueError:
                lines = 0
            # Detect rename: "old => new"
            if "=>" in path:
                change_type = "renamed"
                path = path.split("=>")[-1].strip().strip("}")
            elif added_str == "0" and removed_str != "0":
                change_type = "deleted"
            elif removed_str == "0" and added_str != "0":
                change_type = "added"
            else:
                change_type = "modified"
            files.append((path.strip(), lines, change_type))
        return files
    except Exception:
        return []


def _git_diff_since_date(repo: str, since_date: str) -> list[tuple[str, int, str]]:
    """Return changed files since a date string (e.g. '2026-04-01')."""
    try:
        result = subprocess.run(
            ["git", "-C", repo, "rev-list", "-1", f"--before={since_date}", "HEAD"],
            capture_output=True, text=True, timeout=15,
        )
        base_commit = result.stdout.strip()
        if not base_commit:
            return []
        diff_result = subprocess.run(
            ["git", "-C", repo, "diff", "--numstat", base_commit, "HEAD"],
            capture_output=True, text=True, timeout=30,
        )
        files = []
        for line in diff_result.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            try:
                lines = int(parts[0]) + int(parts[1])
            except ValueError:
                lines = 0
            path = parts[2].strip()
            change_type = "modified"
            files.append((path, lines, change_type))
        return files
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse_freshness(
    store: TraceStore,
    since_date: str = "",
    scope_glob: str = "",
    repo: str = ".",
) -> FreshnessReport:
    """Compute context freshness relative to the last session."""
    # Find last session timestamp
    all_metas = store.list_sessions()
    last_meta = all_metas[-1] if all_metas else None
    last_ts = last_meta.started_at if last_meta else None
    last_sid = last_meta.session_id if last_meta else ""

    # Determine scope
    scope_source = "all files"
    scope_globs: list[str] = []
    if scope_glob:
        scope_globs = [scope_glob]
        scope_source = "--scope flag"
    else:
        scope_globs = _parse_scope_from_agents_md()
        if scope_globs:
            scope_source = "CLAUDE.md / AGENTS.md"

    # Get changed files
    if since_date:
        raw_files = _git_diff_since_date(repo, since_date)
    elif last_ts:
        raw_files = _git_diff_since(repo, last_ts)
    else:
        raw_files = []

    # Build stale file list
    stale: list[StaleFile] = []
    for path, lines, change_type in raw_files:
        in_scope = True
        if scope_globs:
            in_scope = any(fnmatch.fnmatch(path, g) for g in scope_globs)
        stale.append(StaleFile(
            path=path,
            lines_changed=lines,
            change_type=change_type,
            in_scope=in_scope,
        ))

    # Sort: in-scope first, then by lines changed descending
    stale.sort(key=lambda f: (not f.in_scope, -f.lines_changed))

    files_in_scope = sum(1 for f in stale if f.in_scope)
    total = len(stale)

    # Freshness score: 100 if nothing changed, decreases with scope changes
    if total == 0:
        score = 100
    else:
        scope_weight = files_in_scope / max(total, 1)
        large_changes = sum(1 for f in stale if f.in_scope and f.lines_changed > 100)
        score = max(0, int(100 - scope_weight * 60 - large_changes * 10))

    reading_minutes = sum(
        max(1.0, f.lines_changed / 200.0) for f in stale if f.in_scope
    )

    return FreshnessReport(
        last_session_ts=last_ts,
        last_session_id=last_sid,
        files_changed_total=total,
        files_in_scope=files_in_scope,
        stale_files=stale,
        freshness_score=score,
        reading_minutes=reading_minutes,
        scope_source=scope_source,
        scope_glob=scope_glob or (", ".join(scope_globs[:3]) if scope_globs else "**"),
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_freshness(report: FreshnessReport, out: TextIO = sys.stdout) -> None:
    w = out.write
    sep = "─" * 55

    w(f"\nContext Freshness Report\n{sep}\n")

    if report.last_session_ts:
        age_h = int((time.time() - report.last_session_ts) / 3600)
        age_str = f"{age_h}h ago" if age_h < 48 else f"{age_h // 24}d ago"
        w(f"Last session: {report.last_session_id[:12]} ({age_str})\n")
    else:
        w("Last session: none found\n")

    w(f"Scope: {report.scope_glob}  [{report.scope_source}]\n")
    w(f"Files changed: {report.files_changed_total} total, "
      f"{report.files_in_scope} in scope\n")

    score = report.freshness_score
    bar_filled = score // 5
    bar = "█" * bar_filled + "░" * (20 - bar_filled)
    icon = "✅" if score >= 80 else ("⚠️ " if score >= 50 else "❌")
    w(f"Freshness score: {icon} {score}/100  [{bar}]\n")
    w(f"{sep}\n\n")

    if not report.stale_files:
        w("✅ Context is fully fresh — no changes since last session.\n\n")
        return

    in_scope = [f for f in report.stale_files if f.in_scope]
    out_of_scope = [f for f in report.stale_files if not f.in_scope]

    if in_scope:
        w("Stale files in scope:\n\n")
        for f in in_scope[:15]:
            icon = "❌" if f.lines_changed > 200 else "⚠️ "
            w(f"  {icon} {f.path}\n")
            w(f"       {f.change_type} · {f.lines_changed} lines changed\n")
        if len(in_scope) > 15:
            w(f"  ... and {len(in_scope) - 15} more\n")
        w("\n")

    if out_of_scope:
        w(f"Out-of-scope changes: {len(out_of_scope)} files\n\n")

    h = int(report.reading_minutes // 60)
    m = int(report.reading_minutes % 60)
    time_str = f"{h}h {m}min" if h else f"{int(m)}min"
    w(f"Estimated catch-up time: {time_str}\n")
    w(f"{sep}\n\n")


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def cmd_freshness(args: argparse.Namespace) -> int:
    store = TraceStore(args.trace_dir)
    since = getattr(args, "since", "") or ""
    scope = getattr(args, "scope", "") or ""
    repo = getattr(args, "repo", ".") or "."

    report = analyse_freshness(store, since_date=since, scope_glob=scope, repo=repo)
    format_freshness(report)
    return 0
