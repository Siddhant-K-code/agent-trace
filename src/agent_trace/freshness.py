"""Context freshness check: flag stale context before a session starts.

Compares HEAD with the commit the agent last saw. The base is the newest
first-parent commit of HEAD made at or before the start of the newest session,
or at or before 00:00 UTC on the --since date. When no commit is that old, the
base is the empty tree, so every file in HEAD counts as added.

Usage:
    agent-strace freshness
    agent-strace freshness --since 2026-04-01
    agent-strace freshness --scope "src/**"
    agent-strace freshness --repo ../other-repo
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, TextIO

from .inert import inert
from .models import usable_time
from .store import TraceStore


class FreshnessError(Exception):
    """freshness cannot compare the repository with the last session."""


class NoSessionsError(FreshnessError):
    """The trace store has no session and no --since date was given."""


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
    freshness_score: int        # 0 to 100 (100 = fully fresh)
    reading_minutes: float
    scope_source: str           # "CLAUDE.md" | "AGENTS.md" | "--scope flag" | "all files"
    scope_glob: str


# ---------------------------------------------------------------------------
# Scope detection
# ---------------------------------------------------------------------------

_SCOPE_FILES = ("CLAUDE.md", "AGENTS.md")
_SCOPE_HEADING = re.compile(r"(scope|files|include|watch)", re.I)
# A "*" bullet needs a space or tab after it, so a glob such as *.py stays
# whole. A "-" line loses every leading "-", space and tab.
_BULLET = re.compile(r"^(?:\*[ \t]+|[- \t]+)")


def _scope_globs(text: str) -> list[str]:
    """Return the globs listed under scope headings in ``text``."""
    globs: list[str] = []
    in_scope = False
    for line in text.splitlines():
        stripped = line.strip()
        if _SCOPE_HEADING.search(stripped) and stripped.endswith(":"):
            in_scope = True
            continue
        if in_scope:
            if stripped.startswith("-") or stripped.startswith("*"):
                glob = _BULLET.sub("", stripped).strip("`")
                if glob:
                    globs.append(glob)
            elif stripped and not stripped.startswith("#"):
                in_scope = False
    return globs


def _first_scope(paths: Iterable[Path]) -> list[str]:
    """Return the scope globs of the first file in ``paths`` that lists any."""
    for p in paths:
        if not p.is_file():
            continue
        globs = _scope_globs(p.read_text(encoding="utf-8", errors="replace"))
        if globs:
            return globs
    return []


def _parse_scope_from_agents_md(path: str = "CLAUDE.md") -> list[str]:
    """Extract file globs from CLAUDE.md / AGENTS.md scope sections.

    Reads CLAUDE.md, AGENTS.md and then ``path`` from the current directory,
    and returns the globs of the first one that lists any.
    """
    return _first_scope(Path(name) for name in ("CLAUDE.md", "AGENTS.md", path))


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

_GIT_TIMEOUT = 30
# git cannot read an ISO date after 2099, so a later time is clamped to this.
_LATEST_GIT_TIME = 4102444799  # 2099-12-31T23:59:59Z
_CHANGE_TYPES = {
    "A": "added",
    "C": "added",
    "D": "deleted",
    "M": "modified",
    "T": "modified",
    "R": "renamed",
}
_SINCE_FORM = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


def _parse_since(text: str) -> float:
    """Return 00:00:00 UTC on the YYYY-MM-DD date ``text`` as a Unix time."""
    try:
        if not _SINCE_FORM.fullmatch(text):
            raise ValueError(text)
        day = datetime.strptime(text, "%Y-%m-%d")
    except ValueError:
        raise FreshnessError(
            f"--since must be a date in the form YYYY-MM-DD, not {text!r}"
        ) from None
    return day.replace(tzinfo=timezone.utc).timestamp()


def _run_git(cwd: str, args: list[str], where: str) -> subprocess.CompletedProcess:
    """Run git in ``cwd``. ``where`` names the repository in an error.

    Raises FreshnessError when git cannot start or times out.
    """
    try:
        return subprocess.run(
            ["git", "-C", cwd, *args],
            input=b"",
            capture_output=True,
            timeout=_GIT_TIMEOUT,
        )
    except FileNotFoundError:
        raise FreshnessError(
            f"git was not found on PATH, so freshness cannot read {where}"
        ) from None
    except subprocess.TimeoutExpired:
        raise FreshnessError(
            f"git {args[0]} timed out after {_GIT_TIMEOUT}s in {where}"
        ) from None
    except OSError as exc:
        raise FreshnessError(f"cannot run git in {where}: {inert(exc)}") from None


def _git_reason(result: subprocess.CompletedProcess) -> str:
    """Return the first line git wrote to stderr, or its exit code."""
    for line in result.stderr.decode("utf-8", "replace").splitlines():
        if line.strip():
            return inert(line.strip())
    return f"exit code {result.returncode}"


def _git(cwd: str, args: list[str], where: str) -> bytes:
    """Return the output of git in ``cwd``. Raise FreshnessError if it fails."""
    result = _run_git(cwd, args, where)
    if result.returncode != 0:
        raise FreshnessError(f"git {args[0]} failed in {where}: {_git_reason(result)}")
    return result.stdout


def _line_count(text: bytes) -> int:
    # git writes "-" for both counts of a binary file.
    return int(text) if text.isdigit() else 0


def _parse_diff(out: bytes, where: str) -> list[tuple[str, int, str]]:
    """Read the output of ``git diff -z --raw --numstat``.

    git writes every raw entry, then every numstat entry. A raw entry is a
    header that starts with ":" and ends with the status letter, then the
    path, or the old and new paths for a rename or copy. A numstat entry is
    the added count, the removed count and the path, split by tabs. For a
    rename or copy the path is empty, and the old and new paths follow.
    """
    fields = out.split(b"\0")
    entries: list[tuple[str, bytes]] = []
    counts: dict[bytes, int] = {}
    i = 0
    try:
        while fields[i].startswith(b":"):
            status = fields[i].rsplit(b" ", 1)[-1][:1].decode("ascii", "replace")
            if status in ("R", "C"):
                entries.append((status, fields[i + 2]))
                i += 3
            else:
                entries.append((status, fields[i + 1]))
                i += 2
        while fields[i]:
            added, removed, path = fields[i].split(b"\t", 2)
            if path:
                i += 1
            else:
                path = fields[i + 2]
                i += 3
            counts[path] = _line_count(added) + _line_count(removed)
    except (IndexError, ValueError):
        raise FreshnessError(f"cannot read the output of git diff in {where}") from None
    return [
        (
            path.decode("utf-8", "surrogateescape"),
            counts.get(path, 0),
            _CHANGE_TYPES.get(status, "modified"),
        )
        for status, path in entries
    ]


def _changes_since(repo: str, since_ts: float) -> tuple[str, list[tuple[str, int, str]]]:
    """Return the top level of ``repo`` and the files changed since ``since_ts``.

    Each file is (path, lines_changed, change_type), with the path relative
    to the top level. The base is the newest first-parent commit of HEAD made
    at or before ``since_ts``. When there is none, the base is the empty tree.
    """
    repo = repo or "."
    where = inert(os.path.abspath(repo))
    if not os.path.exists(repo):
        raise FreshnessError(f"{where} does not exist")
    if not os.path.isdir(repo):
        raise FreshnessError(f"{where} is not a directory")
    out = _git(repo, ["rev-parse", "--show-toplevel"], where)
    top = os.fsdecode(out[:-1] if out.endswith(b"\n") else out)
    if not top:
        raise FreshnessError(f"{where} is not in a git work tree")
    where = inert(top)

    result = _run_git(top, ["rev-parse", "--verify", "--quiet", "HEAD^{commit}"], where)
    if result.returncode != 0:
        if not result.stderr.strip():
            raise FreshnessError(f"the git repository at {where} has no commits")
        raise FreshnessError(f"git rev-parse failed in {where}: {_git_reason(result)}")
    head = result.stdout.decode("ascii", "replace").strip()

    base = ""
    # No commit is dated before 1970, so an earlier time needs no lookup.
    if since_ts >= 0:
        before = datetime.fromtimestamp(
            min(since_ts, _LATEST_GIT_TIME), tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        # "--" stops git from reading a revision as a file of the same name.
        base = _git(
            top,
            ["rev-list", "-1", "--first-parent", f"--before={before}", head, "--"],
            where,
        ).decode("ascii", "replace").strip()
        if not base:
            shallow = _git(top, ["rev-parse", "--is-shallow-repository"], where)
            if shallow.strip() == b"true":
                raise FreshnessError(
                    f"the git repository at {where} is a shallow clone with no "
                    f"commit at or before {before}. Run git fetch --unshallow, "
                    "or pass a later --since."
                )
    if not base:
        # The empty tree ID depends on the repository's hash algorithm.
        base = _git(top, ["hash-object", "-t", "tree", "--stdin"], where).decode(
            "ascii", "replace"
        ).strip()

    out = _git(
        top,
        ["diff", "-z", "-M", "--no-color", "--raw", "--numstat", base, head, "--"],
        where,
    )
    return top, _parse_diff(out, where)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse_freshness(
    store: TraceStore,
    since_date: str = "",
    scope_glob: str = "",
    repo: str = ".",
) -> FreshnessReport:
    """Compute context freshness relative to the last session.

    Raises NoSessionsError when the store has no session and ``since_date``
    is empty. Raises FreshnessError when ``since_date`` is not a YYYY-MM-DD
    date, when the newest session has no usable start time, or when git
    fails.
    """
    since_ts = _parse_since(since_date) if since_date else None

    # Find last session timestamp
    last_meta = store.get_latest_session()
    last_ts = usable_time(last_meta.started_at) if last_meta else None
    last_sid = last_meta.session_id if last_meta else ""
    if since_ts is None:
        if last_meta is None:
            raise NoSessionsError("No sessions found.")
        if last_ts is None:
            raise FreshnessError(
                f"session {inert(last_sid)} has no usable start time. "
                "Pass --since YYYY-MM-DD to compare with a date."
            )
        since_ts = last_ts

    # Get changed files. Paths are relative to the repository's top level.
    top, raw_files = _changes_since(repo, since_ts)

    # Determine scope
    scope_source = "all files"
    scope_globs: list[str] = []
    if scope_glob:
        scope_globs = [scope_glob]
        scope_source = "--scope flag"
    else:
        scope_globs = _first_scope(Path(top, name) for name in _SCOPE_FILES)
        if scope_globs:
            scope_source = "CLAUDE.md / AGENTS.md"

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

    session = inert(report.last_session_id[:12])
    if report.last_session_ts is not None:
        age_h = int((time.time() - report.last_session_ts) / 3600)
        age_str = f"{age_h}h ago" if age_h < 48 else f"{age_h // 24}d ago"
        w(f"Last session: {session} ({age_str})\n")
    elif report.last_session_id:
        w(f"Last session: {session} (start time unknown)\n")
    else:
        w("Last session: none found\n")

    w(f"Scope: {inert(report.scope_glob)}  [{report.scope_source}]\n")
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
            w(f"  {icon} {inert(f.path)}\n")
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

    try:
        report = analyse_freshness(store, since_date=since, scope_glob=scope, repo=repo)
    except NoSessionsError:
        sys.stderr.write(
            "No sessions found. Pass --since YYYY-MM-DD to compare with a date.\n"
        )
        return 1
    except FreshnessError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1
    format_freshness(report, out=sys.stdout)
    return 0
