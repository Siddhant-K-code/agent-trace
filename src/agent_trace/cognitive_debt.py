"""Cognitive debt reporting for agent-written code.

The report is intentionally local and best-effort: it derives agent-written
lines from trace events, then uses git history when available to estimate how
much of that code has review evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time as _time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, TextIO

from .budget_report import _parse_date
from .models import EventType, SessionMeta, TraceEvent
from .store import TraceStore


@dataclass
class FileDebt:
    path: str
    agent_written_lines: int
    human_reviewed_lines: int = 0
    review_signals: list[str] = field(default_factory=list)

    @property
    def zero_review(self) -> bool:
        return self.agent_written_lines > 0 and self.human_reviewed_lines <= 0


@dataclass
class SessionDebt:
    session_id: str
    started_at: float
    branch: str
    author: str
    agent_written_lines: int
    human_reviewed_lines: int
    files: list[FileDebt]
    git_available: bool = True

    @property
    def debt_score(self) -> float:
        if self.agent_written_lines <= 0:
            return 0.0
        unreviewed = max(0, self.agent_written_lines - self.human_reviewed_lines)
        return unreviewed / self.agent_written_lines

    @property
    def zero_review_files(self) -> list[FileDebt]:
        return [f for f in self.files if f.zero_review]


@dataclass
class CognitiveDebtReport:
    window_start: float
    window_end: float
    group_by: str
    threshold: float
    sessions: list[SessionDebt]
    rows: dict[str, dict]
    git_available: bool

    @property
    def total_agent_written_lines(self) -> int:
        return sum(s.agent_written_lines for s in self.sessions)

    @property
    def total_reviewed_lines(self) -> int:
        return sum(s.human_reviewed_lines for s in self.sessions)

    @property
    def average_score(self) -> float:
        if not self.sessions:
            return 0.0
        return sum(s.debt_score for s in self.sessions) / len(self.sessions)

    @property
    def high_debt_sessions(self) -> list[SessionDebt]:
        return [s for s in self.sessions if s.debt_score > self.threshold]

    @property
    def zero_review_files(self) -> list[tuple[SessionDebt, FileDebt]]:
        rows: list[tuple[SessionDebt, FileDebt]] = []
        for session in self.sessions:
            for file_debt in session.zero_review_files:
                rows.append((session, file_debt))
        return rows


def _git(cwd: str, *args: str, timeout: float = 2.0) -> str:
    if not shutil.which("git"):
        return ""
    try:
        result = subprocess.run(
            ["git", "-C", cwd, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _meta_attr(meta: SessionMeta) -> dict:
    attr = getattr(meta, "attribution", {}) or {}
    return attr if isinstance(attr, dict) else {}


def _working_dir(meta: SessionMeta) -> str:
    attr = _meta_attr(meta)
    return str(attr.get("working_dir") or os.getcwd())


def _branch(meta: SessionMeta, cwd: str) -> str:
    attr = _meta_attr(meta)
    branch = attr.get("git_branch") or _git(cwd, "branch", "--show-current")
    if branch:
        return str(branch)
    match = re.search(r"branch:\s*([^,)]+)", getattr(meta, "command", "") or "")
    if match:
        return match.group(1).strip()
    return "(unknown)"


def _author(meta: SessionMeta, cwd: str) -> str:
    attr = _meta_attr(meta)
    return (
        str(attr.get("git_author") or "")
        or _git(cwd, "config", "user.email")
        or str(attr.get("os_user") or "")
        or os.environ.get("USER", "")
        or os.environ.get("USERNAME", "")
        or "(unknown)"
    )


def _path_from_event(event: TraceEvent) -> str:
    if event.event_type == EventType.FILE_WRITE:
        return str(
            event.data.get("path")
            or event.data.get("file_path")
            or event.data.get("uri")
            or ""
        )
    if event.event_type != EventType.TOOL_CALL:
        return ""
    tool = str(event.data.get("tool_name", "")).lower()
    args = event.data.get("arguments", {}) or {}
    if not isinstance(args, dict):
        return ""
    write_tools = {"write", "edit", "multiedit", "write_file", "edit_file", "replace", "str_replace"}
    if tool in write_tools or "write" in tool or "edit" in tool or "replace" in tool:
        return str(args.get("file_path") or args.get("path") or "")
    return ""


def _line_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        if not value:
            return 0
        return max(1, len(value.splitlines()))
    if isinstance(value, list):
        return sum(_line_count(item) for item in value)
    if isinstance(value, dict):
        keys = (
            "content",
            "text",
            "new_text",
            "new_string",
            "replacement",
        )
        return sum(_line_count(value.get(key)) for key in keys)
    return 0


def _patch_added_lines(text: str) -> int:
    added = 0
    for line in text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
    return added


def _event_line_count(event: TraceEvent) -> int:
    data = event.data
    args = data.get("arguments", {}) if isinstance(data.get("arguments", {}), dict) else {}
    total = _line_count(data) + _line_count(args)
    for source in (data, args):
        for key in ("patch", "diff"):
            value = source.get(key) if isinstance(source, dict) else None
            if isinstance(value, str):
                total += _patch_added_lines(value)
    return max(1, total)


def modified_file_lines(events: list[TraceEvent]) -> dict[str, int]:
    files: dict[str, int] = {}
    for event in events:
        path = _path_from_event(event)
        if not path:
            continue
        files[path] = files.get(path, 0) + _event_line_count(event)
    return files


def _git_available(cwd: str) -> bool:
    return bool(_git(cwd, "rev-parse", "--is-inside-work-tree"))


def _review_for_file(cwd: str, path: str, since_ts: float, agent_lines: int) -> tuple[int, list[str]]:
    if not _git_available(cwd):
        return 0, ["git_unavailable"]

    since = f"@{int(since_ts)}"
    raw = _git(
        cwd,
        "log",
        f"--since={since}",
        "--format=%H%x00%ae%x00%s%x00%b%x1e",
        "--",
        path,
        timeout=3.0,
    )
    if not raw:
        return 0, ["not_committed"]

    signals: set[str] = {"committed_after_session"}
    authors: set[str] = set()
    factor = 0.15
    for record in raw.split("\x1e"):
        if not record.strip():
            continue
        parts = record.strip().split("\x00", 3)
        if len(parts) < 4:
            continue
        _, author, subject, body = parts
        authors.add(author)
        text = f"{subject}\n{body}".lower()
        if "merge pull request" in text or "pull request" in text or re.search(r"\(#\d+\)", subject):
            signals.add("pull_request_merge")
            factor = max(factor, 0.35)
        if "co-authored-by:" in text:
            signals.add("multiple_authors")
            factor = max(factor, 0.45)
        if "reviewed-by:" in text:
            signals.add("reviewed_by_trailer")
            factor = max(factor, 0.75)
        if "review" in text or "comment" in text or "line" in text:
            signals.add("review_reference")
            factor = max(factor, 0.55)
    if len(authors) > 1:
        signals.add("multiple_commit_authors")
        factor = max(factor, 0.50)

    return min(agent_lines, int(round(agent_lines * factor))), sorted(signals)


def _github_repo(cwd: str) -> tuple[str, str] | None:
    remote = _git(cwd, "remote", "get-url", "origin")
    if not remote:
        return None
    match = re.search(r"github\.com[:/]([^/]+)/([^/.]+)(?:\.git)?$", remote)
    if not match:
        return None
    return match.group(1), match.group(2)


def _github_json(path: str, token: str) -> dict | list:
    req = urllib.request.Request(
        f"https://api.github.com{path}",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "agent-strace-cognitive-debt",
        },
    )
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def _github_review_for_file(
    cwd: str,
    branch: str,
    path: str,
    token: str,
    agent_lines: int,
) -> tuple[int, list[str]]:
    repo = _github_repo(cwd)
    if not repo or not token or not branch or branch == "(unknown)":
        return 0, []
    owner, name = repo
    query = urllib.parse.urlencode({
        "q": f"repo:{owner}/{name} is:pr is:merged head:{branch}",
        "per_page": "5",
    })
    try:
        search = _github_json(f"/search/issues?{query}", token)
        items = search.get("items", []) if isinstance(search, dict) else []
        if not items:
            return 0, []
        number = items[0].get("number")
        if not number:
            return 0, []

        factor = 0.45
        signals = {"github_merged_pr"}
        reviews = _github_json(f"/repos/{owner}/{name}/pulls/{number}/reviews?per_page=100", token)
        if isinstance(reviews, list) and reviews:
            factor = max(factor, 0.75)
            signals.add("github_reviews")
        comments = _github_json(f"/repos/{owner}/{name}/pulls/{number}/comments?per_page=100", token)
        if isinstance(comments, list) and comments:
            factor = max(factor, 0.85)
            signals.add("github_review_comments")
            if any(comment.get("path") == path for comment in comments if isinstance(comment, dict)):
                factor = 1.0
                signals.add("github_line_comments")
        return min(agent_lines, int(round(agent_lines * factor))), sorted(signals)
    except (OSError, urllib.error.HTTPError, urllib.error.URLError, ValueError, KeyError, TypeError):
        return 0, []


def analyze_session_debt(store: TraceStore, meta: SessionMeta, github_token: str = "") -> SessionDebt:
    cwd = _working_dir(meta)
    branch = _branch(meta, cwd)
    author = _author(meta, cwd)
    git_ok = _git_available(cwd)
    try:
        events = store.load_events(meta.session_id)
    except Exception:
        events = []

    files = []
    total_written = 0
    total_reviewed = 0
    for path, lines in sorted(modified_file_lines(events).items()):
        reviewed, signals = _review_for_file(cwd, path, meta.started_at, lines)
        gh_reviewed, gh_signals = _github_review_for_file(cwd, branch, path, github_token, lines)
        if gh_reviewed > reviewed:
            reviewed = gh_reviewed
        signals = sorted(set(signals) | set(gh_signals))
        files.append(FileDebt(
            path=path,
            agent_written_lines=lines,
            human_reviewed_lines=reviewed,
            review_signals=signals,
        ))
        total_written += lines
        total_reviewed += reviewed

    return SessionDebt(
        session_id=meta.session_id,
        started_at=meta.started_at,
        branch=branch,
        author=author,
        agent_written_lines=total_written,
        human_reviewed_lines=total_reviewed,
        files=files,
        git_available=git_ok,
    )


def _row_key(group_by: str, session: SessionDebt) -> str:
    if group_by == "branch":
        return session.branch
    return session.author


def build_cognitive_debt_report(
    store: TraceStore,
    window_start: float,
    window_end: float,
    session_id: str = "",
    group_by: str = "author",
    threshold: float = 0.7,
    github_token: str = "",
) -> CognitiveDebtReport:
    metas = []
    if session_id:
        full_id = store.find_session(session_id) or session_id
        if store.session_exists(full_id):
            metas = [store.load_meta(full_id)]
    else:
        metas = [
            meta for meta in store.list_sessions()
            if window_start <= meta.started_at < window_end
        ]

    sessions = [analyze_session_debt(store, meta, github_token=github_token) for meta in metas]
    rows: dict[str, dict] = {}
    for session in sessions:
        key = _row_key(group_by, session)
        if key not in rows:
            rows[key] = {
                "group": key,
                "sessions": 0,
                "agent_written_lines": 0,
                "human_reviewed_lines": 0,
                "score": 0.0,
                "high_debt_sessions": 0,
            }
        row = rows[key]
        row["sessions"] += 1
        row["agent_written_lines"] += session.agent_written_lines
        row["human_reviewed_lines"] += session.human_reviewed_lines
        if session.debt_score > threshold:
            row["high_debt_sessions"] += 1

    for row in rows.values():
        written = row["agent_written_lines"]
        reviewed = row["human_reviewed_lines"]
        row["score"] = 0.0 if written <= 0 else max(0, written - reviewed) / written

    rows = dict(sorted(rows.items(), key=lambda item: item[1]["score"], reverse=True))
    return CognitiveDebtReport(
        window_start=window_start,
        window_end=window_end,
        group_by=group_by,
        threshold=threshold,
        sessions=sessions,
        rows=rows,
        git_available=all(s.git_available for s in sessions) if sessions else _git_available(os.getcwd()),
    )


def _fmt_date(ts: float) -> str:
    import datetime
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def _fmt_score(score: float) -> str:
    return f"{score:.2f}"


def format_cognitive_debt_text(report: CognitiveDebtReport, out: TextIO = sys.stdout) -> None:
    w = out.write
    w(f"Cognitive debt report - {_fmt_date(report.window_start)} to {_fmt_date(report.window_end)}\n")
    if not report.git_available:
        w("Review scoring: git unavailable for one or more sessions; review lines default to zero.\n")
    w("\n")
    if not report.sessions:
        w("No sessions with agent-written files in this period.\n")
        return

    w(f"{'Session':<14}  {'Branch':<24}  {'Lines':>7}  {'Reviewed':>8}  {'Score':>5}\n")
    w("-" * 70 + "\n")
    for session in sorted(report.sessions, key=lambda s: s.debt_score, reverse=True):
        if session.agent_written_lines <= 0:
            continue
        marker = "!" if session.debt_score > report.threshold else " "
        w(
            f"{session.session_id[:12]:<14}  {session.branch[:24]:<24}  "
            f"{session.agent_written_lines:>7}  {session.human_reviewed_lines:>8}  "
            f"{_fmt_score(session.debt_score):>5} {marker}\n"
        )
    w("-" * 70 + "\n")
    w(f"Average score: {_fmt_score(report.average_score)}\n")

    if report.rows:
        w(f"\nBy {report.group_by}:\n")
        for row in report.rows.values():
            w(
                f"  {row['group']}: score {_fmt_score(row['score'])}, "
                f"{row['agent_written_lines']} lines, {row['human_reviewed_lines']} reviewed\n"
            )

    high = report.high_debt_sessions
    if high:
        w(f"\nHigh-debt sessions (score > {report.threshold:g}):\n")
        for session in high[:5]:
            w(
                f"  {session.session_id[:12]} - {session.agent_written_lines} agent-written lines, "
                f"{session.human_reviewed_lines} reviewed\n"
            )

    zero_review = report.zero_review_files
    if zero_review:
        w("\nFiles with zero human review:\n")
        for session, file_debt in zero_review[:10]:
            w(f"  {file_debt.path} ({file_debt.agent_written_lines} lines, session {session.session_id[:12]})\n")


def format_cognitive_debt_json(report: CognitiveDebtReport) -> str:
    return json.dumps({
        "window_start": report.window_start,
        "window_end": report.window_end,
        "group_by": report.group_by,
        "threshold": report.threshold,
        "git_available": report.git_available,
        "average_score": report.average_score,
        "total_agent_written_lines": report.total_agent_written_lines,
        "total_reviewed_lines": report.total_reviewed_lines,
        "rows": list(report.rows.values()),
        "sessions": [
            {
                "session_id": session.session_id,
                "started_at": session.started_at,
                "branch": session.branch,
                "author": session.author,
                "agent_written_lines": session.agent_written_lines,
                "human_reviewed_lines": session.human_reviewed_lines,
                "score": session.debt_score,
                "git_available": session.git_available,
                "files": [
                    {
                        "path": file_debt.path,
                        "agent_written_lines": file_debt.agent_written_lines,
                        "human_reviewed_lines": file_debt.human_reviewed_lines,
                        "review_signals": file_debt.review_signals,
                    }
                    for file_debt in session.files
                ],
            }
            for session in report.sessions
        ],
    }, indent=2)


def cmd_cognitive_debt(args: argparse.Namespace) -> int:
    store = TraceStore(args.trace_dir)
    now = _time.time()
    until = _parse_date(getattr(args, "until", "")) if getattr(args, "until", "") else now
    since_arg = getattr(args, "since", "") or "30d"
    since = _parse_date(since_arg)
    threshold = float(getattr(args, "threshold", 0.7))
    group_by = getattr(args, "by", "author")
    report = build_cognitive_debt_report(
        store,
        window_start=since,
        window_end=until,
        session_id=getattr(args, "session", "") or "",
        group_by=group_by,
        threshold=threshold,
        github_token=getattr(args, "github_token", "") or "",
    )

    if getattr(args, "format", "text") == "json":
        sys.stdout.write(format_cognitive_debt_json(report) + "\n")
    else:
        format_cognitive_debt_text(report)
    return 0
