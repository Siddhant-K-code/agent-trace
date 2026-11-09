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
from pathlib import Path
from typing import Any, TextIO

from .budget_report import _parse_date
from .event_data import require_object_event_data
from .models import EventType, SessionMeta, TraceEvent
from .store import TraceStore, _write_atomic


COGNITIVE_DEBT_SCHEMA_VERSION = 1
COGNITIVE_DEBT_BADGE_SCHEMA_VERSION = 1


class CognitiveDebtInputError(ValueError):
    """The report cannot safely interpret trace or review evidence."""


@dataclass
class FileDebt:
    path: str
    agent_written_lines: int
    human_reviewed_lines: int = 0
    review_signals: list[str] = field(default_factory=list)
    review_state: str = "known"

    @property
    def unreviewed_lines(self) -> int:
        if self.review_state != "known":
            return 0
        return max(0, self.agent_written_lines - self.human_reviewed_lines)

    @property
    def unknown_review_lines(self) -> int:
        return self.agent_written_lines if self.review_state != "known" else 0

    @property
    def zero_review(self) -> bool:
        return (
            self.review_state == "known"
            and self.agent_written_lines > 0
            and self.human_reviewed_lines <= 0
        )


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
    repository: str = "(unknown)"
    unknown_authorship_lines: int = 0

    @property
    def unreviewed_lines(self) -> int:
        return max(
            0,
            self.agent_written_lines
            - self.human_reviewed_lines
            - self.unknown_review_lines,
        )

    @property
    def unknown_review_lines(self) -> int:
        return self.unknown_authorship_lines + sum(
            file_debt.unknown_review_lines for file_debt in self.files
        )

    @property
    def authorship_state(self) -> str:
        if self.agent_written_lines <= 0:
            return "unknown"
        if self.unknown_authorship_lines <= 0:
            return "known"
        if self.unknown_authorship_lines >= self.agent_written_lines:
            return "unknown"
        return "partial"

    @property
    def review_state(self) -> str:
        if self.agent_written_lines <= 0:
            return "unknown"
        if self.unknown_review_lines <= 0:
            return "known"
        if self.unknown_review_lines >= self.agent_written_lines:
            return "unknown"
        return "partial"

    @property
    def debt_score(self) -> float | None:
        if self.agent_written_lines <= 0 or self.unknown_review_lines > 0:
            return None
        return self.unreviewed_lines / self.agent_written_lines

    @property
    def zero_review_files(self) -> list[FileDebt]:
        return [f for f in self.files if f.zero_review]

    def status(self, threshold: float) -> str:
        return _coverage_status(
            total_lines=self.agent_written_lines,
            unreviewed_lines=self.unreviewed_lines,
            unknown_review_lines=self.unknown_review_lines,
            threshold=threshold,
        )


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
    def total_unreviewed_lines(self) -> int:
        return sum(s.unreviewed_lines for s in self.sessions)

    @property
    def total_unknown_review_lines(self) -> int:
        return sum(s.unknown_review_lines for s in self.sessions)

    @property
    def total_unknown_authorship_lines(self) -> int:
        return sum(s.unknown_authorship_lines for s in self.sessions)

    @property
    def reviewed_percent(self) -> float | None:
        return _percent(self.total_reviewed_lines, self.total_agent_written_lines)

    @property
    def unreviewed_percent(self) -> float | None:
        return _percent(self.total_unreviewed_lines, self.total_agent_written_lines)

    @property
    def unknown_review_percent(self) -> float | None:
        return _percent(self.total_unknown_review_lines, self.total_agent_written_lines)

    @property
    def debt_score(self) -> float | None:
        if self.total_agent_written_lines <= 0 or self.total_unknown_review_lines > 0:
            return None
        return self.total_unreviewed_lines / self.total_agent_written_lines

    @property
    def average_score(self) -> float | None:
        return self.debt_score

    @property
    def authorship_state(self) -> str:
        if self.total_agent_written_lines <= 0:
            return "unknown"
        if self.total_unknown_authorship_lines <= 0:
            return "known"
        if self.total_unknown_authorship_lines >= self.total_agent_written_lines:
            return "unknown"
        return "partial"

    @property
    def review_state(self) -> str:
        if self.total_agent_written_lines <= 0:
            return "unknown"
        if self.total_unknown_review_lines <= 0:
            return "known"
        if self.total_unknown_review_lines >= self.total_agent_written_lines:
            return "unknown"
        return "partial"

    @property
    def status(self) -> str:
        return _coverage_status(
            total_lines=self.total_agent_written_lines,
            unreviewed_lines=self.total_unreviewed_lines,
            unknown_review_lines=self.total_unknown_review_lines,
            threshold=self.threshold,
        )

    @property
    def status_label(self) -> str:
        return _status_label(self.status)

    @property
    def high_debt_sessions(self) -> list[SessionDebt]:
        return [
            session
            for session in self.sessions
            if session.debt_score is not None and session.debt_score > self.threshold
        ]

    @property
    def zero_review_files(self) -> list[tuple[SessionDebt, FileDebt]]:
        rows: list[tuple[SessionDebt, FileDebt]] = []
        for session in self.sessions:
            for file_debt in session.zero_review_files:
                rows.append((session, file_debt))
        return rows

    @property
    def unknown_review_files(self) -> list[tuple[SessionDebt, FileDebt]]:
        rows: list[tuple[SessionDebt, FileDebt]] = []
        for session in self.sessions:
            for file_debt in session.files:
                if file_debt.review_state != "known":
                    rows.append((session, file_debt))
        return rows


def _percent(part: int, total: int) -> float | None:
    if total <= 0:
        return None
    return part * 100.0 / total


def _coverage_status(
    *,
    total_lines: int,
    unreviewed_lines: int,
    unknown_review_lines: int,
    threshold: float,
) -> str:
    if total_lines <= 0 or unknown_review_lines > 0:
        return "unknown"
    if unreviewed_lines <= 0:
        return "clean"
    if unreviewed_lines / total_lines > threshold:
        return "above_threshold"
    return "partial"


def _status_label(status: str) -> str:
    return {
        "clean": "Review coverage complete",
        "partial": "Review coverage partial",
        "above_threshold": "Unreviewed coverage above threshold",
        "unknown": "Review coverage unknown",
    }[status]


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


def _repository(meta: SessionMeta, cwd: str) -> str:
    attr = _meta_attr(meta)
    remote = str(attr.get("git_repo") or "") or _git(cwd, "remote", "get-url", "origin")
    match = re.search(r"(?:^|[:/])([^/:]+)/([^/]+?)(?:\.git)?$", remote)
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    root = _git(cwd, "rev-parse", "--show-toplevel")
    return Path(root).name if root else "(unknown)"


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


_WRITE_TOOLS = {
    "write",
    "edit",
    "multiedit",
    "write_file",
    "edit_file",
    "replace",
    "str_replace",
}


def _is_write_event(event: TraceEvent) -> bool:
    if event.event_type == EventType.FILE_WRITE:
        return True
    if event.event_type != EventType.TOOL_CALL:
        return False
    tool_value = event.data.get("tool_name", "")
    if not isinstance(tool_value, str):
        raise CognitiveDebtInputError("write tool name must be a string")
    tool = tool_value.lower()
    return (
        tool in _WRITE_TOOLS
        or "write" in tool
        or "edit" in tool
        or "replace" in tool
    )


def _event_path_value(data: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = data.get(key)
        if value in (None, ""):
            continue
        if not isinstance(value, str):
            raise CognitiveDebtInputError(f"write event {key} must be a string")
        return value
    return ""


def _path_from_event(event: TraceEvent) -> str:
    if event.event_type == EventType.FILE_WRITE:
        return _event_path_value(event.data, ("path", "file_path", "uri"))
    if event.event_type != EventType.TOOL_CALL:
        return ""
    if not _is_write_event(event):
        return ""
    args = event.data.get("arguments", {})
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise CognitiveDebtInputError("write tool arguments must be an object")
    return _event_path_value(args, ("file_path", "path"))


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


# modified_file_lines reads the data of every event of these types.
MODIFIED_FILE_DATA_EVENT_TYPES = frozenset({EventType.FILE_WRITE, EventType.TOOL_CALL})


def modified_file_lines(events: list[TraceEvent]) -> dict[str, int]:
    files, _ = _modified_file_coverage(events)
    return files


def _modified_file_coverage(events: list[TraceEvent]) -> tuple[dict[str, int], int]:
    files: dict[str, int] = {}
    unknown_authorship_lines = 0
    for event in events:
        if not _is_write_event(event):
            continue
        lines = _event_line_count(event)
        path = _path_from_event(event)
        if not path:
            unknown_authorship_lines += lines
            continue
        files[path] = files.get(path, 0) + lines
    return files, unknown_authorship_lines


def _git_available(cwd: str) -> bool:
    return bool(_git(cwd, "rev-parse", "--is-inside-work-tree"))


def _review_log(cwd: str, path: str, since_ts: float) -> str:
    since = f"@{int(since_ts)}"
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                cwd,
                "log",
                "--all",
                f"--since={since}",
                "--format=%H%x00%ae%x00%s%x00%b%x1e",
                "--",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CognitiveDebtInputError("local review history could not be read") from exc
    if result.returncode != 0:
        raise CognitiveDebtInputError("local review history could not be read")
    return result.stdout.strip()


def _review_for_file(
    cwd: str,
    path: str,
    since_ts: float,
    agent_lines: int,
) -> tuple[int, list[str], bool]:
    if not _git_available(cwd):
        return 0, ["git_unavailable"], False

    raw = _review_log(cwd, path, since_ts)
    if not raw:
        return 0, ["not_committed"], True

    signals: set[str] = {"committed_after_session"}
    authors: set[str] = set()
    factor = 0.15
    for record in raw.split("\x1e"):
        if not record.strip():
            continue
        parts = record.strip().split("\x00", 3)
        if len(parts) != 4 or not parts[0] or not parts[1]:
            raise CognitiveDebtInputError("local review history was malformed")
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

    return min(agent_lines, int(round(agent_lines * factor))), sorted(signals), True


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
        if not isinstance(search, dict) or not isinstance(search.get("items"), list):
            raise CognitiveDebtInputError("GitHub review search response was malformed")
        items = search["items"]
        if not items:
            return 0, []
        if not isinstance(items[0], dict):
            raise CognitiveDebtInputError("GitHub review search response was malformed")
        number = items[0].get("number")
        if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
            raise CognitiveDebtInputError("GitHub review search response was malformed")

        factor = 0.45
        signals = {"github_merged_pr"}
        reviews = _github_json(f"/repos/{owner}/{name}/pulls/{number}/reviews?per_page=100", token)
        if not isinstance(reviews, list) or any(not isinstance(item, dict) for item in reviews):
            raise CognitiveDebtInputError("GitHub reviews response was malformed")
        if reviews:
            factor = max(factor, 0.75)
            signals.add("github_reviews")
        comments = _github_json(f"/repos/{owner}/{name}/pulls/{number}/comments?per_page=100", token)
        if not isinstance(comments, list) or any(not isinstance(item, dict) for item in comments):
            raise CognitiveDebtInputError("GitHub review comments response was malformed")
        if comments:
            factor = max(factor, 0.85)
            signals.add("github_review_comments")
            if any(comment.get("path") == path for comment in comments):
                factor = 1.0
                signals.add("github_line_comments")
        return min(agent_lines, int(round(agent_lines * factor))), sorted(signals)
    except CognitiveDebtInputError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError, ValueError) as exc:
        raise CognitiveDebtInputError("GitHub review response was malformed") from exc
    except (OSError, urllib.error.HTTPError, urllib.error.URLError):
        return 0, []


def analyze_session_debt(
    store: TraceStore,
    meta: SessionMeta,
    github_token: str = "",
    *,
    name_session: bool = False,
) -> SessionDebt:
    """Measure one session's cognitive debt.

    Raises NonObjectEventDataError when a file_write or tool_call event's
    data is not an object; ``name_session`` adds the session ID to that error.
    """
    cwd = _working_dir(meta)
    branch = _branch(meta, cwd)
    author = _author(meta, cwd)
    repository = _repository(meta, cwd)
    git_ok = _git_available(cwd)
    try:
        events = store.load_events(meta.session_id)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CognitiveDebtInputError(
            f"session {meta.session_id} trace could not be read"
        ) from exc
    require_object_event_data(
        events,
        event_types=MODIFIED_FILE_DATA_EVENT_TYPES,
        session_id=meta.session_id if name_session else None,
    )

    files = []
    modified_files, unknown_authorship_lines = _modified_file_coverage(events)
    total_written = unknown_authorship_lines
    total_reviewed = 0
    for path, lines in sorted(modified_files.items()):
        reviewed, signals, review_known = _review_for_file(
            cwd, path, meta.started_at, lines
        )
        gh_reviewed, gh_signals = _github_review_for_file(cwd, branch, path, github_token, lines)
        if gh_reviewed > reviewed:
            reviewed = gh_reviewed
        if gh_reviewed > 0 or gh_signals:
            review_known = True
        signals = sorted(set(signals) | set(gh_signals))
        files.append(FileDebt(
            path=path,
            agent_written_lines=lines,
            human_reviewed_lines=reviewed,
            review_signals=signals,
            review_state="known" if review_known else "unknown",
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
        repository=repository,
        unknown_authorship_lines=unknown_authorship_lines,
    )


def _row_key(group_by: str, session: SessionDebt) -> str:
    if group_by == "branch":
        return session.branch
    return session.repository


def build_cognitive_debt_report(
    store: TraceStore,
    window_start: float,
    window_end: float,
    session_id: str = "",
    group_by: str = "repository",
    threshold: float = 0.7,
    github_token: str = "",
) -> CognitiveDebtReport:
    if not 0.0 <= threshold <= 1.0:
        raise CognitiveDebtInputError("threshold must be between 0 and 1")
    if window_start >= window_end:
        raise CognitiveDebtInputError("reporting window start must be before its end")
    if group_by == "author":
        group_by = "repository"
    if group_by not in {"repository", "branch"}:
        raise CognitiveDebtInputError("grouping must be repository or branch")

    try:
        metas = []
        if session_id:
            full_id = store.find_session(session_id) or session_id
            if store.session_exists(full_id):
                metas = [store.load_meta(full_id)]
        else:
            metas = [
                meta for meta in store.list_sessions_strict()
                if window_start <= meta.started_at < window_end
            ]
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CognitiveDebtInputError("trace input could not be read safely") from exc

    sessions = [
        analyze_session_debt(
            store, meta, github_token=github_token, name_session=not session_id
        )
        for meta in metas
    ]
    rows: dict[str, dict] = {}
    for session in sessions:
        key = _row_key(group_by, session)
        if key not in rows:
            rows[key] = {
                "group": key,
                "sessions": 0,
                "agent_written_lines": 0,
                "human_reviewed_lines": 0,
                "reviewed_lines": 0,
                "unreviewed_lines": 0,
                "unknown_review_lines": 0,
                "unknown_authorship_lines": 0,
                "score": None,
                "high_debt_sessions": 0,
            }
        row = rows[key]
        row["sessions"] += 1
        row["agent_written_lines"] += session.agent_written_lines
        row["human_reviewed_lines"] += session.human_reviewed_lines
        row["reviewed_lines"] += session.human_reviewed_lines
        row["unreviewed_lines"] += session.unreviewed_lines
        row["unknown_review_lines"] += session.unknown_review_lines
        row["unknown_authorship_lines"] += session.unknown_authorship_lines
        if session.debt_score is not None and session.debt_score > threshold:
            row["high_debt_sessions"] += 1

    for row in rows.values():
        written = row["agent_written_lines"]
        unknown = row["unknown_review_lines"]
        row["score"] = (
            None
            if written <= 0 or unknown > 0
            else row["unreviewed_lines"] / written
        )
        row["status"] = _coverage_status(
            total_lines=written,
            unreviewed_lines=row["unreviewed_lines"],
            unknown_review_lines=unknown,
            threshold=threshold,
        )
        row["status_label"] = _status_label(row["status"])

    rows = dict(sorted(rows.items()))
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


def _fmt_score(score: float | None) -> str:
    return "unknown" if score is None else f"{score:.2f}"


def format_cognitive_debt_text(report: CognitiveDebtReport, out: TextIO = sys.stdout) -> None:
    w = out.write
    w(f"Cognitive debt report - {_fmt_date(report.window_start)} to {_fmt_date(report.window_end)}\n")
    w("Review coverage only. This does not measure code quality or causation.\n")
    if not report.git_available:
        w("Review state: git unavailable for one or more workflows; affected lines are unknown.\n")
    w(f"Status: {report.status_label} ({report.status})\n")
    w(
        "Coverage: "
        f"{report.total_reviewed_lines} reviewed, "
        f"{report.total_unreviewed_lines} unreviewed, "
        f"{report.total_unknown_review_lines} unknown "
        f"of {report.total_agent_written_lines} agent-written lines\n"
    )
    w("\n")
    if not report.sessions:
        w("No workflow trace evidence was available for this period.\n")
        return

    w(
        f"{'Workflow':<14}  {'Branch':<20}  {'Lines':>7}  {'Reviewed':>8}  "
        f"{'Unreviewed':>10}  {'Unknown':>7}  {'Status':<15}\n"
    )
    w("-" * 96 + "\n")
    for session in sorted(report.sessions, key=lambda item: (item.started_at, item.session_id)):
        if session.agent_written_lines <= 0:
            continue
        w(
            f"{session.session_id[:12]:<14}  {session.branch[:20]:<20}  "
            f"{session.agent_written_lines:>7}  {session.human_reviewed_lines:>8}  "
            f"{session.unreviewed_lines:>10}  {session.unknown_review_lines:>7}  "
            f"{session.status(report.threshold):<15}\n"
        )
    w("-" * 96 + "\n")
    w(f"Repository unreviewed ratio: {_fmt_score(report.debt_score)}\n")

    if report.rows:
        w(f"\nBy {report.group_by}:\n")
        for row in report.rows.values():
            w(
                f"  {row['group']}: {row['reviewed_lines']} reviewed, "
                f"{row['unreviewed_lines']} unreviewed, "
                f"{row['unknown_review_lines']} unknown, {row['status']}\n"
            )

    high = report.high_debt_sessions
    if high:
        w(f"\nWorkflows above the unreviewed threshold ({report.threshold:g}):\n")
        for session in high[:5]:
            w(
                f"  {session.session_id[:12]}: {session.agent_written_lines} agent-written lines, "
                f"{session.human_reviewed_lines} reviewed\n"
            )

    zero_review = report.zero_review_files
    if zero_review:
        w("\nFiles with zero human review:\n")
        for session, file_debt in zero_review[:10]:
            w(f"  {file_debt.path} ({file_debt.agent_written_lines} lines, session {session.session_id[:12]})\n")

    unknown_review = report.unknown_review_files
    if unknown_review:
        w("\nFiles with unknown review state:\n")
        for session, file_debt in unknown_review[:10]:
            w(
                f"  {file_debt.path} "
                f"({file_debt.agent_written_lines} lines, workflow {session.session_id[:12]})\n"
            )


def _coverage_payload(
    *,
    total_lines: int,
    reviewed_lines: int,
    unreviewed_lines: int,
    unknown_review_lines: int,
    unknown_authorship_lines: int,
) -> dict[str, int | float | None | str]:
    if total_lines <= 0:
        authorship_state = "unknown"
        review_state = "unknown"
    else:
        authorship_state = (
            "known"
            if unknown_authorship_lines <= 0
            else "unknown"
            if unknown_authorship_lines >= total_lines
            else "partial"
        )
        review_state = (
            "known"
            if unknown_review_lines <= 0
            else "unknown"
            if unknown_review_lines >= total_lines
            else "partial"
        )
    return {
        "agent_written_lines": total_lines,
        "authorship_state": authorship_state,
        "review_state": review_state,
        "reviewed_lines": reviewed_lines,
        "reviewed_percent": _percent(reviewed_lines, total_lines),
        "unreviewed_lines": unreviewed_lines,
        "unreviewed_percent": _percent(unreviewed_lines, total_lines),
        "unknown_review_lines": unknown_review_lines,
        "unknown_review_percent": _percent(unknown_review_lines, total_lines),
        "unknown_authorship_lines": unknown_authorship_lines,
        "unreviewed_ratio": (
            None
            if total_lines <= 0 or unknown_review_lines > 0
            else unreviewed_lines / total_lines
        ),
    }


def format_cognitive_debt_json(report: CognitiveDebtReport) -> str:
    ordered_sessions = sorted(
        report.sessions, key=lambda item: (item.started_at, item.session_id)
    )
    workflows = [
        {
            "workflow_id": session.session_id,
            "started_at": session.started_at,
            "repository": session.repository,
            "branch": session.branch,
            "status": session.status(report.threshold),
            "status_label": _status_label(session.status(report.threshold)),
            "git_available": session.git_available,
            "coverage": _coverage_payload(
                total_lines=session.agent_written_lines,
                reviewed_lines=session.human_reviewed_lines,
                unreviewed_lines=session.unreviewed_lines,
                unknown_review_lines=session.unknown_review_lines,
                unknown_authorship_lines=session.unknown_authorship_lines,
            ),
            "files": [
                {
                    "path": file_debt.path,
                    "review_state": file_debt.review_state,
                    "coverage": _coverage_payload(
                        total_lines=file_debt.agent_written_lines,
                        reviewed_lines=file_debt.human_reviewed_lines,
                        unreviewed_lines=file_debt.unreviewed_lines,
                        unknown_review_lines=file_debt.unknown_review_lines,
                        unknown_authorship_lines=0,
                    ),
                    "review_signals": file_debt.review_signals,
                }
                for file_debt in session.files
            ],
        }
        for session in ordered_sessions
    ]
    payload = {
        "schema_version": COGNITIVE_DEBT_SCHEMA_VERSION,
        "metric": "cognitive_debt",
        "scope": {
            "aggregate": "repository",
            "detail": "workflow",
            "group_by": report.group_by,
        },
        "status": report.status,
        "status_label": report.status_label,
        "window_start": report.window_start,
        "window_end": report.window_end,
        "threshold": report.threshold,
        "git_available": report.git_available,
        "coverage": _coverage_payload(
            total_lines=report.total_agent_written_lines,
            reviewed_lines=report.total_reviewed_lines,
            unreviewed_lines=report.total_unreviewed_lines,
            unknown_review_lines=report.total_unknown_review_lines,
            unknown_authorship_lines=report.total_unknown_authorship_lines,
        ),
        "groups": list(report.rows.values()),
        "workflows": workflows,
        "interpretation": {
            "measures_review_coverage": True,
            "measures_code_quality": False,
            "claims_causation": False,
            "ranks_individuals": False,
        },
        "group_by": report.group_by,
        "average_score": report.average_score,
        "total_agent_written_lines": report.total_agent_written_lines,
        "total_reviewed_lines": report.total_reviewed_lines,
        "total_unreviewed_lines": report.total_unreviewed_lines,
        "total_unknown_review_lines": report.total_unknown_review_lines,
        "rows": list(report.rows.values()),
        "sessions": [
            {
                "session_id": session.session_id,
                "started_at": session.started_at,
                "branch": session.branch,
                "agent_written_lines": session.agent_written_lines,
                "human_reviewed_lines": session.human_reviewed_lines,
                "unreviewed_lines": session.unreviewed_lines,
                "unknown_review_lines": session.unknown_review_lines,
                "score": session.debt_score,
                "git_available": session.git_available,
                "files": [
                    {
                        "path": file_debt.path,
                        "agent_written_lines": file_debt.agent_written_lines,
                        "human_reviewed_lines": file_debt.human_reviewed_lines,
                        "unreviewed_lines": file_debt.unreviewed_lines,
                        "unknown_review_lines": file_debt.unknown_review_lines,
                        "review_state": file_debt.review_state,
                        "review_signals": file_debt.review_signals,
                    }
                    for file_debt in session.files
                ],
            }
            for session in ordered_sessions
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _badge_percent(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.0f}%"


def format_cognitive_debt_badge(report: CognitiveDebtReport) -> str:
    if report.total_agent_written_lines <= 0:
        message = "review coverage unknown"
    elif report.total_unknown_review_lines > 0:
        message = (
            f"{_badge_percent(report.reviewed_percent)} reviewed, "
            f"{_badge_percent(report.unreviewed_percent)} unreviewed, "
            f"{_badge_percent(report.unknown_review_percent)} unknown"
        )
    else:
        message = (
            f"{_badge_percent(report.reviewed_percent)} reviewed, "
            f"{_badge_percent(report.unreviewed_percent)} unreviewed"
        )
    color = {
        "clean": "brightgreen",
        "partial": "yellow",
        "above_threshold": "orange",
        "unknown": "lightgrey",
    }[report.status]
    return json.dumps(
        {
            "schemaVersion": COGNITIVE_DEBT_BADGE_SCHEMA_VERSION,
            "label": "cognitive debt",
            "message": message,
            "color": color,
        },
        indent=2,
        sort_keys=True,
    )


def cmd_cognitive_debt(args: argparse.Namespace) -> int:
    store = TraceStore(args.trace_dir)
    now = _time.time()
    try:
        until = (
            _parse_date(getattr(args, "until", ""))
            if getattr(args, "until", "")
            else now
        )
        since_arg = getattr(args, "since", "") or "30d"
        since = _parse_date(since_arg)
    except (TypeError, ValueError) as exc:
        raise CognitiveDebtInputError("reporting window is invalid") from exc
    threshold = float(getattr(args, "threshold", 0.7))
    group_by = getattr(args, "by", "repository")
    report = build_cognitive_debt_report(
        store,
        window_start=since,
        window_end=until,
        session_id=getattr(args, "session", "") or "",
        group_by=group_by,
        threshold=threshold,
        github_token=getattr(args, "github_token", "") or "",
    )

    output_format = getattr(args, "format", "text")
    if output_format == "json":
        rendered = format_cognitive_debt_json(report)
    elif output_format == "badge":
        rendered = format_cognitive_debt_badge(report)
    else:
        from io import StringIO
        buffer = StringIO()
        format_cognitive_debt_text(report, out=buffer)
        rendered = buffer.getvalue().rstrip("\n")

    output_path = getattr(args, "output", "") or ""
    if output_path:
        try:
            _write_atomic(Path(output_path), rendered + "\n")
        except (OSError, ValueError) as exc:
            raise CognitiveDebtInputError("report output could not be written safely") from exc
    else:
        sys.stdout.write(rendered + "\n")
    return 0
