"""Cognitive debt reporting for agent-written code.

The report is intentionally local and best-effort: it derives agent-written
lines from trace events, then uses git history when available to estimate how
much of that code has review evidence.
"""

from __future__ import annotations

import argparse
import datetime
import difflib
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
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from .budget_report import _parse_date
from .event_data import (
    EDIT_TOOLS,
    FILE_EVENT_PATH_KEYS,
    PATH_KEYS,
    is_edit,
    patch_paths,
    require_object_event_data,
    tool_arguments,
)
from .inert import inert
from .models import EventType, SessionMeta, TraceEvent, usable_text, usable_time
from .pairing import pair_tool_events
from .store import (
    AmbiguousSessionPrefixError,
    InvalidSessionIDError,
    TraceStore,
    _write_atomic,
)


COGNITIVE_DEBT_SCHEMA_VERSION = 1
COGNITIVE_DEBT_BADGE_SCHEMA_VERSION = 1


class CognitiveDebtInputError(ValueError):
    """The report cannot safely interpret trace or review evidence."""


class CognitiveDebtSessionNotFoundError(LookupError):
    """``--session`` names no session the store can read."""

    def __init__(self, session_id: str) -> None:
        super().__init__(f"Session not found: {session_id}")
        self.session_id = session_id


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


_REMOTE_URL = re.compile(
    r"^(?:[a-z][a-z0-9+.-]*://)?(?:[^@/]+@)?([^/:]+)(?::\d+)?[:/](.+?)(?:\.git)?/*$",
    re.IGNORECASE,
)


def _remote_identity(url: str) -> str:
    """Return a git remote URL as lower-case ``host/owner/repo``, or "" if it is not one."""
    match = _REMOTE_URL.match(url.strip())
    if not match:
        return ""
    return f"{match.group(1)}/{match.group(2).strip('/')}".lower()


def _remote_identities(cwd: str) -> set[str]:
    raw = _git(cwd, "config", "--get-regexp", r"^remote\..*\.url$")
    identities = set()
    for line in raw.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            identity = _remote_identity(parts[1])
            if identity:
                identities.add(identity)
    return identities


@dataclass(frozen=True)
class _Checkout:
    """Where git runs for one session.

    ``cwd`` is the directory git runs in. ``recorded_dir`` is the session's
    recorded working directory, or "". ``fallback`` is True when the recorded
    directory is missing and ``cwd`` is the current checkout of the same
    repository.
    """

    cwd: str
    recorded_dir: str
    fallback: bool = False


_COMMIT_ID = re.compile(r"^[0-9a-fA-F]{4,64}$")
_WINDOWS_DRIVE_PATH = re.compile(r"^[A-Za-z]:[\\/]")


def _local_file_path(path: str) -> str | None:
    """Return a local path for *path*, or None for an unsupported URI."""
    if _WINDOWS_DRIVE_PATH.match(path) or path.startswith("\\\\"):
        return path
    parsed = urllib.parse.urlsplit(path)
    if not parsed.scheme:
        return path
    if (
        parsed.scheme.lower() != "file"
        or parsed.query
        or parsed.fragment
        or parsed.netloc.lower() not in ("", "localhost")
    ):
        return None
    local = urllib.request.url2pathname(urllib.parse.unquote(parsed.path))
    return local or None


def _may_be_subdirectory(toplevel: str, recorded: str, commit: str) -> bool:
    """Return True if the recorded directory may have sat below its repository root.

    The trace does not record where ``working_dir`` sat in its repository. A
    trailing part of it (``pkg``, ``project/pkg``, ...) that is a directory in
    the current checkout, or in the session's start commit, is a place it may
    have been, so mapping it onto the checkout root would be a guess.
    """
    parts = Path(recorded).parts[1:]
    has_commit = bool(_COMMIT_ID.match(commit)) and bool(
        _git(toplevel, "rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}")
    )
    for count in range(1, len(parts) + 1):
        tail = parts[-count:]
        if os.path.isdir(os.path.join(toplevel, *tail)):
            return True
        if has_commit and _git(toplevel, "cat-file", "-t", f"{commit}:{'/'.join(tail)}") == "tree":
            return True
    return False


def _resolve_checkout(meta: SessionMeta) -> _Checkout:
    attr = _meta_attr(meta)
    recorded = str(attr.get("working_dir") or "")
    if not recorded:
        return _Checkout(os.getcwd(), "")
    if os.path.isdir(recorded) or not os.path.isabs(recorded):
        return _Checkout(recorded, recorded)
    expected = _remote_identity(str(attr.get("git_repo") or ""))
    toplevel = _git(os.getcwd(), "rev-parse", "--show-toplevel") if expected else ""
    if (
        toplevel
        and expected in _remote_identities(toplevel)
        and not _may_be_subdirectory(toplevel, recorded, str(attr.get("git_commit") or ""))
    ):
        return _Checkout(toplevel, recorded, fallback=True)
    return _Checkout(recorded, recorded)


def _is_under(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:
        return False


def _inside(path: str, root: str) -> bool:
    """Return True if *path* is *root* or below it, as written or with symlinks resolved."""
    def norm(value: str) -> str:
        return os.path.normcase(os.path.normpath(value))

    def real(value: str) -> str:
        return os.path.normcase(os.path.realpath(value))

    return _is_under(norm(path), norm(root)) or _is_under(real(path), real(root))


def _in_any_repository(path: str) -> bool:
    """Return True if *path* or a directory above it holds a ``.git`` entry."""
    current = os.path.dirname(os.path.abspath(path))
    while True:
        if os.path.lexists(os.path.join(current, ".git")):
            return True
        parent = os.path.dirname(current)
        if parent == current:
            return False
        current = parent


def _checkout_path(checkout: _Checkout, path: str) -> str | None:
    """Return *path* as git in ``checkout.cwd`` must see it.

    Local file URIs are converted before containment checks. Outside fallback
    mode the recorded path is otherwise used as is. In fallback mode a path
    under the recorded directory is moved to the same place in the current
    checkout. None means the path is unsupported or the current checkout has
    no such file.
    """
    path = _local_file_path(path)
    if path is None:
        return None
    if not checkout.fallback:
        return path
    try:
        rel = os.path.relpath(
            os.path.normpath(os.path.join(checkout.recorded_dir, path)),
            os.path.normpath(checkout.recorded_dir),
        )
    except ValueError:
        return None
    if rel in (os.curdir, os.pardir) or rel.startswith(os.pardir + os.sep) or os.path.isabs(rel):
        return None
    mapped = os.path.join(checkout.cwd, rel)
    return mapped if os.path.exists(mapped) else None


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
    match = re.search(r"branch:\s*([^,)]+)", usable_text(getattr(meta, "command", "")))
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


# Tool names that write files when they are the last part of an MCP tool name,
# such as mcp__filesystem__write_file. Single words such as "edit" or "create"
# are too common in other MCP tools to count there.
_MCP_WRITE_TOOLS = frozenset(name for name in EDIT_TOOLS if "_" in name) - {"delete_file"}
_MCP_SEPARATORS = ("__", "/", "-", ".")
# Tools that remove files. They write no lines.
_DELETE_TOOLS = frozenset({"delete_file"})
# Argument keys that hold the whole new text of a file or cell.
_FULL_CONTENT_KEYS = ("content", "file_text", "new_source", "code_edit", "text")
# (old text, new text) argument keys, one pair per tool family.
_EDIT_PAIRS = (
    ("old_string", "new_string"),
    ("old_str", "new_str"),
    ("old_text", "new_text"),
    ("oldText", "newText"),
)
# Argument keys that can hold apply_patch text.
_PATCH_TEXT_KEYS = ("command", "input", "patch")
# Event and argument keys that can hold a unified diff.
_DIFF_KEYS = ("diff", "patch")
_PATCH_SECTION = re.compile(r"^\*\*\* (Add File|Update File|Delete File|Move to): (.*)$")
# Above this many old plus new lines an edit counts every new line instead of
# running a line diff.
_MAX_DIFF_LINES = 20_000


def _tool_name(event: TraceEvent) -> str:
    tool_value = event.data.get("tool_name", "")
    if not isinstance(tool_value, str):
        raise CognitiveDebtInputError("write tool name must be a string")
    name = tool_value.lower()
    if name in EDIT_TOOLS or name == "str_replace":
        return name
    for separator in _MCP_SEPARATORS:
        if separator in name:
            last = name.rsplit(separator, 1)[1]
            if last in _MCP_WRITE_TOOLS:
                return last
    return name


def _write_arguments(event: TraceEvent) -> tuple[Mapping[str, Any], str | None]:
    for key in ("arguments", "tool_input"):
        value = event.data.get(key)
        if value is not None and not isinstance(value, (Mapping, str)):
            raise CognitiveDebtInputError("write tool arguments must be an object")
    return tool_arguments(event)


def _is_write_event(event: TraceEvent) -> bool:
    if event.event_type == EventType.FILE_WRITE:
        return True
    if event.event_type != EventType.TOOL_CALL:
        return False
    name = _tool_name(event)
    if not is_edit(event, name):
        return False
    return tool_arguments(event)[0].get("dryRun") is not True


def _event_path_value(data: Mapping[str, Any], keys: tuple[str, ...]) -> str:
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
        return _event_path_value(event.data, FILE_EVENT_PATH_KEYS)
    if event.event_type != EventType.TOOL_CALL:
        return ""
    if not _is_write_event(event):
        return ""
    args, _ = _write_arguments(event)
    return _event_path_value(args, PATH_KEYS)


def _added_lines(old: str, new: str) -> int:
    """Return how many lines of *new* a line diff from *old* adds or changes."""
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    if not old_lines or len(old_lines) + len(new_lines) > _MAX_DIFF_LINES:
        return len(new_lines)
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    return sum(
        j2 - j1
        for tag, _i1, _i2, j1, j2 in matcher.get_opcodes()
        if tag in ("replace", "insert")
    )


def _diff_added_lines(text: str) -> int:
    """Return the added lines in a unified diff, without ``+++`` file headers."""
    added = 0
    previous = ""
    for line in text.splitlines():
        if line.startswith("+") and not (line.startswith("+++") and previous.startswith("---")):
            added += 1
        previous = line
    return added


def _patch_sections(text: str) -> list[tuple[str, int, bool]]:
    """Return (path, added lines, deleted) for each file an apply_patch text names.

    A ``*** Move to:`` line renames the file of the section it is in. Every
    line in a section that starts with ``+`` is an added line.
    """
    sections: list[list[Any]] = []
    current: list[Any] | None = None
    for line in text.splitlines():
        match = _PATCH_SECTION.match(line)
        if match:
            kind, path = match.group(1), match.group(2).strip()
            if kind == "Move to":
                if current is not None and path:
                    current[0] = path
                continue
            current = [path, 0, kind == "Delete File"]
            sections.append(current)
            continue
        if current is not None and line.startswith("+"):
            current[1] += 1
    return [(path, added, deleted) for path, added, deleted in sections]


def _patch_text(args: Mapping[str, Any], raw: str | None) -> str | None:
    candidates = [args.get(key) for key in _PATCH_TEXT_KEYS]
    candidates.append(raw)
    for value in candidates:
        if isinstance(value, str) and patch_paths(value):
            return value
    return None


def _pair_lines(source: Mapping[str, Any]) -> int | None:
    """Return the lines an (old, new) edit adds, or None if *source* has no new text."""
    for old_key, new_key in _EDIT_PAIRS:
        new = source.get(new_key)
        if isinstance(new, str):
            old = source.get(old_key)
            return _added_lines(old if isinstance(old, str) else "", new)
    replacement = source.get("replacement")
    if isinstance(replacement, str):
        return len(replacement.splitlines())
    return None


def _content_lines(source: Mapping[str, Any]) -> int | None:
    """Return the lines a write's recorded content adds, or None if none is recorded."""
    edits = source.get("edits")
    if isinstance(edits, list):
        counts = [
            _pair_lines(item) for item in edits if isinstance(item, Mapping)
        ]
        known = [count for count in counts if count is not None]
        if known:
            return sum(known)
    lines = _pair_lines(source)
    if lines is not None:
        replacements = source.get("expected_replacements")
        if isinstance(replacements, int) and not isinstance(replacements, bool) and replacements > 1:
            lines *= replacements
        return lines
    for key in _FULL_CONTENT_KEYS:
        value = source.get(key)
        if isinstance(value, str):
            return len(value.splitlines())
    for key in _DIFF_KEYS:
        value = source.get(key)
        if isinstance(value, str) and value:
            return _diff_added_lines(value)
    return None


def _write_line_counts(event: TraceEvent) -> list[tuple[str, int]]:
    """Return (path, agent-written lines) for each file one write event changes.

    The path is "" when the event names no file. Every write counts at least
    one line, so a write whose content was not recorded still shows. Deleted
    files count no lines and are left out.
    """
    if event.event_type == EventType.FILE_WRITE:
        path = _path_from_event(event)
        return [(path, max(1, _content_lines(event.data) or 0))]
    name = _tool_name(event)
    args, raw = _write_arguments(event)
    patch = _patch_text(args, raw)
    if patch is not None:
        sections = _patch_sections(patch)
        return [(path, max(1, added)) for path, added, deleted in sections if not deleted]
    path = _event_path_value(args, PATH_KEYS)
    if name in _DELETE_TOOLS or args.get("edit_mode") == "delete":
        return []
    return [(path, max(1, _content_lines(args) or 0))]


def _failed_call_indexes(events: list[TraceEvent]) -> set[int]:
    """Return the indexes of tool calls whose linked result records a failure."""
    pairing = pair_tool_events(events)
    failed: set[int] = set()
    for response_index, call_index in pairing.pairs.items():
        response = events[response_index]
        data = response.data if isinstance(response.data, Mapping) else {}
        if (
            response.event_type == EventType.ERROR
            or data.get("is_error") is True
            or data.get("success") is False
        ):
            failed.add(call_index)
    return failed


# modified_file_lines reads the data of every event of these types.
MODIFIED_FILE_DATA_EVENT_TYPES = frozenset({EventType.FILE_WRITE, EventType.TOOL_CALL})


def modified_file_lines(events: list[TraceEvent]) -> dict[str, int]:
    files, _ = _modified_file_coverage(events)
    return files


@dataclass(frozen=True)
class _ModifiedFileStats:
    lines: int
    latest_write_at: float


def _modified_file_stats(
    events: list[TraceEvent],
) -> tuple[dict[str, _ModifiedFileStats], int]:
    files: dict[str, _ModifiedFileStats] = {}
    unknown_authorship_lines = 0
    failed = _failed_call_indexes(events)
    for index, event in enumerate(events):
        if not _is_write_event(event) or index in failed:
            continue
        timestamp = usable_time(event.timestamp)
        if timestamp is None:
            raise CognitiveDebtInputError("write event timestamp was unusable")
        for path, lines in _write_line_counts(event):
            if not path:
                unknown_authorship_lines += lines
                continue
            previous = files.get(path)
            files[path] = _ModifiedFileStats(
                lines=lines + (previous.lines if previous else 0),
                latest_write_at=max(timestamp, previous.latest_write_at)
                if previous
                else timestamp,
            )
    return files, unknown_authorship_lines


def _modified_file_coverage(events: list[TraceEvent]) -> tuple[dict[str, int], int]:
    files, unknown_authorship_lines = _modified_file_stats(events)
    return (
        {path: stats.lines for path, stats in files.items()},
        unknown_authorship_lines,
    )


def _git_available(cwd: str) -> bool:
    return bool(_git(cwd, "rev-parse", "--is-inside-work-tree"))


def _review_rev(cwd: str, branch: str) -> str | None:
    """Return the commit whose history holds the session's work.

    A known session branch must exist locally or on ``origin``. HEAD is used
    only when the session branch is absent or unknown. None means the required
    revision is unavailable.
    """
    if branch and branch not in ("HEAD", "(unknown)"):
        refs = (f"refs/heads/{branch}", f"refs/remotes/origin/{branch}")
    else:
        refs = ("HEAD",)
    for ref in refs:
        sha = _git(cwd, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
        if sha:
            return sha
    if (
        branch
        and branch not in ("HEAD", "(unknown)")
        and not _git(cwd, "rev-parse", "--verify", "--quiet", "HEAD^{commit}")
        and _git(cwd, "symbolic-ref", "--quiet", "--short", "HEAD") == branch
    ):
        return "(unborn)"
    return None


def _review_log(cwd: str, path: str, since_ts: float, rev: str | None = "HEAD") -> str:
    if rev is None:
        return ""
    since = f"@{max(0, int(since_ts) - 1)}"
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                cwd,
                "log",
                rev,
                f"--since={since}",
                "--format=%H%x00%ct%x00%ae%x00%an%x00%s%x00%b%x1e",
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


# Share of a file's agent-written lines that local git evidence counts as reviewed.
LOCAL_REVIEW_WEIGHTS = {
    "not_committed": 0.0,
    "committed_after_session": 0.0,
    "reviewed_by_trailer": 0.75,
}
_REVIEWED_BY = re.compile(r"^[ \t]*reviewed-by:[ \t]*(.*?)[ \t]*$", re.IGNORECASE | re.MULTILINE)
_TRAILER_EMAIL = re.compile(r"<([^<>]*)>")
_BARE_EMAIL = re.compile(r"[^\s<>@()\[\]{},;:'\"]+@[^\s<>@()\[\]{},;:'\"]+")
# A Reviewed-by trailer naming one of these is a bot or coding agent, not a person.
_AUTOMATED_REVIEWER_MARKERS = (
    "[bot]",
    "copilot",
    "claude",
    "anthropic",
    "openai",
    "codex",
    "cursor",
    "gemini",
    "devin",
)


def _names_other_person(reviewer: str, author_email: str, author_name: str) -> bool:
    """Return True if a Reviewed-by value names a person other than the commit author."""
    value = reviewer.strip().lower()
    if not value or any(marker in value for marker in _AUTOMATED_REVIEWER_MARKERS):
        return False
    email = _TRAILER_EMAIL.search(value)
    if email:
        address = email.group(1).strip()
        return bool(address) and address != author_email.strip().lower()
    bare_emails = [match.strip(".") for match in _BARE_EMAIL.findall(value)]
    if bare_emails:
        return author_email.strip().lower() not in bare_emails
    return value != author_name.strip().lower()


def _review_for_file(
    cwd: str,
    path: str,
    since_ts: float,
    agent_lines: int,
    rev: str | None = "HEAD",
) -> tuple[int, list[str], bool]:
    if not _git_available(cwd):
        return 0, ["git_unavailable"], False
    if rev == "(unborn)":
        return 0, ["not_committed"], True
    if rev is None:
        return 0, ["git_revision_unavailable"], False

    raw = _review_log(cwd, path, since_ts, rev)
    if not raw:
        return 0, ["not_committed"], True

    signals: set[str] = {"committed_after_session"}
    qualifying_commit = False
    for record in raw.split("\x1e"):
        if not record.strip():
            continue
        parts = record.strip().split("\x00", 5)
        if len(parts) != 6 or not parts[0] or not parts[1] or not parts[2]:
            raise CognitiveDebtInputError("local review history was malformed")
        _, commit_time_raw, author_email, author_name, _subject, body = parts
        try:
            commit_time = float(commit_time_raw)
        except ValueError as exc:
            raise CognitiveDebtInputError("local review history was malformed") from exc
        if commit_time <= since_ts:
            continue
        qualifying_commit = True
        if any(
            _names_other_person(match.group(1), author_email, author_name)
            for match in _REVIEWED_BY.finditer(body)
        ):
            signals.add("reviewed_by_trailer")
    if not qualifying_commit:
        return 0, ["not_committed"], True
    factor = max(LOCAL_REVIEW_WEIGHTS[signal] for signal in signals)
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


# Share of a file's agent-written lines that GitHub evidence counts as reviewed.
GITHUB_REVIEW_WEIGHTS = {
    "github_merged_pr": 0.0,
    "github_reviews": 0.75,
    "github_review_comments": 0.85,
    "github_line_comments": 1.0,
}


def _github_login(item: dict) -> tuple[str, bool]:
    """Return (login, is_bot) for the account that wrote a GitHub review or comment."""
    user = item.get("user")
    if not isinstance(user, dict):
        return "", False
    login = user.get("login")
    login = login if isinstance(login, str) else ""
    return login, user.get("type") == "Bot" or login.lower().endswith("[bot]")


def _by_other_person(items: list[dict], pr_author: str) -> list[dict]:
    """Return the reviews or comments written by a person other than the PR author."""
    kept = []
    for item in items:
        login, bot = _github_login(item)
        if not login or bot or login.lower() == pr_author.lower():
            continue
        kept.append(item)
    return kept


def _repo_relative_path(cwd: str, path: str) -> str:
    """Return *path* relative to the top of the repository that holds *cwd*, or ""."""
    toplevel = _git(cwd, "rev-parse", "--show-toplevel")
    if not toplevel:
        return ""
    full = path if os.path.isabs(path) else os.path.join(cwd, path)
    try:
        rel = os.path.relpath(os.path.realpath(full), os.path.realpath(toplevel))
    except ValueError:
        return ""
    if rel == os.pardir or rel.startswith(os.pardir + os.sep) or os.path.isabs(rel):
        return ""
    return rel.replace(os.sep, "/")


def _github_timestamp(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.timestamp()


def _github_review_for_file(
    cwd: str,
    branch: str,
    path: str,
    token: str,
    agent_lines: int,
    latest_write_at: float,
) -> tuple[int, list[str]]:
    repo = _github_repo(cwd)
    if not repo or not token or not branch or branch == "(unknown)":
        return 0, []
    owner, name = repo
    repo_path = _repo_relative_path(cwd, path)
    if not repo_path:
        return 0, []
    query = urllib.parse.urlencode({
        "q": f"repo:{owner}/{name} is:pr is:merged head:{branch}",
        "per_page": "100",
    })
    try:
        search = _github_json(f"/search/issues?{query}", token)
        if (
            not isinstance(search, dict)
            or search.get("incomplete_results") is not False
            or isinstance(search.get("total_count"), bool)
            or not isinstance(search.get("total_count"), int)
            or not isinstance(search.get("items"), list)
        ):
            raise CognitiveDebtInputError("GitHub review search response was malformed")
        items = search["items"]
        if search["total_count"] != len(items) or len(items) > 100:
            return 0, []
        if not items:
            return 0, []

        candidates: list[tuple[int, str]] = []
        seen_numbers: set[int] = set()
        for item in items:
            if not isinstance(item, dict):
                raise CognitiveDebtInputError("GitHub review search response was malformed")
            number = item.get("number")
            if (
                isinstance(number, bool)
                or not isinstance(number, int)
                or number <= 0
                or number in seen_numbers
            ):
                raise CognitiveDebtInputError("GitHub review search response was malformed")
            seen_numbers.add(number)
            detail = _github_json(f"/repos/{owner}/{name}/pulls/{number}", token)
            if not isinstance(detail, dict) or detail.get("number") != number:
                raise CognitiveDebtInputError("GitHub pull request response was malformed")
            base = detail.get("base")
            head = detail.get("head")
            base_repo = base.get("repo") if isinstance(base, dict) else None
            head_repo = head.get("repo") if isinstance(head, dict) else None
            full_name = base_repo.get("full_name") if isinstance(base_repo, dict) else None
            head_full_name = (
                head_repo.get("full_name") if isinstance(head_repo, dict) else None
            )
            head_ref = head.get("ref") if isinstance(head, dict) else None
            merged_at = _github_timestamp(detail.get("merged_at"))
            changed_files = detail.get("changed_files")
            pr_author, _ = _github_login(detail)
            if (
                not isinstance(full_name, str)
                or full_name.lower() != f"{owner}/{name}".lower()
                or not isinstance(head_full_name, str)
                or head_ref != branch
                or detail.get("merged") is not True
                or merged_at is None
                or isinstance(changed_files, bool)
                or not isinstance(changed_files, int)
                or changed_files < 0
                or not pr_author
            ):
                raise CognitiveDebtInputError("GitHub pull request response was malformed")
            if head_full_name.lower() != f"{owner}/{name}".lower():
                continue
            if merged_at <= latest_write_at:
                continue
            pr_files = _github_json(
                f"/repos/{owner}/{name}/pulls/{number}/files?per_page=100",
                token,
            )
            if (
                not isinstance(pr_files, list)
                or len(pr_files) != changed_files
                or any(not isinstance(item, dict) for item in pr_files)
            ):
                return 0, []
            filenames = [item.get("filename") for item in pr_files]
            if any(not isinstance(filename, str) for filename in filenames):
                raise CognitiveDebtInputError("GitHub pull request files response was malformed")
            if repo_path in filenames:
                candidates.append((number, pr_author))
        if len(candidates) != 1:
            return 0, []

        number, pr_author = candidates[0]
        reviews = _github_json(f"/repos/{owner}/{name}/pulls/{number}/reviews?per_page=100", token)
        if (
            not isinstance(reviews, list)
            or len(reviews) >= 100
            or any(not isinstance(item, dict) for item in reviews)
        ):
            raise CognitiveDebtInputError("GitHub reviews response was malformed")
        submitted = []
        for review in reviews:
            state = review.get("state")
            if not isinstance(state, str):
                raise CognitiveDebtInputError("GitHub reviews response was malformed")
            if state == "PENDING":
                continue
            submitted_at = _github_timestamp(review.get("submitted_at"))
            if submitted_at is None:
                raise CognitiveDebtInputError("GitHub reviews response was malformed")
            if submitted_at > latest_write_at:
                submitted.append(review)

        signals = {"github_merged_pr"}
        if _by_other_person(submitted, pr_author):
            signals.add("github_reviews")
        comments = _github_json(f"/repos/{owner}/{name}/pulls/{number}/comments?per_page=100", token)
        if (
            not isinstance(comments, list)
            or len(comments) >= 100
            or any(not isinstance(item, dict) for item in comments)
        ):
            raise CognitiveDebtInputError("GitHub review comments response was malformed")
        recent_comments = []
        for comment in comments:
            created_at = _github_timestamp(comment.get("created_at"))
            if created_at is None:
                raise CognitiveDebtInputError("GitHub review comments response was malformed")
            if created_at > latest_write_at:
                recent_comments.append(comment)
        comments = _by_other_person(recent_comments, pr_author)
        if comments:
            signals.add("github_review_comments")
            if any(comment.get("path") == repo_path for comment in comments):
                signals.add("github_line_comments")
        factor = max(GITHUB_REVIEW_WEIGHTS[signal] for signal in signals)
        return min(agent_lines, int(round(agent_lines * factor))), sorted(signals)
    except CognitiveDebtInputError:
        return 0, []
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError, ValueError):
        return 0, []
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
    checkout = _resolve_checkout(meta)
    cwd = checkout.cwd
    branch = _branch(meta, cwd)
    author = _author(meta, checkout.recorded_dir or cwd)
    repository = _repository(meta, cwd)
    git_ok = _git_available(cwd)
    rev = _review_rev(cwd, branch) if git_ok else None
    toplevel = _git(cwd, "rev-parse", "--show-toplevel") if git_ok else ""
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
    modified_files, unknown_authorship_lines = _modified_file_stats(events)
    total_written = unknown_authorship_lines
    total_reviewed = 0
    for path, stats in sorted(modified_files.items()):
        lines = stats.lines
        git_path = _checkout_path(checkout, path)
        outside = bool(
            git_path is not None
            and toplevel
            and not _inside(os.path.join(cwd, git_path), toplevel)
        )
        if outside and not _in_any_repository(os.path.join(cwd, git_path)):
            # A file in no git repository, such as an agent's plan file, is not
            # repository code.
            continue
        if outside:
            reviewed, signals, review_known = 0, ["outside_repository"], False
        elif git_path is None:
            reviewed, signals, review_known = 0, ["not_in_checkout"], False
        else:
            reviewed, signals, review_known = _review_for_file(
                cwd, git_path, stats.latest_write_at, lines, rev=rev
            )
        gh_reviewed, gh_signals = (
            (0, [])
            if outside or git_path is None
            else _github_review_for_file(
                cwd,
                branch,
                git_path,
                github_token,
                lines,
                stats.latest_write_at,
            )
        )
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
                # Refuse it as list_sessions_strict refuses one without
                # --session.
                if usable_time(metas[0].started_at) is None:
                    raise ValueError("started_at is not a usable time")
        else:
            # The report does not filter by tenant, so the tenant
            # administration check on untagged metadata does not apply.
            metas = [
                meta
                for meta in store.list_sessions_strict(strict_event_tenants=False)
                if window_start <= meta.started_at < window_end
            ]
    except (AmbiguousSessionPrefixError, InvalidSessionIDError):
        # Their messages name the argument, so the user can fix it.
        raise
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CognitiveDebtInputError("trace input could not be read safely") from exc
    if session_id and not metas:
        # session_exists is False for a missing session and for one the
        # store refuses to read.
        raise CognitiveDebtSessionNotFoundError(session_id)

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
            f"{session.session_id[:12]:<14}  {inert(session.branch[:20]):<20}  "
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
                f"  {inert(row['group'])}: {row['reviewed_lines']} reviewed, "
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
            w(f"  {inert(file_debt.path)} ({file_debt.agent_written_lines} lines, session {session.session_id[:12]})\n")

    unknown_review = report.unknown_review_files
    if unknown_review:
        w("\nFiles with unknown review state:\n")
        for session, file_debt in unknown_review[:10]:
            w(
                f"  {inert(file_debt.path)} "
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
    try:
        report = build_cognitive_debt_report(
            store,
            window_start=since,
            window_end=until,
            session_id=getattr(args, "session", "") or "",
            group_by=group_by,
            threshold=threshold,
            github_token=getattr(args, "github_token", "") or "",
        )
    except CognitiveDebtSessionNotFoundError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1

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
