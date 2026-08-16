"""Post agent session summaries to GitHub PRs or GitLab merge requests.

The module deliberately uses only the Python standard library.  Rendering and
session analysis are pure/local operations; the small HTTP boundary accepts an
optional opener so callers and tests can prevent or inspect external writes.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .cost import estimate_cost
from .explain import explain_session
from .lint import lint_session
from .models import EventType, SessionMeta, TraceEvent
from .store import TraceStore


COMMENT_MARKER = "<!-- agent-trace-pr-comment -->"
DEFAULT_GITHUB_API_URL = "https://api.github.com"
DEFAULT_GITLAB_API_URL = "https://gitlab.com/api/v4"
_HTTP_TIMEOUT = 15
_MAX_API_PAGES = 1000

_READ_TOOLS = {"read", "read_file", "file_read", "view"}
_WRITE_TOOLS = {
    "write", "write_file", "edit", "edit_file", "create", "multiedit",
    "replace", "str_replace", "str_replace_based_edit_tool", "notebook_edit",
}
_TEST_COMMAND_RE = re.compile(
    r"(?:^|(?:&&|;|\|\|)\s*)"
    r"(?:python\s+-m\s+(?:pytest|unittest)|pytest|tox|nox|go\s+test|"
    r"cargo\s+test|npm\s+(?:run\s+)?test|pnpm\s+(?:run\s+)?test|"
    r"yarn\s+(?:run\s+)?test)(?:\s|$)",
    re.IGNORECASE,
)


class PRCommentError(RuntimeError):
    """A safe, user-facing failure while resolving or posting a comment."""


@dataclass(frozen=True)
class ReviewTarget:
    """A GitHub pull request or GitLab merge request comment target."""

    platform: str
    project: str
    number: int
    api_url: str


@dataclass
class SessionSummary:
    """Reviewer-facing, non-sensitive metrics derived from one trace."""

    session_id: str
    duration_seconds: float
    cost_usd: float
    tool_calls: int
    models: list[str]
    status: str
    files_read: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    phases: list[str] = field(default_factory=list)
    test_runs: int = 0
    context_resets: int = 0
    lint_violations: int = 0
    lint_errors: int = 0
    lint_warnings: int = 0
    redundant_reads: dict[str, int] = field(default_factory=dict)
    errors: int = 0

    @property
    def redundant_read_count(self) -> int:
        return sum(max(0, count - 1) for count in self.redundant_reads.values())


def _path_from_event(event: TraceEvent) -> str:
    args = event.data.get("arguments", {})
    if not isinstance(args, dict):
        args = {}
    return str(
        args.get("file_path")
        or args.get("path")
        or event.data.get("file_path")
        or event.data.get("path")
        or event.data.get("uri")
        or ""
    ).strip()


def _file_activity(events: list[TraceEvent]) -> tuple[list[str], list[str], dict[str, int]]:
    read_counts: Counter[str] = Counter()
    modified: set[str] = set()

    for event in events:
        path = _path_from_event(event)
        if not path:
            continue
        if event.event_type == EventType.FILE_READ:
            read_counts[path] += 1
        elif event.event_type == EventType.FILE_WRITE:
            modified.add(path)
        elif event.event_type == EventType.TOOL_CALL:
            tool = str(event.data.get("tool_name", "")).lower()
            if tool in _READ_TOOLS:
                read_counts[path] += 1
            elif tool in _WRITE_TOOLS or "write" in tool or "edit" in tool:
                modified.add(path)

    redundant = {
        path: count for path, count in sorted(read_counts.items()) if count > 1
    }
    return sorted(read_counts), sorted(modified), redundant


def _models(events: list[TraceEvent]) -> list[str]:
    found: list[str] = []
    for event in events:
        for key in ("model", "model_id", "modelId", "model_name"):
            value = event.data.get(key)
            model = str(value).strip() if value else ""
            if model and model not in found:
                found.append(model)
                break
    return found


def _duration(meta: SessionMeta, events: list[TraceEvent]) -> float:
    if meta.total_duration_ms:
        return max(0.0, meta.total_duration_ms / 1000.0)
    if meta.ended_at is not None and meta.ended_at >= meta.started_at:
        return meta.ended_at - meta.started_at
    if len(events) >= 2:
        return max(0.0, events[-1].timestamp - events[0].timestamp)
    return 0.0


def _status(meta: SessionMeta, events: list[TraceEvent]) -> str:
    ended = meta.ended_at is not None
    failed = False
    for event in events:
        if event.event_type == EventType.SESSION_END:
            ended = True
            exit_code = event.data.get("exit_code")
            if exit_code not in (None, "", 0, "0"):
                failed = True
    if failed:
        return "failed"
    if ended:
        return "completed"
    return "in progress"


def _context_resets(events: list[TraceEvent]) -> int:
    resets = 0
    last_request: float | None = None
    for event in events:
        if event.event_type != EventType.LLM_REQUEST:
            continue
        if last_request is not None and event.timestamp - last_request > 120:
            resets += 1
        last_request = event.timestamp
    return resets


def _test_runs(events: list[TraceEvent]) -> int:
    count = 0
    for event in events:
        if event.event_type != EventType.TOOL_CALL:
            continue
        tool = str(event.data.get("tool_name", "")).lower()
        if tool not in {"bash", "shell", "run", "exec", "exec_command"}:
            continue
        args = event.data.get("arguments", {})
        if not isinstance(args, dict):
            continue
        command = str(args.get("command") or args.get("cmd") or "")
        if _TEST_COMMAND_RE.search(command):
            count += 1
    return count


def _categorical_phases(phases: list[Any]) -> list[str]:
    """Return fixed, non-sensitive labels for explain-session phases.

    ``explain_session`` names phases from user prompts.  Those labels are
    useful locally but must not be copied into an external PR comment.  Derive
    a small vocabulary from structural activity instead.
    """

    labels: list[str] = []

    def add(label: str) -> None:
        if label not in labels:
            labels.append(label)

    for phase in phases:
        events = list(getattr(phase, "events", []) or [])
        reads, writes, _redundant = _file_activity(events)
        test_runs = _test_runs(events)
        has_shell = any(
            event.event_type == EventType.TOOL_CALL
            and str(event.data.get("tool_name", "")).lower()
            in {"bash", "shell", "run", "exec", "exec_command"}
            for event in events
        )
        has_llm = any(
            event.event_type in {EventType.LLM_REQUEST, EventType.LLM_RESPONSE}
            for event in events
        )

        if reads:
            add("explore")
        if writes:
            add("implement")
        if test_runs:
            add("verify")
        elif has_shell:
            add("execute")
        if not reads and not writes and not has_shell and has_llm:
            add("plan")
        if getattr(phase, "failed", False):
            add("recover")

    return labels


def summarize_session(store: TraceStore, session_id: str) -> SessionSummary:
    """Build the structured PR summary for one stored session."""

    meta = store.load_meta(session_id)
    events = store.load_events(session_id)
    models = _models(events)
    files_read, files_modified, redundant = _file_activity(events)
    explanation = explain_session(store, session_id)
    lint = lint_session(store, session_id)
    non_redundant_findings = [
        finding for finding in lint.findings if finding.rule != "redundant-read"
    ]

    try:
        cost = estimate_cost(
            store,
            session_id,
            model=models[0] if models else "sonnet",
        ).total_cost
    except (OSError, ValueError, TypeError):
        cost = 0.0

    error_events = sum(1 for event in events if event.event_type == EventType.ERROR)
    tool_events = sum(1 for event in events if event.event_type == EventType.TOOL_CALL)

    return SessionSummary(
        session_id=session_id,
        duration_seconds=_duration(meta, events),
        cost_usd=cost,
        tool_calls=tool_events or meta.tool_calls,
        models=models,
        status=_status(meta, events),
        files_read=files_read,
        files_modified=files_modified,
        phases=_categorical_phases(explanation.phases),
        test_runs=_test_runs(events),
        context_resets=_context_resets(events),
        lint_violations=len(non_redundant_findings),
        lint_errors=sum(1 for finding in non_redundant_findings if finding.level == "ERROR"),
        lint_warnings=sum(1 for finding in non_redundant_findings if finding.level == "WARN"),
        redundant_reads=redundant,
        errors=max(meta.errors, error_events),
    )


def _git(cwd: str | Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def current_branch(cwd: str | Path = ".") -> str:
    """Return the source branch in CI or the current local Git branch."""

    return (
        os.environ.get("GITHUB_HEAD_REF", "")
        or os.environ.get("CI_MERGE_REQUEST_SOURCE_BRANCH_NAME", "")
        or os.environ.get("CI_COMMIT_REF_NAME", "")
        or _git(cwd, "rev-parse", "--abbrev-ref", "HEAD")
    )


def select_sessions(
    store: TraceStore,
    requested_session: str | None = None,
    *,
    branch: str = "",
    cwd: str | Path = ".",
) -> list[str]:
    """Resolve an explicit session or all sessions attributed to the branch.

    Older/imported traces can lack branch attribution.  For those, the session
    whose start time is closest to the current commit timestamp is selected,
    as specified by the issue, with the latest trace as a final fallback.
    """

    if requested_session:
        session_id = store.find_session(requested_session)
        if not session_id:
            raise PRCommentError(f"Session not found: {requested_session}")
        return [session_id]

    sessions = store.list_sessions()
    if not sessions:
        raise PRCommentError("No agent-trace sessions found")

    active_branch = branch or current_branch(cwd)
    if active_branch and active_branch != "HEAD":
        matching = [
            meta for meta in sessions
            if str((meta.attribution or {}).get("git_branch", "")) == active_branch
        ]
        if matching:
            return [meta.session_id for meta in reversed(matching)]

    commit_timestamp = _git(cwd, "log", "-1", "--format=%ct")
    try:
        timestamp = float(commit_timestamp)
    except (TypeError, ValueError):
        return [sessions[0].session_id]

    closest = min(
        sessions,
        key=lambda meta: (abs(meta.started_at - timestamp), -meta.started_at),
    )
    return [closest.session_id]


_INLINE_MARKDOWN_RE = re.compile(r"([\\*_[\]()~])")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")


def _normalise_display_text(value: Any) -> str:
    return str(value).replace("\r", " ").replace("\n", " ").strip()


def _safe_text(value: Any) -> str:
    """Escape trace-derived text for an inline GitHub/GitLab Markdown context."""

    normalised = _normalise_display_text(value)
    escaped = html.escape(normalised, quote=False)
    escaped = _INLINE_MARKDOWN_RE.sub(r"\\\1", escaped)
    # Prevent trace-derived text from creating bot-authored mentions or ending
    # an inline-code span used elsewhere in the comment.
    return escaped.replace("@", "&#64;").replace("`", "&#96;")


def _safe_code_text(value: Any) -> str:
    """Escape text that will be wrapped in a single-backtick code span."""

    return (
        html.escape(_normalise_display_text(value), quote=False)
        .replace("@", "&#64;")
        .replace("`", "&#96;")
    )


def _table_text(value: Any) -> str:
    return _safe_text(value).replace("|", "\\|").replace("\n", " ")


def _display_path(value: str) -> str:
    """Return a reviewer-useful path without workstation identity details."""

    raw = _normalise_display_text(value).replace("\\", "/")
    if raw.startswith("file://"):
        raw = urllib.parse.unquote(urllib.parse.urlsplit(raw).path)

    raw = _EMAIL_RE.sub("<email>", raw)
    home_prefix = re.match(
        r"^(?:[A-Za-z]:)?/(?:home|Users)/[^/]+/", raw, re.IGNORECASE
    )
    if home_prefix:
        raw = "…/" + raw[home_prefix.end():]
    username = (
        os.environ.get("USER", "")
        or os.environ.get("USERNAME", "")
        or os.environ.get("LOGNAME", "")
    )
    if username and len(username) >= 3:
        raw = re.sub(rf"(?<![A-Za-z0-9_.-]){re.escape(username)}(?![A-Za-z0-9_.-])", "<user>", raw)

    is_absolute = raw.startswith("/") or bool(re.match(r"^[A-Za-z]:/", raw))
    traverses_up = raw == ".." or raw.startswith("../") or "/../" in raw
    if is_absolute or traverses_up:
        parts = [part for part in raw.split("/") if part not in {"", ".", ".."}]
        # The final three components normally retain project directory + file
        # context while hiding usernames and machine-specific roots.
        raw = "…/" + "/".join(parts[-3:]) if parts else "…"
    return raw or "(unknown)"


def _format_path_preview(paths: list[str], limit: int = 8) -> str:
    shown = [f"`{_safe_code_text(_display_path(path))}`" for path in paths[:limit]]
    if len(paths) > limit:
        shown.append(f"+{len(paths) - limit} more")
    return ", ".join(shown)


def _format_duration(seconds: float) -> str:
    whole = max(0, int(round(seconds)))
    if whole < 60:
        return f"{whole}s"
    return f"{whole // 60}m {whole % 60:02d}s"


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    return singular if count == 1 else (plural or f"{singular}s")


def _safe_share_url(raw_url: str, session_id: str) -> str:
    value = raw_url.replace("{session_id}", urllib.parse.quote(session_id, safe=""))
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    path = urllib.parse.quote(parsed.path, safe="/%:@-._~!$&'()*+,;=")
    query = urllib.parse.quote_plus(parsed.query, safe="=&%:@-._~")
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, query, ""))


def render_comment(summaries: list[SessionSummary], share_url: str = "") -> str:
    """Render one idempotently identifiable Markdown PR comment."""

    if not summaries:
        raise ValueError("at least one session summary is required")

    lines = [COMMENT_MARKER, "## agent-trace session summary", ""]
    if len(summaries) == 1:
        summary = summaries[0]
        model = ", ".join(summary.models) if summary.models else "unknown"
        lines.extend([
            "| | |",
            "|---|---|",
            f"| **Session** | `{_table_text(summary.session_id)}` · {_format_duration(summary.duration_seconds)} |",
            f"| **Cost** | ${summary.cost_usd:.4f} ({summary.tool_calls} tool calls) |",
            f"| **Model** | {_table_text(model)} |",
            f"| **Status** | {_table_text(summary.status)} |",
        ])
    else:
        lines.extend([
            "| Session | Duration | Cost | Tool calls | Model | Status |",
            "|---|---:|---:|---:|---|---|",
        ])
        for summary in summaries:
            model = ", ".join(summary.models) if summary.models else "unknown"
            lines.append(
                f"| `{_table_text(summary.session_id)}` | "
                f"{_format_duration(summary.duration_seconds)} | "
                f"${summary.cost_usd:.4f} | {summary.tool_calls} | "
                f"{_table_text(model)} | {_table_text(summary.status)} |"
            )

    files_read = {path for summary in summaries for path in summary.files_read}
    files_modified = {path for summary in summaries for path in summary.files_modified}
    phases: list[str] = []
    for summary in summaries:
        for phase in summary.phases:
            if phase and phase not in phases:
                phases.append(phase)
    test_runs = sum(summary.test_runs for summary in summaries)
    context_resets = sum(summary.context_resets for summary in summaries)
    lines.extend([
        "",
        "### What the agent did",
        f"- Read {len(files_read)} {_plural(len(files_read), 'file')}, modified {len(files_modified)}",
    ])
    if files_modified:
        lines.append(f"- Modified: {_format_path_preview(sorted(files_modified))}")
    if phases:
        phase_text = " → ".join(_safe_text(phase) for phase in phases[:8])
        if len(phases) > 8:
            phase_text += f" → +{len(phases) - 8} more"
        test_text = f" ({test_runs} {_plural(test_runs, 'test run')})" if test_runs else ""
        lines.append(f"- Phases: {phase_text}{test_text}")
    if context_resets:
        lines.append(
            f"- {context_resets} context {_plural(context_resets, 'reset')} detected"
        )

    redundant_total = sum(summary.redundant_read_count for summary in summaries)
    redundant_files: Counter[str] = Counter()
    for summary in summaries:
        for path, count in summary.redundant_reads.items():
            redundant_files[path] += count
    lint_count = sum(summary.lint_violations for summary in summaries)
    lint_errors = sum(summary.lint_errors for summary in summaries)
    lint_warnings = sum(summary.lint_warnings for summary in summaries)
    error_count = sum(summary.errors for summary in summaries)

    lines.extend(["", "### Flags"])
    if redundant_total:
        detail = ", ".join(
            f"`{_safe_code_text(_display_path(path))}` read {count}×"
            for path, count in redundant_files.most_common(3)
        )
        lines.append(
            f"- ⚠️ {redundant_total} redundant {_plural(redundant_total, 'read')} ({detail})"
        )
    else:
        lines.append("- ✅ No redundant reads")
    if lint_count:
        severity = []
        if lint_errors:
            severity.append(f"{lint_errors} {_plural(lint_errors, 'error')}")
        if lint_warnings:
            severity.append(f"{lint_warnings} {_plural(lint_warnings, 'warning')}")
        suffix = f" ({', '.join(severity)})" if severity else ""
        lines.append(
            f"- ⚠️ {lint_count} lint {_plural(lint_count, 'violation')}{suffix}"
        )
    else:
        lines.append("- ✅ No lint violations")
    if error_count:
        lines.append(f"- ❌ {error_count} {_plural(error_count, 'error')}")
    else:
        lines.append("- ✅ No errors")

    lines.extend(["", "<details>", "<summary>Replay this session</summary>", "", "```bash"])
    for summary in summaries:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", summary.session_id)
        lines.append(f"agent-strace replay {safe_id}")
    lines.extend(["```", ""])

    links = [
        _safe_share_url(share_url, summary.session_id)
        for summary in summaries
        if share_url
    ]
    links = [link for link in links if link]
    if len(links) == 1:
        lines.append(f"Or [view the HTML replay](<{links[0]}>)")
    elif links:
        lines.append("HTML replays: " + " · ".join(f"[session {index + 1}](<{link}>)" for index, link in enumerate(links)))
    lines.extend(["", "</details>", ""])
    return "\n".join(lines)


def _validate_api_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value.rstrip("/"))
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise PRCommentError("API URL must be an absolute HTTP(S) URL")
    return value.rstrip("/")


def _request_json(
    method: str,
    url: str,
    token: str,
    *,
    platform: str,
    payload: dict[str, Any] | None = None,
    opener: Callable[..., Any] | None = None,
    token_header: str = "",
) -> Any:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {
        "Accept": "application/vnd.github+json" if platform == "github" else "application/json",
        "User-Agent": "agent-strace-pr-comment",
    }
    if data is not None:
        headers["Content-Type"] = "application/json"
    if token:
        if platform == "github":
            headers["Authorization"] = f"Bearer {token}"
            headers["X-GitHub-Api-Version"] = "2022-11-28"
        else:
            headers[token_header or "PRIVATE-TOKEN"] = token

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    open_url = opener or urllib.request.urlopen
    try:
        with open_url(request, timeout=_HTTP_TIMEOUT) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raise PRCommentError(
            f"{platform.title()} API request failed with HTTP {exc.code}: {exc.reason}"
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise PRCommentError(f"{platform.title()} API request failed: {exc}") from exc

    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PRCommentError(f"{platform.title()} API returned invalid JSON") from exc


def _find_marker_comment(
    list_url: str,
    token: str,
    *,
    platform: str,
    opener: Callable[..., Any] | None = None,
    token_header: str = "",
) -> dict[str, Any] | None:
    """Find the marker across every API page without silently duplicating it."""

    separator = "&" if "?" in list_url else "?"
    for page in range(1, _MAX_API_PAGES + 1):
        items = _request_json(
            "GET",
            f"{list_url}{separator}page={page}",
            token,
            platform=platform,
            opener=opener,
            token_header=token_header,
        )
        if not isinstance(items, list):
            raise PRCommentError(
                f"{platform.title()} API returned an invalid comment list"
            )
        existing = next(
            (
                item for item in items
                if isinstance(item, dict)
                and COMMENT_MARKER in str(item.get("body", ""))
            ),
            None,
        )
        if existing is not None:
            return existing
        if len(items) < 100:
            return None

    raise PRCommentError(
        f"{platform.title()} comment list exceeded {_MAX_API_PAGES} pages"
    )


def _repo_from_remote(remote: str) -> str:
    value = remote.strip().rstrip("/")
    if not value:
        return ""
    if "://" in value:
        path = urllib.parse.urlsplit(value).path
    elif re.match(r"^[^/]+@[^:]+:", value):
        path = value.split(":", 1)[1]
    else:
        path = value
    return path.strip("/").removesuffix(".git")


def _remote_project(cwd: str | Path, remote_name: str) -> str:
    if not remote_name or remote_name == ".":
        return ""
    return _repo_from_remote(_git(cwd, "remote", "get-url", remote_name))


def _github_head_owner(
    event: dict[str, Any],
    project: str,
    branch: str,
    cwd: str | Path,
) -> str:
    """Resolve the account that owns the PR head branch, including forks."""

    pull_request = event.get("pull_request") or {}
    head = pull_request.get("head") or {}
    head_repo = head.get("repo") or {}
    head_owner = head_repo.get("owner") or head.get("user") or {}
    event_owner = head_owner.get("login") if isinstance(head_owner, dict) else ""
    env_head = os.environ.get("GITHUB_HEAD_REPOSITORY", "")
    env_owner = env_head.split("/", 1)[0] if "/" in env_head else ""

    tracked_remote = _git(cwd, "config", "--get", f"branch.{branch}.remote")
    tracked_project = _remote_project(cwd, tracked_remote)
    tracked_owner = tracked_project.split("/", 1)[0] if "/" in tracked_project else ""

    return str(event_owner or env_owner or tracked_owner or project.split("/", 1)[0])


def _valid_project(project: str, platform: str) -> bool:
    segments = project.split("/")
    expected = len(segments) == 2 if platform == "github" else len(segments) >= 2
    return expected and all(re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in segments)


def _github_event() -> dict[str, Any]:
    path = os.environ.get("GITHUB_EVENT_PATH", "")
    if not path:
        return {}
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _event_number(event: dict[str, Any]) -> int:
    value = (event.get("pull_request") or {}).get("number") or event.get("number")
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return number if number > 0 else 0


def resolve_target(
    platform: str,
    token: str,
    *,
    project: str = "",
    number: int | str | None = None,
    api_url: str = "",
    branch: str = "",
    cwd: str | Path = ".",
    opener: Callable[..., Any] | None = None,
    token_header: str = "",
) -> ReviewTarget:
    """Resolve repository/project and open PR/MR number from CI or Git."""

    if platform not in {"github", "gitlab"}:
        raise PRCommentError(f"Unsupported platform: {platform}")

    event = _github_event() if platform == "github" else {}
    event_repo = ((event.get("repository") or {}).get("full_name") or "")
    origin_project = _remote_project(cwd, "origin")
    upstream_project = _remote_project(cwd, "upstream") if platform == "github" else ""
    inferred_project = (
        project
        or (os.environ.get("GITHUB_REPOSITORY", "") if platform == "github" else os.environ.get("CI_PROJECT_PATH", ""))
        or event_repo
        or upstream_project
        or origin_project
    )
    if not _valid_project(inferred_project, platform):
        raise PRCommentError(
            "Could not determine a valid repository; pass --repo OWNER/REPO"
            if platform == "github"
            else "Could not determine a valid project; pass --repo GROUP/PROJECT"
        )

    base_url = _validate_api_url(
        api_url
        or (os.environ.get("GITHUB_API_URL", "") if platform == "github" else os.environ.get("CI_API_V4_URL", ""))
        or (DEFAULT_GITHUB_API_URL if platform == "github" else DEFAULT_GITLAB_API_URL)
    )

    raw_number = number
    if not raw_number:
        raw_number = (
            _event_number(event)
            if platform == "github"
            else os.environ.get("CI_MERGE_REQUEST_IID", "")
        )
    try:
        resolved_number = int(raw_number or 0)
    except (TypeError, ValueError):
        resolved_number = 0

    if resolved_number <= 0:
        source_branch = branch or current_branch(cwd)
        if not source_branch or source_branch == "HEAD":
            raise PRCommentError("Could not determine the current source branch")
        if platform == "github":
            owner = _github_head_owner(
                event, inferred_project, source_branch, cwd
            )
            query = urllib.parse.urlencode({
                "state": "open",
                "head": f"{owner}:{source_branch}",
                "per_page": 100,
            })
            url = f"{base_url}/repos/{inferred_project}/pulls?{query}"
        else:
            encoded_project = urllib.parse.quote(inferred_project, safe="")
            query = urllib.parse.urlencode({
                "state": "opened",
                "source_branch": source_branch,
                "per_page": 100,
            })
            url = f"{base_url}/projects/{encoded_project}/merge_requests?{query}"
        matches = _request_json(
            "GET", url, token, platform=platform, opener=opener, token_header=token_header
        )
        if not isinstance(matches, list) or not matches:
            noun = "pull request" if platform == "github" else "merge request"
            raise PRCommentError(f"No open {noun} found for branch {source_branch!r}")
        key = "number" if platform == "github" else "iid"
        try:
            resolved_number = int(matches[0][key])
        except (KeyError, TypeError, ValueError) as exc:
            raise PRCommentError(f"{platform.title()} API returned an invalid review number") from exc

    return ReviewTarget(platform, inferred_project, resolved_number, base_url)


def post_or_update_comment(
    target: ReviewTarget,
    body: str,
    token: str,
    *,
    opener: Callable[..., Any] | None = None,
    token_header: str = "",
) -> str:
    """Create the marker comment, or update it when it already exists."""

    if not token:
        raise PRCommentError(f"A {target.platform.title()} token is required to post comments")
    if COMMENT_MARKER not in body:
        raise ValueError("comment body is missing the idempotency marker")

    if target.platform == "github":
        comments_url = (
            f"{target.api_url}/repos/{target.project}/issues/"
            f"{target.number}/comments?per_page=100"
        )
        existing = _find_marker_comment(
            comments_url, token, platform="github", opener=opener
        )
        if existing:
            comment_id = int(existing["id"])
            url = f"{target.api_url}/repos/{target.project}/issues/comments/{comment_id}"
            method = "PATCH"
            action = "updated"
        else:
            url = comments_url.split("?", 1)[0]
            method = "POST"
            action = "created"
        _request_json(
            method, url, token, platform="github", payload={"body": body}, opener=opener
        )
        return action

    if target.platform == "gitlab":
        encoded_project = urllib.parse.quote(target.project, safe="")
        notes_url = (
            f"{target.api_url}/projects/{encoded_project}/merge_requests/"
            f"{target.number}/notes?per_page=100"
        )
        existing = _find_marker_comment(
            notes_url, token, platform="gitlab", opener=opener,
            token_header=token_header,
        )
        if existing:
            note_id = int(existing["id"])
            url = notes_url.split("?", 1)[0] + f"/{note_id}"
            method = "PUT"
            action = "updated"
        else:
            url = notes_url.split("?", 1)[0]
            method = "POST"
            action = "created"
        _request_json(
            method, url, token, platform="gitlab", payload={"body": body},
            opener=opener, token_header=token_header,
        )
        return action

    raise PRCommentError(f"Unsupported platform: {target.platform}")


def _platform_token(platform: str) -> tuple[str, str]:
    if platform == "github":
        return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN", ""), ""
    # CI_JOB_TOKEN can read MR notes but GitLab does not permit it to create or
    # update them. Posting requires a personal/project access token.
    private_token = (
        os.environ.get("GITLAB_TOKEN")
        or os.environ.get("GITLAB_ACCESS_TOKEN")
        or os.environ.get("PRIVATE_TOKEN", "")
    )
    return private_token, "PRIVATE-TOKEN"


def cmd_pr_comment(args: argparse.Namespace) -> int:
    """CLI handler for ``agent-strace pr-comment``."""

    store = TraceStore(args.trace_dir)
    platform = str(getattr(args, "platform", "github") or "github").lower()
    requested = getattr(args, "session_id", None)
    branch = current_branch()

    try:
        session_ids = select_sessions(store, requested, branch=branch)
        summaries = [summarize_session(store, session_id) for session_id in session_ids]
        share_url = (
            getattr(args, "share_url", "")
            or os.environ.get("AGENT_STRACE_REPLAY_URL", "")
        )
        body = render_comment(summaries, share_url=share_url)

        if getattr(args, "dry_run", False):
            sys.stdout.write(body)
            return 0

        token, token_header = _platform_token(platform)
        if not token:
            names = (
                "GITHUB_TOKEN or GH_TOKEN"
                if platform == "github"
                else "GITLAB_TOKEN, GITLAB_ACCESS_TOKEN, or PRIVATE_TOKEN "
                "(personal/project access token)"
            )
            raise PRCommentError(f"Set {names} to post the comment")

        target = resolve_target(
            platform,
            token,
            project=getattr(args, "repo", "") or "",
            number=getattr(args, "pr", None),
            api_url=getattr(args, "api_url", "") or "",
            branch=branch,
            token_header=token_header,
        )
        action = post_or_update_comment(
            target, body, token, token_header=token_header
        )
    except (PRCommentError, OSError, ValueError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"pr-comment: {exc}\n")
        return 1

    noun = "PR" if platform == "github" else "MR"
    sys.stdout.write(f"{action.title()} agent-trace comment on {noun} #{target.number}.\n")
    return 0


__all__ = [
    "COMMENT_MARKER",
    "PRCommentError",
    "ReviewTarget",
    "SessionSummary",
    "cmd_pr_comment",
    "current_branch",
    "post_or_update_comment",
    "render_comment",
    "resolve_target",
    "select_sessions",
    "summarize_session",
]
