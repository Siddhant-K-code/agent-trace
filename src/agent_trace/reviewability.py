"""Pull-request reviewability check (``agent-strace review``).

Reports what the captured authoring sessions show about a branch: rationale,
scope, validation, stewardship, and evidence health. Each finding has a
status, an evidence source, and a confidence. A repository threshold decides
which findings fail the check.

The output is deterministic. It never prints prompt text, command strings, or
tool output. It reports counts, fixed labels, and display-safe file paths.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence, TextIO

from .approval import list_requests
from .event_data import NonObjectEventDataError, require_object_event_data
from .health import CaptureMatrix, assess_session, detect_provider, load_capture_matrix
from .models import EventType, SessionMeta, TraceEvent
from .pr_comment import (
    PRCommentError,
    _display_path,
    _file_activity,
    _path_from_event,
    _safe_code_text,
    _table_text,
    current_branch,
    select_sessions_with_link,
)
from .project_budget import _parse_simple_yaml
from .store import TraceStore

SCHEMA_VERSION = 1

STATUSES = ("present", "partial", "missing", "inconclusive")
CONFIDENCES = ("low", "medium", "high")
FINDINGS = ("rationale", "scope", "validation", "stewardship", "evidence")
FINDING_LABELS = {
    "rationale": "Rationale",
    "scope": "Scope",
    "validation": "Validation",
    "stewardship": "Stewardship",
    "evidence": "Evidence health",
}
LINK_LABELS = {
    "explicit": "named on the command line",
    "branch": "session git branch matches the PR branch",
    "commit_time": "closest session start to the head commit time (no branch match)",
    "latest": "latest session (no branch or commit to match)",
    "none": "no session linked",
}
DEFAULT_CONFIG_PATHS = (".agent-strace.yaml", ".agent-strace.yml", ".agent-strace.json")
DOCS_URL = "https://github.com/Siddhant-K-code/agent-trace/blob/main/docs/review.md"

_STRONG_LINKS = frozenset({"explicit", "branch"})
_SHELL_TOOLS = frozenset({
    "bash", "shell", "run", "exec", "exec_command", "execute_command",
    "terminal", "run_command", "run_in_terminal", "powershell",
})
_EXIT_CODE_RE = re.compile(r"exit(?:ed with)?[ _]code['\"]?\s*[:=]?\s*(-?\d+)", re.IGNORECASE)
_SEGMENT_SPLIT_RE = re.compile(r"&&|\|\||;|\|")
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_PERMISSION_MODE_RE = re.compile(r"^[A-Za-z][A-Za-z_-]{0,31}$")
_PATH_PREVIEW = 5

# Each entry is (token prefix, label, kind). The label is a fixed string, so
# the report never repeats a recorded command.
_VALIDATION_PREFIXES: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("pytest",), "pytest", "test"),
    (("py.test",), "pytest", "test"),
    (("python", "-m", "pytest"), "pytest", "test"),
    (("python3", "-m", "pytest"), "pytest", "test"),
    (("python", "-m", "unittest"), "unittest", "test"),
    (("python3", "-m", "unittest"), "unittest", "test"),
    (("tox",), "tox", "test"),
    (("nox",), "nox", "test"),
    (("go", "test"), "go test", "test"),
    (("cargo", "test"), "cargo test", "test"),
    (("cargo", "nextest"), "cargo nextest", "test"),
    (("npm", "test"), "npm test", "test"),
    (("npm", "run", "test"), "npm test", "test"),
    (("pnpm", "test"), "pnpm test", "test"),
    (("pnpm", "run", "test"), "pnpm test", "test"),
    (("yarn", "test"), "yarn test", "test"),
    (("yarn", "run", "test"), "yarn test", "test"),
    (("bun", "test"), "bun test", "test"),
    (("deno", "test"), "deno test", "test"),
    (("jest",), "jest", "test"),
    (("vitest",), "vitest", "test"),
    (("mocha",), "mocha", "test"),
    (("rspec",), "rspec", "test"),
    (("phpunit",), "phpunit", "test"),
    (("mvn", "test"), "mvn test", "test"),
    (("mvn", "verify"), "mvn verify", "test"),
    (("gradle", "test"), "gradle test", "test"),
    (("./gradlew", "test"), "gradle test", "test"),
    (("dotnet", "test"), "dotnet test", "test"),
    (("swift", "test"), "swift test", "test"),
    (("mix", "test"), "mix test", "test"),
    (("make", "test"), "make test", "test"),
    (("ruff",), "ruff", "check"),
    (("flake8",), "flake8", "check"),
    (("pylint",), "pylint", "check"),
    (("mypy",), "mypy", "check"),
    (("pyright",), "pyright", "check"),
    (("black", "--check"), "black --check", "check"),
    (("eslint",), "eslint", "check"),
    (("tsc",), "tsc", "check"),
    (("prettier", "--check"), "prettier --check", "check"),
    (("golangci-lint",), "golangci-lint", "check"),
    (("go", "vet"), "go vet", "check"),
    (("go", "build"), "go build", "check"),
    (("cargo", "clippy"), "cargo clippy", "check"),
    (("cargo", "check"), "cargo check", "check"),
    (("cargo", "build"), "cargo build", "check"),
    (("npm", "run", "lint"), "npm run lint", "check"),
    (("npm", "run", "typecheck"), "npm run typecheck", "check"),
    (("npm", "run", "build"), "npm run build", "check"),
    (("pnpm", "lint"), "pnpm lint", "check"),
    (("pnpm", "run", "lint"), "pnpm lint", "check"),
    (("pnpm", "build"), "pnpm build", "check"),
    (("pnpm", "run", "build"), "pnpm build", "check"),
    (("yarn", "lint"), "yarn lint", "check"),
    (("yarn", "build"), "yarn build", "check"),
    (("make", "lint"), "make lint", "check"),
    (("make", "check"), "make check", "check"),
    (("shellcheck",), "shellcheck", "check"),
    (("dotnet", "build"), "dotnet build", "check"),
)
_WRAPPERS: tuple[tuple[str, ...], ...] = (
    ("uv", "run"), ("poetry", "run"), ("pipenv", "run"), ("hatch", "run"),
    ("bundle", "exec"), ("npx",), ("bunx",), ("pnpm", "exec"), ("yarn", "exec"),
)


class ReviewConfigError(ValueError):
    """The review threshold in a flag or config file is not valid."""


@dataclass(frozen=True)
class Finding:
    """One reviewability finding."""

    name: str
    status: str
    confidence: str
    source: str
    detail: str
    facts: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return FINDING_LABELS[self.name]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "confidence": self.confidence,
            "source": self.source,
            "detail": self.detail,
            "facts": self.facts,
        }


@dataclass(frozen=True)
class Threshold:
    """Which findings fail the check."""

    fail_on: tuple[str, ...] = ("missing",)
    min_confidence: str = "medium"
    findings: tuple[str, ...] = FINDINGS
    source: str = "default"

    def fails(self, finding: Finding) -> bool:
        return (
            finding.name in self.findings
            and finding.status in self.fail_on
            and CONFIDENCES.index(finding.confidence) >= CONFIDENCES.index(self.min_confidence)
        )

    def describe(self) -> str:
        if not self.fail_on or not self.findings:
            return "never fails"
        statuses = " or ".join(f"`{status}`" for status in self.fail_on)
        text = f"fails on {statuses} at `{self.min_confidence}` or higher confidence"
        if tuple(self.findings) != FINDINGS:
            text += " for " + ", ".join(self.findings)
        return text

    def to_dict(self) -> dict[str, Any]:
        return {
            "fail_on": list(self.fail_on),
            "min_confidence": self.min_confidence,
            "findings": list(self.findings),
            "source": self.source,
        }


@dataclass(frozen=True)
class LinkedSession:
    session_id: str
    link: str


@dataclass(frozen=True)
class ReviewReport:
    """The reviewability report for the sessions linked to a branch."""

    branch: str
    sessions: tuple[LinkedSession, ...]
    findings: tuple[Finding, ...]
    threshold: Threshold

    @property
    def failing(self) -> tuple[Finding, ...]:
        return tuple(finding for finding in self.findings if self.threshold.fails(finding))

    @property
    def passed(self) -> bool:
        return not self.failing

    @property
    def exit_code(self) -> int:
        return 0 if self.passed else 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "result": "pass" if self.passed else "fail",
            "branch": self.branch or None,
            "sessions": [
                {"session_id": linked.session_id, "link": linked.link} for linked in self.sessions
            ],
            "threshold": self.threshold.to_dict(),
            "failing": [finding.name for finding in self.failing],
            "findings": [finding.to_dict() for finding in self.findings],
        }


# ---------------------------------------------------------------------------
# Threshold configuration
# ---------------------------------------------------------------------------

def _split_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items = [str(item) for item in value]
    else:
        items = str(value).strip().strip("[]").split(",")
    return [item.strip().strip("'\"").lower() for item in items if item.strip()]


def _parse_fail_on(value: Any) -> tuple[str, ...]:
    values = _split_values(value)
    if values == ["none"]:
        return ()
    for item in values:
        if item not in STATUSES or item == "present":
            raise ReviewConfigError(
                f"invalid fail_on value {item!r}; use missing, partial, inconclusive, or none"
            )
    return tuple(status for status in STATUSES if status in values)


def _parse_confidence(value: Any) -> str:
    text = str(value).strip().lower()
    if text not in CONFIDENCES:
        raise ReviewConfigError(f"invalid min_confidence {value!r}; use low, medium, or high")
    return text


def _parse_findings(value: Any) -> tuple[str, ...]:
    values = _split_values(value)
    for item in values:
        if item not in FINDINGS:
            raise ReviewConfigError(
                f"invalid finding {item!r}; use {', '.join(FINDINGS)}"
            )
    return tuple(name for name in FINDINGS if name in values)


def _read_review_block(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = _parse_simple_yaml(text)
    if not isinstance(data, dict):
        return {}
    block = data.get("review")
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise ReviewConfigError(f"{path}: 'review' must be a mapping")
    return block


def load_threshold(
    config_path: str | None = None,
    *,
    fail_on: str | None = None,
    min_confidence: str | None = None,
    findings: str | None = None,
) -> Threshold:
    """Build the threshold from defaults, the config file, then flags.

    Flags override the ``review`` block of ``.agent-strace.yaml``. A config
    path given explicitly must exist.
    """
    threshold = Threshold()
    values: dict[str, Any] = {}
    source = "default"

    if config_path:
        path = Path(config_path)
        if not path.is_file():
            raise ReviewConfigError(f"config file not found: {config_path}")
        candidates = [path]
    else:
        candidates = [Path(name) for name in DEFAULT_CONFIG_PATHS if Path(name).is_file()][:1]
    for path in candidates:
        try:
            values = _read_review_block(path)
        except OSError as exc:
            raise ReviewConfigError(f"could not read {path}: {exc}") from exc
        if values:
            source = str(path)

    if fail_on is not None:
        values["fail_on"] = fail_on
    if min_confidence is not None:
        values["min_confidence"] = min_confidence
    if findings is not None:
        values["findings"] = findings
    if any(flag is not None for flag in (fail_on, min_confidence, findings)):
        source = "flags" if source == "default" else f"{source} + flags"

    unknown = set(values) - {"fail_on", "min_confidence", "findings"}
    if unknown:
        raise ReviewConfigError(
            f"unknown review setting {sorted(unknown)[0]!r}; "
            "use fail_on, min_confidence, or findings"
        )
    return Threshold(
        fail_on=_parse_fail_on(values["fail_on"]) if "fail_on" in values else threshold.fail_on,
        min_confidence=(
            _parse_confidence(values["min_confidence"])
            if "min_confidence" in values
            else threshold.min_confidence
        ),
        findings=_parse_findings(values["findings"]) if "findings" in values else threshold.findings,
        source=source,
    )


# ---------------------------------------------------------------------------
# Session evidence
# ---------------------------------------------------------------------------

@dataclass
class _Session:
    session_id: str
    link: str
    meta: SessionMeta
    events: list[TraceEvent]
    provider: str
    matrix_entry: dict[str, Any] | None

    @property
    def is_subagent(self) -> bool:
        return bool(self.meta.parent_session_id) or self.meta.depth > 0

    def captures(self, capability: str) -> bool:
        entry = self.matrix_entry
        return entry is not None and entry["coverage"].get(capability) == "captured"


def _session_cwd(events: Sequence[TraceEvent]) -> str:
    for event in events:
        if event.event_type == EventType.SESSION_START:
            cwd = event.data.get("cwd")
            if isinstance(cwd, str) and cwd.strip():
                return cwd.strip().replace("\\", "/").rstrip("/")
    return ""


def _normalise_path(path: str, cwd: str) -> str:
    value = path.strip().replace("\\", "/")
    if cwd and value.startswith(cwd + "/"):
        value = value[len(cwd) + 1:]
    while value.startswith("./"):
        value = value[2:]
    return value


def _same_file(left: str, right: str) -> bool:
    return left == right or left.endswith("/" + right) or right.endswith("/" + left)


def _named_in(path: str, texts: Sequence[str]) -> bool:
    candidates = {path}
    base = path.rsplit("/", 1)[-1]
    if "." in base and len(base) >= 4:
        candidates.add(base)
    for candidate in candidates:
        pattern = re.compile(r"(?<![\w.-])" + re.escape(candidate) + r"(?![\w-])")
        if any(pattern.search(text) for text in texts):
            return True
    return False


def _prompt_text(event: TraceEvent) -> str:
    value = event.data.get("prompt")
    return value if isinstance(value, str) else ""


def _command_text(event: TraceEvent) -> str:
    args = event.data.get("arguments")
    if not isinstance(args, dict):
        args = event.data.get("tool_input")
    if not isinstance(args, dict):
        args = {}
    command = args.get("command") or args.get("cmd") or event.data.get("command") or ""
    if isinstance(command, list):
        command = " ".join(str(part) for part in command)
    return command if isinstance(command, str) else ""


def _is_shell_call(event: TraceEvent) -> bool:
    if event.event_type != EventType.TOOL_CALL:
        return False
    tool = str(event.data.get("tool_name") or "").lower()
    return tool in _SHELL_TOOLS or bool(_command_text(event))


def _tokens(segment: str) -> list[str]:
    try:
        tokens = shlex.split(segment)
    except ValueError:
        tokens = segment.split()
    while tokens and _ENV_ASSIGN_RE.match(tokens[0]):
        tokens.pop(0)
    changed = True
    while changed and tokens:
        changed = False
        for wrapper in _WRAPPERS:
            if tuple(token.lower() for token in tokens[: len(wrapper)]) == wrapper:
                tokens = tokens[len(wrapper):]
                changed = True
                break
    return tokens


def classify_validation(command: str) -> list[tuple[str, str]]:
    """Return ``(label, kind)`` for each known test or check in a command."""
    found: list[tuple[str, str]] = []
    for segment in _SEGMENT_SPLIT_RE.split(command):
        tokens = [token.lower() for token in _tokens(segment)]
        if not tokens:
            continue
        if "/" in tokens[0] and not tokens[0].startswith("./"):
            tokens[0] = tokens[0].rsplit("/", 1)[-1]
        for prefix, label, kind in _VALIDATION_PREFIXES:
            if tuple(tokens[: len(prefix)]) == prefix:
                if (label, kind) not in found:
                    found.append((label, kind))
                break
    return found


def _exit_code(event: TraceEvent) -> int | None:
    value = event.data.get("exit_code")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
        return int(value.strip())
    for key in ("result", "error"):
        text = event.data.get(key)
        if isinstance(text, str):
            match = _EXIT_CODE_RE.search(text)
            if match:
                return int(match.group(1))
    return None


def _outcome(call: TraceEvent, results: dict[str, list[TraceEvent]]) -> tuple[str, int | None]:
    linked = results.get(call.event_id, [])
    if not linked:
        return "no result recorded", None
    result = linked[-1]
    code = _exit_code(result)
    if result.event_type == EventType.ERROR:
        return "failed", code
    if code is not None:
        return ("passed" if code == 0 else "failed"), code
    if result.data.get("success") is False:
        return "failed", None
    return "succeeded", None


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

def _root_sessions(sessions: Sequence[_Session]) -> list[_Session]:
    roots = [session for session in sessions if not session.is_subagent]
    return roots or list(sessions)


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def _inconclusive(name: str, detail: str, source: str = "no linked session") -> Finding:
    return Finding(name, "inconclusive", "low", source, detail)


def assess_rationale(sessions: Sequence[_Session]) -> Finding:
    if not sessions:
        return _inconclusive("rationale", "No session is linked, so no request was found.")
    roots = _root_sessions(sessions)
    with_prompt = 0
    prompts = 0
    empty = 0
    redacted = 0
    unknown_capture = 0
    for session in roots:
        events = [e for e in session.events if e.event_type == EventType.USER_PROMPT]
        texts = [_prompt_text(e) for e in events]
        prompts += sum(1 for text in texts if text.strip())
        empty += sum(1 for text in texts if not text.strip())
        redacted += sum(1 for e in events if e.redacted)
        if any(text.strip() for text in texts):
            with_prompt += 1
        elif not session.captures("prompt"):
            unknown_capture += 1
    facts = {
        "sessions": len(roots),
        "sessions_with_request": with_prompt,
        "prompts": prompts,
        "empty_prompts": empty,
        "redacted_prompts": redacted,
        "content_shown": False,
    }
    source = "user_prompt events (content not shown)"
    redaction = f", {redacted} redacted" if redacted else ""
    if with_prompt == len(roots):
        detail = (
            f"{_plural(prompts, 'prompt')} captured in "
            f"{_plural(len(roots), 'session')}{redaction}."
        )
        return Finding("rationale", "present", "high", source, detail, facts)
    if with_prompt:
        detail = (
            f"{with_prompt} of {len(roots)} sessions have a captured request{redaction}."
        )
        return Finding("rationale", "partial", "medium", source, detail, facts)
    if unknown_capture:
        detail = "No request captured, and the capture matrix does not confirm prompt capture."
        return Finding("rationale", "inconclusive", "low", source, detail, facts)
    if empty:
        detail = "Prompt events were captured but are empty."
        return Finding("rationale", "partial", "medium", source, detail, facts)
    detail = "No request captured, and the provider captures prompts."
    return Finding("rationale", "missing", "high", source, detail, facts)


def assess_scope(sessions: Sequence[_Session]) -> Finding:
    if not sessions:
        return _inconclusive("scope", "No session is linked, so no file activity was found.")
    written: dict[str, bool] = {}
    for session in sessions:
        cwd = _session_cwd(session.events)
        files_read, files_written, _ = _file_activity(session.events)
        reads = [_normalise_path(path, cwd) for path in files_read]
        prompts = [
            _prompt_text(e) for e in session.events if e.event_type == EventType.USER_PROMPT
        ]
        for raw in files_written:
            path = _normalise_path(raw, cwd)
            grounded = _named_in(path, prompts) or any(_same_file(path, read) for read in reads)
            written[path] = written.get(path, False) or grounded
    source = "file and edit tool events, prompt path mentions, file reads"
    if not written:
        facts = {"files_written": 0, "grounded": 0, "ungrounded": []}
        detail = "No file writes captured. The session may not have edited files."
        return Finding("scope", "inconclusive", "low", source, detail, facts)
    ungrounded = sorted(path for path, grounded in written.items() if not grounded)
    grounded_count = len(written) - len(ungrounded)
    facts = {
        "files_written": len(written),
        "grounded": grounded_count,
        "ungrounded": [_display_path(path) for path in ungrounded],
    }
    if not ungrounded:
        detail = (
            f"All {_plural(len(written), 'written file')} were named in a prompt "
            "or read in the session."
        )
        return Finding("scope", "present", "medium", source, detail, facts)
    detail = (
        f"{grounded_count} of {_plural(len(written), 'written file')} were named in a prompt "
        "or read in the session."
    )
    status = "partial" if grounded_count else "missing"
    return Finding("scope", status, "medium", source, detail, facts)


def assess_validation(sessions: Sequence[_Session]) -> Finding:
    if not sessions:
        return _inconclusive("validation", "No session is linked, so no test runs were found.")
    runs: list[dict[str, Any]] = []
    shell_calls = 0
    tool_calls = 0
    writes_after_last = 0
    for session in sessions:
        results: dict[str, list[TraceEvent]] = {}
        for event in session.events:
            if event.parent_id and event.event_type in (EventType.TOOL_RESULT, EventType.ERROR):
                results.setdefault(event.parent_id, []).append(event)
        last_validation_ts: float | None = None
        write_times: list[float] = []
        for event in session.events:
            if event.event_type == EventType.TOOL_CALL:
                tool_calls += 1
            if event.event_type == EventType.FILE_WRITE or (
                event.event_type == EventType.TOOL_CALL
                and _path_from_event(event)
                and _file_activity([event])[1]
            ):
                write_times.append(event.timestamp)
            if not _is_shell_call(event):
                continue
            shell_calls += 1
            labels = classify_validation(_command_text(event))
            if not labels:
                continue
            outcome, code = _outcome(event, results)
            for label, kind in labels:
                runs.append({
                    "label": label,
                    "kind": kind,
                    "outcome": outcome,
                    "exit_code": code,
                    "session_id": session.session_id,
                    "timestamp": event.timestamp,
                })
            last_validation_ts = event.timestamp
        if last_validation_ts is not None:
            writes_after_last += sum(1 for ts in write_times if ts > last_validation_ts)

    source = "shell tool calls and linked results inside the session (not CI)"
    if not runs:
        facts = {"runs": [], "shell_commands": shell_calls, "writes_after_last_run": 0}
        if tool_calls:
            detail = (
                f"No known test or check command among "
                f"{_plural(shell_calls, 'shell command')}."
            )
            return Finding("validation", "missing", "medium", source, detail, facts)
        detail = "No tool calls captured, so test runs cannot be assessed."
        return Finding("validation", "inconclusive", "low", source, detail, facts)

    latest: dict[str, dict[str, Any]] = {}
    for run in sorted(runs, key=lambda item: item["timestamp"]):
        latest[run["label"]] = run
    final = [latest[label] for label in sorted(latest)]
    facts = {
        "runs": [
            {key: run[key] for key in ("label", "kind", "outcome", "exit_code", "session_id")}
            for run in runs
        ],
        "latest": [
            {key: run[key] for key in ("label", "kind", "outcome", "exit_code")} for run in final
        ],
        "shell_commands": shell_calls,
        "writes_after_last_run": writes_after_last,
    }

    def describe(run: dict[str, Any]) -> str:
        code = "" if run["exit_code"] is None else f", exit {run['exit_code']}"
        return f"{run['label']} {run['outcome']}{code}"

    parts = [describe(run) for run in final]
    detail = "Last run: " + "; ".join(parts) + "."
    if writes_after_last:
        detail += f" {_plural(writes_after_last, 'file write')} after the last run."
    ok = all(run["outcome"] in ("passed", "succeeded") for run in final)
    if ok and not writes_after_last:
        exact = all(run["exit_code"] is not None for run in final)
        return Finding("validation", "present", "high" if exact else "medium", source, detail, facts)
    return Finding("validation", "partial", "medium", source, detail, facts)


def assess_stewardship(sessions: Sequence[_Session], store: TraceStore) -> Finding:
    if not sessions:
        return _inconclusive("stewardship", "No session is linked, so no human turns were found.")
    roots = _root_sessions(sessions)
    turns = 0
    interventions = 0
    modes: set[str] = set()
    unknown_capture = 0
    for session in roots:
        seen_tool = False
        session_turns = 0
        for event in session.events:
            if event.event_type == EventType.TOOL_CALL:
                seen_tool = True
            elif event.event_type == EventType.USER_PROMPT:
                session_turns += 1
                if seen_tool:
                    interventions += 1
            mode = event.data.get("permission_mode")
            if isinstance(mode, str) and _PERMISSION_MODE_RE.match(mode):
                modes.add(mode)
        turns += session_turns
        if not session_turns and not session.captures("prompt"):
            unknown_capture += 1
    ids = {session.session_id for session in sessions}
    try:
        requests = [req for req in list_requests(store) if req.session_id in ids]
    except OSError:
        requests = []
    approved = sum(1 for req in requests if req.state == "approved")
    denied = sum(1 for req in requests if req.state == "denied")
    pending = sum(1 for req in requests if req.state == "pending")
    facts = {
        "human_turns": turns,
        "interventions": interventions,
        "approvals_approved": approved,
        "approvals_denied": denied,
        "approvals_pending": pending,
        "permission_modes": sorted(modes),
    }
    source = "user_prompt events, approval requests, recorded permission modes"
    detail = (
        f"{_plural(turns, 'human turn')}, {interventions} after the first tool call, "
        f"{approved + denied} approval {'decision' if approved + denied == 1 else 'decisions'}"
    )
    if pending:
        detail += f", {pending} pending"
    if modes:
        detail += f". Permission modes: {', '.join(sorted(modes))}"
    detail += "."
    if turns == 0:
        if unknown_capture:
            return Finding("stewardship", "inconclusive", "low", source, detail, facts)
        return Finding("stewardship", "missing", "high", source, detail, facts)
    if turns >= 2 or interventions or approved or denied:
        return Finding("stewardship", "present", "medium", source, detail, facts)
    return Finding("stewardship", "partial", "medium", source, detail, facts)


def assess_evidence(
    sessions: Sequence[_Session],
    store: TraceStore,
    matrix: CaptureMatrix,
) -> Finding:
    if not sessions:
        return _inconclusive(
            "evidence",
            "No session is linked to this branch.",
            source="session link",
        )
    reports = [
        assess_session(store, s.session_id, s.meta, s.events, matrix) for s in sessions
    ]
    statuses = [report.result.status.value for report in reports]
    redaction = sorted({report.redaction.mode for report in reports})
    links = sorted({session.link for session in sessions})
    providers = sorted({report.result.provider or "unknown" for report in reports})
    facts = {
        "sessions": [
            {
                "session_id": report.session_id,
                "status": report.result.status.value,
                "provider": report.result.provider or None,
                "redaction": report.redaction.mode,
                "reason_codes": sorted({reason.code for reason in report.result.reasons}),
            }
            for report in reports
        ],
        "link": links,
    }
    counts = ", ".join(
        f"{statuses.count(status)} {status}"
        for status in ("healthy", "partial", "unknown", "invalid")
        if statuses.count(status)
    )
    detail = (
        f"Health: {counts} ({', '.join(providers)}). "
        f"Redaction: {', '.join(mode.replace('_', ' ') for mode in redaction)}. "
        f"Linked by: {', '.join(link.replace('_', ' ') for link in links)}."
    )
    source = "agent-strace health, capture matrix, session link"
    weak_link = not set(links) <= _STRONG_LINKS
    if "invalid" in statuses:
        status, confidence = "missing", "high"
    elif "unknown" in statuses:
        status, confidence = "inconclusive", "medium"
    elif "partial" in statuses:
        status, confidence = "partial", "high"
    else:
        status, confidence = "present", "high"
    if weak_link:
        confidence = "low"
        if status == "present":
            status = "partial"
    return Finding("evidence", status, confidence, source, detail, facts)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _load_sessions(
    store: TraceStore,
    linked: Sequence[LinkedSession],
    matrix: CaptureMatrix,
) -> list[_Session]:
    sessions: list[_Session] = []
    for item in linked:
        meta = store.load_meta(item.session_id)
        events = store.load_events(item.session_id)
        require_object_event_data(events, session_id=item.session_id)
        provider = detect_provider(meta, events).provider
        entry = matrix.entry(provider) if provider else None
        sessions.append(_Session(item.session_id, item.link, meta, events, provider, entry))
    return sessions


def link_sessions(
    store: TraceStore,
    requested: Iterable[str] = (),
    *,
    branch: str = "",
) -> list[LinkedSession]:
    """Resolve the sessions to review and how each one was linked."""
    requested = [value for value in requested if value]
    if requested:
        linked: list[LinkedSession] = []
        for value in requested:
            session_id = store.find_session(value)
            if not session_id:
                raise PRCommentError(f"Session not found: {value}")
            if session_id not in [item.session_id for item in linked]:
                linked.append(LinkedSession(session_id, "explicit"))
        return linked
    if not store.list_sessions():
        return []
    ids, link = select_sessions_with_link(store, branch=branch)
    return [LinkedSession(session_id, link) for session_id in ids]


def build_report(
    store: TraceStore,
    linked: Sequence[LinkedSession],
    threshold: Threshold,
    *,
    branch: str = "",
    matrix: CaptureMatrix | None = None,
) -> ReviewReport:
    matrix = matrix or load_capture_matrix()
    sessions = _load_sessions(store, linked, matrix)
    findings = (
        assess_rationale(sessions),
        assess_scope(sessions),
        assess_validation(sessions),
        assess_stewardship(sessions, store),
        assess_evidence(sessions, store, matrix),
    )
    return ReviewReport(branch, tuple(linked), findings, threshold)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _scope_paths(finding: Finding) -> list[str]:
    paths = finding.facts.get("ungrounded") or []
    return [str(path) for path in paths]


def format_text(report: ReviewReport, out: TextIO) -> None:
    w = out.write
    w(f"Reviewability: {'pass' if report.passed else 'fail'}\n")
    w(f"Branch:    {report.branch or 'unknown'}\n")
    if report.sessions:
        for linked in report.sessions:
            w(f"Session:   {linked.session_id} ({LINK_LABELS[linked.link]})\n")
    else:
        w(f"Session:   none ({LINK_LABELS['none']})\n")
    w(f"Threshold: {report.threshold.describe().replace('`', '')} ({report.threshold.source})\n\n")
    for finding in report.findings:
        mark = "  FAIL" if report.threshold.fails(finding) else ""
        w(f"{finding.label:<16} {finding.status:<13} {finding.confidence:<7}{mark}\n")
        w(f"  {finding.detail}\n")
        paths = _scope_paths(finding)
        if paths:
            shown = paths[:_PATH_PREVIEW]
            more = f" (+{len(paths) - len(shown)} more)" if len(paths) > len(shown) else ""
            w(f"  Not named or read: {', '.join(shown)}{more}\n")
        w(f"  Source: {finding.source}\n")
    w("\nCaptured evidence only. CI results are not read. See docs/review.md.\n")


def render_markdown(report: ReviewReport, *, heading: str = "### Reviewability") -> str:
    """Render a compact Markdown section for a PR comment or step summary."""
    lines = [heading, ""]
    lines.extend([
        "| Finding | Status | Confidence | Evidence | Source |",
        "|---|---|---|---|---|",
    ])
    for finding in report.findings:
        status = finding.status
        if report.threshold.fails(finding):
            status = f"**{status}** (fails)"
        detail = _table_text(finding.detail)
        paths = _scope_paths(finding)
        if paths:
            shown = ", ".join(f"`{_safe_code_text(path)}`" for path in paths[:_PATH_PREVIEW])
            more = f" +{len(paths) - _PATH_PREVIEW} more" if len(paths) > _PATH_PREVIEW else ""
            detail += f" Not named or read: {shown}{more}"
        lines.append(
            f"| {finding.label} | {status} | {finding.confidence} | {detail} | "
            f"{_table_text(finding.source)} |"
        )
    result = "pass" if report.passed else "fail"
    lines.extend([
        "",
        f"Check: **{result}**. Threshold {report.threshold.describe()} "
        f"({_table_text(report.threshold.source)}).",
        "",
        "<sub>Deterministic summary of captured session evidence. Validation is "
        "in-session only, not CI. Missing capture is reported, not guessed. "
        f"[How to read this]({DOCS_URL})</sub>",
        "",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_threshold_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the threshold flags shared by ``review`` and ``pr-comment --review``."""
    parser.add_argument(
        "--fail-on", dest="fail_on", default=None, metavar="STATUSES",
        help="comma-separated statuses that fail: missing, partial, inconclusive, "
             "or none (default: missing)",
    )
    parser.add_argument(
        "--min-confidence", dest="min_confidence", default=None,
        choices=CONFIDENCES,
        help="lowest confidence that can fail the check (default: medium)",
    )
    parser.add_argument(
        "--findings", default=None, metavar="NAMES",
        help="comma-separated findings the threshold applies to (default: all)",
    )
    parser.add_argument(
        "--config", default=None, metavar="FILE",
        help="config file with a review block (default: .agent-strace.yaml if present)",
    )


def threshold_from_args(args: argparse.Namespace) -> Threshold:
    return load_threshold(
        getattr(args, "config", None),
        fail_on=getattr(args, "fail_on", None),
        min_confidence=getattr(args, "min_confidence", None),
        findings=getattr(args, "findings", None),
    )


def cmd_review(args: argparse.Namespace) -> int:
    """CLI handler for ``agent-strace review``."""
    try:
        threshold = threshold_from_args(args)
    except ReviewConfigError as exc:
        sys.stderr.write(f"review: {exc}\n")
        return 2

    store = TraceStore(args.trace_dir)
    branch = getattr(args, "branch", "") or current_branch()
    try:
        linked = link_sessions(store, getattr(args, "session_id", None) or (), branch=branch)
        report = build_report(store, linked, threshold, branch=branch)
    except NonObjectEventDataError:
        raise
    except (PRCommentError, OSError, ValueError) as exc:
        sys.stderr.write(f"review: {exc}\n")
        return 2

    output = getattr(args, "format", "text")
    if output == "json":
        sys.stdout.write(json.dumps(report.to_dict(), indent=2) + "\n")
    elif output == "markdown":
        sys.stdout.write(render_markdown(report, heading="## agent-trace reviewability"))
    else:
        format_text(report, sys.stdout)
    return report.exit_code


__all__ = [
    "CONFIDENCES",
    "FINDINGS",
    "STATUSES",
    "Finding",
    "LinkedSession",
    "ReviewConfigError",
    "ReviewReport",
    "Threshold",
    "add_threshold_arguments",
    "build_report",
    "classify_validation",
    "cmd_review",
    "link_sessions",
    "load_threshold",
    "render_markdown",
    "threshold_from_args",
]
