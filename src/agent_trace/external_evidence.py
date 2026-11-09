"""External evidence signals for ``health`` and ``review``.

Reads evidence that other tools captured and reports one signal per item, with
a source, a status, and a confidence:

- Entire checkpoints: the ``Entire-Checkpoint`` commit trailer and the local
  checkpoint store, including a trailer whose checkpoint is not pushed.
- Copilot cloud agent: the ``Agent-Logs-Url`` commit trailer.
- Claude Code OpenTelemetry: an exported logs file, including ``safety_event``.

Statuses are ``present``, ``missing``, ``unpushed``, and ``inconclusive``.
Signals are facts. They never change an evidence-health status, a review
finding, the review threshold, or an exit code. A trailer alone never shows
that evidence is complete.

Everything is local: ``git`` reads of this clone and the given file. No
network. The output never contains prompt text, command strings, or tool
input; it reports counts, fixed labels, checkpoint IDs, and session links.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

SIGNAL_STATUSES = ("present", "missing", "unpushed", "inconclusive")
SIGNAL_CONFIDENCES = ("low", "medium", "high")

SOURCE_ENTIRE = "entire_checkpoint"
SOURCE_COPILOT = "copilot_agent_logs"
SOURCE_CLAUDE_OTEL = "claude_code_otel"
SOURCE_COMMITS = "commit_trailers"
SOURCE_LABELS = {
    SOURCE_ENTIRE: "Entire checkpoint",
    SOURCE_COPILOT: "Copilot agent logs",
    SOURCE_CLAUDE_OTEL: "Claude Code OTel",
    SOURCE_COMMITS: "Commit trailers",
}

NOT_GRADED = "Reported, not graded: never changes a status or the exit code."

MAX_COMMITS = 500
SAFETY_EVENT_MIN_VERSION = (2, 1, 307)

ENTIRE_TRAILER = "Entire-Checkpoint"
COPILOT_TRAILER = "Agent-Logs-Url"
ENTIRE_REF_PREFIX = "refs/entire/checkpoints/"
ENTIRE_V1_BRANCH = "entire/checkpoints/v1"
ENTIRE_PUSH_QUEUE = "entire-checkpoint-push-queue.jsonl"
ENTIRE_PUSH_POLICY = os.path.join("entire", "checkpoint-push-policy")
ENTIRE_SETTINGS = (".entire/settings.json", ".entire/settings.local.json")

_RANGE_RE = re.compile(r"^[A-Za-z0-9._/~^@{}+-]{1,256}$")
_LEGACY_ID_RE = re.compile(r"^[0-9a-f]{12}$")
_ULID_RE = re.compile(r"^[0-7][0-9ABCDEFGHJKMNPQRSTVWXYZ]{25}$")
_ENTIRE_LINE_RE = re.compile(
    r"^" + ENTIRE_TRAILER + r":\s*([0-9a-f]{12}|[0-9ABCDEFGHJKMNPQRSTVWXYZ]{26})\s*$"
)
_ENTIRE_ANY_RE = re.compile(r"^\s*(?:[*-]\s+)?" + ENTIRE_TRAILER + r":", re.IGNORECASE)
_TRAILER_LINE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]*:\s?\S")
_GIT_GENERATED = ("Signed-off-by: ", "(cherry picked from commit ")
_COPILOT_LINE_RE = re.compile(
    r"^\s*(?:[*-]\s+)?" + COPILOT_TRAILER + r":\s*(\S+)\s*$", re.IGNORECASE
)
_COPILOT_URL_RE = re.compile(
    r"^https://[A-Za-z0-9.-]+(?::[0-9]{1,5})?/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+"
    r"/sessions/[A-Za-z0-9_-]{1,128}/?$"
)
_SAFE_URL_RE = re.compile(r"^https://[A-Za-z0-9._~:/?#@!$&'*+,;=%-]{1,512}$")
_LABEL_RE = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
_VERSION_RE = re.compile(r"^(\d{1,4})\.(\d{1,4})\.(\d{1,6})(?:[-+][0-9A-Za-z.-]{0,32})?$")

_CLAUDE_EVENT_NAMES = frozenset({
    "user_prompt", "assistant_response", "tool_result", "tool_decision",
    "api_request", "api_error", "api_refusal", "api_request_body",
    "api_response_body", "safety_event",
})
_SAFETY_DECISIONS = ("reject", "accept", "sent_back")


class ExternalEvidenceError(ValueError):
    """An external evidence flag is not valid."""


@dataclass(frozen=True)
class ExternalSignal:
    """One piece of evidence another tool captured."""

    source: str
    status: str
    confidence: str
    detail: str
    subject: str = ""
    facts: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return SOURCE_LABELS[self.source]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "subject": self.subject or None,
            "status": self.status,
            "confidence": self.confidence,
            "detail": self.detail,
            "facts": self.facts,
        }


@dataclass(frozen=True)
class ExternalEvidence:
    """The signals read for one ``health`` or ``review`` run."""

    commits: str
    otel_logs: str
    signals: tuple[ExternalSignal, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "commits": self.commits or None,
            "otel_logs": Path(self.otel_logs).name if self.otel_logs else None,
            "graded": False,
            "signals": [signal.to_dict() for signal in self.signals],
        }

    def inputs(self, code: Any = None) -> str:
        wrap = code or (lambda text: text)
        parts = []
        if self.commits:
            parts.append(f"commits {wrap(self.commits)}")
        if self.otel_logs:
            parts.append(f"OTel logs {wrap(Path(self.otel_logs).name)}")
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _git(args: Sequence[str], cwd: str | None) -> str | None:
    """Run a read-only git command. Return stdout, or None on any failure."""
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0")
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            env=env,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def validate_range(value: str) -> str:
    text = (value or "").strip()
    if not text:
        raise ExternalEvidenceError("--commits needs a git revision range, such as origin/main..HEAD")
    if text.startswith("-") or not _RANGE_RE.match(text):
        raise ExternalEvidenceError(
            "--commits must be a plain git revision range, such as origin/main..HEAD"
        )
    return text


def read_commits(rev_range: str, cwd: str | None = None) -> list[tuple[str, str]] | None:
    """Return ``(sha, message)`` pairs for the range, newest first, or None."""
    out = _git(
        ["log", f"--max-count={MAX_COMMITS + 1}", "--format=%x1e%H%x1f%B", rev_range, "--"],
        cwd,
    )
    if out is None:
        return None
    commits: list[tuple[str, str]] = []
    for chunk in out.split("\x1e"):
        if "\x1f" not in chunk:
            continue
        sha, message = chunk.split("\x1f", 1)
        commits.append((sha.strip(), message))
    return commits


# ---------------------------------------------------------------------------
# Trailers
# ---------------------------------------------------------------------------

def final_trailer_block(message: str) -> list[str]:
    """Return the lines of the message's final trailer block, if it has one.

    Follows git's rule as Entire 0.13.1 applies it: the last paragraph, never
    the subject, where every line is a trailer or a continuation, or where a
    git-generated trailer allows up to three other lines per trailer line.
    """
    lines = message.replace("\r\n", "\n").split("\n")
    end = len(lines)
    while end > 0 and not lines[end - 1].strip():
        end -= 1
    start = end
    while start > 0 and lines[start - 1].strip():
        start -= 1
    if start == 0 or start == end:
        return []
    block = lines[start:end]
    trailers = others = 0
    generated = continuable = False
    for line in block:
        if line.startswith(_GIT_GENERATED):
            trailers += 1
            generated = continuable = True
        elif _TRAILER_LINE_RE.match(line):
            trailers += 1
            continuable = True
        elif line[:1] in (" ", "\t") and continuable:
            continue
        else:
            others += 1
            continuable = False
    if trailers == 0:
        return []
    if others and not (generated and others <= 3 * trailers):
        return []
    return block


def is_checkpoint_id(value: str) -> bool:
    return bool(_LEGACY_ID_RE.match(value) or _ULID_RE.match(value))


def entire_checkpoints(message: str) -> tuple[list[str], int]:
    """Return linked checkpoint IDs and the count of unlinked trailer lines.

    Only lines in the final trailer block link a commit. An
    ``Entire-Checkpoint`` line elsewhere, or with an invalid ID, does not.
    """
    linked: list[str] = []
    block_lines = final_trailer_block(message)
    block = set(block_lines)
    for line in block_lines:
        match = _ENTIRE_LINE_RE.match(line)
        if match and is_checkpoint_id(match.group(1)) and match.group(1) not in linked:
            linked.append(match.group(1))
    unlinked = 0
    for line in message.replace("\r\n", "\n").split("\n"):
        if not _ENTIRE_ANY_RE.match(line):
            continue
        match = _ENTIRE_LINE_RE.match(line)
        if line in block and match and is_checkpoint_id(match.group(1)):
            continue
        unlinked += 1
    return linked, unlinked


def copilot_log_urls(message: str) -> list[str]:
    """Return every ``Agent-Logs-Url`` value, including squash-merge bodies."""
    urls: list[str] = []
    for line in message.replace("\r\n", "\n").split("\n"):
        match = _COPILOT_LINE_RE.match(line)
        if match and match.group(1) not in urls:
            urls.append(match.group(1))
    return urls


# ---------------------------------------------------------------------------
# Entire local checkpoint store
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EntireStore:
    """What this clone holds for Entire, read without the network."""

    refs: dict[str, str]
    queued: frozenset[str]
    queue_found: bool
    policy: str
    push_sessions: bool | None
    v1_branch: bool
    v1_tracking: tuple[str, ...]

    def ref_for(self, checkpoint_id: str) -> str:
        return f"{ENTIRE_REF_PREFIX}{checkpoint_id[-2:]}/{checkpoint_id}"


def _common_dir(cwd: str | None) -> Path | None:
    out = _git(["rev-parse", "--git-common-dir"], cwd)
    if out is None or not out.strip():
        return None
    path = Path(out.strip())
    if not path.is_absolute():
        path = Path(cwd or os.getcwd()) / path
    return path


def _read_queue(common: Path) -> tuple[frozenset[str], bool]:
    path = common / ENTIRE_PUSH_QUEUE
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return frozenset(), False
    except OSError:
        return frozenset(), False
    refs = set()
    for line in text.splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        ref = entry.get("ref") if isinstance(entry, dict) else None
        if isinstance(ref, str) and ref.startswith(ENTIRE_REF_PREFIX):
            refs.add(ref)
    return frozenset(refs), True


def _read_policy(common: Path) -> str:
    """Return automatic, manual, or unreadable. A missing file is automatic."""
    try:
        text = (common / ENTIRE_PUSH_POLICY).read_text(encoding="utf-8")
    except FileNotFoundError:
        return "automatic"
    except (OSError, UnicodeDecodeError):
        return "unreadable"
    value = text.strip()
    return value if value in ("automatic", "manual") else "unreadable"


def _read_push_sessions(toplevel: Path | None) -> bool | None:
    """Return ``strategy_options.push_sessions``; the local file wins."""
    if toplevel is None:
        return None
    value: bool | None = None
    for name in ENTIRE_SETTINGS:
        try:
            data = json.loads((toplevel / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        options = data.get("strategy_options") if isinstance(data, dict) else None
        if isinstance(options, dict) and isinstance(options.get("push_sessions"), bool):
            value = options["push_sessions"]
    return value


def read_entire_store(cwd: str | None = None) -> EntireStore | None:
    common = _common_dir(cwd)
    if common is None:
        return None
    refs: dict[str, str] = {}
    out = _git(["for-each-ref", "--format=%(refname) %(objectname)", ENTIRE_REF_PREFIX], cwd)
    for line in (out or "").splitlines():
        name, _, sha = line.partition(" ")
        if name.startswith(ENTIRE_REF_PREFIX):
            refs[name] = sha
    queued, queue_found = _read_queue(common)
    top = _git(["rev-parse", "--show-toplevel"], cwd)
    toplevel = Path(top.strip()) if top and top.strip() else None
    v1 = _git(["rev-parse", "--verify", "--quiet", f"refs/heads/{ENTIRE_V1_BRANCH}^{{commit}}"], cwd)
    tracking_out = _git(
        ["for-each-ref", "--format=%(refname)", f"refs/remotes/*/{ENTIRE_V1_BRANCH}"], cwd
    )
    tracking = tuple(sorted(line for line in (tracking_out or "").splitlines() if line))
    return EntireStore(
        refs=refs,
        queued=queued,
        queue_found=queue_found,
        policy=_read_policy(common),
        push_sessions=_read_push_sessions(toplevel),
        v1_branch=v1 is not None,
        v1_tracking=tracking,
    )


def _v1_path(checkpoint_id: str) -> str:
    return f"{checkpoint_id[:2]}/{checkpoint_id[2:]}/metadata.json"


def _tree_has(rev: str, path: str, cwd: str | None) -> bool:
    return _git(["cat-file", "-e", f"{rev}:{path}"], cwd) is not None


def assess_entire_checkpoint(
    checkpoint_id: str,
    commits: Sequence[str],
    store: EntireStore | None,
    cwd: str | None = None,
) -> ExternalSignal:
    """Resolve one linked checkpoint against the local Entire store."""
    facts: dict[str, Any] = {"commits": list(commits)}
    trailer = f"Trailer present in {_plural(len(commits), 'commit')}"
    if store is None:
        return ExternalSignal(
            SOURCE_ENTIRE, "inconclusive", "low",
            f"{trailer}. Not a git clone, so the checkpoint store was not read.",
            checkpoint_id, facts,
        )
    facts.update(push_policy=store.policy, push_sessions=store.push_sessions)
    held = []
    if store.policy == "manual":
        held.append("push policy manual: only entire checkpoint push uploads")
    elif store.policy == "unreadable":
        held.append("push policy unreadable: Entire fails closed and holds pushes")
    if store.push_sessions is False:
        held.append("push_sessions is false: checkpoints stay local")
    hold = f" ({'; '.join(held)})" if held else ""

    ref = store.ref_for(checkpoint_id)
    if ref in store.refs:
        facts.update(backend="git-refs", ref=ref, queued=ref in store.queued)
        if ref in store.queued:
            return ExternalSignal(
                SOURCE_ENTIRE, "unpushed", "high",
                f"{trailer}, not pushed: the checkpoint ref is local and waiting in "
                f"Entire's push queue{hold}.",
                checkpoint_id, facts,
            )
        if store.push_sessions is False:
            return ExternalSignal(
                SOURCE_ENTIRE, "unpushed", "medium",
                f"{trailer}. The checkpoint ref is local and push_sessions is false, so "
                "Entire does not push it.",
                checkpoint_id, facts,
            )
        return ExternalSignal(
            SOURCE_ENTIRE, "present", "medium",
            f"{trailer}. The checkpoint ref is local and not queued for push{hold}. "
            "The remote was not checked.",
            checkpoint_id, facts,
        )

    if store.v1_branch and _tree_has(f"refs/heads/{ENTIRE_V1_BRANCH}", _v1_path(checkpoint_id), cwd):
        facts.update(backend="git-branch", branch=ENTIRE_V1_BRANCH)
        on_remote = [
            tracking for tracking in store.v1_tracking
            if _tree_has(tracking, _v1_path(checkpoint_id), cwd)
        ]
        facts["remote_tracking"] = list(store.v1_tracking)
        if on_remote:
            facts["on_remote_tracking"] = on_remote
            return ExternalSignal(
                SOURCE_ENTIRE, "present", "medium",
                f"{trailer}. The checkpoint is on {ENTIRE_V1_BRANCH} and on the "
                "remote-tracking branch as of the last fetch.",
                checkpoint_id, facts,
            )
        if store.v1_tracking:
            return ExternalSignal(
                SOURCE_ENTIRE, "unpushed", "medium",
                f"{trailer}, not pushed: the checkpoint is on the local {ENTIRE_V1_BRANCH} "
                f"branch but not on its remote-tracking branch{hold}. A stale fetch can "
                "overstate this.",
                checkpoint_id, facts,
            )
        return ExternalSignal(
            SOURCE_ENTIRE, "unpushed", "low",
            f"{trailer}. The checkpoint is on the local {ENTIRE_V1_BRANCH} branch with no "
            f"remote-tracking branch, which Entire counts as not pushed{hold}. A fetch may "
            "show otherwise.",
            checkpoint_id, facts,
        )

    facts["backend"] = None
    return ExternalSignal(
        SOURCE_ENTIRE, "inconclusive", "low",
        f"{trailer}. The checkpoint is not in this clone. It may be on the remote; "
        "fetch Entire's checkpoint refs to read it.",
        checkpoint_id, facts,
    )


# ---------------------------------------------------------------------------
# Commit trailer signals
# ---------------------------------------------------------------------------

def _plural(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


def assess_commits(rev_range: str, cwd: str | None = None) -> list[ExternalSignal]:
    """Read Entire and Copilot trailers in the range and resolve each one."""
    commits = read_commits(rev_range, cwd)
    if commits is None:
        return [ExternalSignal(
            SOURCE_COMMITS, "inconclusive", "low",
            "Could not read the commit range in this clone. A shallow checkout or a "
            "missing base ref causes this; fetch more history.",
            rev_range, {"commits_read": 0},
        )]
    truncated = len(commits) > MAX_COMMITS
    commits = commits[:MAX_COMMITS]
    if not commits:
        return [ExternalSignal(
            SOURCE_COMMITS, "inconclusive", "medium",
            "The range has no commits, so no trailer was read.",
            rev_range, {"commits_read": 0},
        )]
    read = f"{_plural(len(commits), 'commit')} read"
    if truncated:
        read += f" (capped at {MAX_COMMITS})"
    absent_confidence = "medium" if truncated else "high"

    checkpoint_commits: dict[str, list[str]] = {}
    unlinked_commits: list[str] = []
    url_commits: dict[str, list[str]] = {}
    for sha, message in reversed(commits):
        linked, unlinked = entire_checkpoints(message)
        for checkpoint_id in linked:
            checkpoint_commits.setdefault(checkpoint_id, []).append(sha)
        if unlinked:
            unlinked_commits.append(sha)
        for url in copilot_log_urls(message):
            url_commits.setdefault(url, []).append(sha)

    signals: list[ExternalSignal] = []
    if checkpoint_commits:
        store = read_entire_store(cwd)
        for checkpoint_id, shas in checkpoint_commits.items():
            signals.append(assess_entire_checkpoint(checkpoint_id, shas, store, cwd))
    if unlinked_commits:
        signals.append(ExternalSignal(
            SOURCE_ENTIRE, "inconclusive", "medium",
            f"{_plural(len(unlinked_commits), 'commit')} "
            f"{'carries' if len(unlinked_commits) == 1 else 'carry'} an {ENTIRE_TRAILER} line "
            "outside the final trailer block or with an invalid ID. Entire 0.13.1 and "
            "later do not link those commits.",
            "", {"commits": unlinked_commits},
        ))
    if not checkpoint_commits and not unlinked_commits:
        signals.append(ExternalSignal(
            SOURCE_ENTIRE, "missing", absent_confidence,
            f"No {ENTIRE_TRAILER} trailer; {read}.",
            "", {"commits_read": len(commits), "truncated": truncated},
        ))

    for url, shas in url_commits.items():
        facts = {"commits": shas}
        if _COPILOT_URL_RE.match(url):
            signals.append(ExternalSignal(
                SOURCE_COPILOT, "present", "low",
                f"{COPILOT_TRAILER} in {_plural(len(shas), 'commit')}. The session log is "
                "on GitHub and was not read; the trailer alone does not show complete "
                "evidence.",
                url, facts,
            ))
        else:
            signals.append(ExternalSignal(
                SOURCE_COPILOT, "inconclusive", "medium",
                f"{COPILOT_TRAILER} in {_plural(len(shas), 'commit')} is not a GitHub "
                "agent session link.",
                url if _SAFE_URL_RE.match(url) else "", facts,
            ))
    if not url_commits:
        signals.append(ExternalSignal(
            SOURCE_COPILOT, "missing", absent_confidence,
            f"No {COPILOT_TRAILER} trailer; {read}.",
            "", {"commits_read": len(commits), "truncated": truncated},
        ))
    return signals


# ---------------------------------------------------------------------------
# Claude Code OpenTelemetry logs
# ---------------------------------------------------------------------------

def _any_value(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    for key in ("stringValue", "boolValue", "intValue", "doubleValue"):
        if key in value:
            return value[key]
    return None


def _attributes(raw: Any) -> dict[str, Any]:
    if isinstance(raw, list):
        out: dict[str, Any] = {}
        for item in raw:
            if isinstance(item, dict) and isinstance(item.get("key"), str):
                out[item["key"]] = _any_value(item.get("value"))
        return out
    if isinstance(raw, dict):
        return {str(key): _any_value(value) for key, value in raw.items()}
    return {}


def _otlp_records(payload: dict[str, Any]) -> Iterator[tuple[dict[str, Any], Any, dict[str, Any]]]:
    for resource_logs in payload.get("resourceLogs") or ():
        if not isinstance(resource_logs, dict):
            continue
        raw_resource = resource_logs.get("resource")
        resource = _attributes(raw_resource.get("attributes")) if isinstance(raw_resource, dict) else {}
        for scope_logs in resource_logs.get("scopeLogs") or ():
            if not isinstance(scope_logs, dict):
                continue
            for record in scope_logs.get("logRecords") or ():
                if isinstance(record, dict):
                    yield _attributes(record.get("attributes")), _any_value(record.get("body")), resource


def _flat_record(item: dict[str, Any]) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    attrs = _attributes(item.get("attributes")) if "attributes" in item else {}
    for key, value in item.items():
        if key not in ("attributes", "resource", "body"):
            attrs.setdefault(key, _any_value(value))
    resource = _attributes(item.get("resource")) if isinstance(item.get("resource"), (dict, list)) else {}
    return attrs, _any_value(item.get("body")), resource


def _records(payload: Any) -> Iterator[tuple[dict[str, Any], Any, dict[str, Any]]]:
    if isinstance(payload, dict):
        if "resourceLogs" in payload:
            yield from _otlp_records(payload)
        else:
            yield _flat_record(payload)
    elif isinstance(payload, list):
        for item in payload:
            yield from _records(item)


def read_otel_records(path: str) -> tuple[list[tuple[dict[str, Any], Any, dict[str, Any]]], int]:
    """Read an OTLP/JSON logs export or JSON Lines. Return records and bad lines."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    try:
        return list(_records(json.loads(text))), 0
    except ValueError:
        pass
    records: list[tuple[dict[str, Any], Any, dict[str, Any]]] = []
    bad = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            records.extend(_records(json.loads(line)))
        except ValueError:
            bad += 1
    return records, bad


def _claude_event(attrs: dict[str, Any], body: Any, resource: dict[str, Any]) -> str | None:
    name = attrs.get("event.name")
    body_text = body if isinstance(body, str) else ""
    if not isinstance(name, str) or not name:
        name = body_text
    if name.startswith("claude_code."):
        name = name[len("claude_code."):]
    service = resource.get("service.name") or attrs.get("service.name")
    if body_text.startswith("claude_code.") or service == "claude-code":
        return name if _LABEL_RE.match(name or "") else "unrecognized"
    if not service and name in _CLAUDE_EVENT_NAMES:
        return name
    return None


def _label(value: Any) -> str:
    text = str(value) if value is not None else ""
    return text if _LABEL_RE.match(text) else "unrecognized"


def _version(value: Any) -> tuple[int, int, int] | None:
    match = _VERSION_RE.match(str(value or ""))
    return (int(match.group(1)), int(match.group(2)), int(match.group(3))) if match else None


def _counts(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def assess_otel_logs(path: str) -> ExternalSignal:
    """Summarize Claude Code ``safety_event`` records in an exported logs file."""
    name = Path(path).name
    try:
        records, bad = read_otel_records(path)
    except OSError:
        return ExternalSignal(
            SOURCE_CLAUDE_OTEL, "inconclusive", "low",
            "Could not read the logs file.", name, {"records": 0},
        )

    events: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    sessions: set[str] = set()
    versions: set[tuple[int, int, int]] = set()
    safety_tool_uses: set[str] = set()
    rejected_tool_uses: list[str] = []
    rejects_untracked = 0
    for attrs, body, resource in records:
        event = _claude_event(attrs, body, resource)
        if event is None:
            continue
        events[event] += 1
        session = attrs.get("session.id")
        if isinstance(session, str) and session:
            sessions.add(session)
        version = _version(attrs.get("app.version") or resource.get("service.version"))
        if version:
            versions.add(version)
        tool_use = attrs.get("tool_use_id")
        if event == "safety_event":
            categories[_label(attrs.get("category"))] += 1
            decision = _label(attrs.get("decision"))
            decisions[decision if decision in _SAFETY_DECISIONS else "unrecognized"] += 1
            if isinstance(tool_use, str) and tool_use:
                safety_tool_uses.add(tool_use)
        elif event == "tool_decision" and attrs.get("decision") == "reject":
            if isinstance(tool_use, str) and tool_use:
                rejected_tool_uses.append(tool_use)
            else:
                rejects_untracked += 1

    total = sum(events.values())
    rejects = len(rejected_tool_uses) + rejects_untracked
    without = sum(1 for tool_use in rejected_tool_uses if tool_use not in safety_tool_uses)
    facts: dict[str, Any] = {
        "records": len(records),
        "unparsed_lines": bad,
        "claude_code_events": total,
        "event_counts": _counts(events),
        "sessions": len(sessions),
        "versions": [".".join(str(part) for part in v) for v in sorted(versions)],
        "safety_events": events.get("safety_event", 0),
        "safety_categories": _counts(categories),
        "safety_decisions": _counts(decisions),
        "tool_decision_rejects": rejects,
        "rejects_without_safety_event": without,
    }
    if total == 0:
        return ExternalSignal(
            SOURCE_CLAUDE_OTEL, "missing", "high" if not bad else "medium",
            "No Claude Code events in the file.", name, facts,
        )

    scope = f"{_plural(total, 'Claude Code event')} across {_plural(len(sessions), 'session')}"
    unmatched = (
        f" {_plural(without, 'rejected tool decision')} "
        f"{'has' if without == 1 else 'have'} no matching safety event."
        if without else ""
    )
    safety = events.get("safety_event", 0)
    if safety:
        shown = ", ".join(f"{decisions[d]} {d.replace('_', ' ')}" for d in _SAFETY_DECISIONS if decisions[d])
        kinds = ", ".join(f"{kind} {count}" for kind, count in sorted(categories.items()))
        return ExternalSignal(
            SOURCE_CLAUDE_OTEL, "present", "medium",
            f"{_plural(safety, 'safety event')} ({shown or 'no decision'}); categories: "
            f"{kinds}. {scope}.{unmatched} Some refusals log no safety event, so the "
            "count is a floor.",
            name, facts,
        )
    minimum = ".".join(str(part) for part in SAFETY_EVENT_MIN_VERSION)
    if versions and max(versions) < SAFETY_EVENT_MIN_VERSION:
        return ExternalSignal(
            SOURCE_CLAUDE_OTEL, "inconclusive", "medium",
            f"No safety events. {scope}. Every recorded Claude Code version predates "
            f"{minimum}, which added safety_event.",
            name, facts,
        )
    rejected = f" {_plural(rejects, 'tool decision')} rejected." if rejects else ""
    return ExternalSignal(
        SOURCE_CLAUDE_OTEL, "inconclusive", "medium",
        f"No safety events. {scope}.{rejected} Absence does not show that nothing was "
        f"refused: some refusals log only tool_decision, and safety_event needs Claude "
        f"Code {minimum} or later.",
        name, facts,
    )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def collect_external_evidence(
    commits: str = "",
    otel_logs: str = "",
    *,
    cwd: str | None = None,
) -> ExternalEvidence | None:
    """Read the requested sources. Return None when none was requested."""
    rev_range = validate_range(commits) if commits else ""
    if not rev_range and not otel_logs:
        return None
    signals: list[ExternalSignal] = []
    if rev_range:
        signals.extend(assess_commits(rev_range, cwd))
    if otel_logs:
        signals.append(assess_otel_logs(otel_logs))
    return ExternalEvidence(rev_range, otel_logs, tuple(signals))


def add_external_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the external evidence flags shared by ``health``, ``review``, and ``pr-comment``."""
    parser.add_argument(
        "--commits", default="", metavar="RANGE",
        help="git revision range to read Entire-Checkpoint and Agent-Logs-Url trailers "
             "from, such as origin/main..HEAD (reported, not graded)",
    )
    parser.add_argument(
        "--otel-logs", dest="otel_logs", default="", metavar="FILE",
        help="Claude Code OpenTelemetry logs export (OTLP JSON or JSON Lines) to read "
             "safety_event from (reported, not graded)",
    )


def external_from_args(args: argparse.Namespace, *, cwd: str | None = None) -> ExternalEvidence | None:
    return collect_external_evidence(
        getattr(args, "commits", "") or "",
        getattr(args, "otel_logs", "") or "",
        cwd=cwd,
    )


def signal_lines(evidence: ExternalEvidence, *, code: Any = None) -> list[str]:
    """One line per signal: ``label subject: status (confidence). detail``."""
    wrap = code or (lambda text: text)
    lines = []
    for signal in evidence.signals:
        subject = f" {wrap(signal.subject)}" if signal.subject else ""
        lines.append(
            f"{signal.label}{subject}: {signal.status} ({signal.confidence}). {signal.detail}"
        )
    return lines


__all__ = [
    "ExternalEvidence",
    "ExternalEvidenceError",
    "ExternalSignal",
    "SIGNAL_CONFIDENCES",
    "SIGNAL_STATUSES",
    "SOURCE_LABELS",
    "add_external_arguments",
    "assess_commits",
    "assess_entire_checkpoint",
    "assess_otel_logs",
    "collect_external_evidence",
    "copilot_log_urls",
    "entire_checkpoints",
    "external_from_args",
    "final_trailer_block",
    "read_entire_store",
    "signal_lines",
]
