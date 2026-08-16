"""Privacy-minimized assignment bundles and deterministic rubric scoring.

Assignment bundles are deliberately separate from the ordinary share HTML.
The ordinary replay contains prompts, results, commands, and paths; assignment
mode instead builds every artifact from an allowlisted, relative-time event
projection.  Bundles provide internal digest consistency, not issuer
authenticity, and scoring is a process-telemetry review aid rather than an
employment decision or an assessment of task correctness.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import html
import io
import json
import math
import os
import re
import secrets
import stat
import struct
import sys
import unicodedata
import zipfile
import zlib
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from . import __version__
from .eval.config import _parse_yaml_value
from .models import EventType, SessionMeta, TraceEvent
from .store import TraceStore, validate_stored_id


class AssignmentError(ValueError):
    """Raised when an assignment bundle or rubric fails closed."""


BUNDLE_SCHEMA = "agent-strace-assignment-bundle/v1"
TRACE_EVENT_SCHEMA = "agent-strace-assignment-trace-event/v1"
STATS_SCHEMA = "agent-strace-assignment-stats/v1"
COST_SCHEMA = "agent-strace-assignment-cost/v1"
LINT_SCHEMA = "agent-strace-assignment-lint/v1"
SCORE_SCHEMA = "agent-strace-assignment-score/v1"

PAYLOAD_MEMBERS = (
    "trace.ndjson",
    "replay.html",
    "stats.json",
    "cost.json",
    "lint.json",
)
ZIP_MEMBERS = ("manifest.json", *PAYLOAD_MEMBERS)
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
ZIP_EXTRACT_VERSION = 20
ZIP_CREATE_VERSION = (3 << 8) | 20
ZIP_DOS_TIME = 0
ZIP_DOS_DATE = 33

MAX_ARCHIVE_BYTES = 24 * 1024 * 1024
MAX_TOTAL_UNCOMPRESSED = 18 * 1024 * 1024
MAX_TRACE_BYTES = 8 * 1024 * 1024
MAX_HTML_BYTES = 8 * 1024 * 1024
MAX_JSON_BYTES = 512 * 1024
MAX_SOURCE_META_BYTES = 512 * 1024
MAX_SOURCE_TRACE_BYTES = 64 * 1024 * 1024
MAX_SOURCE_EVENT_BYTES = 1024 * 1024
MAX_EVENTS = 50_000
MAX_JSON_DEPTH = 32
MAX_JSON_STRING = 16_384
MAX_COMPRESSION_RATIO = 200
MAX_SUBMISSIONS = 100
MAX_COMPARISON_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_COMPARISON_EVENTS = 200_000
MAX_RUBRIC_BYTES = 64 * 1024
MAX_CRITERIA = 64
MAX_DURATION_SECONDS = 366 * 24 * 60 * 60
MAX_TOKEN_COUNT = 1_000_000_000_000

_MEMBER_LIMITS = {
    "manifest.json": MAX_JSON_BYTES,
    "trace.ndjson": MAX_TRACE_BYTES,
    "replay.html": MAX_HTML_BYTES,
    "stats.json": MAX_JSON_BYTES,
    "cost.json": MAX_JSON_BYTES,
    "lint.json": MAX_JSON_BYTES,
}

_BIDI_CONTROLS = {
    "\u061c", "\u200e", "\u200f", "\u202a", "\u202b", "\u202c",
    "\u202d", "\u202e", "\u2066", "\u2067", "\u2068", "\u2069",
}
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_REF_RE = re.compile(r"^(?:Event|Tool|Resource|Model)-[0-9]{3,6}$")
_SEMVER_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9.-]+)?$")
_RUBRIC_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,79}$")

_TOOL_READ = {"read", "read_file", "file_read", "view", "open_file"}
_TOOL_WRITE = {
    "write", "write_file", "file_write", "edit", "create", "multiedit",
    "str_replace", "str_replace_based_edit_tool", "notebook_edit",
}
_TOOL_SHELL = {"bash", "shell", "exec", "execute", "terminal", "command"}
_TOOL_SEARCH = {"grep", "glob", "search", "find", "websearch", "web_search"}
_TOOL_NETWORK = {"webfetch", "web_fetch", "http", "request", "fetch", "curl"}
_TOOL_AGENT = {"agent", "task", "subagent", "delegate"}
_TOOL_CATEGORIES = {"read", "write", "shell", "search", "network", "agent", "other"}
_RECORDED_STATUSES = {"completed", "timeout", "terminated", "not_recorded"}

_LINT_MESSAGES = {
    "tool-loop": "Repeated tool-call sequence observed in minimized telemetry.",
    "reasoning-spiral": "Repeated model-call sequence observed without an intervening tool call.",
    "budget-proximity": "Recorded token estimate approached the configured lint budget.",
    "context-saturation": "Recorded context estimate approached the configured lint threshold.",
    "post-compaction-regression": "A post-compaction behavior signal was observed.",
    "redundant-read": "A repeated anonymous resource-read signal was observed.",
    "error-retry-loop": "Repeated error-and-retry behavior was observed.",
    "no-output": "No recognized write activity was observed.",
}
_ASSIGNMENT_SCORABLE_LINT_RULES = frozenset(
    set(_LINT_MESSAGES) - {"budget-proximity", "post-compaction-regression"}
)
_ASSIGNMENT_UNAVAILABLE_LINT_RULES = (
    "budget-proximity",
    "post-compaction-regression",
)

_LIMITATIONS = (
    "The bundle contains minimized process telemetry, not prompts, results, commands, paths, file contents, environment data, or candidate identity.",
    "Completed is a recorded process-termination status; it does not establish task completion or correctness.",
    "Cost is an offline estimate from aggregate token estimates and bundled rates; the minimized trace cannot independently reproduce source content lengths.",
    "SHA-256 member digests detect internal inconsistency but provide no issuer signature, provenance proof, or authenticity guarantee.",
    "Scores are deterministic rubric arithmetic for human review, not a hiring decision or a measure of candidate, protected-attribute, task, or outcome quality.",
)

_METRIC_SEMANTICS = {
    "cost_usd": "offline estimated USD cost from aggregate token estimates",
    "duration_seconds": "span between the minimum and maximum validated event timestamps",
    "error_count": "count of explicit error events",
    "lint_violations": "count of deterministic findings over minimized telemetry",
    "redundant_read_ratio": "repeated anonymous resource reads after the first divided by all anonymous resource reads",
    "session_status": "recorded process termination only; not task completion or correctness",
    "tool_call_count": "count of recorded tool-call events",
}

# Compatibility contract: values and assets below are part of bundle schema v1.
# Never edit them in place. Add a new version/profile and dispatch explicitly so
# already-issued submissions remain verifiable after ordinary product updates.
_ASSIGNMENT_PRICING_PROFILES = {
    "bundled-default-v1": {
        "input_rate_per_million": 3.0,
        "output_rate_per_million": 15.0,
        "pricing_snapshot_date": "2026-08-16",
    },
}
_CURRENT_ASSIGNMENT_PRICING_PROFILE = "bundled-default-v1"
_ASSIGNMENT_V1_INPUT_TYPES = {
    EventType.USER_PROMPT,
    EventType.LLM_REQUEST,
    EventType.TOOL_CALL,
}
_ASSIGNMENT_V1_OUTPUT_TYPES = {
    EventType.ASSISTANT_RESPONSE,
    EventType.LLM_RESPONSE,
    EventType.TOOL_RESULT,
}

_ASSIGNMENT_CSS_V1 = """
:root{color-scheme:light;font-family:ui-monospace,SFMono-Regular,Consolas,monospace}
body{max-width:1100px;margin:0 auto;padding:24px;color:#172033;background:#f7f8fb}
.header,.event,.phase,p,ul{background:#fff;border:1px solid #d9deea;border-radius:8px;padding:14px}
.meta-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px}
.meta-label{font-size:.75rem;color:#59647a;text-transform:uppercase}.meta-value{font-weight:700}
.event{margin:8px 0}.event summary{cursor:pointer}.ts{display:inline-block;min-width:90px;color:#59647a}
.badge{margin-right:10px;color:#263b70}.event-summary{color:#333}.event-detail{overflow:auto;white-space:pre-wrap}
.search-bar{display:flex;gap:8px;margin:16px 0}.search-input{width:100%;padding:9px}
.footer{margin-top:24px;color:#59647a;font-size:.8rem}
""".strip()

_ASSIGNMENT_JS_V1 = """
(()=>{const q=document.getElementById('search-input');if(!q)return;const rows=[...document.querySelectorAll('.event')];
const count=document.getElementById('search-count');function filter(){const value=q.value.toLowerCase();let shown=0;
for(const row of rows){const match=!value||row.textContent.toLowerCase().includes(value);row.hidden=!match;if(match)shown++;}
count.textContent=`${shown}/${rows.length}`;}q.addEventListener('input',filter);filter();})();
""".strip()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_file_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AssignmentError("duplicate JSON object key")
        result[key] = value
    return result


def _validate_json_tree(
    value: Any,
    *,
    depth: int = 0,
    max_string: int = MAX_JSON_STRING,
    allow_text_controls: bool = False,
) -> None:
    if depth > MAX_JSON_DEPTH:
        raise AssignmentError("JSON nesting exceeds the supported depth")
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AssignmentError("JSON numbers must be finite")
        return
    if isinstance(value, str):
        if len(value) > max_string:
            raise AssignmentError("JSON string exceeds the supported length")
        if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
            raise AssignmentError("JSON string contains an unsupported surrogate")
        if not allow_text_controls and any(
            char in _BIDI_CONTROLS or ord(char) < 0x20 for char in value
        ):
            raise AssignmentError("JSON string contains unsupported control characters")
        return
    if isinstance(value, list):
        if len(value) > MAX_EVENTS:
            raise AssignmentError("JSON array exceeds the supported item count")
        for item in value:
            _validate_json_tree(
                item,
                depth=depth + 1,
                max_string=max_string,
                allow_text_controls=allow_text_controls,
            )
        return
    if isinstance(value, dict):
        if len(value) > 256:
            raise AssignmentError("JSON object exceeds the supported key count")
        for key, item in value.items():
            if (
                not isinstance(key, str)
                or not key
                or len(key) > 128
                or any(char in _BIDI_CONTROLS or ord(char) < 0x20 for char in key)
            ):
                raise AssignmentError("JSON object contains an invalid key")
            _validate_json_tree(
                item,
                depth=depth + 1,
                max_string=max_string,
                allow_text_controls=allow_text_controls,
            )
        return
    raise AssignmentError("JSON contains an unsupported value")


def _strict_json_bytes(
    raw: bytes,
    *,
    label: str,
    max_bytes: int,
    source_text: bool = False,
) -> Any:
    if not raw or len(raw) > max_bytes:
        raise AssignmentError(f"{label} size is invalid")
    try:
        text = raw.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_int=_parse_bounded_json_int,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                AssignmentError("JSON numbers must be finite")
            ),
        )
    except UnicodeDecodeError as exc:
        raise AssignmentError(f"{label} must be UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise AssignmentError(f"{label} is not valid JSON") from exc
    except AssignmentError:
        raise
    except (RecursionError, ValueError, OverflowError) as exc:
        raise AssignmentError(f"{label} contains an unsupported JSON value") from exc
    _validate_json_tree(
        value,
        max_string=max_bytes if source_text else MAX_JSON_STRING,
        allow_text_controls=source_text,
    )
    return value


def _parse_bounded_json_int(raw: str) -> int:
    digits = raw.lstrip("-")
    if not digits or len(digits) > 40:
        raise AssignmentError("JSON integer exceeds the supported length")
    try:
        return int(raw)
    except ValueError as exc:
        raise AssignmentError("JSON integer is invalid") from exc


def _require_exact_keys(
    value: Any,
    required: Iterable[str],
    optional: Iterable[str] = (),
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AssignmentError(f"{label} must be an object")
    required_set = set(required)
    allowed = required_set | set(optional)
    keys = set(value)
    if keys != required_set | (keys & set(optional)) or not required_set <= keys:
        raise AssignmentError(f"{label} has missing or unknown fields")
    if not keys <= allowed:
        raise AssignmentError(f"{label} has missing or unknown fields")
    return value


def _number(
    value: Any,
    *,
    label: str,
    minimum: float = 0.0,
    maximum: float = 1_000_000_000_000.0,
    integer: bool = False,
) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssignmentError(f"{label} must be a number")
    if isinstance(value, int):
        if value < minimum or value > maximum:
            raise AssignmentError(f"{label} is outside the supported range")
        if integer:
            return value
        return float(value)
    if not math.isfinite(value) or value < minimum or value > maximum:
        raise AssignmentError(f"{label} is outside the supported range")
    if integer:
        raise AssignmentError(f"{label} must be an integer")
    return value


def _validate_no_symlink_ancestors(path: Path) -> None:
    absolute = path.absolute()
    for candidate in (absolute, *absolute.parents):
        if candidate.is_symlink():
            raise AssignmentError("refusing a path with a symlink component")
        if candidate.exists() and not candidate.is_dir() and candidate != absolute:
            raise AssignmentError("path ancestor is not a directory")


def _open_directory_chain(path_value: str | Path, *, create: bool = False) -> int:
    """Open every absolute path component relative to its anchored parent fd."""
    path = Path(path_value).absolute()
    parts = path.parts
    if not parts or not path.is_absolute():
        raise AssignmentError("directory path is invalid")
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        current_fd = os.open(path.anchor, flags)
    except OSError as exc:
        raise AssignmentError("directory root could not be opened safely") from exc
    try:
        for part in parts[1:]:
            if part in {"", ".", ".."}:
                raise AssignmentError("directory path contains an unsafe component")
            try:
                next_fd = os.open(part, flags, dir_fd=current_fd)
            except FileNotFoundError:
                if not create:
                    raise AssignmentError("directory does not exist")
                try:
                    os.mkdir(part, 0o700, dir_fd=current_fd)
                    next_fd = os.open(part, flags, dir_fd=current_fd)
                except OSError as exc:
                    raise AssignmentError("directory could not be created safely") from exc
            except OSError as exc:
                raise AssignmentError("directory has an unsafe component") from exc
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except Exception:
        os.close(current_fd)
        raise


@contextlib.contextmanager
def _locked_store_directory(store: TraceStore):
    """Yield an anchored store directory while holding its ordinary lock file."""
    _validate_no_symlink_ancestors(store.base_dir)
    base_fd = _open_directory_chain(store.base_dir)
    lock_flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        lock_flags |= os.O_NOFOLLOW
    try:
        try:
            lock_fd = os.open(".store.lock", lock_flags, 0o600, dir_fd=base_fd)
        except OSError as exc:
            raise AssignmentError("trace store lock could not be opened safely") from exc
        try:
            if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
                raise AssignmentError("trace store lock is not a regular file")
            try:
                import fcntl
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
            except (ImportError, OSError):
                pass
            try:
                yield base_fd
            finally:
                try:
                    import fcntl
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                except (ImportError, OSError):
                    pass
        finally:
            os.close(lock_fd)
    finally:
        os.close(base_fd)


def _read_regular_nofollow(path_value: str | Path, *, max_bytes: int, label: str) -> bytes:
    path = Path(path_value)
    if path.name in {"", ".", ".."}:
        raise AssignmentError(f"{label} path is invalid")
    parent_fd = _open_directory_chain(path.parent)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path.name, flags, dir_fd=parent_fd)
    except OSError as exc:
        os.close(parent_fd)
        raise AssignmentError(f"{label} could not be opened safely") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0 or before.st_size > max_bytes:
            raise AssignmentError(f"{label} size or type is invalid")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        if len(raw) > max_bytes or (
            before.st_dev, before.st_ino, before.st_mode, before.st_size,
            before.st_mtime_ns, before.st_ctime_ns,
        ) != (
            after.st_dev, after.st_ino, after.st_mode, after.st_size,
            after.st_mtime_ns, after.st_ctime_ns,
        ):
            raise AssignmentError(f"{label} changed while it was read")
        return raw
    finally:
        os.close(descriptor)
        os.close(parent_fd)


def _read_child_file(
    directory_fd: int,
    name: str,
    *,
    max_bytes: int,
    label: str,
) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise AssignmentError(f"{label} could not be opened safely") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size < 0 or before.st_size > max_bytes:
            raise AssignmentError(f"{label} size or type is invalid")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        if len(raw) > max_bytes or (
            before.st_dev, before.st_ino, before.st_mode, before.st_size,
            before.st_mtime_ns, before.st_ctime_ns,
        ) != (
            after.st_dev, after.st_ino, after.st_mode, after.st_size,
            after.st_mtime_ns, after.st_ctime_ns,
        ):
            raise AssignmentError(f"{label} changed while it was read")
        return raw
    finally:
        os.close(descriptor)


def _snapshot_session(store: TraceStore, session_id: str) -> tuple[SessionMeta, list[TraceEvent]]:
    """Read one bounded source snapshot under the store lock without following links."""
    session_id = validate_stored_id(session_id, "stored session ID")
    if store.base_dir.is_symlink():
        raise AssignmentError("refusing a symlinked trace store")
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW

    with _locked_store_directory(store) as base_fd:
        try:
            session_fd = os.open(session_id, directory_flags, dir_fd=base_fd)
        except OSError as exc:
            raise AssignmentError("session directory could not be opened safely") from exc
        try:
            meta_raw = _read_child_file(
                session_fd,
                "meta.json",
                max_bytes=MAX_SOURCE_META_BYTES,
                label="session metadata",
            )
            events_raw = _read_child_file(
                session_fd,
                "events.ndjson",
                max_bytes=MAX_SOURCE_TRACE_BYTES,
                label="session event stream",
            )
        finally:
            os.close(session_fd)

    meta_value = _strict_json_bytes(
        meta_raw,
        label="session metadata",
        max_bytes=MAX_SOURCE_META_BYTES,
        source_text=True,
    )
    if not isinstance(meta_value, dict):
        raise AssignmentError("session metadata must be an object")
    try:
        meta = SessionMeta.from_json(json.dumps(meta_value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise AssignmentError("session metadata schema is invalid") from exc
    if meta.session_id != session_id:
        raise AssignmentError("session metadata identity does not match its directory")

    events: list[TraceEvent] = []
    if events_raw:
        lines = events_raw.splitlines()
        if len(lines) > MAX_EVENTS:
            raise AssignmentError("session has too many events for assignment mode")
        for line in lines:
            if not line or len(line) > MAX_SOURCE_EVENT_BYTES:
                raise AssignmentError("session event line size is invalid")
            value = _strict_json_bytes(
                line,
                label="session event",
                max_bytes=MAX_SOURCE_EVENT_BYTES,
                source_text=True,
            )
            if not isinstance(value, dict):
                raise AssignmentError("session event must be an object")
            try:
                event = TraceEvent.from_json(json.dumps(value, allow_nan=False))
            except (TypeError, ValueError) as exc:
                raise AssignmentError("session event schema is invalid") from exc
            if event.session_id and event.session_id != session_id:
                raise AssignmentError("session event identity does not match its directory")
            if not isinstance(event.data, dict):
                raise AssignmentError("session event data must be an object")
            event.session_id = session_id
            events.append(event)
    return meta, events


def _atomic_write_private(path_value: str | Path, data: bytes) -> None:
    """Atomically replace one regular output through an anchored parent fd."""
    path = Path(path_value)
    if path.name in {"", ".", ".."}:
        raise AssignmentError("--output must name a file")
    parent_fd = _open_directory_chain(path.parent, create=True)

    temporary_name = f".{path.name}.{secrets.token_hex(12)}.tmp"
    temporary_fd = -1
    try:
        try:
            existing = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None and not stat.S_ISREG(existing.st_mode):
            raise AssignmentError("output must be a non-symlinked regular file")

        create_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            create_flags |= os.O_NOFOLLOW
        temporary_fd = os.open(
            temporary_name, create_flags, 0o600, dir_fd=parent_fd
        )
        view = memoryview(data)
        while view:
            written = os.write(temporary_fd, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(temporary_fd)
        os.close(temporary_fd)
        temporary_fd = -1

        try:
            current = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            current = None
        if current is not None and not stat.S_ISREG(current.st_mode):
            raise AssignmentError("output target changed during the write")
        os.replace(
            temporary_name,
            path.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        temporary_name = ""
        try:
            os.fsync(parent_fd)
        except OSError:
            pass
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        if temporary_name:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except OSError:
                pass
        os.close(parent_fd)


def _tool_category(value: Any) -> str:
    if not isinstance(value, str):
        return "other"
    normalized = value.strip().lower().replace("-", "_")
    tail = normalized.rsplit("__", 1)[-1].rsplit(".", 1)[-1]
    if tail in _TOOL_READ or tail.endswith(("read_file", "file_read")):
        return "read"
    if tail in _TOOL_WRITE or tail.endswith(("write_file", "file_write")):
        return "write"
    if tail in _TOOL_SHELL or any(word in tail for word in ("shell", "terminal")):
        return "shell"
    if tail in _TOOL_SEARCH or any(word in tail for word in ("search", "grep", "glob")):
        return "search"
    if tail in _TOOL_NETWORK or any(word in tail for word in ("http", "fetch")):
        return "network"
    if tail in _TOOL_AGENT or any(word in tail for word in ("agent", "delegate")):
        return "agent"
    return "other"


def _source_resource(event: TraceEvent, category: str) -> str:
    data = event.data if isinstance(event.data, dict) else {}
    if event.event_type in {EventType.FILE_READ, EventType.FILE_WRITE}:
        for key in ("path", "file_path", "uri"):
            value = data.get(key)
            if isinstance(value, str) and value:
                return value
    if event.event_type == EventType.TOOL_CALL and category in {"read", "write"}:
        arguments = data.get("arguments")
        if isinstance(arguments, dict):
            for key in ("file_path", "path", "uri"):
                value = arguments.get(key)
                if isinstance(value, str) and value:
                    return value
    return ""


def _source_model(event: TraceEvent) -> str:
    data = event.data if isinstance(event.data, dict) else {}
    value = data.get("model")
    return value if isinstance(value, str) and value else ""


def _safe_token_fields(data: dict[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    payloads = [data]
    usage = data.get("usage")
    if isinstance(usage, dict):
        payloads.insert(0, usage)
    for output_key, candidates in {
        "input_tokens": ("input_tokens", "prompt_tokens"),
        "output_tokens": ("output_tokens", "completion_tokens"),
        "total_tokens": ("total_tokens",),
    }.items():
        for payload in payloads:
            value = next((payload.get(key) for key in candidates if key in payload), None)
            if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= MAX_TOKEN_COUNT:
                result[output_key] = value
                break
    return result


def _recorded_session_status(events: list[TraceEvent]) -> str:
    end = next((event for event in reversed(events) if event.event_type == EventType.SESSION_END), None)
    if end is None:
        return "not_recorded"
    data = end.data if isinstance(end.data, dict) else {}
    explicit = data.get("status") or data.get("session_status")
    if isinstance(explicit, str):
        normalized = explicit.strip().lower().replace("-", "_")
        if normalized in {"completed", "complete", "success", "succeeded"}:
            return "completed"
        if normalized in {"timeout", "timed_out"}:
            return "timeout"
        if normalized in {
            "killed", "cancelled", "canceled", "interrupted", "terminated",
            "failed", "failure", "aborted", "error",
        }:
            return "terminated"
    reason = " ".join(
        value for key in ("reason", "stop_reason", "termination_reason")
        if isinstance((value := data.get(key)), str)
    ).lower()
    if "timeout" in reason or "timed out" in reason:
        return "timeout"
    if any(word in reason for word in ("kill", "cancel", "interrupt", "terminate")):
        return "terminated"
    exit_code = data.get("exit_code")
    if isinstance(exit_code, (int, float)) and not isinstance(exit_code, bool) and math.isfinite(float(exit_code)):
        if float(exit_code) == 124:
            return "timeout"
        if float(exit_code) != 0:
            return "terminated"
    return "completed"


def _alias(mapping: dict[str, str], raw: str, prefix: str, width: int = 3) -> str:
    if raw not in mapping:
        mapping[raw] = f"{prefix}-{len(mapping) + 1:0{width}d}"
    return mapping[raw]


def _sanitize_events(events: list[TraceEvent]) -> list[dict[str, Any]]:
    if len(events) > MAX_EVENTS:
        raise AssignmentError("session has too many events for assignment mode")
    timestamps: list[float] = []
    for event in events:
        if isinstance(event.timestamp, bool) or not isinstance(event.timestamp, (int, float)):
            raise AssignmentError("event timestamp must be numeric")
        timestamp = float(event.timestamp)
        if not math.isfinite(timestamp) or abs(timestamp) > 253_402_300_799:
            raise AssignmentError("event timestamp is outside the supported range")
        timestamps.append(timestamp)
    base = min(timestamps) if timestamps else 0.0
    if timestamps and max(timestamps) - base > MAX_DURATION_SECONDS:
        raise AssignmentError("event timestamp span exceeds assignment bounds")

    event_id_counts: dict[str, int] = {}
    for index, event in enumerate(events, 1):
        if isinstance(event.event_id, str) and event.event_id:
            event_id_counts[event.event_id] = event_id_counts.get(event.event_id, 0) + 1
    event_refs: dict[str, str] = {}
    event_positions: dict[str, int] = {}
    for index, event in enumerate(events, 1):
        raw_id = event.event_id if isinstance(event.event_id, str) else ""
        if raw_id and event_id_counts.get(raw_id) == 1:
            event_refs[raw_id] = f"Event-{index:06d}"
            event_positions[raw_id] = index
    tool_refs: dict[str, str] = {}
    resource_refs: dict[str, str] = {}
    model_refs: dict[str, str] = {}
    status = _recorded_session_status(events)
    sanitized: list[dict[str, Any]] = []

    for index, (event, timestamp) in enumerate(zip(events, timestamps), 1):
        offset_ms = int(round((timestamp - base) * 1000))
        if offset_ms < 0 or offset_ms > MAX_DURATION_SECONDS * 1000:
            raise AssignmentError("event relative timestamp is outside assignment bounds")
        item: dict[str, Any] = {
            "schema": TRACE_EVENT_SCHEMA,
            "sequence": index,
            "event_ref": f"Event-{index:06d}",
            "event_type": event.event_type.value,
            "offset_ms": offset_ms,
        }
        if event.duration_ms is not None:
            duration = _number(
                event.duration_ms,
                label="event duration",
                maximum=MAX_DURATION_SECONDS * 1000,
            )
            item["duration_ms"] = int(round(float(duration)))
        if (
            isinstance(event.parent_id, str)
            and event.parent_id in event_refs
            and event_positions[event.parent_id] < index
        ):
            item["parent_ref"] = event_refs[event.parent_id]

        source_data = event.data if isinstance(event.data, dict) else {}
        data: dict[str, Any] = {"content_omitted": True}
        if event.event_type == EventType.TOOL_CALL:
            raw_tool = source_data.get("tool_name")
            tool_key = raw_tool if isinstance(raw_tool, str) and raw_tool else "@unknown"
            category = _tool_category(raw_tool)
            data = {
                "arguments_omitted": True,
                "content_omitted": True,
                "tool_category": category,
                "tool_ref": _alias(tool_refs, tool_key, "Tool"),
            }
            resource = _source_resource(event, category)
            if resource:
                data["resource_ref"] = _alias(resource_refs, resource, "Resource")
        elif event.event_type == EventType.TOOL_RESULT:
            data = {"content_omitted": True, "outcome_recorded": True}
            if isinstance(source_data.get("is_error"), bool):
                data["is_error"] = source_data["is_error"]
        elif event.event_type in {EventType.FILE_READ, EventType.FILE_WRITE}:
            category = "read" if event.event_type == EventType.FILE_READ else "write"
            resource = _source_resource(event, category)
            data = {"content_omitted": True}
            if resource:
                data["resource_ref"] = _alias(resource_refs, resource, "Resource")
        elif event.event_type in {EventType.LLM_REQUEST, EventType.LLM_RESPONSE}:
            data = {"content_omitted": True, **_safe_token_fields(source_data)}
            model = _source_model(event)
            if model:
                data["model_ref"] = _alias(model_refs, model, "Model")
        elif event.event_type == EventType.ERROR:
            data = {"content_omitted": True, "error_recorded": True}
        elif event.event_type == EventType.SESSION_END:
            data = {"content_omitted": True, "recorded_status": status}
        item["data"] = data
        sanitized.append(item)

    encoded = _encode_trace(sanitized)
    if len(encoded) > MAX_TRACE_BYTES:
        raise AssignmentError("minimized trace exceeds the assignment bundle size limit")
    return sanitized


def _encode_trace(events: list[dict[str, Any]]) -> bytes:
    if not events:
        return b""
    return b"".join(_canonical_json_bytes(item) + b"\n" for item in events)


def _validate_ref(value: Any, prefix: str) -> str:
    if not isinstance(value, str) or not _REF_RE.fullmatch(value) or not value.startswith(prefix + "-"):
        raise AssignmentError(f"minimized trace contains an invalid {prefix.lower()} reference")
    return value


def _validate_trace_data(event_type: EventType, value: Any) -> dict[str, Any]:
    if event_type == EventType.TOOL_CALL:
        data = _require_exact_keys(
            value,
            ("arguments_omitted", "content_omitted", "tool_category", "tool_ref"),
            ("resource_ref",),
            label="tool-call data",
        )
        if data["arguments_omitted"] is not True or data["content_omitted"] is not True:
            raise AssignmentError("tool-call omission flags are invalid")
        if data["tool_category"] not in _TOOL_CATEGORIES:
            raise AssignmentError("tool-call category is invalid")
        _validate_ref(data["tool_ref"], "Tool")
        if "resource_ref" in data:
            _validate_ref(data["resource_ref"], "Resource")
        return data
    if event_type == EventType.TOOL_RESULT:
        data = _require_exact_keys(
            value,
            ("content_omitted", "outcome_recorded"),
            ("is_error",),
            label="tool-result data",
        )
        if data["content_omitted"] is not True or data["outcome_recorded"] is not True:
            raise AssignmentError("tool-result flags are invalid")
        if "is_error" in data and not isinstance(data["is_error"], bool):
            raise AssignmentError("tool-result error flag is invalid")
        return data
    if event_type in {EventType.FILE_READ, EventType.FILE_WRITE}:
        data = _require_exact_keys(
            value, ("content_omitted",), ("resource_ref",), label="file-event data"
        )
        if data["content_omitted"] is not True:
            raise AssignmentError("file-event omission flag is invalid")
        if "resource_ref" in data:
            _validate_ref(data["resource_ref"], "Resource")
        return data
    if event_type in {EventType.LLM_REQUEST, EventType.LLM_RESPONSE}:
        data = _require_exact_keys(
            value,
            ("content_omitted",),
            ("model_ref", "input_tokens", "output_tokens", "total_tokens"),
            label="model-event data",
        )
        if data["content_omitted"] is not True:
            raise AssignmentError("model-event omission flag is invalid")
        if "model_ref" in data:
            _validate_ref(data["model_ref"], "Model")
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            if key in data:
                _number(data[key], label=key, maximum=MAX_TOKEN_COUNT, integer=True)
        return data
    if event_type == EventType.ERROR:
        data = _require_exact_keys(
            value, ("content_omitted", "error_recorded"), label="error-event data"
        )
        if data["content_omitted"] is not True or data["error_recorded"] is not True:
            raise AssignmentError("error-event flags are invalid")
        return data
    if event_type == EventType.SESSION_END:
        data = _require_exact_keys(
            value, ("content_omitted", "recorded_status"), label="session-end data"
        )
        if data["content_omitted"] is not True or data["recorded_status"] not in _RECORDED_STATUSES:
            raise AssignmentError("session-end status is invalid")
        return data
    data = _require_exact_keys(value, ("content_omitted",), label="event data")
    if data["content_omitted"] is not True:
        raise AssignmentError("event omission flag is invalid")
    return data


def _parse_trace(raw: bytes) -> list[dict[str, Any]]:
    if len(raw) > MAX_TRACE_BYTES:
        raise AssignmentError("minimized trace exceeds the supported size")
    if not raw:
        return []
    if not raw.endswith(b"\n"):
        raise AssignmentError("minimized trace must end with a newline")
    lines = raw.splitlines()
    if len(lines) > MAX_EVENTS:
        raise AssignmentError("minimized trace has too many events")
    events: list[dict[str, Any]] = []
    seen_refs: set[str] = set()
    anonymous_refs: dict[str, set[str]] = {
        "Tool": set(),
        "Resource": set(),
        "Model": set(),
    }
    for index, line in enumerate(lines, 1):
        if not line or len(line) > MAX_SOURCE_EVENT_BYTES:
            raise AssignmentError("minimized trace line size is invalid")
        value = _strict_json_bytes(line, label="minimized trace event", max_bytes=MAX_SOURCE_EVENT_BYTES)
        item = _require_exact_keys(
            value,
            ("schema", "sequence", "event_ref", "event_type", "offset_ms", "data"),
            ("duration_ms", "parent_ref"),
            label="minimized trace event",
        )
        if item["schema"] != TRACE_EVENT_SCHEMA:
            raise AssignmentError("minimized trace schema is unsupported")
        if _number(item["sequence"], label="event sequence", maximum=MAX_EVENTS, integer=True) != index:
            raise AssignmentError("minimized trace sequence is not canonical")
        event_ref = _validate_ref(item["event_ref"], "Event")
        if event_ref != f"Event-{index:06d}" or event_ref in seen_refs:
            raise AssignmentError("minimized trace event references are not canonical")
        if "parent_ref" in item:
            parent_ref = _validate_ref(item["parent_ref"], "Event")
            if parent_ref not in seen_refs:
                raise AssignmentError("minimized trace parent reference is not an earlier event")
        seen_refs.add(event_ref)
        try:
            event_type = EventType(item["event_type"])
        except (TypeError, ValueError) as exc:
            raise AssignmentError("minimized trace event type is invalid") from exc
        _number(
            item["offset_ms"],
            label="event offset",
            maximum=MAX_DURATION_SECONDS * 1000,
            integer=True,
        )
        if "duration_ms" in item:
            _number(
                item["duration_ms"],
                label="event duration",
                maximum=MAX_DURATION_SECONDS * 1000,
                integer=True,
            )
        data = _validate_trace_data(event_type, item["data"])
        for key, prefix in (
            ("tool_ref", "Tool"),
            ("resource_ref", "Resource"),
            ("model_ref", "Model"),
        ):
            if key not in data:
                continue
            reference = data[key]
            known = anonymous_refs[prefix]
            if reference not in known:
                expected = f"{prefix}-{len(known) + 1:03d}"
                if reference != expected:
                    raise AssignmentError(
                        "minimized trace anonymous references are not canonical"
                    )
                known.add(reference)
        if line != _canonical_json_bytes(item):
            raise AssignmentError("minimized trace encoding is not canonical")
        events.append(item)
    return events


def _lint_payload(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the frozen, privacy-minimized assignment-v1 lint signals."""
    findings: list[dict[str, Any]] = []

    def add(rule: str, level: str, start: int | None = None, end: int | None = None) -> None:
        entry: dict[str, Any] = {
            "level": level,
            "message": _LINT_MESSAGES[rule],
            "rule": rule,
        }
        if start is not None:
            entry["event_start"] = start
        if end is not None:
            entry["event_end"] = end
        findings.append(entry)

    # tool-loop v1: five consecutive calls with the same anonymous tool ref.
    run_ref: str | None = None
    run_start = 0
    run_length = 0
    for index, item in enumerate(events, 1):
        if item["event_type"] != EventType.TOOL_CALL.value:
            if run_length >= 5:
                add("tool-loop", "WARN", run_start, run_start + run_length - 1)
            run_ref, run_length = None, 0
            continue
        reference = item["data"]["tool_ref"]
        if reference == run_ref:
            run_length += 1
        else:
            if run_length >= 5:
                add("tool-loop", "WARN", run_start, run_start + run_length - 1)
            run_ref, run_start, run_length = reference, index, 1
    if run_length >= 5:
        add("tool-loop", "WARN", run_start, run_start + run_length - 1)

    # reasoning-spiral v1 mirrors the public rule's three model/assistant
    # events without an intervening tool call. Other event types do not reset.
    model_types = {
        EventType.LLM_REQUEST.value,
        EventType.LLM_RESPONSE.value,
        EventType.ASSISTANT_RESPONSE.value,
    }
    run_start, run_length = 0, 0
    for index, item in enumerate(events, 1):
        if item["event_type"] in model_types:
            if run_length == 0:
                run_start = index
            run_length += 1
        elif item["event_type"] == EventType.TOOL_CALL.value:
            if run_length >= 3:
                add("reasoning-spiral", "WARN", run_start, run_start + run_length - 1)
            run_length = 0
    if run_length >= 3:
        add("reasoning-spiral", "WARN", run_start, run_start + run_length - 1)

    # context-saturation v1 uses only allowlisted request token fields and a
    # frozen 200k context / 80% threshold. It reports once.
    cumulative_input = 0
    for index, item in enumerate(events, 1):
        if item["event_type"] == EventType.LLM_REQUEST.value:
            cumulative_input += int(item["data"].get("input_tokens", 0))
            if cumulative_input >= 160_000:
                add("context-saturation", "INFO", index)
                break

    # resource-repeat v1: a finding for each resource read at least 3 times.
    read_counts: dict[str, int] = {}
    for reference in _read_resource_refs(events):
        read_counts[reference] = read_counts.get(reference, 0) + 1
    for reference in sorted(read_counts):
        if read_counts[reference] >= 3:
            add("redundant-read", "INFO")

    # error-retry-loop v1 associates explicit errors with the most recent tool.
    last_tool: str | None = None
    error_counts: dict[str, int] = {}
    for item in events:
        if item["event_type"] == EventType.TOOL_CALL.value:
            last_tool = item["data"]["tool_ref"]
        elif item["event_type"] == EventType.ERROR.value and last_tool:
            error_counts[last_tool] = error_counts.get(last_tool, 0) + 1
        elif (
            item["event_type"] == EventType.TOOL_RESULT.value
            and item["data"].get("is_error") is True
            and last_tool
        ):
            error_counts[last_tool] = error_counts.get(last_tool, 0) + 1
    for reference in sorted(error_counts):
        if error_counts[reference] >= 3:
            add("error-retry-loop", "WARN")

    has_end = any(item["event_type"] == EventType.SESSION_END.value for item in events)
    has_write = any(
        item["event_type"] == EventType.FILE_WRITE.value
        or (
            item["event_type"] == EventType.TOOL_CALL.value
            and item["data"]["tool_category"] == "write"
        )
        for item in events
    )
    if has_end and not has_write:
        add("no-output", "WARN")

    # budget-proximity and post-compaction-regression intentionally have no
    # v1 signal: the required raw budget/compaction context is privacy-omitted.
    findings.sort(key=lambda item: (
        {"ERROR": 0, "WARN": 1, "INFO": 2}[item["level"]],
        item["rule"],
        item.get("event_start", MAX_EVENTS + 1),
        item.get("event_end", MAX_EVENTS + 1),
    ))
    rule_counts = {
        rule: sum(1 for finding in findings if finding["rule"] == rule)
        for rule in sorted(_ASSIGNMENT_SCORABLE_LINT_RULES)
    }
    return {
        "schema": LINT_SCHEMA,
        "session_ref": "Session-001",
        "finding_count": len(findings),
        "errors": sum(1 for item in findings if item["level"] == "ERROR"),
        "warnings": sum(1 for item in findings if item["level"] == "WARN"),
        "infos": sum(1 for item in findings if item["level"] == "INFO"),
        "rule_counts": rule_counts,
        "unavailable_rules": list(_ASSIGNMENT_UNAVAILABLE_LINT_RULES),
        "findings": findings,
        "limitation": "Findings are deterministic process signals over minimized telemetry, not candidate-quality or task-correctness judgments.",
    }


def _cost_payload(events: list[TraceEvent]) -> dict[str, Any]:
    input_tokens = 0
    output_tokens = 0
    for event in events:
        serialized = json.dumps(
            event.data,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        estimate = max(1, len(serialized) // 4)
        incoming = estimate if event.event_type in _ASSIGNMENT_V1_INPUT_TYPES else 0
        outgoing = estimate if event.event_type in _ASSIGNMENT_V1_OUTPUT_TYPES else 0
        input_tokens += incoming
        output_tokens += outgoing
    if input_tokens > MAX_TOKEN_COUNT or output_tokens > MAX_TOKEN_COUNT:
        raise AssignmentError("aggregate token estimate exceeds assignment bounds")
    profile_name = _CURRENT_ASSIGNMENT_PRICING_PROFILE
    rates = _ASSIGNMENT_PRICING_PROFILES[profile_name]
    total = round(
        input_tokens / 1_000_000 * rates["input_rate_per_million"]
        + output_tokens / 1_000_000 * rates["output_rate_per_million"],
        12,
    )
    return {
        "schema": COST_SCHEMA,
        "currency": "USD",
        "estimated": True,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_rate_per_million": rates["input_rate_per_million"],
        "output_rate_per_million": rates["output_rate_per_million"],
        "total_cost_usd": total,
        "pricing_profile": profile_name,
        "pricing_snapshot_date": rates["pricing_snapshot_date"],
        "limitation": "Offline estimate only; aggregate token estimates are derived before content minimization and are not provider billing records.",
    }


def _read_resource_refs(events: list[dict[str, Any]]) -> list[str]:
    refs: list[str] = []
    for item in events:
        event_type = EventType(item["event_type"])
        data = item["data"]
        if event_type == EventType.FILE_READ and "resource_ref" in data:
            refs.append(data["resource_ref"])
        elif (
            event_type == EventType.TOOL_CALL
            and data.get("tool_category") == "read"
            and "resource_ref" in data
        ):
            refs.append(data["resource_ref"])
    return refs


def _stats_payload(
    events: list[dict[str, Any]],
    cost: dict[str, Any],
    lint: dict[str, Any],
) -> dict[str, Any]:
    offsets = [item["offset_ms"] for item in events]
    duration = round((max(offsets) - min(offsets)) / 1000.0, 3) if offsets else 0.0
    end_statuses = [
        item["data"]["recorded_status"]
        for item in events
        if item["event_type"] == EventType.SESSION_END.value
    ]
    status = end_statuses[-1] if end_statuses else "not_recorded"
    reads = _read_resource_refs(events)
    seen: set[str] = set()
    repeated = 0
    for resource in reads:
        if resource in seen:
            repeated += 1
        else:
            seen.add(resource)
    models = {
        item["data"]["model_ref"]
        for item in events
        if "model_ref" in item["data"]
    }
    return {
        "schema": STATS_SCHEMA,
        "session_ref": "Session-001",
        "event_count": len(events),
        "duration_seconds": duration,
        "session_status": status,
        "tool_call_count": sum(
            1 for item in events if item["event_type"] == EventType.TOOL_CALL.value
        ),
        "error_count": sum(
            1 for item in events if item["event_type"] == EventType.ERROR.value
        ),
        "redundant_read_ratio": round(repeated / len(reads), 12) if reads else 0.0,
        "model_count": len(models),
        "model_refs": sorted(models),
        "estimated_cost_usd": cost["total_cost_usd"],
        "lint_violation_count": lint["finding_count"],
        "metric_semantics": dict(_METRIC_SEMANTICS),
    }


def _event_summary(item: dict[str, Any]) -> str:
    event_type = item["event_type"]
    data = item["data"]
    if event_type == EventType.TOOL_CALL.value:
        resource = f"; {data['resource_ref']}" if "resource_ref" in data else ""
        return f"{data['tool_category']} ({data['tool_ref']}{resource}); arguments omitted"
    if event_type == EventType.TOOL_RESULT.value:
        return "recorded tool outcome; content omitted"
    if event_type in {EventType.FILE_READ.value, EventType.FILE_WRITE.value}:
        return f"{data.get('resource_ref', 'anonymous resource')}; content omitted"
    if event_type in {EventType.LLM_REQUEST.value, EventType.LLM_RESPONSE.value}:
        bits = [data.get("model_ref", "anonymous model")]
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            if key in data:
                bits.append(f"{key}={data[key]}")
        return "; ".join(bits) + "; content omitted"
    if event_type == EventType.ERROR.value:
        return "recorded error; content omitted"
    if event_type == EventType.SESSION_END.value:
        return f"recorded process status: {data['recorded_status']}"
    return "content omitted"


def _render_assignment_html(
    events: list[dict[str, Any]],
    stats: dict[str, Any],
    cost: dict[str, Any],
    lint: dict[str, Any],
) -> bytes:
    rows = []
    for item in events:
        details = html.escape(
            json.dumps(item["data"], sort_keys=True, indent=2, ensure_ascii=False),
            quote=True,
        )
        event_type = html.escape(item["event_type"], quote=True)
        summary = html.escape(_event_summary(item), quote=True)
        rows.append(
            '<details class="event" data-type="{event_type}">'
            '<summary><span class="ts">+{offset:.3f}s</span>'
            '<span class="badge badge-white">{event_type}</span>'
            '<span class="event-summary">{summary}</span></summary>'
            '<pre class="event-detail">{details}</pre></details>'.format(
                event_type=event_type,
                offset=item["offset_ms"] / 1000.0,
                summary=summary,
                details=details,
            )
        )
    event_html = "\n".join(rows)
    limitation_items = "".join(f"<li>{html.escape(text)}</li>" for text in _LIMITATIONS)
    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>agent-strace assignment replay</title>
<style>{_ASSIGNMENT_CSS_V1}</style>
</head>
<body>
<div class="header">
  <h1>agent-strace minimized assignment replay</h1>
  <div class="meta-grid">
    <div class="meta-item"><div class="meta-label">Session</div><div class="meta-value">Session-001</div></div>
    <div class="meta-item"><div class="meta-label">Events</div><div class="meta-value">{stats['event_count']}</div></div>
    <div class="meta-item"><div class="meta-label">Duration</div><div class="meta-value">{stats['duration_seconds']:.3f}s</div></div>
    <div class="meta-item"><div class="meta-label">Recorded status</div><div class="meta-value">{html.escape(stats['session_status'])}</div></div>
    <div class="meta-item"><div class="meta-label">Estimated cost</div><div class="meta-value">${cost['total_cost_usd']:.6f}</div></div>
    <div class="meta-item"><div class="meta-label">Lint signals</div><div class="meta-value">{lint['finding_count']}</div></div>
  </div>
</div>
<p>All source prompts, results, commands, paths, file contents, environment data, identities, and custom model identifiers were omitted before this viewer was built.</p>
<div class="search-bar" id="filter-bar">
  <input id="search-input" class="search-input" type="search" placeholder="Search minimized events" autocomplete="off" spellcheck="false">
  <div class="filter-chips"></div><span class="search-count" id="search-count"></span>
</div>
<h2>Minimized event stream</h2>
<details class="phase" open><summary class="phase-header">Recorded process telemetry</summary><div class="events-list">{event_html}</div></details>
<h2>Interpretation limitations</h2><ul>{limitation_items}</ul>
<div class="footer">Generated deterministically from the sanitized trace copy. No external resources.</div>
<script>{_ASSIGNMENT_JS_V1}</script>
</body>
</html>"""
    raw = (document + "\n").encode("utf-8")
    if len(raw) > MAX_HTML_BYTES:
        raise AssignmentError("assignment replay exceeds the bundle size limit")
    return raw


def _manifest_core(
    stats: dict[str, Any], cost: dict[str, Any], lint: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema": BUNDLE_SCHEMA,
        "bundle_version": 1,
        "producer": {"name": "agent-strace", "version": __version__},
        "submission_ref": "Submission",
        "payload_members": list(PAYLOAD_MEMBERS),
        "summary": {
            "session_ref": stats["session_ref"],
            "event_count": stats["event_count"],
            "duration_seconds": stats["duration_seconds"],
            "session_status": stats["session_status"],
            "tool_call_count": stats["tool_call_count"],
            "error_count": stats["error_count"],
            "redundant_read_ratio": stats["redundant_read_ratio"],
            "model_count": stats["model_count"],
            "estimated_cost_usd": cost["total_cost_usd"],
            "lint_violation_count": lint["finding_count"],
        },
        "privacy": {
            "relative_timestamps": True,
            "anonymous_event_tool_resource_and_model_refs": True,
            "raw_identity_omitted": True,
            "prompts_results_commands_paths_file_contents_environment_urls_omitted": True,
        },
        "integrity": {
            "algorithm": "SHA-256",
            "member_digest_scope": "exact uncompressed payload bytes",
            "bundle_digest_scope": "canonical manifest core and member digests; manifest itself excluded",
            "authenticity": "not_provided",
        },
        "limitations": list(_LIMITATIONS),
    }


def _bundle_digest(core: dict[str, Any], member_digests: dict[str, str]) -> str:
    return _digest(_canonical_json_bytes({
        "manifest_core": core,
        "member_sha256": member_digests,
    }))


def build_assignment_bundle(store: TraceStore, session_id: str) -> bytes:
    """Build deterministic assignment ZIP bytes for one resolved session ID."""
    _meta, raw_events = _snapshot_session(store, session_id)
    minimized = _sanitize_events(raw_events)
    trace_raw = _encode_trace(minimized)
    cost = _cost_payload(raw_events)
    lint = _lint_payload(minimized)
    stats = _stats_payload(minimized, cost, lint)
    payloads: dict[str, bytes] = {
        "trace.ndjson": trace_raw,
        "stats.json": _json_file_bytes(stats),
        "cost.json": _json_file_bytes(cost),
        "lint.json": _json_file_bytes(lint),
        "replay.html": _render_assignment_html(minimized, stats, cost, lint),
    }
    member_digests = {name: _digest(payloads[name]) for name in PAYLOAD_MEMBERS}
    core = _manifest_core(stats, cost, lint)
    manifest = {
        **core,
        "member_sha256": member_digests,
        "bundle_digest": _bundle_digest(core, member_digests),
    }
    members = {"manifest.json": _json_file_bytes(manifest), **payloads}

    output = io.BytesIO()
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9, strict_timestamps=True
    ) as archive:
        archive.comment = b""
        for name in ZIP_MEMBERS:
            info = zipfile.ZipInfo(name, date_time=ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            info.internal_attr = 0
            info.extra = b""
            info.comment = b""
            archive.writestr(info, members[name], compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    data = output.getvalue()
    if len(data) > MAX_ARCHIVE_BYTES:
        raise AssignmentError("assignment bundle exceeds the archive size limit")
    # Exercise the same no-extraction hostile-input validator used by score so
    # generator metadata, compression, digests, and derived payloads cannot
    # silently drift apart.
    load_assignment_bundle(data)
    return data


def cmd_share_assignment(args: argparse.Namespace) -> int:
    """CLI handler used by ``share --assignment``."""
    if any(
        getattr(args, name, False)
        for name in ("stdout", "open", "postmortem")
    ):
        sys.stderr.write(
            "Error: --assignment cannot be combined with --stdout, --open, or --postmortem\n"
        )
        return 1
    try:
        store = TraceStore(args.trace_dir)
        session_id = getattr(args, "session_id", None) or store.get_latest_session_id()
        if not session_id:
            raise AssignmentError("no sessions were found")
        full_id = store.find_session(session_id)
        if not full_id:
            raise AssignmentError("the requested session was not found")
        bundle = build_assignment_bundle(store, full_id)
        output = getattr(args, "output", None) or "assignment-submission.zip"
        if Path(output).suffix != ".zip":
            raise AssignmentError("assignment output must use a .zip suffix")
        _atomic_write_private(output, bundle)
        sys.stderr.write(f"Created assignment bundle: {output} ({len(bundle) // 1024}KB)\n")
        return 0
    except (AssignmentError, OSError, ValueError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1


# ---------------------------------------------------------------------------
# Hostile bundle validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedAssignment:
    bundle_digest: str
    archive_sha256: str
    events: list[dict[str, Any]]
    stats: dict[str, Any]
    cost: dict[str, Any]
    lint: dict[str, Any]


def _validate_zip_envelope(raw: bytes) -> tuple[int, int]:
    if len(raw) < 22 or len(raw) > MAX_ARCHIVE_BYTES or not raw.startswith(b"PK\x03\x04"):
        raise AssignmentError("submission archive envelope is invalid")
    if raw[-22:-18] != b"PK\x05\x06":
        raise AssignmentError("submission archive has a preamble, trailer, or comment")
    try:
        (
            _signature,
            disk_number,
            central_disk,
            entries_on_disk,
            total_entries,
            central_size,
            central_offset,
            comment_length,
        ) = struct.unpack("<4s4H2LH", raw[-22:])
    except struct.error as exc:
        raise AssignmentError("submission archive envelope is invalid") from exc
    if (
        disk_number != 0
        or central_disk != 0
        or entries_on_disk != len(ZIP_MEMBERS)
        or total_entries != len(ZIP_MEMBERS)
        or comment_length != 0
        or central_offset <= 0
        or central_offset + central_size != len(raw) - 22
    ):
        raise AssignmentError("submission archive layout is not canonical")
    return central_offset, central_size


def _validate_zip_spans(
    raw: bytes,
    infos: list[zipfile.ZipInfo],
    central_offset: int,
    central_size: int,
) -> None:
    """Reject bytes hidden outside the six exact local/central ZIP records."""
    cursor = 0
    for info in infos:
        if info.header_offset != cursor or cursor + 30 > central_offset:
            raise AssignmentError("submission archive local-member spans are not canonical")
        try:
            (
                signature,
                extract_version,
                flags,
                method,
                dos_time,
                dos_date,
                crc,
                compressed_size,
                file_size,
                name_length,
                extra_length,
            ) = struct.unpack("<4s5H3L2H", raw[cursor:cursor + 30])
        except struct.error as exc:
            raise AssignmentError("submission archive local header is truncated") from exc
        name_start = cursor + 30
        name_end = name_start + name_length
        data_start = name_end + extra_length
        data_end = data_start + compressed_size
        if (
            signature != b"PK\x03\x04"
            or extract_version != ZIP_EXTRACT_VERSION
            or flags != info.flag_bits
            or method != info.compress_type
            or dos_time != ZIP_DOS_TIME
            or dos_date != ZIP_DOS_DATE
            or crc != info.CRC
            or compressed_size != info.compress_size
            or file_size != info.file_size
            or extra_length != 0
            or raw[name_start:name_end] != info.filename.encode("ascii")
            or data_end > central_offset
        ):
            raise AssignmentError("submission archive local header is not canonical")
        cursor = data_end
    if cursor != central_offset:
        raise AssignmentError("submission archive contains bytes outside canonical members")

    cursor = central_offset
    central_end = central_offset + central_size
    for info in infos:
        if cursor + 46 > central_end:
            raise AssignmentError("submission archive central directory is truncated")
        try:
            values = struct.unpack("<4s6H3L5H2L", raw[cursor:cursor + 46])
        except struct.error as exc:
            raise AssignmentError("submission archive central header is truncated") from exc
        (
            signature,
            create_version,
            extract_version,
            flags,
            method,
            dos_time,
            dos_date,
            crc,
            compressed_size,
            file_size,
            name_length,
            extra_length,
            comment_length,
            disk_start,
            internal_attr,
            external_attr,
            local_offset,
        ) = values
        name_start = cursor + 46
        name_end = name_start + name_length
        record_end = name_end + extra_length + comment_length
        if (
            signature != b"PK\x01\x02"
            or create_version != ZIP_CREATE_VERSION
            or extract_version != ZIP_EXTRACT_VERSION
            or flags != info.flag_bits
            or method != info.compress_type
            or dos_time != ZIP_DOS_TIME
            or dos_date != ZIP_DOS_DATE
            or crc != info.CRC
            or compressed_size != info.compress_size
            or file_size != info.file_size
            or extra_length != 0
            or comment_length != 0
            or disk_start != 0
            or internal_attr != info.internal_attr
            or external_attr != info.external_attr
            or local_offset != info.header_offset
            or raw[name_start:name_end] != info.filename.encode("ascii")
            or record_end > central_end
        ):
            raise AssignmentError("submission archive central header is not canonical")
        cursor = record_end
    if cursor != central_end:
        raise AssignmentError("submission archive central directory has hidden bytes")


def _validate_member_name(name: str) -> None:
    if (
        not name
        or "\\" in name
        or "\x00" in name
        or name.startswith(("/", "~"))
        or re.match(r"^[A-Za-z]:", name)
    ):
        raise AssignmentError("submission archive contains an unsafe member name")
    pure = PurePosixPath(name)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise AssignmentError("submission archive contains an unsafe member name")


def _inflate_exact_member(raw: bytes, info: zipfile.ZipInfo) -> bytes:
    """Consume every declared raw-DEFLATE byte and validate its exact output."""
    try:
        name_length, extra_length = struct.unpack(
            "<2H", raw[info.header_offset + 26:info.header_offset + 30]
        )
    except struct.error as exc:
        raise AssignmentError("submission archive local header is truncated") from exc
    data_start = info.header_offset + 30 + name_length + extra_length
    compressed = raw[data_start:data_start + info.compress_size]
    if len(compressed) != info.compress_size:
        raise AssignmentError("submission archive compressed member is truncated")
    inflater = zlib.decompressobj(-zlib.MAX_WBITS)
    try:
        value = inflater.decompress(compressed, info.file_size + 1)
    except zlib.error as exc:
        raise AssignmentError("submission archive DEFLATE stream is invalid") from exc
    if (
        not inflater.eof
        or inflater.unused_data
        or inflater.unconsumed_tail
        or len(value) != info.file_size
        or (zlib.crc32(value) & 0xFFFFFFFF) != info.CRC
    ):
        raise AssignmentError("submission archive DEFLATE stream is not canonical")
    return value


def _read_zip_members(raw: bytes) -> dict[str, bytes]:
    central_offset, central_size = _validate_zip_envelope(raw)
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw), "r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise AssignmentError("submission archive is not a valid ZIP") from exc
    with archive:
        if archive.comment:
            raise AssignmentError("submission archive comments are not allowed")
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if names != list(ZIP_MEMBERS):
            raise AssignmentError("submission archive members or order are not canonical")
        if len(names) != len(set(names)) or len({name.casefold() for name in names}) != len(names):
            raise AssignmentError("submission archive contains duplicate member names")
        _validate_zip_spans(raw, infos, central_offset, central_size)

        total = 0
        payloads: dict[str, bytes] = {}
        for info in infos:
            _validate_member_name(info.filename)
            if info.is_dir() or info.filename not in _MEMBER_LIMITS:
                raise AssignmentError("submission archive contains an unsupported member")
            mode = (info.external_attr >> 16) & 0xFFFF
            if (
                info.create_system != 3
                or stat.S_IFMT(mode) != stat.S_IFREG
                or stat.S_IMODE(mode) != 0o600
                or info.date_time != ZIP_TIMESTAMP
                or info.compress_type != zipfile.ZIP_DEFLATED
                or info.flag_bits != 0
                or info.extra
                or info.comment
                or info.internal_attr != 0
            ):
                raise AssignmentError("submission archive member metadata is not canonical")
            if (
                info.file_size < 0
                or (info.file_size == 0 and info.filename != "trace.ndjson")
                or info.file_size > _MEMBER_LIMITS[info.filename]
            ):
                raise AssignmentError("submission archive member size is invalid")
            if info.compress_size <= 0 or (
                info.file_size > 0
                and info.file_size / info.compress_size > MAX_COMPRESSION_RATIO
            ):
                raise AssignmentError("submission archive compression ratio is unsafe")
            total += info.file_size
            if total > MAX_TOTAL_UNCOMPRESSED:
                raise AssignmentError("submission archive expands beyond the supported size")
            inflated = _inflate_exact_member(raw, info)
            try:
                with archive.open(info, "r") as member:
                    chunks: list[bytes] = []
                    remaining = _MEMBER_LIMITS[info.filename] + 1
                    while remaining > 0:
                        chunk = member.read(min(64 * 1024, remaining))
                        if not chunk:
                            break
                        chunks.append(chunk)
                        remaining -= len(chunk)
                    value = b"".join(chunks)
                    if member.read(1) or len(value) != info.file_size:
                        raise AssignmentError("submission archive member size changed while reading")
            except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                raise AssignmentError("submission archive member could not be read safely") from exc
            if value != inflated:
                raise AssignmentError("submission archive member decoding is inconsistent")
            payloads[info.filename] = value
        return payloads


def _validate_cost(value: Any) -> dict[str, Any]:
    expected_keys = (
        "schema", "currency", "estimated", "input_tokens", "output_tokens",
        "input_rate_per_million", "output_rate_per_million", "total_cost_usd",
        "pricing_profile", "pricing_snapshot_date", "limitation",
    )
    cost = _require_exact_keys(value, expected_keys, label="cost payload")
    if cost["schema"] != COST_SCHEMA or cost["currency"] != "USD" or cost["estimated"] is not True:
        raise AssignmentError("cost payload identity is invalid")
    input_tokens = _number(
        cost["input_tokens"], label="input token estimate", maximum=MAX_TOKEN_COUNT, integer=True
    )
    output_tokens = _number(
        cost["output_tokens"], label="output token estimate", maximum=MAX_TOKEN_COUNT, integer=True
    )
    input_rate = _number(
        cost["input_rate_per_million"], label="input token rate", maximum=1_000_000
    )
    output_rate = _number(
        cost["output_rate_per_million"], label="output token rate", maximum=1_000_000
    )
    total = _number(cost["total_cost_usd"], label="estimated cost", maximum=1_000_000_000)
    expected_total = round(
        input_tokens / 1_000_000 * input_rate
        + output_tokens / 1_000_000 * output_rate,
        12,
    )
    if total != expected_total:
        raise AssignmentError("cost payload arithmetic is inconsistent")
    profile_name = cost["pricing_profile"]
    if not isinstance(profile_name, str) or profile_name not in _ASSIGNMENT_PRICING_PROFILES:
        raise AssignmentError("cost pricing profile is unsupported")
    expected_rates = _ASSIGNMENT_PRICING_PROFILES[profile_name]
    if (
        input_rate != expected_rates["input_rate_per_million"]
        or output_rate != expected_rates["output_rate_per_million"]
        or cost["pricing_snapshot_date"] != expected_rates["pricing_snapshot_date"]
    ):
        raise AssignmentError("cost pricing profile rates or snapshot date are invalid")
    if cost["limitation"] != (
        "Offline estimate only; aggregate token estimates are derived before content "
        "minimization and are not provider billing records."
    ):
        raise AssignmentError("cost payload limitation is not canonical")
    return cost


def _validate_manifest(
    value: Any,
    manifest_raw: bytes,
    payloads: dict[str, bytes],
    stats: dict[str, Any],
    cost: dict[str, Any],
    lint: dict[str, Any],
) -> str:
    manifest = _require_exact_keys(
        value,
        (
            "schema", "bundle_version", "producer", "submission_ref",
            "payload_members", "summary", "privacy", "integrity", "limitations",
            "member_sha256", "bundle_digest",
        ),
        label="assignment manifest",
    )
    if manifest["schema"] != BUNDLE_SCHEMA or manifest["bundle_version"] != 1:
        raise AssignmentError("assignment manifest schema is unsupported")
    producer = _require_exact_keys(
        manifest["producer"], ("name", "version"), label="manifest producer"
    )
    if (
        producer["name"] != "agent-strace"
        or not isinstance(producer["version"], str)
        or not _SEMVER_RE.fullmatch(producer["version"])
    ):
        raise AssignmentError("manifest producer is invalid")
    if manifest["submission_ref"] != "Submission" or manifest["payload_members"] != list(PAYLOAD_MEMBERS):
        raise AssignmentError("assignment manifest payload declaration is invalid")
    digests = _require_exact_keys(
        manifest["member_sha256"], PAYLOAD_MEMBERS, label="manifest member digests"
    )
    for name in PAYLOAD_MEMBERS:
        if not isinstance(digests[name], str) or not _SHA256_RE.fullmatch(digests[name]):
            raise AssignmentError("manifest member digest is invalid")
        if not secrets.compare_digest(digests[name], _digest(payloads[name])):
            raise AssignmentError("submission payload digest mismatch")

    expected_core = _manifest_core(stats, cost, lint)
    expected_core["producer"]["version"] = producer["version"]
    actual_core = {key: manifest[key] for key in expected_core}
    if actual_core != expected_core:
        raise AssignmentError("assignment manifest fields are not canonical")
    if not isinstance(manifest["bundle_digest"], str) or not _SHA256_RE.fullmatch(
        manifest["bundle_digest"]
    ):
        raise AssignmentError("manifest bundle digest is invalid")
    expected_digest = _bundle_digest(expected_core, digests)
    if not secrets.compare_digest(manifest["bundle_digest"], expected_digest):
        raise AssignmentError("manifest bundle digest mismatch")
    expected_manifest = {
        **expected_core,
        "member_sha256": digests,
        "bundle_digest": expected_digest,
    }
    if manifest_raw != _json_file_bytes(expected_manifest):
        raise AssignmentError("assignment manifest encoding or primitive types are not canonical")
    return expected_digest


def load_assignment_bundle(raw: bytes) -> LoadedAssignment:
    """Validate an untrusted assignment ZIP without extracting or opening HTML."""
    payloads = _read_zip_members(raw)
    manifest_value = _strict_json_bytes(
        payloads["manifest.json"], label="assignment manifest", max_bytes=MAX_JSON_BYTES
    )

    # Only the member-digest object is consulted before payload verification.
    # All remaining manifest values stay untrusted until recomputation below.
    if not isinstance(manifest_value, dict):
        raise AssignmentError("assignment manifest must be an object")
    digest_value = manifest_value.get("member_sha256")
    digests = _require_exact_keys(
        digest_value, PAYLOAD_MEMBERS, label="manifest member digests"
    )
    for name in PAYLOAD_MEMBERS:
        claimed = digests[name]
        if not isinstance(claimed, str) or not _SHA256_RE.fullmatch(claimed):
            raise AssignmentError("manifest member digest is invalid")
        if not secrets.compare_digest(claimed, _digest(payloads[name])):
            raise AssignmentError("submission payload digest mismatch")

    events = _parse_trace(payloads["trace.ndjson"])
    cost_value = _strict_json_bytes(
        payloads["cost.json"], label="cost payload", max_bytes=MAX_JSON_BYTES
    )
    cost = _validate_cost(cost_value)
    if payloads["cost.json"] != _json_file_bytes(cost):
        raise AssignmentError("cost payload encoding or primitive types are not canonical")
    lint_value = _strict_json_bytes(
        payloads["lint.json"], label="lint payload", max_bytes=MAX_JSON_BYTES
    )
    expected_lint = _lint_payload(events)
    if payloads["lint.json"] != _json_file_bytes(expected_lint):
        raise AssignmentError("lint payload does not match the minimized trace")
    lint = expected_lint
    stats_value = _strict_json_bytes(
        payloads["stats.json"], label="stats payload", max_bytes=MAX_JSON_BYTES
    )
    expected_stats = _stats_payload(events, cost, lint)
    if payloads["stats.json"] != _json_file_bytes(expected_stats):
        raise AssignmentError("stats payload does not match the minimized trace and aggregates")
    stats = expected_stats
    expected_html = _render_assignment_html(events, stats, cost, lint)
    if not secrets.compare_digest(payloads["replay.html"], expected_html):
        raise AssignmentError("replay payload is not the canonical minimized viewer")
    bundle_digest = _validate_manifest(
        manifest_value, payloads["manifest.json"], payloads, stats, cost, lint
    )
    return LoadedAssignment(
        bundle_digest=bundle_digest,
        archive_sha256=_digest(raw),
        events=events,
        stats=stats,
        cost=cost,
        lint=lint,
    )


# ---------------------------------------------------------------------------
# Strict stdlib-only rubric parsing
# ---------------------------------------------------------------------------


SUPPORTED_SCORERS = (
    "session_status",
    "cost_usd",
    "duration_seconds",
    "error_count",
    "tool_call_count",
    "redundant_read_ratio",
    "lint_violations",
)
_NUMERIC_SCORERS = set(SUPPORTED_SCORERS) - {"session_status"}


@dataclass(frozen=True)
class AssignmentCriterion:
    name: str
    scorer: str
    weight: int
    fail_on: str
    threshold: float | int | None = None
    expected: str | None = None
    rule: str | None = None


@dataclass(frozen=True)
class AssignmentRubric:
    task: str
    criteria: tuple[AssignmentCriterion, ...]
    max_cost_usd: float | None = None
    max_duration_minutes: float | None = None


def _safe_rubric_text(value: Any, *, label: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise AssignmentError(f"{label} must be text")
    text = value.strip()
    if not text or len(text) > maximum:
        raise AssignmentError(f"{label} length is invalid")
    if any(
        char in _BIDI_CONTROLS
        or ord(char) < 0x20
        or ord(char) == 0x7F
        or unicodedata.category(char) in {"Cc", "Cf", "Cs"}
        or (ord(char) & 0xFFFF) in {0xFFFE, 0xFFFF}
        for char in text
    ):
        raise AssignmentError(f"{label} contains unsupported control characters")
    return text


def _rubric_scalar(raw: str, *, line_number: int) -> Any:
    value = raw.strip()
    if not value:
        raise AssignmentError(f"rubric line {line_number} is missing a scalar value")
    if value.startswith('"'):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise AssignmentError(f"rubric line {line_number} has an invalid quoted string") from exc
        if not isinstance(parsed, str):
            raise AssignmentError(f"rubric line {line_number} must use a scalar")
        return parsed
    if value.startswith("'"):
        if len(value) < 2 or not value.endswith("'"):
            raise AssignmentError(f"rubric line {line_number} has an invalid quoted string")
        return value[1:-1].replace("''", "'")
    if "#" in value:
        raise AssignmentError("inline rubric comments are unsupported; use a full comment line")
    if value[0] in "[{&*!|>@`" or value in {"---", "..."}:
        raise AssignmentError(f"rubric line {line_number} uses unsupported YAML syntax")
    parsed = _parse_yaml_value(value)
    if isinstance(parsed, float) and not math.isfinite(parsed):
        raise AssignmentError(f"rubric line {line_number} must contain a finite number")
    return parsed


def _parse_rubric_yaml(raw: bytes) -> dict[str, Any]:
    if not raw or len(raw) > MAX_RUBRIC_BYTES:
        raise AssignmentError("rubric size is invalid")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise AssignmentError("rubric must be UTF-8") from exc
    if "\r" in text or "\t" in text or "\x00" in text:
        raise AssignmentError("rubric contains unsupported whitespace or control characters")
    lines = text.split("\n")
    if len(lines) > 1000:
        raise AssignmentError("rubric has too many lines")
    top: dict[str, Any] = {}
    criteria: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    in_criteria = False

    def add_pair(target: dict[str, Any], content: str, line_number: int) -> None:
        if ":" not in content:
            raise AssignmentError(f"rubric line {line_number} must be key: value")
        key, raw_value = content.split(":", 1)
        key = key.strip()
        if not re.fullmatch(r"[a-z][a-z0-9_]*", key) or key in target:
            raise AssignmentError(f"rubric line {line_number} has an invalid or duplicate key")
        target[key] = _rubric_scalar(raw_value, line_number=line_number)

    for line_number, line in enumerate(lines, 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if any(char in _BIDI_CONTROLS or (ord(char) < 0x20 and char != "\n") for char in line):
            raise AssignmentError(f"rubric line {line_number} contains a control character")
        indent = len(line) - len(line.lstrip(" "))
        content = line[indent:]
        if indent == 0:
            if in_criteria:
                raise AssignmentError("top-level rubric keys must precede criteria")
            if ":" not in content:
                raise AssignmentError(f"rubric line {line_number} must be key: value")
            key, raw_value = content.split(":", 1)
            key = key.strip()
            if key == "criteria":
                if raw_value.strip() or key in top:
                    raise AssignmentError("criteria must be one indented list")
                top[key] = criteria
                in_criteria = True
                current = None
            else:
                add_pair(top, content, line_number)
        elif indent == 2 and content.startswith("- ") and in_criteria:
            if len(criteria) >= MAX_CRITERIA:
                raise AssignmentError("rubric has too many criteria")
            current = {}
            criteria.append(current)
            add_pair(current, content[2:].strip(), line_number)
        elif indent == 4 and in_criteria and current is not None and not content.startswith("- "):
            add_pair(current, content, line_number)
        else:
            raise AssignmentError(f"rubric line {line_number} has unsupported indentation or structure")
    return top


def load_assignment_rubric(path_value: str | Path) -> AssignmentRubric:
    raw = _read_regular_nofollow(
        path_value, max_bytes=MAX_RUBRIC_BYTES, label="rubric file"
    )
    value = _parse_rubric_yaml(raw)
    rubric = _require_exact_keys(
        value,
        ("task", "criteria"),
        ("max_cost_usd", "max_duration_minutes"),
        label="rubric",
    )
    task = _safe_rubric_text(rubric["task"], label="rubric task", maximum=200)
    max_cost: float | None = None
    max_duration: float | None = None
    if "max_cost_usd" in rubric:
        max_cost = float(_number(
            rubric["max_cost_usd"], label="max_cost_usd", maximum=1_000_000_000
        ))
    if "max_duration_minutes" in rubric:
        max_duration = float(_number(
            rubric["max_duration_minutes"],
            label="max_duration_minutes",
            maximum=MAX_DURATION_SECONDS / 60,
        ))
    raw_criteria = rubric["criteria"]
    if not isinstance(raw_criteria, list) or not raw_criteria or len(raw_criteria) > MAX_CRITERIA:
        raise AssignmentError("rubric criteria must be a non-empty bounded list")
    names: set[str] = set()
    criteria: list[AssignmentCriterion] = []
    total_weight = 0.0
    allowed = {"name", "scorer", "expected", "threshold", "fail_on", "weight", "rule"}
    for item in raw_criteria:
        if not isinstance(item, dict) or not set(item) <= allowed:
            raise AssignmentError("rubric criterion has unknown fields")
        if not {"name", "scorer", "weight"} <= set(item):
            raise AssignmentError("rubric criterion is missing required fields")
        name = _safe_rubric_text(item["name"], label="criterion name", maximum=80)
        if not _RUBRIC_NAME_RE.fullmatch(name) or name in names:
            raise AssignmentError("criterion name is invalid or duplicated")
        names.add(name)
        scorer = item["scorer"]
        if not isinstance(scorer, str) or scorer not in SUPPORTED_SCORERS:
            raise AssignmentError("criterion scorer is unsupported")
        weight = int(_number(
            item["weight"],
            label="criterion weight",
            minimum=1,
            maximum=1000,
            integer=True,
        ))
        total_weight += weight
        if total_weight > 10_000:
            raise AssignmentError("rubric total weight exceeds the supported bound")

        if scorer == "session_status":
            if "threshold" in item or "rule" in item or "expected" not in item:
                raise AssignmentError("session_status requires expected and does not accept threshold or rule")
            expected = item["expected"]
            fail_on = item.get("fail_on", "not_equal")
            if expected not in _RECORDED_STATUSES or fail_on != "not_equal":
                raise AssignmentError("session_status expected or fail_on is invalid")
            criteria.append(AssignmentCriterion(
                name=name,
                scorer=scorer,
                weight=weight,
                fail_on=fail_on,
                expected=expected,
            ))
            continue

        if "expected" in item or "threshold" not in item:
            raise AssignmentError("numeric criterion requires threshold and does not accept expected")
        fail_on = item.get("fail_on", "above")
        if fail_on not in {"above", "below"}:
            raise AssignmentError("numeric criterion fail_on must be above or below")
        integer_metric = scorer in {"error_count", "tool_call_count", "lint_violations"}
        threshold_value = _number(
            item["threshold"],
            label="criterion threshold",
            maximum=1.0 if scorer == "redundant_read_ratio" else 1_000_000_000,
            integer=integer_metric,
        )
        rule: str | None = None
        if scorer == "lint_violations":
            if "rule" in item:
                if (
                    not isinstance(item["rule"], str)
                    or item["rule"] not in _ASSIGNMENT_SCORABLE_LINT_RULES
                ):
                    raise AssignmentError(
                        "lint_violations rule is unsupported or unavailable in minimized telemetry"
                    )
                rule = item["rule"]
        elif "rule" in item:
            raise AssignmentError("rule is accepted only by lint_violations")
        criteria.append(AssignmentCriterion(
            name=name,
            scorer=scorer,
            weight=weight,
            fail_on=fail_on,
            threshold=threshold_value,
            rule=rule,
        ))
    return AssignmentRubric(
        task=task,
        criteria=tuple(criteria),
        max_cost_usd=max_cost,
        max_duration_minutes=max_duration,
    )


# ---------------------------------------------------------------------------
# Deterministic scoring and CLI formatting
# ---------------------------------------------------------------------------


def _metric_value(bundle: LoadedAssignment, criterion: AssignmentCriterion) -> int | float | str:
    if criterion.scorer == "cost_usd":
        return bundle.cost["total_cost_usd"]
    if criterion.scorer == "duration_seconds":
        return bundle.stats["duration_seconds"]
    if criterion.scorer == "error_count":
        return bundle.stats["error_count"]
    if criterion.scorer == "tool_call_count":
        return bundle.stats["tool_call_count"]
    if criterion.scorer == "redundant_read_ratio":
        return bundle.stats["redundant_read_ratio"]
    if criterion.scorer == "session_status":
        return bundle.stats["session_status"]
    if criterion.scorer == "lint_violations":
        if criterion.rule:
            return bundle.lint["rule_counts"][criterion.rule]
        return bundle.lint["finding_count"]
    raise AssertionError("unsupported assignment scorer")


def _score_one(
    bundle: LoadedAssignment,
    rubric: AssignmentRubric,
    submission_ref: str,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    awarded = 0
    total = sum(item.weight for item in rubric.criteria)
    for criterion in rubric.criteria:
        value = _metric_value(bundle, criterion)
        if criterion.scorer == "session_status":
            met = value == criterion.expected
            boundary: dict[str, Any] = {
                "expected": criterion.expected,
                "fail_on": criterion.fail_on,
            }
        else:
            assert criterion.threshold is not None
            met = (
                float(value) <= float(criterion.threshold)
                if criterion.fail_on == "above"
                else float(value) >= float(criterion.threshold)
            )
            boundary = {
                "threshold": criterion.threshold,
                "fail_on": criterion.fail_on,
            }
        points = criterion.weight if met else 0
        awarded += points
        evidence: dict[str, Any] = {"value": value}
        if criterion.rule:
            evidence["lint_rule"] = criterion.rule
        results.append({
            "name": criterion.name,
            "scorer": criterion.scorer,
            "weight": criterion.weight,
            "criterion_met": met,
            "points_awarded": points,
            "evidence": evidence,
            **boundary,
            "limitation": _METRIC_SEMANTICS[criterion.scorer],
        })
    percent = float(
        (Decimal(awarded) * Decimal(100) / Decimal(total)).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
    )
    return {
        "submission_ref": submission_ref,
        "bundle_digest": bundle.bundle_digest,
        "score_percent": percent,
        "points_awarded": awarded,
        "total_weight": total,
        "summary": {
            "estimated_cost_usd": bundle.cost["total_cost_usd"],
            "duration_seconds": bundle.stats["duration_seconds"],
            "session_status": bundle.stats["session_status"],
            "error_count": bundle.stats["error_count"],
            "lint_violation_count": bundle.lint["finding_count"],
        },
        "criteria": results,
    }


def build_score_report(
    bundles: list[LoadedAssignment],
    rubric: AssignmentRubric,
    *,
    compare: bool,
) -> dict[str, Any]:
    if not bundles:
        raise AssignmentError("no assignment submissions were selected")
    ordered = sorted(bundles, key=lambda item: item.bundle_digest)
    if len({item.bundle_digest for item in ordered}) != len(ordered):
        raise AssignmentError("comparison contains duplicate assignment bundles")
    scored = [
        _score_one(bundle, rubric, f"Submission-{index:03d}")
        for index, bundle in enumerate(ordered, 1)
    ]
    ranked = sorted(
        scored,
        key=lambda item: (
            -Fraction(item["points_awarded"], item["total_weight"]),
            item["bundle_digest"],
        ),
    )
    previous_score: Fraction | None = None
    previous_rank = 0
    for index, item in enumerate(ranked, 1):
        exact_score = Fraction(item["points_awarded"], item["total_weight"])
        if previous_score is None or exact_score != previous_score:
            previous_rank = index
            previous_score = exact_score
        item["rubric_rank"] = previous_rank
    return {
        "schema": SCORE_SCHEMA,
        "task": rubric.task,
        "mode": "comparison" if compare else "single",
        "submission_count": len(ranked),
        "rubric_context": {
            "max_cost_usd": rubric.max_cost_usd,
            "max_duration_minutes": rubric.max_duration_minutes,
            "context_only": True,
        },
        "submissions": ranked,
        "limitations": list(_LIMITATIONS) + [
            "Rubric rank orders only the declared process-telemetry arithmetic; tied scores share a rank.",
            "Protected attributes, candidate identity, work product, and task outputs are absent and must not be inferred from this report.",
        ],
    }


def format_score_json(report: dict[str, Any]) -> str:
    return json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _format_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def format_score_text(report: dict[str, Any]) -> str:
    lines = [
        f"Assignment process review — {report['task']}",
        f"{report['submission_count']} submission(s)",
        "Review aid only — not a hiring decision or task/candidate-quality assessment.",
        "",
    ]
    if report["mode"] == "comparison":
        lines.append(
            f"{'Submission':<16} {'Cost est.':>10} {'Duration':>10} "
            f"{'Status':<14} {'Errors':>7} {'Lint':>6} {'Score':>9} {'Rank':>6}"
        )
        lines.append("-" * 86)
        for item in report["submissions"]:
            summary = item["summary"]
            lines.append(
                f"{item['submission_ref']:<16} "
                f"${summary['estimated_cost_usd']:>9.4f} "
                f"{summary['duration_seconds']:>9.1f}s "
                f"{summary['session_status']:<14} "
                f"{summary['error_count']:>7} "
                f"{summary['lint_violation_count']:>6} "
                f"{item['score_percent']:>8.2f} "
                f"{item['rubric_rank']:>6}"
            )
        lines.extend(["", "Bundle digest map:"])
        lines.extend(
            f"- {item['submission_ref']}: {item['bundle_digest']}"
            for item in report["submissions"]
        )
    else:
        item = report["submissions"][0]
        lines.extend([
            f"{item['submission_ref']} — {item['score_percent']:.2f}/100 rubric arithmetic",
            f"Bundle digest: {item['bundle_digest']}",
            "",
            f"{'Criterion':<28} {'Scorer':<24} {'Evidence':>12} {'Weight':>8} {'Result':>10}",
            "-" * 86,
        ])
        for result in item["criteria"]:
            lines.append(
                f"{result['name']:<28} {result['scorer']:<24} "
                f"{_format_metric(result['evidence']['value']):>12} "
                f"{result['weight']:>8.2f} "
                f"{('MET' if result['criterion_met'] else 'NOT MET'):>10}"
            )
    lines.extend(["", "Limitations:"])
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def _load_submission_inputs(path_value: str | Path, *, compare: bool) -> list[LoadedAssignment]:
    path = Path(path_value)
    if compare:
        directory_fd = _open_directory_chain(path)
        try:
            try:
                names = os.listdir(directory_fd)
            except OSError as exc:
                raise AssignmentError("submission directory could not be read safely") from exc
            zip_names = sorted(
                (name for name in names if name.endswith(".zip")),
                key=lambda name: name.encode("utf-8", errors="strict"),
            )
            if not zip_names or len(zip_names) > MAX_SUBMISSIONS:
                raise AssignmentError("submission directory ZIP count is invalid")
            bundles: list[LoadedAssignment] = []
            total_archive_bytes = 0
            total_events = 0
            for name in zip_names:
                try:
                    metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                except OSError as exc:
                    raise AssignmentError("submission directory contains an unreadable ZIP entry") from exc
                if not stat.S_ISREG(metadata.st_mode):
                    raise AssignmentError("submission directory contains an unsafe ZIP entry")
                raw = _read_child_file(
                    directory_fd,
                    name,
                    max_bytes=MAX_ARCHIVE_BYTES,
                    label="submission archive",
                )
                total_archive_bytes += len(raw)
                if total_archive_bytes > MAX_COMPARISON_ARCHIVE_BYTES:
                    raise AssignmentError("comparison archives exceed the aggregate size limit")
                bundle = load_assignment_bundle(raw)
                total_events += len(bundle.events)
                if total_events > MAX_COMPARISON_EVENTS:
                    raise AssignmentError("comparison traces exceed the aggregate event limit")
                bundles.append(bundle)
            return bundles
        finally:
            os.close(directory_fd)
    if path.suffix != ".zip":
        raise AssignmentError("submission archive must use a .zip suffix")
    raw = _read_regular_nofollow(path, max_bytes=MAX_ARCHIVE_BYTES, label="submission archive")
    return [load_assignment_bundle(raw)]


def cmd_score(args: argparse.Namespace) -> int:
    """Score one assignment bundle or compare a direct directory of bundles."""
    try:
        rubric = load_assignment_rubric(getattr(args, "rubric"))
        compare = bool(getattr(args, "compare", False))
        bundles = _load_submission_inputs(getattr(args, "submission"), compare=compare)
        report = build_score_report(bundles, rubric, compare=compare)
        if getattr(args, "format", "text") == "json":
            sys.stdout.write(format_score_json(report))
        else:
            sys.stdout.write(format_score_text(report))
        return 0
    except (AssignmentError, OSError, ValueError, zipfile.BadZipFile) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1
