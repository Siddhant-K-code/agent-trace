"""Session data retention management.

Enforces configurable retention policies on the local session store:
  - max_age_days: delete sessions older than N days
  - max_sessions: keep only the most recent N sessions
  - max_size_mb: delete oldest sessions when total storage exceeds limit

Config is read from .agent-strace.yaml (optional). All limits are applied
in order: age first, then count, then size. Deletions are logged to
.agent-traces/retention.log when on_delete=log.

Open sessions are never deleted and do not count toward any limit. A session
is open when its metadata has no end time or a hook ``.active-session*``
marker names it. An open session with no writes for OPEN_SESSION_IDLE_DAYS
is treated as abandoned (for example, its agent crashed) and is handled like
a closed session.

Usage:
    agent-strace retention status
    agent-strace retention clean --dry-run
    agent-strace retention clean
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from .models import SessionMeta
from .store import TraceStore, DEFAULT_TRACE_DIR


# An open session with no writes for this long is treated as abandoned.
OPEN_SESSION_IDLE_DAYS = 7


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RetentionConfig:
    max_age_days: int | None = None       # delete sessions older than N days
    max_sessions: int | None = None       # keep only the most recent N sessions
    max_size_mb: float | None = None      # delete oldest when storage > N MB
    on_delete: str = "log"                # "log" | "silent"
    log_path: str = ""                    # defaults to <trace_dir>/retention.log

    @classmethod
    def from_dict(cls, d: dict) -> "RetentionConfig":
        r = d.get("retention", d)  # accept both top-level and nested
        return cls(
            max_age_days=r.get("max_age_days"),
            max_sessions=r.get("max_sessions"),
            max_size_mb=r.get("max_size_mb"),
            on_delete=str(r.get("on_delete", "log")),
            log_path=str(r.get("log_path", "")),
        )

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "RetentionConfig":
        """Load from .agent-strace.yaml, falling back to defaults."""
        paths = []
        if config_path:
            paths.append(Path(config_path))
        paths += [Path(".agent-strace.yaml"), Path(".agent-strace.yml")]

        for p in paths:
            if p.exists():
                try:
                    data = _parse_simple_yaml(p.read_text())
                    return cls.from_dict(data)
                except Exception:
                    pass
        return cls()


def _parse_simple_yaml(text: str) -> dict:
    """Parse a minimal YAML subset sufficient for retention config.

    Handles: top-level keys with scalar values and one level of nesting.
    """
    result: dict = {}
    current_section: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.lstrip()

        if not stripped or stripped.startswith("#"):
            continue

        indent = len(line) - len(stripped)

        if indent == 0:
            if stripped.endswith(":"):
                current_section = stripped[:-1].strip()
                result[current_section] = {}
            elif ":" in stripped:
                k, _, v = stripped.partition(":")
                result[k.strip()] = _coerce(v.strip())
                current_section = None
        elif indent > 0 and current_section and ":" in stripped:
            k, _, v = stripped.partition(":")
            if isinstance(result.get(current_section), dict):
                result[current_section][k.strip()] = _coerce(v.strip())

    return result


def _coerce(value: str) -> int | float | str | None:
    """Coerce a YAML scalar string to a Python type."""
    if value in ("null", "~", ""):
        return None
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip('"').strip("'")


# ---------------------------------------------------------------------------
# Storage size helpers
# ---------------------------------------------------------------------------

def _session_size_bytes(session_dir: Path) -> int:
    """Return total bytes used by a session directory."""
    total = 0
    try:
        for f in session_dir.iterdir():
            if f.is_file():
                total += f.stat().st_size
    except OSError:
        pass
    return total


def _store_size_bytes(
    store: TraceStore, sessions: list[SessionMeta] | None = None,
) -> int:
    """Return bytes used by listed sessions plus their checkpoints.

    Other store content (datasets, approvals, logs, unlisted directories) is
    not counted: retention cannot free it, so it must not drive the size limit.
    """
    if sessions is None:
        sessions = store.list_sessions()
    return sum(
        _retained_session_size_bytes(store, meta.session_id) for meta in sessions
    )


def _retained_session_size_bytes(store: TraceStore, session_id: str) -> int:
    """Return session storage plus its compaction checkpoint sidecar."""
    total = _session_size_bytes(store._session_dir(session_id))
    checkpoint = store.checkpoint_path(session_id)
    try:
        total += checkpoint.stat().st_size
    except OSError:
        pass
    return total


# ---------------------------------------------------------------------------
# Retention logic
# ---------------------------------------------------------------------------

def _active_marker_session_ids(store: TraceStore) -> set[str]:
    """Return session IDs named by hook ``.active-session*`` markers."""
    roots = [store.base_dir]
    if store.storage_root != store.base_dir:
        roots.append(store.storage_root)
    session_ids: set[str] = set()
    for root in roots:
        try:
            markers = list(root.glob(".active-session*"))
        except OSError:
            continue
        for marker in markers:
            try:
                if marker.is_symlink() or not marker.is_file():
                    continue
                session_id = marker.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeDecodeError):
                continue
            if session_id:
                session_ids.add(session_id)
    return session_ids


def _last_write_time(store: TraceStore, meta: SessionMeta) -> float:
    """Return the newest file modification time in the session directory."""
    latest = meta.started_at
    try:
        for f in store._session_dir(meta.session_id).iterdir():
            if f.is_symlink() or not f.is_file():
                continue
            latest = max(latest, f.stat().st_mtime)
    except (OSError, ValueError):
        pass
    return latest


def _is_open(
    store: TraceStore,
    meta: SessionMeta,
    marker_ids: set[str],
    now: float,
) -> bool:
    """Return True when retention must keep the session because it may be running."""
    if meta.ended_at is not None and meta.session_id not in marker_ids:
        return False
    idle_seconds = now - _last_write_time(store, meta)
    return idle_seconds < OPEN_SESSION_IDLE_DAYS * 86400


def _split_open_sessions(
    store: TraceStore, sessions: list[SessionMeta], now: float,
) -> tuple[list[SessionMeta], list[SessionMeta]]:
    """Return (eligible, open) sessions, preserving order."""
    marker_ids = _active_marker_session_ids(store)
    eligible: list[SessionMeta] = []
    open_sessions: list[SessionMeta] = []
    for meta in sessions:
        if _is_open(store, meta, marker_ids, now):
            open_sessions.append(meta)
        else:
            eligible.append(meta)
    return eligible, open_sessions


@dataclass
class RetentionStatus:
    session_count: int
    oldest_session_id: str
    oldest_session_age_days: float
    total_size_bytes: int
    sessions_to_delete: list[str] = field(default_factory=list)
    bytes_to_free: int = 0
    open_session_count: int = 0

    @property
    def total_size_mb(self) -> float:
        return self.total_size_bytes / (1024 * 1024)

    @property
    def bytes_to_free_mb(self) -> float:
        return self.bytes_to_free / (1024 * 1024)


def compute_sessions_to_delete(
    store: TraceStore,
    config: RetentionConfig,
) -> list[str]:
    """Return session IDs that should be deleted under the given policy.

    Sessions are sorted oldest-first. Policies are applied in order:
    age, count, size. A session marked for deletion by any policy is
    included once in the result. Open sessions are never selected and do
    not count toward the count or size limits.
    """
    sessions = store.list_sessions()
    if not sessions:
        return []

    now = time.time()
    eligible, _ = _split_open_sessions(store, sessions, now)
    # list_sessions() returns newest-first; reverse for oldest-first processing
    sessions_oldest_first = list(reversed(eligible))
    to_delete: set[str] = set()

    # --- Age policy ---
    if config.max_age_days is not None:
        cutoff = now - config.max_age_days * 86400
        for meta in sessions_oldest_first:
            if meta.started_at < cutoff:
                to_delete.add(meta.session_id)

    # --- Count policy ---
    if config.max_sessions is not None:
        surviving = [m for m in sessions_oldest_first if m.session_id not in to_delete]
        # sessions_oldest_first[0] is oldest; keep the most recent max_sessions
        excess = len(surviving) - config.max_sessions
        if excess > 0:
            for meta in surviving[:excess]:
                to_delete.add(meta.session_id)

    # --- Size policy ---
    if config.max_size_mb is not None:
        max_bytes = config.max_size_mb * 1024 * 1024
        # Measure only what this policy can free: the same per-session
        # measure the loop below subtracts.
        sizes = {
            meta.session_id: _retained_session_size_bytes(store, meta.session_id)
            for meta in sessions_oldest_first
        }
        total = sum(
            size for sid, size in sizes.items() if sid not in to_delete
        )
        if total > max_bytes:
            surviving = [m for m in sessions_oldest_first if m.session_id not in to_delete]
            for meta in surviving:
                if total <= max_bytes:
                    break
                to_delete.add(meta.session_id)
                total -= sizes[meta.session_id]

    # Return in oldest-first order for deterministic output
    ordered = [m.session_id for m in sessions_oldest_first if m.session_id in to_delete]
    return ordered


def get_retention_status(store: TraceStore, config: RetentionConfig) -> RetentionStatus:
    sessions = store.list_sessions()
    if not sessions:
        return RetentionStatus(
            session_count=0,
            oldest_session_id="",
            oldest_session_age_days=0.0,
            total_size_bytes=0,
        )

    oldest = min(sessions, key=lambda m: m.started_at)
    now = time.time()
    age_days = (now - oldest.started_at) / 86400
    total_bytes = _store_size_bytes(store, sessions)
    _, open_sessions = _split_open_sessions(store, sessions, now)
    to_delete = compute_sessions_to_delete(store, config)
    bytes_to_free = sum(
        _retained_session_size_bytes(store, sid) for sid in to_delete
    )

    return RetentionStatus(
        session_count=len(sessions),
        oldest_session_id=oldest.session_id,
        oldest_session_age_days=age_days,
        total_size_bytes=total_bytes,
        sessions_to_delete=to_delete,
        bytes_to_free=bytes_to_free,
        open_session_count=len(open_sessions),
    )


@dataclass
class _DeletionResult:
    deleted: list[str] = field(default_factory=list)
    kept_open: list[str] = field(default_factory=list)
    bytes_freed: int = 0


def _delete_sessions(
    store: TraceStore,
    session_ids: list[str],
    config: RetentionConfig,
    log_path: str = "",
) -> _DeletionResult:
    """Delete sessions under the store write lock, rechecking each is closed."""
    result = _DeletionResult()
    effective_log = log_path or config.log_path or str(store.base_dir / "retention.log")

    with store.write_lock():
        try:
            # Validate shared record files before deleting any trace data.
            store.external_session_records(set(session_ids))
        except (OSError, ValueError) as exc:
            sys.stderr.write(f"[retention] refusing to delete sessions: {exc}\n")
            return result

        # A session selected earlier may have been resumed since; recheck
        # under the lock that hooks also take before they write.
        now = time.time()
        marker_ids = _active_marker_session_ids(store)
        for sid in session_ids:
            try:
                session_dir = store._session_dir(sid)
                checkpoint_path = store.checkpoint_path(sid)
                if session_dir.exists():
                    store._session_file(sid, "events.ndjson", "event stream")
                    meta_path = store._session_file(sid, "meta.json", "session metadata")
                    if meta_path.exists():
                        meta = store.load_meta(sid)
                        if _is_open(store, meta, marker_ids, now):
                            result.kept_open.append(sid)
                            continue
                size = _retained_session_size_bytes(store, sid)
                checkpoint_path.unlink(missing_ok=True)
                if not session_dir.exists():
                    continue
                shutil.rmtree(session_dir)
                result.deleted.append(sid)
                result.bytes_freed += size
            except (OSError, ValueError) as exc:
                sys.stderr.write(f"[retention] failed to delete {sid}: {exc}\n")

        if not result.deleted:
            return result
        deleted_ids = set(result.deleted)
        try:
            store._delete_external_session_records_locked(deleted_ids)
        except (OSError, ValueError) as exc:
            sys.stderr.write(f"[retention] failed to remove session records: {exc}\n")
        store._cleanup_deleted_hook_state(deleted_ids)
        if config.on_delete == "log":
            for sid in result.deleted:
                _log_deletion(sid, effective_log)

    return result


def delete_sessions(
    store: TraceStore,
    session_ids: list[str],
    config: RetentionConfig,
    log_path: str = "",
) -> int:
    """Delete the given closed sessions. Returns count of sessions deleted.

    Open sessions are skipped. Each deleted session's checkpoint, approval
    records, dataset entries and hook state are removed with it.
    """
    return len(_delete_sessions(store, session_ids, config, log_path).deleted)


def _log_deletion(session_id: str, log_path: str) -> None:
    """Append a deletion record to the retention log."""
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(p, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"deleted_at": ts, "session_id": session_id}) + "\n")


# ---------------------------------------------------------------------------
# CLI handlers
# ---------------------------------------------------------------------------

def cmd_retention_status(args: argparse.Namespace, out: TextIO = sys.stdout) -> int:
    store = TraceStore(args.trace_dir)
    config = RetentionConfig.load(getattr(args, "config", None))
    status = get_retention_status(store, config)

    if status.session_count == 0:
        out.write("No sessions found.\n")
        return 0

    out.write(f"Sessions:    {status.session_count}\n")
    out.write(f"Oldest:      {status.oldest_session_id[:16]} "
              f"({status.oldest_session_age_days:.1f} days ago)\n")
    out.write(f"Size:        {status.total_size_mb:.1f} MB\n")
    if status.open_session_count:
        out.write(f"Open:        {status.open_session_count} session(s), "
                  "kept and not counted by limits\n")

    if status.sessions_to_delete:
        out.write(f"\nPolicy would delete: {len(status.sessions_to_delete)} session(s) "
                  f"({status.bytes_to_free_mb:.1f} MB)\n")
        out.write("Run `agent-strace retention clean` to apply.\n")
    else:
        out.write("\nNo sessions exceed retention limits.\n")

    return 0


def cmd_retention_clean(args: argparse.Namespace, out: TextIO = sys.stdout) -> int:
    store = TraceStore(args.trace_dir)
    config = RetentionConfig.load(getattr(args, "config", None))

    # Allow CLI overrides
    if getattr(args, "max_age_days", None) is not None:
        config.max_age_days = args.max_age_days
    if getattr(args, "max_sessions", None) is not None:
        config.max_sessions = args.max_sessions
    if getattr(args, "max_size_mb", None) is not None:
        config.max_size_mb = args.max_size_mb

    to_delete = compute_sessions_to_delete(store, config)
    _, open_sessions = _split_open_sessions(store, store.list_sessions(), time.time())
    open_note = (
        f"Kept: {len(open_sessions)} open session(s)\n" if open_sessions else ""
    )

    if not to_delete:
        out.write("Nothing to delete — all sessions are within retention limits.\n")
        out.write(open_note)
        return 0

    dry_run = getattr(args, "dry_run", False)
    if dry_run:
        bytes_to_free = sum(
            _retained_session_size_bytes(store, sid) for sid in to_delete
        )
        mb_to_free = bytes_to_free / (1024 * 1024)
        out.write(f"Would delete: {len(to_delete)} session(s) ({mb_to_free:.1f} MB)\n")
        for sid in to_delete:
            out.write(f"  {sid}\n")
        out.write(open_note)
        return 0

    result = _delete_sessions(store, to_delete, config)
    mb_freed = result.bytes_freed / (1024 * 1024)
    out.write(f"Deleted: {len(result.deleted)} session(s) ({mb_freed:.1f} MB freed)\n")
    kept = len({m.session_id for m in open_sessions} | set(result.kept_open))
    if kept:
        out.write(f"Kept: {kept} open session(s)\n")
    return 0


def cmd_retention(args: argparse.Namespace) -> int:
    sub = getattr(args, "retention_command", None)
    if sub == "status":
        return cmd_retention_status(args)
    if sub == "clean":
        return cmd_retention_clean(args)
    sys.stderr.write(
        "Usage: agent-strace retention <status|clean> [--dry-run]\n"
        "Run `agent-strace retention --help` for details.\n"
    )
    return 1
