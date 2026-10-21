"""Backfill ``meta.json`` for session directories captured without one.

Before the subagent capture fix, GitHub Copilot CLI subagents (the ``task``
tool) wrote ``events.ndjson`` under their own session ID but never fired
``sessionStart``, so no ``meta.json`` was created.  Every reader that lists
sessions skips those directories, and strict tenant administration refuses
the whole store.

``agent-strace backfill-meta`` finds those directories and derives metadata
from the events.  It is a dry run unless ``--apply`` is given.  It never
modifies ``events.ndjson``, never replaces an existing ``meta.json``, takes
the store write lock while writing, and refuses sessions covered by a tenant
erasure tombstone or a pending tenant journal.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .hooks import _PROVIDER_AGENT
from .models import EventType, SessionMeta, TraceEvent
from .store import (
    _TENANT_ADMIN_AUXILIARY_DIRS,
    TraceStore,
    _fsync_directory,
    validate_session_id,
    validate_stored_id,
    validate_tenant_id,
)

BACKFILL_SOURCE = "backfilled"
DEFAULT_AGENT_NAME = "agent-cli"

_HOOK_AGENT_NAMES = frozenset(_PROVIDER_AGENT.values())


@dataclass
class BackfillItem:
    """One metaless session directory and what happened to it."""

    session_id: str
    path: str
    workspace_id: str = ""
    status: str = "skipped"  # would_write | written | skipped | failed
    reason: str = ""
    detail: str = ""
    event_count: int = 0
    agent_name_source: str = ""
    meta: dict | None = None

    def to_dict(self) -> dict:
        data = {
            "session_id": self.session_id,
            "path": self.path,
            "workspace_id": self.workspace_id,
            "status": self.status,
        }
        if self.reason:
            data["reason"] = self.reason
        if self.detail:
            data["detail"] = self.detail
        if self.meta is not None:
            data["event_count"] = self.event_count
            data["agent_name_source"] = self.agent_name_source
            data["meta"] = self.meta
        return data


@dataclass
class BackfillReport:
    trace_dir: str
    applied: bool
    stores: list[str] = field(default_factory=list)
    sessions_with_meta: int = 0
    ignored_entries: int = 0
    items: list[BackfillItem] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def count(self, status: str) -> int:
        return sum(1 for item in self.items if item.status == status)

    def skip_reasons(self) -> dict[str, int]:
        reasons: dict[str, int] = {}
        for item in self.items:
            if item.status in ("skipped", "failed"):
                reasons[item.reason] = reasons.get(item.reason, 0) + 1
        return dict(sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0])))

    def to_dict(self) -> dict:
        return {
            "trace_dir": self.trace_dir,
            "mode": "apply" if self.applied else "dry_run",
            "stores": self.stores,
            "sessions_with_meta": self.sessions_with_meta,
            "ignored_entries": self.ignored_entries,
            "found": len(self.items),
            "would_write": self.count("would_write"),
            "written": self.count("written"),
            "skipped": self.count("skipped"),
            "failed": self.count("failed"),
            "skip_reasons": self.skip_reasons(),
            "errors": self.errors,
            "sessions": [item.to_dict() for item in self.items],
        }


# ---------------------------------------------------------------------------
# Deriving metadata from events
# ---------------------------------------------------------------------------

def infer_agent_name(events: list[TraceEvent]) -> str:
    """Return the hook agent name recorded in the events, or ``""``.

    Conflicting providers return ``""``.
    """
    names = hook_agent_names(events)
    return next(iter(names)) if len(names) == 1 else ""


def hook_agent_names(events: list[TraceEvent]) -> set[str]:
    """Return every hook agent name recorded in the events.

    Only hook-written ``session_start`` and ``assistant_response`` (stop)
    events carry the hook provider.  Other events may name an LLM provider,
    which must not be mistaken for the capturing agent.
    """
    names: set[str] = set()
    for event in events:
        data = event.data if isinstance(event.data, dict) else {}
        if event.event_type == EventType.SESSION_START:
            mode = data.get("mode")
            if isinstance(mode, str) and mode.endswith("-hooks"):
                name = mode[: -len("-hooks")]
                if name in _HOOK_AGENT_NAMES:
                    names.add(name)
        elif not (
            event.event_type == EventType.ASSISTANT_RESPONSE
            and data.get("hook_event") == "stop"
        ):
            continue
        provider = data.get("provider")
        if isinstance(provider, str) and provider in _PROVIDER_AGENT:
            names.add(_PROVIDER_AGENT[provider])
    return names


def derive_meta(
    session_id: str,
    events: list[TraceEvent],
    *,
    agent_name: str = "",
    tenant_id: str = "",
    workspace_id: str = "",
) -> tuple[SessionMeta, str]:
    """Build the metadata the hooks would have written for ``events``.

    Returns the metadata and where ``agent_name`` came from: ``events``,
    ``flag`` (the ``agent_name`` argument) or ``default``.
    """
    if not events:
        raise ValueError("cannot derive metadata from an empty event stream")
    inferred = infer_agent_name(events)
    if inferred:
        name, source = inferred, "events"
    elif agent_name:
        name, source = agent_name, "flag"
    else:
        name, source = DEFAULT_AGENT_NAME, "default"
    timestamps = [float(event.timestamp) for event in events]
    started_at = min(timestamps)
    ended_at = max(timestamps)
    meta = SessionMeta(
        session_id=session_id,
        started_at=started_at,
        ended_at=ended_at,
        agent_name=name,
        command=f"{name} ({BACKFILL_SOURCE})",
        # Same counters the hooks maintain: one per PreToolUse and one per
        # failed PostToolUse.  llm_request is counted like the proxy paths.
        tool_calls=sum(1 for e in events if e.event_type == EventType.TOOL_CALL),
        llm_requests=sum(1 for e in events if e.event_type == EventType.LLM_REQUEST),
        errors=sum(1 for e in events if e.event_type == EventType.ERROR),
        total_duration_ms=(ended_at - started_at) * 1000,
        workspace_id=workspace_id,
        # Attribution describes who started the capture.  Collecting it now
        # would record the operator running the backfill instead.
        attribution={},
        # Explicit so AGENT_STRACE_TENANT_ID never tags historical data.
        tenant_id=tenant_id,
    )
    return meta, source


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

class _Skip(Exception):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _read_events(path: Path, session_id: str) -> tuple[list[TraceEvent], str]:
    """Parse an event stream strictly.  Returns events and the tenant ID."""
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _Skip("unparseable_events", f"not UTF-8: {exc.reason}") from exc
    except OSError as exc:
        raise _Skip("unreadable_events", exc.strerror or str(exc)) from exc
    events: list[TraceEvent] = []
    tenants: set[str] = set()
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = TraceEvent.from_json(line)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError) as exc:
            raise _Skip("unparseable_events", f"line {number}: {type(exc).__name__}") from exc
        if not _is_number(event.timestamp):
            raise _Skip("unparseable_events", f"line {number}: invalid timestamp")
        if event.session_id and event.session_id != session_id:
            raise _Skip("session_id_mismatch", f"line {number}")
        if event.tenant_id:
            try:
                tenants.add(validate_tenant_id(event.tenant_id))
            except (ValueError, TypeError) as exc:
                raise _Skip("invalid_tenant", f"line {number}") from exc
        events.append(event)
    if not events:
        raise _Skip("empty_events")
    if len(tenants) > 1:
        raise _Skip("mixed_tenants", f"{len(tenants)} tenant IDs in one session")
    return events, next(iter(tenants), "")


def _pending_journal_sessions(store: TraceStore) -> tuple[set[str], set[str]]:
    """Return session IDs named by pending tenant delete and tag journals.

    Raises ``_Skip`` when a journal cannot be read: its intent is unknown,
    so no session in the store can be safely backfilled.
    """
    deleting: set[str] = set()
    tagging: set[str] = set()
    try:
        journal_dir = store._journal_dir()
    except ValueError as exc:
        raise _Skip("unreadable_journal", str(exc)) from exc
    if not journal_dir.exists():
        return deleting, tagging
    for path in sorted(journal_dir.iterdir(), key=lambda p: p.name):
        is_delete = path.name.startswith("delete-") and path.name.endswith(".json")
        is_tag = path.name.startswith("tag-") and path.name.endswith(".json")
        if not (is_delete or is_tag):
            continue
        try:
            if path.is_symlink() or not path.is_file():
                raise ValueError("not a regular file")
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict):
                raise ValueError("not an object")
            if is_delete:
                for item in record.get("sessions", []):
                    if isinstance(item, dict) and not item.get("deleted"):
                        deleting.add(str(item.get("session_id", "")))
            else:
                tagging.add(str(record.get("session_id", "")))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise _Skip("unreadable_journal", f"{path.name}: {exc}") from exc
    return deleting, tagging


def _check_erasure_audit(store: TraceStore) -> None:
    """Fail closed when the erasure tombstone file exists but is unreadable.

    ``TraceStore._session_was_erased`` treats read errors as "not erased";
    backfill must not recreate an erased session on that basis.
    """
    try:
        path = store._store_file("tenant-deletions.ndjson", "tenant deletion audit")
        if os.path.lexists(path):
            if path.is_symlink() or not path.is_file():
                raise ValueError("not a regular file")
            path.read_bytes()
    except (OSError, ValueError) as exc:
        raise _Skip("unreadable_erasure_audit", str(exc)) from exc


def _child_tenants(store: TraceStore) -> dict[str, set[str]]:
    """Map parent session ID to the tenant IDs of its existing children."""
    children: dict[str, set[str]] = {}
    for meta in store.list_sessions():
        if meta.parent_session_id:
            children.setdefault(meta.parent_session_id, set()).add(meta.tenant_id)
    return children


def _write_meta_exclusive(store: TraceStore, session_id: str, meta: SessionMeta) -> None:
    """Create ``meta.json`` without following symlinks or replacing a file."""
    path = store._session_file(session_id, "meta.json", "session metadata")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o666)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(meta.to_json())
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    _fsync_directory(path.parent)


def _examine(
    store: TraceStore,
    entry: Path,
    *,
    agent_name: str,
    deleting: set[str],
    tagging: set[str],
    children: dict[str, set[str]],
) -> tuple[BackfillItem, SessionMeta | None]:
    item = BackfillItem(
        session_id=entry.name,
        path="",
        workspace_id=store.workspace_id,
    )
    try:
        session_id = validate_session_id(entry.name)
    except ValueError as exc:
        raise _Skip("invalid_session_id", str(exc)) from exc
    try:
        store._session_dir(session_id)
        events_path = store._session_file(session_id, "events.ndjson", "event stream")
    except ValueError as exc:
        reason = "events_symlink" if "symlink" in str(exc) else "events_not_file"
        raise _Skip(reason, str(exc)) from exc
    if not events_path.exists():
        raise _Skip("no_events")
    if os.path.lexists(entry / "identity.json"):
        # The signature covers meta fields; rebuilt meta would not match and
        # would hide that the signed metadata was removed.
        raise _Skip("signed_without_meta", "identity.json present")
    if store._session_was_erased(session_id):
        raise _Skip("erased", "covered by a tenant erasure tombstone")
    if session_id in deleting:
        raise _Skip("pending_erasure", "named by a pending tenant delete journal")
    if session_id in tagging:
        raise _Skip("pending_tag", "named by a pending tenant tag journal")
    events, tenant_id = _read_events(events_path, session_id)
    child_tenants = children.get(session_id, set())
    if child_tenants and child_tenants != {tenant_id}:
        raise _Skip("cross_tenant_child", "a child session has a different tenant")
    meta, source = derive_meta(
        session_id,
        events,
        agent_name=agent_name,
        tenant_id=tenant_id,
        workspace_id=store.workspace_id,
    )
    store._redact_meta(meta)
    item.event_count = len(events)
    item.agent_name_source = source
    item.meta = json.loads(meta.to_json())
    return item, meta


def _scan_store(
    store: TraceStore,
    report: BackfillReport,
    *,
    apply: bool,
    agent_name: str,
    root: Path,
) -> None:
    base = store.base_dir
    if base.is_symlink():
        report.errors.append(f"{_relative(base, root)}: refusing symlinked trace store")
        return
    if not base.exists():
        return
    if not base.is_dir():
        report.errors.append(f"{_relative(base, root)}: trace store is not a directory")
        return

    journal_error: _Skip | None = None
    try:
        _check_erasure_audit(store)
        deleting, tagging = _pending_journal_sessions(store)
    except _Skip as exc:
        journal_error = exc
        deleting, tagging = set(), set()
    children = _child_tenants(store)

    for entry in sorted(base.iterdir(), key=lambda item: item.name):
        name = entry.name
        if name.startswith("."):
            continue  # lock, gitignore, hook state, journals, approvals
        if name in _TENANT_ADMIN_AUXILIARY_DIRS and not entry.is_symlink():
            continue
        if not store.workspace_id and name == "workspaces":
            continue  # scanned as separate stores
        rel = _relative(entry, root)
        if entry.is_symlink():
            report.items.append(BackfillItem(
                session_id=name, path=rel, workspace_id=store.workspace_id,
                reason="symlink", detail="symlinked entries are never followed",
            ))
            continue
        if not entry.is_dir():
            report.ignored_entries += 1
            continue
        meta_path = entry / "meta.json"
        if os.path.lexists(meta_path):
            if meta_path.is_symlink() or not meta_path.is_file():
                report.items.append(BackfillItem(
                    session_id=name, path=rel, workspace_id=store.workspace_id,
                    reason="unsafe_meta", detail="meta.json is not a regular file",
                ))
            else:
                report.sessions_with_meta += 1
            continue
        if not os.path.lexists(entry / "events.ndjson"):
            report.ignored_entries += 1
            continue

        try:
            if journal_error is not None:
                raise journal_error
            item, meta = _examine(
                store, entry,
                agent_name=agent_name,
                deleting=deleting, tagging=tagging, children=children,
            )
        except _Skip as exc:
            report.items.append(BackfillItem(
                session_id=name, path=rel, workspace_id=store.workspace_id,
                reason=exc.reason, detail=exc.detail,
            ))
            continue
        item.path = rel
        if not apply:
            item.status = "would_write"
        else:
            try:
                _write_meta_exclusive(store, item.session_id, meta)
                item.status = "written"
            except FileExistsError:
                item.status = "skipped"
                item.reason = "meta_exists"
                item.detail = "meta.json appeared before the write"
                item.meta = None
            except (OSError, ValueError) as exc:
                item.status = "failed"
                item.reason = "write_failed"
                item.detail = str(exc)
                item.meta = None
        report.items.append(item)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _discover_stores(root: Path, report: BackfillReport) -> list[TraceStore]:
    """Return the flat store plus each safe workspace store.

    Mirrors ``tenancy.enumerate_trace_stores`` but reports unsafe workspace
    entries instead of aborting, and ignores ``AGENT_STRACE_WORKSPACE``.
    """
    stores = [TraceStore(root, use_workspace_env=False)]
    workspaces = stores[0].storage_root / "workspaces"
    if workspaces.is_symlink():
        report.errors.append("workspaces: refusing symlinked workspaces root")
        return stores
    if not workspaces.is_dir():
        return stores
    seen = {stores[0].base_dir.resolve()}
    for directory in sorted(workspaces.iterdir(), key=lambda item: item.name):
        rel = _relative(directory, root)
        if directory.is_symlink():
            report.items.append(BackfillItem(
                session_id="", path=rel, workspace_id=directory.name,
                reason="workspace_symlink", detail="symlinked workspaces are never followed",
            ))
            continue
        if not directory.is_dir():
            continue
        try:
            workspace_id = validate_stored_id(directory.name, "stored workspace directory")
            store = TraceStore(
                root,
                workspace_id=workspace_id,
                use_workspace_env=False,
                allow_legacy_workspace_id=True,
            )
        except ValueError as exc:
            report.items.append(BackfillItem(
                session_id="", path=rel, workspace_id="",
                reason="invalid_workspace_id", detail=str(exc),
            ))
            continue
        resolved = store.base_dir.resolve()
        if resolved not in seen:
            seen.add(resolved)
            stores.append(store)
    return stores


def backfill_trace_dir(
    trace_dir: str | os.PathLike[str],
    *,
    apply: bool = False,
    agent_name: str = "",
) -> BackfillReport:
    """Scan the flat store and every workspace store under ``trace_dir``.

    Opening a store retries interrupted tenant tag and delete journals, as it
    does for every reader.  A dry run writes no metadata.
    """
    root = Path(trace_dir)
    report = BackfillReport(trace_dir=str(trace_dir), applied=apply)
    if root.is_symlink():
        report.errors.append("refusing symlinked trace directory")
        return report
    if not root.exists():
        return report
    if not root.is_dir():
        report.errors.append("trace directory is not a directory")
        return report
    try:
        stores = _discover_stores(root, report)
    except (OSError, ValueError) as exc:
        report.errors.append(f"store discovery failed: {exc}")
        return report

    for store in stores:
        report.stores.append(_relative(store.base_dir, root) if store.workspace_id else ".")
        try:
            if apply:
                # Hold the store lock for the whole scan so events read here
                # are the events the written meta describes.
                with store.write_lock():
                    _scan_store(store, report, apply=True, agent_name=agent_name, root=root)
            else:
                _scan_store(store, report, apply=False, agent_name=agent_name, root=root)
        except (OSError, ValueError) as exc:
            report.errors.append(f"{report.stores[-1]}: {exc}")
    return report


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _format_time(timestamp: float | None) -> str:
    if timestamp is None:
        return "-"
    try:
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
    except (OverflowError, OSError, ValueError):
        return str(timestamp)


def format_report(report: BackfillReport) -> str:
    lines: list[str] = []
    mode = "apply" if report.applied else "dry run (nothing written; re-run with --apply to write)"
    lines.append(f"backfill-meta: {mode}")
    lines.append(f"Trace dir: {report.trace_dir}  stores scanned: {len(report.stores)}")
    lines.append("")
    lines.append(f"  sessions with meta:   {report.sessions_with_meta}")
    lines.append(f"  metaless dirs found:  {len(report.items)}")
    if report.applied:
        lines.append(f"  written:              {report.count('written')}")
    else:
        lines.append(f"  would write:          {report.count('would_write')}")
    skipped = report.count("skipped") + report.count("failed")
    lines.append(f"  skipped:              {skipped}")
    for reason, count in report.skip_reasons().items():
        lines.append(f"    {reason:<22}{count}")
    if report.ignored_entries:
        lines.append(f"  non-session entries ignored: {report.ignored_entries}")

    ready = [i for i in report.items if i.status in ("would_write", "written")]
    if ready:
        lines.append("")
        lines.append("Would write:" if not report.applied else "Written:")
        for item in ready:
            meta = item.meta or {}
            tenant = f"  tenant={meta['tenant_id']}" if meta.get("tenant_id") else ""
            duration = (meta.get("total_duration_ms") or 0) / 1000
            lines.append(
                f"  {item.path}  {meta.get('agent_name', '')} [{item.agent_name_source}]"
                f"  {_format_time(meta.get('started_at'))} -> {_format_time(meta.get('ended_at'))}"
                f" ({duration:.1f}s)"
                f"  events={item.event_count} tool_calls={meta.get('tool_calls', 0)}"
                f" errors={meta.get('errors', 0)}{tenant}"
            )
    others = [i for i in report.items if i.status in ("skipped", "failed")]
    if others:
        lines.append("")
        lines.append("Skipped:")
        for item in others:
            detail = f"  ({item.detail})" if item.detail else ""
            lines.append(f"  {item.path}  {item.reason}{detail}")
    if report.errors:
        lines.append("")
        lines.append("Errors:")
        for error in report.errors:
            lines.append(f"  {error}")
    if ready:
        lines.append("")
        lines.append(
            "Backfilled sessions keep their original timestamps, so retention limits "
            "apply to them. Review with: agent-strace retention status"
        )
    return "\n".join(lines) + "\n"


def _validate_agent_name(value: str) -> str:
    value = (value or "").strip()
    if len(value) > 128:
        raise ValueError("--agent-name cannot exceed 128 characters")
    if any(unicodedata.category(char) == "Cc" for char in value):
        raise ValueError("--agent-name cannot contain control characters")
    return value


def cmd_backfill_meta(args: argparse.Namespace) -> int:
    try:
        agent_name = _validate_agent_name(getattr(args, "agent_name", "") or "")
    except ValueError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1
    report = backfill_trace_dir(
        args.trace_dir,
        apply=bool(getattr(args, "apply", False)),
        agent_name=agent_name,
    )
    if getattr(args, "format", "text") == "json":
        sys.stdout.write(json.dumps(report.to_dict(), indent=2) + "\n")
    else:
        sys.stdout.write(format_report(report))
    return 1 if report.errors or report.count("failed") else 0
