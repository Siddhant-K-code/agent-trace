"""Trace storage.

Traces are stored as directories:
  .agent-traces/
    <session-id>/
      meta.json       # session metadata
      events.ndjson   # newline-delimited JSON events

  With workspace isolation:
  .agent-traces/
    workspaces/
      <workspace-id>/
        <session-id>/
          meta.json
          events.ndjson

NDJSON is append-only during capture. Assigning a tenant to an already-active
session performs one atomic rewrite and rebuilds its hash chain. No database.
No dependencies. Just files.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import unicodedata
import uuid
from pathlib import Path

from .models import EventType, SessionMeta, TraceEvent
from .redact import redact_data_with_status, redaction_enabled

DEFAULT_TRACE_DIR = ".agent-traces"

# Env var for workspace isolation — sessions are stored under
# .agent-traces/workspaces/<workspace-id>/ when this is set.
_WORKSPACE_ENV = "AGENT_STRACE_WORKSPACE"
_TENANT_ENV = "AGENT_STRACE_TENANT_ID"
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TENANT_ADMIN_AUXILIARY_DIRS = {
    ".approvals",
    ".tenant-journal",
    "checkpoints",
    "datasets",
}


def _has_control_character(value: str) -> bool:
    return any(unicodedata.category(char) == "Cc" for char in value)


def validate_tenant_id(tenant_id: str) -> str:
    """Return a normalised tenant ID or raise ``ValueError``.

    Tenant IDs are data tags rather than paths, but rejecting control
    characters prevents terminal/log injection and a modest length limit
    keeps exported attributes bounded.
    """
    value = str(tenant_id).strip()
    if not value:
        raise ValueError("tenant ID cannot be empty")
    if len(value) > 256:
        raise ValueError("tenant ID cannot exceed 256 characters")
    if _has_control_character(value):
        raise ValueError("tenant ID cannot contain control characters")
    return value


def validate_session_id(session_id: str) -> str:
    """Validate an opaque session identifier before any path construction."""
    value = str(session_id)
    if not _SAFE_ID_RE.fullmatch(value) or value in {".", ".."}:
        raise ValueError(
            "session ID must be 1-128 ASCII letters, digits, dots, underscores, or hyphens"
        )
    return value


def validate_stored_id(value: str, kind: str = "stored identifier") -> str:
    """Validate a containment-safe legacy directory basename.

    Historical stores may contain spaces, Unicode, or identifiers longer than
    the current external-ingress limit. Those names remain readable as long as
    they are one safe path component and contain no control characters.
    """
    value = str(value)
    if (
        not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or _has_control_character(value)
    ):
        raise ValueError(f"invalid {kind}")
    return value


def validate_workspace_id(workspace_id: str) -> str:
    """Validate a workspace directory identifier."""
    value = str(workspace_id)
    if not _SAFE_ID_RE.fullmatch(value) or value in {".", ".."}:
        raise ValueError(
            "workspace ID must be 1-128 ASCII letters, digits, dots, underscores, or hyphens"
        )
    return value


def _validate_meta_ids(meta: SessionMeta, *, strict_session: bool = False) -> None:
    validator = validate_session_id if strict_session else validate_stored_id
    meta.session_id = validator(meta.session_id)
    if meta.parent_session_id:
        meta.parent_session_id = validate_stored_id(
            meta.parent_session_id, "stored parent session ID"
        )


def _lock_file(file_obj) -> None:
    """Best-effort exclusive lock shared by event append and tenant tagging."""
    try:
        import fcntl
        fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX)
    except (ImportError, OSError):
        pass


def _unlock_file(file_obj) -> None:
    try:
        import fcntl
        fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)
    except (ImportError, OSError):
        pass


@contextlib.contextmanager
def _store_write_lock(base_dir: Path):
    """Serialise event append and atomic session retag operations."""
    base_dir.mkdir(parents=True, exist_ok=True)
    lock_path = base_dir / ".store.lock"
    if lock_path.is_symlink() or (lock_path.exists() and not lock_path.is_file()):
        raise ValueError("unsafe trace store lock path")
    if lock_path.resolve(strict=False).parent != base_dir.resolve():
        raise ValueError("trace store lock path escapes the store")
    with open(lock_path, "a+", encoding="utf-8") as lock_file:
        _lock_file(lock_file)
        try:
            yield
        finally:
            _unlock_file(lock_file)


def _write_atomic(path: Path, text: str) -> None:
    """Write text to a sibling temporary file and atomically replace path."""
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError(f"refusing symlinked atomic write path: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(text)
            temporary.flush()
            os.fsync(temporary.fileno())
        if path.exists():
            os.chmod(temporary_name, path.stat().st_mode)
        os.replace(temporary_name, path)
        temporary_name = ""
        _fsync_directory(path.parent)
    finally:
        if temporary_name:
            try:
                Path(temporary_name).unlink()
            except OSError:
                pass


def _fsync_directory(path: Path) -> None:
    """Best-effort directory fsync for durable rename/unlink operations."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _workspace_base(base_dir: str | Path, workspace_id: str) -> Path:
    """Return the workspace-scoped subdirectory path."""
    return Path(base_dir) / "workspaces" / validate_workspace_id(workspace_id)


class TraceStore:
    def __init__(
        self,
        base_dir: str | Path = DEFAULT_TRACE_DIR,
        workspace_id: str = "",
        redact: bool | None = None,
        use_workspace_env: bool = True,
        allow_legacy_workspace_id: bool = False,
    ):
        """Create a TraceStore.

        workspace_id scopes all reads/writes to a subdirectory:
          <base_dir>/workspaces/<workspace_id>/

        If workspace_id is empty, the AGENT_STRACE_WORKSPACE env var is
        checked. If that is also empty, the flat layout is used.
        """
        self.storage_root = Path(base_dir)
        wid = workspace_id or (
            os.environ.get(_WORKSPACE_ENV, "") if use_workspace_env else ""
        )
        if wid:
            validated_wid = (
                validate_stored_id(wid, "stored workspace ID")
                if allow_legacy_workspace_id else validate_workspace_id(wid)
            )
            workspace_root = self.storage_root / "workspaces"
            candidate = workspace_root / validated_wid
            if workspace_root.is_symlink() or candidate.is_symlink():
                raise ValueError("refusing symlinked workspace path")
            if candidate.resolve(strict=False).parent != workspace_root.resolve():
                raise ValueError("workspace path escapes the trace store")
            self.base_dir = candidate
            self.workspace_id = validated_wid
        else:
            self.base_dir = Path(base_dir)
            self.workspace_id = ""
        self.redact = redaction_enabled() if redact is None else redact
        self._erasure_audit_mtime_ns = -1
        self._erased_session_hashes: set[str] = set()
        self._recover_tag_journals()
        self._recover_delete_journals()

    def _warn_redacted(self, item: str) -> None:
        sys.stderr.write(f"agent-strace: redacted secrets from {item}\n")

    def _redact_event(self, event: TraceEvent) -> None:
        if not self.redact:
            return
        redacted_data, changed = redact_data_with_status(event.data)
        if changed:
            event.data = redacted_data
            if not event.redacted:
                self._warn_redacted("trace event")
            event.redacted = True

    def _redact_meta(self, meta: SessionMeta) -> None:
        if not self.redact:
            return
        redacted, changed = redact_data_with_status({
            "agent_name": meta.agent_name,
            "command": meta.command,
            "attribution": meta.attribution,
        })
        if changed:
            already_redacted = meta.redacted
            meta.agent_name = redacted.get("agent_name", meta.agent_name)
            meta.command = redacted.get("command", meta.command)
            meta.attribution = redacted.get("attribution", meta.attribution)
            meta.redacted = True
            if not already_redacted:
                self._warn_redacted("session metadata")

    def _session_dir(self, session_id: str) -> Path:
        session_id = validate_stored_id(session_id, "stored session ID")
        candidate = self.base_dir / session_id
        if candidate.is_symlink():
            raise ValueError("refusing symlinked session directory")
        base_resolved = self.base_dir.resolve()
        candidate_resolved = candidate.resolve(strict=False)
        if candidate_resolved.parent != base_resolved:
            raise ValueError("session path escapes the trace store")
        return candidate

    def _store_file(self, name: str, kind: str) -> Path:
        """Return a non-symlinked regular-file location under the store."""
        candidate = self.base_dir / name
        if candidate.is_symlink():
            raise ValueError(f"refusing symlinked {kind}")
        if candidate.resolve(strict=False).parent != self.base_dir.resolve():
            raise ValueError(f"{kind} escapes the trace store")
        if candidate.exists() and not candidate.is_file():
            raise ValueError(f"{kind} is not a regular file")
        return candidate

    def _auxiliary_dir(self, name: str, kind: str) -> Path:
        """Return a safe direct child directory without following symlinks."""
        candidate = self.base_dir / name
        if candidate.is_symlink():
            raise ValueError(f"refusing symlinked {kind}")
        if candidate.resolve(strict=False).parent != self.base_dir.resolve():
            raise ValueError(f"{kind} escapes the trace store")
        if candidate.exists() and not candidate.is_dir():
            raise ValueError(f"{kind} is not a directory")
        return candidate

    def _session_file(self, session_id: str, name: str, kind: str) -> Path:
        directory = self._session_dir(session_id)
        candidate = directory / name
        if candidate.is_symlink():
            raise ValueError(f"refusing symlinked {kind}")
        if candidate.resolve(strict=False).parent != directory.resolve(strict=False):
            raise ValueError(f"{kind} escapes the session directory")
        if candidate.exists() and not candidate.is_file():
            raise ValueError(f"{kind} is not a regular file")
        return candidate

    def checkpoint_path(self, session_id: str) -> Path:
        """Return a checkpoint path after rejecting symlink components."""
        session_id = validate_stored_id(session_id, "stored session ID")
        directory = self._auxiliary_dir("checkpoints", "checkpoint directory")
        candidate = directory / f"{session_id}.md"
        if candidate.is_symlink():
            raise ValueError("refusing symlinked checkpoint")
        if candidate.resolve(strict=False).parent != directory.resolve(strict=False):
            raise ValueError("checkpoint escapes the trace store")
        if candidate.exists() and not candidate.is_file():
            raise ValueError("checkpoint is not a regular file")
        return candidate

    def _external_record_plan(self, session_ids: set[str]) -> tuple[list[dict], list[Path], list[tuple[Path, str]]]:
        """Collect matching external records and precompute safe deletion edits."""
        records: list[dict] = []
        remove_files: list[Path] = []
        rewrites: list[tuple[Path, str]] = []

        approvals = self._auxiliary_dir(".approvals", "approval directory")
        if approvals.exists():
            for path in sorted(approvals.glob("*.json")):
                if path.is_symlink():
                    raise ValueError("refusing symlinked approval record")
                if not path.is_file():
                    raise ValueError("approval record is not a regular file")
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict) and record.get("session_id") in session_ids:
                    records.append({
                        "kind": "approval",
                        "path": path.relative_to(self.base_dir).as_posix(),
                        "record": record,
                    })
                    remove_files.append(path)

        def scan_jsonl(path: Path, kind: str) -> None:
            if path.is_symlink():
                raise ValueError(f"refusing symlinked {kind} records")
            if path.exists() and not path.is_file():
                raise ValueError(f"{kind} records are not a regular file")
            if not path.exists():
                return
            with open(path, "r", encoding="utf-8", newline="") as record_file:
                lines = record_file.readlines()
            kept: list[str] = []
            changed = False
            for line_number, line in enumerate(lines, 1):
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    kept.append(line)
                    continue
                if isinstance(record, dict) and record.get("session_id") in session_ids:
                    records.append({
                        "kind": kind,
                        "path": path.relative_to(self.base_dir).as_posix(),
                        "line": line_number,
                        "record": record,
                    })
                    changed = True
                else:
                    kept.append(line)
            if changed:
                rewrites.append((path, "".join(kept)))

        datasets = self._auxiliary_dir("datasets", "dataset directory")
        if datasets.exists():
            for path in sorted(datasets.glob("*.jsonl")):
                scan_jsonl(path, "dataset")
        scan_jsonl(self._store_file("retention.log", "retention log"), "retention")
        return records, remove_files, rewrites

    def external_session_records(self, session_ids: set[str]) -> list[dict]:
        """Return approval, dataset, and retention records for sessions."""
        records, _, _ = self._external_record_plan(session_ids)
        return records

    def _delete_external_session_records_locked(self, session_ids: set[str]) -> None:
        """Atomically filter external per-session records, preserving others."""
        _, remove_files, rewrites = self._external_record_plan(session_ids)
        for path, text in rewrites:
            _write_atomic(path, text)
        touched_directories: set[Path] = set()
        for path in remove_files:
            path.unlink(missing_ok=True)
            touched_directories.add(path.parent)
        for directory in touched_directories:
            _fsync_directory(directory)

    def write_lock(self):
        """Return the store-wide write lock context manager."""
        return _store_write_lock(self.base_dir)

    def _session_was_erased(self, session_id: str) -> bool:
        """Return True when a tenant-erasure tombstone covers session_id."""
        audit_path = self._store_file(
            "tenant-deletions.ndjson", "tenant deletion audit"
        )
        if not audit_path.exists():
            return False
        digest = hashlib.sha256(session_id.encode()).hexdigest()
        try:
            mtime_ns = audit_path.stat().st_mtime_ns
            if mtime_ns == self._erasure_audit_mtime_ns:
                return digest in self._erased_session_hashes
            erased_hashes: set[str] = set()
            for line in audit_path.read_text(encoding="utf-8").splitlines():
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                values = record.get("session_id_sha256", [])
                if isinstance(values, list):
                    erased_hashes.update(str(value) for value in values)
            self._erased_session_hashes = erased_hashes
            self._erasure_audit_mtime_ns = mtime_ns
        except OSError:
            return False
        return digest in self._erased_session_hashes

    def _journal_dir(self) -> Path:
        return self._auxiliary_dir(".tenant-journal", "tenant journal directory")

    def _audit_has_operation(self, path: Path, operation_id: str) -> bool:
        if path.is_symlink():
            raise ValueError("refusing symlinked tenant audit")
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    if json.loads(line).get("operation_id") == operation_id:
                        return True
                except json.JSONDecodeError:
                    continue
        except OSError:
            return False
        return False

    def _append_tenant_audit(self, record: dict) -> Path:
        path = self._store_file("tenant-audit.ndjson", "tenant audit")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as audit_file:
            audit_file.write(json.dumps(record, separators=(",", ":")) + "\n")
            audit_file.flush()
            os.fsync(audit_file.fileno())
        return path

    def _complete_tag_journal_locked(self, journal_path: Path, record: dict) -> int:
        """Idempotently roll a durable tenant-tag intent forward."""
        session_id = validate_stored_id(record.get("session_id", ""), "stored session ID")
        tenant_id = validate_tenant_id(record.get("tenant_id", ""))
        operation_id = str(record.get("operation_id", ""))
        if not operation_id:
            raise ValueError("tenant journal is missing operation_id")
        path = self._session_file(session_id, "events.ndjson", "event stream")
        meta_path = self._session_file(session_id, "meta.json", "session metadata")
        meta = self.load_meta(session_id)
        if meta.tenant_id and meta.tenant_id != tenant_id:
            raise ValueError(
                f"session {session_id} is already assigned to another tenant"
            )
        raw_text = path.read_text(encoding="utf-8") if path.exists() else ""
        raw_lines = [line for line in raw_text.splitlines() if line]
        tagged_lines: list[str] = []
        previous_line = ""
        for raw_line in raw_lines:
            event = TraceEvent.from_json(raw_line)
            if event.session_id and event.session_id != session_id:
                raise ValueError("event session ID does not match its directory")
            event.session_id = session_id
            if event.tenant_id and event.tenant_id != tenant_id:
                raise ValueError(f"event {event.event_id} belongs to another tenant")
            event.tenant_id = tenant_id
            event.prev_hash = (
                hashlib.sha256(previous_line.encode()).hexdigest()
                if previous_line else ""
            )
            serialised = event.to_json()
            tagged_lines.append(serialised)
            previous_line = serialised
        tagged_text = "\n".join(tagged_lines) + ("\n" if tagged_lines else "")
        new_terminal_hash = (
            hashlib.sha256(tagged_lines[-1].encode()).hexdigest()
            if tagged_lines else ""
        )

        _write_atomic(path, tagged_text)
        record["phase"] = "events_replaced"
        record["new_terminal_hash"] = new_terminal_hash
        _write_atomic(journal_path, json.dumps(record, separators=(",", ":")))

        meta.tenant_id = tenant_id
        _write_atomic(meta_path, meta.to_json())
        record["phase"] = "metadata_replaced"
        _write_atomic(journal_path, json.dumps(record, separators=(",", ":")))

        record["phase"] = "commit"
        _write_atomic(journal_path, json.dumps(record, separators=(",", ":")))

        audit_path = self._store_file("tenant-audit.ndjson", "tenant audit")
        if not self._audit_has_operation(audit_path, operation_id):
            self._append_tenant_audit({
                "action": "session_tenant_assigned",
                "assigned_at": time.time(),
                "operation_id": operation_id,
                "session_id_sha256": hashlib.sha256(session_id.encode()).hexdigest(),
                "tenant_id_sha256": hashlib.sha256(tenant_id.encode()).hexdigest(),
                "previous_terminal_hash": record.get("previous_terminal_hash", ""),
                "new_terminal_hash": new_terminal_hash,
            })
        record["phase"] = "committed"
        _write_atomic(journal_path, json.dumps(record, separators=(",", ":")))
        journal_path.unlink(missing_ok=True)
        _fsync_directory(journal_path.parent)
        return len(tagged_lines)

    def _recover_tag_journals(self) -> None:
        journal_dir = self._journal_dir()
        if not journal_dir.exists() or journal_dir.is_symlink():
            return
        with _store_write_lock(self.base_dir):
            for journal_path in sorted(journal_dir.glob("tag-*.json")):
                if journal_path.is_symlink():
                    continue
                try:
                    record = json.loads(journal_path.read_text(encoding="utf-8"))
                    if record.get("operation") != "tag_session":
                        continue
                    self._complete_tag_journal_locked(journal_path, record)
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    # Keep the journal for an operator or a later successful
                    # recovery attempt; never guess through corrupt intent.
                    continue

    def _deletion_audit_has(self, operation_id: str, status: str) -> bool:
        path = self._store_file(
            "tenant-deletions.ndjson", "tenant deletion audit"
        )
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    record.get("operation_id") == operation_id
                    and record.get("status") == status
                ):
                    return True
        except OSError:
            return False
        return False

    def _append_deletion_audit(self, record: dict) -> Path:
        path = self._store_file(
            "tenant-deletions.ndjson", "tenant deletion audit"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as audit_file:
            audit_file.write(json.dumps(record, separators=(",", ":")) + "\n")
            audit_file.flush()
            os.fsync(audit_file.fileno())
        return path

    def _hashed_deletion_record(self, journal: dict, status: str) -> dict:
        tenant_id = journal["tenant_id"]
        items = journal.get("sessions", [])
        failed = [item["session_id"] for item in items if item.get("failed")]
        deleted = [item for item in items if item.get("deleted")]
        record = {
            "action": "tenant_data_deletion",
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "operation_id": journal["operation_id"],
            "tenant_id_sha256": hashlib.sha256(tenant_id.encode()).hexdigest(),
            "session_id_sha256": [
                hashlib.sha256(item["session_id"].encode()).hexdigest()
                for item in items
            ],
            "workspace_id_sha256": (
                hashlib.sha256(self.workspace_id.encode()).hexdigest()
                if self.workspace_id else ""
            ),
            "sessions_deleted": len(deleted),
            "events_deleted": sum(
                int(item.get("event_count", 0)) for item in deleted
            ),
            "status": status,
        }
        if failed:
            record["failed_session_id_sha256"] = [
                hashlib.sha256(session_id.encode()).hexdigest()
                for session_id in failed
            ]
        return record

    def _cleanup_deleted_hook_state(self, deleted_ids: set[str]) -> None:
        """Remove scoped hook state and unambiguous legacy root markers."""
        def suffix_session_id(suffix: str) -> str | None:
            if suffix.startswith(".v2."):
                parts = suffix.split(".")
                if len(parts) != 5 or not re.fullmatch(r"[0-9a-f]{64}", parts[4]):
                    return None
                try:
                    session_id = bytes.fromhex(parts[3]).decode("utf-8")
                    return validate_stored_id(session_id, "hook state session ID")
                except (UnicodeDecodeError, ValueError):
                    return None
            if not suffix.startswith("."):
                return None
            try:
                raw_session_id = validate_stored_id(
                    suffix[1:], "legacy hook state suffix"
                )
            except ValueError:
                return None
            return raw_session_id[:16]

        def has_surviving_session(session_id: str) -> bool:
            # Root markers predate workspace-scoped state. Do not guess which
            # workspace they belong to when the same session ID survives in
            # any flat or workspace store.
            flat = self.storage_root / session_id
            if not flat.is_symlink() and (flat / "meta.json").is_file():
                return True
            workspaces = self.storage_root / "workspaces"
            if not workspaces.exists() or workspaces.is_symlink():
                return False
            try:
                directories = workspaces.iterdir()
                for workspace in directories:
                    if workspace.is_symlink() or not workspace.is_dir():
                        continue
                    candidate = workspace / session_id
                    if not candidate.is_symlink() and (candidate / "meta.json").is_file():
                        return True
            except OSError:
                # Preserve legacy state if uniqueness cannot be established.
                return True
            return False

        def cleanup(root: Path, *, require_unambiguous: bool) -> None:
            if not root.exists():
                return
            for marker in root.glob(".active-session*"):
                if marker.is_symlink():
                    continue
                try:
                    session_id = marker.read_text(encoding="utf-8").strip()
                    if session_id not in deleted_ids:
                        continue
                    if require_unambiguous and has_surviving_session(session_id):
                        continue
                    suffix = marker.name[len(".active-session"):]
                    marker.unlink()
                    (root / f".pending-calls{suffix}.json").unlink(missing_ok=True)
                except OSError:
                    continue
            # Markerless pending calls are deleted only when their safe,
            # attributable suffix maps to one of the erased local sessions.
            for pending in root.glob(".pending-calls*.json"):
                if pending.is_symlink():
                    continue
                suffix = pending.name[len(".pending-calls"):-len(".json")]
                marker = root / f".active-session{suffix}"
                try:
                    if (
                        not marker.exists()
                        and suffix_session_id(suffix) in deleted_ids
                    ):
                        pending.unlink()
                except OSError:
                    continue

        if self.base_dir != self.storage_root:
            cleanup(self.base_dir, require_unambiguous=False)
        cleanup(self.storage_root, require_unambiguous=True)

    def _complete_delete_journal_locked(
        self, journal_path: Path, journal: dict,
    ) -> dict:
        """Idempotently roll a durable tenant-deletion intent forward."""
        tenant_id = validate_tenant_id(journal.get("tenant_id", ""))
        operation_id = str(journal.get("operation_id", ""))
        if not operation_id:
            raise ValueError("deletion journal is missing operation_id")
        items = journal.get("sessions", [])
        if not isinstance(items, list):
            raise ValueError("deletion journal sessions must be a list")
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("invalid deletion journal session")
            validate_stored_id(item.get("session_id", ""), "stored session ID")

        # Validate every remaining core path before removing shared external
        # records. A symlinked checkpoint, meta, or event stream must leave
        # both the trace and any outside target untouched.
        for item in items:
            if item.get("deleted"):
                continue
            session_id = item["session_id"]
            session_dir = self._session_dir(session_id)
            self.checkpoint_path(session_id)
            if session_dir.exists():
                meta = self.load_meta(session_id)
                self._session_file(session_id, "events.ndjson", "event stream")
                if meta.tenant_id != tenant_id:
                    raise ValueError(
                        "deletion journal tenant does not match session"
                    )
        if not journal.get("external_records_cleaned"):
            self._external_record_plan({item["session_id"] for item in items})

        # Persist a hash-only tombstone only after every path passes preflight,
        # but before deleting trace bytes or external records.
        if not self._deletion_audit_has(operation_id, "pending"):
            self._append_deletion_audit(
                self._hashed_deletion_record(journal, "pending")
            )
        self._erasure_audit_mtime_ns = -1

        if not journal.get("external_records_cleaned"):
            self._delete_external_session_records_locked({
                item["session_id"] for item in items
            })
            journal["external_records_cleaned"] = True
            journal["phase"] = "external_records_cleaned"
            _write_atomic(journal_path, json.dumps(journal, separators=(",", ":")))

        for item in items:
            if item.get("deleted"):
                continue
            session_id = item["session_id"]
            try:
                session_dir = self._session_dir(session_id)
                if session_dir.exists():
                    meta = self.load_meta(session_id)
                    if meta.tenant_id != tenant_id:
                        raise ValueError(
                            "deletion journal tenant does not match session"
                        )
                    checkpoint = self.checkpoint_path(session_id)
                    checkpoint.unlink(missing_ok=True)
                    shutil.rmtree(session_dir)
                item["deleted"] = True
                item.pop("failed", None)
            except (OSError, ValueError):
                item["failed"] = True
            _write_atomic(journal_path, json.dumps(journal, separators=(",", ":")))

        deleted_ids = {
            item["session_id"] for item in items if item.get("deleted")
        }
        self._cleanup_deleted_hook_state(deleted_ids)
        failed = [item["session_id"] for item in items if item.get("failed")]
        status = "partial" if failed else "completed"
        if not self._deletion_audit_has(operation_id, status):
            self._append_deletion_audit(
                self._hashed_deletion_record(journal, status)
            )
        self._erasure_audit_mtime_ns = -1
        if failed:
            journal["phase"] = "partial"
            _write_atomic(journal_path, json.dumps(journal, separators=(",", ":")))
        else:
            journal_path.unlink(missing_ok=True)
            _fsync_directory(journal_path.parent)
        deleted_items = [item for item in items if item.get("deleted")]
        return {
            "tenant_id": tenant_id,
            "deleted_sessions": len(deleted_items),
            "deleted_events": sum(
                int(item.get("event_count", 0)) for item in deleted_items
            ),
            "failed_session_ids": tuple(failed),
            "audit_path": self._store_file(
                "tenant-deletions.ndjson", "tenant deletion audit"
            ),
        }

    def _recover_delete_journals(self) -> None:
        journal_dir = self._journal_dir()
        if not journal_dir.exists() or journal_dir.is_symlink():
            return
        with _store_write_lock(self.base_dir):
            for journal_path in sorted(journal_dir.glob("delete-*.json")):
                if journal_path.is_symlink():
                    continue
                try:
                    journal = json.loads(journal_path.read_text(encoding="utf-8"))
                    if journal.get("operation") != "delete_tenant":
                        continue
                    self._complete_delete_journal_locked(journal_path, journal)
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    # Keep malformed or temporarily unexecutable intent for
                    # an operator or a later successful startup.
                    continue

    def recover_delete_journals(self) -> None:
        """Retry interrupted tenant deletions for this scoped store."""
        self._recover_delete_journals()

    def _enforce_parent_tenant(self, meta: SessionMeta, *, inherit: bool) -> None:
        """Resolve a parent in this store and enforce a safe tenant boundary."""
        if not meta.parent_session_id:
            return
        if meta.parent_session_id == meta.session_id:
            raise ValueError("a session cannot be its own parent")
        try:
            parent = self.load_meta(meta.parent_session_id)
        except FileNotFoundError as exc:
            raise ValueError("parent session must exist in the same scoped store") from exc
        if parent.tenant_id == meta.tenant_id:
            return  # includes the explicit legacy untagged→untagged case
        if inherit and parent.tenant_id and not meta.tenant_id:
            meta.tenant_id = parent.tenant_id
            return
        if not parent.tenant_id and meta.tenant_id:
            raise ValueError("tag the legacy untagged parent before linking a tagged child")
        raise ValueError("cross-tenant parent link rejected")

    def _enforce_tag_tenant_links(self, meta: SessionMeta, tenant_id: str) -> None:
        """Reject a tag that would split an existing parent/child tree."""
        if meta.parent_session_id:
            try:
                parent = self.load_meta(meta.parent_session_id)
            except FileNotFoundError as exc:
                raise ValueError(
                    "parent session must exist in the same scoped store"
                ) from exc
            if parent.tenant_id != tenant_id:
                raise ValueError(
                    "tagging session would create a cross-tenant parent link"
                )
        for child in self.list_sessions():
            if child.parent_session_id != meta.session_id:
                continue
            if child.tenant_id != tenant_id:
                raise ValueError(
                    "tagging session would create a cross-tenant child link"
                )

    def create_session(self, meta: SessionMeta) -> Path:
        # Stamp workspace_id onto meta so it's visible in exports/reports
        with _store_write_lock(self.base_dir):
            _validate_meta_ids(meta, strict_session=True)
            if self._session_was_erased(meta.session_id):
                raise ValueError(f"session {meta.session_id} was erased and cannot be recreated")
            if self.workspace_id and not getattr(meta, "workspace_id", ""):
                meta.workspace_id = self.workspace_id
            if not meta.tenant_id:
                meta.tenant_id = os.environ.get(_TENANT_ENV, "").strip()
            if meta.tenant_id:
                meta.tenant_id = validate_tenant_id(meta.tenant_id)
            self._enforce_parent_tenant(meta, inherit=True)
            self._redact_meta(meta)
            d = self._session_dir(meta.session_id)
            existing_meta_path = self._session_file(
                meta.session_id, "meta.json", "session metadata"
            )
            if existing_meta_path.exists():
                existing = SessionMeta.from_json(existing_meta_path.read_text())
                if existing.tenant_id and not meta.tenant_id:
                    meta.tenant_id = existing.tenant_id
                elif existing.tenant_id != meta.tenant_id:
                    raise ValueError(
                        f"session {meta.session_id} tenant cannot be changed during creation"
                    )
                if existing.parent_session_id != meta.parent_session_id:
                    raise ValueError("use update_meta to change a session parent")
            d.mkdir(parents=True, exist_ok=True)
            self._session_file(
                meta.session_id, "meta.json", "session metadata"
            ).write_text(meta.to_json())
            # create empty events file
            self._session_file(
                meta.session_id, "events.ndjson", "event stream"
            ).touch()
            return d

    def append_event(self, session_id: str, event: TraceEvent) -> None:
        try:
            session_id = validate_session_id(session_id)
        except ValueError:
            session_id = validate_stored_id(session_id, "stored session ID")
            if not self._session_file(
                session_id, "meta.json", "session metadata"
            ).exists():
                raise
        if event.session_id and event.session_id != session_id:
            raise ValueError("event session ID does not match target session")
        event.session_id = session_id
        # The only supported empty→tagged metadata transition is tag_session,
        # which atomically updates historical events and records hash evidence.
        if event.tenant_id:
            if self._session_was_erased(session_id):
                raise ValueError(f"session {session_id} was erased and cannot accept events")
            existing = self.load_meta(session_id)
            if not existing.tenant_id:
                self.tag_session(session_id, event.tenant_id)
        f = self._session_file(session_id, "events.ndjson", "event stream")
        with _store_write_lock(self.base_dir):
            if self._session_was_erased(session_id):
                raise ValueError(f"session {session_id} was erased and cannot accept events")
            # Session metadata is authoritative.  This prevents a changed
            # environment (or malformed remote event) from mixing tenants in
            # one session. An untagged legacy session adopts its first tag.
            try:
                meta = self.load_meta(session_id)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                meta = None
            if meta is not None:
                if meta.tenant_id:
                    if event.tenant_id and event.tenant_id != meta.tenant_id:
                        raise ValueError(
                            f"event tenant {event.tenant_id!r} does not match "
                            f"session tenant {meta.tenant_id!r}"
                        )
                    event.tenant_id = validate_tenant_id(meta.tenant_id)
            elif event.tenant_id:
                event.tenant_id = validate_tenant_id(event.tenant_id)
            self._redact_event(event)
            f.parent.mkdir(parents=True, exist_ok=True)
            with open(f, "a+", encoding="utf-8") as fh:
                # Compute the hash and append while holding one lock so a
                # concurrent tenant-tag operation cannot split the chain.
                if not event.prev_hash:
                    fh.seek(0)
                    text = fh.read()
                    last_line = text.rstrip("\n").rsplit("\n", 1)[-1] if text.strip() else ""
                    event.prev_hash = hashlib.sha256(last_line.encode()).hexdigest() if last_line else ""
                fh.write(event.to_json() + "\n")
                fh.flush()

    def update_meta(self, meta: SessionMeta) -> None:
        _validate_meta_ids(meta)
        with _store_write_lock(self.base_dir):
            f = self._session_file(
                meta.session_id, "meta.json", "session metadata"
            )
            if not f.exists():
                raise FileNotFoundError(f"session not found: {meta.session_id}")
            existing = SessionMeta.from_json(f.read_text())
            if existing.session_id != meta.session_id:
                raise ValueError("metadata session ID does not match its directory")
            if existing.tenant_id != meta.tenant_id:
                if not existing.tenant_id and meta.tenant_id:
                    raise ValueError("use tag_session for the first tenant assignment")
                raise ValueError("session tenant cannot be cleared or changed")
            if meta.tenant_id:
                meta.tenant_id = validate_tenant_id(meta.tenant_id)
            self._enforce_parent_tenant(meta, inherit=False)
            self._redact_meta(meta)
            _write_atomic(f, meta.to_json())

    def load_meta(self, session_id: str) -> SessionMeta:
        f = self._session_file(session_id, "meta.json", "session metadata")
        meta = SessionMeta.from_json(f.read_text())
        if meta.session_id != validate_stored_id(session_id, "stored session ID"):
            raise ValueError("metadata session ID does not match its directory")
        _validate_meta_ids(meta)
        return meta

    def load_events(self, session_id: str) -> list[TraceEvent]:
        session_id = validate_stored_id(session_id, "stored session ID")
        f = self._session_file(session_id, "events.ndjson", "event stream")
        meta = self.load_meta(session_id)
        events = []
        for line in f.read_text().strip().splitlines():
            if line:
                event = TraceEvent.from_json(line)
                if event.session_id and event.session_id != session_id:
                    raise ValueError("event session ID does not match its directory")
                event.session_id = session_id
                if meta.tenant_id and event.tenant_id and event.tenant_id != meta.tenant_id:
                    raise ValueError("event tenant does not match session metadata")
                events.append(event)
        # Session-level tenancy is authoritative and also makes the single
        # pre-watch event of an active session tenant-aware to all readers.
        tenant_id = meta.tenant_id
        if tenant_id:
            for event in events:
                event.tenant_id = tenant_id
        return events

    def list_sessions(self, tenant_id: str | None = None) -> list[SessionMeta]:
        """Return valid sessions sorted newest first by started_at, then descending session ID."""
        if not self.base_dir.exists():
            return []
        sessions = []
        for d in self.base_dir.iterdir():
            if d.is_symlink() or not d.is_dir():
                continue
            meta_file = d / "meta.json"
            if meta_file.is_symlink():
                continue
            if meta_file.exists() and meta_file.is_file():
                try:
                    meta = SessionMeta.from_json(meta_file.read_text())
                    _validate_meta_ids(meta)
                    validate_stored_id(d.name, "stored session directory")
                    if meta.session_id != d.name:
                        continue
                    if tenant_id is None or meta.tenant_id == tenant_id:
                        sessions.append(meta)
                except (json.JSONDecodeError, TypeError, ValueError, OSError):
                    continue
        return sorted(
            sessions,
            key=lambda meta: (meta.started_at, meta.session_id),
            reverse=True,
        )

    def list_sessions_strict(
        self, tenant_id: str | None = None,
    ) -> list[SessionMeta]:
        """Strictly discover sessions for report/export/erasure workflows.

        Ordinary listing intentionally skips corrupt entries. Tenant
        administration must instead fail closed so it cannot report an
        incomplete result or claim deletion succeeded while data was omitted.
        """
        if self.base_dir.is_symlink():
            raise ValueError("refusing symlinked trace store")
        if not self.base_dir.exists():
            return []
        if not self.base_dir.is_dir():
            raise ValueError("trace store is not a directory")

        sessions: list[SessionMeta] = []
        for entry in sorted(self.base_dir.iterdir(), key=lambda item: item.name):
            if entry.is_symlink():
                raise ValueError(f"unsafe symlinked trace store entry: {entry.name}")
            if not entry.is_dir():
                continue

            meta_path = entry / "meta.json"
            events_path = entry / "events.ndjson"
            is_flat_workspace_root = (
                not self.workspace_id and entry.name == "workspaces"
            )
            is_auxiliary = entry.name in _TENANT_ADMIN_AUXILIARY_DIRS
            if (is_flat_workspace_root or is_auxiliary) and not (
                meta_path.exists() or events_path.exists()
            ):
                continue

            try:
                session_id = validate_stored_id(
                    entry.name, "stored session directory"
                )
            except ValueError as exc:
                raise ValueError(
                    f"unsafe session directory in tenant store: {entry.name!r}"
                ) from exc
            if not meta_path.exists() or not events_path.exists():
                raise ValueError(
                    f"unrecognized or incomplete session directory: {session_id}"
                )
            if meta_path.is_symlink() or events_path.is_symlink():
                raise ValueError(f"unsafe symlinked core file for session {session_id}")
            if not meta_path.is_file() or not events_path.is_file():
                raise ValueError(f"invalid core file for session {session_id}")
            try:
                meta = self.load_meta(session_id)
                self.load_events(session_id)
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"corrupt session in tenant store: {session_id}"
                ) from exc
            if tenant_id is None or meta.tenant_id == tenant_id:
                sessions.append(meta)

        return sorted(
            sessions,
            key=lambda meta: (meta.started_at, meta.session_id),
            reverse=True,
        )

    def get_latest_session(self, tenant_id: str | None = None) -> SessionMeta | None:
        """Return the newest session metadata, or None when the store is empty."""
        sessions = self.list_sessions(tenant_id=tenant_id)
        if not sessions:
            return None
        return sessions[0]

    def get_latest_session_id(self, tenant_id: str | None = None) -> str | None:
        """Return the newest session ID, or None when the store is empty."""
        latest = self.get_latest_session(tenant_id=tenant_id)
        if not latest:
            return None
        return latest.session_id

    def session_exists(self, session_id: str) -> bool:
        return self._session_file(
            session_id, "meta.json", "session metadata"
        ).exists()

    def find_session(self, prefix: str, tenant_id: str | None = None) -> str | None:
        """Find a session by prefix match."""
        validate_stored_id(prefix, "session ID prefix")
        if not self.base_dir.exists():
            return None
        for d in self.base_dir.iterdir():
            if d.is_symlink() or not d.is_dir() or not d.name.startswith(prefix):
                continue
            try:
                validate_stored_id(d.name, "stored session directory")
                meta = self.load_meta(d.name)
                if tenant_id is None or meta.tenant_id == tenant_id:
                    return d.name
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        return None

    def tag_session(self, session_id: str, tenant_id: str) -> int:
        """Assign *tenant_id* to a session and every persisted event.

        Returns the number of existing events tagged.  Retagging a session to
        a different tenant is rejected so a query boundary cannot be moved by
        accident.  The event hash chain is rebuilt while holding the same
        advisory lock used by append_event.
        """
        session_id = validate_stored_id(session_id, "stored session ID")
        tenant_id = validate_tenant_id(tenant_id)
        path = self._session_file(session_id, "events.ndjson", "event stream")
        meta_path = self._session_file(session_id, "meta.json", "session metadata")
        with _store_write_lock(self.base_dir):
            meta = self.load_meta(session_id)
            self._store_file("tenant-audit.ndjson", "tenant audit")
            if meta.tenant_id and meta.tenant_id != tenant_id:
                raise ValueError(
                    f"session {session_id} is already assigned to tenant {meta.tenant_id!r}"
                )
            self._enforce_tag_tenant_links(meta, tenant_id)
            raw_text = path.read_text(encoding="utf-8") if path.exists() else ""
            raw_lines = [line for line in raw_text.splitlines() if line]
            for raw_line in raw_lines:
                event = TraceEvent.from_json(raw_line)
                if event.session_id and event.session_id != session_id:
                    raise ValueError("event session ID does not match its directory")
                if event.tenant_id and event.tenant_id != tenant_id:
                    raise ValueError(f"event {event.event_id} belongs to another tenant")
            previous_terminal_hash = (
                hashlib.sha256(raw_lines[-1].encode()).hexdigest() if raw_lines else ""
            )
            original_meta_text = meta_path.read_text(encoding="utf-8")
            operation_id = uuid.uuid4().hex
            journal_dir = self._journal_dir()
            journal_dir.mkdir(parents=True, exist_ok=True)
            journal_path = journal_dir / f"tag-{operation_id}.json"
            record = {
                "operation": "tag_session",
                "operation_id": operation_id,
                "session_id": session_id,
                "tenant_id": tenant_id,
                "phase": "intent",
                "previous_terminal_hash": previous_terminal_hash,
            }
            _write_atomic(journal_path, json.dumps(record, separators=(",", ":")))
            try:
                return self._complete_tag_journal_locked(journal_path, record)
            except Exception:
                # Once both trace files are durable, commit/committed phases
                # are roll-forward only. This prevents a post-audit failure
                # from rolling data back while leaving false audit evidence.
                if record.get("phase") in {"commit", "committed"}:
                    raise
                # Pre-commit exceptions can safely restore the exact input.
                _write_atomic(path, raw_text)
                if meta_path.read_text(encoding="utf-8") != original_meta_text:
                    _write_atomic(meta_path, original_meta_text)
                journal_path.unlink(missing_ok=True)
                _fsync_directory(journal_path.parent)
                raise

    def annotations_path(self, session_id: str) -> Path:
        """Return the path to the annotations sidecar file."""
        return self._session_file(
            session_id, "annotations.jsonl", "annotations sidecar"
        )
