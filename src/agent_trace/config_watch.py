"""AGENTS.md change detector: flag sessions that ran after a config change.

Hashes a set of config files (AGENTS.md, system prompts, tool configs) each
time ``agent-strace config-watch snapshot`` runs, and detects when those files
changed between snapshots. Nothing takes snapshots automatically: run
``snapshot`` yourself, for example after each edit or in CI. Sessions that
started after a snapshot recorded a change are flagged as "potentially
affected" so teams know to re-evaluate agent behaviour.

Usage:
    # Snapshot current config state
    agent-strace config-watch snapshot

    # Check whether config has changed since last snapshot
    agent-strace config-watch check

    # Show full change history
    agent-strace config-watch history

    # Show which sessions ran after each config change
    agent-strace config-watch affected [--since 7d]

Config files tracked by default (``DEFAULT_WATCH_PATHS``):
    AGENTS.md, CLAUDE.md, .agent-strace-policy.json, .agent-strace-lint.json,
    .agent-watch.json, system_prompt.txt, system_prompt.md, .cursorrules,
    .github/copilot-instructions.md

Additional paths can be added via --watch or .agent-strace-watch.json.

The CLI watches files in the current directory, so run it from the repository
root. Snapshots are kept in ``.config-snapshots.json`` in the trace directory,
each tagged with the directory it hashed, so several repositories can share
one trace directory. The snapshots that older versions kept in
``.agent-traces`` next to the trace directory are still read, but never
written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from .fileutil import file_lock, write_atomic
from .inert import inert
from .models import usable_time
from .store import DEFAULT_TRACE_DIR, TraceStore


# ---------------------------------------------------------------------------
# Default watched paths (relative to workspace root)
# ---------------------------------------------------------------------------

DEFAULT_WATCH_PATHS: list[str] = [
    "AGENTS.md",
    "CLAUDE.md",
    ".agent-strace-policy.json",
    ".agent-strace-lint.json",
    ".agent-watch.json",
    "system_prompt.txt",
    "system_prompt.md",
    ".cursorrules",
    ".github/copilot-instructions.md",
]

_SNAPSHOT_NAME = ".config-snapshots.json"
_WATCH_CONFIG_FILE = ".agent-strace-watch.json"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FileSnapshot:
    path: str
    sha256: str          # "" if file does not exist
    mtime: float         # 0.0 if file does not exist
    exists: bool


@dataclass
class ConfigSnapshot:
    snapshot_id: str
    timestamp: float
    files: list[FileSnapshot]
    session_id: str = ""   # session that triggered this snapshot (if any)
    label: str = ""        # human label (e.g. "before deploy")
    workspace: str = ""    # real path of the hashed directory; "" in older snapshots

    def to_dict(self) -> dict:
        d: dict = {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "label": self.label,
        }
        if self.workspace:
            d["workspace"] = self.workspace
        d["files"] = [
            {"path": f.path, "sha256": f.sha256,
             "mtime": f.mtime, "exists": f.exists}
            for f in self.files
        ]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ConfigSnapshot":
        files = [
            FileSnapshot(
                path=f["path"],
                sha256=f.get("sha256", ""),
                mtime=f.get("mtime", 0.0),
                exists=f.get("exists", False),
            )
            for f in d.get("files", [])
        ]
        return cls(
            snapshot_id=d["snapshot_id"],
            timestamp=d["timestamp"],
            session_id=d.get("session_id", ""),
            label=d.get("label", ""),
            workspace=d.get("workspace", ""),
            files=files,
        )


@dataclass
class FileDiff:
    path: str
    change: str   # "added" | "removed" | "modified" | "unchanged"
    old_sha: str = ""
    new_sha: str = ""


@dataclass
class SnapshotDiff:
    snapshot_a_id: str
    snapshot_b_id: str
    timestamp_a: float
    timestamp_b: float
    changes: list[FileDiff]

    @property
    def has_changes(self) -> bool:
        return any(c.change != "unchanged" for c in self.changes)

    @property
    def changed_paths(self) -> list[str]:
        return [c.path for c in self.changes if c.change != "unchanged"]


# ---------------------------------------------------------------------------
# File hashing
# ---------------------------------------------------------------------------

def _hash_file(path: Path) -> str:
    """SHA-256 of file contents, or '' if file does not exist."""
    try:
        data = path.read_bytes()
        return hashlib.sha256(data).hexdigest()
    except (OSError, IOError):
        return ""


def _snapshot_file(root: Path, rel_path: str) -> FileSnapshot:
    p = root / rel_path
    exists = p.exists()
    if not exists:
        return FileSnapshot(path=rel_path, sha256="", mtime=0.0, exists=False)
    return FileSnapshot(
        path=rel_path,
        sha256=_hash_file(p),
        mtime=p.stat().st_mtime,
        exists=True,
    )


# ---------------------------------------------------------------------------
# Snapshot storage
# ---------------------------------------------------------------------------

def _snapshot_path(workspace_root: Path, trace_dir: str | Path | None = None) -> Path:
    if trace_dir is None:
        trace_dir = workspace_root / DEFAULT_TRACE_DIR
    return Path(trace_dir) / _SNAPSHOT_NAME


class SnapshotFileError(ValueError):
    """The snapshot file exists but cannot be read or parsed as a JSON list."""


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _snapshot_problem(d: object) -> str:
    """Return why *d* is not a usable snapshot entry, or '' if it is."""
    if not isinstance(d, dict):
        return "not a JSON object"
    if not isinstance(d.get("snapshot_id"), str):
        return "snapshot_id is missing or not a string"
    if not _is_number(d.get("timestamp")):
        return "timestamp is missing or not a number"
    for name in ("session_id", "label", "workspace"):
        if not isinstance(d.get(name, ""), str):
            return f"{name} is not a string"
    files = d.get("files", [])
    if not isinstance(files, list):
        return "files is not a list"
    for f in files:
        if not isinstance(f, dict) or not isinstance(f.get("path"), str):
            return "a files item has no string path"
        if not isinstance(f.get("sha256", ""), str):
            return "a files item has a non-string sha256"
        if not _is_number(f.get("mtime", 0.0)):
            return "a files item has a non-number mtime"
        if not isinstance(f.get("exists", False), bool):
            return "a files item has a non-boolean exists"
    return ""


def _read_snapshot_entries(workspace_root: Path, trace_dir: str | Path | None = None) -> list:
    """Return the raw entries of the snapshot file ([] if it does not exist).

    Raises SnapshotFileError if the file exists but is not a readable JSON list.
    """
    p = _snapshot_path(workspace_root, trace_dir)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise SnapshotFileError(
            f"cannot parse {inert(p)}: {inert(type(exc).__name__)}"
        ) from exc
    if not isinstance(data, list):
        raise SnapshotFileError(f"cannot parse {inert(p)}: top level is not a JSON list")
    return data


def _owned_snapshots(
    workspace_root: Path,
    trace_dir: str | Path,
    data: list,
) -> list[ConfigSnapshot]:
    """Return the valid snapshots of *workspace_root* in *data*, the entries
    of the snapshot file in *trace_dir*, skipping (with a warning) entries
    that do not parse.

    Entries of other directories are skipped without a warning. An entry
    with no ``workspace`` was written before snapshots were tagged, into
    ``<directory>/.agent-traces``, so it belongs to the directory that holds
    *trace_dir*.
    """
    here = os.path.normcase(os.path.realpath(workspace_root))
    legacy_owner = os.path.normcase(
        os.path.realpath(os.path.dirname(os.path.abspath(trace_dir)))
    )
    snapshots: list[ConfigSnapshot] = []
    for index, d in enumerate(data):
        problem = _snapshot_problem(d)
        if problem:
            sys.stderr.write(
                f"warning: skipping snapshot {index} in "
                f"{inert(_snapshot_path(workspace_root, trace_dir))}: {problem}\n"
            )
            continue
        workspace = d.get("workspace", "")
        owner = os.path.normcase(workspace) if workspace else legacy_owner
        if owner == here:
            snapshots.append(ConfigSnapshot.from_dict(d))
    return snapshots


def _same_path(a: str | Path, b: str | Path) -> bool:
    return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))


def _legacy_snapshots(workspace_root: Path, trace_dir: str | Path) -> list[ConfigSnapshot]:
    """Return the snapshots of *workspace_root* that older versions kept for
    *trace_dir*, in ``.agent-traces`` of the directory that holds it.

    Only untagged entries are read: a tagged entry there belongs to the
    trace dir that wrote it. In the default layout that file is the
    snapshot file of *trace_dir*, so this returns []. An unreadable file
    yields [] and a warning on stderr. The file is never written.
    """
    legacy_dir = Path(os.path.dirname(os.path.abspath(trace_dir))) / DEFAULT_TRACE_DIR
    if _same_path(_snapshot_path(workspace_root, legacy_dir),
                  _snapshot_path(workspace_root, trace_dir)):
        return []
    try:
        data = _read_snapshot_entries(workspace_root, legacy_dir)
    except SnapshotFileError as exc:
        sys.stderr.write(f"warning: {exc}; treating as no snapshots\n")
        return []
    return [s for s in _owned_snapshots(workspace_root, legacy_dir, data) if not s.workspace]


def _load_snapshots(
    workspace_root: Path,
    trace_dir: str | Path | None = None,
) -> list[ConfigSnapshot]:
    """Return the valid snapshots of *workspace_root* in *trace_dir*,
    skipping (with a warning) entries that do not parse. An unreadable file
    yields no snapshots and a warning on stderr.

    The snapshots that older versions kept for *trace_dir* come first (see
    ``_legacy_snapshots``).
    """
    if trace_dir is None:
        trace_dir = workspace_root / DEFAULT_TRACE_DIR
    try:
        data = _read_snapshot_entries(workspace_root, trace_dir)
    except SnapshotFileError as exc:
        sys.stderr.write(f"warning: {exc}; treating as no snapshots\n")
        return []
    legacy = _legacy_snapshots(workspace_root, trace_dir)
    return legacy + _owned_snapshots(workspace_root, trace_dir, data)


def _save_snapshots(workspace_root: Path, snapshots: list[ConfigSnapshot]) -> None:
    p = _snapshot_path(workspace_root)
    write_atomic(p, json.dumps([s.to_dict() for s in snapshots], indent=2), allow_symlinks=True)


# ---------------------------------------------------------------------------
# Watch path resolution
# ---------------------------------------------------------------------------

def _load_watch_paths(workspace_root: Path, extra: list[str] | None = None) -> list[str]:
    """Merge default paths, .agent-strace-watch.json, and CLI extras."""
    paths = list(DEFAULT_WATCH_PATHS)

    cfg_file = workspace_root / _WATCH_CONFIG_FILE
    if cfg_file.exists():
        try:
            cfg = json.loads(cfg_file.read_text())
            paths.extend(cfg.get("watch", []))
        except Exception:
            pass

    if extra:
        paths.extend(extra)

    # Deduplicate preserving order
    seen: set[str] = set()
    result: list[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

def take_snapshot(
    workspace_root: Path,
    watch_paths: list[str],
    session_id: str = "",
    label: str = "",
    trace_dir: str | Path | None = None,
) -> ConfigSnapshot:
    """Hash all watched files and append a new snapshot.

    The snapshot is written to ``.config-snapshots.json`` in *trace_dir*
    (default ``<workspace_root>/.agent-traces``) and tagged with the real
    path of *workspace_root*. Existing entries are kept as they are,
    including ones that do not parse. Raises SnapshotFileError, without
    writing, if the snapshot file exists but is not a readable JSON list.
    """
    import uuid
    snap_id = uuid.uuid4().hex[:12]
    files = [_snapshot_file(workspace_root, p) for p in watch_paths]
    snapshot = ConfigSnapshot(
        snapshot_id=snap_id,
        timestamp=time.time(),
        session_id=session_id,
        label=label,
        workspace=os.path.realpath(workspace_root),
        files=files,
    )
    path = _snapshot_path(workspace_root, trace_dir)
    with file_lock(path):
        try:
            entries = _read_snapshot_entries(workspace_root, trace_dir)
        except SnapshotFileError as exc:
            raise SnapshotFileError(
                f"{exc}; refusing to write so existing snapshots are not lost. "
                "Fix or remove the file and retry."
            ) from exc
        entries.append(snapshot.to_dict())
        write_atomic(path, json.dumps(entries, indent=2), allow_symlinks=True)
    return snapshot


def diff_snapshots(a: ConfigSnapshot, b: ConfigSnapshot) -> SnapshotDiff:
    """Compare two snapshots and return per-file changes."""
    a_map = {f.path: f for f in a.files}
    b_map = {f.path: f for f in b.files}
    all_paths = sorted(set(a_map) | set(b_map))

    changes: list[FileDiff] = []
    for path in all_paths:
        fa = a_map.get(path)
        fb = b_map.get(path)

        if fa is None:
            changes.append(FileDiff(path=path, change="added",
                                    old_sha="", new_sha=fb.sha256 if fb else ""))
        elif fb is None:
            changes.append(FileDiff(path=path, change="removed",
                                    old_sha=fa.sha256, new_sha=""))
        elif not fa.exists and not fb.exists:
            pass  # both absent — skip
        elif not fa.exists and fb.exists:
            changes.append(FileDiff(path=path, change="added",
                                    old_sha="", new_sha=fb.sha256))
        elif fa.exists and not fb.exists:
            changes.append(FileDiff(path=path, change="removed",
                                    old_sha=fa.sha256, new_sha=""))
        elif fa.sha256 != fb.sha256:
            changes.append(FileDiff(path=path, change="modified",
                                    old_sha=fa.sha256, new_sha=fb.sha256))
        else:
            changes.append(FileDiff(path=path, change="unchanged",
                                    old_sha=fa.sha256, new_sha=fb.sha256))

    return SnapshotDiff(
        snapshot_a_id=a.snapshot_id,
        snapshot_b_id=b.snapshot_id,
        timestamp_a=a.timestamp,
        timestamp_b=b.timestamp,
        changes=changes,
    )


def find_affected_sessions(
    store: TraceStore,
    workspace_root: Path,
    since: float | None = None,
) -> list[tuple[str, str, list[str]]]:
    """Return (session_id, started_at_str, changed_paths) for sessions that
    ran after a config change relative to the previous snapshot.

    A session whose start is not a usable time is left out, since it cannot
    be placed after any change.
    """
    snapshots = _load_snapshots(workspace_root, store.storage_root)
    if len(snapshots) < 2:
        return []

    # Build a timeline of config changes: (timestamp, changed_paths)
    change_events: list[tuple[float, list[str]]] = []
    for i in range(1, len(snapshots)):
        diff = diff_snapshots(snapshots[i - 1], snapshots[i])
        if diff.has_changes:
            change_events.append((snapshots[i].timestamp, diff.changed_paths))

    if not change_events:
        return []

    all_sessions = store.list_sessions()

    results: list[tuple[str, str, list[str]]] = []
    for meta in all_sessions:
        started = usable_time(meta.started_at)
        if started is None:
            continue
        if since and started < since:
            continue
        # Find the most recent config change before this session
        relevant: list[str] = []
        for change_ts, changed_paths in change_events:
            if change_ts <= started:
                relevant = changed_paths  # keep the most recent
        if relevant:
            import datetime
            dt = datetime.datetime.fromtimestamp(started)
            results.append((meta.session_id, dt.strftime("%Y-%m-%d %H:%M"), relevant))

    return results


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt_ts(ts: float) -> str:
    import datetime
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def format_check(diff: SnapshotDiff, out: TextIO = sys.stdout) -> None:
    if not diff.has_changes:
        out.write("✓  No config changes since last snapshot.\n")
        return

    out.write(f"Config changed since snapshot {diff.snapshot_a_id[:8]} "
              f"({_fmt_ts(diff.timestamp_a)}):\n\n")
    for c in diff.changes:
        if c.change == "unchanged":
            continue
        symbol = {"added": "+", "removed": "-", "modified": "~"}.get(c.change, "?")
        out.write(f"  {symbol}  {c.path}  ({c.change})\n")
    out.write(f"\n{len(diff.changed_paths)} file(s) changed.\n")


def format_history(snapshots: list[ConfigSnapshot], out: TextIO = sys.stdout) -> None:
    if not snapshots:
        out.write("No snapshots recorded yet. Run: agent-strace config-watch snapshot\n")
        return

    out.write(f"Config snapshot history ({len(snapshots)} snapshot(s)):\n\n")
    for i, snap in enumerate(snapshots):
        label = f"  [{snap.label}]" if snap.label else ""
        session = f"  session={snap.session_id[:12]}" if snap.session_id else ""
        out.write(f"  {snap.snapshot_id[:8]}  {_fmt_ts(snap.timestamp)}{label}{session}\n")
        if i > 0:
            diff = diff_snapshots(snapshots[i - 1], snap)
            for c in diff.changes:
                if c.change != "unchanged":
                    sym = {"added": "+", "removed": "-", "modified": "~"}.get(c.change, "?")
                    out.write(f"             {sym}  {c.path}\n")
    out.write("\n")


def format_affected(
    affected: list[tuple[str, str, list[str]]],
    out: TextIO = sys.stdout,
) -> None:
    if not affected:
        out.write("No sessions found that ran after a config change.\n")
        return

    out.write(f"{len(affected)} session(s) ran after a config change:\n\n")
    for session_id, started_at, changed_paths in affected:
        out.write(f"  {session_id[:12]}  {started_at}  "
                  f"(after change to: {', '.join(changed_paths)})\n")
    out.write(
        "\nRun `agent-strace drift` to compare behaviour before and after the change.\n"
    )


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def current_workspace() -> Path | None:
    """Return the current directory, the one the CLI watches.

    If it cannot be read, for example because it was deleted, print an
    error and return None.
    """
    try:
        return Path.cwd()
    except OSError as exc:
        sys.stderr.write(
            f"error: cannot read the current directory: {inert(type(exc).__name__)}\n"
        )
        return None


def cmd_config_watch(args: argparse.Namespace) -> int:
    workspace_root = current_workspace()
    if workspace_root is None:
        return 1
    trace_dir = Path(args.trace_dir)
    store = TraceStore(trace_dir)
    subcommand = getattr(args, "config_watch_command", None)
    extra_paths = getattr(args, "watch", None) or []
    watch_paths = _load_watch_paths(workspace_root, extra_paths)
    fmt = getattr(args, "format", "text")

    if subcommand == "snapshot" or subcommand is None:
        label = getattr(args, "label", "") or ""
        try:
            snap = take_snapshot(workspace_root, watch_paths, label=label, trace_dir=trace_dir)
        except SnapshotFileError as exc:
            sys.stderr.write(f"error: {exc}\n")
            return 1
        existing = _load_snapshots(workspace_root, trace_dir)
        sys.stdout.write(
            f"Snapshot {snap.snapshot_id[:8]} recorded "
            f"({sum(1 for f in snap.files if f.exists)} file(s) hashed).\n"
        )
        # Show diff vs previous if one exists
        if len(existing) >= 2:
            diff = diff_snapshots(existing[-2], existing[-1])
            if diff.has_changes:
                sys.stdout.write("\nChanges from previous snapshot:\n")
                format_check(diff, sys.stdout)
        return 0

    elif subcommand == "check":
        snapshots = _load_snapshots(workspace_root, trace_dir)
        if not snapshots:
            sys.stderr.write(
                f"No snapshots yet for {inert(str(workspace_root))}. "
                "Run: agent-strace config-watch snapshot\n"
            )
            return 1
        # Compare current state against latest snapshot.
        # Use the union of: paths in the latest snapshot + any CLI --watch extras.
        latest = snapshots[-1]
        snapshot_paths = [f.path for f in latest.files]
        check_paths = list(dict.fromkeys(snapshot_paths + (extra_paths or [])))
        current_files = [_snapshot_file(workspace_root, p) for p in check_paths]
        import uuid
        current_snap = ConfigSnapshot(
            snapshot_id=uuid.uuid4().hex[:12],
            timestamp=time.time(),
            files=current_files,
        )
        diff = diff_snapshots(latest, current_snap)
        if fmt == "json":
            sys.stdout.write(json.dumps({
                "has_changes": diff.has_changes,
                "changed_paths": diff.changed_paths,
                "since_snapshot": latest.snapshot_id,
                "since_timestamp": latest.timestamp,
                "changes": [
                    {"path": c.path, "change": c.change}
                    for c in diff.changes if c.change != "unchanged"
                ],
            }, indent=2) + "\n")
        else:
            format_check(diff, sys.stdout)
        return 1 if diff.has_changes else 0

    elif subcommand == "history":
        snapshots = _load_snapshots(workspace_root, trace_dir)
        if fmt == "json":
            sys.stdout.write(json.dumps([s.to_dict() for s in snapshots], indent=2) + "\n")
        else:
            format_history(snapshots, sys.stdout)
        return 0

    elif subcommand == "affected":
        since_str = getattr(args, "since", None)
        since_ts: float | None = None
        if since_str:
            from .lint import _parse_since
            since_ts = _parse_since(since_str)
        affected = find_affected_sessions(store, workspace_root, since=since_ts)
        if fmt == "json":
            sys.stdout.write(json.dumps([
                {"session_id": sid, "started_at": ts, "changed_paths": paths}
                for sid, ts, paths in affected
            ], indent=2) + "\n")
        else:
            format_affected(affected, sys.stdout)
        return 0

    else:
        sys.stderr.write(f"Unknown config-watch subcommand: {subcommand!r}\n")
        return 1
