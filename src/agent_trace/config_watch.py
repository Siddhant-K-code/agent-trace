"""AGENTS.md change detector — auto-flag drift after config changes.

Snapshots a set of config files (AGENTS.md, system prompts, tool configs)
at session boundaries and detects when those files change between sessions.
Sessions after a config change are flagged as "potentially affected" so
teams know to re-evaluate agent behaviour.

Usage:
    # Snapshot current config state
    agent-strace config-watch snapshot

    # Check whether config has changed since last snapshot
    agent-strace config-watch check

    # Show full change history
    agent-strace config-watch history

    # Show which sessions ran after each config change
    agent-strace config-watch affected [--since 7d]

Config files tracked by default:
    AGENTS.md, .agent-strace-policy.json, .agent-strace-lint.json,
    .agent-watch.json, system_prompt.txt, system_prompt.md

Additional paths can be added via --watch or .agent-strace-watch.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from .store import TraceStore


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

_SNAPSHOT_FILE = ".agent-traces/.config-snapshots.json"
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

    def to_dict(self) -> dict:
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "label": self.label,
            "files": [
                {"path": f.path, "sha256": f.sha256,
                 "mtime": f.mtime, "exists": f.exists}
                for f in self.files
            ],
        }

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

def _snapshot_path(workspace_root: Path) -> Path:
    return workspace_root / _SNAPSHOT_FILE


def _load_snapshots(workspace_root: Path) -> list[ConfigSnapshot]:
    p = _snapshot_path(workspace_root)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
        return [ConfigSnapshot.from_dict(d) for d in data]
    except Exception:
        return []


def _save_snapshots(workspace_root: Path, snapshots: list[ConfigSnapshot]) -> None:
    p = _snapshot_path(workspace_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([s.to_dict() for s in snapshots], indent=2))


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
) -> ConfigSnapshot:
    """Hash all watched files and append a new snapshot."""
    import uuid
    snap_id = uuid.uuid4().hex[:12]
    files = [_snapshot_file(workspace_root, p) for p in watch_paths]
    snapshot = ConfigSnapshot(
        snapshot_id=snap_id,
        timestamp=time.time(),
        session_id=session_id,
        label=label,
        files=files,
    )
    existing = _load_snapshots(workspace_root)
    existing.append(snapshot)
    _save_snapshots(workspace_root, existing)
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
    ran after a config change relative to the previous snapshot."""
    snapshots = _load_snapshots(workspace_root)
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
    if since:
        all_sessions = [s for s in all_sessions if s.started_at >= since]

    results: list[tuple[str, str, list[str]]] = []
    for meta in all_sessions:
        # Find the most recent config change before this session
        relevant: list[str] = []
        for change_ts, changed_paths in change_events:
            if change_ts <= meta.started_at:
                relevant = changed_paths  # keep the most recent
        if relevant:
            import datetime
            dt = datetime.datetime.fromtimestamp(meta.started_at)
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

def cmd_config_watch(args: argparse.Namespace) -> int:
    workspace_root = Path(args.trace_dir).parent  # .agent-traces/ → workspace root
    store = TraceStore(Path(args.trace_dir))
    subcommand = getattr(args, "config_watch_command", None)
    extra_paths = getattr(args, "watch", None) or []
    watch_paths = _load_watch_paths(workspace_root, extra_paths)
    fmt = getattr(args, "format", "text")

    if subcommand == "snapshot" or subcommand is None:
        label = getattr(args, "label", "") or ""
        snap = take_snapshot(workspace_root, watch_paths, label=label)
        existing = _load_snapshots(workspace_root)
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
        snapshots = _load_snapshots(workspace_root)
        if not snapshots:
            sys.stderr.write("No snapshots yet. Run: agent-strace config-watch snapshot\n")
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
        snapshots = _load_snapshots(workspace_root)
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
