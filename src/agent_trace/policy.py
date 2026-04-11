"""Policy suggestion: auto-generate .agent-scope.json from observed traces.

Analyses one or more sessions and produces a minimal allow-list policy that
covers exactly the files read/written and commands run. The output is a valid
.agent-scope.json that can be used directly with `agent-strace audit`.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from .models import EventType, TraceEvent
from .store import TraceStore


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PolicySuggestion:
    session_ids: list[str]
    files_read: list[str]
    files_written: list[str]
    commands: list[str]
    network_hosts: list[str]
    # Collapsed glob patterns (e.g. src/**/*.py instead of individual files)
    file_read_patterns: list[str]
    file_write_patterns: list[str]
    cmd_patterns: list[str]


# ---------------------------------------------------------------------------
# Pattern collapsing helpers
# ---------------------------------------------------------------------------

_COMMON_DIRS = [
    "src", "tests", "lib", "app", "pkg", "internal",
    "components", "pages", "utils", "helpers",
]


def _collapse_paths(paths: list[str]) -> list[str]:
    """Collapse a list of file paths into minimal glob patterns.

    Groups files by directory prefix and emits ``dir/**`` when 3+ files share
    the same top-level directory. Individual files are kept as-is otherwise.
    """
    if not paths:
        return []

    from collections import Counter
    dir_counts: Counter = Counter()
    for p in paths:
        parts = Path(p).parts
        if len(parts) > 1:
            dir_counts[parts[0]] += 1

    collapsed: list[str] = []
    covered: set[str] = set()

    for top_dir, count in dir_counts.items():
        if count >= 3:
            collapsed.append(f"{top_dir}/**")
            covered.update(p for p in paths if Path(p).parts[0] == top_dir)

    for p in paths:
        if p not in covered:
            collapsed.append(p)

    return sorted(set(collapsed))


def _collapse_commands(cmds: list[str]) -> list[str]:
    """Collapse commands to their base executable (first token).

    e.g. ``pytest tests/foo.py -x`` → ``pytest *``
    Deduplicates by base command and emits ``<cmd> *`` patterns.
    """
    if not cmds:
        return []

    bases: dict[str, list[str]] = {}
    for cmd in cmds:
        base = cmd.strip().split()[0] if cmd.strip() else cmd
        bases.setdefault(base, []).append(cmd)

    patterns: list[str] = []
    for base, variants in bases.items():
        if len(variants) == 1 and variants[0] == base:
            patterns.append(base)
        else:
            patterns.append(f"{base} *")

    return sorted(patterns)


# ---------------------------------------------------------------------------
# Observation pass
# ---------------------------------------------------------------------------

def _extract_url_host(cmd: str) -> list[str]:
    """Extract hostnames from URLs in a shell command."""
    import re
    hosts = []
    for m in re.finditer(r"https?://([^/\s\"']+)", cmd):
        host = m.group(1).split(":")[0]
        hosts.append(host)
    return hosts


def observe_session(store: TraceStore, session_id: str) -> dict:
    """Return raw observations from a single session."""
    events = store.load_events(session_id)
    files_read: list[str] = []
    files_written: list[str] = []
    commands: list[str] = []
    network_hosts: list[str] = []

    for event in events:
        if event.event_type != EventType.TOOL_CALL:
            continue
        name = event.data.get("tool_name", "").lower()
        args = event.data.get("arguments", {}) or {}

        if name in ("read", "view", "grep", "glob"):
            path = str(
                args.get("file_path") or args.get("path") or args.get("pattern") or ""
            ).strip()
            if path and not path.startswith("/proc") and not path.startswith("/sys"):
                files_read.append(path)

        elif name in ("write", "edit", "create"):
            path = str(args.get("file_path") or args.get("path") or "").strip()
            if path:
                files_written.append(path)

        elif name == "bash":
            cmd = str(args.get("command", "")).strip()
            if cmd:
                commands.append(cmd)
                network_hosts.extend(_extract_url_host(cmd))

    return {
        "files_read": files_read,
        "files_written": files_written,
        "commands": commands,
        "network_hosts": network_hosts,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def suggest_policy(
    store: TraceStore,
    session_ids: list[str],
) -> PolicySuggestion:
    """Analyse *session_ids* and return a PolicySuggestion."""
    all_reads: list[str] = []
    all_writes: list[str] = []
    all_cmds: list[str] = []
    all_hosts: list[str] = []

    for sid in session_ids:
        obs = observe_session(store, sid)
        all_reads.extend(obs["files_read"])
        all_writes.extend(obs["files_written"])
        all_cmds.extend(obs["commands"])
        all_hosts.extend(obs["network_hosts"])

    # Deduplicate preserving order
    def _dedup(lst: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for x in lst:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    reads = _dedup(all_reads)
    writes = _dedup(all_writes)
    cmds = _dedup(all_cmds)
    hosts = _dedup(all_hosts)

    return PolicySuggestion(
        session_ids=session_ids,
        files_read=reads,
        files_written=writes,
        commands=cmds,
        network_hosts=hosts,
        file_read_patterns=_collapse_paths(reads),
        file_write_patterns=_collapse_paths(writes),
        cmd_patterns=_collapse_commands(cmds),
    )


def render_policy_json(suggestion: PolicySuggestion) -> dict:
    """Convert a PolicySuggestion to the .agent-scope.json dict format."""
    policy: dict = {}

    files: dict = {}
    if suggestion.file_read_patterns:
        files["read"] = {"allow": suggestion.file_read_patterns}
    if suggestion.file_write_patterns:
        files["write"] = {"allow": suggestion.file_write_patterns}
    if files:
        policy["files"] = files

    if suggestion.cmd_patterns:
        policy["commands"] = {"allow": suggestion.cmd_patterns}

    if suggestion.network_hosts:
        policy["network"] = {
            "deny_all": True,
            "allow": sorted(set(suggestion.network_hosts)),
        }

    return policy


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_suggestion(
    suggestion: PolicySuggestion,
    out: TextIO = sys.stdout,
    dry_run: bool = False,
) -> None:
    w = out.write
    n = len(suggestion.session_ids)
    w(f"\nPolicy suggestion from {n} session{'s' if n != 1 else ''}:\n\n")

    if suggestion.file_read_patterns:
        w("  Files read (allow):\n")
        for p in suggestion.file_read_patterns:
            w(f"    {p}\n")
        w("\n")

    if suggestion.file_write_patterns:
        w("  Files written (allow):\n")
        for p in suggestion.file_write_patterns:
            w(f"    {p}\n")
        w("\n")

    if suggestion.cmd_patterns:
        w("  Commands (allow):\n")
        for p in suggestion.cmd_patterns:
            w(f"    {p}\n")
        w("\n")

    if suggestion.network_hosts:
        w("  Network hosts (allow):\n")
        for h in sorted(set(suggestion.network_hosts)):
            w(f"    {h}\n")
        w("\n")

    if dry_run:
        w("Generated policy (dry run):\n\n")
        w(json.dumps(render_policy_json(suggestion), indent=2))
        w("\n\n")


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def cmd_policy(args: argparse.Namespace) -> int:
    store = TraceStore(args.trace_dir)

    session_ids_raw: list[str] = getattr(args, "session_ids", []) or []

    # If no sessions specified, use all sessions
    if not session_ids_raw:
        all_sessions = store.list_sessions()
        if not all_sessions:
            sys.stderr.write("No sessions found.\n")
            return 1
        session_ids_raw = [s.session_id for s in all_sessions]

    # Default: dry-run when no --output given
    if not output_path and not dry_run:
        dry_run = True

    # Resolve prefixes
    resolved: list[str] = []
    for sid in session_ids_raw:
        full = store.find_session(sid)
        if not full:
            sys.stderr.write(f"Session not found: {sid}\n")
            return 1
        resolved.append(full)

    suggestion = suggest_policy(store, resolved)

    output_path = getattr(args, "output", None)
    dry_run = getattr(args, "dry_run", False)

    if dry_run or not output_path:
        format_suggestion(suggestion, dry_run=True)
        return 0

    policy_dict = render_policy_json(suggestion)
    out_path = Path(output_path)
    out_path.write_text(json.dumps(policy_dict, indent=2) + "\n")
    sys.stdout.write(f"Policy written to {out_path}\n")
    return 0
