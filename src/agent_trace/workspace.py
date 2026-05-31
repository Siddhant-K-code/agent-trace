"""Workspace isolation for agent-trace.

Workspaces scope sessions into separate subdirectories so multiple teams,
environments, or projects can share a single AGENT_TRACE_DIR without
seeing each other's sessions.

Layout:
  .agent-traces/
    workspaces/
      <workspace-id>/
        <session-id>/
          meta.json
          events.ndjson

Usage:
    # List workspaces
    agent-strace workspace list

    # Activate a workspace in the current shell
    eval $(agent-strace workspace use staging)

    # Create a new workspace
    agent-strace workspace new staging

    # Delete a workspace
    agent-strace workspace rm staging --force

Env var: AGENT_STRACE_WORKSPACE
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from .store import DEFAULT_TRACE_DIR, _WORKSPACE_ENV, _workspace_base


def _workspaces_root(trace_dir: str) -> Path:
    return Path(trace_dir) / "workspaces"


def list_workspaces(trace_dir: str = DEFAULT_TRACE_DIR) -> list[str]:
    """Return sorted list of workspace IDs."""
    root = _workspaces_root(trace_dir)
    if not root.exists():
        return []
    return sorted(d.name for d in root.iterdir() if d.is_dir())


def create_workspace(workspace_id: str,
                     trace_dir: str = DEFAULT_TRACE_DIR) -> Path:
    """Create a workspace directory. Idempotent."""
    path = _workspace_base(trace_dir, workspace_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def delete_workspace(workspace_id: str,
                     trace_dir: str = DEFAULT_TRACE_DIR) -> bool:
    """Delete a workspace and all its sessions. Returns True if it existed."""
    path = _workspace_base(trace_dir, workspace_id)
    if not path.exists():
        return False
    shutil.rmtree(path)
    return True


def workspace_session_count(workspace_id: str,
                             trace_dir: str = DEFAULT_TRACE_DIR) -> int:
    """Return the number of sessions in a workspace."""
    path = _workspace_base(trace_dir, workspace_id)
    if not path.exists():
        return 0
    return sum(1 for d in path.iterdir() if d.is_dir() and (d / "meta.json").exists())


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def cmd_workspace(args: argparse.Namespace) -> int:
    trace_dir = args.trace_dir
    sub = getattr(args, "workspace_cmd", None)

    if sub == "list":
        workspaces = list_workspaces(trace_dir)
        if not workspaces:
            sys.stdout.write("No workspaces found.\n")
            sys.stdout.write(
                f"Create one with: agent-strace workspace new <name>\n"
                f"Or set {_WORKSPACE_ENV}=<name> to use one automatically.\n"
            )
            return 0
        sys.stdout.write(f"\n{'Workspace':<32}  Sessions\n")
        sys.stdout.write(f"{'─'*32}  {'─'*8}\n")
        for wid in workspaces:
            count = workspace_session_count(wid, trace_dir)
            sys.stdout.write(f"{wid:<32}  {count}\n")
        sys.stdout.write("\n")
        return 0

    elif sub == "use":
        wid = args.workspace_id
        # Print a shell export the user can eval
        sys.stdout.write(f"export {_WORKSPACE_ENV}={wid}\n")
        sys.stderr.write(
            f"# Run: eval $(agent-strace workspace use {wid})\n"
            f"# Or add to your shell profile.\n"
        )
        return 0

    elif sub == "new":
        wid = args.workspace_id
        path = create_workspace(wid, trace_dir)
        sys.stdout.write(f"Workspace '{wid}' created at {path}\n")
        sys.stdout.write(f"Activate: eval $(agent-strace workspace use {wid})\n")
        return 0

    elif sub == "rm":
        wid = args.workspace_id
        count = workspace_session_count(wid, trace_dir)
        if count and not getattr(args, "force", False):
            sys.stderr.write(
                f"Workspace '{wid}' contains {count} session(s). "
                f"Use --force to delete.\n"
            )
            return 1
        existed = delete_workspace(wid, trace_dir)
        if existed:
            sys.stdout.write(f"Workspace '{wid}' deleted.\n")
        else:
            sys.stderr.write(f"Workspace '{wid}' not found.\n")
            return 1
        return 0

    else:
        sys.stderr.write("Usage: agent-strace workspace <list|use|new|rm>\n")
        return 1
