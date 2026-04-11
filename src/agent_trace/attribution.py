"""Session attribution: record spawning user, process ancestry, and agent provider.

Captures who/what started a session:
  - OS user (login name)
  - Process ancestry (parent process chain up to a recognisable agent)
  - Agent provider detected from environment variables or process name
  - Git context (repo, branch, commit) when available

Attribution is stored in SessionMeta.attribution (a plain dict) and written
to meta.json at session start. It is never updated after that.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Attribution:
    """Who/what started this session."""
    # OS-level identity
    os_user: str = ""
    hostname: str = ""
    # Process ancestry
    pid: int = 0
    ppid: int = 0
    process_name: str = ""          # name of the immediate parent process
    process_ancestry: list[str] = field(default_factory=list)  # [grandparent, parent, self]
    # Agent provider detected from env / process name
    agent_provider: str = ""        # "claude-code" | "cursor" | "copilot" | "unknown"
    agent_version: str = ""
    # Git context
    git_repo: str = ""
    git_branch: str = ""
    git_commit: str = ""
    # Working directory at session start
    working_dir: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v or v == 0}

    @classmethod
    def from_dict(cls, d: dict) -> "Attribution":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _detect_os_user() -> str:
    try:
        import pwd
        return pwd.getpwuid(os.getuid()).pw_name
    except Exception:
        return os.environ.get("USER", os.environ.get("USERNAME", ""))


def _detect_hostname() -> str:
    try:
        import socket
        return socket.gethostname()
    except Exception:
        return ""


def _detect_process_ancestry() -> list[str]:
    """Walk up the process tree and return names of ancestors."""
    ancestry: list[str] = []
    try:
        pid = os.getpid()
        # Read /proc on Linux; fall back gracefully on macOS/Windows
        for _ in range(8):  # max 8 levels up
            comm_path = Path(f"/proc/{pid}/comm")
            stat_path = Path(f"/proc/{pid}/stat")
            if comm_path.exists():
                name = comm_path.read_text().strip()
                ancestry.append(name)
                # Read parent PID from stat
                stat = stat_path.read_text()
                # Format: pid (name) state ppid ...
                ppid = int(stat.split(")")[1].split()[1])
                if ppid <= 1:
                    break
                pid = ppid
            else:
                break
    except Exception:
        pass
    return list(reversed(ancestry))  # oldest ancestor first


def _detect_agent_provider() -> tuple[str, str]:
    """Return (provider_name, version) by inspecting env vars and process names."""
    # Claude Code
    if os.environ.get("CLAUDE_CODE_SESSION") or os.environ.get("ANTHROPIC_API_KEY"):
        version = os.environ.get("CLAUDE_CODE_VERSION", "")
        return "claude-code", version

    # Cursor
    if os.environ.get("CURSOR_SESSION_ID") or os.environ.get("CURSOR_TRACE_ID"):
        return "cursor", ""

    # GitHub Copilot
    if os.environ.get("GITHUB_COPILOT_TOKEN") or os.environ.get("COPILOT_AGENT"):
        return "copilot", ""

    # Cline / Continue
    if os.environ.get("CLINE_SESSION") or os.environ.get("CONTINUE_SESSION"):
        return "cline", ""

    # Check process ancestry for known agent names
    ancestry = _detect_process_ancestry()
    for name in ancestry:
        name_lower = name.lower()
        if "claude" in name_lower:
            return "claude-code", ""
        if "cursor" in name_lower:
            return "cursor", ""
        if "copilot" in name_lower:
            return "copilot", ""

    return "unknown", ""


def _detect_git_context() -> tuple[str, str, str]:
    """Return (repo_url, branch, commit_sha) from the current working directory."""
    try:
        import subprocess
        cwd = os.getcwd()

        def _git(*args: str) -> str:
            result = subprocess.run(
                ["git"] + list(args),
                capture_output=True, text=True, cwd=cwd, timeout=2,
            )
            return result.stdout.strip() if result.returncode == 0 else ""

        repo = _git("remote", "get-url", "origin")
        branch = _git("rev-parse", "--abbrev-ref", "HEAD")
        commit = _git("rev-parse", "--short", "HEAD")
        return repo, branch, commit
    except Exception:
        return "", "", ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def collect_attribution() -> Attribution:
    """Collect attribution data for the current process."""
    os_user = _detect_os_user()
    hostname = _detect_hostname()
    ancestry = _detect_process_ancestry()
    provider, version = _detect_agent_provider()
    repo, branch, commit = _detect_git_context()

    return Attribution(
        os_user=os_user,
        hostname=hostname,
        pid=os.getpid(),
        ppid=os.getppid() if hasattr(os, "getppid") else 0,
        process_name=ancestry[-1] if ancestry else "",
        process_ancestry=ancestry,
        agent_provider=provider,
        agent_version=version,
        git_repo=repo,
        git_branch=branch,
        git_commit=commit,
        working_dir=os.getcwd(),
    )


def format_attribution(attr: Attribution) -> str:
    """Return a compact human-readable summary."""
    parts = []
    if attr.os_user:
        parts.append(f"user={attr.os_user}")
    if attr.hostname:
        parts.append(f"host={attr.hostname}")
    if attr.agent_provider and attr.agent_provider != "unknown":
        v = f"/{attr.agent_version}" if attr.agent_version else ""
        parts.append(f"agent={attr.agent_provider}{v}")
    if attr.git_branch:
        parts.append(f"branch={attr.git_branch}")
    if attr.git_commit:
        parts.append(f"commit={attr.git_commit}")
    return "  ".join(parts) if parts else "(no attribution)"
