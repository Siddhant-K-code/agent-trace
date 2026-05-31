"""Human-in-the-loop approval workflow.

When a watch rule fires with action=require_approval, the agent is paused
(SIGSTOP) and an approval request is written to the trace directory.
A human reviews the request and approves or denies it via the CLI or
VS Code extension. The watcher polls for the decision and resumes or
kills the agent accordingly.

Approval requests are stored as JSON files:
  .agent-traces/.approvals/<request-id>.json

States: pending → approved | denied

Usage:
    # List pending approval requests
    agent-strace approval list

    # Approve a request (resumes the agent)
    agent-strace approval approve <request-id>

    # Deny a request (kills the agent)
    agent-strace approval deny <request-id> [--reason TEXT]

    # Show details of a request
    agent-strace approval show <request-id>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from .store import TraceStore

ApprovalState = Literal["pending", "approved", "denied"]

_APPROVALS_DIR = ".approvals"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ApprovalRequest:
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    session_id: str = ""
    rule_name: str = ""
    tool_name: str = ""
    tool_input: dict = field(default_factory=dict)
    event_id: str = ""
    agent_pid: int = 0
    state: ApprovalState = "pending"
    created_at: float = field(default_factory=time.time)
    decided_at: float | None = None
    decided_by: str = ""
    reason: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "ApprovalRequest":
        d = json.loads(text)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def _approvals_dir(store: TraceStore) -> Path:
    return store.base_dir / _APPROVALS_DIR


def _request_path(store: TraceStore, request_id: str) -> Path:
    return _approvals_dir(store) / f"{request_id}.json"


def save_request(store: TraceStore, req: ApprovalRequest) -> Path:
    d = _approvals_dir(store)
    d.mkdir(parents=True, exist_ok=True)
    path = _request_path(store, req.request_id)
    path.write_text(req.to_json())
    return path


def load_request(store: TraceStore, request_id: str) -> ApprovalRequest | None:
    path = _request_path(store, request_id)
    if not path.exists():
        return None
    try:
        return ApprovalRequest.from_json(path.read_text())
    except Exception:
        return None


def list_requests(store: TraceStore,
                  state: ApprovalState | None = None) -> list[ApprovalRequest]:
    """Return all approval requests, optionally filtered by state."""
    d = _approvals_dir(store)
    if not d.exists():
        return []
    requests = []
    for f in sorted(d.glob("*.json")):
        try:
            req = ApprovalRequest.from_json(f.read_text())
            if state is None or req.state == state:
                requests.append(req)
        except Exception:
            continue
    return requests


def find_request(store: TraceStore, prefix: str) -> ApprovalRequest | None:
    """Find a request by ID prefix."""
    d = _approvals_dir(store)
    if not d.exists():
        return None
    for f in d.glob(f"{prefix}*.json"):
        try:
            return ApprovalRequest.from_json(f.read_text())
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Approval workflow
# ---------------------------------------------------------------------------

def create_approval_request(
    store: TraceStore,
    session_id: str,
    rule_name: str,
    tool_name: str = "",
    tool_input: dict | None = None,
    event_id: str = "",
    agent_pid: int = 0,
) -> ApprovalRequest:
    """Create and persist a new pending approval request."""
    req = ApprovalRequest(
        session_id=session_id,
        rule_name=rule_name,
        tool_name=tool_name,
        tool_input=tool_input or {},
        event_id=event_id,
        agent_pid=agent_pid,
    )
    save_request(store, req)
    return req


def approve_request(
    store: TraceStore,
    request_id: str,
    decided_by: str = "",
    resume_agent: bool = True,
) -> tuple[bool, str]:
    """Approve a pending request. Optionally resumes the agent via SIGCONT.

    Returns (ok, message).
    """
    req = find_request(store, request_id)
    if not req:
        return False, f"Request not found: {request_id}"
    if req.state != "pending":
        return False, f"Request already {req.state}"

    req.state = "approved"
    req.decided_at = time.time()
    req.decided_by = decided_by or os.environ.get("USER", "")
    save_request(store, req)

    if resume_agent and req.agent_pid:
        try:
            import signal
            os.kill(req.agent_pid, signal.SIGCONT)
        except (ProcessLookupError, PermissionError):
            pass

    return True, f"Approved — agent {req.agent_pid} resumed"


def deny_request(
    store: TraceStore,
    request_id: str,
    reason: str = "",
    decided_by: str = "",
    kill_agent: bool = True,
) -> tuple[bool, str]:
    """Deny a pending request. Optionally kills the agent.

    Returns (ok, message).
    """
    req = find_request(store, request_id)
    if not req:
        return False, f"Request not found: {request_id}"
    if req.state != "pending":
        return False, f"Request already {req.state}"

    req.state = "denied"
    req.decided_at = time.time()
    req.decided_by = decided_by or os.environ.get("USER", "")
    req.reason = reason
    save_request(store, req)

    if kill_agent and req.agent_pid:
        try:
            import signal
            os.kill(req.agent_pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

    return True, f"Denied — agent {req.agent_pid} terminated"


def poll_for_decision(
    store: TraceStore,
    request_id: str,
    timeout: float = 300.0,
    poll_interval: float = 1.0,
) -> ApprovalState:
    """Block until a decision is made or timeout expires.

    Returns the final state: 'approved', 'denied', or 'pending' (timeout).
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        req = load_request(store, request_id)
        if req and req.state != "pending":
            return req.state
        time.sleep(poll_interval)
    return "pending"


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def cmd_approval(args: argparse.Namespace) -> int:
    store = TraceStore(args.trace_dir)
    sub = getattr(args, "approval_cmd", None)

    if sub == "list":
        state_filter = getattr(args, "state", None) or None
        requests = list_requests(store, state=state_filter)
        if not requests:
            label = f" ({state_filter})" if state_filter else ""
            sys.stdout.write(f"No approval requests{label}.\n")
            return 0
        sys.stdout.write(f"\n{'ID':<14} {'State':<10} {'Rule':<24} {'Tool':<20} {'Age'}\n")
        sys.stdout.write(f"{'─'*14} {'─'*10} {'─'*24} {'─'*20} {'─'*8}\n")
        for req in requests:
            age = int(time.time() - req.created_at)
            age_str = f"{age//60}m{age%60}s" if age < 3600 else f"{age//3600}h"
            sys.stdout.write(
                f"{req.request_id:<14} {req.state:<10} {req.rule_name[:24]:<24} "
                f"{req.tool_name[:20]:<20} {age_str}\n"
            )
        sys.stdout.write("\n")
        return 0

    elif sub == "show":
        req = find_request(store, args.request_id)
        if not req:
            sys.stderr.write(f"Request not found: {args.request_id}\n")
            return 1
        sys.stdout.write(json.dumps(asdict(req), indent=2) + "\n")
        return 0

    elif sub == "approve":
        ok, msg = approve_request(
            store, args.request_id,
            decided_by=getattr(args, "by", "") or "",
            resume_agent=not getattr(args, "no_resume", False),
        )
        if ok:
            sys.stdout.write(f"✅ {msg}\n")
            return 0
        sys.stderr.write(f"❌ {msg}\n")
        return 1

    elif sub == "deny":
        ok, msg = deny_request(
            store, args.request_id,
            reason=getattr(args, "reason", "") or "",
            decided_by=getattr(args, "by", "") or "",
            kill_agent=not getattr(args, "no_kill", False),
        )
        if ok:
            sys.stdout.write(f"✅ {msg}\n")
            return 0
        sys.stderr.write(f"❌ {msg}\n")
        return 1

    else:
        sys.stderr.write("Usage: agent-strace approval list|show|approve|deny\n")
        return 1
