"""Human-in-the-loop approval workflow.

When a watch rule fires with action=require_approval, the agent is paused
(SIGSTOP) and an approval request is written to the trace directory.
A human reviews the request and approves or denies it via the CLI or
VS Code extension. The watcher polls for the decision and resumes or
kills the agent accordingly.

Approval requests are stored as JSON files:
  .agent-traces/.approvals/<request-id>.json

States: pending → approved | denied

The allowed states and transitions are defined once in ``APPROVAL_STATES``
and ``_TRANSITIONS``. Decisions are applied under the store write lock, so
exactly one of several concurrent approve/deny calls wins.

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

from .store import TraceStore, _write_atomic

ApprovalState = Literal["pending", "approved", "denied"]

PENDING: ApprovalState = "pending"
APPROVED: ApprovalState = "approved"
DENIED: ApprovalState = "denied"

APPROVAL_STATES: frozenset[str] = frozenset({PENDING, APPROVED, DENIED})
DECIDED_STATES: frozenset[str] = frozenset({APPROVED, DENIED})

# Allowed state transitions. Decided states are terminal.
_TRANSITIONS: dict[str, frozenset[str]] = {
    PENDING: frozenset({APPROVED, DENIED}),
    APPROVED: frozenset(),
    DENIED: frozenset(),
}

PollOutcome = Literal["approved", "denied", "timeout", "not_found", "invalid"]

_APPROVALS_DIR = ".approvals"


def can_transition(current: str, target: str) -> bool:
    """Return True when ``current`` -> ``target`` is an allowed transition."""
    return target in _TRANSITIONS.get(current, frozenset())


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
    state: ApprovalState = PENDING
    created_at: float = field(default_factory=time.time)
    decided_at: float | None = None
    decided_by: str = ""
    reason: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "ApprovalRequest":
        """Parse a stored request. Raises ``ValueError`` on an unknown state."""
        d = json.loads(text)
        if not isinstance(d, dict):
            raise ValueError("approval record must be a JSON object")
        state = d.get("state", PENDING)
        if not isinstance(state, str) or state not in APPROVAL_STATES:
            raise ValueError(f"invalid approval state: {state!r}")
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class PollResult:
    """Result of :func:`wait_for_decision`.

    ``outcome`` is one of:
      - ``approved`` / ``denied``: a decision was recorded
      - ``timeout``: the request was still pending when the timeout expired
      - ``not_found``: no request file exists for the ID
      - ``invalid``: the request file exists but cannot be parsed or has an
        unknown state
    """

    outcome: PollOutcome
    request: ApprovalRequest | None = None

    @property
    def decided(self) -> bool:
        return self.outcome in DECIDED_STATES

    @property
    def timed_out(self) -> bool:
        return self.outcome == "timeout"


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def _approvals_dir(store: TraceStore) -> Path:
    return store.base_dir / _APPROVALS_DIR


def _request_path(store: TraceStore, request_id: str) -> Path:
    return _approvals_dir(store) / f"{request_id}.json"


def save_request(store: TraceStore, req: ApprovalRequest) -> Path:
    """Persist a request atomically so concurrent readers never see a partial file."""
    if req.state not in APPROVAL_STATES:
        raise ValueError(f"invalid approval state: {req.state!r}")
    path = _request_path(store, req.request_id)
    _write_atomic(path, req.to_json())
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


def _find_request_path(
    store: TraceStore, request_id: str,
) -> tuple[Path | None, ApprovalRequest | None, str]:
    """Resolve an ID or prefix to (path, request, error) for an update.

    An exact ID match is preferred over prefix matches. ``error`` is set when
    matching files exist but none of them is a valid record.
    """
    d = _approvals_dir(store)
    if not request_id or not d.is_dir():
        return None, None, ""
    candidates = sorted(
        p for p in d.glob(f"{request_id}*.json")
        if p.parent == d and not p.is_symlink()
    )
    exact = _request_path(store, request_id)
    if exact in candidates:
        candidates.remove(exact)
        candidates.insert(0, exact)
    error = ""
    for path in candidates:
        try:
            return path, ApprovalRequest.from_json(path.read_text()), ""
        except (OSError, ValueError) as exc:
            error = error or str(exc)
    return None, None, error


def _decide(
    store: TraceStore,
    request_id: str,
    target: ApprovalState,
    decided_by: str,
    reason: str | None = None,
) -> tuple[ApprovalRequest | None, str]:
    """Apply ``pending -> target`` atomically under the store write lock.

    Returns (request, "") on success or (None, error message) otherwise. The
    load, state check, and write happen under one lock, so when several
    deciders race exactly one succeeds and the rest see the recorded state.
    """
    if not _approvals_dir(store).is_dir():
        return None, f"Request not found: {request_id}"
    with store.write_lock():
        path, req, error = _find_request_path(store, request_id)
        if req is None or path is None:
            if error:
                return None, f"Request {request_id} has an invalid record: {error}"
            return None, f"Request not found: {request_id}"
        if not can_transition(req.state, target):
            return None, f"Request already {req.state}"
        req.state = target
        req.decided_at = time.time()
        req.decided_by = decided_by or os.environ.get("USER", "")
        if reason is not None:
            req.reason = reason
        _write_atomic(path, req.to_json())
    return req, ""


def approve_request(
    store: TraceStore,
    request_id: str,
    decided_by: str = "",
    resume_agent: bool = True,
) -> tuple[bool, str]:
    """Approve a pending request. Optionally resumes the agent via SIGCONT.

    Returns (ok, message). If the request was already decided, including by
    a concurrent call, returns (False, "Request already <state>").
    """
    req, error = _decide(store, request_id, APPROVED, decided_by)
    if req is None:
        return False, error

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

    Returns (ok, message). If the request was already decided, including by
    a concurrent call, returns (False, "Request already <state>").
    """
    req, error = _decide(store, request_id, DENIED, decided_by, reason=reason)
    if req is None:
        return False, error

    if kill_agent and req.agent_pid:
        try:
            import signal
            os.kill(req.agent_pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

    return True, f"Denied — agent {req.agent_pid} terminated"


def _check_decision(store: TraceStore, request_id: str) -> PollResult | None:
    """Return a final PollResult, or None while the request is still pending."""
    path = _request_path(store, request_id)
    try:
        text = path.read_text()
    except FileNotFoundError:
        return PollResult("not_found")
    except OSError:
        return PollResult("invalid")
    try:
        req = ApprovalRequest.from_json(text)
    except ValueError:
        return PollResult("invalid")
    if req.state == PENDING:
        return None
    return PollResult(req.state, req)


def wait_for_decision(
    store: TraceStore,
    request_id: str,
    timeout: float = 300.0,
    poll_interval: float = 1.0,
) -> PollResult:
    """Block until the request is decided, missing, invalid, or the timeout expires.

    Unlike :func:`poll_for_decision`, every outcome is distinct: a decision
    (``approved``/``denied``), ``timeout`` (still pending at the deadline),
    ``not_found`` (no request file), or ``invalid`` (unreadable record or
    unknown state). ``not_found`` and ``invalid`` are returned immediately
    instead of waiting for the timeout. The request is always checked at
    least once, and once more at the deadline.
    """
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        result = _check_decision(store, request_id)
        if result is not None:
            return result
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return PollResult("timeout", load_request(store, request_id))
        time.sleep(max(0.0, min(poll_interval, remaining)))


def poll_for_decision(
    store: TraceStore,
    request_id: str,
    timeout: float = 300.0,
    poll_interval: float = 1.0,
) -> ApprovalState:
    """Block until a decision is made or timeout expires.

    Returns 'approved' or 'denied' when decided. Returns 'pending' only when
    the timeout expired with the request still undecided.

    Raises ``LookupError`` if the request file does not exist and
    ``ValueError`` if it cannot be parsed or has an unknown state, rather
    than reporting 'pending'. Use :func:`wait_for_decision` to get these
    outcomes as values.
    """
    result = wait_for_decision(store, request_id, timeout, poll_interval)
    if result.outcome == "not_found":
        raise LookupError(f"approval request not found: {request_id}")
    if result.outcome == "invalid":
        raise ValueError(f"approval request has an invalid record: {request_id}")
    if result.outcome == "timeout":
        return PENDING
    return result.outcome  # type: ignore[return-value]


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
