"""Role-based access control for agent-trace.

Stores role assignments in a JSON file alongside the trace store.
Supports org-level and workspace-scoped roles.

Org-level roles (in order of privilege):
    owner   — full access including billing and SSO config
    admin   — manage policies, identities, workspaces
    member  — read sessions, run agents
    viewer  — read-only
    machine — for agent identities (programmatic access)

Workspace-scoped roles (override org-level for a specific workspace):
    workspace:admin   — full access within workspace
    workspace:member  — read/run within workspace
    workspace:viewer  — read-only within workspace

CLI:
    agent-strace rbac assign --user alice@example.com --role admin
    agent-strace rbac assign --group eng@example.com --role member
    agent-strace rbac assign --user bob@example.com --role workspace:member --workspace prod
    agent-strace rbac revoke --user bob@example.com --workspace prod
    agent-strace rbac list
    agent-strace rbac check --user alice@example.com --action read_sessions
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .fileutil import file_lock, write_atomic
from .inert import inert
from .store import DEFAULT_TRACE_DIR, resolve_trace_dir

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ORG_ROLES = ("owner", "admin", "member", "viewer", "machine")
WORKSPACE_ROLES = ("workspace:admin", "workspace:member", "workspace:viewer")
ALL_ROLES = ORG_ROLES + WORKSPACE_ROLES

# Privilege order for org-level roles (higher index = more privilege)
_ORG_RANK = {r: i for i, r in enumerate(("viewer", "machine", "member", "admin", "owner"))}

# Actions and the minimum org-level role required (when no workspace override)
_ACTION_MIN_ROLE: dict[str, str] = {
    "read_sessions": "viewer",
    "run_agent": "member",
    "annotate": "member",
    "manage_policies": "admin",
    "manage_identities": "admin",
    "manage_workspaces": "admin",
    "export_compliance": "admin",
    "manage_rbac": "owner",
    "manage_billing": "owner",
    "manage_sso": "owner",
}

_RBAC_FILE = "rbac.json"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RoleAssignment:
    """A single role assignment for a user or group."""
    principal: str          # email or group identifier
    principal_type: str     # "user" or "group"
    role: str               # one of ALL_ROLES
    workspace_id: str = ""  # empty = org-level
    assigned_by: str = ""
    assigned_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RoleAssignment":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

class RBACFileError(ValueError):
    """rbac.json exists but cannot be read or parsed as a list of entries."""


_STR_FIELDS = ("principal", "principal_type", "role", "workspace_id", "assigned_by")


def _entry_problem(d: object) -> str:
    """Return why *d* is not a usable assignment entry, or '' if it is."""
    if not isinstance(d, dict):
        return "not a JSON object"
    for name in ("principal", "principal_type", "role"):
        if name not in d:
            return f"missing {name}"
    for name in _STR_FIELDS:
        if name in d and not isinstance(d[name], str):
            return f"{name} is not a string"
    assigned_at = d.get("assigned_at", 0.0)
    if isinstance(assigned_at, bool) or not isinstance(assigned_at, (int, float)):
        return "assigned_at is not a number"
    return ""


def _same_key(d: object, principal: str, workspace_id: str) -> bool:
    """True if raw entry *d* names *principal* in *workspace_id*."""
    return (
        isinstance(d, dict)
        and d.get("principal") == principal
        and d.get("workspace_id", "") == workspace_id
    )


class RBACStore:
    """Persist role assignments to a JSON file in the trace store directory.

    An entry that cannot be parsed is skipped with a warning on stderr and is
    written back unchanged by ``assign`` and ``revoke``, unless the call
    replaces or revokes the same principal and workspace. If the file itself
    cannot be read or parsed, reads see no assignments (so every permission
    check denies) and ``assign`` / ``revoke`` raise :class:`RBACFileError`
    instead of overwriting it.
    """

    def __init__(self, base_dir: str | Path = DEFAULT_TRACE_DIR) -> None:
        self._path = Path(base_dir) / _RBAC_FILE
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._warned: set[str] = set()

    def _warn(self, message: str) -> None:
        if message not in self._warned:
            self._warned.add(message)
            sys.stderr.write(f"warning: {message}\n")

    def _read(self) -> tuple[list[RoleAssignment], list[object]]:
        """Return (valid assignments, raw entries that failed to parse).

        Raises RBACFileError if the file exists but is not a readable JSON list.
        """
        where = inert(self._path)
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return [], []
        except (OSError, UnicodeDecodeError, ValueError, RecursionError) as exc:
            raise RBACFileError(
                f"cannot parse {where}: {inert(type(exc).__name__)}"
            ) from exc
        if not isinstance(data, list):
            raise RBACFileError(f"cannot parse {where}: top level is not a JSON list")
        assignments: list[RoleAssignment] = []
        bad: list[object] = []
        for index, d in enumerate(data):
            problem = _entry_problem(d)
            if problem:
                bad.append(d)
                self._warn(f"skipping entry {index} in {where}: {problem}")
            else:
                assignments.append(RoleAssignment.from_dict(d))
        return assignments, bad

    def _load(self) -> list[RoleAssignment]:
        """Return valid assignments; an unreadable file yields none (fail closed)."""
        try:
            return self._read()[0]
        except RBACFileError as exc:
            self._warn(f"{exc}; treating as no role assignments")
            return []

    def read_error(self) -> str | None:
        """Return why rbac.json cannot be parsed, or None if it can (or is absent)."""
        try:
            self._read()
        except RBACFileError as exc:
            return str(exc)
        return None

    def _read_for_update(self) -> tuple[list[RoleAssignment], list[object]]:
        try:
            return self._read()
        except RBACFileError as exc:
            raise RBACFileError(
                f"{exc}; refusing to write so existing assignments are not lost. "
                "Fix or remove the file and retry."
            ) from exc

    def _save(self, assignments: list[RoleAssignment], bad: list[object] = ()) -> None:
        write_atomic(
            self._path,
            json.dumps([a.to_dict() for a in assignments] + list(bad), indent=2),
            allow_symlinks=True,
        )

    def list_assignments(
        self,
        workspace_id: str = "",
        principal: str = "",
    ) -> list[RoleAssignment]:
        """Return all assignments, optionally filtered."""
        assignments = self._load()
        if workspace_id:
            assignments = [a for a in assignments if a.workspace_id == workspace_id]
        if principal:
            assignments = [a for a in assignments if a.principal == principal]
        return assignments

    def assign(
        self,
        principal: str,
        principal_type: str,
        role: str,
        workspace_id: str = "",
        assigned_by: str = "",
    ) -> RoleAssignment:
        """Create or update a role assignment. Returns the assignment."""
        if role not in ALL_ROLES:
            raise ValueError(f"Unknown role: {role!r}. Valid roles: {ALL_ROLES}")
        if workspace_id and role not in WORKSPACE_ROLES:
            raise ValueError(
                f"Role {role!r} is an org-level role. "
                f"Use a workspace role (workspace:admin/member/viewer) with --workspace."
            )
        if not workspace_id and role in WORKSPACE_ROLES:
            raise ValueError(
                f"Role {role!r} requires --workspace to be specified."
            )

        with file_lock(self._path):
            assignments, bad = self._read_for_update()
            # Remove any existing assignment for this principal+workspace
            assignments = [
                a for a in assignments
                if not (a.principal == principal and a.workspace_id == workspace_id)
            ]
            bad = [d for d in bad if not _same_key(d, principal, workspace_id)]
            new = RoleAssignment(
                principal=principal,
                principal_type=principal_type,
                role=role,
                workspace_id=workspace_id,
                assigned_by=assigned_by,
            )
            assignments.append(new)
            self._save(assignments, bad)
        return new

    def revoke(self, principal: str, workspace_id: str = "") -> bool:
        """Remove a role assignment. Returns True if something was removed."""
        with file_lock(self._path):
            assignments, bad = self._read_for_update()
            before = len(assignments) + len(bad)
            assignments = [
                a for a in assignments
                if not (a.principal == principal and a.workspace_id == workspace_id)
            ]
            bad = [d for d in bad if not _same_key(d, principal, workspace_id)]
            if len(assignments) + len(bad) < before:
                self._save(assignments, bad)
                return True
            return False

    def get_role(self, principal: str, workspace_id: str = "") -> str | None:
        """Return the role for *principal* in *workspace_id* (or org-level)."""
        for a in self._load():
            if a.principal == principal and a.workspace_id == workspace_id:
                return a.role
        return None

    def effective_role(self, principal: str, workspace_id: str = "") -> str | None:
        """Return the effective role for *principal*, considering workspace override.

        If a workspace-scoped assignment exists, return it.
        Otherwise fall back to the org-level assignment.
        """
        if workspace_id:
            ws_role = self.get_role(principal, workspace_id)
            if ws_role:
                return ws_role
        return self.get_role(principal, "")


# ---------------------------------------------------------------------------
# Permission check
# ---------------------------------------------------------------------------

def can(
    store: RBACStore,
    principal: str,
    action: str,
    workspace_id: str = "",
) -> bool:
    """Return True if *principal* is allowed to perform *action*.

    Workspace-scoped roles map to org-level equivalents for the check:
        workspace:admin   → admin
        workspace:member  → member
        workspace:viewer  → viewer
    """
    role = store.effective_role(principal, workspace_id)
    if role is None:
        return False

    # Normalise workspace roles to their org equivalent
    ws_map = {
        "workspace:admin": "admin",
        "workspace:member": "member",
        "workspace:viewer": "viewer",
    }
    effective = ws_map.get(role, role)

    min_role = _ACTION_MIN_ROLE.get(action, "owner")
    return _ORG_RANK.get(effective, -1) >= _ORG_RANK.get(min_role, 999)


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def cmd_rbac(args: argparse.Namespace) -> int:
    base_dir = getattr(args, "dir", None) or resolve_trace_dir(getattr(args, "trace_dir", None))
    store = RBACStore(base_dir)
    sub = getattr(args, "rbac_cmd", None)

    if sub == "assign":
        principal = args.user or args.group
        if not principal:
            sys.stderr.write("error: --user or --group required\n")
            return 1
        principal_type = "group" if args.group else "user"
        workspace_id = getattr(args, "workspace", "") or ""
        try:
            a = store.assign(
                principal=principal,
                principal_type=principal_type,
                role=args.role,
                workspace_id=workspace_id,
                assigned_by=getattr(args, "by", "") or "",
            )
        except ValueError as exc:
            sys.stderr.write(f"error: {exc}\n")
            return 1
        scope = f" (workspace: {workspace_id})" if workspace_id else " (org-level)"
        sys.stdout.write(f"Assigned {a.role} to {principal}{scope}\n")
        return 0

    if sub == "revoke":
        principal = args.user or getattr(args, "group", None)
        if not principal:
            sys.stderr.write("error: --user or --group required\n")
            return 1
        workspace_id = getattr(args, "workspace", "") or ""
        try:
            removed = store.revoke(principal, workspace_id)
        except RBACFileError as exc:
            sys.stderr.write(f"error: {exc}\n")
            return 1
        if removed:
            scope = f" (workspace: {workspace_id})" if workspace_id else " (org-level)"
            sys.stdout.write(f"Revoked role for {principal}{scope}\n")
        else:
            sys.stdout.write(f"No assignment found for {principal}\n")
        return 0

    if sub == "list":
        workspace_id = getattr(args, "workspace", "") or ""
        error = store.read_error()
        if error:
            sys.stderr.write(f"error: {error}\n")
            return 1
        assignments = store.list_assignments(workspace_id=workspace_id)
        if not assignments:
            sys.stdout.write("No role assignments found.\n")
            return 0
        fmt = "{:<40} {:<8} {:<20} {:<20}\n"
        sys.stdout.write(fmt.format("Principal", "Type", "Role", "Workspace"))
        sys.stdout.write("-" * 92 + "\n")
        for a in sorted(assignments, key=lambda x: (x.workspace_id, x.principal)):
            sys.stdout.write(fmt.format(
                a.principal[:39], a.principal_type[:7],
                a.role[:19], a.workspace_id or "(org-level)",
            ))
        return 0

    if sub == "check":
        principal = args.user
        action = args.action
        workspace_id = getattr(args, "workspace", "") or ""
        allowed = can(store, principal, action, workspace_id)
        result = "allowed" if allowed else "denied"
        sys.stdout.write(f"{principal} → {action}: {result}\n")
        return 0 if allowed else 1

    sys.stderr.write("Usage: agent-strace rbac <assign|revoke|list|check>\n")
    return 1
