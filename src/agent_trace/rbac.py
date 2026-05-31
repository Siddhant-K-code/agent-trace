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

from .store import DEFAULT_TRACE_DIR

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

class RBACStore:
    """Persist role assignments to a JSON file in the trace store directory."""

    def __init__(self, base_dir: str | Path = DEFAULT_TRACE_DIR) -> None:
        self._path = Path(base_dir) / _RBAC_FILE
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> list[RoleAssignment]:
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text())
            return [RoleAssignment.from_dict(d) for d in data]
        except Exception:
            return []

    def _save(self, assignments: list[RoleAssignment]) -> None:
        self._path.write_text(json.dumps([a.to_dict() for a in assignments], indent=2))

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

        assignments = self._load()
        # Remove any existing assignment for this principal+workspace
        assignments = [
            a for a in assignments
            if not (a.principal == principal and a.workspace_id == workspace_id)
        ]
        new = RoleAssignment(
            principal=principal,
            principal_type=principal_type,
            role=role,
            workspace_id=workspace_id,
            assigned_by=assigned_by,
        )
        assignments.append(new)
        self._save(assignments)
        return new

    def revoke(self, principal: str, workspace_id: str = "") -> bool:
        """Remove a role assignment. Returns True if something was removed."""
        assignments = self._load()
        before = len(assignments)
        assignments = [
            a for a in assignments
            if not (a.principal == principal and a.workspace_id == workspace_id)
        ]
        if len(assignments) < before:
            self._save(assignments)
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
    base_dir = getattr(args, "dir", None) or DEFAULT_TRACE_DIR
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
        removed = store.revoke(principal, workspace_id)
        if removed:
            scope = f" (workspace: {workspace_id})" if workspace_id else " (org-level)"
            sys.stdout.write(f"Revoked role for {principal}{scope}\n")
        else:
            sys.stdout.write(f"No assignment found for {principal}\n")
        return 0

    if sub == "list":
        workspace_id = getattr(args, "workspace", "") or ""
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
