"""Tests for rbac.py — role assignments, permission checks, and CLI."""

from __future__ import annotations

import json
import sys
from io import StringIO

import pytest

from agent_trace.rbac import (
    RBACStore,
    RoleAssignment,
    can,
    cmd_rbac,
    ALL_ROLES,
    ORG_ROLES,
    WORKSPACE_ROLES,
)


# ---------------------------------------------------------------------------
# RoleAssignment dataclass
# ---------------------------------------------------------------------------

class TestRoleAssignment:
    def test_to_dict_roundtrip(self):
        a = RoleAssignment(principal="alice@example.com", principal_type="user", role="admin")
        d = a.to_dict()
        b = RoleAssignment.from_dict(d)
        assert b.principal == a.principal
        assert b.role == a.role
        assert b.workspace_id == ""

    def test_workspace_scoped(self):
        a = RoleAssignment(principal="bob@example.com", principal_type="user",
                           role="workspace:member", workspace_id="prod")
        assert a.workspace_id == "prod"


# ---------------------------------------------------------------------------
# RBACStore — assign / revoke / list / get_role
# ---------------------------------------------------------------------------

class TestRBACStore:
    def test_empty_store(self, tmp_path):
        store = RBACStore(tmp_path)
        assert store.list_assignments() == []

    def test_assign_org_level(self, tmp_path):
        store = RBACStore(tmp_path)
        a = store.assign("alice@example.com", "user", "admin")
        assert a.role == "admin"
        assert a.workspace_id == ""

    def test_assign_persisted(self, tmp_path):
        store = RBACStore(tmp_path)
        store.assign("alice@example.com", "user", "member")
        store2 = RBACStore(tmp_path)
        assignments = store2.list_assignments()
        assert len(assignments) == 1
        assert assignments[0].principal == "alice@example.com"

    def test_assign_overwrites_existing(self, tmp_path):
        store = RBACStore(tmp_path)
        store.assign("alice@example.com", "user", "member")
        store.assign("alice@example.com", "user", "admin")
        assignments = store.list_assignments()
        assert len(assignments) == 1
        assert assignments[0].role == "admin"

    def test_assign_workspace_role(self, tmp_path):
        store = RBACStore(tmp_path)
        a = store.assign("bob@example.com", "user", "workspace:member", workspace_id="prod")
        assert a.role == "workspace:member"
        assert a.workspace_id == "prod"

    def test_assign_invalid_role_raises(self, tmp_path):
        store = RBACStore(tmp_path)
        with pytest.raises(ValueError, match="Unknown role"):
            store.assign("alice@example.com", "user", "superuser")

    def test_assign_org_role_with_workspace_raises(self, tmp_path):
        store = RBACStore(tmp_path)
        with pytest.raises(ValueError, match="workspace role"):
            store.assign("alice@example.com", "user", "admin", workspace_id="prod")

    def test_assign_workspace_role_without_workspace_raises(self, tmp_path):
        store = RBACStore(tmp_path)
        with pytest.raises(ValueError, match="requires --workspace"):
            store.assign("alice@example.com", "user", "workspace:admin")

    def test_revoke_existing(self, tmp_path):
        store = RBACStore(tmp_path)
        store.assign("alice@example.com", "user", "member")
        removed = store.revoke("alice@example.com")
        assert removed is True
        assert store.list_assignments() == []

    def test_revoke_nonexistent_returns_false(self, tmp_path):
        store = RBACStore(tmp_path)
        assert store.revoke("nobody@example.com") is False

    def test_revoke_workspace_scoped(self, tmp_path):
        store = RBACStore(tmp_path)
        store.assign("bob@example.com", "user", "workspace:viewer", workspace_id="dev")
        removed = store.revoke("bob@example.com", workspace_id="dev")
        assert removed is True

    def test_revoke_workspace_does_not_remove_org(self, tmp_path):
        store = RBACStore(tmp_path)
        store.assign("alice@example.com", "user", "member")
        store.assign("alice@example.com", "user", "workspace:admin", workspace_id="prod")
        store.revoke("alice@example.com", workspace_id="prod")
        assignments = store.list_assignments()
        assert len(assignments) == 1
        assert assignments[0].workspace_id == ""

    def test_get_role_org(self, tmp_path):
        store = RBACStore(tmp_path)
        store.assign("alice@example.com", "user", "admin")
        assert store.get_role("alice@example.com") == "admin"

    def test_get_role_missing(self, tmp_path):
        store = RBACStore(tmp_path)
        assert store.get_role("nobody@example.com") is None

    def test_list_filter_by_workspace(self, tmp_path):
        store = RBACStore(tmp_path)
        store.assign("alice@example.com", "user", "member")
        store.assign("bob@example.com", "user", "workspace:member", workspace_id="prod")
        ws_only = store.list_assignments(workspace_id="prod")
        assert len(ws_only) == 1
        assert ws_only[0].principal == "bob@example.com"

    def test_list_filter_by_principal(self, tmp_path):
        store = RBACStore(tmp_path)
        store.assign("alice@example.com", "user", "member")
        store.assign("bob@example.com", "user", "viewer")
        alice_only = store.list_assignments(principal="alice@example.com")
        assert len(alice_only) == 1

    def test_group_assignment(self, tmp_path):
        store = RBACStore(tmp_path)
        a = store.assign("eng@example.com", "group", "member")
        assert a.principal_type == "group"

    def test_effective_role_falls_back_to_org(self, tmp_path):
        store = RBACStore(tmp_path)
        store.assign("alice@example.com", "user", "admin")
        # No workspace-specific role → falls back to org
        assert store.effective_role("alice@example.com", "prod") == "admin"

    def test_effective_role_workspace_overrides_org(self, tmp_path):
        store = RBACStore(tmp_path)
        store.assign("alice@example.com", "user", "admin")
        store.assign("alice@example.com", "user", "workspace:viewer", workspace_id="prod")
        assert store.effective_role("alice@example.com", "prod") == "workspace:viewer"


# ---------------------------------------------------------------------------
# Permission checks
# ---------------------------------------------------------------------------

class TestCan:
    def test_owner_can_do_everything(self, tmp_path):
        store = RBACStore(tmp_path)
        store.assign("owner@example.com", "user", "owner")
        for action in ("read_sessions", "manage_policies", "manage_rbac", "manage_billing"):
            assert can(store, "owner@example.com", action) is True

    def test_viewer_can_only_read(self, tmp_path):
        store = RBACStore(tmp_path)
        store.assign("viewer@example.com", "user", "viewer")
        assert can(store, "viewer@example.com", "read_sessions") is True
        assert can(store, "viewer@example.com", "manage_policies") is False
        assert can(store, "viewer@example.com", "run_agent") is False

    def test_member_can_run_but_not_manage(self, tmp_path):
        store = RBACStore(tmp_path)
        store.assign("member@example.com", "user", "member")
        assert can(store, "member@example.com", "run_agent") is True
        assert can(store, "member@example.com", "manage_policies") is False

    def test_admin_can_manage_policies(self, tmp_path):
        store = RBACStore(tmp_path)
        store.assign("admin@example.com", "user", "admin")
        assert can(store, "admin@example.com", "manage_policies") is True
        assert can(store, "admin@example.com", "manage_rbac") is False

    def test_no_assignment_denied(self, tmp_path):
        store = RBACStore(tmp_path)
        assert can(store, "nobody@example.com", "read_sessions") is False

    def test_workspace_admin_can_manage_policies(self, tmp_path):
        store = RBACStore(tmp_path)
        store.assign("alice@example.com", "user", "workspace:admin", workspace_id="prod")
        assert can(store, "alice@example.com", "manage_policies", workspace_id="prod") is True

    def test_workspace_viewer_cannot_run(self, tmp_path):
        store = RBACStore(tmp_path)
        store.assign("alice@example.com", "user", "workspace:viewer", workspace_id="prod")
        assert can(store, "alice@example.com", "run_agent", workspace_id="prod") is False

    def test_workspace_override_restricts_org_admin(self, tmp_path):
        store = RBACStore(tmp_path)
        store.assign("alice@example.com", "user", "admin")
        store.assign("alice@example.com", "user", "workspace:viewer", workspace_id="restricted")
        # In the restricted workspace, only viewer-level access
        assert can(store, "alice@example.com", "manage_policies", workspace_id="restricted") is False
        assert can(store, "alice@example.com", "read_sessions", workspace_id="restricted") is True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _run_cmd(argv: list[str], tmp_path) -> tuple[int, str]:
    """Run cmd_rbac with the given argv, return (exit_code, stdout)."""
    import argparse
    from agent_trace.rbac import cmd_rbac, ALL_ROLES, WORKSPACE_ROLES

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="rbac_cmd")

    p_assign = sub.add_parser("assign")
    p_assign.add_argument("--user", default="")
    p_assign.add_argument("--group", default="")
    p_assign.add_argument("--role", required=True)
    p_assign.add_argument("--workspace", default="")
    p_assign.add_argument("--by", default="")
    p_assign.add_argument("--dir", default=str(tmp_path))

    p_revoke = sub.add_parser("revoke")
    p_revoke.add_argument("--user", default="")
    p_revoke.add_argument("--group", default="")
    p_revoke.add_argument("--workspace", default="")
    p_revoke.add_argument("--dir", default=str(tmp_path))

    p_list = sub.add_parser("list")
    p_list.add_argument("--workspace", default="")
    p_list.add_argument("--dir", default=str(tmp_path))

    p_check = sub.add_parser("check")
    p_check.add_argument("--user", required=True)
    p_check.add_argument("--action", required=True)
    p_check.add_argument("--workspace", default="")
    p_check.add_argument("--dir", default=str(tmp_path))

    args = parser.parse_args(argv)
    buf = StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        rc = cmd_rbac(args)
    finally:
        sys.stdout = old_stdout
    return rc, buf.getvalue()


class TestCLI:
    def test_assign_and_list(self, tmp_path):
        rc, out = _run_cmd(["assign", "--user", "alice@example.com", "--role", "admin",
                            "--dir", str(tmp_path)], tmp_path)
        assert rc == 0
        assert "admin" in out

        rc2, out2 = _run_cmd(["list", "--dir", str(tmp_path)], tmp_path)
        assert rc2 == 0
        assert "alice@example.com" in out2

    def test_assign_workspace_role(self, tmp_path):
        rc, out = _run_cmd(["assign", "--user", "bob@example.com",
                            "--role", "workspace:member", "--workspace", "prod",
                            "--dir", str(tmp_path)], tmp_path)
        assert rc == 0
        assert "workspace:member" in out

    def test_revoke(self, tmp_path):
        _run_cmd(["assign", "--user", "alice@example.com", "--role", "member",
                  "--dir", str(tmp_path)], tmp_path)
        rc, out = _run_cmd(["revoke", "--user", "alice@example.com",
                            "--dir", str(tmp_path)], tmp_path)
        assert rc == 0
        assert "Revoked" in out

    def test_check_allowed(self, tmp_path):
        _run_cmd(["assign", "--user", "alice@example.com", "--role", "admin",
                  "--dir", str(tmp_path)], tmp_path)
        rc, out = _run_cmd(["check", "--user", "alice@example.com",
                            "--action", "manage_policies",
                            "--dir", str(tmp_path)], tmp_path)
        assert rc == 0
        assert "allowed" in out

    def test_check_denied(self, tmp_path):
        _run_cmd(["assign", "--user", "viewer@example.com", "--role", "viewer",
                  "--dir", str(tmp_path)], tmp_path)
        rc, out = _run_cmd(["check", "--user", "viewer@example.com",
                            "--action", "manage_policies",
                            "--dir", str(tmp_path)], tmp_path)
        assert rc == 1
        assert "denied" in out

    def test_list_empty(self, tmp_path):
        rc, out = _run_cmd(["list", "--dir", str(tmp_path)], tmp_path)
        assert rc == 0
        assert "No role assignments" in out
