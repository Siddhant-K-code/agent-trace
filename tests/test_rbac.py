"""Tests for rbac.py — role assignments, permission checks, and CLI."""

from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from io import StringIO

sys.path.insert(0, "src")

from agent_trace.rbac import (
    RBACStore,
    RoleAssignment,
    can,
    cmd_rbac,
)

try:
    from _concurrency import run_concurrently
except ImportError:
    from tests._concurrency import run_concurrently


def _run_cmd(argv: list[str], tmp_dir: str) -> tuple[int, str]:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="rbac_cmd")

    p_assign = sub.add_parser("assign")
    p_assign.add_argument("--user", default="")
    p_assign.add_argument("--group", default="")
    p_assign.add_argument("--role", required=True)
    p_assign.add_argument("--workspace", default="")
    p_assign.add_argument("--by", default="")
    p_assign.add_argument("--dir", default=tmp_dir)

    p_revoke = sub.add_parser("revoke")
    p_revoke.add_argument("--user", default="")
    p_revoke.add_argument("--group", default="")
    p_revoke.add_argument("--workspace", default="")
    p_revoke.add_argument("--dir", default=tmp_dir)

    p_list = sub.add_parser("list")
    p_list.add_argument("--workspace", default="")
    p_list.add_argument("--dir", default=tmp_dir)

    p_check = sub.add_parser("check")
    p_check.add_argument("--user", required=True)
    p_check.add_argument("--action", required=True)
    p_check.add_argument("--workspace", default="")
    p_check.add_argument("--dir", default=tmp_dir)

    args = parser.parse_args(argv)
    buf = StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        rc = cmd_rbac(args)
    finally:
        sys.stdout = old
    return rc, buf.getvalue()


class TestRoleAssignment(unittest.TestCase):
    def test_to_dict_roundtrip(self):
        a = RoleAssignment(principal="alice@example.com", principal_type="user", role="admin")
        b = RoleAssignment.from_dict(a.to_dict())
        self.assertEqual(b.principal, a.principal)
        self.assertEqual(b.role, a.role)
        self.assertEqual(b.workspace_id, "")

    def test_workspace_scoped(self):
        a = RoleAssignment(principal="bob@example.com", principal_type="user",
                           role="workspace:member", workspace_id="prod")
        self.assertEqual(a.workspace_id, "prod")


class TestRBACStore(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def test_empty_store(self):
        self.assertEqual(RBACStore(self._tmp).list_assignments(), [])

    def test_assign_org_level(self):
        a = RBACStore(self._tmp).assign("alice@example.com", "user", "admin")
        self.assertEqual(a.role, "admin")
        self.assertEqual(a.workspace_id, "")

    def test_assign_persisted(self):
        RBACStore(self._tmp).assign("alice@example.com", "user", "member")
        assignments = RBACStore(self._tmp).list_assignments()
        self.assertEqual(len(assignments), 1)
        self.assertEqual(assignments[0].principal, "alice@example.com")

    def test_assign_overwrites_existing(self):
        store = RBACStore(self._tmp)
        store.assign("alice@example.com", "user", "member")
        store.assign("alice@example.com", "user", "admin")
        assignments = store.list_assignments()
        self.assertEqual(len(assignments), 1)
        self.assertEqual(assignments[0].role, "admin")

    def test_assign_workspace_role(self):
        a = RBACStore(self._tmp).assign("bob@example.com", "user",
                                        "workspace:member", workspace_id="prod")
        self.assertEqual(a.role, "workspace:member")
        self.assertEqual(a.workspace_id, "prod")

    def test_assign_invalid_role_raises(self):
        with self.assertRaises(ValueError):
            RBACStore(self._tmp).assign("alice@example.com", "user", "superuser")

    def test_assign_org_role_with_workspace_raises(self):
        with self.assertRaises(ValueError):
            RBACStore(self._tmp).assign("alice@example.com", "user", "admin",
                                        workspace_id="prod")

    def test_assign_workspace_role_without_workspace_raises(self):
        with self.assertRaises(ValueError):
            RBACStore(self._tmp).assign("alice@example.com", "user", "workspace:admin")

    def test_revoke_existing(self):
        store = RBACStore(self._tmp)
        store.assign("alice@example.com", "user", "member")
        self.assertTrue(store.revoke("alice@example.com"))
        self.assertEqual(store.list_assignments(), [])

    def test_revoke_nonexistent_returns_false(self):
        self.assertFalse(RBACStore(self._tmp).revoke("nobody@example.com"))

    def test_revoke_workspace_scoped(self):
        store = RBACStore(self._tmp)
        store.assign("bob@example.com", "user", "workspace:viewer", workspace_id="dev")
        self.assertTrue(store.revoke("bob@example.com", workspace_id="dev"))

    def test_revoke_workspace_does_not_remove_org(self):
        store = RBACStore(self._tmp)
        store.assign("alice@example.com", "user", "member")
        store.assign("alice@example.com", "user", "workspace:admin", workspace_id="prod")
        store.revoke("alice@example.com", workspace_id="prod")
        assignments = store.list_assignments()
        self.assertEqual(len(assignments), 1)
        self.assertEqual(assignments[0].workspace_id, "")

    def test_get_role_org(self):
        store = RBACStore(self._tmp)
        store.assign("alice@example.com", "user", "admin")
        self.assertEqual(store.get_role("alice@example.com"), "admin")

    def test_get_role_missing(self):
        self.assertIsNone(RBACStore(self._tmp).get_role("nobody@example.com"))

    def test_list_filter_by_workspace(self):
        store = RBACStore(self._tmp)
        store.assign("alice@example.com", "user", "member")
        store.assign("bob@example.com", "user", "workspace:member", workspace_id="prod")
        ws_only = store.list_assignments(workspace_id="prod")
        self.assertEqual(len(ws_only), 1)
        self.assertEqual(ws_only[0].principal, "bob@example.com")

    def test_list_filter_by_principal(self):
        store = RBACStore(self._tmp)
        store.assign("alice@example.com", "user", "member")
        store.assign("bob@example.com", "user", "viewer")
        self.assertEqual(len(store.list_assignments(principal="alice@example.com")), 1)

    def test_group_assignment(self):
        a = RBACStore(self._tmp).assign("eng@example.com", "group", "member")
        self.assertEqual(a.principal_type, "group")

    def test_effective_role_falls_back_to_org(self):
        store = RBACStore(self._tmp)
        store.assign("alice@example.com", "user", "admin")
        self.assertEqual(store.effective_role("alice@example.com", "prod"), "admin")

    def test_effective_role_workspace_overrides_org(self):
        store = RBACStore(self._tmp)
        store.assign("alice@example.com", "user", "admin")
        store.assign("alice@example.com", "user", "workspace:viewer", workspace_id="prod")
        self.assertEqual(store.effective_role("alice@example.com", "prod"), "workspace:viewer")


_ASSIGN_SCRIPT = """
from agent_trace.rbac import RBACStore
store = RBACStore(sys.argv[2])
wait_for_start()
for principal in sys.argv[3:]:
    store.assign(principal, "user", "viewer")
"""


class TestRBACStoreConcurrency(unittest.TestCase):
    def test_concurrent_assignments_are_not_lost(self):
        tmp = tempfile.mkdtemp()
        groups = [[f"user{i}-{n}@example.com" for n in range(10)] for i in range(5)]
        run_concurrently(_ASSIGN_SCRIPT, [[tmp, *group] for group in groups])
        principals = {a.principal for a in RBACStore(tmp).list_assignments()}
        self.assertEqual(principals, {p for group in groups for p in group})


class TestCan(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def test_owner_can_do_everything(self):
        store = RBACStore(self._tmp)
        store.assign("owner@example.com", "user", "owner")
        for action in ("read_sessions", "manage_policies", "manage_rbac", "manage_billing"):
            self.assertTrue(can(store, "owner@example.com", action))

    def test_viewer_can_only_read(self):
        store = RBACStore(self._tmp)
        store.assign("viewer@example.com", "user", "viewer")
        self.assertTrue(can(store, "viewer@example.com", "read_sessions"))
        self.assertFalse(can(store, "viewer@example.com", "manage_policies"))
        self.assertFalse(can(store, "viewer@example.com", "run_agent"))

    def test_member_can_run_but_not_manage(self):
        store = RBACStore(self._tmp)
        store.assign("member@example.com", "user", "member")
        self.assertTrue(can(store, "member@example.com", "run_agent"))
        self.assertFalse(can(store, "member@example.com", "manage_policies"))

    def test_admin_can_manage_policies(self):
        store = RBACStore(self._tmp)
        store.assign("admin@example.com", "user", "admin")
        self.assertTrue(can(store, "admin@example.com", "manage_policies"))
        self.assertFalse(can(store, "admin@example.com", "manage_rbac"))

    def test_no_assignment_denied(self):
        self.assertFalse(can(RBACStore(self._tmp), "nobody@example.com", "read_sessions"))

    def test_workspace_admin_can_manage_policies(self):
        store = RBACStore(self._tmp)
        store.assign("alice@example.com", "user", "workspace:admin", workspace_id="prod")
        self.assertTrue(can(store, "alice@example.com", "manage_policies", workspace_id="prod"))

    def test_workspace_viewer_cannot_run(self):
        store = RBACStore(self._tmp)
        store.assign("alice@example.com", "user", "workspace:viewer", workspace_id="prod")
        self.assertFalse(can(store, "alice@example.com", "run_agent", workspace_id="prod"))

    def test_workspace_override_restricts_org_admin(self):
        store = RBACStore(self._tmp)
        store.assign("alice@example.com", "user", "admin")
        store.assign("alice@example.com", "user", "workspace:viewer", workspace_id="r")
        self.assertFalse(can(store, "alice@example.com", "manage_policies", workspace_id="r"))
        self.assertTrue(can(store, "alice@example.com", "read_sessions", workspace_id="r"))


class TestCLI(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def test_assign_and_list(self):
        rc, out = _run_cmd(["assign", "--user", "alice@example.com", "--role", "admin",
                            "--dir", self._tmp], self._tmp)
        self.assertEqual(rc, 0)
        self.assertIn("admin", out)
        rc2, out2 = _run_cmd(["list", "--dir", self._tmp], self._tmp)
        self.assertEqual(rc2, 0)
        self.assertIn("alice@example.com", out2)

    def test_assign_workspace_role(self):
        rc, out = _run_cmd(["assign", "--user", "bob@example.com",
                            "--role", "workspace:member", "--workspace", "prod",
                            "--dir", self._tmp], self._tmp)
        self.assertEqual(rc, 0)
        self.assertIn("workspace:member", out)

    def test_revoke(self):
        _run_cmd(["assign", "--user", "alice@example.com", "--role", "member",
                  "--dir", self._tmp], self._tmp)
        rc, out = _run_cmd(["revoke", "--user", "alice@example.com",
                            "--dir", self._tmp], self._tmp)
        self.assertEqual(rc, 0)
        self.assertIn("Revoked", out)

    def test_check_allowed(self):
        _run_cmd(["assign", "--user", "alice@example.com", "--role", "admin",
                  "--dir", self._tmp], self._tmp)
        rc, out = _run_cmd(["check", "--user", "alice@example.com",
                            "--action", "manage_policies",
                            "--dir", self._tmp], self._tmp)
        self.assertEqual(rc, 0)
        self.assertIn("allowed", out)

    def test_check_denied(self):
        _run_cmd(["assign", "--user", "viewer@example.com", "--role", "viewer",
                  "--dir", self._tmp], self._tmp)
        rc, out = _run_cmd(["check", "--user", "viewer@example.com",
                            "--action", "manage_policies",
                            "--dir", self._tmp], self._tmp)
        self.assertEqual(rc, 1)
        self.assertIn("denied", out)

    def test_list_empty(self):
        rc, out = _run_cmd(["list", "--dir", self._tmp], self._tmp)
        self.assertEqual(rc, 0)
        self.assertIn("No role assignments", out)


if __name__ == "__main__":
    unittest.main()
