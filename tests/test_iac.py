"""Tests for iac.py — config parsing, apply, and config-diff."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path

sys.path.insert(0, "src")

from agent_trace.iac import (
    _parse_yaml,
    _coerce,
    load_config,
    config_diff,
    _apply_workspaces,
    _apply_rbac,
    _apply_identities,
    _apply_team_budgets,
    cmd_apply,
    cmd_config_diff,
)
from agent_trace.workspace import list_workspaces, create_workspace
from agent_trace.rbac import RBACStore

try:
    from _concurrency import run_concurrently
except ImportError:
    from tests._concurrency import run_concurrently


def _run(cmd_fn, argv_dict: dict) -> tuple[int, str]:
    ns = argparse.Namespace(**argv_dict)
    buf = StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        rc = cmd_fn(ns)
    finally:
        sys.stdout = old
    return rc, buf.getvalue()


class TestParseYaml(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(_parse_yaml(""), {})

    def test_simple_key_value(self):
        self.assertEqual(_parse_yaml("foo: bar\n")["foo"], "bar")

    def test_int_coercion(self):
        self.assertEqual(_parse_yaml("count: 42\n")["count"], 42)

    def test_float_coercion(self):
        self.assertAlmostEqual(_parse_yaml("amount: 3.14\n")["amount"], 3.14)

    def test_bool_coercion(self):
        self.assertIs(_parse_yaml("flag: true\n")["flag"], True)

    def test_list_of_dicts(self):
        yaml = "workspaces:\n  - name: production\n  - name: staging\n"
        result = _parse_yaml(yaml)
        self.assertEqual(result["workspaces"],
                         [{"name": "production"}, {"name": "staging"}])

    def test_nested_list_item(self):
        yaml = ("team_budgets:\n"
                "  - team: backend\n"
                "    monthly: 500.0\n"
                "    alert_threshold: 0.8\n")
        result = _parse_yaml(yaml)
        self.assertEqual(result["team_budgets"][0]["team"], "backend")
        self.assertAlmostEqual(result["team_budgets"][0]["monthly"], 500.0)
        self.assertAlmostEqual(result["team_budgets"][0]["alert_threshold"], 0.8)

    def test_inline_comment_stripped(self):
        self.assertEqual(_parse_yaml("foo: bar  # comment\n")["foo"], "bar")

    def test_full_config(self):
        yaml = ("workspaces:\n  - name: prod\n"
                "identities:\n  - name: billing-agent\n    team: backend\n"
                "rbac:\n  - user: alice@example.com\n    role: admin\n"
                "       - group: eng@example.com\n    role: member\n")
        result = _parse_yaml(yaml)
        self.assertEqual(len(result["workspaces"]), 1)
        self.assertEqual(len(result["identities"]), 1)
        self.assertGreaterEqual(len(result["rbac"]), 1)


class TestCoerce(unittest.TestCase):
    def test_true(self):
        self.assertIs(_coerce("true"), True)

    def test_false(self):
        self.assertIs(_coerce("false"), False)

    def test_int(self):
        self.assertEqual(_coerce("42"), 42)

    def test_float(self):
        self.assertAlmostEqual(_coerce("3.14"), 3.14)

    def test_string(self):
        self.assertEqual(_coerce("hello"), "hello")

    def test_quoted_string(self):
        self.assertEqual(_coerce('"hello world"'), "hello world")


class TestLoadConfig(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def test_missing_file_returns_empty(self):
        self.assertEqual(load_config(Path(self._tmp) / "nope.yaml"), {})

    def test_loads_yaml(self):
        f = Path(self._tmp) / ".agent-strace.yaml"
        f.write_text("workspaces:\n  - name: prod\n")
        self.assertEqual(load_config(f)["workspaces"], [{"name": "prod"}])


class TestApplyWorkspaces(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def test_creates_new_workspace(self):
        changes = _apply_workspaces({"workspaces": [{"name": "prod"}]},
                                    self._tmp, dry_run=False)
        self.assertTrue(any("+ workspace: prod" in c for c in changes))
        self.assertIn("prod", list_workspaces(self._tmp))

    def test_dry_run_does_not_create(self):
        _apply_workspaces({"workspaces": [{"name": "prod"}]}, self._tmp, dry_run=True)
        self.assertNotIn("prod", list_workspaces(self._tmp))

    def test_existing_workspace_no_change(self):
        create_workspace("prod", self._tmp)
        changes = _apply_workspaces({"workspaces": [{"name": "prod"}]},
                                    self._tmp, dry_run=False)
        self.assertTrue(any("exists" in c for c in changes))

    def test_empty_workspaces(self):
        self.assertEqual(_apply_workspaces({}, self._tmp, dry_run=False), [])


class TestApplyRbac(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def test_assigns_new_role(self):
        changes = _apply_rbac({"rbac": [{"user": "alice@example.com", "role": "admin"}]},
                               self._tmp, dry_run=False)
        self.assertTrue(any("+ rbac" in c for c in changes))
        self.assertEqual(RBACStore(self._tmp).get_role("alice@example.com"), "admin")

    def test_dry_run_does_not_assign(self):
        _apply_rbac({"rbac": [{"user": "alice@example.com", "role": "admin"}]},
                    self._tmp, dry_run=True)
        self.assertIsNone(RBACStore(self._tmp).get_role("alice@example.com"))

    def test_no_change_when_role_matches(self):
        RBACStore(self._tmp).assign("alice@example.com", "user", "admin")
        changes = _apply_rbac({"rbac": [{"user": "alice@example.com", "role": "admin"}]},
                               self._tmp, dry_run=False)
        self.assertTrue(any("no change" in c for c in changes))

    def test_workspace_role(self):
        changes = _apply_rbac(
            {"rbac": [{"user": "bob@example.com", "role": "workspace:member",
                       "workspace": "prod"}]},
            self._tmp, dry_run=False)
        self.assertTrue(any("workspace:member" in c for c in changes))


class TestApplyIdentities(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def test_creates_identity(self):
        changes = _apply_identities(
            {"identities": [{"name": "billing-agent", "team": "backend"}]},
            self._tmp, dry_run=False)
        self.assertTrue(any("+ identity: billing-agent" in c for c in changes))
        data = json.loads((Path(self._tmp) / "identities.json").read_text())
        self.assertIn("billing-agent", data)

    def test_dry_run_no_file(self):
        _apply_identities({"identities": [{"name": "billing-agent", "team": "backend"}]},
                          self._tmp, dry_run=True)
        self.assertFalse((Path(self._tmp) / "identities.json").exists())
        self.assertFalse((Path(self._tmp) / "identities.json.lock").exists())

    def test_concurrent_applies_keep_every_identity(self):
        script = """
from agent_trace.iac import _apply_identities
wait_for_start()
for name in sys.argv[3:]:
    _apply_identities({"identities": [{"name": name}]}, sys.argv[2], dry_run=False)
"""
        groups = [[f"agent-{i}-{n}" for n in range(10)] for i in range(5)]
        run_concurrently(script, [[self._tmp, *group] for group in groups])
        data = json.loads((Path(self._tmp) / "identities.json").read_text())
        self.assertEqual(set(data), {name for group in groups for name in group})


class TestApplyTeamBudgets(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def test_creates_budget(self):
        changes = _apply_team_budgets(
            {"team_budgets": [{"team": "backend", "monthly": 500.0}]},
            self._tmp, dry_run=False)
        self.assertTrue(any("+ team_budget" in c for c in changes))
        data = json.loads((Path(self._tmp) / "team_budgets.json").read_text())
        self.assertIn("backend/", data)

    def test_dry_run_no_file(self):
        _apply_team_budgets({"team_budgets": [{"team": "backend", "monthly": 500.0}]},
                            self._tmp, dry_run=True)
        self.assertFalse((Path(self._tmp) / "team_budgets.json").exists())


class TestConfigDiff(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def test_no_drift(self):
        create_workspace("prod", self._tmp)
        RBACStore(self._tmp).assign("alice@example.com", "user", "admin")
        cfg = {"workspaces": [{"name": "prod"}],
               "rbac": [{"user": "alice@example.com", "role": "admin"}]}
        self.assertEqual(config_diff(cfg, self._tmp), ["No drift detected."])

    def test_missing_workspace(self):
        lines = config_diff({"workspaces": [{"name": "prod"}]}, self._tmp)
        self.assertTrue(any("+ workspace: prod" in l for l in lines))

    def test_extra_workspace(self):
        create_workspace("orphan", self._tmp)
        lines = config_diff({}, self._tmp)
        self.assertTrue(any("- workspace: orphan" in l for l in lines))

    def test_missing_rbac(self):
        lines = config_diff({"rbac": [{"user": "alice@example.com", "role": "admin"}]},
                            self._tmp)
        self.assertTrue(any("+ rbac: alice@example.com" in l for l in lines))

    def test_changed_rbac(self):
        RBACStore(self._tmp).assign("alice@example.com", "user", "viewer")
        lines = config_diff({"rbac": [{"user": "alice@example.com", "role": "admin"}]},
                            self._tmp)
        self.assertTrue(any("~ rbac" in l and "viewer" in l and "admin" in l
                            for l in lines))


class TestCLIApply(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def test_apply_creates_workspace(self):
        cfg_file = Path(self._tmp) / ".agent-strace.yaml"
        cfg_file.write_text("workspaces:\n  - name: prod\n")
        rc, out = _run(cmd_apply, {"config": str(cfg_file), "dry_run": False,
                                   "server": "", "auth_key": "", "dir": self._tmp})
        self.assertEqual(rc, 0)
        self.assertIn("+ workspace: prod", out)

    def test_apply_dry_run(self):
        cfg_file = Path(self._tmp) / ".agent-strace.yaml"
        cfg_file.write_text("workspaces:\n  - name: prod\n")
        rc, out = _run(cmd_apply, {"config": str(cfg_file), "dry_run": True,
                                   "server": "", "auth_key": "", "dir": self._tmp})
        self.assertEqual(rc, 0)
        self.assertIn("dry-run", out)
        self.assertNotIn("prod", list_workspaces(self._tmp))

    def test_apply_missing_config(self):
        rc, _ = _run(cmd_apply, {"config": str(Path(self._tmp) / "missing.yaml"),
                                 "dry_run": False, "server": "", "auth_key": "",
                                 "dir": self._tmp})
        self.assertEqual(rc, 1)


class TestCLIConfigDiff(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def test_config_diff_no_drift(self):
        create_workspace("prod", self._tmp)
        cfg_file = Path(self._tmp) / ".agent-strace.yaml"
        cfg_file.write_text("workspaces:\n  - name: prod\n")
        rc, out = _run(cmd_config_diff, {"config": str(cfg_file), "dir": self._tmp})
        self.assertEqual(rc, 0)
        self.assertIn("No drift", out)

    def test_config_diff_shows_missing(self):
        cfg_file = Path(self._tmp) / ".agent-strace.yaml"
        cfg_file.write_text("workspaces:\n  - name: prod\n")
        rc, out = _run(cmd_config_diff, {"config": str(cfg_file), "dir": self._tmp})
        self.assertEqual(rc, 0)
        self.assertIn("prod", out)


if __name__ == "__main__":
    unittest.main()
