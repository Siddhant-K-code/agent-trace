"""Tests for iac.py — config parsing, apply, and config-diff."""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path

import pytest

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
    DEFAULT_CONFIG,
)
from agent_trace.workspace import list_workspaces
from agent_trace.rbac import RBACStore


# ---------------------------------------------------------------------------
# YAML parser
# ---------------------------------------------------------------------------

class TestParseYaml:
    def test_empty(self):
        assert _parse_yaml("") == {}

    def test_simple_key_value(self):
        result = _parse_yaml("foo: bar\n")
        assert result["foo"] == "bar"

    def test_int_coercion(self):
        result = _parse_yaml("count: 42\n")
        assert result["count"] == 42

    def test_float_coercion(self):
        result = _parse_yaml("amount: 3.14\n")
        assert result["amount"] == pytest.approx(3.14)

    def test_bool_coercion(self):
        result = _parse_yaml("flag: true\n")
        assert result["flag"] is True

    def test_list_of_dicts(self):
        yaml = (
            "workspaces:\n"
            "  - name: production\n"
            "  - name: staging\n"
        )
        result = _parse_yaml(yaml)
        assert result["workspaces"] == [{"name": "production"}, {"name": "staging"}]

    def test_nested_list_item(self):
        yaml = (
            "team_budgets:\n"
            "  - team: backend\n"
            "    monthly: 500.0\n"
            "    alert_threshold: 0.8\n"
        )
        result = _parse_yaml(yaml)
        assert result["team_budgets"][0]["team"] == "backend"
        assert result["team_budgets"][0]["monthly"] == pytest.approx(500.0)
        assert result["team_budgets"][0]["alert_threshold"] == pytest.approx(0.8)

    def test_inline_comment_stripped(self):
        result = _parse_yaml("foo: bar  # this is a comment\n")
        assert result["foo"] == "bar"

    def test_full_config(self):
        yaml = (
            "workspaces:\n"
            "  - name: prod\n"
            "identities:\n"
            "  - name: billing-agent\n"
            "    team: backend\n"
            "    workspace: prod\n"
            "rbac:\n"
            "  - user: alice@example.com\n"
            "    role: admin\n"
            "  - group: eng@example.com\n"
            "    role: member\n"
        )
        result = _parse_yaml(yaml)
        assert len(result["workspaces"]) == 1
        assert len(result["identities"]) == 1
        assert len(result["rbac"]) == 2
        assert result["rbac"][0]["user"] == "alice@example.com"
        assert result["rbac"][1]["group"] == "eng@example.com"


class TestCoerce:
    def test_true(self):
        assert _coerce("true") is True

    def test_false(self):
        assert _coerce("false") is False

    def test_int(self):
        assert _coerce("42") == 42

    def test_float(self):
        assert _coerce("3.14") == pytest.approx(3.14)

    def test_string(self):
        assert _coerce("hello") == "hello"

    def test_quoted_string(self):
        assert _coerce('"hello world"') == "hello world"


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_missing_file_returns_empty(self, tmp_path):
        result = load_config(tmp_path / "nonexistent.yaml")
        assert result == {}

    def test_loads_yaml(self, tmp_path):
        f = tmp_path / ".agent-strace.yaml"
        f.write_text("workspaces:\n  - name: prod\n")
        result = load_config(f)
        assert result["workspaces"] == [{"name": "prod"}]


# ---------------------------------------------------------------------------
# Apply helpers
# ---------------------------------------------------------------------------

class TestApplyWorkspaces:
    def test_creates_new_workspace(self, tmp_path):
        cfg = {"workspaces": [{"name": "prod"}]}
        changes = _apply_workspaces(cfg, str(tmp_path), dry_run=False)
        assert any("+ workspace: prod" in c for c in changes)
        assert "prod" in list_workspaces(str(tmp_path))

    def test_dry_run_does_not_create(self, tmp_path):
        cfg = {"workspaces": [{"name": "prod"}]}
        _apply_workspaces(cfg, str(tmp_path), dry_run=True)
        assert "prod" not in list_workspaces(str(tmp_path))

    def test_existing_workspace_no_change(self, tmp_path):
        from agent_trace.workspace import create_workspace
        create_workspace("prod", str(tmp_path))
        cfg = {"workspaces": [{"name": "prod"}]}
        changes = _apply_workspaces(cfg, str(tmp_path), dry_run=False)
        assert any("exists" in c for c in changes)

    def test_empty_workspaces(self, tmp_path):
        changes = _apply_workspaces({}, str(tmp_path), dry_run=False)
        assert changes == []


class TestApplyRbac:
    def test_assigns_new_role(self, tmp_path):
        cfg = {"rbac": [{"user": "alice@example.com", "role": "admin"}]}
        changes = _apply_rbac(cfg, str(tmp_path), dry_run=False)
        assert any("+ rbac" in c for c in changes)
        store = RBACStore(str(tmp_path))
        assert store.get_role("alice@example.com") == "admin"

    def test_dry_run_does_not_assign(self, tmp_path):
        cfg = {"rbac": [{"user": "alice@example.com", "role": "admin"}]}
        _apply_rbac(cfg, str(tmp_path), dry_run=True)
        store = RBACStore(str(tmp_path))
        assert store.get_role("alice@example.com") is None

    def test_no_change_when_role_matches(self, tmp_path):
        store = RBACStore(str(tmp_path))
        store.assign("alice@example.com", "user", "admin")
        cfg = {"rbac": [{"user": "alice@example.com", "role": "admin"}]}
        changes = _apply_rbac(cfg, str(tmp_path), dry_run=False)
        assert any("no change" in c for c in changes)

    def test_workspace_role(self, tmp_path):
        cfg = {"rbac": [{"user": "bob@example.com", "role": "workspace:member",
                          "workspace": "prod"}]}
        changes = _apply_rbac(cfg, str(tmp_path), dry_run=False)
        assert any("workspace:member" in c for c in changes)


class TestApplyIdentities:
    def test_creates_identity(self, tmp_path):
        cfg = {"identities": [{"name": "billing-agent", "team": "backend"}]}
        changes = _apply_identities(cfg, str(tmp_path), dry_run=False)
        assert any("+ identity: billing-agent" in c for c in changes)
        data = json.loads((tmp_path / "identities.json").read_text())
        assert "billing-agent" in data

    def test_dry_run_no_file(self, tmp_path):
        cfg = {"identities": [{"name": "billing-agent", "team": "backend"}]}
        _apply_identities(cfg, str(tmp_path), dry_run=True)
        assert not (tmp_path / "identities.json").exists()


class TestApplyTeamBudgets:
    def test_creates_budget(self, tmp_path):
        cfg = {"team_budgets": [{"team": "backend", "monthly": 500.0}]}
        changes = _apply_team_budgets(cfg, str(tmp_path), dry_run=False)
        assert any("+ team_budget" in c for c in changes)
        data = json.loads((tmp_path / "team_budgets.json").read_text())
        assert "backend/" in data

    def test_dry_run_no_file(self, tmp_path):
        cfg = {"team_budgets": [{"team": "backend", "monthly": 500.0}]}
        _apply_team_budgets(cfg, str(tmp_path), dry_run=True)
        assert not (tmp_path / "team_budgets.json").exists()


# ---------------------------------------------------------------------------
# config_diff
# ---------------------------------------------------------------------------

class TestConfigDiff:
    def test_no_drift(self, tmp_path):
        from agent_trace.workspace import create_workspace
        create_workspace("prod", str(tmp_path))
        store = RBACStore(str(tmp_path))
        store.assign("alice@example.com", "user", "admin")
        cfg = {
            "workspaces": [{"name": "prod"}],
            "rbac": [{"user": "alice@example.com", "role": "admin"}],
        }
        lines = config_diff(cfg, str(tmp_path))
        assert lines == ["No drift detected."]

    def test_missing_workspace(self, tmp_path):
        cfg = {"workspaces": [{"name": "prod"}]}
        lines = config_diff(cfg, str(tmp_path))
        assert any("+ workspace: prod" in l for l in lines)

    def test_extra_workspace(self, tmp_path):
        from agent_trace.workspace import create_workspace
        create_workspace("orphan", str(tmp_path))
        cfg = {}
        lines = config_diff(cfg, str(tmp_path))
        assert any("- workspace: orphan" in l for l in lines)

    def test_missing_rbac(self, tmp_path):
        cfg = {"rbac": [{"user": "alice@example.com", "role": "admin"}]}
        lines = config_diff(cfg, str(tmp_path))
        assert any("+ rbac: alice@example.com" in l for l in lines)

    def test_changed_rbac(self, tmp_path):
        store = RBACStore(str(tmp_path))
        store.assign("alice@example.com", "user", "viewer")
        cfg = {"rbac": [{"user": "alice@example.com", "role": "admin"}]}
        lines = config_diff(cfg, str(tmp_path))
        assert any("~ rbac" in l and "viewer" in l and "admin" in l for l in lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _run(cmd_fn, argv_dict: dict, tmp_path) -> tuple[int, str]:
    import argparse
    ns = argparse.Namespace(**argv_dict)
    buf = StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        rc = cmd_fn(ns)
    finally:
        sys.stdout = old
    return rc, buf.getvalue()


class TestCLIApply:
    def test_apply_creates_workspace(self, tmp_path):
        cfg_file = tmp_path / ".agent-strace.yaml"
        cfg_file.write_text("workspaces:\n  - name: prod\n")
        rc, out = _run(cmd_apply, {
            "config": str(cfg_file), "dry_run": False,
            "server": "", "auth_key": "", "dir": str(tmp_path),
        }, tmp_path)
        assert rc == 0
        assert "+ workspace: prod" in out

    def test_apply_dry_run(self, tmp_path):
        cfg_file = tmp_path / ".agent-strace.yaml"
        cfg_file.write_text("workspaces:\n  - name: prod\n")
        rc, out = _run(cmd_apply, {
            "config": str(cfg_file), "dry_run": True,
            "server": "", "auth_key": "", "dir": str(tmp_path),
        }, tmp_path)
        assert rc == 0
        assert "dry-run" in out
        ws_names = {w.name for w in list_workspaces(str(tmp_path))}
        assert "prod" not in ws_names

    def test_apply_missing_config(self, tmp_path):
        rc, _ = _run(cmd_apply, {
            "config": str(tmp_path / "missing.yaml"), "dry_run": False,
            "server": "", "auth_key": "", "dir": str(tmp_path),
        }, tmp_path)
        assert rc == 1


class TestCLIConfigDiff:
    def test_config_diff_no_drift(self, tmp_path):
        from agent_trace.workspace import create_workspace
        create_workspace("prod", str(tmp_path))
        cfg_file = tmp_path / ".agent-strace.yaml"
        cfg_file.write_text("workspaces:\n  - name: prod\n")
        rc, out = _run(cmd_config_diff, {
            "config": str(cfg_file), "dir": str(tmp_path),
        }, tmp_path)
        assert rc == 0
        assert "No drift" in out

    def test_config_diff_shows_missing(self, tmp_path):
        cfg_file = tmp_path / ".agent-strace.yaml"
        cfg_file.write_text("workspaces:\n  - name: prod\n")
        rc, out = _run(cmd_config_diff, {
            "config": str(cfg_file), "dir": str(tmp_path),
        }, tmp_path)
        assert rc == 0
        assert "prod" in out
