"""Infrastructure-as-code support for agent-trace.

Reads ``.agent-strace.yaml`` (or a custom path) and applies the declared
configuration to a local store or a hosted collector via HTTP.

Supported config sections:
    workspaces:   list of workspace IDs to create
    policies:     named policy files to register (path on disk)
    identities:   agent identities (name, team, workspace)
    team_budgets: team spending limits
    rbac:         role assignments

CLI:
    agent-strace apply [--config .agent-strace.yaml] [--server URL] [--dry-run]
    agent-strace config-diff [--config .agent-strace.yaml] [--server URL]

When ``--server`` is omitted the config is applied to the local store only
(workspaces and RBAC are written to disk; policies are validated).

Example ``.agent-strace.yaml``:

    workspaces:
      - name: production
      - name: staging

    identities:
      - name: billing-agent
        team: backend
        workspace: production

    team_budgets:
      - team: backend
        workspace: production
        monthly: 500.0
        alert_threshold: 0.8

    rbac:
      - user: alice@example.com
        role: admin
      - group: eng@example.com
        role: member
      - user: bob@example.com
        role: workspace:member
        workspace: production
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .store import DEFAULT_TRACE_DIR
from .workspace import create_workspace, list_workspaces
from .rbac import RBACStore

DEFAULT_CONFIG = ".agent-strace.yaml"

# ---------------------------------------------------------------------------
# YAML-free parser (stdlib only)
# ---------------------------------------------------------------------------

def _parse_yaml(text: str) -> dict:
    """Minimal YAML parser for the subset used in .agent-strace.yaml.

    Supports:
    - Top-level keys
    - List items (- key: value)
    - Nested key: value pairs under list items
    - String, float, int, bool values
    - Inline comments (#)

    Does NOT support: anchors, multi-line strings, flow style, nested dicts.
    """
    result: dict[str, Any] = {}
    current_key: str | None = None
    current_list: list | None = None
    current_item: dict | None = None

    for raw_line in text.splitlines():
        # Strip inline comments
        line = raw_line.split("#")[0].rstrip()
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip())

        if indent == 0:
            # Top-level key
            if ":" in line:
                k, _, v = line.partition(":")
                k = k.strip()
                v = v.strip()
                if v:
                    result[k] = _coerce(v)
                    current_key = None
                    current_list = None
                    current_item = None
                else:
                    current_key = k
                    current_list = []
                    result[k] = current_list
                    current_item = None
        elif indent == 2 and line.lstrip().startswith("- "):
            # List item start
            rest = line.lstrip()[2:].strip()
            if current_list is None:
                continue
            if ":" in rest:
                k, _, v = rest.partition(":")
                current_item = {k.strip(): _coerce(v.strip())}
            else:
                current_item = {}
            current_list.append(current_item)
        elif indent >= 4 and current_item is not None:
            # Nested key under list item
            stripped = line.strip()
            if ":" in stripped:
                k, _, v = stripped.partition(":")
                current_item[k.strip()] = _coerce(v.strip())

    return result


def _coerce(v: str) -> Any:
    """Convert a string value to int, float, bool, or str."""
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    # Strip surrounding quotes
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    return v


def load_config(path: str | Path) -> dict:
    """Load and parse an .agent-strace.yaml file. Returns empty dict if missing."""
    p = Path(path)
    if not p.exists():
        return {}
    return _parse_yaml(p.read_text())


# ---------------------------------------------------------------------------
# Local apply helpers
# ---------------------------------------------------------------------------

def _apply_workspaces(cfg: dict, base_dir: str, dry_run: bool) -> list[str]:
    changes = []
    existing = set(list_workspaces(base_dir))
    for ws in cfg.get("workspaces", []):
        name = ws.get("name", "")
        if not name:
            continue
        if name not in existing:
            if not dry_run:
                create_workspace(name, base_dir)
            changes.append(f"+ workspace: {name}")
        else:
            changes.append(f"  workspace: {name} (exists)")
    return changes


def _apply_rbac(cfg: dict, base_dir: str, dry_run: bool) -> list[str]:
    changes = []
    store = RBACStore(base_dir)
    for entry in cfg.get("rbac", []):
        principal = entry.get("user") or entry.get("group", "")
        if not principal:
            continue
        ptype = "group" if "group" in entry else "user"
        role = entry.get("role", "")
        workspace_id = entry.get("workspace", "")
        if not role:
            continue
        existing = store.get_role(principal, workspace_id)
        if existing == role:
            changes.append(f"  rbac: {principal} → {role} (no change)")
        else:
            if not dry_run:
                try:
                    store.assign(principal, ptype, role, workspace_id=workspace_id,
                                 assigned_by="agent-strace apply")
                except ValueError as exc:
                    changes.append(f"! rbac: {principal} → {role}: {exc}")
                    continue
            action = "+" if existing is None else "~"
            changes.append(f"{action} rbac: {principal} → {role}"
                           + (f" (workspace: {workspace_id})" if workspace_id else ""))
    return changes


def _apply_identities(cfg: dict, base_dir: str, dry_run: bool) -> list[str]:
    """Write identity entries to .agent-traces/identities.json."""
    changes = []
    identities_path = Path(base_dir) / "identities.json"
    existing: dict = {}
    if identities_path.exists():
        try:
            existing = json.loads(identities_path.read_text())
        except Exception:
            existing = {}

    for entry in cfg.get("identities", []):
        name = entry.get("name", "")
        if not name:
            continue
        if existing.get(name) == entry:
            changes.append(f"  identity: {name} (no change)")
        else:
            action = "+" if name not in existing else "~"
            if not dry_run:
                existing[name] = entry
            changes.append(f"{action} identity: {name}")

    if not dry_run and changes:
        identities_path.parent.mkdir(parents=True, exist_ok=True)
        identities_path.write_text(json.dumps(existing, indent=2))
    return changes


def _apply_team_budgets(cfg: dict, base_dir: str, dry_run: bool) -> list[str]:
    """Write team budget entries to .agent-traces/team_budgets.json."""
    changes = []
    budgets_path = Path(base_dir) / "team_budgets.json"
    existing: dict = {}
    if budgets_path.exists():
        try:
            existing = json.loads(budgets_path.read_text())
        except Exception:
            existing = {}

    for entry in cfg.get("team_budgets", []):
        team = entry.get("team", "")
        if not team:
            continue
        key = f"{team}/{entry.get('workspace', '')}"
        if existing.get(key) == entry:
            changes.append(f"  team_budget: {key} (no change)")
        else:
            action = "+" if key not in existing else "~"
            if not dry_run:
                existing[key] = entry
            changes.append(f"{action} team_budget: {key} monthly={entry.get('monthly', '?')}")

    if not dry_run and changes:
        budgets_path.parent.mkdir(parents=True, exist_ok=True)
        budgets_path.write_text(json.dumps(existing, indent=2))
    return changes


# ---------------------------------------------------------------------------
# Remote apply (HTTP)
# ---------------------------------------------------------------------------

def _http_get(url: str, auth_key: str = "") -> Any:
    headers = {}
    if auth_key:
        headers["Authorization"] = f"Bearer {auth_key}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"GET {url} → {e.code}") from e


def _http_post(url: str, data: dict, auth_key: str = "") -> Any:
    body = json.dumps(data).encode()
    headers = {"Content-Type": "application/json"}
    if auth_key:
        headers["Authorization"] = f"Bearer {auth_key}"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"POST {url} → {e.code}") from e


def _apply_remote(cfg: dict, server: str, auth_key: str, dry_run: bool) -> list[str]:
    """Push config sections to a hosted collector via HTTP."""
    changes = []
    base = server.rstrip("/")

    # Workspaces
    try:
        remote_ws = {w["name"] for w in _http_get(f"{base}/api/workspaces", auth_key)}
    except Exception:
        remote_ws = set()

    for ws in cfg.get("workspaces", []):
        name = ws.get("name", "")
        if not name:
            continue
        if name in remote_ws:
            changes.append(f"  workspace: {name} (exists on server)")
        else:
            if not dry_run:
                try:
                    _http_post(f"{base}/api/workspaces", {"name": name}, auth_key)
                except Exception as exc:
                    changes.append(f"! workspace: {name}: {exc}")
                    continue
            changes.append(f"+ workspace: {name}")

    # RBAC
    for entry in cfg.get("rbac", []):
        principal = entry.get("user") or entry.get("group", "")
        role = entry.get("role", "")
        workspace_id = entry.get("workspace", "")
        if not principal or not role:
            continue
        if not dry_run:
            try:
                _http_post(f"{base}/api/rbac", {
                    "principal": principal,
                    "principal_type": "group" if "group" in entry else "user",
                    "role": role,
                    "workspace_id": workspace_id,
                }, auth_key)
            except Exception as exc:
                changes.append(f"! rbac: {principal} → {role}: {exc}")
                continue
        changes.append(f"+ rbac: {principal} → {role}")

    return changes


# ---------------------------------------------------------------------------
# Config diff
# ---------------------------------------------------------------------------

def config_diff(cfg: dict, base_dir: str) -> list[str]:
    """Compare local config against the current local store state."""
    lines = []

    # Workspaces
    existing_ws = set(list_workspaces(base_dir))
    declared_ws = {ws.get("name", "") for ws in cfg.get("workspaces", [])} - {""}
    for name in sorted(declared_ws - existing_ws):
        lines.append(f"+ workspace: {name}  (declared, not created)")
    for name in sorted(existing_ws - declared_ws):
        lines.append(f"- workspace: {name}  (exists, not in config)")

    # RBAC
    store = RBACStore(base_dir)
    existing_rbac = {(a.principal, a.workspace_id): a.role
                     for a in store.list_assignments()}
    for entry in cfg.get("rbac", []):
        principal = entry.get("user") or entry.get("group", "")
        role = entry.get("role", "")
        workspace_id = entry.get("workspace", "")
        if not principal or not role:
            continue
        current = existing_rbac.get((principal, workspace_id))
        if current is None:
            lines.append(f"+ rbac: {principal} → {role}  (declared, not assigned)")
        elif current != role:
            lines.append(f"~ rbac: {principal}: {current} → {role}  (role changed)")

    if not lines:
        lines.append("No drift detected.")
    return lines


# ---------------------------------------------------------------------------
# CLI handlers
# ---------------------------------------------------------------------------

def cmd_apply(args: argparse.Namespace) -> int:
    config_path = getattr(args, "config", DEFAULT_CONFIG) or DEFAULT_CONFIG
    dry_run = getattr(args, "dry_run", False)
    server = getattr(args, "server", "") or ""
    auth_key = getattr(args, "auth_key", "") or ""
    base_dir = getattr(args, "dir", None) or DEFAULT_TRACE_DIR

    cfg = load_config(config_path)
    if not cfg:
        sys.stderr.write(f"No config found at {config_path}\n")
        return 1

    prefix = "[dry-run] " if dry_run else ""
    sys.stdout.write(f"{prefix}Applying {config_path}...\n")

    all_changes: list[str] = []

    if server:
        all_changes += _apply_remote(cfg, server, auth_key, dry_run)
    else:
        all_changes += _apply_workspaces(cfg, base_dir, dry_run)
        all_changes += _apply_rbac(cfg, base_dir, dry_run)
        all_changes += _apply_identities(cfg, base_dir, dry_run)
        all_changes += _apply_team_budgets(cfg, base_dir, dry_run)

    for line in all_changes:
        sys.stdout.write(f"  {line}\n")

    added = sum(1 for l in all_changes if l.strip().startswith("+"))
    changed = sum(1 for l in all_changes if l.strip().startswith("~"))
    errors = sum(1 for l in all_changes if l.strip().startswith("!"))

    sys.stdout.write(
        f"\n{prefix}Done. +{added} added, ~{changed} updated, !{errors} errors\n"
    )
    return 1 if errors else 0


def cmd_config_diff(args: argparse.Namespace) -> int:
    config_path = getattr(args, "config", DEFAULT_CONFIG) or DEFAULT_CONFIG
    base_dir = getattr(args, "dir", None) or DEFAULT_TRACE_DIR

    cfg = load_config(config_path)
    if not cfg:
        sys.stderr.write(f"No config found at {config_path}\n")
        return 1

    lines = config_diff(cfg, base_dir)
    for line in lines:
        sys.stdout.write(f"  {line}\n")
    return 0
