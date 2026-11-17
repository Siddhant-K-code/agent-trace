"""Tests for iac.py — config parsing, apply, and config-diff."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path
from typing import Any
from unittest import mock

sys.path.insert(0, "src")

from agent_trace import iac
from agent_trace.iac import (
    _parse_yaml,
    _coerce,
    load_config,
    config_diff,
    _apply_workspaces,
    _apply_rbac,
    _apply_identities,
    _apply_team_budgets,
    _http_get,
    cmd_apply,
    cmd_config_diff,
)
from agent_trace.workspace import list_workspaces, create_workspace
from agent_trace.rbac import RBACStore

try:
    from _concurrency import run_concurrently
except ImportError:
    from tests._concurrency import run_concurrently

try:
    from _redirect_servers import ALL_REDIRECTS, ORIGIN_NOTE, REDIRECT_NOTE, RedirectServers
except ImportError:
    from tests._redirect_servers import (
        ALL_REDIRECTS, ORIGIN_NOTE, REDIRECT_NOTE, RedirectServers,
    )


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

    def test_apply_server_post_redirect_is_an_error(self):
        # urlopen re-sent the POST as a GET with no body and the key, and the
        # workspace was reported as added.
        servers = RedirectServers(self, a_reply=[])
        cfg_file = Path(self._tmp) / ".agent-strace.yaml"
        cfg_file.write_text("workspaces:\n  - name: prod\n")
        for status in ALL_REDIRECTS:
            with self.subTest(status=status):
                servers.redirect(status)
                rc, out = _run(cmd_apply, {"config": str(cfg_file), "dry_run": False,
                                           "server": servers.url, "auth_key": "iac-key",
                                           "dir": self._tmp})
                self.assertEqual(rc, 1)
                self.assertIn(
                    f"  ! workspace: prod: POST {servers.url}/api/workspaces → "
                    f"{status}{REDIRECT_NOTE}\n",
                    out,
                )
                self.assertNotIn("+ workspace: prod", out)
                self.assertEqual(servers.a.methods(), ["GET", "POST"])
                self.assertEqual(servers.b.received(), [])

    def test_apply_server_get_redirect_to_another_origin_is_refused(self):
        # urlopen sent the key to B and read B's list, so prod was reported
        # as already on the server. Then the refusal counted as an empty
        # list, and prod was POSTed to A.
        servers = RedirectServers(
            self, methods=("GET",), reply=[{"name": "prod"}], a_reply=[]
        )
        cfg_file = Path(self._tmp) / ".agent-strace.yaml"
        cfg_file.write_text("workspaces:\n  - name: prod\n")
        for status in ALL_REDIRECTS:
            with self.subTest(status=status):
                servers.redirect(status)
                rc, out = _run(cmd_apply, {"config": str(cfg_file), "dry_run": False,
                                           "server": servers.url, "auth_key": "iac-key",
                                           "dir": self._tmp})
                self.assertEqual(rc, 1)
                self.assertIn(
                    f"  {_LIST_FAILED}GET {servers.url}/api/workspaces → "
                    f"{status}{ORIGIN_NOTE}\n",
                    out,
                )
                self.assertNotIn("exists on server", out)
                self.assertEqual(servers.a.methods(), ["GET"])
                self.assertEqual(servers.b.received(), [])


# A server's raw status line. Shown raw, it erases the line, writes to the
# clipboard (OSC 52), rings the bell and moves the cursor back.
_BAD_STATUS_LINE = "\x1b[2K\x1b]52;c;aWQK\x07EVIL 200 OK\r\n"
_BAD_STATUS_LINE_SHOWN = "\\x1b[2K\\x1b]52;c;aWQK\\x07EVIL 200 OK\\r\\n"
_LIST_FAILED = "! workspaces: could not list them on the server, so nothing was applied: "
_NOT_A_LIST = "the reply is not a list of workspaces with names"
_RBAC = "rbac:\n  - user: alice@example.com\n    role: admin\n"
_PROD_AND_RBAC = "workspaces:\n  - name: prod\n" + _RBAC
_PROD_AND_STAGING = "workspaces:\n  - name: prod\n  - name: staging\n"
_ALICE = "alice@example.com → admin"
_ALICE_POST = {
    "principal": "alice@example.com", "principal_type": "user",
    "role": "admin", "workspace_id": "",
}


def _http_reply(status: str, body: bytes = b"") -> bytes:
    """Return a whole HTTP/1.0 response with *status* and *body*."""
    return f"HTTP/1.0 {status}\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body


def _json_reply(value: Any) -> bytes:
    return _http_reply("200 OK", json.dumps(value).encode())


class _ScriptedServer(ThreadingHTTPServer):
    """Answer each request with the bytes scripted for its method and path.

    An entry in *script* is the whole response, status line included, so it
    can be a bad status line or a body that is not JSON. Empty bytes close
    the connection with no reply. A request with no entry gets 200 and {}.
    """

    daemon_threads = True
    block_on_close = False

    def __init__(self, script: dict[str, bytes]) -> None:
        super().__init__(("127.0.0.1", 0), _ScriptedHandler)
        self.url = f"http://127.0.0.1:{self.server_address[1]}"
        self.script = script
        self.lock = threading.Lock()
        self._received: list[tuple[str, str, Any]] = []

    def handle_error(self, request, client_address) -> None:
        pass  # the client gave up and closed the connection

    def record(self, method: str, path: str, body: bytes) -> None:
        with self.lock:
            self._received.append((method, path, json.loads(body) if body else None))

    def requests(self) -> list[tuple[str, str, Any]]:
        """Return the method, path and JSON body of each request, in order."""
        with self.lock:
            return list(self._received)

    def clear(self) -> None:
        with self.lock:
            self._received.clear()


class _ScriptedHandler(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:
        pass

    def _answer(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        self.server.record(self.command, self.path, self.rfile.read(length))
        key = f"{self.command} {self.path}"
        self.wfile.write(self.server.script.get(key, _json_reply({})))

    do_GET = do_POST = _answer


class TestApplyServerListing(unittest.TestCase):
    """apply --server lists the server's workspaces before it sends anything."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self._tmp = tmp.name
        self._config = Path(self._tmp) / ".agent-strace.yaml"
        # A proxy from the environment or the system must not see these.
        environ = mock.patch.dict(os.environ, {"no_proxy": "*"})
        environ.start()
        self.addCleanup(environ.stop)

    def _serve(self, script: dict[str, bytes]) -> _ScriptedServer:
        server = _ScriptedServer(script)
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True,
        )
        thread.start()
        self.addCleanup(thread.join, 5)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server

    def _apply(self, config: str, server: str, dry_run: bool) -> tuple[int, str]:
        self._config.write_text(config)
        return _run(cmd_apply, {"config": str(self._config), "dry_run": dry_run,
                                "server": server, "auth_key": "iac-key", "dir": self._tmp})

    def _assert_output(
        self, out: str, dry_run: bool, *changes: str, added: int = 0, errors: int = 0,
    ) -> None:
        """Check that apply printed its header, *changes* and the summary, and nothing else."""
        prefix = "[dry-run] " if dry_run else ""
        header, _, rest = out.partition("\n")
        self.assertEqual(header.rstrip("."), f"{prefix}Applying {self._config}")
        lines = "".join(f"  {change}\n" for change in changes)
        self.assertEqual(
            rest, f"{lines}\n{prefix}Done. +{added} added, ~0 updated, !{errors} errors\n"
        )

    def _assert_no_raw_controls(self, out: str) -> None:
        for char in ("\x1b", "\x07", "\r"):
            self.assertNotIn(char, out)

    def test_failed_listing_sends_nothing_and_exits_1(self):
        # Each failure counted as a server with no workspaces. --dry-run
        # showed prod as new and exited 0, and a real run POSTed prod and
        # the RBAC entry.
        server = self._serve({})
        url = f"{server.url}/api/workspaces"
        replies = [
            ("500", _http_reply("500 Internal Server Error"), f"GET {url} → 500"),
            ("no reply", b"", "Remote end closed connection without response"),
            ("bad status line", b"NOT HTTP\r\n", "NOT HTTP\\r\\n"),
            ("invalid JSON", _http_reply("200 OK", b"not json"),
             "Expecting value: line 1 column 1 (char 0)"),
        ]
        for shape in ({}, {"name": "prod"}, None, ["prod"], [{"id": 1}], [{"name": 5}],
                      [{"name": "prod"}, {"id": 1}]):
            replies.append((json.dumps(shape), _json_reply(shape), _NOT_A_LIST))
        for label, reply, reason in replies:
            for dry_run in (False, True):
                with self.subTest(reply=label, dry_run=dry_run):
                    server.script["GET /api/workspaces"] = reply
                    server.clear()
                    rc, out = self._apply(_PROD_AND_RBAC, server.url, dry_run)
                    self._assert_output(out, dry_run, _LIST_FAILED + reason, errors=1)
                    self.assertEqual(rc, 1)
                    self.assertEqual(server.requests(), [("GET", "/api/workspaces", None)])

    def test_refused_connection_sends_nothing_and_exits_1(self):
        # A port that was listening a moment ago and is now closed. A bound
        # socket that never listens is not enough: macOS drops the SYN.
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            url = f"http://127.0.0.1:{listener.getsockname()[1]}"
        for dry_run in (False, True):
            with self.subTest(dry_run=dry_run), \
                    mock.patch.object(iac, "_http_get", wraps=iac._http_get) as get, \
                    mock.patch.object(iac, "_http_post", wraps=iac._http_post) as post:
                rc, out = self._apply(_PROD_AND_RBAC, url, dry_run)
                reason = re.search(r"<urlopen error [^\n]+>", out)
                self.assertIsNotNone(reason, out)
                self._assert_output(out, dry_run, _LIST_FAILED + reason.group(), errors=1)
                self.assertEqual(rc, 1)
                get.assert_called_once_with(f"{url}/api/workspaces", "iac-key")
                post.assert_not_called()

    def test_failed_listing_shows_the_reason_as_escapes(self):
        # http.client gives the server's raw status line as the error text.
        server = self._serve({"GET /api/workspaces": _BAD_STATUS_LINE.encode("latin-1")})
        for dry_run in (False, True):
            with self.subTest(dry_run=dry_run):
                server.clear()
                rc, out = self._apply(_PROD_AND_RBAC, server.url, dry_run)
                self._assert_output(out, dry_run, _LIST_FAILED + _BAD_STATUS_LINE_SHOWN, errors=1)
                self._assert_no_raw_controls(out)
                self.assertEqual(rc, 1)
                self.assertEqual(server.requests(), [("GET", "/api/workspaces", None)])

    def test_failed_workspace_post_shows_the_reason_as_escapes(self):
        server = self._serve({
            "GET /api/workspaces": _json_reply([]),
            "POST /api/workspaces": _BAD_STATUS_LINE.encode("latin-1"),
        })
        rc, out = self._apply("workspaces:\n  - name: prod\n", server.url, False)
        self._assert_output(out, False, f"! workspace: prod: {_BAD_STATUS_LINE_SHOWN}", errors=1)
        self._assert_no_raw_controls(out)
        self.assertEqual(rc, 1)
        self.assertEqual(server.requests(), [
            ("GET", "/api/workspaces", None),
            ("POST", "/api/workspaces", {"name": "prod"}),
        ])

    def test_failed_rbac_post_shows_the_reason_as_escapes(self):
        server = self._serve({
            "GET /api/workspaces": _json_reply([{"name": "prod"}]),
            "POST /api/rbac": _BAD_STATUS_LINE.encode("latin-1"),
        })
        rc, out = self._apply(_PROD_AND_RBAC, server.url, False)
        self._assert_output(
            out,
            False,
            "  workspace: prod (exists on server)",
            f"! rbac: {_ALICE}: {_BAD_STATUS_LINE_SHOWN}",
            errors=1,
        )
        self._assert_no_raw_controls(out)
        self.assertEqual(rc, 1)
        self.assertEqual(server.requests(), [
            ("GET", "/api/workspaces", None),
            ("POST", "/api/rbac", _ALICE_POST),
        ])

    def test_rbac_only_config_makes_no_get(self):
        # The listing only matters for workspaces with names, so an apply
        # without them does not depend on it.
        server = self._serve({"GET /api/workspaces": _http_reply("500 Internal Server Error")})
        configs = {"rbac only": _RBAC, "no workspace name": "workspaces:\n  - team: a\n" + _RBAC}
        for label, config in configs.items():
            for dry_run in (False, True):
                with self.subTest(config=label, dry_run=dry_run):
                    server.clear()
                    rc, out = self._apply(config, server.url, dry_run)
                    self._assert_output(out, dry_run, f"+ rbac: {_ALICE}", added=1)
                    self.assertEqual(rc, 0)
                    posts = [] if dry_run else [("POST", "/api/rbac", _ALICE_POST)]
                    self.assertEqual(server.requests(), posts)

    def test_get_is_made_when_a_later_workspace_has_a_name(self):
        # The first workspace has no name. The GET must still be made for prod.
        server = self._serve({"GET /api/workspaces": _json_reply([{"name": "prod"}])})
        for dry_run in (False, True):
            with self.subTest(dry_run=dry_run):
                server.clear()
                rc, out = self._apply(
                    "workspaces:\n  - team: a\n  - name: prod\n", server.url, dry_run,
                )
                self._assert_output(out, dry_run, "  workspace: prod (exists on server)")
                self.assertEqual(rc, 0)
                self.assertEqual(server.requests(), [("GET", "/api/workspaces", None)])

    def test_listed_workspace_exists_and_the_others_are_posted(self):
        server = self._serve({"GET /api/workspaces": _json_reply([{"name": "prod"}])})
        for dry_run in (False, True):
            with self.subTest(dry_run=dry_run):
                server.clear()
                rc, out = self._apply(_PROD_AND_STAGING + _RBAC, server.url, dry_run)
                self._assert_output(
                    out,
                    dry_run,
                    "  workspace: prod (exists on server)",
                    "+ workspace: staging",
                    f"+ rbac: {_ALICE}",
                    added=2,
                )
                self.assertEqual(rc, 0)
                posts = [] if dry_run else [
                    ("POST", "/api/workspaces", {"name": "staging"}),
                    ("POST", "/api/rbac", _ALICE_POST),
                ]
                self.assertEqual(server.requests(), [("GET", "/api/workspaces", None)] + posts)

    def test_empty_listing_posts_every_workspace(self):
        server = self._serve({"GET /api/workspaces": _json_reply([])})
        for dry_run in (False, True):
            with self.subTest(dry_run=dry_run):
                server.clear()
                rc, out = self._apply(_PROD_AND_STAGING, server.url, dry_run)
                self._assert_output(
                    out, dry_run, "+ workspace: prod", "+ workspace: staging", added=2,
                )
                self.assertEqual(rc, 0)
                posts = [] if dry_run else [
                    ("POST", "/api/workspaces", {"name": "prod"}),
                    ("POST", "/api/workspaces", {"name": "staging"}),
                ]
                self.assertEqual(server.requests(), [("GET", "/api/workspaces", None)] + posts)


class TestHTTPGet(unittest.TestCase):
    def setUp(self):
        self.servers = RedirectServers(self, methods=("GET",), reply=[{"name": "prod"}])
        self.url = self.servers.url + "/api/workspaces"

    def test_redirect_to_another_origin_is_an_error(self):
        # urlopen sent the key to the host in Location.
        servers = self.servers
        for status in ALL_REDIRECTS:
            with self.subTest(status=status):
                servers.redirect(status)
                with self.assertRaises(RuntimeError) as caught:
                    _http_get(self.url, "iac-key")
                self.assertEqual(
                    str(caught.exception), f"GET {self.url} → {status}{ORIGIN_NOTE}"
                )
                # Closed, so Python 3.14 prints no ResourceWarning for it.
                self.assertTrue(caught.exception.__cause__.fp.closed)
                self.assertEqual(servers.a.methods(), ["GET"])
                self.assertEqual(servers.b.received(), [])

    def test_redirect_to_the_same_origin_is_followed(self):
        servers = self.servers
        servers.redirect(301, servers.moved)
        self.assertEqual(_http_get(self.url, "iac-key"), [{"name": "prod"}])
        self.assertEqual(
            [(r.path, r.headers["Authorization"]) for r in servers.a.received()],
            [
                ("/api/workspaces", "Bearer iac-key"),
                ("/moved/api/workspaces", "Bearer iac-key"),
            ],
        )
        self.assertEqual(servers.b.received(), [])


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
