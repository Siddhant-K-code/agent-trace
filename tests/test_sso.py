"""Tests for sso.py — OIDC helpers, session store, token storage, and CLI."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import tempfile
import time
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, "src")

from agent_trace.sso import (
    SessionStore,
    OIDCConfig,
    parse_session_cookie,
    make_session_cookie,
    decode_id_token_claims,
    build_auth_url,
    save_auth_token,
    load_auth_token,
    clear_auth_token,
    cmd_auth,
    _SESSION_COOKIE,
)
from agent_trace import sso

try:
    from _concurrency import run_concurrently
except ImportError:
    from tests._concurrency import run_concurrently


def _make_jwt(claims: dict) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"RS256"}').rstrip(b"=").decode()
    payload = base64.urlsafe_b64encode(
        json.dumps(claims).encode()
    ).rstrip(b"=").decode()
    return f"{header}.{payload}.fakesig"


def _run_auth(argv_dict: dict, stdin_input: str = "") -> tuple[int, str]:
    ns = argparse.Namespace(**argv_dict)
    buf = StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        if stdin_input:
            with patch("builtins.input", return_value=stdin_input):
                rc = cmd_auth(ns)
        else:
            rc = cmd_auth(ns)
    finally:
        sys.stdout = old
    return rc, buf.getvalue()


class TestSessionStore(unittest.TestCase):
    def test_create_and_get(self):
        store = SessionStore()
        token = store.create({"sub": "u1", "email": "alice@example.com"})
        session = store.get(token)
        self.assertIsNotNone(session)
        self.assertEqual(session["email"], "alice@example.com")

    def test_get_missing_returns_none(self):
        self.assertIsNone(SessionStore().get("nonexistent"))

    def test_get_expired_returns_none(self):
        store = SessionStore()
        token = store.create({"sub": "u1"}, ttl=-1)
        self.assertIsNone(store.get(token))

    def test_delete(self):
        store = SessionStore()
        token = store.create({"sub": "u1"})
        store.delete(token)
        self.assertIsNone(store.get(token))

    def test_purge_expired(self):
        store = SessionStore()
        t1 = store.create({"sub": "u1"}, ttl=-1)
        t2 = store.create({"sub": "u2"}, ttl=3600)
        self.assertEqual(store.purge_expired(), 1)
        self.assertIsNotNone(store.get(t2))

    def test_unique_tokens(self):
        store = SessionStore()
        tokens = {store.create({"sub": f"u{i}"}) for i in range(10)}
        self.assertEqual(len(tokens), 10)


class TestOIDCConfig(unittest.TestCase):
    def test_not_configured_when_empty(self):
        self.assertFalse(OIDCConfig().is_configured())

    def test_configured_when_all_set(self):
        cfg = OIDCConfig(issuer="https://accounts.google.com",
                         client_id="cid", client_secret="csec")
        self.assertTrue(cfg.is_configured())

    def test_not_configured_missing_secret(self):
        self.assertFalse(OIDCConfig(issuer="https://accounts.google.com",
                                    client_id="cid").is_configured())


class TestCookieHelpers(unittest.TestCase):
    def test_parse_session_cookie(self):
        header = f"other=val; {_SESSION_COOKIE}=abc123; foo=bar"
        self.assertEqual(parse_session_cookie(header), "abc123")

    def test_parse_missing_returns_empty(self):
        self.assertEqual(parse_session_cookie("other=val"), "")

    def test_parse_empty_header(self):
        self.assertEqual(parse_session_cookie(""), "")

    def test_make_session_cookie_contains_token(self):
        cookie = make_session_cookie("mytoken123")
        self.assertIn("mytoken123", cookie)
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=Lax", cookie)

    def test_make_session_cookie_max_age(self):
        self.assertIn("Max-Age=1800", make_session_cookie("tok", ttl=1800))


class TestDecodeIdToken(unittest.TestCase):
    def test_decode_claims(self):
        claims = {"sub": "user123", "email": "alice@example.com", "iat": 1000}
        decoded = decode_id_token_claims(_make_jwt(claims))
        self.assertEqual(decoded["sub"], "user123")
        self.assertEqual(decoded["email"], "alice@example.com")

    def test_invalid_jwt_raises(self):
        with self.assertRaises(ValueError):
            decode_id_token_claims("not.a.valid.jwt.format.extra")


class TestBuildAuthUrl(unittest.TestCase):
    def test_contains_required_params(self):
        discovery = {"authorization_endpoint": "https://idp.example.com/auth"}
        url = build_auth_url(discovery, "client123", "https://app/callback",
                             state="st", nonce="nc")
        self.assertIn("client_id=client123", url)
        self.assertIn("response_type=code", url)
        self.assertIn("state=st", url)
        self.assertIn("nonce=nc", url)
        self.assertIn("redirect_uri=", url)

    def test_custom_scopes(self):
        discovery = {"authorization_endpoint": "https://idp.example.com/auth"}
        url = build_auth_url(discovery, "cid", "https://app/cb",
                             state="s", nonce="n", scopes=["openid", "email"])
        self.assertTrue("openid" in url and "email" in url)


class TestTokenStorage(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._auth_file = Path(self._tmp) / "auth.json"
        self._patcher = patch("agent_trace.sso._AUTH_FILE", self._auth_file)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_save_and_load(self):
        save_auth_token("https://c.example.com", "tok123", time.time() + 3600)
        self.assertEqual(load_auth_token("https://c.example.com"), "tok123")

    def test_load_missing_returns_none(self):
        self.assertIsNone(load_auth_token("https://c.example.com"))

    def test_load_expired_returns_none(self):
        save_auth_token("https://c.example.com", "tok123", time.time() - 1)
        self.assertIsNone(load_auth_token("https://c.example.com"))

    def test_clear_token(self):
        save_auth_token("https://c.example.com", "tok123", time.time() + 3600)
        self.assertTrue(clear_auth_token("https://c.example.com"))
        self.assertIsNone(load_auth_token("https://c.example.com"))

    def test_clear_nonexistent_returns_false(self):
        self.assertFalse(clear_auth_token("https://nobody.example.com"))

    def test_multiple_servers(self):
        save_auth_token("https://a.example.com", "tok_a", time.time() + 3600)
        save_auth_token("https://b.example.com", "tok_b", time.time() + 3600)
        self.assertEqual(load_auth_token("https://a.example.com"), "tok_a")
        self.assertEqual(load_auth_token("https://b.example.com"), "tok_b")


_AUTH_WORKER_SCRIPT = """
import time
from pathlib import Path
from agent_trace import sso
sso._AUTH_FILE = Path(sys.argv[2])
wait_for_start()
mode, servers = sys.argv[3], sys.argv[4:]
for server in servers:
    if mode == "login":
        sso.save_auth_token(server, "tok-" + server, time.time() + 3600)
    else:
        assert sso.clear_auth_token(server), server
"""


class TestTokenStorageLocking(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._auth_file = Path(self._tmp.name) / "home" / "auth.json"
        self._patcher = patch("agent_trace.sso._AUTH_FILE", self._auth_file)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self._tmp.cleanup()

    def test_concurrent_logins_keep_every_token(self):
        workers, per_worker = 6, 10
        args = [
            [str(self._auth_file), "login",
             *(f"https://w{i}-{n}.example.com" for n in range(per_worker))]
            for i in range(workers)
        ]
        run_concurrently(_AUTH_WORKER_SCRIPT, args)
        data = json.loads(self._auth_file.read_text())
        self.assertEqual(len(data), workers * per_worker)
        for server, entry in data.items():
            self.assertEqual(entry["token"], "tok-" + server)

    def test_concurrent_login_and_logout_do_not_corrupt_file(self):
        keep = [f"https://keep-{i}.example.com" for i in range(20)]
        drop = [f"https://drop-{i}.example.com" for i in range(20)]
        for server in drop:
            save_auth_token(server, "old", time.time() + 3600)
        auth = str(self._auth_file)
        run_concurrently(
            _AUTH_WORKER_SCRIPT,
            [
                [auth, "login", *keep[:10]],
                [auth, "login", *keep[10:]],
                [auth, "logout", *drop[:10]],
                [auth, "logout", *drop[10:]],
            ],
        )
        data = json.loads(self._auth_file.read_text())
        self.assertEqual(sorted(data), sorted(keep))

    @unittest.skipIf(os.name == "nt", "POSIX file modes")
    def test_token_file_is_always_0600(self):
        self._auth_file.parent.mkdir(parents=True)
        self._auth_file.write_text("{}")
        self._auth_file.chmod(0o644)
        save_auth_token("https://a.example.com", "tok", time.time() + 3600)
        self.assertEqual(self._auth_file.stat().st_mode & 0o777, 0o600)
        self._auth_file.chmod(0o644)
        self.assertTrue(clear_auth_token("https://a.example.com"))
        self.assertEqual(self._auth_file.stat().st_mode & 0o777, 0o600)

    def test_write_leaves_no_temp_files_and_keeps_format(self):
        save_auth_token("https://a.example.com", "tok", 123.0)
        names = sorted(p.name for p in self._auth_file.parent.iterdir())
        self.assertEqual(names, ["auth.json", "auth.json.lock"])
        self.assertEqual(
            self._auth_file.read_text(),
            json.dumps({"https://a.example.com": {"token": "tok", "expires_at": 123.0}}, indent=2),
        )

    def test_login_reports_lock_timeout(self):
        err = StringIO()
        with patch.object(sso, "file_lock", side_effect=TimeoutError("busy")), \
                patch("sys.stderr", err):
            rc, _ = _run_auth(
                {"auth_cmd": "login", "server": "https://c.example.com", "force": True},
                stdin_input="tok",
            )
        self.assertEqual(rc, 1)
        self.assertIn("could not store token", err.getvalue())

    def test_logout_reports_lock_timeout(self):
        save_auth_token("https://c.example.com", "tok", time.time() + 3600)
        err = StringIO()
        with patch.object(sso, "file_lock", side_effect=TimeoutError("busy")), \
                patch("sys.stderr", err):
            rc, _ = _run_auth({"auth_cmd": "logout", "server": "https://c.example.com"})
        self.assertEqual(rc, 1)
        self.assertIn("could not update", err.getvalue())
        self.assertEqual(load_auth_token("https://c.example.com"), "tok")


class TestCLIAuth(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._auth_file = Path(self._tmp) / "auth.json"
        self._patcher = patch("agent_trace.sso._AUTH_FILE", self._auth_file)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_logout_no_token(self):
        rc, out = _run_auth({"auth_cmd": "logout", "server": "https://c.example.com"})
        self.assertEqual(rc, 0)
        self.assertIn("No stored token", out)

    def test_logout_removes_token(self):
        save_auth_token("https://c.example.com", "tok", time.time() + 3600)
        rc, out = _run_auth({"auth_cmd": "logout", "server": "https://c.example.com"})
        self.assertEqual(rc, 0)
        self.assertIn("Logged out", out)

    def test_status_no_tokens(self):
        rc, out = _run_auth({"auth_cmd": "status", "server": ""})
        self.assertEqual(rc, 0)
        self.assertIn("No stored tokens", out)

    def test_status_specific_server_logged_in(self):
        save_auth_token("https://c.example.com", "tok", time.time() + 3600)
        rc, out = _run_auth({"auth_cmd": "status", "server": "https://c.example.com"})
        self.assertEqual(rc, 0)
        self.assertIn("Logged in", out)

    def test_status_specific_server_not_logged_in(self):
        rc, out = _run_auth({"auth_cmd": "status", "server": "https://c.example.com"})
        self.assertEqual(rc, 0)
        self.assertIn("Not logged in", out)

    def test_login_stores_token(self):
        rc, out = _run_auth(
            {"auth_cmd": "login", "server": "https://c.example.com", "force": False},
            stdin_input="mytoken123",
        )
        self.assertEqual(rc, 0)
        self.assertIn("Logged in", out)
        self.assertEqual(load_auth_token("https://c.example.com"), "mytoken123")

    def test_login_already_logged_in(self):
        save_auth_token("https://c.example.com", "existing", time.time() + 3600)
        rc, out = _run_auth(
            {"auth_cmd": "login", "server": "https://c.example.com", "force": False})
        self.assertEqual(rc, 0)
        self.assertIn("Already logged in", out)

    def test_no_subcommand_returns_error(self):
        rc, _ = _run_auth({"auth_cmd": None})
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
