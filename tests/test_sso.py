"""Tests for sso.py — OIDC helpers, session store, token storage, and CLI."""

from __future__ import annotations

import json
import sys
import time
from io import StringIO
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

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
    _SESSION_TTL,
)


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------

class TestSessionStore:
    def test_create_and_get(self):
        store = SessionStore()
        claims = {"sub": "user123", "email": "alice@example.com"}
        token = store.create(claims)
        session = store.get(token)
        assert session is not None
        assert session["email"] == "alice@example.com"
        assert session["sub"] == "user123"

    def test_get_missing_returns_none(self):
        store = SessionStore()
        assert store.get("nonexistent") is None

    def test_get_expired_returns_none(self):
        store = SessionStore()
        token = store.create({"sub": "u1"}, ttl=-1)  # already expired
        assert store.get(token) is None

    def test_delete(self):
        store = SessionStore()
        token = store.create({"sub": "u1"})
        store.delete(token)
        assert store.get(token) is None

    def test_purge_expired(self):
        store = SessionStore()
        t1 = store.create({"sub": "u1"}, ttl=-1)
        t2 = store.create({"sub": "u2"}, ttl=3600)
        purged = store.purge_expired()
        assert purged == 1
        assert store.get(t2) is not None

    def test_unique_tokens(self):
        store = SessionStore()
        tokens = {store.create({"sub": f"u{i}"}) for i in range(10)}
        assert len(tokens) == 10


# ---------------------------------------------------------------------------
# OIDCConfig
# ---------------------------------------------------------------------------

class TestOIDCConfig:
    def test_not_configured_when_empty(self):
        cfg = OIDCConfig()
        assert cfg.is_configured() is False

    def test_configured_when_all_set(self):
        cfg = OIDCConfig(issuer="https://accounts.google.com",
                         client_id="cid", client_secret="csec")
        assert cfg.is_configured() is True

    def test_not_configured_missing_secret(self):
        cfg = OIDCConfig(issuer="https://accounts.google.com", client_id="cid")
        assert cfg.is_configured() is False


# ---------------------------------------------------------------------------
# Cookie helpers
# ---------------------------------------------------------------------------

class TestCookieHelpers:
    def test_parse_session_cookie(self):
        header = f"other=val; {_SESSION_COOKIE}=abc123; foo=bar"
        assert parse_session_cookie(header) == "abc123"

    def test_parse_missing_returns_empty(self):
        assert parse_session_cookie("other=val") == ""

    def test_parse_empty_header(self):
        assert parse_session_cookie("") == ""

    def test_make_session_cookie_contains_token(self):
        cookie = make_session_cookie("mytoken123")
        assert "mytoken123" in cookie
        assert "HttpOnly" in cookie
        assert "SameSite=Lax" in cookie

    def test_make_session_cookie_max_age(self):
        cookie = make_session_cookie("tok", ttl=1800)
        assert "Max-Age=1800" in cookie


# ---------------------------------------------------------------------------
# JWT decode (no signature verification)
# ---------------------------------------------------------------------------

class TestDecodeIdToken:
    def _make_jwt(self, claims: dict) -> str:
        import base64
        header = base64.urlsafe_b64encode(b'{"alg":"RS256"}').rstrip(b"=").decode()
        payload = base64.urlsafe_b64encode(
            json.dumps(claims).encode()
        ).rstrip(b"=").decode()
        return f"{header}.{payload}.fakesig"

    def test_decode_claims(self):
        claims = {"sub": "user123", "email": "alice@example.com", "iat": 1000}
        token = self._make_jwt(claims)
        decoded = decode_id_token_claims(token)
        assert decoded["sub"] == "user123"
        assert decoded["email"] == "alice@example.com"

    def test_invalid_jwt_raises(self):
        with pytest.raises(ValueError, match="Invalid JWT"):
            decode_id_token_claims("not.a.valid.jwt.format.extra")


# ---------------------------------------------------------------------------
# build_auth_url
# ---------------------------------------------------------------------------

class TestBuildAuthUrl:
    def test_contains_required_params(self):
        discovery = {"authorization_endpoint": "https://idp.example.com/auth"}
        url = build_auth_url(discovery, "client123", "https://app/callback",
                             state="st", nonce="nc")
        assert "client_id=client123" in url
        assert "response_type=code" in url
        assert "state=st" in url
        assert "nonce=nc" in url
        assert "redirect_uri=" in url

    def test_custom_scopes(self):
        discovery = {"authorization_endpoint": "https://idp.example.com/auth"}
        url = build_auth_url(discovery, "cid", "https://app/cb",
                             state="s", nonce="n", scopes=["openid", "email"])
        assert "scope=openid+email" in url or "scope=openid%20email" in url


# ---------------------------------------------------------------------------
# Token storage
# ---------------------------------------------------------------------------

class TestTokenStorage:
    def test_save_and_load(self, tmp_path, monkeypatch):
        auth_file = tmp_path / "auth.json"
        monkeypatch.setattr("agent_trace.sso._AUTH_FILE", auth_file)
        save_auth_token("https://collector.example.com", "tok123", time.time() + 3600)
        loaded = load_auth_token("https://collector.example.com")
        assert loaded == "tok123"

    def test_load_missing_returns_none(self, tmp_path, monkeypatch):
        auth_file = tmp_path / "auth.json"
        monkeypatch.setattr("agent_trace.sso._AUTH_FILE", auth_file)
        assert load_auth_token("https://collector.example.com") is None

    def test_load_expired_returns_none(self, tmp_path, monkeypatch):
        auth_file = tmp_path / "auth.json"
        monkeypatch.setattr("agent_trace.sso._AUTH_FILE", auth_file)
        save_auth_token("https://collector.example.com", "tok123", time.time() - 1)
        assert load_auth_token("https://collector.example.com") is None

    def test_clear_token(self, tmp_path, monkeypatch):
        auth_file = tmp_path / "auth.json"
        monkeypatch.setattr("agent_trace.sso._AUTH_FILE", auth_file)
        save_auth_token("https://collector.example.com", "tok123", time.time() + 3600)
        removed = clear_auth_token("https://collector.example.com")
        assert removed is True
        assert load_auth_token("https://collector.example.com") is None

    def test_clear_nonexistent_returns_false(self, tmp_path, monkeypatch):
        auth_file = tmp_path / "auth.json"
        monkeypatch.setattr("agent_trace.sso._AUTH_FILE", auth_file)
        assert clear_auth_token("https://nobody.example.com") is False

    def test_multiple_servers(self, tmp_path, monkeypatch):
        auth_file = tmp_path / "auth.json"
        monkeypatch.setattr("agent_trace.sso._AUTH_FILE", auth_file)
        save_auth_token("https://a.example.com", "tok_a", time.time() + 3600)
        save_auth_token("https://b.example.com", "tok_b", time.time() + 3600)
        assert load_auth_token("https://a.example.com") == "tok_a"
        assert load_auth_token("https://b.example.com") == "tok_b"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _run_auth(argv_dict: dict, tmp_path, stdin_input: str = "") -> tuple[int, str]:
    import argparse
    ns = argparse.Namespace(**argv_dict)
    buf = StringIO()
    old_out = sys.stdout
    sys.stdout = buf
    try:
        if stdin_input:
            with patch("builtins.input", return_value=stdin_input):
                rc = cmd_auth(ns)
        else:
            rc = cmd_auth(ns)
    finally:
        sys.stdout = old_out
    return rc, buf.getvalue()


class TestCLIAuth:
    def test_logout_no_token(self, tmp_path, monkeypatch):
        auth_file = tmp_path / "auth.json"
        monkeypatch.setattr("agent_trace.sso._AUTH_FILE", auth_file)
        rc, out = _run_auth({"auth_cmd": "logout", "server": "https://c.example.com"},
                            tmp_path)
        assert rc == 0
        assert "No stored token" in out

    def test_logout_removes_token(self, tmp_path, monkeypatch):
        auth_file = tmp_path / "auth.json"
        monkeypatch.setattr("agent_trace.sso._AUTH_FILE", auth_file)
        save_auth_token("https://c.example.com", "tok", time.time() + 3600)
        rc, out = _run_auth({"auth_cmd": "logout", "server": "https://c.example.com"},
                            tmp_path)
        assert rc == 0
        assert "Logged out" in out

    def test_status_no_tokens(self, tmp_path, monkeypatch):
        auth_file = tmp_path / "auth.json"
        monkeypatch.setattr("agent_trace.sso._AUTH_FILE", auth_file)
        rc, out = _run_auth({"auth_cmd": "status", "server": ""}, tmp_path)
        assert rc == 0
        assert "No stored tokens" in out

    def test_status_specific_server_logged_in(self, tmp_path, monkeypatch):
        auth_file = tmp_path / "auth.json"
        monkeypatch.setattr("agent_trace.sso._AUTH_FILE", auth_file)
        save_auth_token("https://c.example.com", "tok", time.time() + 3600)
        rc, out = _run_auth({"auth_cmd": "status", "server": "https://c.example.com"},
                            tmp_path)
        assert rc == 0
        assert "Logged in" in out

    def test_status_specific_server_not_logged_in(self, tmp_path, monkeypatch):
        auth_file = tmp_path / "auth.json"
        monkeypatch.setattr("agent_trace.sso._AUTH_FILE", auth_file)
        rc, out = _run_auth({"auth_cmd": "status", "server": "https://c.example.com"},
                            tmp_path)
        assert rc == 0
        assert "Not logged in" in out

    def test_login_stores_token(self, tmp_path, monkeypatch):
        auth_file = tmp_path / "auth.json"
        monkeypatch.setattr("agent_trace.sso._AUTH_FILE", auth_file)
        rc, out = _run_auth(
            {"auth_cmd": "login", "server": "https://c.example.com", "force": False},
            tmp_path,
            stdin_input="mytoken123",
        )
        assert rc == 0
        assert "Logged in" in out
        assert load_auth_token("https://c.example.com") == "mytoken123"

    def test_login_already_logged_in(self, tmp_path, monkeypatch):
        auth_file = tmp_path / "auth.json"
        monkeypatch.setattr("agent_trace.sso._AUTH_FILE", auth_file)
        save_auth_token("https://c.example.com", "existing", time.time() + 3600)
        rc, out = _run_auth(
            {"auth_cmd": "login", "server": "https://c.example.com", "force": False},
            tmp_path,
        )
        assert rc == 0
        assert "Already logged in" in out

    def test_no_subcommand_returns_error(self, tmp_path):
        rc, _ = _run_auth({"auth_cmd": None}, tmp_path)
        assert rc == 1
