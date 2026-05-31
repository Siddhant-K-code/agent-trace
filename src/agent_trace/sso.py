"""SSO (OIDC) support for agent-trace server and CLI.

Provides:
- OIDC discovery and token validation (optional dep: authlib or PyJWT)
- Short-lived session cookie issuance after successful IdP auth
- CLI ``agent-strace auth login`` stores token in ~/.agent-strace/auth.json
- ``--auth oidc`` flag on ``agent-strace server`` enables SSO enforcement
- ``--enforce-sso`` disables API key fallback

OIDC flow:
    1. Client hits a protected endpoint → 302 to /auth/login
    2. /auth/login → 302 to IdP authorization URL
    3. IdP → 302 to /auth/callback?code=...
    4. /auth/callback exchanges code for tokens, validates, issues session cookie
    5. Subsequent requests carry the session cookie

Token validation uses the IdP's JWKS endpoint. Requires ``authlib`` or
``PyJWT`` as an optional extra:

    pip install agent-trace[oidc]

Without the optional dep, the server still starts but token signature
verification is skipped (suitable for development/testing only).

CLI token storage:
    ~/.agent-strace/auth.json  — {token, server, expires_at}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_AUTH_FILE = Path.home() / ".agent-strace" / "auth.json"
_SESSION_COOKIE = "at_session"
_SESSION_TTL = 3600  # 1 hour


# ---------------------------------------------------------------------------
# OIDC discovery
# ---------------------------------------------------------------------------

def discover_oidc(issuer: str) -> dict:
    """Fetch OIDC discovery document from ``{issuer}/.well-known/openid-configuration``."""
    url = issuer.rstrip("/") + "/.well-known/openid-configuration"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.loads(r.read())
    except Exception as exc:
        raise RuntimeError(f"OIDC discovery failed for {issuer}: {exc}") from exc


def build_auth_url(
    discovery: dict,
    client_id: str,
    redirect_uri: str,
    state: str,
    nonce: str,
    scopes: list[str] | None = None,
) -> str:
    """Build the IdP authorization URL for the OIDC code flow."""
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": " ".join(scopes or ["openid", "email", "profile"]),
        "state": state,
        "nonce": nonce,
    }
    return discovery["authorization_endpoint"] + "?" + urllib.parse.urlencode(params)


def exchange_code(
    discovery: dict,
    client_id: str,
    client_secret: str,
    code: str,
    redirect_uri: str,
) -> dict:
    """Exchange an authorization code for tokens. Returns the token response dict."""
    data = urllib.parse.urlencode({
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "client_secret": client_secret,
    }).encode()
    req = urllib.request.Request(
        discovery["token_endpoint"],
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"Token exchange failed: {exc.code} {body}") from exc


def decode_id_token_claims(id_token: str) -> dict:
    """Decode JWT claims from an id_token without signature verification.

    For production use, install ``authlib`` or ``PyJWT`` and call
    ``verify_id_token()`` instead.
    """
    parts = id_token.split(".")
    if len(parts) != 3:
        raise ValueError("Invalid JWT format")
    # Base64url decode the payload (add padding as needed)
    payload = parts[1]
    payload += "=" * (4 - len(payload) % 4)
    import base64
    decoded = base64.urlsafe_b64decode(payload)
    return json.loads(decoded)


def verify_id_token(id_token: str, discovery: dict, client_id: str) -> dict:
    """Verify id_token signature using authlib or PyJWT (optional deps).

    Falls back to unverified decode with a warning if neither is installed.
    Returns the claims dict.
    """
    # Try authlib first
    try:
        from authlib.jose import JsonWebToken, KeySet  # type: ignore
        import urllib.request as _ur
        jwks_data = json.loads(_ur.urlopen(discovery["jwks_uri"], timeout=10).read())
        jwt = JsonWebToken(["RS256", "ES256"])
        key_set = KeySet.import_key_set(jwks_data)
        claims = jwt.decode(id_token, key_set)
        claims.validate()
        return dict(claims)
    except ImportError:
        pass

    # Try PyJWT
    try:
        import jwt as pyjwt  # type: ignore
        import urllib.request as _ur
        jwks_data = json.loads(_ur.urlopen(discovery["jwks_uri"], timeout=10).read())
        jwks_client = pyjwt.PyJWKClient(discovery["jwks_uri"])
        signing_key = jwks_client.get_signing_key_from_jwt(id_token)
        return pyjwt.decode(id_token, signing_key.key,
                            algorithms=["RS256", "ES256"],
                            audience=client_id)
    except ImportError:
        pass

    # No verification library — decode without verification (dev only)
    sys.stderr.write(
        "[sso] WARNING: id_token signature not verified. "
        "Install agent-trace[oidc] for production use.\n"
    )
    return decode_id_token_claims(id_token)


# ---------------------------------------------------------------------------
# Session store (in-memory, keyed by opaque session token)
# ---------------------------------------------------------------------------

class SessionStore:
    """In-memory session store for SSO sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, dict] = {}

    def create(self, claims: dict, ttl: int = _SESSION_TTL) -> str:
        token = secrets.token_hex(32)
        self._sessions[token] = {
            "claims": claims,
            "expires_at": time.time() + ttl,
            "email": claims.get("email", ""),
            "sub": claims.get("sub", ""),
        }
        return token

    def get(self, token: str) -> dict | None:
        session = self._sessions.get(token)
        if session is None:
            return None
        if time.time() > session["expires_at"]:
            del self._sessions[token]
            return None
        return session

    def delete(self, token: str) -> None:
        self._sessions.pop(token, None)

    def purge_expired(self) -> int:
        now = time.time()
        expired = [k for k, v in self._sessions.items() if now > v["expires_at"]]
        for k in expired:
            del self._sessions[k]
        return len(expired)


# ---------------------------------------------------------------------------
# OIDCConfig — parsed from server flags
# ---------------------------------------------------------------------------

@dataclass
class OIDCConfig:
    issuer: str = ""
    client_id: str = ""
    client_secret: str = ""
    redirect_uri: str = ""
    enforce: bool = False          # --enforce-sso: reject API key fallback
    discovery: dict = field(default_factory=dict)

    def is_configured(self) -> bool:
        return bool(self.issuer and self.client_id and self.client_secret)

    def load_discovery(self) -> None:
        if self.issuer and not self.discovery:
            self.discovery = discover_oidc(self.issuer)


# ---------------------------------------------------------------------------
# OIDC middleware helpers (used by server.py handler)
# ---------------------------------------------------------------------------

def parse_session_cookie(cookie_header: str) -> str:
    """Extract the at_session value from a Cookie header."""
    for part in cookie_header.split(";"):
        part = part.strip()
        if part.startswith(f"{_SESSION_COOKIE}="):
            return part[len(f"{_SESSION_COOKIE}="):]
    return ""


def make_session_cookie(token: str, ttl: int = _SESSION_TTL) -> str:
    """Build a Set-Cookie header value for the session token."""
    return (
        f"{_SESSION_COOKIE}={token}; "
        f"HttpOnly; SameSite=Lax; Max-Age={ttl}; Path=/"
    )


def redirect_response(location: str) -> tuple[int, dict[str, str], bytes]:
    """Return (status, headers, body) for a 302 redirect."""
    return 302, {"Location": location}, b""


def login_page_html(auth_url: str) -> str:
    """Minimal login redirect page."""
    return (
        "<!DOCTYPE html><html><head>"
        f'<meta http-equiv="refresh" content="0;url={auth_url}">'
        "</head><body>"
        f'<p>Redirecting to identity provider… <a href="{auth_url}">click here</a></p>'
        "</body></html>"
    )


# ---------------------------------------------------------------------------
# CLI token storage
# ---------------------------------------------------------------------------

def save_auth_token(server: str, token: str, expires_at: float) -> None:
    """Persist a CLI auth token to ~/.agent-strace/auth.json."""
    _AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if _AUTH_FILE.exists():
        try:
            data = json.loads(_AUTH_FILE.read_text())
        except Exception:
            data = {}
    data[server] = {"token": token, "expires_at": expires_at}
    _AUTH_FILE.write_text(json.dumps(data, indent=2))
    _AUTH_FILE.chmod(0o600)


def load_auth_token(server: str) -> str | None:
    """Load a valid CLI auth token for *server*, or None if missing/expired."""
    if not _AUTH_FILE.exists():
        return None
    try:
        data = json.loads(_AUTH_FILE.read_text())
        entry = data.get(server, {})
        if not entry:
            return None
        if time.time() > entry.get("expires_at", 0):
            return None
        return entry.get("token")
    except Exception:
        return None


def clear_auth_token(server: str) -> bool:
    """Remove the stored token for *server*. Returns True if something was removed."""
    if not _AUTH_FILE.exists():
        return False
    try:
        data = json.loads(_AUTH_FILE.read_text())
        if server in data:
            del data[server]
            _AUTH_FILE.write_text(json.dumps(data, indent=2))
            return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def cmd_auth(args: argparse.Namespace) -> int:
    sub = getattr(args, "auth_cmd", None)

    if sub == "login":
        server = getattr(args, "server", "").rstrip("/")
        if not server:
            sys.stderr.write("error: --server required\n")
            return 1

        # Check if already logged in
        existing = load_auth_token(server)
        if existing and not getattr(args, "force", False):
            sys.stdout.write(f"Already logged in to {server}. Use --force to re-authenticate.\n")
            return 0

        # In a real implementation this would open a browser and start a local
        # HTTP server to receive the callback. Here we print the instructions
        # and accept a token pasted from the browser (suitable for headless envs).
        sys.stdout.write(
            f"To authenticate with {server}:\n"
            f"  1. Open {server}/auth/login in your browser\n"
            f"  2. Complete the IdP login\n"
            f"  3. Copy the token from the success page\n"
            f"  4. Paste it here and press Enter: "
        )
        sys.stdout.flush()
        try:
            token = input().strip()
        except (EOFError, KeyboardInterrupt):
            sys.stdout.write("\nAborted.\n")
            return 1

        if not token:
            sys.stderr.write("error: no token provided\n")
            return 1

        expires_at = time.time() + _SESSION_TTL
        save_auth_token(server, token, expires_at)
        sys.stdout.write(f"Logged in to {server}. Token stored in {_AUTH_FILE}\n")
        return 0

    if sub == "logout":
        server = getattr(args, "server", "").rstrip("/")
        if not server:
            sys.stderr.write("error: --server required\n")
            return 1
        removed = clear_auth_token(server)
        if removed:
            sys.stdout.write(f"Logged out from {server}\n")
        else:
            sys.stdout.write(f"No stored token for {server}\n")
        return 0

    if sub == "status":
        server = getattr(args, "server", "").rstrip("/")
        if server:
            token = load_auth_token(server)
            if token:
                sys.stdout.write(f"Logged in to {server}\n")
            else:
                sys.stdout.write(f"Not logged in to {server}\n")
        else:
            if not _AUTH_FILE.exists():
                sys.stdout.write("No stored tokens.\n")
                return 0
            try:
                data = json.loads(_AUTH_FILE.read_text())
                if not data:
                    sys.stdout.write("No stored tokens.\n")
                else:
                    for srv, entry in data.items():
                        expired = time.time() > entry.get("expires_at", 0)
                        status = "expired" if expired else "valid"
                        sys.stdout.write(f"  {srv}: {status}\n")
            except Exception:
                sys.stdout.write("Could not read auth file.\n")
        return 0

    sys.stderr.write("Usage: agent-strace auth <login|logout|status>\n")
    return 1
