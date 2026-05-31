"""Per-agent machine identity with HMAC-signed session tokens.

Generates a stable machine identity (UUID stored in ~/.agent-trace/identity.json)
and signs session metadata with HMAC-SHA256 so that a remote collector or reviewer
can verify that a trace came from a specific named agent instance.

Usage:
    # Generate or show the current machine identity
    agent-strace identity show

    # Sign a session (adds HMAC signature to session meta)
    agent-strace identity sign [session-id]

    # Verify a session's signature
    agent-strace identity verify [session-id]

The identity key is stored at ~/.agent-trace/identity.json (user-scoped).
Override with AGENT_TRACE_IDENTITY_FILE env var.

Zero runtime dependencies — stdlib hmac / hashlib / uuid only.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .store import TraceStore


DEFAULT_IDENTITY_DIR = Path.home() / ".agent-trace"
_IDENTITY_FILE_ENV = "AGENT_TRACE_IDENTITY_FILE"


# ---------------------------------------------------------------------------
# Identity schema
# ---------------------------------------------------------------------------

@dataclass
class AgentIdentity:
    """Stable per-machine identity for an agent instance."""
    identity_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    agent_name: str = ""
    created_at: float = field(default_factory=time.time)
    # HMAC key stored as hex; never transmitted in signatures
    _key_hex: str = field(default_factory=lambda: uuid.uuid4().hex + uuid.uuid4().hex)

    def to_public_dict(self) -> dict:
        """Return the public portion (no key)."""
        return {
            "identity_id": self.identity_id,
            "agent_name": self.agent_name,
            "created_at": self.created_at,
        }

    def to_json(self) -> str:
        d = asdict(self)
        d["key_hex"] = d.pop("_key_hex")
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "AgentIdentity":
        d = json.loads(text)
        key = d.pop("key_hex", d.pop("_key_hex", uuid.uuid4().hex * 2))
        obj = cls(
            identity_id=d.get("identity_id", uuid.uuid4().hex),
            agent_name=d.get("agent_name", ""),
            created_at=d.get("created_at", time.time()),
        )
        obj._key_hex = key
        return obj

    @property
    def key_bytes(self) -> bytes:
        return bytes.fromhex(self._key_hex)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _identity_path() -> Path:
    env = os.environ.get(_IDENTITY_FILE_ENV, "")
    if env:
        return Path(env)
    return DEFAULT_IDENTITY_DIR / "identity.json"


def load_or_create_identity(agent_name: str = "") -> AgentIdentity:
    """Load the machine identity, creating one if it doesn't exist."""
    path = _identity_path()
    if path.exists():
        try:
            identity = AgentIdentity.from_json(path.read_text())
            if agent_name and not identity.agent_name:
                identity.agent_name = agent_name
            return identity
        except Exception:
            pass
    identity = AgentIdentity(agent_name=agent_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(identity.to_json())
    return identity


def load_identity() -> AgentIdentity | None:
    """Load existing identity; return None if not found."""
    path = _identity_path()
    if not path.exists():
        return None
    try:
        return AgentIdentity.from_json(path.read_text())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Signing and verification
# ---------------------------------------------------------------------------

def _sign_payload(identity: AgentIdentity, payload: dict) -> str:
    """Return HMAC-SHA256 hex digest of the canonical JSON payload."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hmac.new(
        identity.key_bytes,
        canonical.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _sig_path(store: TraceStore, session_id: str) -> Path:
    """Return the path to the identity sidecar file for a session."""
    return store.base_dir / session_id / "identity.json"


def _load_sig(store: TraceStore, session_id: str) -> dict | None:
    p = _sig_path(store, session_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _save_sig(store: TraceStore, session_id: str, sig_data: dict) -> None:
    p = _sig_path(store, session_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(sig_data, indent=2))


def sign_session(store: TraceStore, session_id: str,
                 identity: AgentIdentity) -> bool:
    """Add an HMAC signature sidecar to a session. Returns True on success."""
    try:
        meta = store.load_meta(session_id)
    except Exception:
        return False
    if not meta:
        return False

    payload = {
        "session_id": meta.session_id,
        "agent_name": meta.agent_name,
        "started_at": meta.started_at,
        "ended_at": meta.ended_at,
        "tool_calls": meta.tool_calls,
        "errors": meta.errors,
    }
    sig = _sign_payload(identity, payload)
    _save_sig(store, session_id, {
        "identity_id": identity.identity_id,
        "agent_name": identity.agent_name,
        "signature": sig,
        "signed_at": time.time(),
    })
    return True


def verify_session(store: TraceStore, session_id: str,
                   identity: AgentIdentity) -> tuple[bool, str]:
    """Verify a session's HMAC signature. Returns (ok, reason)."""
    try:
        meta = store.load_meta(session_id)
    except Exception:
        return False, "session not found"
    if not meta:
        return False, "session not found"

    stored = _load_sig(store, session_id)
    if not stored:
        return False, "no signature on session"

    if stored.get("identity_id") != identity.identity_id:
        return False, (
            f"identity mismatch: session signed by {stored.get('identity_id')}, "
            f"verifying with {identity.identity_id}"
        )

    payload = {
        "session_id": meta.session_id,
        "agent_name": meta.agent_name,
        "started_at": meta.started_at,
        "ended_at": meta.ended_at,
        "tool_calls": meta.tool_calls,
        "errors": meta.errors,
    }
    expected = _sign_payload(identity, payload)
    actual = stored.get("signature", "")

    if not hmac.compare_digest(expected, actual):
        return False, "signature mismatch — session may have been tampered with"

    return True, "ok"


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def cmd_identity(args: argparse.Namespace) -> int:
    store = TraceStore(args.trace_dir)
    sub = getattr(args, "identity_cmd", None)

    if sub == "show":
        identity = load_identity()
        if not identity:
            sys.stdout.write("No identity found. Run: agent-strace identity show\n")
            sys.stdout.write(f"Identity will be created at: {_identity_path()}\n")
            identity = load_or_create_identity()
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(identity.created_at))
        sys.stdout.write(f"\nAgent identity\n\n")
        sys.stdout.write(f"  ID:         {identity.identity_id}\n")
        sys.stdout.write(f"  Name:       {identity.agent_name or '(unnamed)'}\n")
        sys.stdout.write(f"  Created:    {ts}\n")
        sys.stdout.write(f"  Key file:   {_identity_path()}\n\n")
        return 0

    elif sub == "sign":
        session_id = getattr(args, "session_id", None)
        if not session_id:
            session_id = store.get_latest_session_id()
        if not session_id:
            sys.stderr.write("No sessions found.\n")
            return 1
        full_id = store.find_session(session_id)
        if not full_id:
            sys.stderr.write(f"Session not found: {session_id}\n")
            return 1
        identity = load_or_create_identity(getattr(args, "agent_name", "") or "")
        ok = sign_session(store, full_id, identity)
        if ok:
            sys.stdout.write(f"Signed session {full_id[:12]} with identity {identity.identity_id[:12]}\n")
            return 0
        sys.stderr.write("Failed to sign session.\n")
        return 1

    elif sub == "verify":
        session_id = getattr(args, "session_id", None)
        if not session_id:
            session_id = store.get_latest_session_id()
        if not session_id:
            sys.stderr.write("No sessions found.\n")
            return 1
        full_id = store.find_session(session_id)
        if not full_id:
            sys.stderr.write(f"Session not found: {session_id}\n")
            return 1
        identity = load_identity()
        if not identity:
            sys.stderr.write("No identity found. Run: agent-strace identity show\n")
            return 1
        ok, reason = verify_session(store, full_id, identity)
        if ok:
            sys.stdout.write(f"✅ Session {full_id[:12]} signature verified\n")
            return 0
        sys.stderr.write(f"❌ Verification failed: {reason}\n")
        return 1

    else:
        sys.stderr.write("Usage: agent-strace identity <show|sign|verify>\n")
        return 1
