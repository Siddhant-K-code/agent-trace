"""Tests for machine identity and session signing (issue #141)."""

import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.identity import (
    AgentIdentity,
    load_or_create_identity,
    load_identity,
    sign_session,
    verify_session,
    _sign_payload,
    _IDENTITY_FILE_ENV,
)
from agent_trace.models import SessionMeta, TraceEvent
from agent_trace.models import EventType
from agent_trace.store import TraceStore


def _make_store_with_session(tmpdir: str) -> tuple[TraceStore, str]:
    store = TraceStore(tmpdir)
    meta = SessionMeta(agent_name="test-agent")
    meta.started_at = time.time() - 60
    meta.ended_at = time.time()
    meta.tool_calls = 3
    meta.errors = 0
    store.create_session(meta)
    store.update_meta(meta)
    return store, meta.session_id


class TestAgentIdentity(unittest.TestCase):
    def test_new_identity_has_unique_id(self):
        a = AgentIdentity()
        b = AgentIdentity()
        self.assertNotEqual(a.identity_id, b.identity_id)

    def test_key_bytes_is_32_bytes(self):
        identity = AgentIdentity()
        self.assertEqual(len(identity.key_bytes), 32)

    def test_to_public_dict_excludes_key(self):
        identity = AgentIdentity(agent_name="my-agent")
        pub = identity.to_public_dict()
        self.assertIn("identity_id", pub)
        self.assertIn("agent_name", pub)
        self.assertNotIn("_key_hex", pub)
        self.assertNotIn("key_hex", pub)

    def test_json_roundtrip(self):
        identity = AgentIdentity(agent_name="roundtrip-agent")
        loaded = AgentIdentity.from_json(identity.to_json())
        self.assertEqual(loaded.identity_id, identity.identity_id)
        self.assertEqual(loaded.agent_name, identity.agent_name)
        self.assertEqual(loaded._key_hex, identity._key_hex)


class TestIdentityPersistence(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._orig_env = os.environ.get(_IDENTITY_FILE_ENV)
        os.environ[_IDENTITY_FILE_ENV] = os.path.join(self.tmpdir, "identity.json")

    def tearDown(self):
        if self._orig_env is None:
            os.environ.pop(_IDENTITY_FILE_ENV, None)
        else:
            os.environ[_IDENTITY_FILE_ENV] = self._orig_env

    def test_creates_identity_on_first_call(self):
        identity = load_or_create_identity("my-agent")
        self.assertIsNotNone(identity)
        self.assertEqual(identity.agent_name, "my-agent")
        path = os.environ[_IDENTITY_FILE_ENV]
        self.assertTrue(os.path.exists(path))

    def test_loads_same_identity_on_second_call(self):
        first = load_or_create_identity("agent-a")
        second = load_or_create_identity("agent-a")
        self.assertEqual(first.identity_id, second.identity_id)
        self.assertEqual(first._key_hex, second._key_hex)

    def test_load_identity_returns_none_when_missing(self):
        result = load_identity()
        self.assertIsNone(result)


class TestSignAndVerify(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_sign_and_verify_succeeds(self):
        store, sid = _make_store_with_session(self.tmpdir)
        identity = AgentIdentity(agent_name="signer")
        ok = sign_session(store, sid, identity)
        self.assertTrue(ok)
        verified, reason = verify_session(store, sid, identity)
        self.assertTrue(verified)
        self.assertEqual(reason, "ok")

    def test_verify_fails_with_wrong_identity(self):
        store, sid = _make_store_with_session(self.tmpdir)
        signer = AgentIdentity(agent_name="signer")
        verifier = AgentIdentity(agent_name="other")
        sign_session(store, sid, signer)
        ok, reason = verify_session(store, sid, verifier)
        self.assertFalse(ok)
        self.assertIn("identity mismatch", reason)

    def test_verify_fails_on_unsigned_session(self):
        store, sid = _make_store_with_session(self.tmpdir)
        identity = AgentIdentity()
        ok, reason = verify_session(store, sid, identity)
        self.assertFalse(ok)
        self.assertIn("no signature", reason)

    def test_verify_fails_after_metadata_tampered(self):
        store, sid = _make_store_with_session(self.tmpdir)
        identity = AgentIdentity(agent_name="signer")
        sign_session(store, sid, identity)

        # Tamper: change tool_calls after signing
        meta = store.load_meta(sid)
        meta.tool_calls = 999
        store.update_meta(meta)

        ok, reason = verify_session(store, sid, identity)
        self.assertFalse(ok)
        self.assertIn("signature mismatch", reason)

    def test_sign_nonexistent_session_returns_false(self):
        store = TraceStore(self.tmpdir)
        identity = AgentIdentity()
        # Use a valid hex session_id format that simply doesn't exist on disk
        ok = sign_session(store, "deadbeefdeadbeef", identity)
        self.assertFalse(ok)

    def test_hmac_is_deterministic(self):
        identity = AgentIdentity()
        payload = {"session_id": "abc", "tool_calls": 5}
        sig1 = _sign_payload(identity, payload)
        sig2 = _sign_payload(identity, payload)
        self.assertEqual(sig1, sig2)

    def test_hmac_differs_for_different_payloads(self):
        identity = AgentIdentity()
        sig1 = _sign_payload(identity, {"tool_calls": 5})
        sig2 = _sign_payload(identity, {"tool_calls": 6})
        self.assertNotEqual(sig1, sig2)


if __name__ == "__main__":
    unittest.main()
