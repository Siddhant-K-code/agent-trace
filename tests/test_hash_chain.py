"""Tests for tamper-evident hash chain audit (issue #143)."""

import hashlib
import json
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.audit import verify_chain, ChainVerifyResult
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore


def _make_session(store: TraceStore, n_events: int = 3) -> str:
    meta = SessionMeta(agent_name="test")
    meta.started_at = time.time() - 60
    meta.ended_at = time.time()
    store.create_session(meta)
    for i in range(n_events):
        store.append_event(meta.session_id, TraceEvent(
            event_type=EventType.TOOL_CALL,
            session_id=meta.session_id,
            data={"tool_name": f"tool-{i}"},
        ))
    store.update_meta(meta)
    return meta.session_id


class TestHashChainAppend(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = TraceStore(self.tmpdir)

    def test_first_event_has_empty_prev_hash(self):
        sid = _make_session(self.store, n_events=1)
        events = self.store.load_events(sid)
        self.assertEqual(events[0].prev_hash, "")

    def test_second_event_has_prev_hash(self):
        sid = _make_session(self.store, n_events=2)
        events = self.store.load_events(sid)
        self.assertNotEqual(events[1].prev_hash, "")

    def test_prev_hash_is_sha256_hex(self):
        sid = _make_session(self.store, n_events=2)
        events = self.store.load_events(sid)
        h = events[1].prev_hash
        self.assertEqual(len(h), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in h))

    def test_prev_hash_matches_sha256_of_previous_line(self):
        sid = _make_session(self.store, n_events=3)
        events_path = self.store._session_dir(sid) / "events.ndjson"
        lines = [l for l in events_path.read_text().splitlines() if l.strip()]
        # Check event[2].prev_hash == sha256(lines[1])
        expected = hashlib.sha256(lines[1].encode()).hexdigest()
        obj = json.loads(lines[2])
        self.assertEqual(obj["prev_hash"], expected)


class TestVerifyChain(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = TraceStore(self.tmpdir)

    def test_intact_chain_returns_ok(self):
        sid = _make_session(self.store, n_events=5)
        result = verify_chain(self.store, sid)
        self.assertTrue(result.ok)
        self.assertEqual(result.total_events, 5)

    def test_empty_session_returns_ok(self):
        meta = SessionMeta(agent_name="empty")
        self.store.create_session(meta)
        result = verify_chain(self.store, meta.session_id)
        self.assertTrue(result.ok)
        self.assertEqual(result.total_events, 0)

    def test_tampered_event_detected(self):
        sid = _make_session(self.store, n_events=4)
        events_path = self.store._session_dir(sid) / "events.ndjson"
        lines = events_path.read_text().splitlines()

        # Tamper: modify the second event's data
        obj = json.loads(lines[1])
        obj["data"]["tool_name"] = "TAMPERED"
        lines[1] = json.dumps(obj)
        events_path.write_text("\n".join(lines) + "\n")

        result = verify_chain(self.store, sid)
        self.assertFalse(result.ok)
        self.assertIsNotNone(result.broken_at)
        self.assertGreater(result.broken_at, 0)

    def test_inserted_event_detected(self):
        sid = _make_session(self.store, n_events=3)
        events_path = self.store._session_dir(sid) / "events.ndjson"
        lines = events_path.read_text().splitlines()

        # Insert a fake event between lines[0] and lines[1]
        fake = json.loads(lines[0])
        fake["event_id"] = "fakeevent00"
        fake["prev_hash"] = ""  # wrong hash
        lines.insert(1, json.dumps(fake))
        events_path.write_text("\n".join(lines) + "\n")

        result = verify_chain(self.store, sid)
        self.assertFalse(result.ok)

    def test_pre_v062_events_without_hash_skipped(self):
        """Events without prev_hash (legacy) must not cause a false failure."""
        meta = SessionMeta(agent_name="legacy")
        self.store.create_session(meta)
        # Write events manually without prev_hash
        events_path = self.store._session_dir(meta.session_id) / "events.ndjson"
        for i in range(3):
            obj = {"event_type": "tool_call", "event_id": f"ev{i:04d}",
                   "timestamp": time.time(), "data": {}}
            events_path.open("a").write(json.dumps(obj) + "\n")

        result = verify_chain(self.store, meta.session_id)
        self.assertTrue(result.ok)

    def test_result_format_ok(self):
        import io
        sid = _make_session(self.store, n_events=2)
        result = verify_chain(self.store, sid)
        buf = io.StringIO()
        result.format(buf)
        self.assertIn("intact", buf.getvalue())

    def test_result_format_broken(self):
        import io
        result = ChainVerifyResult(
            session_id="abc", ok=False, total_events=5,
            broken_at=2, broken_event_id="ev0002"
        )
        buf = io.StringIO()
        result.format(buf)
        self.assertIn("broken", buf.getvalue())
        self.assertIn("ev0002", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
