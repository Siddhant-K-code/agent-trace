"""Tests for human-in-the-loop approval workflow (issue #137)."""

import json
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.approval import (
    ApprovalRequest,
    create_approval_request,
    approve_request,
    deny_request,
    list_requests,
    load_request,
    find_request,
    save_request,
    poll_for_decision,
)
from agent_trace.store import TraceStore


class TestApprovalRequest(unittest.TestCase):
    def test_new_request_is_pending(self):
        req = ApprovalRequest()
        self.assertEqual(req.state, "pending")

    def test_unique_ids(self):
        a = ApprovalRequest()
        b = ApprovalRequest()
        self.assertNotEqual(a.request_id, b.request_id)

    def test_json_roundtrip(self):
        req = ApprovalRequest(session_id="sess1", rule_name="no-network",
                              tool_name="Bash", agent_pid=1234)
        loaded = ApprovalRequest.from_json(req.to_json())
        self.assertEqual(loaded.request_id, req.request_id)
        self.assertEqual(loaded.session_id, req.session_id)
        self.assertEqual(loaded.rule_name, req.rule_name)
        self.assertEqual(loaded.agent_pid, req.agent_pid)


class TestApprovalStorage(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = TraceStore(self.tmpdir)

    def test_save_and_load(self):
        req = ApprovalRequest(session_id="s1", rule_name="test-rule")
        save_request(self.store, req)
        loaded = load_request(self.store, req.request_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.request_id, req.request_id)

    def test_load_missing_returns_none(self):
        result = load_request(self.store, "nonexistent")
        self.assertIsNone(result)

    def test_find_by_prefix(self):
        req = ApprovalRequest(session_id="s1")
        save_request(self.store, req)
        found = find_request(self.store, req.request_id[:4])
        self.assertIsNotNone(found)
        self.assertEqual(found.request_id, req.request_id)

    def test_list_all(self):
        for i in range(3):
            save_request(self.store, ApprovalRequest(session_id=f"s{i}"))
        requests = list_requests(self.store)
        self.assertEqual(len(requests), 3)

    def test_list_filtered_by_state(self):
        r1 = ApprovalRequest(session_id="s1", state="pending")
        r2 = ApprovalRequest(session_id="s2", state="approved")
        r3 = ApprovalRequest(session_id="s3", state="denied")
        for r in (r1, r2, r3):
            save_request(self.store, r)
        pending = list_requests(self.store, state="pending")
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].session_id, "s1")

    def test_list_empty_store(self):
        self.assertEqual(list_requests(self.store), [])


class TestApproveRequest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = TraceStore(self.tmpdir)

    def test_approve_pending_request(self):
        req = create_approval_request(self.store, "sess1", "rule-a",
                                      agent_pid=0)
        ok, msg = approve_request(self.store, req.request_id,
                                  decided_by="alice", resume_agent=False)
        self.assertTrue(ok)
        updated = load_request(self.store, req.request_id)
        self.assertEqual(updated.state, "approved")
        self.assertEqual(updated.decided_by, "alice")
        self.assertIsNotNone(updated.decided_at)

    def test_approve_nonexistent_returns_false(self):
        ok, msg = approve_request(self.store, "nonexistent", resume_agent=False)
        self.assertFalse(ok)
        self.assertIn("not found", msg)

    def test_approve_already_decided_returns_false(self):
        req = ApprovalRequest(state="denied")
        save_request(self.store, req)
        ok, msg = approve_request(self.store, req.request_id, resume_agent=False)
        self.assertFalse(ok)
        self.assertIn("already", msg)

    def test_approve_sends_sigcont(self):
        req = create_approval_request(self.store, "s1", "rule", agent_pid=99999)
        with patch("os.kill") as mock_kill:
            approve_request(self.store, req.request_id, resume_agent=True)
        import signal
        mock_kill.assert_called_once_with(99999, signal.SIGCONT)

    def test_approve_no_resume_skips_sigcont(self):
        req = create_approval_request(self.store, "s1", "rule", agent_pid=99999)
        with patch("os.kill") as mock_kill:
            approve_request(self.store, req.request_id, resume_agent=False)
        mock_kill.assert_not_called()


class TestDenyRequest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = TraceStore(self.tmpdir)

    def test_deny_pending_request(self):
        req = create_approval_request(self.store, "sess1", "rule-b",
                                      agent_pid=0)
        ok, msg = deny_request(self.store, req.request_id,
                               reason="too risky", decided_by="bob",
                               kill_agent=False)
        self.assertTrue(ok)
        updated = load_request(self.store, req.request_id)
        self.assertEqual(updated.state, "denied")
        self.assertEqual(updated.reason, "too risky")
        self.assertEqual(updated.decided_by, "bob")

    def test_deny_sends_sigterm(self):
        req = create_approval_request(self.store, "s1", "rule", agent_pid=99999)
        with patch("os.kill") as mock_kill:
            deny_request(self.store, req.request_id, kill_agent=True)
        import signal
        mock_kill.assert_called_once_with(99999, signal.SIGTERM)

    def test_deny_no_kill_skips_sigterm(self):
        req = create_approval_request(self.store, "s1", "rule", agent_pid=99999)
        with patch("os.kill") as mock_kill:
            deny_request(self.store, req.request_id, kill_agent=False)
        mock_kill.assert_not_called()

    def test_deny_nonexistent_returns_false(self):
        ok, msg = deny_request(self.store, "nonexistent", kill_agent=False)
        self.assertFalse(ok)


class TestPollForDecision(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = TraceStore(self.tmpdir)

    def test_returns_approved_when_approved(self):
        req = create_approval_request(self.store, "s1", "rule", agent_pid=0)
        approve_request(self.store, req.request_id, resume_agent=False)
        result = poll_for_decision(self.store, req.request_id,
                                   timeout=1.0, poll_interval=0.01)
        self.assertEqual(result, "approved")

    def test_returns_denied_when_denied(self):
        req = create_approval_request(self.store, "s1", "rule", agent_pid=0)
        deny_request(self.store, req.request_id, kill_agent=False)
        result = poll_for_decision(self.store, req.request_id,
                                   timeout=1.0, poll_interval=0.01)
        self.assertEqual(result, "denied")

    def test_returns_pending_on_timeout(self):
        req = create_approval_request(self.store, "s1", "rule", agent_pid=0)
        result = poll_for_decision(self.store, req.request_id,
                                   timeout=0.05, poll_interval=0.01)
        self.assertEqual(result, "pending")


class TestCreateApprovalRequest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = TraceStore(self.tmpdir)

    def test_creates_pending_request(self):
        req = create_approval_request(
            self.store, "sess1", "no-network",
            tool_name="Bash", tool_input={"command": "curl http://example.com"},
            agent_pid=1234,
        )
        self.assertEqual(req.state, "pending")
        self.assertEqual(req.tool_name, "Bash")
        self.assertEqual(req.agent_pid, 1234)

    def test_persisted_to_disk(self):
        req = create_approval_request(self.store, "s1", "rule")
        loaded = load_request(self.store, req.request_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.request_id, req.request_id)


if __name__ == "__main__":
    unittest.main()
