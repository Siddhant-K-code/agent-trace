"""Tests for human-in-the-loop approval workflow (issue #137)."""

import json
import os
import sys
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace import approval as approval_mod
from agent_trace.approval import (
    APPROVAL_STATES,
    APPROVED,
    DECIDED_STATES,
    DENIED,
    PENDING,
    ApprovalRequest,
    can_transition,
    create_approval_request,
    approve_request,
    deny_request,
    list_requests,
    load_request,
    find_request,
    save_request,
    poll_for_decision,
    wait_for_decision,
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

    def test_raises_when_request_missing(self):
        start = time.monotonic()
        with self.assertRaises(LookupError):
            poll_for_decision(self.store, "nonexistent",
                              timeout=5.0, poll_interval=0.01)
        self.assertLess(time.monotonic() - start, 1.0)

    def test_raises_when_state_invalid(self):
        req = create_approval_request(self.store, "s1", "rule")
        path = approval_mod._request_path(self.store, req.request_id)
        data = json.loads(path.read_text())
        data["state"] = "maybe"
        path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            poll_for_decision(self.store, req.request_id,
                              timeout=5.0, poll_interval=0.01)


class TestWaitForDecision(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = TraceStore(self.tmpdir)

    def test_timeout_is_distinct_from_decision(self):
        req = create_approval_request(self.store, "s1", "rule")
        result = wait_for_decision(self.store, req.request_id,
                                   timeout=0.05, poll_interval=0.01)
        self.assertEqual(result.outcome, "timeout")
        self.assertTrue(result.timed_out)
        self.assertFalse(result.decided)
        self.assertEqual(result.request.state, PENDING)

    def test_returns_decision(self):
        req = create_approval_request(self.store, "s1", "rule")
        deny_request(self.store, req.request_id, kill_agent=False)
        result = wait_for_decision(self.store, req.request_id,
                                   timeout=1.0, poll_interval=0.01)
        self.assertEqual(result.outcome, DENIED)
        self.assertTrue(result.decided)
        self.assertFalse(result.timed_out)
        self.assertEqual(result.request.request_id, req.request_id)

    def test_sees_decision_made_while_waiting(self):
        req = create_approval_request(self.store, "s1", "rule")
        timer = threading.Timer(
            0.05, approve_request, args=(self.store, req.request_id),
            kwargs={"resume_agent": False},
        )
        timer.start()
        try:
            result = wait_for_decision(self.store, req.request_id,
                                       timeout=5.0, poll_interval=0.01)
        finally:
            timer.join()
        self.assertEqual(result.outcome, APPROVED)

    def test_zero_timeout_checks_once(self):
        req = create_approval_request(self.store, "s1", "rule")
        approve_request(self.store, req.request_id, resume_agent=False)
        result = wait_for_decision(self.store, req.request_id, timeout=0)
        self.assertEqual(result.outcome, APPROVED)

    def test_missing_request_returns_not_found_immediately(self):
        start = time.monotonic()
        result = wait_for_decision(self.store, "nonexistent",
                                   timeout=5.0, poll_interval=0.01)
        self.assertEqual(result.outcome, "not_found")
        self.assertIsNone(result.request)
        self.assertLess(time.monotonic() - start, 1.0)

    def test_invalid_state_returns_invalid(self):
        req = create_approval_request(self.store, "s1", "rule")
        path = approval_mod._request_path(self.store, req.request_id)
        data = json.loads(path.read_text())
        data["state"] = "approved-ish"
        path.write_text(json.dumps(data))
        result = wait_for_decision(self.store, req.request_id,
                                   timeout=5.0, poll_interval=0.01)
        self.assertEqual(result.outcome, "invalid")
        self.assertFalse(result.decided)


class TestStateDefinition(unittest.TestCase):
    def test_states(self):
        self.assertEqual(APPROVAL_STATES, {"pending", "approved", "denied"})
        self.assertEqual(DECIDED_STATES, {"approved", "denied"})

    def test_transitions(self):
        self.assertTrue(can_transition(PENDING, APPROVED))
        self.assertTrue(can_transition(PENDING, DENIED))
        for decided in DECIDED_STATES:
            for target in APPROVAL_STATES:
                self.assertFalse(can_transition(decided, target))
        self.assertFalse(can_transition(PENDING, PENDING))
        self.assertFalse(can_transition("bogus", APPROVED))

    def test_from_json_rejects_unknown_state(self):
        payload = json.loads(ApprovalRequest().to_json())
        for bad in ("maybe", "", None, ["approved"], 1):
            payload["state"] = bad
            with self.assertRaises(ValueError, msg=repr(bad)):
                ApprovalRequest.from_json(json.dumps(payload))

    def test_from_json_rejects_non_object(self):
        with self.assertRaises(ValueError):
            ApprovalRequest.from_json("[]")

    def test_from_json_reads_existing_records(self):
        record = {
            "request_id": "abc123", "session_id": "s1", "rule_name": "r",
            "tool_name": "", "tool_input": {}, "event_id": "", "agent_pid": 0,
            "state": "approved", "created_at": 1.0, "decided_at": 2.0,
            "decided_by": "alice", "reason": "", "future_field": "ignored",
        }
        req = ApprovalRequest.from_json(json.dumps(record))
        self.assertEqual(req.state, APPROVED)
        self.assertEqual(req.decided_by, "alice")

    def test_save_rejects_unknown_state(self):
        store = TraceStore(tempfile.mkdtemp())
        with self.assertRaises(ValueError):
            save_request(store, ApprovalRequest(state="maybe"))

    def test_list_skips_invalid_records(self):
        store = TraceStore(tempfile.mkdtemp())
        good = create_approval_request(store, "s1", "rule")
        bad = create_approval_request(store, "s2", "rule")
        path = approval_mod._request_path(store, bad.request_id)
        data = json.loads(path.read_text())
        data["state"] = "maybe"
        path.write_text(json.dumps(data))
        ids = [r.request_id for r in list_requests(store)]
        self.assertEqual(ids, [good.request_id])
        self.assertIsNone(load_request(store, bad.request_id))

    def test_decide_on_invalid_record_reports_invalid(self):
        store = TraceStore(tempfile.mkdtemp())
        req = create_approval_request(store, "s1", "rule")
        path = approval_mod._request_path(store, req.request_id)
        data = json.loads(path.read_text())
        data["state"] = "maybe"
        path.write_text(json.dumps(data))
        ok, msg = approve_request(store, req.request_id, resume_agent=False)
        self.assertFalse(ok)
        self.assertIn("invalid", msg)
        self.assertEqual(json.loads(path.read_text())["state"], "maybe")

    def test_decide_ignores_matches_outside_approvals_dir(self):
        base = tempfile.mkdtemp()
        store = TraceStore(base)
        create_approval_request(store, "s1", "rule")
        outside = ApprovalRequest(request_id="escape")
        outside_path = os.path.join(base, "escape.json")
        with open(outside_path, "w") as f:
            f.write(outside.to_json())
        ok, msg = approve_request(store, "../escape", resume_agent=False)
        self.assertFalse(ok)
        self.assertIn("not found", msg)
        with open(outside_path) as f:
            self.assertEqual(json.load(f)["state"], PENDING)

    def test_save_is_atomic_and_leaves_no_temp_files(self):
        store = TraceStore(tempfile.mkdtemp())
        req = create_approval_request(store, "s1", "rule")
        approve_request(store, req.request_id, resume_agent=False)
        names = sorted(p.name for p in approval_mod._approvals_dir(store).iterdir())
        self.assertEqual(names, [f"{req.request_id}.json"])


_RACE_SCRIPT = """
import sys
sys.path.insert(0, sys.argv[1])
from agent_trace.approval import approve_request, deny_request
from agent_trace.store import TraceStore
store = TraceStore(sys.argv[2])
if sys.argv[4] == "approve":
    ok, msg = approve_request(store, sys.argv[3], resume_agent=False)
else:
    ok, msg = deny_request(store, sys.argv[3], kill_agent=False)
print("OK" if ok else msg)
"""


class TestConcurrentDecisions(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = TraceStore(self.tmpdir)

    def _race_threads(self, n: int = 6) -> tuple[list, str]:
        req = create_approval_request(self.store, "s1", "rule")
        barrier = threading.Barrier(n)
        results: list = [None] * n

        def decide(i: int) -> None:
            barrier.wait()
            if i % 2:
                results[i] = approve_request(
                    self.store, req.request_id, decided_by=f"u{i}",
                    resume_agent=False)
            else:
                results[i] = deny_request(
                    self.store, req.request_id, decided_by=f"u{i}",
                    kill_agent=False)

        threads = [threading.Thread(target=decide, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return results, req.request_id

    def test_exactly_one_thread_wins(self):
        real_write = approval_mod._write_atomic

        def slow_write(path, text):
            # Widen the read-check-write window so an unlocked
            # implementation would let several deciders through.
            time.sleep(0.02)
            real_write(path, text)

        with patch.object(approval_mod, "_write_atomic", side_effect=slow_write):
            results, request_id = self._race_threads()

        winners = [r for r in results if r[0]]
        self.assertEqual(len(winners), 1, results)
        final = load_request(self.store, request_id)
        self.assertIn(final.state, DECIDED_STATES)
        expected_win = "Approved" if final.state == APPROVED else "Denied"
        self.assertTrue(winners[0][1].startswith(expected_win))
        for ok, msg in results:
            if not ok:
                self.assertEqual(msg, f"Request already {final.state}")

    def test_exactly_one_process_wins(self):
        req = create_approval_request(self.store, "s1", "rule")
        src = os.path.join(os.path.dirname(__file__), "..", "src")
        procs = [
            subprocess.Popen(
                [sys.executable, "-c", _RACE_SCRIPT, src, self.tmpdir,
                 req.request_id, "approve" if i % 2 else "deny"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            for i in range(4)
        ]
        outputs = []
        for p in procs:
            out, err = p.communicate(timeout=30)
            self.assertEqual(p.returncode, 0, err)
            outputs.append(out.strip())
        final = load_request(self.store, req.request_id)
        self.assertEqual(outputs.count("OK"), 1, outputs)
        self.assertEqual(
            sorted(o for o in outputs if o != "OK"),
            [f"Request already {final.state}"] * 3,
        )


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
