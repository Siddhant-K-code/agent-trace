"""Tests for human-in-the-loop approval workflow (issue #137)."""

import argparse
import io
import json
import os
import signal
import sys
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import _escapes
except ImportError:
    from tests import _escapes

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


class TestDecisionWithoutAgentPid(unittest.TestCase):
    """A request from watch without --agent-pid records the decision only."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = TraceStore(self.tmpdir)

    def _decide(self, decide, pid):
        req = ApprovalRequest(agent_pid=pid)
        save_request(self.store, req)
        with patch("os.kill") as mock_kill:
            ok, msg = decide(self.store, req.request_id)
        mock_kill.assert_not_called()
        return ok, msg, load_request(self.store, req.request_id)

    def test_approve_and_deny_record_without_signalling(self):
        for decide, state in ((approve_request, APPROVED), (deny_request, DENIED)):
            for pid in (0, None, -1, "1234", True):
                with self.subTest(decide=decide.__name__, pid=pid):
                    ok, msg, updated = self._decide(decide, pid)
                    self.assertTrue(ok)
                    self.assertEqual(updated.state, state)
                    self.assertIn("no agent was signalled", msg)
                    self.assertIn("--agent-pid", msg)
                    self.assertNotIn("agent 0", msg)
                    self.assertNotIn("resumed", msg)
                    self.assertNotIn("terminated", msg)

    def test_cli_exits_zero_and_explains(self):
        import argparse
        import io
        for sub, state in (("approve", APPROVED), ("deny", DENIED)):
            with self.subTest(sub=sub):
                req = create_approval_request(self.store, "s1", "rule", agent_pid=0)
                args = argparse.Namespace(
                    trace_dir=self.tmpdir, approval_cmd=sub,
                    request_id=req.request_id,
                )
                out = io.StringIO()
                with patch("os.kill") as mock_kill, patch("sys.stdout", out):
                    code = approval_mod.cmd_approval(args)
                self.assertEqual(code, 0)
                mock_kill.assert_not_called()
                self.assertIn("no agent was signalled", out.getvalue())
                self.assertNotIn("agent 0", out.getvalue())
                self.assertEqual(load_request(self.store, req.request_id).state, state)

    def test_opt_out_flags_are_reported(self):
        req = create_approval_request(self.store, "s1", "rule", agent_pid=99999)
        with patch("os.kill") as mock_kill:
            ok, msg = approve_request(self.store, req.request_id, resume_agent=False)
        mock_kill.assert_not_called()
        self.assertTrue(ok)
        self.assertIn("agent 99999 not signalled (--no-resume)", msg)
        req = create_approval_request(self.store, "s1", "rule", agent_pid=99999)
        with patch("os.kill") as mock_kill:
            ok, msg = deny_request(self.store, req.request_id, kill_agent=False)
        mock_kill.assert_not_called()
        self.assertIn("agent 99999 not signalled (--no-kill)", msg)

    def test_signal_success_and_failure_messages(self):
        req = create_approval_request(self.store, "s1", "rule", agent_pid=99999)
        with patch("os.kill"):
            ok, msg = approve_request(self.store, req.request_id)
        self.assertTrue(ok)
        self.assertTrue(msg.endswith("agent 99999 resumed"))
        req = create_approval_request(self.store, "s1", "rule", agent_pid=99999)
        with patch("os.kill", side_effect=ProcessLookupError):
            ok, msg = deny_request(self.store, req.request_id)
        self.assertTrue(ok)
        self.assertIn("agent 99999 was not signalled: no such process", msg)
        self.assertEqual(load_request(self.store, req.request_id).state, DENIED)


class TestDecisionWithReusedAgentPid(unittest.TestCase):
    """The agent exited and its PID now belongs to another process."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = TraceStore(self.tmpdir)

    def _request(self):
        return create_approval_request(
            self.store, "s1", "rule", agent_pid=99999, agent_start_time="proc:100",
        )

    def test_approve_and_deny_record_without_signalling(self):
        for decide, state in ((approve_request, APPROVED), (deny_request, DENIED)):
            with self.subTest(decide=decide.__name__):
                req = self._request()
                with patch("agent_trace.approval.pid_reused", return_value=True) as reused, \
                     patch("os.kill") as mock_kill:
                    ok, msg = decide(self.store, req.request_id)
                reused.assert_called_once_with(99999, "proc:100")
                mock_kill.assert_not_called()
                self.assertTrue(ok)
                self.assertIn(
                    "decision recorded, but agent 99999 was not signalled: "
                    "the PID now belongs to another process", msg,
                )
                self.assertNotIn("resumed", msg)
                self.assertNotIn("terminated", msg)
                self.assertEqual(load_request(self.store, req.request_id).state, state)

    def test_cli_exits_zero_and_explains(self):
        import argparse
        import io
        req = self._request()
        args = argparse.Namespace(trace_dir=self.tmpdir, approval_cmd="deny",
                                  request_id=req.request_id)
        out = io.StringIO()
        with patch("agent_trace.approval.pid_reused", return_value=True), \
             patch("os.kill") as mock_kill, patch("sys.stdout", out):
            self.assertEqual(approval_mod.cmd_approval(args), 0)
        mock_kill.assert_not_called()
        self.assertIn("was not signalled", out.getvalue())

    def test_matching_identity_signals(self):
        import signal
        req = self._request()
        with patch("agent_trace.approval.pid_reused", return_value=False), \
             patch("os.kill") as mock_kill:
            ok, msg = deny_request(self.store, req.request_id)
        mock_kill.assert_called_once_with(99999, signal.SIGTERM)
        self.assertTrue(msg.endswith("agent 99999 terminated"))

    def test_opt_out_does_not_look_up_identity(self):
        req = self._request()
        with patch("agent_trace.approval.pid_reused") as reused, patch("os.kill"):
            ok, msg = deny_request(self.store, req.request_id, kill_agent=False)
        reused.assert_not_called()
        self.assertIn("not signalled (--no-kill)", msg)

    def test_identity_is_stored_and_loaded(self):
        req = self._request()
        raw = json.loads((Path(self.tmpdir) / ".approvals" / f"{req.request_id}.json").read_text())
        self.assertEqual(raw["agent_start_time"], "proc:100")
        self.assertEqual(load_request(self.store, req.request_id).agent_start_time, "proc:100")


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
        # Records written before agent_start_time existed load with it empty.
        self.assertEqual(req.agent_start_time, "")

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
        self.assertEqual(msg, "invalid request ID")
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


_real_scandir = os.scandir


class _DescendingScandir:
    """os.scandir that lists entries in descending name order.

    The file system may list names in any order. A fixed order that is not
    sorted shows whether the lookup sorts the matches itself.
    """

    def __init__(self, *args, **kwargs):
        with _real_scandir(*args, **kwargs) as entries:
            self._entries = iter(sorted(entries, key=lambda e: e.name, reverse=True))

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._entries)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def close(self):
        pass


class TestRequestIdLookup(unittest.TestCase):
    """show, approve and deny resolve a request ID or prefix the same way."""

    AMBIGUOUS = "Request ID prefix abc matches 2 requests: abc111111111, abc222222222"

    def setUp(self):
        self.trace_dir = self.new_trace_dir()
        self.store = TraceStore(self.trace_dir)

    def new_trace_dir(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return tmp.name

    def save(self, request_id, pid=0, store=None):
        save_request(store or self.store,
                     ApprovalRequest(request_id=request_id, agent_pid=pid))

    def save_three(self):
        self.save("abc111111111", 1111)
        self.save("abc222222222", 2222)
        self.save("def333333333", 3333)

    def set_state(self, request_id, state):
        path = approval_mod._request_path(self.store, request_id)
        data = json.loads(path.read_text())
        data["state"] = state
        path.write_text(json.dumps(data))

    def state(self, request_id, store=None):
        path = approval_mod._request_path(store or self.store, request_id)
        return json.loads(path.read_text())["state"]

    def snapshot(self):
        d = approval_mod._approvals_dir(self.store)
        return {p.name: p.read_bytes() for p in d.iterdir()}

    def run_cli(self, sub, request_id, trace_dir=None):
        # no_resume and no_kill stay off, so a request resolved by mistake
        # would also be signalled.
        args = argparse.Namespace(
            trace_dir=trace_dir or self.trace_dir, approval_cmd=sub,
            request_id=request_id, by="", reason="",
            no_resume=False, no_kill=False,
        )
        out, err = io.StringIO(), io.StringIO()
        with patch("os.kill") as kill, patch("os.scandir", _DescendingScandir), \
             patch("sys.stdout", out), patch("sys.stderr", err):
            code = approval_mod.cmd_approval(args)
        return code, out.getvalue(), err.getvalue(), kill

    def run_ok(self, sub, request_id, trace_dir=None):
        code, out, err, kill = self.run_cli(sub, request_id, trace_dir)
        self.assertEqual((code, err), (0, ""))
        return out, kill

    def assert_refused(self, sub, request_id, message, trace_dir=None):
        code, out, err, kill = self.run_cli(sub, request_id, trace_dir)
        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertEqual(err, ("" if sub == "show" else "❌ ") + message + "\n")
        kill.assert_not_called()

    def test_ambiguous_prefix_is_refused(self):
        self.save_three()
        before = self.snapshot()
        for sub in ("show", "approve", "deny"):
            with self.subTest(sub=sub):
                self.assert_refused(sub, "abc", self.AMBIGUOUS)
        self.assertEqual(self.snapshot(), before)

    def test_refusal_names_five_ids_at_most(self):
        named = "abc100000000, abc200000000, abc300000000, abc400000000, abc500000000"
        for n in (3, 1, 5, 2, 4):
            self.save(f"abc{n}00000000")
        for count, more in ((5, ""), (6, " and 1 more"), (7, " and 2 more")):
            if count > 5:
                self.save(f"abc{count}00000000")
            with self.subTest(matches=count):
                self.assert_refused(
                    "show", "abc",
                    f"Request ID prefix abc matches {count} requests: {named}{more}",
                )

    def test_exact_id_wins_over_longer_ids(self):
        decided = {"approve": (APPROVED, signal.SIGCONT), "deny": (DENIED, signal.SIGTERM)}
        for sub in ("show", "approve", "deny"):
            with self.subTest(sub=sub):
                trace_dir = self.new_trace_dir()
                store = TraceStore(trace_dir)
                self.save("abc", 1111, store)
                self.save("abc111111111", 2222, store)
                out, kill = self.run_ok(sub, "abc", trace_dir)
                if sub == "show":
                    self.assertEqual(json.loads(out)["request_id"], "abc")
                    kill.assert_not_called()
                else:
                    state, sig = decided[sub]
                    kill.assert_called_once_with(1111, sig)
                    self.assertEqual(self.state("abc", store), state)
                    self.assertEqual(self.state("abc111111111", store), PENDING)
        # An exact ID whose record cannot be read still wins. The lookup does
        # not fall through to the longer ID.
        self.save("abc", 1111)
        self.save("abc111111111", 2222)
        self.set_state("abc", "maybe")
        before = self.snapshot()
        for sub in ("show", "approve", "deny"):
            with self.subTest(sub=sub, record="invalid"):
                self.assert_refused(
                    sub, "abc",
                    "Request abc has an invalid record: invalid approval state: 'maybe'",
                )
        self.assertEqual(self.snapshot(), before)

    def test_unique_prefix_resolves_the_same_request_everywhere(self):
        self.save("abc111111111", 1111)
        self.save("abc222222222", 2222)
        self.save("[d]444444444", 4444)
        out, _ = self.run_ok("show", "abc1")
        self.assertEqual(json.loads(out)["request_id"], "abc111111111")
        _, kill = self.run_ok("approve", "abc1")
        kill.assert_called_once_with(1111, signal.SIGCONT)
        self.assertEqual(self.state("abc111111111"), APPROVED)
        _, kill = self.run_ok("deny", "abc2")
        kill.assert_called_once_with(2222, signal.SIGTERM)
        self.assertEqual(self.state("abc222222222"), DENIED)
        # Glob characters are plain text, so [d] names the ID that starts with it.
        out, _ = self.run_ok("show", "[d]")
        self.assertEqual(json.loads(out)["request_id"], "[d]444444444")
        _, kill = self.run_ok("deny", "[d]")
        kill.assert_called_once_with(4444, signal.SIGTERM)
        self.assertEqual(self.state("[d]444444444"), DENIED)

    def test_wildcards_match_nothing(self):
        self.save_three()
        before = self.snapshot()
        for arg in ("*", "?bc2*", "[d]"):
            for sub in ("show", "approve", "deny"):
                with self.subTest(arg=arg, sub=sub):
                    self.assert_refused(sub, arg, f"Request not found: {arg}")
        self.assertEqual(self.snapshot(), before)

    def test_unsafe_argument_is_refused_before_any_file_access(self):
        outside = Path(self.trace_dir) / "zz-outside.json"
        outside.write_text(ApprovalRequest(request_id="zz-outside").to_json())
        before = outside.read_bytes()
        no_access = AssertionError("file access")
        for arg in ("../zz", "a/b", "a\\b", "", ".", "..", "abc\x1b[2J"):
            with self.subTest(arg=arg), \
                 patch.object(approval_mod, "TraceStore", side_effect=no_access), \
                 patch.object(approval_mod, "_approvals_dir", side_effect=no_access):
                for sub in ("show", "approve", "deny"):
                    self.assert_refused(sub, arg, "invalid request ID")
                self.assertIsNone(find_request(self.store, arg))
                self.assertEqual(approve_request(self.store, arg, resume_agent=False),
                                 (False, "invalid request ID"))
                self.assertEqual(deny_request(self.store, arg, kill_agent=False),
                                 (False, "invalid request ID"))
        self.assertEqual(outside.read_bytes(), before)

    def test_unreadable_match_still_makes_prefix_ambiguous(self):
        self.save_three()
        approval_mod._request_path(self.store, "abc111111111").write_text("{not json")
        before = self.snapshot()
        for sub in ("show", "approve", "deny"):
            with self.subTest(sub=sub):
                self.assert_refused(sub, "abc", self.AMBIGUOUS)
        self.assertEqual(self.snapshot(), before)

    def test_single_invalid_match_reports_the_record(self):
        self.save("abc111111111", 1111)
        self.save("def333333333", 3333)
        self.set_state("abc111111111", "maybe")
        before = self.snapshot()
        for sub in ("show", "approve", "deny"):
            with self.subTest(sub=sub):
                self.assert_refused(
                    sub, "abc",
                    "Request abc111111111 has an invalid record: "
                    "invalid approval state: 'maybe'",
                )
        self.assertEqual(self.snapshot(), before)

    def test_symlink_counts_as_a_match_and_is_never_followed(self):
        # (symlink ID, regular ID or None, message for the prefix abc)
        cases = [
            ("abc111111111", None,
             "Request abc111111111 has an invalid record: is a symlink"),
            ("abc111111111", "abc222222222", self.AMBIGUOUS),
            ("abc", "abc111111111", "Request abc has an invalid record: is a symlink"),
        ]
        for link_id, regular_id, message in cases:
            for sub in ("show", "approve", "deny"):
                with self.subTest(link=link_id, regular=regular_id, sub=sub):
                    trace_dir = self.new_trace_dir()
                    store = TraceStore(trace_dir)
                    target = Path(self.new_trace_dir()) / f"{link_id}.json"
                    target.write_text(
                        ApprovalRequest(request_id=link_id, agent_pid=1111).to_json())
                    link = approval_mod._request_path(store, link_id)
                    link.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        link.symlink_to(target)
                    except (OSError, NotImplementedError):
                        self.skipTest("cannot create a symlink here")
                    if regular_id:
                        self.save(regular_id, 2222, store)
                    before = target.read_bytes()
                    self.assert_refused(sub, "abc", message, trace_dir)
                    self.assertTrue(link.is_symlink())
                    self.assertEqual(target.read_bytes(), before)
                    if regular_id:
                        self.assertEqual(self.state(regular_id, store), PENDING)

    def test_messages_show_ids_as_text(self):
        # A request file name can hold escape codes, and these messages name
        # IDs taken from file names.
        name = "abc" + _escapes.PAYLOAD
        try:
            self.save(name)
        except OSError:
            self.skipTest("the file system refuses this name")
        self.save("abc111111111")
        self.set_state(name, "maybe")
        shown = "abc" + _escapes.PAYLOAD_TEXT
        before = self.snapshot()
        for sub in ("show", "approve", "deny"):
            with self.subTest(sub=sub):
                self.assert_refused(
                    sub, "abc",
                    f"Request ID prefix abc matches 2 requests: abc111111111, {shown}",
                )
                self.assert_refused(
                    sub, "abcA",
                    f"Request {shown} has an invalid record: "
                    "invalid approval state: 'maybe'",
                )
                self.assert_refused(sub, "zz\u202e", r"Request not found: zz\u202e")
        self.assertEqual(self.snapshot(), before)


def _request(**fields):
    values = dict(request_id="plain", state=PENDING, rule_name="plain", tool_name="plain")
    values.update(fields)
    return ApprovalRequest(**values)


# (label, kind, build): build(value) returns a request with value in the
# column the label names. The kinds are in tests/_escapes.py.
APPROVAL_LIST_CASES = [
    ("ID", "text", lambda v: _request(request_id=v)),
    ("State", "text", lambda v: _request(state=v)),
    ("Rule", "mid", lambda v: _request(rule_name=v)),
    ("Tool", "mid", lambda v: _request(tool_name=v)),
]

# (label, cut, build): approval list keeps cut characters of the value in the
# column the label names.
APPROVAL_LIST_CUTS = [
    ("Rule", 24, lambda v: _request(rule_name=v)),
    ("Tool", 20, lambda v: _request(tool_name=v)),
]


class TestApprovalListShowsEscapesAsText(unittest.TestCase):
    """A tool name in an approval request comes from the agent or an MCP
    server. approval list shows escape codes in it as text."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.trace_dir = tmp.name

    def list_output(self):
        args = argparse.Namespace(trace_dir=self.trace_dir, approval_cmd="list", state=None)
        with patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(approval_mod.cmd_approval(args), 0)
        return out.getvalue()

    def render(self, request):
        # The state is checked when a request is loaded, so pass the request
        # in directly to cover every column.
        with patch.object(approval_mod, "list_requests", return_value=[request]):
            return self.list_output()

    def test_each_column_shows_escapes_as_text(self):
        _escapes.check_each_value(self, APPROVAL_LIST_CASES, self.render)

    def test_plain_text_prints_unchanged(self):
        _escapes.check_plain_text_unchanged(self, APPROVAL_LIST_CASES, self.render)

    def test_value_as_long_as_the_cut_shows_in_full(self):
        _escapes.check_each_cut(self, APPROVAL_LIST_CUTS, self.render)

    def test_stored_tool_name_shows_as_text(self):
        create_approval_request(TraceStore(self.trace_dir), "esc-a", "ask-before-shell",
                                tool_name="x\x1b]52;c;aWQK\x07")
        output = self.list_output()
        self.assertIn(r"x\x1b]52;c;aWQK\x07", output)
        self.assertNotIn("\x1b", output)
        self.assertNotIn("\x07", output)

    def test_sgr_code_shows_as_text(self):
        output = self.render(_request(tool_name="a\x1b[8mhidden"))
        self.assertIn(r"a\x1b[8mhidden", output)
        self.assertNotIn("\x1b", output)


if __name__ == "__main__":
    unittest.main()
