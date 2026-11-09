"""Tests for who owns an agent pause in watch (editor, rule, approval)."""

import io
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_trace.approval import approve_request, deny_request, load_request
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore
from agent_trace.watch import (
    PAUSE_APPROVAL,
    PAUSE_EDITOR,
    PAUSE_RULE,
    NannyRule,
    WatcherConfig,
    WatchState,
    _IDLE_SENTINEL,
    _check_pause_file,
    _dispatch_alert,
    _dispatch_nanny_rule,
    watch_session,
)

SESSION = "pause-session"
PID = 424242


class PauseTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = TraceStore(self._tmp.name, redact=False)
        self.store.create_session(SessionMeta(session_id=SESSION, started_at=1.0))
        self.state = WatchState(agent_pid=PID, session_id=SESSION)
        self.config = WatcherConfig(on_violation="terminal", alert_log=str(Path(self._tmp.name) / "alerts.log"))
        self.out = io.StringIO()

        patches = {
            "pause": patch("agent_trace.watch._pause_process"),
            "resume": patch("agent_trace.watch._resume_process"),
            "kill": patch("agent_trace.watch._kill_process"),
            "os_kill": patch("os.kill"),
            "terminal": patch("agent_trace.watch._alert_terminal"),
        }
        self.mocks = {}
        for name, p in patches.items():
            self.mocks[name] = p.start()
            self.addCleanup(p.stop)

    @property
    def marker(self) -> Path:
        return Path(self._tmp.name) / ".pause-request"

    def tick(self, n: int = 1) -> None:
        for _ in range(n):
            _check_pause_file(self.store, self.state, self.out)

    def rule_pause(self) -> None:
        _dispatch_alert("rule fired", self.config, self.state, action="pause",
                        store=self.store, session_id=SESSION)

    def require_approval(self) -> str:
        _dispatch_alert("needs approval", self.config, self.state,
                        action="require_approval", notify="big-change",
                        store=self.store, session_id=SESSION)
        return self.state.approval_request_id


class TestRulePause(PauseTestCase):
    def test_rule_pause_survives_idle_ticks_without_marker(self):
        self.rule_pause()
        self.mocks["pause"].assert_called_once_with(PID)
        self.assertEqual(self.state.pause_owner, PAUSE_RULE)

        self.tick(5)

        self.mocks["resume"].assert_not_called()
        self.assertTrue(self.state.paused)

    def test_editor_pause_then_resume_releases_rule_pause(self):
        self.rule_pause()
        self.marker.write_text(SESSION)
        self.tick()
        self.assertEqual(self.state.pause_owner, PAUSE_EDITOR)
        self.assertEqual(self.mocks["pause"].call_count, 1)

        self.marker.unlink()
        self.tick()
        self.mocks["resume"].assert_called_once_with(PID)
        self.assertFalse(self.state.paused)

    def test_rule_pause_during_editor_pause_keeps_editor_owner(self):
        self.marker.write_text(SESSION)
        self.tick()
        self.rule_pause()
        self.assertEqual(self.state.pause_owner, PAUSE_EDITOR)
        self.assertEqual(self.mocks["pause"].call_count, 1)


class TestEditorPause(PauseTestCase):
    def test_marker_pauses_and_removal_resumes(self):
        self.marker.write_text(SESSION)
        self.tick(3)
        self.mocks["pause"].assert_called_once_with(PID)
        self.assertEqual(self.state.pause_owner, PAUSE_EDITOR)

        self.marker.unlink()
        self.tick(3)
        self.mocks["resume"].assert_called_once_with(PID)
        self.assertIsNone(self.state.pause_owner)

    def test_empty_marker_applies_to_any_session(self):
        self.marker.write_text("")
        self.tick()
        self.assertEqual(self.state.pause_owner, PAUSE_EDITOR)

    def test_marker_for_other_session_is_ignored(self):
        self.marker.write_text("some-other-session")
        self.tick()
        self.mocks["pause"].assert_not_called()
        self.assertFalse(self.state.paused)

    def test_no_pid_means_no_signal(self):
        self.state.agent_pid = None
        self.marker.write_text(SESSION)
        self.tick()
        self.mocks["pause"].assert_not_called()
        self.assertFalse(self.state.paused)


class TestApprovalPause(PauseTestCase):
    def test_approval_pause_creates_request_and_owns_pause(self):
        request_id = self.require_approval()
        self.assertIsNotNone(request_id)
        self.assertEqual(self.state.pause_owner, PAUSE_APPROVAL)
        self.mocks["pause"].assert_called_once_with(PID)
        req = load_request(self.store, request_id)
        self.assertEqual(req.state, "pending")
        self.assertEqual(req.session_id, SESSION)
        self.assertEqual(req.agent_pid, PID)

    def test_idle_ticks_and_editor_marker_do_not_release_approval(self):
        self.require_approval()
        self.tick(3)
        self.marker.write_text(SESSION)
        self.tick()
        self.marker.unlink()
        self.tick()
        self.mocks["resume"].assert_not_called()
        self.assertEqual(self.state.pause_owner, PAUSE_APPROVAL)

    def test_approval_decision_releases_pause_and_later_rules_fire(self):
        request_id = self.require_approval()
        ok, _ = approve_request(self.store, request_id, resume_agent=False)
        self.assertTrue(ok)

        self.tick()
        self.assertIsNone(self.state.pause_owner)
        self.assertIsNone(self.state.approval_request_id)
        # approve_request owns the SIGCONT; watch must not send a second one.
        self.mocks["resume"].assert_not_called()

        self.rule_pause()
        self.assertEqual(self.mocks["pause"].call_count, 2)
        self.assertEqual(self.state.pause_owner, PAUSE_RULE)

    def test_denial_keeps_agent_stopped_but_releasable(self):
        request_id = self.require_approval()
        deny_request(self.store, request_id, kill_agent=False)
        self.tick()
        self.assertEqual(self.state.pause_owner, PAUSE_RULE)
        self.assertIsNone(self.state.approval_request_id)
        self.mocks["resume"].assert_not_called()

        # A later approval rule still gates the agent with a new request.
        second = self.require_approval()
        self.assertNotEqual(second, request_id)
        self.assertEqual(self.state.pause_owner, PAUSE_APPROVAL)

    def test_editor_can_release_after_denial(self):
        request_id = self.require_approval()
        deny_request(self.store, request_id, kill_agent=False)
        self.tick()
        self.marker.write_text(SESSION)
        self.tick()
        self.marker.unlink()
        self.tick()
        self.mocks["resume"].assert_called_once_with(PID)
        self.assertFalse(self.state.paused)

    def test_exited_agent_clears_pause(self):
        request_id = self.require_approval()
        deny_request(self.store, request_id, kill_agent=False)

        def fake_kill(pid, sig):
            if sig == 0:
                raise ProcessLookupError
        self.mocks["os_kill"].side_effect = fake_kill
        self.tick()
        self.assertIsNone(self.state.pause_owner)

    def test_editor_marker_reapplies_after_approval(self):
        self.marker.write_text(SESSION)
        self.tick()
        request_id = self.require_approval()
        self.assertEqual(self.state.pause_owner, PAUSE_APPROVAL)
        self.assertEqual(self.mocks["pause"].call_count, 1)

        approve_request(self.store, request_id, resume_agent=False)
        self.tick()
        self.assertEqual(self.state.pause_owner, PAUSE_EDITOR)
        self.assertEqual(self.mocks["pause"].call_count, 2)

    def test_second_approval_while_pending_does_not_create_request(self):
        first = self.require_approval()
        second = self.require_approval()
        self.assertEqual(first, second)
        self.assertEqual(len(list((Path(self._tmp.name) / ".approvals").glob("*.json"))), 1)

    def test_failed_request_falls_back_to_rule_owner(self):
        with patch("agent_trace.approval.create_approval_request", side_effect=OSError("disk full")):
            self.require_approval()
        self.assertEqual(self.state.pause_owner, PAUSE_RULE)
        self.assertIsNone(self.state.approval_request_id)

    def test_dry_run_neither_pauses_nor_creates_request(self):
        _dispatch_alert("needs approval", self.config, self.state,
                        action="require_approval", dry_run=True,
                        store=self.store, session_id=SESSION)
        self.mocks["pause"].assert_not_called()
        self.assertFalse(self.state.paused)
        self.assertFalse((Path(self._tmp.name) / ".approvals").exists())

    def test_nanny_rule_require_approval_creates_request(self):
        rule = NannyRule(name="gate", condition="files_modified > 0", action="require_approval")
        event = TraceEvent(event_type=EventType.TOOL_CALL, timestamp=1.0, session_id=SESSION, data={})
        _dispatch_nanny_rule(rule, event, self.store, SESSION, self.state, self.config)
        self.assertEqual(self.state.pause_owner, PAUSE_APPROVAL)
        self.assertIsNotNone(load_request(self.store, self.state.approval_request_id))


class TestWatchSessionLoop(PauseTestCase):
    def _run(self, items, **kwargs):
        real_state = WatchState

        def make_state(**kw):
            self.loop_state = real_state(agent_pid=PID, **kw)
            return self.loop_state

        with patch("agent_trace.watch._tail_events", return_value=iter(items)), \
             patch("agent_trace.watch.WatchState", side_effect=make_state):
            watch_session(self.store, SESSION, self.config, out=self.out, **kwargs)

    def test_rule_pause_is_not_undone_by_idle_ticks(self):
        rule = NannyRule(name="stop", condition="files_modified > 0", action="pause")
        edit = TraceEvent(
            event_type=EventType.TOOL_CALL, timestamp=time.time(), session_id=SESSION,
            data={"tool_name": "Write", "arguments": {"file_path": "a.py"}},
        )
        self._run([edit] + [_IDLE_SENTINEL] * 5, nanny_rules=[rule])
        self.mocks["pause"].assert_called_once_with(PID)
        self.mocks["resume"].assert_not_called()
        self.assertEqual(self.loop_state.pause_owner, PAUSE_RULE)

    def test_idle_timeout_does_not_stop_watch_while_paused(self):
        self.marker.write_text(SESSION)

        def items():
            yield _IDLE_SENTINEL  # applies the editor pause
            for _ in range(3):
                time.sleep(0.1)
                yield _IDLE_SENTINEL

        self._run(items(), max_idle_seconds=0.05)
        self.assertNotIn("stopping", self.out.getvalue())
        self.assertEqual(self.loop_state.pause_owner, PAUSE_EDITOR)

    def test_idle_timeout_still_stops_unpaused_watch(self):
        def items():
            yield _IDLE_SENTINEL
            time.sleep(0.1)
            yield _IDLE_SENTINEL

        self._run(items(), max_idle_seconds=0.05)
        self.assertIn("stopping", self.out.getvalue())


if __name__ == "__main__":
    unittest.main()
