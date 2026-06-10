"""Tests for per-project rolling budget guardrails."""

from __future__ import annotations

import io
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.project_budget import (
    ProjectBudgetConfig,
    enforce_new_session_budget,
    load_project_budget_config,
    project_budget_status,
    rolling_project_spend,
)
from agent_trace.store import TraceStore
from agent_trace.watch import WatcherConfig, watch_session


def _add_session(store: TraceStore, started_at: float, payload_size: int = 4000) -> str:
    meta = SessionMeta(agent_name="agent", command="test")
    meta.started_at = started_at
    session_path = store.create_session(meta)
    session_id = session_path.name
    store.append_event(
        session_id,
        TraceEvent(
            event_type=EventType.SESSION_START,
            timestamp=started_at,
            session_id=session_id,
            data={},
        ),
    )
    store.append_event(
        session_id,
        TraceEvent(
            event_type=EventType.ASSISTANT_RESPONSE,
            timestamp=started_at + 1,
            session_id=session_id,
            data={"text": "x" * payload_size},
        ),
    )
    store.append_event(
        session_id,
        TraceEvent(
            event_type=EventType.SESSION_END,
            timestamp=started_at + 2,
            session_id=session_id,
            data={},
        ),
    )
    return session_id


class TestProjectBudgetConfig(unittest.TestCase):
    def test_loads_budget_block_from_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".agent-strace.yaml"
            path.write_text(
                "\n".join([
                    "budget:",
                    "  weekly: 20.00",
                    "  warn_at: 0.75",
                    "  stop_at: 1.00",
                    "  per_session_max: 5.50",
                ])
            )

            cfg = load_project_budget_config(path)

        self.assertEqual(cfg.weekly, 20.0)
        self.assertEqual(cfg.warn_at, 0.75)
        self.assertEqual(cfg.stop_at, 1.0)
        self.assertEqual(cfg.per_session_max, 5.5)

    def test_missing_config_disables_budget(self) -> None:
        cfg = load_project_budget_config("/no/such/.agent-strace.yaml")
        self.assertFalse(cfg.enabled)


class TestRollingProjectSpend(unittest.TestCase):
    def test_rolling_spend_excludes_old_sessions(self) -> None:
        now = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            store = TraceStore(Path(tmp))
            recent_id = _add_session(store, now - 60)
            _add_session(store, now - 8 * 24 * 60 * 60)

            total = rolling_project_spend(store, now=now)
            without_recent = rolling_project_spend(
                store, now=now, exclude_session_id=recent_id
            )

        self.assertGreater(total, 0.0)
        self.assertEqual(without_recent, 0.0)

    def test_status_warns_and_stops_at_thresholds(self) -> None:
        now = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            store = TraceStore(Path(tmp))
            _add_session(store, now - 60, payload_size=20000)
            spent = rolling_project_spend(store, now=now)
            cfg = ProjectBudgetConfig(
                weekly=spent,
                warn_at=0.50,
                stop_at=1.00,
            )

            status = project_budget_status(store, cfg, now=now)

        self.assertTrue(status.should_warn)
        self.assertTrue(status.should_stop)

    def test_enforce_new_session_budget_refuses_over_stop(self) -> None:
        now = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            store = TraceStore(Path(tmp))
            _add_session(store, now - 60, payload_size=20000)
            spent = rolling_project_spend(store, now=now)
            cfg = ProjectBudgetConfig(weekly=spent / 2, stop_at=1.00)
            out = io.StringIO()

            allowed = enforce_new_session_budget(store, cfg, out, now=now)

        self.assertFalse(allowed)
        self.assertIn("refusing new session", out.getvalue())


class TestWatchProjectBudget(unittest.TestCase):
    def _fake_tail(self, events: list[TraceEvent]):
        def _gen(_path, poll_interval=0.5):
            yield from events
        return _gen

    def test_watch_session_writes_project_budget_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TraceStore(Path(tmp) / "traces")
            session_path = store.create_session(SessionMeta(agent_name="agent"))
            session_id = session_path.name
            out = io.StringIO()
            alert_log = Path(tmp) / "alerts.log"
            cfg = WatcherConfig(alert_log=str(alert_log))
            budget = ProjectBudgetConfig(weekly=1.0, warn_at=0.0)
            events = [
                TraceEvent(
                    event_type=EventType.SESSION_END,
                    timestamp=time.time(),
                    session_id=session_id,
                    data={},
                )
            ]

            with patch("agent_trace.watch._tail_events", self._fake_tail(events)):
                watch_session(
                    store,
                    session_id,
                    cfg,
                    out=out,
                    project_budget_config=budget,
                )

            self.assertIn("ProjectBudget", out.getvalue())
            self.assertIn("ProjectBudget", alert_log.read_text())

    def test_per_session_max_cost_violation_uses_kill_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TraceStore(Path(tmp))
            session_path = store.create_session(SessionMeta(agent_name="agent"))
            session_id = session_path.name
            cfg = WatcherConfig(max_cost_dollars=0.0)
            budget = ProjectBudgetConfig(
                weekly=10.0,
                warn_at=2.0,
                per_session_max=1.0,
            )
            events = [
                TraceEvent(
                    event_type=EventType.ASSISTANT_RESPONSE,
                    timestamp=time.time(),
                    session_id=session_id,
                    data={"text": "x" * 10000},
                ),
                TraceEvent(
                    event_type=EventType.SESSION_END,
                    timestamp=time.time(),
                    session_id=session_id,
                    data={},
                ),
            ]

            with patch("agent_trace.watch._tail_events", self._fake_tail(events)), \
                 patch("agent_trace.watch._dispatch_alert") as dispatch:
                watch_session(
                    store,
                    session_id,
                    cfg,
                    out=io.StringIO(),
                    project_budget_config=budget,
                )

        self.assertEqual(dispatch.call_args.kwargs["action"], "kill")


if __name__ == "__main__":
    unittest.main()
