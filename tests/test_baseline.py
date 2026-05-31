"""Tests for baseline anomaly detection (issue #134)."""

import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.baseline import (
    BaselineProfile,
    MetricStats,
    AnomalyResult,
    build_baseline,
    check_session,
    save_baseline,
    load_baseline,
    _mean,
    _stddev,
    _z_score,
)
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore


def _make_store_with_sessions(tmpdir: str, n: int = 10, tool_calls: int = 5) -> TraceStore:
    """Create a store with n completed sessions."""
    store = TraceStore(tmpdir)
    for i in range(n):
        meta = SessionMeta(agent_name="test-agent")
        meta.tool_calls = tool_calls
        meta.llm_requests = 2
        meta.started_at = time.time() - 86400 * (i + 1)
        meta.ended_at = meta.started_at + 60.0
        meta.total_duration_ms = 60_000.0
        store.create_session(meta)
        store.append_event(meta.session_id, TraceEvent(
            event_type=EventType.SESSION_START, session_id=meta.session_id,
        ))
        store.update_meta(meta)
    return store


class TestStatHelpers(unittest.TestCase):
    def test_mean(self):
        self.assertAlmostEqual(_mean([1.0, 2.0, 3.0]), 2.0)

    def test_mean_empty(self):
        self.assertEqual(_mean([]), 0.0)

    def test_stddev(self):
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        mean = _mean(values)
        sd = _stddev(values, mean)
        self.assertAlmostEqual(sd, 2.0, places=0)

    def test_stddev_single_value(self):
        self.assertEqual(_stddev([5.0], 5.0), 0.0)

    def test_z_score_zero_stddev(self):
        self.assertEqual(_z_score(10.0, 5.0, 0.0), 0.0)

    def test_z_score_normal(self):
        z = _z_score(7.0, 5.0, 2.0)
        self.assertAlmostEqual(z, 1.0)


class TestBuildBaseline(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_builds_from_sessions(self):
        store = _make_store_with_sessions(self.tmpdir, n=10)
        profile = build_baseline(store, since_days=30)
        self.assertGreater(profile.session_count, 0)
        self.assertGreater(profile.tool_calls.mean, 0)

    def test_empty_store_returns_zero_profile(self):
        store = TraceStore(self.tmpdir)
        profile = build_baseline(store, since_days=30)
        self.assertEqual(profile.session_count, 0)

    def test_excludes_sessions_outside_window(self):
        store = TraceStore(self.tmpdir)
        # Create a session 60 days ago (outside 30-day window)
        meta = SessionMeta(agent_name="old")
        meta.started_at = time.time() - 86400 * 60
        meta.ended_at = meta.started_at + 60
        meta.total_cost_usd = 99.0
        meta.tool_calls = 100
        store.create_session(meta)
        store.update_meta(meta)
        profile = build_baseline(store, since_days=30)
        self.assertEqual(profile.session_count, 0)


class TestBaselinePersistence(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_save_and_load_roundtrip(self):
        profile = BaselineProfile(session_count=5)
        profile.cost_usd = MetricStats(mean=1.5, stddev=0.3, min=1.0, max=2.0, count=5)
        path = os.path.join(self.tmpdir, "baseline.json")
        save_baseline(profile, path)
        loaded = load_baseline(path)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.session_count, 5)
        self.assertAlmostEqual(loaded.cost_usd.mean, 1.5)

    def test_load_missing_returns_none(self):
        result = load_baseline("/nonexistent/path/baseline.json")
        self.assertIsNone(result)


class TestCheckSession(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _make_profile(self, mean_tools: float = 5.0, stddev_tools: float = 1.0) -> BaselineProfile:
        profile = BaselineProfile(session_count=20)
        profile.cost_usd = MetricStats(mean=0.0, stddev=0.0, min=0.0, max=0.0, count=20)
        profile.tool_calls = MetricStats(mean=mean_tools, stddev=stddev_tools,
                                         min=2.0, max=10.0, count=20)
        profile.duration_ms = MetricStats(mean=60000.0, stddev=5000.0,
                                          min=10000.0, max=120000.0, count=20)
        profile.error_rate = MetricStats(mean=0.0, stddev=0.0, min=0.0, max=0.0, count=20)
        profile.llm_requests = MetricStats(mean=2.0, stddev=0.5, min=1.0, max=5.0, count=20)
        return profile

    def _make_session(self, tool_calls: int = 5) -> str:
        store = TraceStore(self.tmpdir)
        meta = SessionMeta(agent_name="test")
        meta.tool_calls = tool_calls
        meta.llm_requests = 2
        meta.started_at = time.time() - 120
        meta.ended_at = time.time()
        meta.total_duration_ms = 60_000.0
        store.create_session(meta)
        store.update_meta(meta)
        return meta.session_id

    def test_normal_session_not_anomalous(self):
        store = TraceStore(self.tmpdir)
        sid = self._make_session(tool_calls=5)
        profile = self._make_profile(mean_tools=5.0, stddev_tools=1.0)
        result = check_session(store, sid, profile, sigma=2.0)
        self.assertFalse(result.anomalous)

    def test_high_tool_call_session_flagged(self):
        store = TraceStore(self.tmpdir)
        sid = self._make_session(tool_calls=100)  # far above mean of 5
        profile = self._make_profile(mean_tools=5.0, stddev_tools=1.0)
        result = check_session(store, sid, profile, sigma=2.0)
        self.assertTrue(result.anomalous)
        self.assertIn("tool_calls", result.deviations)
        self.assertGreater(result.deviations["tool_calls"], 2.0)

    def test_deviations_dict_populated(self):
        store = TraceStore(self.tmpdir)
        sid = self._make_session(tool_calls=5)
        profile = self._make_profile()
        result = check_session(store, sid, profile)
        self.assertIsInstance(result.deviations, dict)
        self.assertIn("tool_calls", result.deviations)

    def test_insufficient_baseline_data_skipped(self):
        """Metrics with count < 3 should not contribute to anomaly detection."""
        store = TraceStore(self.tmpdir)
        sid = self._make_session(tool_calls=999)
        profile = BaselineProfile(session_count=2)
        profile.tool_calls = MetricStats(mean=5.0, stddev=1.0, min=2.0, max=10.0, count=2)
        result = check_session(store, sid, profile, sigma=2.0)
        self.assertFalse(result.anomalous)  # count < 3, skipped


if __name__ == "__main__":
    unittest.main()
