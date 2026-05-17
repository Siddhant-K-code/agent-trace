"""Tests for behavioral drift detection."""

import json
import tempfile
import time
import unittest
from pathlib import Path

from agent_trace.drift import (
    BehavioralFingerprint,
    DistStats,
    SessionMetrics,
    _dist_stats,
    _js_divergence,
    _stats_divergence,
    compute_drift,
    compute_fingerprint,
    print_report,
)
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(tmp: str) -> TraceStore:
    return TraceStore(tmp)


def _add_session(store: TraceStore, session_id: str, events: list[TraceEvent], started_at: float | None = None) -> None:
    ts = started_at or time.time()
    meta = SessionMeta(
        session_id=session_id,
        started_at=ts,
        ended_at=ts + 60,
    )
    store.create_session(meta)
    for ev in events:
        store.append_event(session_id, ev)


def _tool_call(name: str, ts: float = 0.0) -> TraceEvent:
    return TraceEvent(event_type=EventType.TOOL_CALL, timestamp=ts, data={"tool_name": name})


def _error(ts: float = 0.0) -> TraceEvent:
    return TraceEvent(event_type=EventType.ERROR, timestamp=ts, data={"message": "fail"})


def _decision(ts: float = 0.0) -> TraceEvent:
    return TraceEvent(event_type=EventType.DECISION, timestamp=ts, data={"choice": "A"})


def _file_write(path: str, ts: float = 0.0) -> TraceEvent:
    return TraceEvent(event_type=EventType.FILE_WRITE, timestamp=ts, data={"path": path})


# ---------------------------------------------------------------------------
# Unit tests: statistical helpers
# ---------------------------------------------------------------------------

class TestJsDivergence(unittest.TestCase):
    def test_identical_distributions_zero(self):
        p = {"a": 0.5, "b": 0.5}
        self.assertAlmostEqual(_js_divergence(p, p), 0.0, places=5)

    def test_completely_different_distributions(self):
        p = {"a": 1.0}
        q = {"b": 1.0}
        score = _js_divergence(p, q)
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_partial_overlap(self):
        p = {"a": 0.7, "b": 0.3}
        q = {"a": 0.3, "b": 0.7}
        score = _js_divergence(p, q)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_empty_distributions(self):
        self.assertEqual(_js_divergence({}, {}), 0.0)

    def test_one_empty(self):
        p = {"a": 1.0}
        score = _js_divergence(p, {})
        self.assertGreater(score, 0.0)


class TestStatsDivergence(unittest.TestCase):
    def test_identical_stats_zero(self):
        s = DistStats(mean=1.0, p50=1.0, p95=2.0)
        self.assertAlmostEqual(_stats_divergence(s, s), 0.0, places=5)

    def test_different_stats_nonzero(self):
        a = DistStats(mean=1.0, p50=1.0, p95=2.0)
        b = DistStats(mean=5.0, p50=5.0, p95=8.0)
        score = _stats_divergence(a, b)
        self.assertGreater(score, 0.0)

    def test_both_zero(self):
        s = DistStats(mean=0.0, p50=0.0, p95=0.0)
        self.assertEqual(_stats_divergence(s, s), 0.0)


class TestDistStats(unittest.TestCase):
    def test_empty(self):
        s = _dist_stats([])
        self.assertEqual(s.mean, 0.0)

    def test_single_value(self):
        s = _dist_stats([5.0])
        self.assertEqual(s.mean, 5.0)
        self.assertEqual(s.p50, 5.0)
        self.assertEqual(s.p95, 5.0)

    def test_multiple_values(self):
        s = _dist_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(s.mean, 3.0)
        self.assertAlmostEqual(s.p50, 3.0)
        self.assertGreaterEqual(s.p95, 4.0)


# ---------------------------------------------------------------------------
# Unit tests: fingerprint computation
# ---------------------------------------------------------------------------

class TestComputeFingerprint(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = _make_store(self.tmp)

    def test_empty_sessions(self):
        fp = compute_fingerprint(self.store, [])
        self.assertEqual(fp.sessions, 0)

    def test_single_session_tool_mix(self):
        events = [
            _tool_call("Bash", 0.0),
            _tool_call("Bash", 1.0),
            _tool_call("Read", 2.0),
        ]
        _add_session(self.store, "s1", events, started_at=time.time())
        fp = compute_fingerprint(self.store, ["s1"])
        self.assertEqual(fp.sessions, 1)
        self.assertAlmostEqual(fp.tool_mix.get("Bash", 0), 2 / 3, places=3)
        self.assertAlmostEqual(fp.tool_mix.get("Read", 0), 1 / 3, places=3)

    def test_error_rate_computed(self):
        events = [_tool_call("Bash"), _error(), _error()]
        _add_session(self.store, "s2", events, started_at=time.time())
        fp = compute_fingerprint(self.store, ["s2"])
        # 2 errors / 1 tool call = 2.0 (capped at mean)
        self.assertGreater(fp.error_rate.mean, 0.0)

    def test_blast_radius_from_file_writes(self):
        events = [
            _file_write("src/a.py"),
            _file_write("src/b.py"),
            _file_write("src/a.py"),  # duplicate — should not count twice
        ]
        _add_session(self.store, "s3", events, started_at=time.time())
        fp = compute_fingerprint(self.store, ["s3"])
        self.assertEqual(fp.blast_radius.mean, 2.0)

    def test_decision_depth(self):
        events = [_decision(), _decision(), _decision()]
        _add_session(self.store, "s4", events, started_at=time.time())
        fp = compute_fingerprint(self.store, ["s4"])
        self.assertEqual(fp.decision_depth.mean, 3.0)

    def test_fingerprint_serialization_roundtrip(self):
        events = [_tool_call("Bash"), _tool_call("Read")]
        _add_session(self.store, "s5", events, started_at=time.time())
        fp = compute_fingerprint(self.store, ["s5"], fingerprint_id="test_fp")
        json_str = fp.to_json()
        restored = BehavioralFingerprint.from_json(json_str)
        self.assertEqual(restored.fingerprint_id, "test_fp")
        self.assertEqual(restored.sessions, 1)
        self.assertAlmostEqual(
            restored.tool_mix.get("Bash", 0),
            fp.tool_mix.get("Bash", 0),
            places=3,
        )


# ---------------------------------------------------------------------------
# Unit tests: drift computation
# ---------------------------------------------------------------------------

class TestComputeDrift(unittest.TestCase):
    def _make_fp(self, tool_mix: dict, error_mean: float = 0.0) -> BehavioralFingerprint:
        fp = BehavioralFingerprint(
            fingerprint_id="test",
            sessions=10,
            period_start="2026-04-01",
            period_end="2026-04-15",
            tool_mix=tool_mix,
            error_rate=DistStats(mean=error_mean, p50=error_mean, p95=error_mean * 2),
            retry_rate=DistStats(),
            blast_radius=DistStats(),
            session_duration_s=DistStats(mean=60.0, p50=60.0, p95=90.0),
            decision_depth=DistStats(),
        )
        return fp

    def test_identical_fingerprints_low_drift(self):
        fp = self._make_fp({"Bash": 0.5, "Read": 0.5})
        report = compute_drift(fp, fp, threshold=0.20)
        self.assertLess(report.overall_score, 0.05)
        self.assertFalse(report.exceeded)

    def test_completely_different_tool_mix_high_drift(self):
        baseline = self._make_fp({"Bash": 1.0})
        current = self._make_fp({"Read": 1.0})
        report = compute_drift(baseline, current, threshold=0.20)
        self.assertGreater(report.overall_score, 0.20)
        self.assertTrue(report.exceeded)

    def test_error_rate_drift_detected(self):
        baseline = self._make_fp({"Bash": 0.5, "Read": 0.5}, error_mean=0.05)
        current = self._make_fp({"Bash": 0.5, "Read": 0.5}, error_mean=0.50)
        report = compute_drift(baseline, current, threshold=0.20)
        error_dim = next(d for d in report.dimensions if d.name == "error_rate")
        self.assertGreater(error_dim.score, 0.0)

    def test_report_label_stable(self):
        fp = self._make_fp({"Bash": 0.5, "Read": 0.5})
        report = compute_drift(fp, fp, threshold=0.20)
        self.assertEqual(report.label, "stable")

    def test_report_label_high(self):
        baseline = self._make_fp({"Bash": 1.0})
        current = self._make_fp({"Read": 1.0})
        report = compute_drift(baseline, current, threshold=0.20)
        self.assertEqual(report.label, "high")

    def test_all_dimensions_present(self):
        fp = self._make_fp({"Bash": 1.0})
        report = compute_drift(fp, fp)
        names = {d.name for d in report.dimensions}
        expected = {"tool_mix", "error_rate", "retry_rate", "blast_radius", "session_duration_s", "decision_depth"}
        self.assertEqual(names, expected)


# ---------------------------------------------------------------------------
# Integration: end-to-end with store
# ---------------------------------------------------------------------------

class TestDriftEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = _make_store(self.tmp)

    def _populate(self, prefix: str, n: int, tool: str, ts_base: float) -> list[str]:
        ids = []
        for i in range(n):
            sid = f"{prefix}_{i}"
            events = [_tool_call(tool, ts_base + i * 10)]
            _add_session(self.store, sid, events, started_at=ts_base + i * 10)
            ids.append(sid)
        return ids

    def test_stable_sessions_low_drift(self):
        now = time.time()
        baseline_ids = self._populate("base", 5, "Bash", now - 200)
        current_ids = self._populate("curr", 5, "Bash", now - 100)
        baseline_fp = compute_fingerprint(self.store, baseline_ids)
        current_fp = compute_fingerprint(self.store, current_ids)
        report = compute_drift(baseline_fp, current_fp, threshold=0.20)
        self.assertLess(report.overall_score, 0.20)

    def test_shifted_tool_mix_high_drift(self):
        now = time.time()
        baseline_ids = self._populate("base2", 5, "Bash", now - 200)
        current_ids = self._populate("curr2", 5, "Write", now - 100)
        baseline_fp = compute_fingerprint(self.store, baseline_ids)
        current_fp = compute_fingerprint(self.store, current_ids)
        report = compute_drift(baseline_fp, current_fp, threshold=0.20)
        self.assertGreater(report.overall_score, 0.20)

    def test_print_report_no_crash(self):
        import io
        now = time.time()
        baseline_ids = self._populate("base3", 3, "Bash", now - 200)
        current_ids = self._populate("curr3", 3, "Read", now - 100)
        baseline_fp = compute_fingerprint(self.store, baseline_ids)
        current_fp = compute_fingerprint(self.store, current_ids)
        report = compute_drift(baseline_fp, current_fp)
        buf = io.StringIO()
        print_report(report, out=buf)
        output = buf.getvalue()
        self.assertIn("drift", output.lower())
        self.assertIn("overall", output.lower())

    def test_fingerprint_save_and_load(self):
        now = time.time()
        ids = self._populate("fp_test", 3, "Bash", now - 100)
        fp = compute_fingerprint(self.store, ids, fingerprint_id="saved_fp")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(fp.to_json())
            path = f.name
        loaded = BehavioralFingerprint.from_json(Path(path).read_text())
        self.assertEqual(loaded.fingerprint_id, "saved_fp")
        self.assertEqual(loaded.sessions, 3)


if __name__ == "__main__":
    unittest.main()
