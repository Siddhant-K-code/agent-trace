"""Tests for baseline anomaly detection (issue #134)."""

import io
import json
import os
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.baseline import (
    BaselineProfile,
    MetricStats,
    AnomalyResult,
    build_baseline,
    check_session,
    cmd_baseline,
    save_baseline,
    load_baseline,
    _mean,
    _stddev,
    _z_score,
)
from agent_trace.cli import build_parser
from agent_trace.cost import DEFAULT_MODEL, _dollars, estimate_cost
from agent_trace.hooks import _PROVIDER_ENV, hook_main
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore

try:
    from test_capture_matrix_fixtures import (
        FIXTURES,
        load_fixture,
        matcher_applies,
        registered_hooks,
    )
    from test_cost import _model_request
except ImportError:
    from tests.test_capture_matrix_fixtures import (
        FIXTURES,
        load_fixture,
        matcher_applies,
        registered_hooks,
    )
    from tests.test_cost import _model_request


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

    def test_skips_session_with_non_numeric_start_time(self):
        store = TraceStore(self.tmpdir)
        meta = SessionMeta(agent_name="edited")
        store.create_session(meta)
        meta.ended_at = meta.started_at + 60
        store.update_meta(meta)
        meta_path = Path(self.tmpdir) / meta.session_id / "meta.json"
        raw = json.loads(meta_path.read_text())
        raw["started_at"] = "yesterday"
        meta_path.write_text(json.dumps(raw))
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


def _priced_session(
    store: TraceStore,
    input_tokens: int,
    *,
    minutes_ago: float,
    tool_call_data=None,
) -> str:
    """Create a completed Codex-style session whose model request records its usage.

    Every metric except cost is the same for every session made here, so a
    cost difference is the only thing a baseline check can flag.
    """
    meta = SessionMeta(agent_name="codex")
    meta.tool_calls = 1
    meta.llm_requests = 1
    meta.started_at = time.time() - 60 * minutes_ago
    meta.ended_at = meta.started_at + 30.0
    meta.total_duration_ms = 30_000.0
    store.create_session(meta)
    sid, ts = meta.session_id, meta.started_at
    if tool_call_data is None:
        tool_call_data = {"tool_name": "Bash", "arguments": {"command": "ls"}}
    events = [
        TraceEvent(event_type=EventType.USER_PROMPT, timestamp=ts, session_id=sid,
                   data={"prompt": "list the files"}),
        _model_request(ts + 1, input_tokens, 10, session_id=sid),
        TraceEvent(event_type=EventType.TOOL_CALL, timestamp=ts + 2, session_id=sid,
                   data=tool_call_data),
        TraceEvent(event_type=EventType.TOOL_RESULT, timestamp=ts + 3, session_id=sid,
                   data={"result": "README.md"}),
        TraceEvent(event_type=EventType.ASSISTANT_RESPONSE, timestamp=ts + 4, session_id=sid,
                   data={"text": "One file."}),
    ]
    for event in events:
        store.append_event(sid, event)
    store.update_meta(meta)
    return sid


class TestBaselineCost(unittest.TestCase):
    """Baselines price each session exactly as ``agent-strace cost`` does."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.trace_dir = tmp.name
        self.store = TraceStore(tmp.name)
        self.normal = [
            _priced_session(self.store, tokens, minutes_ago=i + 2)
            for i, tokens in enumerate((1000, 1100, 1200, 1300, 1400, 1500))
        ]

    def test_cost_matches_estimate_cost(self):
        profile = build_baseline(self.store, since_days=30)
        costs = [estimate_cost(self.store, sid).total_cost for sid in self.normal]
        self.assertEqual(profile.session_count, 6)
        self.assertEqual(profile.cost_usd.count, 6)
        self.assertGreater(profile.cost_usd.mean, 0)
        self.assertGreater(profile.cost_usd.stddev, 0)
        self.assertAlmostEqual(profile.cost_usd.mean, sum(costs) / len(costs), places=12)
        # Priced from the recorded usage at the default model, as `cost` does.
        self.assertAlmostEqual(profile.cost_usd.min, _dollars(1000, 10, DEFAULT_MODEL), places=12)
        self.assertAlmostEqual(profile.cost_usd.max, _dollars(1500, 10, DEFAULT_MODEL), places=12)

    def test_check_flags_cost_outlier(self):
        profile = build_baseline(self.store, since_days=30)
        outlier = _priced_session(self.store, 200_000, minutes_ago=1)
        result = check_session(self.store, outlier, profile, sigma=2.0)
        self.assertTrue(result.anomalous)
        self.assertGreater(result.deviations["cost_usd"], 2.0)
        others = {m: z for m, z in result.deviations.items() if m != "cost_usd"}
        self.assertEqual(set(others.values()), {0.0}, others)

    def test_typical_cost_is_not_flagged(self):
        profile = build_baseline(self.store, since_days=30)
        typical = _priced_session(self.store, 1250, minutes_ago=1)
        result = check_session(self.store, typical, profile, sigma=2.0)
        self.assertFalse(result.anomalous)
        self.assertLess(result.deviations["cost_usd"], 1.0)

    def test_unpriceable_session_adds_no_cost_sample(self):
        # `agent-strace cost` refuses a tool call whose data is not an object.
        _priced_session(self.store, 1000, minutes_ago=1, tool_call_data=["ls"])
        profile = build_baseline(self.store, since_days=30)
        self.assertEqual(profile.session_count, 7)
        self.assertEqual(profile.tool_calls.count, 7)
        self.assertEqual(profile.cost_usd.count, 6)
        self.assertAlmostEqual(profile.cost_usd.min, _dollars(1000, 10, DEFAULT_MODEL), places=12)

    def test_check_does_not_compare_cost_of_unpriceable_session(self):
        profile = build_baseline(self.store, since_days=30)
        sid = _priced_session(self.store, 200_000, minutes_ago=1, tool_call_data=["ls"])
        result = check_session(self.store, sid, profile, sigma=2.0)
        self.assertNotIn("cost_usd", result.deviations)
        self.assertIn("tool_calls", result.deviations)

    def test_pricing_error_is_not_recorded_as_zero_cost(self):
        with patch("agent_trace.baseline.estimate_cost", side_effect=TypeError("bug")):
            with self.assertRaises(TypeError):
                build_baseline(self.store, since_days=30)

    def test_cli_update_and_check_flag_cost_outlier(self):
        parser = build_parser()
        path = os.path.join(self.trace_dir, "baseline.json")
        with redirect_stdout(io.StringIO()):
            rc = cmd_baseline(parser.parse_args(
                ["--trace-dir", self.trace_dir, "baseline", "update", "--output", path]
            ))
        self.assertEqual(rc, 0)
        self.assertGreater(json.loads(Path(path).read_text())["cost_usd"]["mean"], 0)

        outlier = _priced_session(self.store, 200_000, minutes_ago=1)
        with patch.object(AnomalyResult, "format", autospec=True) as fmt:
            rc = cmd_baseline(parser.parse_args(
                ["--trace-dir", self.trace_dir, "baseline", "check", outlier, "--baseline", path]
            ))
        self.assertEqual(rc, 1)
        result = fmt.call_args.args[0]
        self.assertEqual(result.session_id, outlier)
        self.assertGreater(result.deviations["cost_usd"], result.sigma_threshold)


class TestBaselineFromCodexFixture(unittest.TestCase):
    """A baseline of replayed Codex hook recordings prices their recorded usage."""

    def setUp(self):
        env = patch.dict(os.environ, {"DO_NOT_TRACK": "1"}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        for name in (
            *_PROVIDER_ENV.values(),
            "AGENT_TRACE_REDACT",
            "AGENT_STRACE_TENANT_ID",
            "AGENT_STRACE_WORKSPACE",
            "AGENT_STRACE_ENDPOINT",
        ):
            os.environ.pop(name, None)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        trace_dir = str(Path(tmp.name) / ".agent-traces")
        os.environ["AGENT_TRACE_DIR"] = trace_dir
        registered = registered_hooks("codex")
        self.recorded: dict[str, tuple[int, int]] = {}
        for scenario in load_fixture(FIXTURES["codex"])["scenarios"].values():
            for hook in scenario["hooks"]:
                event, payload = hook["event"], hook["payload"]
                usage = payload.get("token_usage") if event == "PostModelRequest" else None
                if usage:
                    inp, out = self.recorded.get(payload["session_id"], (0, 0))
                    self.recorded[payload["session_id"]] = (
                        inp + usage["input_tokens"], out + usage["output_tokens"],
                    )
                if event not in registered or not matcher_applies(
                    registered[event][0], event, payload
                ):
                    continue
                stdin = io.StringIO(json.dumps(payload))
                with patch.object(sys, "stdin", stdin), redirect_stderr(io.StringIO()):
                    hook_main(["--provider", "codex", registered[event][1]])
        self.store = TraceStore(trace_dir)

    def test_baseline_cost_matches_cost_command(self):
        sessions = self.store.list_sessions()
        self.assertEqual(len(sessions), 3)
        costs = [estimate_cost(self.store, m.session_id).total_cost for m in sessions]
        expected = [_dollars(inp, out, DEFAULT_MODEL) for inp, out in self.recorded.values()]
        for cost, want in zip(sorted(costs), sorted(expected)):
            self.assertAlmostEqual(cost, want, places=12)

        profile = build_baseline(self.store, since_days=30)
        self.assertEqual(profile.session_count, 3)
        self.assertEqual(profile.cost_usd.count, 3)
        self.assertGreater(profile.cost_usd.min, 0)
        self.assertAlmostEqual(profile.cost_usd.mean, sum(costs) / 3, places=12)
        self.assertAlmostEqual(profile.cost_usd.min, min(costs), places=12)
        self.assertAlmostEqual(profile.cost_usd.max, max(costs), places=12)


if __name__ == "__main__":
    unittest.main()
