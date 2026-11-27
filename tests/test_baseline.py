"""Tests for baseline anomaly detection (issue #134)."""

import io
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from functools import partialmethod
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
from agent_trace.cost import DEFAULT_MODEL, TokenUsage, _usage_dollars, estimate_cost
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
    from test_unusable_event_timestamps import restamp, set_meta
except ImportError:
    from tests.test_capture_matrix_fixtures import (
        FIXTURES,
        load_fixture,
        matcher_applies,
        registered_hooks,
    )
    from tests.test_cost import _model_request
    from tests.test_unusable_event_timestamps import restamp, set_meta

try:
    import _escapes
except ImportError:
    from tests import _escapes


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


def _priced(input_tokens: int) -> float:
    """The cost of one _priced_session request at the default model."""
    usage = TokenUsage(input_tokens, 10, cache_read_tokens=input_tokens // 2)
    return _usage_dollars(usage, DEFAULT_MODEL)


# Event timestamps that are not a Unix time in seconds. On main NaN and the
# time in milliseconds were priced, and the text raised TypeError.
UNUSABLE_STAMPS = (float("nan"), 1_795_111_200_000.0, "2026-01-01T00:00:00Z")


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
        # Half of each request's input was a cache read.
        self.assertAlmostEqual(profile.cost_usd.min, _priced(1000), places=12)
        self.assertAlmostEqual(profile.cost_usd.max, _priced(1500), places=12)

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
        self.assertAlmostEqual(profile.cost_usd.min, _priced(1000), places=12)

    def test_check_does_not_compare_cost_of_unpriceable_session(self):
        profile = build_baseline(self.store, since_days=30)
        sid = _priced_session(self.store, 200_000, minutes_ago=1, tool_call_data=["ls"])
        result = check_session(self.store, sid, profile, sigma=2.0)
        self.assertNotIn("cost_usd", result.deviations)
        self.assertIn("tool_calls", result.deviations)

    def test_session_with_unusable_timestamp_adds_no_cost_sample(self):
        # `agent-strace cost` refuses an event timestamp that is not a Unix
        # time in seconds.
        for value in UNUSABLE_STAMPS:
            with self.subTest(value=value):
                sid = _priced_session(self.store, 1000, minutes_ago=1)
                restamp(Path(self.trace_dir) / sid, "ALL", value)
                try:
                    profile = build_baseline(self.store, since_days=30)
                finally:
                    shutil.rmtree(Path(self.trace_dir) / sid)
                self.assertEqual(profile.session_count, 7)
                self.assertEqual(profile.tool_calls.count, 7)
                self.assertEqual(profile.cost_usd.count, 6)

    def test_check_does_not_compare_cost_of_session_with_unusable_timestamp(self):
        profile = build_baseline(self.store, since_days=30)
        for value in UNUSABLE_STAMPS:
            with self.subTest(value=value):
                sid = _priced_session(self.store, 200_000, minutes_ago=1)
                restamp(Path(self.trace_dir) / sid, "ALL", value)
                result = check_session(self.store, sid, profile, sigma=2.0)
                self.assertNotIn("cost_usd", result.deviations)
                self.assertIn("tool_calls", result.deviations)

    def test_cli_update_leaves_out_cost_of_session_with_text_timestamp(self):
        sid = _priced_session(self.store, 1000, minutes_ago=1)
        restamp(Path(self.trace_dir) / sid, "ALL", "2026-01-01T00:00:00Z")
        path = os.path.join(self.trace_dir, "baseline.json")
        err = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(err):
            rc = cmd_baseline(build_parser().parse_args(
                ["--trace-dir", self.trace_dir, "baseline", "update", "--output", path]
            ))
        self.assertEqual((rc, err.getvalue()), (0, ""))
        saved = json.loads(Path(path).read_text())
        self.assertEqual(saved["session_count"], 7)
        self.assertEqual(saved["cost_usd"]["count"], 6)

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


class TestOpenSessions(unittest.TestCase):
    """A session with no end time is scored on what it recorded so far."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.trace_dir = tmp.name
        self.store = TraceStore(tmp.name)
        for i, calls in enumerate((10, 11, 12, 13, 14)):
            meta = SessionMeta(agent_name="normal")
            meta.tool_calls = calls
            meta.llm_requests = 2
            meta.started_at = time.time() - 3600 * (i + 2)
            meta.ended_at = meta.started_at + 60 + i
            meta.total_duration_ms = (60 + i) * 1000.0
            self.store.create_session(meta)
            self.store.update_meta(meta)
        self.profile = build_baseline(self.store, since_days=30)

    def _open_session(self, tool_calls=0, tool_call_events=0, span_s=30.0) -> str:
        meta = SessionMeta(agent_name="killed")
        meta.tool_calls = tool_calls
        meta.started_at = time.time() - 600
        self.store.create_session(meta)
        sid = meta.session_id
        self.store.append_event(sid, TraceEvent(
            event_type=EventType.SESSION_START, session_id=sid,
            timestamp=meta.started_at,
        ))
        for n in range(tool_call_events):
            self.store.append_event(sid, TraceEvent(
                event_type=EventType.TOOL_CALL, session_id=sid,
                timestamp=meta.started_at + 1 + n * 0.001,
                data={"tool_name": "Bash", "arguments": {"command": "ls"}},
            ))
        self.store.append_event(sid, TraceEvent(
            event_type=EventType.ASSISTANT_RESPONSE, session_id=sid,
            timestamp=meta.started_at + span_s, data={"text": "working"},
        ))
        self.store.update_meta(meta)
        return sid

    def _cli_check(self, sid: str) -> tuple[int, AnomalyResult, str]:
        path = os.path.join(self.trace_dir, "baseline.json")
        save_baseline(self.profile, path)
        args = build_parser().parse_args(
            ["--trace-dir", self.trace_dir, "baseline", "check", sid, "--baseline", path]
        )
        with patch.object(AnomalyResult, "format", autospec=True) as fmt:
            rc = cmd_baseline(args)
        result = fmt.call_args.args[0]
        out = io.StringIO()
        result.format(out=out)
        return rc, result, out.getvalue()

    def test_runaway_open_session_is_flagged(self):
        self.assertEqual(self.profile.session_count, 5)
        sid = self._open_session(tool_calls=5000)
        result = check_session(self.store, sid, self.profile, sigma=2.0)
        self.assertTrue(result.scored)
        self.assertTrue(result.open_session)
        self.assertTrue(result.anomalous)
        self.assertGreater(result.deviations["tool_calls"], 2.0)

    def test_tool_call_events_count_when_counters_were_never_written(self):
        # A proxy killed mid-run never writes its in-memory counters.
        sid = self._open_session(tool_calls=0, tool_call_events=200)
        result = check_session(self.store, sid, self.profile, sigma=2.0)
        self.assertTrue(result.anomalous)
        self.assertGreater(result.deviations["tool_calls"], 2.0)

    def test_duration_runs_to_last_event(self):
        sid = self._open_session(tool_calls=12, span_s=3600.0)
        result = check_session(self.store, sid, self.profile, sigma=2.0)
        self.assertTrue(result.anomalous)
        self.assertGreater(result.deviations["duration_ms"], 2.0)
        self.assertEqual(result.deviations["tool_calls"], 0.0)

    def test_unusable_times_give_no_duration(self):
        # On main 10**400 raised OverflowError. The time in milliseconds gave
        # a duration of about 57,000 years, and a started_at of -5 one of
        # about 57 years.
        cases = (
            ("timestamp", 10**400), ("timestamp", 1_795_111_200_000.0),
            ("started_at", 10**400), ("started_at", -5),
        )
        for field_name, value in cases:
            with self.subTest(field=field_name, value=value):
                sid = self._open_session(tool_calls=12, span_s=3600.0)
                if field_name == "timestamp":
                    restamp(Path(self.trace_dir) / sid, "ALL", value)
                else:
                    set_meta(Path(self.trace_dir) / sid, started_at=value)
                result = check_session(self.store, sid, self.profile, sigma=2.0)
                self.assertTrue(result.scored)
                self.assertTrue(result.open_session)
                self.assertFalse(result.anomalous)
                self.assertNotIn("duration_ms", result.deviations)
                self.assertEqual(result.deviations["tool_calls"], 0.0)

    def test_session_below_the_mean_so_far_is_not_flagged(self):
        sid = self._open_session(tool_calls=1, span_s=5.0)
        result = check_session(self.store, sid, self.profile, sigma=2.0)
        self.assertTrue(result.scored)
        self.assertTrue(result.open_session)
        self.assertFalse(result.anomalous)
        self.assertEqual(result.deviations["tool_calls"], 0.0)
        self.assertEqual(result.deviations["duration_ms"], 0.0)

    def test_cli_exits_1_and_marks_open_session(self):
        sid = self._open_session(tool_calls=5000)
        rc, result, text = self._cli_check(sid)
        self.assertEqual(rc, 1)
        self.assertIn("[ANOMALOUS]", text)
        self.assertIn("open session", text)

    def test_cli_marks_typical_open_session(self):
        sid = self._open_session(tool_calls=1, span_s=5.0)
        rc, result, text = self._cli_check(sid)
        self.assertEqual(rc, 0)
        self.assertIn("[OK]  open session", text)
        self.assertIn("No end time recorded", text)

    def test_unmeasurable_session_is_unscored_not_ok(self):
        sid = self._open_session(tool_calls=5000)
        meta_path = Path(self.trace_dir) / sid / "meta.json"
        raw = json.loads(meta_path.read_text())
        raw["tool_calls"] = "many"
        meta_path.write_text(json.dumps(raw))
        rc, result, text = self._cli_check(sid)
        self.assertEqual(rc, 1)
        self.assertFalse(result.scored)
        self.assertFalse(result.anomalous)
        self.assertIn("[UNSCORED]", text)
        self.assertIn("non-numeric counters", text)
        self.assertNotIn("[OK]", text)


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
        self.recorded: dict[str, TokenUsage] = {}
        for scenario in load_fixture(FIXTURES["codex"])["scenarios"].values():
            for hook in scenario["hooks"]:
                event, payload = hook["event"], hook["payload"]
                usage = payload.get("token_usage") if event == "PostModelRequest" else None
                if usage:
                    total = self.recorded.get(payload["session_id"], TokenUsage())
                    self.recorded[payload["session_id"]] = total + TokenUsage(
                        input_tokens=usage["input_tokens"],
                        output_tokens=usage["output_tokens"],
                        cache_read_tokens=usage["cached_input_tokens"],
                        cache_write_tokens=usage["cache_write_input_tokens"],
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
        expected = [_usage_dollars(usage, DEFAULT_MODEL) for usage in self.recorded.values()]
        for cost, want in zip(sorted(costs), sorted(expected)):
            self.assertAlmostEqual(cost, want, places=12)

        profile = build_baseline(self.store, since_days=30)
        self.assertEqual(profile.session_count, 3)
        self.assertEqual(profile.cost_usd.count, 3)
        self.assertGreater(profile.cost_usd.min, 0)
        self.assertAlmostEqual(profile.cost_usd.mean, sum(costs) / 3, places=12)
        self.assertAlmostEqual(profile.cost_usd.min, min(costs), places=12)
        self.assertAlmostEqual(profile.cost_usd.max, max(costs), places=12)


# Python 3.14 limits JSON nesting by stack size, not by a count, so it can
# parse 100000 levels. A million levels raise RecursionError on 3.10+.
TOO_DEEP = 1_000_000
NOT_FINITE = "it has a number that is not finite"
NOT_JSON = ("it is not valid JSON "
            "(Expecting property name enclosed in double quotes at line 1 column 2)")


def _nested(depth):
    """A profile file that nests depth levels: an object around lists."""
    return b'{"session_count": ' + b"[" * (depth - 1) + b"]" * (depth - 1) + b"}"


REFUSED_PROFILES = [
    ("empty", b"", "it is not valid JSON (Expecting value at line 1 column 1)"),
    ("bad JSON", b"{", NOT_JSON),
    ("not UTF-8", b"\xff\xfe{}", "it is not valid UTF-8"),
    ("list", b"[]", "it is not a JSON object"),
    ("null", b"null", "it is not a JSON object"),
    ("number", b"5", "it is not a JSON object"),
    ("deep list", b"[" * 129 + b"]" * 129, "it is nested too deeply"),
    ("depth 129", _nested(129), "it is nested too deeply"),
    ("too deep to parse", b"[" * TOO_DEEP + b"]" * TOO_DEEP, "it is nested too deeply"),
    ("NaN", b'{"tool_calls": {"mean": NaN}}', NOT_FINITE),
    ("Infinity", b'{"cost_usd": {"stddev": Infinity}}', NOT_FINITE),
    ("-Infinity", b'{"created_at": -Infinity}', NOT_FINITE),
    ("float too large", b'{"duration_ms": {"max": 1e400}}', NOT_FINITE),
    ("4301 digits", b'{"session_count": 1' + b"0" * 4300 + b"}",
     "it has a number with too many digits"),
    ("metric number", b'{"tool_calls": 5}', "tool_calls is not an object"),
    ("metric null", b'{"cost_usd": null}', "cost_usd is not an object"),
    ("metric list", b'{"llm_requests": []}', "llm_requests is not an object"),
    ("mean text", b'{"tool_calls": {"mean": "x"}}', "tool_calls.mean is not a finite number"),
    ("mean null", b'{"duration_ms": {"mean": null}}',
     "duration_ms.mean is not a finite number"),
    ("mean bool", b'{"error_rate": {"mean": true}}', "error_rate.mean is not a finite number"),
    ("mean too large", b'{"tool_calls": {"mean": 1' + b"0" * 400 + b"}}",
     "tool_calls.mean is not a finite number"),
    ("min text", b'{"cost_usd": {"min": "x"}}', "cost_usd.min is not a finite number"),
    ("max null", b'{"llm_requests": {"max": null}}', "llm_requests.max is not a finite number"),
    ("stddev text", b'{"tool_calls": {"stddev": "x"}}',
     "tool_calls.stddev is not a number of 0 or more"),
    ("stddev negative", b'{"tool_calls": {"stddev": -1.5}}',
     "tool_calls.stddev is not a number of 0 or more"),
    ("stddev bool", b'{"duration_ms": {"stddev": true}}',
     "duration_ms.stddev is not a number of 0 or more"),
    ("stddev too large", b'{"cost_usd": {"stddev": 1' + b"0" * 400 + b"}}",
     "cost_usd.stddev is not a number of 0 or more"),
    ("count text", b'{"tool_calls": {"count": "x"}}',
     "tool_calls.count is not a number of 0 or more"),
    ("count null", b'{"error_rate": {"count": null}}',
     "error_rate.count is not a number of 0 or more"),
    ("count bool", b'{"tool_calls": {"count": true}}',
     "tool_calls.count is not a number of 0 or more"),
    ("count negative", b'{"llm_requests": {"count": -5}}',
     "llm_requests.count is not a number of 0 or more"),
]

ACCEPTED_PROFILES = [
    ("empty object", b"{}"),
    ("depth 128", _nested(128)),
    ("4300 digits and a sign", b'{"session_count": -1' + b"0" * 4299 + b"}"),
    ("negative stats", b'{"tool_calls": {"mean": -12, "stddev": 1, "min": -14, "max": -10, '
     b'"count": 5}}'),
    ("fractional count", b'{"tool_calls": {"mean": 12, "stddev": 1.5, "count": 2.5}}'),
    ("float count", b'{"tool_calls": {"mean": 12, "stddev": 1.5, "count": 5.0}}'),
    ("zero stddev", b'{"tool_calls": {"mean": 12, "stddev": 0, "count": 5}}'),
    ("any created_at and session_count",
     b'{"created_at": "x", "session_count": {"n": [true, null]}}'),
    ("huge numbers", b'{"duration_ms": {"mean": 1e308, "stddev": 1e308, "min": -1e308, '
     b'"max": 1e308, "count": 1e308}}'),
    ("extra keys", b'{"note": "x", "tool_calls": {"mean": 12, "note": "x"}}'),
]


def _open_error(path):
    """The reason open() gives for path, or None when it opens."""
    try:
        with open(path, "rb"):
            return None
    except OSError as exc:
        return exc.strerror


class TestUnreadableBaselineFile(unittest.TestCase):
    """check and show read a profile file the same way. When they cannot read
    one, they print one line on stderr that names it, print nothing on stdout
    and exit 1."""

    def setUp(self):
        store_dir = tempfile.TemporaryDirectory()
        self.addCleanup(store_dir.cleanup)
        self.trace_dir = store_dir.name
        files_dir = tempfile.TemporaryDirectory()
        self.addCleanup(files_dir.cleanup)
        self.files = Path(files_dir.name)
        store = TraceStore(self.trace_dir)
        for i, calls in enumerate((10, 11, 12, 13, 14)):
            meta = SessionMeta(agent_name="normal")
            meta.tool_calls = calls
            meta.started_at = time.time() - 3600 * (i + 2)
            meta.ended_at = meta.started_at + 60
            store.create_session(meta)
            store.update_meta(meta)
        self.sid = meta.session_id
        self.good = self.files / "good.json"
        save_baseline(build_baseline(store, since_days=30), str(self.good))

    def _run(self, argv):
        args = build_parser().parse_args(["--trace-dir", self.trace_dir, "baseline", *argv])
        out, err = io.StringIO(), io.StringIO()
        # AnomalyResult.format writes to the sys.stdout that baseline.py saw on
        # import. Point it at the buffer.
        with patch("sys.stdout", out), patch("sys.stderr", err), \
                patch.object(AnomalyResult, "format",
                             partialmethod(AnomalyResult.format, out=out)):
            code = cmd_baseline(args)
        return code, out.getvalue(), err.getvalue()

    def _check(self, path):
        return self._run(["check", self.sid, "--baseline", path])

    def _show(self, path):
        return self._run(["show", "--baseline", path])

    def _forms(self, path):
        return {"check": lambda: self._check(path), "show": lambda: self._show(path)}

    def _file(self, name, data):
        path = self.files / name
        path.write_bytes(data)
        return str(path)

    def _open_reason(self, path):
        reason = _open_error(path)
        self.assertIsNotNone(reason, "the test file opened")
        return reason

    def _assert_refused(self, label, path, reason, shown=None):
        shown = path if shown is None else shown
        line = f"Could not read baseline file {shown}: {reason}\n"
        for form, run in self._forms(path).items():
            with self.subTest(case=label, form=form):
                self.assertEqual(run(), (1, "", line))

    def test_refuses_a_file_it_cannot_use(self):
        for label, data, reason in REFUSED_PROFILES:
            self._assert_refused(label, self._file("baseline.json", data), reason)

    def test_reads_a_file_it_can_use(self):
        for label, data in ACCEPTED_PROFILES:
            path = self._file("baseline.json", data)
            for form, run in self._forms(path).items():
                with self.subTest(case=label, form=form):
                    code, out, err = run()
                    self.assertIn(code, (0, 1))
                    self.assertEqual(err, "")
                    self.assertNotEqual(out, "")

    def test_no_file_at_the_path_keeps_the_not_found_lines(self):
        missing = self.files / "missing.json"
        paths = {"missing": str(missing)}
        link = self.files / "dangling.json"
        try:
            link.symlink_to(missing)
        except OSError:
            pass
        else:
            paths["dangling symlink"] = str(link)
        for label, path in paths.items():
            with self.subTest(case=label):
                self.assertEqual(self._check(path), (
                    1, "", f"No baseline found at {path}. Run: agent-strace baseline update\n"))
                self.assertEqual(self._show(path), (1, "", f"No baseline found at {path}\n"))

    def test_refuses_a_path_it_cannot_open(self):
        folder = self.files / "dir"
        folder.mkdir()
        cases = {"directory": folder, "path through a file": self.good / "baseline.json"}
        loop = self.files / "loop.json"
        try:
            loop.symlink_to(loop)
        except OSError:
            pass
        else:
            cases["symlink loop"] = loop
        for label, path in cases.items():
            self._assert_refused(label, str(path), self._open_reason(path))

    def test_refuses_a_path_it_has_no_permission_to_read(self):
        locked = Path(self._file("locked.json", self.good.read_bytes()))
        folder = self.files / "locked"
        folder.mkdir()
        inside = folder / "baseline.json"
        inside.write_bytes(self.good.read_bytes())
        for path in (locked, folder):
            path.chmod(0)
            self.addCleanup(path.chmod, 0o700)
        if _open_error(locked) is None:
            self.skipTest("this user can read a file with mode 0")
        for label, path in (("mode 0 file", locked), ("file in a mode 0 directory", inside)):
            self._assert_refused(label, str(path), self._open_reason(path))

    def test_empty_path(self):
        self._assert_refused("empty path", "", "the path is empty", shown="''")

    def test_path_shows_escapes_as_text(self):
        path = self.files / f"bad{_escapes.PAYLOAD}.json"
        try:
            path.write_bytes(b"{")
        except OSError:
            self.skipTest("this file system refuses control characters in a file name")
        shown = f"{self.files}/bad{_escapes.PAYLOAD_TEXT}.json"
        self._assert_refused("escapes in the path", str(path), NOT_JSON, shown=shown)

    def test_reads_a_file_that_starts_with_a_byte_order_mark(self):
        good = self._forms(str(self.good))
        bom = self._forms(self._file("bom.json", b"\xef\xbb\xbf" + self.good.read_bytes()))
        for form, run in good.items():
            with self.subTest(form=form):
                expected = run()
                self.assertEqual(expected[2], "")
                self.assertEqual(bom[form](), expected)

    def test_show_says_build_time_unknown_for_an_unusable_created_at(self):
        for label, value in (("null", b"null"), ("bool", b"true"), ("negative", b"-1"),
                             ("text", b'"x"'), ("too late", b"1e300")):
            path = self._file("baseline.json",
                              b'{"created_at": ' + value + b', "session_count": 5}')
            with self.subTest(created_at=label):
                code, out, err = self._show(path)
                self.assertEqual((code, err), (0, ""))
                self.assertIn("\nBaseline (5 sessions, build time unknown)\n", out)

    def test_show_prints_a_build_time_of_0(self):
        built = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(0))
        for label, data in (("0", b'{"created_at": 0, "session_count": 5}'),
                            ("missing", b'{"session_count": 5}')):
            with self.subTest(created_at=label):
                code, out, err = self._show(self._file("baseline.json", data))
                self.assertEqual((code, err), (0, ""))
                self.assertIn(f"\nBaseline (5 sessions, built {built})\n", out)

    def test_show_prints_escapes_in_the_session_count_as_text(self):
        data = json.dumps({"session_count": _escapes.PAYLOAD}).encode()
        code, out, err = self._show(self._file("baseline.json", data))
        self.assertEqual((code, err), (0, ""))
        self.assertIn(f"\nBaseline ({_escapes.PAYLOAD_TEXT} sessions, built ", out)


if __name__ == "__main__":
    unittest.main()
