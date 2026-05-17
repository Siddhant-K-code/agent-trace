"""Tests for Langfuse and OTLP metrics export."""

import base64
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent_trace.langfuse_export import (
    EvalScore,
    LangfuseConfig,
    OtlpMetricsConfig,
    _events_to_langfuse_observations,
    _iso,
    _load_eval_scores,
    _otlp_gauge,
    _scores_to_langfuse,
    _session_metrics,
    _session_to_langfuse_trace,
    export_metrics_to_otlp,
    export_session_to_langfuse,
)
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(tmp: str) -> TraceStore:
    return TraceStore(tmp)


def _add_session(
    store: TraceStore,
    session_id: str,
    events: list[TraceEvent] | None = None,
    started_at: float | None = None,
    total_tokens: int = 1000,
    total_duration_ms: float = 60_000,
) -> SessionMeta:
    ts = started_at or time.time()
    meta = SessionMeta(
        session_id=session_id,
        started_at=ts,
        ended_at=ts + 60,
        agent_name="test-agent",
        total_tokens=total_tokens,
        total_duration_ms=total_duration_ms,
    )
    store.create_session(meta)
    for ev in (events or []):
        store.append_event(session_id, ev)
    return meta


def _write_eval_json(store: TraceStore, session_id: str, results: list[dict]) -> None:
    path = store.base_dir / session_id / "eval.json"
    path.write_text(json.dumps({"results": results}))


def _tool_call(name: str, ts: float = 0.0, event_id: str = "tc1") -> TraceEvent:
    return TraceEvent(
        event_type=EventType.TOOL_CALL,
        timestamp=ts,
        event_id=event_id,
        data={"tool_name": name, "arguments": {"path": "src/a.py"}},
    )


def _tool_result(name: str, ts: float = 1.0, parent_id: str = "tc1") -> TraceEvent:
    return TraceEvent(
        event_type=EventType.TOOL_RESULT,
        timestamp=ts,
        parent_id=parent_id,
        data={"tool_name": name, "result": "ok"},
    )


def _llm_req(ts: float = 0.0, event_id: str = "lr1") -> TraceEvent:
    return TraceEvent(
        event_type=EventType.LLM_REQUEST,
        timestamp=ts,
        event_id=event_id,
        data={"model": "claude-3", "input_tokens": 100},
    )


def _llm_resp(ts: float = 1.0, parent_id: str = "lr1") -> TraceEvent:
    return TraceEvent(
        event_type=EventType.LLM_RESPONSE,
        timestamp=ts,
        parent_id=parent_id,
        data={"output_tokens": 50, "text": "done"},
    )


def _error_ev(ts: float = 0.0) -> TraceEvent:
    return TraceEvent(
        event_type=EventType.ERROR,
        timestamp=ts,
        data={"message": "exit 1"},
    )


def _write_ev(path: str, ts: float = 0.0) -> TraceEvent:
    return TraceEvent(
        event_type=EventType.FILE_WRITE,
        timestamp=ts,
        data={"path": path},
    )


# ---------------------------------------------------------------------------
# LangfuseConfig
# ---------------------------------------------------------------------------

class TestLangfuseConfig(unittest.TestCase):
    def test_configured_true_when_keys_set(self):
        cfg = LangfuseConfig(public_key="pk", secret_key="sk")
        self.assertTrue(cfg.configured)

    def test_configured_false_when_keys_missing(self):
        cfg = LangfuseConfig()
        self.assertFalse(cfg.configured)

    def test_auth_header_is_basic(self):
        cfg = LangfuseConfig(public_key="pk", secret_key="sk")
        self.assertTrue(cfg.auth_header.startswith("Basic "))
        decoded = base64.b64decode(cfg.auth_header[6:]).decode()
        self.assertEqual(decoded, "pk:sk")


# ---------------------------------------------------------------------------
# Eval score loading
# ---------------------------------------------------------------------------

class TestLoadEvalScores(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = _make_store(self.tmp)

    def test_no_eval_json_returns_empty(self):
        _add_session(self.store, "s1")
        scores = _load_eval_scores(self.store, "s1")
        self.assertEqual(scores, [])

    def test_eval_json_parsed(self):
        _add_session(self.store, "s2")
        _write_eval_json(self.store, "s2", [
            {"scorer": "no_errors", "score": 1.0, "threshold": 1.0, "passed": True},
            {"scorer": "cost_under", "score": 0.8, "threshold": 0.9, "passed": False},
        ])
        scores = _load_eval_scores(self.store, "s2")
        self.assertEqual(len(scores), 2)
        self.assertEqual(scores[0].judge, "no_errors")
        self.assertEqual(scores[0].score, 1.0)
        self.assertTrue(scores[0].passed)
        self.assertFalse(scores[1].passed)

    def test_malformed_eval_json_returns_empty(self):
        _add_session(self.store, "s3")
        (self.store.base_dir / "s3" / "eval.json").write_text("not json")
        scores = _load_eval_scores(self.store, "s3")
        self.assertEqual(scores, [])


# ---------------------------------------------------------------------------
# Langfuse trace / observation building
# ---------------------------------------------------------------------------

class TestSessionToLangfuseTrace(unittest.TestCase):
    def test_trace_has_required_fields(self):
        meta = SessionMeta(
            session_id="abc123",
            started_at=1748000000.0,
            agent_name="claude",
            total_tokens=500,
        )
        body = _session_to_langfuse_trace(meta, [])
        self.assertEqual(body["id"], "abc123")
        self.assertIn("name", body)
        self.assertIn("timestamp", body)
        self.assertIn("metadata", body)

    def test_metadata_contains_token_count(self):
        meta = SessionMeta(session_id="x", started_at=time.time(), total_tokens=1234)
        body = _session_to_langfuse_trace(meta, [])
        self.assertEqual(body["metadata"]["total_tokens"], 1234)


class TestEventsToLangfuseObservations(unittest.TestCase):
    def test_tool_call_result_pair_becomes_span(self):
        events = [_tool_call("Bash", ts=0.0, event_id="tc1"), _tool_result("Bash", ts=1.0, parent_id="tc1")]
        obs = _events_to_langfuse_observations("sess1", events)
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0]["type"], "SPAN")
        self.assertIn("tool/Bash", obs[0]["name"])

    def test_llm_request_response_pair_becomes_generation(self):
        events = [_llm_req(ts=0.0, event_id="lr1"), _llm_resp(ts=1.0, parent_id="lr1")]
        obs = _events_to_langfuse_observations("sess1", events)
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0]["type"], "GENERATION")
        self.assertEqual(obs[0]["usage"]["input"], 100)
        self.assertEqual(obs[0]["usage"]["output"], 50)

    def test_unmatched_tool_call_produces_no_observation(self):
        events = [_tool_call("Bash", ts=0.0, event_id="tc_orphan")]
        obs = _events_to_langfuse_observations("sess1", events)
        self.assertEqual(obs, [])

    def test_empty_events_returns_empty(self):
        obs = _events_to_langfuse_observations("sess1", [])
        self.assertEqual(obs, [])


class TestScoresToLangfuse(unittest.TestCase):
    def test_scores_mapped_correctly(self):
        scores = [
            EvalScore(judge="no_errors", score=1.0, passed=True, threshold=1.0),
            EvalScore(judge="cost_under", score=0.7, passed=False, threshold=0.9),
        ]
        bodies = _scores_to_langfuse("sess1", scores)
        self.assertEqual(len(bodies), 2)
        self.assertEqual(bodies[0]["name"], "no_errors")
        self.assertEqual(bodies[0]["value"], 1.0)
        self.assertEqual(bodies[0]["traceId"], "sess1")
        self.assertEqual(bodies[1]["name"], "cost_under")

    def test_empty_scores_returns_empty(self):
        self.assertEqual(_scores_to_langfuse("sess1", []), [])


# ---------------------------------------------------------------------------
# OTLP gauge building
# ---------------------------------------------------------------------------

class TestOtlpGauge(unittest.TestCase):
    def test_gauge_structure(self):
        g = _otlp_gauge("agent_strace.session.cost_usd", 0.042, {"session_id": "abc"}, 1748000000000)
        self.assertEqual(g["name"], "agent_strace.session.cost_usd")
        self.assertIn("gauge", g)
        dp = g["gauge"]["dataPoints"][0]
        self.assertAlmostEqual(dp["asDouble"], 0.042)
        self.assertEqual(dp["timeUnixNano"], "1748000000000")

    def test_attributes_encoded(self):
        g = _otlp_gauge("m", 1.0, {"k": "v"}, 0)
        attrs = g["gauge"]["dataPoints"][0]["attributes"]
        self.assertEqual(attrs[0]["key"], "k")
        self.assertEqual(attrs[0]["value"]["stringValue"], "v")


# ---------------------------------------------------------------------------
# Session metrics extraction
# ---------------------------------------------------------------------------

class TestSessionMetrics(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = _make_store(self.tmp)

    def test_cost_computed_from_tokens(self):
        meta = _add_session(self.store, "s1", total_tokens=1_000_000)
        m = _session_metrics(self.store, meta)
        self.assertAlmostEqual(m["agent_strace.session.cost_usd"], 3.0)

    def test_error_rate_computed(self):
        events = [_tool_call("Bash"), _error_ev()]
        meta = _add_session(self.store, "s2", events=events)
        m = _session_metrics(self.store, meta)
        self.assertGreater(m["agent_strace.session.error_rate"], 0.0)

    def test_blast_radius_from_writes(self):
        events = [_write_ev("a.py"), _write_ev("b.py"), _write_ev("a.py")]
        meta = _add_session(self.store, "s3", events=events)
        m = _session_metrics(self.store, meta)
        self.assertEqual(m["agent_strace.session.blast_radius"], 2.0)

    def test_all_metric_keys_present(self):
        meta = _add_session(self.store, "s4")
        m = _session_metrics(self.store, meta)
        expected_keys = {
            "agent_strace.session.cost_usd",
            "agent_strace.session.error_rate",
            "agent_strace.session.retry_rate",
            "agent_strace.session.blast_radius",
            "agent_strace.session.duration_s",
            "agent_strace.session.tool_calls",
        }
        self.assertEqual(set(m.keys()), expected_keys)


# ---------------------------------------------------------------------------
# ISO timestamp helper
# ---------------------------------------------------------------------------

class TestIso(unittest.TestCase):
    def test_returns_z_suffix(self):
        s = _iso(1748000000.0)
        self.assertTrue(s.endswith("Z"))

    def test_parseable(self):
        from datetime import datetime
        s = _iso(1748000000.0)
        # Should not raise
        datetime.fromisoformat(s.replace("Z", "+00:00"))


# ---------------------------------------------------------------------------
# Integration: export_session_to_langfuse (mocked HTTP)
# ---------------------------------------------------------------------------

class TestExportSessionToLangfuse(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = _make_store(self.tmp)

    def _mock_urlopen(self, status: int = 200):
        mock_resp = MagicMock()
        mock_resp.status = status
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    def test_export_calls_langfuse_endpoint(self):
        events = [_tool_call("Bash", event_id="tc1"), _tool_result("Bash", parent_id="tc1")]
        _add_session(self.store, "sess_lf", events=events)
        cfg = LangfuseConfig(public_key="pk", secret_key="sk")

        with patch("urllib.request.urlopen", return_value=self._mock_urlopen(200)) as mock_open:
            result = export_session_to_langfuse(self.store, "sess_lf", cfg, include_scores=False)

        self.assertTrue(result)
        mock_open.assert_called_once()
        call_args = mock_open.call_args[0][0]
        self.assertIn("/api/public/ingestion", call_args.full_url)

    def test_export_includes_scores_when_present(self):
        _add_session(self.store, "sess_scores")
        _write_eval_json(self.store, "sess_scores", [
            {"scorer": "no_errors", "score": 1.0, "threshold": 1.0, "passed": True}
        ])
        cfg = LangfuseConfig(public_key="pk", secret_key="sk")

        posted_bodies = []

        def _capture_urlopen(req, timeout=30):
            posted_bodies.append(json.loads(req.data))
            return self._mock_urlopen(200)

        with patch("urllib.request.urlopen", side_effect=_capture_urlopen):
            export_session_to_langfuse(self.store, "sess_scores", cfg, include_scores=True)

        batch = posted_bodies[0]["batch"]
        score_items = [b for b in batch if b["type"] == "score-create"]
        self.assertEqual(len(score_items), 1)
        self.assertEqual(score_items[0]["body"]["name"], "no_errors")

    def test_export_returns_false_on_http_error(self):
        _add_session(self.store, "sess_fail")
        cfg = LangfuseConfig(public_key="pk", secret_key="sk")

        import urllib.error
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("connection refused")):
            result = export_session_to_langfuse(self.store, "sess_fail", cfg)

        self.assertFalse(result)

    def test_missing_session_returns_false(self):
        cfg = LangfuseConfig(public_key="pk", secret_key="sk")
        result = export_session_to_langfuse(self.store, "nonexistent", cfg)
        self.assertFalse(result)


# ---------------------------------------------------------------------------
# Integration: export_metrics_to_otlp (mocked HTTP)
# ---------------------------------------------------------------------------

class TestExportMetricsToOtlp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = _make_store(self.tmp)

    def _mock_urlopen(self, status: int = 200):
        mock_resp = MagicMock()
        mock_resp.status = status
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    def test_metrics_posted_to_otlp_endpoint(self):
        _add_session(self.store, "m1", total_tokens=500)
        cfg = OtlpMetricsConfig(endpoint="http://localhost:4318")

        with patch("urllib.request.urlopen", return_value=self._mock_urlopen(200)) as mock_open:
            result = export_metrics_to_otlp(self.store, ["m1"], cfg, include_scores=False)

        self.assertTrue(result)
        mock_open.assert_called_once()
        call_args = mock_open.call_args[0][0]
        self.assertIn("/v1/metrics", call_args.full_url)

    def test_payload_contains_cost_metric(self):
        _add_session(self.store, "m2", total_tokens=1_000_000)
        cfg = OtlpMetricsConfig(endpoint="http://localhost:4318")

        posted_bodies = []

        def _capture(req, timeout=30):
            posted_bodies.append(json.loads(req.data))
            return self._mock_urlopen(200)

        with patch("urllib.request.urlopen", side_effect=_capture):
            export_metrics_to_otlp(self.store, ["m2"], cfg, include_scores=False)

        metrics = posted_bodies[0]["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]
        names = [m["name"] for m in metrics]
        self.assertIn("agent_strace.session.cost_usd", names)

    def test_eval_scores_included_as_metrics(self):
        _add_session(self.store, "m3")
        _write_eval_json(self.store, "m3", [
            {"scorer": "no_errors", "score": 1.0, "threshold": 1.0, "passed": True}
        ])
        cfg = OtlpMetricsConfig(endpoint="http://localhost:4318")

        posted_bodies = []

        def _capture(req, timeout=30):
            posted_bodies.append(json.loads(req.data))
            return self._mock_urlopen(200)

        with patch("urllib.request.urlopen", side_effect=_capture):
            export_metrics_to_otlp(self.store, ["m3"], cfg, include_scores=True)

        metrics = posted_bodies[0]["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]
        names = [m["name"] for m in metrics]
        self.assertIn("agent_strace.eval.score", names)

    def test_empty_session_list_returns_true(self):
        cfg = OtlpMetricsConfig(endpoint="http://localhost:4318")
        result = export_metrics_to_otlp(self.store, [], cfg)
        self.assertTrue(result)

    def test_http_error_returns_false(self):
        _add_session(self.store, "m4")
        cfg = OtlpMetricsConfig(endpoint="http://localhost:4318")

        import urllib.error
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            result = export_metrics_to_otlp(self.store, ["m4"], cfg)

        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
