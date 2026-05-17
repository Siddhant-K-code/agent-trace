"""Tests for eval extensions: LLM judge scorer, dataset auto-sampling, eval --ci baseline."""

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent_trace.eval.config import EvalConfig
from agent_trace.eval.dataset import DatasetEntry, add_entry, auto_populate, list_entries
from agent_trace.eval.runner import (
    EvalReport,
    _load_baseline,
    _save_baseline,
    _write_github_summary,
)
from agent_trace.eval.scorers import ScoreResult, run_scorer, score_llm_judge
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
        total_tokens=total_tokens,
        total_duration_ms=total_duration_ms,
    )
    store.create_session(meta)
    for ev in (events or []):
        store.append_event(session_id, ev)
    return meta


def _tool(name: str, ts: float = 0.0) -> TraceEvent:
    return TraceEvent(event_type=EventType.TOOL_CALL, timestamp=ts, data={"tool_name": name})


def _error(ts: float = 0.0) -> TraceEvent:
    return TraceEvent(event_type=EventType.ERROR, timestamp=ts, data={"message": "fail"})


def _write(path: str, ts: float = 0.0) -> TraceEvent:
    return TraceEvent(event_type=EventType.FILE_WRITE, timestamp=ts, data={"path": path})


def _make_report(scores: list[tuple[str, float, float, bool]]) -> EvalReport:
    """Build an EvalReport from (scorer, score, threshold, passed) tuples."""
    results = [
        ScoreResult(scorer=s, score=sc, threshold=th, passed=p, reason="")
        for s, sc, th, p in scores
    ]
    return EvalReport(
        session_id="test",
        results=results,
        config=EvalConfig.default(),
    )


# ---------------------------------------------------------------------------
# LLM judge scorer
# ---------------------------------------------------------------------------

class TestScoreLlmJudge(unittest.TestCase):
    def _events(self) -> list[TraceEvent]:
        return [_tool("Bash"), _tool("Read")]

    def _mock_urlopen(self, score: float = 0.9, reason: str = "looks good"):
        resp_body = json.dumps({
            "choices": [{"message": {"content": json.dumps({"score": score, "reason": reason})}}]
        }).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = resp_body
        mock_resp.status = 200
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    def test_returns_score_from_llm(self):
        with patch("urllib.request.urlopen", return_value=self._mock_urlopen(0.85)):
            result = score_llm_judge(
                self._events(),
                prompt="Did the agent complete the task?",
                base_url="http://localhost:11434/v1",
                api_key="test",
                model="llama3",
                threshold=0.80,
            )
        self.assertAlmostEqual(result.score, 0.85)
        self.assertTrue(result.passed)
        self.assertEqual(result.scorer, "llm_judge")

    def test_fails_below_threshold(self):
        with patch("urllib.request.urlopen", return_value=self._mock_urlopen(0.5)):
            result = score_llm_judge(
                self._events(),
                prompt="Did the agent complete the task?",
                base_url="http://localhost:11434/v1",
                api_key="test",
                threshold=0.80,
            )
        self.assertFalse(result.passed)

    def test_missing_credentials_returns_failure(self):
        result = score_llm_judge(
            self._events(),
            prompt="test",
            base_url="",
            api_key="",
        )
        self.assertEqual(result.score, 0.0)
        self.assertFalse(result.passed)
        self.assertIn("required", result.reason)

    def test_http_error_returns_failure(self):
        import urllib.error
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            result = score_llm_judge(
                self._events(),
                prompt="test",
                base_url="http://localhost:11434/v1",
                api_key="key",
            )
        self.assertEqual(result.score, 0.0)
        self.assertFalse(result.passed)

    def test_malformed_json_response_returns_failure(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"choices": [{"message": {"content": "not json"}}]}'
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = score_llm_judge(
                self._events(),
                prompt="test",
                base_url="http://localhost:11434/v1",
                api_key="key",
            )
        self.assertEqual(result.score, 0.0)

    def test_score_clamped_to_0_1(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "choices": [{"message": {"content": json.dumps({"score": 1.5, "reason": "great"})}}]
        }).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = score_llm_judge(
                self._events(), prompt="test",
                base_url="http://localhost:11434/v1", api_key="key",
            )
        self.assertLessEqual(result.score, 1.0)

    def test_markdown_fences_stripped(self):
        content = "```json\n" + json.dumps({"score": 0.7, "reason": "ok"}) + "\n```"
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "choices": [{"message": {"content": content}}]
        }).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = score_llm_judge(
                self._events(), prompt="test",
                base_url="http://localhost:11434/v1", api_key="key",
            )
        self.assertAlmostEqual(result.score, 0.7)

    def test_dispatched_via_run_scorer(self):
        """llm_judge is reachable through the standard run_scorer dispatch."""
        import os
        events = self._events()
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "choices": [{"message": {"content": json.dumps({"score": 0.9, "reason": "ok"})}}]
        }).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = run_scorer(
                "llm_judge",
                {
                    "prompt": "Did the agent succeed?",
                    "base_url": "http://localhost:11434/v1",
                    "api_key": "key",
                    "model": "llama3",
                    "threshold": 0.8,
                },
                events,
            )
        self.assertEqual(result.scorer, "llm_judge")
        self.assertAlmostEqual(result.score, 0.9)


# ---------------------------------------------------------------------------
# Dataset auto-sampling
# ---------------------------------------------------------------------------

class TestDatasetAutoPopulate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = _make_store(self.tmp)
        self.ds_path = Path(self.tmp) / "test.jsonl"

    def test_has_errors_filter(self):
        now = time.time()
        _add_session(self.store, "s_err", [_error()], started_at=now - 100)
        _add_session(self.store, "s_ok", [_tool("Bash")], started_at=now - 100)
        added = auto_populate(self.store, self.ds_path, "has-errors", since_days=1)
        self.assertEqual(added, 1)
        entries = list_entries(self.ds_path)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].session_id, "s_err")

    def test_high_retry_filter(self):
        now = time.time()
        # 3 consecutive same-tool calls = 2 retries / 3 calls = 67% > 30%
        events = [_tool("Bash"), _tool("Bash"), _tool("Bash")]
        _add_session(self.store, "s_retry", events, started_at=now - 100)
        _add_session(self.store, "s_ok", [_tool("Bash"), _tool("Read")], started_at=now - 100)
        added = auto_populate(self.store, self.ds_path, "high-retry", since_days=1)
        self.assertEqual(added, 1)
        self.assertEqual(list_entries(self.ds_path)[0].session_id, "s_retry")

    def test_cost_above_filter(self):
        now = time.time()
        # 1M tokens * $3/M = $3.00 > $1.00
        _add_session(self.store, "s_expensive", total_tokens=1_000_000, started_at=now - 100)
        _add_session(self.store, "s_cheap", total_tokens=100, started_at=now - 100)
        added = auto_populate(self.store, self.ds_path, "cost-above:1.00", since_days=1)
        self.assertEqual(added, 1)
        self.assertEqual(list_entries(self.ds_path)[0].session_id, "s_expensive")

    def test_wide_blast_filter(self):
        now = time.time()
        events = [_write(f"src/file{i}.py") for i in range(11)]
        _add_session(self.store, "s_wide", events, started_at=now - 100)
        _add_session(self.store, "s_narrow", [_write("src/a.py")], started_at=now - 100)
        added = auto_populate(self.store, self.ds_path, "wide-blast", since_days=1)
        self.assertEqual(added, 1)
        self.assertEqual(list_entries(self.ds_path)[0].session_id, "s_wide")

    def test_long_duration_filter(self):
        now = time.time()
        _add_session(self.store, "s_long", total_duration_ms=600_000, started_at=now - 100)
        _add_session(self.store, "s_short", total_duration_ms=10_000, started_at=now - 100)
        added = auto_populate(self.store, self.ds_path, "long-duration:300s", since_days=1)
        self.assertEqual(added, 1)
        self.assertEqual(list_entries(self.ds_path)[0].session_id, "s_long")

    def test_low_eval_score_filter(self):
        now = time.time()
        _add_session(self.store, "s_bad", started_at=now - 100)
        _add_session(self.store, "s_good", started_at=now - 100)
        # Write eval.json for both
        (self.store.base_dir / "s_bad" / "eval.json").write_text(
            json.dumps({"results": [{"scorer": "x", "score": 0.3, "threshold": 1.0, "passed": False}]})
        )
        (self.store.base_dir / "s_good" / "eval.json").write_text(
            json.dumps({"results": [{"scorer": "x", "score": 0.95, "threshold": 1.0, "passed": True}]})
        )
        added = auto_populate(self.store, self.ds_path, "low-eval-score:0.5", since_days=1)
        self.assertEqual(added, 1)
        self.assertEqual(list_entries(self.ds_path)[0].session_id, "s_bad")

    def test_since_filter_excludes_old_sessions(self):
        now = time.time()
        _add_session(self.store, "s_old", [_error()], started_at=now - 86400 * 10)
        _add_session(self.store, "s_new", [_error()], started_at=now - 100)
        added = auto_populate(self.store, self.ds_path, "has-errors", since_days=1)
        self.assertEqual(added, 1)
        self.assertEqual(list_entries(self.ds_path)[0].session_id, "s_new")

    def test_no_duplicates_added(self):
        now = time.time()
        _add_session(self.store, "s_err", [_error()], started_at=now - 100)
        auto_populate(self.store, self.ds_path, "has-errors", since_days=1)
        added_second = auto_populate(self.store, self.ds_path, "has-errors", since_days=1)
        self.assertEqual(added_second, 0)
        self.assertEqual(len(list_entries(self.ds_path)), 1)

    def test_label_applied(self):
        now = time.time()
        _add_session(self.store, "s_err", [_error()], started_at=now - 100)
        auto_populate(self.store, self.ds_path, "has-errors", since_days=1, label="my-label")
        entries = list_entries(self.ds_path)
        self.assertEqual(entries[0].label, "my-label")

    def test_unknown_filter_returns_zero(self):
        now = time.time()
        _add_session(self.store, "s1", [_error()], started_at=now - 100)
        added = auto_populate(self.store, self.ds_path, "nonexistent-filter", since_days=1)
        self.assertEqual(added, 0)


# ---------------------------------------------------------------------------
# eval --ci baseline
# ---------------------------------------------------------------------------

class TestEvalCiBaseline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_save_and_load_baseline(self):
        report = _make_report([
            ("no_errors", 1.0, 1.0, True),
            ("cost_under", 0.8, 0.9, False),
        ])
        path = str(Path(self.tmp) / "baseline.json")
        _save_baseline(path, report)
        loaded = _load_baseline(path)
        self.assertAlmostEqual(loaded["no_errors"], 1.0)
        self.assertAlmostEqual(loaded["cost_under"], 0.8)

    def test_load_missing_baseline_returns_empty(self):
        loaded = _load_baseline("/nonexistent/path/baseline.json")
        self.assertEqual(loaded, {})

    def test_load_malformed_baseline_returns_empty(self):
        path = str(Path(self.tmp) / "bad.json")
        Path(path).write_text("not json")
        loaded = _load_baseline(path)
        self.assertEqual(loaded, {})

    def test_github_summary_written(self):
        report = _make_report([
            ("no_errors", 0.9, 0.8, True),
            ("cost_under", 0.6, 0.8, False),
        ])
        baseline = {"no_errors": 0.7, "cost_under": 0.8}
        import os
        orig = os.getcwd()
        os.chdir(self.tmp)
        try:
            _write_github_summary(report, baseline, tolerance=0.0)
            summary = Path(".agent-traces/eval-summary.md").read_text()
        finally:
            os.chdir(orig)
        self.assertIn("agent-strace eval", summary)
        self.assertIn("no_errors", summary)
        self.assertIn("cost_under", summary)
        self.assertIn("FAIL", summary)

    def test_github_summary_pass(self):
        report = _make_report([("no_errors", 1.0, 1.0, True)])
        import os
        orig = os.getcwd()
        os.chdir(self.tmp)
        try:
            _write_github_summary(report, {}, tolerance=0.0)
            summary = Path(".agent-traces/eval-summary.md").read_text()
        finally:
            os.chdir(orig)
        self.assertIn("PASS", summary)

    def test_github_summary_shows_delta(self):
        report = _make_report([("no_errors", 0.9, 0.8, True)])
        baseline = {"no_errors": 0.7}
        import os
        orig = os.getcwd()
        os.chdir(self.tmp)
        try:
            _write_github_summary(report, baseline, tolerance=0.0)
            summary = Path(".agent-traces/eval-summary.md").read_text()
        finally:
            os.chdir(orig)
        # Delta should be +20%
        self.assertIn("+20%", summary)

    def test_github_summary_no_baseline_shows_dashes(self):
        report = _make_report([("no_errors", 1.0, 1.0, True)])
        import os
        orig = os.getcwd()
        os.chdir(self.tmp)
        try:
            _write_github_summary(report, {}, tolerance=0.0)
            summary = Path(".agent-traces/eval-summary.md").read_text()
        finally:
            os.chdir(orig)
        self.assertIn("—", summary)


if __name__ == "__main__":
    unittest.main()
