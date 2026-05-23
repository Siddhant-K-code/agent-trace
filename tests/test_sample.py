"""Tests for dataset auto-sampler (issue #94)."""

import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.sample import (
    SessionScore,
    _score_session,
    _sample_worst,
    _sample_diverse,
    _sample_random,
    _sample_recent,
    _session_to_jsonl_record,
    run_sample,
)
from agent_trace.store import TraceStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store() -> tuple[TraceStore, str]:
    tmpdir = tempfile.mkdtemp()
    return TraceStore(tmpdir), tmpdir


def _add_session(
    store: TraceStore,
    age_days: float = 0.0,
    errors: int = 0,
    tool_calls: int = 5,
    retries: int = 0,
) -> SessionMeta:
    meta = SessionMeta()
    meta.started_at = time.time() - age_days * 86400
    store.create_session(meta)
    base_ts = meta.started_at

    for i in range(tool_calls):
        ev = TraceEvent(
            event_type=EventType.TOOL_CALL,
            session_id=meta.session_id,
            timestamp=base_ts + i,
            data={"tool_name": "Bash", "arguments": {"command": f"echo {i}"}},
        )
        store.append_event(meta.session_id, ev)

    for _ in range(retries):
        ev = TraceEvent(
            event_type=EventType.TOOL_CALL,
            session_id=meta.session_id,
            timestamp=base_ts + tool_calls,
            data={"tool_name": "Bash", "arguments": {"command": "echo 0"}},
        )
        store.append_event(meta.session_id, ev)

    for _ in range(errors):
        ev = TraceEvent(
            event_type=EventType.ERROR,
            session_id=meta.session_id,
            timestamp=base_ts + tool_calls + 1,
            data={"message": "something failed"},
        )
        store.append_event(meta.session_id, ev)

    return meta


def _make_score(
    session_id: str = "abc",
    started_at: float | None = None,
    error_rate: float = 0.0,
    retry_rate: float = 0.0,
    blast_radius: int = 0,
    cost_estimate: float = 0.0,
    worst_score: float = 0.0,
) -> SessionScore:
    return SessionScore(
        session_id=session_id,
        started_at=started_at or time.time(),
        error_rate=error_rate,
        retry_rate=retry_rate,
        blast_radius=blast_radius,
        cost_estimate=cost_estimate,
        duration_s=10.0,
        tool_calls=5,
        worst_score=worst_score,
    )


# ---------------------------------------------------------------------------
# _score_session
# ---------------------------------------------------------------------------

class TestScoreSession(unittest.TestCase):
    def _make_events(self, tool_calls=5, errors=0, retries=0):
        events = []
        base = time.time()
        for i in range(tool_calls):
            events.append(TraceEvent(
                event_type=EventType.TOOL_CALL,
                timestamp=base + i,
                data={"tool_name": "Bash", "arguments": {"command": f"echo {i}"}},
            ))
        for _ in range(retries):
            events.append(TraceEvent(
                event_type=EventType.TOOL_CALL,
                timestamp=base + tool_calls,
                data={"tool_name": "Bash", "arguments": {"command": "echo 0"}},
            ))
        for _ in range(errors):
            events.append(TraceEvent(
                event_type=EventType.ERROR,
                timestamp=base + tool_calls + 1,
                data={"message": "fail"},
            ))
        return events

    def test_no_errors_zero_error_rate(self):
        meta = SessionMeta()
        events = self._make_events(tool_calls=5, errors=0)
        score = _score_session(meta.session_id, meta, events)
        self.assertEqual(score.error_rate, 0.0)

    def test_errors_increase_error_rate(self):
        meta = SessionMeta()
        events = self._make_events(tool_calls=5, errors=2)
        score = _score_session(meta.session_id, meta, events)
        self.assertGreater(score.error_rate, 0.0)

    def test_worst_score_higher_for_bad_session(self):
        meta = SessionMeta()
        good = _score_session(meta.session_id, meta, self._make_events(5, 0, 0))
        bad = _score_session(meta.session_id, meta, self._make_events(5, 3, 2))
        self.assertGreater(bad.worst_score, good.worst_score)

    def test_empty_events_no_crash(self):
        meta = SessionMeta()
        score = _score_session(meta.session_id, meta, [])
        self.assertEqual(score.tool_calls, 0)
        self.assertEqual(score.error_rate, 0.0)


# ---------------------------------------------------------------------------
# Sampling strategies
# ---------------------------------------------------------------------------

class TestSampleWorst(unittest.TestCase):
    def test_returns_n_sessions(self):
        scores = [_make_score(str(i), worst_score=float(i)) for i in range(10)]
        result = _sample_worst(scores, 3)
        self.assertEqual(len(result), 3)

    def test_returns_highest_worst_score(self):
        scores = [_make_score(str(i), worst_score=float(i)) for i in range(10)]
        result = _sample_worst(scores, 3)
        ids = {s.session_id for s in result}
        self.assertIn("9", ids)
        self.assertIn("8", ids)
        self.assertIn("7", ids)

    def test_n_larger_than_pool(self):
        scores = [_make_score(str(i)) for i in range(3)]
        result = _sample_worst(scores, 10)
        self.assertEqual(len(result), 3)


class TestSampleDiverse(unittest.TestCase):
    def test_returns_n_sessions(self):
        scores = [
            _make_score(str(i), error_rate=i * 0.1, retry_rate=(9 - i) * 0.1)
            for i in range(10)
        ]
        result = _sample_diverse(scores, 4)
        self.assertEqual(len(result), 4)

    def test_no_duplicates(self):
        scores = [_make_score(str(i), error_rate=i * 0.1) for i in range(10)]
        result = _sample_diverse(scores, 5)
        ids = [s.session_id for s in result]
        self.assertEqual(len(ids), len(set(ids)))

    def test_empty_input(self):
        result = _sample_diverse([], 5)
        self.assertEqual(result, [])

    def test_n_zero(self):
        scores = [_make_score(str(i)) for i in range(5)]
        result = _sample_diverse(scores, 0)
        self.assertEqual(result, [])


class TestSampleRandom(unittest.TestCase):
    def test_returns_n_sessions(self):
        scores = [_make_score(str(i)) for i in range(20)]
        result = _sample_random(scores, 5, seed=42)
        self.assertEqual(len(result), 5)

    def test_reproducible_with_seed(self):
        scores = [_make_score(str(i)) for i in range(20)]
        r1 = [s.session_id for s in _sample_random(scores, 5, seed=42)]
        r2 = [s.session_id for s in _sample_random(scores, 5, seed=42)]
        self.assertEqual(r1, r2)

    def test_n_larger_than_pool(self):
        scores = [_make_score(str(i)) for i in range(3)]
        result = _sample_random(scores, 10, seed=0)
        self.assertEqual(len(result), 3)


class TestSampleRecent(unittest.TestCase):
    def test_returns_most_recent(self):
        now = time.time()
        scores = [_make_score(str(i), started_at=now - i * 3600) for i in range(10)]
        result = _sample_recent(scores, 3)
        ids = {s.session_id for s in result}
        self.assertIn("0", ids)  # most recent
        self.assertIn("1", ids)
        self.assertIn("2", ids)
        self.assertNotIn("9", ids)  # oldest

    def test_returns_n_sessions(self):
        now = time.time()
        scores = [_make_score(str(i), started_at=now - i) for i in range(10)]
        result = _sample_recent(scores, 4)
        self.assertEqual(len(result), 4)


# ---------------------------------------------------------------------------
# _session_to_jsonl_record
# ---------------------------------------------------------------------------

class TestSessionToJsonlRecord(unittest.TestCase):
    def test_record_has_required_fields(self):
        meta = SessionMeta()
        events = [TraceEvent(
            event_type=EventType.TOOL_CALL,
            session_id=meta.session_id,
            data={"tool_name": "Bash"},
        )]
        score = _make_score(meta.session_id)
        record = _session_to_jsonl_record(meta, events, score)
        self.assertIn("session_id", record)
        self.assertIn("events", record)
        self.assertIn("score", record)
        self.assertEqual(len(record["events"]), 1)

    def test_record_is_json_serialisable(self):
        meta = SessionMeta()
        score = _make_score(meta.session_id)
        record = _session_to_jsonl_record(meta, [], score)
        # Should not raise
        json.dumps(record)


# ---------------------------------------------------------------------------
# run_sample (integration)
# ---------------------------------------------------------------------------

class TestRunSample(unittest.TestCase):
    def setUp(self):
        self.store, self.tmpdir = _make_store()
        self.output = os.path.join(self.tmpdir, "out.jsonl")

    def test_worst_strategy_writes_jsonl(self):
        for i in range(5):
            _add_session(self.store, errors=i, tool_calls=5)
        out = io.StringIO()
        rc = run_sample(self.store, "worst", n=3, output_path=self.output, out=out)
        self.assertEqual(rc, 0)
        lines = Path(self.output).read_text().strip().splitlines()
        self.assertEqual(len(lines), 3)
        for line in lines:
            record = json.loads(line)
            self.assertIn("session_id", record)
            self.assertIn("events", record)

    def test_recent_strategy(self):
        for i in range(5):
            _add_session(self.store, age_days=float(i))
        out = io.StringIO()
        rc = run_sample(self.store, "recent", n=2, output_path=self.output, out=out)
        self.assertEqual(rc, 0)
        lines = Path(self.output).read_text().strip().splitlines()
        self.assertEqual(len(lines), 2)

    def test_random_strategy(self):
        for i in range(10):
            _add_session(self.store)
        out = io.StringIO()
        rc = run_sample(self.store, "random", n=4, output_path=self.output, seed=7, out=out)
        self.assertEqual(rc, 0)
        lines = Path(self.output).read_text().strip().splitlines()
        self.assertEqual(len(lines), 4)

    def test_diverse_strategy(self):
        for i in range(8):
            _add_session(self.store, errors=i % 3, tool_calls=5)
        out = io.StringIO()
        rc = run_sample(self.store, "diverse", n=3, output_path=self.output, out=out)
        self.assertEqual(rc, 0)
        lines = Path(self.output).read_text().strip().splitlines()
        self.assertEqual(len(lines), 3)

    def test_empty_store_returns_error(self):
        out = io.StringIO()
        rc = run_sample(self.store, "worst", n=5, output_path=self.output, out=out)
        self.assertEqual(rc, 1)
        self.assertIn("No sessions", out.getvalue())

    def test_n_larger_than_sessions(self):
        _add_session(self.store)
        _add_session(self.store)
        out = io.StringIO()
        rc = run_sample(self.store, "worst", n=100, output_path=self.output, out=out)
        self.assertEqual(rc, 0)
        lines = Path(self.output).read_text().strip().splitlines()
        self.assertEqual(len(lines), 2)

    def test_deduplicate_flag(self):
        # Add two sessions with identical tool call sequences
        for _ in range(3):
            _add_session(self.store, tool_calls=3)
        out = io.StringIO()
        rc = run_sample(
            self.store, "worst", n=10, output_path=self.output,
            deduplicate=True, out=out,
        )
        self.assertEqual(rc, 0)
        lines = Path(self.output).read_text().strip().splitlines()
        # All three have identical sequences — only 1 should survive dedup
        self.assertEqual(len(lines), 1)


# ---------------------------------------------------------------------------
# CLI registration
# ---------------------------------------------------------------------------

class TestSampleCLIRegistered(unittest.TestCase):
    def test_sample_in_help(self):
        import sys
        from agent_trace.cli import main
        old_argv = sys.argv
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.argv = ["agent-strace", "--help"]
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            try:
                main()
            except SystemExit:
                pass
            output = sys.stdout.getvalue() + sys.stderr.getvalue()
        finally:
            sys.argv = old_argv
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        self.assertIn("sample", output)


if __name__ == "__main__":
    unittest.main()
