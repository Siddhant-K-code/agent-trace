"""Tests for behavioral drift detection."""

import json
import argparse
import io
import tempfile
import time
import unittest
from pathlib import Path
from functools import partial
from unittest.mock import patch

from agent_trace.cli import build_parser
from agent_trace.drift import (
    BehavioralFingerprint,
    DimensionDrift,
    DistStats,
    DriftReport,
    SessionMetrics,
    cmd_drift,
    cmd_fingerprint,
    _dist_stats,
    _js_divergence,
    _stats_divergence,
    compute_drift,
    compute_fingerprint,
    print_fingerprint,
    print_report,
)
from agent_trace.cost import estimate_cost
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore

try:
    from test_dashboard import _codex_session
except ImportError:  # pragma: no cover - package-style test discovery
    from tests.test_dashboard import _codex_session

try:
    import _escapes
except ImportError:
    from tests import _escapes


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
        expected = {
            "tool_mix",
            "error_rate",
            "retry_rate",
            "tool_calls",
            "cost_usd",
            "blast_radius",
            "session_duration_s",
            "decision_depth",
        }
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

    def test_suggested_next_steps_parse_and_run(self):
        import contextlib
        import shlex

        from agent_trace.cli import (
            _normalise_eval_argv,
            _normalise_policy_argv,
            _normalise_skills_argv,
        )
        from agent_trace.eval import cmd_eval
        from agent_trace.optimize import _resolve_sessions

        now = time.time()
        baseline_ids = self._populate("base4", 3, "Bash", now - 200)
        _add_session(self.store, "curr4_err", [_error(now - 50)], started_at=now - 50)
        current_ids = self._populate("curr4", 3, "Write", now - 100) + ["curr4_err"]
        report = compute_drift(
            compute_fingerprint(self.store, baseline_ids),
            compute_fingerprint(self.store, current_ids),
        )
        self.assertTrue(report.exceeded)
        buf = io.StringIO()
        print_report(report, out=buf)
        lines = buf.getvalue().split("Suggested next step:\n", 1)[1].splitlines()
        commands = [line.strip() for line in lines if line.strip().startswith("agent-strace ")]
        self.assertEqual(len(commands), 2)

        parser = build_parser()
        parsed = []
        for command in commands:
            argv = ["--trace-dir", self.tmp, *shlex.split(command)[1:]]
            argv = _normalise_eval_argv(_normalise_skills_argv(_normalise_policy_argv(argv)))
            with self.subTest(command=command), \
                    contextlib.redirect_stderr(io.StringIO()):
                try:
                    parsed.append(parser.parse_args(argv))
                except SystemExit:
                    self.fail(f"suggested command does not parse: {command}")

        collect, follow_up = parsed
        self.assertEqual((collect.command, collect.eval_command, collect.dataset_command),
                         ("eval", "dataset", "auto"))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cmd_eval(collect), 0)
        self.assertEqual(follow_up.command, "optimize")
        self.assertEqual(_resolve_sessions(self.store, follow_up), ["curr4_err"])

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


class TestFingerprintCommand(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = _make_store(self.tmp)

    def _args(self, **kwargs):
        defaults = dict(
            trace_dir=self.tmp,
            sessions=20,
            output=None,
            id="",
            compare=None,
            threshold=0.20,
            format="text",
        )
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_print_fingerprint_contains_profile(self):
        now = time.time()
        _add_session(self.store, "fp_print", [_tool_call("Bash", now)], started_at=now)
        fp = compute_fingerprint(self.store, ["fp_print"], fingerprint_id="fp")
        out = io.StringIO()
        print_fingerprint(fp, out)
        text = out.getvalue()
        self.assertIn("Behavioral fingerprint", text)
        self.assertIn("Tool call profile", text)
        self.assertIn("Bash", text)

    def test_cmd_fingerprint_json(self):
        now = time.time()
        _add_session(self.store, "fp_json", [_tool_call("Read", now)], started_at=now)
        captured = io.StringIO()

        with patch("sys.stdout", captured):
            result = cmd_fingerprint(self._args(format="json"))

        self.assertEqual(result, 0)
        data = json.loads(captured.getvalue())
        self.assertEqual(data["sessions"], 1)
        self.assertIn("Read", data["tool_mix"])
        self.assertIn("tool_calls", data)
        self.assertIn("cost_usd", data)

    def test_cmd_fingerprint_writes_output(self):
        now = time.time()
        _add_session(self.store, "fp_out", [_tool_call("Write", now)], started_at=now)
        output = Path(self.tmp) / "fingerprint.json"

        result = cmd_fingerprint(self._args(output=str(output), id="saved"))

        self.assertEqual(result, 0)
        data = json.loads(output.read_text())
        self.assertEqual(data["fingerprint_id"], "saved")
        self.assertEqual(data["sessions"], 1)

    def test_cmd_fingerprint_json_with_output_keeps_stdout_parseable(self):
        now = time.time()
        _add_session(self.store, "fp_json_out", [_tool_call("Read", now)], started_at=now)
        output = Path(self.tmp) / "fingerprint.json"
        captured = io.StringIO()

        with patch("sys.stdout", captured):
            result = cmd_fingerprint(self._args(output=str(output), format="json"))

        self.assertEqual(result, 0)
        data = json.loads(captured.getvalue())
        self.assertEqual(data["sessions"], 1)
        self.assertTrue(output.exists())

    def test_cmd_fingerprint_compare_json(self):
        first = BehavioralFingerprint(
            fingerprint_id="a",
            sessions=2,
            period_start="2026-01-01",
            period_end="2026-01-02",
            tool_mix={"Bash": 1.0},
        )
        second = BehavioralFingerprint(
            fingerprint_id="b",
            sessions=2,
            period_start="2026-01-03",
            period_end="2026-01-04",
            tool_mix={"Read": 1.0},
        )
        a_path = Path(self.tmp) / "a.json"
        b_path = Path(self.tmp) / "b.json"
        a_path.write_text(first.to_json())
        b_path.write_text(second.to_json())
        captured = io.StringIO()

        with patch("sys.stdout", captured):
            result = cmd_fingerprint(self._args(compare=[str(a_path), str(b_path)], format="json"))

        self.assertEqual(result, 1)
        data = json.loads(captured.getvalue())
        self.assertEqual(data["label"], "high")

    def test_parser_registers_fingerprint(self):
        parser = build_parser()
        args = parser.parse_args(["fingerprint", "--sessions", "5", "--output", "fp.json"])
        self.assertEqual(args.command, "fingerprint")
        self.assertEqual(args.sessions, 5)
        self.assertEqual(args.output, "fp.json")


class TestFingerprintUnpricedSessions(unittest.TestCase):
    """cost_usd is the `agent-strace cost` total; refused sessions are not $0."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = _make_store(tmp.name)

    def test_unpriceable_session_is_left_out_of_cost_usd(self):
        priced = _codex_session(self.store, started_at=1000.0)
        unpriced = _codex_session(self.store, prompt_data=[], started_at=2000.0)
        want = estimate_cost(self.store, priced).total_cost

        fp = compute_fingerprint(self.store, [priced, unpriced], "fp")
        self.assertEqual(fp.sessions, 2)
        self.assertEqual(fp.unpriced_sessions, 1)
        self.assertEqual(fp.cost_usd, _dist_stats([want]))

        data = fp.to_dict()
        self.assertEqual(data["unpriced_sessions"], 1)
        self.assertEqual(BehavioralFingerprint.from_json(json.dumps(data)).unpriced_sessions, 1)
        out = io.StringIO()
        print_fingerprint(fp, out)
        self.assertIn("Sessions not priced: 1 (left out of cost usd)", out.getvalue())

    def test_no_priced_session_has_null_cost_usd(self):
        sid = _codex_session(self.store, prompt_data=[], started_at=1000.0)
        fp = compute_fingerprint(self.store, [sid], "fp")
        self.assertIsNone(fp.cost_usd)
        data = fp.to_dict()
        self.assertIsNone(data["cost_usd"])
        self.assertIsNone(BehavioralFingerprint.from_json(json.dumps(data)).cost_usd)
        out = io.StringIO()
        print_fingerprint(fp, out)
        cost_row = next(line for line in out.getvalue().splitlines() if "cost usd" in line)
        self.assertEqual(cost_row.split()[-3:], ["n/a", "n/a", "n/a"])

    def test_unpriced_baseline_does_not_score_cost_drift(self):
        baseline = compute_fingerprint(
            self.store, [_codex_session(self.store, prompt_data=[], started_at=1000.0)], "base"
        )
        current = compute_fingerprint(
            self.store, [_codex_session(self.store, started_at=2000.0)], "cur"
        )
        report = compute_drift(baseline, current)
        cost = next(d for d in report.dimensions if d.name == "cost_usd")
        self.assertIsNone(cost.score)
        self.assertEqual(cost.label, "n/a")
        self.assertEqual(cost.baseline_summary, "n/a")
        self.assertEqual((report.baseline_unpriced, report.current_unpriced), (1, 0))
        scored = [d.score for d in report.dimensions if d.score is not None]
        self.assertEqual(len(scored), len(report.dimensions) - 1)
        self.assertAlmostEqual(
            report.overall_score, round((2 * scored[0] + sum(scored[1:])) / (len(scored) + 1), 3)
        )
        out = io.StringIO()
        print_report(report, out)
        self.assertIn("Sessions not priced: 1 baseline, 0 current (left out of cost_usd)", out.getvalue())
        self.assertIn("cost_usd                  n/a", out.getvalue())

    def test_pricing_bugs_are_not_hidden(self):
        sid = _codex_session(self.store, started_at=1000.0)
        with patch("agent_trace.drift.session_cost", side_effect=TypeError("bug")):
            with self.assertRaises(TypeError):
                compute_fingerprint(self.store, [sid], "fp")


def _drift_report(baseline_period="2026-01-01 to 2026-01-07",
                  current_period="2026-01-08 to 2026-01-14",
                  baseline_sessions=3, current_sessions=3,
                  baseline_unpriced=0, current_unpriced=0,
                  name="tool_mix", label="stable",
                  baseline_summary="Read=100%", current_summary="Bash=100%"):
    dimension = DimensionDrift(name, 0.1, label, baseline_summary, current_summary)
    return DriftReport(0.1, 0.2, baseline_sessions, current_sessions, baseline_period,
                       current_period, [dimension], baseline_unpriced, current_unpriced)


def _render_fingerprint(fp):
    out = io.StringIO()
    print_fingerprint(fp, out)
    return out.getvalue()


def _render_drift(report):
    out = io.StringIO()
    print_report(report, out)
    return out.getvalue()


# (label, kind, build): build(value) returns a fingerprint or a report with
# value in the place the label names. The kinds are in tests/_escapes.py. A
# fingerprint file can hold a string where a count belongs.
FINGERPRINT_CASES = [
    ("tool name", "text", lambda v: BehavioralFingerprint(sessions=1, tool_mix={v: 1.0})),
]

DRIFT_REPORT_CASES = [
    ("baseline period", "text", lambda v: _drift_report(baseline_period=v)),
    ("current period", "text", lambda v: _drift_report(current_period=v)),
    ("baseline sessions", "text", lambda v: _drift_report(baseline_sessions=v)),
    ("current sessions", "text", lambda v: _drift_report(current_sessions=v)),
    ("baseline sessions not priced", "text", lambda v: _drift_report(baseline_unpriced=v)),
    ("current sessions not priced", "text", lambda v: _drift_report(current_unpriced=v)),
    ("dimension name", "text", lambda v: _drift_report(name=v)),
    ("high dimension name", "text", lambda v: _drift_report(name=v, label="high")),
    ("dimension label", "text", lambda v: _drift_report(label=v)),
    ("baseline summary", "text", lambda v: _drift_report(baseline_summary=v, label="high")),
    ("current summary", "text", lambda v: _drift_report(current_summary=v, label="high")),
]


class TestTraceTextShowsAsText(unittest.TestCase):
    """A tool name in a trace, or any value in a fingerprint file, can hold
    escape codes and line breaks. fingerprint and drift show them as text. The
    JSON and the fingerprint files they write keep the raw text."""

    def setUp(self):
        store_dir = tempfile.TemporaryDirectory()
        self.addCleanup(store_dir.cleanup)
        self.tmp = store_dir.name
        files_dir = tempfile.TemporaryDirectory()
        self.addCleanup(files_dir.cleanup)
        self.files = Path(files_dir.name)

    def test_each_value_shows_escapes_as_text(self):
        _escapes.check_each_value(self, FINGERPRINT_CASES, _render_fingerprint)
        _escapes.check_each_value(self, DRIFT_REPORT_CASES, _render_drift)

    def test_plain_text_prints_unchanged(self):
        _escapes.check_plain_text_unchanged(self, FINGERPRINT_CASES, _render_fingerprint)
        _escapes.check_plain_text_unchanged(self, DRIFT_REPORT_CASES, _render_drift)

    def test_sgr_code_shows_as_text(self):
        outputs = (
            _render_fingerprint(BehavioralFingerprint(sessions=1, tool_mix={"a\x1b[8mhidden": 1.0})),
            _render_drift(_drift_report(current_period="a\x1b[8mhidden")),
        )
        for output in outputs:
            self.assertIn(r"a\x1b[8mhidden", output)
            self.assertNotIn("\x1b", output)

    def _crafted(self, name, tool):
        path = self.files / name
        path.write_text(json.dumps({
            "fingerprint_id": _escapes.PAYLOAD,
            "sessions": _escapes.PAYLOAD,
            "period": {"start": _escapes.PAYLOAD, "end": _escapes.PAYLOAD},
            "tool_mix": {tool: 1.0},
            "unpriced_sessions": _escapes.PAYLOAD,
        }))
        return str(path)

    def _run(self, command, args):
        # The commands call the printers without out, so the printers write to
        # the sys.stdout that drift.py saw on import. Point them at the buffer.
        captured = io.StringIO()
        with patch("sys.stdout", captured), \
                patch("agent_trace.drift.print_report", partial(print_report, out=captured)), \
                patch("agent_trace.drift.print_fingerprint", partial(print_fingerprint, out=captured)):
            command(args)
        return captured.getvalue()

    def _fingerprint(self, fmt="text", compare=None, output=None):
        return self._run(cmd_fingerprint, argparse.Namespace(
            trace_dir=self.tmp, sessions=20, output=output, id="",
            compare=compare, threshold=0.20, format=fmt))

    def _drift(self, fmt="table", baseline=None, save_baseline=None):
        return self._run(cmd_drift, argparse.Namespace(
            trace_dir=self.tmp, threshold=0.20, baseline=baseline, since="30d",
            current=None, baseline_range=None, save_baseline=save_baseline, format=fmt))

    def _add_sessions(self, tool):
        store = _make_store(self.tmp)
        now = time.time()
        for i in range(2):
            _add_session(store, f"s{i}", [_tool_call(tool, now - 100 + i)], started_at=now - 100 + i)

    def _compare_files(self):
        return [self._crafted("a.json", _escapes.PAYLOAD),
                self._crafted("b.json", _escapes.PAYLOAD + "2")]

    def _assert_printable(self, output):
        self.assertNotIn("\x1b", output)
        for line in output.split("\n"):
            self.assertTrue(line.isprintable(), ascii(line))

    def test_compare_shows_fingerprint_file_text_as_text(self):
        output = self._fingerprint(compare=self._compare_files())
        shown = _escapes.PAYLOAD_TEXT
        self.assertIn(f"Baseline: {shown} to {shown} ({shown} sessions)", output)
        self.assertIn(f"Current:  {shown} to {shown} ({shown} sessions)", output)
        self.assertIn(f"Sessions not priced: {shown} baseline, {shown} current", output)
        self.assertIn(f"    Baseline: {shown}=100%", output)
        self.assertIn(f"    Current:  {shown}2=100%", output)
        self._assert_printable(output)

    def test_drift_shows_baseline_file_text_as_text(self):
        self._add_sessions("Read")
        output = self._drift(baseline=self._crafted("baseline.json", _escapes.PAYLOAD))
        shown = _escapes.PAYLOAD_TEXT
        self.assertIn(f"Baseline: {shown} to {shown} ({shown} sessions)", output)
        self.assertIn(f"Sessions not priced: {shown} baseline,", output)
        self.assertIn(f"    Baseline: {shown}=100%", output)
        self._assert_printable(output)

    def test_unreadable_file_error_shows_as_text(self):
        path = self._crafted("a.json", "Read")
        args = argparse.Namespace(trace_dir=self.tmp, sessions=20, output=None, id="",
                                  compare=[path, path], threshold=0.20, format="text")
        err = io.StringIO()
        with patch("agent_trace.drift.BehavioralFingerprint.from_json",
                   side_effect=ValueError(_escapes.PAYLOAD)):
            with patch("sys.stderr", err):
                self.assertEqual(cmd_fingerprint(args), 1)
        self.assertIn(_escapes.PAYLOAD_TEXT, err.getvalue())
        self._assert_printable(err.getvalue())

    def test_tool_name_in_trace_shows_as_text(self):
        self._add_sessions(_escapes.PAYLOAD)
        output = self._fingerprint()
        self.assertIn(f"  {_escapes.PAYLOAD_TEXT} ", output)
        self._assert_printable(output)

    def test_json_and_files_keep_the_raw_text(self):
        self._add_sessions(_escapes.PAYLOAD)
        data = json.loads(self._fingerprint(fmt="json"))
        self.assertEqual(list(data["tool_mix"]), [_escapes.PAYLOAD])

        saved = self.files / "fp.json"
        self._fingerprint(output=str(saved))
        self.assertEqual(list(json.loads(saved.read_text())["tool_mix"]), [_escapes.PAYLOAD])

        baseline = self.files / "baseline.json"
        self._drift(save_baseline=str(baseline))
        self.assertEqual(list(json.loads(baseline.read_text())["tool_mix"]), [_escapes.PAYLOAD])

        data = json.loads(self._fingerprint(fmt="json", compare=self._compare_files()))
        self.assertEqual(data["baseline_sessions"], _escapes.PAYLOAD)
        self.assertEqual(data["current_unpriced_sessions"], _escapes.PAYLOAD)
        self.assertEqual(data["dimensions"][0]["first"], _escapes.PAYLOAD + "=100%")
        self.assertEqual(data["dimensions"][0]["second"], _escapes.PAYLOAD + "2=100%")

        data = json.loads(self._drift(fmt="json", baseline=self._crafted("b.json", "Read")))
        self.assertEqual(data["baseline_sessions"], _escapes.PAYLOAD)
        self.assertEqual(data["baseline_unpriced_sessions"], _escapes.PAYLOAD)


# Python 3.14 limits JSON nesting by stack size, not by a count, so it can
# parse 100000 levels. A million levels raise RecursionError on 3.10+.
TOO_DEEP = 1_000_000
NOT_FINITE = "it has a number that is not finite"
BAD_SHARE = "a tool_mix value is not a number of 0 or more"


def _nested(depth):
    """A fingerprint file that nests depth levels: an object around lists."""
    return b'{"sessions": ' + b"[" * (depth - 1) + b"]" * (depth - 1) + b"}"


REFUSED_FINGERPRINTS = [
    ("empty", b"", "it is not valid JSON (Expecting value at line 1 column 1)"),
    ("bad JSON", b"{", "it is not valid JSON "
     "(Expecting property name enclosed in double quotes at line 1 column 2)"),
    ("not UTF-8", b"\xff\xfe{}", "it is not valid UTF-8"),
    ("list", b"[]", "it is not a JSON object"),
    ("null", b"null", "it is not a JSON object"),
    ("number", b"5", "it is not a JSON object"),
    ("deep list", b"[" * 129 + b"]" * 129, "it is nested too deeply"),
    ("depth 129", _nested(129), "it is nested too deeply"),
    ("too deep to parse", b"[" * TOO_DEEP + b"]" * TOO_DEEP, "it is nested too deeply"),
    ("NaN", b'{"sessions": NaN}', NOT_FINITE),
    ("Infinity", b'{"error_rate": {"mean": -Infinity}}', NOT_FINITE),
    ("float too large", b'{"sessions": 1e400}', NOT_FINITE),
    ("4301 digits", b'{"sessions": 1' + b"0" * 4300 + b"}",
     "it has a number with too many digits"),
    ("period text", b'{"period": "x"}', "period is not an object"),
    ("period null", b'{"period": null}', "period is not an object"),
    ("stat number", b'{"error_rate": 0.5}', "error_rate is not an object"),
    ("stat null", b'{"retry_rate": null}', "retry_rate is not an object"),
    ("cost text", b'{"cost_usd": "x"}', "cost_usd is not an object"),
    ("stat text", b'{"tool_calls": {"mean": "x"}}', "tool_calls.mean is not a finite number"),
    ("stat null value", b'{"blast_radius": {"p50": null}}',
     "blast_radius.p50 is not a finite number"),
    ("stat bool", b'{"decision_depth": {"p95": true}}',
     "decision_depth.p95 is not a finite number"),
    ("stat too large", b'{"session_duration_s": {"mean": 1' + b"0" * 400 + b"}}",
     "session_duration_s.mean is not a finite number"),
    ("tool_mix list", b'{"tool_mix": []}', "tool_mix is not an object"),
    ("tool_mix text", b'{"tool_mix": {"Read": "x"}}', BAD_SHARE),
    ("tool_mix negative", b'{"tool_mix": {"Read": -0.5}}', BAD_SHARE),
    ("tool_mix bool", b'{"tool_mix": {"Read": true}}', BAD_SHARE),
    ("tool_mix too large", b'{"tool_mix": {"Read": 1' + b"0" * 400 + b"}}", BAD_SHARE),
]

ACCEPTED_FINGERPRINTS = [
    ("empty object", b"{}"),
    ("depth 128", _nested(128)),
    ("4300 digits and a sign", b'{"sessions": -1' + b"0" * 4299 + b"}"),
    ("negative stat", b'{"session_duration_s": {"mean": -5, "p50": -5, "p95": -5}}'),
    ("null cost", b'{"cost_usd": null}'),
    ("null period fields", b'{"period": {"start": null, "end": null}}'),
    ("zero share", b'{"tool_mix": {"Read": 0, "Edit": 1}}'),
    ("huge numbers", b'{"error_rate": {"mean": 1e308, "p50": -1e308, "p95": 1e308}, '
     b'"tool_mix": {"Read": 1e308, "Edit": 1e308}}'),
    ("extra keys", b'{"note": "x", "error_rate": {"mean": 0.5, "note": "x"}}'),
]


def _strict_json(text):
    """Load text as JSON, and refuse a bare NaN or Infinity in it."""
    def refuse(name):
        raise ValueError(f"the output holds a bare {name}")
    return json.loads(text, parse_constant=refuse)


def _open_error(path):
    """The reason open() gives for path, or None when it opens."""
    try:
        with open(path, "rb"):
            return None
    except OSError as exc:
        return exc.strerror


class TestUnreadableFingerprintFile(unittest.TestCase):
    """drift --baseline and fingerprint --compare read a fingerprint file the
    same way. When they cannot read one, they print one line on stderr that
    names it, print nothing on stdout and exit 1."""

    def setUp(self):
        store_dir = tempfile.TemporaryDirectory()
        self.addCleanup(store_dir.cleanup)
        self.tmp = store_dir.name
        files_dir = tempfile.TemporaryDirectory()
        self.addCleanup(files_dir.cleanup)
        self.files = Path(files_dir.name)
        store = _make_store(self.tmp)
        now = time.time()
        for i in range(4):
            _add_session(store, f"s{i}", [
                _tool_call("Read", now - 100 + i),
                _tool_call("Edit" if i % 2 else "Bash", now - 99 + i),
            ], started_at=now - 100 + i)
        day = 86400
        self.last_30_days = (time.strftime("%Y-%m-%d", time.gmtime(now - 30 * day)) + ":"
                             + time.strftime("%Y-%m-%d", time.gmtime(now + day)))
        self.good = self.files / "good.json"
        code, _, err = self._run(cmd_fingerprint, argparse.Namespace(
            trace_dir=self.tmp, sessions=20, output=str(self.good), id="good",
            compare=None, threshold=0.20, format="text"))
        self.assertEqual((code, err), (0, ""))

    def _run(self, command, args):
        # The printers write to the sys.stdout that drift.py saw on import.
        # Point them at the buffer.
        out, err = io.StringIO(), io.StringIO()
        with patch("sys.stdout", out), patch("sys.stderr", err), \
                patch("agent_trace.drift.print_report", partial(print_report, out=out)), \
                patch("agent_trace.drift.print_fingerprint", partial(print_fingerprint, out=out)):
            code = command(args)
        return code, out.getvalue(), err.getvalue()

    def _drift(self, baseline, fmt="table", since="30d", current=None, save_baseline=None):
        return self._run(cmd_drift, argparse.Namespace(
            trace_dir=self.tmp, threshold=0.20, baseline=baseline, since=since,
            current=current, baseline_range=None, save_baseline=save_baseline, format=fmt))

    def _compare(self, first, second, fmt="text"):
        return self._run(cmd_fingerprint, argparse.Namespace(
            trace_dir=self.tmp, sessions=20, output=None, id="",
            compare=[first, second], threshold=0.20, format=fmt))

    def _forms(self, path):
        """The six forms that read path, as names and calls."""
        good = str(self.good)
        return {
            "drift --since": lambda: self._drift(path),
            "drift --since json": lambda: self._drift(path, fmt="json"),
            "drift --current": lambda: self._drift(path, since=None, current=self.last_30_days),
            "compare first": lambda: self._compare(path, good),
            "compare second": lambda: self._compare(good, path),
            "compare second json": lambda: self._compare(good, path, fmt="json"),
        }

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
        line = f"Could not read fingerprint file {shown}: {reason}\n"
        for form, run in self._forms(path).items():
            with self.subTest(case=label, form=form):
                self.assertEqual(run(), (1, "", line))

    def test_refuses_a_file_it_cannot_use(self):
        for label, data, reason in REFUSED_FINGERPRINTS:
            self._assert_refused(label, self._file("fingerprint.json", data), reason)

    def test_reads_a_file_it_can_use(self):
        for label, data in ACCEPTED_FINGERPRINTS:
            path = self._file("fingerprint.json", data)
            for form, run in self._forms(path).items():
                with self.subTest(case=label, form=form):
                    code, out, err = run()
                    self.assertIn(code, (0, 1))
                    self.assertEqual(err, "")
                    self.assertNotEqual(out, "")
                    if form.endswith("json"):
                        _strict_json(out)

    def test_missing_file(self):
        path = str(self.files / "missing.json")
        self._assert_refused("missing", path, self._open_reason(path))

    def test_directory(self):
        path = self.files / "dir"
        path.mkdir()
        self._assert_refused("directory", str(path), self._open_reason(path))

    def test_file_it_has_no_permission_to_read(self):
        path = self.files / "locked.json"
        path.write_bytes(self.good.read_bytes())
        path.chmod(0)
        self.addCleanup(path.chmod, 0o600)
        reason = _open_error(path)
        if reason is None:
            self.skipTest("this user can read a file with mode 0")
        self._assert_refused("mode 0", str(path), reason)

    def test_empty_path(self):
        self._assert_refused("empty path", "", "the path is empty", shown="''")

    def test_path_shows_escapes_as_text(self):
        path = str(self.files / f"missing{_escapes.PAYLOAD}.json")
        shown = f"{self.files}/missing{_escapes.PAYLOAD_TEXT}.json"
        self._assert_refused("escapes in the path", path, self._open_reason(path), shown=shown)

    def test_reads_a_file_that_starts_with_a_byte_order_mark(self):
        good = self._forms(str(self.good))
        bom = self._forms(self._file("bom.json", b"\xef\xbb\xbf" + self.good.read_bytes()))
        for form, run in good.items():
            with self.subTest(form=form):
                expected = run()
                self.assertEqual(expected[2], "")
                self.assertEqual(bom[form](), expected)

    def test_save_baseline_writes_nothing_when_it_cannot_read_the_baseline(self):
        missing = str(self.files / "missing.json")
        saved = self.files / "saved.json"
        line = f"Could not read fingerprint file {missing}: {self._open_reason(missing)}\n"
        self.assertEqual(self._drift(missing, save_baseline=str(saved)), (1, "", line))
        self.assertFalse(saved.exists())

    def test_compare_names_the_first_file_it_cannot_read(self):
        first = self._file("first.json", b"{")
        second = self._file("second.json", b"[]")
        line = (f"Could not read fingerprint file {first}: it is not valid JSON "
                "(Expecting property name enclosed in double quotes at line 1 column 2)\n")
        self.assertEqual(self._compare(first, second), (1, "", line))


if __name__ == "__main__":
    unittest.main()
