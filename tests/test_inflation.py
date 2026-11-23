"""Tests for issue #44: token inflation calculator."""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from agent_trace.cli import build_parser
from agent_trace.cost import get_model_pricing
from agent_trace.inflation import (
    TOKENIZER_FACTORS,
    _resolve_factor,
    analyse_inflation,
    cmd_inflation,
    format_inflation,
)
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore

try:  # keeps product telemetry off in this process and its children
    import _no_telemetry
except ImportError:  # pragma: no cover - run as a package module
    from tests import _no_telemetry

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))

# json.dumps adds two quotes, and 4 characters make 1 token.
PROMPT_1000 = (EventType.USER_PROMPT, {"content": "x" * 3998})
REPLY_500 = (EventType.ASSISTANT_RESPONSE, {"content": "y" * 1998})
TOOL_ONLY = (EventType.TOOL_CALL, {"tool_name": "Bash"})
FACTOR_LIST = ", ".join(sorted(TOKENIZER_FACTORS))


def _make_store(tmp_dir: str) -> TraceStore:
    return TraceStore(os.path.join(tmp_dir, "traces"))


def _add_session(store: TraceStore, *events: tuple[EventType, dict]) -> None:
    meta = SessionMeta()
    store.create_session(meta)
    for event_type, data in events:
        store.append_event(
            meta.session_id,
            TraceEvent(event_type=event_type, session_id=meta.session_id, data=data),
        )


def _report_text(report) -> str:
    buf = io.StringIO()
    format_inflation(report, out=buf)
    return buf.getvalue()


def _row(text: str, label: str) -> str:
    for line in text.splitlines():
        if line.strip().startswith(label):
            return line
    raise AssertionError(f"no {label!r} row in:\n{text}")


def _cells(row: str) -> list[str]:
    """Return the baseline, inflated and delta cells of a table row."""
    return [row[26:40].strip(), row[42:56].strip(), row[58:66].strip()]


def _run(trace_dir: str, *argv: str) -> tuple[int, str, str]:
    args = build_parser().parse_args(["--trace-dir", trace_dir, "inflation", *argv])
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cmd_inflation(args)
    return code, out.getvalue(), err.getvalue()


def _run_cli(trace_dir: str, *argv: str) -> subprocess.CompletedProcess:
    """Run agent-strace inflation in a child process, as a user runs it."""
    env = _no_telemetry.without_telemetry({
        key: value
        for key, value in os.environ.items()
        if not key.startswith("AGENT_STRACE_") and key != "AGENT_TRACE_DIR"
    })
    env["PYTHONPATH"] = os.pathsep.join(item for item in (SRC, env.get("PYTHONPATH", "")) if item)
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "-m", "agent_trace.entry", "--trace-dir", trace_dir, "inflation", *argv],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=120,
    )


def _populate(store: TraceStore, n: int = 5) -> None:
    for _ in range(n):
        meta = SessionMeta()
        store.create_session(meta)
        sid = meta.session_id
        for event in [
            TraceEvent(
                event_type=EventType.LLM_REQUEST,
                session_id=sid,
                data={
                    "model": "claude-opus-4-6",
                    "system": "You are a helpful assistant.",
                    "tools": [{"name": "read_file"}],
                    "messages": [{"role": "user", "content": "x" * 200}],
                },
            ),
            TraceEvent(
                event_type=EventType.USER_PROMPT,
                session_id=sid,
                data={"content": "x" * 400},
            ),
        ]:
            store.append_event(sid, event)


class TestInflationFactors(unittest.TestCase):
    def test_resolve_factor_baseline(self):
        from agent_trace.inflation import _resolve_factor
        self.assertEqual(_resolve_factor("claude-opus-4-6"), 1.0)

    def test_resolve_factor_inflated(self):
        from agent_trace.inflation import _resolve_factor
        self.assertEqual(_resolve_factor("claude-opus-4-7"), 1.38)

    def test_resolve_factor_prefix_match(self):
        from agent_trace.inflation import _resolve_factor
        self.assertEqual(_resolve_factor("claude-opus-4-7-20260101"), 1.38)

    def test_resolve_factor_unknown_has_no_factor(self):
        self.assertIsNone(_resolve_factor("unknown-model-xyz"))

    def test_dated_and_prefixed_names_use_the_base_model(self):
        cases = {
            "gpt-4o-2024-08-06": 1.05,
            "gpt-4o-mini-2024-07-18": 1.05,
            "anthropic/claude-opus-4-7": 1.38,
            "Claude-Opus-4-7@20260101": 1.38,
            "gpt-4-turbo": 1.00,
            "gemini-2.0-flash-001": 0.95,
        }
        for model, factor in cases.items():
            with self.subTest(model=model):
                self.assertEqual(_resolve_factor(model), factor)

    def test_partial_and_empty_names_have_no_factor(self):
        for model in ("claude", "", "claude-opus-4", "claude-opus-47", "gpt-4o2"):
            with self.subTest(model=model):
                self.assertIsNone(_resolve_factor(model))


class TestInflationAnalysis(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_analyse_returns_report(self):
        from agent_trace.inflation import analyse_inflation
        store = _make_store(self._tmp)
        _populate(store, 5)
        report = analyse_inflation(
            store,
            model_baseline="claude-opus-4-6",
            model_inflated="claude-opus-4-7",
        )
        self.assertGreaterEqual(report.session_count, 1)
        self.assertGreater(report.factor_inflated, report.factor_baseline)
        self.assertGreaterEqual(report.avg_tokens_inflated, report.avg_tokens_baseline)

    def test_inflation_factor_ratio_correct(self):
        from agent_trace.inflation import analyse_inflation, _resolve_factor
        store = _make_store(self._tmp)
        _populate(store, 3)
        report = analyse_inflation(store)
        if report.avg_tokens_baseline > 0:
            ratio = report.avg_tokens_inflated / report.avg_tokens_baseline
            expected = _resolve_factor("claude-opus-4-7") / _resolve_factor("claude-opus-4-6")
            self.assertAlmostEqual(ratio, expected, places=2)

    def test_by_content_type_all_present(self):
        from agent_trace.inflation import analyse_inflation, CONTENT_TYPES
        store = _make_store(self._tmp)
        _populate(store, 3)
        report = analyse_inflation(store)
        ct_names = {ct.content_type for ct in report.by_content_type}
        for expected in CONTENT_TYPES:
            self.assertIn(expected, ct_names)

    def test_system_prompt_bucket_populated(self):
        from agent_trace.inflation import analyse_inflation
        store = _make_store(self._tmp)
        _populate(store, 3)
        report = analyse_inflation(store)
        sp = next(ct for ct in report.by_content_type if ct.content_type == "system_prompt")
        self.assertGreater(sp.tokens_baseline, 0)

    def test_tool_definitions_bucket_populated(self):
        from agent_trace.inflation import analyse_inflation
        store = _make_store(self._tmp)
        _populate(store, 3)
        report = analyse_inflation(store)
        td = next(ct for ct in report.by_content_type if ct.content_type == "tool_definitions")
        self.assertGreater(td.tokens_baseline, 0)

    def test_format_output(self):
        from agent_trace.inflation import analyse_inflation, format_inflation
        store = _make_store(self._tmp)
        _populate(store, 3)
        report = analyse_inflation(store)
        buf = io.StringIO()
        format_inflation(report, out=buf)
        output = buf.getvalue()
        self.assertIn("Token Inflation Report", output)
        self.assertIn("Monthly", output)

    def test_empty_store(self):
        from agent_trace.inflation import analyse_inflation
        store = _make_store(self._tmp)
        report = analyse_inflation(store)
        self.assertEqual(report.session_count, 0)
        self.assertEqual(report.avg_tokens_baseline, 0)

    def test_analyse_refuses_a_model_without_a_factor(self):
        store = _make_store(self._tmp)
        _populate(store, 1)
        with self.assertRaisesRegex(ValueError, "'claude-opus-47' has no tokenizer factor"):
            analyse_inflation(store, model_inflated="claude-opus-47")

    def test_cli_has_inflation_command(self):
        from agent_trace.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "inflation",
            "--compare", "claude-opus-4-6,claude-opus-4-7",
            "--sessions", "20",
        ])
        self.assertEqual(args.compare, "claude-opus-4-6,claude-opus-4-7")
        self.assertEqual(args.sessions, 20)


class _StoreTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self.trace_dir = os.path.join(self._tmp, "traces")
        self.store = TraceStore(self.trace_dir)


class TestPricing(_StoreTestCase):
    def setUp(self):
        super().setUp()
        _add_session(self.store, PROMPT_1000, REPLY_500)

    def test_costs_use_bundled_input_and_output_rates(self):
        baseline = get_model_pricing("claude-opus-4-6")
        for model, factor in TOKENIZER_FACTORS.items():
            with self.subTest(model=model):
                report = analyse_inflation(self.store, model_inflated=model)
                self.assertAlmostEqual(
                    report.avg_cost_baseline,
                    (1000 * baseline.input_per_million + 500 * baseline.output_per_million) / 1e6,
                    places=12,
                )
                price = get_model_pricing(model)
                if price is None:
                    self.assertIsNone(report.avg_cost_inflated)
                    self.assertIsNone(report.monthly_cost_inflated)
                    continue
                expected = (
                    (1000 * price.input_per_million + 500 * price.output_per_million)
                    * factor / 1e6
                )
                self.assertAlmostEqual(report.avg_cost_inflated, expected, places=12)
                self.assertAlmostEqual(report.monthly_cost_inflated, expected * 8 * 30, places=10)

    def test_model_without_price_shows_na_and_keeps_token_rows(self):
        for pair, column in ((("claude-opus-4-6", "gpt-4"), 1), (("gpt-4", "claude-opus-4-6"), 0)):
            with self.subTest(pair=pair):
                text = _report_text(analyse_inflation(self.store, *pair))
                self.assertEqual(_cells(_row(text, "User Messages")), ["1,000", "1,000", "+0.0%"])
                self.assertEqual(_cells(_row(text, "Assistant Messages")), ["500", "500", "+0.0%"])
                for label in ("Per session (cost)", "Daily (8 sessions)", "Monthly"):
                    cells = _cells(_row(text, label))
                    self.assertEqual(cells[column], "n/a", label)
                    self.assertTrue(cells[1 - column].startswith("$"), label)
                    self.assertEqual(cells[2], "n/a", label)


class TestSessionCount(_StoreTestCase):
    def test_sessions_without_content_count_as_none(self):
        for _ in range(3):
            _add_session(self.store, TOOL_ONLY)
        report = analyse_inflation(self.store)
        self.assertEqual(report.session_count, 0)
        self.assertEqual(report.avg_tokens_baseline, 0)
        self.assertEqual(report.avg_cost_inflated, 0)
        self.assertEqual(report.monthly_cost_inflated, 0)
        self.assertIn("Token Inflation Report (0 sessions)\n", _report_text(report))

    def test_one_session_is_singular(self):
        _add_session(self.store, PROMPT_1000)
        _add_session(self.store, TOOL_ONLY)
        text = _report_text(analyse_inflation(self.store))
        self.assertIn("Token Inflation Report (1 session)\n", text)


def _request(system: str, user: str) -> tuple[EventType, dict]:
    return (EventType.LLM_REQUEST, {"system": system, "messages": [{"role": "user", "content": user}]})


class TestBreakdown(_StoreTestCase):
    def _types(self, report) -> dict:
        return {ct.content_type: ct for ct in report.by_content_type}

    def test_counts_are_rounded_not_truncated(self):
        # 150 tokens at 1.38 is 206.99999999999997.
        _add_session(self.store, (EventType.USER_PROMPT, {"content": "u" * 598}))
        user = self._types(analyse_inflation(self.store))["user_messages"]
        self.assertEqual((user.tokens_baseline, user.tokens_inflated), (150, 207))

    def test_largest_source_adds_the_most_tokens(self):
        cases = {
            # 50 -> 69 and 100 -> 138 are both +38.0%. User messages add more.
            "s" * 198: (69, 138),
            # 2 -> 3 is +50%, but it adds 1 token and user messages add 38.
            "s" * 6: (3, 138),
        }
        for system, counts in cases.items():
            with self.subTest(system_tokens=(len(system) + 2) // 4):
                store = _make_store(tempfile.mkdtemp(dir=self._tmp))
                _add_session(store, _request(system, "u" * 398))
                report = analyse_inflation(store)
                types = self._types(report)
                self.assertEqual(
                    (types["system_prompt"].tokens_inflated, types["user_messages"].tokens_inflated),
                    counts,
                )
                self.assertEqual(report.largest_inflation_source, "user_messages")
                self.assertIn("Largest inflation source: User Messages\n", _report_text(report))

    def test_tie_keeps_the_first_type(self):
        # The system prompt and user messages both add 38 tokens.
        _add_session(self.store, _request("s" * 398, "u" * 398))
        report = analyse_inflation(self.store)
        self.assertEqual(report.largest_inflation_source, "system_prompt")

    def test_no_largest_line_when_no_type_adds_tokens(self):
        _add_session(self.store, PROMPT_1000)
        for pair in (("claude-opus-4-7", "claude-opus-4-6"), ("claude-opus-4-6", "claude-sonnet-4-5")):
            with self.subTest(pair=pair):
                report = analyse_inflation(self.store, *pair)
                self.assertEqual(report.largest_inflation_source, "")
                self.assertNotIn("Largest inflation source", _report_text(report))


class TestCommandRefusals(_StoreTestCase):
    def setUp(self):
        super().setUp()
        _add_session(self.store, PROMPT_1000)

    def _assert_refused(self, argv: list[str], message: str) -> None:
        code, out, err = _run(self.trace_dir, *argv)
        self.assertEqual(code, 2, err)
        self.assertEqual(out, "")
        self.assertEqual(err, f"Error: {message}\n")

    def test_compare_must_be_two_names_separated_by_one_comma(self):
        for value in (
            "claude-opus-4-8",
            "claude-opus-4-6,claude-opus-4-7,gpt-4o",
            "claude-opus-4-6,,claude-opus-4-7",
            ",claude-opus-4-7",
            "claude-opus-4-6, ",
        ):
            with self.subTest(value=value):
                self._assert_refused(
                    ["--compare", value],
                    f"--compare must be two model names separated by one comma, got {value!r}",
                )

    def test_compare_model_needs_a_tokenizer_factor(self):
        for value, model in (
            ("claude-opus-4-6,claude-opus-47", "claude-opus-47"),
            ("claude-opus-4-6,claude-opus-4-8", "claude-opus-4-8"),
            ("claude,claude-opus-4-7", "claude"),
            ("a,b", "a"),
        ):
            with self.subTest(value=value):
                self._assert_refused(
                    ["--compare", value],
                    f"--compare model {model!r} has no tokenizer factor. "
                    f"Models with a factor: {FACTOR_LIST}",
                )

    def test_sessions_must_be_one_or_more(self):
        for value in ("0", "-1"):
            with self.subTest(value=value):
                self._assert_refused(["--sessions", value], f"--sessions must be 1 or more, got {value}")

    def test_daily_sessions_must_be_finite_and_not_negative(self):
        for value, shown in (
            ("-3", "-3"),
            ("-0.5", "-0.5"),
            ("nan", "nan"),
            ("inf", "inf"),
            ("-inf", "-inf"),
            ("1e400", "inf"),
        ):
            with self.subTest(value=value):
                self._assert_refused(
                    [f"--daily-sessions={value}"],
                    f"--daily-sessions must be a finite number, 0 or more, got {shown}",
                )

    def test_flags_are_checked_before_any_session_is_read(self):
        with mock.patch.object(TraceStore, "list_sessions", return_value=[]) as list_sessions, \
                mock.patch.object(TraceStore, "load_events", return_value=[]) as load_events:
            for argv in (["--compare", "a,b"], ["--sessions", "0"], ["--daily-sessions", "nan"]):
                with self.subTest(argv=argv):
                    self.assertEqual(_run(self.trace_dir, *argv)[0], 2)
        list_sessions.assert_not_called()
        load_events.assert_not_called()

    def test_compare_names_are_stripped(self):
        with mock.patch("agent_trace.inflation.analyse_inflation", wraps=analyse_inflation) as analyse:
            code, _, err = _run(self.trace_dir, "--compare", " claude-opus-4-6 , claude-opus-4-7 ")
        self.assertEqual((code, err), (0, ""))
        kwargs = analyse.call_args.kwargs
        self.assertEqual(
            (kwargs["model_baseline"], kwargs["model_inflated"]),
            ("claude-opus-4-6", "claude-opus-4-7"),
        )


class TestCommandOutput(_StoreTestCase):
    def setUp(self):
        super().setUp()
        _add_session(self.store, PROMPT_1000)

    def test_default_pair_reports_38_percent_and_the_largest_source(self):
        result = _run_cli(self.trace_dir)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(_cells(_row(result.stdout, "User Messages"))[2], "+38.0%")
        self.assertEqual(_cells(_row(result.stdout, "Per session (tokens)"))[2], "+38.0%")
        self.assertIn("  Daily (8 sessions)  ", result.stdout)
        self.assertIn("Largest inflation source: User Messages\n", result.stdout)

    def test_daily_sessions_zero_projects_no_cost(self):
        # -0 is 0, so the label and the costs show no minus sign.
        for value in ("0", "-0"):
            with self.subTest(value=value):
                result = _run_cli(self.trace_dir, f"--daily-sessions={value}")
                self.assertEqual(result.returncode, 0, result.stderr)
                daily = _row(result.stdout, "Daily (")
                self.assertEqual(daily[2:24].rstrip(), "Daily (0 sessions)")
                for row in (daily, _row(result.stdout, "Monthly")):
                    self.assertEqual(row[26:].split(), ["$", "0.00", "$", "0.00", "$", "+0.00"])

    def test_daily_sessions_label_shows_the_value_unrounded(self):
        for value, label in (("2.5", "Daily (2.5 sessions)"), ("1", "Daily (1 session)")):
            with self.subTest(value=value):
                result = _run_cli(self.trace_dir, "--daily-sessions", value)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(_row(result.stdout, "Daily (")[2:24].rstrip(), label)
