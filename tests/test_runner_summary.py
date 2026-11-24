"""Tests for reading a test run's outcome from the runner's own summary."""

from __future__ import annotations

import unittest

from agent_trace.models import EventType, TraceEvent
from agent_trace.runner_summary import output_text, parse_summary, summary_outcome

RULE = "-" * 70
UNITTEST_OK = f"..\n{RULE}\nRan 2 tests in 0.000s\n\nOK\n"
UNITTEST_FAILED = (
    f"F.\n{'=' * 70}\nFAIL: test_x (test_mod.T.test_x)\n{RULE}\nTraceback (most recent call last):\n"
    f"AssertionError: 1 != 2\n\n{RULE}\nRan 2 tests in 0.001s\n\nFAILED (failures=1)\n"
)


def _result(text: object = None, *, event_type: EventType = EventType.TOOL_RESULT,
            redacted: bool = False, **data) -> TraceEvent:
    if text is not None:
        data["result"] = text
    return TraceEvent(event_type=event_type, timestamp=1.0, data=data, redacted=redacted)


class TestUnittestSummary(unittest.TestCase):
    def test_ok_and_failed(self):
        self.assertEqual(parse_summary("unittest", UNITTEST_OK), "passed")
        self.assertEqual(parse_summary("unittest", UNITTEST_FAILED), "failed")
        self.assertEqual(
            parse_summary("unittest", f"Ran 3 tests in 0.1s\n\nFAILED (errors=1, skipped=1)"),
            "failed",
        )
        self.assertEqual(parse_summary("unittest", "Ran 1 test in 0.002s\n\nOK (skipped=0)"), "passed")

    def test_newlines_collapsed(self):
        # The container quickstart quoted the result on one line.
        text = f".. {RULE} Ran 2 tests in 0.000s OK"
        self.assertEqual(parse_summary("unittest", text), "passed")

    def test_summary_followed_by_more_output(self):
        text = UNITTEST_OK + "<shellId: 2 completed with exit code 0>\nmore lines\n"
        self.assertEqual(parse_summary("unittest", text), "passed")

    def test_no_tests_or_all_skipped_is_not_a_pass(self):
        for text in (
            "Ran 0 tests in 0.000s\n\nNO TESTS RAN",
            "Ran 0 tests in 0.000s\n\nOK",
            "Ran 2 tests in 0.000s\n\nOK (skipped=2)",
        ):
            with self.subTest(text=text):
                self.assertIsNone(parse_summary("unittest", text))

    def test_missing_truncated_repeated_or_contradictory(self):
        for text in (
            "",
            "..\n" + RULE,
            "Ran 2 tests in 0.000s\n",
            "Ran 2 tests in 0.000s\n\nFAILED (failures=1",
            UNITTEST_OK + UNITTEST_OK,
            UNITTEST_OK + UNITTEST_FAILED,
            "Ran 2 tests in 0.000s\n\nFAILED (skipped=1)",
            "Ran 2 tests in 0.000s\n\nOK (failures=1)",
            "Ran 2 tests in 0.000s\n\nOK (failures=1, failures=1)",
            "Ran 2 tests in 0.000s\n\nmaybe",
        ):
            with self.subTest(text=text):
                self.assertIsNone(parse_summary("unittest", text))


class TestPytestSummary(unittest.TestCase):
    def test_pass_and_fail(self):
        cases = {
            "tests/test_a.py ...\n\n===== 3 passed in 0.12s =====\n": "passed",
            "...\n3 passed, 1 warning in 0.05s\n": "passed",
            "= 2 passed, 1 skipped, 1 deselected in 1.20s (0:00:01) =": "passed",
            "FAILED tests/test_a.py::test_x - assert 1 == 2\n== 1 failed, 2 passed in 0.31s ==": "failed",
            "== 1 error in 0.10s ==": "failed",
            "== 2 passed, 3 errors in 0.10s ==": "failed",
            "\x1b[32m==== \x1b[1m4 passed\x1b[0m\x1b[32m in 0.02s ====\x1b[0m": "passed",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(parse_summary("pytest", text), expected)

    def test_no_pass_or_unreadable(self):
        for text in (
            "== no tests ran in 0.01s ==",
            "== 2 skipped in 0.01s ==",
            "== 3 passed ==",
            "collected 3 items\n\ntests/test_a.py ..",
            "3 passed in 0.1s\n3 passed in 0.1s",
            "== 3 passed, 1 passed in 0.1s ==",
            "== 3 passed, 1 exploded in 0.1s ==",
            "== 3 passed in 0.1s ==\nFAIL Required test coverage of 90% not reached. Total coverage: 80.00%",
        ):
            with self.subTest(text=text):
                self.assertIsNone(parse_summary("pytest", text))

    def test_runner_must_match(self):
        self.assertIsNone(parse_summary("unittest", "== 3 passed in 0.1s =="))
        self.assertIsNone(parse_summary("pytest", UNITTEST_OK))
        self.assertIsNone(parse_summary("go test", "ok  \texample.com/pkg\t0.01s"))


class TestJestAndVitestSummary(unittest.TestCase):
    def test_jest(self):
        passed = "Test Suites: 2 passed, 2 total\nTests:       5 passed, 5 total\nTime: 1.2 s"
        failed = "Test Suites: 1 failed, 1 passed, 2 total\nTests:       1 failed, 4 passed, 5 total"
        suite_only = "Test Suites: 1 failed, 1 total\nTests:       0 total"
        filtered = "Test Suites: 1 skipped, 1 passed, 1 of 2 total\nTests:       1 skipped, 3 passed, 4 total"
        self.assertEqual(parse_summary("jest", passed), "passed")
        self.assertEqual(parse_summary("jest", failed), "failed")
        self.assertEqual(parse_summary("jest", suite_only), "failed")
        self.assertEqual(parse_summary("jest", filtered), "passed")
        for text in (
            "Tests:       5 passed, 5 total",
            "Test Suites: 2 passed, 2 total\nTests:       5 passed, 6 total",
            passed + "\n" + passed,
            passed + '\nJest: "global" coverage threshold for lines (90%) not met: 80%',
        ):
            with self.subTest(text=text):
                self.assertIsNone(parse_summary("jest", text))

    def test_vitest(self):
        passed = " Test Files  1 passed (1)\n      Tests  3 passed (3)\n   Duration  0.5s"
        failed = " Test Files  1 failed | 1 passed (2)\n      Tests  1 failed | 4 passed (5)"
        errors = passed + "\n     Errors  1 error"
        self.assertEqual(parse_summary("vitest", passed), "passed")
        self.assertEqual(parse_summary("vitest", failed), "failed")
        self.assertEqual(parse_summary("vitest", errors), "failed")
        for text in (
            " Test Files  1 passed (1)\n      Tests  no tests",
            " Test Files  1 passed (1)\n      Tests  3 passed (4)",
            "      Tests  3 passed (3)",
            passed + "\nERROR: Coverage for lines (80%) does not meet global threshold (90%)",
        ):
            with self.subTest(text=text):
                self.assertIsNone(parse_summary("vitest", text))


class TestOutputText(unittest.TestCase):
    def test_reads_result_output_and_streams(self):
        self.assertEqual(output_text(_result("Ran 1 test")), "Ran 1 test")
        self.assertEqual(output_text(_result(output="a\r\nb")), "a\nb")
        self.assertEqual(output_text(_result(stdout="out", stderr="err")), "out\nerr")
        self.assertEqual(output_text(_result(content_preview="preview")), "preview")

    def test_dict_repr_is_unwrapped(self):
        text = repr({"stdout": "", "stderr": UNITTEST_OK, "interrupted": False, "isImage": False})
        self.assertEqual(summary_outcome("unittest", _result(text)), "passed")
        self.assertIsNone(summary_outcome("unittest", _result("{'stdout': 'Ran 2 tests in 0.0s\\n\\nOK'")))

    def test_unreadable_results_give_nothing(self):
        cases = {
            "missing": None,
            "error event": _result(error=UNITTEST_FAILED, event_type=EventType.ERROR),
            "redacted flag": _result(UNITTEST_OK, redacted=True),
            "redaction marker": _result(UNITTEST_OK + "token [REDACTED:api-key]"),
            "content omitted": TraceEvent(event_type=EventType.TOOL_RESULT, timestamp=1.0,
                                          data={"content_omitted": True, "outcome_recorded": True}),
            "truncated by hook": _result(UNITTEST_OK + "x" * 50 + "... (truncated)"),
            "truncated flag": _result(UNITTEST_OK, truncated=True),
            "cut by an import": _result(UNITTEST_OK + "x" * 50 + "..."),
            "output_truncated flag": _result(UNITTEST_OK, output_truncated=True),
            "non-text": _result(123),
        }
        for name, event in cases.items():
            with self.subTest(case=name):
                self.assertIsNone(summary_outcome("unittest", event))

    def test_dots_before_the_end_or_untruncated_output_still_read(self):
        verbose = f"test_a (m.T) ... ok\n\n{RULE}\nRan 1 test in 0.000s\n\nOK\n"
        self.assertEqual(summary_outcome("unittest", _result(verbose)), "passed")
        self.assertEqual(summary_outcome("unittest", _result(UNITTEST_OK, output_truncated=False)),
                         "passed")

    def test_unsupported_runner(self):
        self.assertIsNone(summary_outcome("npm test", _result("Tests: 1 passed, 1 total")))


if __name__ == "__main__":
    unittest.main()
