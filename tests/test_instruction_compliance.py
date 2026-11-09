import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.cli import _normalise_skills_argv, build_parser
from agent_trace.context_score import context_file_load_events
from agent_trace.instruction_compliance import ActionExpectation, analyze_compliance_run
from agent_trace.models import EventType, TraceEvent
from agent_trace.skills_eval import build_report, cmd_skills_eval, format_markdown, load_cases


REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS = REPO_ROOT / "examples" / "instruction-compliance"
T0 = 1_800_100_000.0


def _event(event_type, offset, event_id, **data):
    return TraceEvent(
        event_type=event_type,
        timestamp=T0 + offset,
        event_id=event_id,
        data=data,
    )


def _run_corpus(fmt):
    argv = [
        "skills",
        "eval",
        str(CORPUS / "cases.json"),
        "--from-traces",
        str(CORPUS / "traces"),
        "--workspace",
        str(CORPUS / "fixture"),
        "--format",
        fmt,
    ]
    args = build_parser().parse_args(_normalise_skills_argv(argv))
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cmd_skills_eval(args)
    return code, out.getvalue(), err.getvalue()


class ComplianceCorpusTests(unittest.TestCase):
    def test_public_corpus_is_offline_and_deterministic(self):
        cases = load_cases(CORPUS / "cases.json")
        self.assertEqual(len(cases), 3)
        self.assertTrue(any(case.should_trigger for case in cases))
        self.assertTrue(any(not case.should_trigger for case in cases))
        self.assertTrue(all(case.instructions for case in cases))
        self.assertTrue(all(case.expected_actions for case in cases))

        first = _run_corpus("json")
        second = _run_corpus("json")
        self.assertEqual(first, second)
        code, output, error = first
        report = json.loads(output)

        self.assertEqual(code, 0, error)
        self.assertNotIn("agent", report)
        self.assertIs(report["compliance"]["not_a_score"], True)
        self.assertEqual(report["totals"]["should_trigger"]["loaded"], 2)
        self.assertEqual(report["totals"]["should_not_trigger"]["clean"], 1)
        self.assertEqual(report["compliance"]["evidence_counts"]["loads"], {
            "observed": 5,
            "inferred": 1,
            "missing": 0,
            "unavailable": 0,
        })
        inferred = next(case for case in report["cases"] if case["id"] == "availability-only-inference")
        instruction = next(
            item
            for item in inferred["run_results"][0]["compliance"]["loads"]
            if item["kind"] == "instruction"
        )
        self.assertEqual(
            (instruction["actual"], instruction["evidence"]),
            ("available_load_unobserved", "inferred"),
        )
        self.assertIsNone(instruction["expectation_met"])

    def test_markdown_is_concise_and_score_free(self):
        code, output, error = _run_corpus("markdown")

        self.assertEqual(code, 0, error)
        self.assertIn("# Skill trigger eval", output)
        self.assertIn("## Instruction compliance evidence", output)
        self.assertIn("| `draft-release-notes` | 2/0/0/0 | 3/0/0/0 | 1/0/0/0 |", output)
        self.assertIn("not an outcome or developer score", output)
        self.assertNotIn("\u2014", output)

    def test_markdown_formatter_has_stable_empty_warning_behavior(self):
        report = {
            "cases": [{
                "id": "clean",
                "skill": "release-notes",
                "should_trigger": False,
                "runs": 1,
                "loaded": 0,
                "scored": 1,
            }],
            "warnings": [],
        }

        output = format_markdown(report)

        self.assertNotIn("## Warnings", output)
        self.assertTrue(output.endswith("\n"))

    def test_compliance_warnings_reach_top_level_report(self):
        counts = {
            "observed": 0,
            "inferred": 0,
            "missing": 0,
            "unavailable": 1,
        }
        case = {
            "id": "broken",
            "skill": "release-notes",
            "should_trigger": True,
            "runs": 1,
            "loaded": 0,
            "not_loaded": 0,
            "unscored": 1,
            "compliance": {
                "loads": counts,
                "follow": counts,
                "validation": counts,
                "warnings": ["run-1: broken-session: skill analysis unavailable (ValueError)"],
            },
        }

        report = build_report([case], mode="from_traces", cases_file="cases.json")

        self.assertEqual(report["warnings"], [
            "broken: run-1: broken-session: skill analysis unavailable (ValueError)",
        ])
        self.assertEqual(report["compliance"]["warnings"], [
            "run-1: broken-session: skill analysis unavailable (ValueError)",
        ])


class ComplianceEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "AGENTS.md").write_text("Run tests after release changes.\n")
        skill = self.root / ".github" / "skills" / "release-notes"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            "---\nname: release-notes\ndescription: test\n---\n"
            "# Release notes\n\nRead `reference.md`.\n"
        )
        (skill / "reference.md").write_text("format\n")

    def tearDown(self):
        self.temp.cleanup()

    def test_missing_and_unavailable_are_distinct(self):
        events = [
            _event(EventType.SESSION_START, 0, "start"),
            _event(EventType.USER_PROMPT, 1, "prompt", prompt="Draft notes"),
            _event(
                EventType.TOOL_CALL,
                2,
                "skill",
                tool_name="skill",
                arguments={"skill": "release-notes"},
            ),
            _event(EventType.SESSION_END, 3, "end"),
        ]
        report = analyze_compliance_run(
            [("session", events)],
            workspace=str(self.root),
            skill="release-notes",
            should_trigger=True,
            instructions=("AGENTS.md",),
            expected_actions=(
                ActionExpectation("write-notes", "file_write", path="RELEASE_NOTES.md"),
            ),
            user_dirs=(),
        )

        instruction = next(item for item in report["loads"] if item["kind"] == "instruction")
        self.assertEqual(instruction["evidence"], "inferred")
        self.assertEqual(report["follow"]["skill"]["evidence"], "missing")
        self.assertEqual(report["follow"]["instruction_actions"][0]["evidence"], "missing")

        (self.root / "AGENTS.md").unlink()
        unavailable = analyze_compliance_run(
            [],
            workspace=str(self.root),
            skill="release-notes",
            should_trigger=True,
            instructions=("AGENTS.md",),
            expected_actions=(
                ActionExpectation("write-notes", "file_write", path="RELEASE_NOTES.md"),
            ),
            user_dirs=(),
        )

        instruction = next(item for item in unavailable["loads"] if item["kind"] == "instruction")
        self.assertEqual(instruction["evidence"], "unavailable")
        self.assertEqual(unavailable["follow"]["skill"]["evidence"], "unavailable")
        self.assertEqual(unavailable["follow"]["instruction_actions"][0]["evidence"], "unavailable")
        self.assertEqual(unavailable["validation"]["evidence"], "unavailable")

    def test_observed_post_load_validation_facts(self):
        events = [
            _event(EventType.SESSION_START, 0, "start"),
            _event(EventType.FILE_READ, 1, "agents", path="AGENTS.md"),
            _event(
                EventType.TOOL_CALL,
                2,
                "skill",
                tool_name="skill",
                arguments={"skill": "release-notes"},
            ),
            _event(EventType.FILE_WRITE, 3, "write", path="RELEASE_NOTES.md"),
            _event(
                EventType.TOOL_CALL,
                4,
                "test",
                tool_name="bash",
                arguments={"command": "pytest -q"},
                is_test=True,
            ),
            TraceEvent(
                event_type=EventType.TOOL_RESULT,
                timestamp=T0 + 5,
                event_id="result",
                parent_id="test",
                data={"exit_code": 0, "success": True},
            ),
            _event(EventType.SESSION_END, 6, "end"),
        ]
        report = analyze_compliance_run(
            [("session", events)],
            workspace=str(self.root),
            skill="release-notes",
            should_trigger=True,
            instructions=("AGENTS.md",),
            expected_actions=(
                ActionExpectation("write-notes", "file_write", path="RELEASE_NOTES.md"),
                ActionExpectation("tests", "test"),
            ),
            user_dirs=(),
        )

        self.assertEqual(report["evidence_counts"]["loads"]["observed"], 2)
        self.assertEqual(report["evidence_counts"]["follow"]["observed"], 2)
        self.assertEqual(report["validation"]["evidence"], "observed")
        self.assertEqual(report["validation"]["tests_run"], 1)
        self.assertEqual(report["validation"]["last_test"]["result"]["exit_code"], 0)
        self.assertTrue(report["validation"]["session_end_recorded"])

    def test_instruction_load_requires_a_read_event(self):
        events = [
            _event(EventType.FILE_WRITE, 0, "write", path="AGENTS.md"),
            _event(
                EventType.TOOL_CALL,
                1,
                "edit",
                tool_name="edit",
                arguments={"path": "AGENTS.md"},
            ),
            _event(
                EventType.TOOL_CALL,
                2,
                "read",
                tool_name="view",
                arguments={"path": "AGENTS.md"},
            ),
        ]

        loads = context_file_load_events(events, self.root, "AGENTS.md")

        self.assertEqual([event.event_id for event in loads], ["read"])

    def test_analysis_failure_marks_skill_evidence_unavailable(self):
        events = [
            _event(EventType.SESSION_START, 0, "start"),
            _event(EventType.USER_PROMPT, 1, "prompt", prompt="Draft notes"),
            _event(EventType.SESSION_END, 2, "end"),
        ]
        with mock.patch(
            "agent_trace.instruction_compliance.analyze_session",
            side_effect=ValueError("trace detail must not be emitted"),
        ):
            report = analyze_compliance_run(
                [("broken-session", events)],
                workspace=str(self.root),
                skill="release-notes",
                should_trigger=True,
                instructions=(),
                expected_actions=(
                    ActionExpectation("write-notes", "file_write", path="RELEASE_NOTES.md"),
                ),
                user_dirs=(),
            )

        skill = next(item for item in report["loads"] if item["kind"] == "skill")
        self.assertEqual((skill["actual"], skill["evidence"]), ("unavailable", "unavailable"))
        self.assertIsNone(skill["expectation_met"])
        self.assertEqual(report["follow"]["skill"]["actual"], "analysis_unavailable")
        self.assertEqual(report["follow"]["skill"]["evidence"], "unavailable")
        self.assertEqual(report["follow"]["instruction_actions"][0]["evidence"], "unavailable")
        self.assertEqual(report["validation"]["evidence"], "unavailable")
        self.assertEqual(report["analysis"]["sessions_analyzed"], 0)
        self.assertEqual(report["analysis"]["sessions_unavailable"], [{
            "session": "broken-session",
            "error_type": "ValueError",
        }])
        self.assertEqual(report["warnings"], [
            "broken-session: skill analysis unavailable (ValueError)",
        ])
        self.assertNotIn("trace detail", json.dumps(report))

    def test_malformed_event_gap_is_unavailable_not_missing(self):
        malformed = TraceEvent(
            event_type=EventType.FILE_WRITE,
            timestamp="not-a-timestamp",
            event_id="malformed",
            data={"path": "RELEASE_NOTES.md"},
        )
        report = analyze_compliance_run(
            [("broken-session", [
                _event(EventType.SESSION_START, 0, "start"),
                malformed,
                _event(EventType.SESSION_END, 2, "end"),
            ])],
            workspace=str(self.root),
            skill="release-notes",
            should_trigger=True,
            instructions=(),
            expected_actions=(
                ActionExpectation("write-notes", "file_write", path="RELEASE_NOTES.md"),
            ),
            user_dirs=(),
        )

        skill = next(item for item in report["loads"] if item["kind"] == "skill")
        self.assertEqual(skill["evidence"], "unavailable")
        self.assertEqual(report["follow"]["skill"]["evidence"], "unavailable")
        self.assertEqual(report["follow"]["instruction_actions"][0]["evidence"], "unavailable")
        self.assertEqual(report["validation"]["evidence"], "unavailable")
        self.assertEqual(report["analysis"]["sessions_unavailable"], [{
            "session": "broken-session",
            "error_type": "MalformedEventsSkipped",
            "events_skipped": 1,
        }])
        self.assertEqual(report["warnings"], [
            "broken-session: compliance evidence unavailable for 1 malformed event(s)",
        ])

    def test_non_object_event_data_is_an_unavailable_gap(self):
        malformed = TraceEvent(
            event_type=EventType.TOOL_CALL,
            timestamp=T0 + 1,
            event_id="malformed",
            data=["not", "an", "object"],
        )
        report = analyze_compliance_run(
            [("broken-session", [
                _event(EventType.SESSION_START, 0, "start"),
                malformed,
                _event(EventType.SESSION_END, 2, "end"),
            ])],
            workspace=str(self.root),
            skill="release-notes",
            should_trigger=True,
            instructions=(),
            expected_actions=(),
            user_dirs=(),
        )

        skill = next(item for item in report["loads"] if item["kind"] == "skill")
        self.assertEqual(skill["evidence"], "unavailable")
        self.assertEqual(report["follow"]["skill"]["evidence"], "unavailable")
        self.assertEqual(report["validation"]["evidence"], "unavailable")
        self.assertEqual(report["analysis"]["sessions_unavailable"], [{
            "session": "broken-session",
            "error_type": "MalformedEventsNormalized",
            "events_unavailable": 1,
        }])
        self.assertEqual(report["warnings"], [
            "broken-session: compliance evidence unavailable for 1 malformed event field set(s)",
        ])

    def test_ambiguous_validation_results_are_unavailable(self):
        base = [
            _event(EventType.SESSION_START, 0, "start"),
            _event(
                EventType.TOOL_CALL,
                1,
                "skill",
                tool_name="skill",
                arguments={"skill": "release-notes"},
            ),
            _event(
                EventType.TOOL_CALL,
                3,
                "test",
                tool_name="bash",
                arguments={"command": "pytest -q"},
                is_test=True,
            ),
        ]
        cases = {
            "multiple_linked_results": [
                TraceEvent(
                    event_type=EventType.TOOL_RESULT,
                    timestamp=T0 + 4,
                    event_id="result-1",
                    parent_id="test",
                    data={"exit_code": 0},
                ),
                TraceEvent(
                    event_type=EventType.ERROR,
                    timestamp=T0 + 5,
                    event_id="result-2",
                    parent_id="test",
                    data={"message": "late error"},
                ),
            ],
            "result_before_call": [
                TraceEvent(
                    event_type=EventType.TOOL_RESULT,
                    timestamp=T0 + 2,
                    event_id="result-before",
                    parent_id="test",
                    data={"exit_code": 0},
                ),
            ],
            "duplicate_call_id": [
                _event(
                    EventType.TOOL_CALL,
                    4,
                    "test",
                    tool_name="bash",
                    arguments={"command": "pytest -q"},
                    is_test=True,
                ),
                TraceEvent(
                    event_type=EventType.TOOL_RESULT,
                    timestamp=T0 + 5,
                    event_id="result",
                    parent_id="test",
                    data={"exit_code": 0},
                ),
            ],
        }
        for reason, extra in cases.items():
            with self.subTest(reason=reason):
                report = analyze_compliance_run(
                    [("session", [*base, *extra, _event(EventType.SESSION_END, 6, "end")])],
                    workspace=str(self.root),
                    skill="release-notes",
                    should_trigger=True,
                    instructions=(),
                    expected_actions=(),
                    user_dirs=(),
                )

                self.assertEqual(report["validation"]["evidence"], "unavailable")
                self.assertTrue(report["validation"]["ambiguous_result_correlation"])
                self.assertEqual(report["validation"]["last_test"]["result"]["reason"], reason)
                self.assertIn(reason, report["warnings"][0])

    def test_earlier_ambiguous_test_makes_validation_unavailable(self):
        events = [
            _event(EventType.SESSION_START, 0, "start"),
            _event(
                EventType.TOOL_CALL,
                1,
                "skill",
                tool_name="skill",
                arguments={"skill": "release-notes"},
            ),
            _event(
                EventType.TOOL_CALL,
                2,
                "test-1",
                tool_name="bash",
                arguments={"command": "pytest -q"},
                is_test=True,
            ),
            TraceEvent(
                event_type=EventType.TOOL_RESULT,
                timestamp=T0 + 3,
                event_id="result-1a",
                parent_id="test-1",
                data={"exit_code": 0},
            ),
            TraceEvent(
                event_type=EventType.ERROR,
                timestamp=T0 + 4,
                event_id="result-1b",
                parent_id="test-1",
                data={"message": "duplicate outcome"},
            ),
            _event(
                EventType.TOOL_CALL,
                5,
                "test-2",
                tool_name="bash",
                arguments={"command": "pytest -q"},
                is_test=True,
            ),
            TraceEvent(
                event_type=EventType.TOOL_RESULT,
                timestamp=T0 + 6,
                event_id="result-2",
                parent_id="test-2",
                data={"exit_code": 0},
            ),
            _event(EventType.SESSION_END, 7, "end"),
        ]

        report = analyze_compliance_run(
            [("session", events)],
            workspace=str(self.root),
            skill="release-notes",
            should_trigger=True,
            instructions=(),
            expected_actions=(),
            user_dirs=(),
        )

        self.assertEqual(report["validation"]["evidence"], "unavailable")
        self.assertEqual(report["validation"]["ambiguous_result_reasons"], ["multiple_linked_results"])
        self.assertEqual(report["validation"]["result_correlations"], {
            "observed": 1,
            "unavailable": 1,
        })
        self.assertEqual(report["validation"]["last_test"]["result"]["evidence"], "observed")


class ComplianceCaseValidationTests(unittest.TestCase):
    def test_invalid_compliance_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cases.json"
            path.write_text(json.dumps([{
                "id": "bad",
                "skill": "release-notes",
                "prompt": "draft",
                "should_trigger": True,
                "instructions": ["README.md"],
                "expected_actions": [
                    {"id": "write", "kind": "file_write"},
                    {"id": "tool", "kind": "tool_call", "path": "x"},
                ],
            }]))

            with self.assertRaises(ValueError) as context:
                load_cases(path)
            message = str(context.exception)
            self.assertIn("instructions[1]", message)
            self.assertIn("expected_actions[1].path", message)


if __name__ == "__main__":
    unittest.main()
