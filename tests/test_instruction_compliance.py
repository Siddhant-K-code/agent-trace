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
from agent_trace.models import EventType, SessionMeta, TraceEvent
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


def _call(offset, event_id, tool_name, arguments):
    return _event(EventType.TOOL_CALL, offset, event_id, tool_name=tool_name, arguments=arguments)


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
        self.assertEqual(report["validation"]["last_test"]["result"]["runner_summary"], {})

    def test_last_test_reports_runner_summary_when_exit_code_is_masked(self):
        events = [
            _event(EventType.SESSION_START, 0, "start"),
            _event(EventType.FILE_READ, 1, "agents", path="AGENTS.md"),
            _event(EventType.TOOL_CALL, 2, "skill", tool_name="skill",
                   arguments={"skill": "release-notes"}),
            _event(EventType.TOOL_CALL, 3, "test", tool_name="bash",
                   arguments={"command": "python -m pytest -q 2>&1 | tail -3"}),
            TraceEvent(event_type=EventType.TOOL_RESULT, timestamp=T0 + 4, event_id="result",
                       parent_id="test", data={"result": "FAILED tests/test_a.py::test_x\n"
                                                         "== 1 failed, 2 passed in 0.31s =="}),
            _event(EventType.SESSION_END, 5, "end"),
        ]
        report = analyze_compliance_run(
            [("session", events)],
            workspace=str(self.root),
            skill="release-notes",
            should_trigger=True,
            instructions=("AGENTS.md",),
            expected_actions=(ActionExpectation("tests", "test"),),
            user_dirs=(),
        )

        result = report["validation"]["last_test"]["result"]
        self.assertEqual(result["evidence"], "observed")
        self.assertEqual(result["runner_summary"], {"pytest": "failed"})
        self.assertNotIn("test_a.py", json.dumps(report))

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


class HookRecordedComplianceTests(unittest.TestCase):
    """Reads and writes in the shapes the hooks record.

    The shapes come from tests/fixtures/capture_matrix. Claude Code records
    Read, Write and Bash calls. Copilot CLI records view, create, bash and
    apply_patch calls, and stores apply_patch arguments as the patch text.
    Codex records only Bash calls. Paths are absolute.
    """

    MET = ("observed", True)
    MISSING = ("missing", False)
    LOADED = ("loaded", "observed")

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.root = self.base / "workspace"
        skill = self.root / ".github" / "skills" / "release-notes"
        skill.mkdir(parents=True)
        (self.root / "AGENTS.md").write_text("Run tests after release changes.\n")
        (skill / "SKILL.md").write_text("---\nname: release-notes\ndescription: test\n---\n# Release notes\n")

    def _path(self, relative):
        return str(self.root / relative)

    def _session(self, cwd, *events):
        return [
            _event(EventType.SESSION_START, 0, "start", cwd=cwd),
            *events,
            _event(EventType.SESSION_END, 99, "end"),
        ]

    def _analyze(self, sessions, *expected):
        return analyze_compliance_run(
            [(f"session-{index}", events) for index, events in enumerate(sessions)],
            workspace=str(self.root),
            skill="release-notes",
            should_trigger=True,
            instructions=("AGENTS.md",),
            expected_actions=expected,
            user_dirs=(),
        )

    @staticmethod
    def _actions(report):
        return {
            item["id"]: (item["evidence"], item["expectation_met"])
            for item in report["follow"]["instruction_actions"]
        }

    @staticmethod
    def _instruction_load(report):
        item = next(item for item in report["loads"] if item["kind"] == "instruction")
        return item["actual"], item["evidence"]

    def test_claude_read_and_copilot_view_meet_file_read(self):
        shapes = {
            "claude": ("Read", "file_path", "Skill"),
            "copilot": ("view", "path", "skill"),
        }
        for agent, (read_tool, key, skill_tool) in shapes.items():
            with self.subTest(agent=agent):
                report = self._analyze(
                    [self._session(
                        str(self.root),
                        _call(1, "agents", read_tool, {key: self._path("AGENTS.md")}),
                        _call(2, "skill", skill_tool, {"skill": "release-notes"}),
                        _call(3, "read", read_tool, {key: self._path("docs/format.md")}),
                    )],
                    ActionExpectation("read-format", "file_read", path="docs/*.md"),
                )

                self.assertEqual(self._instruction_load(report), self.LOADED)
                self.assertEqual(self._actions(report), {"read-format": self.MET})

    def test_editor_view_is_a_read_and_not_a_write(self):
        for tool in ("str_replace_based_edit_tool", "str_replace_editor"):
            with self.subTest(tool=tool):
                report = self._analyze(
                    [self._session(
                        str(self.root),
                        _call(1, "agents", tool, {"command": "view", "path": self._path("AGENTS.md")}),
                        _call(2, "skill", "skill", {"skill": "release-notes"}),
                        _call(3, "view", tool, {"command": "view", "path": self._path("CHANGELOG.md")}),
                        _call(4, "edit", tool, {
                            "command": "str_replace",
                            "path": self._path("NOTES.md"),
                            "old_str": "a",
                            "new_str": "b",
                        }),
                    )],
                    ActionExpectation("read-changelog", "file_read", path="CHANGELOG.md"),
                    ActionExpectation("write-changelog", "file_write", path="CHANGELOG.md"),
                    ActionExpectation("write-notes", "file_write", path="NOTES.md"),
                )

                self.assertEqual(self._instruction_load(report), self.LOADED)
                self.assertEqual(self._actions(report), {
                    "read-changelog": self.MET,
                    "write-changelog": self.MISSING,
                    "write-notes": self.MET,
                })

    def test_codex_shell_reads_count_as_reads(self):
        report = self._analyze(
            [self._session(
                str(self.root),
                _call(1, "agents", "Bash", {"command": "cat AGENTS.md"}),
                _call(2, "skill", "Bash", {"command": "sed -n 1,200p .github/skills/release-notes/SKILL.md"}),
                _call(3, "read", "Bash", {"command": "cd docs && sed -n 1,40p ../CHANGELOG.md"}),
                _call(4, "edit", "Bash", {"command": "sed -i s/a/b/ NOTES.md"}),
            )],
            ActionExpectation("read-changelog", "file_read", path="CHANGELOG.md"),
            ActionExpectation("read-notes", "file_read", path="NOTES.md"),
        )

        self.assertEqual(self._instruction_load(report), self.LOADED)
        self.assertEqual(self._actions(report), {
            "read-changelog": self.MET,
            "read-notes": self.MISSING,
        })

    def test_apply_patch_text_names_the_files_it_writes(self):
        patch = "*** Begin Patch\n*** Add File: RELEASE_NOTES.md\n+notes\n*** End Patch\n"
        report = self._analyze(
            [self._session(
                str(self.root),
                _call(1, "agents", "view", {"path": self._path("AGENTS.md")}),
                _call(2, "skill", "skill", {"skill": "release-notes"}),
                _call(3, "patch", "apply_patch", patch),
            )],
            ActionExpectation("write-notes", "file_write", path="RELEASE_NOTES.md"),
        )

        self.assertEqual(self._actions(report), {"write-notes": self.MET})

    def test_notebook_edit_writes_its_notebook(self):
        report = self._analyze(
            [self._session(
                str(self.root),
                _call(1, "agents", "Read", {"file_path": self._path("AGENTS.md")}),
                _call(2, "skill", "Skill", {"skill": "release-notes"}),
                _call(3, "notebook", "NotebookEdit", {
                    "notebook_path": self._path("analysis.ipynb"),
                    "new_source": "x",
                }),
            )],
            ActionExpectation("edit-notebook", "file_write", path="*.ipynb"),
        )

        self.assertEqual(self._actions(report), {"edit-notebook": self.MET})

    def test_replayed_run_resolves_paths_against_its_recorded_cwd(self):
        # --save-traces keeps the run. Its paths sit under its old temp
        # workspace, which is gone when --from-traces scores it.
        old = self.base / "gone" / "workspace"
        cases = self.base / "cases.json"
        cases.write_text(json.dumps([{
            "id": "draft",
            "skill": "release-notes",
            "prompt": "Draft the release notes.",
            "should_trigger": True,
            "instructions": ["AGENTS.md"],
            "expected_actions": [
                {"id": "write-notes", "kind": "file_write", "path": "RELEASE_NOTES.md"},
            ],
        }]))
        events = [
            _event(EventType.SESSION_START, 0, "start", provider="claude", cwd=str(old)),
            _call(1, "agents", "Read", {"file_path": str(old / "AGENTS.md")}),
            _call(2, "skill", "Skill", {"skill": "release-notes"}),
            _call(3, "write", "Write", {"file_path": str(old / "RELEASE_NOTES.md"), "content": "notes"}),
            _event(EventType.SESSION_END, 4, "end"),
        ]
        session = self.base / "traces" / "draft" / "run-1" / "s-draft"
        session.mkdir(parents=True)
        (session / "events.ndjson").write_text("".join(event.to_json() + "\n" for event in events))
        (session / "meta.json").write_text(SessionMeta(
            session_id="s-draft",
            started_at=T0,
            ended_at=T0 + 4,
            agent_name="claude",
        ).to_json())

        argv = [
            "skills",
            "eval",
            str(cases),
            "--from-traces",
            str(self.base / "traces"),
            "--workspace",
            str(self.root),
            "--format",
            "json",
        ]
        args = build_parser().parse_args(_normalise_skills_argv(argv))
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cmd_skills_eval(args)

        self.assertEqual(code, 0, err.getvalue())
        compliance = json.loads(out.getvalue())["cases"][0]["run_results"][0]["compliance"]
        self.assertEqual(self._instruction_load(compliance), self.LOADED)
        self.assertEqual(self._actions(compliance), {"write-notes": self.MET})

    def test_each_session_resolves_against_its_own_cwd(self):
        site = self.root / "site"
        report = self._analyze(
            [
                self._session(
                    str(self.root),
                    _call(1, "agents", "Read", {"file_path": self._path("AGENTS.md")}),
                    _call(2, "skill", "Skill", {"skill": "release-notes"}),
                ),
                self._session(
                    str(site),
                    _call(3, "write", "Write", {"file_path": str(site / "RELEASE_NOTES.md"), "content": "notes"}),
                ),
            ],
            ActionExpectation("write-notes", "file_write", path="RELEASE_NOTES.md"),
        )

        self.assertEqual(self._actions(report), {"write-notes": self.MET})

    def test_dot_prefixes_resolve_instead_of_being_stripped(self):
        report = self._analyze(
            [self._session(
                str(self.root),
                _event(EventType.FILE_READ, 1, "agents", path="AGENTS.md"),
                _call(2, "skill", "skill", {"skill": "release-notes"}),
                _event(EventType.FILE_READ, 3, "dotdir", path="./.github/skills/release-notes/references/format.md"),
                _event(EventType.FILE_READ, 4, "outside", path="../src/x.py"),
                _call(5, "create", "create", {"path": "./.github/release.md", "file_text": "notes"}),
                _call(6, "write", "Write", {"file_path": "../src/y.py", "content": "y"}),
            )],
            ActionExpectation("read-dotdir", "file_read", path=".github/*"),
            ActionExpectation("read-outside", "file_read", path="src/*"),
            ActionExpectation("write-dotdir", "file_write", path=".github/*"),
            ActionExpectation("write-outside", "file_write", path="src/*"),
        )

        self.assertEqual(self._actions(report), {
            "read-dotdir": self.MET,
            "read-outside": self.MISSING,
            "write-dotdir": self.MET,
            "write-outside": self.MISSING,
        })

    def test_instruction_load_reads_every_path_key_and_json_text(self):
        events = [
            _call(1, "target-file", "read_file", {"target_file": "AGENTS.md"}),
            _call(2, "absolute-path", "view", {"absolute_path": self._path("AGENTS.md")}),
            _call(3, "json-text", "view", json.dumps({"path": "AGENTS.md"})),
            _event(EventType.TOOL_CALL, 4, "tool-input", tool_name="Read", tool_input={"file_path": "AGENTS.md"}),
        ]

        loads = context_file_load_events(events, self.root, "AGENTS.md")

        self.assertEqual(
            [event.event_id for event in loads],
            ["target-file", "absolute-path", "json-text", "tool-input"],
        )

    def test_absolute_hook_paths_under_the_workspace_still_match(self):
        report = self._analyze(
            [self._session(
                str(self.root),
                _call(1, "agents", "view", {"path": self._path("AGENTS.md")}),
                _call(2, "skill", "skill", {"skill": "release-notes"}),
                _call(3, "write", "Write", {"file_path": self._path("RELEASE_NOTES.md"), "content": "notes"}),
            )],
            ActionExpectation("write-notes", "file_write", path="RELEASE_NOTES.md"),
        )

        self.assertEqual(self._instruction_load(report), self.LOADED)
        self.assertEqual(self._actions(report), {"write-notes": self.MET})

    def test_shell_redirect_is_not_a_write(self):
        report = self._analyze(
            [self._session(
                str(self.root),
                _call(1, "agents", "Read", {"file_path": self._path("AGENTS.md")}),
                _call(2, "skill", "Skill", {"skill": "release-notes"}),
                _call(3, "shell", "Bash", {"command": "echo x > RELEASE_NOTES.md"}),
            )],
            ActionExpectation("write-notes", "file_write", path="RELEASE_NOTES.md"),
        )

        self.assertEqual(self._actions(report), {"write-notes": self.MISSING})


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
