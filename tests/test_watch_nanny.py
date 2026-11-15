"""Tests for issue #48: agent nanny rule-based kill switch."""

from __future__ import annotations

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

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore

# Names from agent_trace.watch that main does not have are imported inside
# the tests, so the tests that guard unchanged behavior still run there.


def _tool_call(tool_name: str, args: dict) -> TraceEvent:
    return TraceEvent(
        event_type=EventType.TOOL_CALL,
        data={"tool_name": tool_name, "arguments": args},
    )


_COST_KILL = {"name": "cost cap", "condition": "cost_usd > 0.50", "action": "kill"}
_COST_KILL_YAML = (
    "rules:\n"
    "  - name: cost cap\n"
    "    condition: cost_usd > 0.50\n"
    "    action: kill\n"
)
_ENV = 'file_path matches "**/production.env"'
_ACTIONS = "action must be one of alert, terminal, file, pause, kill, require_approval"
_NUMBER = "condition must end with one plain number, such as 5 or 0.50, and nothing after it"
_GLOB = (
    'condition must be file_path matches "GLOB" with a non-empty GLOB in quotes '
    "and nothing after it"
)
_FORM = 'condition must be METRIC OP NUMBER or file_path matches "GLOB"'
_BUILT_INS = "the built-in rules are mcp-poisoning, loop, budget, timeout and cognitive-debt"


def _unknown_metric(metric: str) -> str:
    return (
        f"condition has an unknown metric {metric!r}; use files_modified, cost_usd, "
        "consecutive_test_failures or duration_minutes"
    )


def _unknown_key(key: str) -> str:
    return f"unknown key {key!r}; a rule has only name, condition, action and notify"


def _json_error(text: str) -> str:
    """The refusal for JSON the json module rejects; its message differs
    between Python versions."""
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        return f"it is not valid JSON ({exc.msg} at line {exc.lineno} column {exc.colno})"
    raise AssertionError(f"{text!r} is valid JSON")


def _summary(rules) -> list[tuple[str, str, str, str]]:
    return [(rule.name, rule.condition, rule.action, rule.notify) for rule in rules]


class TestNannyRuleParsing(unittest.TestCase):
    def test_parse_numeric_condition(self):
        from agent_trace.watch import NannyRule
        rule = NannyRule(name="cost", condition="cost_usd > 5.00", action="kill")
        self.assertEqual(rule._metric, "cost_usd")
        self.assertEqual(rule._op, ">")
        self.assertEqual(rule._threshold, 5.0)

    def test_parse_file_path_matches(self):
        from agent_trace.watch import NannyRule
        rule = NannyRule(name="sensitive", condition='file_path matches "/etc/**"', action="kill")
        self.assertEqual(rule._metric, "file_path")
        self.assertEqual(rule._op, "matches")
        self.assertEqual(rule._pattern, "/etc/**")

    def test_evaluate_numeric_true(self):
        from agent_trace.watch import NannyRule
        rule = NannyRule(name="cost", condition="cost_usd > 5.00", action="kill")
        self.assertTrue(rule.evaluate({"cost_usd": 6.0}))

    def test_evaluate_numeric_false(self):
        from agent_trace.watch import NannyRule
        rule = NannyRule(name="cost", condition="cost_usd > 5.00", action="kill")
        self.assertFalse(rule.evaluate({"cost_usd": 3.0}))

    def test_evaluate_files_modified_over(self):
        from agent_trace.watch import NannyRule
        rule = NannyRule(name="files", condition="files_modified > 20", action="pause")
        self.assertTrue(rule.evaluate({"files_modified": 21}))

    def test_evaluate_files_modified_at_limit(self):
        from agent_trace.watch import NannyRule
        rule = NannyRule(name="files", condition="files_modified > 20", action="pause")
        self.assertFalse(rule.evaluate({"files_modified": 20}))

    def test_evaluate_file_path_matches(self):
        from agent_trace.watch import NannyRule
        rule = NannyRule(name="sensitive", condition='file_path matches "/etc/**"', action="kill")
        event = TraceEvent(
            event_type=EventType.TOOL_CALL,
            data={"tool_name": "write", "arguments": {"file_path": "/etc/passwd"}},
        )
        self.assertTrue(rule.evaluate({}, event))

    def test_evaluate_file_path_no_match(self):
        from agent_trace.watch import NannyRule
        rule = NannyRule(name="sensitive", condition='file_path matches "/etc/**"', action="kill")
        event = TraceEvent(
            event_type=EventType.TOOL_CALL,
            data={"tool_name": "write", "arguments": {"file_path": "/home/user/file.py"}},
        )
        self.assertFalse(rule.evaluate({}, event))

    def test_conditions_parse_as_before(self):
        # Guard: these conditions were parsed the same way before watch
        # checked them.
        from agent_trace.watch import NannyRule
        cases = (
            ("cost_usd > 0.50", ("cost_usd", ">", 0.5, "")),
            ("files_modified>=20", ("files_modified", ">=", 20, "")),
            ("  duration_minutes  <  30  ", ("duration_minutes", "<", 30, "")),
            ("consecutive_test_failures <= 3", ("consecutive_test_failures", "<=", 3, "")),
            ("cost_usd == .5", ("cost_usd", "==", 0.5, "")),
            ("cost_usd > 5.", ("cost_usd", ">", 5.0, "")),
            ("cost_usd > 0", ("cost_usd", ">", 0, "")),
            (_ENV, ("file_path", "matches", None, "**/production.env")),
            ("file_path  matches  '/etc/*'", ("file_path", "matches", None, "/etc/*")),
            ('file_path matches "it\'s *"', ("file_path", "matches", None, "it's *")),
            ("file_path matches '\"*\"'", ("file_path", "matches", None, '"*"')),
        )
        for condition, parsed in cases:
            with self.subTest(condition=condition):
                rule = NannyRule(name="r", condition=condition, action="kill")
                self.assertEqual(
                    (rule._metric, rule._op, rule._threshold, rule._pattern), parsed,
                )
                self.assertIs(type(rule._threshold), type(parsed[2]))

    def test_conditions_it_cannot_evaluate_are_refused(self):
        from agent_trace.watch import NannyRule, WatchRulesError
        cases = (
            ("cost_usd > $0.50", _NUMBER),  # R04
            ("cost_usd > 0.5.0", _NUMBER),  # R08
            ("cost_usd > 0.5 and files_modified > 3", _NUMBER),  # R12
            ("cost_usd > 0.50 USD", _NUMBER),
            ("cost_usd > -1", _NUMBER),
            ("cost_usd > +1", _NUMBER),
            ("cost_usd > 1e3", _NUMBER),
            ("cost_usd > 1_000", _NUMBER),
            ("cost_usd > inf", _NUMBER),
            ("cost_usd > nan", _NUMBER),
            ("cost_usd > .", _NUMBER),
            ("cost_usd >", _NUMBER),
            ("cost_usd > \u0663", _NUMBER),  # ARABIC-INDIC DIGIT THREE
            ("cost > 0.50", _unknown_metric("cost")),  # R05
            ("Cost_usd > 0.50", _unknown_metric("Cost_usd")),
            ("file_path > 1", _unknown_metric("file_path")),
            ("cost_usd => 1", "condition has an unknown operator '=>'; use >, >=, <, <= or =="),
            ("cost_usd = 1", "condition has an unknown operator '='; use >, >=, <, <= or =="),
            ("cost_usd != 1", "condition has an unknown operator '!='; use >, >=, <, <= or =="),
            ("cost_usd >> 1", "condition has an unknown operator '>>'; use >, >=, <, <= or =="),
            ("file_path matches **/production.env", _GLOB),
            ('file_path matches ""', _GLOB),
            ("file_path matches ''", _GLOB),
            ('file_path matches "a" or "b"', _GLOB),
            ('file_path matches "a', _GLOB),
            ("file_path matches 'a\"", _GLOB),
            ('file_path matches "a"b"', _GLOB),
            ("file_path matches", _GLOB),
            ("", _FORM),
            ("   ", _FORM),
            ("always", _FORM),
            ("cost_usd 5", _FORM),
            ("file_path contains 'x'", _FORM),
            ("cost_usd \u2265 1", _FORM),
        )
        for condition, problem in cases:
            with self.subTest(condition=condition):
                with self.assertRaises(WatchRulesError) as ctx:
                    NannyRule(name="r", condition=condition, action="kill")
                self.assertEqual(str(ctx.exception), problem)
        for condition in (["cost_usd > 0.50"], None, 5, {"cost_usd": 0.5}):
            with self.subTest(condition=condition):
                with self.assertRaises(WatchRulesError) as ctx:
                    NannyRule(name="r", condition=condition, action="kill")
                self.assertEqual(str(ctx.exception), "condition must be a string")

    def test_conditions_may_space_their_parts_freely(self):
        from agent_trace.watch import NannyRule
        cases = (
            ("cost_usd>\n0.50", ("cost_usd", ">", 0.5, "")),
            ('file_path matches"**/.env"', ("file_path", "matches", None, "**/.env")),
            ("file_path\tmatches\t'**/.env'", ("file_path", "matches", None, "**/.env")),
        )
        for condition, parsed in cases:
            with self.subTest(condition=condition):
                rule = NannyRule(name="r", condition=condition, action="kill")
                self.assertEqual(
                    (rule._metric, rule._op, rule._threshold, rule._pattern), parsed,
                )


class TestNannyRuleLoading(unittest.TestCase):
    def test_load_json_rules(self):
        import tempfile, os
        from agent_trace.watch import _load_nanny_rules
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"rules": [
                {"name": "cost-limit", "condition": "cost_usd > 5.00", "action": "kill"},
                {"name": "too-many-files", "condition": "files_modified > 20", "action": "pause"},
            ]}, f)
            path = f.name
        try:
            rules = _load_nanny_rules(path)
            self.assertEqual(len(rules), 2)
            self.assertEqual(rules[0].name, "cost-limit")
            self.assertEqual(rules[1].action, "pause")
        finally:
            os.unlink(path)

    def test_load_yaml_rules(self):
        import tempfile, os
        from agent_trace.watch import _load_nanny_rules
        yaml_text = (
            "rules:\n"
            "  - name: cost-limit\n"
            "    condition: cost_usd > 5.00\n"
            "    action: kill\n"
            "  - name: too-many-files\n"
            "    condition: files_modified > 20\n"
            "    action: pause\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            path = f.name
        try:
            rules = _load_nanny_rules(path)
            self.assertEqual(len(rules), 2)
            self.assertEqual(rules[0].name, "cost-limit")
            self.assertEqual(rules[1].name, "too-many-files")
        finally:
            os.unlink(path)

    def test_load_missing_file(self):
        from agent_trace.watch import WatchRulesError, _load_nanny_rules
        path = "/nonexistent/path/rules.yaml"
        with self.assertRaises(OSError) as read_error:
            Path(path).read_bytes()
        with self.assertRaises(WatchRulesError) as ctx:
            _load_nanny_rules(path)
        self.assertEqual(str(ctx.exception), f"cannot read it ({read_error.exception.strerror})")


class TestNannyRulesFile(unittest.TestCase):
    """_load_nanny_rules returns every rule in a file as written, or raises
    WatchRulesError: it never drops, skips or changes a rule."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def _write(self, data, name="rules.json") -> str:
        """Write *data* to *name* and return its path: bytes as given, str as
        UTF-8, anything else as JSON."""
        if isinstance(data, str):
            data = data.encode("utf-8")
        elif not isinstance(data, bytes):
            data = json.dumps(data).encode("utf-8")
        path = self.root / name
        path.write_bytes(data)
        return str(path)

    def _load(self, data, name="rules.json"):
        from agent_trace.watch import _load_nanny_rules
        return _summary(_load_nanny_rules(self._write(data, name)))

    def _assert_path_refused(self, path, problem):
        from agent_trace.watch import WatchRulesError, _load_nanny_rules
        with self.assertRaises(WatchRulesError) as ctx:
            _load_nanny_rules(path)
        self.assertEqual(str(ctx.exception), problem)
        self.assertNotIn("SECRET", str(ctx.exception))

    def _assert_refused(self, data, problem, name="rules.json"):
        self._assert_path_refused(self._write(data, name), problem)

    def test_files_that_loaded_before_load_the_same(self):
        # Guard: these files loaded the same way before watch checked them.
        json_rules = {"rules": [
            _COST_KILL,
            {"name": "env", "condition": "file_path matches '**/.env'", "action": "pause",
             "notify": "slack:#alerts"},
            {"condition": "files_modified >= 3", "action": "alert"},
        ]}
        json_expected = [
            ("cost cap", "cost_usd > 0.50", "kill", ""),
            ("env", "file_path matches '**/.env'", "pause", "slack:#alerts"),
            ("unnamed", "files_modified >= 3", "alert", ""),
        ]
        yaml_text = (
            "# watch rules\n"
            "rules:\n"
            "  - name: cost cap\n"
            "    condition: cost_usd > 0.50\n"
            "    action: kill\n"
            "\n"
            '  - name: "env"\n'
            # R29: a single-quoted value that holds double quotes.
            "    condition: 'file_path matches \"**/.env\"'\n"
            "    action: pause\n"
            "    notify: slack:#alerts\n"
            "  - condition: files_modified >= 3\n"
            "    action: alert\n"
        )
        yaml_expected = [
            ("cost cap", "cost_usd > 0.50", "kill", ""),
            ("env", 'file_path matches "**/.env"', "pause", "slack:#alerts"),
            ("unnamed", "files_modified >= 3", "alert", ""),
        ]
        cases = (
            ("rules.json", json_rules, json_expected),
            ("rules.JSON", json_rules, json_expected),
            ("rules", json_rules, json_expected),
            ("rules.yaml", yaml_text, yaml_expected),
            ("rules.yml", yaml_text, yaml_expected),
            ("rules.txt", yaml_text, yaml_expected),
            ("rules", yaml_text, yaml_expected),
            ("rules.yaml", _COST_KILL_YAML, [("cost cap", "cost_usd > 0.50", "kill", "")]),
        )
        for name, data, expected in cases:
            with self.subTest(name=name, data=data):
                self.assertEqual(self._load(data, name), expected)

    def test_yaml_values_are_read_as_yaml_reads_them(self):
        cases = (
            # R21: a comment after a value is not part of it.
            (
                "rules:\n"
                "  - name: cost cap # the cap\n"
                "    condition: cost_usd > 0.50\n"
                "    action: kill  # stop the agent\n",
                [("cost cap", "cost_usd > 0.50", "kill", "")],
            ),
            # R28: an unquoted value keeps the quotes inside it.
            (
                "rules:\n"
                "  - name: protected env\n"
                f"    condition: {_ENV}\n"
                "    action: kill\n",
                [("protected env", _ENV, "kill", "")],
            ),
            # A quoted value loses one pair of quotes and nothing else.
            (
                "rules:\n"
                "  - name: '\"quoted\" name'\n"
                "    condition: \"file_path matches 'C:\\\\*'\"\n"
                '    action: "kill"  # a comment after quotes\n'
                "    notify: 'it''s #here'\n",
                [('"quoted" name', "file_path matches 'C:\\*'", "kill", "it's #here")],
            ),
            (
                "rules:\n"
                "  - name: \"say \\\"stop\\\"\"\n"
                "    condition: cost_usd > 1\n"
                "    action: alert\n"
                "    notify: slack:#alerts\n",
                [('say "stop"', "cost_usd > 1", "alert", "slack:#alerts")],
            ),
            # A leading ---, rules at column 0, a rule that starts with a bare
            # dash, comments, blank lines, trailing spaces and CRLF.
            (
                "---\r\n"
                "# watch rules\r\n"
                "rules: # all of them\r\n"
                "\r\n"
                "- name: a  \r\n"
                "  condition: files_modified >= 3\r\n"
                "  action: pause\r\n"
                "-\r\n"
                "   condition: duration_minutes > 30\r\n"
                "   # still the second rule\r\n"
                "   action: alert\r\n",
                [
                    ("a", "files_modified >= 3", "pause", ""),
                    ("unnamed", "duration_minutes > 30", "alert", ""),
                ],
            ),
        )
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(self._load(text, "rules.yaml"), expected)

    def test_top_level_problems_are_refused(self):
        cases = [
            ([_COST_KILL], "the top level must be a JSON object"),  # R09
            ({}, "the top level has no rules key"),  # R13
            ({"rule": [_COST_KILL]},
             "the top level has an unknown key 'rule'; it must hold only rules"),
            ({"rules": [_COST_KILL], "version": 1},
             "the top level has an unknown key 'version'; it must hold only rules"),
            ('{"rules": [], "rules": [%s]}' % json.dumps(_COST_KILL),
             "the top level has 'rules' twice"),
        ]
        for value in ("SECRET", None, 1, 0.5, True):
            cases.append((json.dumps(value), "the top level must be a JSON object"))
        for value in ([], {}, None, "SECRET", 1, True):
            cases.append(({"rules": value}, "rules must be a non-empty list"))
        for data, problem in cases:
            with self.subTest(data=data):
                self._assert_refused(data, problem)

    def test_rule_problems_are_refused_with_position_and_name(self):
        good = {"name": "big", "condition": "files_modified > 100", "action": "alert"}
        cases = [
            ([dict(_COST_KILL, condition="cost_usd > $0.50")],  # R04
             f"rule 1 'cost cap': {_NUMBER}"),
            ([dict(_COST_KILL, condition="cost > 0.50")],  # R05
             f"rule 1 'cost cap': {_unknown_metric('cost')}"),
            ([good, dict(_COST_KILL, condition="cost_usd > 0.5.0")],  # R08
             f"rule 2 'cost cap': {_NUMBER}"),
            ([dict(_COST_KILL, condition="cost_usd > 0.5 and files_modified > 3")],  # R12
             f"rule 1 'cost cap': {_NUMBER}"),
            ([dict(_COST_KILL, condition=["cost_usd > 0.50"])],  # R15
             "rule 1 'cost cap': condition must be a string"),
            ([dict(_COST_KILL, condition=None)], "rule 1 'cost cap': condition must be a string"),
            ([{"name": "cost cap", "condition": "cost_usd > 0.50", "acton": "kill"}],  # R20
             f"rule 1 'cost cap': {_unknown_key('acton')}"),
            ([{"name": "cost cap", "condtion": "cost_usd > 0.50", "action": "kill"}],  # R26
             f"rule 1 'cost cap': {_unknown_key('condtion')}"),
            ([dict(_COST_KILL, notify="slack:#SECRET", notfy="slack:#SECRET")],
             f"rule 1 'cost cap': {_unknown_key('notfy')}"),
            ([{"name": "cost cap", "action": "kill"}], "rule 1 'cost cap': it has no condition"),
            # Before, a rule without an action was an alert rule.
            ([{"name": "cost cap", "condition": "cost_usd > 0.50"}],
             "rule 1 'cost cap': it has no action"),
            ([good, "kill"], "rule 2 must be an object"),
            ([None], "rule 1 must be an object"),
            ([["SECRET"]], "rule 1 must be an object"),
            # A rule without a name, or with an empty one, is named by its
            # position. A name is shown with repr, so it stays on one line.
            ([{"condition": "cost > 1", "action": "kill"}],
             f"rule 1: {_unknown_metric('cost')}"),
            ([dict(_COST_KILL, name="", action="kil")], f"rule 1: {_ACTIONS}"),
            ([dict(_COST_KILL, name="two\nlines", action="kil")],
             f"rule 1 'two\\nlines': {_ACTIONS}"),
            ('{"rules": [{"name": "cost cap", "condition": "cost_usd > 0.50", '
             '"action": "alert", "action": "kill"}]}',
             "rule 1 'cost cap': it has 'action' twice"),
        ]
        # R06, R07 and R27 among them.
        for action in ("kil", "Kill", "KILL", " kill", "kill ", "", "block", "notify", "SECRET",
                       ["kill"], None, 1, True, {"kill": 1}):
            cases.append(([dict(_COST_KILL, action=action, notify="slack:#SECRET")],
                          f"rule 1 'cost cap': {_ACTIONS}"))
        for notify in (["SECRET"], None, 1, {"SECRET": 1}):
            cases.append(([dict(_COST_KILL, notify=notify)],
                          "rule 1 'cost cap': notify must be a string"))
        for name in (None, 1, ["SECRET"]):
            cases.append(([dict(_COST_KILL, name=name)], "rule 1: name must be a string"))
        for rules, problem in cases:
            with self.subTest(rules=rules):
                self._assert_refused(
                    rules if isinstance(rules, str) else {"rules": rules}, problem,
                )

    def test_files_it_cannot_read_as_text_are_refused(self):
        missing = self.root / "missing.yaml"
        with self.assertRaises(OSError) as ctx:
            missing.read_bytes()
        self._assert_path_refused(str(missing), f"cannot read it ({ctx.exception.strerror})")
        self._assert_path_refused("", "the path is empty")
        for name in ("rulesdir", "rules.json"):  # R18
            folder = self.root / "folders" / name
            folder.mkdir(parents=True)
            with self.subTest(folder=name):
                self._assert_path_refused(str(folder), "it is a directory, not a rules file")
        text = json.dumps({"rules": [_COST_KILL]})
        cases = (
            # R19
            (b'{"rules": [{"name": "caf\xe9", "condition": "cost_usd > 0.50", '
             b'"action": "kill"}]}', "it is not valid UTF-8"),
            (text.encode("utf-16"), "it is not valid UTF-8"),
            (b"\xef\xbb\xbf" + text.encode("utf-8"),
             "it starts with a byte order mark; save it without one"),
            (b"", "it is empty"),
            (b" \n\t\r\n", "it is empty"),
        )
        for name in ("rules.json", "rules.yaml"):
            for data, problem in cases:
                with self.subTest(name=name, data=data[:20]):
                    self._assert_refused(data, problem, name)

    @unittest.skipUnless(hasattr(os, "geteuid"), "needs POSIX file permissions")
    def test_file_without_read_permission_is_refused(self):
        locked = Path(self._write({"rules": [_COST_KILL]}))
        locked.chmod(0)
        self.addCleanup(locked.chmod, 0o600)
        try:
            locked.read_bytes()
        except PermissionError as exc:
            reason = exc.strerror
        else:
            self.skipTest("this user can read a file without read permission")
        self._assert_path_refused(str(locked), f"cannot read it ({reason})")

    def test_json_errors_are_refused_and_never_read_as_yaml(self):
        cases = (
            # R03: a trailing comma.
            ("rules.json", json.dumps({"rules": [_COST_KILL]})[:-1] + ",}"),
            # A .json file is JSON, even when it holds YAML.
            ("rules.json", _COST_KILL_YAML),
            # So is a file whose text starts with { or [.
            ("rules.yaml", json.dumps({"rules": [_COST_KILL]})[:-2] + ",]}"),
            ("rules.yaml", "\n  {rules: [{name: cost cap}]}\n"),
            ("rules", "[1, 2,]"),
            ("rules.json", '{"rules": [{"name": "cost cap", "notify": "SECRET'),
            ("rules.json", "{'rules': []}"),
        )
        for name, text in cases:
            with self.subTest(name=name, text=text):
                self._assert_refused(text, _json_error(text), name)
        self._assert_refused(b'{"rules": ' + b"[" * 1_000_000, "it is nested too deeply")
        if hasattr(sys, "get_int_max_str_digits"):
            self._assert_refused(
                b'{"rules": [{"name": ' + b"1" * 5000 + b"}]}",
                "it has a number with too many digits",
            )

    def test_yaml_lines_outside_the_subset_are_refused(self):
        head = "rules:\n  - name: cost cap\n    condition: cost_usd > 0.50\n"
        value = "a value cannot start with {!r} unless it is quoted"
        colon = "a value cannot contain ': ' or end with ':' unless it is quoted"
        cases = (
            # R11
            ("rule:\n  - name: cost cap\n    condition: cost_usd > 0.50\n    action: kill\n",
             "line 1: the top level has an unknown key 'rule'; it must hold only rules"),
            ("version: 1\n" + _COST_KILL_YAML,
             "line 1: the top level has an unknown key 'version'; it must hold only rules"),
            (_COST_KILL_YAML + "version: 1\n",
             "line 5: the top level has an unknown key 'version'; it must hold only rules"),
            (_COST_KILL_YAML + "rules:\n", "line 5: the top level has 'rules' twice"),
            (_COST_KILL_YAML + "---\n", "line 5: expected - at the start of a rule"),
            ("---\n---\n" + _COST_KILL_YAML, "line 2: expected rules:"),
            ("SECRET\n", "line 1: expected rules:"),
            ("  rules:\n", "line 1: unexpected indentation"),
            ("rules: []\n", "line 1: list the rules on the lines below rules:"),
            ("rules: [{name: SECRET}]\n", "line 1: list the rules on the lines below rules:"),
            ("rules:\n", "rules must be a non-empty list"),
            ("rules:\n# none yet\n", "rules must be a non-empty list"),
            ("rules:\n  name: cost cap\n", "line 2: expected - at the start of a rule"),
            ("rules:\n  -name: cost cap\n", "line 2: expected - at the start of a rule"),
            ("rules:\nname: cost cap\n",
             "line 2: the top level has an unknown key 'name'; it must hold only rules"),
            (_COST_KILL_YAML + " - name: b\n", "line 5: unexpected indentation"),
            (head + "     action: kill\n", "line 4: unexpected indentation"),
            (head + "  action: kill\n", "line 4: unexpected indentation"),
            (head + "    action:\n      kill\n", "line 5: unexpected indentation"),
            (head + "\taction: kill\n", "line 4: indent with spaces, not tabs"),
            ("rules:\n  - \tname: cost cap\n", "line 2: indent with spaces, not tabs"),
            (head + "    action kill\n", "line 4: expected key: value"),
            (head + "    action:kill\n", "line 4: expected key: value"),
            (head + '    "action": kill\n', "line 4: expected key: value"),
            (head + "    action: 'kill\n", "line 4: the closing quote is missing"),
            (head + '    notify: "SECRET\n', "line 4: the closing quote is missing"),
            (head + '    action: "kill" now\n',
             "line 4: only a comment may follow the closing quote"),
            (head + '    action: "kill"# now\n',
             "line 4: only a comment may follow the closing quote"),
            (head + '    notify: "\\SECRET"\n',
             'line 4: in double quotes a backslash may only escape " or \\; '
             "use single quotes for other backslashes"),
            (head + "    action: [kill]\n", "line 4: " + value.format("[")),
            (head + "    notify: {SECRET: 1}\n", "line 4: " + value.format("{")),
            (head + "    action: >\n      kill\n", "line 4: " + value.format(">")),
            (head + "    action: |\n      kill\n", "line 4: " + value.format("|")),
            (head + "    action: &a kill\n", "line 4: " + value.format("&")),
            (head + "    action: *a\n", "line 4: " + value.format("*")),
            (head + "    action: !!str kill\n", "line 4: " + value.format("!")),
            (head + "    action: - kill\n", "line 4: " + value.format("-")),
            (head + "    action: kill: now\n", "line 4: " + colon),
            (head + "    notify: SECRET:\n", "line 4: " + colon),
        )
        for text, problem in cases:
            with self.subTest(text=text):
                self._assert_refused(text, problem, "rules.yaml")

    def test_yaml_rule_problems_are_refused_with_position_and_name(self):
        head = "rules:\n  - name: cost cap\n    condition: cost_usd > 0.50\n"
        cases = (
            # R26
            ("rules:\n  - name: cost cap\n    condtion: cost_usd > 0.50\n    action: kill\n",
             f"rule 1 'cost cap': {_unknown_key('condtion')}"),
            (head + "    action: alert\n    action: kill\n",
             "rule 1 'cost cap': it has 'action' twice"),
            (head, "rule 1 'cost cap': it has no action"),
            (head + "    action: Kill\n", f"rule 1 'cost cap': {_ACTIONS}"),
            (head + "    action: kill # SECRET\n    notify: ~\n",
             "rule 1 'cost cap': notify must be a string"),
            (head + "    action:\n", f"rule 1 'cost cap': {_ACTIONS}"),
            (head + "    action: null\n", f"rule 1 'cost cap': {_ACTIONS}"),
            ("rules:\n  - name:\n    condition: cost_usd > 0.50\n    action: kill\n",
             "rule 1: name must be a string"),
            ("rules:\n  -\n", "rule 1 must be an object"),
            (_COST_KILL_YAML + "  -\n  - name: b\n", "rule 2 must be an object"),
        )
        for text, problem in cases:
            with self.subTest(text=text):
                self._assert_refused(text, problem, "rules.yaml")


class TestWatchStateNanny(unittest.TestCase):
    def test_nanny_metrics(self):
        from agent_trace.watch import WatchState
        state = WatchState(start_time=time.time() - 120)
        state.files_modified = 5
        state.estimated_cost = 2.5
        state.consecutive_test_failures = 2
        metrics = state.nanny_metrics()
        self.assertEqual(metrics["files_modified"], 5)
        self.assertEqual(metrics["cost_usd"], 2.5)
        self.assertEqual(metrics["consecutive_test_failures"], 2)
        self.assertGreaterEqual(metrics["duration_minutes"], 2.0)

    def test_check_event_tracks_files_modified(self):
        from agent_trace.watch import WatcherConfig, WatchState, check_event
        config = WatcherConfig()
        state = WatchState()
        check_event(_tool_call("write", {"file_path": "src/foo.py"}), config, state)
        check_event(_tool_call("write", {"file_path": "src/bar.py"}), config, state)
        check_event(_tool_call("write", {"file_path": "src/foo.py"}), config, state)  # duplicate
        self.assertEqual(state.files_modified, 2)

    def test_check_event_tracks_test_failures(self):
        from agent_trace.watch import WatcherConfig, WatchState, check_event
        config = WatcherConfig()
        state = WatchState()
        fail_event = TraceEvent(
            event_type=EventType.TOOL_RESULT,
            data={"content": "FAILED: test_foo AssertionError", "is_error": True},
        )
        check_event(fail_event, config, state)
        check_event(fail_event, config, state)
        self.assertEqual(state.consecutive_test_failures, 2)

    def test_cli_has_rules_and_dry_run_flags(self):
        from agent_trace.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["watch", "--rules", "rules.yaml", "--dry-run"])
        self.assertEqual(args.rules, "rules.yaml")
        self.assertTrue(args.dry_run)


class TestWatchRulesFlag(unittest.TestCase):
    """cmd_watch refuses a --rules value it cannot use with one line and
    status 1, before it reads the session or writes anything, so watch never
    runs with fewer or different rules than the user wrote."""

    # Flags that must not change which problem is reported.
    _OTHER_FLAGS = (
        "--agent-pid", "1", "--on-violation", "kill", "--max-cost", "1", "--timeout", "1m",
    )

    @classmethod
    def setUpClass(cls):
        from agent_trace.cli import build_parser
        cls.parser = build_parser()

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.trace_dir = self.root / "traces"
        store = TraceStore(self.trace_dir, redact=False)
        store.create_session(SessionMeta(session_id="rules-flag", started_at=1.0))
        store.append_event("rules-flag", TraceEvent(
            event_type=EventType.TOOL_CALL, timestamp=2.0, session_id="rules-flag",
            data={"tool_name": "Bash", "arguments": {"command": "ls"}},
        ))
        # A writable TraceStore writes this file back, so a refusal that
        # opened the store first changes the snapshot.
        (self.trace_dir / ".gitignore").unlink(missing_ok=True)
        # --rules reads files relative to the working directory, and cmd_watch
        # reads project budget files from it.
        cwd = os.getcwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, cwd)
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("AGENT_STRACE_TENANT_ID", None)

    def _write(self, data, name="rules.json"):
        """Write *data* to *name* in the working directory: bytes as given,
        str as UTF-8, anything else as JSON."""
        if isinstance(data, str):
            data = data.encode("utf-8")
        elif not isinstance(data, bytes):
            data = json.dumps(data).encode("utf-8")
        (self.root / name).write_bytes(data)

    def _snapshot(self):
        return {
            path.relative_to(self.root).as_posix(): path.read_bytes() if path.is_file() else None
            for path in sorted(self.root.rglob("*"))
        }

    def _run(self, value, *flags, session="rules-flag"):
        from agent_trace.watch import cmd_watch
        args = self.parser.parse_args(
            ["--trace-dir", str(self.trace_dir), "watch", session, f"--rules={value}", *flags],
        )
        out, err = io.StringIO(), io.StringIO()
        with patch("agent_trace.watch.watch_session") as watch_session_mock, \
                redirect_stdout(out), redirect_stderr(err):
            status = cmd_watch(args)
        return status, out.getvalue(), err.getvalue(), watch_session_mock

    def _watched_with(self, value, *flags):
        """Return the config and rules cmd_watch passes to watch_session."""
        status, out, err, watch_session_mock = self._run(value, *flags)
        self.assertEqual((status, out, err), (0, "", ""))
        watch_session_mock.assert_called_once()
        call = watch_session_mock.call_args
        return call.args[2], call.kwargs["nanny_rules"]

    def _assert_refused(self, value, line):
        """Assert cmd_watch prints *line*, exits 1 and writes nothing, with or
        without --dry-run, a session that exists and other flags."""
        for session, flags in (
            ("rules-flag", ()),
            ("rules-flag", ("--dry-run",)),
            ("missing-session", self._OTHER_FLAGS),
            ("missing-session", (*self._OTHER_FLAGS, "--dry-run")),
        ):
            with self.subTest(value=value, session=session, flags=flags):
                before = self._snapshot()
                status, out, err, watch_session_mock = self._run(value, *flags, session=session)
                self.assertEqual((status, out, err), (1, "", line))
                self.assertNotIn("SECRET", err)
                watch_session_mock.assert_not_called()
                self.assertEqual(self._snapshot(), before)

    def test_rules_files_it_cannot_use_are_refused(self):
        trailing_comma = json.dumps({"rules": [_COST_KILL]})[:-1] + ",}"
        cases = (
            ("rules.json", trailing_comma, _json_error(trailing_comma)),  # R03
            ("rules.json", {"rules": [dict(_COST_KILL, condition="cost_usd > $0.50")]},  # R04
             f"rule 1 'cost cap': {_NUMBER}"),
            ("rules.json", {"rules": [dict(_COST_KILL, condition="cost > 0.50")]},  # R05
             f"rule 1 'cost cap': {_unknown_metric('cost')}"),
            ("rules.json", {"rules": [dict(_COST_KILL, action="kil")]},  # R06
             f"rule 1 'cost cap': {_ACTIONS}"),
            ("rules.json", {"rules": [dict(_COST_KILL, action="Kill")]},  # R07
             f"rule 1 'cost cap': {_ACTIONS}"),
            ("rules.json", {"rules": [  # R08
                {"name": "big", "condition": "files_modified > 100", "action": "alert"},
                dict(_COST_KILL, condition="cost_usd > 0.5.0"),
            ]}, f"rule 2 'cost cap': {_NUMBER}"),
            ("rules.json", [_COST_KILL], "the top level must be a JSON object"),  # R09
            ("rules.yaml", _COST_KILL_YAML.replace("rules:", "rule:"),  # R11
             "line 1: the top level has an unknown key 'rule'; it must hold only rules"),
            ("rules.json", {"rules": [dict(  # R12
                _COST_KILL, condition="cost_usd > 0.5 and files_modified > 3",
            )]}, f"rule 1 'cost cap': {_NUMBER}"),
            ("rules.json", {}, "the top level has no rules key"),  # R13
            ("rules.json", {"rules": [dict(_COST_KILL, condition=["cost_usd > 0.50"])]},  # R15
             "rule 1 'cost cap': condition must be a string"),
            ("rules.json", json.dumps(  # R19
                {"rules": [dict(_COST_KILL, name="caf\u00e9")]}, ensure_ascii=False,
            ).encode("latin-1"), "it is not valid UTF-8"),
            ("rules.json", {"rules": [{  # R20
                "name": "cost cap", "condition": "cost_usd > 0.50", "acton": "kill",
            }]}, f"rule 1 'cost cap': {_unknown_key('acton')}"),
            ("rules.yaml", _COST_KILL_YAML.replace("condition:", "condtion:"),  # R26
             f"rule 1 'cost cap': {_unknown_key('condtion')}"),
            ("rules.json", {"rules": [dict(_COST_KILL, action=["kill"])]},  # R27
             f"rule 1 'cost cap': {_ACTIONS}"),
            # Before, a rule without an action was an alert rule.
            ("rules.json", {"rules": [{"name": "cost cap", "condition": "cost_usd > 0.50"}]},
             "rule 1 'cost cap': it has no action"),
            ("rules.json", {"rules": [dict(_COST_KILL, action="kil", notify="slack:#SECRET")]},
             f"rule 1 'cost cap': {_ACTIONS}"),
            ("rules.yaml", _COST_KILL_YAML + '    notify: "SECRET\n',
             "line 5: the closing quote is missing"),
        )
        for name, data, problem in cases:
            self._write(data, name)
            self._assert_refused(name, f"[watch] invalid --rules file {name}: {problem}\n")
        absolute = str(self.root / "rules.json")
        self._write({"rules": [dict(_COST_KILL, action="kil")]})
        self._assert_refused(
            absolute, f"[watch] invalid --rules file {absolute}: rule 1 'cost cap': {_ACTIONS}\n",
        )
        (self.root / "rulesdir").mkdir()
        self._assert_refused(  # R18
            "rulesdir", "[watch] invalid --rules file rulesdir: it is a directory, not a rules file\n",
        )

    def test_missing_rules_files_are_refused(self):
        for value in ("missing.yaml", "missing.JSON", "rules/missing", "rules\\missing",
                      str(self.root / "missing.yml"), "rules\n.json"):
            with self.assertRaises(OSError) as ctx:
                Path(value).read_bytes()
            shown = value if value.isprintable() else repr(value)
            self._assert_refused(
                value,
                f"[watch] invalid --rules file {shown}: cannot read it ({ctx.exception.strerror})\n",
            )

    def test_values_that_are_neither_a_rules_file_nor_built_in_rules_are_refused(self):
        self._write({"rules": [_COST_KILL]})  # rules.json exists
        not_both = (
            "is not a built-in rule; --rules takes one rules file or a list of built-in "
            "rules, not both"
        )
        cases = (
            # An unset variable must not drop the rules.
            ("", "it is empty"),
            ("   ", "it is empty"),
            (",", "it lists no rules"),  # R24
            (" , ,", "it lists no rules"),
            ("budget:2,,loop", "item 2 is empty"),
            ("budget:2,", "item 2 is empty"),
            (",budget:2", "item 1 is empty"),
            ("rules.jsn", f"'rules.jsn' is not a built-in rule or an existing file; {_BUILT_INS}"),  # R02
            ("Budget:5", f"'Budget:5' is not a built-in rule or an existing file; {_BUILT_INS}"),
            ("loop,kill", f"'kill' is not a built-in rule or an existing file; {_BUILT_INS}"),
            ("missing.json,budget:2",
             f"'missing.json' is not a built-in rule or an existing file; {_BUILT_INS}"),
            ("rules.json,budget:2", f"'rules.json' {not_both}"),  # R25
            ("budget:2, rules.json", f"'rules.json' {not_both}"),
            ("budget:2,budget:3", "'budget:3' lists budget again; list each built-in rule once"),
            ("loop,loop:3", "'loop:3' lists loop again; list each built-in rule once"),
        )
        for value, problem in cases:
            self._assert_refused(value, f"[watch] invalid --rules value: {problem}\n")

    def test_built_in_rules_with_values_they_cannot_use_are_refused(self):
        budget = "N in budget:N must be a finite number, 0 or more"
        timeout = "D in timeout:D must be a duration more than 0, such as 30m or 2h"
        loop = "N and W in loop:N and loop:N/W must be whole numbers, 1 or more"
        debt = "N in cognitive-debt:N must be a number from 0 to 1"
        cases = [
            ("mcp-poisoning:1", "mcp-poisoning takes no value"),
            ("mcp-poisoning:", "mcp-poisoning takes no value"),
        ]
        # R14, R22 and R23 among them.
        cases += [(item, budget) for item in (
            "budget:2x", "budget:nan", "budget:-1", "budget:inf", "budget", "budget:",
            "budget:$$5", "budget:$-1", "budget:1e400",
        )]
        cases += [(item, timeout) for item in (
            "timeout", "timeout:", "timeout:0", "timeout:0s", "timeout:-5", "timeout:soon",
            "timeout:inf", "timeout:nan",
        )]
        cases += [(item, loop) for item in (
            "loop:", "loop:0", "loop:3/0", "loop:-1", "loop:1.5", "loop:3/", "loop:/3",
            "loop:x", "loop:3/4/5", "loop:+4", "loop:4_000",
        )]
        cases += [(item, debt) for item in (
            "cognitive-debt:", "cognitive-debt:2", "cognitive-debt:-0.1", "cognitive-debt:nan",
            "cognitive-debt:inf",
        )]
        for item, problem in cases:
            other = "loop" if item.startswith("mcp-poisoning") else "mcp-poisoning"
            for value in (item, f"{other}, {item}"):
                self._assert_refused(value, f"[watch] invalid --rules value: {item!r}: {problem}\n")

    def test_rules_files_that_loaded_before_still_load(self):
        # Guard: these values loaded the same rules before watch checked them.
        env_kill = {"name": "protected env", "condition": _ENV, "action": "kill"}
        cost = [("cost cap", "cost_usd > 0.50", "kill", "")]
        env = [("protected env", _ENV, "kill", "")]
        cases = (
            ("rules.json", {"rules": [_COST_KILL]}, "rules.json", cost),  # R01
            ("rules.yaml", _COST_KILL_YAML, "rules.yaml", cost),  # R10
            ("rules.json", {"rules": [env_kill]}, "rules.json", env),  # R16
            ("rules.yaml", (  # R29
                "rules:\n  - name: protected env\n"
                f"    condition: '{_ENV}'\n    action: kill\n"
            ), "rules.yaml", env),
            # A file named like a built-in rule, or with a comma in its name.
            ("loop", {"rules": [_COST_KILL]}, "loop", cost),
            ("a,b.json", {"rules": [_COST_KILL]}, "a,b.json", cost),
            ("rules.json", {"rules": [_COST_KILL]}, str(self.root / "rules.json"), cost),
        )
        for name, data, value, expected in cases:
            with self.subTest(value=value, data=data):
                self._write(data, name)
                config, rules = self._watched_with(value)
                self.assertEqual(_summary(rules), expected)
                self.assertEqual(config.built_in_rules, set())

    def test_built_in_rules_are_applied(self):
        # Guard: these lists set the same values before watch checked them.
        from agent_trace.watch import WatcherConfig
        default = WatcherConfig()
        fields = (
            "built_in_rules", "loop_threshold", "loop_window", "max_cost_dollars",
            "max_duration_seconds", "cognitive_debt_threshold",
        )
        cases = (
            (" mcp-poisoning , loop:4/12, budget:$2.50 ,timeout:30m,cognitive-debt:0.6",
             ({"mcp-poisoning", "loop", "cognitive-debt"}, 4, 12, 2.5, 1800.0, 0.6)),
            ("loop", ({"loop"}, default.loop_threshold, default.loop_window,
                      default.max_cost_dollars, default.max_duration_seconds, None)),
            ("loop:7,budget:0,timeout:90",
             ({"loop"}, 7, default.loop_window, 0.0, 90.0, None)),
            ("budget: 5,timeout:1h30m,cognitive-debt:1",
             ({"cognitive-debt"}, default.loop_threshold, default.loop_window, 5.0, 5400.0, 1.0)),
            ("budget:1e3,cognitive-debt:0",
             ({"cognitive-debt"}, default.loop_threshold, default.loop_window, 1000.0,
              default.max_duration_seconds, 0.0)),
        )
        for value, expected in cases:
            with self.subTest(value=value):
                config, rules = self._watched_with(value)
                self.assertIsNone(rules)
                self.assertEqual(tuple(getattr(config, field) for field in fields), expected)

    def test_built_in_values_override_flags_and_the_project_budget(self):
        # Guard: built-in values were applied after the flags and the project
        # budget before watch checked them.
        config_file = self.root / "watch.json"
        config_file.write_text(json.dumps({"budget": {
            "weekly": 50, "warn_at": 0.5, "stop_at": 0.9, "per_session_max": 4,
        }}))
        cases = (
            ("loop", ("--config", str(config_file)), "max_cost_dollars", 4.0),
            ("budget:3", ("--config", str(config_file)), "max_cost_dollars", 3.0),
            ("budget:3", ("--budget", "7"), "max_cost_dollars", 3.0),
            ("budget:3", ("--max-cost", "7"), "max_cost_dollars", 3.0),
            ("timeout:30m", ("--timeout", "2h"), "max_duration_seconds", 1800.0),
            ("timeout:30m", ("--max-duration", "7200"), "max_duration_seconds", 1800.0),
            ("loop:4", ("--loop-threshold", "9"), "loop_threshold", 4),
            ("loop:4/12", ("--loop-window", "30"), "loop_window", 12),
        )
        for value, flags, field, expected in cases:
            with self.subTest(value=value, flags=flags):
                config, _ = self._watched_with(value, *flags)
                self.assertEqual(getattr(config, field), expected)

    def test_yaml_comments_and_quotes_inside_values_now_load(self):
        cases = (
            # R21: the comment is not part of the action.
            (_COST_KILL_YAML.replace("action: kill", "action: kill  # stop the agent"),
             [("cost cap", "cost_usd > 0.50", "kill", "")],
             {"cost_usd": 3.0}, None),
            # R28: an unquoted value keeps the closing quote.
            (f"rules:\n  - name: protected env\n    condition: {_ENV}\n    action: kill\n",
             [("protected env", _ENV, "kill", "")],
             {}, _tool_call("Write", {"file_path": "/app/production.env", "content": "x"})),
        )
        for text, expected, metrics, event in cases:
            with self.subTest(text=text):
                self._write(text, "rules.yaml")
                _, rules = self._watched_with("rules.yaml")
                self.assertEqual(_summary(rules), expected)
                self.assertTrue(rules[0].evaluate(metrics, event))

    def test_cognitive_debt_without_a_value_keeps_the_configured_threshold(self):
        # Before, a bare cognitive-debt was an unknown built-in rule.
        from agent_trace.watch import WatcherConfig
        config, _ = self._watched_with("cognitive-debt")
        self.assertEqual(config.built_in_rules, {"cognitive-debt"})
        self.assertEqual(config.cognitive_debt_threshold, WatcherConfig().cognitive_debt_threshold)
        config_file = self.root / "watch.json"
        config_file.write_text(json.dumps({"cognitive_debt_threshold": 0.3}))
        config, _ = self._watched_with("loop,cognitive-debt", "--config", str(config_file))
        self.assertEqual(config.built_in_rules, {"loop", "cognitive-debt"})
        self.assertEqual(config.cognitive_debt_threshold, 0.3)
