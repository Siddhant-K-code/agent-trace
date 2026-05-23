"""Tests for trace anonymization (issue #95)."""

import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_trace.anonymize import (
    AnonymizationResult,
    AnonymizationRule,
    _apply_rules_to_string,
    _anonymize_value,
    _build_builtin_rules,
    _load_custom_rules,
    _parse_custom_rules,
    anonymize_event,
    anonymize_session,
    cmd_anonymize_export,
)
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore
import re


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store() -> tuple[TraceStore, str]:
    tmpdir = tempfile.mkdtemp()
    return TraceStore(tmpdir), tmpdir


def _add_session(store: TraceStore, events_data: list[dict] | None = None) -> SessionMeta:
    meta = SessionMeta()
    store.create_session(meta)
    for d in (events_data or []):
        ev = TraceEvent(
            event_type=EventType.TOOL_CALL,
            session_id=meta.session_id,
            data=d,
        )
        store.append_event(meta.session_id, ev)
    return meta


def _make_rule(pattern: str, replacement: str, desc: str = "test") -> AnonymizationRule:
    return AnonymizationRule(
        pattern=re.compile(pattern),
        replacement=replacement,
        description=desc,
    )


# ---------------------------------------------------------------------------
# AnonymizationResult
# ---------------------------------------------------------------------------

class TestAnonymizationResult(unittest.TestCase):
    def test_total_replacements_zero_initially(self):
        r = AnonymizationResult()
        self.assertEqual(r.total_replacements, 0)

    def test_record_increments_count(self):
        r = AnonymizationResult()
        r.record("emails", 3)
        r.record("emails", 2)
        self.assertEqual(r.rules_applied["emails"], 5)
        self.assertEqual(r.total_replacements, 5)

    def test_multiple_categories(self):
        r = AnonymizationResult()
        r.record("emails", 2)
        r.record("paths", 5)
        self.assertEqual(r.total_replacements, 7)


# ---------------------------------------------------------------------------
# _apply_rules_to_string
# ---------------------------------------------------------------------------

class TestApplyRulesToString(unittest.TestCase):
    def test_replaces_match(self):
        rule = _make_rule(r"\bfoo\b", "<FOO>")
        result = AnonymizationResult()
        out = _apply_rules_to_string("foo bar foo", [rule], result)
        self.assertEqual(out, "<FOO> bar <FOO>")
        self.assertEqual(result.rules_applied["test"], 2)

    def test_no_match_unchanged(self):
        rule = _make_rule(r"xyz", "<XYZ>")
        result = AnonymizationResult()
        out = _apply_rules_to_string("hello world", [rule], result)
        self.assertEqual(out, "hello world")
        self.assertEqual(result.total_replacements, 0)

    def test_multiple_rules_applied_in_order(self):
        rules = [
            _make_rule(r"foo", "<FOO>", "foo"),
            _make_rule(r"bar", "<BAR>", "bar"),
        ]
        result = AnonymizationResult()
        out = _apply_rules_to_string("foo bar", rules, result)
        self.assertEqual(out, "<FOO> <BAR>")


# ---------------------------------------------------------------------------
# _anonymize_value
# ---------------------------------------------------------------------------

class TestAnonymizeValue(unittest.TestCase):
    def test_string_value(self):
        rule = _make_rule(r"secret@example\.com", "<email>")
        result = AnonymizationResult()
        out = _anonymize_value("contact secret@example.com", [rule], result)
        self.assertEqual(out, "contact <email>")

    def test_nested_dict(self):
        rule = _make_rule(r"alice", "<user>")
        result = AnonymizationResult()
        data = {"user": "alice", "nested": {"owner": "alice"}}
        out = _anonymize_value(data, [rule], result)
        self.assertEqual(out["user"], "<user>")
        self.assertEqual(out["nested"]["owner"], "<user>")

    def test_list_of_strings(self):
        rule = _make_rule(r"alice", "<user>")
        result = AnonymizationResult()
        out = _anonymize_value(["alice", "bob", "alice"], [rule], result)
        self.assertEqual(out, ["<user>", "bob", "<user>"])

    def test_non_string_scalar_unchanged(self):
        rule = _make_rule(r"42", "<num>")
        result = AnonymizationResult()
        out = _anonymize_value(42, [rule], result)
        self.assertEqual(out, 42)  # int, not string — unchanged


# ---------------------------------------------------------------------------
# _build_builtin_rules — email detection
# ---------------------------------------------------------------------------

class TestBuiltinRules(unittest.TestCase):
    def test_email_rule_present(self):
        rules = _build_builtin_rules()
        descriptions = [r.description for r in rules]
        self.assertIn("email addresses", descriptions)

    def test_email_replaced(self):
        rules = _build_builtin_rules()
        result = AnonymizationResult()
        out = _apply_rules_to_string("contact user@example.com for help", rules, result)
        self.assertNotIn("user@example.com", out)
        self.assertIn("<email>", out)

    def test_home_dir_replaced(self):
        home = str(Path.home())
        if home == "/" or not home:
            self.skipTest("home dir is root, skip")
        rules = _build_builtin_rules()
        result = AnonymizationResult()
        path = f"{home}/projects/myapp/config.py"
        out = _apply_rules_to_string(path, rules, result)
        self.assertNotIn(home, out)
        self.assertIn("~/", out)


# ---------------------------------------------------------------------------
# _parse_custom_rules
# ---------------------------------------------------------------------------

class TestParseCustomRules(unittest.TestCase):
    def test_parses_pattern_and_replacement(self):
        yaml = "rules:\n  - pattern: 'ACME Corp'\n    replacement: '<company>'\n"
        rules = _parse_custom_rules(yaml)
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0].replacement, "<company>")

    def test_invalid_regex_skipped(self):
        yaml = "rules:\n  - pattern: '[invalid'\n    replacement: '<x>'\n"
        rules = _parse_custom_rules(yaml)
        self.assertEqual(len(rules), 0)

    def test_empty_yaml_returns_empty(self):
        rules = _parse_custom_rules("")
        self.assertEqual(rules, [])

    def test_multiple_rules(self):
        yaml = (
            "rules:\n"
            "  - pattern: 'foo'\n    replacement: '<foo>'\n"
            "  - pattern: 'bar'\n    replacement: '<bar>'\n"
        )
        rules = _parse_custom_rules(yaml)
        self.assertEqual(len(rules), 2)


# ---------------------------------------------------------------------------
# anonymize_event
# ---------------------------------------------------------------------------

class TestAnonymizeEvent(unittest.TestCase):
    def test_original_event_unchanged(self):
        ev = TraceEvent(
            event_type=EventType.TOOL_CALL,
            data={"tool_name": "Bash", "arguments": {"command": "echo user@example.com"}},
        )
        rule = _make_rule(r"user@example\.com", "<email>")
        result = AnonymizationResult()
        new_ev = anonymize_event(ev, [rule], result)
        # Original unchanged
        self.assertIn("user@example.com", ev.data["arguments"]["command"])
        # New event anonymized
        self.assertIn("<email>", new_ev.data["arguments"]["command"])

    def test_event_id_preserved(self):
        ev = TraceEvent(event_type=EventType.TOOL_CALL, data={"x": "y"})
        result = AnonymizationResult()
        new_ev = anonymize_event(ev, [], result)
        self.assertEqual(new_ev.event_id, ev.event_id)


# ---------------------------------------------------------------------------
# anonymize_session
# ---------------------------------------------------------------------------

class TestAnonymizeSession(unittest.TestCase):
    def test_anonymizes_email_in_events(self):
        store, _ = _make_store()
        meta = _add_session(store, [
            {"tool_name": "Bash", "arguments": {"command": "git config user.email alice@corp.com"}},
        ])
        events, result = anonymize_session(store, meta.session_id)
        cmd = events[0].data["arguments"]["command"]
        self.assertNotIn("alice@corp.com", cmd)
        self.assertIn("<email>", cmd)

    def test_result_records_replacements(self):
        store, _ = _make_store()
        meta = _add_session(store, [
            {"tool_name": "Bash", "arguments": {"command": "echo bob@example.com"}},
        ])
        _, result = anonymize_session(store, meta.session_id)
        self.assertGreater(result.total_replacements, 0)

    def test_empty_session_no_crash(self):
        store, _ = _make_store()
        meta = _add_session(store, [])
        events, result = anonymize_session(store, meta.session_id)
        self.assertEqual(events, [])
        self.assertEqual(result.total_replacements, 0)


# ---------------------------------------------------------------------------
# cmd_anonymize_export
# ---------------------------------------------------------------------------

class TestCmdAnonymizeExport(unittest.TestCase):
    def _make_args(self, store: TraceStore, session_id: str,
                   output: str = "", dry_run: bool = False) -> object:
        import argparse
        args = argparse.Namespace()
        args.trace_dir = str(store.base_dir)
        args.session_id = session_id
        args.anonymize_config = None
        args.output = output
        args.dry_run = dry_run
        return args

    def test_writes_output_file(self):
        store, tmpdir = _make_store()
        meta = _add_session(store, [{"tool_name": "Bash", "arguments": {"command": "echo hi"}}])
        output = os.path.join(tmpdir, "anon.json")
        args = self._make_args(store, meta.session_id, output=output)
        out = io.StringIO()
        rc = cmd_anonymize_export(args, out=out)
        self.assertEqual(rc, 0)
        self.assertTrue(Path(output).exists())
        data = json.loads(Path(output).read_text())
        self.assertEqual(data["session_id"], meta.session_id)
        self.assertTrue(data["anonymized"])

    def test_dry_run_does_not_write(self):
        store, tmpdir = _make_store()
        meta = _add_session(store, [{"tool_name": "Bash", "arguments": {"command": "echo hi"}}])
        output = os.path.join(tmpdir, "anon.json")
        args = self._make_args(store, meta.session_id, output=output, dry_run=True)
        out = io.StringIO()
        rc = cmd_anonymize_export(args, out=out)
        self.assertEqual(rc, 0)
        self.assertFalse(Path(output).exists())

    def test_missing_session_returns_error(self):
        store, _ = _make_store()
        import argparse
        args = argparse.Namespace()
        args.trace_dir = str(store.base_dir)
        args.session_id = "nonexistent"
        args.anonymize_config = None
        args.output = ""
        args.dry_run = False
        out = io.StringIO()
        rc = cmd_anonymize_export(args, out=out)
        self.assertEqual(rc, 1)


# ---------------------------------------------------------------------------
# CLI: --anonymize flag registered on export
# ---------------------------------------------------------------------------

class TestAnonymizeCLIFlag(unittest.TestCase):
    def test_anonymize_flag_in_export_help(self):
        import sys
        from agent_trace.cli import main
        old_argv = sys.argv
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.argv = ["agent-strace", "export", "--help"]
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
        self.assertIn("--anonymize", output)


if __name__ == "__main__":
    unittest.main()
