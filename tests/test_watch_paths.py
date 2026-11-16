"""Tests for the file paths, edits and loop keys watch reads from events.

The fixtures mirror what the hooks store. GitHub Copilot CLI sends an
apply_patch call's arguments as the patch text, and the hooks store that text
as ``arguments``. The hooks store the session's working directory only on
SESSION_START events, so relative patch paths resolve against it.
"""

import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

try:  # keeps product telemetry off in this process and its children
    import _no_telemetry  # noqa: F401
except ImportError:  # pragma: no cover - run as a package module
    from tests import _no_telemetry  # noqa: F401

from agent_trace import watch as watch_mod
from agent_trace.hooks import hook_main
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore
from agent_trace.watch import (
    NannyRule,
    OperationRule,
    WatcherConfig,
    WatchState,
    _event_key,
    check_event,
    watch_session,
)

# Names added with this behavior are imported inside the tests that use them.

SESSION = "paths-session"
ENV = 'file_path matches "**/production.env"'
NOTEBOOK = 'file_path matches "**/secret.ipynb"'
SECRETS = 'file_path matches "/app/secrets/*"'
ENV_DENIED = "ScopeWatcher: write to /app/config/production.env denied by policy"
LOOP = "LoopWatcher: identical tool call repeated 3 times in last 10 events"


def _patch(*lines: str, newline: str = "\n") -> str:
    return newline.join(("*** Begin Patch", *lines, "*** End Patch", ""))


ADD_ENV = _patch("*** Add File: config/production.env", "+X=1")
UPDATE_SECRET = _patch("*** Update File: secrets/key.pem", "@@", "-a", "+b")
MOVE_ENV = _patch(
    "*** Update File: src/old.env", "*** Move to: config/production.env", "@@", "-a", "+b",
)
DELETE_ENV = _patch("*** Delete File: config/production.env")
UPDATE_A = _patch("*** Update File: a.py", "@@", "-a", "+b")
UPDATE_B = _patch("*** Update File: b.py", "@@", "-a", "+b")
EDIT_A = {"path": "/app/a.py", "old_str": "a", "new_str": "b"}

# A preToolUse payload GitHub Copilot CLI 1.0.112 sent for apply_patch.
CAPTURED_CWD = "/private/tmp/ntcap-lDkd/ws"
CAPTURED_PAYLOAD = {
    "sessionId": "df8bda97-daf1-4810-b9aa-8f81b3bfd326",
    "timestamp": 1794837303265,
    "cwd": CAPTURED_CWD,
    "toolCallId": "custom_call_is0htq5DLxf8Px6AQsgX7ZgH",
    "toolName": "apply_patch",
    "toolArgs": (
        "*** Begin Patch\n*** Add File: notes/hello.txt\n+hi\n*** Delete File: README.md\n"
        "*** Add File: README.md\n+demo\n*** End Patch\n"
    ),
}
CAPTURED_SESSION = "df8bda97-daf1-48"


def _call(tool: str, args, key: str = "arguments") -> TraceEvent:
    return TraceEvent(
        event_type=EventType.TOOL_CALL, timestamp=2.0, data={"tool_name": tool, key: args},
    )


def _stored(tool: str, args) -> TraceEvent:
    """Return a tool call as watch reads it back from the events file."""
    return TraceEvent.from_json(_call(tool, args).to_json())


def _file(event_type: EventType, **data) -> TraceEvent:
    return TraceEvent(event_type=event_type, timestamp=2.0, data=data)


def _start(cwd=None, source: str = "new") -> TraceEvent:
    data = {"source": source}
    if cwd is not None:
        data["cwd"] = cwd
    return TraceEvent(event_type=EventType.SESSION_START, timestamp=1.0, data=data)


def _end() -> TraceEvent:
    return TraceEvent(event_type=EventType.SESSION_END, timestamp=3.0, data={"exit_code": 0})


def _rule(condition: str) -> NannyRule:
    return NannyRule(name="r", condition=condition, action="kill")


def _fired(condition: str) -> str:
    return f"NannyRule 'r': {condition} → kill"


class _WatchCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.policy = self.root / "agent-scope.json"
        self._stores = 0

    def new_store(self, attribution=None) -> TraceStore:
        self._stores += 1
        store = TraceStore(str(self.root / f"store{self._stores}"), redact=False)
        store.create_session(
            SessionMeta(session_id=SESSION, started_at=1.0, attribution=attribution or {}),
        )
        return store

    def config(self, **overrides) -> WatcherConfig:
        overrides.setdefault("scope_policy", str(self.policy))
        overrides.setdefault("alert_log", str(self.root / "alerts.log"))
        return WatcherConfig(**overrides)

    def deny(self, *patterns: str) -> None:
        self.policy.write_text(json.dumps({"files": {"write": {"deny": list(patterns)}}}))

    def check(self, *events: TraceEvent, config=None):
        """Run check_event over *events*; return the state and every violation."""
        state = WatchState()
        config = config or self.config()
        violations = []
        for event in events:
            violations.extend(check_event(event, config, state))
        return state, violations

    def watch(self, events, conditions=(), config=None, store=None) -> list[str]:
        """Run watch_session over *events* and a SESSION_END; return the alerts."""
        out = io.StringIO()
        with mock.patch(
            "agent_trace.watch._tail_events", return_value=iter([*events, _end()]),
        ), mock.patch("agent_trace.watch._dispatch_alert") as alert:
            watch_session(
                store or self.new_store(), SESSION, config or self.config(), out=out,
                nanny_rules=[_rule(c) for c in conditions],
            )
        self.assertIn("[watch] Session ended (", out.getvalue())
        return [entry.args[0] for entry in alert.call_args_list]

    def run_watch(self, store, conditions=(), during=None, session_id=SESSION):
        """Run watch_session on the real events file; return the alerts and output.

        Once watch reaches the end of the file, *during* writes events while
        it runs, then a SESSION_END ends the watch.
        """
        real_tail = watch_mod._tail_events

        def tail(path, poll_interval=0.5):
            inner = real_tail(path, poll_interval=0.01)
            idle = 0
            try:
                for item in inner:
                    if item is watch_mod._IDLE_SENTINEL:
                        idle += 1
                        if idle == 1:
                            if during:
                                during()
                            store.append_event(session_id, _end())
                        elif idle > 50:
                            raise AssertionError("watch never read the SESSION_END")
                    yield item
            finally:
                inner.close()

        out = io.StringIO()
        with mock.patch("agent_trace.watch._tail_events", side_effect=tail), mock.patch(
            "agent_trace.watch._dispatch_alert",
        ) as alert:
            watch_session(
                store, session_id, self.config(), out=out, poll_interval=0.01,
                nanny_rules=[_rule(c) for c in conditions],
            )
        self.assertIn("[watch] Session ended (", out.getvalue())
        return [entry.args[0] for entry in alert.call_args_list], out.getvalue()


class TestFilePathRule(_WatchCase):
    """file_path matches tests every path a tool call or file event names."""

    def test_claude_write_and_copilot_edit_still_fire(self):
        self.assertEqual(
            self.watch([_call("Write", {"file_path": "/app/production.env", "content": "x"})], [ENV]),
            [_fired(ENV)],
        )
        self.assertEqual(
            self.watch([_call("edit", {"path": "/app/production.env", "old_str": "a", "new_str": "b"})], [ENV]),
            [_fired(ENV)],
        )

    def test_every_path_argument_is_read(self):
        for key in ("file_path", "path", "absolute_path", "filePath", "target_file", "notebook_path", "uri"):
            with self.subTest(key=key):
                self.assertTrue(_rule(ENV).evaluate({}, _call("some_tool", {key: "/app/production.env"})))

    def test_notebook_edit(self):
        event = _call("NotebookEdit", {"notebook_path": "/app/secret.ipynb", "new_source": "x"})
        self.assertEqual(self.watch([event], [NOTEBOOK]), [_fired(NOTEBOOK)])

    def test_file_write_and_file_read_events(self):
        for event_type in (EventType.FILE_WRITE, EventType.FILE_READ):
            for key in ("path", "file_path", "uri"):
                with self.subTest(event_type=event_type.value, key=key):
                    event = _file(event_type, **{key: "/app/production.env"})
                    self.assertEqual(self.watch([event], [ENV]), [_fired(ENV)])

    def test_arguments_stored_as_text_or_under_tool_input(self):
        args = {"file_path": "/app/production.env", "content": "x"}
        cases = (
            ("JSON text", lambda: _call("Write", json.dumps(args))),
            ("Python literal", lambda: _call("Write", repr(args))),
            ("tool_input", lambda: _call("Write", args, key="tool_input")),
            ("tool_input JSON text", lambda: _call("Write", json.dumps(args), key="tool_input")),
        )
        for label, event in cases:
            with self.subTest(label):
                self.assertEqual(self.watch([event()], [ENV]), [_fired(ENV)])

    def test_json_text_over_the_python_literal_limit(self):
        from agent_trace.event_data import MAX_ARGUMENT_CHARS

        text = json.dumps({"file_path": "/app/production.env", "content": "x" * MAX_ARGUMENT_CHARS})
        self.assertEqual(self.watch([_call("Write", text)], [ENV]), [_fired(ENV)])
        state, _ = self.check(_call("Write", text))
        self.assertEqual(state.files_modified_set, {"/app/production.env"})

    def test_apply_patch_add_move_and_delete(self):
        for label, text in (("Add File", ADD_ENV), ("Move to", MOVE_ENV), ("Delete File", DELETE_ENV)):
            with self.subTest(label):
                self.assertEqual(self.watch([_call("apply_patch", text)], [ENV]), [_fired(ENV)])

    def test_the_file_a_patch_moves_from_counts_too(self):
        old = 'file_path matches "**/old.env"'
        self.assertEqual(self.watch([_call("apply_patch", MOVE_ENV)], [old]), [_fired(old)])

    def test_crlf_patch_and_tool_name_case(self):
        crlf = _patch("*** Add File: config/production.env", "+X=1", newline="\r\n")
        for tool, text in (("apply_patch", crlf), ("Apply_Patch", ADD_ENV)):
            with self.subTest(tool=tool, crlf="\r" in text):
                self.assertEqual(self.watch([_call(tool, text)], [ENV]), [_fired(ENV)])

    def test_patch_content_lines_are_not_read_as_paths(self):
        text = _patch(
            "*** Update File: notes.md",
            "@@ *** Move to: config/production.env",
            "-*** Add File: config/production.env",
            "+*** Delete File: config/production.env",
            " *** Update File: config/production.env",
        )
        self.assertEqual(self.watch([_call("apply_patch", text)], [ENV]), [])

    def test_patch_text_is_read_only_for_apply_patch(self):
        cases = (
            ("Write with patch text", lambda: _call("Write", ADD_ENV)),
            ("bash with patch text", lambda: _call("bash", ADD_ENV)),
            ("patch under another key", lambda: _call("apply_patch", {"input": ADD_ENV})),
        )
        for label, event in cases:
            with self.subTest(label):
                self.assertEqual(self.watch([event()], [ENV]), [])

    def test_relative_patch_path_resolves_against_the_session_cwd(self):
        update = _call("apply_patch", UPDATE_SECRET)
        self.assertEqual(self.watch([_start("/app"), update], [SECRETS]), [_fired(SECRETS)])

    def test_relative_patch_path_without_a_session_cwd_does_not_fire(self):
        self.assertEqual(self.watch([_call("apply_patch", UPDATE_SECRET)], [SECRETS]), [])

    def test_resume_moves_the_cwd(self):
        to_app = [_start("/old"), _start("/app", "resume"), _call("apply_patch", UPDATE_SECRET)]
        self.assertEqual(self.watch(to_app, [SECRETS]), [_fired(SECRETS)])
        away = [_start("/app"), _start("/other", "resume"), _call("apply_patch", UPDATE_SECRET)]
        self.assertEqual(self.watch(away, [SECRETS]), [])

    def test_a_start_without_a_usable_cwd_keeps_the_last_one(self):
        for cwd in ("app", "", None, ["/app"], 42):
            with self.subTest(cwd=cwd):
                events = [_start("/app"), _start(cwd, "resume"), _call("apply_patch", UPDATE_SECRET)]
                self.assertEqual(self.watch(events, [SECRETS]), [_fired(SECRETS)])

    def test_dot_segments_collapse(self):
        cases = (
            ("relative with ..", "/app/src", _call("apply_patch", _patch(
                "*** Update File: ../secrets/key.pem", "@@", "-a", "+b",
            ))),
            ("relative with .", "/app", _call("apply_patch", _patch(
                "*** Update File: ./secrets/./key.pem", "@@", "-a", "+b",
            ))),
            ("absolute with ..", None, _call("Write", {"file_path": "/app/src/../secrets/key.pem"})),
        )
        for label, cwd, event in cases:
            with self.subTest(label):
                events = [_start(cwd), event] if cwd else [event]
                self.assertEqual(self.watch(events, [SECRETS]), [_fired(SECRETS)])

    def test_without_a_session_cwd_paths_match_as_given(self):
        # Neither watch's own directory nor the session's recorded
        # working_dir is used as a base for relative paths.
        app = Path(os.path.realpath(self.root)) / "app"
        app.mkdir()
        previous = os.getcwd()
        os.chdir(app)
        self.addCleanup(os.chdir, previous)
        store = self.new_store(attribution={"working_dir": str(app)})
        absolute = f'file_path matches "{app}/secrets/*"'
        as_given = 'file_path matches "secrets/*"'
        alerts = self.watch([_call("apply_patch", UPDATE_SECRET)], [absolute, as_given], store=store)
        self.assertEqual(alerts, [_fired(as_given)])

    def test_windows_cwd_and_paths(self):
        rule = 'file_path matches "C:/repo/config/*"'
        cases = (
            ("relative path", [_start("C:\\repo"), _call("edit", {"path": "config\\production.env"})]),
            ("patch path", [_start("C:\\repo"), _call("apply_patch", ADD_ENV)]),
            ("absolute path", [_call("edit", {"path": "C:\\repo\\config\\production.env"})]),
        )
        for label, events in cases:
            with self.subTest(label):
                self.assertEqual(self.watch(events, [rule]), [_fired(rule)])

    def test_uri_is_matched_as_stored(self):
        stored = 'file_path matches "file:///app/secrets/*"'
        event = _file(EventType.FILE_READ, uri="file:///app/secrets/key.pem")
        self.assertEqual(self.watch([_start("/app"), event], [stored, SECRETS]), [_fired(stored)])

    def test_evaluate_keeps_its_two_argument_form(self):
        rule = _rule(SECRETS)
        update = _call("apply_patch", UPDATE_SECRET)
        self.assertFalse(rule.evaluate({}, update))
        self.assertTrue(rule.evaluate({}, update, cwd="/app"))
        self.assertFalse(rule.evaluate({}))
        self.assertFalse(rule.evaluate({}, _start("/app")))


class TestFilesModified(_WatchCase):
    """files_modified counts the distinct resolved paths of edits."""

    def test_every_edit_counts(self):
        cases = (
            ("Write", {"file_path": "/app/a.py", "content": "x"}),
            ("Edit", {"file_path": "/app/a.py", "old_string": "a", "new_string": "b"}),
            ("MultiEdit", {"file_path": "/app/a.py", "edits": []}),
            ("NotebookEdit", {"notebook_path": "/app/a.ipynb", "new_source": "x"}),
            ("write_file", {"file_path": "/app/a.py", "content": "x"}),
            ("replace", {"file_path": "/app/a.py", "old_string": "a", "new_string": "b"}),
            ("edit", {"path": "/app/a.py", "old_str": "a", "new_str": "b"}),
            ("create", {"path": "/app/a.py", "file_text": "x"}),
            ("str_replace", {"path": "/app/a.py", "old_str": "a", "new_str": "b"}),
            ("str_replace_editor", {"command": "str_replace", "path": "/app/a.py", "old_str": "a", "new_str": "b"}),
            ("str_replace_based_edit_tool", {"command": "create", "path": "/app/a.py", "file_text": "x"}),
            ("insert", {"path": "/app/a.py", "insert_line": 1, "new_str": "x"}),
            ("edit_file", {"target_file": "/app/a.py", "code_edit": "x"}),
            ("search_replace", {"file_path": "/app/a.py", "old_string": "a", "new_string": "b"}),
            ("write_to_file", {"path": "/app/a.py", "content": "x"}),
            ("file_edit", {"filePath": "/app/a.py", "content": "x"}),
            ("delete_file", {"target_file": "/app/a.py"}),
            ("patch", {"path": "/app/a.py", "diff": "x"}),
            ("apply_patch", _patch("*** Update File: /app/a.py", "@@", "-a", "+b")),
        )
        for tool, args in cases:
            with self.subTest(tool=tool):
                state, _ = self.check(_call(tool, args))
                self.assertEqual(state.files_modified, 1)
        state, _ = self.check(_file(EventType.FILE_WRITE, path="/app/a.py"))
        self.assertEqual(state.files_modified, 1)

    def test_edit_tools_trigger_files_modified_rules(self):
        cases = (
            ("MultiEdit", lambda: _call("MultiEdit", {"file_path": "/app/a.py", "edits": []})),
            ("write_file", lambda: _call("write_file", {"file_path": "/app/a.py", "content": "x"})),
            ("replace", lambda: _call("replace", {"file_path": "/app/a.py", "old_string": "a", "new_string": "b"})),
            ("apply_patch", lambda: _call("apply_patch", ADD_ENV)),
        )
        for label, event in cases:
            with self.subTest(label):
                self.assertEqual(
                    self.watch([event()], ["files_modified > 0"]), [_fired("files_modified > 0")],
                )

    def test_captured_copilot_patch(self):
        text = CAPTURED_PAYLOAD["toolArgs"]
        state, _ = self.check(_start(CAPTURED_CWD), _call("apply_patch", text))
        self.assertEqual(
            state.files_modified_set,
            {f"{CAPTURED_CWD}/notes/hello.txt", f"{CAPTURED_CWD}/README.md"},
        )
        state, _ = self.check(_call("apply_patch", text))
        self.assertEqual(state.files_modified_set, {"notes/hello.txt", "README.md"})

    def test_one_file_named_two_ways_counts_once(self):
        for name in ("a.py", "./a.py", "sub/../a.py"):
            with self.subTest(name=name):
                patch_a = _patch(f"*** Update File: {name}", "@@", "-a", "+b")
                state, _ = self.check(_start("/app"), _call("edit", EDIT_A), _call("apply_patch", patch_a))
                self.assertEqual(state.files_modified, 1)
        events = [_start("/app"), _call("edit", EDIT_A), _call("apply_patch", UPDATE_A)]
        self.assertEqual(self.watch(events, ["files_modified > 1"]), [])

    def test_two_files_fire_files_modified_above_one(self):
        events = [_start("/app"), _call("edit", EDIT_A), _call("apply_patch", UPDATE_B)]
        self.assertEqual(self.watch(events, ["files_modified > 1"]), [_fired("files_modified > 1")])

    def test_reads_and_shell_writes_are_not_edits(self):
        cases = (
            ("editor view", lambda: _call("str_replace_editor", {"command": "view", "path": "/app/a.py"})),
            ("editor view as JSON text", lambda: _call(
                "str_replace_editor", json.dumps({"command": "view", "path": "/app/a.py"}),
            )),
            ("based edit tool view", lambda: _call(
                "str_replace_based_edit_tool", {"command": "view", "path": "/app/a.py"},
            )),
            ("Read", lambda: _call("Read", {"file_path": "/app/a.py"})),
            ("view", lambda: _call("view", {"path": "/app/a.py"})),
            ("FILE_READ", lambda: _file(EventType.FILE_READ, path="/app/a.py")),
            ("shell write", lambda: _call("bash", {"command": "echo x > /app/a.py"})),
        )
        for label, event in cases:
            with self.subTest(label):
                state, _ = self.check(event())
                self.assertEqual(state.files_modified, 0)
        view = _call("str_replace_editor", {"command": "view", "path": "/app/a.py"})
        self.assertEqual(self.watch([view], ["files_modified > 0"]), [])

    def test_cognitive_debt_counts_apply_patch(self):
        config = self.config(built_in_rules={"cognitive-debt"})
        _, violations = self.check(_call("apply_patch", ADD_ENV), config=config)
        self.assertEqual(violations, [
            "CognitiveDebtWatcher: live debt score 1.00 "
            "(1 modified files, no review yet; threshold: 0.8)",
        ])


class TestPolicyWriteDeny(_WatchCase):
    """--policy files.write.deny checks every path of every edit."""

    def setUp(self):
        super().setUp()
        self.deny("**/production.env")

    def scope(self, *events: TraceEvent) -> list[str]:
        return self.check(*events)[1]

    def test_copilot_edit_message_is_unchanged(self):
        event = _call("edit", {"path": "/app/config/production.env", "old_str": "a", "new_str": "b"})
        self.assertEqual(self.scope(event), [ENV_DENIED])

    def test_apply_patch(self):
        self.assertEqual(self.scope(_start("/app"), _call("apply_patch", ADD_ENV)), [ENV_DENIED])
        self.assertEqual(
            self.scope(_call("apply_patch", ADD_ENV)),
            ["ScopeWatcher: write to config/production.env denied by policy"],
        )

    def test_every_edit_is_checked(self):
        target = "/app/config/production.env"
        cases = (
            ("write_file", lambda: _call("write_file", {"file_path": target, "content": "x"})),
            ("FILE_WRITE", lambda: _file(EventType.FILE_WRITE, path=target)),
            ("MultiEdit", lambda: _call("MultiEdit", {"file_path": target, "edits": []})),
            ("Write as JSON text", lambda: _call("Write", json.dumps({"file_path": target, "content": "x"}))),
            ("str_replace", lambda: _call("str_replace", {"path": target, "old_str": "a", "new_str": "b"})),
            ("patch Move to", lambda: _call("apply_patch", MOVE_ENV)),
            ("patch Delete File", lambda: _call("apply_patch", DELETE_ENV)),
        )
        for label, event in cases:
            with self.subTest(label):
                self.assertEqual(self.scope(_start("/app"), event()), [ENV_DENIED])

    def test_the_file_a_patch_moves_from_is_checked(self):
        self.deny("**/old.env")
        self.assertEqual(
            self.scope(_start("/app"), _call("apply_patch", MOVE_ENV)),
            ["ScopeWatcher: write to /app/src/old.env denied by policy"],
        )

    def test_relative_deny_pattern_matches_the_path_as_given(self):
        self.deny("config/production.env")
        self.assertEqual(self.scope(_start("/app"), _call("apply_patch", ADD_ENV)), [ENV_DENIED])

    def test_reads_are_not_checked(self):
        target = "/app/config/production.env"
        cases = (
            ("FILE_READ", lambda: _file(EventType.FILE_READ, path=target)),
            ("view", lambda: _call("view", {"path": target})),
            ("editor view", lambda: _call("str_replace_editor", {"command": "view", "path": target})),
            ("Read", lambda: _call("Read", {"file_path": target})),
        )
        for label, event in cases:
            with self.subTest(label):
                self.assertEqual(self.scope(event()), [])

    def test_arguments_that_are_not_objects_never_raise(self):
        for tool in ("Write", "apply_patch"):
            for value in ("plain text", 42, None, ["/app/config/production.env"], True):
                with self.subTest(tool=tool, value=value):
                    self.assertEqual(self.scope(_call(tool, value)), [])

    def test_one_alert_per_denied_file(self):
        self.deny("**/production.env", "**/*.pem")
        both = _patch(
            "*** Add File: config/production.env", "+X=1",
            "*** Update File: secrets/key.pem", "@@", "-a", "+b",
        )
        again = _call("edit", {"path": "/app/config/production.env", "old_str": "a", "new_str": "b"})
        self.assertEqual(
            self.scope(_start("/app"), _call("apply_patch", both), again),
            [ENV_DENIED, "ScopeWatcher: write to /app/secrets/key.pem denied by policy"],
        )


class TestLoopKey(_WatchCase):
    """Different calls get different loop keys; identical calls share one."""

    def test_different_patches_are_not_a_loop(self):
        config = self.config(on_violation="kill")
        new_files = [_call("apply_patch", _patch(f"*** Add File: notes/n{i}.txt", f"+{i}")) for i in range(3)]
        self.assertEqual(self.watch(new_files, config=config), [])
        hunks = [
            _call("apply_patch", _patch("*** Update File: src/app.py", "@@", f"-{i}", f"+{i + 1}"))
            for i in range(3)
        ]
        self.assertEqual(self.watch(hunks, config=config), [])

    def test_identical_patches_are_a_loop(self):
        alerts = self.watch(
            [_call("apply_patch", ADD_ENV) for _ in range(3)], config=self.config(on_violation="kill"),
        )
        self.assertEqual(len(alerts), 1, alerts)
        self.assertTrue(alerts[0].startswith(LOOP), alerts)

    def test_object_arguments_keep_their_key(self):
        args = {"command": "ls", "timeout": 5}
        digest = hashlib.sha256(json.dumps(args, sort_keys=True).encode("utf-8")).hexdigest()[:12]
        self.assertEqual(_event_key(_call("bash", args)), f"bash:ls:{digest}")

    def test_patch_text_is_hashed_as_text(self):
        empty = hashlib.sha256(b"{}").hexdigest()[:12]
        key = _event_key(_call("apply_patch", UPDATE_A))
        self.assertEqual(key, _event_key(_call("apply_patch", UPDATE_A)))
        self.assertNotEqual(key, _event_key(_call("apply_patch", UPDATE_B)))
        self.assertFalse(key.endswith(empty), key)
        self.assertTrue(key.startswith("apply_patch:a.py:"), key)
        hunks = {
            _event_key(_call("apply_patch", _patch("*** Update File: src/app.py", "@@", f"-{i}", f"+{i + 1}")))
            for i in range(3)
        }
        self.assertEqual(len(hunks), 3)

    def test_other_values_are_hashed_as_stored(self):
        empty = hashlib.sha256(b"{}").hexdigest()[:12]
        keys = set()
        for value in ("text", 42, None, [1], True):
            with self.subTest(value=value):
                key = _event_key(_call("some_tool", value))
                self.assertFalse(key.endswith(empty), key)
                keys.add(key)
        self.assertEqual(len(keys), 5)

    def test_tool_input_is_hashed(self):
        first = _event_key(_call("Write", {"file_path": "/app/a.py"}, key="tool_input"))
        second = _event_key(_call("Write", {"file_path": "/app/b.py"}, key="tool_input"))
        self.assertNotEqual(first, second)

    # JSON allows "\ud800", and TraceEvent.from_json decodes it to a lone
    # surrogate, which strict UTF-8 cannot encode.
    def test_a_lone_surrogate_in_a_patch(self):
        event = _stored("apply_patch", _patch("*** Add File: notes/a\ud800.txt", "+x"))
        self.assertIn("\ud800", event.data["arguments"])
        self.assertEqual(self.check(event)[1], [])
        write = _call("Write", {"file_path": "/app/production.env", "content": "x"})
        self.assertEqual(self.watch([event, write], [ENV]), [_fired(ENV)])

    def test_identical_patches_with_a_lone_surrogate_are_a_loop(self):
        calls = [_stored("apply_patch", _patch("*** Add File: notes/a\ud800.txt", "+x")) for _ in range(3)]
        violations = self.check(*calls)[1]
        self.assertEqual(len(violations), 1, violations)
        self.assertTrue(violations[0].startswith(LOOP), violations)

    def test_patches_that_differ_only_by_a_lone_surrogate_are_not_a_loop(self):
        calls = [
            _stored("apply_patch", _patch("*** Add File: notes/a.txt", body))
            for body in ("+x\ud800", "+x\ud801", "+x\ud802")
        ]
        keys = {_event_key(call) for call in calls}
        self.assertEqual(len(keys), 3, keys)
        self.assertTrue(all(key.startswith("apply_patch:notes/a.txt:") for key in keys), keys)
        self.assertEqual(self.check(*calls)[1], [])

    def test_preview_reads_arguments_stored_as_text(self):
        key = _event_key(_call("bash", json.dumps({"command": "npm test"})))
        self.assertTrue(key.startswith("bash:npm test:"), key)

    def test_retry_watcher_reads_arguments_stored_as_text(self):
        config = self.config(max_retries=2, loop_threshold=99)
        calls = [_call("bash", json.dumps({"command": "npm test"})) for _ in range(3)]
        self.assertEqual(
            self.check(*calls, config=config)[1], ["RetryWatcher: command ran 3 times: npm test"],
        )

    def test_operation_watcher_reads_arguments_stored_as_text(self):
        config = self.config(operation_rules=[OperationRule("write", "/etc/*", "block", "system files")])
        call = _call("write", json.dumps({"file_path": "/etc/hosts", "content": "x"}))
        self.assertEqual(
            self.check(call, config=config)[1], ["OperationWatcher: block write '/etc/hosts': system files"],
        )


class TestStartupCwd(_WatchCase):
    """Relative paths resolve against a cwd written before watch started."""

    def events_file(self, store) -> Path:
        return store._session_file(SESSION, "events.ndjson", "event stream")

    def test_reads_the_latest_absolute_cwd(self):
        from agent_trace.watch import _startup_session_cwd

        path = self.root / "events.ndjson"
        lines = [
            _start("/old").to_json(),
            _start("/app", "resume").to_json(),
            _call("session_start", {"cwd": "/tool"}).to_json(),
            _start("relative", "resume").to_json(),
            _start("", "resume").to_json(),
            _start(None, "resume").to_json(),
            _start(["/list"], "resume").to_json(),
            json.dumps({"event_type": "session_start", "timestamp": 1.0, "data": "/text"}),
            '{"event_type":"session_start","data":{"cwd":"/truncated',
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.assertEqual(_startup_session_cwd(path, path.stat().st_size), "/app")

    def test_reads_only_up_to_the_offset(self):
        from agent_trace.watch import _startup_session_cwd

        path = self.root / "events.ndjson"
        first = (_start("/old").to_json() + "\n").encode("utf-8")
        second = (_start("/app", "resume").to_json() + "\n").encode("utf-8")
        path.write_bytes(first + second)
        self.assertEqual(_startup_session_cwd(path, 0), "")
        self.assertEqual(_startup_session_cwd(path, len(first)), "/old")
        self.assertEqual(_startup_session_cwd(path, len(first) + 10), "/old")
        self.assertEqual(_startup_session_cwd(path, len(first) + len(second)), "/app")
        self.assertEqual(_startup_session_cwd(self.root / "missing.ndjson", 10), "")

    def test_an_event_written_during_the_startup_read_is_tailed(self):
        from agent_trace.watch import _StartupCwd

        store = self.new_store()
        store.append_event(SESSION, _start("/old"))
        real_read = watch_mod._startup_session_cwd

        def append_then_read(path, end):
            store.append_event(SESSION, _start("/new", "resume"))
            return real_read(path, end)

        with mock.patch("agent_trace.watch._startup_session_cwd", side_effect=append_then_read):
            tail = watch_mod._tail_events(self.events_file(store), poll_interval=0.01)
            try:
                first, second = next(tail), next(tail)
            finally:
                tail.close()
        self.assertIsInstance(first, _StartupCwd)
        self.assertEqual(first.cwd, "/old")
        self.assertEqual(second.event_type, EventType.SESSION_START)
        self.assertEqual(second.data["cwd"], "/new")

    def test_cwd_from_a_start_written_before_watch(self):
        store = self.new_store()
        store.append_event(SESSION, _start("/app"))
        alerts, _ = self.run_watch(
            store, [SECRETS],
            during=lambda: store.append_event(SESSION, _call("apply_patch", UPDATE_SECRET)),
        )
        self.assertEqual(alerts, [_fired(SECRETS)])

    def test_the_latest_start_before_watch_wins(self):
        store = self.new_store()
        store.append_event(SESSION, _start("/old"))
        store.append_event(SESSION, _start("/app", "resume"))
        alerts, _ = self.run_watch(
            store, [SECRETS],
            during=lambda: store.append_event(SESSION, _call("apply_patch", UPDATE_SECRET)),
        )
        self.assertEqual(alerts, [_fired(SECRETS)])

    def test_a_start_tailed_after_watch_replaces_the_startup_cwd(self):
        for before, after, fires in (("/old", "/app", True), ("/app", "/other", False)):
            with self.subTest(before=before, after=after):
                store = self.new_store()
                store.append_event(SESSION, _start(before))

                def during(store=store, after=after):
                    store.append_event(SESSION, _start(after, "resume"))
                    store.append_event(SESSION, _call("apply_patch", UPDATE_SECRET))

                alerts, _ = self.run_watch(store, [SECRETS], during=during)
                self.assertEqual(alerts, [_fired(SECRETS)] if fires else [])

    def test_events_written_before_watch_are_not_checked(self):
        store = self.new_store()
        store.append_event(SESSION, _start("/app"))
        store.append_event(SESSION, _call("apply_patch", UPDATE_SECRET))
        alerts, out = self.run_watch(store, [SECRETS, "files_modified > 0"])
        self.assertEqual(alerts, [])
        self.assertIn("[watch] Session ended (1 events,", out)


class TestCapturedCopilotHooks(_WatchCase):
    """watch reads the apply_patch calls the real Copilot hooks store."""

    def setUp(self):
        super().setUp()
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        self.store_dir = self.root / "hooks"
        os.environ["AGENT_TRACE_DIR"] = str(self.store_dir)
        os.environ["DO_NOT_TRACK"] = "1"
        for name in ("AGENT_TRACE_COPILOT_SESSION_ID", "AGENT_TRACE_REDACT", "AGENT_STRACE_ENDPOINT"):
            os.environ.pop(name, None)

    def hook(self, kind: str, payload: dict) -> None:
        with mock.patch.object(sys, "stdin", io.StringIO(json.dumps(payload))), redirect_stdout(io.StringIO()):
            hook_main(["--provider", "copilot", kind])

    def session_start(self, cwd: str, source: str) -> None:
        self.hook("session-start", {"sessionId": CAPTURED_PAYLOAD["sessionId"], "cwd": cwd, "source": source})

    def test_patch_from_the_pre_tool_hook(self):
        self.session_start(CAPTURED_CWD, "new")
        store = TraceStore(str(self.store_dir), redact=False)
        conditions = [f'file_path matches "{CAPTURED_CWD}/README.md"', "files_modified > 1"]
        alerts, _ = self.run_watch(
            store, conditions, during=lambda: self.hook("pre-tool", CAPTURED_PAYLOAD),
            session_id=CAPTURED_SESSION,
        )
        self.assertEqual(alerts, [_fired(c) for c in conditions])
        calls = [e for e in store.load_events(CAPTURED_SESSION) if e.event_type == EventType.TOOL_CALL]
        self.assertEqual(calls[0].data["arguments"], CAPTURED_PAYLOAD["toolArgs"])

    def test_resume_hook_moves_the_cwd(self):
        self.session_start("/old", "new")
        store = TraceStore(str(self.store_dir), redact=False)

        def during():
            self.session_start(CAPTURED_CWD, "resume")
            self.hook("pre-tool", CAPTURED_PAYLOAD)

        condition = f'file_path matches "{CAPTURED_CWD}/notes/*"'
        alerts, _ = self.run_watch(store, [condition], during=during, session_id=CAPTURED_SESSION)
        self.assertEqual(alerts, [_fired(condition)])


if __name__ == "__main__":
    unittest.main()
