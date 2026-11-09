"""Tests for the ``agent-strace`` console entry and its hook fast path.

``agent_trace.entry:main`` parses plain ``hook`` command lines itself and
sends everything else to ``agent_trace.cli:main``.  These tests check that the
fast path accepts only command lines the full parser reads the same way, that
hook processes behave identically through either entry point, and that the
fast path never imports the full CLI.
"""

import argparse
import contextlib
import io
import itertools
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
sys.path.insert(0, str(_SRC))

from agent_trace import cli, copilot_cloud, entry, hooks, skills_eval, telemetry


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _full_parse(argv):
    """Run ``cli.main`` on ``argv`` and return (hook args, exit code, output)."""
    calls = []
    out, err = io.StringIO(), io.StringIO()
    code = None
    with patch.object(sys, "argv", ["agent-strace", *argv]), \
            patch.object(cli, "hook_main", side_effect=calls.append), \
            patch.object(cli, "build_parser", _cached_parser), \
            contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            cli.main()
        except SystemExit as exc:
            code = exc.code
    return (calls[0] if calls else None), code, out.getvalue() + err.getvalue()


_PARSER = []
_BUILD_PARSER = cli.build_parser


def _cached_parser():
    # Building the full parser takes tens of milliseconds; reuse one.
    if not _PARSER:
        _PARSER.append(_BUILD_PARSER())
    return _PARSER[0]


def _hook_subparser():
    parser = cli.build_parser()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices["hook"]
    raise AssertionError("no hook subcommand")


class TestParseHookArgv(unittest.TestCase):
    def test_providers_match_hooks_and_cli(self):
        self.assertEqual(entry.HOOK_PROVIDERS, tuple(hooks._PROVIDER_AGENT))
        provider = next(a for a in _hook_subparser()._actions if a.dest == "provider")
        self.assertEqual(list(provider.choices), list(entry.HOOK_PROVIDERS))
        self.assertEqual(provider.default, "claude")
        self.assertEqual(entry._METADATA_ONLY_FLAG, hooks.METADATA_ONLY_FLAG)

    def test_fast_path_knows_every_hook_option(self):
        # A new `hook` option must be added to entry.parse_hook_argv too, or
        # hooks that use it silently lose the fast path.
        options = {opt for action in _hook_subparser()._actions for opt in action.option_strings}
        self.assertEqual(options, {"-h", "--help", "--provider", hooks.METADATA_ONLY_FLAG})

    def test_accepted_forms(self):
        cases = {
            ("hook",): ["--provider", "claude"],
            ("hook", "pre-tool"): ["--provider", "claude", "pre-tool"],
            ("hook", "--provider", "codex", "PostModelRequest"): ["--provider", "codex", "PostModelRequest"],
            ("hook", "--provider=copilot", "userPromptSubmitted"): ["--provider", "copilot", "userPromptSubmitted"],
            ("hook", "pre-tool", "--provider", "gemini"): ["--provider", "gemini", "pre-tool"],
            ("hook", "--provider", "codex", "--provider", "cursor", "stop"): ["--provider", "cursor", "stop"],
            ("hook", "--provider", "copilot"): ["--provider", "copilot"],
            ("hook", "--metadata-only", "pre-tool"): ["--provider", "claude", "--metadata-only", "pre-tool"],
            ("hook", "--provider", "copilot", "--metadata-only", "preToolUse"):
                ["--provider", "copilot", "--metadata-only", "preToolUse"],
            ("hook", "stop", "--metadata-only", "--metadata-only"): ["--provider", "claude", "--metadata-only", "stop"],
        }
        for argv, expected in cases.items():
            with self.subTest(argv=argv):
                self.assertEqual(entry.parse_hook_argv(list(argv)), expected)

    def test_other_command_lines_use_the_full_parser(self):
        cases = [
            [],
            ["list"],
            ["--trace-dir", "x", "hook", "pre-tool"],
            ["--version"],
            ["hooks", "pre-tool"],
            ["hook", "-h"],
            ["hook", "--help"],
            ["hook", "--prov", "codex", "pre-tool"],
            ["hook", "--provider"],
            ["hook", "--provider", "bogus", "pre-tool"],
            ["hook", "--provider="],
            ["hook", "--", "pre-tool"],
            ["hook", "-"],
            ["hook", ""],
            ["hook", "-1"],
            ["hook", "pre-tool", "extra"],
            ["hook", "pre-tool", "--trace-dir", "x"],
            ["hook", "--metadata", "pre-tool"],
            ["hook", "--metadata-only=1", "pre-tool"],
            ["hook", "--provider", "--metadata-only", "pre-tool"],
        ]
        for argv in cases:
            with self.subTest(argv=argv):
                self.assertIsNone(entry.parse_hook_argv(argv))

    def test_every_accepted_line_matches_the_full_parser(self):
        vocabulary = [
            "--provider", "--provider=codex", "--provider=copilot", "--provider=", "--provider=bogus",
            "copilot", "claude", "gemini", "bogus", "hook", "pre-tool", "PostModelRequest", "session-start",
            "-h", "--help", "--prov", "--prov=gemini", "--", "-", "", "-1", "--x", "a b", "-p",
            "--metadata-only", "--metadata", "--metadata-only=1",
        ]
        accepted = 0
        for length in range(4):
            for tail in itertools.product(vocabulary, repeat=length):
                argv = ["hook", *tail]
                fast = entry.parse_hook_argv(argv)
                if fast is None:
                    continue
                accepted += 1
                with self.subTest(argv=argv):
                    self.assertEqual(_full_parse(argv), (fast, 0, ""))
        self.assertGreater(accepted, 200)


def _hook_commands(node):
    if isinstance(node, dict):
        for value in node.values():
            yield from _hook_commands(value)
    elif isinstance(node, list):
        for value in node:
            yield from _hook_commands(value)
    elif isinstance(node, str) and "agent-strace hook" in node:
        yield node


def _exec_hook_args(node):
    """Yield ``args`` lists of exec-form hook handlers."""
    if isinstance(node, dict):
        if isinstance(node.get("args"), list) and node["args"][:1] == ["hook"]:
            yield node["args"]
        for value in node.values():
            yield from _exec_hook_args(value)
    elif isinstance(node, list):
        for value in node:
            yield from _exec_hook_args(value)


def _hook_argv(command):
    tokens = shlex.split(command)
    while tokens and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", tokens[0]):
        tokens.pop(0)
    if tokens[:1] == ["agent-strace"]:
        return tokens[1:]
    index = tokens.index("-m")
    return tokens[index + 2:]


class TestGeneratedHookCommands(unittest.TestCase):
    """Every hook command agent-strace writes must take the fast path."""

    def assert_fast_path(self, command):
        argv = _hook_argv(command) if isinstance(command, str) else command
        fast = entry.parse_hook_argv(argv)
        self.assertIsNotNone(fast, command)
        self.assertEqual(_full_parse(argv), (fast, 0, ""), command)

    def test_setup_configs(self):
        generators = [
            cli._claude_hooks_config,
            cli._codex_hooks_config,
            cli._copilot_hooks_config,
            cli._gemini_hooks_config,
            cli._cursor_hooks_config,
        ]
        modes = [
            {"redact": False, "no_redact": False},
            {"redact": True, "no_redact": False},
            {"redact": False, "no_redact": True},
        ]
        for generator, mode in itertools.product(generators, modes):
            commands = list(_hook_commands(generator(argparse.Namespace(**mode))))
            self.assertTrue(commands, generator.__name__)
            for command in commands:
                with self.subTest(generator=generator.__name__, command=command):
                    self.assert_fast_path(command)

    def test_managed_configs(self):
        for name in ("copilot", "claude"):
            args = list(_exec_hook_args(cli._managed_hooks_config(name, "/opt/agent-strace/bin/agent-strace")))
            self.assertTrue(args, name)
            for argv in args:
                with self.subTest(cli=name, argv=argv):
                    self.assertIn(hooks.METADATA_ONLY_FLAG, argv)
                    self.assert_fast_path(argv)

    def test_plugin_hook_files(self):
        for name in (
            "hooks.json",
            "hooks/hooks.json",
            "plugins/agent-strace/com.github.copilot/hooks/hooks.json",
        ):
            commands = list(_hook_commands(json.loads((_ROOT / name).read_text())))
            self.assertTrue(commands, name)
            for command in commands:
                with self.subTest(file=name, command=command):
                    self.assert_fast_path(command)

    def test_skills_eval_hooks_use_the_entry_module(self):
        prefix = skills_eval.hook_command_prefix()
        self.assertIn("-m agent_trace.entry hook", prefix)
        self.assert_fast_path(f"{prefix} --provider copilot pre-tool")


@unittest.skipUnless(shutil.which("bash"), "needs bash")
class TestCopilotCloudHookScript(unittest.TestCase):
    """The cloud hook script uses the entry module when the install has one."""

    def run_script(self, modules):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "pkg" / "agent_trace"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("")
            for name in modules:
                (package / f"{name}.py").write_text(f"import sys\nprint({name!r}, sys.argv[1:])\n")
            bin_dir = root / "bin"
            bin_dir.mkdir()
            python3 = bin_dir / "python3"
            python3.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
            python3.chmod(0o755)
            script = root / "agent-strace-cloud.sh"
            script.write_text(copilot_cloud.hook_script())
            proc = subprocess.run(
                [shutil.which("bash"), str(script), "pre-tool"],
                input="{}",
                capture_output=True,
                text=True,
                env={"PATH": str(bin_dir), "AGENT_STRACE_INSTALL_DIR": str(root / "pkg"), "AGENT_STRACE_INSTALL": "0"},
                timeout=60,
            )
            self.assertEqual((proc.returncode, proc.stderr), (0, ""))
            return proc.stdout

    def test_prefers_the_entry_module(self):
        self.assertEqual(self.run_script(("cli", "entry")), "entry ['hook', '--provider', 'copilot', 'pre-tool']\n")

    def test_falls_back_to_the_cli_module(self):
        self.assertEqual(self.run_script(("cli",)), "cli ['hook', '--provider', 'copilot', 'pre-tool']\n")


# ---------------------------------------------------------------------------
# In-process dispatch
# ---------------------------------------------------------------------------

class TestEntryMain(unittest.TestCase):
    def run_main(self, main, argv):
        out, err = io.StringIO(), io.StringIO()
        code = "returned"
        with patch.object(sys, "argv", ["agent-strace", *argv]), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                main()
            except SystemExit as exc:
                code = exc.code
        return code, out.getvalue(), err.getvalue()

    def test_hook_lines_go_to_hook_main(self):
        with patch("agent_trace.hooks.hook_main") as hook_main, patch("agent_trace.cli.main") as cli_main:
            code, _, _ = self.run_main(entry.main, ["hook", "--provider", "codex", "PostModelRequest"])
        self.assertEqual(code, 0)
        hook_main.assert_called_once_with(["--provider", "codex", "PostModelRequest"])
        cli_main.assert_not_called()

    def test_other_lines_go_to_cli_main(self):
        for argv in (["list"], ["hook", "-h"], ["--trace-dir", "x", "hook", "pre-tool"]):
            with self.subTest(argv=argv), patch("agent_trace.hooks.hook_main") as hook_main, \
                    patch("agent_trace.cli.main", return_value=None) as cli_main:
                code, _, _ = self.run_main(entry.main, argv)
            self.assertEqual(code, "returned")
            cli_main.assert_called_once_with()
            hook_main.assert_not_called()

    def test_telemetry_notice_is_never_shown_for_hooks(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(os.environ, {"AGENT_TRACE_DIR": tmp}), \
                patch("agent_trace.telemetry.maybe_show_telemetry_notice") as notice, \
                patch("agent_trace.telemetry.capture") as capture, \
                patch("agent_trace.hooks.hook_main"):
            for main in (entry.main, cli.main):
                self.assertEqual(self.run_main(main, ["hook", "--provider", "copilot", "pre-tool"])[0], 0)
            notice.assert_not_called()
            capture.assert_not_called()
            # Control: other commands still show it, through either entry point.
            with patch("agent_trace.cli.cmd_list", return_value=0):
                for main in (entry.main, cli.main):
                    self.run_main(main, ["list"])
            self.assertEqual(notice.call_count, 2)


# ---------------------------------------------------------------------------
# Hook processes: old entry point versus new
# ---------------------------------------------------------------------------

_COMMANDS = {
    # What the console script ran before (agent_trace.cli:main) and now.
    "cli": [sys.executable, "-c", "import sys; from agent_trace.cli import main; sys.exit(main())"],
    "entry": [sys.executable, "-c", "import sys; from agent_trace.entry import main; sys.exit(main())"],
    "cli-module": [sys.executable, "-m", "agent_trace.cli"],
    "entry-module": [sys.executable, "-m", "agent_trace.entry"],
}

_SECRET = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
_MASKED_KEYS = {
    "timestamp", "started_at", "ended_at", "updated_at", "duration_ms", "total_duration_ms",
    "prev_hash", "pid", "ppid", "uuid", "$insert_id",
}
_EVENT_ID = re.compile(r"[0-9a-f]{12}")


def _base_env():
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("AGENT_STRACE_", "AGENT_TRACE_")) and key != "DO_NOT_TRACK"
    }
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_SRC), os.environ.get("PYTHONPATH")]))
    env["AGENT_STRACE_TELEMETRY"] = "0"
    return env


class _Normaliser:
    """Mask values that differ between runs; number event ids by first use."""

    def __init__(self, root):
        self.root = str(root)
        self.ids = {}

    def __call__(self, value, key=None):
        if isinstance(value, dict):
            return {self(k): self(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [self(v) for v in value]
        if key in _MASKED_KEYS and value is not None:
            return "<masked>"
        if isinstance(value, str):
            if _EVENT_ID.fullmatch(value):
                return self.ids.setdefault(value, f"<event {len(self.ids)}>")
            return value.replace(self.root, "<root>")
        return value

    def text(self, raw):
        try:
            return self(json.loads(raw))
        except ValueError:
            pass
        lines = []
        for line in raw.splitlines():
            try:
                lines.append(self(json.loads(line)))
            except ValueError:
                lines.append(self(line))
        return lines

    def tree(self, root):
        return {
            self(str(path)): self.text(path.read_text(errors="replace"))
            for path in sorted(p for p in Path(root).rglob("*") if p.is_file())
        }


class _Recorder(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.server.received.append((self.path, self.headers.get("Content-Type"), body.decode()))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *args):
        pass


@contextlib.contextmanager
def _http_server():
    server = HTTPServer(("127.0.0.1", 0), _Recorder)
    server.received = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _closed_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _lifecycle(provider, session):
    """Return (event, payload) steps that exercise every hook handler."""
    cwd = "{work}"
    if provider == "copilot":
        base = {"sessionId": session, "cwd": cwd}

        def tool(use_id, result=None):
            payload = {**base, "toolName": "bash", "toolArgs": {"command": f"echo {use_id}"}, "toolUseId": use_id}
            if result is not None:
                payload["toolResult"] = {"resultType": result, "textResultForLlm": "ok"}
            return payload

        return [
            ("sessionStart", {**base, "source": "new", "initialPrompt": "start"}),
            ("userPromptSubmitted", {**base, "prompt": f"use {_SECRET}"}),
            ("preToolUse", tool("t1")),
            ("postToolUse", tool("t1", "success")),
            ("preToolUse", tool("t2")),
            ("postToolUseFailure", {**tool("t2", "failure"), "error": {"message": "boom"}}),
            ("subagentStart", {**base, "agentName": "helper"}),
            ("subagentStop", {**base, "agentName": "helper"}),
            ("agentStop", {**base, "stopReason": "end_turn"}),
            ("sessionEnd", {**base, "reason": "complete"}),
        ]
    if provider == "cursor":
        base = {"conversation_id": session, "generation_id": "g1", "cwd": cwd, "workspace_roots": [cwd]}
        return [
            ("session-start", base),
            ("before-submit-prompt", {**base, "prompt": "hello"}),
            ("before-shell-execution", {**base, "command": "ls"}),
            ("after-shell-execution", {**base, "command": "ls", "output": "a"}),
            ("after-file-edit", {**base, "file_path": f"{cwd}/a.txt", "edits": [{"old_string": "", "new_string": "x"}]}),
            ("after-agent-response", {**base, "text": "done"}),
            ("stop", {**base, "status": "completed"}),
            ("session-end", base),
        ]
    base = {"session_id": session, "cwd": cwd}

    def tool(use_id, response=None):
        payload = {**base, "tool_name": "Bash", "tool_input": {"command": f"echo {use_id}"}, "tool_use_id": use_id}
        if response is not None:
            payload["tool_response"] = response
        return payload

    names = {
        "claude": ["session-start", "user-prompt", "pre-tool", "post-tool", "post-tool-failure",
                   "subagent-start", "subagent-stop", "stop", "session-end"],
        "codex": ["SessionStart", "UserPromptSubmit", "PreToolUse", "PostToolUse", "PostToolUseFailure",
                  "SubagentStart", "SubagentStop", "Stop", "SessionEnd"],
        "gemini": ["session-start", "before-agent", "before-tool", "after-tool", "post-tool-failure",
                   "subagent-start", "subagent-stop", "after-agent", "session-end"],
    }[provider]
    steps = [
        (names[0], {**base, "source": "startup", "model": "m-1"}),
        (names[1], {**base, "prompt": f"use {_SECRET}"}),
        (names[2], tool("t1")),
        (names[3], tool("t1", {"stdout": "ok"})),
        (names[2], tool("t2")),
        (names[4], {**tool("t2"), "error": "boom"}),
        (names[5], {**base, "agent_id": "a1", "agent_type": "helper"}),
        (names[6], {**base, "agent_id": "a1", "agent_type": "helper"}),
    ]
    if provider == "codex":
        steps.append(("PostModelRequest", {**base, "model": "m-1", "turn_id": "turn-1"}))
    steps += [
        (names[7], {**base, "stop_reason": "end_turn"}),
        (names[8], {**base, "reason": "exit"}),
    ]
    return steps


class TestHookProcessEquivalence(unittest.TestCase):
    """Run the same hook sequence through the old and new entry points."""

    maxDiff = None

    def run_steps(self, mode, steps, env=None, use_trace_dir=True, servers=None):
        """Run ``steps`` in a fresh directory and return normalised results."""
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            root = Path(tmp).resolve()
            work = root / "work"
            work.mkdir()
            proc_env = _base_env()
            proc_env["AGENT_STRACE_TELEMETRY_CONFIG"] = str(root / "telemetry.json")
            if use_trace_dir:
                proc_env["AGENT_TRACE_DIR"] = str(root / "traces")
            received = {}
            for name, variable in (servers or {}).items():
                server = stack.enter_context(_http_server())
                proc_env[variable] = f"http://127.0.0.1:{server.server_port}"
                received[name] = server.received
            proc_env.update(env or {})
            normalise = _Normaliser(root)
            results = []
            for argv, stdin in steps:
                if callable(stdin):
                    stdin(root)
                    continue
                if not isinstance(stdin, str):
                    stdin = json.dumps(stdin).replace("{work}", str(work))
                proc = subprocess.run(
                    _COMMANDS[mode] + argv,
                    input=stdin,
                    capture_output=True,
                    text=True,
                    cwd=work,
                    env=proc_env,
                    timeout=60,
                )
                results.append({
                    "argv": argv,
                    "returncode": proc.returncode,
                    "stdout": normalise.text(proc.stdout),
                    "stderr": normalise.text(proc.stderr),
                    "files": normalise.tree(root),
                })
            for name, items in received.items():
                results.append({
                    name: [(path, ctype, normalise.text(body)) for path, ctype, body in items],
                })
            return results

    def compare(self, steps, modes=("cli", "entry"), **kwargs):
        with ThreadPoolExecutor(len(modes)) as pool:
            runs = list(pool.map(lambda mode: self.run_steps(mode, steps, **kwargs), modes))
        for mode, run in zip(modes[1:], runs[1:]):
            for expected, actual in zip(runs[0], run):
                self.assertEqual(actual, expected, f"{mode} differs from {modes[0]} at {expected.get('argv')}")
            self.assertEqual(len(run), len(runs[0]))
        return runs[0]

    @staticmethod
    def hook_steps(provider, session="sess-1"):
        prefix = ["hook"] if provider == "claude" else ["hook", "--provider", provider]
        return [(prefix + [event], payload) for event, payload in _lifecycle(provider, session)]

    def events(self, run, session="sess-1"):
        files = run[-1]["files"]
        name = next(k for k in files if k.endswith(f"{session}/events.ndjson"))
        return files[name]

    def test_provider_lifecycles(self):
        for provider in entry.HOOK_PROVIDERS:
            with self.subTest(provider=provider):
                run = self.compare(self.hook_steps(provider))
                self.assertTrue(all(step["returncode"] == 0 for step in run))
                events = [e["event_type"] for e in self.events(run)]
                self.assertEqual(events[0], "session_start")
                self.assertEqual(events[-1], "session_end")
                self.assertIn("tool_call", events)
                self.assertIn("tool_result", events)
                if provider == "codex":
                    self.assertIn("llm_response", events)
                if provider in ("copilot", "claude"):
                    # test_metadata_only_lifecycles asserts this content is dropped.
                    self.assertIn("echo t1", json.dumps(self.events(run)))

    def test_metadata_only_lifecycles(self):
        for provider in ("copilot", "claude"):
            with self.subTest(provider=provider):
                steps = [
                    (argv[:-1] + [hooks.METADATA_ONLY_FLAG, argv[-1]], payload)
                    for argv, payload in self.hook_steps(provider)
                ]
                run = self.compare(steps)
                self.assertTrue(all(step["returncode"] == 0 for step in run))
                events = self.events(run)
                self.assertIn("tool_call", [e["event_type"] for e in events])
                self.assertNotIn("echo t1", json.dumps(events))

    def test_module_entry_points(self):
        run = self.compare(self.hook_steps("copilot"), modes=("cli", "entry", "cli-module", "entry-module"))
        self.assertIn("tool_call", [e["event_type"] for e in self.events(run)])

    def test_redaction_modes(self):
        steps = self.hook_steps("claude")[:4]
        for env, redacted in (({}, True), ({"AGENT_TRACE_REDACT": "1"}, True), ({"AGENT_TRACE_NO_REDACT": "1"}, False)):
            with self.subTest(env=env):
                run = self.compare(steps, env=env)
                written = json.dumps(self.events(run))
                self.assertEqual(_SECRET in written, not redacted)
                self.assertEqual("[REDACTED" in written, redacted)

    def test_trace_dir_from_payload_cwd(self):
        run = self.compare(self.hook_steps("codex")[:4], use_trace_dir=False)
        self.assertTrue(any("/work/.agent-traces/" in name for name in run[-1]["files"]))

    def test_streaming_to_endpoint(self):
        run = self.compare(self.hook_steps("copilot"), servers={"collector": "AGENT_STRACE_ENDPOINT"})
        received = run[-1]["collector"]
        paths = [path for path, _, _ in received]
        self.assertIn("/events", paths)
        self.assertIn("/sessions", paths)
        self.assertIn("redacted secrets", json.dumps(run))
        self.assertNotIn(_SECRET, json.dumps(received))

    def test_streaming_endpoint_unreachable(self):
        endpoint = f"http://127.0.0.1:{_closed_port()}"
        run = self.compare(self.hook_steps("codex")[:4], env={"AGENT_STRACE_ENDPOINT": endpoint})
        self.assertIn("remote send failed", json.dumps(run))
        self.assertTrue(all(step["returncode"] == 0 for step in run))

    def test_session_telemetry(self):
        def seed(root):
            (root / "telemetry.json").write_text(json.dumps({"anonymous_id": "0" * 32}))

        steps = [([], seed)] + self.hook_steps("codex")
        run = self.compare(
            steps,
            env={"AGENT_STRACE_TELEMETRY": "1"},
            servers={"posthog": "AGENT_STRACE_TELEMETRY_HOST"},
        )
        captured = run[-1]["posthog"]
        self.assertEqual(len(captured), 1)
        path, _, body = captured[0]
        self.assertEqual(path, "/i/v0/e/")
        self.assertEqual(body["event"], telemetry.SESSION_COMPLETED)
        self.assertEqual(body["distinct_id"], "0" * 32)

    def test_errors_and_fallbacks(self):
        payload = {"session_id": "s", "cwd": "{work}", "tool_name": "Bash", "tool_input": {}}
        steps = [
            (["hook"], payload),
            (["hook", "--provider", "codex"], payload),
            (["hook", "no-such-event"], payload),
            (["hook", "--provider", "bogus", "pre-tool"], payload),
            (["hook", "-h"], payload),
            (["hook", "pre-tool", "extra"], payload),
            (["hook", "--prov", "copilot", "pre-tool"], payload),
            (["hook", "--provider=gemini", "pre-tool"], payload),
            (["--trace-dir", "elsewhere", "hook", "pre-tool"], payload),
            (["hook", "pre-tool"], "not json"),
            (["hook", "pre-tool"], ""),
        ]
        run = self.compare(steps)
        codes = [step["returncode"] for step in run]
        self.assertEqual(codes, [1, 1, 1, 2, 0, 2, 0, 0, 0, 0, 0])

    def test_non_object_payload(self):
        # A JSON payload that is not an object raises before any handler runs.
        # Tracebacks name the entry module, so compare the outcome only.
        steps = [(["hook", "pre-tool"], "[1, 2]")]
        runs = [self.run_steps(mode, steps) for mode in ("cli", "entry")]
        for run in runs:
            run[0]["stderr"] = run[0]["stderr"][-1:]
        self.assertEqual(runs[1], runs[0])
        self.assertEqual(runs[0][0]["returncode"], 1)


# ---------------------------------------------------------------------------
# Regression: the hook path stays light
# ---------------------------------------------------------------------------

_MODULE_DUMP = (
    "import atexit, os, sys\n"
    "atexit.register(lambda: open(os.environ['MODULES_OUT'], 'w').write('\\n'.join(sorted(sys.modules))))\n"
)


class TestHookPathModules(unittest.TestCase):
    """Module-set checks, independent of machine speed."""

    def loaded_modules(self, launcher, argv, payload):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "modules.txt"
            env = _base_env()
            env.update({
                "AGENT_TRACE_DIR": str(Path(tmp) / "traces"),
                "AGENT_STRACE_TELEMETRY_CONFIG": str(Path(tmp) / "telemetry.json"),
                "MODULES_OUT": str(out),
            })
            proc = subprocess.run(
                [sys.executable, "-c", _MODULE_DUMP + launcher, *argv],
                input=json.dumps({**payload, "cwd": tmp}),
                capture_output=True,
                text=True,
                cwd=tmp,
                env=env,
                timeout=60,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            return set(out.read_text().split())

    def test_hook_path_does_not_import_the_cli(self):
        entry_launcher = "from agent_trace.entry import main\nsys.exit(main())\n"
        module_launcher = "import runpy\nrunpy.run_module('agent_trace.entry', run_name='__main__')\n"
        tool = {"session_id": "s1", "tool_name": "Bash", "tool_input": {"command": "ls"}, "tool_use_id": "t1"}
        cases = [
            (["hook", "session-start"], {"session_id": "s1"}),
            (["hook", "pre-tool"], tool),
            (["hook", "post-tool"], {**tool, "tool_response": "ok"}),
            (["hook", "--provider", "codex", "PostModelRequest"], {"session_id": "s1", "model": "m"}),
            (["hook", "--provider", "codex", "SessionEnd"], {"session_id": "s1"}),
            (["hook", "--provider", "copilot", "userPromptSubmitted"], {"sessionId": "s1", "prompt": "hi"}),
            (["hook", "--provider=gemini", "before-tool"], tool),
            (["hook", "--provider", "cursor", "before-shell-execution"], {"conversation_id": "s1", "command": "ls"}),
            (["hook", "--provider", "copilot", "--metadata-only", "preToolUse"], {**tool, "sessionId": "s1"}),
        ]
        for launcher in (entry_launcher, module_launcher):
            for argv, payload in cases:
                with self.subTest(launcher=launcher.splitlines()[0], argv=argv):
                    modules = self.loaded_modules(launcher, argv, payload)
                    self.assertIn("agent_trace.hooks", modules)
                    self.assertNotIn("agent_trace.cli", modules)
                    # The slowest imports the fast path avoids.
                    for name in ("argparse", "urllib.request", "http.client", "agent_trace.server"):
                        self.assertNotIn(name, modules)

    def test_full_parser_path_still_imports_the_cli(self):
        # Control for the check above: a fallback line loads the full CLI.
        launcher = "from agent_trace.entry import main\nsys.exit(main())\n"
        modules = self.loaded_modules(launcher, ["hook", "--prov", "claude", "pre-tool"], {"session_id": "s1"})
        self.assertIn("agent_trace.cli", modules)


if __name__ == "__main__":
    unittest.main()
