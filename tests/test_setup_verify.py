"""Offline Codex and Claude Code setup verification tests; no actual provider configuration touched."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_trace import __version__, setup_verify
from agent_trace.cli import (
    _claude_config_dir,
    _claude_hooks_config,
    _claude_managed_settings_dir,
    _codex_hooks_config,
    build_parser,
    cmd_setup,
    main,
)
from agent_trace.setup_verify import (
    check_codex_hooks_feature,
    verify_claude_setup,
    verify_codex_setup,
)

# Every config.toml case runs under tomllib (3.11+) and the 3.10 fallback reader.
_PARSERS = ("fallback",) if setup_verify._tomllib is None else ("tomllib", "fallback")
_NOT_SET = "hooks not set; enabled by default"


def _args(**updates):
    values = {
        "cli": "codex",
        "verify": True,
        "redact": False,
        "no_redact": False,
        "global_config": False,
    }
    values.update(updates)
    return argparse.Namespace(**values)


class CodexSetupVerifyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.hooks_path = Path(self.tmp.name) / "hooks.json"
        self.gemini_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.gemini_tmp.cleanup)
        self.env = patch.dict(os.environ, {
            "CODEX_HOME": self.tmp.name,
            "GEMINI_CONFIG_DIR": self.gemini_tmp.name,
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        os.environ.pop("CODEX_CONFIG_DIR", None)

    def _write_config(self):
        self.hooks_path.write_text(
            json.dumps(_codex_hooks_config(_args())), encoding="utf-8"
        )

    def _probe_ok(self):
        return patch(
            "agent_trace.setup_verify.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=["/bin/agent-strace", "--version"],
                returncode=0,
                stdout=f"agent-strace {__version__}\n",
                stderr="",
            ),
        )

    def test_clean_config_verifies_without_rewriting_files_or_running_hooks(self):
        self._write_config()
        with patch("agent_trace.setup_verify.shutil.which", return_value="/bin/agent-strace"), self._probe_ok() as run:
            before = self.hooks_path.read_bytes()
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                status = cmd_setup(_args())
            self.assertEqual(status, 0)
            self.assertEqual(self.hooks_path.read_bytes(), before)
            self.assertFalse(Path(self.tmp.name, "hooks.json.agent-strace.bak").exists())
            self.assertIn("static hook setup verified", out.getvalue())
            self.assertIn("does not prove", out.getvalue())
            run.assert_called_once()
            self.assertEqual(run.call_args.args[0], ["/bin/agent-strace", "--version"])
            self.assertEqual(run.call_args.kwargs["timeout"], 5)

    def test_missing_config_is_an_error_and_creates_nothing(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(cmd_setup(_args()), 1)
        self.assertIn("Missing Codex hooks config", err.getvalue())
        self.assertFalse(self.hooks_path.exists())
        self.assertEqual(list(Path(self.tmp.name).iterdir()), [])

    def test_malformed_json_and_invalid_structure_fail(self):
        for content in ("{garbage", "[]", '{"hooks":[]}'):
            with self.subTest(content=content):
                self.hooks_path.write_text(content)
                ok, issues = verify_codex_setup(
                    self.hooks_path, _codex_hooks_config(_args()), __version__
                )
                self.assertFalse(ok)
                self.assertTrue(issues)

    def test_missing_or_stale_hook_command_fails_without_modification(self):
        config = _codex_hooks_config(_args())
        config["hooks"]["Stop"][0]["hooks"][0]["command"] = "agent-strace hook --provider codex old-stop"
        self.hooks_path.write_text(json.dumps(config))
        with patch("agent_trace.setup_verify.shutil.which", return_value="/bin/agent-strace"), self._probe_ok():
            contents = self.hooks_path.read_bytes()
            ok, problems = verify_codex_setup(
                self.hooks_path, _codex_hooks_config(_args()), __version__
            )
            self.assertFalse(ok)
            self.assertTrue(any("Stop:" in p for p in problems))
            self.assertEqual(contents, self.hooks_path.read_bytes())

    def test_wrong_event_matcher_or_hook_type_cannot_appear_verified(self):
        for mutation in ("matcher", "type"):
            with self.subTest(mutation=mutation):
                config = _codex_hooks_config(_args())
                if mutation == "matcher":
                    config["hooks"]["PreToolUse"][0]["matcher"] = "NeverMatchingTool"
                else:
                    config["hooks"]["PreToolUse"][0]["hooks"][0]["type"] = "prompt"
                self.hooks_path.write_text(json.dumps(config))
                with patch("agent_trace.setup_verify.shutil.which", return_value="/bin/agent-strace"), self._probe_ok():
                    ok, problems = verify_codex_setup(
                        self.hooks_path, _codex_hooks_config(_args()), __version__
                    )
                self.assertFalse(ok)
                self.assertTrue(any("PreToolUse:" in msg for msg in problems))

    def test_non_agent_hooks_are_allowed_without_removal(self):
        config = _codex_hooks_config(_args())
        config["hooks"]["Stop"][0]["hooks"].append(
            {"type": "command", "command": "my-existing-user-tool --run"}
        )
        self.hooks_path.write_text(json.dumps(config))
        with patch("agent_trace.setup_verify.shutil.which", return_value="/bin/agent-strace"), self._probe_ok():
            ok, problems = verify_codex_setup(
                self.hooks_path, _codex_hooks_config(_args()), __version__
            )
        self.assertTrue(ok, problems)

    def test_missing_executable_and_mismatched_version_fail(self):
        self._write_config()
        with patch("agent_trace.setup_verify.shutil.which", return_value=None):
            ok, problems = verify_codex_setup(
                self.hooks_path, _codex_hooks_config(_args()), __version__
            )
            self.assertFalse(ok)
            self.assertIn("not on PATH", " ".join(problems))

        bad = subprocess.CompletedProcess(
            args=["/bin/agent-strace", "--version"],
            returncode=0,
            stdout="agent-strace 0.01.0\n",
            stderr="",
        )
        with patch("agent_trace.setup_verify.shutil.which", return_value="/bin/agent-strace"), patch(
            "agent_trace.setup_verify.subprocess.run", return_value=bad
        ):
            ok, problems = verify_codex_setup(
                self.hooks_path, _codex_hooks_config(_args()), __version__
            )
            self.assertFalse(ok)
            self.assertIn("version mismatch", " ".join(problems))

    def test_timeout_is_reported_and_does_not_write_any_file(self):
        self._write_config()
        with patch("agent_trace.setup_verify.shutil.which", return_value="/bin/agent-strace"), patch(
            "agent_trace.setup_verify.subprocess.run",
            side_effect=subprocess.TimeoutExpired(["agent-strace", "--version"], 5),
        ):
            ok, problems = verify_codex_setup(
                self.hooks_path, _codex_hooks_config(_args()), __version__
            )
        self.assertFalse(ok)
        self.assertIn("Cannot run", " ".join(problems))

    def test_non_codex_verification_fails_without_creating_provider_config(self):
        args = _args(cli="gemini")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(cmd_setup(args), 2)
        self.assertIn("only --cli codex", err.getvalue())
        self.assertEqual(list(Path(self.gemini_tmp.name).iterdir()), [])

    def test_entry_point_records_verify_subcommand_in_telemetry(self):
        with patch.object(
            __import__("sys"), "argv",
            ["agent-strace", "setup", "--cli", "codex", "--verify"],
        ), patch("agent_trace.cli._product_telemetry.capture") as capture, patch(
            "agent_trace.cli._product_telemetry.maybe_show_telemetry_notice"
        ) as notice:
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                with self.assertRaises(SystemExit) as result:
                    main()
        self.assertEqual(result.exception.code, 1)  # no config
        capture.assert_called_once()
        props = capture.call_args.args[1]
        self.assertEqual(props["command"], "setup")
        self.assertEqual(props["integration"], "codex")
        self.assertEqual(props["subcommand"], "verify")
        self.assertFalse(props["success"])
        self.assertEqual(props["exit_code"], 1)
        self.assertNotIn("hooks_path", props)
        notice.assert_called_once()
        self.assertFalse(self.hooks_path.exists())

    def test_redaction_modes_verify_without_matching_cli_flags(self):
        for mode in ("default", "redact", "no_redact"):
            with self.subTest(mode=mode):
                args = _args(verify=False, redact=mode == "redact", no_redact=mode == "no_redact")
                self.hooks_path.write_text(json.dumps(_codex_hooks_config(args)))
                details = {}
                with patch("agent_trace.setup_verify.shutil.which", return_value="/bin/agent-strace"), self._probe_ok():
                    ok, issues = verify_codex_setup(
                        self.hooks_path, _codex_hooks_config(_args()), __version__, details=details
                    )
                self.assertTrue(ok, issues)
                self.assertEqual(details["redaction"], mode.replace("_", "-"))
                self.assertEqual(details["event_count"], 5)
                self.assertEqual(details["executable"], "/bin/agent-strace")

    def test_mixed_modes_are_rejected(self):
        config = _codex_hooks_config(_args())
        config["hooks"]["Stop"][0]["hooks"][0]["command"] = (
            "AGENT_TRACE_REDACT=1 " + config["hooks"]["Stop"][0]["hooks"][0]["command"]
        )
        self.hooks_path.write_text(json.dumps(config))
        with patch("agent_trace.setup_verify.shutil.which", return_value="/bin/agent-strace"), self._probe_ok():
            ok, issues = verify_codex_setup(self.hooks_path, _codex_hooks_config(_args()), __version__)
        self.assertFalse(ok)
        self.assertIn("inconsistent redaction", " ".join(issues))

    def test_documented_handler_fields_and_match_all_aliases_are_accepted(self):
        for alias in (None, "", "*", ".*"):
            with self.subTest(matcher=alias):
                config = _codex_hooks_config(_args())
                for event in ("PreToolUse", "PostToolUse", "SessionStart"):
                    if alias is None:
                        config["hooks"][event][0].pop("matcher", None)
                    else:
                        config["hooks"][event][0]["matcher"] = alias
                for event in ("Stop", "UserPromptSubmit"):
                    config["hooks"][event][0]["matcher"] = "*"
                    config["hooks"][event][0]["hooks"][0]["timeout"] = 30
                    config["hooks"][event][0]["hooks"][0]["statusMessage"] = "running"
                self.hooks_path.write_text(json.dumps(config))
                with patch("agent_trace.setup_verify.shutil.which", return_value="/bin/agent-strace"), self._probe_ok():
                    ok, issues = verify_codex_setup(
                        self.hooks_path, _codex_hooks_config(_args()), __version__
                    )
                self.assertTrue(ok, issues)

    def test_session_start_extra_sources_are_accepted(self):
        config = _codex_hooks_config(_args())
        config["hooks"]["SessionStart"][0]["matcher"] = "startup|resume|clear|compact|fork"
        self.hooks_path.write_text(json.dumps(config))
        with patch("agent_trace.setup_verify.shutil.which", return_value="/bin/agent-strace"), self._probe_ok():
            ok, issues = verify_codex_setup(self.hooks_path, _codex_hooks_config(_args()), __version__)
        self.assertTrue(ok, issues)

    def test_wrong_matcher_is_not_falsely_reported_as_missing_command(self):
        config = _codex_hooks_config(_args())
        config["hooks"]["PreToolUse"][0]["matcher"] = "Bash"
        self.hooks_path.write_text(json.dumps(config))
        with patch("agent_trace.setup_verify.shutil.which", return_value="/bin/agent-strace"), self._probe_ok():
            ok, issues = verify_codex_setup(self.hooks_path, _codex_hooks_config(_args()), __version__)
        self.assertFalse(ok)
        joined = " ".join(issues)
        self.assertIn("PreToolUse: matcher is 'Bash'", joined)
        self.assertNotIn("PreToolUse: missing", joined)

    def test_wrong_event_and_duplicate_hook_commands_fail(self):
        for mutation in ("wrong_event", "duplicate_group", "duplicate_handler"):
            with self.subTest(mutation=mutation):
                config = _codex_hooks_config(_args())
                stop = config["hooks"]["Stop"][0]
                if mutation == "wrong_event":
                    config["hooks"].setdefault("SessionEnd", []).append(json.loads(json.dumps(stop)))
                elif mutation == "duplicate_group":
                    config["hooks"]["Stop"].append(json.loads(json.dumps(stop)))
                else:
                    stop["hooks"].append(json.loads(json.dumps(stop["hooks"][0])))
                self.hooks_path.write_text(json.dumps(config))
                with patch("agent_trace.setup_verify.shutil.which", return_value="/bin/agent-strace"), self._probe_ok():
                    ok, problems = verify_codex_setup(
                        self.hooks_path, _codex_hooks_config(_args()), __version__
                    )
                self.assertFalse(ok)
                self.assertIn("wrong event" if mutation == "wrong_event" else "duplicate", " ".join(problems))
                self.assertIn("agent-strace hook --provider codex stop", " ".join(problems))

    def test_missing_command_has_actionable_diagnostics_without_unsafe_overwrite_advice(self):
        config = _codex_hooks_config(_args())
        config["hooks"]["Stop"][0]["hooks"] = [
            {"type": "command", "command": "my-own-hook --keep"}
        ]
        self.hooks_path.write_text(json.dumps(config))
        with patch("agent_trace.setup_verify.shutil.which", return_value="/bin/agent-strace"), self._probe_ok():
            ok, issues = verify_codex_setup(
                self.hooks_path, _codex_hooks_config(_args()), __version__
            )
        self.assertFalse(ok)
        joined = " ".join(issues)
        self.assertIn("Stop: missing", joined)
        self.assertIn("agent-strace hook --provider codex stop", joined)
        self.assertNotIn("Re-run setup", joined)

    def test_codex_config_environment_precedence_and_executable_summary(self):
        from agent_trace.cli import _codex_config_dir
        with tempfile.TemporaryDirectory() as alternate:
            with patch.dict(os.environ, {"CODEX_CONFIG_DIR": "", "CODEX_HOME": alternate}):
                self.assertEqual(_codex_config_dir(), Path(alternate))
            with patch.dict(os.environ, {"CODEX_CONFIG_DIR": self.tmp.name, "CODEX_HOME": alternate}):
                self.assertEqual(_codex_config_dir(), Path(alternate))
        self._write_config()
        output = io.StringIO()
        with patch("agent_trace.setup_verify.shutil.which", return_value="/bin/agent-strace"), self._probe_ok():
            with contextlib.redirect_stdout(output):
                self.assertEqual(cmd_setup(_args()), 0)
        self.assertIn("5/5 events, redaction: default", output.getvalue())
        self.assertIn("/bin/agent-strace (agent-strace", output.getvalue())
        self.assertIn("\n"+"hooks:      ", output.getvalue())
        self.assertEqual(output.getvalue().count(str(self.hooks_path)), 1)
        self.assertIn("\n"+"executable: ", output.getvalue())

    def test_hooks_disabled_in_config_toml_fails_without_writes(self):
        self._write_config()
        config_toml = Path(self.tmp.name) / "config.toml"
        config_toml.write_text("[features]\nhooks = false\n", encoding="utf-8")
        before = (config_toml.read_bytes(), self.hooks_path.read_bytes())
        err, out = io.StringIO(), io.StringIO()
        with patch("agent_trace.setup_verify.shutil.which", return_value="/bin/agent-strace"), self._probe_ok():
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
                self.assertEqual(cmd_setup(_args()), 1)
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(
            err.getvalue(),
            f"Setup verification failed: Codex hooks are disabled in {config_toml}: "
            "[features].hooks = false. Remove that setting or set [features].hooks = true, "
            "then restart Codex.\n",
        )
        self.assertEqual((config_toml.read_bytes(), self.hooks_path.read_bytes()), before)
        self.assertEqual(sorted(p.name for p in Path(self.tmp.name).iterdir()), ["config.toml", "hooks.json"])

    def test_config_toml_is_read_from_the_resolved_codex_dir(self):
        with tempfile.TemporaryDirectory() as codex_home:
            Path(codex_home, "hooks.json").write_text(json.dumps(_codex_hooks_config(_args())))
            Path(codex_home, "config.toml").write_text("features.codex_hooks = false\n")
            err = io.StringIO()
            with patch.dict(os.environ, {"CODEX_CONFIG_DIR": "", "CODEX_HOME": codex_home}), patch(
                "agent_trace.setup_verify.shutil.which", return_value="/bin/agent-strace"
            ), self._probe_ok(), contextlib.redirect_stderr(err):
                self.assertEqual(cmd_setup(_args()), 1)
            self.assertIn(
                f"disabled in {Path(codex_home, 'config.toml')}: [features].codex_hooks = false",
                err.getvalue(),
            )

    def test_config_toml_problem_is_reported_with_hooks_json_problems(self):
        config_toml = Path(self.tmp.name) / "config.toml"
        config_toml.write_text("[features]\ncodex_hooks = false\n", encoding="utf-8")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(cmd_setup(_args()), 1)
        self.assertIn("[features].codex_hooks = false", err.getvalue())
        self.assertIn("Missing Codex hooks config", err.getvalue())
        self.assertEqual([p.name for p in Path(self.tmp.name).iterdir()], ["config.toml"])

    def test_verified_summary_reports_config_toml_state_and_unchecked_layers(self):
        self._write_config()
        config_toml = Path(self.tmp.name) / "config.toml"
        for content, state in (
            (None, "not found; hooks enabled by default"),
            ("[features]\nhooks = true\n", "[features].hooks = true"),
            ('model = "o3"\n[profiles.fast.features]\nhooks = false\n', _NOT_SET),
        ):
            with self.subTest(state=state):
                if content is not None:
                    config_toml.write_text(content, encoding="utf-8")
                out = io.StringIO()
                with patch("agent_trace.setup_verify.shutil.which", return_value="/bin/agent-strace"), self._probe_ok():
                    with contextlib.redirect_stdout(out):
                        self.assertEqual(cmd_setup(_args()), 0)
                self.assertIn(f"\nconfig:     {config_toml} ({state})\n", out.getvalue())
                self.assertIn(
                    "Not checked: Codex profiles, -c overrides, and project, system, "
                    "or managed config layers.", out.getvalue()
                )
                self.assertIn("does not prove", out.getvalue())

    def test_cli_parser_accepts_verify_with_codex(self):
        parsed = build_parser().parse_args(["setup", "--cli", "codex", "--verify"])
        self.assertEqual(parsed.cli, "codex")
        self.assertTrue(parsed.verify)


_REALISTIC_CONFIG = '''\
# Codex user config with look-alike settings that must not count.
model = "gpt-5-codex"
model_reasoning_effort = 'high'
notify = ["python3", "/Users/me/.codex/notify.py"]
instructions = """
Prefer small diffs.
[features]
hooks = false
"""
banner = \'\'\'
codex_hooks = false\'\'\'
quoted = "say \\"hooks = false\\" # not a comment"

[shell_environment_policy]
inherit = "core"
set = { PATH = "/usr/bin", "DOTTED.KEY" = 'x' }

[mcp_servers.docs]
command = "npx"
args = [
  "-y",  # hooks = false
  "@acme/docs-mcp",
  { nested = [1, 2.5, 1979-05-27T07:32:00Z] },
]
env = { "API_KEY" = "redacted" }
startup_timeout_sec = 10.5

[profiles.fast]
model = "o4-mini"
features.hooks = false

[profiles.slow.features]
codex_hooks = false

[[skills.config]]
path = "/tmp/a"
features.hooks = false
'''


class CodexHooksFeatureTests(unittest.TestCase):
    """config.toml [features] checks; nothing is written."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config_path = Path(self.tmp.name) / "config.toml"

    def _check(self, parser, text=None):
        if text is not None:
            self.config_path.write_text(text, encoding="utf-8")
        module = setup_verify._tomllib if parser == "tomllib" else None
        with patch.object(setup_verify, "_tomllib", module):
            return check_codex_hooks_feature(self.config_path)

    def test_disabled_hooks_fail_naming_file_key_and_fix(self):
        cases = {
            "[features]\nhooks = false\n": ("[features].hooks = false.", "Remove that setting"),
            "features.hooks = false\n": ("[features].hooks = false.",),
            "features = { hooks = false }\n": ("[features].hooks = false.",),
            '[ features ] # c\n"hooks"\t=\tfalse # off\n': ("[features].hooks = false.",),
            "[features]\ncodex_hooks = false\n": (
                "[features].codex_hooks = false (codex_hooks is the legacy name of hooks).",
            ),
            "[features]\nhooks = false\ncodex_hooks = true\n": ("[features].hooks = false.",),
            "[features]\ncodex_hooks = false\nhooks = false\n": (
                "[features].hooks = false and [features].codex_hooks = false",
                "Remove both settings",
            ),
            _REALISTIC_CONFIG + "\n[features]\nhooks = false\n": ("[features].hooks = false.",),
        }
        for parser in _PARSERS:
            for text, fragments in cases.items():
                with self.subTest(parser=parser, config=text[-60:]):
                    problems, summary = self._check(parser, text)
                    self.assertEqual(summary, "")
                    self.assertEqual(len(problems), 1, problems)
                    self.assertTrue(problems[0].startswith(
                        f"Codex hooks are disabled in {self.config_path}: "
                    ), problems[0])
                    self.assertIn("set [features].hooks = true, then restart Codex.", problems[0])
                    for fragment in fragments:
                        self.assertIn(fragment, problems[0])

    def test_enabled_unset_or_other_tables_pass(self):
        cases = {
            "": _NOT_SET,
            "# hooks = false\n": _NOT_SET,
            "[features]\nother_feature = false\n": _NOT_SET,
            "[features]\nhooks = true\n": "[features].hooks = true",
            "[features]\ncodex_hooks = true\n": "[features].codex_hooks = true",
            # Codex applies hooks after its legacy alias, regardless of file order.
            "[features]\nhooks = true\ncodex_hooks = false\n": "[features].hooks = true",
            "[features]\ncodex_hooks = false\nhooks = true\n": "[features].hooks = true",
            "[profiles.work.features]\nhooks = false\n": _NOT_SET,
            "[profiles.work]\nfeatures.hooks = false\n": _NOT_SET,
            "profiles.work.features.hooks = false\n": _NOT_SET,
            "[features.sub]\nhooks = false\n": _NOT_SET,
            "[[agents]]\nfeatures.hooks = false\n": _NOT_SET,
            _REALISTIC_CONFIG: _NOT_SET,
            "features.codex_hooks = true\n" + _REALISTIC_CONFIG: "[features].codex_hooks = true",
        }
        for parser in _PARSERS:
            for text, summary in cases.items():
                with self.subTest(parser=parser, config=text[:60]):
                    self.assertEqual(self._check(parser, text), ([], summary))

    def test_missing_config_passes_as_codex_default(self):
        for parser in _PARSERS:
            with self.subTest(parser=parser):
                self.assertEqual(self._check(parser), ([], "not found; hooks enabled by default"))
        self.assertEqual(list(Path(self.tmp.name).iterdir()), [])

    def test_bom_and_crlf_are_accepted(self):
        self.config_path.write_bytes(b"\xef\xbb\xbf[features]\r\nhooks = false\r\n")
        for parser in _PARSERS:
            with self.subTest(parser=parser):
                problems, _ = self._check(parser)
                self.assertEqual(len(problems), 1)
                self.assertIn("[features].hooks = false.", problems[0])

    def test_unparseable_config_fails_naming_the_file(self):
        cases = (
            "[features\nhooks = false\n",
            "[features]\nhooks = \n",
            "[features]\nhooks = false extra\n",
            "[features]\nhooks = false\nhooks = true\n",
            "[features]\nhooks = 'unterminated\n",
            'note = """never closed\n',
            "args = [1, 2\n[features]\n",
        )
        for parser in _PARSERS:
            for text in cases:
                with self.subTest(parser=parser, config=text):
                    problems, summary = self._check(parser, text)
                    self.assertEqual(summary, "")
                    self.assertEqual(len(problems), 1, problems)
                    self.assertTrue(problems[0].startswith(
                        f"Cannot parse Codex config {self.config_path} as TOML: "
                    ), problems[0])
                    self.assertTrue(problems[0].endswith(
                        "Codex rejects invalid config files; fix the syntax and run this check again."
                    ))

    def test_fallback_parse_errors_report_line_numbers(self):
        problems, _ = self._check("fallback", "model = 'o3'\n\n[features]\nhooks = \n")
        self.assertIn("expected a value (line 4)", problems[0])
        problems, _ = self._check("fallback", "[features]\nhooks = false\nhooks = true\n")
        self.assertIn("duplicate key features.hooks (line 3)", problems[0])

    def test_non_boolean_hook_values_fail(self):
        cases = {
            '[features]\nhooks = "false"\n': "hooks",
            "[features]\ncodex_hooks = 0\n": "codex_hooks",
            "[features.hooks]\nenabled = false\n": "hooks",
            "features = { hooks = { enabled = true } }\n": "hooks",
            "[features]\nhooks = [false]\n": "hooks",
        }
        for parser in _PARSERS:
            for text, key in cases.items():
                with self.subTest(parser=parser, config=text):
                    self.assertEqual(self._check(parser, text), (
                        [f"Invalid Codex config {self.config_path}: [features].{key} must be true or false."],
                        "",
                    ))

    def test_unreadable_config_fails(self):
        self.config_path.write_bytes(b"[features]\nhooks = \xff\n")
        problems, _ = check_codex_hooks_feature(self.config_path)
        self.assertTrue(problems[0].startswith(f"Cannot read Codex config {self.config_path}: "))
        self.config_path.unlink()
        self.config_path.mkdir()
        problems, _ = check_codex_hooks_feature(self.config_path)
        self.assertTrue(problems[0].startswith(f"Cannot read Codex config {self.config_path}: "))

    def test_fallback_matches_tomllib_on_valid_configs(self):
        if setup_verify._tomllib is None:
            self.skipTest("tomllib requires Python 3.11+")
        corpus = [
            _REALISTIC_CONFIG,
            _REALISTIC_CONFIG + "\n[features]\nhooks = false\n",
            _REALISTIC_CONFIG + "\n[features]\ncodex_hooks = false\nhooks = true\n",
            "features.codex_hooks = false\n" + _REALISTIC_CONFIG,
            "features = { hooks = true, codex_hooks = false }\n" + _REALISTIC_CONFIG,
            'x = """a\\"""b"""\ny = """c""""\nz = \'\'\'d\'\'\'\'\'\n[features]\nhooks = false\n',
        ]
        for text in corpus:
            with self.subTest(config=text[:60]):
                self.config_path.write_text(text, encoding="utf-8")
                setup_verify._tomllib.loads(text)  # The corpus must be valid TOML.
                self.assertEqual(self._check("fallback"), self._check("tomllib"))



class ClaudeSetupVerifyTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.config_dir = self.root / "config"
        self.project = self.root / "project"
        self.managed = self.root / "managed"
        for directory in (self.config_dir, self.project, self.managed):
            directory.mkdir()
        self.settings_path = self.config_dir / "settings.json"
        self.project_path = self.project / ".claude" / "settings.json"
        self.local_path = self.project / ".claude" / "settings.local.json"
        self.managed_path = self.managed / "managed-settings.json"
        self.which = patch("agent_trace.setup_verify.shutil.which", return_value="/bin/agent-strace")
        self.run = patch(
            "agent_trace.setup_verify.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=["/bin/agent-strace", "--version"],
                returncode=0,
                stdout=f"agent-strace {__version__}\n",
                stderr="",
            ),
        )
        for active in (
            patch.dict(os.environ, {"CLAUDE_CONFIG_DIR": str(self.config_dir)}),
            patch("agent_trace.cli._claude_managed_settings_dir", return_value=self.managed),
            patch("agent_trace.cli.Path.cwd", return_value=self.project),
            self.which,
            self.run,
        ):
            active.start()
            self.addCleanup(active.stop)

    def _hooks(self, **flags):
        return _claude_hooks_config(_args(cli="claude", **flags))

    def _write(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")

    def _install(self, **extra):
        self._write(self.settings_path, {**self._hooks(), **extra})

    def _verify(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            status = cmd_setup(_args(cli="claude"))
        return status, out.getvalue(), err.getvalue()

    def _problems(self, settings=None):
        if settings is not None:
            self._write(self.settings_path, settings)
        ok, problems = verify_claude_setup(
            self.settings_path,
            self._hooks(),
            __version__,
            project_dir=self.project,
            managed_dir=self.managed,
        )
        self.assertEqual(ok, not problems)
        return " ".join(problems)

    def _snapshot(self):
        return {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}

    def test_clean_settings_verify_read_only_with_summary(self):
        self._install(model="opus")
        before = self._snapshot()
        status, out, err = self._verify()
        self.assertEqual((status, err), (0, ""))
        self.assertEqual(self._snapshot(), before)
        for expected in (
            "Claude Code static hook setup verified.\n",
            f"settings:   {self.settings_path} (7/7 events, redaction: default)\n",
            f"project:    {self.project_path} (not found)\n",
            f"local:      {self.local_path} (not found)\n",
            f"managed:    {self.managed} (no managed settings files)\n",
            f"executable: /bin/agent-strace (agent-strace {__version__})\n",
            "Not checked: other settings values (run claude doctor)",
            "MDM and server-managed policy",
            "This does not prove Claude Code trusted this folder, ran the hooks or captured an event",
        ):
            self.assertIn(expected, out)

    def test_redaction_modes_verify_without_matching_cli_flags(self):
        for flag, mode in (("redact", "redact"), ("no_redact", "no-redact")):
            with self.subTest(mode=mode):
                self._write(self.settings_path, self._hooks(**{flag: True}))
                status, out, _ = self._verify()
                self.assertEqual(status, 0)
                self.assertIn(f"redaction: {mode})", out)

    def test_missing_user_settings_fail_without_creating_files(self):
        status, out, err = self._verify()
        self.assertEqual((status, out), (1, ""))
        self.assertIn(
            f"Setup verification failed: Missing Claude Code user settings: {self.settings_path}. "
            "Run agent-strace setup --cli claude and add the printed hooks to that file.",
            err,
        )
        self.assertEqual(self._snapshot(), {})

    def test_invalid_or_empty_user_settings_fail(self):
        unrelated = {"hooks": {"PreToolUse": [{"matcher": "", "hooks": [
            {"type": "command", "command": "other-tool pre"}]}]}}
        for settings, expected in (
            ("{", "Cannot read valid Claude Code user settings JSON at"),
            ("[]", "the top level must be a JSON object"),
            ({"hooks": []}, "hooks must be an object"),
            ("", "No AgentTrace hooks in Claude Code user settings"),
            ({"model": "opus"}, "No AgentTrace hooks in Claude Code user settings"),
            (unrelated, "No AgentTrace hooks in Claude Code user settings"),
        ):
            with self.subTest(settings=settings):
                self.assertIn(expected, self._problems(settings))

    def test_missing_stale_misplaced_duplicate_or_mistyped_hooks_fail(self):
        def edited(change):
            settings = self._hooks()
            change(settings["hooks"])
            return settings

        stale = {"hooks": [{"type": "command", "command": "agent-strace hook notification"}]}
        for change, expected in (
            (lambda h: h.pop("SessionEnd"), "SessionEnd: missing"),
            (lambda h: h.setdefault("Notification", []).append(stale), "stale or unexpected"),
            (lambda h: h["PostToolUse"].append(h["PreToolUse"][0]), "assigned to the wrong event"),
            (lambda h: h["Stop"].append(h["Stop"][0]), "duplicate AgentTrace hook command"),
            (lambda h: h["Stop"][0]["hooks"][0].update(type="http"), "has handler type 'http'"),
            (lambda h: h.update(Stop={}), "Stop: hooks must be a list."),
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, self._problems(edited(change)))

    def test_narrowed_matchers_fail_and_match_all_forms_pass(self):
        for event, matcher in (("PreToolUse", "Bash"), ("SessionStart", "startup"), ("SessionEnd", "clear")):
            with self.subTest(event=event):
                settings = self._hooks()
                settings["hooks"][event][0]["matcher"] = matcher
                problems = self._problems(settings)
                self.assertIn(f"{event}: matcher is {matcher!r}, expected a match-all matcher", problems)

        settings = self._hooks()
        settings["hooks"]["PreToolUse"][0]["matcher"] = "*"
        settings["hooks"]["PostToolUse"][0]["matcher"] = ".*"
        del settings["hooks"]["PostToolUseFailure"][0]["matcher"]
        settings["hooks"]["SessionStart"][0]["matcher"] = ""
        settings["hooks"]["UserPromptSubmit"][0]["matcher"] = "ignored"
        self.assertEqual(self._problems(settings), "")

    def test_unrelated_hooks_and_extra_handler_fields_are_allowed(self):
        settings = self._hooks()
        mine = {"type": "command", "command": "~/bin/lint.sh"}
        settings["hooks"]["PreToolUse"].append({"matcher": "Bash", "hooks": [mine]})
        settings["hooks"]["PostToolUse"][0]["hooks"].append(mine)
        settings["hooks"]["Notification"] = [{"hooks": [mine]}]
        settings["hooks"]["Stop"][0]["hooks"][0].update(timeout=30, statusMessage="trace")
        settings["hooks"]["SessionEnd"][0]["hooks"][0]["async"] = True
        self._write(self.settings_path, settings)
        before = self.settings_path.read_bytes()
        status, _, err = self._verify()
        self.assertEqual((status, err), (0, ""))
        self.assertEqual(self.settings_path.read_bytes(), before)

    def test_if_and_args_handler_fields_fail(self):
        for event, field, value, expected in (
            ("PreToolUse", "if", "Bash(git *)", "runs it only for matching tool calls"),
            ("Stop", "if", "Bash(*)", "so this hook never runs"),
            ("SessionStart", "args", [], 'has "args"'),
        ):
            with self.subTest(event=event, field=field):
                settings = self._hooks()
                settings["hooks"][event][0]["hooks"][0][field] = value
                problems = self._problems(settings)
                self.assertIn(f"{event}: 'agent-strace hook ", problems)
                self.assertIn(expected, problems)

    def test_mixed_redaction_modes_fail(self):
        settings = self._hooks()
        settings["hooks"]["Stop"] = self._hooks(redact=True)["hooks"]["Stop"]
        self.assertIn(
            "Claude Code AgentTrace hooks use inconsistent redaction modes: default, redact.",
            self._problems(settings),
        )

    def test_disable_all_hooks_follows_settings_precedence(self):
        local_fix = f"set disableAllHooks to false in {self.local_path} for this project only"
        for user, project, local, expected in (
            (True, None, None, f"Claude Code hooks are disabled in {self.settings_path}: disableAllHooks = true."),
            (None, True, None, local_fix),
            (None, None, True, f"Claude Code hooks are disabled in {self.local_path}"),
            (True, False, None, "turn them back on only in this project, so other projects run no hooks"),
            (False, True, False, ""),
            (False, False, None, ""),
            ("yes", None, None, f"Invalid Claude Code user settings {self.settings_path}: disableAllHooks must be true or false."),
            (None, "null", None, f"Invalid Claude Code project settings {self.project_path}: disableAllHooks"),
        ):
            with self.subTest(user=user, project=project, local=local):
                for path, value in ((self.project_path, project), (self.local_path, local)):
                    path.unlink(missing_ok=True)
                    if value is not None:
                        self._write(path, '{"disableAllHooks": null}' if value == "null" else {"disableAllHooks": value})
                extra = {} if user is None else {"disableAllHooks": user}
                problems = self._problems({**self._hooks(), **extra})
                if expected:
                    self.assertIn(expected, problems)
                else:
                    self.assertEqual(problems, "")

    def test_overridden_disable_all_hooks_is_reported_in_the_summary(self):
        self._install()
        self._write(self.project_path, {"disableAllHooks": True})
        self._write(self.local_path, {"disableAllHooks": False})
        status, out, _ = self._verify()
        self.assertEqual(status, 0)
        self.assertIn(
            f"project:    {self.project_path} (found; disableAllHooks = true, overridden by local settings)",
            out,
        )
        self.assertIn(f"local:      {self.local_path} (found; disableAllHooks = false)", out)

    def test_managed_disable_all_hooks_wins_and_invalid_values_are_ignored(self):
        for managed, user, expected in (
            (True, None, f"Claude Code hooks are disabled by managed settings {self.managed_path}"),
            (False, True, ""),
            ("true", None, ""),
            (None, None, ""),
        ):
            with self.subTest(managed=managed, user=user):
                self._write(self.managed_path, {"disableAllHooks": managed})
                extra = {} if user is None else {"disableAllHooks": user}
                problems = self._problems({**self._hooks(), **extra})
                if expected:
                    self.assertIn(expected, problems)
                    self.assertIn("Ask your administrator", problems)
                else:
                    self.assertEqual(problems, "")
        self._write(self.managed_path, {"disableAllHooks": "true"})
        status, out, _ = self._verify()
        self.assertEqual(status, 0)
        self.assertIn('disableAllHooks = "true", ignored because it is not true or false', out)

    def test_allow_managed_hooks_only_blocks_user_hooks_and_fails_closed(self):
        for value, expected in (
            (True, "set allowManagedHooksOnly to true. Only managed hooks run"),
            ("true", 'set allowManagedHooksOnly to "true". Only managed hooks run'),
            (1, "set allowManagedHooksOnly to 1, which Claude Code reads as true because it is invalid"),
            ([], "set allowManagedHooksOnly to [], which Claude Code reads as true because it is invalid"),
            (False, ""),
            ("false", ""),
            (None, ""),
        ):
            with self.subTest(value=value):
                self._write(self.managed_path, {"allowManagedHooksOnly": value})
                problems = self._problems(self._hooks())
                if expected:
                    self.assertIn(f"Claude Code managed settings {self.managed_path} {expected}", problems)
                    self.assertIn(f"the AgentTrace hooks in {self.settings_path} are blocked", problems)
                else:
                    self.assertEqual(problems, "")

    def test_managed_drop_ins_merge_in_name_order(self):
        self._install()
        drop_ins = self.managed / "managed-settings.d"
        self._write(self.managed_path, {"allowManagedHooksOnly": True})
        self._write(drop_ins / "20-hooks.json", {"allowManagedHooksOnly": False})
        self._write(drop_ins / ".hidden.json", {"allowManagedHooksOnly": True})
        self._write(drop_ins / "30-notes.txt", {"allowManagedHooksOnly": True})
        self._write(drop_ins / "10-base.json", {"allowManagedHooksOnly": True})
        status, out, _ = self._verify()
        self.assertEqual(status, 0)
        self.assertIn(f"managed:    {self.managed} (3 files read; allowManagedHooksOnly = false)", out)

        self._write(drop_ins / "20-hooks.json", {"allowManagedHooksOnly": None})
        self.assertIn(f"Claude Code managed settings {drop_ins / '10-base.json'} set", self._problems())
        self._write(drop_ins / "05-first.json", {"allowManagedHooksOnly": False})
        self._write(self.managed_path, "")
        self.assertIn(str(drop_ins / "10-base.json"), self._problems())

    def test_unparseable_or_unreadable_managed_settings(self):
        self._install()
        self._write(self.managed_path, "{")
        problems = self._problems()
        self.assertIn(f"Cannot read valid Claude Code managed settings JSON at {self.managed_path}", problems)
        self.assertIn("Claude Code refuses to start until it is fixed.", problems)
        self._write(self.managed_path, "[1]")
        self.assertIn("the top level must be a JSON object", self._problems())

        self._write(self.managed_path, {"allowManagedHooksOnly": True})
        real_read_text = Path.read_text

        def read_text(path, *args, **kwargs):
            if path == self.managed_path:
                raise PermissionError(13, "Permission denied", str(path))
            return real_read_text(path, *args, **kwargs)

        with patch.object(Path, "read_text", autospec=True, side_effect=read_text):
            status, out, err = self._verify()
        self.assertEqual((status, err), (0, ""))
        self.assertIn("(1 not readable, so Claude Code skips it)", out)

    def test_unreadable_user_settings_fail(self):
        self._install()
        real_read_text = Path.read_text

        def read_text(path, *args, **kwargs):
            if path == self.settings_path:
                raise PermissionError(13, "Permission denied", str(path))
            return real_read_text(path, *args, **kwargs)

        with patch.object(Path, "read_text", autospec=True, side_effect=read_text):
            self.assertIn(f"Cannot read Claude Code user settings {self.settings_path}", self._problems())

    def test_exact_copies_in_other_settings_files_pass_and_different_ones_fail(self):
        self._install()
        self._write(self.project_path, self._hooks())
        status, out, _ = self._verify()
        self.assertEqual(status, 0)
        self.assertIn(f"project:    {self.project_path} (found; 7 AgentTrace hooks identical to user settings)", out)

        self._write(self.local_path, self._hooks(redact=True))
        problems = self._problems()
        self.assertIn(
            f"Claude Code local settings {self.local_path} have AgentTrace hooks for PostToolUse, "
            "PostToolUseFailure, PreToolUse, SessionEnd, SessionStart, Stop, UserPromptSubmit "
            "that are not exact copies of the user settings hooks.",
            problems,
        )
        self.assertIn("Remove them, or make them identical to the user settings hooks.", problems)
        self.local_path.unlink()

        changed = self._hooks()
        changed["hooks"]["PreToolUse"][0]["hooks"][0]["timeout"] = 5
        self._write(self.project_path, changed)
        self.assertIn("have AgentTrace hooks for PreToolUse that are not exact copies", self._problems())
        self._write(self.project_path, "{")
        self.assertIn(f"Cannot read valid Claude Code project settings JSON at {self.project_path}", self._problems())
        self.project_path.unlink()

        self._write(self.managed_path, self._hooks(no_redact=True))
        self.assertIn("Ask your administrator to remove them", self._problems())
        self._write(self.managed_path, self._hooks())
        status, out, _ = self._verify()
        self.assertEqual(status, 0)
        self.assertIn("(1 file read; 7 AgentTrace hooks identical to user settings)", out)

    def test_project_only_hooks_fail_and_point_to_user_settings(self):
        self._write(self.settings_path, {})
        self._write(self.project_path, self._hooks())
        problems = self._problems()
        self.assertIn(f"No AgentTrace hooks in Claude Code user settings {self.settings_path}", problems)
        self.assertIn(f"Claude Code project settings {self.project_path} have AgentTrace hooks", problems)

    def test_project_settings_that_are_the_user_settings_are_not_read_twice(self):
        user_dir = self.project / ".claude"
        user_dir.mkdir()
        self._write(user_dir / "settings.json", {**self._hooks(), "disableAllHooks": False})
        with patch.dict(os.environ, {"CLAUDE_CONFIG_DIR": str(user_dir)}):
            status, out, _ = self._verify()
        self.assertEqual(status, 0)
        self.assertIn(f"settings:   {user_dir / 'settings.json'} (7/7 events", out)
        self.assertIn(f"project:    {self.project_path} (same file as user settings)", out)

    def test_claude_config_dir_defaults_to_home_and_honors_override(self):
        home = self.root / "home"
        with patch.dict(os.environ, {"HOME": str(home), "USERPROFILE": str(home), "CLAUDE_CONFIG_DIR": ""}):
            self.assertEqual(_claude_config_dir(), home / ".claude")
            os.environ.pop("CLAUDE_CONFIG_DIR")
            self.assertEqual(_claude_config_dir(), home / ".claude")
        self.assertEqual(_claude_config_dir(), self.config_dir)

    def test_managed_settings_dir_follows_the_platform(self):
        for platform, expected in (
            ("darwin", Path("/Library/Application Support/ClaudeCode")),
            ("win32", Path(r"C:\Program Files\ClaudeCode")),
            ("linux", Path("/etc/claude-code")),
        ):
            with self.subTest(platform=platform), patch("agent_trace.cli.sys.platform", platform):
                self.assertEqual(_claude_managed_settings_dir(), expected)

    def test_missing_executable_and_mismatched_version_fail(self):
        self._install()
        with patch("agent_trace.setup_verify.shutil.which", return_value=None):
            self.assertIn(
                "The agent-strace executable is not on PATH. Install this AgentTrace release "
                "into the same environment used by Claude Code.",
                self._problems(),
            )
        with patch(
            "agent_trace.setup_verify.subprocess.run",
            return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout="agent-strace 0.1.0\n", stderr=""),
        ):
            self.assertIn("AgentTrace CLI version mismatch", self._problems())

    def test_entry_point_records_claude_verify_in_telemetry(self):
        with patch.object(
            __import__("sys"), "argv", ["agent-strace", "setup", "--cli", "claude", "--verify"],
        ), patch("agent_trace.cli._product_telemetry.capture") as capture, patch(
            "agent_trace.cli._product_telemetry.maybe_show_telemetry_notice"
        ):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as result:
                    main()
        self.assertEqual(result.exception.code, 1)  # no settings
        props = capture.call_args.args[1]
        self.assertEqual((props["command"], props["integration"], props["subcommand"]), ("setup", "claude", "verify"))
        self.assertEqual(props["exit_code"], 1)
        self.assertEqual(self._snapshot(), {})

    def test_cli_parser_accepts_verify_with_claude(self):
        args = build_parser().parse_args(["setup", "--cli", "claude", "--verify"])
        self.assertEqual((args.cli, args.verify), ("claude", True))

    def test_other_providers_and_all_are_rejected_without_writes(self):
        env = {name: str(self.root / name) for name in ("CURSOR_CONFIG_DIR", "COPILOT_HOME", "GEMINI_CONFIG_DIR", "CODEX_HOME")}
        with patch.dict(os.environ, env):
            for cli in ("all", "cursor", "copilot", "gemini"):
                with self.subTest(cli=cli):
                    err = io.StringIO()
                    with contextlib.redirect_stderr(err):
                        self.assertEqual(cmd_setup(_args(cli=cli)), 2)
                    self.assertIn("supports only --cli codex or --cli claude, one provider per run", err.getvalue())
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["config", "managed", "project"])


if __name__ == "__main__":
    unittest.main()
