"""Offline Codex setup verification tests; no actual provider configuration touched."""

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

from agent_trace import __version__
from agent_trace.cli import _codex_hooks_config, build_parser, cmd_setup, main
from agent_trace.setup_verify import verify_codex_setup


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
            "CODEX_CONFIG_DIR": self.tmp.name,
            "GEMINI_CONFIG_DIR": self.gemini_tmp.name,
        })
        self.env.start()
        self.addCleanup(self.env.stop)

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
        self.assertEqual(list(Path(self.tmp.name).iterdir()), [])

    def test_entry_point_skips_product_telemetry_in_read_only_verify_mode(self):
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
        self.assertEqual(list(Path(self.gemini_tmp.name).iterdir()), [])

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
                for event in ("PreToolUse", "PostToolUse"):
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
                self.assertEqual(_codex_config_dir(), Path(self.tmp.name))
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

    def test_cli_parser_accepts_verify_with_codex(self):
        parsed = build_parser().parse_args(["setup", "--cli", "codex", "--verify"])
        self.assertEqual(parsed.cli, "codex")
        self.assertTrue(parsed.verify)


if __name__ == "__main__":
    unittest.main()
