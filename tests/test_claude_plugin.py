"""Tests for the native Claude Code plugin package."""

import argparse
import json
import unittest
from pathlib import Path

from agent_trace import __version__
from agent_trace.cli import _claude_hooks_config


ROOT = Path(__file__).resolve().parents[1]
CLAUDE_PLUGIN_DIR = ROOT / ".claude-plugin"
HOOKS_PATH = ROOT / "hooks" / "hooks.json"


def _args(*, redact=False, no_redact=False):
    return argparse.Namespace(redact=redact, no_redact=no_redact)


def _commands(config):
    return {
        event: group["hooks"][0]["command"]
        for event, groups in config["hooks"].items()
        for group in groups
    }


class TestClaudePlugin(unittest.TestCase):
    def test_manifest_has_supported_metadata_and_reuses_components(self):
        manifest = json.loads((CLAUDE_PLUGIN_DIR / "plugin.json").read_text())

        self.assertEqual(manifest["name"], "agent-strace")
        self.assertEqual(manifest["displayName"], "AgentTrace")
        self.assertEqual(manifest["version"], __version__)
        self.assertEqual(manifest["license"], "MIT")
        self.assertTrue(manifest["homepage"].startswith("https://"))
        self.assertTrue(manifest["repository"].startswith("https://"))

        self.assertTrue((ROOT / "skills" / "agent-strace" / "SKILL.md").is_file())
        agent = (ROOT / "agents" / "trace-analyst.agent.md").read_text()
        self.assertTrue(agent.startswith("---\nname: trace-analyst\n"))
        self.assertTrue(HOOKS_PATH.is_file())

    def test_marketplace_references_repository_root_plugin(self):
        manifest = json.loads((CLAUDE_PLUGIN_DIR / "plugin.json").read_text())
        marketplace = json.loads(
            (CLAUDE_PLUGIN_DIR / "marketplace.json").read_text()
        )

        self.assertEqual(marketplace["name"], "agent-trace")
        self.assertEqual(len(marketplace["plugins"]), 1)
        entry = marketplace["plugins"][0]
        self.assertEqual(entry["name"], manifest["name"])
        self.assertEqual(entry["version"], __version__)
        self.assertEqual(entry["source"], "./")
        self.assertEqual((ROOT / entry["source"]).resolve(), ROOT.resolve())

    def test_all_plugin_versions_stay_synchronized(self):
        copilot_manifest = json.loads((ROOT / "plugin.json").read_text())
        copilot_marketplace = json.loads(
            (ROOT / ".github" / "plugin" / "marketplace.json").read_text()
        )
        claude_manifest = json.loads(
            (CLAUDE_PLUGIN_DIR / "plugin.json").read_text()
        )
        claude_marketplace = json.loads(
            (CLAUDE_PLUGIN_DIR / "marketplace.json").read_text()
        )

        versions = {
            __version__,
            copilot_manifest["version"],
            copilot_marketplace["metadata"]["version"],
            copilot_marketplace["plugins"][0]["version"],
            claude_manifest["version"],
            claude_marketplace["plugins"][0]["version"],
        }
        self.assertEqual(versions, {__version__})

    def test_hooks_match_default_manual_claude_setup(self):
        plugin_hooks = json.loads(HOOKS_PATH.read_text())

        self.assertEqual(plugin_hooks, _claude_hooks_config(_args()))
        self.assertEqual(
            set(plugin_hooks["hooks"]),
            {
                "UserPromptSubmit",
                "PreToolUse",
                "PostToolUse",
                "PostToolUseFailure",
                "Stop",
                "SessionStart",
                "SessionEnd",
            },
        )

    def test_hook_commands_use_supported_events_and_safe_redaction_default(self):
        commands = _commands(json.loads(HOOKS_PATH.read_text()))

        self.assertEqual(
            commands,
            {
                "UserPromptSubmit": "agent-strace hook user-prompt",
                "PreToolUse": "agent-strace hook pre-tool",
                "PostToolUse": "agent-strace hook post-tool",
                "PostToolUseFailure": "agent-strace hook post-tool-failure",
                "Stop": "agent-strace hook stop",
                "SessionStart": "agent-strace hook session-start",
                "SessionEnd": "agent-strace hook session-end",
            },
        )
        for command in commands.values():
            self.assertNotIn("--provider", command)
            self.assertNotIn("NO_REDACT", command)
            self.assertNotIn("--no-redact", command)

    def test_manual_redaction_modes_only_prefix_default_commands(self):
        default = _commands(_claude_hooks_config(_args()))
        redact = _commands(_claude_hooks_config(_args(redact=True)))
        no_redact = _commands(_claude_hooks_config(_args(no_redact=True)))

        for event, command in default.items():
            self.assertEqual(redact[event], f"AGENT_TRACE_REDACT=1 {command}")
            self.assertEqual(no_redact[event], f"AGENT_TRACE_NO_REDACT=1 {command}")

    def test_plugin_documentation_covers_lifecycle_and_privacy(self):
        guide = (ROOT / "docs" / "claude-code-plugin.md").read_text()

        for heading in (
            "## Install",
            "## Verify",
            "## Update",
            "## Uninstall",
            "## Duplicate hook avoidance",
            "## Privacy boundary",
            "## Known provider gaps",
            "## Release and publication",
        ):
            self.assertIn(heading, guide)
        self.assertNotIn("\N{EM DASH}", guide)


if __name__ == "__main__":
    unittest.main()
