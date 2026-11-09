"""Tests for the GitHub Copilot CLI plugin package."""

import argparse
import json
import unittest
from pathlib import Path

from agent_trace import __version__
from agent_trace.cli import _copilot_hooks_config, _gemini_extension_manifest


ROOT = Path(__file__).resolve().parents[1]


class TestCopilotPlugin(unittest.TestCase):
    def test_manifest_references_existing_components(self):
        manifest = json.loads((ROOT / "plugin.json").read_text())

        self.assertEqual(manifest["name"], "agent-strace")
        self.assertEqual(manifest["version"], __version__)
        for field in ("agents", "skills", "hooks"):
            self.assertTrue((ROOT / manifest[field]).exists())

    def test_marketplace_references_root_plugin(self):
        marketplace = json.loads(
            (ROOT / ".github" / "plugin" / "marketplace.json").read_text()
        )

        self.assertEqual(marketplace["name"], "agent-trace")
        self.assertEqual(marketplace["metadata"]["version"], __version__)
        self.assertEqual(len(marketplace["plugins"]), 1)
        self.assertEqual(marketplace["plugins"][0]["name"], "agent-strace")
        self.assertEqual(marketplace["plugins"][0]["version"], __version__)
        self.assertEqual(marketplace["plugins"][0]["source"], ".")

    def test_gemini_extension_manifest_matches_setup(self):
        manifest = json.loads((ROOT / "gemini-extension.json").read_text())

        self.assertEqual(manifest, _gemini_extension_manifest())
        self.assertEqual(manifest["version"], __version__)
        # Copilot legacy plugins also probe hooks/hooks.json; Gemini hooks are
        # written by `agent-strace setup --cli gemini` instead.
        self.assertFalse((ROOT / "hooks").exists())

    def test_mcp_registry_server_json_matches_package(self):
        server = json.loads((ROOT / "server.json").read_text())
        name = "io.github.Siddhant-K-code/agent-trace"

        self.assertEqual(server["name"], name)
        self.assertLessEqual(len(server["description"]), 100)
        self.assertEqual(server["version"], __version__)
        self.assertEqual(len(server["packages"]), 1)
        package = server["packages"][0]
        self.assertEqual(package["registryType"], "pypi")
        self.assertEqual(package["identifier"], "agent-strace")
        self.assertEqual(package["version"], __version__)
        self.assertEqual(package["packageArguments"], [{"type": "positional", "value": "mcp"}])
        # PyPI ownership marker the MCP Registry checks in the package README.
        self.assertIn(f"mcp-name: {name}", (ROOT / "README.md").read_text())

    def test_plugin_hooks_match_generated_copilot_hooks(self):
        plugin_hooks = json.loads((ROOT / "hooks.json").read_text())
        args = argparse.Namespace(redact=False, no_redact=False)

        self.assertEqual(plugin_hooks, _copilot_hooks_config(args))

    def test_agent_and_skill_have_required_frontmatter(self):
        agent = (ROOT / "agents" / "trace-analyst.agent.md").read_text()
        skill = (ROOT / "skills" / "agent-strace" / "SKILL.md").read_text()

        self.assertTrue(agent.startswith("---\nname: trace-analyst\n"))
        self.assertIn("\ndescription:", agent)
        self.assertTrue(skill.startswith("---\nname: agent-strace\n"))
        self.assertIn("\ndescription:", skill)


if __name__ == "__main__":
    unittest.main()
