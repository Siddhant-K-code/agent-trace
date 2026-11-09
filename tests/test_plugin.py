"""Tests for the GitHub Copilot CLI plugin packages and registry metadata."""

import argparse
import json
import unittest
from pathlib import Path

from agent_trace import __version__
from agent_trace.cli import _claude_hooks_config, _copilot_hooks_config


ROOT = Path(__file__).resolve().parents[1]
# Agent Plugins 1.0 package for marketplaces that require that layout
# (github/awesome-copilot). The repo root stays a legacy Copilot plugin
# and a Claude Code plugin; the two cannot share one root because both
# clients would probe hooks/hooks.json.
AGENT_PLUGIN_DIR = ROOT / "plugins" / "agent-strace"
AGENT_PLUGINS_FIELDS = {
    "$schema", "name", "version", "description", "author", "homepage",
    "repository", "license", "keywords", "extensions",
}
# Copilot CLI probes these manifests in order and uses the first it finds.
COPILOT_LEGACY_MANIFESTS = (
    ".plugin/plugin.json", "plugin.json", ".github/plugin/plugin.json",
    ".claude-plugin/plugin.json",
)


def _hook_args():
    return argparse.Namespace(redact=False, no_redact=False)


def _files(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


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

    def test_plugin_hooks_match_generated_copilot_hooks(self):
        plugin_hooks = json.loads((ROOT / "hooks.json").read_text())

        self.assertEqual(plugin_hooks, _copilot_hooks_config(_hook_args()))

    def test_agent_and_skill_have_required_frontmatter(self):
        agent = (ROOT / "agents" / "trace-analyst.agent.md").read_text()
        skill = (ROOT / "skills" / "agent-strace" / "SKILL.md").read_text()

        self.assertTrue(agent.startswith("---\nname: trace-analyst\n"))
        self.assertIn("\ndescription:", agent)
        self.assertTrue(skill.startswith("---\nname: agent-strace\n"))
        self.assertIn("\ndescription:", skill)


class TestSharedRootLayout(unittest.TestCase):
    """Copilot CLI and Claude Code both load the repo root as a plugin."""

    def test_copilot_resolves_root_manifest_with_copilot_hooks(self):
        found = [p for p in COPILOT_LEGACY_MANIFESTS if (ROOT / p).is_file()]
        self.assertEqual(found[0], "plugin.json")
        manifest = json.loads((ROOT / found[0]).read_text())

        self.assertNotIn("$schema", manifest)
        # An explicit hooks path stops Copilot from falling back to the
        # Claude-format hooks/hooks.json.
        self.assertEqual(manifest["hooks"], "hooks.json")
        self.assertEqual(
            json.loads((ROOT / manifest["hooks"]).read_text()),
            _copilot_hooks_config(_hook_args()),
        )

    def test_claude_resolves_default_hooks_with_claude_hooks(self):
        manifest = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text())

        self.assertNotIn("hooks", manifest)
        self.assertEqual(
            json.loads((ROOT / "hooks" / "hooks.json").read_text()),
            _claude_hooks_config(_hook_args()),
        )

    def test_root_does_not_register_other_client_components(self):
        # Root MCP configs would start agent-strace mcp in the plugin cache,
        # not the project. A root gemini-extension.json would make Gemini CLI
        # load the Claude-format hooks/hooks.json.
        for name in ("mcp.json", ".mcp.json", "com.github.copilot", "gemini-extension.json"):
            self.assertFalse((ROOT / name).exists(), name)


class TestAgentPluginsPackage(unittest.TestCase):
    def test_manifest_is_agent_plugins_1_0(self):
        manifest = json.loads((AGENT_PLUGIN_DIR / "plugin.json").read_text())
        root = json.loads((ROOT / "plugin.json").read_text())

        self.assertEqual(
            manifest["$schema"],
            "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json",
        )
        self.assertLessEqual(set(manifest), AGENT_PLUGINS_FIELDS)
        self.assertEqual(manifest["version"], __version__)
        for field in ("name", "description", "author", "homepage", "repository", "license", "keywords"):
            self.assertEqual(manifest[field], root[field], field)

    def test_package_contains_only_copilot_components(self):
        skill_files = [f"skills/{f}" for f in _files(ROOT / "skills")]
        agent_files = [f"com.github.copilot/agents/{f}" for f in _files(ROOT / "agents")]
        self.assertEqual(
            _files(AGENT_PLUGIN_DIR),
            sorted(["plugin.json", "com.github.copilot/hooks/hooks.json", *skill_files, *agent_files]),
        )

    def test_components_match_root_copies(self):
        for rel in _files(ROOT / "skills"):
            self.assertEqual(
                (AGENT_PLUGIN_DIR / "skills" / rel).read_bytes(),
                (ROOT / "skills" / rel).read_bytes(), rel,
            )
        for rel in _files(ROOT / "agents"):
            self.assertEqual(
                (AGENT_PLUGIN_DIR / "com.github.copilot" / "agents" / rel).read_bytes(),
                (ROOT / "agents" / rel).read_bytes(), rel,
            )
        self.assertEqual(
            json.loads((AGENT_PLUGIN_DIR / "com.github.copilot" / "hooks" / "hooks.json").read_text()),
            _copilot_hooks_config(_hook_args()),
        )


class TestMcpRegistryMetadata(unittest.TestCase):
    def test_server_json_targets_package(self):
        server = json.loads((ROOT / "server.json").read_text())
        name = "io.github.Siddhant-K-code/agent-trace"

        self.assertEqual(server["name"], name)
        self.assertLessEqual(len(server["description"]), 100)
        self.assertEqual(len(server["packages"]), 1)
        package = server["packages"][0]
        self.assertEqual(package["registryType"], "pypi")
        self.assertEqual(package["identifier"], "agent-strace")
        self.assertEqual(package["packageArguments"], [{"type": "positional", "value": "mcp"}])
        # PyPI ownership marker the MCP Registry checks in the package README.
        self.assertIn(f"mcp-name: {name}", (ROOT / "README.md").read_text())

    def test_server_json_version_is_not_an_older_release(self):
        # server.json names the first release that ships these MCP tools and
        # the ownership marker. It may lead __version__ until that release,
        # which then syncs both values (see AGENTS.md).
        server = json.loads((ROOT / "server.json").read_text())
        version = server["version"]

        self.assertEqual(server["packages"][0]["version"], version)
        as_tuple = lambda v: tuple(int(x) for x in v.split("."))
        self.assertGreaterEqual(as_tuple(version), as_tuple(__version__))


if __name__ == "__main__":
    unittest.main()
