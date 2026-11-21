"""Tests for the Codex plugin package and its repo marketplace."""

import argparse
import json
import re
import unittest
from pathlib import Path
from urllib.parse import urlparse

from agent_trace import __version__
from agent_trace.cli import _codex_hooks_config


ROOT = Path(__file__).resolve().parents[1]
# Kept out of the repo root so the package stays under third-party scan
# limits, and out of plugins/agent-strace so Copilot never loads Codex hooks.
CODEX_PLUGIN_DIR = ROOT / "plugins" / "agent-strace-codex"
MANIFEST_PATH = CODEX_PLUGIN_DIR / ".codex-plugin" / "plugin.json"
MARKETPLACE_PATH = ROOT / ".agents" / "plugins" / "marketplace.json"

KEBAB = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
SEMVER = re.compile(r"^\d+\.\d+\.\d+$")
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
# codex-marketplace.com drops any other interface.category and guesses one.
MARKETPLACE_CATEGORIES = {
    "Coding", "Development", "Design", "Productivity", "Communication",
    "DevOps", "Data", "Finance", "Writing", "AI & Agents", "Utilities",
}
CODEX_CAPABILITIES = {"Interactive", "Read", "Write"}
INSTALLATION_POLICIES = {"AVAILABLE", "INSTALLED_BY_DEFAULT", "NOT_AVAILABLE"}
AUTHENTICATION_POLICIES = {"ON_INSTALL", "ON_USE", "ON_FIRST_USE"}
SCAN_FILE_LIMIT = 128


def _args():
    return argparse.Namespace(redact=False, no_redact=False)


def _files(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def _manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _assert_https_url(case, value, field):
    parsed = urlparse(value)
    case.assertEqual(parsed.scheme, "https", field)
    case.assertTrue(parsed.netloc, field)


def _inside(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


class TestCodexPluginManifest(unittest.TestCase):
    def test_manifest_has_required_marketplace_fields(self):
        manifest = _manifest()

        self.assertRegex(manifest["name"], KEBAB)
        self.assertEqual(manifest["name"], "agent-strace")
        self.assertRegex(manifest["version"], SEMVER)
        self.assertEqual(manifest["version"], __version__)
        self.assertIsInstance(manifest["description"], str)
        self.assertIn("Codex", manifest["description"])

        _assert_https_url(self, manifest["homepage"], "homepage")
        _assert_https_url(self, manifest["author"]["url"], "author.url")
        self.assertRegex(manifest["author"]["email"], EMAIL)

        claude = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text())
        for field in ("author", "homepage", "repository", "license"):
            self.assertEqual(manifest[field], claude[field], field)

    def test_component_paths_are_strings_inside_the_package(self):
        manifest = _manifest()

        self.assertEqual(manifest["skills"], "./skills/")
        self.assertEqual(manifest["hooks"], "./hooks/hooks.json")
        for field in ("skills", "hooks"):
            value = manifest[field]
            self.assertIsInstance(value, str, field)
            self.assertTrue(value.startswith("./"), field)
            target = CODEX_PLUGIN_DIR / value
            self.assertTrue(target.exists(), field)
            self.assertTrue(_inside(target, CODEX_PLUGIN_DIR), field)
        # A plugin MCP server would start in the plugin cache and read the
        # wrong .agent-traces/, so no plugin registers one.
        self.assertNotIn("mcpServers", manifest)
        self.assertNotIn("apps", manifest)

    def test_interface_listing_is_valid_and_has_no_images(self):
        listing = _manifest()["interface"]

        self.assertEqual(listing["displayName"], "AgentTrace")
        self.assertLessEqual(len(listing["shortDescription"]), 125)
        self.assertTrue(listing["longDescription"])
        self.assertIn(listing["category"], MARKETPLACE_CATEGORIES)
        self.assertLessEqual(set(listing["capabilities"]), CODEX_CAPABILITIES)
        _assert_https_url(self, listing["websiteURL"], "interface.websiteURL")
        self.assertIsInstance(listing["defaultPrompt"], list)
        self.assertLessEqual(len(listing["defaultPrompt"]), 3)
        for field in ("composerIcon", "logo", "screenshots"):
            self.assertNotIn(field, listing)


class TestCodexPluginComponents(unittest.TestCase):
    def test_hooks_match_default_manual_codex_setup(self):
        hooks = json.loads((CODEX_PLUGIN_DIR / "hooks" / "hooks.json").read_text())

        self.assertEqual(hooks, _codex_hooks_config(_args()))
        commands = [
            handler["command"]
            for groups in hooks["hooks"].values()
            for group in groups
            for handler in group["hooks"]
        ]
        self.assertTrue(commands)
        for command in commands:
            self.assertTrue(command.startswith("agent-strace hook --provider codex "), command)

    def test_skill_matches_shared_copy(self):
        shared = _files(ROOT / "skills")

        self.assertEqual(_files(CODEX_PLUGIN_DIR / "skills"), shared)
        for rel in shared:
            self.assertEqual(
                (CODEX_PLUGIN_DIR / "skills" / rel).read_bytes(),
                (ROOT / "skills" / rel).read_bytes(), rel,
            )

    def test_package_is_small_and_text_only(self):
        files = _files(CODEX_PLUGIN_DIR)

        self.assertEqual(
            files,
            sorted([
                ".codex-plugin/plugin.json",
                "hooks/hooks.json",
                *(f"skills/{f}" for f in _files(ROOT / "skills")),
            ]),
        )
        self.assertLess(len(files), SCAN_FILE_LIMIT)
        for rel in files:
            data = (CODEX_PLUGIN_DIR / rel).read_bytes()
            self.assertNotIn(b"\0", data, rel)
            data.decode("utf-8")


class TestCodexRepoMarketplace(unittest.TestCase):
    def test_marketplace_entry_points_at_codex_package(self):
        marketplace = json.loads(MARKETPLACE_PATH.read_text(encoding="utf-8"))

        self.assertEqual(marketplace["name"], "agent-trace")
        self.assertTrue(marketplace["interface"]["displayName"])
        self.assertEqual(len(marketplace["plugins"]), 1)
        entry = marketplace["plugins"][0]
        self.assertEqual(entry["name"], _manifest()["name"])
        self.assertEqual(
            entry["source"], {"source": "local", "path": "./plugins/agent-strace-codex"}
        )
        self.assertIn(entry["policy"]["installation"], INSTALLATION_POLICIES)
        self.assertIn(entry["policy"]["authentication"], AUTHENTICATION_POLICIES)
        self.assertIn(entry["category"], MARKETPLACE_CATEGORIES)

        source = ROOT / entry["source"]["path"]
        self.assertTrue(_inside(source, ROOT))
        self.assertEqual(source.resolve(), CODEX_PLUGIN_DIR.resolve())

    def test_plural_install_resolves_only_codex_packages(self):
        # `npx codex-marketplace add owner/repo --plugins` uses the entries
        # whose path starts with plugins/. Without this file it loads every
        # plugins/* directory and fails on the Copilot package, which has no
        # .codex-plugin/plugin.json.
        marketplace = json.loads(MARKETPLACE_PATH.read_text(encoding="utf-8"))

        resolved = []
        for entry in marketplace["plugins"]:
            path = entry["source"]["path"].removeprefix("./")
            if not path.startswith("plugins/"):
                continue
            manifest = json.loads((ROOT / path / ".codex-plugin" / "plugin.json").read_text())
            self.assertEqual(manifest["name"], entry["name"])
            resolved.append((ROOT / path).resolve())
        self.assertEqual(resolved, [CODEX_PLUGIN_DIR.resolve()])


if __name__ == "__main__":
    unittest.main()
