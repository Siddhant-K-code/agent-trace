"""Static checks for release and repository trust configuration."""

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


class TestReleaseWorkflows(unittest.TestCase):
    def test_core_release_dispatches_only_pypi_at_created_tag(self):
        workflow = (WORKFLOWS / "release.yml").read_text()

        self.assertIn("Verify release version parity", workflow)
        self.assertIn("gh workflow run publish.yml", workflow)
        self.assertNotIn("gh workflow run publish-vscode.yml", workflow)
        self.assertEqual(workflow.count('--ref "$TAG"'), 1)
        self.assertEqual(workflow.count('-f tag="$TAG"'), 1)
        self.assertNotIn("vscode-extension", workflow)
        self.assertIn('--exclude="vscode-v*"', workflow)
        self.assertIn("plugins/agent-strace/plugin.json", workflow)
        self.assertIn(".claude-plugin/plugin.json", workflow)
        self.assertIn(".claude-plugin/marketplace.json", workflow)
        self.assertIn("server.json PyPI package", workflow)
        self.assertIn("generated Gemini extension", workflow)
        self.assertIn("agent-strace-interoperability-v1.zip", workflow)
        self.assertIn("src/agent_trace/corpus/v1", workflow)

    def test_publish_workflows_require_tag_and_checkout_that_tag(self):
        for name in ("publish.yml", "publish-vscode.yml"):
            with self.subTest(workflow=name):
                workflow = (WORKFLOWS / name).read_text()
                self.assertIn("workflow_dispatch:", workflow)
                self.assertIn("required: true", workflow)
                self.assertIn("PUBLISH_TAG:", workflow)
                self.assertIn("ref: ${{ env.PUBLISH_TAG }}", workflow)
                self.assertIn("Verify publish target", workflow)

    def test_pypi_publish_ignores_extension_releases(self):
        workflow = (WORKFLOWS / "publish.yml").read_text()

        self.assertIn(
            "!startsWith(github.event.release.tag_name, 'vscode-v')",
            workflow,
        )

    def test_mcp_registry_publish_follows_pypi_with_oidc(self):
        workflow = (WORKFLOWS / "publish.yml").read_text()
        header, _, jobs = workflow.partition("\njobs:\n")
        pypi_job, _, registry_job = jobs.partition("\n  mcp-registry:\n")

        self.assertNotIn("id-token", header)
        self.assertIn("id-token: write", pypi_job)
        self.assertTrue(registry_job)
        self.assertIn("needs: publish", registry_job)
        self.assertIn("id-token: write", registry_job)
        self.assertNotIn("contents: write", registry_job)
        self.assertIn("ref: ${{ env.PUBLISH_TAG }}", registry_job)
        self.assertIn("persist-credentials: false", registry_job)
        self.assertRegex(registry_job, r"MCP_PUBLISHER_VERSION: v\d+\.\d+\.\d+\n")
        self.assertRegex(
            registry_job, r"MCP_PUBLISHER_SHA256_LINUX_AMD64: [0-9a-f]{64}\n"
        )
        self.assertRegex(
            registry_job, r"MCP_PUBLISHER_SHA256_LINUX_ARM64: [0-9a-f]{64}\n"
        )
        self.assertIn("sha256sum -c", registry_job)
        self.assertIn('["packages"][0]["version"]', registry_job)
        self.assertIn("https://pypi.org/pypi/agent-strace/${VERSION}/json", registry_job)
        self.assertIn("seq 1 30", registry_job)
        self.assertIn("registry.modelcontextprotocol.io/v0/servers/", registry_job)
        self.assertNotIn("continue-on-error", registry_job)
        self.assertLess(
            registry_job.index("mcp-publisher login github-oidc"),
            registry_job.index("mcp-publisher publish"),
        )
        for step in (
            "Wait for PyPI to serve this version",
            "Install mcp-publisher",
            "Log in to MCP Registry",
            "Publish to MCP Registry",
        ):
            with self.subTest(step=step):
                block = registry_job.split(f"- name: {step}\n", 1)[1]
                self.assertTrue(
                    block.lstrip().startswith(
                        "if: steps.registry.outputs.published != 'true'"
                    )
                )

    def test_extension_publish_is_deterministic_and_reports_failures(self):
        workflow = (WORKFLOWS / "publish-vscode.yml").read_text()

        self.assertIn('tags:\n      - "vscode-v*"', workflow)
        self.assertNotIn("types: [published]", workflow)
        self.assertIn('test "$PUBLISH_TAG" = "vscode-v$VERSION"', workflow)
        self.assertIn("npm ci", workflow)
        self.assertIn("@vscode/vsce@4.1.0", workflow)
        self.assertIn("ovsx@1.3.0", workflow)
        self.assertEqual(workflow.count("--skip-duplicate"), 2)
        self.assertNotIn("continue-on-error", workflow)
        self.assertNotIn("npm install sharp", workflow)

    def test_extension_manifest_and_lock_versions_match(self):
        manifest = json.loads(
            (ROOT / "vscode-extension" / "package.json").read_text()
        )
        lock = json.loads(
            (ROOT / "vscode-extension" / "package-lock.json").read_text()
        )

        self.assertEqual(manifest["version"], lock["version"])
        self.assertEqual(manifest["version"], lock["packages"][""]["version"])

    def test_scorecard_publishes_public_results_with_limited_permissions(self):
        workflow = (WORKFLOWS / "scorecard.yml").read_text()

        self.assertIn("publish_results: true", workflow)
        self.assertIn("contents: read", workflow)
        self.assertIn("id-token: write", workflow)
        self.assertIn("security-events: write", workflow)
        self.assertIn("persist-credentials: false", workflow)
        self.assertNotIn("contents: write", workflow)

    def test_release_actions_are_pinned_to_commits(self):
        for name in (
            "release.yml",
            "publish.yml",
            "publish-vscode.yml",
            "scorecard.yml",
        ):
            with self.subTest(workflow=name):
                workflow = (WORKFLOWS / name).read_text()
                actions = re.findall(r"^\s*uses:\s+\S+@(\S+)", workflow, re.MULTILINE)
                self.assertTrue(actions)
                for revision in actions:
                    self.assertRegex(revision, r"^[0-9a-f]{40}$")

    def test_repository_trust_files_exist(self):
        paths = (
            "SECURITY.md",
            "CONTRIBUTING.md",
            "CODE_OF_CONDUCT.md",
            ".github/PULL_REQUEST_TEMPLATE.md",
            ".github/ISSUE_TEMPLATE/bug_report.yml",
            ".github/ISSUE_TEMPLATE/feature_request.yml",
            ".github/ISSUE_TEMPLATE/config.yml",
        )

        for path in paths:
            with self.subTest(path=path):
                self.assertTrue((ROOT / path).is_file())


if __name__ == "__main__":
    unittest.main()
