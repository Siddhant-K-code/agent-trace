"""Feature landing pages keep local links and CLI examples valid."""

from __future__ import annotations

import argparse
import re
import shlex
import unittest
from pathlib import Path

from agent_trace.cli import _normalise_skills_argv, build_parser


ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
LANDING_DOCS = (
    "index.md",
    "claude-code-session-replay.md",
    "codex-trace-viewer.md",
    "github-copilot-cli-observability.md",
    "mcp-trace-recording.md",
    "ai-agent-pr-evidence.md",
    "agent-instruction-compliance.md",
    "cognitive-debt-review-coverage.md",
    "releases.md",
)
LOCAL_LINK = re.compile(r"\[[^\]]+\]\((?!https?://|mailto:)([^)#]+)(?:#[^)]+)?\)")


def _subcommands(parser: argparse.ArgumentParser) -> set[str]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices)
    return set()


def _agent_strace_commands(text: str):
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        command = lines[index].strip()
        lineno = index + 1
        index += 1
        if not command.startswith("agent-strace "):
            continue
        while command.endswith("\\") and index < len(lines):
            command = command[:-1].rstrip() + " " + lines[index].strip()
            index += 1
        yield lineno, command


class FeatureLandingDocsTest(unittest.TestCase):
    def test_local_links_resolve(self):
        problems = []
        for name in LANDING_DOCS:
            path = DOCS / name
            text = path.read_text(encoding="utf-8")
            for target in LOCAL_LINK.findall(text):
                resolved = (path.parent / target).resolve()
                if not resolved.exists():
                    problems.append(f"{name}: {target}")
        self.assertEqual(problems, [])

    def test_agent_strace_examples_parse(self):
        parser = build_parser()
        commands = _subcommands(parser)
        problems = []
        checked = 0
        for name in LANDING_DOCS:
            text = (DOCS / name).read_text(encoding="utf-8")
            for lineno, command in _agent_strace_commands(text):
                checked += 1
                argv = _normalise_skills_argv(shlex.split(command)[1:])
                if argv == ["--version"]:
                    continue
                if not argv or argv[0] not in commands:
                    problems.append(f"{name}:{lineno}: unknown command")
                    continue
                try:
                    parser.parse_args(argv)
                except SystemExit:
                    problems.append(f"{name}:{lineno}: {command}")
        self.assertGreaterEqual(checked, 25)
        self.assertEqual(problems, [])

    def test_landing_pages_use_no_em_dash(self):
        problems = [
            name
            for name in LANDING_DOCS
            if "\N{EM DASH}" in (DOCS / name).read_text(encoding="utf-8")
        ]
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
