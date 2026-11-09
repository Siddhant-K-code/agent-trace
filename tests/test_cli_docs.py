"""CLI docs must only document flags that the CLI parser accepts."""

from __future__ import annotations

import argparse
import ast
import re
import unittest
from pathlib import Path

from agent_trace.cli import (
    _command_index,
    _normalise_eval_argv,
    _normalise_policy_argv,
    _normalise_skills_argv,
    build_parser,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
COMMANDS_MD = REPO_ROOT / "docs" / "commands.md"
OTHER_DOCS = (
    "docs/agents-md-integration.md",
    "docs/security.md",
    "docs/server.md",
)
DOCSTRING_MODULES = ("a2a.py", "iac.py", "mcp_server.py", "standup.py")

_HEADING = re.compile(r"^###\s+`([^`]+)`")
_FLAG = re.compile(r"(?<![\w-])(--?[A-Za-z][\w-]*)")
_CHOICES = re.compile(r"(?<![\w-])(--?[A-Za-z][\w-]*)[ =]([\w.-]+(?:\\?\|[\w.-]+)+)")
_SEPARATOR = re.compile(r"(?:^|\s)--(?:\s|$)")


def _subparsers(parser: argparse.ArgumentParser) -> dict:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices
    return {}


def _parser_tree(root: argparse.ArgumentParser) -> dict[tuple[str, ...], argparse.ArgumentParser]:
    tree = {}

    def walk(parser, path):
        tree[path] = parser
        for name, child in _subparsers(parser).items():
            walk(child, path + (name,))

    walk(root, ())
    return tree


def _resolve(root, tokens: list[str]) -> tuple[str, ...]:
    """Return the deepest subcommand path named by *tokens*."""
    tokens = tokens[_command_index(tokens):]
    path: tuple[str, ...] = ()
    parser = root
    for token in tokens:
        children = _subparsers(parser)
        if token in children:
            path += (token,)
            parser = children[token]
        elif not token.startswith("-"):
            break
    return path


def _usage_tokens(text: str) -> list[str]:
    tokens = text.split()
    return _normalise_eval_argv(_normalise_skills_argv(_normalise_policy_argv(tokens)))


def find_mismatches(markdown: str) -> tuple[list[str], int]:
    """Return (mismatches, flags_checked) for usage blocks and flag tables."""
    root = build_parser()
    tree = _parser_tree(root)
    root_actions = list(root._actions)

    def actions_for(path, subtree):
        actions = list(root_actions)
        for key, parser in tree.items():
            if key == path or (subtree and key[: len(path)] == path):
                actions.extend(parser._actions)
        return actions

    def check(lineno, label, text, actions):
        nonlocal checked
        options = {opt for action in actions for opt in action.option_strings}
        for flag in _FLAG.findall(text):
            checked += 1
            if flag not in options:
                problems.append(f"line {lineno}: {label}: {flag} is not a CLI option")
        for flag, values in _CHOICES.findall(text):
            action = next((a for a in actions if flag in a.option_strings), None)
            if action is None or not action.choices:
                continue
            documented = {v for v in values.replace("\\|", "|").split("|") if v}
            unknown = documented - {str(c) for c in action.choices}
            if unknown:
                problems.append(
                    f"line {lineno}: {label}: {flag} does not accept {sorted(unknown)}"
                )

    problems: list[str] = []
    checked = 0
    heading: tuple[str, ...] | None = None
    in_code = False
    usage_path: tuple[str, ...] | None = None

    for lineno, line in enumerate(markdown.splitlines(), 1):
        if line.startswith("```"):
            in_code = not in_code
            usage_path = None
            continue

        if in_code:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                usage_path = None
                continue
            index = line.find("agent-strace ")
            if index >= 0:
                text = line[index + len("agent-strace "):]
                usage_path = _resolve(root, _usage_tokens(text.split(" | ")[0]))
            elif usage_path is not None and line[:1].isspace():
                text = line
            else:
                usage_path = None
                continue
            text = _SEPARATOR.split(text)[0]
            label = "agent-strace " + " ".join(usage_path)
            check(lineno, label, text, actions_for(usage_path, subtree=False))
            continue

        match = _HEADING.match(line)
        if match:
            heading = _resolve(root, _normalise_skills_argv(match.group(1).split()))
            continue

        if heading is not None and re.match(r"^\|\s*`-", line):
            first_cell = line.split("|")[1]
            flags = " ".join(re.findall(r"`([^`]+)`", first_cell))
            label = "table for agent-strace " + " ".join(heading)
            check(lineno, label, flags, actions_for(heading, subtree=True))

    return problems, checked


def _docstring_usage(path: Path) -> str:
    """Return the ``agent-strace`` lines of a module docstring as a code block."""
    doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
    lines = [line for line in doc.splitlines() if "agent-strace " in line]
    return "\n".join(["```", *lines, "```"])


class CommandsDocFlagsTest(unittest.TestCase):
    def test_commands_md_flags_exist_in_parser(self):
        problems, checked = find_mismatches(COMMANDS_MD.read_text(encoding="utf-8"))
        self.assertGreater(checked, 200)
        self.assertEqual(problems, [], "\n" + "\n".join(problems))

    def test_other_docs_flags_exist_in_parser(self):
        for name in OTHER_DOCS:
            with self.subTest(doc=name):
                text = (REPO_ROOT / name).read_text(encoding="utf-8")
                problems, checked = find_mismatches(text)
                self.assertGreater(checked, 0)
                self.assertEqual(problems, [], "\n" + "\n".join(problems))

    def test_module_docstring_flags_exist_in_parser(self):
        for name in DOCSTRING_MODULES:
            with self.subTest(module=name):
                usage = _docstring_usage(REPO_ROOT / "src" / "agent_trace" / name)
                self.assertIn("agent-strace ", usage)
                problems, _ = find_mismatches(usage)
                self.assertEqual(problems, [], "\n" + "\n".join(problems))

    def test_checker_reports_unknown_flags_and_choices(self):
        markdown = "\n".join([
            "### `auth`",
            "```",
            "agent-strace auth login --host URL",
            "agent-strace mcp [--transport stdio|http]",
            "agent-strace record --name N -- npx -y server",
            "```",
            "| Flag | Description |",
            "|---|---|",
            "| `--client-id ID` | Not an auth option |",
        ])
        problems, _ = find_mismatches(markdown)
        self.assertEqual(len(problems), 3, problems)
        self.assertIn("--host", problems[0])
        self.assertIn("['http']", problems[1])
        self.assertIn("--client-id", problems[2])

    def test_checker_follows_cli_argv_normalisation(self):
        markdown = "\n".join([
            "```",
            "agent-strace policy [SESSION_ID ...] [--output FILE] [--dry-run]",
            "agent-strace eval [session-id] [--baseline FILE]",
            "agent-strace skills eval CASES_FILE --runs N",
            "agent-strace --trace-dir DIR list",
            "```",
        ])
        problems, checked = find_mismatches(markdown)
        self.assertEqual(problems, [])
        self.assertEqual(checked, 5)


if __name__ == "__main__":
    unittest.main()
