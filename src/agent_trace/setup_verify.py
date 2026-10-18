"""Read-only verification of installed Codex hooks (not runtime capture)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def verify_codex_setup(
    hooks_path: Path, expected: dict, version: str
) -> tuple[bool, list[str]]:
    """Check a config and the AgentTrace CLI without running provider hooks."""
    problems: list[str] = []

    if not hooks_path.is_file():
        problems.append(
            f"Missing Codex hooks config: {hooks_path}. "
            "Run agent-strace setup --cli codex to install it."
        )
        return False, problems

    try:
        installed = json.loads(hooks_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        problems.append(f"Cannot read valid Codex hooks JSON at {hooks_path}: {exc}")
        return False, problems

    hooks = installed.get("hooks") if isinstance(installed, dict) else None
    if not isinstance(hooks, dict):
        problems.append(f"Invalid Codex hooks structure in {hooks_path}: missing hooks object.")
        return False, problems

    # Match the full generated event group (including matcher and hook type),
    # not just the command text. A wrong matcher can silently suppress capture.
    expected_commands: set[str] = set()
    for event, groups in expected["hooks"].items():
        expected_commands.update(
            entry["command"] for group in groups for entry in group["hooks"]
        )
        actual_groups = hooks.get(event)
        if not isinstance(actual_groups, list):
            actual_groups = []

        def matches_group(installed_group: object, expected_group: dict) -> bool:
            if not isinstance(installed_group, dict):
                return False
            if installed_group.get("matcher") != expected_group.get("matcher"):
                return False
            entries = installed_group.get("hooks")
            return isinstance(entries, list) and all(
                required in entries for required in expected_group["hooks"]
            )

        if not all(
            any(matches_group(actual, expected_group) for actual in actual_groups)
            for expected_group in groups
        ):
            problems.append(
                f"{event}: expected AgentTrace hook command, type or matcher "
                "missing or stale. Re-run agent-strace setup --cli codex "
                "after reviewing your backup."
            )

    for event, groups in hooks.items():
        if not isinstance(groups, list):
            continue
        for group in groups:
            if not isinstance(group, dict) or not isinstance(group.get("hooks"), list):
                continue
            for entry in group["hooks"]:
                if not isinstance(entry, dict):
                    continue
                command = entry.get("command")
                if (
                    isinstance(command, str)
                    and "agent-strace hook " in command
                    and command not in expected_commands
                ):
                    problems.append(f"{event}: stale or unexpected AgentTrace hook command.")
                    break

    executable = shutil.which("agent-strace")
    if executable is None:
        problems.append(
            "The agent-strace executable is not on PATH. Install this "
            "AgentTrace release into the same environment used by Codex."
        )
    else:
        try:
            probe = subprocess.run(
                [executable, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            problems.append(f"Cannot run {executable} --version: {exc}")
        else:
            expected_version = f"agent-strace {version}"
            if probe.returncode != 0 or probe.stdout.strip() != expected_version:
                problems.append(
                    f"AgentTrace CLI version mismatch: expected {expected_version!r}; "
                    f"got {probe.stdout.strip()!r} (exit {probe.returncode})."
                )

    return not problems, problems
