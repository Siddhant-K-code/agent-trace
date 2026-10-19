"""Read-only verification of installed Codex hooks (not runtime capture)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

_REDACTION_PREFIXES = {
    "AGENT_TRACE_REDACT=1 ": "redact",
    "AGENT_TRACE_NO_REDACT=1 ": "no-redact",
}
_MATCH_ALL = (None, "", "*", ".*")
_IGNORED_MATCHER_EVENTS = ("Stop", "UserPromptSubmit")


def _command_and_mode(command: str) -> tuple[str, str]:
    """Normalize installed redaction flags without hiding the event action."""
    for prefix, mode in _REDACTION_PREFIXES.items():
        if command.startswith(prefix):
            return command[len(prefix):], mode
    return command, "default"


def _matcher_matches(event: str, actual: object, expected: object) -> bool:
    if event in _IGNORED_MATCHER_EVENTS:
        return True
    if expected == ".*":
        return actual in _MATCH_ALL
    return actual == expected


def verify_codex_setup(
    hooks_path: Path,
    expected: dict,
    version: str,
    *,
    details: dict | None = None,
) -> tuple[bool, list[str]]:
    """Verify the correct event, command, and type exactly once, without writes.

    Only installed AgentTrace entries are inspected; arbitrary user hooks and
    documented Codex fields on handlers remain valid.
    """
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
        return False, [f"Cannot read valid Codex hooks JSON at {hooks_path}: {exc}"]

    hooks = installed.get("hooks") if isinstance(installed, dict) else None
    if not isinstance(hooks, dict):
        return False, [f"Invalid Codex hooks structure in {hooks_path}: missing hooks object."]

    # Verification flags must not change the expected mode: installation with
    # --redact or --no-redact must verify with default CLI arguments.
    expected_by_event: dict[str, dict[str, dict]] = {}
    all_expected_commands: set[str] = set()
    for event, groups in expected["hooks"].items():
        commands = {}
        for group in groups:
            for handler in group["hooks"]:
                command, _ = _command_and_mode(handler["command"])
                commands[command] = {"type": handler["type"], "matcher": group.get("matcher")}
                all_expected_commands.add(command)
        expected_by_event[event] = commands

    seen: set[tuple[str, str]] = set()
    satisfied: set[tuple[str, str]] = set()
    modes: set[str] = set()
    for event, groups in hooks.items():
        if not isinstance(groups, list):
            if event in expected_by_event:
                problems.append(f"{event}: hooks must be a list.")
            continue
        for group in groups:
            if not isinstance(group, dict):
                continue
            entries = group.get("hooks")
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                command = entry.get("command")
                if not isinstance(command, str):
                    continue
                canonical, mode = _command_and_mode(command)
                if "agent-strace hook " not in canonical:
                    continue  # Unrelated user hook: allowed without modification.
                key = (event, canonical)
                if key in seen:
                    problems.append(f"{event}: duplicate AgentTrace hook command {command!r}.")
                    continue
                seen.add(key)
                required = expected_by_event.get(event, {}).get(canonical)
                if required is None:
                    if canonical in all_expected_commands:
                        problems.append(
                            f"{event}: AgentTrace hook {command!r} is assigned to the wrong event."
                        )
                    else:
                        problems.append(
                            f"{event}: stale or unexpected AgentTrace hook command {command!r}."
                        )
                    continue

                valid = True
                if entry.get("type") != required["type"]:
                    problems.append(
                        f"{event}: {command!r} has handler type {entry.get('type')!r}, "
                        f"expected {required['type']!r}."
                    )
                    valid = False
                if not _matcher_matches(event, group.get("matcher"), required["matcher"]):
                    problems.append(
                        f"{event}: matcher is {group.get('matcher')!r}, "
                        f"expected {required['matcher']!r} for {command!r}."
                    )
                    valid = False
                if valid:
                    satisfied.add(key)
                    modes.add(mode)

    for event, commands in expected_by_event.items():
        for command in commands:
            if (event, command) not in satisfied:
                problems.append(
                    f"{event}: missing {json.dumps({'type': commands[command]['type'], 'command': command})}."
                )

    if len(modes) > 1:
        problems.append(
            "Codex AgentTrace hooks use inconsistent redaction modes: "
            + ", ".join(sorted(modes))
            + ". Install one mode consistently."
        )

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

    if not problems and details is not None:
        details.update(
            event_count=len(expected_by_event),
            event_total=len(expected_by_event),
            redaction=next(iter(modes), "default"),
            executable=executable,
            version=f"agent-strace {version}",
        )
    return not problems, problems
