"""Experimental skill trigger eval: did a skill load when it should have?

``agent-strace skills`` reports which skills loaded in sessions that were
already recorded.  It cannot say whether a skill *should* have loaded.  This
module runs deliberate test prompts (some that should load a skill and some
that should not) through a headless agent several times.  Each run is captured
by agent-strace hooks into a throwaway trace store, and the eval counts the
runs in which the model loaded the skill.

Load detection is delegated to :func:`agent_trace.skills.analyze_session`;
there is no second detector.  ``--from-traces`` scores pre-recorded run
directories instead of calling an agent, so tests and CI never need a model.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .models import EventType, TraceEvent
from .skills import USER_SKILL_DIRS, _prompt_loads, analyze_session

SCHEMA_VERSION = 1
DEFAULT_RUNS = 3
DEFAULT_TIMEOUT = 300.0
_KILL_GRACE_S = 5.0

PRESETS: dict[str, str] = {
    # Copilot CLI loads workspace skills only from trusted folders: --add-dir
    # trusts the run workspace and --plugin-dir loads temporary hooks.
    "copilot": (
        "copilot -p {prompt} --add-dir {workspace} "
        "--plugin-dir {hooks_plugin_dir} --no-ask-user -s"
    ),
    "claude": "claude -p {prompt} --settings {claude_settings}",
}
PLACEHOLDERS = ("prompt", "workspace", "trace_dir", "hooks_plugin_dir", "claude_settings")
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_]+)\}")

# Flags that approve every tool call.  They are never added by a preset; a
# template that contains one is the user's explicit choice and gets a warning.
APPROVE_ALL_FLAGS = frozenset({
    "--allow-all-tools",
    "--allow-all",
    "--yolo",
    "--dangerously-skip-permissions",
    "--allow-dangerously-skip-permissions",
})

_SCRUBBED_ENV = (
    # Events must land in the run's local trace dir, not a remote collector
    # or a workspace-scoped subdirectory.
    "AGENT_STRACE_ENDPOINT",
    "AGENT_STRACE_WORKSPACE",
    "AGENT_TRACE_CLAUDE_SESSION_ID",
    "AGENT_TRACE_CODEX_SESSION_ID",
    "AGENT_TRACE_GEMINI_SESSION_ID",
    "AGENT_TRACE_CURSOR_SESSION_ID",
    "AGENT_TRACE_COPILOT_SESSION_ID",
    # Never approve every tool implicitly through the inherited environment.
    "COPILOT_ALLOW_ALL",
    # Lets `claude -p` start when the eval itself runs inside Claude Code.
    "CLAUDECODE",
)

_CASE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_CASE_KEYS = frozenset({"id", "skill", "prompt", "should_trigger", "fixture_dir", "runs", "note"})
_RUN_INFO_KEYS = ("exit_code", "timed_out", "duration_s", "agent", "agent_version", "error")

CAVEATS = (
    "Non-deterministic: the same prompt can load the skill in one run and not the next.",
    "Results depend on the agent, model, and version, and on every other skill, plugin, "
    "and instruction file the agent can see.",
    "Small N: a few runs per case show direction, not a precise rate.",
    "The prompt wording is the test: a rephrased task can trigger differently.",
    "Only observable loads count (a skill tool call or a SKILL.md read seen by hooks).",
)


class EvalInputError(ValueError):
    """Invalid cases file, template, or option combination."""


@dataclass(frozen=True)
class EvalCase:
    id: str
    skill: str
    prompt: str
    should_trigger: bool
    fixture_dir: str = ""
    runs: int | None = None
    note: str = ""

    def explicit_invocation(self) -> str:
        """Return how the prompt invokes the skill by name (``/skill`` or ``$skill``)."""
        found = _prompt_loads(self.prompt, [self.skill])
        return found[0][1] if found else ""


@dataclass
class AgentSpec:
    name: str
    template: str
    tokens: list[str]
    timeout: float
    version: str = ""


# --------------------------------------------------------------------------- #
# Cases file
# --------------------------------------------------------------------------- #


def load_cases(path: str | Path) -> list[EvalCase]:
    """Parse and validate a JSON cases file."""
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EvalInputError(f"cannot read cases file {source}: {exc.strerror or type(exc).__name__}") from None
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        detail = f"line {exc.lineno} column {exc.colno}: {exc.msg}" if isinstance(exc, json.JSONDecodeError) else "not UTF-8"
        raise EvalInputError(f"cases file is not valid JSON ({detail})") from None
    if isinstance(raw, dict) and "cases" in raw:
        raw = raw["cases"]
    if not isinstance(raw, list) or not raw:
        raise EvalInputError('cases file must be a non-empty JSON list of cases, or {"cases": [...]}')

    base = source.resolve().parent
    problems: list[str] = []
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for index, item in enumerate(raw, start=1):
        where = f"case {index}"
        problems_before = len(problems)
        if not isinstance(item, dict):
            problems.append(f"{where}: must be an object")
            continue
        case_id = item.get("id")
        if isinstance(case_id, str) and case_id:
            where = f"case {index} ({case_id})"
        unknown = sorted(set(item) - _CASE_KEYS)
        if unknown:
            problems.append(f"{where}: unknown field(s): {', '.join(unknown)}")
        if not isinstance(case_id, str) or not _CASE_ID_RE.match(case_id):
            problems.append(f"{where}: id must be 1-128 letters, digits, '.', '_' or '-', starting with a letter or digit")
        elif case_id in seen:
            problems.append(f"{where}: duplicate id")
        else:
            seen.add(case_id)
        skill = item.get("skill")
        if not isinstance(skill, str) or not skill.strip():
            problems.append(f"{where}: skill must be a non-empty string")
        prompt = item.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            problems.append(f"{where}: prompt must be a non-empty string")
        should_trigger = item.get("should_trigger")
        if not isinstance(should_trigger, bool):
            problems.append(f"{where}: should_trigger must be true or false")
        fixture = item.get("fixture_dir") or ""
        if not isinstance(fixture, str):
            problems.append(f"{where}: fixture_dir must be a string path")
            fixture = ""
        runs = item.get("runs")
        if runs is not None and (isinstance(runs, bool) or not isinstance(runs, int) or runs < 1):
            problems.append(f"{where}: runs must be a positive integer")
        note = item.get("note") or ""
        if not isinstance(note, str):
            problems.append(f"{where}: note must be a string")
            note = ""
        if len(problems) > problems_before:
            continue
        fixture_path = str((base / os.path.expanduser(fixture)).resolve()) if fixture else ""
        cases.append(EvalCase(
            id=case_id,
            skill=skill.strip(),
            prompt=prompt,
            should_trigger=should_trigger,
            fixture_dir=fixture_path,
            runs=runs,
            note=note,
        ))
    if problems:
        raise EvalInputError("invalid cases file:\n  " + "\n  ".join(problems))
    return cases


# --------------------------------------------------------------------------- #
# Agent command
# --------------------------------------------------------------------------- #


def parse_template(template: str) -> list[str]:
    """Split an agent command template into argv tokens and validate placeholders."""
    try:
        tokens = shlex.split(template)
    except ValueError as exc:
        raise EvalInputError(f"cannot parse agent command: {exc}") from None
    if not tokens:
        raise EvalInputError("agent command is empty")
    names = {name for token in tokens for name in _PLACEHOLDER_RE.findall(token)}
    unknown = sorted(names - set(PLACEHOLDERS))
    if unknown:
        allowed = ", ".join("{" + name + "}" for name in PLACEHOLDERS)
        raise EvalInputError(
            f"unknown placeholder(s) in agent command: {', '.join('{' + n + '}' for n in unknown)} "
            f"(allowed: {allowed})"
        )
    if "prompt" not in names:
        raise EvalInputError("agent command must contain {prompt}")
    if _PLACEHOLDER_RE.search(tokens[0]):
        raise EvalInputError("the agent program itself cannot be a placeholder")
    return tokens


def render_command(tokens: Iterable[str], values: dict[str, str]) -> list[str]:
    """Substitute placeholders token by token; values are never re-split or shell-parsed."""
    return [_PLACEHOLDER_RE.sub(lambda match: values[match.group(1)], token) for token in tokens]


def approve_all_flags(tokens: Iterable[str]) -> list[str]:
    tokens = list(tokens)
    found = [token for token in tokens if token.split("=", 1)[0] in APPROVE_ALL_FLAGS]
    for index, token in enumerate(tokens):
        if token == "--permission-mode=bypassPermissions" or (
            token == "--permission-mode" and tokens[index + 1:index + 2] == ["bypassPermissions"]
        ):
            found.append("--permission-mode bypassPermissions")
    return found


def hook_command_prefix() -> str:
    """Run hooks with this interpreter and this copy of agent_trace."""
    package_parent = Path(__file__).resolve().parent.parent
    return (
        f"PYTHONPATH={shlex.quote(str(package_parent))} "
        f"{shlex.quote(sys.executable)} -m agent_trace.cli hook"
    )


def _rewrite_hook_commands(node: Any, prefix: str) -> Any:
    if isinstance(node, dict):
        return {
            key: (
                prefix + value[len("agent-strace hook"):]
                if key == "command" and isinstance(value, str) and value.startswith("agent-strace hook")
                else _rewrite_hook_commands(value, prefix)
            )
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [_rewrite_hook_commands(item, prefix) for item in node]
    return node


def write_hook_configs(root: Path) -> dict[str, str]:
    """Write a Copilot hooks plugin and a Claude settings file under ``root``."""
    from .cli import _claude_hooks_config, _copilot_hooks_config  # cli imports this module

    args = argparse.Namespace(redact=False, no_redact=False)
    prefix = hook_command_prefix()
    plugin_dir = root / "hooks-plugin"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.json").write_text(json.dumps({
        "name": "agent-strace-skills-eval",
        "description": "Temporary agent-strace hooks for one skills eval run",
        "version": "0.0.0",
        "hooks": "hooks.json",
    }, indent=2) + "\n")
    (plugin_dir / "hooks.json").write_text(
        json.dumps(_rewrite_hook_commands(_copilot_hooks_config(args), prefix), indent=2) + "\n"
    )
    settings = root / "claude-settings.json"
    settings.write_text(
        json.dumps(_rewrite_hook_commands(_claude_hooks_config(args), prefix), indent=2) + "\n"
    )
    return {"hooks_plugin_dir": str(plugin_dir), "claude_settings": str(settings)}


def agent_env(trace_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    for key in _SCRUBBED_ENV:
        env.pop(key, None)
    env["AGENT_TRACE_DIR"] = str(trace_dir)
    return env


def _terminate(proc: subprocess.Popen) -> None:
    if os.name != "posix":
        proc.kill()
        proc.wait()
        return
    # The agent runs in its own process group so hook and tool children die too.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=_KILL_GRACE_S)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    proc.wait()


def _last_line(path: Path, limit: int = 200) -> str:
    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines()]
    except OSError:
        return ""
    lines = [line for line in lines if line]
    return lines[-1][:limit] if lines else ""


def run_agent(argv: list[str], *, cwd: Path, env: dict[str, str], timeout: float, log_dir: Path) -> dict[str, Any]:
    """Run one headless agent process with a hard timeout. Never uses a shell."""
    outcome: dict[str, Any] = {"exit_code": None, "timed_out": False, "duration_s": 0.0, "error": ""}
    stdout_path = log_dir / "agent.stdout.log"
    stderr_path = log_dir / "agent.stderr.log"
    started = time.monotonic()
    with open(stdout_path, "wb") as out, open(stderr_path, "wb") as err:
        try:
            proc = subprocess.Popen(
                argv,
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                start_new_session=os.name == "posix",
            )
        except OSError as exc:
            outcome["error"] = f"could not start agent: {exc.strerror or type(exc).__name__}"
            return outcome
        try:
            outcome["exit_code"] = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            outcome["timed_out"] = True
            _terminate(proc)
            outcome["exit_code"] = proc.returncode
    outcome["duration_s"] = round(time.monotonic() - started, 2)
    if outcome["timed_out"]:
        outcome["error"] = f"timed out after {timeout:g}s"
    elif outcome["exit_code"] != 0:
        detail = _last_line(stderr_path) or _last_line(stdout_path)
        outcome["error"] = f"agent exited with status {outcome['exit_code']}" + (f": {detail}" if detail else "")
    return outcome


def agent_version(program: str) -> str:
    try:
        completed = subprocess.run(
            [program, "--version"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    lines = [line.strip() for line in (completed.stdout or "").splitlines() if line.strip()]
    return lines[0][:120].rstrip(".") if completed.returncode == 0 and lines else ""


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def find_session_dirs(trace_dir: Path) -> list[Path]:
    """Return session dirs under a trace store, including dirs without meta.json.

    Sessions whose hooks never fired a session start (for example Copilot CLI
    subagents recorded before metadata was created lazily) have only
    ``events.ndjson``; ``list`` and ``skills`` skip them, the eval does not.
    """
    roots = [trace_dir]
    workspaces = trace_dir / "workspaces"
    if workspaces.is_dir() and not workspaces.is_symlink():
        roots.extend(sorted(p for p in workspaces.iterdir() if p.is_dir() and not p.is_symlink()))
    found: list[Path] = []
    for root in roots:
        if not root.is_dir() or root.is_symlink():
            continue
        for child in sorted(root.iterdir()):
            if child.name == "workspaces" or child.is_symlink() or not child.is_dir():
                continue
            events = child / "events.ndjson"
            if events.is_file() and not events.is_symlink():
                found.append(child)
    return found


def read_events(path: Path) -> list[TraceEvent]:
    events: list[TraceEvent] = []
    try:
        handle = open(path, encoding="utf-8", errors="replace")
    except OSError:
        return events
    with handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(TraceEvent.from_json(line))
            except (ValueError, KeyError, TypeError, AttributeError):
                continue
    return events


def skill_matches(loaded_name: str, wanted: str) -> bool:
    if loaded_name == wanted:
        return True
    return loaded_name.rsplit(":", 1)[-1] == wanted.rsplit(":", 1)[-1]


def score_trace_dir(
    trace_dir: Path,
    skill: str,
    *,
    workspace: str = "",
    extra_dirs: Iterable[str] = (),
    user_dirs: Iterable[str] = USER_SKILL_DIRS,
) -> dict[str, Any]:
    """Score one run's trace store: did the model load ``skill`` anywhere in it?"""
    extra_dirs = list(extra_dirs)
    user_dirs = list(user_dirs)
    methods: set[str] = set()
    other_skills: set[str] = set()
    load_sessions: list[dict[str, Any]] = []
    user_invoked = False
    sessions = events_total = without_meta = without_start = unreadable = 0
    for session_dir in find_session_dirs(trace_dir):
        events = read_events(session_dir / "events.ndjson")
        if not events:
            continue
        sessions += 1
        events_total += len(events)
        has_meta = (session_dir / "meta.json").is_file()
        has_start = any(event.event_type == EventType.SESSION_START for event in events)
        without_meta += not has_meta
        without_start += not has_start
        try:
            report = analyze_session(
                events,
                session_id=session_dir.name,
                workspace=workspace,
                extra_dirs=extra_dirs,
                user_dirs=user_dirs,
            )
        except (ValueError, TypeError, KeyError, OSError):
            unreadable += 1
            continue
        model_load = False
        for entry in report["loaded"]:
            name = str(entry.get("name", ""))
            by_model = [load for load in entry.get("loads", []) if load.get("invoked_by") == "model"]
            if skill_matches(name, skill):
                for load in by_model:
                    methods.add(str(load.get("method", "")))
                    model_load = True
                if len(by_model) < len(entry.get("loads", [])):
                    user_invoked = True
            elif by_model:
                other_skills.add(name)
        if model_load:
            load_sessions.append({
                "session": session_dir.name,
                "has_meta": has_meta,
                "has_session_start": has_start,
            })
    return {
        "loaded": bool(methods),
        "methods": sorted(methods),
        "user_invoked": user_invoked,
        "sessions": sessions,
        "sessions_without_meta": without_meta,
        "sessions_without_session_start": without_start,
        "sessions_unreadable": unreadable,
        "events": events_total,
        "load_sessions": load_sessions,
        "other_skills_loaded": sorted(other_skills),
    }


def classify_run(result: dict[str, Any]) -> dict[str, Any]:
    """Set ``status`` to loaded, not_loaded, or unscored (with a reason)."""
    reason = ""
    if not result.get("loaded"):
        if result.get("error"):
            reason = str(result["error"])
        elif result.get("timed_out"):
            reason = "timed out"
        elif result.get("exit_code") not in (None, 0):
            reason = f"agent exited with status {result['exit_code']}"
        elif not result.get("events"):
            reason = "no trace events captured (did the hooks run?)"
    result["status"] = "loaded" if result.get("loaded") else ("unscored" if reason else "not_loaded")
    result["unscored_reason"] = reason
    return result


def summarize_case(case: EvalCase, runs: list[dict[str, Any]]) -> dict[str, Any]:
    loaded = sum(1 for run in runs if run["status"] == "loaded")
    not_loaded = sum(1 for run in runs if run["status"] == "not_loaded")
    unscored = sum(1 for run in runs if run["status"] == "unscored")
    scored = loaded + not_loaded
    return {
        "id": case.id,
        "skill": case.skill,
        "should_trigger": case.should_trigger,
        "prompt": case.prompt,
        "note": case.note,
        "explicit_invocation": case.explicit_invocation(),
        "runs": len(runs),
        "scored": scored,
        "loaded": loaded,
        "not_loaded": not_loaded,
        "unscored": unscored,
        "trigger_rate": round(loaded / scored, 4) if scored else None,
        "user_invoked_runs": sum(1 for run in runs if run.get("user_invoked")),
        "run_results": runs,
    }


def compute_totals(case_reports: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    should = {"cases": 0, "runs": 0, "loaded": 0, "missed": 0, "unscored": 0}
    should_not = {"cases": 0, "runs": 0, "loaded": 0, "clean": 0, "unscored": 0}
    for case in case_reports:
        bucket = should if case["should_trigger"] else should_not
        bucket["cases"] += 1
        bucket["runs"] += case["runs"]
        bucket["loaded"] += case["loaded"]
        bucket["unscored"] += case["unscored"]
        bucket["missed" if case["should_trigger"] else "clean"] += case["not_loaded"]
    return {"should_trigger": should, "should_not_trigger": should_not}


def gate_failures(report: dict[str, Any], *, fail_on_miss: bool, fail_on_false_trigger: bool) -> list[str]:
    """Return case-level reasons for a non-zero exit. Unscored runs fail an enabled gate."""
    failures: list[str] = []
    for case in report["cases"]:
        if case["should_trigger"] and fail_on_miss:
            if case["not_loaded"] or case["unscored"] or not case["runs"]:
                failures.append(
                    f"{case['id']}: should trigger; {case['not_loaded']} missed, "
                    f"{case['unscored']} unscored of {case['runs']} runs"
                )
        if not case["should_trigger"] and fail_on_false_trigger:
            if case["loaded"] or case["unscored"] or not case["runs"]:
                failures.append(
                    f"{case['id']}: should not trigger; {case['loaded']} loaded, "
                    f"{case['unscored']} unscored of {case['runs']} runs"
                )
    return failures


# --------------------------------------------------------------------------- #
# Live runs and recorded runs
# --------------------------------------------------------------------------- #


def _copy_sessions(trace_dir: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for session_dir in find_session_dirs(trace_dir):
        relative = session_dir.relative_to(trace_dir)
        target = destination / relative
        target.mkdir(parents=True, exist_ok=True)
        for name in ("events.ndjson", "meta.json"):
            source = session_dir / name
            if source.is_file() and not source.is_symlink():
                shutil.copy2(source, target / name)


def run_case_live(
    case: EvalCase,
    run_no: int,
    spec: AgentSpec,
    *,
    extra_dirs: Iterable[str] = (),
    user_dirs: Iterable[str] = USER_SKILL_DIRS,
    keep_workspaces: bool = False,
    save_root: Path | None = None,
) -> dict[str, Any]:
    root = Path(tempfile.mkdtemp(prefix=f"agent-strace-skills-eval-{case.id}-"))
    workspace = root / "workspace"
    traces = root / "traces"
    try:
        if case.fixture_dir:
            shutil.copytree(case.fixture_dir, workspace, symlinks=True)
        else:
            workspace.mkdir()
        traces.mkdir()
        values = {
            "prompt": case.prompt,
            "workspace": str(workspace),
            "trace_dir": str(traces),
            **write_hook_configs(root),
        }
        outcome = run_agent(
            render_command(spec.tokens, values),
            cwd=workspace,
            env=agent_env(traces),
            timeout=spec.timeout,
            log_dir=root,
        )
        result = {"run": f"run-{run_no}", **outcome}
        result.update(score_trace_dir(
            traces, case.skill, workspace=str(workspace), extra_dirs=extra_dirs, user_dirs=user_dirs,
        ))
        classify_run(result)
        if save_root is not None:
            destination = save_root / case.id / f"run-{run_no}"
            _copy_sessions(traces, destination)
            info = {key: outcome[key] for key in ("exit_code", "timed_out", "duration_s", "error")}
            info.update({"agent": spec.name, "agent_version": spec.version})
            (destination / "run.json").write_text(json.dumps(info, indent=2) + "\n")
        if keep_workspaces:
            result["kept_dir"] = str(root)
        return result
    finally:
        if not keep_workspaces:
            shutil.rmtree(root, ignore_errors=True)


def _natural_key(name: str) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def _read_run_info(run_dir: Path, warnings: list[str]) -> dict[str, Any]:
    path = run_dir / "run.json"
    if not path.is_file() or path.is_symlink():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        warnings.append(f"{run_dir}: run.json is not valid JSON; ignored")
        return {}
    if not isinstance(data, dict):
        return {}
    info: dict[str, Any] = {}
    if isinstance(data.get("exit_code"), int) and not isinstance(data.get("exit_code"), bool):
        info["exit_code"] = data["exit_code"]
    if isinstance(data.get("timed_out"), bool):
        info["timed_out"] = data["timed_out"]
    if isinstance(data.get("duration_s"), (int, float)) and not isinstance(data.get("duration_s"), bool):
        info["duration_s"] = data["duration_s"]
    for key in ("agent", "agent_version", "error"):
        if isinstance(data.get(key), str) and data[key]:
            info[key] = data[key]
    return info


def recorded_run_dirs(root: Path, case_id: str) -> list[Path]:
    """Run dirs for one case: ``ROOT/<case_id>/<run>/``, or the case dir itself if it is a store."""
    case_dir = root / case_id
    if not case_dir.is_dir() or case_dir.is_symlink():
        return []
    children = [
        child for child in case_dir.iterdir()
        if child.is_dir() and not child.is_symlink() and not child.name.startswith(".")
    ]
    if any((child / "events.ndjson").is_file() for child in children):
        return [case_dir]
    return sorted(children, key=lambda path: _natural_key(path.name))


def score_recorded_case(
    case: EvalCase,
    root: Path,
    *,
    workspace: str = "",
    extra_dirs: Iterable[str] = (),
    user_dirs: Iterable[str] = USER_SKILL_DIRS,
    warnings: list[str] | None = None,
) -> list[dict[str, Any]]:
    warnings = warnings if warnings is not None else []
    extra_dirs = list(extra_dirs)
    user_dirs = list(user_dirs)
    results = []
    for run_dir in recorded_run_dirs(root, case.id):
        result: dict[str, Any] = {"run": run_dir.name, "exit_code": None, "timed_out": False, "duration_s": None, "error": ""}
        result.update(_read_run_info(run_dir, warnings))
        result.update(score_trace_dir(
            run_dir, case.skill, workspace=workspace, extra_dirs=extra_dirs, user_dirs=user_dirs,
        ))
        results.append(classify_run(result))
    if not results:
        warnings.append(f"{case.id}: no recorded runs under {root / case.id}")
    return results


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def build_report(
    case_reports: list[dict[str, Any]],
    *,
    mode: str,
    cases_file: str,
    agent: AgentSpec | None = None,
    source: str = "",
    default_runs: int | None = None,
    warnings: Iterable[str] = (),
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experimental": True,
        "mode": mode,
        "cases_file": cases_file,
    }
    if agent is not None:
        report["agent"] = {
            "name": agent.name,
            "command": agent.template,
            "version": agent.version,
            "timeout_s": agent.timeout,
        }
        report["default_runs"] = default_runs
    if source:
        report["source"] = source
    report["cases"] = case_reports
    report["totals"] = compute_totals(case_reports)
    report["warnings"] = list(warnings)
    report["caveats"] = list(CAVEATS)
    return report


def _rate(case: dict[str, Any]) -> str:
    if not case["scored"]:
        return "-"
    return f"{round(100 * case['loaded'] / case['scored'])}% ({case['loaded']}/{case['scored']})"


def format_text(report: dict[str, Any]) -> str:
    lines = ["Skill trigger eval (experimental)"]
    agent = report.get("agent")
    if agent:
        version = f" {agent['version']}" if agent.get("version") else ""
        lines.append(
            f"  agent: {agent['name']}{version}   default runs: {report.get('default_runs')}   "
            f"timeout: {agent['timeout_s']:g}s"
        )
    if report.get("source"):
        lines.append(f"  recorded runs: {report['source']}")
    lines.append("")

    headers = ("CASE", "SKILL", "EXPECT", "RUNS", "LOADED", "RATE", "")
    rows = []
    for case in report["cases"]:
        extra = []
        if case["unscored"]:
            extra.append(f"{case['unscored']} unscored")
        if case["user_invoked_runs"]:
            extra.append(f"{case['user_invoked_runs']} user-invoked")
        rows.append((
            case["id"],
            case["skill"],
            "trigger" if case["should_trigger"] else "no trigger",
            str(case["runs"]),
            str(case["loaded"]),
            _rate(case),
            ", ".join(extra),
        ))
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    right = {3, 4}

    def render(row: tuple[str, ...]) -> str:
        cells = [cell.rjust(widths[i]) if i in right else cell.ljust(widths[i]) for i, cell in enumerate(row)]
        return ("  " + "  ".join(cells)).rstrip()

    lines.append(render(headers))
    lines.extend(render(row) for row in rows)
    lines.append("")

    should = report["totals"]["should_trigger"]
    should_not = report["totals"]["should_not_trigger"]
    lines.append(
        f"  should trigger:     {should['runs']} runs, {should['loaded']} loaded, "
        f"{should['missed']} missed, {should['unscored']} unscored"
    )
    lines.append(
        f"  should not trigger: {should_not['runs']} runs, {should_not['loaded']} loaded, "
        f"{should_not['clean']} clean, {should_not['unscored']} unscored"
    )

    notes = []
    for case in report["cases"]:
        if case["explicit_invocation"]:
            notes.append(f"{case['id']}: prompt names the skill ({case['explicit_invocation']}); not a trigger test")
        for run in case["run_results"]:
            label = f"{case['id']} {run['run']}"
            if run["status"] == "unscored":
                notes.append(f"{label}: unscored, {run['unscored_reason']}")
            if run.get("load_sessions") and all(not s["has_session_start"] for s in run["load_sessions"]):
                notes.append(f"{label}: loaded only in a session without session start (likely a subagent)")
            if run.get("kept_dir"):
                notes.append(f"{label}: kept {run['kept_dir']}")
    notes.extend(report.get("warnings", []))
    if notes:
        lines.append("")
        lines.append("  Notes:")
        lines.extend(f"    {note}" for note in notes)

    lines.append("")
    lines.append(
        "  Counts only. Non-deterministic; depends on agent, model and version; small N; "
        "the prompt wording is the test."
    )
    lines.append("  See docs/skills-eval.md.")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _select_cases(cases: list[EvalCase], wanted: list[str]) -> list[EvalCase]:
    if not wanted:
        return cases
    known = {case.id for case in cases}
    missing = [case_id for case_id in wanted if case_id not in known]
    if missing:
        raise EvalInputError(f"unknown case id(s): {', '.join(missing)}")
    return [case for case in cases if case.id in set(wanted)]


def _agent_spec(args: argparse.Namespace) -> AgentSpec:
    timeout = float(args.timeout if args.timeout is not None else DEFAULT_TIMEOUT)
    if timeout <= 0:
        raise EvalInputError("--timeout must be positive")
    if args.agent_cmd:
        name, template = "custom", args.agent_cmd
    else:
        name, template = args.agent, PRESETS[args.agent]
    tokens = parse_template(template)
    if shutil.which(tokens[0]) is None:
        raise EvalInputError(f"agent program not found on PATH: {tokens[0]}")
    return AgentSpec(name=name, template=template, tokens=tokens, timeout=timeout)


def cmd_skills_eval(args: argparse.Namespace) -> int:
    as_json = getattr(args, "format", "text") == "json"
    extra_dirs = list(getattr(args, "skills_dirs", None) or [])
    warnings: list[str] = []
    try:
        cases = _select_cases(load_cases(args.cases_file), list(getattr(args, "case_ids", None) or []))
        if args.runs is not None and args.runs < 1:
            raise EvalInputError("--runs must be at least 1")

        if args.from_traces:
            root = Path(args.from_traces)
            if not root.is_dir():
                raise EvalInputError(f"--from-traces directory not found: {root}")
            for flag, value in (("--runs", args.runs), ("--timeout", args.timeout), ("--save-traces", args.save_traces)):
                if value is not None:
                    warnings.append(f"{flag} is ignored with --from-traces")
            known = {case.id for case in cases}
            for child in sorted(root.iterdir()):
                if child.is_dir() and not child.name.startswith(".") and child.name not in known and not getattr(args, "case_ids", None):
                    warnings.append(f"{child.name}: recorded runs with no matching case; ignored")
            case_reports = [
                summarize_case(case, score_recorded_case(
                    case, root, workspace=args.workspace or "", extra_dirs=extra_dirs, warnings=warnings,
                ))
                for case in cases
            ]
            report = build_report(
                case_reports, mode="from_traces", cases_file=str(args.cases_file),
                source=str(root), warnings=warnings,
            )
        else:
            spec = _agent_spec(args)
            if args.workspace:
                warnings.append("--workspace is ignored for live runs; each run uses its own workspace")
            for case in cases:
                if case.fixture_dir and not Path(case.fixture_dir).is_dir():
                    raise EvalInputError(f"{case.id}: fixture_dir not found: {case.fixture_dir}")
            save_root = Path(args.save_traces) if args.save_traces else None
            if save_root is not None and save_root.exists() and (not save_root.is_dir() or any(save_root.iterdir())):
                raise EvalInputError(f"--save-traces directory must be new or empty: {save_root}")
            flagged = approve_all_flags(spec.tokens)
            if flagged:
                sys.stderr.write(
                    f"agent-strace: warning: the agent command approves every tool call ({', '.join(flagged)})\n"
                )
            if spec.name in PRESETS:
                spec.version = agent_version(spec.tokens[0])
            default_runs = args.runs or DEFAULT_RUNS
            total = sum(case.runs or default_runs for case in cases)
            done = 0
            case_reports = []
            for case in cases:
                runs = []
                for run_no in range(1, (case.runs or default_runs) + 1):
                    done += 1
                    sys.stderr.write(f"[{done}/{total}] {case.id} run {run_no} ... ")
                    sys.stderr.flush()
                    result = run_case_live(
                        case, run_no, spec,
                        extra_dirs=extra_dirs,
                        keep_workspaces=bool(args.keep_workspaces),
                        save_root=save_root,
                    )
                    reason = f" ({result['unscored_reason']})" if result["status"] == "unscored" else ""
                    sys.stderr.write(f"{result['status']}{reason} {result['duration_s']}s\n")
                    runs.append(result)
                case_reports.append(summarize_case(case, runs))
            report = build_report(
                case_reports, mode="live", cases_file=str(args.cases_file),
                agent=spec, default_runs=default_runs, warnings=warnings,
            )
    except EvalInputError as exc:
        sys.stderr.write(f"agent-strace skills eval: {exc}\n")
        return 2

    if as_json:
        sys.stdout.write(json.dumps(report, indent=2) + "\n")
    else:
        sys.stdout.write(format_text(report))

    failures = gate_failures(
        report,
        fail_on_miss=bool(getattr(args, "fail_on_miss", False)),
        fail_on_false_trigger=bool(getattr(args, "fail_on_false_trigger", False)),
    )
    if failures:
        sys.stderr.write("agent-strace skills eval: gate failed\n")
        for failure in failures:
            sys.stderr.write(f"  {failure}\n")
        return 1
    return 0


__all__ = [
    "APPROVE_ALL_FLAGS",
    "CAVEATS",
    "DEFAULT_RUNS",
    "DEFAULT_TIMEOUT",
    "PLACEHOLDERS",
    "PRESETS",
    "SCHEMA_VERSION",
    "AgentSpec",
    "EvalCase",
    "EvalInputError",
    "agent_env",
    "approve_all_flags",
    "build_report",
    "classify_run",
    "cmd_skills_eval",
    "compute_totals",
    "find_session_dirs",
    "format_text",
    "gate_failures",
    "hook_command_prefix",
    "load_cases",
    "parse_template",
    "read_events",
    "recorded_run_dirs",
    "render_command",
    "run_agent",
    "run_case_live",
    "score_recorded_case",
    "score_trace_dir",
    "skill_matches",
    "summarize_case",
    "write_hook_configs",
]
