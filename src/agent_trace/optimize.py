"""Propose improvements to AGENTS.md and skill files from trace failures.

Analyzes a session or dataset of sessions, clusters failures by root cause,
and proposes concrete additions to instruction files (AGENTS.md, CLAUDE.md,
skill .md files). Uses an LLM for clustering and proposal generation via any
OpenAI-compatible endpoint. Falls back to heuristic-only mode when no LLM
is configured.

No new dependencies. LLM calls use urllib.request.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from .models import EventType, TraceEvent
from .store import TraceStore


# ---------------------------------------------------------------------------
# Failure pattern extraction (heuristic, no LLM)
# ---------------------------------------------------------------------------

@dataclass
class FailurePattern:
    name: str
    description: str
    session_ids: list[str]
    example_events: list[str]   # short human-readable descriptions
    proposed_instruction: str   # the text to add to the target file


def _extract_patterns_heuristic(
    sessions: dict[str, list[TraceEvent]],
) -> list[FailurePattern]:
    """Identify failure clusters from trace structure without an LLM."""
    patterns: list[FailurePattern] = []

    blind_retry_sessions: list[str] = []
    blind_retry_examples: list[str] = []

    scope_sessions: list[str] = []
    scope_examples: list[str] = []

    error_no_change_sessions: list[str] = []
    error_no_change_examples: list[str] = []

    for sid, events in sessions.items():
        # Pattern 1: blind retry — same tool called 3+ times consecutively
        prev_tool: str | None = None
        run = 0
        for ev in events:
            if ev.event_type == EventType.TOOL_CALL:
                name = ev.data.get("tool_name", "")
                if name == prev_tool:
                    run += 1
                    if run >= 2 and sid not in blind_retry_sessions:
                        blind_retry_sessions.append(sid)
                        blind_retry_examples.append(
                            f"session {sid[:8]}: {name!r} called {run+1}+ times consecutively"
                        )
                else:
                    prev_tool = name
                    run = 0

        # Pattern 2: error with no subsequent change — error followed by
        # the exact same tool call (no write/edit between them)
        last_error_tool2: str | None = None
        last_tool2: str | None = None
        for ev in events:
            if ev.event_type in (EventType.FILE_WRITE,):
                last_error_tool2 = None  # any file write clears the error context
            elif ev.event_type == EventType.TOOL_CALL:
                name = ev.data.get("tool_name", "")
                if name.lower() in ("write", "edit", "create"):
                    last_error_tool2 = None  # write-type tool call also clears context
                elif (
                    last_error_tool2
                    and name == last_error_tool2
                    and sid not in error_no_change_sessions
                ):
                    error_no_change_sessions.append(sid)
                    error_no_change_examples.append(
                        f"session {sid[:8]}: retried {name!r} after error without changing approach"
                    )
                last_tool2 = name
            elif ev.event_type == EventType.ERROR:
                last_error_tool2 = last_tool2

        # Pattern 3: wide blast radius — more than 8 distinct files written
        files_written: set[str] = set()
        for ev in events:
            if ev.event_type == EventType.FILE_WRITE:
                path = ev.data.get("path") or ev.data.get("file_path") or ""
                if path:
                    files_written.add(path)
        if len(files_written) > 8 and sid not in scope_sessions:
            scope_sessions.append(sid)
            scope_examples.append(
                f"session {sid[:8]}: wrote to {len(files_written)} distinct files"
            )

    if blind_retry_sessions:
        patterns.append(FailurePattern(
            name="blind-retry",
            description="Agent retried the same tool consecutively without changing approach",
            session_ids=blind_retry_sessions,
            example_events=blind_retry_examples[:3],
            proposed_instruction=(
                "## Retry policy\n"
                "After any failed command, read the full error output before retrying.\n"
                "If the same command fails twice with the same error, stop and report\n"
                "rather than attempting a third time."
            ),
        ))

    if error_no_change_sessions:
        patterns.append(FailurePattern(
            name="error-no-change",
            description="Agent retried a tool after an error without modifying its approach",
            session_ids=error_no_change_sessions,
            example_events=error_no_change_examples[:3],
            proposed_instruction=(
                "## Error handling\n"
                "When a tool call produces an error, diagnose the cause before retrying.\n"
                "Do not call the same tool with the same arguments after an error."
            ),
        ))

    if scope_sessions:
        patterns.append(FailurePattern(
            name="wide-blast-radius",
            description="Agent wrote to an unusually large number of files in a single session",
            session_ids=scope_sessions,
            example_events=scope_examples[:3],
            proposed_instruction=(
                "## Scope discipline\n"
                "Limit file writes to files directly relevant to the task.\n"
                "If more than 5 files need to change, confirm scope with the user before proceeding."
            ),
        ))

    return patterns


# ---------------------------------------------------------------------------
# LLM-assisted clustering and proposal
# ---------------------------------------------------------------------------

def _build_llm_prompt(
    sessions: dict[str, list[TraceEvent]],
    target_file: str,
    existing_content: str,
) -> str:
    """Build a prompt for the LLM to cluster failures and propose instructions."""
    session_summaries = []
    for sid, events in list(sessions.items())[:10]:  # cap at 10 sessions
        errors = [e for e in events if e.event_type == EventType.ERROR]
        tool_calls = [e for e in events if e.event_type == EventType.TOOL_CALL]
        # Detect consecutive same-tool runs
        runs: list[str] = []
        prev = None
        run_len = 0
        for ev in tool_calls:
            name = ev.data.get("tool_name", "")
            if name == prev:
                run_len += 1
            else:
                if run_len >= 2 and prev:
                    runs.append(f"{prev} x{run_len+1}")
                prev = name
                run_len = 0
        summary = (
            f"Session {sid[:8]}: "
            f"{len(tool_calls)} tool calls, "
            f"{len(errors)} errors"
        )
        if runs:
            summary += f", consecutive runs: {', '.join(runs[:3])}"
        if errors:
            msgs = [e.data.get("message", "")[:60] for e in errors[:2]]
            summary += f", error samples: {'; '.join(msgs)}"
        session_summaries.append(summary)

    summaries_text = "\n".join(f"- {s}" for s in session_summaries)

    existing_snippet = existing_content[:2000] if existing_content else "(file does not exist yet)"

    return f"""You are analyzing agent execution traces to improve an instruction file.

Target file: {target_file}
Existing content (first 2000 chars):
{existing_snippet}

Session summaries (failures and anomalies):
{summaries_text}

Task:
1. Identify 1-3 distinct failure patterns from the session summaries.
2. For each pattern, propose a short, concrete addition to {target_file}.
   - Additions must be plain Markdown, 2-5 lines each.
   - Do not rewrite existing content. Only propose new sections or lines.
   - Be specific: name the tool, file type, or behavior to constrain.

Respond with JSON only, no prose:
{{
  "patterns": [
    {{
      "name": "short-slug",
      "description": "one sentence describing the failure pattern",
      "affected_sessions": ["sid1", "sid2"],
      "proposed_addition": "## Section title\\nInstruction text here."
    }}
  ]
}}"""


def _call_llm(prompt: str, base_url: str, model: str, api_key: str) -> str | None:
    """Call an OpenAI-compatible chat endpoint. Returns the response text or None."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 1024,
    }).encode()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
    except (urllib.error.URLError, KeyError, json.JSONDecodeError, IndexError):
        return None


def _parse_llm_patterns(
    response: str,
    sessions: dict[str, list[TraceEvent]],
) -> list[FailurePattern]:
    """Parse LLM JSON response into FailurePattern objects."""
    try:
        # Strip markdown code fences if present
        text = response.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
        if text.endswith("```"):
            text = "\n".join(text.split("\n")[:-1])
        data = json.loads(text)
        patterns = []
        for p in data.get("patterns", []):
            name = str(p.get("name", "unknown"))
            desc = str(p.get("description", ""))
            affected = [str(s) for s in p.get("affected_sessions", [])]
            addition = str(p.get("proposed_addition", ""))
            if not addition:
                continue
            # Match affected session IDs to full IDs in sessions dict
            matched = [
                sid for sid in sessions
                if any(sid.startswith(a) or a.startswith(sid[:8]) for a in affected)
            ]
            patterns.append(FailurePattern(
                name=name,
                description=desc,
                session_ids=matched or list(sessions.keys())[:3],
                example_events=[],
                proposed_instruction=addition,
            ))
        return patterns
    except (json.JSONDecodeError, KeyError, TypeError):
        return []


# ---------------------------------------------------------------------------
# Diff generation
# ---------------------------------------------------------------------------

@dataclass
class OptimizeProposal:
    target_file: str
    patterns: list[FailurePattern]
    existing_content: str

    def diff_text(self) -> str:
        lines = []
        for p in self.patterns:
            lines.append(f"\n# Cluster: {p.name} ({len(p.session_ids)} session(s))")
            lines.append(f"# {p.description}")
            for ex in p.example_events[:2]:
                lines.append(f"#   {ex}")
            lines.append("")
            for instruction_line in p.proposed_instruction.splitlines():
                lines.append(f"+ {instruction_line}")
        return "\n".join(lines)

    def apply(self) -> str:
        """Return the new file content with all proposed additions appended."""
        additions = "\n\n".join(p.proposed_instruction for p in self.patterns)
        separator = "\n\n---\n\n" if self.existing_content.strip() else ""
        return self.existing_content.rstrip() + separator + additions + "\n"


# ---------------------------------------------------------------------------
# Core optimize function
# ---------------------------------------------------------------------------

def run_optimize(
    store: TraceStore,
    session_ids: list[str],
    target_file: str,
    llm_base_url: str = "",
    llm_model: str = "gpt-4o-mini",
    llm_api_key: str = "",
) -> OptimizeProposal:
    """Analyze sessions and return a proposal for the target file."""
    sessions: dict[str, list[TraceEvent]] = {}
    for sid in session_ids:
        try:
            sessions[sid] = store.load_events(sid)
        except Exception:
            continue

    target_path = Path(target_file)
    existing_content = target_path.read_text() if target_path.exists() else ""

    patterns: list[FailurePattern] = []

    # Try LLM first if configured
    if llm_base_url and llm_api_key and sessions:
        prompt = _build_llm_prompt(sessions, target_file, existing_content)
        response = _call_llm(prompt, llm_base_url, llm_model, llm_api_key)
        if response:
            patterns = _parse_llm_patterns(response, sessions)

    # Fall back to heuristics (also used to supplement LLM results)
    if not patterns:
        patterns = _extract_patterns_heuristic(sessions)

    return OptimizeProposal(
        target_file=target_file,
        patterns=patterns,
        existing_content=existing_content,
    )


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------

def print_proposal(proposal: OptimizeProposal, out: TextIO = sys.stdout) -> None:
    w = out.write
    n_sessions = sum(len(p.session_ids) for p in proposal.patterns)
    w(f"\nagent-strace optimize — {len(proposal.patterns)} cluster(s) from {n_sessions} session(s)\n")
    w(f"Target: {proposal.target_file}\n")
    w("─" * 60 + "\n")

    if not proposal.patterns:
        w("No failure patterns detected. No changes proposed.\n\n")
        return

    for i, p in enumerate(proposal.patterns, 1):
        w(f"\nCluster {i}: {p.name} ({len(p.session_ids)} session(s))\n")
        w(f"  {p.description}\n")
        for ex in p.example_events[:2]:
            w(f"  Example: {ex}\n")
        w("\n  Proposed addition:\n")
        for line in p.proposed_instruction.splitlines():
            w(f"  + {line}\n")

    w("\n─" * 60 + "\n")
    w(f"Apply? Run with --apply to write changes to {proposal.target_file}\n\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_sessions(store: TraceStore, args: argparse.Namespace) -> list[str]:
    """Return session IDs from --session, --dataset, or latest."""
    session_arg = getattr(args, "session_id", None)
    dataset_arg = getattr(args, "dataset", None)

    if dataset_arg:
        from .eval.dataset import list_entries
        ds_path = Path(dataset_arg)
        if not ds_path.exists():
            # Try default datasets dir
            ds_path = store.base_dir / "datasets" / f"{dataset_arg}.jsonl"
        entries = list_entries(ds_path)
        return [e.session_id for e in entries]

    if session_arg:
        found = store.find_session(session_arg)
        return [found] if found else []

    # Default: latest session
    latest = store.get_latest_session_id()
    return [latest] if latest else []


def cmd_optimize(args: argparse.Namespace) -> int:
    import os
    store = TraceStore(args.trace_dir)
    target = getattr(args, "target", "AGENTS.md") or "AGENTS.md"
    dry_run = getattr(args, "dry_run", False)
    apply_flag = getattr(args, "apply", False)

    session_ids = _resolve_sessions(store, args)
    if not session_ids:
        sys.stderr.write("No sessions found.\n")
        return 1

    # LLM config from env or args
    base_url = (
        getattr(args, "base_url", None)
        or os.environ.get("OPENAI_BASE_URL", "")
        or os.environ.get("AGENT_STRACE_LLM_URL", "")
    )
    model = (
        getattr(args, "model", None)
        or os.environ.get("AGENT_STRACE_LLM_MODEL", "gpt-4o-mini")
    )
    api_key = (
        getattr(args, "api_key", None)
        or os.environ.get("OPENAI_API_KEY", "")
        or os.environ.get("AGENT_STRACE_LLM_KEY", "")
    )

    proposal = run_optimize(
        store,
        session_ids,
        target_file=target,
        llm_base_url=base_url,
        llm_model=model,
        llm_api_key=api_key,
    )

    if not proposal.patterns:
        sys.stdout.write("No failure patterns detected. No changes proposed.\n")
        return 0

    if dry_run or not apply_flag:
        print_proposal(proposal)
        if not apply_flag:
            return 0

    # Apply
    new_content = proposal.apply()
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(new_content)
    sys.stdout.write(f"Written {len(proposal.patterns)} addition(s) to {target}\n")
    return 0
