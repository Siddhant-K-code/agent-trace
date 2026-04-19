"""CLI entry point.

Usage:
    agent-strace record [--redact] -- <server-command> [args...]
    agent-strace record-http [--redact] --url <remote-url> [--port <local-port>]
    agent-strace setup [--redact] [--global]
    agent-strace hook <event>
    agent-strace replay [session-id]
    agent-strace list
    agent-strace inspect <session-id>
    agent-strace export <session-id> [--format json|csv|otlp]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time

from . import __version__
from .hooks import hook_main
from .http_proxy import HTTPProxyServer
from .annotate import cmd_annotate
from .audit import cmd_audit
from .cost import cmd_cost
from .curve import cmd_curve
from .dashboard import cmd_dashboard
from .shadow_ai import cmd_audit_tools
from .inflation import cmd_inflation
from .diff import cmd_diff
from .eval import cmd_eval
from .explain import cmd_explain
from .jsonl_import import cmd_import
from .policy import cmd_policy
from .postmortem import cmd_postmortem
from .share import cmd_share
from .token_budget import cmd_token_budget
from .watch import cmd_watch
from .why import cmd_why
from .models import EventType, SessionMeta, TraceEvent
from .proxy import MCPProxy
from .replay import format_event, format_summary, list_sessions, replay_session
from .store import TraceStore
from .subagent import cmd_replay_tree, cmd_stats_tree
from .why import cmd_why


def _print_live_event(event: TraceEvent) -> None:
    """Print event to stderr during recording."""
    line = format_event(event)
    sys.stderr.write(f"\r{line}\n")
    sys.stderr.flush()


def cmd_record(args: argparse.Namespace) -> int:
    """Record an MCP server session."""
    store = TraceStore(args.trace_dir)

    meta = SessionMeta(
        agent_name=args.name or "",
        command=" ".join(args.command),
    )
    store.create_session(meta)

    if not args.quiet:
        sys.stderr.write(
            f"agent-strace: recording session {meta.session_id}\n"
            f"agent-strace: command: {' '.join(args.command)}\n"
        )

    on_event = _print_live_event if args.verbose else None

    proxy = MCPProxy(
        server_command=args.command,
        store=store,
        session_meta=meta,
        on_event=on_event,
        redact=args.redact,
    )

    returncode = proxy.run()

    if not args.quiet:
        sys.stderr.write(
            f"\nagent-strace: session {meta.session_id} complete\n"
            f"agent-strace: {meta.tool_calls} tool calls, "
            f"{meta.llm_requests} llm requests, "
            f"{meta.errors} errors\n"
            f"agent-strace: replay with: agent-trace replay {meta.session_id}\n"
        )

    return returncode


def cmd_record_http(args: argparse.Namespace) -> int:
    """Record a remote MCP server session over HTTP/SSE."""
    store = TraceStore(args.trace_dir)

    meta = SessionMeta(
        agent_name=args.name or "",
        command=f"http-proxy -> {args.url}",
    )
    store.create_session(meta)

    if not args.quiet:
        sys.stderr.write(
            f"agent-strace: recording HTTP session {meta.session_id}\n"
            f"agent-strace: proxying http://127.0.0.1:{args.port} -> {args.url}\n"
        )

    on_event = _print_live_event if args.verbose else None

    proxy = HTTPProxyServer(
        remote_url=args.url,
        local_port=args.port,
        store=store,
        session_meta=meta,
        on_event=on_event,
        redact=args.redact,
    )

    proxy.run()

    if not args.quiet:
        sys.stderr.write(
            f"\nagent-strace: session {meta.session_id} complete\n"
            f"agent-strace: {meta.tool_calls} tool calls, "
            f"{meta.llm_requests} llm requests, "
            f"{meta.errors} errors\n"
            f"agent-strace: replay with: agent-strace replay {meta.session_id}\n"
        )

    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    """Replay a recorded session."""
    # Delegate to tree replay when subagent flags are set
    if getattr(args, "expand_subagents", False) or getattr(args, "tree", False):
        return cmd_replay_tree(args)

    store = TraceStore(args.trace_dir)

    session_id = args.session_id
    if not session_id:
        session_id = store.get_latest_session_id()
        if not session_id:
            sys.stderr.write("No sessions found.\n")
            return 1

    # support prefix matching
    if not store.session_exists(session_id):
        found = store.find_session(session_id)
        if found:
            session_id = found
        else:
            sys.stderr.write(f"Session not found: {session_id}\n")
            return 1

    event_filter = None
    if args.filter:
        try:
            event_filter = {EventType(f) for f in args.filter.split(",")}
        except ValueError as e:
            sys.stderr.write(f"Invalid filter: {e}\n")
            return 1

    replay_session(
        store=store,
        session_id=session_id,
        event_filter=event_filter,
        speed=args.speed,
        live=args.live,
    )
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List all recorded sessions."""
    store = TraceStore(args.trace_dir)
    list_sessions(store)
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Inspect a session: show full event data as JSON."""
    store = TraceStore(args.trace_dir)

    session_id = args.session_id
    if not store.session_exists(session_id):
        found = store.find_session(session_id)
        if found:
            session_id = found
        else:
            sys.stderr.write(f"Session not found: {session_id}\n")
            return 1

    meta = store.load_meta(session_id)
    events = store.load_events(session_id)

    output = {
        "session": json.loads(meta.to_json()),
        "events": [json.loads(e.to_json()) for e in events],
    }

    sys.stdout.write(json.dumps(output, indent=2) + "\n")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Export a session to JSON, CSV, or OTLP."""
    store = TraceStore(args.trace_dir)

    session_id = args.session_id
    if not store.session_exists(session_id):
        found = store.find_session(session_id)
        if found:
            session_id = found
        else:
            sys.stderr.write(f"Session not found: {session_id}\n")
            return 1

    events = store.load_events(session_id)

    if args.format == "json":
        output = [json.loads(e.to_json()) for e in events]
        sys.stdout.write(json.dumps(output, indent=2) + "\n")

    elif args.format == "csv":
        writer = csv.writer(sys.stdout)
        writer.writerow(["timestamp", "event_type", "event_id", "parent_id", "duration_ms", "data"])
        for e in events:
            writer.writerow([
                e.timestamp,
                e.event_type.value,
                e.event_id,
                e.parent_id,
                e.duration_ms or "",
                json.dumps(e.data),
            ])

    elif args.format == "ndjson":
        for e in events:
            sys.stdout.write(e.to_json() + "\n")

    elif args.format == "otlp":
        from .otlp import export_otlp, session_to_otlp

        endpoint = args.endpoint
        if not endpoint:
            # No endpoint: dump OTLP JSON to stdout
            meta = store.load_meta(session_id)
            payload = session_to_otlp(meta, events, service_name=args.service_name)
            sys.stdout.write(json.dumps(payload, indent=2) + "\n")
            return 0

        # Build headers from --header flags
        headers = {}
        for h in (args.header or []):
            if ":" in h:
                key, val = h.split(":", 1)
                headers[key.strip()] = val.strip()

        ok = export_otlp(
            store=store,
            session_id=session_id,
            endpoint=endpoint,
            headers=headers,
            service_name=args.service_name,
        )
        return 0 if ok else 1

    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    """Show statistics for a session."""
    if getattr(args, "include_subagents", False):
        return cmd_stats_tree(args)

    store = TraceStore(args.trace_dir)

    session_id = args.session_id
    if not session_id:
        session_id = store.get_latest_session_id()
        if not session_id:
            sys.stderr.write("No sessions found.\n")
            return 1

    if not store.session_exists(session_id):
        found = store.find_session(session_id)
        if found:
            session_id = found
        else:
            sys.stderr.write(f"Session not found: {session_id}\n")
            return 1

    events = store.load_events(session_id)
    meta = store.load_meta(session_id)

    # tool call frequency
    tool_counts: dict[str, int] = {}
    tool_durations: dict[str, list[float]] = {}
    result_events = {e.parent_id: e for e in events if e.event_type == EventType.TOOL_RESULT}

    for e in events:
        if e.event_type == EventType.TOOL_CALL:
            name = e.data.get("tool_name", "unknown")
            tool_counts[name] = tool_counts.get(name, 0) + 1
            # find matching result
            result = result_events.get(e.event_id)
            if result and result.duration_ms:
                tool_durations.setdefault(name, []).append(result.duration_ms)

    print(format_summary(meta))
    print()

    if tool_counts:
        print(f"  Tool Call Frequency:")
        for name, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
            avg_ms = ""
            if name in tool_durations:
                durations = tool_durations[name]
                avg = sum(durations) / len(durations)
                avg_ms = f"  avg: {avg:.0f}ms"
            print(f"    {name:<30} {count:>4}x{avg_ms}")

    # error summary
    errors = [e for e in events if e.event_type == EventType.ERROR]
    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for e in errors:
            msg = e.data.get("message", "unknown")
            print(f"    {msg[:80]}")

    print()
    return 0


def cmd_setup(args: argparse.Namespace) -> None:
    """Generate Claude Code hooks configuration."""
    redact_env = ""
    if args.redact:
        redact_env = "AGENT_TRACE_REDACT=1 "

    cmd_prefix = f"{redact_env}agent-strace hook"

    config = {
        "hooks": {
            "UserPromptSubmit": [{
                "hooks": [{"type": "command", "command": f"{cmd_prefix} user-prompt"}],
            }],
            "PreToolUse": [{
                "matcher": "",
                "hooks": [{"type": "command", "command": f"{cmd_prefix} pre-tool"}],
            }],
            "PostToolUse": [{
                "matcher": "",
                "hooks": [{"type": "command", "command": f"{cmd_prefix} post-tool"}],
            }],
            "PostToolUseFailure": [{
                "matcher": "",
                "hooks": [{"type": "command", "command": f"{cmd_prefix} post-tool-failure"}],
            }],
            "Stop": [{
                "hooks": [{"type": "command", "command": f"{cmd_prefix} stop"}],
            }],
            "SessionStart": [{
                "hooks": [{"type": "command", "command": f"{cmd_prefix} session-start"}],
            }],
            "SessionEnd": [{
                "hooks": [{"type": "command", "command": f"{cmd_prefix} session-end"}],
            }],
        }
    }

    output = json.dumps(config, indent=2)

    if args.global_config:
        sys.stderr.write("Add this to ~/.claude/settings.json:\n\n")
    else:
        sys.stderr.write("Add this to .claude/settings.json:\n\n")

    sys.stdout.write(output + "\n")
    sys.stderr.write(
        "\nThis captures the full Claude Code session: user prompts, "
        "assistant responses, and every tool call (Bash, Edit, Write, "
        "Read, Agent, and all MCP tools).\n"
        "Replay with: agent-strace replay\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-strace",
        description="strace for AI agents. Capture and replay every tool call.",
    )
    parser.add_argument("--version", action="version", version=f"agent-strace {__version__}")
    parser.add_argument(
        "--trace-dir",
        default=".agent-traces",
        help="directory to store traces (default: .agent-traces)",
    )

    sub = parser.add_subparsers(dest="command")

    # record
    p_record = sub.add_parser("record", help="record an MCP server session (stdio)")
    p_record.add_argument("--name", "-n", help="name for this agent/session")
    p_record.add_argument("--redact", action="store_true", help="redact secrets from trace data")
    p_record.add_argument("--verbose", "-v", action="store_true", help="print events to stderr during recording")
    p_record.add_argument("--quiet", "-q", action="store_true", help="suppress all output except errors")
    p_record.add_argument("command", nargs=argparse.REMAINDER, help="MCP server command to run")

    # record-http
    p_record_http = sub.add_parser("record-http", help="record a remote MCP server session (HTTP/SSE)")
    p_record_http.add_argument("--url", "-u", required=True, help="remote MCP server URL")
    p_record_http.add_argument("--port", "-p", type=int, default=5100, help="local proxy port (default: 5100)")
    p_record_http.add_argument("--name", "-n", help="name for this agent/session")
    p_record_http.add_argument("--redact", action="store_true", help="redact secrets from trace data")
    p_record_http.add_argument("--verbose", "-v", action="store_true", help="print events to stderr during recording")
    p_record_http.add_argument("--quiet", "-q", action="store_true", help="suppress all output except errors")

    # replay
    p_replay = sub.add_parser("replay", help="replay a recorded session")
    p_replay.add_argument("session_id", nargs="?", help="session ID (default: latest)")
    p_replay.add_argument("--filter", "-f", help="comma-separated event types to show")
    p_replay.add_argument("--speed", "-s", type=float, default=0, help="replay speed multiplier (0=instant)")
    p_replay.add_argument("--live", "-l", action="store_true", help="replay with timing delays")
    p_replay.add_argument("--expand-subagents", action="store_true",
                          help="inline subagent sessions under their parent tool_call")
    p_replay.add_argument("--tree", action="store_true",
                          help="show session hierarchy tree without full event replay")

    # list
    sub.add_parser("list", help="list all recorded sessions")

    # inspect
    p_inspect = sub.add_parser("inspect", help="inspect a session as raw JSON")
    p_inspect.add_argument("session_id", help="session ID or prefix")

    # export
    p_export = sub.add_parser("export", help="export a session")
    p_export.add_argument("session_id", help="session ID or prefix")
    p_export.add_argument("--format", choices=["json", "csv", "ndjson", "otlp"], default="json")
    p_export.add_argument("--endpoint", help="OTLP collector URL (e.g. http://localhost:4318)")
    p_export.add_argument("--header", action="append", help="HTTP header for OTLP (e.g. 'x-honeycomb-team: KEY')")
    p_export.add_argument("--service-name", default="agent-trace", help="OTel service name (default: agent-trace)")

    # stats
    p_stats = sub.add_parser("stats", help="show session statistics")
    p_stats.add_argument("session_id", nargs="?", help="session ID (default: latest)")
    p_stats.add_argument("--include-subagents", action="store_true",
                         help="roll up stats across all subagent sessions")

    # hook (called by Claude Code hooks system)
    p_hook = sub.add_parser("hook", help="handle a Claude Code hook event (internal)")
    p_hook.add_argument("event", nargs="?", help="hook event: session-start, session-end, pre-tool, post-tool, post-tool-failure")

    # setup (generate Claude Code hooks config)
    p_setup = sub.add_parser("setup", help="generate Claude Code hooks configuration")
    p_setup.add_argument("--redact", action="store_true", help="enable secret redaction")
    p_setup.add_argument("--global", dest="global_config", action="store_true", help="output config for ~/.claude/settings.json (all projects)")

    # import (Claude Code JSONL session logs)
    p_import = sub.add_parser("import", help="import a Claude Code JSONL session log")
    p_import.add_argument("path", nargs="?", help="path to .jsonl session file")
    p_import.add_argument("--discover", action="store_true", help="list available Claude Code sessions")
    p_import.add_argument("--claude-dir", default="~/.claude", help="Claude config directory (default: ~/.claude)")

    # explain
    p_explain = sub.add_parser("explain", help="explain a session in plain English")
    p_explain.add_argument("session_id", nargs="?", help="session ID or prefix (default: latest)")

    # diff
    p_diff = sub.add_parser("diff", help="compare two sessions structurally")
    p_diff.add_argument("session_a", help="first session ID or prefix")
    p_diff.add_argument("session_b", help="second session ID or prefix")

    # why
    p_why = sub.add_parser("why", help="trace the causal chain for a specific event")
    p_why.add_argument("session_id", nargs="?", help="session ID or prefix (default: latest)")
    p_why.add_argument("event_number", type=int, help="1-based event number (from replay output)")

    # cost
    p_cost = sub.add_parser("cost", help="estimate token cost for a session")
    p_cost.add_argument("session_id", nargs="?", help="session ID or prefix (default: latest)")
    p_cost.add_argument("--model", default="sonnet",
                        choices=["sonnet", "opus", "haiku", "gpt4", "gpt4o"],
                        help="model pricing to use (default: sonnet)")
    p_cost.add_argument("--input-price", type=float, dest="input_price",
                        help="custom input price per 1M tokens (overrides --model)")
    p_cost.add_argument("--output-price", type=float, dest="output_price",
                        help="custom output price per 1M tokens (overrides --model)")

    # audit
    p_audit = sub.add_parser("audit", help="check session tool calls against a policy file")
    p_audit.add_argument("session_id", nargs="?", help="session ID or prefix (default: latest)")
    p_audit.add_argument("--policy", default=".agent-scope.json",
                         help="path to policy file (default: .agent-scope.json)")

    # share
    p_share = sub.add_parser("share", help="generate a self-contained HTML replay of a session")
    p_share.add_argument("session_id", nargs="?", help="session ID or prefix (default: latest)")
    p_share.add_argument("--output", "-o", help="output file path (default: session-<id>.html)")
    p_share.add_argument("--stdout", action="store_true", help="write HTML to stdout instead of a file")
    p_share.add_argument("--open", action="store_true", help="open the HTML file in the browser after creation")
    p_share.add_argument("--postmortem", action="store_true", help="include postmortem analysis in the HTML")

    # postmortem
    p_postmortem = sub.add_parser("postmortem", help="generate a structured postmortem for a failed session")
    p_postmortem.add_argument("session_id", nargs="?", help="session ID or prefix (default: latest)")
    p_postmortem.add_argument("--agents-md", default="AGENTS.md",
                              help="path to AGENTS.md for violation detection (default: AGENTS.md)")

    # eval
    p_eval = sub.add_parser("eval", help="score, compare, and regression-test agent sessions")
    eval_sub = p_eval.add_subparsers(dest="eval_command")

    p_eval_run = eval_sub.add_parser("run", help="score a session against configured scorers")
    p_eval_run.add_argument("session_id", nargs="?", help="session ID or prefix (default: latest)")
    p_eval_run.add_argument("--format", choices=["table", "json"], default="table")
    p_eval_run.add_argument("--config", default=".agent-evals.yaml",
                            help="eval config file (default: .agent-evals.yaml)")

    p_eval_compare = eval_sub.add_parser("compare", help="compare two sessions across all scorers")
    p_eval_compare.add_argument("session_a", help="first session ID or prefix")
    p_eval_compare.add_argument("session_b", help="second session ID or prefix")
    p_eval_compare.add_argument("--config", default=".agent-evals.yaml")

    p_eval_ci = eval_sub.add_parser("ci", help="run evals and exit 1 if any scorer fails")
    p_eval_ci.add_argument("session_id", nargs="?", help="session ID or prefix (default: latest)")
    p_eval_ci.add_argument("--config", default=".agent-evals.yaml")

    p_eval_dataset = eval_sub.add_parser("dataset", help="manage eval datasets")
    dataset_sub = p_eval_dataset.add_subparsers(dest="dataset_command")
    p_ds_add = dataset_sub.add_parser("add", help="add a session to the dataset")
    p_ds_add.add_argument("--session", required=True, help="session ID to add")
    p_ds_add.add_argument("--label", default="", help="human-readable label")
    p_ds_add.add_argument("--dataset", default=".agent-traces/datasets/default.jsonl")
    dataset_sub.add_parser("list", help="list dataset entries").add_argument(
        "--dataset", default=".agent-traces/datasets/default.jsonl"
    )
    p_ds_export = dataset_sub.add_parser("export", help="export dataset to JSONL")
    p_ds_export.add_argument("--dataset", default=".agent-traces/datasets/default.jsonl")

    # watch
    p_watch = sub.add_parser("watch", help="monitor a live session with circuit breakers")
    p_watch.add_argument("session_id", nargs="?", help="session ID to watch (default: latest active)")
    p_watch.add_argument("--max-retries", type=int, default=5, help="max retries before alert (default: 5)")
    p_watch.add_argument("--max-cost", type=float, default=10.0, help="max cost in dollars (default: 10)")
    p_watch.add_argument("--max-duration", type=int, default=1800, help="max duration in seconds (default: 1800)")
    p_watch.add_argument("--on-violation", choices=["terminal", "file", "kill"], default="terminal",
                         help="action on violation (default: terminal)")
    p_watch.add_argument("--webhook", help="webhook URL for alerts")
    p_watch.add_argument("--config", help="path to .agent-watch.json config file")
    p_watch.add_argument("--max-context-pct", type=int, default=90, dest="max_context_pct",
                         help="alert when context window is this %% full (default: 90)")
    p_watch.add_argument("--rules", metavar="RULES_FILE",
                         help="YAML/JSON rules file for rule-based kill switch (nanny mode)")
    p_watch.add_argument("--dry-run", action="store_true", dest="dry_run",
                         help="evaluate rules without taking action (for testing)")

    # policy
    p_policy = sub.add_parser("policy", help="suggest a .agent-scope.json policy from observed traces")
    p_policy.add_argument("session_ids", nargs="*", help="session IDs to analyse (default: all)")
    p_policy.add_argument("--output", "-o", default=".agent-scope.json",
                          help="output path (default: .agent-scope.json)")
    p_policy.add_argument("--dry-run", action="store_true", help="print policy without writing")

    # dashboard
    p_dash = sub.add_parser("dashboard", help="aggregate view across sessions")
    p_dash.add_argument("--limit", type=int, default=50, help="max sessions to show (default: 50)")
    p_dash.add_argument("--agent", default="", help="filter by agent name")
    p_dash.add_argument("--output", "-o", help="write HTML dashboard to this file")

    # annotate
    p_ann = sub.add_parser("annotate", help="attach notes, labels, and bookmarks to trace events")
    p_ann.add_argument("session_id", nargs="?", help="session ID or prefix (default: latest)")
    p_ann.add_argument("--event", help="event ID to annotate")
    p_ann.add_argument("--at", help="time offset to annotate (e.g. 2m14s)")
    p_ann.add_argument("--note", help="text note to attach")
    p_ann.add_argument("--label", help="label chip (e.g. root-cause, decision, retry)")
    p_ann.add_argument("--author", help="author name or email")
    p_ann.add_argument("--list", action="store_true", help="list all annotations for the session")
    p_ann.add_argument("--delete", metavar="ANNOTATION_ID", help="delete an annotation by ID")

    # token-budget
    p_tb = sub.add_parser("token-budget", help="show context window usage for a session")
    p_tb.add_argument("session_id", nargs="?", help="session ID or prefix (default: latest)")
    p_tb.add_argument("--warning-threshold", type=float, default=0.9, dest="warning_threshold",
                      help="warn at this fraction of context limit (default: 0.9)")

    # audit-tools (shadow AI detection)
    p_at = sub.add_parser("audit-tools", help="detect which AI tools are active in a repository")
    p_at.add_argument("--repo", default=".", help="path to git repository (default: .)")
    p_at.add_argument("--since", default="90 days ago",
                      help="how far back to scan git history (default: '90 days ago')")
    p_at.add_argument("--approved", default="",
                      help="comma-separated list of approved tool names")

    # curve (personal cost curve)
    p_curve = sub.add_parser("curve", help="show your personal agent cost-efficiency curve by task type")
    p_curve.add_argument("--min-sessions", type=int, default=20, dest="min_sessions",
                         help="minimum sessions for meaningful analysis (default: 20)")
    p_curve.add_argument("--export", choices=["csv"], default="",
                         help="export results (csv)")

    # inflation (token inflation calculator)
    p_inf = sub.add_parser("inflation", help="measure tokenizer cost impact across model versions")
    p_inf.add_argument("--compare", default="claude-opus-4-6,claude-opus-4-7",
                       help="comma-separated model pair to compare (default: claude-opus-4-6,claude-opus-4-7)")
    p_inf.add_argument("--sessions", type=int, default=30,
                       help="number of recent sessions to analyse (default: 30)")
    p_inf.add_argument("--daily-sessions", type=float, default=8.0, dest="daily_sessions",
                       help="assumed sessions per day for projections (default: 8)")

    # diff --semantic and --eval-config flags (extend existing diff parser)
    p_diff.add_argument("--semantic", action="store_true",
                        help="semantic outcome-level diff (files, cost, errors)")
    p_diff.add_argument("--compare", action="store_true",
                        help="rich side-by-side comparison table with verdict")
    p_diff.add_argument("--eval-config", default=".agent-evals.yaml", dest="eval_config",
                        help="eval config for score comparison")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # hook subcommand is handled separately (reads stdin)
    if args.command == "hook":
        hook_main([args.event] if args.event else [])
        sys.exit(0)

    if args.command == "setup":
        cmd_setup(args)
        sys.exit(0)

    handlers = {
        "record": cmd_record,
        "record-http": cmd_record_http,
        "replay": cmd_replay,
        "list": cmd_list,
        "inspect": cmd_inspect,
        "export": cmd_export,
        "stats": cmd_stats,
        "import": cmd_import,
        "explain": cmd_explain,
        "cost": cmd_cost,
        "diff": cmd_diff,
        "why": cmd_why,
        "audit": cmd_audit,
        "share": cmd_share,
        "postmortem": cmd_postmortem,
        "eval": cmd_eval,
        "watch": cmd_watch,
        "policy": cmd_policy,
        "dashboard": cmd_dashboard,
        "annotate": cmd_annotate,
        "token-budget": cmd_token_budget,
        "audit-tools": cmd_audit_tools,
        "curve": cmd_curve,
        "inflation": cmd_inflation,
    }

    handler = handlers.get(args.command)
    if handler:
        sys.exit(handler(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
