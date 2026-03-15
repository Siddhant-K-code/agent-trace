"""CLI entry point.

Usage:
    agent-strace record -- <server-command> [args...]
    agent-strace replay [session-id]
    agent-strace list
    agent-strace inspect <session-id>
    agent-strace export <session-id> [--format json|csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time

from . import __version__
from .models import EventType, SessionMeta, TraceEvent
from .proxy import MCPProxy
from .replay import format_event, format_summary, list_sessions, replay_session
from .store import TraceStore


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


def cmd_replay(args: argparse.Namespace) -> int:
    """Replay a recorded session."""
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
    """Export a session to JSON or CSV."""
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

    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    """Show statistics for a session."""
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
    p_record = sub.add_parser("record", help="record an MCP server session")
    p_record.add_argument("--name", "-n", help="name for this agent/session")
    p_record.add_argument("--verbose", "-v", action="store_true", help="print events to stderr during recording")
    p_record.add_argument("--quiet", "-q", action="store_true", help="suppress all output except errors")
    p_record.add_argument("command", nargs=argparse.REMAINDER, help="MCP server command to run")

    # replay
    p_replay = sub.add_parser("replay", help="replay a recorded session")
    p_replay.add_argument("session_id", nargs="?", help="session ID (default: latest)")
    p_replay.add_argument("--filter", "-f", help="comma-separated event types to show")
    p_replay.add_argument("--speed", "-s", type=float, default=0, help="replay speed multiplier (0=instant)")
    p_replay.add_argument("--live", "-l", action="store_true", help="replay with timing delays")

    # list
    sub.add_parser("list", help="list all recorded sessions")

    # inspect
    p_inspect = sub.add_parser("inspect", help="inspect a session as raw JSON")
    p_inspect.add_argument("session_id", help="session ID or prefix")

    # export
    p_export = sub.add_parser("export", help="export a session")
    p_export.add_argument("session_id", help="session ID or prefix")
    p_export.add_argument("--format", choices=["json", "csv", "ndjson"], default="json")

    # stats
    p_stats = sub.add_parser("stats", help="show session statistics")
    p_stats.add_argument("session_id", nargs="?", help="session ID (default: latest)")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    handlers = {
        "record": cmd_record,
        "replay": cmd_replay,
        "list": cmd_list,
        "inspect": cmd_inspect,
        "export": cmd_export,
        "stats": cmd_stats,
    }

    handler = handlers.get(args.command)
    if handler:
        sys.exit(handler(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
