"""CLI entry point.

Usage:
    agent-strace record [--no-redact] -- <server-command> [args...]
    agent-strace record-http [--no-redact] --url <remote-url> [--port <local-port>]
    agent-strace setup [--no-redact] [--global]
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
import os
import sys
import time
from pathlib import Path

from . import __version__
from .hooks import hook_main
from .http_proxy import HTTPProxyServer
from .a2a import cmd_a2a_tree
from .mcp_server import cmd_mcp
from .annotate import cmd_annotate
from .approval import cmd_approval
from .rbac import cmd_rbac
from .iac import cmd_apply, cmd_config_diff
from .sso import cmd_auth
from .baseline import cmd_baseline
from .compliance import cmd_audit_readiness, cmd_compliance, cmd_export_eu_ai_act, cmd_verify_export
from .drift import cmd_drift, cmd_fingerprint
from .identity import cmd_identity
from .workspace import cmd_workspace
from .langfuse_export import cmd_export_scores
from .oncall import cmd_oncall
from .optimize import cmd_optimize
from .freshness import cmd_freshness
from .standup import cmd_standup
from .audit import cmd_audit, verify_chain
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
from .project_budget import enforce_new_session_budget, load_project_budget_config
from .share import cmd_share
from .token_budget import cmd_token_budget
from .anonymize import cmd_anonymize_export
from .integrations import detect_and_instrument, _INTEGRATIONS
from .budget_report import cmd_budget_report
from .team_report import cmd_team_report
from .compare import cmd_compare
from .freeze import cmd_freeze, cmd_regression
from .timeline import cmd_timeline
from .config_watch import cmd_config_watch
from .lint import cmd_lint
from .mcp_scan import cmd_mcp_scan
from .retention import cmd_retention
from .sample import cmd_sample
from .server import cmd_server
from .watch import cmd_watch
from .why import cmd_why
from .models import EventType, SessionMeta, TraceEvent
from .proxy import MCPProxy
from .replay import format_event, format_summary, list_sessions, replay_session
from .store import TraceStore
from .subagent import cmd_replay_tree, cmd_stats_tree, cmd_tree
from .why import cmd_why


def _print_live_event(event: TraceEvent) -> None:
    """Print event to stderr during recording."""
    line = format_event(event)
    sys.stderr.write(f"\r{line}\n")
    sys.stderr.flush()


def _redact_setting(args: argparse.Namespace) -> bool | None:
    """Return explicit redaction setting, or None to use env/defaults."""
    if getattr(args, "no_redact", False):
        return False
    if getattr(args, "redact", False):
        return True
    return None


def _parent_session_id(args: argparse.Namespace) -> str:
    return (
        getattr(args, "parent", None)
        or os.environ.get("AGENT_STRACE_PARENT_SESSION", "")
    )


def _resolve_parent_session_id(store: TraceStore, args: argparse.Namespace) -> str:
    raw = _parent_session_id(args)
    if not raw:
        return ""
    return store.find_session(raw) or raw


def _parent_event_id() -> str:
    return os.environ.get("AGENT_STRACE_PARENT_EVENT", "")


def _parent_depth(store: TraceStore, parent_session_id: str) -> int:
    if not parent_session_id:
        return 0
    try:
        parent_meta = store.load_meta(parent_session_id)
        return parent_meta.depth + 1
    except Exception:
        return 1


def cmd_record(args: argparse.Namespace) -> int:
    """Record an MCP server session."""
    store = TraceStore(args.trace_dir, redact=_redact_setting(args))
    budget_config = load_project_budget_config()
    if budget_config.enabled and not enforce_new_session_budget(
        store, budget_config, sys.stderr
    ):
        return 1

    server_cmd = args.server_cmd
    # Strip leading '--' separator added by argparse REMAINDER
    if server_cmd and server_cmd[0] == "--":
        server_cmd = server_cmd[1:]

    parent_session_id = _resolve_parent_session_id(store, args)
    meta = SessionMeta(
        agent_name=args.name or "",
        command=" ".join(server_cmd),
        parent_session_id=parent_session_id,
        parent_event_id=_parent_event_id(),
        depth=_parent_depth(store, parent_session_id),
    )
    store.create_session(meta)

    if not args.quiet:
        sys.stderr.write(
            f"agent-strace: recording session {meta.session_id}\n"
            f"agent-strace: command: {' '.join(server_cmd)}\n"
        )

    on_event = _print_live_event if args.verbose else None

    proxy = MCPProxy(
        server_command=server_cmd,
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
    store = TraceStore(args.trace_dir, redact=_redact_setting(args))
    budget_config = load_project_budget_config()
    if budget_config.enabled and not enforce_new_session_budget(
        store, budget_config, sys.stderr
    ):
        return 1

    parent_session_id = _resolve_parent_session_id(store, args)
    meta = SessionMeta(
        agent_name=args.name or "",
        command=f"http-proxy -> {args.url}",
        parent_session_id=parent_session_id,
        parent_event_id=_parent_event_id(),
        depth=_parent_depth(store, parent_session_id),
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

    fmt = getattr(args, "format", "terminal") or "terminal"

    # --diff: side-by-side HTML diff viewer
    diff_session_id = getattr(args, "diff", "") or ""
    if diff_session_id:
        from .replay import replay_to_html_diff
        diff_full = store.find_session(diff_session_id) or diff_session_id
        output_path = getattr(args, "output", "") or \
            f"diff-{session_id[:8]}-vs-{diff_full[:8]}.html"
        replay_to_html_diff(store, session_id, diff_full, output_path=output_path)
        sys.stdout.write(f"Diff HTML written to {output_path}\n")
        return 0

    if fmt == "html":
        from .replay import replay_to_html
        output_path = getattr(args, "output", "") or f"session-{session_id[:12]}.html"
        replay_to_html(store, session_id, output_path=output_path)
        sys.stdout.write(f"HTML replay written to {output_path}\n")
        return 0

    replay_session(
        store=store,
        session_id=session_id,
        event_filter=event_filter,
        speed=args.speed,
        live=args.live,
        limit=getattr(args, "limit", None),
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
    # Route to Langfuse/OTLP export when --scores, --metrics, or --backend is set
    if getattr(args, "scores", False) or getattr(args, "metrics", False) or getattr(args, "backend", None):
        return cmd_export_scores(args)

    # Route to anonymized export when --anonymize is set
    if getattr(args, "anonymize", False):
        return cmd_anonymize_export(args)

    if getattr(args, "format", "") == "eu-ai-act":
        return cmd_export_eu_ai_act(args)
    store = TraceStore(args.trace_dir)

    session_id = args.session_id
    if not session_id:
        session_id = store.get_latest_session_id()
        if not session_id:
            sys.stderr.write("No sessions found.\n")
            return 1
    elif not store.session_exists(session_id):
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

    elif args.format in ("otlp", "otlp-genai"):
        from .otlp import export_otlp, session_to_otlp, session_to_otlp_genai

        use_genai = args.format == "otlp-genai"
        endpoint = args.endpoint

        # When --endpoint is set and format is plain otlp, default to otlp-genai
        # for better backend compatibility (backwards-compat: explicit --format otlp
        # always uses the legacy mapping)
        if endpoint and not use_genai:
            use_genai = False  # explicit --format otlp keeps legacy behaviour

        if not endpoint:
            # No endpoint: write OTLP JSON to --output file or stdout
            meta = store.load_meta(session_id)
            if use_genai:
                payload = session_to_otlp_genai(meta, events, service_name=args.service_name)
            else:
                payload = session_to_otlp(meta, events, service_name=args.service_name)
            output_path = getattr(args, "output", "") or ""
            if not output_path:
                fmt_suffix = "otlp-genai" if use_genai else "otlp"
                output_path = f"trace-{session_id[:12]}-{fmt_suffix}.json"
            with open(output_path, "w") as f:
                f.write(json.dumps(payload, indent=2) + "\n")
            sys.stderr.write(f"OTLP payload written to {output_path}\n")
            sys.stderr.write(f"Send to a collector: agent-strace export --format {args.format} --endpoint <url>\n")
            return 0

        # Build headers from --header flags
        headers = {}
        for h in (args.header or []):
            if ":" in h:
                key, val = h.split(":", 1)
                headers[key.strip()] = val.strip()

        if use_genai:
            # Export using GenAI conventions
            import urllib.request, urllib.error
            meta = store.load_meta(session_id)
            payload = session_to_otlp_genai(meta, events, service_name=args.service_name)
            body = json.dumps(payload).encode("utf-8")
            url = endpoint.rstrip("/") + "/v1/traces"
            req_headers = {"Content-Type": "application/json"}
            req_headers.update(headers)
            req = urllib.request.Request(url, data=body, headers=req_headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    ok = resp.status in (200, 202)
                    sys.stderr.write(f"Exported {len(events)} events to {url} (HTTP {resp.status})\n")
                    return 0 if ok else 1
            except Exception as exc:
                sys.stderr.write(f"OTLP GenAI export failed: {exc}\n")
                return 1
        else:
            ok = export_otlp(
                store=store,
                session_id=session_id,
                endpoint=endpoint,
                headers=headers,
                service_name=args.service_name,
            )
            return 0 if ok else 1

    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify a session hash chain or an exported EU AI Act package."""
    if getattr(args, "from_export", ""):
        return cmd_verify_export(args)

    store = TraceStore(args.trace_dir)
    session_id = getattr(args, "session_id", None) or store.get_latest_session_id()
    if not session_id:
        sys.stderr.write("No sessions found.\n")
        return 1
    full_id = store.find_session(session_id) or session_id
    if not store.session_exists(full_id):
        sys.stderr.write(f"Session not found: {session_id}\n")
        return 1

    result = verify_chain(store, full_id)
    if getattr(args, "format", "text") == "json":
        sys.stdout.write(json.dumps({
            "session_id": result.session_id,
            "ok": result.ok,
            "total_events": result.total_events,
            "broken_at": result.broken_at,
            "broken_event_id": result.broken_event_id,
        }, indent=2) + "\n")
    else:
        result.format(sys.stdout)
    return 0 if result.ok else 1


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


def _hook_command_prefix(args: argparse.Namespace, provider: str = "claude") -> str:
    redact_env = ""
    if args.no_redact:
        redact_env = "AGENT_TRACE_NO_REDACT=1 "
    elif args.redact:
        redact_env = "AGENT_TRACE_REDACT=1 "
    provider_arg = "" if provider == "claude" else f"--provider {provider} "
    return f"{redact_env}agent-strace hook {provider_arg}".rstrip()


def _claude_hooks_config(args: argparse.Namespace) -> dict:
    cmd_prefix = _hook_command_prefix(args, provider="claude")
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
    return config


def _codex_hooks_config(args: argparse.Namespace) -> dict:
    cmd_prefix = _hook_command_prefix(args, provider="codex")
    return {
        "hooks": {
            "SessionStart": [{
                "matcher": "startup|resume|clear|compact",
                "hooks": [{
                    "type": "command",
                    "command": f"{cmd_prefix} session-start",
                }],
            }],
            "UserPromptSubmit": [{
                "hooks": [{
                    "type": "command",
                    "command": f"{cmd_prefix} user-prompt",
                }],
            }],
            "PreToolUse": [{
                "matcher": ".*",
                "hooks": [{
                    "type": "command",
                    "command": f"{cmd_prefix} pre-tool",
                }],
            }],
            "PostToolUse": [{
                "matcher": ".*",
                "hooks": [{
                    "type": "command",
                    "command": f"{cmd_prefix} post-tool",
                }],
            }],
            "Stop": [{
                "hooks": [{
                    "type": "command",
                    "command": f"{cmd_prefix} stop",
                }],
            }],
        }
    }


def _gemini_hooks_config(args: argparse.Namespace) -> dict:
    cmd_prefix = _hook_command_prefix(args, provider="gemini")
    return {
        "hooks": {
            "SessionStart": [{
                "matcher": "*",
                "hooks": [{
                    "name": "agent-strace-session-start",
                    "type": "command",
                    "command": f"{cmd_prefix} session-start",
                    "timeout": 5000,
                }],
            }],
            "BeforeAgent": [{
                "matcher": "*",
                "hooks": [{
                    "name": "agent-strace-user-prompt",
                    "type": "command",
                    "command": f"{cmd_prefix} user-prompt",
                    "timeout": 5000,
                }],
            }],
            "BeforeTool": [{
                "matcher": "*",
                "hooks": [{
                    "name": "agent-strace-tool-call",
                    "type": "command",
                    "command": f"{cmd_prefix} pre-tool",
                    "timeout": 5000,
                }],
            }],
            "AfterTool": [{
                "matcher": "*",
                "hooks": [{
                    "name": "agent-strace-tool-result",
                    "type": "command",
                    "command": f"{cmd_prefix} post-tool",
                    "timeout": 5000,
                }],
            }],
            "AfterAgent": [{
                "matcher": "*",
                "hooks": [{
                    "name": "agent-strace-assistant-response",
                    "type": "command",
                    "command": f"{cmd_prefix} stop",
                    "timeout": 5000,
                }],
            }],
            "SessionEnd": [{
                "matcher": "*",
                "hooks": [{
                    "name": "agent-strace-session-end",
                    "type": "command",
                    "command": f"{cmd_prefix} session-end",
                    "timeout": 5000,
                }],
            }],
        }
    }


def _gemini_extension_manifest() -> dict:
    return {
        "name": "agent-strace",
        "version": __version__,
        "description": "Capture and replay Gemini CLI sessions with agent-strace",
    }


def _gemini_config_dir() -> Path:
    return Path(os.environ.get("GEMINI_CONFIG_DIR", "~/.gemini")).expanduser()


def _write_gemini_extension(args: argparse.Namespace) -> tuple[Path, Path]:
    extension_dir = _gemini_config_dir() / "extensions" / "agent-strace"
    hooks_dir = extension_dir / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = extension_dir / "gemini-extension.json"
    hooks_path = hooks_dir / "hooks.json"
    manifest_path.write_text(json.dumps(_gemini_extension_manifest(), indent=2) + "\n")
    hooks_path.write_text(json.dumps(_gemini_hooks_config(args), indent=2) + "\n")
    return manifest_path, hooks_path


def cmd_setup(args: argparse.Namespace) -> None:
    """Generate hooks configuration for supported agent CLIs."""
    cli = getattr(args, "cli", "claude") or "claude"

    configs: list[tuple[str, str, dict]] = []
    if cli in ("claude", "all"):
        configs.append(("Claude Code", "~/.claude/settings.json", _claude_hooks_config(args)))
    if cli in ("codex", "all"):
        configs.append(("OpenAI Codex", "~/.codex/hooks.json", _codex_hooks_config(args)))
    if cli in ("gemini", "all"):
        manifest_path, hooks_path = _write_gemini_extension(args)
        sys.stderr.write(
            f"Wrote Gemini CLI extension manifest: {manifest_path}\n"
            f"Wrote Gemini CLI hooks config: {hooks_path}\n"
        )

    for idx, (name, path, config) in enumerate(configs):
        if idx:
            sys.stdout.write("\n")
        sys.stderr.write(f"Add this to {path} for {name}:\n\n")
        sys.stdout.write(json.dumps(config, indent=2) + "\n")

    if cli == "gemini":
        sys.stdout.write(json.dumps(_gemini_hooks_config(args), indent=2) + "\n")

    sys.stderr.write(
        "\nThis captures full agent sessions: user prompts, assistant "
        "responses, and hook-visible tool calls.\n"
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
    record_redaction = p_record.add_mutually_exclusive_group()
    record_redaction.add_argument(
        "--redact",
        action="store_true",
        help="redact secrets from trace data (default)",
    )
    record_redaction.add_argument(
        "--no-redact",
        action="store_true",
        help="disable automatic secret redaction",
    )
    p_record.add_argument("--verbose", "-v", action="store_true", help="print events to stderr during recording")
    p_record.add_argument("--quiet", "-q", action="store_true", help="suppress all output except errors")
    p_record.add_argument("--parent", metavar="SESSION",
                          help="parent session ID for subagent correlation")
    p_record.add_argument("server_cmd", nargs=argparse.REMAINDER, help="MCP server command to run")

    # record-http
    p_record_http = sub.add_parser("record-http", help="record a remote MCP server session (HTTP/SSE)")
    p_record_http.add_argument("--url", "-u", required=True, help="remote MCP server URL")
    p_record_http.add_argument("--port", "-p", type=int, default=5100, help="local proxy port (default: 5100)")
    p_record_http.add_argument("--name", "-n", help="name for this agent/session")
    record_http_redaction = p_record_http.add_mutually_exclusive_group()
    record_http_redaction.add_argument(
        "--redact",
        action="store_true",
        help="redact secrets from trace data (default)",
    )
    record_http_redaction.add_argument(
        "--no-redact",
        action="store_true",
        help="disable automatic secret redaction",
    )
    p_record_http.add_argument("--verbose", "-v", action="store_true", help="print events to stderr during recording")
    p_record_http.add_argument("--quiet", "-q", action="store_true", help="suppress all output except errors")
    p_record_http.add_argument("--parent", metavar="SESSION",
                               help="parent session ID for subagent correlation")

    # replay
    p_replay = sub.add_parser("replay", help="replay a recorded session")
    p_replay.add_argument("session_id", nargs="?", help="session ID (default: latest)")
    p_replay.add_argument("--filter", "-f", help="comma-separated event types to show")
    p_replay.add_argument("--speed", "-s", type=float, default=0, help="replay speed multiplier (0=instant)")
    p_replay.add_argument("--live", "-l", action="store_true", help="replay with timing delays")
    p_replay.add_argument("--limit", "-n", type=int, default=None, metavar="N",
                          help="cap output at N events (default: all); useful for quick inspection of large sessions")
    p_replay.add_argument("--format", choices=["terminal", "html"], default="terminal",
                          help="output format: terminal timeline or self-contained HTML viewer (default: terminal)")
    p_replay.add_argument("--diff", metavar="SESSION_B",
                         help="generate side-by-side HTML diff against SESSION_B")
    p_replay.add_argument("--output", "-o", default="",
                          help="output file path for --format html (default: session-<id>.html)")
    p_replay.add_argument("--expand-subagents", action="store_true",
                          help="inline subagent sessions under their parent tool_call")
    p_replay.add_argument("--tree", action="store_true",
                          help="show session hierarchy tree without full event replay")

    # tree
    p_tree = sub.add_parser("tree", help="show a parent/child session hierarchy")
    p_tree.add_argument("session_id", nargs="?", help="root session ID or prefix (default: latest)")
    p_tree.add_argument("--format", choices=["text", "json"], default="text",
                        help="output format (default: text)")

    # list
    sub.add_parser("list", help="list all recorded sessions")

    # inspect
    p_inspect = sub.add_parser("inspect", help="inspect a session as raw JSON")
    p_inspect.add_argument("session_id", help="session ID or prefix")

    # export
    p_export = sub.add_parser("export", help="export a session")
    p_export.add_argument("session_id", nargs="?", help="session ID or prefix")
    p_export.add_argument("--format", choices=["json", "csv", "ndjson", "otlp", "otlp-genai", "eu-ai-act"],
                          default="json",
                          help="output format (otlp-genai uses strict OTel GenAI semantic conventions)")
    p_export.add_argument("--endpoint", help="OTLP collector URL (e.g. http://localhost:4318)")
    p_export.add_argument("--header", action="append", help="HTTP header for OTLP (e.g. 'x-honeycomb-team: KEY')")
    p_export.add_argument("--service-name", default="agent-trace", help="OTel service name (default: agent-trace)")
    # Langfuse / OTLP metrics flags
    p_export.add_argument("--scores", action="store_true",
                          help="include eval scores in export")
    p_export.add_argument("--metrics", action="store_true",
                          help="export behavioral metrics as OTLP gauges")
    p_export.add_argument("--backend", choices=["langfuse", "otlp"],
                          help="export backend: langfuse or otlp")
    p_export.add_argument("--since", metavar="Nd",
                          help="export sessions from the last N days (e.g. 7d)")
    p_export.add_argument("--until", metavar="DATE",
                          help="upper time bound for batch exports (ISO date or timestamp)")
    p_export.add_argument("--all", action="store_true",
                          help="export all sessions in the selected time window")
    p_export.add_argument("--langfuse-public-key", dest="langfuse_public_key", metavar="KEY",
                          help="Langfuse public key (overrides LANGFUSE_PUBLIC_KEY)")
    p_export.add_argument("--langfuse-secret-key", dest="langfuse_secret_key", metavar="KEY",
                          help="Langfuse secret key (overrides LANGFUSE_SECRET_KEY)")
    p_export.add_argument("--langfuse-host", dest="langfuse_host", metavar="URL",
                          help="Langfuse host (default: https://cloud.langfuse.com)")
    p_export.add_argument("--otlp-endpoint", dest="otlp_endpoint", metavar="URL",
                          help="OTLP metrics endpoint (overrides OTEL_EXPORTER_OTLP_ENDPOINT)")
    p_export.add_argument("--otlp-headers", dest="otlp_headers", metavar="HEADERS",
                          help="OTLP headers as key=value,key=value")
    p_export.add_argument("--anonymize", action="store_true",
                          help="strip identifying information (paths, hostnames, emails, usernames) from the export")
    p_export.add_argument("--anonymize-config", dest="anonymize_config", metavar="FILE",
                          help="path to custom anonymization rules YAML file")
    p_export.add_argument("--output", "-o", default="",
                          help="output file path")
    p_export.add_argument("--dry-run", action="store_true",
                          help="show what would be anonymized without writing output (use with --anonymize)")

    # stats
    p_stats = sub.add_parser("stats", help="show session statistics")
    p_stats.add_argument("session_id", nargs="?", help="session ID (default: latest)")
    p_stats.add_argument("--include-subagents", action="store_true",
                         help="roll up stats across all subagent sessions")

    # hook (called by agent CLI hooks systems)
    p_hook = sub.add_parser("hook", help="handle an agent CLI hook event (internal)")
    p_hook.add_argument("--provider", choices=["claude", "codex", "gemini"], default="claude",
                        help="hook provider (default: claude)")
    p_hook.add_argument("event", nargs="?", help="hook event: session-start, session-end, pre-tool, post-tool, post-tool-failure")

    # setup (generate agent CLI hooks config)
    p_setup = sub.add_parser("setup", help="generate agent CLI hooks configuration")
    setup_redaction = p_setup.add_mutually_exclusive_group()
    setup_redaction.add_argument(
        "--redact",
        action="store_true",
        help="enable secret redaction explicitly (default)",
    )
    setup_redaction.add_argument(
        "--no-redact",
        action="store_true",
        help="disable automatic secret redaction in generated hooks",
    )
    p_setup.add_argument("--global", dest="global_config", action="store_true", help="output config for ~/.claude/settings.json (all projects)")
    p_setup.add_argument("--cli", choices=["claude", "codex", "gemini", "all"], default="claude",
                         help="agent CLI to configure (default: claude)")

    # import (Claude Code JSONL session logs)
    p_import = sub.add_parser("import", help="import a Claude Code JSONL session log")
    p_import.add_argument("path", nargs="?", help="path to .jsonl session file")
    p_import.add_argument("--discover", action="store_true", help="list available Claude Code sessions")
    p_import.add_argument("--claude-dir", default="~/.claude", help="Claude config directory (default: ~/.claude)")

    # explain
    p_explain = sub.add_parser("explain", help="explain a session in plain English")
    p_explain.add_argument("session_id", nargs="?", help="session ID or prefix (default: latest)")

    # timeline
    p_timeline = sub.add_parser("timeline",
                                 help="structured chronological view of a session by phase")
    p_timeline.add_argument("session_id", nargs="?",
                             help="session ID or prefix (default: latest)")
    p_timeline.add_argument("--model", default="sonnet",
                             choices=["sonnet", "opus", "haiku", "gpt4", "gpt4o"],
                             help="model pricing for cost estimates (default: sonnet)")
    p_timeline.add_argument("--format", choices=["text", "json"], default="text",
                             help="output format (default: text)")

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
    p_audit.add_argument("--verify-chain", dest="verify_chain", action="store_true",
                         help="verify SHA-256 hash chain integrity before policy audit")
    p_audit.add_argument("--policy", default=".agent-scope.json",
                         help="path to policy file (default: .agent-scope.json)")

    # verify
    p_verify = sub.add_parser("verify", help="verify session or exported trace integrity")
    p_verify.add_argument("session_id", nargs="?", help="session ID or prefix")
    p_verify.add_argument("--from-export", dest="from_export", metavar="FILE",
                          help="verify hash chain from an EU AI Act export")
    p_verify.add_argument("--format", choices=["text", "json"], default="text")

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
    p_postmortem.add_argument("--list", action="store_true", help="list crashed sessions")
    p_postmortem.add_argument("--stale-after", type=float, default=30.0, metavar="SECONDS",
                              help="heartbeat age before a live session is treated as crashed (default: 30)")
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
    p_eval_ci.add_argument("--baseline", metavar="FILE",
                           help="compare scores against a saved baseline JSON")
    p_eval_ci.add_argument("--save-baseline", dest="save_baseline", metavar="FILE",
                           help="save current scores as a baseline and exit")
    p_eval_ci.add_argument("--tolerance", type=float, default=0.0, metavar="N",
                           help="allow up to N regression vs baseline before failing (default: 0)")
    p_eval_ci.add_argument("--github-summary", dest="github_summary", action="store_true",
                           help="write PR-comment Markdown to .agent-traces/eval-summary.md")

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
    p_ds_auto = dataset_sub.add_parser("auto", help="auto-populate dataset from sessions by signal filter")
    p_ds_auto.add_argument("--name", default="default", help="dataset name (default: default)")
    p_ds_auto.add_argument("--dataset", default="", help="explicit dataset path (overrides --name)")
    p_ds_auto.add_argument("--filter", default="has-errors",
                           help="filter: has-errors, high-retry, cost-above:N, wide-blast, "
                                "long-duration:Ns, low-eval-score:N (default: has-errors)")
    p_ds_auto.add_argument("--since", default="7d", metavar="Nd",
                           help="look back N days (default: 7d)")
    p_ds_auto.add_argument("--label", default="", help="label for added entries")

    # watch
    p_watch = sub.add_parser("watch", help="monitor a live session with circuit breakers")
    p_watch.add_argument("session_id", nargs="?", help="session ID to watch (default: latest active)")
    p_watch.add_argument("--max-retries", type=int, default=5, help="max retries before alert (default: 5)")
    p_watch.add_argument("--max-cost", type=float, default=10.0, help="max cost in dollars (default: 10)")
    p_watch.add_argument("--max-duration", type=int, default=1800, help="max duration in seconds (default: 1800)")
    p_watch.add_argument("--timeout", metavar="DURATION",
                         help="wall-clock timeout before killing the agent (e.g. 30m, 2h, 90s); "
                              "alias for --max-duration with human-readable units")
    p_watch.add_argument("--budget", type=float, metavar="DOLLARS",
                         help="token-cost ceiling in dollars before killing the agent; "
                              "alias for --max-cost")
    p_watch.add_argument("--loop-threshold", type=int, metavar="N",
                         help="alert when an identical tool call repeats N times (default: 3)")
    p_watch.add_argument("--loop-window", type=int, metavar="N",
                         help="events to scan for repeated identical tool calls (default: 10)")
    p_watch.add_argument("--on-death", dest="on_death", metavar="CMD",
                         help="command to run after the agent is killed; "
                              "{post_mortem_path} is substituted with the post-mortem JSON path")
    p_watch.add_argument("--on-violation", choices=["terminal", "file", "kill"], default="terminal",
                         help="action on violation (default: terminal)")
    p_watch.add_argument("--webhook", help="webhook URL for alerts")
    p_watch.add_argument("--config", help="path to .agent-watch.json config file")
    p_watch.add_argument("--max-context-pct", type=int, default=90, dest="max_context_pct",
                         help="alert when context window is this %% full (default: 90)")
    p_watch.add_argument("--policy", metavar="POLICY_FILE",
                         help="path to scope policy file (default: .agent-scope.json)")
    p_watch.add_argument("--rules", metavar="RULES_FILE",
                         help="YAML/JSON rules file for rule-based kill switch (nanny mode)")
    p_watch.add_argument("--dry-run", action="store_true", dest="dry_run",
                         help="evaluate rules without taking action (for testing)")
    p_watch.add_argument("--stream-otlp", dest="stream_otlp", action="store_true",
                         help="stream events as OTLP JSON spans instead of raw NDJSON")
    p_watch.add_argument("--stream-to", dest="stream_to", metavar="URL",
                         help="push events in real-time to this HTTP endpoint as NDJSON")
    p_watch.add_argument("--stream-batch-size", dest="stream_batch_size", type=int, default=10,
                         metavar="N",
                         help="number of events per batch when streaming (default: 10)")
    p_watch.add_argument("--stream-flush-interval", dest="stream_flush_interval", type=float,
                         default=2.0, metavar="SECONDS",
                         help="max seconds between flushes when streaming (default: 2.0)")

    # mcp-scan
    p_mcp_scan = sub.add_parser("mcp-scan", help="scan runtime MCP tool poisoning indicators")
    p_mcp_scan.add_argument("--session", metavar="ID",
                            help="session ID or prefix to scan (default: all recent sessions)")
    p_mcp_scan.add_argument("--since", default="7d", metavar="DURATION_OR_DATE",
                            help="scan sessions since this duration/date (default: 7d)")
    p_mcp_scan.add_argument("--watch", action="store_true",
                            help="watch the latest or selected session for live MCP poisoning alerts")
    p_mcp_scan.add_argument("--patterns", metavar="FILE",
                            help="additional regex pattern file (default: ~/.agent-strace/mcp-patterns.txt)")
    p_mcp_scan.add_argument("--project-root", default=".",
                            help="project root for shadow-write detection (default: .)")
    p_mcp_scan.add_argument("--format", choices=["text", "json"], default="text",
                            help="output format (default: text)")

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
    p_dash.add_argument("--trend", action="store_true",
                        help="show quality and behavioral metrics over time")
    p_dash.add_argument("--since", metavar="Nd",
                        help="limit trend to sessions from the last N days (e.g. 30d)")
    p_dash.add_argument("--html", metavar="FILE",
                        help="write self-contained HTML trend report to FILE")
    dash_sub = p_dash.add_subparsers(dest="dash_command")
    p_dash_ann = dash_sub.add_parser("annotate", help="add a timeline annotation")
    p_dash_ann.add_argument("--date", required=True, help="date in YYYY-MM-DD format")
    p_dash_ann.add_argument("--note", required=True, help="annotation text")

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
    p_ann.add_argument("--filter-label", dest="filter_label", metavar="LABEL",
                       help="filter listed annotations by label")
    p_ann.add_argument("--filter-author", dest="filter_author", metavar="AUTHOR",
                       help="filter listed annotations by author")
    p_ann.add_argument("--since", metavar="Nd",
                       help="filter listed annotations created in the last N days (e.g. 7d)")
    p_ann.add_argument("--export-format", dest="export_format", choices=["json"],
                       help="output format for --list (default: terminal)")

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

    # a2a-tree (A2A agent call graph)
    p_a2a = sub.add_parser("a2a-tree", help="show the agent-to-agent call graph for a session")
    p_a2a.add_argument("session_id", nargs="?", help="session ID or prefix (default: latest)")
    p_a2a.add_argument("--format", choices=["text", "json"], default="text",
                       help="output format: text tree or OTLP-compatible JSON spans (default: text)")

    # oncall (on-call readiness report)
    p_oncall = sub.add_parser("oncall", help="show agent-modified files you haven't reviewed before on-call")
    p_oncall.add_argument("--rotation-start", required=True, dest="rotation_start",
                          metavar="DATE", help="on-call rotation start date (YYYY-MM-DD)")
    p_oncall.add_argument("--scope", default="**", help="file glob to limit scope (default: **)")
    p_oncall.add_argument("--repo", default=".", help="path to git repository (default: .)")
    p_oncall.add_argument("--since-days", type=int, default=30, dest="since_days",
                          help="how many days of sessions to scan (default: 30)")

    # freshness (context freshness check)
    p_fresh = sub.add_parser("freshness", help="check how stale the agent context is since last session")
    p_fresh.add_argument("--since", default="", help="check changes since this date (YYYY-MM-DD)")
    p_fresh.add_argument("--scope", default="", help="file glob to limit scope")
    p_fresh.add_argument("--repo", default=".", help="path to git repository (default: .)")

    # baseline (statistical baseline and anomaly detection)
    p_baseline = sub.add_parser("baseline", help="build and check statistical session baselines")
    baseline_sub = p_baseline.add_subparsers(dest="baseline_cmd")

    p_bl_update = baseline_sub.add_parser("update", help="build baseline from recent sessions")
    p_bl_update.add_argument("--since", type=float, default=30.0, dest="since_days",
                             metavar="DAYS", help="sessions from the last N days (default: 30)")
    p_bl_update.add_argument("--output", default=".agent-traces/baseline.json",
                             help="output path (default: .agent-traces/baseline.json)")

    p_bl_check = baseline_sub.add_parser("check", help="check a session against the baseline")
    p_bl_check.add_argument("session_id", nargs="?", help="session ID or prefix (default: latest)")
    p_bl_check.add_argument("--baseline", dest="baseline_path",
                            default=".agent-traces/baseline.json",
                            help="baseline file (default: .agent-traces/baseline.json)")
    p_bl_check.add_argument("--sigma", type=float, default=2.0,
                            help="anomaly threshold in standard deviations (default: 2.0)")

    p_bl_show = baseline_sub.add_parser("show", help="show baseline statistics")
    p_bl_show.add_argument("--baseline", dest="baseline_path",
                           default=".agent-traces/baseline.json",
                           help="baseline file (default: .agent-traces/baseline.json)")

    # approval (human-in-the-loop)
    p_appr = sub.add_parser("approval", help="manage human-in-the-loop approval requests")
    appr_sub = p_appr.add_subparsers(dest="approval_cmd")

    p_appr_list = appr_sub.add_parser("list", help="list approval requests")
    p_appr_list.add_argument("--state", choices=["pending", "approved", "denied"],
                             help="filter by state (default: all)")

    p_appr_show = appr_sub.add_parser("show", help="show details of a request")
    p_appr_show.add_argument("request_id", help="request ID or prefix")

    p_appr_approve = appr_sub.add_parser("approve", help="approve a pending request")
    p_appr_approve.add_argument("request_id", help="request ID or prefix")
    p_appr_approve.add_argument("--by", default="", help="approver name")
    p_appr_approve.add_argument("--no-resume", dest="no_resume", action="store_true",
                                help="approve without sending SIGCONT to the agent")

    p_appr_deny = appr_sub.add_parser("deny", help="deny a pending request")
    p_appr_deny.add_argument("request_id", help="request ID or prefix")
    p_appr_deny.add_argument("--reason", default="", help="reason for denial")
    p_appr_deny.add_argument("--by", default="", help="reviewer name")
    p_appr_deny.add_argument("--no-kill", dest="no_kill", action="store_true",
                             help="deny without sending SIGTERM to the agent")

    # rbac (role-based access control)
    p_rbac = sub.add_parser("rbac", help="manage role-based access control assignments")
    rbac_sub = p_rbac.add_subparsers(dest="rbac_cmd")

    p_rbac_assign = rbac_sub.add_parser("assign", help="assign a role to a user or group")
    p_rbac_assign.add_argument("--user", metavar="EMAIL", default="", help="user email")
    p_rbac_assign.add_argument("--group", metavar="GROUP", default="", help="group identifier")
    p_rbac_assign.add_argument("--role", required=True,
                               help="role to assign (owner/admin/member/viewer/machine or workspace:*)")
    p_rbac_assign.add_argument("--workspace", metavar="ID", default="",
                               help="workspace ID for workspace-scoped role")
    p_rbac_assign.add_argument("--by", default="", help="assigner name for audit trail")

    p_rbac_revoke = rbac_sub.add_parser("revoke", help="revoke a role assignment")
    p_rbac_revoke.add_argument("--user", metavar="EMAIL", default="", help="user email")
    p_rbac_revoke.add_argument("--group", metavar="GROUP", default="", help="group identifier")
    p_rbac_revoke.add_argument("--workspace", metavar="ID", default="",
                               help="workspace ID (omit for org-level)")

    p_rbac_list = rbac_sub.add_parser("list", help="list all role assignments")
    p_rbac_list.add_argument("--workspace", metavar="ID", default="",
                             help="filter by workspace ID")

    p_rbac_check = rbac_sub.add_parser("check", help="check if a user can perform an action")
    p_rbac_check.add_argument("--user", metavar="EMAIL", required=True)
    p_rbac_check.add_argument("--action", required=True,
                              help="action to check (e.g. read_sessions, manage_policies)")
    p_rbac_check.add_argument("--workspace", metavar="ID", default="",
                              help="workspace context for the check")

    # auth (SSO login/logout/status)
    p_auth = sub.add_parser("auth", help="authenticate with a hosted collector via SSO")
    auth_sub = p_auth.add_subparsers(dest="auth_cmd")

    p_auth_login = auth_sub.add_parser("login", help="log in to a hosted collector")
    p_auth_login.add_argument("--server", metavar="URL", required=True,
                              help="hosted collector URL")
    p_auth_login.add_argument("--force", action="store_true",
                              help="re-authenticate even if already logged in")

    p_auth_logout = auth_sub.add_parser("logout", help="remove stored token")
    p_auth_logout.add_argument("--server", metavar="URL", required=True,
                               help="hosted collector URL")

    p_auth_status = auth_sub.add_parser("status", help="show stored token status")
    p_auth_status.add_argument("--server", metavar="URL", default="",
                               help="check a specific server (omit for all)")

    # apply (IaC — apply .agent-strace.yaml to local store or hosted collector)
    p_apply = sub.add_parser("apply", help="apply .agent-strace.yaml config to local store or hosted collector")
    p_apply.add_argument("--config", metavar="FILE", default=".agent-strace.yaml",
                         help="config file (default: .agent-strace.yaml)")
    p_apply.add_argument("--server", metavar="URL", default="",
                         help="hosted collector URL (omit for local apply)")
    p_apply.add_argument("--auth-key", metavar="KEY", dest="auth_key", default="",
                         help="API key for hosted collector")
    p_apply.add_argument("--dry-run", action="store_true", dest="dry_run",
                         help="show planned changes without applying")
    p_apply.add_argument("--dir", metavar="DIR", default=None,
                         help="trace store directory (default: .agent-traces)")

    # config-diff (IaC — show drift between config file and local store)
    p_cdiff = sub.add_parser("config-diff",
                              help="show drift between .agent-strace.yaml and current store state")
    p_cdiff.add_argument("--config", metavar="FILE", default=".agent-strace.yaml",
                         help="config file (default: .agent-strace.yaml)")
    p_cdiff.add_argument("--dir", metavar="DIR", default=None,
                         help="trace store directory (default: .agent-traces)")

    # compliance (compliance export)
    p_comp = sub.add_parser("compliance", help="export compliance reports (EU AI Act, SOC 2, HIPAA)")
    comp_sub = p_comp.add_subparsers(dest="compliance_cmd")
    p_comp_exp = comp_sub.add_parser("export", help="export compliance report")
    p_comp_exp.add_argument("session_id", nargs="?",
                            help="session ID or prefix (default: all recent sessions)")
    p_comp_exp.add_argument("--framework", choices=["eu-ai-act", "soc2", "hipaa", "all"],
                            default="all", help="compliance framework (default: all)")
    p_comp_exp.add_argument("--since", metavar="Nd",
                            help="export sessions from last N days (e.g. 30d)")
    p_comp_exp.add_argument("--output", "-o", metavar="FILE",
                            help="write JSON report to FILE instead of stdout")

    # audit-readiness
    p_ready = sub.add_parser("audit-readiness", help="check EU AI Act audit export readiness")
    p_ready.add_argument("--retention-days", type=float, default=90.0,
                         help="required retention coverage in days (default: 90)")
    p_ready.add_argument("--format", choices=["text", "json"], default="text")

    # workspace (workspace isolation)
    p_ws = sub.add_parser("workspace", help="manage isolated workspaces")
    ws_sub = p_ws.add_subparsers(dest="workspace_cmd")
    ws_sub.add_parser("list", help="list all workspaces")
    p_ws_use = ws_sub.add_parser("use", help="print shell export for a workspace")
    p_ws_use.add_argument("workspace_id", help="workspace ID to activate")
    p_ws_new = ws_sub.add_parser("new", help="create a new workspace")
    p_ws_new.add_argument("workspace_id", help="workspace ID to create")
    p_ws_rm = ws_sub.add_parser("rm", help="delete a workspace and all its sessions")
    p_ws_rm.add_argument("workspace_id", help="workspace ID to delete")
    p_ws_rm.add_argument("--force", action="store_true", help="skip confirmation")

    # identity (per-agent machine identity and session signing)
    p_identity = sub.add_parser("identity", help="manage agent machine identity and sign sessions")
    identity_sub = p_identity.add_subparsers(dest="identity_cmd")

    identity_sub.add_parser("show", help="show or create the machine identity")

    p_id_sign = identity_sub.add_parser("sign", help="sign a session with the machine identity")
    p_id_sign.add_argument("session_id", nargs="?", help="session ID or prefix (default: latest)")
    p_id_sign.add_argument("--agent-name", dest="agent_name", default="",
                           help="agent name to embed in the identity")

    p_id_verify = identity_sub.add_parser("verify", help="verify a session's signature")
    p_id_verify.add_argument("session_id", nargs="?", help="session ID or prefix (default: latest)")

    # drift (behavioral drift detection)
    p_drift = sub.add_parser("drift", help="detect behavioral drift across sessions")
    p_drift.add_argument("--since", metavar="Nd", help="analyse sessions from the last N days (e.g. 30d)")
    p_drift.add_argument("--baseline", metavar="FILE",
                         help="path to a saved behavioral fingerprint JSON, or a date range YYYY-MM-DD:YYYY-MM-DD")
    p_drift.add_argument("--current", metavar="RANGE",
                         help="current window as YYYY-MM-DD:YYYY-MM-DD")
    p_drift.add_argument("--baseline-range", metavar="RANGE", dest="baseline_range",
                         help="baseline window as YYYY-MM-DD:YYYY-MM-DD")
    p_drift.add_argument("--save-baseline", metavar="FILE", dest="save_baseline",
                         help="save current fingerprint to FILE and exit")
    p_drift.add_argument("--threshold", type=float, default=0.20,
                         help="drift score above which to alert (default: 0.20)")
    p_drift.add_argument("--format", choices=["table", "json"], default="table",
                         help="output format (default: table)")

    # fingerprint (behavioral profile)
    p_fingerprint = sub.add_parser("fingerprint", help="characterize an agent's behavioral profile")
    p_fingerprint.add_argument("--sessions", type=int, default=20, metavar="N",
                               help="number of recent sessions to analyze (default: 20)")
    p_fingerprint.add_argument("--output", "-o", metavar="FILE",
                               help="write fingerprint JSON to FILE")
    p_fingerprint.add_argument("--id", default="", metavar="ID",
                               help="fingerprint ID to store in JSON")
    p_fingerprint.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"),
                               help="compare two saved fingerprint JSON files")
    p_fingerprint.add_argument("--threshold", type=float, default=0.20,
                               help="comparison score above which to alert (default: 0.20)")
    p_fingerprint.add_argument("--format", choices=["text", "json"], default="text",
                               help="output format (default: text)")

    # freeze/regression (tool-call sequence fixtures)
    p_freeze = sub.add_parser("freeze", help="freeze a session's tool-call sequence as a fixture")
    p_freeze.add_argument("session_id", nargs="?", help="session ID or prefix (default: latest)")
    p_freeze.add_argument("--output", "-o", metavar="FILE",
                          help="write fixture JSON to FILE")
    p_freeze.add_argument("--task", default="", help="task description to store in the fixture")
    p_freeze.add_argument("--format", choices=["text", "json"], default="text",
                          help="output format (default: text)")

    p_regression = sub.add_parser("regression", help="compare a session against a frozen fixture")
    p_regression.add_argument("fixture_file", help="fixture JSON written by agent-strace freeze")
    p_regression.add_argument("session_id", nargs="?", help="session ID or prefix (default: latest)")
    p_regression.add_argument("--threshold", type=float, default=0.0,
                              help="allowed divergence before failing (default: 0.0)")
    p_regression.add_argument("--format", choices=["text", "json"], default="text",
                              help="output format (default: text)")

    # optimize (propose AGENTS.md / skill file improvements from trace failures)
    p_opt = sub.add_parser("optimize", help="propose instruction file improvements from trace failures")
    p_opt.add_argument("session_id", nargs="?", help="session ID or prefix (default: latest)")
    p_opt.add_argument("--target", default="AGENTS.md",
                       help="instruction file to improve (default: AGENTS.md)")
    p_opt.add_argument("--dataset", metavar="NAME",
                       help="use a named dataset instead of a single session")
    p_opt.add_argument("--dry-run", action="store_true", dest="dry_run",
                       help="show proposed diff without writing")
    p_opt.add_argument("--apply", action="store_true",
                       help="write proposed additions to the target file")
    p_opt.add_argument("--base-url", dest="base_url", metavar="URL",
                       help="OpenAI-compatible LLM base URL (overrides OPENAI_BASE_URL)")
    p_opt.add_argument("--model", metavar="MODEL", default="gpt-4o-mini",
                       help="LLM model name (default: gpt-4o-mini)")
    p_opt.add_argument("--api-key", dest="api_key", metavar="KEY",
                       help="LLM API key (overrides OPENAI_API_KEY)")

    # standup (agent standup report)
    p_standup = sub.add_parser("standup", help="plain-English summary of what the agent did")
    p_standup.add_argument("session_id", nargs="?", help="session ID or prefix (default: latest)")
    p_standup.add_argument("--no-llm", action="store_true", dest="no_llm",
                           help="structured output only, no LLM narrative (default)")

    # mcp (MCP server — expose traces as queryable tools)
    p_mcp = sub.add_parser(
        "mcp",
        help="start an MCP server that exposes session traces as queryable tools",
    )
    p_mcp.add_argument(
        "--transport",
        choices=["stdio"],
        default="stdio",
        help="transport protocol (default: stdio)",
    )

    # auto (auto-instrumentation)
    p_auto = sub.add_parser(
        "auto",
        help="run a command with auto-instrumentation for agent frameworks",
    )
    p_auto.add_argument(
        "--framework", "-f",
        metavar="NAME",
        help=(
            "framework to instrument: "
            + ", ".join(sorted(set(_INTEGRATIONS.keys())))
            + " (or 'detect' to auto-detect)"
        ),
    )
    p_auto.add_argument("--detect", action="store_true",
                        help="auto-detect and instrument all installed frameworks")
    p_auto.add_argument("server_cmd", nargs=argparse.REMAINDER,
                        help="command to run with instrumentation")

    # server (event collector)
    p_server = sub.add_parser(
        "server",
        help="start a server-side event collector (receives events from remote agents)",
    )
    p_server.add_argument("--port", type=int, default=4317,
                          help="port to listen on (default: 4317)")
    p_server.add_argument("--host", default="0.0.0.0",
                          help="host to bind to (default: 0.0.0.0)")
    p_server.add_argument("--storage", metavar="DIR",
                          help="storage directory for traces "
                               "(default: $AGENT_STRACE_STORAGE or .agent-traces)")
    p_server.add_argument("--dashboard", action="store_true",
                          help="serve a browser-based session dashboard at /")
    p_server.add_argument("--auth-key", metavar="KEY", dest="auth_key",
                          help="require Authorization: Bearer <KEY> on all requests "
                               "(also read from AGENT_STRACE_AUTH_KEY env var)")
    p_server.add_argument("--auth", choices=["oidc"], default="",
                          help="enable SSO authentication (oidc)")
    p_server.add_argument("--oidc-issuer", metavar="URL", dest="oidc_issuer", default="",
                          help="OIDC issuer URL (e.g. https://accounts.google.com)")
    p_server.add_argument("--oidc-client-id", metavar="ID", dest="oidc_client_id", default="",
                          help="OIDC client ID")
    p_server.add_argument("--oidc-client-secret", metavar="SECRET",
                          dest="oidc_client_secret", default="",
                          help="OIDC client secret")
    p_server.add_argument("--enforce-sso", action="store_true", dest="enforce_sso",
                          help="reject API key fallback when SSO is configured")
    p_server_sub = p_server.add_subparsers(dest="server_subcommand")
    p_server_sub.add_parser("keygen", help="generate a new ast_-prefixed API key")

    # sample
    p_sample = sub.add_parser(
        "sample",
        help="export worst/diverse/random/recent sessions as a JSONL regression suite",
    )
    p_sample.add_argument(
        "--strategy",
        choices=["worst", "diverse", "random", "recent"],
        default="worst",
        help="sampling strategy (default: worst)",
    )
    p_sample.add_argument("--n", type=int, default=20, metavar="N",
                          help="number of sessions to sample (default: 20)")
    p_sample.add_argument("--output", "-o", default="sample.jsonl",
                          help="output JSONL file path (default: sample.jsonl)")
    p_sample.add_argument("--deduplicate", action="store_true",
                          help="skip sessions with identical tool call sequences")
    p_sample.add_argument("--seed", type=int, default=None,
                          help="random seed for reproducible random sampling")

    # config-watch
    p_cw = sub.add_parser("config-watch",
                          help="detect AGENTS.md and config file changes between sessions")
    cw_sub = p_cw.add_subparsers(dest="config_watch_command")

    cw_snap = cw_sub.add_parser("snapshot", help="record a snapshot of current config files")
    cw_snap.add_argument("--label", metavar="TEXT",
                         help="human-readable label for this snapshot")
    cw_snap.add_argument("--watch", metavar="PATH", action="append",
                         help="additional file to watch (repeatable)")

    cw_check = cw_sub.add_parser("check",
                                  help="check whether config has changed since last snapshot")
    cw_check.add_argument("--watch", metavar="PATH", action="append",
                          help="additional file to watch (repeatable)")
    cw_check.add_argument("--format", choices=["text", "json"], default="text")

    cw_hist = cw_sub.add_parser("history", help="show full snapshot history")
    cw_hist.add_argument("--format", choices=["text", "json"], default="text")

    cw_aff = cw_sub.add_parser("affected",
                                help="list sessions that ran after a config change")
    cw_aff.add_argument("--since", metavar="DURATION",
                        help="only sessions newer than this (e.g. 7d, 24h)")
    cw_aff.add_argument("--format", choices=["text", "json"], default="text")

    # compare
    p_compare = sub.add_parser("compare", help="session-to-session regression report")
    p_compare.add_argument("session_id_a", nargs="?",
                           help="first session ID (baseline)")
    p_compare.add_argument("session_id_b", nargs="?",
                           help="second session ID (candidate)")
    p_compare.add_argument("--rerun", action="store_true",
                           help="re-run the original prompt and compare live")
    p_compare.add_argument("--model", metavar="MODEL",
                           help="model to use for --rerun")
    p_compare.add_argument("--tag", metavar="TAG",
                           help="compare the last N sessions matching this tag")
    p_compare.add_argument("--last", type=int, default=2, metavar="N",
                           help="number of tagged sessions to compare (default: 2)")
    p_compare.add_argument("--format", choices=["text", "json"], default="text",
                           dest="format", help="output format (default: text)")

    # budget-report
    p_budget = sub.add_parser("budget-report", help="weekly spend digest across sessions")
    p_budget.add_argument("--since", metavar="DATE",
                          help="start of window (ISO date or duration like 7d; default: 7 days ago)")
    p_budget.add_argument("--until", metavar="DATE",
                          help="end of window (ISO date; default: now)")
    p_budget.add_argument("--team", metavar="TEAM",
                         help="filter report to a specific team name")
    p_budget.add_argument("--format", choices=["text", "markdown", "json"], default="text",
                          dest="format", help="output format (default: text)")
    p_budget.add_argument("--endpoint", metavar="URL",
                          help="remote collector endpoint (not yet implemented)")

    # team-report
    p_team_report = sub.add_parser("team-report", help="cost attribution by git author, branch, or PR")
    p_team_report.add_argument("--since", metavar="DATE",
                               help="start of window (ISO date or duration like 7d; default: 7 days ago)")
    p_team_report.add_argument("--until", metavar="DATE",
                               help="end of window (ISO date or duration like 7d; default: now)")
    p_team_report.add_argument("--by", choices=["author", "branch", "pr"], default="author",
                               help="group by git author, branch, or PR (default: author)")
    p_team_report.add_argument("--export", choices=["text", "csv", "json"], default="text",
                               help="output format (default: text)")
    p_team_report.add_argument("--outlier-threshold", type=float, default=2.0,
                               help="flag sessions above N times average cost (default: 2.0)")

    # lint
    p_lint = sub.add_parser("lint", help="analyse a session for bad behaviour patterns")
    p_lint.add_argument("session_id", nargs="?",
                        help="session ID to lint (default: latest)")
    p_lint.add_argument("--all", action="store_true", dest="all",
                        help="lint all sessions in the store")
    p_lint.add_argument("--since", metavar="DURATION",
                        help="with --all, only sessions newer than this (e.g. 7d, 24h)")
    p_lint.add_argument("--strict", action="store_true",
                        help="exit with code 1 on any WARN or ERROR (for CI)")
    p_lint.add_argument("--format", choices=["text", "json"], default="text",
                        help="output format (default: text)")
    p_lint.add_argument("--config", metavar="FILE",
                        help="path to .agent-strace-lint.json config file")

    # retention
    p_ret = sub.add_parser("retention", help="manage session data retention")
    ret_sub = p_ret.add_subparsers(dest="retention_command")

    ret_sub.add_parser("status", help="show retention status and what would be deleted")

    p_ret_clean = ret_sub.add_parser("clean", help="delete sessions that exceed retention limits")
    p_ret_clean.add_argument("--dry-run", action="store_true",
                             help="show what would be deleted without deleting")
    p_ret_clean.add_argument("--max-age-days", type=int, dest="max_age_days", metavar="N",
                             help="delete sessions older than N days")
    p_ret_clean.add_argument("--max-sessions", type=int, dest="max_sessions", metavar="N",
                             help="keep only the most recent N sessions")
    p_ret_clean.add_argument("--max-size-mb", type=float, dest="max_size_mb", metavar="MB",
                             help="delete oldest sessions when storage exceeds MB")
    p_ret_clean.add_argument("--config", metavar="FILE",
                             help="path to .agent-strace.yaml config file")

    # diff --semantic and --eval-config flags (extend existing diff parser)
    p_diff.add_argument("--semantic", action="store_true",
                        help="semantic outcome-level diff (files, cost, errors)")
    p_diff.add_argument("--compare", action="store_true",
                        help="rich side-by-side comparison table with verdict")
    p_diff.add_argument("--eval-config", default=".agent-evals.yaml", dest="eval_config",
                        help="eval config for score comparison")

    return parser


def cmd_auto(args: argparse.Namespace) -> int:
    """Run a command with auto-instrumentation applied."""
    import subprocess
    import os

    command = getattr(args, "server_cmd", [])
    # Strip leading '--' separator
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        sys.stderr.write("Usage: agent-strace auto [--framework NAME] -- <command>\n")
        return 1

    # Determine which frameworks to instrument
    if getattr(args, "detect", False):
        env_val = "detect"
    elif getattr(args, "framework", None):
        env_val = args.framework
    else:
        env_val = "detect"

    env = os.environ.copy()
    env["AGENT_STRACE_AUTO_INSTRUMENT"] = env_val

    # Inject agent_trace.auto into PYTHONSTARTUP or sitecustomize is complex;
    # instead set PYTHONPATH and use -c to import auto before the script.
    # Simplest approach: set env var and let the user's code import agent_trace.auto,
    # or use python -c "import agent_trace.auto; exec(open(script).read())"
    # For subprocess execution, we prepend the auto-import via PYTHONSTARTUP.
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("import agent_trace.auto\n")
        startup_path = f.name

    env["PYTHONSTARTUP"] = startup_path

    try:
        result = subprocess.run(command, env=env)
        return result.returncode
    finally:
        try:
            os.unlink(startup_path)
        except OSError:
            pass


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # hook subcommand is handled separately (reads stdin)
    if args.command == "hook":
        hook_args = ["--provider", getattr(args, "provider", "claude")]
        if args.event:
            hook_args.append(args.event)
        hook_main(hook_args)
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
        "timeline": cmd_timeline,
        "cost": cmd_cost,
        "diff": cmd_diff,
        "why": cmd_why,
        "audit": cmd_audit,
        "verify": cmd_verify,
        "share": cmd_share,
        "postmortem": cmd_postmortem,
        "eval": cmd_eval,
        "watch": cmd_watch,
        "mcp-scan": cmd_mcp_scan,
        "policy": cmd_policy,
        "dashboard": cmd_dashboard,
        "annotate": cmd_annotate,
        "token-budget": cmd_token_budget,
        "audit-tools": cmd_audit_tools,
        "curve": cmd_curve,
        "inflation": cmd_inflation,
        "a2a-tree": cmd_a2a_tree,
        "baseline": cmd_baseline,
        "drift": cmd_drift,
        "fingerprint": cmd_fingerprint,
        "tree": cmd_tree,
        "identity": cmd_identity,
        "workspace": cmd_workspace,
        "compliance": cmd_compliance,
        "audit-readiness": cmd_audit_readiness,
        "approval": cmd_approval,
        "rbac": cmd_rbac,
        "apply": cmd_apply,
        "config-diff": cmd_config_diff,
        "auth": cmd_auth,
        "optimize": cmd_optimize,
        "oncall": cmd_oncall,
        "freshness": cmd_freshness,
        "standup": cmd_standup,
        "mcp": cmd_mcp,
        "auto": cmd_auto,
        "budget-report": cmd_budget_report,
        "team-report": cmd_team_report,
        "compare": cmd_compare,
        "freeze": cmd_freeze,
        "regression": cmd_regression,
        "config-watch": cmd_config_watch,
        "lint": cmd_lint,
        "retention": cmd_retention,
        "sample": cmd_sample,
        "server": cmd_server,
    }

    handler = handlers.get(args.command)
    if handler:
        sys.exit(handler(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
