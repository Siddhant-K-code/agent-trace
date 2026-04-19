"""Trace replay and display.

Renders a captured trace as a human-readable timeline.
Supports filtering by event type and time range.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from .models import EventType, TraceEvent, SessionMeta
from .store import TraceStore


# ANSI colors
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    GRAY = "\033[90m"


EVENT_COLORS = {
    EventType.SESSION_START: C.GREEN,
    EventType.SESSION_END: C.GREEN,
    EventType.TOOL_CALL: C.CYAN,
    EventType.TOOL_RESULT: C.BLUE,
    EventType.LLM_REQUEST: C.MAGENTA,
    EventType.LLM_RESPONSE: C.MAGENTA,
    EventType.FILE_READ: C.YELLOW,
    EventType.FILE_WRITE: C.YELLOW,
    EventType.DECISION: C.WHITE,
    EventType.ERROR: C.RED,
    EventType.USER_PROMPT: C.GREEN,
    EventType.ASSISTANT_RESPONSE: C.MAGENTA,
}

EVENT_ICONS = {
    EventType.SESSION_START: "▶",
    EventType.SESSION_END: "■",
    EventType.TOOL_CALL: "→",
    EventType.TOOL_RESULT: "←",
    EventType.LLM_REQUEST: "⬆",
    EventType.LLM_RESPONSE: "⬇",
    EventType.FILE_READ: "📖",
    EventType.FILE_WRITE: "📝",
    EventType.DECISION: "◆",
    EventType.ERROR: "✗",
    EventType.USER_PROMPT: "👤",
    EventType.ASSISTANT_RESPONSE: "🤖",
}


def _format_timestamp(ts: float, base_ts: float | None = None) -> str:
    """Format timestamp as relative offset or absolute time."""
    if base_ts is not None:
        offset = ts - base_ts
        if offset < 60:
            return f"+{offset:6.2f}s"
        minutes = int(offset // 60)
        seconds = offset % 60
        return f"+{minutes}m{seconds:05.2f}s"
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%H:%M:%S.%f")[:-3]


def _format_duration(ms: float | None) -> str:
    if ms is None:
        return ""
    if ms < 1000:
        return f" ({ms:.0f}ms)"
    return f" ({ms / 1000:.2f}s)"


def _strip_markdown(text: str) -> str:
    """Remove markdown formatting for terminal display."""
    import re
    # Code blocks (fenced)
    text = re.sub(r'```[\s\S]*?```', '', text)
    # Bold and italic
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    text = re.sub(r'_(.+?)_', r'\1', text)
    # Inline code
    text = re.sub(r'`(.+?)`', r'\1', text)
    # Headers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Links [text](url) -> text
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
    # Markdown tables: strip separator rows and pipe formatting
    text = re.sub(r'^\|?[-:| ]+\|[-:| ]+\|?$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\|(.+)\|$', lambda m: m.group(1).replace('|', ', ').strip(), text, flags=re.MULTILINE)
    # List markers
    text = re.sub(r'^[\s]*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[\s]*\d+\.\s+', '', text, flags=re.MULTILINE)
    # Collapse multiple spaces and newlines
    text = re.sub(r'\n{2,}', ' ', text)
    text = text.replace('\n', ' ')
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def _tool_call_detail(tool_name: str, args: dict) -> str:
    """Extract the most useful detail from tool call arguments."""
    if not args:
        return ""
    name = tool_name.lower()

    # Bash: show the command
    if name == "bash" and "command" in args:
        cmd = str(args["command"])
        if len(cmd) > 120:
            cmd = cmd[:120] + "..."
        return f"$ {cmd}"

    # Read/Write: show the file path
    if name in ("read", "write") and "file_path" in args:
        return str(args["file_path"])

    # Edit: show file path and a hint of what changed
    if name == "edit":
        path = args.get("file_path", "")
        old = args.get("old_string", args.get("old_text", ""))
        if path and old:
            old_preview = str(old)[:60].replace("\n", " ")
            if len(str(old)) > 60:
                old_preview += "..."
            return f"{path} (replacing: {old_preview})"
        if path:
            return str(path)

    # Glob/Grep: show the pattern
    if name == "glob" and "pattern" in args:
        return str(args["pattern"])
    if name == "grep" and "pattern" in args:
        path = args.get("path", "")
        return f"/{args['pattern']}/ {path}".strip()

    # WebFetch: show URL
    if name == "webfetch" and "url" in args:
        return str(args["url"])

    # WebSearch: show query
    if name == "websearch" and "query" in args:
        return str(args["query"])

    # Agent: show the task
    if name == "agent" and "prompt" in args:
        prompt = str(args["prompt"])
        if len(prompt) > 120:
            prompt = prompt[:120] + "..."
        return prompt

    # MCP tools: show first string arg value
    for key, val in args.items():
        if isinstance(val, str) and val and len(val) < 150:
            return f"{key}: {val}"

    # Fallback: show arg keys
    return ", ".join(args.keys())


def format_event(event: TraceEvent, base_ts: float | None = None) -> str:
    """Format a single event as a colored terminal line."""
    color = EVENT_COLORS.get(event.event_type, C.WHITE)
    icon = EVENT_ICONS.get(event.event_type, " ")
    ts = _format_timestamp(event.timestamp, base_ts)
    duration = _format_duration(event.duration_ms)

    parts = [
        f"{C.GRAY}{ts}{C.RESET}",
        f"{color}{icon}{C.RESET}",
    ]

    if event.event_type == EventType.TOOL_CALL:
        name = event.data.get("tool_name", "?")
        args = event.data.get("arguments", {})
        parts.append(f"{color}{C.BOLD}tool_call{C.RESET} {C.WHITE}{name}{C.RESET}")
        # Show the most useful argument value inline
        detail = _tool_call_detail(name, args)
        if detail:
            parts.append(f"\n{C.GRAY}{'':>14}  {detail}{C.RESET}")

    elif event.event_type == EventType.TOOL_RESULT:
        name = event.data.get("tool_name", "")
        result = event.data.get("result", "")
        preview = event.data.get("content_preview", "")
        types = event.data.get("content_types", [])
        type_str = ",".join(types) if types else ""
        parts.append(f"{color}tool_result{C.RESET}")
        if name:
            parts.append(f"{C.DIM}{name}{C.RESET}")
        if type_str:
            parts.append(f"{C.DIM}[{type_str}]{C.RESET}")
        parts.append(f"{duration}")
        # Show a preview of the result
        output = result or preview
        if output:
            output = _strip_markdown(str(output))
            output_preview = output[:120]
            if len(output) > 120:
                output_preview += "..."
            parts.append(f"\n{C.GRAY}{'':>14}  {output_preview}{C.RESET}")

    elif event.event_type == EventType.LLM_REQUEST:
        model = event.data.get("model", "")
        count = event.data.get("message_count", 0)
        parts.append(f"{color}{C.BOLD}llm_request{C.RESET}")
        if model:
            parts.append(f"{C.DIM}{model}{C.RESET}")
        parts.append(f"{C.DIM}({count} messages){C.RESET}")

    elif event.event_type == EventType.LLM_RESPONSE:
        tokens = event.data.get("total_tokens", 0)
        parts.append(f"{color}llm_response{C.RESET}")
        if tokens:
            parts.append(f"{C.DIM}({tokens} tokens){C.RESET}")
        parts.append(f"{duration}")

    elif event.event_type == EventType.FILE_READ:
        uri = event.data.get("uri", "")
        parts.append(f"{color}file_read{C.RESET} {C.DIM}{uri}{C.RESET}")

    elif event.event_type == EventType.FILE_WRITE:
        uri = event.data.get("uri", "")
        parts.append(f"{color}file_write{C.RESET} {C.DIM}{uri}{C.RESET}")

    elif event.event_type == EventType.ERROR:
        msg = event.data.get("message", "") or event.data.get("error", "")
        name = event.data.get("tool_name", "")
        code = event.data.get("code", "")
        parts.append(f"{color}{C.BOLD}error{C.RESET}")
        if name:
            parts.append(f"{C.RED}{name}{C.RESET}")
        if code:
            parts.append(f"{C.DIM}(code: {code}){C.RESET}")
        if msg:
            msg = _strip_markdown(str(msg))
            msg_preview = msg[:150]
            if len(msg) > 150:
                msg_preview += "..."
            parts.append(f"\n{C.GRAY}{'':>14}  {C.RED}{msg_preview}{C.RESET}")

    elif event.event_type == EventType.SESSION_START:
        cmd = event.data.get("command", [])
        parts.append(f"{color}{C.BOLD}session_start{C.RESET}")
        if cmd:
            parts.append(f"{C.DIM}{' '.join(cmd)}{C.RESET}")

    elif event.event_type == EventType.SESSION_END:
        exit_code = event.data.get("exit_code", "?")
        parts.append(f"{color}{C.BOLD}session_end{C.RESET}")
        parts.append(f"{C.DIM}exit={exit_code}{C.RESET}")
        parts.append(f"{duration}")

    elif event.event_type == EventType.DECISION:
        choice = event.data.get("choice", "")
        reason = event.data.get("reason", "")
        parts.append(f"{color}{C.BOLD}decision{C.RESET} {C.WHITE}{choice}{C.RESET}")
        if reason:
            parts.append(f"\n{C.GRAY}{'':>14}  reason: {reason[:120]}{C.RESET}")

    elif event.event_type == EventType.USER_PROMPT:
        prompt = event.data.get("prompt", "")
        preview = prompt[:150]
        if len(prompt) > 150:
            preview += "..."
        parts.append(f"{color}{C.BOLD}user_prompt{C.RESET}")
        parts.append(f"\n{C.GRAY}{'':>14}  \"{preview}\"{C.RESET}")

    elif event.event_type == EventType.ASSISTANT_RESPONSE:
        text = event.data.get("text", "")
        text = _strip_markdown(text)
        preview = text[:200]
        if len(text) > 200:
            preview += "..."
        parts.append(f"{color}{C.BOLD}assistant_response{C.RESET}")
        if preview:
            parts.append(f"\n{C.GRAY}{'':>14}  \"{preview}\"{C.RESET}")

    return " ".join(parts)


def format_summary(meta: SessionMeta) -> str:
    """Format session summary."""
    started = datetime.fromtimestamp(meta.started_at, tz=timezone.utc)
    duration = meta.total_duration_ms / 1000 if meta.total_duration_ms else 0

    lines = [
        f"",
        f"{C.BOLD}Session Summary{C.RESET}",
        f"{C.GRAY}{'─' * 50}{C.RESET}",
        f"  Session:    {meta.session_id}",
        f"  Started:    {started.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"  Duration:   {duration:.2f}s",
        f"  Tool calls: {meta.tool_calls}",
        f"  LLM reqs:   {meta.llm_requests}",
        f"  Errors:     {meta.errors}",
    ]

    if meta.agent_name:
        lines.insert(4, f"  Agent:      {meta.agent_name}")
    if meta.command:
        lines.insert(4, f"  Command:    {meta.command}")

    lines.append(f"{C.GRAY}{'─' * 50}{C.RESET}")
    return "\n".join(lines)


def replay_session(
    store: TraceStore,
    session_id: str,
    event_filter: set[EventType] | None = None,
    speed: float = 1.0,
    live: bool = False,
    out: TextIO = sys.stdout,
) -> None:
    """Replay a trace session to the terminal."""
    meta = store.load_meta(session_id)
    events = store.load_events(session_id)

    if not events:
        out.write(f"No events found for session {session_id}\n")
        return

    if event_filter:
        events = [e for e in events if e.event_type in event_filter]

    base_ts = events[0].timestamp if events else None

    out.write(format_summary(meta) + "\n\n")

    prev_ts = base_ts
    for event in events:
        if live and prev_ts and speed > 0:
            delay = (event.timestamp - prev_ts) / speed
            if delay > 0:
                time.sleep(min(delay, 2.0))  # cap at 2s between events
            prev_ts = event.timestamp

        out.write(format_event(event, base_ts) + "\n")

    out.write("\n")


def list_sessions(store: TraceStore, out: TextIO = sys.stdout) -> None:
    """List all captured sessions."""
    sessions = store.list_sessions()

    if not sessions:
        out.write("No traces found.\n")
        return

    out.write(f"\n{C.BOLD}Captured Sessions{C.RESET}\n")
    out.write(f"{C.GRAY}{'─' * 70}{C.RESET}\n")
    out.write(
        f"  {C.DIM}{'ID':<18} {'Started':<22} {'Duration':>10} {'Tools':>6} {'LLM':>5} {'Err':>4}{C.RESET}\n"
    )
    out.write(f"{C.GRAY}{'─' * 70}{C.RESET}\n")

    for meta in sessions:
        started = datetime.fromtimestamp(meta.started_at, tz=timezone.utc)
        duration = meta.total_duration_ms / 1000 if meta.total_duration_ms else 0
        err_color = C.RED if meta.errors > 0 else C.DIM

        out.write(
            f"  {meta.session_id:<18} "
            f"{started.strftime('%Y-%m-%d %H:%M:%S'):<22} "
            f"{duration:>9.1f}s "
            f"{meta.tool_calls:>6} "
            f"{meta.llm_requests:>5} "
            f"{err_color}{meta.errors:>4}{C.RESET}\n"
        )

    out.write(f"{C.GRAY}{'─' * 70}{C.RESET}\n")
    out.write(f"  {len(sessions)} session(s)\n\n")


# ---------------------------------------------------------------------------
# HTML replay viewer (#40)
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>agent-strace: {session_id}</title>
<style>
  :root {{
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --dim: #8b949e; --green: #3fb950; --red: #f85149;
    --yellow: #d29922; --blue: #58a6ff; --purple: #bc8cff; --cyan: #39d353;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px; }}
  header {{ background: var(--surface); border-bottom: 1px solid var(--border); padding: 12px 20px; display: flex; align-items: center; gap: 16px; position: sticky; top: 0; z-index: 10; }}
  header h1 {{ font-size: 14px; font-weight: 600; }}
  .meta {{ color: var(--dim); font-size: 12px; }}
  .cost-counter {{ margin-left: auto; color: var(--green); font-weight: 600; }}
  .controls {{ display: flex; gap: 8px; }}
  button {{ background: var(--surface); border: 1px solid var(--border); color: var(--text); padding: 4px 12px; border-radius: 6px; cursor: pointer; font-size: 12px; }}
  button:hover {{ background: var(--border); }}
  #timeline {{ padding: 16px 20px; max-width: 1100px; margin: 0 auto; }}
  .event {{ border: 1px solid var(--border); border-radius: 8px; margin-bottom: 8px; overflow: hidden; opacity: 0; transform: translateY(4px); transition: opacity 0.2s, transform 0.2s; }}
  .event.visible {{ opacity: 1; transform: none; }}
  .event-header {{ display: flex; align-items: center; gap: 10px; padding: 8px 12px; cursor: pointer; user-select: none; }}
  .event-header:hover {{ background: rgba(255,255,255,0.03); }}
  .ts {{ color: var(--dim); font-size: 11px; min-width: 70px; }}
  .icon {{ font-size: 14px; min-width: 20px; text-align: center; }}
  .type {{ font-weight: 600; font-size: 12px; min-width: 130px; }}
  .summary {{ color: var(--dim); font-size: 12px; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .dur {{ color: var(--dim); font-size: 11px; margin-left: auto; }}
  .event-body {{ display: none; padding: 10px 12px; border-top: 1px solid var(--border); background: rgba(0,0,0,0.2); }}
  .event-body.open {{ display: block; }}
  pre {{ white-space: pre-wrap; word-break: break-all; font-size: 12px; color: var(--dim); max-height: 300px; overflow-y: auto; }}
  .type-tool_call {{ border-left: 3px solid var(--cyan); }}
  .type-tool_result {{ border-left: 3px solid var(--blue); }}
  .type-llm_request {{ border-left: 3px solid var(--purple); }}
  .type-llm_response {{ border-left: 3px solid var(--purple); }}
  .type-session_start, .type-session_end {{ border-left: 3px solid var(--green); }}
  .type-error {{ border-left: 3px solid var(--red); }}
  .type-user_prompt {{ border-left: 3px solid var(--green); }}
  .type-assistant_response {{ border-left: 3px solid var(--purple); }}
  .type-file_read, .type-file_write {{ border-left: 3px solid var(--yellow); }}
  #scrubber {{ width: 100%; margin: 8px 0; accent-color: var(--blue); }}
  .paused {{ opacity: 0.5; }}
</style>
</head>
<body>
<header>
  <div>
    <h1>agent-strace replay</h1>
    <div class="meta">{session_id} &nbsp;·&nbsp; {event_count} events &nbsp;·&nbsp; {duration}</div>
  </div>
  <div class="controls">
    <button id="btn-play" onclick="togglePlay()">⏸ Pause</button>
    <button onclick="showAll()">Show all</button>
  </div>
  <div class="cost-counter">$<span id="cost">0.0000</span></div>
</header>
<div id="timeline">
  <input type="range" id="scrubber" min="0" max="{max_idx}" value="0" oninput="scrubTo(this.value)">
</div>
<script>
const EVENTS = {events_json};
const COSTS = {costs_json};
let playing = true;
let currentIdx = 0;
let timer = null;

function fmtTs(offset) {{
  if (offset < 60) return '+' + offset.toFixed(1) + 's';
  return '+' + Math.floor(offset/60) + 'm' + (offset%60).toFixed(0).padStart(2,'0') + 's';
}}

function icon(type) {{
  const m = {{tool_call:'→',tool_result:'←',llm_request:'⬆',llm_response:'⬇',
    session_start:'▶',session_end:'■',file_read:'📖',file_write:'📝',
    error:'✗',user_prompt:'👤',assistant_response:'🤖',decision:'◆'}};
  return m[type] || '·';
}}

function summary(ev) {{
  const d = ev.data || {{}};
  if (ev.event_type === 'tool_call') return (d.tool_name||'') + ' ' + JSON.stringify(d.arguments||{{}}).slice(0,60);
  if (ev.event_type === 'tool_result') return String(d.content||d.result||'').slice(0,80);
  if (ev.event_type === 'llm_request') return (d.model||'') + ' (' + (d.message_count||0) + ' msgs)';
  if (ev.event_type === 'llm_response') return (d.total_tokens||0) + ' tokens';
  if (ev.event_type === 'user_prompt') return String(d.content||'').slice(0,80);
  if (ev.event_type === 'assistant_response') return String(d.content||'').slice(0,80);
  return '';
}}

function renderEvent(ev, idx) {{
  const div = document.createElement('div');
  div.className = 'event type-' + ev.event_type;
  div.id = 'ev-' + idx;
  const dur = ev.duration_ms ? (ev.duration_ms < 1000 ? ev.duration_ms.toFixed(0)+'ms' : (ev.duration_ms/1000).toFixed(1)+'s') : '';
  div.innerHTML = `
    <div class="event-header" onclick="toggle(${{idx}})">
      <span class="ts">${{fmtTs(ev._offset||0)}}</span>
      <span class="icon">${{icon(ev.event_type)}}</span>
      <span class="type">${{ev.event_type}}</span>
      <span class="summary">${{summary(ev)}}</span>
      <span class="dur">${{dur}}</span>
    </div>
    <div class="event-body" id="body-${{idx}}">
      <pre>${{JSON.stringify(ev.data||{{}}, null, 2)}}</pre>
    </div>`;
  return div;
}}

function toggle(idx) {{
  const body = document.getElementById('body-' + idx);
  body.classList.toggle('open');
}}

function showEvent(idx) {{
  if (idx >= EVENTS.length) {{ playing = false; return; }}
  const ev = EVENTS[idx];
  const div = renderEvent(ev, idx);
  document.getElementById('timeline').appendChild(div);
  requestAnimationFrame(() => div.classList.add('visible'));
  document.getElementById('cost').textContent = COSTS[idx].toFixed(4);
  document.getElementById('scrubber').value = idx;
  currentIdx = idx + 1;
}}

function showAll() {{
  clearTimeout(timer);
  playing = false;
  document.getElementById('btn-play').textContent = '▶ Play';
  while (currentIdx < EVENTS.length) showEvent(currentIdx);
}}

function scrubTo(val) {{
  clearTimeout(timer);
  playing = false;
  document.getElementById('btn-play').textContent = '▶ Play';
  const tl = document.getElementById('timeline');
  // Remove events after scrubber range
  const existing = tl.querySelectorAll('.event');
  existing.forEach(el => el.remove());
  currentIdx = 0;
  for (let i = 0; i <= parseInt(val); i++) showEvent(i);
}}

function togglePlay() {{
  playing = !playing;
  document.getElementById('btn-play').textContent = playing ? '⏸ Pause' : '▶ Play';
  if (playing) scheduleNext();
}}

function scheduleNext() {{
  if (!playing || currentIdx >= EVENTS.length) return;
  const ev = EVENTS[currentIdx];
  const next = EVENTS[currentIdx + 1];
  const delay = next ? Math.min((next._offset - ev._offset) * 1000 / 4, 800) : 0;
  showEvent(currentIdx);
  timer = setTimeout(scheduleNext, Math.max(delay, 80));
}}

// Start playback
scheduleNext();
</script>
</body>
</html>
"""


def replay_to_html(
    store: TraceStore,
    session_id: str,
    output_path: str | None = None,
) -> str:
    """Generate a self-contained HTML replay viewer for a session.

    Returns the HTML string. If output_path is given, also writes to disk.
    """
    import json as _json
    from .cost import _dollars, _event_tokens

    meta = store.load_meta(session_id)
    events = store.load_events(session_id)

    if not events:
        return "<html><body>No events found.</body></html>"

    base_ts = events[0].timestamp
    duration_s = events[-1].timestamp - base_ts
    if duration_s < 60:
        duration_str = f"{duration_s:.0f}s"
    else:
        duration_str = f"{int(duration_s)//60}m {int(duration_s)%60:02d}s"

    # Build event list with offset and running cost
    events_data = []
    costs_data = []
    running_cost = 0.0
    for e in events:
        d = dict(e.data)
        d_out = {
            "event_type": e.event_type.value,
            "event_id": e.event_id,
            "_offset": round(e.timestamp - base_ts, 3),
            "duration_ms": e.duration_ms,
            "data": d,
        }
        inp, out = _event_tokens(e)
        running_cost += _dollars(inp, out, "sonnet")
        events_data.append(d_out)
        costs_data.append(round(running_cost, 6))

    html = _HTML_TEMPLATE.format(
        session_id=session_id[:16],
        event_count=len(events),
        duration=duration_str,
        max_idx=len(events) - 1,
        events_json=_json.dumps(events_data, separators=(",", ":")),
        costs_json=_json.dumps(costs_data, separators=(",", ":")),
    )

    if output_path:
        Path(output_path).write_text(html, encoding="utf-8")

    return html
