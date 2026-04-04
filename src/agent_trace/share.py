"""Shareable HTML replay generator.

Produces a self-contained HTML file from a session trace.
No external dependencies. No CDN links. All CSS and JS inlined.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import TextIO

from .cost import estimate_cost
from .explain import explain_session
from .models import EventType, TraceEvent, SessionMeta
from .store import TraceStore


# ---------------------------------------------------------------------------
# Event rendering helpers
# ---------------------------------------------------------------------------

_EVENT_BADGE_CLASS = {
    EventType.SESSION_START: "badge-green",
    EventType.SESSION_END: "badge-green",
    EventType.TOOL_CALL: "badge-cyan",
    EventType.TOOL_RESULT: "badge-blue",
    EventType.LLM_REQUEST: "badge-purple",
    EventType.LLM_RESPONSE: "badge-purple",
    EventType.FILE_READ: "badge-yellow",
    EventType.FILE_WRITE: "badge-yellow",
    EventType.DECISION: "badge-white",
    EventType.ERROR: "badge-red",
    EventType.USER_PROMPT: "badge-green",
    EventType.ASSISTANT_RESPONSE: "badge-purple",
}


def _esc(text: str) -> str:
    return html.escape(str(text))


def _fmt_ts(offset: float) -> str:
    if offset < 60:
        return f"+{offset:.2f}s"
    m = int(offset) // 60
    s = offset % 60
    return f"+{m}m{s:05.2f}s"


def _event_summary(event: TraceEvent) -> str:
    d = event.data
    et = event.event_type
    if et == EventType.TOOL_CALL:
        name = d.get("tool_name", "?")
        args = d.get("arguments", {}) or {}
        if name.lower() == "bash":
            cmd = str(args.get("command", ""))[:120]
            return f"{name}: $ {cmd}"
        path = args.get("file_path") or args.get("path") or ""
        if path:
            return f"{name}: {path}"
        return name
    if et == EventType.TOOL_RESULT:
        result = str(d.get("result", d.get("content_preview", "")))[:100]
        return result or "(empty)"
    if et == EventType.USER_PROMPT:
        return str(d.get("prompt", ""))[:120]
    if et == EventType.ASSISTANT_RESPONSE:
        return str(d.get("text", ""))[:120]
    if et == EventType.ERROR:
        return str(d.get("message", d.get("error", "error")))[:120]
    if et == EventType.LLM_REQUEST:
        model = d.get("model", "")
        count = d.get("message_count", 0)
        return f"{model} ({count} messages)" if model else f"{count} messages"
    if et == EventType.LLM_RESPONSE:
        tokens = d.get("total_tokens", 0)
        return f"{tokens} tokens" if tokens else ""
    if et == EventType.FILE_READ:
        return str(d.get("uri", ""))
    if et == EventType.FILE_WRITE:
        return str(d.get("uri", ""))
    if et == EventType.SESSION_START:
        cmd = d.get("command", [])
        return " ".join(cmd) if isinstance(cmd, list) else str(cmd)
    if et == EventType.SESSION_END:
        return f"exit={d.get('exit_code', '?')}"
    return ""


def _render_event(event: TraceEvent, base_ts: float, is_error: bool = False) -> str:
    badge_cls = _EVENT_BADGE_CLASS.get(event.event_type, "badge-white")
    offset = event.event_type and (event.timestamp - base_ts)
    ts_str = _fmt_ts(event.timestamp - base_ts)
    summary = _esc(_event_summary(event))
    et_label = _esc(event.event_type.value)
    detail_json = _esc(json.dumps(event.data, indent=2))
    detail_id = f"ev-{event.event_id}"
    error_cls = " event-error" if is_error else ""
    expanded = " open" if is_error else ""

    return f"""
    <details class="event{error_cls}"{expanded} data-type="{et_label}">
      <summary>
        <span class="ts">{_esc(ts_str)}</span>
        <span class="badge {badge_cls}">{et_label}</span>
        <span class="event-summary">{summary}</span>
      </summary>
      <pre class="event-detail">{detail_json}</pre>
    </details>"""


def _render_phase(phase, base_ts: float, phase_cost: float | None = None) -> str:
    status_cls = " phase-failed" if phase.failed else ""
    status_label = " — FAILED" if phase.failed else ""
    cost_str = f"  ${phase_cost:.4f}" if phase_cost is not None else ""
    events_html = ""
    for event in phase.events:
        is_error = event.event_type == EventType.ERROR
        events_html += _render_event(event, base_ts, is_error=is_error)

    files_html = ""
    if phase.files_read:
        files_html += f'<div class="phase-meta">Read: {_esc(", ".join(phase.files_read[:8]))}</div>'
    if phase.files_written:
        files_html += f'<div class="phase-meta">Wrote: {_esc(", ".join(phase.files_written[:8]))}</div>'
    if phase.commands:
        cmds = [c[:80] for c in phase.commands[:5]]
        files_html += f'<div class="phase-meta">Ran: {_esc(", ".join(cmds))}</div>'

    return f"""
  <details class="phase{status_cls}" open>
    <summary class="phase-header">
      Phase {phase.index}: {_esc(phase.name)}{_esc(status_label)}
      <span class="phase-meta-inline">
        {phase.event_count} events · {phase.duration:.1f}s{_esc(cost_str)}
      </span>
    </summary>
    {files_html}
    <div class="events-list">
      {events_html}
    </div>
  </details>"""


# ---------------------------------------------------------------------------
# CSS + JS (inlined, no external URLs)
# ---------------------------------------------------------------------------

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #0d1117; color: #c9d1d9;
  font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', Consolas, monospace;
  font-size: 13px; line-height: 1.6; padding: 16px;
}
a { color: #58a6ff; }
h1 { font-size: 18px; color: #e6edf3; margin-bottom: 8px; }
h2 { font-size: 14px; color: #8b949e; margin: 16px 0 8px; text-transform: uppercase; letter-spacing: 0.05em; }
.header { border-bottom: 1px solid #21262d; padding-bottom: 16px; margin-bottom: 16px; }
.meta-grid { display: flex; flex-wrap: wrap; gap: 16px; margin-top: 8px; }
.meta-item { background: #161b22; border: 1px solid #21262d; border-radius: 6px; padding: 8px 12px; }
.meta-label { color: #8b949e; font-size: 11px; text-transform: uppercase; }
.meta-value { color: #e6edf3; font-size: 14px; font-weight: bold; }
.badge {
  display: inline-block; padding: 1px 6px; border-radius: 4px;
  font-size: 11px; font-weight: bold; margin-right: 6px; white-space: nowrap;
}
.badge-green  { background: #1a4731; color: #3fb950; }
.badge-cyan   { background: #0d2d3a; color: #39c5cf; }
.badge-blue   { background: #0d1f3a; color: #58a6ff; }
.badge-purple { background: #2d1f4a; color: #bc8cff; }
.badge-yellow { background: #3a2d0d; color: #e3b341; }
.badge-red    { background: #3a0d0d; color: #f85149; }
.badge-white  { background: #21262d; color: #c9d1d9; }
/* ── Search / filter bar ── */
.search-bar {
  position: sticky; top: 0; z-index: 100;
  background: #0d1117; border-bottom: 1px solid #21262d;
  padding: 10px 0 12px; margin-bottom: 12px;
  display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
}
.search-input {
  flex: 1; min-width: 200px;
  background: #161b22; border: 1px solid #30363d; border-radius: 6px;
  color: #c9d1d9; font-family: inherit; font-size: 13px;
  padding: 6px 10px; outline: none;
}
.search-input:focus { border-color: #58a6ff; }
.search-input::placeholder { color: #484f58; }
.filter-chips { display: flex; flex-wrap: wrap; gap: 4px; }
.chip {
  padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: bold;
  cursor: pointer; border: 1px solid transparent; user-select: none;
  transition: opacity 0.1s;
}
.chip.active  { opacity: 1; }
.chip.inactive { opacity: 0.35; }
.chip-all     { background: #21262d; color: #c9d1d9; border-color: #30363d; }
.chip-tool_call       { background: #0d2d3a; color: #39c5cf; }
.chip-tool_result     { background: #0d1f3a; color: #58a6ff; }
.chip-llm_request     { background: #2d1f4a; color: #bc8cff; }
.chip-llm_response    { background: #2d1f4a; color: #bc8cff; }
.chip-file_read       { background: #3a2d0d; color: #e3b341; }
.chip-file_write      { background: #3a2d0d; color: #e3b341; }
.chip-user_prompt     { background: #1a4731; color: #3fb950; }
.chip-assistant_response { background: #2d1f4a; color: #bc8cff; }
.chip-error           { background: #3a0d0d; color: #f85149; }
.chip-decision        { background: #21262d; color: #c9d1d9; }
.chip-session_start   { background: #1a4731; color: #3fb950; }
.chip-session_end     { background: #1a4731; color: #3fb950; }
.search-count { color: #484f58; font-size: 11px; white-space: nowrap; }
.search-highlight { background: #3a2d0d; color: #e3b341; border-radius: 2px; padding: 0 1px; }
.event-hidden { display: none !important; }
.phase-all-hidden > .events-list { display: none; }
/* ── Phases ── */
.phase {
  border: 1px solid #21262d; border-radius: 8px;
  margin-bottom: 12px; overflow: hidden;
}
.phase-failed { border-color: #6e2020; }
.phase-header {
  background: #161b22; padding: 10px 14px; cursor: pointer;
  display: flex; align-items: center; gap: 8px; list-style: none;
  color: #e6edf3; font-weight: bold;
}
.phase-failed > summary { background: #1f1010; color: #f85149; }
.phase-meta-inline { color: #8b949e; font-size: 11px; font-weight: normal; margin-left: auto; }
.phase-meta { color: #8b949e; font-size: 11px; padding: 4px 14px; border-bottom: 1px solid #21262d; }
.events-list { padding: 4px 8px; }
.event { border-bottom: 1px solid #161b22; }
.event > summary {
  padding: 5px 6px; cursor: pointer; display: flex; align-items: baseline;
  gap: 6px; list-style: none; white-space: nowrap; overflow: hidden;
}
.event > summary:hover { background: #161b22; }
.event-error > summary { background: #1a0d0d; }
.event-error > summary:hover { background: #200f0f; }
.ts { color: #484f58; font-size: 11px; min-width: 80px; }
.event-summary { color: #8b949e; overflow: hidden; text-overflow: ellipsis; }
.event-detail {
  background: #010409; color: #c9d1d9; padding: 10px 14px;
  font-size: 12px; overflow-x: auto; white-space: pre; border-top: 1px solid #21262d;
}
.cost-table { width: 100%; border-collapse: collapse; margin-top: 8px; }
.cost-table th { color: #8b949e; font-size: 11px; text-align: left; padding: 4px 8px; border-bottom: 1px solid #21262d; }
.cost-table td { padding: 4px 8px; border-bottom: 1px solid #161b22; }
.cost-wasted { color: #f85149; }
.footer { margin-top: 24px; padding-top: 16px; border-top: 1px solid #21262d; color: #484f58; font-size: 11px; }
@media (max-width: 768px) {
  body { padding: 8px; font-size: 12px; }
  .meta-grid { gap: 8px; }
  .event > summary { white-space: normal; }
  .search-bar { position: static; }
}
"""

_JS = """
(function() {
  // Collect all event types present in this session
  var allEvents = Array.from(document.querySelectorAll('details.event'));
  var typesPresent = {};
  allEvents.forEach(function(el) {
    var badge = el.querySelector('.badge');
    if (badge) typesPresent[badge.textContent.trim()] = true;
  });

  // Build filter chip set (only types that exist in this session)
  var filterBar = document.getElementById('filter-bar');
  var chipsContainer = filterBar.querySelector('.filter-chips');
  var activeTypes = new Set(['__all__']);

  // "All" chip
  var allChip = document.createElement('span');
  allChip.className = 'chip chip-all active';
  allChip.dataset.type = '__all__';
  allChip.textContent = 'all';
  chipsContainer.appendChild(allChip);

  Object.keys(typesPresent).sort().forEach(function(t) {
    var chip = document.createElement('span');
    var cls = t.replace(/_/g, '_');
    chip.className = 'chip chip-' + cls + ' active';
    chip.dataset.type = t;
    chip.textContent = t.replace(/_/g, '\u200b_'); // allow wrap at underscores
    chipsContainer.appendChild(chip);
  });

  var searchInput = document.getElementById('search-input');
  var countEl = document.getElementById('search-count');

  function escapeRe(s) {
    return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  function stripHighlights(el) {
    el.querySelectorAll('.search-highlight').forEach(function(h) {
      h.replaceWith(document.createTextNode(h.textContent));
    });
    // Normalize adjacent text nodes
    el.normalize();
  }

  function highlightText(el, re) {
    // Only highlight in .event-summary and .badge text nodes
    var targets = el.querySelectorAll('.event-summary, .badge');
    targets.forEach(function(target) {
      target.childNodes.forEach(function(node) {
        if (node.nodeType !== 3) return; // text nodes only
        var text = node.textContent;
        if (!re.test(text)) return;
        re.lastIndex = 0;
        var frag = document.createDocumentFragment();
        var last = 0, m;
        while ((m = re.exec(text)) !== null) {
          if (m.index > last) frag.appendChild(document.createTextNode(text.slice(last, m.index)));
          var mark = document.createElement('mark');
          mark.className = 'search-highlight';
          mark.textContent = m[0];
          frag.appendChild(mark);
          last = re.lastIndex;
          if (m[0].length === 0) { re.lastIndex++; break; }
        }
        if (last < text.length) frag.appendChild(document.createTextNode(text.slice(last)));
        node.parentNode.replaceChild(frag, node);
      });
    });
  }

  function applyFilters() {
    var query = searchInput.value.trim().toLowerCase();
    var re = query ? new RegExp(escapeRe(query), 'gi') : null;
    var showAll = activeTypes.has('__all__');
    var visible = 0;

    allEvents.forEach(function(el) {
      // Remove old highlights
      stripHighlights(el);

      var badge = el.querySelector('.badge');
      var evType = badge ? badge.textContent.trim() : '';
      var summaryEl = el.querySelector('.event-summary');
      var summaryText = summaryEl ? summaryEl.textContent.toLowerCase() : '';
      var detailEl = el.querySelector('.event-detail');
      var detailText = detailEl ? detailEl.textContent.toLowerCase() : '';

      // Type filter
      var typeMatch = showAll || activeTypes.has(evType);

      // Text filter: match against badge + summary + detail JSON
      var textMatch = !query ||
        evType.toLowerCase().includes(query) ||
        summaryText.includes(query) ||
        detailText.includes(query);

      if (typeMatch && textMatch) {
        el.classList.remove('event-hidden');
        visible++;
        if (re) highlightText(el, re);
        // Auto-open if text matched inside detail
        if (re && detailText.includes(query) && !el.open) {
          el.open = true;
        }
      } else {
        el.classList.add('event-hidden');
      }
    });

    // Hide phases where all events are hidden
    document.querySelectorAll('details.phase').forEach(function(phase) {
      var phaseEvents = phase.querySelectorAll('details.event');
      var anyVisible = Array.from(phaseEvents).some(function(e) {
        return !e.classList.contains('event-hidden');
      });
      if (phaseEvents.length > 0) {
        phase.classList.toggle('event-hidden', !anyVisible);
      }
    });

    var total = allEvents.length;
    countEl.textContent = query || !showAll
      ? visible + ' / ' + total + ' events'
      : total + ' events';
  }

  // Chip click handler
  chipsContainer.addEventListener('click', function(e) {
    var chip = e.target.closest('.chip');
    if (!chip) return;
    var type = chip.dataset.type;

    if (type === '__all__') {
      // Toggle: if all active → deactivate all type chips, keep only "all"
      // If "all" clicked while some inactive → activate everything
      var allActive = Array.from(chipsContainer.querySelectorAll('.chip:not(.chip-all)'))
        .every(function(c) { return c.classList.contains('active'); });
      activeTypes.clear();
      activeTypes.add('__all__');
      chipsContainer.querySelectorAll('.chip').forEach(function(c) {
        c.classList.toggle('active', true);
        c.classList.toggle('inactive', false);
      });
    } else {
      // Clicking a type chip: deselect "all", toggle this type
      activeTypes.delete('__all__');
      var allChipEl = chipsContainer.querySelector('.chip-all');
      allChipEl.classList.remove('active');
      allChipEl.classList.add('inactive');

      if (activeTypes.has(type)) {
        activeTypes.delete(type);
        chip.classList.remove('active');
        chip.classList.add('inactive');
      } else {
        activeTypes.add(type);
        chip.classList.add('active');
        chip.classList.remove('inactive');
      }

      // If all type chips are now active, restore "all"
      var typeChips = Array.from(chipsContainer.querySelectorAll('.chip:not(.chip-all)'));
      if (typeChips.every(function(c) { return activeTypes.has(c.dataset.type); })) {
        activeTypes.add('__all__');
        allChipEl.classList.add('active');
        allChipEl.classList.remove('inactive');
      }
    }
    applyFilters();
  });

  // Search input handler (debounced)
  var debounceTimer;
  searchInput.addEventListener('input', function() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(applyFilters, 120);
  });

  // Keyboard shortcut: / to focus search
  document.addEventListener('keydown', function(e) {
    if (e.key === '/' && document.activeElement !== searchInput) {
      e.preventDefault();
      searchInput.focus();
      searchInput.select();
    }
    if (e.key === 'Escape' && document.activeElement === searchInput) {
      searchInput.value = '';
      applyFilters();
      searchInput.blur();
    }
  });

  // Initial count
  applyFilters();
})();
"""


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

def render_html(
    store: TraceStore,
    session_id: str,
    postmortem_html: str = "",
) -> str:
    meta = store.load_meta(session_id)
    events = store.load_events(session_id)
    explain = explain_session(store, session_id)

    try:
        cost_result = estimate_cost(store, session_id)
        total_cost_str = f"${cost_result.total_cost:.4f}"
        phase_costs = {pc.phase_index: pc.cost_dollars for pc in cost_result.phase_costs}
        wasted_cost = cost_result.wasted_cost
    except Exception:
        cost_result = None
        total_cost_str = "n/a"
        phase_costs = {}
        wasted_cost = 0.0

    base_ts = events[0].timestamp if events else meta.started_at
    started = datetime.fromtimestamp(meta.started_at, tz=timezone.utc)
    duration = meta.total_duration_ms / 1000 if meta.total_duration_ms else 0.0
    status = "FAILED" if any(p.failed for p in explain.phases) else "OK"
    status_color = "#f85149" if status == "FAILED" else "#3fb950"

    # Header meta items
    meta_items = [
        ("Session", meta.session_id),
        ("Started", started.strftime("%Y-%m-%d %H:%M:%S UTC")),
        ("Duration", f"{duration:.1f}s"),
        ("Events", str(len(events))),
        ("Tool calls", str(meta.tool_calls)),
        ("LLM requests", str(meta.llm_requests)),
        ("Errors", str(meta.errors)),
        ("Est. cost", total_cost_str),
        ("Status", status),
    ]
    if meta.agent_name:
        meta_items.insert(1, ("Agent", meta.agent_name))

    meta_html = ""
    for label, value in meta_items:
        color = status_color if label == "Status" else ""
        style = f' style="color:{color}"' if color else ""
        meta_html += f"""
      <div class="meta-item">
        <div class="meta-label">{_esc(label)}</div>
        <div class="meta-value"{style}>{_esc(value)}</div>
      </div>"""

    # Phases
    phases_html = ""
    for phase in explain.phases:
        pc = phase_costs.get(phase.index)
        phases_html += _render_phase(phase, base_ts, phase_cost=pc)

    # Cost table
    cost_html = ""
    if cost_result and cost_result.phase_costs:
        rows = ""
        for pc in cost_result.phase_costs:
            wasted_cls = ' class="cost-wasted"' if pc.failed else ""
            rows += f"""
        <tr{wasted_cls}>
          <td>Phase {pc.phase_index}</td>
          <td>{_esc(pc.phase_name[:40])}</td>
          <td>{pc.input_tokens:,}</td>
          <td>{pc.output_tokens:,}</td>
          <td>${pc.cost_dollars:.4f}{"  ← wasted" if pc.failed else ""}</td>
        </tr>"""
        cost_html = f"""
    <h2>Cost Breakdown</h2>
    <table class="cost-table">
      <thead><tr><th>#</th><th>Phase</th><th>Input tok</th><th>Output tok</th><th>Cost</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    <p style="margin-top:8px;color:#8b949e">
      Total: {cost_result.input_tokens:,} input + {cost_result.output_tokens:,} output tokens
      = <strong>{total_cost_str}</strong>
      {f'· <span class="cost-wasted">Wasted: ${wasted_cost:.4f}</span>' if wasted_cost > 0 else ""}
    </p>"""

    generated = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>agent-trace: {_esc(meta.session_id)}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="header">
  <h1>agent-trace session replay</h1>
  <div class="meta-grid">{meta_html}
  </div>
</div>

{postmortem_html}

<div class="search-bar" id="filter-bar">
  <input
    id="search-input"
    class="search-input"
    type="search"
    placeholder="Search events… (press / to focus, Esc to clear)"
    autocomplete="off"
    spellcheck="false"
  >
  <div class="filter-chips"></div>
  <span class="search-count" id="search-count"></span>
</div>

<h2>Session Phases</h2>
{phases_html}

{cost_html}

<div class="footer">
  Generated by agent-strace · {generated} · {len(events)} events
</div>
<script>{_JS}</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def cmd_share(args: argparse.Namespace) -> int:
    store = TraceStore(args.trace_dir)

    session_id = args.session_id
    if not session_id:
        session_id = store.get_latest_session_id()
    if not session_id:
        sys.stderr.write("No sessions found.\n")
        return 1
    full_id = store.find_session(session_id)
    if not full_id:
        sys.stderr.write(f"Session not found: {session_id}\n")
        return 1

    postmortem_html = ""
    if getattr(args, "postmortem", False):
        from .postmortem import analyze_session, render_postmortem_html
        report = analyze_session(store, full_id)
        postmortem_html = render_postmortem_html(report)

    content = render_html(store, full_id, postmortem_html=postmortem_html)

    if getattr(args, "stdout", False):
        sys.stdout.write(content)
        return 0

    out_path = getattr(args, "output", None) or f"session-{full_id[:12]}.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)

    size_kb = len(content.encode()) // 1024
    sys.stderr.write(f"Created: {out_path} ({size_kb}KB)\n")

    if getattr(args, "open", False):
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", out_path], check=False)
            elif sys.platform.startswith("linux"):
                subprocess.run(["xdg-open", out_path], check=False)
            elif sys.platform == "win32":
                os.startfile(out_path)  # type: ignore[attr-defined]
        except Exception:
            pass

    return 0
