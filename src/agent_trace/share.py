"""Portable HTML evidence packet generator.

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
from .disclosure import (
    DisclosurePreview,
    NonObjectEventDataError,
    require_object_event_data,
)
from .evidence_packet import EvidencePacket, build_evidence_packet
from .explain import ExplainResult, build_phases
from .models import EventType, TraceEvent, SessionMeta
from .store import TraceStore
from .subagent import build_tree, aggregate_stats, SessionNode
from .annotate import LABEL_COLOURS, DEFAULT_LABEL_COLOUR


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
    EventType.SUBAGENT_START: "badge-cyan",
    EventType.SUBAGENT_STOP: "badge-cyan",
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
        if d.get("content_omitted"):
            effect = d.get("effect", "tool")
            attempt = d.get("attempt", 1)
            suffix = ""
            if d.get("recovery_attempt"):
                suffix = f" recovery attempt {attempt}"
            elif d.get("retry"):
                suffix = f" retry {attempt}"
            return f"{name}: {effect} input omitted{suffix}"
        args = d.get("arguments", {}) or {}
        if name.lower() == "bash":
            cmd = str(args.get("command", ""))[:120]
            return f"{name}: $ {cmd}"
        path = args.get("file_path") or args.get("path") or ""
        if path:
            return f"{name}: {path}"
        return name
    if et == EventType.TOOL_RESULT:
        if d.get("content_omitted"):
            outcome = d.get("outcome", "recorded")
            suffix = " and recovered" if d.get("recovered") else ""
            return f"tool result {outcome}{suffix}; value omitted"
        result = str(d.get("result", d.get("content_preview", "")))[:100]
        return result or "(empty)"
    if et == EventType.USER_PROMPT:
        if d.get("content_omitted"):
            return "prompt omitted"
        return str(d.get("prompt", ""))[:120]
    if et == EventType.ASSISTANT_RESPONSE:
        if d.get("content_omitted"):
            return "response omitted"
        return str(d.get("text", ""))[:120]
    if et == EventType.ERROR:
        if d.get("content_omitted"):
            return "failure recorded; details omitted"
        return str(d.get("message", d.get("error", "error")))[:120]
    if et == EventType.LLM_REQUEST:
        model = d.get("model", "")
        count = d.get("message_count", 0)
        return f"{model} ({count} messages)" if model else f"{count} messages"
    if et == EventType.LLM_RESPONSE:
        tokens = d.get("total_tokens", 0)
        return f"{tokens} tokens" if tokens else ""
    if et == EventType.FILE_READ:
        if d.get("content_omitted"):
            return f"file read; {d.get('path_kind', 'path')} value omitted"
        return str(d.get("uri", ""))
    if et == EventType.FILE_WRITE:
        if d.get("content_omitted"):
            return f"file write; {d.get('path_kind', 'path')} value omitted"
        return str(d.get("uri", ""))
    if et == EventType.SESSION_START:
        cmd = d.get("command", [])
        return " ".join(cmd) if isinstance(cmd, list) else str(cmd)
    if et == EventType.SESSION_END:
        return f"exit={d.get('exit_code', '?')}"
    if et in (EventType.SUBAGENT_START, EventType.SUBAGENT_STOP):
        if d.get("content_omitted"):
            agent = str(d.get("agent_type") or "subagent")
            return f"{agent}; content omitted"
        agent = str(d.get("agent_type") or d.get("agent_name") or "subagent")
        child = d.get("child_session_id", "")
        return f"{agent} child={child}" if child else agent
    return ""


def _render_event(event: TraceEvent, base_ts: float, is_error: bool = False) -> str:
    badge_cls = _EVENT_BADGE_CLASS.get(event.event_type, "badge-white")
    ts_str = _fmt_ts(event.timestamp - base_ts)
    summary = _esc(_event_summary(event))
    et_label = _esc(event.event_type.value)
    detail_json = _esc(json.dumps(event.data, indent=2))
    error_cls = " event-error" if is_error else ""
    expanded = " open" if is_error else ""

    return f"""
    <details id="ev-{_esc(event.event_id)}" class="event{error_cls}"{expanded} data-type="{et_label}">
      <summary>
        <span class="ts">{_esc(ts_str)}</span>
        <span class="badge {badge_cls}">{et_label}</span>
        <span class="event-summary">{summary}</span>
      </summary>
      <pre class="event-detail">{detail_json}</pre>
    </details>"""


def _render_phase(phase, base_ts: float, phase_cost: float | None = None) -> str:
    status_cls = " phase-failed" if phase.failed else ""
    status_label = " - FAILED" if phase.failed else ""
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

_JS = r"""
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

def _render_subagent_tree_html(tree: "SessionNode", stats: object) -> str:
    """Render a collapsible subagent hierarchy section for the share HTML."""

    def _node_html(node: "SessionNode", depth: int = 0) -> str:
        indent = depth * 20
        sid = node.meta.session_id[:12]
        dur = node.meta.total_duration_ms / 1000 if node.meta.total_duration_ms else 0
        err_badge = (
            f'<span style="color:#f85149;margin-left:8px">✗ {node.meta.errors} err</span>'
            if node.meta.errors else ""
        )
        children_html = "".join(_node_html(c, depth + 1) for c in node.children)
        toggle = "▶" if node.children else "·"
        return f"""
<div class="tree-node" style="margin-left:{indent}px">
  <span class="tree-toggle">{toggle}</span>
  <span class="tree-sid">{_esc(sid)}</span>
  <span class="tree-meta">
    depth={node.meta.depth}
    &nbsp;·&nbsp;{dur:.1f}s
    &nbsp;·&nbsp;{node.meta.tool_calls} tools
    &nbsp;·&nbsp;{node.meta.total_tokens:,} tok
    {err_badge}
  </span>
  {children_html}
</div>"""

    tree_body = _node_html(tree)
    agg = stats
    return f"""
<h2>Subagent Tree</h2>
<div style="background:#161b22;border:1px solid #30363d;border-radius:6px;padding:16px;margin-bottom:20px;font-size:.85em">
  <div style="color:#8b949e;margin-bottom:12px">
    {getattr(agg,'session_count',1)} sessions &nbsp;·&nbsp;
    {getattr(agg,'tool_calls',0)} total tool calls &nbsp;·&nbsp;
    {getattr(agg,'total_tokens',0):,} total tokens
    {f'&nbsp;·&nbsp;<span style="color:#f85149">{agg.errors} errors</span>' if getattr(agg,'errors',0) else ''}
  </div>
  <div class="subagent-tree">{tree_body}</div>
</div>
<style>
.tree-node{{padding:3px 0;font-family:monospace}}
.tree-toggle{{color:#8b949e;margin-right:6px;cursor:pointer}}
.tree-sid{{color:#58a6ff;font-weight:bold}}
.tree-meta{{color:#8b949e;margin-left:10px;font-size:.9em}}
</style>"""


def _packet_explain(packet: EvidencePacket) -> ExplainResult:
    events = list(packet.events)
    base_ts = events[0].timestamp if events else packet.meta.started_at
    phases = build_phases(events, base_ts)
    for phase in phases:
        if not phase.name:
            phase.name = f"Phase {phase.index}"
    duration = packet.meta.total_duration_ms / 1000 if packet.meta.total_duration_ms else 0.0
    return ExplainResult(
        session_id=packet.meta.session_id,
        total_duration=duration,
        total_events=len(events),
        phases=phases,
        total_retries=packet.behavior.retry_count,
        wasted_seconds=sum(phase.duration for phase in phases if phase.failed),
    )


def _render_packet_summary(packet: EvidencePacket) -> str:
    behavior = packet.behavior
    health = packet.manifest["evidence_health"]
    reasons = "".join(
        (
            f"<li><strong>{_esc(reason['code'])}</strong>"
            f"{': ' + _esc(reason['message']) if reason.get('message') else ''}</li>"
        )
        for reason in health["reasons"]
    )
    if not reasons:
        reasons = "<li>No evidence gaps or structural issues were detected.</li>"
    behavior_rows = (
        ("Failure events", behavior.failure_event_count),
        ("Failed operations", behavior.failed_operation_count),
        ("Retries", behavior.retry_count),
        ("Recovery attempts", behavior.recovery_attempt_count),
        ("Recovered operations", behavior.recovered_operation_count),
        ("Unresolved failed operations", behavior.unresolved_failed_operation_count),
    )
    behavior_html = "".join(
        f"<tr><td>{_esc(label)}</td><td>{value}</td></tr>"
        for label, value in behavior_rows
    )
    privacy = packet.manifest["privacy"]
    content_mode = privacy["content"]
    annotation_mode = privacy["annotations"]
    redaction = privacy["redaction"]
    anonymization = privacy["anonymization"]
    return f"""
<h2>Evidence Health</h2>
<div class="phase-meta">
  Status: <strong>{_esc(health["status"])}</strong>
  | Provider: {_esc(packet.manifest["provenance"]["provider"])}
  | Capture: {_esc(packet.manifest["provenance"]["capture_method"])}
</div>
<ul style="margin:8px 0 16px 20px">{reasons}</ul>

<h2>Failure, Retry, and Recovery</h2>
<table class="cost-table">
  <tbody>{behavior_html}</tbody>
</table>

<h2>Disclosure</h2>
<div class="phase-meta">
  Content: {_esc(content_mode)}
  | Annotations: {_esc(annotation_mode)}
  | Redaction: {_esc(redaction["mode"])}
  | Anonymization: {_esc(anonymization["mode"])}
</div>"""


def render_html(
    store: TraceStore,
    session_id: str,
    postmortem_html: str = "",
    *,
    include_content: bool = False,
    packet: EvidencePacket | None = None,
) -> str:
    packet = packet or build_evidence_packet(
        store,
        session_id,
        include_content=include_content,
    )
    content_included = packet.preview.mode == "include_content"
    if postmortem_html and not content_included:
        raise ValueError("postmortem HTML requires include_content mode")
    meta = packet.meta
    events = list(packet.events)
    require_object_event_data(events)
    explain = _packet_explain(packet)

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
        ("Mode", packet.preview.mode),
        ("Started", started.strftime("%Y-%m-%d %H:%M:%S UTC")),
        ("Duration", f"{duration:.1f}s"),
        ("Events", str(len(events))),
        ("Tool calls", str(meta.tool_calls)),
        ("LLM requests", str(meta.llm_requests)),
        ("Errors", str(meta.errors)),
        ("Est. cost", total_cost_str),
        ("Status", status),
        ("Evidence health", packet.health.result.status.value),
        ("Provider", packet.health.result.provider or "unknown"),
        ("Capture", packet.health.result.capture_method or "unknown"),
        ("Redaction", packet.health.redaction.mode),
    ]
    if content_included and meta.agent_name:
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
            phase_name = pc.phase_name
            if not content_included and 0 < pc.phase_index <= len(explain.phases):
                phase_name = explain.phases[pc.phase_index - 1].name
            rows += f"""
        <tr{wasted_cls}>
          <td>Phase {pc.phase_index}</td>
          <td>{_esc(phase_name[:40])}</td>
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

    # Subagent tree section
    tree_html = ""
    if content_included:
        try:
            tree = build_tree(store, session_id)
            if tree.children:
                stats = aggregate_stats(tree)
                tree_html = _render_subagent_tree_html(tree, stats)
        except Exception:
            pass

    # Annotations
    annotations = packet.annotations

    # Bookmark sidebar
    bookmarks_html = ""
    if annotations:
        items = ""
        for ann in annotations:
            label_str = f"[{ann.label}] " if ann.label else ""
            note_str = ann.note[:60] if ann.note else ann.annotation_id
            colour = LABEL_COLOURS.get(ann.label, DEFAULT_LABEL_COLOUR)
            items += (
                f'<li><a href="#ev-{_esc(ann.event_id)}" style="color:{colour}">'
                f'{_esc(label_str)}{_esc(note_str)}</a></li>\n'
            )
        bookmarks_html = f"""
<div class="bookmarks">
  <h3>Bookmarks</h3>
  <ul>{items}</ul>
</div>"""

    generated = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    packet_summary_html = _render_packet_summary(packet)
    manifest_json = json.dumps(
        packet.manifest,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    ).replace("</", "<\\/")
    if content_included:
        warning_html = """
<div style="background:#3a0d0d;border:1px solid #f85149;border-radius:6px;padding:12px;margin-bottom:16px">
  <strong>Full-content packet.</strong>
  This file may contain prompts, responses, commands, paths, tool inputs, tool results,
  secrets, personal data, and proprietary material.
</div>"""
    else:
        warning_html = """
<div style="background:#0d2d3a;border:1px solid #39c5cf;border-radius:6px;padding:12px;margin-bottom:16px">
  <strong>Privacy-minimized packet.</strong>
  Content-rich values are omitted. Only annotations explicitly marked shareable are included.
</div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>agent-trace: {_esc(meta.session_id)}</title>
<style>{_CSS}</style>
</head>
<body>
{warning_html}
<div class="header">
  <h1>agent-strace evidence packet</h1>
  <div class="meta-grid">{meta_html}
  </div>
</div>
{bookmarks_html}

{postmortem_html}

{tree_html}

{packet_summary_html}

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

<h2>Disclosure Manifest</h2>
<details class="phase">
  <summary class="phase-header">
    Schema {_esc(packet.manifest["schema"])} v{packet.manifest["schema_version"]}
  </summary>
  <pre class="event-detail">{_esc(manifest_json)}</pre>
</details>
<script type="application/json" id="agent-evidence-manifest">{manifest_json}</script>

<div class="footer">
  Generated by agent-strace {_esc(packet.manifest["tool"]["version"])}
  | {generated} | {len(events)} events
</div>
<script>{_JS}</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Disclosure preview (--dry-run)
# ---------------------------------------------------------------------------

_DISCLOSURE_TEXT_ROWS = (
    ("events", "event_count"),
    ("prompts", "prompt_count"),
    ("responses", "response_count"),
    ("tool inputs", "tool_input_count"),
    ("tool results", "tool_result_count"),
    ("command-bearing events", "command_count"),
    ("path-bearing events", "path_count"),
    ("included annotations", "annotation_count"),
    ("source annotations", "source_annotation_count"),
    ("omitted annotations", "omitted_annotation_count"),
    ("content-rich events", "content_rich_event_count"),
    ("minimized content-rich events", "minimized_content_rich_event_count"),
    ("redacted events", "redacted_event_count"),
    ("unredacted content-rich events", "unredacted_content_rich_event_count"),
    ("sensitive-field warnings", "unresolved_sensitive_field_warning_count"),
    ("anonymization actions", "anonymization_action_count"),
    ("known evidence gaps", "known_gap_count"),
)


def build_share_disclosure_preview(
    store: TraceStore,
    session_id: str,
    *,
    include_content: bool = False,
) -> DisclosurePreview:
    """Count what the selected share mode would disclose."""
    return build_evidence_packet(
        store,
        session_id,
        include_content=include_content,
    ).preview


def format_disclosure_preview(session_id: str, preview: DisclosurePreview) -> str:
    """Render a disclosure preview as text. Prints counts, never values."""
    counts = preview.to_dict()
    width = max(len(label) for label, _ in _DISCLOSURE_TEXT_ROWS) + 1
    lines = [
        "Share disclosure preview (dry run): no HTML written or opened.",
        f"Session: {session_id}",
        f"Mode: {preview.mode}",
        f"Evidence health: {preview.evidence_health_status}",
        f"Provider: {preview.provider}",
        f"Capture method: {preview.capture_method}",
        f"Capture matrix: {preview.capture_matrix_version}",
        f"Redaction: {preview.redaction_mode}",
        f"Anonymization: {preview.anonymization_mode}",
        "",
    ]
    lines.extend(
        f"  {label + ':':<{width}}  {counts[key]}" for label, key in _DISCLOSURE_TEXT_ROWS
    )
    lines.extend([
        "",
        "Included field categories:",
        *(f"  - {category}" for category in preview.included_field_categories),
        "",
        "Omitted field categories:",
        *(
            (f"  - {category}" for category in preview.omitted_field_categories)
            if preview.omitted_field_categories
            else ("  - none",)
        ),
        "",
        "Warnings:",
        *(f"  - {warning}" for warning in preview.warnings),
        "",
        "Counts only. No prompt, response, command, path, result, or annotation text is shown.",
        "Re-run without --dry-run to write the selected evidence packet.",
    ])
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def cmd_share(args: argparse.Namespace) -> int:
    dry_run = getattr(args, "dry_run", False)
    preview_format = getattr(args, "format", None)
    include_content = bool(getattr(args, "include_content", False))
    if preview_format is not None and not dry_run:
        sys.stderr.write("Error: --format requires --dry-run\n")
        return 1

    if getattr(args, "assignment", False):
        if include_content:
            sys.stderr.write(
                "Error: --include-content cannot be combined with --assignment; "
                "assignment bundles use their own privacy format\n"
            )
            return 1
        if dry_run:
            sys.stderr.write(
                "Error: --dry-run previews ordinary share HTML and cannot be combined "
                "with --assignment\n"
            )
            return 1
        assignment_store = TraceStore(args.trace_dir)
        assignment_session_id = args.session_id or assignment_store.get_latest_session_id()
        if assignment_session_id:
            assignment_full_id = assignment_store.find_session(assignment_session_id)
            if assignment_full_id:
                try:
                    require_object_event_data(
                        assignment_store.load_events(assignment_full_id)
                    )
                except NonObjectEventDataError:
                    sys.stderr.write("Error: session event data must be an object\n")
                    return 1
        from .assignment import cmd_share_assignment
        return cmd_share_assignment(args)

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

    # One check before any mode runs, so --dry-run refuses exactly the sessions
    # that share refuses, and nothing is written, printed, or opened.
    try:
        require_object_event_data(store.load_events(full_id))
    except NonObjectEventDataError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1

    if getattr(args, "postmortem", False) and not include_content:
        sys.stderr.write(
            "Error: --postmortem includes content-rich values and now requires "
            "--include-content. The default packet already preserves value-free "
            "failure, retry, and recovery signals.\n"
        )
        return 1

    try:
        packet = build_evidence_packet(
            store,
            full_id,
            include_content=include_content,
        )
    except (TypeError, ValueError) as exc:
        sys.stderr.write(f"Error: could not build evidence packet: {exc}\n")
        return 1

    if dry_run:
        preview = packet.preview
        if preview_format == "json":
            payload = {"session_id": full_id, **preview.to_dict()}
            sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        else:
            sys.stdout.write(format_disclosure_preview(full_id, preview))
        ignored = [
            flag
            for flag, attr in (("--output", "output"), ("--stdout", "stdout"), ("--open", "open"))
            if getattr(args, attr, None)
        ]
        if ignored:
            sys.stderr.write(
                f"Note: --dry-run ignores {', '.join(ignored)}; no HTML was written or opened.\n"
            )
        return 0

    postmortem_html = ""
    if getattr(args, "postmortem", False):
        from .postmortem import analyze_session, render_postmortem_html
        report = analyze_session(store, full_id)
        postmortem_html = render_postmortem_html(report)

    content = render_html(
        store,
        full_id,
        postmortem_html=postmortem_html,
        include_content=include_content,
        packet=packet,
    )

    if getattr(args, "stdout", False):
        if include_content:
            sys.stderr.write(
                "Warning: writing a full-content evidence packet to stdout; "
                "review it for sensitive values before sharing.\n"
            )
        else:
            sys.stderr.write(
                "Writing a privacy-minimized evidence packet. Use --include-content "
                "only when a full replay is required.\n"
            )
        sys.stdout.write(content)
        return 0

    out_path = getattr(args, "output", None) or f"session-{full_id[:12]}.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)

    size_kb = len(content.encode()) // 1024
    mode_label = "full-content" if include_content else "privacy-minimized"
    sys.stderr.write(f"Created {mode_label} evidence packet: {out_path} ({size_kb}KB)\n")
    if include_content:
        sys.stderr.write(
            "Warning: this packet may contain sensitive values. Review it before sharing.\n"
        )
    else:
        sys.stderr.write(
            "For the legacy full replay, re-run with --include-content.\n"
        )

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
