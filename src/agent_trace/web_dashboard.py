"""Web dashboard UI for ``agent-strace server --dashboard``.

Serves a browser-based session explorer on top of the existing collector
HTTP server. Zero new dependencies. All HTML/CSS/JS is embedded and served
by the stdlib HTTP handler.

Routes added when dashboard=True:
    GET /           → session list page
    GET /session/<id>  → session detail page
    GET /cost       → cost / team breakdown page
    GET /violations → policy violations page
    GET /health     → server health page
    GET /api/sessions              → JSON session list
    GET /api/sessions/<id>/events  → JSON events for a session
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING

from .compare import DECISION_DATA_EVENT_TYPES, decision_divergence
from .diff import compare_sessions
from .event_data import require_object_event_data
from .explain import PHASE_DATA_EVENT_TYPES
from .otlp import otlp_data_indexes, session_to_otlp
from .review_surface import (
    build_minimized_review_payload,
    build_session_review,
    render_minimized_review_html,
)

if TYPE_CHECKING:
    from .store import TraceStore

# The compare route builds explain phases and decision divergence for both
# sessions, so it reads the data of every event of these types in each.
COMPARE_DATA_EVENT_TYPES = PHASE_DATA_EVENT_TYPES | DECISION_DATA_EVENT_TYPES

# ---------------------------------------------------------------------------
# Shared page shell
# ---------------------------------------------------------------------------

_SHELL = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} | agent-trace</title>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f1117;color:#e2e8f0;min-height:100vh}}
a{{color:#60a5fa;text-decoration:none}}a:hover{{text-decoration:underline}}
header{{background:#1e2130;border-bottom:1px solid #2d3148;padding:12px 24px;display:flex;align-items:center;gap:16px}}
header h1{{font-size:1.1rem;font-weight:600;color:#f1f5f9}}
.subtitle{{font-size:.8rem;color:#64748b}}
nav{{margin-left:auto;display:flex;gap:4px}}
nav a{{font-size:.85rem;color:#94a3b8;padding:4px 10px;border-radius:4px}}
nav a:hover,nav a.active{{background:#2d3148;color:#e2e8f0;text-decoration:none}}
main{{padding:24px;max-width:1400px;margin:0 auto}}
.card{{background:#1e2130;border:1px solid #2d3148;border-radius:8px;padding:20px;margin-bottom:16px}}
.card h2{{font-size:.9rem;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em;margin-bottom:16px}}
table{{width:100%;border-collapse:collapse;font-size:.85rem}}
th{{text-align:left;padding:8px 12px;color:#64748b;font-weight:500;border-bottom:1px solid #2d3148}}
td{{padding:8px 12px;border-bottom:1px solid #1a1f2e;vertical-align:middle}}
tr:hover td{{background:#252a3a}}
.badge{{display:inline-block;padding:2px 8px;border-radius:12px;font-size:.75rem;font-weight:500}}
.ok{{background:#14532d;color:#86efac}}
.err{{background:#450a0a;color:#fca5a5}}
.warn{{background:#451a03;color:#fdba74}}
.info{{background:#1e3a5f;color:#93c5fd}}
.purple{{background:#3b0764;color:#d8b4fe}}
.stat-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;margin-bottom:20px}}
.stat{{background:#252a3a;border-radius:6px;padding:14px;text-align:center}}
.stat .val{{font-size:1.6rem;font-weight:700;color:#f1f5f9}}
.stat .lbl{{font-size:.75rem;color:#64748b;margin-top:4px}}
.filter-bar{{display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap;align-items:center}}
.filter-bar input,.filter-bar select{{background:#252a3a;border:1px solid #2d3148;color:#e2e8f0;padding:6px 10px;border-radius:4px;font-size:.85rem}}
.filter-bar input:focus,.filter-bar select:focus{{outline:none;border-color:#60a5fa}}
.empty{{text-align:center;padding:48px;color:#475569}}
.mono{{font-family:'SF Mono',Consolas,monospace;font-size:.8rem}}
.tl{{list-style:none}}
.tl li{{display:flex;gap:12px;padding:8px 0;border-bottom:1px solid #1a1f2e;font-size:.82rem}}
.tl .ts{{color:#475569;min-width:80px;font-family:monospace}}
.tl .et{{min-width:130px}}
.tl .ed{{color:#94a3b8;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:500px}}
.review-hero{{background:#171b28;border:1px solid #30364c;border-top:3px solid #5ea6ff;padding:22px;margin-bottom:16px}}
.review-title-row{{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;flex-wrap:wrap}}
.review-title{{font-size:1.55rem;line-height:1.2;color:#f8fafc;margin:0 0 6px}}
.review-kicker{{color:#8aa2c8;font-size:.82rem;margin-bottom:7px}}
.health-stamp{{display:inline-flex;align-items:center;gap:7px;padding:7px 11px;border:1px solid currentColor;font-size:.78rem;font-weight:650}}
.health-stamp.healthy{{color:#86efac}}.health-stamp.partial{{color:#fbbf24}}
.health-stamp.invalid{{color:#fda4af}}.health-stamp.unknown{{color:#cbd5e1}}
.review-actions{{display:flex;gap:8px;flex-wrap:wrap}}
.action{{appearance:none;background:#242a3b;border:1px solid #3a435e;color:#dbeafe;padding:7px 10px;font:inherit;font-size:.8rem;cursor:pointer}}
.action:hover{{background:#30384e;text-decoration:none}}.action.primary{{background:#155e75;border-color:#0891b2;color:#ecfeff}}
.action:focus-visible,.filter-chip:focus-visible,.relation-link:focus-visible,.event-card:focus-visible{{outline:2px solid #7dd3fc;outline-offset:2px}}
.identity-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-top:20px}}
.identity-cell{{border-top:1px solid #30364c;padding-top:9px;min-width:0}}
.identity-label{{display:block;color:#6f7f9d;font-size:.72rem;margin-bottom:3px}}
.identity-value{{display:block;color:#e5edf9;font-size:.84rem;overflow-wrap:anywhere}}
.gap-summary{{margin-top:16px;border-top:1px solid #30364c;padding-top:12px}}
.gap-summary summary{{cursor:pointer;color:#fbbf24;font-size:.82rem}}
.gap-list{{margin:10px 0 0 18px;color:#cbd5e1;font-size:.8rem}}.gap-list li{{margin:5px 0}}
.review-workbench{{display:grid;grid-template-columns:230px minmax(0,1fr);gap:16px;align-items:start}}
.review-rail{{position:sticky;top:16px;background:#171b28;border:1px solid #30364c;padding:14px}}
.rail-heading{{font-size:.82rem;color:#cbd5e1;margin:0 0 10px}}
.review-search{{width:100%;background:#0f131d;border:1px solid #39425a;color:#e5edf9;padding:8px 9px;margin-bottom:12px}}
.filter-stack{{display:grid;gap:5px}}.filter-chip{{display:flex;justify-content:space-between;gap:8px;width:100%;background:transparent;border:0;border-left:3px solid transparent;color:#94a3b8;padding:6px 8px;text-align:left;cursor:pointer}}
.filter-chip:hover{{background:#202638;color:#e2e8f0}}.filter-chip.active{{background:#202b3e;border-left-color:#5ea6ff;color:#f8fafc}}
.filter-count{{color:#64748b;font-variant-numeric:tabular-nums}}
.compare-control{{border-top:1px solid #30364c;margin-top:14px;padding-top:14px}}
.compare-control select{{width:100%;margin:7px 0;background:#0f131d;border:1px solid #39425a;color:#e5edf9;padding:7px}}
.evidence-pane{{min-width:0}}.evidence-toolbar{{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-bottom:10px;color:#94a3b8;font-size:.8rem}}
.keyboard-hint{{color:#5f708d}}.evidence-list{{position:relative;display:grid;gap:8px}}
.evidence-list::before{{content:"";position:absolute;left:25px;top:18px;bottom:18px;width:1px;background:#34405a}}
.event-card{{position:relative;display:grid;grid-template-columns:50px minmax(0,1fr);background:#171b28;border:1px solid #2d354b;min-width:0}}
.event-card.selected{{border-color:#7dd3fc;box-shadow:0 0 0 1px #7dd3fc}}.event-card.gap{{border-color:#9a531c;background:#1c1a20}}
.event-sequence{{position:relative;z-index:1;display:flex;align-items:flex-start;justify-content:center;padding-top:14px;color:#8aa2c8;font:700 .72rem 'SF Mono',Consolas,monospace}}
.event-sequence span{{display:flex;align-items:center;justify-content:center;width:25px;height:25px;border:1px solid #46516d;background:#171b28;border-radius:50%}}
.event-content{{padding:13px 14px 14px 4px;min-width:0}}
.event-heading{{display:flex;align-items:flex-start;justify-content:space-between;gap:10px}}
.event-type{{font-size:.82rem;color:#dbeafe;font-weight:650}}.event-time{{font:normal .72rem 'SF Mono',Consolas,monospace;color:#687894;white-space:nowrap}}
.event-summary{{color:#aebbd0;font-size:.82rem;line-height:1.45;margin:7px 0;white-space:pre-wrap;overflow-wrap:anywhere}}
.category-row,.relationship-row,.annotation-row{{display:flex;gap:6px;flex-wrap:wrap;align-items:center}}
.category-tag{{font-size:.68rem;color:#a8b8d2;border:1px solid #3c465f;padding:2px 6px}}.category-tag.gaps{{color:#fdba74;border-color:#9a531c}}
.relationship-row{{margin-top:8px;color:#73839f;font-size:.73rem}}.relation-link{{background:transparent;border:0;color:#7dd3fc;padding:0;cursor:pointer;font:inherit}}
.unresolved{{color:#fdba74}}.annotation{{margin-top:9px;padding:8px 10px;background:#20283a;border-left:3px solid #8b5cf6;color:#cbd5e1;font-size:.78rem}}
.annotation.correction{{border-left-color:#f59e0b}}.annotation-meta{{color:#8291aa;font-size:.69rem;margin-bottom:3px}}
.raw-details{{margin-top:10px}}.raw-details summary{{cursor:pointer;color:#7c8aa4;font-size:.72rem}}.raw-details pre{{margin-top:7px;padding:10px;background:#0d1018;border:1px solid #262d40;color:#a9b8cf;white-space:pre-wrap;overflow-wrap:anywhere;font-size:.72rem}}
.compare-panel{{display:none;background:#171b28;border:1px solid #3a435e;padding:16px;margin-bottom:12px}}.compare-panel.visible{{display:block}}
.compare-panel h3{{font-size:.95rem;margin-bottom:10px}}.compare-verdict{{color:#bae6fd;margin-bottom:12px}}.compare-grid{{display:grid;grid-template-columns:1.4fr 1fr 1fr;gap:1px;background:#30364c}}
.compare-grid span{{background:#171b28;padding:7px 9px;font-size:.75rem}}.error-message{{padding:14px;border:1px solid #7f1d1d;background:#2b151a;color:#fecaca}}
@media(max-width:780px){{.review-workbench{{grid-template-columns:1fr}}.review-rail{{position:static}}.filter-stack{{grid-template-columns:repeat(2,minmax(0,1fr))}}.keyboard-hint{{display:none}}}}
.back{{display:inline-flex;align-items:center;gap:6px;margin-bottom:16px;font-size:.85rem;color:#94a3b8}}
#loading{{text-align:center;padding:48px;color:#475569}}
</style>
</head>
<body>
<header>
  <h1>agent-trace</h1>
  <span class="subtitle">dashboard</span>
  <nav>
    <a href="/" class="{n_sessions}">Sessions</a>
    <a href="/cost" class="{n_cost}">Cost</a>
    <a href="/violations" class="{n_violations}">Violations</a>
    <a href="/health" class="{n_health}">Health</a>
  </nav>
</header>
<main>{body}</main>
<script>{script}</script>
</body>
</html>
"""


def _page(title: str, active: str, body: str, script: str) -> str:
    nav = {f"n_{p}": ("active" if p == active else "")
           for p in ("sessions", "cost", "violations", "health")}
    return _SHELL.format(title=title, body=body, script=script, **nav)


# ---------------------------------------------------------------------------
# Sessions list page
# ---------------------------------------------------------------------------

_SESSIONS_BODY = """\
<div class="stat-grid">
  <div class="stat"><div class="val" id="s-total">—</div><div class="lbl">Sessions</div></div>
  <div class="stat"><div class="val" id="s-tools">—</div><div class="lbl">Tool Calls</div></div>
  <div class="stat"><div class="val" id="s-errors">—</div><div class="lbl">Errors</div></div>
  <div class="stat"><div class="val" id="s-tokens">—</div><div class="lbl">Tokens</div></div>
</div>
<div class="card">
  <h2>Sessions</h2>
  <div class="filter-bar">
    <input id="q" placeholder="Filter by agent / session ID…" style="flex:1;min-width:180px">
    <select id="ws-f"><option value="">All workspaces</option></select>
    <select id="team-f"><option value="">All teams</option></select>
    <select id="sort">
      <option value="new">Newest first</option>
      <option value="old">Oldest first</option>
      <option value="tools">Most tool calls</option>
      <option value="err">Most errors</option>
    </select>
  </div>
  <div id="loading">Loading sessions…</div>
  <table id="tbl" style="display:none">
    <thead><tr>
      <th>Session ID</th><th>Agent</th><th>Workspace</th><th>Team</th>
      <th>Started</th><th>Duration</th><th>Tools</th><th>Errors</th><th>Status</th>
    </tr></thead>
    <tbody id="tbody"></tbody>
  </table>
  <div class="empty" id="empty" style="display:none">No sessions match the filter.</div>
</div>"""

_SESSIONS_SCRIPT = r"""
let _all = [];
async function load() {
  const r = await fetch('/api/sessions');
  _all = await r.json();
  document.getElementById('s-total').textContent = _all.length;
  document.getElementById('s-tools').textContent = _all.reduce((a,s)=>a+(s.tool_calls||0),0);
  document.getElementById('s-errors').textContent = _all.reduce((a,s)=>a+(s.errors||0),0);
  document.getElementById('s-tokens').textContent = _all.reduce((a,s)=>a+(s.total_tokens||0),0);
  const ws=[...new Set(_all.map(s=>s.workspace_id).filter(Boolean))];
  const teams=[...new Set(_all.map(s=>s.team).filter(Boolean))];
  const wsEl=document.getElementById('ws-f');
  ws.forEach(w=>{const o=document.createElement('option');o.value=w;o.textContent=w;wsEl.appendChild(o);});
  const tEl=document.getElementById('team-f');
  teams.forEach(t=>{const o=document.createElement('option');o.value=t;o.textContent=t;tEl.appendChild(o);});
  render();
  document.getElementById('loading').style.display='none';
}
function fmtDur(ms){if(!ms)return'—';if(ms<1000)return ms.toFixed(0)+'ms';if(ms<60000)return(ms/1000).toFixed(1)+'s';return(ms/60000).toFixed(1)+'m';}
function fmtTs(ts){return ts?new Date(ts*1000).toLocaleString():'—';}
function badge(s){if(s.errors>0)return'<span class="badge err">errors</span>';if(!s.ended_at)return'<span class="badge warn">running</span>';return'<span class="badge ok">done</span>';}
function render(){
  const q=document.getElementById('q').value.toLowerCase();
  const ws=document.getElementById('ws-f').value;
  const team=document.getElementById('team-f').value;
  const sort=document.getElementById('sort').value;
  let rows=_all.filter(s=>{
    if(q&&!s.session_id.includes(q)&&!(s.agent_name||'').toLowerCase().includes(q))return false;
    if(ws&&s.workspace_id!==ws)return false;
    if(team&&s.team!==team)return false;
    return true;
  });
  rows.sort((a,b)=>{
    if(sort==='old')return(a.started_at||0)-(b.started_at||0);
    if(sort==='tools')return(b.tool_calls||0)-(a.tool_calls||0);
    if(sort==='err')return(b.errors||0)-(a.errors||0);
    return(b.started_at||0)-(a.started_at||0);
  });
  const tbody=document.getElementById('tbody');
  tbody.innerHTML='';
  rows.forEach(s=>{
    const dur=s.ended_at?fmtDur((s.ended_at-s.started_at)*1000):'—';
    const tr=document.createElement('tr');
    tr.innerHTML=`<td><a href="/session/${s.session_id}" class="mono">${s.session_id.slice(0,12)}</a></td>
      <td>${s.agent_name||'—'}</td><td>${s.workspace_id||'—'}</td><td>${s.team||'—'}</td>
      <td class="mono">${fmtTs(s.started_at)}</td><td>${dur}</td>
      <td>${s.tool_calls||0}</td><td>${s.errors||0}</td><td>${badge(s)}</td>`;
    tbody.appendChild(tr);
  });
  document.getElementById('empty').style.display=rows.length?'none':'';
  document.getElementById('tbl').style.display=rows.length?'':'none';
}
['q','ws-f','team-f','sort'].forEach(id=>document.getElementById(id).addEventListener('change',render));
document.getElementById('q').addEventListener('input',render);
load();
"""


def render_sessions_page() -> str:
    return _page("Sessions", "sessions", _SESSIONS_BODY, _SESSIONS_SCRIPT)


# ---------------------------------------------------------------------------
# Session detail page
# ---------------------------------------------------------------------------

_DETAIL_BODY = """\
<a href="/" class="back">← All sessions</a>
<section class="review-hero" aria-labelledby="review-title">
  <div class="review-title-row">
    <div>
      <div class="review-kicker">Local evidence review</div>
      <h2 class="review-title" id="review-title">Loading session</h2>
      <span class="health-stamp unknown" id="health-stamp">Evidence health: loading</span>
    </div>
    <div class="review-actions">
      <a class="action" id="share-action" href="#">Save minimized share</a>
      <a class="action" id="otlp-action" href="#">Download OTLP JSON</a>
    </div>
  </div>
  <div class="identity-grid" id="identity-grid"></div>
  <details class="gap-summary" id="gap-summary">
    <summary id="gap-title">Capture gaps</summary>
    <ul class="gap-list" id="gap-list"></ul>
  </details>
</section>
<div id="loading">Loading evidence...</div>
<div class="review-workbench" id="review-workbench" style="display:none">
  <aside class="review-rail" aria-label="Evidence filters">
    <h3 class="rail-heading">Find evidence</h3>
    <input class="review-search" id="review-search" type="search" placeholder="Search recorded fields">
    <div class="filter-stack" id="filter-stack"></div>
    <div class="compare-control">
      <h3 class="rail-heading">Compare locally</h3>
      <select id="compare-session" aria-label="Session to compare"></select>
      <button class="action primary" id="compare-action" type="button">Compare sessions</button>
    </div>
  </aside>
  <section class="evidence-pane" aria-label="Ordered evidence">
    <div class="compare-panel" id="compare-panel"></div>
    <div class="evidence-toolbar">
      <span id="evidence-count"></span>
      <span class="keyboard-hint">J/K or arrows move, Enter opens raw fields, / searches</span>
    </div>
    <div class="evidence-list" id="evidence-list"></div>
  </section>
</div>"""

_DETAIL_SCRIPT = r"""
const sid=decodeURIComponent(window.location.pathname.split('/').pop());
const state={review:null,filter:'all',search:'',visible:[],selected:-1};
const labels={all:'All evidence',files:'Files',commands:'Commands',tools:'Tools',tests:'Tests',
  failures:'Failures',retry:'Retries',recovery:'Recovery',annotations:'Annotations',
  corrections:'Corrections',privacy:'Privacy changes',gaps:'Gaps'};
const listEl=document.getElementById('evidence-list');
const searchEl=document.getElementById('review-search');

function node(tag,className,text){
  const el=document.createElement(tag);
  if(className)el.className=className;
  if(text!==undefined&&text!==null)el.textContent=String(text);
  return el;
}
function fmtDate(ts){return ts?new Date(ts*1000).toLocaleString():'Not recorded';}
function fmtTime(ts){return ts?new Date(ts*1000).toISOString().slice(11,23):'';}
function fmtDur(ms){
  if(ms===null||ms===undefined)return'In progress';
  if(ms<1000)return Math.round(ms)+' ms';
  if(ms<60000)return(ms/1000).toFixed(1)+' s';
  return Math.floor(ms/60000)+' m '+Math.round((ms%60000)/1000)+' s';
}
function identity(label,value){
  const cell=node('div','identity-cell');
  cell.append(node('span','identity-label',label),node('span','identity-value',value||'Not recorded'));
  return cell;
}
function renderHeader(review){
  const session=review.session;
  const liveCapture=review.live_capture||{};
  document.getElementById('review-title').textContent=`Session ${session.session_id}`;
  const health=review.evidence_health;
  const stamp=document.getElementById('health-stamp');
  stamp.className=`health-stamp ${health.status||'unknown'}`;
  stamp.textContent=`Evidence health: ${health.status||'unknown'}`;
  const grid=document.getElementById('identity-grid');
  grid.replaceChildren(
    identity('Provider',session.provider),
    identity('Capture method',session.capture_method),
    identity('Agent',session.agent_name),
    identity('Started',fmtDate(session.started_at)),
    identity('Ended',fmtDate(session.ended_at)),
    identity('Duration',fmtDur(session.duration_ms)),
    identity('Redaction mode',review.privacy.redaction_mode.replaceAll('_',' ')),
    identity('Content mode',review.privacy.content_mode.replaceAll('_',' ')),
    identity(
      'Live capture evidence',
      liveCapture.supported
        ? String(liveCapture.status||'unknown').replaceAll('_',' ')
        : 'Not available for this provider'
    )
  );
  const gaps=review.capture_gaps||[];
  document.getElementById('gap-title').textContent=`Capture gaps and limits (${gaps.length})`;
  const gapList=document.getElementById('gap-list');
  gapList.replaceChildren();
  const unresolvedAnnotations=[...(review.unresolved_annotations||[])];
  if(!gaps.length){
    gapList.appendChild(node('li','', 'No observed structural gaps or declared provider limits.'));
  }else{
    gaps.forEach(gap=>{
      const item=node('li');
      const code=node('span','mono',gap.code+': ');
      item.append(code,document.createTextNode(gap.message));
      if(gap.code==='unresolved_annotation'&&unresolvedAnnotations.length){
        const annotation=unresolvedAnnotations.shift();
        const detail=annotation.note||annotation.label||annotation.annotation_id;
        item.appendChild(document.createTextNode(` Annotation: ${detail}`));
      }
      if(gap.event_id){
        const jump=node('button','relation-link',` ${gap.event_id}`);
        jump.type='button';
        jump.addEventListener('click',()=>selectEvent(gap.event_id,true));
        item.appendChild(jump);
      }
      gapList.appendChild(item);
    });
  }
  document.getElementById('share-action').href=`/api/sessions/${encodeURIComponent(session.session_id)}/share?mode=minimized`;
  document.getElementById('otlp-action').href=`/api/sessions/${encodeURIComponent(session.session_id)}/otlp`;
}
function renderFilters(review){
  const stack=document.getElementById('filter-stack');
  stack.replaceChildren();
  const filters=[{id:'all',count:review.items.length},...review.filters];
  filters.forEach(entry=>{
    const button=node('button','filter-chip'+(entry.id===state.filter?' active':''));
    button.type='button';
    button.dataset.filter=entry.id;
    button.append(node('span','',labels[entry.id]||entry.id),node('span','filter-count',entry.count));
    button.addEventListener('click',()=>{
      state.filter=entry.id;
      renderFilters(review);
      renderEvidence();
    });
    stack.appendChild(button);
  });
}
function relationshipButton(label,eventId){
  const button=node('button','relation-link',`${label} ${eventId}`);
  button.type='button';
  button.addEventListener('click',()=>selectEvent(eventId,true));
  return button;
}
function renderItem(item){
  const card=node('article','event-card'+(item.categories.includes('gaps')?' gap':''));
  card.tabIndex=-1;
  card.dataset.eventId=item.event_id;
  card.id=`event-${item.sequence}`;
  const seq=node('div','event-sequence');
  seq.appendChild(node('span','',item.sequence));
  const content=node('div','event-content');
  const heading=node('div','event-heading');
  heading.append(node('span','event-type',item.event_type.replaceAll('_',' ')),node('span','event-time',fmtTime(item.timestamp)));
  content.appendChild(heading);
  content.appendChild(node('div','event-summary',item.summary));
  const categories=node('div','category-row');
  item.categories.forEach(category=>categories.appendChild(node('span','category-tag '+category,labels[category]||category)));
  content.appendChild(categories);
  const relation=node('div','relationship-row');
  if(item.parent_id){
    if(item.relationship==='linked')relation.appendChild(relationshipButton('Parent',item.parent_id));
    else relation.appendChild(node('span','unresolved',`Unresolved parent ${item.parent_id} (${item.relationship_issue||'unknown'})`));
  }else if(item.relationship_issue==='missing_parent'){
    relation.appendChild(node('span','unresolved','Missing parent relationship'));
  }
  item.child_ids.forEach(child=>relation.appendChild(relationshipButton('Child',child)));
  if(item.missing_child)relation.appendChild(node('span','unresolved','Missing terminal result'));
  if(relation.childNodes.length)content.appendChild(relation);
  item.annotations.forEach(annotation=>{
    const box=node('div','annotation'+(annotation.correction?' correction':''));
    const meta=[annotation.label,annotation.author].filter(Boolean).join(' | ')||'Annotation';
    box.append(node('div','annotation-meta',meta),node('div','',annotation.note||'No note recorded'));
    content.appendChild(box);
  });
  const details=node('details','raw-details');
  details.append(node('summary','',`Source ${item.event_id} | raw provider fields`));
  const pre=node('pre');
  pre.textContent=JSON.stringify(item.data,null,2);
  details.appendChild(pre);
  content.appendChild(details);
  card.append(seq,content);
  card.addEventListener('click',event=>{
    if(event.target instanceof HTMLButtonElement||event.target instanceof HTMLElement&&event.target.closest('details'))return;
    selectEvent(item.event_id,false);
  });
  return card;
}
function renderEvidence(){
  if(!state.review)return;
  const query=state.search.trim().toLowerCase();
  state.visible=state.review.items.filter(item=>{
    if(state.filter!=='all'&&!item.categories.includes(state.filter))return false;
    if(!query)return true;
    return [item.event_id,item.event_type,item.summary,JSON.stringify(item.data)]
      .some(value=>String(value).toLowerCase().includes(query));
  });
  listEl.replaceChildren(...state.visible.map(renderItem));
  document.getElementById('evidence-count').textContent=`${state.visible.length} of ${state.review.items.length} evidence items`;
  state.selected=-1;
  const target=new URL(window.location.href).searchParams.get('event');
  if(target)selectEvent(target,true);
}
function selectEvent(eventId,scroll){
  const index=state.visible.findIndex(item=>item.event_id===eventId);
  if(index<0&&state.review){
    const item=state.review.items.find(candidate=>candidate.event_id===eventId);
    if(item){
      state.filter='all';
      state.search='';
      searchEl.value='';
      renderFilters(state.review);
      renderEvidence();
      return selectEvent(eventId,scroll);
    }
    return;
  }
  document.querySelectorAll('.event-card.selected').forEach(el=>el.classList.remove('selected'));
  state.selected=index;
  const card=listEl.querySelector(`[data-event-id="${CSS.escape(eventId)}"]`);
  if(card){
    card.classList.add('selected');
    card.focus({preventScroll:!scroll});
    if(scroll)card.scrollIntoView({block:'center',behavior:'smooth'});
  }
  const url=new URL(window.location.href);
  url.searchParams.set('event',eventId);
  history.replaceState(null,'',url);
}
function moveSelection(delta){
  if(!state.visible.length)return;
  const next=state.selected<0?(delta>0?0:state.visible.length-1):
    Math.max(0,Math.min(state.visible.length-1,state.selected+delta));
  selectEvent(state.visible[next].event_id,true);
}
function renderCompare(report){
  const panel=document.getElementById('compare-panel');
  panel.replaceChildren();
  panel.classList.add('visible');
  panel.append(node('h3','',`${report.label_a} compared with ${report.label_b}`));
  panel.append(node('div','compare-verdict',report.verdict));
  const grid=node('div','compare-grid');
  const rows=[
    ['Metric',report.label_a,report.label_b],
    ['Duration',fmtDur(report.duration_a*1000),fmtDur(report.duration_b*1000)],
    ['Cost',`$${report.cost_a.toFixed(4)}`,`$${report.cost_b.toFixed(4)}`],
    ['Tool calls',report.tool_calls_a,report.tool_calls_b],
    ['Files modified',report.files_modified_a,report.files_modified_b],
    ['Errors',report.errors_a,report.errors_b],
    ['Decision divergence',report.decision_divergence,report.decision_divergence]
  ];
  rows.flat().forEach(value=>grid.appendChild(node('span','',value)));
  panel.appendChild(grid);
}
async function loadCompareSessions(){
  const response=await fetch('/api/sessions');
  const sessions=await response.json();
  const select=document.getElementById('compare-session');
  select.replaceChildren();
  sessions.filter(session=>session.session_id!==sid).forEach(session=>{
    const option=node('option','',`${session.session_id.slice(0,12)} ${session.agent_name||''}`);
    option.value=session.session_id;
    select.appendChild(option);
  });
  document.getElementById('compare-action').disabled=!select.options.length;
}
async function compare(){
  const other=document.getElementById('compare-session').value;
  if(!other)return;
  const response=await fetch(`/api/sessions/${encodeURIComponent(sid)}/compare?other=${encodeURIComponent(other)}`);
  if(!response.ok){
    const payload=await response.json();
    throw new Error(payload.error||'Comparison failed');
  }
  renderCompare(await response.json());
}
async function load(){
  try{
    const response=await fetch(`/api/sessions/${encodeURIComponent(sid)}/review`);
    if(!response.ok)throw new Error((await response.json()).error||'Session review failed');
    state.review=await response.json();
    renderHeader(state.review);
    renderFilters(state.review);
    renderEvidence();
    await loadCompareSessions();
    document.getElementById('loading').style.display='none';
    document.getElementById('review-workbench').style.display='';
  }catch(error){
    document.getElementById('loading').replaceChildren(node('div','error-message',String(error)));
  }
}
searchEl.addEventListener('input',()=>{state.search=searchEl.value;renderEvidence();});
document.getElementById('compare-action').addEventListener('click',()=>compare().catch(error=>{
  document.getElementById('compare-panel').replaceChildren(node('div','error-message',String(error)));
  document.getElementById('compare-panel').classList.add('visible');
}));
document.addEventListener('keydown',event=>{
  const typing=event.target instanceof HTMLInputElement||event.target instanceof HTMLSelectElement;
  if(!typing&&(event.key==='j'||event.key==='ArrowDown')){event.preventDefault();moveSelection(1);}
  else if(!typing&&(event.key==='k'||event.key==='ArrowUp')){event.preventDefault();moveSelection(-1);}
  else if(!typing&&event.key==='/'){event.preventDefault();searchEl.focus();}
  else if(event.key==='Escape'){state.search='';searchEl.value='';renderEvidence();searchEl.blur();}
  else if(!typing&&event.key==='Enter'&&state.selected>=0){
    const card=listEl.children[state.selected];
    const details=card&&card.querySelector('details');
    if(details)details.open=!details.open;
  }
});
load();
"""


def render_detail_page(session_id: str) -> str:
    return _page(f"Session {session_id[:12]}", "sessions", _DETAIL_BODY, _DETAIL_SCRIPT)


# ---------------------------------------------------------------------------
# Cost page
# ---------------------------------------------------------------------------

_COST_BODY = """\
<div class="card">
  <h2>By Team</h2>
  <div id="loading">Loading…</div>
  <table id="team-tbl" style="display:none">
    <thead><tr><th>Team</th><th>Sessions</th><th>Tool Calls</th><th>LLM Requests</th><th>Tokens</th></tr></thead>
    <tbody id="team-tbody"></tbody>
  </table>
</div>
<div class="card">
  <h2>By Agent</h2>
  <table id="agent-tbl" style="display:none">
    <thead><tr><th>Agent</th><th>Sessions</th><th>Tool Calls</th><th>Errors</th></tr></thead>
    <tbody id="agent-tbody"></tbody>
  </table>
</div>"""

_COST_SCRIPT = r"""
async function load(){
  const r=await fetch('/api/sessions');
  const sessions=await r.json();
  const teams={},agents={};
  sessions.forEach(s=>{
    const t=s.team||'(none)';
    if(!teams[t])teams[t]={n:0,tools:0,llm:0,tokens:0};
    teams[t].n++;teams[t].tools+=s.tool_calls||0;teams[t].llm+=s.llm_requests||0;teams[t].tokens+=s.total_tokens||0;
    const a=s.agent_name||'(unknown)';
    if(!agents[a])agents[a]={n:0,tools:0,errors:0};
    agents[a].n++;agents[a].tools+=s.tool_calls||0;agents[a].errors+=s.errors||0;
  });
  const tb=document.getElementById('team-tbody');
  Object.entries(teams).sort((a,b)=>b[1].tools-a[1].tools).forEach(([t,d])=>{
    const tr=document.createElement('tr');
    tr.innerHTML=`<td>${t}</td><td>${d.n}</td><td>${d.tools}</td><td>${d.llm}</td><td>${d.tokens}</td>`;
    tb.appendChild(tr);
  });
  document.getElementById('team-tbl').style.display='';
  const ab=document.getElementById('agent-tbody');
  Object.entries(agents).sort((a,b)=>b[1].tools-a[1].tools).forEach(([a,d])=>{
    const tr=document.createElement('tr');
    tr.innerHTML=`<td>${a}</td><td>${d.n}</td><td>${d.tools}</td><td>${d.errors}</td>`;
    ab.appendChild(tr);
  });
  document.getElementById('agent-tbl').style.display='';
  document.getElementById('loading').style.display='none';
}
load();
"""


def render_cost_page() -> str:
    return _page("Cost", "cost", _COST_BODY, _COST_SCRIPT)


# ---------------------------------------------------------------------------
# Violations page
# ---------------------------------------------------------------------------

_VIOLATIONS_BODY = """\
<div class="card">
  <h2>Policy Violations</h2>
  <div id="loading">Loading…</div>
  <table id="tbl" style="display:none">
    <thead><tr><th>Session</th><th>Agent</th><th>Tool</th><th>Action</th><th>Time</th></tr></thead>
    <tbody id="tbody"></tbody>
  </table>
  <div class="empty" id="empty" style="display:none">No policy violations recorded.</div>
</div>"""

_VIOLATIONS_SCRIPT = r"""
async function load(){
  const r=await fetch('/api/sessions');
  const sessions=await r.json();
  const rows=[];
  await Promise.all(sessions.map(async s=>{
    const er=await fetch(`/api/sessions/${s.session_id}/events`);
    const events=await er.json();
    events.forEach(e=>{
      if(e.event_type==='policy_violation'||(e.data&&e.data.policy_action)){
        rows.push({sid:s.session_id,agent:s.agent_name||'—',
          tool:(e.data&&e.data.tool_name)||'—',
          action:(e.data&&e.data.policy_action)||e.event_type,ts:e.timestamp});
      }
    });
  }));
  document.getElementById('loading').style.display='none';
  if(!rows.length){document.getElementById('empty').style.display='';return;}
  rows.sort((a,b)=>b.ts-a.ts);
  const tbody=document.getElementById('tbody');
  rows.forEach(row=>{
    const tr=document.createElement('tr');
    tr.innerHTML=`<td><a href="/session/${row.sid}" class="mono">${row.sid.slice(0,12)}</a></td>
      <td>${row.agent}</td><td>${row.tool}</td>
      <td><span class="badge err">${row.action}</span></td>
      <td class="mono">${new Date(row.ts*1000).toLocaleString()}</td>`;
    tbody.appendChild(tr);
  });
  document.getElementById('tbl').style.display='';
}
load();
"""


def render_violations_page() -> str:
    return _page("Violations", "violations", _VIOLATIONS_BODY, _VIOLATIONS_SCRIPT)


# ---------------------------------------------------------------------------
# Health page
# ---------------------------------------------------------------------------

_HEALTH_BODY = """\
<div class="stat-grid">
  <div class="stat"><div class="val" id="h-total">—</div><div class="lbl">Sessions</div></div>
  <div class="stat"><div class="val" id="h-running">—</div><div class="lbl">Running</div></div>
  <div class="stat"><div class="val" id="h-errors">—</div><div class="lbl">With Errors</div></div>
  <div class="stat"><div class="val" id="h-ws">—</div><div class="lbl">Workspaces</div></div>
</div>
<div class="card">
  <h2>Server Metrics</h2>
  <div id="loading">Loading…</div>
  <table id="tbl" style="display:none">
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody id="tbody"></tbody>
  </table>
</div>"""

_HEALTH_SCRIPT = r"""
async function load(){
  const [hr,sr]=await Promise.all([fetch('/health'),fetch('/api/sessions')]);
  const health=await hr.json();
  const sessions=await sr.json();
  const running=sessions.filter(s=>!s.ended_at).length;
  const withErr=sessions.filter(s=>s.errors>0).length;
  const ws=new Set(sessions.map(s=>s.workspace_id).filter(Boolean)).size;
  document.getElementById('h-total').textContent=sessions.length;
  document.getElementById('h-running').textContent=running;
  document.getElementById('h-errors').textContent=withErr;
  document.getElementById('h-ws').textContent=ws||'—';
  const rows=[
    ['Status',health.status||'ok'],
    ['Total sessions',sessions.length],
    ['Running',running],
    ['With errors',withErr],
    ['Workspaces',ws||0],
  ];
  const tbody=document.getElementById('tbody');
  rows.forEach(([k,v])=>{
    const tr=document.createElement('tr');
    tr.innerHTML=`<td>${k}</td><td>${v}</td>`;
    tbody.appendChild(tr);
  });
  document.getElementById('tbl').style.display='';
  document.getElementById('loading').style.display='none';
}
load();
"""


def render_health_page() -> str:
    return _page("Health", "health", _HEALTH_BODY, _HEALTH_SCRIPT)


# ---------------------------------------------------------------------------
# API helpers (called by the server handler)
# ---------------------------------------------------------------------------

def api_sessions(store: "TraceStore") -> str:
    """Return JSON array of all session metadata."""
    sessions = store.list_sessions()
    return json.dumps([json.loads(m.to_json()) for m in sessions])


def api_session_events(store: "TraceStore", session_id: str) -> str | None:
    """Return JSON array of events for *session_id*, or None if not found."""
    session_id = _resolve_session(store, session_id)
    if session_id is None:
        return None
    events = store.load_events(session_id)
    return json.dumps([json.loads(e.to_json()) for e in events])


def _resolve_session(store: "TraceStore", session_id: str) -> str | None:
    try:
        if store.session_exists(session_id):
            return session_id
        return store.find_session(session_id)
    except ValueError:
        return None


def api_session_review(store: "TraceStore", session_id: str) -> str | None:
    """Return the stable review payload for one local session."""
    full_id = _resolve_session(store, session_id)
    if full_id is None:
        return None
    return json.dumps(build_session_review(store, full_id))


def api_session_compare(
    store: "TraceStore",
    session_id: str,
    other_session_id: str,
) -> str | None:
    """Return an existing deterministic session comparison as JSON."""
    full_id = _resolve_session(store, session_id)
    other_id = _resolve_session(store, other_session_id)
    if full_id is None or other_id is None:
        return None
    # compare_sessions loads both sessions before it reads any event data, so
    # load both before checking either to keep a load error ahead of a refusal.
    loaded = [(sid, store.load_events(sid)) for sid in (full_id, other_id)]
    for sid, events in loaded:
        require_object_event_data(events, event_types=COMPARE_DATA_EVENT_TYPES, session_id=sid)
    report = asdict(compare_sessions(store, full_id, other_id))
    report["decision_divergence"] = decision_divergence(store, full_id, other_id)
    return json.dumps(report)


def api_minimized_review_share(store: "TraceStore", session_id: str) -> str | None:
    """Return a self-contained value-free review file."""
    full_id = _resolve_session(store, session_id)
    if full_id is None:
        return None
    review = build_session_review(store, full_id)
    return render_minimized_review_html(build_minimized_review_payload(review))


def api_session_otlp(store: "TraceStore", session_id: str) -> str | None:
    """Return an OTLP/HTTP JSON payload for local download without uploading."""
    full_id = _resolve_session(store, session_id)
    if full_id is None:
        return None
    meta = store.load_meta(full_id)
    events = store.load_events(full_id)
    require_object_event_data(events, indexes=otlp_data_indexes(events), session_id=full_id)
    return json.dumps(session_to_otlp(meta, events), indent=2) + "\n"
