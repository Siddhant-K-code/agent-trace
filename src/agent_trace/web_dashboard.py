"""Web dashboard UI for ``agent-strace server --dashboard``.

Serves a browser-based session explorer on top of the existing collector
HTTP server. Zero new dependencies — all HTML/CSS/JS is embedded and served
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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .store import TraceStore

# ---------------------------------------------------------------------------
# Shared page shell
# ---------------------------------------------------------------------------

_SHELL = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} — agent-trace</title>
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
<div class="stat-grid">
  <div class="stat"><div class="val" id="d-tools">—</div><div class="lbl">Tool Calls</div></div>
  <div class="stat"><div class="val" id="d-llm">—</div><div class="lbl">LLM Requests</div></div>
  <div class="stat"><div class="val" id="d-errors">—</div><div class="lbl">Errors</div></div>
  <div class="stat"><div class="val" id="d-dur">—</div><div class="lbl">Duration</div></div>
  <div class="stat"><div class="val" id="d-tokens">—</div><div class="lbl">Tokens</div></div>
</div>
<div class="card">
  <h2>Session <span id="sid" class="mono" style="font-size:.85rem;color:#60a5fa"></span></h2>
  <div id="loading">Loading events…</div>
  <ul class="tl" id="tl" style="display:none"></ul>
</div>"""

_DETAIL_SCRIPT = r"""
const sid=window.location.pathname.split('/').pop();
document.getElementById('sid').textContent=sid;
const COLORS={tool_call:'info',tool_result:'ok',llm_request:'purple',llm_response:'purple',
  error:'err',session_start:'ok',session_end:'ok',file_read:'warn',file_write:'warn'};
function fmtTs(ts){return new Date(ts*1000).toISOString().slice(11,23);}
function fmtDur(ms){if(!ms)return'—';if(ms<1000)return ms.toFixed(0)+'ms';if(ms<60000)return(ms/1000).toFixed(1)+'s';return(ms/60000).toFixed(1)+'m';}
function summarise(e){
  const d=e.data||{};
  if(e.event_type==='tool_call')return`${d.tool_name||''} ${JSON.stringify(d.input||{}).slice(0,80)}`;
  if(e.event_type==='tool_result')return(d.output||'').toString().slice(0,100);
  if(e.event_type==='llm_request')return`${d.model||''} — ${(d.prompt||'').slice(0,80)}`;
  if(e.event_type==='llm_response')return(d.text||d.content||'').slice(0,100);
  if(e.event_type==='error')return d.message||'';
  return JSON.stringify(d).slice(0,100);
}
async function load(){
  const r=await fetch(`/api/sessions/${sid}/events`);
  const events=await r.json();
  let tools=0,llm=0,errors=0,tokens=0,t0=null,t1=null;
  events.forEach(e=>{
    if(e.event_type==='tool_call')tools++;
    if(e.event_type==='llm_request')llm++;
    if(e.event_type==='error')errors++;
    tokens+=(e.data&&e.data.total_tokens)||0;
    if(!t0||e.timestamp<t0)t0=e.timestamp;
    if(!t1||e.timestamp>t1)t1=e.timestamp;
  });
  document.getElementById('d-tools').textContent=tools;
  document.getElementById('d-llm').textContent=llm;
  document.getElementById('d-errors').textContent=errors;
  document.getElementById('d-tokens').textContent=tokens||'—';
  document.getElementById('d-dur').textContent=fmtDur(t0&&t1?(t1-t0)*1000:0);
  const ul=document.getElementById('tl');
  events.forEach(e=>{
    const cls=COLORS[e.event_type]||'info';
    const li=document.createElement('li');
    li.innerHTML=`<span class="ts">${fmtTs(e.timestamp)}</span>
      <span class="et"><span class="badge ${cls}">${e.event_type}</span></span>
      <span class="ed">${summarise(e).replace(/</g,'&lt;')}</span>`;
    ul.appendChild(li);
  });
  document.getElementById('loading').style.display='none';
  ul.style.display='';
}
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
    if not store.session_exists(session_id):
        found = store.find_session(session_id)
        if found:
            session_id = found
        else:
            return None
    events = store.load_events(session_id)
    return json.dumps([json.loads(e.to_json()) for e in events])
