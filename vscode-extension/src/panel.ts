/**
 * Side panel webview showing the live event stream.
 *
 * Renders as a VS Code WebviewView in the Explorer sidebar.
 * Receives events via postMessage from the extension host.
 * Includes a Pause / Resume button that posts a command back.
 */

import * as vscode from "vscode";
import { SessionState, TraceEvent } from "./traceStore";

// ---------------------------------------------------------------------------
// Icons / labels per event type
// ---------------------------------------------------------------------------

const EVENT_ICON: Record<string, string> = {
  session_start: "▶",
  session_end: "■",
  user_prompt: "👤",
  assistant_response: "🤖",
  tool_call: "→",
  tool_result: "←",
  llm_request: "⇢",
  llm_response: "⇠",
  file_read: "📖",
  file_write: "✏️",
  decision: "🔀",
  error: "✗",
};

function eventSummary(event: TraceEvent): string {
  const d = event.data;
  switch (event.event_type) {
    case "tool_call":
      return `${d.tool_name ?? ""}  ${_truncate(JSON.stringify(d.arguments ?? ""), 60)}`;
    case "tool_result":
      return _truncate(String(d.output ?? d.result ?? ""), 80);
    case "user_prompt":
      return _truncate(String(d.prompt ?? d.text ?? ""), 80);
    case "assistant_response":
      return _truncate(String(d.text ?? d.response ?? ""), 80);
    case "file_read":
    case "file_write":
      return String(d.path ?? "");
    case "error":
      return _truncate(String(d.message ?? d.error ?? ""), 80);
    default:
      return "";
  }
}

function _truncate(s: string, n: number): string {
  return s.length > n ? s.slice(0, n) + "…" : s;
}

function _relTime(ts: number): string {
  const s = Math.floor(Date.now() / 1000 - ts);
  if (s < 60) { return `${s}s ago`; }
  const m = Math.floor(s / 60);
  return `${m}m ago`;
}

// ---------------------------------------------------------------------------
// EventStreamPanel — WebviewViewProvider
// ---------------------------------------------------------------------------

export class EventStreamPanel implements vscode.WebviewViewProvider {
  public static readonly viewId = "agentTrace.eventStream";

  private view?: vscode.WebviewView;
  private pendingEvents: TraceEvent[] = [];
  private currentState: SessionState | null = null;

  constructor(private readonly extensionUri: vscode.Uri) {}

  resolveWebviewView(
    webviewView: vscode.WebviewView,
    _context: vscode.WebviewViewResolveContext,
    _token: vscode.CancellationToken
  ): void {
    this.view = webviewView;

    webviewView.webview.options = {
      enableScripts: true,
    };

    webviewView.webview.html = this._buildHtml(webviewView.webview);

    // Handle messages from the webview (pause/resume button)
    webviewView.webview.onDidReceiveMessage((msg) => {
      if (msg.command === "pause") {
        vscode.commands.executeCommand("agentTrace.pauseAgent");
      } else if (msg.command === "resume") {
        vscode.commands.executeCommand("agentTrace.resumeAgent");
      }
    });

    // Flush any events that arrived before the view was ready
    if (this.currentState) {
      this._postState(this.currentState);
    }
    for (const e of this.pendingEvents) {
      this._postEvent(e);
    }
    this.pendingEvents = [];
  }

  /** Called when a new session starts — resets the panel. */
  onSessionStart(state: SessionState): void {
    this.currentState = state;
    this.pendingEvents = [];
    this.view?.webview.postMessage({ type: "reset" });
    this._postState(state);
  }

  /** Called when the session ends. */
  onSessionEnd(state: SessionState): void {
    this.currentState = null;
    this._postState({ ...state, activeTool: null });
    this.view?.webview.postMessage({ type: "sessionEnd" });
  }

  /** Called on every new event. */
  pushEvent(state: SessionState, event: TraceEvent): void {
    this.currentState = state;
    if (!this.view) {
      this.pendingEvents.push(event);
      return;
    }
    this._postState(state);
    this._postEvent(event);
  }

  private _postState(state: SessionState): void {
    this.view?.webview.postMessage({
      type: "state",
      cost: state.estimatedCostUsd.toFixed(4),
      toolCalls: state.toolCallCount,
      errors: state.errorCount,
      activeTool: state.activeTool,
      paused: state.paused,
      sessionId: state.sessionId.slice(0, 8),
    });
  }

  private _postEvent(event: TraceEvent): void {
    this.view?.webview.postMessage({
      type: "event",
      icon: EVENT_ICON[event.event_type] ?? "·",
      eventType: event.event_type,
      summary: eventSummary(event),
      time: _relTime(event.timestamp),
      isError: event.event_type === "error",
    });
  }

  // -------------------------------------------------------------------------
  // HTML
  // -------------------------------------------------------------------------

  private _buildHtml(_webview: vscode.Webview): string {
    return /* html */ `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: var(--vscode-font-family);
    font-size: var(--vscode-font-size);
    color: var(--vscode-foreground);
    background: var(--vscode-sideBar-background);
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
  }

  /* Header */
  #header {
    padding: 8px 10px 6px;
    border-bottom: 1px solid var(--vscode-sideBarSectionHeader-border, #333);
    flex-shrink: 0;
  }
  #stats {
    display: flex;
    gap: 12px;
    font-size: 11px;
    color: var(--vscode-descriptionForeground);
    margin-bottom: 6px;
  }
  #stats span { white-space: nowrap; }
  #stats .cost { color: var(--vscode-charts-green, #4ec9b0); font-weight: 600; }
  #stats .errors { color: var(--vscode-errorForeground, #f48771); }
  #active-tool {
    font-size: 11px;
    color: var(--vscode-descriptionForeground);
    min-height: 14px;
    font-style: italic;
  }

  /* Controls */
  #controls {
    display: flex;
    gap: 6px;
    margin-top: 6px;
  }
  button {
    font-size: 11px;
    padding: 2px 8px;
    border: 1px solid var(--vscode-button-border, transparent);
    border-radius: 2px;
    cursor: pointer;
    background: var(--vscode-button-secondaryBackground);
    color: var(--vscode-button-secondaryForeground);
  }
  button:hover { background: var(--vscode-button-secondaryHoverBackground); }
  button.primary {
    background: var(--vscode-button-background);
    color: var(--vscode-button-foreground);
  }
  button.primary:hover { background: var(--vscode-button-hoverBackground); }
  button:disabled { opacity: 0.4; cursor: default; }

  /* Event list */
  #events {
    flex: 1;
    overflow-y: auto;
    padding: 4px 0;
  }
  .event {
    display: grid;
    grid-template-columns: 18px 1fr auto;
    gap: 4px;
    padding: 3px 10px;
    font-size: 11px;
    line-height: 1.4;
    border-bottom: 1px solid var(--vscode-sideBar-background);
  }
  .event:hover { background: var(--vscode-list-hoverBackground); }
  .event.error { color: var(--vscode-errorForeground, #f48771); }
  .event .icon { text-align: center; flex-shrink: 0; }
  .event .summary {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: var(--vscode-foreground);
  }
  .event.error .summary { color: var(--vscode-errorForeground, #f48771); }
  .event .time {
    color: var(--vscode-descriptionForeground);
    white-space: nowrap;
    font-size: 10px;
    align-self: center;
  }

  /* Empty state */
  #empty {
    padding: 20px 10px;
    text-align: center;
    color: var(--vscode-descriptionForeground);
    font-size: 11px;
    line-height: 1.6;
  }

  /* Session ended banner */
  #ended-banner {
    display: none;
    padding: 4px 10px;
    font-size: 11px;
    background: var(--vscode-diffEditor-removedLineBackground, #3a1a1a);
    color: var(--vscode-descriptionForeground);
    text-align: center;
    flex-shrink: 0;
  }
</style>
</head>
<body>

<div id="header">
  <div id="stats">
    <span class="cost" id="stat-cost">$0.0000</span>
    <span id="stat-calls">0 calls</span>
    <span class="errors" id="stat-errors" style="display:none">0 errors</span>
    <span id="stat-session" style="color:var(--vscode-descriptionForeground)"></span>
  </div>
  <div id="active-tool"></div>
  <div id="controls">
    <button class="primary" id="btn-pause" onclick="sendPause()" disabled>Pause</button>
    <button id="btn-resume" onclick="sendResume()" style="display:none">Resume</button>
    <button id="btn-clear" onclick="clearEvents()">Clear</button>
  </div>
</div>

<div id="ended-banner">Session ended</div>

<div id="events">
  <div id="empty">No active session.<br>Start an agent with <code>agent-strace setup</code>.</div>
</div>

<script>
  const vscode = acquireVsCodeApi();
  const eventsEl = document.getElementById('events');
  const emptyEl = document.getElementById('empty');
  const endedBanner = document.getElementById('ended-banner');
  let eventCount = 0;
  let paused = false;

  function sendPause() { vscode.postMessage({ command: 'pause' }); }
  function sendResume() { vscode.postMessage({ command: 'resume' }); }
  function clearEvents() {
    eventsEl.innerHTML = '';
    eventCount = 0;
    emptyEl.style.display = 'block';
    eventsEl.appendChild(emptyEl);
  }

  window.addEventListener('message', (e) => {
    const msg = e.data;

    if (msg.type === 'reset') {
      clearEvents();
      endedBanner.style.display = 'none';
      document.getElementById('btn-pause').disabled = false;
      return;
    }

    if (msg.type === 'sessionEnd') {
      endedBanner.style.display = 'block';
      document.getElementById('btn-pause').disabled = true;
      document.getElementById('btn-resume').style.display = 'none';
      document.getElementById('btn-pause').style.display = 'inline-block';
      return;
    }

    if (msg.type === 'state') {
      document.getElementById('stat-cost').textContent = '$' + msg.cost;
      document.getElementById('stat-calls').textContent = msg.toolCalls + ' calls';
      const errEl = document.getElementById('stat-errors');
      if (msg.errors > 0) {
        errEl.textContent = msg.errors + ' error' + (msg.errors > 1 ? 's' : '');
        errEl.style.display = 'inline';
      } else {
        errEl.style.display = 'none';
      }
      if (msg.sessionId) {
        document.getElementById('stat-session').textContent = msg.sessionId + '…';
      }
      const toolEl = document.getElementById('active-tool');
      toolEl.textContent = msg.activeTool ? '⟳ ' + msg.activeTool : '';

      paused = msg.paused;
      const pauseBtn = document.getElementById('btn-pause');
      const resumeBtn = document.getElementById('btn-resume');
      if (paused) {
        pauseBtn.style.display = 'none';
        resumeBtn.style.display = 'inline-block';
      } else {
        pauseBtn.style.display = 'inline-block';
        resumeBtn.style.display = 'none';
      }
      return;
    }

    if (msg.type === 'event') {
      emptyEl.style.display = 'none';
      eventCount++;

      const row = document.createElement('div');
      row.className = 'event' + (msg.isError ? ' error' : '');

      const icon = document.createElement('span');
      icon.className = 'icon';
      icon.textContent = msg.icon;

      const summary = document.createElement('span');
      summary.className = 'summary';
      summary.textContent = msg.summary || msg.eventType;
      summary.title = msg.summary || msg.eventType;

      const time = document.createElement('span');
      time.className = 'time';
      time.textContent = msg.time;

      row.appendChild(icon);
      row.appendChild(summary);
      row.appendChild(time);
      eventsEl.appendChild(row);

      // Auto-scroll to bottom
      eventsEl.scrollTop = eventsEl.scrollHeight;
    }
  });
</script>
</body>
</html>`;
  }
}
