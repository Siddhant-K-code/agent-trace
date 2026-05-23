/**
 * Live streaming panel (#120).
 *
 * Opens a webview that connects to an agent-strace server via SSE
 * (Server-Sent Events) and displays events in real time.
 *
 * The SSE connection is made inside the webview using the browser-native
 * EventSource API — not Node.js http — so it works within VS Code's
 * webview sandbox. Reconnection uses exponential backoff (1s → 2s → 4s …
 * capped at 30s).
 *
 * The endpoint is read from agentTrace.collectorEndpoint setting.
 * If empty, the panel shows a configuration prompt.
 */

import * as vscode from "vscode";

function _nonce(): string {
  let text = "";
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  for (let i = 0; i < 32; i++) {
    text += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return text;
}

function _renderHtml(endpoint: string, nonce: string): string {
  const sseUrl = endpoint ? `${endpoint.replace(/\/$/, "")}/events/stream` : "";

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="Content-Security-Policy"
    content="default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-${nonce}'; connect-src ${endpoint || "'none'"} http: https:;">
  <style>
    * { box-sizing: border-box; }
    body {
      font-family: var(--vscode-editor-font-family);
      font-size: var(--vscode-editor-font-size);
      color: var(--vscode-foreground);
      background: var(--vscode-editor-background);
      margin: 0; padding: 0;
      display: flex; flex-direction: column; height: 100vh;
    }
    #toolbar {
      display: flex; align-items: center; gap: 8px;
      padding: 6px 12px;
      background: var(--vscode-sideBar-background);
      border-bottom: 1px solid var(--vscode-panel-border);
      flex-shrink: 0;
    }
    #status {
      font-size: 0.85em;
      color: var(--vscode-descriptionForeground);
      flex: 1;
    }
    #status.connected { color: #4caf50; }
    #status.error     { color: #f44336; }
    button {
      padding: 3px 10px; cursor: pointer;
      background: var(--vscode-button-background);
      color: var(--vscode-button-foreground);
      border: none; border-radius: 3px; font-size: 0.85em;
    }
    button:hover { background: var(--vscode-button-hoverBackground); }
    #filter {
      padding: 3px 8px; font-size: 0.85em;
      background: var(--vscode-input-background);
      color: var(--vscode-input-foreground);
      border: 1px solid var(--vscode-input-border);
      border-radius: 3px;
    }
    #events {
      flex: 1; overflow-y: auto; padding: 8px 12px;
    }
    .event {
      display: flex; gap: 10px; padding: 3px 0;
      border-bottom: 1px solid var(--vscode-panel-border);
      font-size: 0.9em;
    }
    .event:last-child { border-bottom: none; }
    .ts   { color: var(--vscode-descriptionForeground); flex-shrink: 0; width: 80px; }
    .type { flex-shrink: 0; width: 140px; font-weight: bold; }
    .type.tool_call         { color: #4fc3f7; }
    .type.llm_response      { color: #a5d6a7; }
    .type.llm_request       { color: #fff176; }
    .type.error             { color: #ef9a9a; }
    .type.session_start     { color: #ce93d8; }
    .type.session_end       { color: #ce93d8; }
    .type.user_prompt       { color: #ffcc80; }
    .data { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
            color: var(--vscode-foreground); }
    .session-link { color: var(--vscode-textLink-foreground); cursor: pointer; text-decoration: underline; }
    #no-endpoint {
      display: flex; flex-direction: column; align-items: center; justify-content: center;
      height: 100%; gap: 12px; color: var(--vscode-descriptionForeground);
    }
    #no-endpoint button { padding: 6px 16px; font-size: 1em; }
  </style>
</head>
<body>
${sseUrl ? `
  <div id="toolbar">
    <span id="status">Connecting…</span>
    <input id="filter" type="text" placeholder="Filter by type…" oninput="applyFilter()">
    <button onclick="clearEvents()">Clear</button>
    <button onclick="togglePause()" id="pauseBtn">Pause</button>
  </div>
  <div id="events"></div>
` : `
  <div id="no-endpoint">
    <p>No collector endpoint configured.</p>
    <p>Set <code>agentTrace.collectorEndpoint</code> in VS Code settings to connect to an agent-strace server.</p>
    <button onclick="openSettings()">Open Settings</button>
  </div>
`}

  <script nonce="${nonce}">
    const vscode = acquireVsCodeApi();
    const SSE_URL = ${JSON.stringify(sseUrl)};

    let paused = false;
    let filterText = "";
    let eventSource = null;
    let reconnectDelay = 1000;
    let reconnectTimer = null;
    let eventCount = 0;
    const MAX_EVENTS = 500;

    function openSettings() {
      vscode.postMessage({ command: 'openSettings' });
    }

    function clearEvents() {
      document.getElementById('events').innerHTML = '';
      eventCount = 0;
    }

    function togglePause() {
      paused = !paused;
      document.getElementById('pauseBtn').textContent = paused ? 'Resume' : 'Pause';
    }

    function applyFilter() {
      filterText = document.getElementById('filter').value.toLowerCase();
      const rows = document.querySelectorAll('.event');
      rows.forEach(row => {
        const type = row.dataset.type || '';
        row.style.display = (!filterText || type.includes(filterText)) ? '' : 'none';
      });
    }

    function setStatus(text, cls) {
      const el = document.getElementById('status');
      if (!el) return;
      el.textContent = text;
      el.className = cls || '';
    }

    function appendEvent(ev) {
      if (paused) return;
      const container = document.getElementById('events');
      if (!container) return;

      // Trim old events
      while (container.children.length >= MAX_EVENTS) {
        container.removeChild(container.firstChild);
      }

      const ts = new Date(ev.timestamp * 1000).toLocaleTimeString();
      const type = ev.event_type || 'unknown';
      const data = JSON.stringify(ev.data || {});
      const summary = data.length > 120 ? data.slice(0, 120) + '…' : data;

      const row = document.createElement('div');
      row.className = 'event';
      row.dataset.type = type;
      if (filterText && !type.includes(filterText)) {
        row.style.display = 'none';
      }

      const sessionId = ev.session_id || '';
      const sessionLink = sessionId
        ? '<span class="session-link" onclick="jumpToSession(' + JSON.stringify(sessionId) + ')">' + sessionId.slice(0, 8) + '</span>'
        : '';

      row.innerHTML =
        '<span class="ts">' + ts + '</span>' +
        '<span class="type ' + type + '">' + type + '</span>' +
        '<span class="data">' + sessionLink + (sessionLink ? '  ' : '') + escHtml(summary) + '</span>';

      container.appendChild(row);
      container.scrollTop = container.scrollHeight;
    }

    function escHtml(s) {
      return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }

    function jumpToSession(sessionId) {
      vscode.postMessage({ command: 'jumpToSession', sessionId });
    }

    function connect() {
      if (!SSE_URL) return;
      if (eventSource) { eventSource.close(); }

      setStatus('Connecting…');
      eventSource = new EventSource(SSE_URL);

      eventSource.onopen = () => {
        reconnectDelay = 1000;
        setStatus('Connected to ' + SSE_URL, 'connected');
      };

      eventSource.onmessage = (e) => {
        try {
          const ev = JSON.parse(e.data);
          appendEvent(ev);
        } catch { /* skip malformed */ }
      };

      eventSource.onerror = () => {
        eventSource.close();
        eventSource = null;
        const delay = Math.min(reconnectDelay, 30000);
        setStatus('Disconnected — reconnecting in ' + (delay / 1000) + 's…', 'error');
        reconnectTimer = setTimeout(() => {
          reconnectDelay = Math.min(reconnectDelay * 2, 30000);
          connect();
        }, delay);
      };
    }

    if (SSE_URL) { connect(); }
  </script>
</body>
</html>`;
}

export class LiveStreamPanel extends vscode.Disposable {
  private panel: vscode.WebviewPanel | null = null;

  constructor(private readonly extensionUri: vscode.Uri) {
    super(() => this._cleanup());
  }

  open(): void {
    if (this.panel) {
      this.panel.reveal();
      return;
    }

    const endpoint = vscode.workspace
      .getConfiguration("agentTrace")
      .get<string>("collectorEndpoint", "");

    const panel = vscode.window.createWebviewPanel(
      "agentTrace.liveStream",
      "agent-trace: Live Stream",
      vscode.ViewColumn.Beside,
      {
        enableScripts: true,
        retainContextWhenHidden: true,
        // Allow connections to the configured endpoint
        localResourceRoots: [],
      }
    );

    const nonce = _nonce();
    panel.webview.html = _renderHtml(endpoint, nonce);

    panel.webview.onDidReceiveMessage((msg) => {
      if (msg.command === "openSettings") {
        vscode.commands.executeCommand(
          "workbench.action.openSettings",
          "agentTrace.collectorEndpoint"
        );
      } else if (msg.command === "jumpToSession") {
        // Reveal in session browser if available
        vscode.commands.executeCommand(
          "agentTrace.revealSession",
          msg.sessionId
        );
      }
    });

    panel.onDidDispose(() => {
      this.panel = null;
    });

    this.panel = panel;
  }

  private _cleanup(): void {
    this.panel?.dispose();
    this.panel = null;
  }

  override dispose(): void {
    this._cleanup();
  }
}
