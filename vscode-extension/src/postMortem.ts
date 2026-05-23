/**
 * Post-mortem viewer panel (#118).
 *
 * Watches .agent-traces/*\/watchdog-postmortem.json for new files and
 * automatically opens a formatted webview panel. Can also be triggered
 * manually via command palette for any past session.
 *
 * The "Copy recovery context" button copies the recovery_context field
 * to clipboard — ready to paste as a system prompt for a follow-up session.
 */

import * as fs from "fs";
import * as path from "path";
import * as vscode from "vscode";

interface PostMortemData {
  session_id: string;
  reason?: string;
  elapsed_seconds?: number;
  cost_at_death?: number;
  last_tool_call?: { tool_name?: string; arguments?: Record<string, unknown> };
  last_llm_response?: { model?: string; content?: string };
  recovery_context?: string;
}

function _fmtDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return m > 0 ? `${m}m ${s.toString().padStart(2, "0")}s` : `${s}s`;
}

function _renderHtml(data: PostMortemData, nonce: string): string {
  const reason = data.reason?.replace(/_/g, " ") ?? "unknown";
  const elapsed = data.elapsed_seconds != null ? _fmtDuration(data.elapsed_seconds) : "—";
  const cost = data.cost_at_death != null ? `$${data.cost_at_death.toFixed(4)}` : "—";
  const sessionShort = (data.session_id ?? "").slice(0, 12);

  const lastTool = data.last_tool_call
    ? `${data.last_tool_call.tool_name ?? "unknown"}: ${JSON.stringify(data.last_tool_call.arguments ?? {}).slice(0, 120)}`
    : "—";

  const lastLlm = data.last_llm_response?.content
    ? `"${data.last_llm_response.content.slice(0, 200)}${data.last_llm_response.content.length > 200 ? "…" : ""}"`
    : "—";

  const recovery = data.recovery_context
    ? `<pre class="recovery">${_escHtml(data.recovery_context)}</pre>`
    : "<p class='muted'>No recovery context available.</p>";

  const copyBtn = data.recovery_context
    ? `<button id="copyBtn" onclick="copyRecovery()">Copy recovery context</button>`
    : "";

  const recoveryJson = data.recovery_context
    ? JSON.stringify(data.recovery_context)
    : "null";

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="Content-Security-Policy"
    content="default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-${nonce}';">
  <style>
    body { font-family: var(--vscode-font-family); font-size: var(--vscode-font-size);
           color: var(--vscode-foreground); padding: 16px; max-width: 700px; }
    h1 { font-size: 1.1em; margin-bottom: 4px; }
    .subtitle { color: var(--vscode-descriptionForeground); margin-bottom: 16px; }
    .section { margin-bottom: 16px; }
    .label { font-weight: bold; margin-bottom: 4px; }
    .value { font-family: var(--vscode-editor-font-family); background: var(--vscode-textBlockQuote-background);
             padding: 6px 10px; border-radius: 4px; word-break: break-all; }
    pre.recovery { white-space: pre-wrap; word-break: break-word;
                   background: var(--vscode-textBlockQuote-background);
                   padding: 10px; border-radius: 4px; max-height: 200px; overflow-y: auto; }
    .muted { color: var(--vscode-descriptionForeground); }
    button { margin-top: 8px; padding: 6px 14px; cursor: pointer;
             background: var(--vscode-button-background);
             color: var(--vscode-button-foreground);
             border: none; border-radius: 3px; }
    button:hover { background: var(--vscode-button-hoverBackground); }
    .row { display: flex; gap: 24px; margin-bottom: 12px; }
    .col { flex: 1; }
  </style>
</head>
<body>
  <h1>Session ${_escHtml(sessionShort)} — TERMINATED</h1>
  <div class="subtitle">Reason: ${_escHtml(reason)}</div>

  <div class="row">
    <div class="col">
      <div class="label">Cost at death</div>
      <div class="value">${_escHtml(cost)}</div>
    </div>
    <div class="col">
      <div class="label">Wall time</div>
      <div class="value">${_escHtml(elapsed)}</div>
    </div>
  </div>

  <div class="section">
    <div class="label">Last tool call</div>
    <div class="value">${_escHtml(lastTool)}</div>
  </div>

  <div class="section">
    <div class="label">Last LLM response</div>
    <div class="value">${_escHtml(lastLlm)}</div>
  </div>

  <div class="section">
    <div class="label">Recovery context</div>
    ${recovery}
    ${copyBtn}
  </div>

  <script nonce="${nonce}">
    const vscode = acquireVsCodeApi();
    const recoveryContext = ${recoveryJson};

    function copyRecovery() {
      vscode.postMessage({ command: 'copyRecovery', text: recoveryContext });
    }
  </script>
</body>
</html>`;
}

function _escHtml(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
          .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

function _nonce(): string {
  let text = "";
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  for (let i = 0; i < 32; i++) {
    text += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return text;
}

export class PostMortemManager extends vscode.Disposable {
  private readonly traceDir: string;
  private dirWatcher: fs.FSWatcher | null = null;
  private sessionWatchers: Map<string, fs.FSWatcher> = new Map();
  private openPanels: Map<string, vscode.WebviewPanel> = new Map();

  constructor(traceDir: string) {
    super(() => this._cleanup());
    this.traceDir = traceDir;
  }

  /** Start watching for new post-mortem files. */
  start(): void {
    if (!fs.existsSync(this.traceDir)) { return; }

    // Watch the trace dir for new session directories
    try {
      this.dirWatcher = fs.watch(this.traceDir, (_evt, filename) => {
        if (filename) {
          this._watchSession(filename);
        }
      });
    } catch { /* ignore */ }

    // Watch existing session directories
    try {
      for (const entry of fs.readdirSync(this.traceDir)) {
        this._watchSession(entry);
      }
    } catch { /* ignore */ }
  }

  private _watchSession(sessionId: string): void {
    if (this.sessionWatchers.has(sessionId)) { return; }
    const sessionDir = path.join(this.traceDir, sessionId);
    try {
      if (!fs.statSync(sessionDir).isDirectory()) { return; }
    } catch { return; }

    const pmPath = path.join(sessionDir, "watchdog-postmortem.json");

    // Check if post-mortem already exists
    if (fs.existsSync(pmPath)) {
      // Don't auto-open for existing files on startup — only new ones
    }

    try {
      const watcher = fs.watch(sessionDir, (_evt, filename) => {
        if (filename === "watchdog-postmortem.json" && fs.existsSync(pmPath)) {
          this._openPanel(sessionId, pmPath);
        }
      });
      this.sessionWatchers.set(sessionId, watcher);
    } catch { /* ignore */ }
  }

  /** Open post-mortem panel for a specific session (command palette trigger). */
  openForSession(sessionId: string): void {
    const pmPath = path.join(this.traceDir, sessionId, "watchdog-postmortem.json");
    if (!fs.existsSync(pmPath)) {
      vscode.window.showWarningMessage(
        `No post-mortem found for session ${sessionId.slice(0, 12)}.`
      );
      return;
    }
    this._openPanel(sessionId, pmPath);
  }

  private _openPanel(sessionId: string, pmPath: string): void {
    // Reuse existing panel if open
    const existing = this.openPanels.get(sessionId);
    if (existing) {
      existing.reveal();
      return;
    }

    let data: PostMortemData;
    try {
      data = JSON.parse(fs.readFileSync(pmPath, "utf8")) as PostMortemData;
    } catch {
      vscode.window.showErrorMessage("agent-trace: failed to read post-mortem file.");
      return;
    }

    const panel = vscode.window.createWebviewPanel(
      "agentTrace.postMortem",
      `Post-mortem: ${sessionId.slice(0, 12)}`,
      vscode.ViewColumn.Beside,
      { enableScripts: true, retainContextWhenHidden: false }
    );

    const nonce = _nonce();
    panel.webview.html = _renderHtml(data, nonce);

    panel.webview.onDidReceiveMessage((msg) => {
      if (msg.command === "copyRecovery" && msg.text) {
        vscode.env.clipboard.writeText(msg.text).then(() => {
          vscode.window.showInformationMessage("Recovery context copied to clipboard.");
        });
      }
    });

    panel.onDidDispose(() => {
      this.openPanels.delete(sessionId);
    });

    this.openPanels.set(sessionId, panel);
  }

  private _cleanup(): void {
    this.dirWatcher?.close();
    this.dirWatcher = null;
    for (const w of this.sessionWatchers.values()) { w.close(); }
    this.sessionWatchers.clear();
    for (const p of this.openPanels.values()) { p.dispose(); }
    this.openPanels.clear();
  }

  override dispose(): void {
    this._cleanup();
  }
}
