/**
 * Side panel webview showing the live event stream.
 *
 * Renders as a VS Code WebviewView in the Explorer sidebar.
 * Receives events via postMessage from the extension host.
 * Includes a Pause / Resume button that posts a command back.
 */

import * as fs from "fs";
import * as path from "path";
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

  private _getNonce(): string {
    let text = "";
    const possible = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    for (let i = 0; i < 32; i++) {
      text += possible.charAt(Math.floor(Math.random() * possible.length));
    }
    return text;
  }

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
  // HTML — loaded from media/panel.html with a per-render CSP nonce
  // -------------------------------------------------------------------------

  private _buildHtml(webview: vscode.Webview): string {
    const htmlPath = path.join(
      this.extensionUri.fsPath,
      "media",
      "panel.html"
    );
    const nonce = this._getNonce();
    let html = fs.readFileSync(htmlPath, "utf8");
    // Inject nonce into CSP meta tag and script tag
    html = html.replace(/NONCE_PLACEHOLDER/g, nonce);
    return html;
  }
}
