/**
 * Review-oriented session panel.
 *
 * The webview renders ordered evidence for active and historical sessions.
 * All actions stay local: comparison runs the CLI, minimized share writes a
 * content-free HTML file, and OTLP export writes JSON without an endpoint.
 */

import { execFile } from "child_process";
import * as fs from "fs";
import * as path from "path";
import * as vscode from "vscode";
import {
  buildReviewModel,
  CliHealth,
  loadAnnotations,
  loadStoredReview,
  renderMinimizedReviewHtml,
  ReviewModel,
} from "./reviewModel";
import { SessionState } from "./traceStore";

interface CliResult {
  stdout: string;
  stderr: string;
  failed: boolean;
}

function runCli(args: string[]): Promise<CliResult> {
  return new Promise((resolve, reject) => {
    execFile(
      "agent-strace",
      args,
      { encoding: "utf8", maxBuffer: 8 * 1024 * 1024 },
      (error, stdout, stderr) => {
        if (error && !stdout && "code" in error && error.code === "ENOENT") {
          reject(new Error("agent-strace CLI was not found on PATH."));
          return;
        }
        resolve({
          stdout: String(stdout ?? ""),
          stderr: String(stderr ?? ""),
          failed: Boolean(error),
        });
      },
    );
  });
}

export class EventStreamPanel implements vscode.WebviewViewProvider {
  public static readonly viewId = "agentTrace.eventStream";

  private view?: vscode.WebviewView;
  private currentState: SessionState | null = null;
  private selectedSessionId: string | null = null;
  private currentModel: ReviewModel | null = null;

  constructor(
    private readonly extensionUri: vscode.Uri,
    private readonly traceDir: string,
  ) {}

  resolveWebviewView(
    webviewView: vscode.WebviewView,
    _context: vscode.WebviewViewResolveContext,
    _token: vscode.CancellationToken,
  ): void {
    this.view = webviewView;
    webviewView.webview.options = { enableScripts: true };
    webviewView.webview.html = this._buildHtml(webviewView.webview);
    webviewView.webview.onDidReceiveMessage((message: unknown) => {
      void this._handleMessage(message);
    });

    if (
      this.selectedSessionId &&
      this.selectedSessionId !== this.currentState?.sessionId
    ) {
      void this.showSession(this.selectedSessionId);
    } else if (this.currentState) {
      this._postLiveReview(this.currentState);
    } else if (this.selectedSessionId) {
      void this.showSession(this.selectedSessionId);
    }
  }

  onSessionStart(state: SessionState): void {
    this.currentState = state;
    this.selectedSessionId = state.sessionId;
    this.view?.webview.postMessage({ type: "reset" });
    this._postLiveReview(state);
    void this._refreshHealth(state.sessionId);
  }

  onSessionEnd(state: SessionState): void {
    this.currentState = null;
    if (this.selectedSessionId !== state.sessionId) {
      return;
    }
    this.selectedSessionId = state.sessionId;
    this._postLiveReview(state);
    this.view?.webview.postMessage({ type: "sessionEnd" });
    void this.showSession(state.sessionId);
  }

  onStateChange(state: SessionState): void {
    this.currentState = state;
    if (this.selectedSessionId === state.sessionId) {
      this._postLiveReview(state);
    }
  }

  async showSession(sessionId: string): Promise<void> {
    this.selectedSessionId = sessionId;
    try {
      const structural = loadStoredReview(this.traceDir, sessionId);
      this._postReview(structural, null);
      const health = await this._loadHealth(sessionId);
      if (this.selectedSessionId !== sessionId) {
        return;
      }
      this._postReview(loadStoredReview(this.traceDir, sessionId, health), null);
    } catch (error) {
      this.view?.webview.postMessage({
        type: "error",
        message: error instanceof Error ? error.message : String(error),
      });
    }
  }

  private _postLiveReview(state: SessionState): void {
    const model = buildReviewModel(
      state.meta,
      state.events,
      loadAnnotations(this.traceDir, state.sessionId),
    );
    this._postReview(model, state);
  }

  private _postReview(model: ReviewModel, liveState: SessionState | null): void {
    this.currentModel = model;
    this.view?.webview.postMessage({
      type: "review",
      model,
      controls: {
        live: Boolean(liveState),
        paused: liveState?.paused ?? false,
        activeTool: liveState?.activeTool ?? null,
        cost: liveState?.estimatedCostUsd.toFixed(4) ?? null,
      },
    });
  }

  private async _refreshHealth(sessionId: string): Promise<void> {
    try {
      const health = await this._loadHealth(sessionId);
      if (
        this.selectedSessionId !== sessionId ||
        this.currentState?.sessionId !== sessionId
      ) {
        return;
      }
      const state = this.currentState;
      const model = buildReviewModel(
        state.meta,
        state.events,
        loadAnnotations(this.traceDir, sessionId),
        health,
      );
      this._postReview(model, state);
    } catch {
      // Structural review remains available when the CLI is not on PATH.
    }
  }

  private async _loadHealth(sessionId: string): Promise<CliHealth> {
    const result = await runCli([
      "--trace-dir",
      this.traceDir,
      "health",
      sessionId,
      "--format",
      "json",
    ]);
    try {
      return JSON.parse(result.stdout) as CliHealth;
    } catch {
      throw new Error(result.stderr.trim() || "Evidence health JSON was unavailable.");
    }
  }

  private async _handleMessage(raw: unknown): Promise<void> {
    if (typeof raw !== "object" || raw === null) {
      return;
    }
    const message = raw as { command?: string };
    if (message.command === "pause") {
      await vscode.commands.executeCommand("agentTrace.pauseAgent");
    } else if (message.command === "resume") {
      await vscode.commands.executeCommand("agentTrace.resumeAgent");
    } else if (message.command === "compare") {
      await this._compareCurrentSession();
    } else if (message.command === "saveMinimizedShare") {
      await this._saveMinimizedShare();
    } else if (message.command === "exportOtlp") {
      await this._exportOtlp();
    } else if (message.command === "verifyLiveCapture") {
      await this._verifyLiveCapture();
    }
  }

  private async _compareCurrentSession(): Promise<void> {
    const current = this.currentModel?.session.session_id;
    if (!current) {
      return;
    }
    const sessions = this._sessionChoices().filter((choice) => choice.id !== current);
    const selected = await vscode.window.showQuickPick(sessions, {
      placeHolder: "Choose a local session to compare",
      matchOnDescription: true,
    });
    if (!selected) {
      return;
    }
    try {
      const result = await runCli([
        "--trace-dir",
        this.traceDir,
        "compare",
        current,
        selected.id,
        "--format",
        "json",
      ]);
      if (result.failed) {
        throw new Error(result.stderr.trim() || "Session comparison failed.");
      }
      this.view?.webview.postMessage({
        type: "comparison",
        report: JSON.parse(result.stdout),
      });
    } catch (error) {
      vscode.window.showErrorMessage(
        `agent-trace: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  private _sessionChoices(): Array<vscode.QuickPickItem & { id: string }> {
    try {
      return fs.readdirSync(this.traceDir).flatMap((id) => {
        const metaPath = path.join(this.traceDir, id, "meta.json");
        try {
          const meta = JSON.parse(fs.readFileSync(metaPath, "utf8")) as {
            agent_name?: string;
            started_at?: number;
          };
          return [{
            id,
            label: id.slice(0, 12),
            description: [
              meta.agent_name,
              meta.started_at ? new Date(meta.started_at * 1000).toLocaleString() : "",
            ].filter(Boolean).join(" | "),
          }];
        } catch {
          return [];
        }
      });
    } catch {
      return [];
    }
  }

  private async _saveMinimizedShare(): Promise<void> {
    if (!this.currentModel) {
      return;
    }
    const target = await vscode.window.showSaveDialog({
      defaultUri: vscode.Uri.file(path.join(
        path.dirname(this.traceDir),
        `review-${this.currentModel.session.session_id.slice(0, 12)}-minimized.html`,
      )),
      filters: { "HTML files": ["html"] },
      saveLabel: "Save minimized review",
    });
    if (!target) {
      return;
    }
    await vscode.workspace.fs.writeFile(
      target,
      Buffer.from(renderMinimizedReviewHtml(this.currentModel), "utf8"),
    );
    vscode.window.showInformationMessage("Privacy-minimized review saved locally.");
  }

  private async _exportOtlp(): Promise<void> {
    const sessionId = this.currentModel?.session.session_id;
    if (!sessionId) {
      return;
    }
    const target = await vscode.window.showSaveDialog({
      defaultUri: vscode.Uri.file(path.join(
        path.dirname(this.traceDir),
        `trace-${sessionId.slice(0, 12)}-otlp.json`,
      )),
      filters: { "JSON files": ["json"] },
      saveLabel: "Export OTLP JSON",
    });
    if (!target) {
      return;
    }
    try {
      const result = await runCli([
        "--trace-dir",
        this.traceDir,
        "export",
        sessionId,
        "--format",
        "otlp",
        "--output",
        target.fsPath,
      ]);
      if (result.failed) {
        throw new Error(result.stderr.trim() || "OTLP export failed.");
      }
      vscode.window.showInformationMessage("OTLP JSON saved locally.");
    } catch (error) {
      vscode.window.showErrorMessage(
        `agent-trace: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  private async _verifyLiveCapture(): Promise<void> {
    const model = this.currentModel;
    if (!model) {
      return;
    }
    const provider = model.session.provider;
    if (provider !== "claude" && provider !== "codex") {
      vscode.window.showWarningMessage(
        "Live capture verification currently supports Claude Code and Codex.",
      );
      return;
    }
    try {
      const result = await runCli([
        "--trace-dir",
        this.traceDir,
        "setup",
        "--cli",
        provider,
        "--verify",
        "--live",
        "--session",
        model.session.session_id,
      ]);
      this.view?.webview.postMessage({
        type: "liveCapture",
        result: {
          ok: !result.failed,
          output: [result.stdout.trim(), result.stderr.trim()]
            .filter(Boolean)
            .join("\n"),
        },
      });
    } catch (error) {
      this.view?.webview.postMessage({
        type: "liveCapture",
        result: {
          ok: false,
          output: error instanceof Error ? error.message : String(error),
        },
      });
    }
  }

  private _getNonce(): string {
    let text = "";
    const possible = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    for (let index = 0; index < 32; index++) {
      text += possible.charAt(Math.floor(Math.random() * possible.length));
    }
    return text;
  }

  private _buildHtml(_webview: vscode.Webview): string {
    const htmlPath = path.join(this.extensionUri.fsPath, "media", "panel.html");
    const nonce = this._getNonce();
    return fs.readFileSync(htmlPath, "utf8").replace(/NONCE_PLACEHOLDER/g, nonce);
  }
}
