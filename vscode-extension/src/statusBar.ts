/**
 * Status bar items for agent-trace.
 *
 * StatusBarManager — live session cost, tool call count, active tool.
 *   Idle (no session): hidden.
 *   Active session:    "$(pulse) agent  $0.0042  47 calls  [Read]"
 *   Paused session:    "$(debug-pause) agent  PAUSED"
 *
 * WatchdogStatusBar — live budget and timeout display when a watchdog session
 * is active. Reads watchdog config from meta.json and polls for cost/elapsed.
 * Switches to a red death-state display when post-mortem.json appears.
 *   Active:  "$(clock) 12m / 30m  $(credit-card) $2.43 / $5.00"
 *   Dead:    "$(error) Budget exceeded — post-mortem ready"
 */

import * as fs from "fs";
import * as path from "path";
import * as vscode from "vscode";
import { SessionState } from "./traceStore";

// ---------------------------------------------------------------------------
// StatusBarManager (existing, unchanged)
// ---------------------------------------------------------------------------

export class StatusBarManager extends vscode.Disposable {
  private readonly item: vscode.StatusBarItem;

  constructor() {
    super(() => this.dispose());
    this.item = vscode.window.createStatusBarItem(
      vscode.StatusBarAlignment.Left,
      100
    );
    this.item.command = "agentTrace.openPanel";
    this.item.tooltip = "agent-trace — click to open event stream";
  }

  update(state: SessionState | null): void {
    if (!state) {
      this.item.hide();
      return;
    }

    const cost = `$${state.estimatedCostUsd.toFixed(4)}`;
    const calls = `${state.toolCallCount} calls`;

    if (state.paused) {
      this.item.text = `$(debug-pause) agent  PAUSED`;
      this.item.backgroundColor = new vscode.ThemeColor(
        "statusBarItem.warningBackground"
      );
    } else {
      const tool = state.activeTool ? `  [${state.activeTool}]` : "";
      this.item.text = `$(pulse) agent  ${cost}  ${calls}${tool}`;
      this.item.backgroundColor = undefined;
    }

    this.item.show();
  }

  hide(): void {
    this.item.hide();
  }

  override dispose(): void {
    this.item.dispose();
  }
}

// ---------------------------------------------------------------------------
// WatchdogStatusBar (#117)
// ---------------------------------------------------------------------------

interface WatchdogConfig {
  timeoutSeconds: number | null;   // from meta.json watchdog_timeout_seconds
  budgetDollars: number | null;    // from meta.json watchdog_budget_dollars
  startedAt: number;               // Unix timestamp
}

function _fmtElapsed(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return m > 0 ? `${m}m ${s.toString().padStart(2, "0")}s` : `${s}s`;
}

function _fmtLimit(seconds: number): string {
  const m = Math.floor(seconds / 60);
  return m > 0 ? `${m}m` : `${seconds}s`;
}

export class WatchdogStatusBar extends vscode.Disposable {
  private readonly item: vscode.StatusBarItem;
  private pollTimer: ReturnType<typeof setInterval> | null = null;
  private pmWatcher: fs.FSWatcher | null = null;
  private traceDir: string = "";
  private sessionId: string | null = null;
  private watchdogCfg: WatchdogConfig | null = null;
  private dead: boolean = false;
  private deathReason: string = "";
  private readonly pollIntervalMs: number;

  constructor(pollIntervalMs?: number) {
    super(() => this._cleanup());
    this.pollIntervalMs = pollIntervalMs ??
      (vscode.workspace.getConfiguration("agentTrace")
        .get<number>("watchdogPollIntervalSeconds", 5) * 1000);

    this.item = vscode.window.createStatusBarItem(
      vscode.StatusBarAlignment.Left,
      99   // just below the main status bar item
    );
    this.item.command = "agentTrace.openPanel";
  }

  /** Called when a new session starts. Reads watchdog config from meta.json. */
  onSessionStart(traceDir: string, state: SessionState): void {
    this._cleanup();
    this.traceDir = traceDir;
    this.sessionId = state.sessionId;
    this.dead = false;
    this.deathReason = "";

    const cfg = this._readWatchdogConfig(state);
    if (!cfg) {
      // No watchdog config — stay hidden
      return;
    }
    this.watchdogCfg = cfg;

    // Watch for post-mortem file
    this._watchPostMortem();

    // Start polling
    this.pollTimer = setInterval(() => this._tick(state), this.pollIntervalMs);
    this._tick(state);
  }

  /** Called on every state change to update cost display. */
  onStateChange(state: SessionState): void {
    if (!this.watchdogCfg || this.dead) { return; }
    this._render(state.estimatedCostUsd);
  }

  /** Called when the session ends. */
  onSessionEnd(): void {
    this._cleanup();
    this.item.hide();
  }

  private _readWatchdogConfig(state: SessionState): WatchdogConfig | null {
    if (!this.traceDir || !this.sessionId) { return null; }
    const metaPath = path.join(this.traceDir, this.sessionId, "meta.json");
    try {
      const meta = JSON.parse(fs.readFileSync(metaPath, "utf8"));
      const timeout = meta.watchdog_timeout_seconds ?? meta.max_duration_seconds ?? null;
      const budget = meta.watchdog_budget_dollars ?? meta.max_cost_dollars ?? null;
      if (!timeout && !budget) { return null; }
      return {
        timeoutSeconds: timeout ? Number(timeout) : null,
        budgetDollars: budget ? Number(budget) : null,
        startedAt: state.meta.started_at,
      };
    } catch {
      return null;
    }
  }

  private _watchPostMortem(): void {
    if (!this.traceDir || !this.sessionId) { return; }
    const pmPath = path.join(this.traceDir, this.sessionId, "watchdog-postmortem.json");
    const sessionDir = path.join(this.traceDir, this.sessionId);

    const checkPm = () => {
      if (!fs.existsSync(pmPath)) { return; }
      try {
        const pm = JSON.parse(fs.readFileSync(pmPath, "utf8"));
        this.dead = true;
        this.deathReason = pm.reason ?? "terminated";
        this._renderDead();
      } catch { /* ignore */ }
    };

    // Check immediately (may already exist)
    checkPm();

    // Watch the session directory for new files
    try {
      this.pmWatcher = fs.watch(sessionDir, (_evt, filename) => {
        if (filename === "watchdog-postmortem.json") { checkPm(); }
      });
    } catch { /* ignore — polling covers this */ }
  }

  private _tick(state: SessionState): void {
    if (this.dead) { return; }
    this._render(state.estimatedCostUsd);
  }

  private _render(costUsd: number): void {
    if (!this.watchdogCfg) { return; }
    const cfg = this.watchdogCfg;
    const elapsed = Date.now() / 1000 - cfg.startedAt;

    const parts: string[] = [];

    if (cfg.timeoutSeconds) {
      parts.push(`$(clock) ${_fmtElapsed(elapsed)} / ${_fmtLimit(cfg.timeoutSeconds)}`);
    }
    if (cfg.budgetDollars) {
      parts.push(`$(credit-card) $${costUsd.toFixed(2)} / $${cfg.budgetDollars.toFixed(2)}`);
    }

    this.item.text = parts.join("  ");
    this.item.backgroundColor = undefined;
    this.item.tooltip = "agent-trace watchdog — click to open session";
    this.item.show();
  }

  private _renderDead(): void {
    const label = this.deathReason.replace(/_/g, " ");
    this.item.text = `$(error) ${label} — post-mortem ready`;
    this.item.backgroundColor = new vscode.ThemeColor("statusBarItem.errorBackground");
    this.item.tooltip = "agent-trace: watchdog terminated session — click to view post-mortem";
    this.item.show();
  }

  private _cleanup(): void {
    if (this.pollTimer) {
      clearInterval(this.pollTimer);
      this.pollTimer = null;
    }
    this.pmWatcher?.close();
    this.pmWatcher = null;
    this.watchdogCfg = null;
    this.sessionId = null;
  }

  override dispose(): void {
    this._cleanup();
    this.item.dispose();
  }
}
