/**
 * Status bar item showing live session cost, tool call count, and active tool.
 *
 * Idle (no session): hidden.
 * Active session:    "$(pulse) agent  $0.0042  47 calls  [Read]"
 * Paused session:    "$(debug-pause) agent  PAUSED"
 */

import * as vscode from "vscode";
import { SessionState } from "./traceStore";

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
