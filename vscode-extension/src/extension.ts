/**
 * agent-trace VS Code extension entry point.
 *
 * Activates when a workspace contains a .agent-traces directory.
 * Zero overhead when no session is active — all watchers are fs.watch
 * based, no polling timers run until a session starts.
 */

import * as path from "path";
import * as vscode from "vscode";
import { DecorationManager } from "./decorations";
import { EventStreamPanel } from "./panel";
import { PauseManager } from "./pauseAgent";
import { StatusBarManager } from "./statusBar";
import { TraceWatcher } from "./traceStore";

export function activate(context: vscode.ExtensionContext): void {
  const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
  if (!workspaceRoot) { return; }

  const config = vscode.workspace.getConfiguration("agentTrace");
  const traceDirSetting = config.get<string>("traceDir", ".agent-traces")!;
  const traceDir = path.isAbsolute(traceDirSetting)
    ? traceDirSetting
    : path.join(workspaceRoot, traceDirSetting);

  // -------------------------------------------------------------------------
  // Core components
  // -------------------------------------------------------------------------

  const watcher = new TraceWatcher(workspaceRoot, traceDirSetting);
  const statusBar = new StatusBarManager();
  const decorations = new DecorationManager();
  const pauseManager = new PauseManager(traceDir);
  const panel = new EventStreamPanel(context.extensionUri);

  // -------------------------------------------------------------------------
  // Webview panel registration
  // -------------------------------------------------------------------------

  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(EventStreamPanel.viewId, panel, {
      webviewOptions: { retainContextWhenHidden: true },
    })
  );

  // -------------------------------------------------------------------------
  // Wire watcher → UI components
  // -------------------------------------------------------------------------

  watcher.onSessionStart((state) => {
    vscode.commands.executeCommand(
      "setContext",
      "agentTrace.sessionActive",
      true
    );
    statusBar.update(state);
    decorations.update(state);
    panel.onSessionStart(state);
  });

  watcher.onSessionEnd((state) => {
    vscode.commands.executeCommand(
      "setContext",
      "agentTrace.sessionActive",
      false
    );
    statusBar.update(null);
    decorations.update(null);
    panel.onSessionEnd(state);
    pauseManager.cleanup();
  });

  watcher.onStateChange((state) => {
    statusBar.update(state);
    if (config.get<boolean>("showGutterAnnotations", true)) {
      decorations.update(state);
    }
  });

  watcher.onEvent(({ state, event }) => {
    panel.pushEvent(state, event);
  });

  // -------------------------------------------------------------------------
  // Commands
  // -------------------------------------------------------------------------

  context.subscriptions.push(
    vscode.commands.registerCommand("agentTrace.pauseAgent", () => {
      const state = watcher.state;
      if (!state) {
        vscode.window.showWarningMessage("agent-trace: no active session to pause.");
        return;
      }
      pauseManager.pause(state);
      statusBar.update(state);
      vscode.window.setStatusBarMessage("agent-trace: agent paused.", 3000);
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("agentTrace.resumeAgent", () => {
      const state = watcher.state;
      if (!state) { return; }
      pauseManager.resume(state);
      statusBar.update(state);
      vscode.window.setStatusBarMessage("agent-trace: agent resumed.", 3000);
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("agentTrace.openPanel", () => {
      vscode.commands.executeCommand("agentTrace.eventStream.focus");
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("agentTrace.clearDecorations", () => {
      decorations.clear();
    })
  );

  // -------------------------------------------------------------------------
  // Config change — re-read traceDir if it changes
  // -------------------------------------------------------------------------

  context.subscriptions.push(
    vscode.workspace.onDidChangeConfiguration((e) => {
      if (e.affectsConfiguration("agentTrace")) {
        vscode.window.showInformationMessage(
          "agent-trace: configuration changed — reload window to apply."
        );
      }
    })
  );

  // -------------------------------------------------------------------------
  // Start watching
  // -------------------------------------------------------------------------

  watcher.start();

  // Register disposables
  context.subscriptions.push(watcher, statusBar, decorations);
}

export function deactivate(): void {
  // Disposables registered via context.subscriptions are cleaned up
  // automatically. PauseManager.cleanup() is called in onSessionEnd,
  // but call it here too in case the window closes mid-session.
}
