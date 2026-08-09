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
import { LiveStreamPanel } from "./liveStream";
import { PostMortemManager } from "./postMortem";
import { SessionTreeProvider } from "./sessionTree";
import { StatusBarManager, WatchdogStatusBar } from "./statusBar";
import { ExtensionTelemetry } from "./telemetry";
import { TraceWatcher } from "./traceStore";

export function activate(context: vscode.ExtensionContext): void {
  const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
  if (!workspaceRoot) {
    return;
  }

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
  const watchdogBar = new WatchdogStatusBar();
  const decorations = new DecorationManager();
  const pauseManager = new PauseManager(traceDir);
  const panel = new EventStreamPanel(context.extensionUri);
  const postMortem = new PostMortemManager(traceDir);
  const liveStream = new LiveStreamPanel(context.extensionUri);
  const sessionTree = new SessionTreeProvider(traceDir);
  const telemetry = new ExtensionTelemetry(context);

  // -------------------------------------------------------------------------
  // Webview panel registration
  // -------------------------------------------------------------------------

  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(EventStreamPanel.viewId, panel, {
      webviewOptions: { retainContextWhenHidden: true },
    }),
  );

  // -------------------------------------------------------------------------
  // Wire watcher → UI components
  // -------------------------------------------------------------------------

  watcher.onSessionStart((state) => {
    telemetry.captureSessionStarted();
    vscode.commands.executeCommand(
      "setContext",
      "agentTrace.sessionActive",
      true,
    );
    statusBar.update(state);
    watchdogBar.onSessionStart(traceDir, state);
    decorations.update(state);
    panel.onSessionStart(state);
  });

  watcher.onSessionEnd((state) => {
    telemetry.captureSessionCompleted(state);
    vscode.commands.executeCommand(
      "setContext",
      "agentTrace.sessionActive",
      false,
    );
    statusBar.update(null);
    watchdogBar.onSessionEnd();
    decorations.update(null);
    panel.onSessionEnd(state);
    pauseManager.cleanup();
  });

  watcher.onStateChange((state) => {
    statusBar.update(state);
    watchdogBar.onStateChange(state);
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
    vscode.commands.registerCommand(
      "agentTrace.pauseAgent",
      telemetry.trackCommand("pause_agent", () => {
        const state = watcher.state;
        if (!state) {
          vscode.window.showWarningMessage(
            "agent-trace: no active session to pause.",
          );
          return;
        }
        pauseManager.pause(state);
        statusBar.update(state);
        vscode.window.setStatusBarMessage("agent-trace: agent paused.", 3000);
      }),
    ),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand(
      "agentTrace.resumeAgent",
      telemetry.trackCommand("resume_agent", () => {
        const state = watcher.state;
        if (!state) {
          return;
        }
        pauseManager.resume(state);
        statusBar.update(state);
        vscode.window.setStatusBarMessage("agent-trace: agent resumed.", 3000);
      }),
    ),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand(
      "agentTrace.openPanel",
      telemetry.trackCommand("open_panel", () => {
        vscode.commands.executeCommand("agentTrace.eventStream.focus");
      }),
    ),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand(
      "agentTrace.clearDecorations",
      telemetry.trackCommand("clear_decorations", () => {
        decorations.clear();
      }),
    ),
  );

  // -------------------------------------------------------------------------
  // Config change — re-read traceDir if it changes
  // -------------------------------------------------------------------------

  context.subscriptions.push(
    vscode.workspace.onDidChangeConfiguration((e) => {
      const telemetryChanged = e.affectsConfiguration(
        "agentTrace.telemetry.enabled",
      );
      if (telemetryChanged) {
        telemetry.handleConfigurationChange();
      }
      if (e.affectsConfiguration("agentTrace") && !telemetryChanged) {
        vscode.window.showInformationMessage(
          "agent-trace: configuration changed — reload window to apply.",
        );
      }
    }),
  );

  context.subscriptions.push(
    vscode.env.onDidChangeTelemetryEnabled((enabled) => {
      telemetry.handleEditorTelemetryChange(enabled);
    }),
  );

  // -------------------------------------------------------------------------
  // Start watching
  // -------------------------------------------------------------------------

  telemetry.initialize();
  watcher.start();

  // Register session browser tree view
  const treeView = vscode.window.registerTreeDataProvider(
    "agentTrace.sessionBrowser",
    sessionTree,
  );

  // Register commands for new features
  context.subscriptions.push(
    vscode.commands.registerCommand(
      "agentTrace.openLiveStream",
      telemetry.trackCommand("open_live_stream", () => {
        liveStream.open();
      }),
    ),
    vscode.commands.registerCommand(
      "agentTrace.openPostMortem",
      telemetry.trackCommand("open_post_mortem", (sessionId?: unknown) => {
        if (typeof sessionId === "string" && sessionId) {
          postMortem.openForSession(sessionId);
        } else {
          // Prompt user to pick a session
          return vscode.window
            .showInputBox({ prompt: "Enter session ID" })
            .then((id) => {
              if (id) {
                postMortem.openForSession(id);
              }
            });
        }
      }),
    ),
    vscode.commands.registerCommand(
      "agentTrace.refreshSessionBrowser",
      telemetry.trackCommand("refresh_session_browser", () => {
        sessionTree.refresh();
      }),
    ),
    vscode.commands.registerCommand(
      "agentTrace.revealSession",
      telemetry.trackCommand("reveal_session", (_sessionId?: unknown) => {
        // Refresh tree and let the user find the session
        sessionTree.refresh();
      }),
    ),
    vscode.commands.registerCommand(
      "agentTrace.enableTelemetry",
      telemetry.trackCommand("enable_telemetry", async () => {
        await telemetry.setEnabled(true, "command");
        vscode.window.showInformationMessage(
          "agent-strace: anonymous product telemetry enabled.",
        );
      }),
    ),
    vscode.commands.registerCommand(
      "agentTrace.disableTelemetry",
      telemetry.trackCommand("disable_telemetry", async () => {
        await telemetry.setEnabled(false, "command");
        vscode.window.showInformationMessage(
          "agent-strace: anonymous product telemetry disabled and anonymous ID deleted.",
        );
      }),
    ),
  );

  // Start post-mortem watcher
  postMortem.start();

  // Register disposables
  context.subscriptions.push(
    watcher,
    statusBar,
    watchdogBar,
    decorations,
    postMortem,
    liveStream,
    treeView,
    telemetry,
    { dispose: () => sessionTree.dispose() },
  );
}

export function deactivate(): void {
  // Disposables registered via context.subscriptions are cleaned up
  // automatically. PauseManager.cleanup() is called in onSessionEnd,
  // but call it here too in case the window closes mid-session.
}
