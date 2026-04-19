/**
 * Gutter annotations and inline text for agent-touched files.
 *
 * Files the agent read:     subtle left-border highlight + inline "← agent read Nx"
 * Files the agent modified: stronger highlight + inline "← agent modified Nx"
 *
 * Decorations are applied to the first line of the file (non-intrusive).
 * Cleared when the session ends or the user runs agentTrace.clearDecorations.
 */

import * as vscode from "vscode";
import { FileAccess, SessionState } from "./traceStore";

// Decoration type for files the agent has read (but not written)
const readDecoration = vscode.window.createTextEditorDecorationType({
  isWholeLine: true,
  overviewRulerColor: new vscode.ThemeColor("editorInfo.foreground"),
  overviewRulerLane: vscode.OverviewRulerLane.Left,
  borderWidth: "0 0 0 2px",
  borderStyle: "solid",
  borderColor: new vscode.ThemeColor("editorInfo.foreground"),
  light: { borderColor: "#4a9eff55" },
  dark: { borderColor: "#4a9eff55" },
});

// Decoration type for files the agent has written
const writeDecoration = vscode.window.createTextEditorDecorationType({
  isWholeLine: true,
  overviewRulerColor: new vscode.ThemeColor("editorWarning.foreground"),
  overviewRulerLane: vscode.OverviewRulerLane.Left,
  borderWidth: "0 0 0 3px",
  borderStyle: "solid",
  borderColor: new vscode.ThemeColor("editorWarning.foreground"),
  light: { borderColor: "#e5a00d88" },
  dark: { borderColor: "#e5a00d88" },
});

export class DecorationManager extends vscode.Disposable {
  private readonly disposables: vscode.Disposable[] = [];

  constructor() {
    super(() => this._disposeAll());

    // Re-apply decorations when the active editor changes
    this.disposables.push(
      vscode.window.onDidChangeActiveTextEditor((editor) => {
        if (editor && this.currentState) {
          this._applyToEditor(editor, this.currentState.fileAccess);
        }
      })
    );
  }

  private currentState: SessionState | null = null;

  update(state: SessionState | null): void {
    this.currentState = state;

    if (!state) {
      this._clearAll();
      return;
    }

    for (const editor of vscode.window.visibleTextEditors) {
      this._applyToEditor(editor, state.fileAccess);
    }
  }

  clear(): void {
    this._clearAll();
    this.currentState = null;
  }

  private _applyToEditor(
    editor: vscode.TextEditor,
    fileAccess: Map<string, FileAccess>
  ): void {
    const filePath = editor.document.uri.fsPath;
    const access = fileAccess.get(filePath);

    if (!access) {
      editor.setDecorations(readDecoration, []);
      editor.setDecorations(writeDecoration, []);
      return;
    }

    const firstLine = new vscode.Range(0, 0, 0, 0);

    if (access.writes > 0) {
      const label = _label(access);
      editor.setDecorations(readDecoration, []);
      editor.setDecorations(writeDecoration, [
        {
          range: firstLine,
          renderOptions: {
            after: {
              contentText: `  ← ${label}`,
              color: new vscode.ThemeColor("editorWarning.foreground"),
              fontStyle: "italic",
              margin: "0 0 0 2em",
            },
          },
        },
      ]);
    } else {
      const label = _label(access);
      editor.setDecorations(writeDecoration, []);
      editor.setDecorations(readDecoration, [
        {
          range: firstLine,
          renderOptions: {
            after: {
              contentText: `  ← ${label}`,
              color: new vscode.ThemeColor("editorInfo.foreground"),
              fontStyle: "italic",
              margin: "0 0 0 2em",
            },
          },
        },
      ]);
    }
  }

  private _clearAll(): void {
    for (const editor of vscode.window.visibleTextEditors) {
      editor.setDecorations(readDecoration, []);
      editor.setDecorations(writeDecoration, []);
    }
  }

  private _disposeAll(): void {
    this._clearAll();
    readDecoration.dispose();
    writeDecoration.dispose();
    for (const d of this.disposables) { d.dispose(); }
  }
}

function _label(access: FileAccess): string {
  const parts: string[] = [];
  if (access.reads > 0) {
    parts.push(`agent read ${access.reads}×`);
  }
  if (access.writes > 0) {
    parts.push(`agent modified ${access.writes}×`);
  }
  return parts.join(", ") + " this session";
}
