/**
 * Session browser tree view (#119).
 *
 * Implements a VS Code TreeDataProvider that shows all sessions grouped by
 * date, with cost, status, and task name at a glance.
 *
 * Tree structure:
 *   AGENT TRACE SESSIONS
 *   ├── Today
 *   │   ├── a84664242afa  $5.03  ⚠ budget_exceeded  refactor-auth
 *   │   └── bf1207728ee6  $2.87  ✓ completed        add-tests
 *   └── Yesterday
 *       └── c91ab3312fde  $4.87  ✓ completed        fix-login-bug
 *
 * Clicking a session opens the replay panel (agentTrace.openPanel).
 * Right-click context menu: Replay, View post-mortem, Export OTLP, Delete.
 * Auto-refreshes when new sessions are written (file watcher on traceDir).
 */

import * as fs from "fs";
import * as path from "path";
import * as vscode from "vscode";

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

interface SessionInfo {
  session_id: string;
  started_at: number;
  ended_at?: number;
  agent_name?: string;
  command?: string;
  tool_calls?: number;
  errors?: number;
}

type SessionStatus = "completed" | "watchdog" | "error" | "in_progress";

interface SessionEntry {
  info: SessionInfo;
  status: SessionStatus;
  costUsd: number;
  hasPostMortem: boolean;
}

// ---------------------------------------------------------------------------
// Tree item types
// ---------------------------------------------------------------------------

export class DateGroupItem extends vscode.TreeItem {
  constructor(
    public readonly label: string,
    public readonly sessions: SessionEntry[]
  ) {
    super(label, vscode.TreeItemCollapsibleState.Expanded);
    this.contextValue = "dateGroup";
    this.description = `${sessions.length} session${sessions.length !== 1 ? "s" : ""}`;
  }
}

export class SessionItem extends vscode.TreeItem {
  constructor(public readonly entry: SessionEntry) {
    const { info, status, costUsd } = entry;
    const shortId = info.session_id.slice(0, 12);
    const name = info.agent_name || info.command || shortId;
    const icon = _statusIcon(status);
    const costStr = `$${costUsd.toFixed(2)}`;

    super(`${icon} ${shortId}`, vscode.TreeItemCollapsibleState.None);

    this.description = `${costStr}  ${name.slice(0, 30)}`;
    this.tooltip = [
      `Session: ${info.session_id}`,
      `Status: ${status}`,
      `Cost: ${costStr}`,
      `Task: ${name}`,
      `Started: ${new Date(info.started_at * 1000).toLocaleString()}`,
    ].join("\n");

    this.contextValue = entry.hasPostMortem ? "sessionWithPostMortem" : "session";

    this.command = {
      command: "agentTrace.openPanel",
      title: "Open session",
      arguments: [info.session_id],
    };
  }
}

function _statusIcon(status: SessionStatus): string {
  switch (status) {
    case "completed":   return "✓";
    case "watchdog":    return "⚠";
    case "error":       return "✗";
    case "in_progress": return "⟳";
  }
}

// ---------------------------------------------------------------------------
// Cost estimation (mirrors Python heuristic)
// ---------------------------------------------------------------------------

function _estimateCost(traceDir: string, sessionId: string): number {
  const eventsFile = path.join(traceDir, sessionId, "events.ndjson");
  try {
    const lines = fs.readFileSync(eventsFile, "utf8").split("\n").filter(Boolean);
    let cost = 0;
    for (const line of lines) {
      try {
        const ev = JSON.parse(line);
        const payload = JSON.stringify(ev.data ?? {});
        const tokens = Math.floor(payload.length / 4);
        if (ev.event_type === "llm_request") {
          cost += tokens * 3.0 / 1_000_000;
        } else if (ev.event_type === "llm_response" || ev.event_type === "assistant_response") {
          cost += tokens * 15.0 / 1_000_000;
        }
      } catch { /* skip malformed */ }
    }
    return cost;
  } catch {
    return 0;
  }
}

// ---------------------------------------------------------------------------
// Session loading
// ---------------------------------------------------------------------------

function _loadSessions(traceDir: string): SessionEntry[] {
  if (!fs.existsSync(traceDir)) { return []; }

  const entries: SessionEntry[] = [];

  try {
    for (const name of fs.readdirSync(traceDir)) {
      const sessionDir = path.join(traceDir, name);
      try { if (!fs.statSync(sessionDir).isDirectory()) { continue; } } catch { continue; }
      const metaFile = path.join(sessionDir, "meta.json");
      if (!fs.existsSync(metaFile)) { continue; }

      let info: SessionInfo;
      try {
        info = JSON.parse(fs.readFileSync(metaFile, "utf8")) as SessionInfo;
      } catch { continue; }

      const hasPostMortem = fs.existsSync(
        path.join(sessionDir, "watchdog-postmortem.json")
      );

      // Determine status
      let status: SessionStatus;
      if (hasPostMortem) {
        status = "watchdog";
      } else if ((info.errors ?? 0) > 0) {
        status = "error";
      } else if (info.ended_at) {
        status = "completed";
      } else {
        status = "in_progress";
      }

      const costUsd = _estimateCost(traceDir, info.session_id);

      entries.push({ info, status, costUsd, hasPostMortem });
    }
  } catch { /* ignore */ }

  // Sort newest first
  return entries.sort((a, b) => b.info.started_at - a.info.started_at);
}

// ---------------------------------------------------------------------------
// Date grouping
// ---------------------------------------------------------------------------

function _dayLabel(ts: number): string {
  const now = new Date();
  const d = new Date(ts * 1000);
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const yesterday = new Date(today.getTime() - 86400_000);
  const sessionDay = new Date(d.getFullYear(), d.getMonth(), d.getDate());

  if (sessionDay.getTime() === today.getTime()) { return "Today"; }
  if (sessionDay.getTime() === yesterday.getTime()) { return "Yesterday"; }
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function _groupByDate(sessions: SessionEntry[]): DateGroupItem[] {
  const groups = new Map<string, SessionEntry[]>();
  for (const s of sessions) {
    const label = _dayLabel(s.info.started_at);
    if (!groups.has(label)) { groups.set(label, []); }
    groups.get(label)!.push(s);
  }
  return Array.from(groups.entries()).map(([label, items]) => new DateGroupItem(label, items));
}

// ---------------------------------------------------------------------------
// TreeDataProvider
// ---------------------------------------------------------------------------

export class SessionTreeProvider
  implements vscode.TreeDataProvider<DateGroupItem | SessionItem>
{
  private readonly _onDidChangeTreeData =
    new vscode.EventEmitter<DateGroupItem | SessionItem | undefined | null | void>();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  private traceDir: string;
  private dirWatcher: fs.FSWatcher | null = null;
  private refreshTimer: ReturnType<typeof setInterval> | null = null;
  private readonly refreshIntervalMs: number;

  constructor(traceDir: string) {
    this.traceDir = traceDir;
    this.refreshIntervalMs =
      (vscode.workspace.getConfiguration("agentTrace")
        .get<number>("sessionBrowserRefreshInterval", 5)) * 1000;

    this._startWatching();
  }

  private _startWatching(): void {
    // File watcher for fast refresh
    if (fs.existsSync(this.traceDir)) {
      try {
        this.dirWatcher = fs.watch(this.traceDir, { recursive: false }, () => {
          this.refresh();
        });
      } catch { /* ignore */ }
    }

    // Polling fallback
    this.refreshTimer = setInterval(() => this.refresh(), this.refreshIntervalMs);
  }

  refresh(): void {
    this._onDidChangeTreeData.fire();
  }

  getTreeItem(element: DateGroupItem | SessionItem): vscode.TreeItem {
    return element;
  }

  getChildren(element?: DateGroupItem | SessionItem): (DateGroupItem | SessionItem)[] {
    if (!element) {
      // Root: return date groups
      const sessions = _loadSessions(this.traceDir);
      return _groupByDate(sessions);
    }
    if (element instanceof DateGroupItem) {
      return element.sessions.map((s) => new SessionItem(s));
    }
    return [];
  }

  dispose(): void {
    this.dirWatcher?.close();
    if (this.refreshTimer) { clearInterval(this.refreshTimer); }
    this._onDidChangeTreeData.dispose();
  }
}
