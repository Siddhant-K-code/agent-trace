/**
 * Watches the agent-trace store for live session activity.
 *
 * Reads .agent-traces/.active-session to find the current session,
 * then tails events.ndjson for new events. Emits typed events that
 * the rest of the extension subscribes to.
 *
 * Zero polling when no session is active — uses fs.watch on the
 * .active-session file and the events.ndjson file.
 */

import * as fs from "fs";
import * as path from "path";
import * as vscode from "vscode";

// ---------------------------------------------------------------------------
// Types mirroring agent-trace's Python models
// ---------------------------------------------------------------------------

export type EventType =
  | "session_start"
  | "session_end"
  | "tool_call"
  | "tool_result"
  | "llm_request"
  | "llm_response"
  | "file_read"
  | "file_write"
  | "decision"
  | "error"
  | "user_prompt"
  | "assistant_response";

export interface TraceEvent {
  event_type: EventType;
  timestamp: number;
  event_id: string;
  session_id: string;
  parent_id?: string;
  duration_ms?: number;
  data: Record<string, unknown>;
}

export interface SessionMeta {
  session_id: string;
  started_at: number;
  ended_at?: number;
  agent_name?: string;
  command?: string;
  tool_calls: number;
  errors: number;
  total_tokens: number;
}

/** Aggregated per-file access counts for the current session. */
export interface FileAccess {
  reads: number;
  writes: number;
}

/** Live state derived from the current session's events. */
export interface SessionState {
  sessionId: string;
  meta: SessionMeta;
  /** Absolute path -> access counts */
  fileAccess: Map<string, FileAccess>;
  toolCallCount: number;
  errorCount: number;
  estimatedCostUsd: number;
  activeTool: string | null;
  paused: boolean;
  events: TraceEvent[];
}

// ---------------------------------------------------------------------------
// Cost estimation (mirrors cost.py heuristic: len/4 tokens, sonnet pricing)
// ---------------------------------------------------------------------------

const INPUT_PRICE_PER_TOKEN = 3.0 / 1_000_000;   // $3 / 1M input tokens
const OUTPUT_PRICE_PER_TOKEN = 15.0 / 1_000_000;  // $15 / 1M output tokens

function estimateEventCost(event: TraceEvent): number {
  const payload = JSON.stringify(event.data);
  const tokens = Math.floor(payload.length / 4);
  if (event.event_type === "llm_request") {
    return tokens * INPUT_PRICE_PER_TOKEN;
  }
  if (event.event_type === "llm_response" || event.event_type === "assistant_response") {
    return tokens * OUTPUT_PRICE_PER_TOKEN;
  }
  return 0;
}

// ---------------------------------------------------------------------------
// TraceWatcher
// ---------------------------------------------------------------------------

export class TraceWatcher extends vscode.Disposable {
  private readonly _onSessionStart = new vscode.EventEmitter<SessionState>();
  private readonly _onSessionEnd = new vscode.EventEmitter<SessionState>();
  private readonly _onEvent = new vscode.EventEmitter<{ state: SessionState; event: TraceEvent }>();
  private readonly _onStateChange = new vscode.EventEmitter<SessionState>();

  readonly onSessionStart = this._onSessionStart.event;
  readonly onSessionEnd = this._onSessionEnd.event;
  readonly onEvent = this._onEvent.event;
  readonly onStateChange = this._onStateChange.event;

  private traceDir: string;
  private activeSessionFile: string;
  private currentSessionId: string | null = null;
  private currentState: SessionState | null = null;
  private eventsFileOffset = 0;

  private activeSessionWatcher: fs.FSWatcher | null = null;
  private eventsWatcher: fs.FSWatcher | null = null;
  private pollTimer: ReturnType<typeof setInterval> | null = null;

  constructor(workspaceRoot: string, traceDir: string) {
    super(() => this.dispose());
    this.traceDir = path.isAbsolute(traceDir)
      ? traceDir
      : path.join(workspaceRoot, traceDir);
    this.activeSessionFile = path.join(this.traceDir, ".active-session");
  }

  /** Start watching. Call once after construction. */
  start(): void {
    this._watchActiveSessionFile();
    // Check immediately in case a session is already running
    this._checkActiveSession();
  }

  get state(): SessionState | null {
    return this.currentState;
  }

  // -------------------------------------------------------------------------
  // Active session file watcher
  // -------------------------------------------------------------------------

  private _watchActiveSessionFile(): void {
    const dir = this.traceDir;

    // Always use polling as primary — fs.watch silently fails on network/FUSE
    // filesystems (e.g. Gitpod, WSL, Docker volumes) without throwing.
    this.pollTimer = setInterval(() => this._checkActiveSession(), 500);

    // Also try fs.watch as a faster secondary trigger (best-effort)
    if (fs.existsSync(dir)) {
      try {
        this.activeSessionWatcher = fs.watch(dir, (_event, filename) => {
          if (filename === ".active-session") {
            this._checkActiveSession();
          }
        });
      } catch {
        // polling already running, ignore
      }
    }
  }

  private _checkActiveSession(): void {
    const sessionId = this._readActiveSession();

    if (sessionId && sessionId !== this.currentSessionId) {
      this._startSession(sessionId);
    } else if (!sessionId && this.currentSessionId) {
      this._endSession();
    }
  }

  private _readActiveSession(): string | null {
    try {
      if (!fs.existsSync(this.activeSessionFile)) { return null; }
      const id = fs.readFileSync(this.activeSessionFile, "utf8").trim();
      return id || null;
    } catch {
      return null;
    }
  }

  // -------------------------------------------------------------------------
  // Session lifecycle
  // -------------------------------------------------------------------------

  private _startSession(sessionId: string): void {
    this.currentSessionId = sessionId;
    this.eventsFileOffset = 0;

    const meta = this._loadMeta(sessionId);
    this.currentState = {
      sessionId,
      meta,
      fileAccess: new Map(),
      toolCallCount: 0,
      errorCount: 0,
      estimatedCostUsd: 0,
      activeTool: null,
      paused: false,
      events: [],
    };

    // Replay any events already written before we started watching
    this._readNewEvents();

    // Watch events.ndjson for appends
    this._watchEventsFile(sessionId);

    this._onSessionStart.fire(this.currentState);
    this._onStateChange.fire(this.currentState);
  }

  private _endSession(): void {
    if (!this.currentState) { return; }
    this.eventsWatcher?.close();
    this.eventsWatcher = null;
    this.currentSessionId = null;
    this._onSessionEnd.fire(this.currentState);
    this.currentState = null;
  }

  // -------------------------------------------------------------------------
  // Events file watcher
  // -------------------------------------------------------------------------

  private _watchEventsFile(sessionId: string): void {
    const eventsFile = path.join(this.traceDir, sessionId, "events.ndjson");

    // Always poll — fs.watch silently fails on network/FUSE filesystems.
    // 300ms gives snappy updates without hammering the FS.
    const timer = setInterval(() => {
      if (this.currentSessionId !== sessionId) {
        clearInterval(timer);
        return;
      }
      this._readNewEvents();
    }, 300);

    // Also try fs.watch as a faster secondary trigger (best-effort)
    if (fs.existsSync(eventsFile)) {
      try {
        this.eventsWatcher = fs.watch(eventsFile, () => {
          this._readNewEvents();
        });
      } catch {
        // polling already running, ignore
      }
    }
  }

  private _readNewEvents(): void {
    if (!this.currentSessionId || !this.currentState) { return; }

    const eventsFile = path.join(
      this.traceDir,
      this.currentSessionId,
      "events.ndjson"
    );

    let content: string;
    try {
      const buf = fs.readFileSync(eventsFile, "utf8");
      content = buf.slice(this.eventsFileOffset);
      this.eventsFileOffset = buf.length;
    } catch {
      return;
    }

    const lines = content.split("\n").filter((l) => l.trim());
    for (const line of lines) {
      try {
        const event: TraceEvent = JSON.parse(line);
        this._applyEvent(event);
      } catch {
        // malformed line — skip
      }
    }
  }

  // -------------------------------------------------------------------------
  // State accumulation
  // -------------------------------------------------------------------------

  private _applyEvent(event: TraceEvent): void {
    if (!this.currentState) { return; }
    const s = this.currentState;

    s.events.push(event);
    s.estimatedCostUsd += estimateEventCost(event);

    switch (event.event_type) {
      case "tool_call": {
        s.toolCallCount++;
        const toolName = (event.data.tool_name as string) ?? "";
        s.activeTool = toolName;
        // Extract file paths from Claude Code hook tool calls (Read/Write/Edit/etc.)
        this._extractToolFilePaths(event);
        break;
      }
      case "tool_result": {
        s.activeTool = null;
        break;
      }
      case "file_read": {
        const p = this._resolveFilePath(event.data.path as string);
        if (p) {
          const acc = s.fileAccess.get(p) ?? { reads: 0, writes: 0 };
          acc.reads++;
          s.fileAccess.set(p, acc);
        }
        break;
      }
      case "file_write": {
        const p = this._resolveFilePath(event.data.path as string);
        if (p) {
          const acc = s.fileAccess.get(p) ?? { reads: 0, writes: 0 };
          acc.writes++;
          s.fileAccess.set(p, acc);
        }
        break;
      }
      case "error": {
        s.errorCount++;
        break;
      }
      case "session_end": {
        s.activeTool = null;
        break;
      }
    }

    this._onEvent.fire({ state: s, event });
    this._onStateChange.fire(s);
  }

  /**
   * Claude Code hooks record file ops as tool_call events with tool_name
   * Read/Write/Edit/Glob. Extract the file path from those.
   */
  private _extractToolFilePaths(event: TraceEvent): void {
    if (!this.currentState) { return; }
    const s = this.currentState;
    const toolName = (event.data.tool_name as string ?? "").toLowerCase();
    const args = (event.data.arguments as Record<string, unknown>) ?? {};

    const filePath = (args.path ?? args.file_path ?? args.filename) as string | undefined;
    if (!filePath) { return; }

    const resolved = this._resolveFilePath(filePath);
    if (!resolved) { return; }

    const acc = s.fileAccess.get(resolved) ?? { reads: 0, writes: 0 };
    if (["read", "view", "grep", "glob"].includes(toolName)) {
      acc.reads++;
    } else if (["write", "edit", "multiedit", "create"].includes(toolName)) {
      acc.writes++;
    }
    s.fileAccess.set(resolved, acc);
  }

  private _resolveFilePath(p: string | undefined): string | null {
    if (!p) { return null; }
    if (path.isAbsolute(p)) { return p; }
    // Relative paths — resolve against workspace root (parent of traceDir)
    const workspaceRoot = path.dirname(this.traceDir);
    return path.resolve(workspaceRoot, p);
  }

  // -------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------

  private _loadMeta(sessionId: string): SessionMeta {
    const metaFile = path.join(this.traceDir, sessionId, "meta.json");
    try {
      return JSON.parse(fs.readFileSync(metaFile, "utf8")) as SessionMeta;
    } catch {
      return {
        session_id: sessionId,
        started_at: Date.now() / 1000,
        tool_calls: 0,
        errors: 0,
        total_tokens: 0,
      };
    }
  }

  override dispose(): void {
    this.activeSessionWatcher?.close();
    this.eventsWatcher?.close();
    if (this.pollTimer) { clearInterval(this.pollTimer); }
    this._onSessionStart.dispose();
    this._onSessionEnd.dispose();
    this._onEvent.dispose();
    this._onStateChange.dispose();
  }
}
