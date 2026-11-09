/**
 * Pause / resume the active agent session.
 *
 * Uses a marker file (.agent-traces/.pause-request) that watch.py polls.
 * The marker holds the session ID it applies to. Writing it requests a
 * pause; deleting it requests a resume. watch.py sends SIGSTOP / SIGCONT
 * to the agent PID on its next poll.
 *
 * watch.py only resumes a pause that the marker started. A pause held by a
 * require_approval rule stays in place until the approval is decided.
 */

import * as fs from "fs";
import * as path from "path";
import type { SessionState } from "./traceStore";

/** watch.py rewrites the session heartbeat every 5 seconds while it runs. */
const HEARTBEAT_FRESH_MS = 15_000;

export interface PauseResult {
  ok: boolean;
  /** Why the request failed (ok=false) or a caveat the user should see. */
  message?: string;
}

function errorText(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function isMissing(err: unknown): boolean {
  return (err as NodeJS.ErrnoException | undefined)?.code === "ENOENT";
}

export class PauseManager {
  constructor(
    private readonly traceDir: string,
    private readonly now: () => number = Date.now,
  ) {}

  private get pauseFile(): string {
    return path.join(this.traceDir, ".pause-request");
  }

  /** Session ID in the marker, "" for an empty marker, null when absent. */
  private readMarker(): string | null {
    try {
      return fs.readFileSync(this.pauseFile, "utf8").trim();
    } catch (err) {
      if (isMissing(err)) { return null; }
      throw err;
    }
  }

  /** True when `agent-strace watch` has written a recent heartbeat for the session. */
  isWatcherRunning(sessionId: string): boolean {
    try {
      const raw = fs.readFileSync(
        path.join(this.traceDir, sessionId, ".heartbeat"),
        "utf8",
      );
      const updatedAt = Number(JSON.parse(raw).updated_at);
      return Number.isFinite(updatedAt) &&
        this.now() - updatedAt * 1000 < HEARTBEAT_FRESH_MS;
    } catch {
      return false;
    }
  }

  /** IDs of pending approval requests for the session. */
  pendingApprovals(sessionId: string): string[] {
    const dir = path.join(this.traceDir, ".approvals");
    let names: string[];
    try {
      names = fs.readdirSync(dir).filter((name) => name.endsWith(".json"));
    } catch {
      return [];
    }
    const ids: string[] = [];
    for (const name of names) {
      try {
        const req = JSON.parse(fs.readFileSync(path.join(dir, name), "utf8"));
        if (req.session_id === sessionId && req.state === "pending") {
          ids.push(String(req.request_id));
        }
      } catch {
        // Skip unreadable or half-written requests.
      }
    }
    return ids;
  }

  /** Align state.paused with the marker, for example after a window reload. */
  syncState(state: SessionState): void {
    try {
      state.paused = this.readMarker() === state.sessionId;
    } catch {
      // Keep the current state if the marker cannot be read.
    }
  }

  pause(state: SessionState): PauseResult {
    if (state.paused) { return { ok: true }; }
    if (!this.isWatcherRunning(state.sessionId)) {
      return {
        ok: false,
        message: "agent-strace watch is not running for this session. Start it, then pause again.",
      };
    }
    try {
      const owner = this.readMarker();
      if (owner && owner !== state.sessionId && this.isWatcherRunning(owner)) {
        return {
          ok: false,
          message: `session ${owner.slice(0, 12)} is paused from the editor. Resume it first.`,
        };
      }
      fs.mkdirSync(this.traceDir, { recursive: true });
      fs.writeFileSync(this.pauseFile, state.sessionId, "utf8");
    } catch (err) {
      return { ok: false, message: `could not write pause request: ${errorText(err)}` };
    }
    state.paused = true;
    return { ok: true };
  }

  resume(state: SessionState): PauseResult {
    try {
      const owner = this.readMarker();
      // Never remove a marker that pauses another session.
      if (owner === state.sessionId || owner === "") {
        fs.unlinkSync(this.pauseFile);
      }
    } catch (err) {
      if (!isMissing(err)) {
        return { ok: false, message: `could not remove pause request: ${errorText(err)}` };
      }
    }
    state.paused = false;
    const pending = this.pendingApprovals(state.sessionId);
    if (pending.length > 0) {
      return {
        ok: true,
        message: `approval ${pending[0]} is pending, so the agent stays paused until it is decided. ` +
          `Run: agent-strace approval approve ${pending[0]}`,
      };
    }
    return { ok: true };
  }

  /** Remove the marker if it belongs to the session that ended. */
  cleanup(sessionId: string): void {
    try {
      if (this.readMarker() === sessionId) {
        fs.unlinkSync(this.pauseFile);
      }
    } catch {
      // ignore
    }
  }
}
