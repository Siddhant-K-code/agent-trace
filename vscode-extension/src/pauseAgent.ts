/**
 * Pause / resume the active agent session.
 *
 * Uses a signal file (.agent-traces/.pause-request) that watch.py polls.
 * Writing the file requests a pause; deleting it requests a resume.
 * This avoids needing a daemon or direct process access from the extension.
 *
 * watch.py checks for this file on every event loop iteration and calls
 * SIGSTOP / SIGCONT on the agent PID accordingly.
 */

import * as fs from "fs";
import * as path from "path";
import { SessionState } from "./traceStore";

export class PauseManager {
  private traceDir: string;

  constructor(traceDir: string) {
    this.traceDir = traceDir;
  }

  private get pauseFile(): string {
    return path.join(this.traceDir, ".pause-request");
  }

  pause(state: SessionState): void {
    if (state.paused) { return; }
    try {
      fs.mkdirSync(this.traceDir, { recursive: true });
      fs.writeFileSync(this.pauseFile, state.sessionId, "utf8");
      state.paused = true;
    } catch {
      // Non-fatal — watch.py may not be running
    }
  }

  resume(state: SessionState): void {
    if (!state.paused) { return; }
    try {
      if (fs.existsSync(this.pauseFile)) {
        fs.unlinkSync(this.pauseFile);
      }
      state.paused = false;
    } catch {
      // Non-fatal
    }
  }

  /** Clean up signal file on extension deactivation. */
  cleanup(): void {
    try {
      if (fs.existsSync(this.pauseFile)) {
        fs.unlinkSync(this.pauseFile);
      }
    } catch {
      // ignore
    }
  }
}
