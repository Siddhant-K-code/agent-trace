import * as assert from "node:assert/strict";
import fs from "node:fs";
import Module from "node:module";
import * as os from "node:os";
import * as path from "node:path";
import { after, mock, test } from "node:test";
import type { SessionState } from "./traceStore";

// Loads the real WatchdogStatusBar against a fake status bar item.

class FakeItem {
  text = "";
  tooltip = "";
  backgroundColor: unknown = undefined;
  command = "";
  visible = false;
  show(): void {
    this.visible = true;
  }
  hide(): void {
    this.visible = false;
  }
  dispose(): void {}
}

const items: FakeItem[] = [];

const vscodeStub = {
  Disposable: class {
    constructor(_dispose: () => void) {}
  },
  ThemeColor: class {
    constructor(readonly id: string) {}
  },
  StatusBarAlignment: { Left: 1, Right: 2 },
  window: {
    createStatusBarItem: () => {
      const item = new FakeItem();
      items.push(item);
      return item;
    },
  },
  workspace: {
    getConfiguration: () => ({ get: <T>(_key: string, fallback: T): T => fallback }),
  },
};

type Load = (request: string, parent: unknown, isMain: boolean) => unknown;
const loader = Module as unknown as { _load: Load };

async function loadWatchdogBar() {
  const originalLoad = loader._load;
  loader._load = function (this: unknown, request, parent, isMain) {
    if (request === "vscode") return vscodeStub;
    return originalLoad.call(this, request, parent, isMain);
  };
  try {
    return (await import("./statusBar")).WatchdogStatusBar;
  } finally {
    loader._load = originalLoad;
  }
}

const tempDirs: string[] = [];
after(() => {
  for (const dir of tempDirs) fs.rmSync(dir, { recursive: true, force: true });
});

function traceDir(sessions: Record<string, Record<string, unknown>>): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "agent-trace-watchdog-"));
  tempDirs.push(dir);
  for (const [id, meta] of Object.entries(sessions)) {
    fs.mkdirSync(path.join(dir, id));
    fs.writeFileSync(path.join(dir, id, "meta.json"), JSON.stringify(meta));
  }
  return dir;
}

function state(sessionId: string): SessionState {
  return {
    sessionId,
    meta: { session_id: sessionId, started_at: Date.now() / 1000 },
    estimatedCostUsd: 1,
  } as unknown as SessionState;
}

test("a direct switch to a session without watchdog limits hides the bar", async () => {
  const WatchdogStatusBar = await loadWatchdogBar();
  const dir = traceDir({ a: { watchdog_budget_dollars: 5 }, b: {} });
  const bar = new WatchdogStatusBar(60_000);
  const item = items.at(-1)!;
  try {
    bar.onSessionStart(dir, state("a"));
    assert.equal(item.visible, true);
    assert.match(item.text, /\$5\.00/);

    bar.onSessionStart(dir, state("b"));

    assert.equal(item.visible, false);
  } finally {
    bar.dispose();
  }
});

test("polling shows the post-mortem when fs.watch is unavailable", async (t) => {
  const WatchdogStatusBar = await loadWatchdogBar();
  const dir = traceDir({ a: { watchdog_budget_dollars: 5 } });
  t.after(() => mock.restoreAll());
  mock.method(fs, "watch", () => {
    throw new Error("fs.watch unavailable");
  });
  const bar = new WatchdogStatusBar(10);
  const item = items.at(-1)!;
  try {
    bar.onSessionStart(dir, state("a"));
    assert.doesNotMatch(item.text, /post-mortem/);

    fs.writeFileSync(
      path.join(dir, "a", "watchdog-postmortem.json"),
      JSON.stringify({ reason: "budget_exceeded" }),
    );
    const deadline = Date.now() + 2_000;
    while (!item.text.includes("post-mortem") && Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, 10));
    }

    assert.match(item.text, /budget exceeded .* post-mortem ready/);
  } finally {
    bar.dispose();
  }
});
