import * as assert from "node:assert/strict";
import Module from "node:module";
import * as os from "node:os";
import { beforeEach, test } from "node:test";
import type { SessionState } from "./traceStore";

// Runs activate() with every collaborator replaced by a recorder, so the
// wiring between watcher events, settings and commands can be checked.

type Fn = (...args: unknown[]) => unknown;

const calls: Array<{ target: string; method: string; args: unknown[] }> = [];
const commands = new Map<string, Fn>();
const watcherCallbacks: Record<string, Fn> = {};
const settings: Record<string, unknown> = {};

function recorder(target: string, overrides: Record<string, Fn> = {}) {
  return class {
    static readonly viewId = "agentTrace.eventStream";
    constructor() {
      return new Proxy(this, {
        get: (_obj, prop) => {
          if (typeof prop !== "string" || prop === "then") return undefined;
          if (prop in overrides) return overrides[prop];
          return (...args: unknown[]) => {
            calls.push({ target, method: prop, args });
          };
        },
      });
    }
  };
}

const disposable = { dispose: () => undefined };

const vscodeStub = {
  TreeItem: class {
    constructor(readonly label: string, readonly collapsibleState: number) {}
  },
  TreeItemCollapsibleState: { None: 0, Collapsed: 1, Expanded: 2 },
  EventEmitter: class {
    event = () => disposable;
    fire(): void {}
    dispose(): void {}
  },
  workspace: {
    workspaceFolders: [{ uri: { fsPath: os.tmpdir() } }],
    getConfiguration: () => ({
      get: <T>(key: string, fallback: T): T =>
        (key in settings ? settings[key] : fallback) as T,
    }),
    onDidChangeConfiguration: () => disposable,
  },
  window: {
    registerWebviewViewProvider: () => disposable,
    registerTreeDataProvider: () => disposable,
    showInputBox: (...args: unknown[]) => {
      calls.push({ target: "window", method: "showInputBox", args });
      return Promise.resolve(undefined);
    },
  },
  commands: {
    registerCommand: (id: string, fn: Fn) => {
      commands.set(id, fn);
      return disposable;
    },
    executeCommand: () => Promise.resolve(undefined),
  },
  env: { onDidChangeTelemetryEnabled: () => disposable },
};

const stubs: Record<string, unknown> = {
  vscode: vscodeStub,
  "./decorations": { DecorationManager: recorder("decorations") },
  "./panel": { EventStreamPanel: recorder("panel") },
  "./pauseAgent": { PauseManager: recorder("pause") },
  "./liveStream": { LiveStreamPanel: recorder("liveStream") },
  "./postMortem": { PostMortemManager: recorder("postMortem") },
  "./statusBar": {
    StatusBarManager: recorder("statusBar"),
    WatchdogStatusBar: recorder("watchdogBar"),
  },
  "./telemetry": {
    ExtensionTelemetry: recorder("telemetry", {
      trackCommand: (_name: unknown, fn: unknown) => fn,
    }),
  },
  "./traceStore": {
    TraceWatcher: recorder("watcher", {
      onSessionStart: (cb: unknown) => (watcherCallbacks.start = cb as Fn),
      onSessionEnd: (cb: unknown) => (watcherCallbacks.end = cb as Fn),
      onStateChange: (cb: unknown) => (watcherCallbacks.change = cb as Fn),
    }),
  },
};

type Load = (request: string, parent: unknown, isMain: boolean) => unknown;
const loader = Module as unknown as { _load: Load };

async function load() {
  const originalLoad = loader._load;
  loader._load = function (this: unknown, request, parent, isMain) {
    if (request in stubs) return stubs[request];
    const loaded = originalLoad.call(this, request, parent, isMain);
    if (request === "./sessionTree") {
      // The real SessionItem and argument parsing, without the provider's
      // refresh timer.
      return { ...(loaded as object), SessionTreeProvider: recorder("sessionTree") };
    }
    return loaded;
  };
  try {
    const extension = await import("./extension");
    const sessionTree = await import("./sessionTree");
    return { activate: extension.activate, SessionItem: sessionTree.SessionItem };
  } finally {
    loader._load = originalLoad;
  }
}

async function activate() {
  const loaded = await load();
  loaded.activate({ subscriptions: [], extensionUri: { fsPath: "" } } as never);
  return loaded;
}

const state = { sessionId: "s1" } as SessionState;
const decorationUpdates = () =>
  calls.filter((c) => c.target === "decorations" && c.method === "update");

beforeEach(() => {
  calls.length = 0;
  commands.clear();
  for (const key of Object.keys(settings)) delete settings[key];
});

test("session start skips gutter annotations when they are turned off", async () => {
  settings.showGutterAnnotations = false;
  await activate();

  watcherCallbacks.start(state);
  watcherCallbacks.change(state);

  assert.deepEqual(decorationUpdates(), []);
});

test("session start applies gutter annotations by default", async () => {
  await activate();

  watcherCallbacks.start(state);

  assert.deepEqual(decorationUpdates().map((c) => c.args), [[state]]);
});

test("the inline View Post-Mortem action opens the clicked session", async () => {
  const { SessionItem } = await activate();
  const item = new SessionItem({
    info: { session_id: "abc123", started_at: 1 },
    status: "watchdog",
    costUsd: 0,
    hasPostMortem: true,
  });

  await commands.get("agentTrace.openPostMortem")!(item);

  const opened = calls.filter((c) => c.target === "postMortem" && c.method === "openForSession");
  assert.deepEqual(opened.map((c) => c.args), [["abc123"]]);
  assert.equal(calls.some((c) => c.method === "showInputBox"), false);
});

test("View Post-Mortem still accepts a session id string", async () => {
  await activate();

  await commands.get("agentTrace.openPostMortem")!("def456");

  const opened = calls.filter((c) => c.target === "postMortem" && c.method === "openForSession");
  assert.deepEqual(opened.map((c) => c.args), [["def456"]]);
});
