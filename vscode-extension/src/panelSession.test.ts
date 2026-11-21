import * as assert from "node:assert/strict";
import Module from "node:module";
import * as path from "node:path";
import { afterEach, beforeEach, mock, test } from "node:test";
import type { SessionState } from "./traceStore";

// Drives EventStreamPanel against a fake webview. The CLI health call and
// the review builders are stand-ins, so only the panel's own state handling
// is under test.

type Callback = (error: Error | null, stdout: string, stderr: string) => void;

const fake = {
  holdHealth: false,
  pendingHealth: [] as Callback[],
  pendingArgs: [] as string[][],
  errorMessages: [] as string[],
  warnings: [] as string[],
  quickPick: null as null | ((choice: { id: string }) => void),
  saveDialog: null as null | ((target: { fsPath: string }) => void),
  saveDefault: "",
  written: [] as string[],
};

// A dialog the test answers later, so it can switch sessions while it is open.
function held<T>(store: (resolve: (value: T) => void) => void): Promise<T> {
  return new Promise<T>((resolve) => store(resolve));
}

const vscodeStub = {
  Uri: { file: (fsPath: string) => ({ fsPath }) },
  window: {
    showQuickPick: () => held<{ id: string }>((r) => { fake.quickPick = r; }),
    showSaveDialog: (options: { defaultUri: { fsPath: string } }) => {
      fake.saveDefault = options.defaultUri.fsPath;
      return held<{ fsPath: string }>((r) => { fake.saveDialog = r; });
    },
    showErrorMessage: (message: string) => {
      fake.errorMessages.push(message);
    },
    showInformationMessage: () => undefined,
    showWarningMessage: (message: string) => {
      fake.warnings.push(message);
    },
  },
  workspace: {
    fs: {
      writeFile: async (_target: unknown, data: Buffer) => {
        fake.written.push(data.toString("utf8"));
      },
    },
  },
};

interface Health {
  status?: string;
}

const reviewModelStub = {
  buildReviewModel: (
    meta: { session_id: string },
    events: unknown[],
    _annotations: unknown,
    health?: Health,
  ) => ({
    session: { session_id: meta.session_id, provider: "claude" },
    events: events.length,
    health: health?.status,
  }),
  loadStoredReview: (_dir: string, sessionId: string, health?: Health) => {
    if (sessionId.startsWith("broken")) {
      throw new Error("Session contains an unsupported event type.");
    }
    return {
      session: { session_id: sessionId, provider: "claude" },
      health: health?.status,
    };
  },
  loadAnnotations: () => [],
  renderMinimizedReviewHtml: (model: { session: { session_id: string } }) =>
    model.session.session_id,
};

const childProcessStub = {
  execFile: (_cmd: string, args: string[], _opts: unknown, callback: Callback) => {
    if (fake.holdHealth) {
      fake.pendingHealth.push(callback);
      fake.pendingArgs.push(args);
    } else {
      callback(null, "{}", "");
    }
  },
};

type Load = (request: string, parent: unknown, isMain: boolean) => unknown;
const loader = Module as unknown as { _load: Load };

async function loadPanelClass() {
  const originalLoad = loader._load;
  loader._load = function (this: unknown, request, parent, isMain) {
    if (request === "vscode") return vscodeStub;
    if (request === "./reviewModel") return reviewModelStub;
    if (request === "child_process") return childProcessStub;
    return originalLoad.call(this, request, parent, isMain);
  };
  try {
    return (await import("./panel")).EventStreamPanel;
  } finally {
    loader._load = originalLoad;
  }
}

interface Posted {
  type: string;
  model?: { session: { session_id: string }; events?: number; health?: string };
  controls?: { live: boolean; paused: boolean };
}

async function openPanel() {
  const EventStreamPanel = await loadPanelClass();
  const panel = new EventStreamPanel(
    { fsPath: path.resolve(__dirname, "..") } as never,
    "/unused-trace-dir",
  );
  const messages: Posted[] = [];
  let onMessage: (message: unknown) => void = () => undefined;
  const view = {
    webview: {
      options: {},
      html: "",
      onDidReceiveMessage: (handler: (message: unknown) => void) => {
        onMessage = handler;
      },
      postMessage: (message: Posted) => {
        messages.push(message);
        return Promise.resolve(true);
      },
    },
  };
  panel.resolveWebviewView(view as never, {} as never, {} as never);
  const send = (message: unknown) => onMessage(message);
  return { panel, messages, send };
}

function liveState(sessionId: string, paused = false, events = 0): SessionState {
  return {
    sessionId,
    meta: { session_id: sessionId },
    events: Array.from({ length: events }, () => ({})),
    paused,
    activeTool: null,
    estimatedCostUsd: 0.5,
  } as unknown as SessionState;
}

const reviews = (messages: Posted[]) => messages.filter((m) => m.type === "review");

// Lets queued promise callbacks run, so a held CLI call has been issued.
const settle = () => new Promise((resolve) => setImmediate(resolve));

// Finishes the held CLI call whose arguments include `marker`: the oldest
// such call, or the newest with `newest`.
function finish(marker: string, stdout: string, stderr = "", newest = false) {
  const matches = fake.pendingArgs
    .map((args, index) => (args.includes(marker) ? index : -1))
    .filter((index) => index >= 0);
  const index = (newest ? matches.at(-1) : matches[0]) ?? -1;
  assert.ok(index >= 0, `no pending CLI call for ${marker}`);
  const [callback] = fake.pendingHealth.splice(index, 1);
  fake.pendingArgs.splice(index, 1);
  callback(stdout ? null : new Error("failed"), stdout, stderr);
}

afterEach(() => {
  mock.timers.reset();
});

beforeEach(() => {
  fake.holdHealth = false;
  fake.pendingHealth = [];
  fake.pendingArgs = [];
  fake.errorMessages = [];
  fake.warnings = [];
  fake.quickPick = null;
  fake.saveDialog = null;
  fake.saveDefault = "";
  fake.written = [];
});

// Starts a comparison for the shown session and picks the other session.
async function startCompare(send: (message: unknown) => void) {
  fake.holdHealth = true;
  send({ command: "compare" });
  await settle();
  fake.quickPick?.({ id: "other" });
  await settle();
}

test("picking the live session in the browser keeps the live controls", async () => {
  const { panel, messages } = await openPanel();
  panel.onSessionStart(liveState("live", true));
  await panel.showSession("old");
  messages.length = 0;

  await panel.showSession("live");

  const posted = reviews(messages);
  assert.ok(posted.length > 0);
  for (const review of posted) {
    assert.equal(review.model?.session.session_id, "live");
    assert.deepEqual(
      { live: review.controls?.live, paused: review.controls?.paused },
      { live: true, paused: true },
    );
  }
});

test("re-picking the session already shown keeps its live controls", async () => {
  const { panel, messages } = await openPanel();
  panel.onSessionStart(liveState("live"));
  messages.length = 0;

  await panel.showSession("live");

  assert.equal(reviews(messages).at(-1)?.controls?.live, true);
});

test("a session that goes live while its stored review loads ends live", async () => {
  const { panel, messages } = await openPanel();
  fake.holdHealth = true;
  const shown = panel.showSession("s1");
  panel.onSessionStart(liveState("s1"));
  // The stored review's health load finishes after the live refresh.
  for (const callback of fake.pendingHealth.reverse()) callback(null, "{}", "");
  await shown;

  assert.equal(reviews(messages).at(-1)?.controls?.live, true);
});

test("switching sessions resets the webview before the new review", async () => {
  const { panel, messages } = await openPanel();
  panel.onSessionStart(liveState("live"));
  messages.length = 0;

  await panel.showSession("old");

  assert.equal(messages[0]?.type, "reset");
  assert.equal(reviews(messages)[0]?.model?.session.session_id, "old");
});

test("showing the selected session again does not reset the webview", async () => {
  const { panel, messages } = await openPanel();
  await panel.showSession("old");
  messages.length = 0;

  await panel.showSession("old");

  assert.equal(messages.some((m) => m.type === "reset"), false);
});

test("a health error for a session no longer selected is not shown", async () => {
  const { panel, messages } = await openPanel();
  fake.holdHealth = true;
  const shownA = panel.showSession("a");
  const shownB = panel.showSession("b");
  finish("b", "{}");
  await shownB;
  finish("a", "", "a lookup failed");
  await shownA;

  assert.equal(messages.some((m) => m.type === "error"), false);
  assert.equal(reviews(messages).at(-1)?.model?.session.session_id, "b");
});

test("a health error for the selected stored session is shown", async () => {
  const { panel, messages } = await openPanel();
  fake.holdHealth = true;
  const shown = panel.showSession("a");
  finish("a", "", "a lookup failed");
  await shown;

  assert.equal(messages.at(-1)?.type, "error");
});

test("a stored health error is not shown once the session went live", async () => {
  const { panel, messages } = await openPanel();
  fake.holdHealth = true;
  const shown = panel.showSession("s1");
  panel.onSessionStart(liveState("s1"));
  finish("s1", "", "lookup failed");
  await shown;

  assert.equal(messages.some((m) => m.type === "error"), false);
});

test("a comparison that finishes after a session switch is not shown", async () => {
  const { panel, messages, send } = await openPanel();
  await panel.showSession("a");
  await startCompare(send);
  const shownB = panel.showSession("b");
  finish("compare", "{}");
  await settle();
  finish("b", "{}");
  await shownB;

  assert.equal(messages.some((m) => m.type === "comparison"), false);
});

test("a live capture check that finishes after a session switch is not shown", async () => {
  const { panel, messages, send } = await openPanel();
  await panel.showSession("a");
  fake.holdHealth = true;
  send({ command: "verifyLiveCapture" });
  await settle();
  const shownB = panel.showSession("b");
  finish("--verify", "ok");
  await settle();
  finish("b", "{}");
  await shownB;

  assert.equal(messages.some((m) => m.type === "liveCapture"), false);
});

test("comparison and live capture results for the selected session are shown", async () => {
  const { panel, messages, send } = await openPanel();
  await panel.showSession("a");
  await startCompare(send);
  finish("compare", "{}");
  send({ command: "verifyLiveCapture" });
  await settle();
  finish("--verify", "ok");
  await settle();

  assert.ok(messages.some((m) => m.type === "comparison"));
  assert.ok(messages.some((m) => m.type === "liveCapture"));
});

test("an older health error for a session shown again is not shown", async () => {
  const { panel, messages } = await openPanel();
  fake.holdHealth = true;
  const first = panel.showSession("a");
  const toB = panel.showSession("b");
  const again = panel.showSession("a");
  finish("b", "{}");
  await toB;
  finish("a", "{}", "", true);
  await again;
  finish("a", "", "old lookup failed");
  await first;

  assert.equal(messages.some((m) => m.type === "error"), false);
  assert.equal(reviews(messages).at(-1)?.model?.session.session_id, "a");
});

test("an older health result does not replace a newer error for that session", async () => {
  const { panel, messages } = await openPanel();
  fake.holdHealth = true;
  const first = panel.showSession("a");
  const toB = panel.showSession("b");
  const again = panel.showSession("a");
  finish("b", "{}");
  await toB;
  finish("a", "", "new lookup failed", true);
  await again;
  finish("a", "{}");
  await first;

  assert.equal(messages.at(-1)?.type, "error");
});

for (const [name, stdout, stderr] of [
  ["fails", "", "compare failed"],
  ["returns invalid JSON", "not json", ""],
]) {
  test(`a comparison that ${name} after a session switch shows no error`, async () => {
    const { panel, send } = await openPanel();
    await panel.showSession("a");
    await startCompare(send);
    const shownB = panel.showSession("b");
    finish("compare", stdout, stderr);
    await settle();
    finish("b", "{}");
    await shownB;

    assert.deepEqual(fake.errorMessages, []);
  });
}

test("a comparison that fails for the selected session shows its error", async () => {
  const { panel, send } = await openPanel();
  await panel.showSession("a");
  await startCompare(send);
  finish("compare", "", "compare failed");
  await settle();

  assert.equal(fake.errorMessages.length, 1);
});

test("a comparison picked after a session switch does not run", async () => {
  const { panel, send } = await openPanel();
  await panel.showSession("a");
  send({ command: "compare" });
  await settle();
  await panel.showSession("b");
  fake.holdHealth = true;
  fake.quickPick?.({ id: "other" });
  await settle();

  assert.equal(fake.pendingArgs.some((args) => args.includes("compare")), false);
});

test("a comparison from an earlier view of the same session is not shown", async () => {
  const { panel, messages, send } = await openPanel();
  await panel.showSession("a");
  await startCompare(send);
  const toB = panel.showSession("b");
  const again = panel.showSession("a");
  finish("compare", "{}");
  await settle();
  finish("b", "{}");
  finish("a", "{}");
  await Promise.all([toB, again]);

  assert.equal(messages.some((m) => m.type === "comparison"), false);
});

test("a minimized share saves the review shown when it was requested", async () => {
  const { panel, send } = await openPanel();
  await panel.showSession("a");
  send({ command: "saveMinimizedShare" });
  await settle();
  await panel.showSession("b");
  fake.saveDialog?.({ fsPath: "/tmp/share.html" });
  await settle();

  assert.match(path.basename(fake.saveDefault), /^review-a-/);
  assert.deepEqual(fake.written, ["a"]);
});

const healthCalls = () => fake.pendingArgs.filter((args) => args.includes("health"));

test("live health refreshes run one at a time and the last one wins", async () => {
  mock.timers.enable({ apis: ["setTimeout"] });
  const { panel, messages } = await openPanel();
  fake.holdHealth = true;
  panel.onSessionStart(liveState("live", false, 1));
  panel.onStateChange(liveState("live", false, 3));
  mock.timers.tick(400);
  panel.onStateChange(liveState("live", false, 4));
  mock.timers.tick(400);
  await settle();

  assert.equal(healthCalls().length, 1);

  finish("health", JSON.stringify({ status: "health-for-1-event" }));
  await settle();
  // The trace grew while the lookup ran, so it runs once more.
  assert.equal(healthCalls().length, 1);
  finish("health", JSON.stringify({ status: "health-for-4-events" }));
  await settle();

  assert.equal(healthCalls().length, 0);
  const last = reviews(messages).at(-1);
  assert.deepEqual(
    { events: last?.model?.events, health: last?.model?.health },
    { events: 4, health: "health-for-4-events" },
  );
});

test("an older stored health result does not replace a newer live one", async () => {
  const { panel, messages } = await openPanel();
  fake.holdHealth = true;
  const shown = panel.showSession("s1");
  panel.onSessionStart(liveState("s1", false, 2));
  finish("s1", JSON.stringify({ status: "new" }), "", true);
  await settle();
  finish("s1", JSON.stringify({ status: "old" }));
  await shown;

  assert.equal(reviews(messages).at(-1)?.model?.health, "new");
  panel.onStateChange(liveState("s1", false, 3));
  assert.equal(reviews(messages).at(-1)?.model?.health, "new");
});

async function showBrokenAfterGood() {
  const opened = await openPanel();
  await opened.panel.showSession("good");
  opened.messages.length = 0;
  await opened.panel.showSession("broken");
  return opened;
}

test("a session that fails to load resets the view and shows the error", async () => {
  const { messages } = await showBrokenAfterGood();

  assert.deepEqual(messages.map((m) => m.type), ["reset", "error"]);
});

for (const command of ["compare", "saveMinimizedShare", "exportOtlp", "verifyLiveCapture"]) {
  test(`${command} refuses after the selected session fails to load`, async () => {
    const { messages, send } = await showBrokenAfterGood();
    fake.holdHealth = true;
    send({ command });
    await settle();

    assert.equal(fake.quickPick, null);
    assert.equal(fake.saveDialog, null);
    assert.deepEqual(fake.pendingArgs, []);
    assert.equal(messages.some((m) => m.type === "liveCapture"), false);
    assert.equal(fake.warnings.length, 1);
  });
}

test("a session shown again that now fails to load drops its old review", async () => {
  const { panel, messages, send } = await openPanel();
  await panel.showSession("flaky");
  const original = reviewModelStub.loadStoredReview;
  reviewModelStub.loadStoredReview = () => {
    throw new Error("unreadable");
  };
  try {
    messages.length = 0;
    await panel.showSession("flaky");
  } finally {
    reviewModelStub.loadStoredReview = original;
  }
  send({ command: "exportOtlp" });
  await settle();

  assert.deepEqual(messages.map((m) => m.type), ["reset", "error"]);
  assert.equal(fake.saveDialog, null);
});
