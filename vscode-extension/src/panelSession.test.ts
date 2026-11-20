import * as assert from "node:assert/strict";
import Module from "node:module";
import * as path from "node:path";
import { beforeEach, test } from "node:test";
import type { SessionState } from "./traceStore";

// Drives EventStreamPanel against a fake webview. The CLI health call and
// the review builders are stand-ins, so only the panel's own state handling
// is under test.

type Callback = (error: Error | null, stdout: string, stderr: string) => void;

const fake = {
  holdHealth: false,
  pendingHealth: [] as Callback[],
  pendingArgs: [] as string[][],
};

const vscodeStub = {
  window: {
    showQuickPick: async () => ({ id: "other" }),
    showErrorMessage: () => undefined,
  },
};

const reviewModelStub = {
  buildReviewModel: (meta: { session_id: string }) => ({
    session: { session_id: meta.session_id, provider: "claude" },
  }),
  loadStoredReview: (_dir: string, sessionId: string) => ({
    session: { session_id: sessionId, provider: "claude" },
  }),
  loadAnnotations: () => [],
  renderMinimizedReviewHtml: () => "",
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
  model?: { session: { session_id: string } };
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

function liveState(sessionId: string, paused = false): SessionState {
  return {
    sessionId,
    meta: { session_id: sessionId },
    events: [],
    paused,
    activeTool: null,
    estimatedCostUsd: 0.5,
  } as unknown as SessionState;
}

const reviews = (messages: Posted[]) => messages.filter((m) => m.type === "review");

// Lets queued promise callbacks run, so a held CLI call has been issued.
const settle = () => new Promise((resolve) => setImmediate(resolve));

// Finishes the held CLI call whose arguments include `marker`.
function finish(marker: string, stdout: string, stderr = "") {
  const index = fake.pendingArgs.findIndex((args) => args.includes(marker));
  assert.ok(index >= 0, `no pending CLI call for ${marker}`);
  const [callback] = fake.pendingHealth.splice(index, 1);
  fake.pendingArgs.splice(index, 1);
  callback(stdout ? null : new Error("failed"), stdout, stderr);
}

beforeEach(() => {
  fake.holdHealth = false;
  fake.pendingHealth = [];
  fake.pendingArgs = [];
});

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
  fake.holdHealth = true;
  send({ command: "compare" });
  await settle();
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
  fake.holdHealth = true;
  send({ command: "compare" });
  await settle();
  finish("compare", "{}");
  send({ command: "verifyLiveCapture" });
  await settle();
  finish("--verify", "ok");
  await settle();

  assert.ok(messages.some((m) => m.type === "comparison"));
  assert.ok(messages.some((m) => m.type === "liveCapture"));
});
