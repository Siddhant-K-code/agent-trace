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
};

const vscodeStub = {};

const reviewModelStub = {
  buildReviewModel: (meta: { session_id: string }) => ({
    session: { session_id: meta.session_id },
  }),
  loadStoredReview: (_dir: string, sessionId: string) => ({
    session: { session_id: sessionId },
  }),
  loadAnnotations: () => [],
  renderMinimizedReviewHtml: () => "",
};

const childProcessStub = {
  execFile: (_cmd: string, _args: string[], _opts: unknown, callback: Callback) => {
    if (fake.holdHealth) {
      fake.pendingHealth.push(callback);
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
  const view = {
    webview: {
      options: {},
      html: "",
      onDidReceiveMessage: () => undefined,
      postMessage: (message: Posted) => {
        messages.push(message);
        return Promise.resolve(true);
      },
    },
  };
  panel.resolveWebviewView(view as never, {} as never, {} as never);
  return { panel, messages };
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

beforeEach(() => {
  fake.holdHealth = false;
  fake.pendingHealth = [];
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
