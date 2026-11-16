import * as assert from "node:assert/strict";
import Module from "node:module";
import { afterEach, beforeEach, test } from "node:test";
import { EXTENSION_ACTIVATED } from "./telemetryCore";

// Load the real telemetry module against stand-ins for the editor API and
// the network, so these tests can never send a real event.
const fake = {
  settingGlobalValue: undefined as boolean | undefined,
  requests: 0,
};

const vscodeStub = {
  env: { appName: "Visual Studio Code", isTelemetryEnabled: true, uiKind: 1 },
  UIKind: { Desktop: 1, Web: 2 },
  version: "1.99.0",
  workspace: {
    getConfiguration: () => ({
      get: <T>(_key: string, fallback: T): T =>
        (fake.settingGlobalValue ?? fallback) as T,
      inspect: () => ({ globalValue: fake.settingGlobalValue }),
    }),
  },
};

const httpsStub = {
  request: () => {
    fake.requests += 1;
    const noop = (): undefined => undefined;
    return { on: noop, setTimeout: noop, end: noop, destroy: noop };
  },
};

type Load = (request: string, parent: unknown, isMain: boolean) => unknown;
const loader = Module as unknown as { _load: Load };

async function loadTelemetry() {
  const originalLoad = loader._load;
  loader._load = function (this: unknown, request, parent, isMain) {
    if (request === "vscode") {
      return vscodeStub;
    }
    if (request === "https") {
      return httpsStub;
    }
    return originalLoad.call(this, request, parent, isMain);
  };
  try {
    return (await import("./telemetry")).ExtensionTelemetry;
  } finally {
    loader._load = originalLoad;
  }
}

function fakeContext(noticeShown: boolean) {
  const values = new Map<string, unknown>([
    ["agentTrace.telemetry.noticeShown", noticeShown],
  ]);
  return {
    values,
    globalState: {
      get: (key: string, fallback?: unknown) =>
        values.has(key) ? values.get(key) : fallback,
      update: async (key: string, value: unknown) => {
        values.set(key, value);
      },
    },
    extension: { packageJSON: { version: "0.0.0" } },
  };
}

async function captureOnce(noticeShown: boolean) {
  const ExtensionTelemetry = await loadTelemetry();
  const context = fakeContext(noticeShown);
  const telemetry = new ExtensionTelemetry(
    context as unknown as ConstructorParameters<typeof ExtensionTelemetry>[0],
  );
  telemetry.capture(EXTENSION_ACTIVATED);
  return context.values;
}

const ENV_NAMES = ["AGENT_STRACE_TELEMETRY", "DO_NOT_TRACK"] as const;
let savedEnv: Record<string, string | undefined> = {};

beforeEach(() => {
  savedEnv = Object.fromEntries(ENV_NAMES.map((name) => [name, process.env[name]]));
  for (const name of ENV_NAMES) {
    delete process.env[name];
  }
  fake.settingGlobalValue = undefined;
  fake.requests = 0;
});

afterEach(() => {
  for (const name of ENV_NAMES) {
    if (savedEnv[name] === undefined) {
      delete process.env[name];
    } else {
      process.env[name] = savedEnv[name];
    }
  }
});

test("sends nothing and creates no ID before the notice or an opt-in", async () => {
  const stored = await captureOnce(false);
  assert.equal(fake.requests, 0);
  assert.equal(stored.has("agentTrace.telemetry.anonymousId"), false);
});

test("sends once the notice was shown", async () => {
  await captureOnce(true);
  assert.equal(fake.requests, 1);
});

test("sends after the setting is explicitly enabled", async () => {
  fake.settingGlobalValue = true;
  await captureOnce(false);
  assert.equal(fake.requests, 1);
});

test("sends with AGENT_STRACE_TELEMETRY=1", async () => {
  process.env.AGENT_STRACE_TELEMETRY = "1";
  await captureOnce(false);
  assert.equal(fake.requests, 1);
});
