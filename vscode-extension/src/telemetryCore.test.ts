import * as assert from "node:assert/strict";
import { test } from "node:test";
import {
  AnalyticsSession,
  COMMAND_COMPLETED,
  SESSION_COMPLETED,
  SESSION_IDLE_MS,
  SESSION_MAX_MS,
  UUIDV7_MAX_MS,
  nextAnalyticsSession,
  sanitiseEventProperties,
  uuidv7,
} from "./telemetryCore";

test("drops unknown and sensitive command properties", () => {
  const clean = sanitiseEventProperties(COMMAND_COMPLETED, {
    command: "open_panel",
    success: true,
    duration_ms: 12.9,
    workspace_path: "/private/repository",
    session_id: "secret-session",
    arguments: "--token secret",
  });

  assert.deepEqual(clean, {
    command: "open_panel",
    success: true,
    duration_ms: 12,
  });
});

test("rejects unknown events and command names", () => {
  assert.equal(sanitiseEventProperties("unknown", {}), null);
  assert.deepEqual(
    sanitiseEventProperties(COMMAND_COMPLETED, {
      command: "open /private/path",
      success: true,
    }),
    { success: true },
  );
});

test("validates and bounds aggregate session metrics", () => {
  assert.deepEqual(
    sanitiseEventProperties(SESSION_COMPLETED, {
      success: false,
      duration_ms: -20,
      tool_call_count: 1_500_000,
      error_count: 3,
      prompt: "never send me",
    }),
    {
      success: false,
      duration_ms: 0,
      tool_call_count: 1_000_000,
      error_count: 3,
    },
  );
});

const UUID7 = /^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

test("builds a UUIDv7 that starts at the given time", () => {
  const now = Date.UTC(2026, 4, 1, 12, 30);
  const id = uuidv7(now, new Uint8Array(10).fill(0xff));

  assert.match(id, UUID7);
  assert.equal(parseInt(id.slice(0, 8) + id.slice(9, 13), 16), now);
  assert.match(uuidv7(now, new Uint8Array(10)), UUID7);
});

test("clamps UUIDv7 times outside the 48-bit range", () => {
  const start = (id: string) => parseInt(id.slice(0, 8) + id.slice(9, 13), 16);
  const random = new Uint8Array(10);
  assert.equal(start(uuidv7(-1, random)), 0);
  assert.equal(start(uuidv7(2 ** 48, random)), UUIDV7_MAX_MS);
  assert.match(uuidv7(2 ** 48, random), UUID7);
});

test("keeps a session until 30 idle minutes or 24 hours pass", () => {
  let counter = 0;
  const newId = (startedAt: number) => `${startedAt}-${counter++}`;
  const start = 1_800_000_000_000;
  let session: AnalyticsSession | undefined;
  const ids = new Set<string>();
  for (let now = start; now <= start + SESSION_MAX_MS; now += 20 * 60 * 1000) {
    session = nextAnalyticsSession(session, now, newId);
    assert.ok(now - session.startedAt < SESSION_MAX_MS);
    ids.add(session.id);
  }
  assert.equal(ids.size, 2);

  const current = nextAnalyticsSession(undefined, start, newId);
  const idle = nextAnalyticsSession(current, start + SESSION_IDLE_MS, newId);
  assert.notEqual(idle.id, current.id);
  assert.equal(idle.startedAt, start + SESSION_IDLE_MS);

  const earlier = nextAnalyticsSession(current, start - 1, newId);
  assert.notEqual(earlier.id, current.id);
  assert.equal(earlier.startedAt, start - 1);

  const later = nextAnalyticsSession(current, start + 60_000, newId);
  assert.equal(later.id, current.id);
  const movedBack = nextAnalyticsSession(later, start + 30_000, newId);
  assert.notEqual(movedBack.id, current.id);
  assert.equal(movedBack.startedAt, start + 30_000);

  const future = { ...current, lastEventAt: start + 10 * 60_000 };
  assert.notEqual(nextAnalyticsSession(future, start + 60_000, newId).id, current.id);
});
