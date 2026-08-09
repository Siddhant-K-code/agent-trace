import * as assert from "node:assert/strict";
import { test } from "node:test";
import {
  COMMAND_COMPLETED,
  SESSION_COMPLETED,
  sanitiseEventProperties,
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
