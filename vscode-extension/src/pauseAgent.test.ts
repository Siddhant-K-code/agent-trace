import * as assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { after, test } from "node:test";
import { PauseManager } from "./pauseAgent";
import type { SessionState } from "./traceStore";

const NOW = 1_700_000_000_000;
const tempDirs: string[] = [];

after(() => {
  for (const dir of tempDirs) {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

function setup(): { dir: string; manager: PauseManager; marker: string } {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "agent-trace-pause-"));
  tempDirs.push(dir);
  return {
    dir,
    manager: new PauseManager(dir, () => NOW),
    marker: path.join(dir, ".pause-request"),
  };
}

function session(id: string, paused = false): SessionState {
  return { sessionId: id, paused } as SessionState;
}

function heartbeat(dir: string, id: string, ageMs = 0): void {
  fs.mkdirSync(path.join(dir, id), { recursive: true });
  fs.writeFileSync(
    path.join(dir, id, ".heartbeat"),
    JSON.stringify({ session_id: id, updated_at: (NOW - ageMs) / 1000 }),
  );
}

test("pause without a running watcher fails and leaves state unpaused", () => {
  const { dir, manager, marker } = setup();
  heartbeat(dir, "s1", 60_000);
  const state = session("s1");
  const result = manager.pause(state);
  assert.equal(result.ok, false);
  assert.equal(state.paused, false);
  assert.equal(fs.existsSync(marker), false);
});

test("pause writes the session ID and resume removes it", () => {
  const { dir, manager, marker } = setup();
  heartbeat(dir, "s1");
  const state = session("s1");
  assert.equal(manager.pause(state).ok, true);
  assert.equal(state.paused, true);
  assert.equal(fs.readFileSync(marker, "utf8"), "s1");

  const result = manager.resume(state);
  assert.deepEqual(result, { ok: true });
  assert.equal(state.paused, false);
  assert.equal(fs.existsSync(marker), false);
});

test("pause refuses to replace a marker for another watched session", () => {
  const { dir, manager, marker } = setup();
  heartbeat(dir, "s1");
  heartbeat(dir, "s2");
  fs.writeFileSync(marker, "s2");
  const state = session("s1");
  assert.equal(manager.pause(state).ok, false);
  assert.equal(state.paused, false);
  assert.equal(fs.readFileSync(marker, "utf8"), "s2");
});

test("pause replaces a stale marker from a session with no watcher", () => {
  const { dir, manager, marker } = setup();
  heartbeat(dir, "s1");
  fs.writeFileSync(marker, "old-session");
  const state = session("s1");
  assert.equal(manager.pause(state).ok, true);
  assert.equal(fs.readFileSync(marker, "utf8"), "s1");
});

test("resume and cleanup never remove another session's marker", () => {
  const { manager, marker } = setup();
  fs.writeFileSync(marker, "s2");
  const state = session("s1", true);
  assert.equal(manager.resume(state).ok, true);
  assert.equal(state.paused, false);
  manager.cleanup("s1");
  assert.equal(fs.readFileSync(marker, "utf8"), "s2");
  manager.cleanup("s2");
  assert.equal(fs.existsSync(marker), false);
});

test("resume failure keeps the session paused", () => {
  const { manager, marker } = setup();
  fs.mkdirSync(marker);
  const state = session("s1", true);
  // A directory at the marker path cannot be read or unlinked as a file.
  const result = manager.resume(state);
  assert.equal(result.ok, false);
  assert.equal(state.paused, true);
});

test("resume warns when an approval is still pending", () => {
  const { dir, manager } = setup();
  fs.mkdirSync(path.join(dir, ".approvals"));
  fs.writeFileSync(
    path.join(dir, ".approvals", "abc123.json"),
    JSON.stringify({ request_id: "abc123", session_id: "s1", state: "pending" }),
  );
  fs.writeFileSync(
    path.join(dir, ".approvals", "def456.json"),
    JSON.stringify({ request_id: "def456", session_id: "s1", state: "approved" }),
  );
  const result = manager.resume(session("s1", true));
  assert.equal(result.ok, true);
  assert.match(result.message ?? "", /abc123/);
});

test("syncState reads the marker", () => {
  const { manager, marker } = setup();
  const state = session("s1");
  fs.writeFileSync(marker, "s1");
  manager.syncState(state);
  assert.equal(state.paused, true);
  fs.writeFileSync(marker, "s2");
  manager.syncState(state);
  assert.equal(state.paused, false);
});
