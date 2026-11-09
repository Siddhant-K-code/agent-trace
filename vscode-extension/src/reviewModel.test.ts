import * as assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { test } from "node:test";
import {
  buildReviewModel,
  loadStoredReview,
  renderMinimizedReviewHtml,
  ReviewAnnotation,
} from "./reviewModel";
import { SessionMeta, TraceEvent } from "./traceStore";

const fixtureDir = path.resolve(
  __dirname,
  "../../src/agent_trace/corpus/v1/fixture-review-001",
);

function fixtureEvents(): TraceEvent[] {
  return fs.readFileSync(path.join(fixtureDir, "events.ndjson"), "utf8")
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line) as TraceEvent);
}

function fixtureAnnotations(): ReviewAnnotation[] {
  return fs.readFileSync(path.join(fixtureDir, "annotations.jsonl"), "utf8")
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line) as ReviewAnnotation);
}

function withStoredSession(
  callback: (root: string, sessionId: string) => void,
  options: {
    metaSessionId?: string;
    eventSessionId?: string;
    data?: unknown;
  } = {},
): void {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "agent-trace-review-"));
  const sessionId = "stored-session";
  const sessionDir = path.join(root, sessionId);
  fs.mkdirSync(sessionDir);
  fs.writeFileSync(path.join(sessionDir, "meta.json"), JSON.stringify({
    session_id: options.metaSessionId ?? sessionId,
    started_at: 1,
    ended_at: 2,
    tool_calls: 0,
    errors: 0,
    total_tokens: 0,
  }));
  fs.writeFileSync(path.join(sessionDir, "events.ndjson"), JSON.stringify({
    event_type: "session_start",
    timestamp: 1,
    event_id: "start",
    session_id: options.eventSessionId ?? sessionId,
    data: options.data ?? {},
  }) + "\n");
  try {
    callback(root, sessionId);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
}

test("representative fixture projects review evidence", () => {
  const events = fixtureEvents();
  const meta: SessionMeta = {
    session_id: "fixture-review-001",
    started_at: events[0].timestamp,
    ended_at: events[events.length - 1].timestamp,
    agent_name: "synthetic-review-agent",
    command: "fixture capture",
    tool_calls: 2,
    errors: 1,
    total_tokens: 0,
  };

  const review = buildReviewModel(meta, events, fixtureAnnotations());
  const byId = new Map(review.items.map((item) => [item.event_id, item]));

  assert.equal(review.session.provider, "synthetic");
  assert.equal(review.session.capture_method, "fixture");
  assert.equal(review.session.duration_ms, 11_000);
  assert.equal(review.privacy.content_mode, "mixed");
  assert.equal(byId.get("ev-orphan")?.relationship, "orphaned");
  assert.equal(byId.get("ev-orphan")?.relationship_issue, "missing_parent");
  assert.ok(byId.get("ev-orphan")?.categories.includes("gaps"));
  assert.ok(byId.get("tool-test-1")?.categories.includes("tests"));
  assert.ok(byId.get("tool-test-2")?.categories.includes("retry"));
  assert.ok(byId.get("result-test-2")?.categories.includes("recovery"));
  assert.ok(byId.get("ev-decision")?.categories.includes("corrections"));
  assert.equal(byId.get("ev-decision")?.annotations[0].correction, true);
});

test("minimized extension share allowlists capture and annotation metadata", () => {
  const events = fixtureEvents();
  const review = buildReviewModel({
    session_id: "fixture-review-001",
    started_at: events[0].timestamp,
    ended_at: events[events.length - 1].timestamp,
    tool_calls: 2,
    errors: 1,
    total_tokens: 0,
  }, events, fixtureAnnotations(), {
    status: "partial",
    provider: "untrusted-provider",
    capture_method: "customer-capture-name",
  });
  const annotated = review.items.find((item) => item.annotations.length > 0);
  assert.ok(annotated);
  annotated.annotations[0].label = "customer-secret-label";

  const html = renderMinimizedReviewHtml(review);

  assert.match(html, /Privacy-minimized evidence review/);
  assert.match(html, /Provider: unknown/);
  assert.match(html, /Capture: unknown/);
  assert.match(html, /Annotation categories: correction 1/);
  assert.doesNotMatch(html, /fixture-review-001/);
  assert.doesNotMatch(html, /src\/example\.py/);
  assert.doesNotMatch(html, /python -m pytest/);
  assert.doesNotMatch(html, /Human correction/);
  assert.doesNotMatch(html, /1 passed/);
  assert.doesNotMatch(html, /untrusted-provider/);
  assert.doesNotMatch(html, /customer-capture-name/);
  assert.doesNotMatch(html, /customer-secret-label/);
});

test("subagent response text is content-rich", () => {
  const event: TraceEvent = {
    event_type: "subagent_stop",
    timestamp: 1,
    event_id: "subagent-stop",
    session_id: "session",
    data: { text: "private subagent response" },
  };
  const review = buildReviewModel({
    session_id: "session",
    started_at: 1,
    ended_at: 2,
    tool_calls: 0,
    errors: 0,
    total_tokens: 0,
  }, [event]);

  assert.equal(review.privacy.content_mode, "recorded");
});

test("unresolved annotations remain visible as capture gaps", () => {
  const events = fixtureEvents();
  const duplicate = { ...events[0], timestamp: events[0].timestamp + 0.1 };
  const annotations: ReviewAnnotation[] = [
    { annotation_id: "missing", event_id: "absent", label: "question" },
    { annotation_id: "duplicate", event_id: events[0].event_id, label: "retry" },
  ];
  const review = buildReviewModel({
    session_id: "fixture-review-001",
    started_at: events[0].timestamp,
    ended_at: events[events.length - 1].timestamp,
    tool_calls: 2,
    errors: 1,
    total_tokens: 0,
  }, [events[0], duplicate, ...events.slice(1)], annotations);

  assert.deepEqual(
    review.unresolved_annotations.map((annotation) => annotation.annotation_id),
    ["missing", "duplicate"],
  );
  assert.equal(
    review.capture_gaps.filter((reason) => reason.code === "unresolved_annotation").length,
    2,
  );
});

test("stored review rejects traversal, symlinks, and mismatched IDs", () => {
  withStoredSession((root) => {
    assert.throws(
      () => loadStoredReview(root, "../escape"),
      /Invalid stored session ID/,
    );
  });

  withStoredSession((root, sessionId) => {
    const target = path.join(root, sessionId);
    const link = path.join(root, "linked-session");
    fs.symlinkSync(target, link, "dir");
    assert.throws(
      () => loadStoredReview(root, "linked-session"),
      /Session directory must be a real directory/,
    );
  });

  withStoredSession((root, sessionId) => {
    const sessionDir = path.join(root, sessionId);
    const realMeta = path.join(root, "real-meta.json");
    fs.renameSync(path.join(sessionDir, "meta.json"), realMeta);
    fs.symlinkSync(realMeta, path.join(sessionDir, "meta.json"));
    assert.throws(
      () => loadStoredReview(root, sessionId),
      /session metadata must be a real file/,
    );
  });

  withStoredSession((root, sessionId) => {
    const sessionDir = path.join(root, sessionId);
    const realEvents = path.join(root, "real-events.ndjson");
    fs.renameSync(path.join(sessionDir, "events.ndjson"), realEvents);
    fs.symlinkSync(realEvents, path.join(sessionDir, "events.ndjson"));
    assert.throws(
      () => loadStoredReview(root, sessionId),
      /event stream must be a real file/,
    );
  });

  withStoredSession((root, sessionId) => {
    const sessionDir = path.join(root, sessionId);
    const realAnnotations = path.join(root, "real-annotations.jsonl");
    fs.writeFileSync(realAnnotations, "");
    fs.symlinkSync(realAnnotations, path.join(sessionDir, "annotations.jsonl"));
    assert.throws(
      () => loadStoredReview(root, sessionId),
      /annotations sidecar must be a real file/,
    );
  });

  withStoredSession((root, sessionId) => {
    assert.throws(
      () => loadStoredReview(root, sessionId),
      /Session metadata ID does not match/,
    );
  }, { metaSessionId: "other-session" });

  withStoredSession((root, sessionId) => {
    assert.throws(
      () => loadStoredReview(root, sessionId),
      /Event session ID does not match/,
    );
  }, { eventSessionId: "other-session" });

  withStoredSession((root, sessionId) => {
    assert.throws(
      () => loadStoredReview(root, sessionId),
      /Session event data must be an object/,
    );
  }, { data: ["not", "an", "object"] });
});
