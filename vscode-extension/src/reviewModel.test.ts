import * as assert from "node:assert/strict";
import * as fs from "node:fs";
import * as path from "node:path";
import { test } from "node:test";
import {
  buildReviewModel,
  renderMinimizedReviewHtml,
  ReviewAnnotation,
} from "./reviewModel";
import { SessionMeta, TraceEvent } from "./traceStore";

const fixtureDir = path.resolve(
  __dirname,
  "../../tests/fixtures/representative_session",
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
  assert.ok(byId.get("ev-error")?.categories.includes("corrections"));
  assert.equal(byId.get("ev-error")?.annotations[0].correction, true);
});

test("minimized extension share omits recorded values", () => {
  const events = fixtureEvents();
  const review = buildReviewModel({
    session_id: "fixture-review-001",
    started_at: events[0].timestamp,
    ended_at: events[events.length - 1].timestamp,
    tool_calls: 2,
    errors: 1,
    total_tokens: 0,
  }, events, fixtureAnnotations());

  const html = renderMinimizedReviewHtml(review);

  assert.match(html, /Privacy-minimized evidence review/);
  assert.doesNotMatch(html, /fixture-review-001/);
  assert.doesNotMatch(html, /src\/example\.py/);
  assert.doesNotMatch(html, /python -m pytest/);
  assert.doesNotMatch(html, /Human correction/);
  assert.doesNotMatch(html, /1 passed/);
});
