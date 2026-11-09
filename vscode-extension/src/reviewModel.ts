import { createHash } from "crypto";
import * as fs from "fs";
import * as path from "path";
import { SessionMeta, TraceEvent } from "./traceStore";

export interface ReviewAnnotation {
  annotation_id: string;
  session_id?: string;
  event_id?: string;
  offset_seconds?: number;
  label?: string;
  note?: string;
  author?: string;
  created_at?: number;
}

export interface HealthReason {
  code: string;
  message: string;
  kind: string;
  event_id?: string | null;
}

export interface CliHealth {
  status: string;
  reasons?: HealthReason[];
  provider?: string;
  capture_method?: string;
  redaction?: {
    mode?: string;
  };
}

export interface ReviewItem {
  sequence: number;
  event_id: string;
  event_type: string;
  timestamp: number;
  duration_ms?: number;
  parent_id?: string;
  child_ids: string[];
  relationship: "none" | "linked" | "orphaned";
  relationship_issue?: string;
  missing_child: boolean;
  categories: string[];
  redacted: boolean;
  source_index: number;
  summary: string;
  data: Record<string, unknown>;
  annotations: Array<ReviewAnnotation & { correction: boolean }>;
}

export interface ReviewModel {
  schema_version: 1;
  session: {
    session_id: string;
    provider: string;
    capture_method: string;
    agent_name?: string;
    command?: string;
    started_at?: number;
    ended_at?: number;
    duration_ms?: number;
    active: boolean;
  };
  evidence_health: {
    status: string;
    reasons: HealthReason[];
  };
  capture_gaps: HealthReason[];
  privacy: {
    redaction_mode: string;
    content_mode: string;
  };
  filters: Array<{ id: string; count: number }>;
  items: ReviewItem[];
}

const FILTERS = [
  "files",
  "commands",
  "tools",
  "tests",
  "failures",
  "retry",
  "recovery",
  "annotations",
  "corrections",
  "privacy",
  "gaps",
];

const CONTENT_TYPES = new Set([
  "user_prompt",
  "assistant_response",
  "llm_request",
  "llm_response",
  "tool_call",
  "tool_result",
  "file_read",
  "file_write",
]);

const EXPECTED_PARENTS: Record<string, Set<string>> = {
  tool_result: new Set(["tool_call"]),
  llm_response: new Set(["llm_request"]),
  error: new Set(["tool_call", "llm_request"]),
};

const EXPECTED_CHILDREN: Record<string, Set<string>> = {
  tool_call: new Set(["tool_result", "error"]),
  llm_request: new Set(["llm_response", "error"]),
};
const COMMAND_TOOLS = new Set([
  "bash",
  "shell",
  "execute_command",
  "exec_command",
  "terminal",
  "run_command",
]);

function commandFor(event: TraceEvent): string {
  if (event.event_type !== "tool_call") {
    return "";
  }
  const direct = event.data.command;
  if (typeof direct === "string") {
    return direct;
  }
  const args = event.data.arguments ?? event.data.tool_input;
  if (typeof args === "object" && args !== null) {
    const record = args as Record<string, unknown>;
    const value = record.command ?? record.cmd;
    return typeof value === "string" ? value : "";
  }
  return "";
}

function isTestCommand(command: string): boolean {
  const tokens = command.trim().toLowerCase().split(/\s+/);
  if (tokens.length === 0) {
    return false;
  }
  return (
    tokens[0] === "pytest" ||
    tokens[0] === "py.test" ||
    tokens.slice(0, 3).join(" ") === "python -m pytest" ||
    tokens.slice(0, 3).join(" ") === "python3 -m pytest" ||
    tokens.slice(0, 2).join(" ") === "npm test" ||
    tokens.slice(0, 2).join(" ") === "bun test" ||
    tokens.slice(0, 2).join(" ") === "cargo test" ||
    tokens.slice(0, 2).join(" ") === "go test"
  );
}

function categoriesFor(event: TraceEvent): Set<string> {
  const categories = new Set<string>();
  if (event.event_type === "file_read" || event.event_type === "file_write") {
    categories.add("files");
  }
  if (event.event_type === "tool_call" || event.event_type === "tool_result") {
    categories.add("tools");
  }
  if (event.event_type === "error") {
    categories.add("failures");
  }
  if (event.redacted) {
    categories.add("privacy");
  }
  const command = commandFor(event);
  const category = event.data.category;
  const toolName = String(event.data.tool_name ?? event.data.tool ?? "").toLowerCase();
  if (event.data.is_test === true || category === "test" || isTestCommand(command)) {
    categories.add("tests");
  }
  if (
    event.data.is_command === true ||
    category === "command" ||
    command ||
    (event.event_type === "tool_call" && COMMAND_TOOLS.has(toolName))
  ) {
    categories.add("commands");
  }
  if (event.data.retry_of || event.data.retry === true) {
    categories.add("retry");
  }
  if (event.data.recovery_of || event.data.recovered === true) {
    categories.add("recovery");
  }
  if (event.data.privacy_transformed === true) {
    categories.add("privacy");
  }
  return categories;
}

function eventSummary(event: TraceEvent): string {
  const data = event.data;
  if (event.event_type === "tool_call") {
    const tool = data.tool_name ?? data.tool ?? "tool";
    const detail = commandFor(event) || JSON.stringify(data.arguments ?? data.tool_input ?? "");
    return detail ? `${String(tool)}: ${detail}` : String(tool);
  }
  if (event.event_type === "tool_result") {
    return String(data.output ?? data.result ?? "tool result");
  }
  if (event.event_type === "file_read" || event.event_type === "file_write") {
    return String(data.path ?? data.file_path ?? event.event_type);
  }
  if (event.event_type === "error") {
    return String(data.message ?? data.error ?? "recorded error");
  }
  if (event.event_type === "user_prompt") {
    return String(data.prompt ?? data.content ?? data.text ?? "user prompt");
  }
  if (event.event_type === "assistant_response") {
    return String(data.response ?? data.content ?? data.text ?? "assistant response");
  }
  if (event.event_type === "decision") {
    return String(data.decision ?? data.reasoning ?? data.text ?? "recorded decision");
  }
  return event.event_type.replace(/_/g, " ");
}

function correctionFor(annotation: ReviewAnnotation): boolean {
  return (
    (annotation.label ?? "").trim().toLowerCase() === "correction" ||
    (annotation.note ?? "").trim().toLowerCase().startsWith("human correction:")
  );
}

function annotationMap(
  annotations: ReviewAnnotation[],
  events: TraceEvent[],
): Map<string, Array<ReviewAnnotation & { correction: boolean }>> {
  const mapped = new Map<string, Array<ReviewAnnotation & { correction: boolean }>>();
  const idCounts = new Map<string, number>();
  for (const event of events) {
    idCounts.set(event.event_id, (idCounts.get(event.event_id) ?? 0) + 1);
  }
  const startedAt = events[0]?.timestamp ?? 0;
  for (const annotation of annotations) {
    let target = annotation.event_id;
    if ((!target || idCounts.get(target) !== 1) && !target && events.length > 0) {
      const offset = annotation.offset_seconds ?? 0;
      target = events.reduce((best, event) => {
        const bestDistance = Math.abs((best.timestamp - startedAt) - offset);
        const distance = Math.abs((event.timestamp - startedAt) - offset);
        return distance < bestDistance ? event : best;
      }, events[0]).event_id;
    }
    if (!target || idCounts.get(target) !== 1) {
      continue;
    }
    const values = mapped.get(target) ?? [];
    values.push({ ...annotation, correction: correctionFor(annotation) });
    mapped.set(target, values);
  }
  return mapped;
}

function structuralProjection(
  events: TraceEvent[],
  annotations: ReviewAnnotation[],
): { items: ReviewItem[]; reasons: HealthReason[] } {
  const ordered = events
    .map((event, sourceIndex) => ({ event, sourceIndex }))
    .sort((a, b) => a.event.timestamp - b.event.timestamp || a.sourceIndex - b.sourceIndex);
  const byId = new Map<string, TraceEvent[]>();
  for (const event of events) {
    const values = byId.get(event.event_id) ?? [];
    values.push(event);
    byId.set(event.event_id, values);
  }
  const annotationsByEvent = annotationMap(annotations, events);
  const childIds = new Map<string, string[]>();
  const reasons: HealthReason[] = [];

  const relationships = ordered.map(({ event }) => {
    let relationship: ReviewItem["relationship"] = "none";
    let relationshipIssue: string | undefined;
    if (EXPECTED_PARENTS[event.event_type]) {
      if (!event.parent_id) {
        relationship = "orphaned";
        relationshipIssue = "missing_parent";
      } else {
        const parents = byId.get(event.parent_id) ?? [];
        if (parents.length === 0) {
          relationship = "orphaned";
          relationshipIssue = "missing_parent";
        } else if (parents.length > 1) {
          relationship = "orphaned";
          relationshipIssue = "ambiguous_parent";
        } else if (!EXPECTED_PARENTS[event.event_type].has(parents[0].event_type)) {
          relationship = "orphaned";
          relationshipIssue = "invalid_parent_type";
        } else {
          relationship = "linked";
          const children = childIds.get(event.parent_id) ?? [];
          children.push(event.event_id);
          childIds.set(event.parent_id, children);
        }
      }
    } else if (event.parent_id) {
      const parents = byId.get(event.parent_id) ?? [];
      if (parents.length === 1) {
        relationship = "linked";
        const children = childIds.get(event.parent_id) ?? [];
        children.push(event.event_id);
        childIds.set(event.parent_id, children);
      } else {
        relationship = "orphaned";
        relationshipIssue = parents.length > 1 ? "ambiguous_parent" : "missing_parent";
      }
    }
    if (relationshipIssue) {
      reasons.push({
        code: relationshipIssue,
        message: `${event.event_type} ${event.event_id} has an unresolved parent relationship.`,
        kind: "observed",
        event_id: event.event_id,
      });
    }
    return { relationship, relationshipIssue };
  });

  const items = ordered.map(({ event, sourceIndex }, index): ReviewItem => {
    const categories = categoriesFor(event);
    const eventAnnotations = annotationsByEvent.get(event.event_id) ?? [];
    if (eventAnnotations.length > 0) {
      categories.add("annotations");
    }
    if (eventAnnotations.some((annotation) => annotation.correction)) {
      categories.add("corrections");
    }
    const children = childIds.get(event.event_id) ?? [];
    const expected = EXPECTED_CHILDREN[event.event_type];
    const missingChild = Boolean(
      expected &&
      !children.some((childId) => {
        const child = byId.get(childId);
        return child?.length === 1 && expected.has(child[0].event_type);
      }),
    );
    if (missingChild || relationships[index].relationship === "orphaned") {
      categories.add("gaps");
    }
    if (
      event.event_type === "tool_result" &&
      relationships[index].relationship === "linked" &&
      event.parent_id
    ) {
      const parent = byId.get(event.parent_id)?.[0];
      if (
        parent?.data.retry_of &&
        (event.data.exit_code === 0 || event.data.success === true)
      ) {
        categories.add("recovery");
      }
    }
    if (missingChild) {
      reasons.push({
        code: "missing_child",
        message: `${event.event_type} ${event.event_id} has no recorded terminal result.`,
        kind: "observed",
        event_id: event.event_id,
      });
    }
    return {
      sequence: index + 1,
      event_id: event.event_id,
      event_type: event.event_type,
      timestamp: event.timestamp,
      duration_ms: event.duration_ms,
      parent_id: event.parent_id,
      child_ids: children,
      relationship: relationships[index].relationship,
      relationship_issue: relationships[index].relationshipIssue,
      missing_child: missingChild,
      categories: [...categories].sort(),
      redacted: Boolean(event.redacted),
      source_index: sourceIndex,
      summary: eventSummary(event),
      data: event.data,
      annotations: eventAnnotations,
    };
  });
  return { items, reasons };
}

function captureIdentity(
  meta: SessionMeta,
  events: TraceEvent[],
  exactHealth?: CliHealth,
): { provider: string; captureMethod: string } {
  const start = events.find((event) => event.event_type === "session_start");
  const provider = String(
    exactHealth?.provider ||
    start?.data.provider ||
    meta.agent_name ||
    "unknown",
  );
  const captureMethod = String(
    exactHealth?.capture_method ||
    start?.data.capture_method ||
    "unknown",
  );
  return { provider, captureMethod };
}

export function buildReviewModel(
  meta: SessionMeta,
  events: TraceEvent[],
  annotations: ReviewAnnotation[] = [],
  exactHealth?: CliHealth,
): ReviewModel {
  const projection = structuralProjection(events, annotations);
  const identity = captureIdentity(meta, events, exactHealth);
  const startedAt = meta.started_at || projection.items[0]?.timestamp;
  const endedEvent = [...events].reverse().find((event) => event.event_type === "session_end");
  const endedAt = meta.ended_at || endedEvent?.timestamp;
  const durationMs = startedAt && endedAt ? Math.max(0, (endedAt - startedAt) * 1000) : undefined;
  const contentEvents = events.filter((event) => CONTENT_TYPES.has(event.event_type));
  const redactedContent = contentEvents.filter((event) => event.redacted).length;
  let contentMode = "recorded";
  if (contentEvents.length === 0) {
    contentMode = "metadata_only";
  } else if (redactedContent === contentEvents.length) {
    contentMode = "redacted";
  } else if (redactedContent > 0) {
    contentMode = "mixed";
  }
  const reasons = exactHealth?.reasons ?? projection.reasons;
  const status = exactHealth?.status ?? (projection.reasons.length > 0 ? "partial" : "unknown");
  const counts = new Map<string, number>();
  for (const item of projection.items) {
    for (const category of item.categories) {
      counts.set(category, (counts.get(category) ?? 0) + 1);
    }
  }
  return {
    schema_version: 1,
    session: {
      session_id: meta.session_id,
      provider: identity.provider,
      capture_method: identity.captureMethod,
      agent_name: meta.agent_name,
      command: meta.command,
      started_at: startedAt,
      ended_at: endedAt,
      duration_ms: durationMs,
      active: !endedAt,
    },
    evidence_health: {
      status,
      reasons,
    },
    capture_gaps: reasons,
    privacy: {
      redaction_mode: exactHealth?.redaction?.mode ?? "not_recorded",
      content_mode: contentMode,
    },
    filters: FILTERS.map((id) => ({ id, count: counts.get(id) ?? 0 })),
    items: projection.items,
  };
}

export function loadAnnotations(traceDir: string, sessionId: string): ReviewAnnotation[] {
  const file = path.join(traceDir, sessionId, "annotations.jsonl");
  try {
    return fs.readFileSync(file, "utf8")
      .split(/\r?\n/)
      .filter(Boolean)
      .flatMap((line) => {
        try {
          return [JSON.parse(line) as ReviewAnnotation];
        } catch {
          return [];
        }
      });
  } catch {
    return [];
  }
}

export function loadStoredReview(
  traceDir: string,
  sessionId: string,
  exactHealth?: CliHealth,
): ReviewModel {
  const sessionDir = path.resolve(traceDir, sessionId);
  const relative = path.relative(path.resolve(traceDir), sessionDir);
  if (!relative || relative.startsWith("..") || path.isAbsolute(relative)) {
    throw new Error("Session path escapes the configured trace directory.");
  }
  const meta = JSON.parse(
    fs.readFileSync(path.join(sessionDir, "meta.json"), "utf8"),
  ) as SessionMeta;
  const events = fs.readFileSync(path.join(sessionDir, "events.ndjson"), "utf8")
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line) as TraceEvent);
  return buildReviewModel(meta, events, loadAnnotations(traceDir, sessionId), exactHealth);
}

export function renderMinimizedReviewHtml(review: ReviewModel): string {
  const sessionRef = createHash("sha256")
    .update(review.session.session_id, "utf8")
    .digest("hex")
    .slice(0, 12);
  const rows = review.items.map((item) => {
    const categories = item.categories.join(", ") || "none";
    const relation = item.relationship_issue
      ? `${item.relationship} (${item.relationship_issue})`
      : item.relationship;
    return `<tr><td>${item.sequence}</td><td>${escapeHtml(item.event_type)}</td><td>${escapeHtml(categories)}</td><td>${escapeHtml(relation)}</td><td>${item.redacted ? "yes" : "no"}</td></tr>`;
  }).join("");
  return `<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Privacy-minimized evidence review</title>
<style>body{font:15px/1.5 system-ui,sans-serif;max-width:980px;margin:40px auto;padding:0 20px;color:#172033;background:#f7f8fb}h1{font-size:1.6rem}.meta{display:flex;gap:24px;flex-wrap:wrap;margin:20px 0;padding:16px;background:#fff;border:1px solid #ccd3df}table{width:100%;border-collapse:collapse;background:#fff}th,td{padding:8px 10px;text-align:left;border-bottom:1px solid #dfe4ec}th{color:#475569}</style>
</head><body><h1>Privacy-minimized evidence review</h1>
<p>This local file omits session identity, absolute timestamps, raw fields, prompts, commands, paths, results, and annotation text.</p>
<div class="meta"><span>Session ref: ${sessionRef}</span><span>Provider: ${escapeHtml(review.session.provider)}</span><span>Evidence health: ${escapeHtml(review.evidence_health.status)}</span><span>Events: ${review.items.length}</span></div>
<table><thead><tr><th>#</th><th>Evidence</th><th>Categories</th><th>Relationship</th><th>Redacted</th></tr></thead><tbody>${rows}</tbody></table>
</body></html>`;
}

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
