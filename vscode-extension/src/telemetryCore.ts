/** Pure helpers and schema for privacy-preserving extension telemetry. */

export const EXTENSION_ACTIVATED = "agent_strace_vscode_extension_activated";
export const COMMAND_COMPLETED = "agent_strace_vscode_command_completed";
export const SESSION_STARTED = "agent_strace_vscode_session_started";
export const SESSION_COMPLETED = "agent_strace_vscode_session_completed";
export const TELEMETRY_ENABLED = "agent_strace_vscode_telemetry_enabled";

export type ExtensionTelemetryEvent =
  | typeof EXTENSION_ACTIVATED
  | typeof COMMAND_COMPLETED
  | typeof SESSION_STARTED
  | typeof SESSION_COMPLETED
  | typeof TELEMETRY_ENABLED;

export type ExtensionCommand =
  | "pause_agent"
  | "resume_agent"
  | "open_panel"
  | "clear_decorations"
  | "open_live_stream"
  | "open_post_mortem"
  | "refresh_session_browser"
  | "reveal_session"
  | "enable_telemetry"
  | "disable_telemetry";

type Validator = (value: unknown) => unknown | undefined;

const SAFE_TEXT = /^[A-Za-z0-9_.:/-]{1,80}$/;
const COMMANDS = new Set<ExtensionCommand>([
  "pause_agent",
  "resume_agent",
  "open_panel",
  "clear_decorations",
  "open_live_stream",
  "open_post_mortem",
  "refresh_session_browser",
  "reveal_session",
  "enable_telemetry",
  "disable_telemetry",
]);
const ENABLEMENT_SOURCES = new Set(["command", "settings", "editor_setting"]);

function booleanValue(value: unknown): boolean | undefined {
  return typeof value === "boolean" ? value : undefined;
}

function boundedInteger(maximum: number): Validator {
  return (value: unknown): number | undefined => {
    if (typeof value !== "number" || !Number.isFinite(value)) {
      return undefined;
    }
    return Math.max(0, Math.min(Math.trunc(value), maximum));
  };
}

function safeText(value: unknown): string | undefined {
  return typeof value === "string" && SAFE_TEXT.test(value) ? value : undefined;
}

function allowedValue(values: ReadonlySet<string>): Validator {
  return (value: unknown): string | undefined =>
    typeof value === "string" && values.has(value) ? value : undefined;
}

const EVENT_FIELDS: Record<
  ExtensionTelemetryEvent,
  Record<string, Validator>
> = {
  [EXTENSION_ACTIVATED]: {},
  [COMMAND_COMPLETED]: {
    command: allowedValue(COMMANDS),
    success: booleanValue,
    duration_ms: boundedInteger(2_678_400_000),
    error_type: safeText,
  },
  [SESSION_STARTED]: {},
  [SESSION_COMPLETED]: {
    success: booleanValue,
    duration_ms: boundedInteger(2_678_400_000),
    tool_call_count: boundedInteger(1_000_000),
    error_count: boundedInteger(1_000_000),
  },
  [TELEMETRY_ENABLED]: {
    source: allowedValue(ENABLEMENT_SOURCES),
  },
};

export function isExtensionTelemetryEvent(
  event: string,
): event is ExtensionTelemetryEvent {
  return Object.prototype.hasOwnProperty.call(EVENT_FIELDS, event);
}

/** Drop every property that is not explicitly allowed for this event. */
export function sanitiseEventProperties(
  event: string,
  properties: Record<string, unknown>,
): Record<string, unknown> | null {
  if (!isExtensionTelemetryEvent(event)) {
    return null;
  }

  const clean: Record<string, unknown> = {};
  for (const [name, validator] of Object.entries(EVENT_FIELDS[event])) {
    if (!Object.prototype.hasOwnProperty.call(properties, name)) {
      continue;
    }
    const value = validator(properties[name]);
    if (value !== undefined) {
      clean[name] = value;
    }
  }
  return clean;
}

// PostHog's own session rules: a new session after 30 minutes without an
// event, and no session longer than 24 hours.
export const SESSION_IDLE_MS = 30 * 60 * 1000;
export const SESSION_MAX_MS = 24 * 60 * 60 * 1000;

/** A random PostHog `$session_id` and when it started and was last used. */
export interface AnalyticsSession {
  readonly id: string;
  readonly startedAt: number;
  readonly lastEventAt: number;
}

/** Return a UUIDv7 (RFC 9562) for `unixMs` built from 10 random bytes. */
export function uuidv7(unixMs: number, random: Uint8Array): string {
  const bytes = new Uint8Array(16);
  let timestamp = Math.max(0, Math.trunc(unixMs));
  for (let index = 5; index >= 0; index -= 1) {
    bytes[index] = timestamp % 256;
    timestamp = Math.floor(timestamp / 256);
  }
  bytes.set(random.subarray(0, 10), 6);
  bytes[6] = 0x70 | (bytes[6] & 0x0f);
  bytes[8] = 0x80 | (bytes[8] & 0x3f);
  const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

/**
 * Return the session an event sent at `now` belongs to.  The session goes
 * on until 30 idle minutes or 24 hours have passed, or the clock moves back
 * before its start; PostHog drops a session ID whose UUIDv7 time is later
 * than the session's first event or more than 24 hours before its last.
 */
export function nextAnalyticsSession(
  current: AnalyticsSession | undefined,
  now: number,
  newId: (startedAt: number) => string,
): AnalyticsSession {
  if (
    current === undefined ||
    now < current.startedAt ||
    now - current.startedAt >= SESSION_MAX_MS ||
    now - Math.max(current.lastEventAt, current.startedAt) >= SESSION_IDLE_MS
  ) {
    return { id: newId(now), startedAt: now, lastEventAt: now };
  }
  return { ...current, lastEventAt: Math.max(current.lastEventAt, now) };
}
