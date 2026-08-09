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
