/** Default-on, privacy-preserving product telemetry for the editor extension. */

import * as crypto from "crypto";
import * as https from "https";
import * as vscode from "vscode";
import {
  COMMAND_COMPLETED,
  EXTENSION_ACTIVATED,
  ExtensionCommand,
  ExtensionTelemetryEvent,
  SESSION_COMPLETED,
  SESSION_STARTED,
  TELEMETRY_ENABLED,
  sanitiseEventProperties,
} from "./telemetryCore";
import { SessionState } from "./traceStore";

// PostHog project tokens are public, write-only event-ingestion tokens.
const POSTHOG_PROJECT_TOKEN =
  "phc_yWjxK7ibJuFfskDuTr4HbXGdyGvFC6V3Y3fRwNDiRK6x";
const POSTHOG_HOST = "https://us.i.posthog.com";
const REQUEST_TIMEOUT_MS = 500;
const TELEMETRY_SCHEMA_VERSION = 1;

const ANONYMOUS_ID_KEY = "agentTrace.telemetry.anonymousId";
const NOTICE_SHOWN_KEY = "agentTrace.telemetry.noticeShown";
const CONFIG_SECTION = "agentTrace";
const CONFIG_KEY = "telemetry.enabled";
const TRUE_VALUES = new Set(["1", "true", "yes", "on", "enabled"]);
const FALSE_VALUES = new Set(["0", "false", "no", "off", "disabled"]);

type EnablementSource = "command" | "settings" | "editor_setting";

function parseBoolean(value: string | undefined): boolean | undefined {
  if (value === undefined) {
    return undefined;
  }
  const normalised = value.trim().toLowerCase();
  if (TRUE_VALUES.has(normalised)) {
    return true;
  }
  if (FALSE_VALUES.has(normalised)) {
    return false;
  }
  return undefined;
}

function editorName(): string {
  const name = vscode.env.appName.toLowerCase();
  if (name.includes("cursor")) {
    return "cursor";
  }
  if (name.includes("windsurf")) {
    return "windsurf";
  }
  if (name.includes("vscodium")) {
    return "vscodium";
  }
  if (name.includes("visual studio code")) {
    return "vscode";
  }
  return "other";
}

function errorType(error: unknown): string {
  if (error instanceof Error && /^[A-Za-z0-9_.:/-]{1,80}$/.test(error.name)) {
    return error.name;
  }
  return "UnknownError";
}

export class ExtensionTelemetry implements vscode.Disposable {
  private anonymousId: string | undefined;
  private disclosurePending = false;
  private settingFromCommand = false;
  private readonly requests = new Set<ReturnType<typeof https.request>>();

  constructor(private readonly context: vscode.ExtensionContext) {
    const storedId = context.globalState.get<string>(ANONYMOUS_ID_KEY);
    if (storedId && /^[0-9a-f]{32}$/.test(storedId)) {
      this.anonymousId = storedId;
    }
  }

  /** Display the one-time disclosure before sending this installation's first event. */
  initialize(): void {
    const noticeShown = this.context.globalState.get<boolean>(
      NOTICE_SHOWN_KEY,
      false,
    );
    if (!this.isEnabled() || noticeShown) {
      this.capture(EXTENSION_ACTIVATED);
      return;
    }

    this.disclosurePending = true;
    void Promise.resolve(
      this.context.globalState.update(NOTICE_SHOWN_KEY, true),
    )
      .catch(() => undefined)
      .then(async () => {
        const choice = await vscode.window.showInformationMessage(
          "agent-strace anonymous extension telemetry is enabled by default. " +
            "Prompts, trace contents, paths, session IDs, and command arguments are never sent.",
          "Disable telemetry",
          "Privacy details",
        );
        if (choice === "Disable telemetry") {
          await this.setEnabled(false, "command");
        } else if (choice === "Privacy details") {
          await vscode.env.openExternal(
            vscode.Uri.parse(
              "https://github.com/Siddhant-K-code/agent-trace/blob/main/docs/telemetry.md",
            ),
          );
        }
      })
      .catch(() => undefined)
      .finally(() => {
        this.disclosurePending = false;
        this.capture(EXTENSION_ACTIVATED);
      });
  }

  isEnabled(): boolean {
    // The editor-wide telemetry switch is a hard privacy boundary.
    if (vscode.env.isTelemetryEnabled === false) {
      return false;
    }
    if (parseBoolean(process.env.DO_NOT_TRACK) === true) {
      return false;
    }

    const environmentOverride = parseBoolean(
      process.env.AGENT_STRACE_TELEMETRY,
    );
    if (environmentOverride === false) {
      return false;
    }

    return vscode.workspace
      .getConfiguration(CONFIG_SECTION)
      .get<boolean>(CONFIG_KEY, true);
  }

  async setEnabled(enabled: boolean, source: EnablementSource): Promise<void> {
    this.settingFromCommand = source === "command";
    try {
      await vscode.workspace
        .getConfiguration(CONFIG_SECTION)
        .update(CONFIG_KEY, enabled, vscode.ConfigurationTarget.Global);
    } finally {
      this.settingFromCommand = false;
    }

    if (!enabled) {
      await this.clearAnonymousId();
      return;
    }
    this.capture(TELEMETRY_ENABLED, { source });
  }

  /** Handle changes made directly in the Settings UI. */
  handleConfigurationChange(): void {
    if (this.settingFromCommand) {
      return;
    }
    if (!this.isEnabled()) {
      void this.clearAnonymousId().catch(() => undefined);
      return;
    }
    this.capture(TELEMETRY_ENABLED, { source: "settings" });
  }

  /** Respect changes to VS Code's global telemetry level immediately. */
  handleEditorTelemetryChange(enabled: boolean): void {
    if (!enabled) {
      void this.clearAnonymousId().catch(() => undefined);
      return;
    }
    if (this.isEnabled()) {
      this.capture(TELEMETRY_ENABLED, { source: "editor_setting" });
    }
  }

  captureSessionStarted(): void {
    this.capture(SESSION_STARTED);
  }

  captureSessionCompleted(state: SessionState): void {
    const startedAtMs = Math.max(0, state.meta.started_at * 1000);
    const endedAtMs = state.meta.ended_at
      ? state.meta.ended_at * 1000
      : Date.now();
    this.capture(SESSION_COMPLETED, {
      success: state.errorCount === 0,
      duration_ms: Math.max(0, endedAtMs - startedAtMs),
      tool_call_count: state.toolCallCount,
      error_count: state.errorCount,
    });
  }

  trackCommand(
    command: ExtensionCommand,
    handler: (...args: unknown[]) => unknown,
  ): (...args: unknown[]) => unknown {
    return (...args: unknown[]): unknown => {
      const startedAt = Date.now();
      try {
        const result = handler(...args);
        if (
          result !== null &&
          typeof result === "object" &&
          "then" in result &&
          typeof (result as { then?: unknown }).then === "function"
        ) {
          return Promise.resolve(result).then(
            (value) => {
              this.capture(COMMAND_COMPLETED, {
                command,
                success: true,
                duration_ms: Date.now() - startedAt,
              });
              return value;
            },
            (error: unknown) => {
              this.capture(COMMAND_COMPLETED, {
                command,
                success: false,
                duration_ms: Date.now() - startedAt,
                error_type: errorType(error),
              });
              throw error;
            },
          );
        }
        this.capture(COMMAND_COMPLETED, {
          command,
          success: true,
          duration_ms: Date.now() - startedAt,
        });
        return result;
      } catch (error) {
        this.capture(COMMAND_COMPLETED, {
          command,
          success: false,
          duration_ms: Date.now() - startedAt,
          error_type: errorType(error),
        });
        throw error;
      }
    };
  }

  capture(
    event: ExtensionTelemetryEvent,
    properties: Record<string, unknown> = {},
  ): void {
    if (!this.isEnabled() || this.disclosurePending) {
      return;
    }
    const clean = sanitiseEventProperties(event, properties);
    if (clean === null) {
      return;
    }

    const body = JSON.stringify({
      api_key: POSTHOG_PROJECT_TOKEN,
      event,
      distinct_id: this.getAnonymousId(),
      properties: {
        ...clean,
        $process_person_profile: false,
        $geoip_disable: true,
        $lib: "agent-strace-vscode",
        $lib_version: this.context.extension.packageJSON.version,
        telemetry_schema_version: TELEMETRY_SCHEMA_VERSION,
        extension_version: this.context.extension.packageJSON.version,
        editor: editorName(),
        editor_version: vscode.version,
        ui_kind: vscode.env.uiKind === vscode.UIKind.Web ? "web" : "desktop",
        remote: Boolean(vscode.env.remoteName),
        os: process.platform,
      },
      timestamp: new Date().toISOString(),
      uuid: crypto.randomUUID(),
    });

    try {
      const req = https.request(
        new URL(POSTHOG_HOST + "/i/v0/e/"),
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Content-Length": Buffer.byteLength(body),
            "User-Agent": `agent-strace-vscode/${this.context.extension.packageJSON.version}`,
          },
        },
        (response) => response.resume(),
      );
      this.requests.add(req);
      req.on("close", () => this.requests.delete(req));
      req.on("error", () => this.requests.delete(req));
      req.setTimeout(REQUEST_TIMEOUT_MS, () => req.destroy());
      req.end(body);
    } catch {
      // Product telemetry is always best-effort and never affects extension UI.
    }
  }

  private getAnonymousId(): string {
    if (!this.anonymousId) {
      this.anonymousId = crypto.randomBytes(16).toString("hex");
      void Promise.resolve(
        this.context.globalState.update(ANONYMOUS_ID_KEY, this.anonymousId),
      ).catch(() => undefined);
    }
    return this.anonymousId;
  }

  private async clearAnonymousId(): Promise<void> {
    this.anonymousId = undefined;
    await this.context.globalState.update(ANONYMOUS_ID_KEY, undefined);
  }

  dispose(): void {
    for (const request of this.requests) {
      request.destroy();
    }
    this.requests.clear();
  }
}
