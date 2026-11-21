import * as assert from "node:assert/strict";
import * as fs from "node:fs";
import * as path from "node:path";
import { test } from "node:test";
import * as vm from "node:vm";
import { buildReviewModel, ReviewModel } from "./reviewModel";
import { SessionMeta, TraceEvent } from "./traceStore";

// Runs the inline script of media/panel.html against a small fake DOM, so
// the webview's message handling can be tested without a browser.

type Listener = (event: unknown) => void;

class FakeElement {
  children: FakeElement[] = [];
  style: Record<string, string> = {};
  dataset: Record<string, string> = {};
  className = "";
  textContent = "";
  value = "";
  disabled = false;
  open = false;
  tabIndex = 0;
  listeners: Record<string, Listener[]> = {};

  constructor(readonly tagName: string) {}

  get childNodes(): FakeElement[] {
    return this.children;
  }
  get classList() {
    const names = () => this.className.split(" ").filter(Boolean);
    return {
      add: (name: string) => {
        if (!names().includes(name)) this.className = [...names(), name].join(" ");
      },
      remove: (name: string) => {
        this.className = names().filter((n) => n !== name).join(" ");
      },
      contains: (name: string) => names().includes(name),
    };
  }
  append(...nodes: FakeElement[]): void {
    this.children.push(...nodes);
  }
  appendChild(child: FakeElement): FakeElement {
    this.children.push(child);
    return child;
  }
  replaceChildren(...nodes: FakeElement[]): void {
    this.children = nodes;
  }
  addEventListener(type: string, listener: Listener): void {
    (this.listeners[type] ??= []).push(listener);
  }
  fire(type: string, event: unknown = {}): void {
    for (const listener of this.listeners[type] ?? []) listener(event);
  }
  descendants(): FakeElement[] {
    return this.children.flatMap((child) => [child, ...child.descendants()]);
  }
  querySelector(tag: string): FakeElement | null {
    return this.descendants().find((element) => element.tagName === tag) ?? null;
  }
  closest(): null {
    return null;
  }
  focus(): void {}
  scrollIntoView(): void {}
}

class FakeButton extends FakeElement {}
class FakeInput extends FakeElement {}

function loadPanel() {
  const html = fs.readFileSync(path.resolve(__dirname, "../media/panel.html"), "utf8");
  const script = /<script nonce="[^"]*">([\s\S]*?)<\/script>/.exec(html)?.[1];
  assert.ok(script, "panel.html has an inline script");

  const byId = new Map<string, FakeElement>();
  const element = (id: string) => {
    if (!byId.has(id)) byId.set(id, new (id === "search" ? FakeInput : FakeElement)("div"));
    return byId.get(id)!;
  };
  const documentListeners: Record<string, Listener[]> = {};
  const windowListeners: Record<string, Listener[]> = {};
  const saved: unknown[] = [];
  const document = {
    getElementById: element,
    createElement: (tag: string) => new (tag === "button" ? FakeButton : FakeElement)(tag),
    addEventListener: (type: string, listener: Listener) => {
      (documentListeners[type] ??= []).push(listener);
    },
    querySelectorAll: (selector: string) => {
      const wanted = selector.split(".").filter(Boolean);
      return [...byId.values()]
        .flatMap((root) => [root, ...root.descendants()])
        .filter((el) => wanted.every((name) => el.classList.contains(name)));
    },
  };
  vm.runInNewContext(script, {
    acquireVsCodeApi: () => ({
      getState: () => undefined,
      // Copied so the object compares equal outside the vm's realm.
      setState: (value: unknown) => saved.push(JSON.parse(JSON.stringify(value))),
      postMessage: () => undefined,
    }),
    document,
    window: {
      addEventListener: (type: string, listener: Listener) => {
        (windowListeners[type] ??= []).push(listener);
      },
    },
    HTMLElement: FakeElement,
    HTMLButtonElement: FakeButton,
    HTMLInputElement: FakeInput,
  });
  return {
    element,
    saved,
    post(message: unknown) {
      for (const listener of windowListeners.message ?? []) listener({ data: message });
    },
    key(key: string) {
      const event = { key, target: new FakeElement("div"), preventDefault: () => undefined };
      for (const listener of documentListeners.keydown ?? []) listener(event);
    },
    selected(): string[] {
      return element("events").children
        .filter((card) => card.classList.contains("selected"))
        .map((card) => card.dataset.eventId);
    },
  };
}

function review(sessionId: string, eventIds: string[]): ReviewModel {
  const events: TraceEvent[] = eventIds.map((eventId, index) => ({
    event_type: "tool_call",
    timestamp: index + 1,
    event_id: eventId,
    session_id: sessionId,
    data: { tool_name: "Bash", arguments: { command: "echo " + eventId } },
  }));
  const meta: SessionMeta = {
    session_id: sessionId,
    started_at: 1,
    tool_calls: events.length,
    errors: 0,
    total_tokens: 0,
  };
  return buildReviewModel(meta, events, []);
}

const controls = { live: true, paused: false, activeTool: null, cost: null };

test("reset clears the previous session's rows, selection, filters and error", () => {
  const panel = loadPanel();
  panel.post({ type: "review", model: review("first", ["a1", "a2", "a3"]), controls });
  panel.key("j");
  panel.key("j");
  assert.deepEqual(panel.selected(), ["a2"]);
  const search = panel.element("search");
  search.value = "echo";
  search.fire("input");
  panel.post({ type: "error", message: "stale error" });
  panel.post({ type: "sessionEnd" });

  panel.post({ type: "reset" });

  assert.equal(panel.element("events").children.length, 0);
  assert.equal(panel.element("filter-chips").children.length, 0);
  assert.equal(panel.element("count").textContent, "");
  assert.equal(panel.element("error").style.display, "none");
  assert.equal(panel.element("error").textContent, "");
  assert.equal(panel.element("ended-banner").style.display, "none");
  assert.equal(search.value, "");
  assert.deepEqual(panel.saved.at(-1), { filter: "all", search: "", selectedEventId: null });
});

test("the review after a reset renders from a clean slate", () => {
  const panel = loadPanel();
  panel.post({ type: "review", model: review("first", ["a1", "a2", "a3"]), controls });
  panel.key("j");
  panel.key("j");
  panel.key("j");
  assert.deepEqual(panel.selected(), ["a3"]);
  const search = panel.element("search");
  search.value = "echo a3";
  search.fire("input");

  panel.post({ type: "reset" });
  panel.post({ type: "review", model: review("second", ["b1", "b2"]), controls });

  const cards = panel.element("events").children.map((card) => card.dataset.eventId);
  assert.deepEqual(cards, ["b1", "b2"]);
  assert.deepEqual(panel.selected(), []);
  assert.equal(panel.element("count").textContent, "2 of 2 evidence items");
  // The selection starts again at the first row, not where the old one was.
  panel.key("j");
  assert.deepEqual(panel.selected(), ["b1"]);
});

test("reset clears the previous session's header and hides its actions", () => {
  const panel = loadPanel();
  panel.post({ type: "review", model: review("first", ["a1"]), controls });
  assert.equal(panel.element("actions").style.display, "");
  assert.equal(panel.element("title").textContent, "Session first");

  panel.post({ type: "reset" });

  assert.equal(panel.element("title").textContent, "No session selected");
  assert.equal(panel.element("fact-start").textContent, "Not recorded");
  assert.equal(panel.element("health").textContent, "unknown");
  assert.equal(panel.element("actions").style.display, "none");
  assert.equal(panel.element("verify-capture").style.display, "none");
  assert.equal(panel.element("pause").disabled, true);
  assert.equal(panel.element("gap-list").children.length, 0);

  panel.post({ type: "review", model: review("second", ["b1"]), controls });
  assert.equal(panel.element("actions").style.display, "");
});
