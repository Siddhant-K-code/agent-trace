import * as assert from "node:assert/strict";
import Module from "node:module";
import { test } from "node:test";
import * as vm from "node:vm";

// The live stream and post-mortem webviews use a CSP of script-src 'nonce-...',
// which blocks inline event handlers such as onclick="...". These tests check
// that the generated HTML has none and that the nonce'd script wires each
// control with addEventListener instead.

type Load = (request: string, parent: unknown, isMain: boolean) => unknown;
const loader = Module as unknown as { _load: Load };

async function loadRenderers() {
  const vscodeStub = { Disposable: class {} };
  const originalLoad = loader._load;
  loader._load = function (this: unknown, request, parent, isMain) {
    if (request === "vscode") {
      return vscodeStub;
    }
    return originalLoad.call(this, request, parent, isMain);
  };
  try {
    const live = await import("./liveStream");
    const pm = await import("./postMortem");
    return {
      liveStream: live.renderLiveStreamHtml,
      postMortem: pm.renderPostMortemHtml,
    };
  } finally {
    loader._load = originalLoad;
  }
}

const INLINE_HANDLER = /\son[a-z]+\s*=/i;
const NONCE = "testnonce";

type Listener = (event: unknown) => void;

class FakeElement {
  children: FakeElement[] = [];
  parent: FakeElement | null = null;
  style: Record<string, string> = {};
  dataset: Record<string, string> = {};
  className = "";
  value = "";
  scrollTop = 0;
  scrollHeight = 0;
  private text = "";
  listeners: Record<string, Listener[]> = {};

  constructor(readonly tagName: string) {}

  get textContent(): string {
    return this.text + this.children.map((c) => c.textContent).join("");
  }
  set textContent(value: string) {
    this.text = value;
    this.children = [];
  }
  set innerHTML(value: string) {
    assert.equal(value, "", "scripts only clear innerHTML");
    this.textContent = "";
  }
  get firstChild(): FakeElement | undefined {
    return this.children[0];
  }
  appendChild(child: FakeElement): FakeElement {
    child.parent = this;
    this.children.push(child);
    return child;
  }
  append(...nodes: (FakeElement | string)[]): void {
    for (const node of nodes) {
      if (typeof node === "string") {
        const textNode = new FakeElement("#text");
        textNode.text = node;
        this.appendChild(textNode);
      } else {
        this.appendChild(node);
      }
    }
  }
  removeChild(child: FakeElement): void {
    this.children = this.children.filter((c) => c !== child);
  }
  closest(selector: string): FakeElement | null {
    const cls = selector.replace(/^\./, "");
    if (this.className.split(" ").includes(cls)) return this;
    return this.parent ? this.parent.closest(selector) : null;
  }
  addEventListener(type: string, listener: Listener): void {
    (this.listeners[type] ??= []).push(listener);
  }
  fire(type: string, target: FakeElement = this): void {
    for (const listener of this.listeners[type] ?? []) listener({ target });
  }
  descendants(): FakeElement[] {
    return this.children.flatMap((c) => [c, ...c.descendants()]);
  }
}

class FakeEventSource {
  static last: FakeEventSource | null = null;
  onopen: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  constructor(readonly url: string) {
    FakeEventSource.last = this;
  }
  close(): void {}
}

function runWebview(html: string) {
  const script = /<script nonce="[^"]*">([\s\S]*?)<\/script>/.exec(html)?.[1];
  assert.ok(script, "webview has a nonce'd script");
  const markup = html.replace(/<script[\s\S]*?<\/script>/g, "");

  // Only elements present in the markup exist, as in a real webview.
  const byId = new Map<string, FakeElement>();
  for (const match of markup.matchAll(/<(\w+)[^>]*\sid="([^"]+)"[^>]*>([^<]*)/g)) {
    const el = new FakeElement(match[1]);
    el.textContent = match[3];
    byId.set(match[2], el);
  }
  const posted: unknown[] = [];
  vm.runInNewContext(script, {
    acquireVsCodeApi: () => ({
      postMessage: (msg: unknown) => posted.push(JSON.parse(JSON.stringify(msg))),
    }),
    document: {
      getElementById: (id: string) => byId.get(id) ?? null,
      createElement: (tag: string) => new FakeElement(tag),
      querySelectorAll: (selector: string) =>
        [...byId.values()]
          .flatMap((el) => [el, ...el.descendants()])
          .filter((el) => el.className.split(" ").includes(selector.replace(/^\./, ""))),
    },
    EventSource: FakeEventSource,
    setTimeout: () => 0,
  });
  const el = (id: string) => {
    const found = byId.get(id);
    assert.ok(found, `#${id} exists`);
    return found;
  };
  return { el, posted };
}

test("webview HTML has no inline event handler attributes", async () => {
  const render = await loadRenderers();
  const pages = {
    "live stream with endpoint": render.liveStream("http://localhost:4318", NONCE),
    "live stream without endpoint": render.liveStream("", NONCE),
    "post-mortem with recovery": render.postMortem(
      { session_id: "abc", recovery_context: "resume here" },
      NONCE,
    ),
    "post-mortem without recovery": render.postMortem({ session_id: "abc" }, NONCE),
  };
  for (const [name, html] of Object.entries(pages)) {
    assert.match(html, /script-src 'nonce-testnonce';/, `${name} keeps the nonce CSP`);
    assert.doesNotMatch(html, INLINE_HANDLER, `${name} has an inline handler`);
  }
});

test("live stream Open Settings button posts openSettings", async () => {
  const { liveStream } = await loadRenderers();
  const { el, posted } = runWebview(liveStream("", NONCE));
  el("settingsBtn").fire("click");
  assert.deepEqual(posted, [{ command: "openSettings" }]);
});

test("live stream Pause, Clear, filter and session links work", async () => {
  const { liveStream } = await loadRenderers();
  const { el, posted } = runWebview(liveStream("http://localhost:4318", NONCE));
  const source = FakeEventSource.last;
  assert.ok(source?.onmessage);
  const events = el("events");

  const sessionId = `x');alert(1);//"<b>`;
  source.onmessage({
    data: JSON.stringify({ timestamp: 1, event_type: "tool_call", session_id: sessionId, data: {} }),
  });
  assert.equal(events.children.length, 1);
  const link = events.descendants().find((c) => c.className === "session-link");
  assert.ok(link, "session link rendered");
  assert.equal(link.textContent, sessionId.slice(0, 8));
  events.fire("click", link);
  assert.deepEqual(posted, [{ command: "jumpToSession", sessionId }]);

  const filter = el("filter");
  filter.value = "llm";
  filter.fire("input");
  assert.equal(events.children[0].style.display, "none");

  el("pauseBtn").fire("click");
  assert.equal(el("pauseBtn").textContent, "Resume");
  source.onmessage({ data: JSON.stringify({ timestamp: 2, event_type: "error" }) });
  assert.equal(events.children.length, 1, "paused stream drops events");
  el("pauseBtn").fire("click");
  assert.equal(el("pauseBtn").textContent, "Pause");

  el("clearBtn").fire("click");
  assert.equal(events.children.length, 0);
});

test("post-mortem Copy recovery context button posts the text", async () => {
  const { postMortem } = await loadRenderers();
  const { el, posted } = runWebview(
    postMortem({ session_id: "abc", recovery_context: "resume here" }, NONCE),
  );
  el("copyBtn").fire("click");
  assert.deepEqual(posted, [{ command: "copyRecovery", text: "resume here" }]);
});
