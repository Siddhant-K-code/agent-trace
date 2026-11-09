import * as assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { test } from "node:test";
import { NdjsonTail, splitCompleteLines } from "./eventTail";

function tempFile(): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "agent-trace-tail-"));
  return path.join(dir, "events.ndjson");
}

test("splitCompleteLines keeps the unterminated remainder", () => {
  const { lines, remainder } = splitCompleteLines(
    Buffer.alloc(0),
    Buffer.from('{"a":1}\n\n{"b":2}\n{"c":'),
  );
  assert.deepEqual(lines, ['{"a":1}', '{"b":2}']);
  assert.equal(remainder.toString(), '{"c":');

  const next = splitCompleteLines(remainder, Buffer.from("3}\n"));
  assert.deepEqual(next.lines, ['{"c":3}']);
  assert.equal(next.remainder.length, 0);
});

test("splitCompleteLines joins a multi-byte character split across chunks", () => {
  const bytes = Buffer.from('{"s":"héllo ✓"}\n');
  const cut = bytes.indexOf(0xe2) + 1; // inside the 3-byte check mark
  const first = splitCompleteLines(Buffer.alloc(0), bytes.subarray(0, cut));
  assert.deepEqual(first.lines, []);
  const second = splitCompleteLines(first.remainder, bytes.subarray(cut));
  assert.deepEqual(second.lines, ['{"s":"héllo ✓"}']);
});

test("missing file yields nothing and is picked up once created", () => {
  const file = tempFile();
  const tail = new NdjsonTail(file);
  assert.deepEqual(tail.read(), { lines: [], reset: false });
  fs.writeFileSync(file, '{"n":1}\n');
  assert.deepEqual(tail.read().lines, ['{"n":1}']);
});

test("partial trailing line is delivered once completed", () => {
  const file = tempFile();
  const tail = new NdjsonTail(file);
  fs.writeFileSync(file, '{"n":1}\n{"n":');
  assert.deepEqual(tail.read().lines, ['{"n":1}']);
  assert.deepEqual(tail.read().lines, []);
  fs.appendFileSync(file, '2}\n{"n":3}\n');
  assert.deepEqual(tail.read().lines, ['{"n":2}', '{"n":3}']);
});

test("multi-byte UTF-8 split across appends is decoded intact", () => {
  const file = tempFile();
  const tail = new NdjsonTail(file);
  const bytes = Buffer.from('{"s":"日本"}\n');
  fs.writeFileSync(file, bytes.subarray(0, 8));
  assert.deepEqual(tail.read().lines, []);
  fs.appendFileSync(file, bytes.subarray(8));
  assert.deepEqual(tail.read().lines, ['{"s":"日本"}']);
});

test("shrinking file resets and rereads from the start", () => {
  const file = tempFile();
  const tail = new NdjsonTail(file);
  fs.writeFileSync(file, '{"n":1}\n{"n":2}\n{"n":3}\n');
  assert.equal(tail.read().lines.length, 3);

  fs.truncateSync(file, 0);
  fs.appendFileSync(file, '{"m":1}\n');
  assert.deepEqual(tail.read(), { lines: ['{"m":1}'], reset: true });
  assert.deepEqual(tail.read(), { lines: [], reset: false });
});

test("replaced file (new inode) resets even when it is larger", () => {
  const file = tempFile();
  const tail = new NdjsonTail(file);
  fs.writeFileSync(file, '{"n":1}\n');
  assert.equal(tail.read().lines.length, 1);

  const tmp = `${file}.tmp`;
  fs.writeFileSync(tmp, '{"m":1}\n{"m":2}\n{"m":3}\n');
  fs.renameSync(tmp, file);
  const result = tail.read();
  assert.equal(result.reset, true);
  assert.deepEqual(result.lines, ['{"m":1}', '{"m":2}', '{"m":3}']);
});

test("flush returns an unterminated final line once", () => {
  const file = tempFile();
  const tail = new NdjsonTail(file);
  fs.writeFileSync(file, '{"n":1}\n{"n":2}');
  assert.deepEqual(tail.read().lines, ['{"n":1}']);
  assert.equal(tail.flush(), '{"n":2}');
  assert.equal(tail.flush(), null);
});

test("reset starts over from the beginning", () => {
  const file = tempFile();
  const tail = new NdjsonTail(file);
  fs.writeFileSync(file, '{"n":1}\n');
  tail.read();
  tail.reset();
  assert.deepEqual(tail.read().lines, ['{"n":1}']);
});
