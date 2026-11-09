/**
 * Incremental reader for an append-only NDJSON file.
 *
 * Offsets are byte positions. Lines are split on the newline byte before
 * UTF-8 decoding, so multi-byte characters are never split across reads and
 * an incomplete trailing line is kept until its newline arrives.
 *
 * Kept free of `vscode` imports so it can be unit tested with node:test.
 */

import * as fs from "fs";

const NEWLINE = 0x0a;
const EMPTY = Buffer.alloc(0);

export interface TailRead {
  /** Complete, non-blank lines appended since the previous read. */
  lines: string[];
  /** True when the file shrank or was replaced and reading restarted at 0. */
  reset: boolean;
}

/**
 * Split `pending + chunk` into complete lines and the unterminated remainder.
 */
export function splitCompleteLines(
  pending: Buffer,
  chunk: Buffer,
): { lines: string[]; remainder: Buffer } {
  const data = pending.length ? Buffer.concat([pending, chunk]) : chunk;
  const lines: string[] = [];
  let start = 0;
  let nl = data.indexOf(NEWLINE, start);
  while (nl !== -1) {
    const line = data.toString("utf8", start, nl);
    if (line.trim()) { lines.push(line); }
    start = nl + 1;
    nl = data.indexOf(NEWLINE, start);
  }
  const remainder = start >= data.length ? EMPTY : Buffer.from(data.subarray(start));
  return { lines, remainder };
}

export class NdjsonTail {
  private offset = 0;
  private pending: Buffer = EMPTY;
  private ino: number | null = null;

  constructor(readonly filePath: string) {}

  /** Forget all progress so the next read starts from the beginning. */
  reset(): void {
    this.offset = 0;
    this.pending = EMPTY;
    this.ino = null;
  }

  /** Read bytes appended since the last call. Missing file yields no lines. */
  read(): TailRead {
    let fd: number;
    try {
      fd = fs.openSync(this.filePath, "r");
    } catch {
      return { lines: [], reset: false };
    }
    let reset = false;
    try {
      const stat = fs.fstatSync(fd);
      const replaced = this.ino !== null && stat.ino !== 0 && stat.ino !== this.ino;
      if (stat.size < this.offset || replaced) {
        this.reset();
        reset = true;
      }
      this.ino = stat.ino;

      const toRead = stat.size - this.offset;
      if (toRead <= 0) { return { lines: [], reset }; }

      const chunk = Buffer.alloc(toRead);
      let got = 0;
      while (got < toRead) {
        const n = fs.readSync(fd, chunk, got, toRead - got, this.offset + got);
        if (n === 0) { break; }
        got += n;
      }
      this.offset += got;

      const { lines, remainder } = splitCompleteLines(this.pending, chunk.subarray(0, got));
      this.pending = remainder;
      return { lines, reset };
    } catch {
      return { lines: [], reset };
    } finally {
      fs.closeSync(fd);
    }
  }

  /** Return and clear any unterminated trailing line (used when a session ends). */
  flush(): string | null {
    const line = this.pending.toString("utf8");
    this.pending = EMPTY;
    return line.trim() ? line : null;
  }
}
