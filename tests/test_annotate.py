"""Tests for replay annotations (issue #26)."""

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.annotate import (
    Annotation,
    _parse_offset,
    add_annotation,
    delete_annotation,
    filter_annotations,
    format_annotations,
    load_annotations,
    cmd_annotate,
)
from agent_trace.cli import build_parser
from agent_trace.models import SessionMeta
from agent_trace.share import cmd_share
from agent_trace.store import TraceStore

try:
    import _escapes
except ImportError:
    from tests import _escapes

try:  # keeps product telemetry off in this process and its children
    import _no_telemetry
except ImportError:  # pragma: no cover - run as a package module
    from tests import _no_telemetry


def _make_store():
    tmpdir = tempfile.mkdtemp()
    store = TraceStore(tmpdir)
    meta = SessionMeta(agent_name="test")
    store.create_session(meta)
    return store, meta.session_id


class TestParseOffset(unittest.TestCase):
    def test_seconds(self):
        self.assertAlmostEqual(_parse_offset("30s"), 30.0)

    def test_minutes_seconds(self):
        self.assertAlmostEqual(_parse_offset("2m14s"), 134.0)

    def test_colon_format(self):
        self.assertAlmostEqual(_parse_offset("1:30"), 90.0)

    def test_plain_number(self):
        self.assertAlmostEqual(_parse_offset("45"), 45.0)

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            _parse_offset("not-a-time")

    def test_offset_that_is_not_finite_raises(self):
        nines = "9" * 400
        for text in ("inf", "-inf", "nan", "1e400", nines + "m", nines + "s", nines + ":00"):
            with self.subTest(text.replace(nines, "<400 nines>")):
                with self.assertRaisesRegex(ValueError, "^Cannot parse offset: "):
                    _parse_offset(text)


class TestAnnotationRoundTrip(unittest.TestCase):
    def test_to_from_json(self):
        a = Annotation(
            event_id="ev-001",
            note="test note",
            label="root-cause",
            shareable=True,
        )
        restored = Annotation.from_json(a.to_json())
        self.assertEqual(restored.event_id, "ev-001")
        self.assertEqual(restored.note, "test note")
        self.assertEqual(restored.label, "root-cause")
        self.assertTrue(restored.shareable)

    def test_legacy_annotation_defaults_to_not_shareable(self):
        restored = Annotation.from_json(
            '{"annotation_id":"legacy","session_id":"s","note":"private"}'
        )
        self.assertFalse(restored.shareable)

    def test_annotation_id_generated(self):
        a = Annotation()
        self.assertTrue(a.annotation_id)
        self.assertEqual(len(a.annotation_id), 12)


class TestAddLoadAnnotations(unittest.TestCase):
    def test_add_and_load(self):
        store, sid = _make_store()
        ann = Annotation(event_id="ev-001", note="hello", label="decision")
        add_annotation(store, sid, ann)

        loaded = load_annotations(store, sid)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].note, "hello")
        self.assertEqual(loaded[0].label, "decision")

    def test_load_empty(self):
        store, sid = _make_store()
        loaded = load_annotations(store, sid)
        self.assertEqual(loaded, [])

    def test_multiple_annotations(self):
        store, sid = _make_store()
        for i in range(3):
            add_annotation(store, sid, Annotation(note=f"note {i}"))
        loaded = load_annotations(store, sid)
        self.assertEqual(len(loaded), 3)

    def test_delete_annotation(self):
        store, sid = _make_store()
        ann = Annotation(note="to delete")
        add_annotation(store, sid, ann)

        found = delete_annotation(store, sid, ann.annotation_id)
        self.assertTrue(found)

        loaded = load_annotations(store, sid)
        self.assertEqual(len(loaded), 0)

    def test_delete_nonexistent(self):
        store, sid = _make_store()
        found = delete_annotation(store, sid, "nonexistent-id")
        self.assertFalse(found)


class TestFormatAnnotations(unittest.TestCase):
    def test_format_empty(self):
        buf = io.StringIO()
        format_annotations([], out=buf)
        self.assertIn("No annotations", buf.getvalue())

    def test_format_with_annotations(self):
        anns = [
            Annotation(
                event_id="ev-001",
                note="root cause",
                label="root-cause",
                shareable=True,
            )
        ]
        buf = io.StringIO()
        format_annotations(anns, out=buf)
        output = buf.getvalue()
        self.assertIn("root cause", output)
        self.assertIn("root-cause", output)
        self.assertIn("shareable", output)

    def test_shareable_line_needs_true(self):
        self.assertIn("\n    shareable\n", _render(_annotation(shareable=True)))
        self.assertNotIn("shareable", _render(_annotation(shareable=1)))

    def test_created_at_that_localtime_rejects_prints_as_a_number(self):
        for created_at, expected in ((1e300, "1e+300"), (float("nan"), "nan")):
            with self.subTest(created_at):
                output = _render(_annotation(created_at=created_at))
                self.assertIn(f"\n  plain  {expected}  [plain]  event=plain\n", output)


class TestFilterAnnotations(unittest.TestCase):
    def _make_annotations(self):
        import time
        now = time.time()
        return [
            Annotation(event_id="e1", label="root-cause", author="alice", note="a", created_at=now - 86400 * 10),
            Annotation(event_id="e2", label="decision",   author="bob",   note="b", created_at=now - 86400 * 3),
            Annotation(event_id="e3", label="root-cause", author="bob",   note="c", created_at=now - 86400 * 1),
        ]

    def test_filter_by_label(self):
        anns = self._make_annotations()
        result = filter_annotations(anns, label="root-cause")
        self.assertEqual(len(result), 2)
        self.assertTrue(all(a.label == "root-cause" for a in result))

    def test_filter_by_author(self):
        anns = self._make_annotations()
        result = filter_annotations(anns, author="bob")
        self.assertEqual(len(result), 2)
        self.assertTrue(all(a.author == "bob" for a in result))

    def test_filter_by_since(self):
        import time
        anns = self._make_annotations()
        since = time.time() - 86400 * 5  # last 5 days
        result = filter_annotations(anns, since=since)
        self.assertEqual(len(result), 2)

    def test_filter_combined(self):
        import time
        anns = self._make_annotations()
        since = time.time() - 86400 * 5
        result = filter_annotations(anns, label="root-cause", author="bob", since=since)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].event_id, "e3")

    def test_no_filter_returns_all(self):
        anns = self._make_annotations()
        result = filter_annotations(anns)
        self.assertEqual(len(result), 3)


class TestShareableAnnotationCli(unittest.TestCase):
    def test_shareable_flag_is_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = TraceStore(tmp)
            meta = SessionMeta(session_id="shareable-annotation", agent_name="test")
            store.create_session(meta)
            args = build_parser().parse_args([
                "--trace-dir",
                tmp,
                "annotate",
                meta.session_id,
                "--note",
                "safe to share",
                "--shareable",
            ])
            with redirect_stdout(io.StringIO()):
                self.assertEqual(cmd_annotate(args), 0)
            annotations = load_annotations(store, meta.session_id)
            self.assertEqual(len(annotations), 1)
            self.assertTrue(annotations[0].shareable)


class TestWatchPolicyFlag(unittest.TestCase):
    """Verify --policy is wired into WatcherConfig."""

    def test_policy_path_passed_to_config(self):
        import argparse
        from agent_trace.watch import WatcherConfig
        # Simulate what cmd_watch does when --policy is provided
        config = WatcherConfig(scope_policy="/tmp/my-policy.json")
        self.assertEqual(config.scope_policy, "/tmp/my-policy.json")

    def test_default_policy_path(self):
        from agent_trace.watch import WatcherConfig
        config = WatcherConfig()
        self.assertEqual(config.scope_policy, ".agent-scope.json")


def _annotation(**fields):
    values = dict(annotation_id="plain", event_id="plain", label="plain", note="plain",
                  author="plain", created_at=0.0)
    values.update(fields)
    return Annotation(**values)


def _render(annotation):
    buf = io.StringIO()
    format_annotations([annotation], out=buf)
    return buf.getvalue()


# (label, kind, build): build(value) returns an annotation with value in the
# field the label names. The kinds are in tests/_escapes.py.
FORMAT_ANNOTATIONS_CASES = [
    ("annotation ID", "text", lambda v: _annotation(annotation_id=v)),
    ("label", "text", lambda v: _annotation(label=v)),
    ("event ID", "text", lambda v: _annotation(event_id=v)),
    ("note", "free", lambda v: _annotation(note=v)),
    ("author", "text", lambda v: _annotation(author=v)),
]


class TestFormatAnnotationsShowsEscapesAsText(unittest.TestCase):
    """Another tool can write any text into an annotations file. annotate
    --list shows escape codes and line breaks in each text field as text."""

    def test_each_value_shows_escapes_as_text(self):
        _escapes.check_each_value(self, FORMAT_ANNOTATIONS_CASES, _render)

    def test_plain_text_prints_unchanged(self):
        _escapes.check_plain_text_unchanged(self, FORMAT_ANNOTATIONS_CASES, _render)

    def test_whitespace_in_free_text_shows_as_one_space(self):
        _escapes.check_whitespace_in_free_text(self, FORMAT_ANNOTATIONS_CASES, _render)

    def test_sgr_code_shows_as_text(self):
        sgr = "a\x1b[8mhidden"
        output = _render(_annotation(annotation_id=sgr, label=sgr, event_id=sgr, note=sgr, author=sgr))
        self.assertEqual(output.count(r"a\x1b[8mhidden"), 5)
        self.assertNotIn("\x1b", output)

    def test_value_that_is_not_a_string_prints_as_before(self):
        cases = [
            ({"note": ["root", "cause"]}, "\n    ['root', 'cause']\n"),
            ({"note": 2.5}, "\n    2.5\n"),
            ({"author": 42}, "\n    42\n"),
            ({"label": ["a", 1]}, "  [['a', 1]]  event=plain\n"),
            ({"event_id": 7}, "  event=7\n"),
            ({"annotation_id": 12}, "\n  12  "),
        ]
        for fields, expected in cases:
            with self.subTest(fields):
                self.assertIn(expected, _render(_annotation(**fields)))

    def test_note_of_only_whitespace_keeps_its_line(self):
        expected = _render(_annotation(note="x")).replace("\n    x\n", "\n    \n")
        self.assertEqual(_render(_annotation(note=" \t\n ")), expected)

    def test_json_export_keeps_the_raw_text(self):
        payload = _escapes.PAYLOAD
        with tempfile.TemporaryDirectory() as tmp:
            store = TraceStore(tmp)
            store.create_session(SessionMeta(session_id="esc-annotate", agent_name="test"))
            add_annotation(store, "esc-annotate", Annotation(
                annotation_id=payload, label=payload, event_id=payload, note=payload, author=payload,
            ))
            args = build_parser().parse_args([
                "--trace-dir", tmp, "annotate", "esc-annotate", "--list", "--export-format", "json",
            ])
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(cmd_annotate(args), 0)
        [loaded] = json.loads(out.getvalue())
        for name in ("annotation_id", "label", "event_id", "note", "author"):
            with self.subTest(name):
                self.assertEqual(loaded[name], payload)


SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
SESSION = "raw-lines"

_GOOD_LINE = '{"annotation_id":"good","label":"fix","note":"GOOD-NOTE","shareable":true}'

# (name, line): load_annotations skips each line. Each one asks to be shared
# and has a BAD- marker, so any output with the marker let the line through.
_BAD_LINES = [
    ("not an object", '["BAD-NOT-OBJECT"]'),
    ("annotation_id is a number", '{"annotation_id":5,"note":"BAD-ID","shareable":true}'),
    ("session_id is a number", '{"session_id":5,"note":"BAD-SESSION","shareable":true}'),
    ("event_id is a number", '{"event_id":5,"note":"BAD-EVENT","shareable":true}'),
    ("label is a list", '{"label":["a"],"note":"BAD-LABEL","shareable":true}'),
    ("note is a number", '{"label":"BAD-NOTE","note":5,"shareable":true}'),
    ("author is a number", '{"author":5,"note":"BAD-AUTHOR","shareable":true}'),
    ("author is null", '{"author":null,"note":"BAD-AUTHOR-NULL","shareable":true}'),
    ("offset_seconds is text", '{"offset_seconds":"x","note":"BAD-OFFSET","shareable":true}'),
    ("offset_seconds is Infinity", '{"offset_seconds":Infinity,"note":"BAD-OFFSET-INF","shareable":true}'),
    ("offset_seconds is false", '{"offset_seconds":false,"note":"BAD-OFFSET-BOOL","shareable":true}'),
    ("created_at is text", '{"created_at":"x","note":"BAD-CREATED","shareable":true}'),
    ("created_at is NaN", '{"created_at":NaN,"note":"BAD-CREATED-NAN","shareable":true}'),
    ("created_at is true", '{"created_at":true,"note":"BAD-CREATED-BOOL","shareable":true}'),
    ("created_at is too big for a float",
     '{"created_at":1' + "0" * 400 + ',"note":"BAD-CREATED-HUGE","shareable":true}'),
    ("number with 5000 digits", '{"created_at":' + "9" * 5000 + ',"note":"BAD-DIGITS","shareable":true}'),
    ("1000000 levels deep",
     '{"label":"BAD-DEEP","note":' + "[" * 1_000_000 + "]" * 1_000_000 + ',"shareable":true}'),
    ("not valid UTF-8", '{"note":"BAD-UTF8-\udcff","shareable":true}'),
]


def _lines(*lines):
    # surrogateescape writes "\udcff" as the byte 0xff, which is not valid UTF-8.
    return "".join(line + "\n" for line in lines).encode("utf-8", "surrogateescape")


def _session_with_annotations(test, data):
    """Return (trace_dir, store, path) for a session whose annotations file holds *data*."""
    tmp = tempfile.TemporaryDirectory()
    test.addCleanup(tmp.cleanup)
    store = TraceStore(tmp.name)
    store.create_session(SessionMeta(session_id=SESSION, agent_name="test"))
    path = store.annotations_path(SESSION)
    path.write_bytes(data)
    return tmp.name, store, path


def _run(handler, trace_dir, *argv):
    args = build_parser().parse_args(["--trace-dir", trace_dir, *argv])
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = handler(args)
    return code, out.getvalue(), err.getvalue()


def _run_cli(trace_dir, *argv):
    """Run agent-strace in a child process.

    annotate --list prints through format_annotations, whose out= default is
    bound at import, so redirect_stdout does not capture that text.
    """
    env = _no_telemetry.without_telemetry({
        key: value
        for key, value in os.environ.items()
        if not key.startswith("AGENT_STRACE_") and key != "AGENT_TRACE_DIR"
    })
    env["PYTHONPATH"] = os.pathsep.join(item for item in (SRC, env.get("PYTHONPATH", "")) if item)
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "-m", "agent_trace.entry", "--trace-dir", trace_dir, *argv],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=120,
    )


class TestLoadSkipsLinesWithTheWrongShape(unittest.TestCase):
    """One bad line must not stop annotate --list or share from showing the
    other annotations."""

    def test_each_bad_line_is_skipped(self):
        for name, bad in _BAD_LINES:
            with self.subTest(name):
                trace_dir, store, _ = _session_with_annotations(self, _lines(bad, _GOOD_LINE))
                self.assertEqual([a.annotation_id for a in load_annotations(store, SESSION)], ["good"])
                for since in ([], ["--since", "3650d"]):
                    code, out, err = _run(cmd_annotate, trace_dir, "annotate", SESSION,
                                          "--list", "--export-format", "json", *since)
                    self.assertEqual(code, 0, err)
                    self.assertEqual([a["annotation_id"] for a in json.loads(out)], ["good"])
                code, out, err = _run(cmd_share, trace_dir, "share", SESSION, "--stdout")
                self.assertEqual(code, 0, err)
                self.assertIn("GOOD-NOTE", out)
                self.assertEqual(out.count('<li><a href="#ev-'), 1)
                self.assertEqual(out.count("BAD-"), 0)

    def test_annotate_list_prints_the_good_line(self):
        trace_dir, _, _ = _session_with_annotations(
            self, _lines(*(line for _, line in _BAD_LINES), _GOOD_LINE))
        result = _run_cli(trace_dir, "annotate", SESSION, "--list")
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertIn("\n1 annotation(s):\n", result.stdout)
        self.assertIn("\n    GOOD-NOTE\n", result.stdout)
        self.assertEqual(result.stdout.count("BAD-"), 0)


class TestOnlyJsonTrueMarksAnAnnotationShareable(unittest.TestCase):
    def test_other_values_load_as_not_shareable(self):
        values = ["false", 1, "true", "yes", 0, None, True]
        _, store, _ = _session_with_annotations(self, _lines(*(
            json.dumps({"annotation_id": f"a{i}", "shareable": value})
            for i, value in enumerate(values)
        )))
        loaded = load_annotations(store, SESSION)
        self.assertEqual([a.annotation_id for a in loaded], [f"a{i}" for i in range(len(values))])
        for annotation, value in zip(loaded, values):
            with self.subTest(value):
                self.assertIs(annotation.shareable, value is True)


class TestAnnotationsFileSplitsOnLfOnly(unittest.TestCase):
    def test_unicode_line_break_in_a_note_keeps_its_line(self):
        note = "one\u2028two\u2029three\x85four"
        aaa = json.dumps({"annotation_id": "aaa", "note": note}, ensure_ascii=False).encode("utf-8")
        _, store, path = _session_with_annotations(self, aaa + b'\n{"annotation_id":"bbb"}\r\n')
        loaded = load_annotations(store, SESSION)
        self.assertEqual([a.annotation_id for a in loaded], ["aaa", "bbb"])
        self.assertEqual(loaded[0].note, note)
        self.assertTrue(delete_annotation(store, SESSION, "bbb"))
        self.assertEqual(path.read_bytes(), aaa + b"\n")


class TestDeleteMatchesTheAnnotationId(unittest.TestCase):
    def test_delete_removes_a_line_that_load_skips(self):
        _, store, path = _session_with_annotations(self, _lines(
            '{"annotation_id":"keep1"}', '{"annotation_id":"bad1","note":5}', '{"annotation_id":"keep2"}'))
        self.assertTrue(delete_annotation(store, SESSION, "bad1"))
        self.assertEqual(path.read_bytes(), _lines('{"annotation_id":"keep1"}', '{"annotation_id":"keep2"}'))

    def test_delete_keeps_every_other_line_as_written(self):
        bad = _lines(*(line for _, line in _BAD_LINES))
        _, store, path = _session_with_annotations(self, bad + _lines(_GOOD_LINE))
        self.assertTrue(delete_annotation(store, SESSION, "good"))
        self.assertEqual(path.read_bytes(), bad)

    def test_delete_removes_a_line_that_is_not_utf8(self):
        _, store, path = _session_with_annotations(self, _lines(
            '{"annotation_id":"keep1"}', '{"annotation_id":"bad1","note":"x\udcff"}', '{"annotation_id":"keep2"}'))
        self.assertTrue(delete_annotation(store, SESSION, "bad1"))
        self.assertEqual(path.read_bytes(), _lines('{"annotation_id":"keep1"}', '{"annotation_id":"keep2"}'))


class TestAnnotateAtWritesALineThatLoads(unittest.TestCase):
    """annotate --at must not write a line that load_annotations skips."""

    def test_offset_that_is_not_finite_exits_1_and_writes_nothing(self):
        for at in ("inf", "nan", "1e400"):
            with self.subTest(at):
                trace_dir, _, path = _session_with_annotations(self, b"")
                code, out, err = _run(cmd_annotate, trace_dir, "annotate", SESSION, "--at", at, "--note", "N")
                self.assertEqual((code, out), (1, ""))
                self.assertEqual(err, f"Cannot parse offset: '{at}'\n")
                self.assertEqual(path.read_bytes(), b"")

    def test_event_id_that_is_not_a_string_is_left_empty(self):
        trace_dir, _, path = _session_with_annotations(self, b"")
        path.with_name("events.ndjson").write_bytes(
            b'{"event_type":"session_start","timestamp":1000.0,"event_id":5}\n'
            b'{"event_type":"tool_call","timestamp":1010.0,"event_id":"abc","data":{"tool_name":"t"}}\n'
        )
        code, _, err = _run(cmd_annotate, trace_dir, "annotate", SESSION, "--at", "0s", "--note", "N-AT0")
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(path.read_bytes())["event_id"], "")
        result = _run_cli(trace_dir, "annotate", SESSION, "--list")
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertIn("  offset=0.0s\n    N-AT0\n", result.stdout)


if __name__ == "__main__":
    unittest.main()
