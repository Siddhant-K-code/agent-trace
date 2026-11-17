"""Tests for replay annotations (issue #26)."""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

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
from agent_trace.store import TraceStore

try:
    import _escapes
except ImportError:
    from tests import _escapes


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
    """An annotations file can hold any JSON value in each field. annotate
    --list shows escape codes and line breaks in them as text."""

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


if __name__ == "__main__":
    unittest.main()
