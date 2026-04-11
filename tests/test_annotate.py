"""Tests for replay annotations (issue #26)."""

import io
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.annotate import (
    Annotation,
    _parse_offset,
    add_annotation,
    delete_annotation,
    format_annotations,
    load_annotations,
)
from agent_trace.models import SessionMeta
from agent_trace.store import TraceStore


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
        a = Annotation(event_id="ev-001", note="test note", label="root-cause")
        restored = Annotation.from_json(a.to_json())
        self.assertEqual(restored.event_id, "ev-001")
        self.assertEqual(restored.note, "test note")
        self.assertEqual(restored.label, "root-cause")

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
        anns = [Annotation(event_id="ev-001", note="root cause", label="root-cause")]
        buf = io.StringIO()
        format_annotations(anns, out=buf)
        output = buf.getvalue()
        self.assertIn("root cause", output)
        self.assertIn("root-cause", output)


if __name__ == "__main__":
    unittest.main()
