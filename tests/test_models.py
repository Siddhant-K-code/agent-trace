"""Tests for trace data model."""

import ctypes
import json
import math
import sys
import os
import threading
import time
import unittest
from datetime import datetime
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.models import EventType, SessionMeta, TraceEvent, usable_time

NAN = float("nan")
INF = float("inf")


def _refuse_constant(token):
    raise ValueError(f"not JSON: {token}")


def strict_loads(text, **kwargs):
    """Parse text as JSON that has no bare NaN, Infinity or -Infinity."""
    return json.loads(text, parse_constant=_refuse_constant, **kwargs)


class TestTraceEvent(unittest.TestCase):
    def test_create_tool_call_event(self):
        event = TraceEvent(
            event_type=EventType.TOOL_CALL,
            session_id="abc123",
            data={"tool_name": "read_file", "arguments": {"path": "/tmp/test.py"}},
        )
        self.assertEqual(event.event_type, EventType.TOOL_CALL)
        self.assertEqual(event.data["tool_name"], "read_file")
        self.assertIsNotNone(event.event_id)
        self.assertIsNotNone(event.timestamp)

    def test_serialize_roundtrip(self):
        event = TraceEvent(
            event_type=EventType.TOOL_CALL,
            session_id="abc123",
            data={"tool_name": "search", "arguments": {"query": "test"}},
        )
        json_str = event.to_json()
        parsed = TraceEvent.from_json(json_str)
        self.assertEqual(parsed.event_type, EventType.TOOL_CALL)
        self.assertEqual(parsed.data["tool_name"], "search")
        self.assertEqual(parsed.session_id, "abc123")

    def test_json_omits_empty_fields(self):
        event = TraceEvent(
            event_type=EventType.ERROR,
            data={"message": "something broke"},
        )
        json_str = event.to_json()
        d = json.loads(json_str)
        self.assertNotIn("parent_id", d)
        self.assertNotIn("duration_ms", d)
        self.assertNotIn("session_id", d)

    def test_all_event_types_valid(self):
        for et in EventType:
            event = TraceEvent(event_type=et, data={})
            json_str = event.to_json()
            parsed = TraceEvent.from_json(json_str)
            self.assertEqual(parsed.event_type, et)


class TestSessionMeta(unittest.TestCase):
    def test_create_session(self):
        meta = SessionMeta(agent_name="test-agent")
        self.assertIsNotNone(meta.session_id)
        self.assertEqual(meta.agent_name, "test-agent")
        self.assertEqual(meta.tool_calls, 0)

    def test_serialize_roundtrip(self):
        meta = SessionMeta(
            agent_name="test",
            tool_calls=5,
            llm_requests=3,
            errors=1,
        )
        json_str = meta.to_json()
        parsed = SessionMeta.from_json(json_str)
        self.assertEqual(parsed.agent_name, "test")
        self.assertEqual(parsed.tool_calls, 5)
        self.assertEqual(parsed.llm_requests, 3)
        self.assertEqual(parsed.errors, 1)


class TestForwardCompatibleReaders(unittest.TestCase):
    """Fields added by newer writers must not break older readers."""

    def test_session_meta_ignores_unknown_fields(self):
        text = json.dumps({
            "session_id": "fwdcompat0000001",
            "agent_name": "claude-code",
            "field_from_a_future_release": {"nested": True},
        })
        meta = SessionMeta.from_json(text)
        self.assertEqual(meta.session_id, "fwdcompat0000001")
        self.assertFalse(hasattr(meta, "field_from_a_future_release"))

    def test_trace_event_ignores_unknown_fields(self):
        line = json.dumps({
            "event_type": "user_prompt",
            "event_id": "evt1",
            "data": {"prompt": "hi"},
            "field_from_a_future_release": "x",
        })
        event = TraceEvent.from_json(line)
        self.assertEqual(event.event_id, "evt1")
        self.assertEqual(event.data, {"prompt": "hi"})

    def test_rewrite_keeps_unknown_fields(self):
        # update_meta and tag_session re-serialise loaded records; a reader
        # that does not know a field must not erase it.
        meta = SessionMeta.from_json(json.dumps({
            "session_id": "fwdcompat0000002",
            "field_from_a_future_release": "kept",
        }))
        meta.tool_calls = 3
        written = json.loads(meta.to_json())
        self.assertEqual(written["field_from_a_future_release"], "kept")
        self.assertEqual(written["tool_calls"], 3)

        event = TraceEvent.from_json(json.dumps({
            "event_type": "user_prompt",
            "field_from_a_future_release": "kept",
        }))
        self.assertEqual(json.loads(event.to_json())["field_from_a_future_release"], "kept")

    def test_records_without_unknown_fields_serialise_unchanged(self):
        meta = SessionMeta.from_json(json.dumps({"session_id": "fwdcompat0000003"}))
        self.assertNotIn("_unknown_fields", json.loads(meta.to_json()))

    def test_empty_conversation_id_is_not_written(self):
        # v0.106.0 and older readers reject the key, so sessions without a
        # native ID must not carry it.
        meta = SessionMeta(session_id="fwdcompat0000004", conversation_id="")
        self.assertNotIn("conversation_id", json.loads(meta.to_json()))
        meta.conversation_id = "native-session-id"
        self.assertEqual(json.loads(meta.to_json())["conversation_id"], "native-session-id")


class _Real(float):
    """A float subclass, as numpy's float64 is."""


def _event(**kwargs):
    kwargs.setdefault("event_type", EventType.TOOL_CALL)
    kwargs.setdefault("timestamp", 1.0)
    kwargs.setdefault("event_id", "e1")
    kwargs.setdefault("session_id", "s1")
    kwargs.setdefault("tenant_id", "")
    return TraceEvent(**kwargs)


def _stop_thread(thread):
    """Raise SystemExit in a thread that is still running, so it stops using memory."""
    api = getattr(ctypes, "pythonapi", None)
    if api is not None:
        api.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread.ident), ctypes.py_object(SystemExit))
        thread.join(5)


class TestStrictJson(unittest.TestCase):
    """to_json writes a NaN or infinite float as null, as JSON.stringify does."""

    def test_non_finite_floats_are_null_at_any_depth_and_the_event_is_unchanged(self):
        deep = [{"y": -INF}, (INF, 1.5)]
        data = {"x": NAN, "deep": deep, "ok": 2.0}
        event = _event(data=data)
        event._unknown_fields = {"future": {"z": [NAN]}}
        before = repr(event), repr(event._unknown_fields)

        written = strict_loads(event.to_json())

        self.assertEqual(
            written["data"], {"x": None, "deep": [{"y": None}, [None, 1.5]], "ok": 2.0}
        )
        self.assertEqual(written["future"], {"z": [None]})
        self.assertEqual((repr(event), repr(event._unknown_fields)), before)
        self.assertIs(event.data, data)
        self.assertIs(data["deep"], deep)
        self.assertIsInstance(deep[1], tuple)

    def test_unknown_fields_are_written_as_strict_json(self):
        event = TraceEvent.from_json(
            '{"event_type":"user_prompt","timestamp":1.0,"event_id":"e1",'
            '"future":{"v":NaN,"w":[Infinity,-Infinity,1]}}'
        )
        self.assertEqual(
            strict_loads(event.to_json())["future"], {"v": None, "w": [None, None, 1]}
        )

    def test_a_float_subclass_is_a_float(self):
        event = _event(
            data={"x": _Real("nan"), "y": [_Real("inf")], "ok": _Real(1.5), _Real("-inf"): "key"},
            duration_ms=_Real("-inf"),
        )
        event._unknown_fields = {"future": _Real("-inf")}

        written = strict_loads(event.to_json())

        self.assertEqual(
            written["data"], {"x": None, "y": [None], "ok": 1.5, "-Infinity": "key"}
        )
        self.assertIsNone(written["future"])
        self.assertNotIn("duration_ms", written)

    def test_a_non_finite_dict_key_keeps_its_text(self):
        event = _event(data={NAN: 1, INF: 2, -INF: 3, "NaN": 4, "v": NAN})

        pairs = strict_loads(event.to_json(), object_pairs_hook=list)

        self.assertEqual(
            dict(pairs)["data"],
            [("NaN", 1), ("Infinity", 2), ("-Infinity", 3), ("NaN", 4), ("v", None)],
        )

    def test_one_list_in_two_places_is_cleaned_in_both(self):
        shared = [NAN, 1.0]
        event = _event(data={})
        event._unknown_fields = {"a": shared, "b": {"c": shared, "d": (shared, shared)}}

        written = strict_loads(event.to_json())

        self.assertEqual(written["a"], [None, 1.0])
        self.assertEqual(written["b"], {"c": [None, 1.0], "d": [[None, 1.0], [None, 1.0]]})

    def test_a_nan_900_levels_deep_is_written_as_null(self):
        deep = [NAN]
        for _ in range(899):
            deep = [deep]
        event = _event(data={})
        event._unknown_fields = {"deep": deep}

        value = strict_loads(event.to_json())["deep"]

        for _ in range(899):
            (value,) = value
        self.assertEqual(value, [None])

    def test_a_non_finite_duration_is_left_out(self):
        for duration in (NAN, INF, -INF):
            with self.subTest(duration_ms=duration):
                written = strict_loads(_event(duration_ms=duration, data={"x": 1}).to_json())
                self.assertNotIn("duration_ms", written)
                self.assertEqual(written["data"], {"x": 1})

    def test_a_non_finite_timestamp_is_written_as_null(self):
        # Left out, it would read back as the current time.
        for timestamp in (NAN, INF, -INF):
            with self.subTest(timestamp=timestamp):
                text = _event(timestamp=timestamp).to_json()
                written = strict_loads(text)
                self.assertIn("timestamp", written)
                self.assertIsNone(written["timestamp"])
                read_back = TraceEvent.from_json(text).timestamp
                self.assertIsNone(read_back)
                self.assertIsNone(usable_time(read_back))

    def test_circular_data_still_raises(self):
        # The NaN comes first, so json.dumps refuses it before it meets the
        # cycle, and the cleaner meets the cycle itself.
        loop = [NAN]
        loop.append(loop)
        event = _event(data={})
        event._unknown_fields = {"loop": loop}
        raised = []

        def write():
            try:
                event.to_json()
            except Exception as exc:
                raised.append(exc)

        thread = threading.Thread(target=write, daemon=True)
        thread.start()
        thread.join(5)
        if thread.is_alive():
            _stop_thread(thread)
            self.fail("to_json did not stop on circular data")
        self.assertEqual([(type(exc), str(exc)) for exc in raised],
                         [(ValueError, "Circular reference detected")])

        loop.pop()
        self.assertEqual(strict_loads(event.to_json())["loop"], [None])

    def test_other_values_pass_through_and_unwritable_values_still_raise(self):
        data = {
            "x": NAN, "s": "text \u2028", "i": 10**30, "f": -0.5, "b": False,
            "n": None, "e": [], "o": {}, "t": (1, "two"),
        }
        self.assertEqual(strict_loads(_event(data=data).to_json())["data"], {
            "x": None, "s": "text \u2028", "i": 10**30, "f": -0.5, "b": False,
            "n": None, "e": [], "o": {}, "t": [1, "two"],
        })

        with self.assertRaises(TypeError) as expected:
            json.dumps({1, 2})
        with self.assertRaises(TypeError) as caught:
            _event(data={"x": NAN, "s": {1, 2}}).to_json()
        self.assertEqual(str(caught.exception), str(expected.exception))

    def test_an_event_without_nan_or_infinity_is_written_as_before(self):
        # Passes on releases before strict JSON too. It guards their bytes.
        event = _event(
            event_type=EventType.TOOL_RESULT,
            timestamp=1700000000.25,
            event_id="e2",
            parent_id="e1",
            duration_ms=12.5,
            data={
                "tool_name": "Bash",
                "result": "ok \u2028 \u00e9",
                "nested": {"list": [1, -2.5, 1e300, None, True, (3, 4.0)], 1.5: "float key", "e": {}},
            },
            redacted=True,
            prev_hash="ab" * 32,
            tenant_id="acme",
        )
        event._unknown_fields = {"future": [1, {"a": 0.1}], "event_type": "ignored"}

        self.assertEqual(event.to_json(), (
            '{"event_type":"tool_result","timestamp":1700000000.25,"event_id":"e2",'
            '"session_id":"s1","parent_id":"e1","duration_ms":12.5,"data":{"tool_name":"Bash",'
            '"result":"ok \\u2028 \\u00e9","nested":{"list":[1,-2.5,1e+300,null,true,[3,4.0]],'
            '"1.5":"float key","e":{}}},"prev_hash":"' + "ab" * 32 + '","redacted":true,'
            '"tenant_id":"acme","future":[1,{"a":0.1}]}'
        ))


# POSIX TZ strings, so no tz database is needed. Each maps to its UTC offset in
# seconds on 1970-01-01.
ZONES = {
    "UTC0": 0,
    "<-12>12": -12 * 3600,
    "<+14>-14": 14 * 3600,
    "<+0530>-5:30": 5 * 3600 + 30 * 60,
    "PST8PDT,M3.2.0,M11.1.0": -8 * 3600,
}


def use_time_zone(test: unittest.TestCase, zone: str) -> None:
    """Set this process's time zone to *zone*, a key of ZONES, until *test* ends."""
    if not hasattr(time, "tzset"):
        test.skipTest("time.tzset is not available on this platform")
    old = os.environ.get("TZ")

    def restore():
        if old is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old
        time.tzset()

    test.addCleanup(restore)
    os.environ["TZ"] = zone
    time.tzset()
    test.assertEqual(time.localtime(0).tm_gmtoff, ZONES[zone], f"TZ={zone!r} did not apply")


class _LocalTimeFails:
    """A datetime whose local conversion raises, as Windows can near the year 3000."""

    @staticmethod
    def fromtimestamp(timestamp, tz=None):
        if tz is None:
            raise OSError(22, "Invalid argument")
        return datetime.fromtimestamp(timestamp, tz)


class _UtcFails:
    """A datetime whose UTC conversion raises and whose local conversion works."""

    @staticmethod
    def fromtimestamp(timestamp, tz=None):
        if tz is not None:
            raise OSError(22, "Invalid argument")
        return datetime.fromtimestamp(timestamp)


class TestUsableTime(unittest.TestCase):
    """usable_time() takes a Unix time in seconds up to 9999-12-30T00:00:00Z."""

    def test_the_limit_is_the_same_in_every_time_zone(self):
        limit = 253402128000.0  # 9999-12-30T00:00:00Z, the limit in the docs
        refused = (
            math.nextafter(limit, math.inf),
            253402300799.0,  # 9999-12-31T23:59:59Z
            253402300800.5,  # 10000-01-01T00:00:00.5Z
            "1700000000", True, None, NAN, -1, 10**400,
        )
        for zone in ZONES:
            use_time_zone(self, zone)
            for value in (0, limit):
                with self.subTest(zone=zone, value=value):
                    result = usable_time(value)
                    self.assertEqual(result, value)
                    self.assertIs(type(result), float)
            for value in refused:
                with self.subTest(zone=zone, value=value):
                    self.assertIsNone(usable_time(value))

    def test_a_time_that_local_or_utc_conversion_refuses_is_not_usable(self):
        # On Windows the C library has a lower limit, near the end of the year
        # 3000. The local and the UTC conversions can fail at different times.
        value = 1_700_000_000.0
        self.assertEqual(usable_time(value), value)
        for stub in (_LocalTimeFails, _UtcFails):
            with self.subTest(stub=stub.__name__), mock.patch("agent_trace.models.datetime", stub):
                self.assertIsNone(usable_time(value))


if __name__ == "__main__":
    unittest.main()
