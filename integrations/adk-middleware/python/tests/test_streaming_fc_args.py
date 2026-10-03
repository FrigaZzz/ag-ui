"""Tests for streaming function call arguments (Mode A, google-adk >= 1.24.0).

These tests verify the EventTranslator correctly handles streaming function call
chunks from Gemini 3+ models when streaming_function_call_arguments=True.
"""

import json
import pytest
from unittest.mock import MagicMock

from ag_ui.core import EventType
from ag_ui_adk import EventTranslator, ADKAgent
from ag_ui_adk.config import PredictStateMapping


def _event_types(events):
    """Extract event type names from a list of events."""
    return [str(ev.type).split('.')[-1] for ev in events]


def _make_adk_event(
    func_calls=None,
    partial=False,
    author="assistant",
    lro_ids=None,
):
    """Create a mock ADK event with function calls."""
    event = MagicMock()
    event.author = author
    event.partial = partial
    event.content = MagicMock()
    event.content.parts = []
    event.get_function_calls = MagicMock(return_value=func_calls or [])
    event.long_running_tool_ids = lro_ids or []
    # get_function_responses should return empty by default
    event.get_function_responses = MagicMock(return_value=[])
    # Prevent MagicMock auto-creating truthy attributes for state/custom handlers
    event.actions = None
    event.custom_data = None
    return event


def _make_func_call(name=None, args=None, partial_args=None, will_continue=None, fc_id=None):
    """Create a mock FunctionCall."""
    fc = MagicMock()
    fc.name = name
    fc.id = fc_id or f"adk-{id(fc)}"
    fc.args = args
    fc.partial_args = partial_args
    fc.will_continue = will_continue
    return fc


def _make_partial_arg(json_path, string_value):
    """Create a mock PartialArg."""
    pa = MagicMock()
    pa.json_path = json_path
    pa.string_value = string_value
    return pa


async def _collect_events(translator, adk_event, thread_id="thread", run_id="run"):
    """Collect all events from a translator.translate() call."""
    events = []
    async for e in translator.translate(adk_event, thread_id, run_id):
        events.append(e)
    return events


async def _final(translator, name, args, fc_id="adk-final"):
    """Send the aggregated (non-partial) call that follows the streamed chunks."""
    fc = _make_func_call(name=name, args=args, fc_id=fc_id)
    return await _collect_events(translator, _make_adk_event(func_calls=[fc], partial=False))


def _args_json(events, tool_call_id=None):
    return "".join(
        e.delta for e in events
        if "TOOL_CALL_ARGS" in str(e.type)
        and (tool_call_id is None or e.tool_call_id == tool_call_id)
    )


# ============================================================================
# First chunk tests
# ============================================================================


@pytest.mark.asyncio
async def test_streaming_fc_first_chunk_emits_start():
    """First chunk with name + will_continue=True emits TOOL_CALL_START."""
    translator = EventTranslator(streaming_function_call_arguments=True)

    fc = _make_func_call(name="write_document", will_continue=True)
    adk_event = _make_adk_event(func_calls=[fc], partial=True)

    events = await _collect_events(translator, adk_event)
    types = _event_types(events)

    assert "TOOL_CALL_START" in types
    start_event = [e for e in events if "TOOL_CALL_START" in str(e.type)][0]
    assert start_event.tool_call_name == "write_document"
    assert start_event.tool_call_id is not None


@pytest.mark.asyncio
async def test_streaming_fc_disabled_by_default():
    """Without flag, partial events with will_continue are skipped."""
    translator = EventTranslator()  # Default: streaming_function_call_arguments=False

    fc = _make_func_call(name="write_document", will_continue=True)
    adk_event = _make_adk_event(func_calls=[fc], partial=True)

    events = await _collect_events(translator, adk_event)
    types = _event_types(events)

    assert "TOOL_CALL_START" not in types


# ============================================================================
# Continuation chunk tests
# ============================================================================


@pytest.mark.asyncio
async def test_streaming_fc_continuation_emits_args():
    """Continuation chunks with partial_args emit TOOL_CALL_ARGS deltas."""
    translator = EventTranslator(streaming_function_call_arguments=True)

    # First chunk
    fc1 = _make_func_call(name="write_document", will_continue=True, fc_id="adk-1")
    event1 = _make_adk_event(func_calls=[fc1], partial=True)
    await _collect_events(translator, event1)

    # Continuation chunk
    pa = _make_partial_arg("$.document", "Hello world")
    fc2 = _make_func_call(partial_args=[pa], will_continue=True, fc_id="adk-2")
    event2 = _make_adk_event(func_calls=[fc2], partial=True)

    events = await _collect_events(translator, event2)
    types = _event_types(events)

    assert "TOOL_CALL_ARGS" in types
    args_event = [e for e in events if "TOOL_CALL_ARGS" in str(e.type)][0]
    assert "document" in args_event.delta
    assert "Hello world" in args_event.delta


@pytest.mark.asyncio
async def test_streaming_fc_multiple_continuations():
    """Multiple continuation chunks accumulate deltas correctly."""
    translator = EventTranslator(streaming_function_call_arguments=True)

    # First chunk
    fc1 = _make_func_call(name="write_document", will_continue=True, fc_id="adk-1")
    event1 = _make_adk_event(func_calls=[fc1], partial=True)
    start_events = await _collect_events(translator, event1)

    # Continuation 1
    pa1 = _make_partial_arg("$.document", "Once upon ")
    fc2 = _make_func_call(partial_args=[pa1], will_continue=True, fc_id="adk-2")
    event2 = _make_adk_event(func_calls=[fc2], partial=True)
    chunk1_events = await _collect_events(translator, event2)

    # Continuation 2
    pa2 = _make_partial_arg("$.document", "a time")
    fc3 = _make_func_call(partial_args=[pa2], will_continue=True, fc_id="adk-3")
    event3 = _make_adk_event(func_calls=[fc3], partial=True)
    chunk2_events = await _collect_events(translator, event3)

    # First continuation has key prefix, second has just the value
    assert len(chunk1_events) >= 1
    assert len(chunk2_events) >= 1
    assert "TOOL_CALL_ARGS" in _event_types(chunk1_events)
    assert "TOOL_CALL_ARGS" in _event_types(chunk2_events)

    # Second delta should just be the escaped text (no key prefix)
    args2 = [e for e in chunk2_events if "TOOL_CALL_ARGS" in str(e.type)][0]
    assert args2.delta == "a time"


# ============================================================================
# End marker tests
# ============================================================================


@pytest.mark.asyncio
async def test_streaming_fc_end_comes_with_final():
    """The end marker emits nothing; the aggregated call closes JSON + TOOL_CALL_END."""
    translator = EventTranslator(streaming_function_call_arguments=True)

    # First chunk
    fc1 = _make_func_call(name="write_document", will_continue=True, fc_id="adk-1")
    event1 = _make_adk_event(func_calls=[fc1], partial=True)
    await _collect_events(translator, event1)

    # Continuation (opens JSON path)
    pa = _make_partial_arg("$.document", "content")
    fc2 = _make_func_call(partial_args=[pa], will_continue=True, fc_id="adk-2")
    event2 = _make_adk_event(func_calls=[fc2], partial=True)
    await _collect_events(translator, event2)

    # End marker
    fc_end = _make_func_call(fc_id="adk-3")  # no name, no partial_args, no will_continue
    event_end = _make_adk_event(func_calls=[fc_end], partial=True)
    assert await _collect_events(translator, event_end) == []

    events = await _final(translator, "write_document", {"document": "content"})
    assert _event_types(events) == ["TOOL_CALL_ARGS", "TOOL_CALL_END"]
    assert events[0].delta == '"}'
    assert events[1].tool_call_id == "adk-1"


# ============================================================================
# Full streaming sequence tests
# ============================================================================


@pytest.mark.asyncio
async def test_streaming_fc_full_sequence():
    """Full streaming sequence produces START, ARGS..., ARGS (close), END."""
    translator = EventTranslator(streaming_function_call_arguments=True)

    # First chunk
    fc1 = _make_func_call(name="write_document", will_continue=True, fc_id="adk-1")
    all_events = await _collect_events(translator, _make_adk_event(func_calls=[fc1], partial=True))

    # Two continuations
    pa1 = _make_partial_arg("$.document", "Hello ")
    fc2 = _make_func_call(partial_args=[pa1], will_continue=True, fc_id="adk-2")
    all_events += await _collect_events(translator, _make_adk_event(func_calls=[fc2], partial=True))

    pa2 = _make_partial_arg("$.document", "World")
    fc3 = _make_func_call(partial_args=[pa2], will_continue=True, fc_id="adk-3")
    all_events += await _collect_events(translator, _make_adk_event(func_calls=[fc3], partial=True))

    # End marker
    fc_end = _make_func_call(fc_id="adk-4")
    all_events += await _collect_events(translator, _make_adk_event(func_calls=[fc_end], partial=True))
    all_events += await _final(translator, "write_document", {"document": "Hello World"})

    types = _event_types(all_events)
    assert types[0] == "TOOL_CALL_START"
    assert types[-1] == "TOOL_CALL_END"
    assert types.count("TOOL_CALL_ARGS") == 3  # open, continuation, close


@pytest.mark.asyncio
async def test_streaming_fc_json_deltas_concatenate():
    """All TOOL_CALL_ARGS deltas concatenate to valid JSON."""
    translator = EventTranslator(streaming_function_call_arguments=True)

    # First chunk
    fc1 = _make_func_call(name="write_document", will_continue=True, fc_id="adk-1")
    all_events = await _collect_events(translator, _make_adk_event(func_calls=[fc1], partial=True))

    # Continuations
    pa1 = _make_partial_arg("$.document", "Hello ")
    fc2 = _make_func_call(partial_args=[pa1], will_continue=True, fc_id="adk-2")
    all_events += await _collect_events(translator, _make_adk_event(func_calls=[fc2], partial=True))

    pa2 = _make_partial_arg("$.document", "World")
    fc3 = _make_func_call(partial_args=[pa2], will_continue=True, fc_id="adk-3")
    all_events += await _collect_events(translator, _make_adk_event(func_calls=[fc3], partial=True))

    # End marker
    fc_end = _make_func_call(fc_id="adk-4")
    all_events += await _collect_events(translator, _make_adk_event(func_calls=[fc_end], partial=True))
    all_events += await _final(translator, "write_document", {"document": "Hello World"})

    # Concatenate all TOOL_CALL_ARGS deltas
    args_deltas = [e.delta for e in all_events if "TOOL_CALL_ARGS" in str(e.type)]
    full_json = "".join(args_deltas)

    # Should be valid JSON
    parsed = json.loads(full_json)
    assert parsed == {"document": "Hello World"}


# ============================================================================
# Duplicate suppression tests
# ============================================================================


@pytest.mark.asyncio
async def test_streaming_fc_suppresses_final_aggregated():
    """Final aggregated (non-partial) event is suppressed after streaming."""
    translator = EventTranslator(streaming_function_call_arguments=True)

    # Stream: first -> end (minimal)
    fc1 = _make_func_call(name="write_document", will_continue=True, fc_id="adk-1")
    await _collect_events(translator, _make_adk_event(func_calls=[fc1], partial=True))

    fc_end = _make_func_call(fc_id="adk-2")
    await _collect_events(translator, _make_adk_event(func_calls=[fc_end], partial=True))

    # Final aggregated (non-partial) event
    fc_final = _make_func_call(
        name="write_document", args={"document": "full content"}, fc_id="adk-final"
    )
    final_event = _make_adk_event(func_calls=[fc_final], partial=False)
    events = await _collect_events(translator, final_event)

    types = _event_types(events)
    # No second call: the final only completes the streamed one
    assert "TOOL_CALL_START" not in types
    assert types.count("TOOL_CALL_END") == 1
    assert {e.tool_call_id for e in events} == {"adk-1"}
    assert json.loads(_args_json(events)) == {"document": "full content"}


@pytest.mark.asyncio
async def test_streaming_fc_confirmed_id_remapped():
    """Confirmed FC id is remapped to streaming id for TOOL_CALL_RESULT."""
    translator = EventTranslator(streaming_function_call_arguments=True)

    # Stream: first -> end
    fc1 = _make_func_call(name="write_document", will_continue=True, fc_id="adk-1")
    start_events = await _collect_events(translator, _make_adk_event(func_calls=[fc1], partial=True))
    streaming_id = start_events[0].tool_call_id

    fc_end = _make_func_call(fc_id="adk-2")
    await _collect_events(translator, _make_adk_event(func_calls=[fc_end], partial=True))

    # Final aggregated triggers ID mapping
    fc_final = _make_func_call(
        name="write_document", args={"document": "content"}, fc_id="adk-final"
    )
    await _collect_events(translator, _make_adk_event(func_calls=[fc_final], partial=False))

    # Check ID mapping exists
    assert "adk-final" in translator._confirmed_to_streaming_id
    assert translator._confirmed_to_streaming_id["adk-final"] == streaming_id


# ============================================================================
# Stable ID tests
# ============================================================================


@pytest.mark.asyncio
async def test_streaming_fc_uses_stable_id():
    """All events in a streaming sequence use the same tool_call_id."""
    translator = EventTranslator(streaming_function_call_arguments=True)

    # First chunk
    fc1 = _make_func_call(name="write_document", will_continue=True, fc_id="adk-1")
    events1 = await _collect_events(translator, _make_adk_event(func_calls=[fc1], partial=True))
    start_id = events1[0].tool_call_id

    # Continuation
    pa = _make_partial_arg("$.document", "hello")
    fc2 = _make_func_call(partial_args=[pa], will_continue=True, fc_id="adk-2")
    events2 = await _collect_events(translator, _make_adk_event(func_calls=[fc2], partial=True))

    # End
    fc_end = _make_func_call(fc_id="adk-3")
    events3 = await _collect_events(translator, _make_adk_event(func_calls=[fc_end], partial=True))

    # All events should use the same stable ID
    all_ids = set()
    for e in events1 + events2 + events3:
        if hasattr(e, 'tool_call_id'):
            all_ids.add(e.tool_call_id)

    assert len(all_ids) == 1
    assert start_id in all_ids


# ============================================================================
# PredictState integration tests
# ============================================================================


@pytest.mark.asyncio
async def test_streaming_fc_with_predict_state():
    """PredictState CustomEvent is emitted before TOOL_CALL_START during streaming."""
    translator = EventTranslator(
        streaming_function_call_arguments=True,
        predict_state=[
            PredictStateMapping(
                state_key="document",
                tool="write_document",
                tool_argument="document",
            )
        ],
    )

    fc = _make_func_call(name="write_document", will_continue=True)
    adk_event = _make_adk_event(func_calls=[fc], partial=True)
    events = await _collect_events(translator, adk_event)

    types = _event_types(events)
    assert "CUSTOM" in types
    assert "TOOL_CALL_START" in types
    # PredictState should come before TOOL_CALL_START
    custom_idx = types.index("CUSTOM")
    start_idx = types.index("TOOL_CALL_START")
    assert custom_idx < start_idx

    custom_event = events[custom_idx]
    assert custom_event.name == "PredictState"


# ============================================================================
# Reset tests
# ============================================================================


@pytest.mark.asyncio
async def test_streaming_fc_resets_on_reset():
    """reset() clears all streaming FC state."""
    translator = EventTranslator(streaming_function_call_arguments=True)

    # Start streaming
    fc1 = _make_func_call(name="write_document", will_continue=True, fc_id="adk-1")
    await _collect_events(translator, _make_adk_event(func_calls=[fc1], partial=True))
    assert translator._fc_arg_streams

    # Reset
    translator.reset()

    # State should be clean
    assert not translator._fc_arg_streams
    assert not translator._confirmed_to_streaming_id


# ============================================================================
# Version gate tests
# ============================================================================


def test_adk_version_gate():
    """_adk_supports_streaming_fc_args() returns True for current ADK (>=1.24.0)."""
    assert ADKAgent._adk_supports_streaming_fc_args() is True


# ============================================================================
# Edge case tests
# ============================================================================


@pytest.mark.asyncio
async def test_streaming_fc_stray_chunk_ignored():
    """Nameless chunks without active streaming are ignored."""
    translator = EventTranslator(streaming_function_call_arguments=True)

    # Send a continuation chunk without a preceding first chunk
    pa = _make_partial_arg("$.document", "orphan")
    fc = _make_func_call(partial_args=[pa], will_continue=True, fc_id="adk-stray")
    adk_event = _make_adk_event(func_calls=[fc], partial=True)

    events = await _collect_events(translator, adk_event)
    types = _event_types(events)

    assert "TOOL_CALL_START" not in types
    assert "TOOL_CALL_ARGS" not in types


@pytest.mark.asyncio
async def test_streaming_fc_special_chars_escaped():
    """Special characters in partial_args are properly JSON-escaped in deltas."""
    translator = EventTranslator(streaming_function_call_arguments=True)

    # First chunk
    fc1 = _make_func_call(name="write_document", will_continue=True, fc_id="adk-1")
    await _collect_events(translator, _make_adk_event(func_calls=[fc1], partial=True))

    # Continuation with special chars
    pa = _make_partial_arg("$.document", 'He said "hello"\nNew line')
    fc2 = _make_func_call(partial_args=[pa], will_continue=True, fc_id="adk-2")
    events = await _collect_events(translator, _make_adk_event(func_calls=[fc2], partial=True))

    # End
    fc_end = _make_func_call(fc_id="adk-3")
    end_events = await _collect_events(translator, _make_adk_event(func_calls=[fc_end], partial=True))
    end_events += await _final(
        translator, "write_document", {"document": 'He said "hello"\nNew line'}
    )

    # Concatenate all args deltas and verify valid JSON
    all_events = events + end_events
    args_deltas = [e.delta for e in all_events if "TOOL_CALL_ARGS" in str(e.type)]
    full_json = "".join(args_deltas)
    parsed = json.loads(full_json)
    assert parsed == {"document": 'He said "hello"\nNew line'}


@pytest.mark.asyncio
async def test_streaming_fc_lro_skipped():
    """LRO function calls in partial events are skipped by streaming detection."""
    translator = EventTranslator(streaming_function_call_arguments=True)

    fc = _make_func_call(name="write_document", will_continue=True, fc_id="lro-1")
    adk_event = _make_adk_event(func_calls=[fc], partial=True, lro_ids=["lro-1"])

    events = await _collect_events(translator, adk_event)
    types = _event_types(events)

    assert "TOOL_CALL_START" not in types


@pytest.mark.asyncio
async def test_streaming_fc_deferred_end_for_stream_tool_call():
    """stream_tool_call=True defers TOOL_CALL_END."""
    translator = EventTranslator(
        streaming_function_call_arguments=True,
        predict_state=[
            PredictStateMapping(
                state_key="document",
                tool="write_document",
                tool_argument="document",
                stream_tool_call=True,
            )
        ],
    )

    # First chunk
    fc1 = _make_func_call(name="write_document", will_continue=True, fc_id="adk-1")
    await _collect_events(translator, _make_adk_event(func_calls=[fc1], partial=True))

    # End marker
    fc_end = _make_func_call(fc_id="adk-2")
    events = await _collect_events(translator, _make_adk_event(func_calls=[fc_end], partial=True))
    types = _event_types(events)

    # TOOL_CALL_END should NOT be emitted (deferred)
    assert "TOOL_CALL_END" not in types


# ============================================================================
# LiteLLM shape: every chunk named, same provider id, no end marker
# ============================================================================


@pytest.mark.asyncio
async def test_litellm_shape_streams_one_valid_call():
    """LiteLLM streams nested args on every chunk with will_continue=True and only
    then sends the aggregated call with the same id: one call, valid JSON args."""
    translator = EventTranslator(streaming_function_call_arguments=True)
    chunks = [
        [_make_partial_arg("$.nodeId", "concept_")],  # first chunk already has args
        [_make_partial_arg("$.nodeId", "51ea")],
        [_make_partial_arg("$.fields.value", 'Idea: "sentinel"')],
        [_make_partial_arg("$.fields.value", " species")],
    ]
    events = []
    for pas in chunks:
        fc = _make_func_call(
            name="revise_fields", partial_args=pas, will_continue=True, fc_id="call_0"
        )
        events += await _collect_events(translator, _make_adk_event(func_calls=[fc], partial=True))
    final_args = {"nodeId": "concept_51ea", "fields": {"value": 'Idea: "sentinel" species'}}
    events += await _final(translator, "revise_fields", final_args, fc_id="call_0")

    types = _event_types(events)
    assert types[0] == "TOOL_CALL_START" and types[-1] == "TOOL_CALL_END"
    assert types.count("TOOL_CALL_START") == 1 and types.count("TOOL_CALL_END") == 1
    assert {e.tool_call_id for e in events} == {"call_0"}
    assert types.count("TOOL_CALL_ARGS") >= len(chunks)  # streamed, not one blob
    assert json.loads(_args_json(events)) == final_args


@pytest.mark.asyncio
async def test_parallel_streamed_calls_stay_separate():
    """Two calls streamed under different ids never merge into one."""
    translator = EventTranslator(streaming_function_call_arguments=True)
    events = []
    for fc_id, text in [("call_0", "a"), ("call_1", "b"), ("call_0", "c"), ("call_1", "d")]:
        fc = _make_func_call(
            name="revise_fields",
            partial_args=[_make_partial_arg("$.value", text)],
            will_continue=True,
            fc_id=fc_id,
        )
        events += await _collect_events(translator, _make_adk_event(func_calls=[fc], partial=True))
    final = _make_adk_event(
        func_calls=[
            _make_func_call(name="revise_fields", args={"value": "ac"}, fc_id="call_0"),
            _make_func_call(name="revise_fields", args={"value": "bd"}, fc_id="call_1"),
        ],
        partial=False,
    )
    events += await _collect_events(translator, final)

    assert _event_types(events).count("TOOL_CALL_START") == 2
    assert json.loads(_args_json(events, "call_0")) == {"value": "ac"}
    assert json.loads(_args_json(events, "call_1")) == {"value": "bd"}


@pytest.mark.asyncio
async def test_unfinished_stream_closed_at_run_end():
    """A stream whose final never arrives is closed into valid JSON."""
    translator = EventTranslator(streaming_function_call_arguments=True)
    fc = _make_func_call(
        name="revise_fields",
        partial_args=[_make_partial_arg("$.value", "cut sh")],
        will_continue=True,
        fc_id="call_0",
    )
    events = await _collect_events(translator, _make_adk_event(func_calls=[fc], partial=True))
    events += [e async for e in translator.close_open_lro_arg_streams()]

    assert _event_types(events)[-1] == "TOOL_CALL_END"
    assert json.loads(_args_json(events)) == {"value": "cut sh"}
    assert not translator.has_open_lro_arg_stream()  # backend streams don't gate LRO drain


@pytest.mark.asyncio
async def test_gemini_shape_sequential_calls_stay_separate():
    """Gemini streams calls one after another, each with nameless continuations
    under fresh ids and an end marker; a call stays open until the aggregated
    final, so the next call's continuations must not land on the first one."""
    translator = EventTranslator(streaming_function_call_arguments=True)
    events = []
    for name_id, cont_id, end_id, key, text in [
        ("g1", "g2", "g3", "x", "A1"),
        ("g4", "g5", "g6", "y", "B1"),
    ]:
        for fc in [
            _make_func_call(name="tool", will_continue=True, fc_id=name_id),
            _make_func_call(
                partial_args=[_make_partial_arg(f"$.{key}", text)],
                will_continue=True,
                fc_id=cont_id,
            ),
            _make_func_call(fc_id=end_id),
        ]:
            events += await _collect_events(translator, _make_adk_event(func_calls=[fc], partial=True))
    final = _make_adk_event(
        func_calls=[
            _make_func_call(name="tool", args={"x": "A1"}, fc_id="f1"),
            _make_func_call(name="tool", args={"y": "B1"}, fc_id="f2"),
        ],
        partial=False,
    )
    events += await _collect_events(translator, final)

    assert _event_types(events).count("TOOL_CALL_START") == 2
    assert _event_types(events).count("TOOL_CALL_END") == 2
    assert json.loads(_args_json(events, "g1")) == {"x": "A1"}
    assert json.loads(_args_json(events, "g4")) == {"y": "B1"}
