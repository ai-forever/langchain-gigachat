"""Adversarial state-machine tests for the primary stream converter."""

from __future__ import annotations

from functools import reduce
from operator import add
from typing import Any, cast

import pytest
from langchain_core.language_models.chat_models import generate_from_stream
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from langchain_gigachat.chat_models._contracts import primary
from langchain_gigachat.chat_models._contracts.primary.types import PrimaryStreamError


def _convert(
    event: dict[str, Any],
    state: primary.StreamState,
) -> ChatGenerationChunk:
    chunk = primary.convert_stream_event(event, state=state)
    assert chunk is not None
    return chunk


def _message(chunk: ChatGenerationChunk) -> AIMessageChunk:
    assert isinstance(chunk.message, AIMessageChunk)
    return cast(AIMessageChunk, chunk.message)


def _aggregate(events: list[dict[str, Any]]) -> AIMessage:
    state = primary.StreamState()
    chunks = [_convert(event, state) for event in events]
    message = reduce(add, chunks).message
    assert isinstance(message, AIMessageChunk)
    return AIMessage(**message.model_dump(exclude={"type"}))


def test_late_tools_state_finalizes_an_idless_client_tool_call() -> None:
    state = primary.StreamState()
    function = _convert(
        {
            "event": "response.message.delta",
            "messages": [
                {
                    "function_call": {
                        "name": "weather",
                        "arguments": '{"city":"Moscow"}',
                    }
                }
            ],
        },
        state,
    )
    done = _convert(
        {
            "event": "response.message.done",
            "tools_state_id": "tools-state-1",
            "finish_reason": "function_call",
        },
        state,
    )

    assert _message(function).tool_call_chunks[0]["id"] is None
    assert _message(done).tool_call_chunks == [
        {
            "name": None,
            "args": "",
            "id": "tools-state-1",
            "index": 0,
            "type": "tool_call_chunk",
        }
    ]
    aggregate = function + done
    assert _message(aggregate).tool_calls == [
        {
            "name": "weather",
            "args": {"city": "Moscow"},
            "id": "tools-state-1",
            "type": "tool_call",
        }
    ]


def test_late_request_id_replaces_provisional_stream_message_id() -> None:
    state = primary.StreamState()
    delta = _convert(
        {
            "event": "response.message.delta",
            "messages": [{"content": [{"text": "answer"}]}],
        },
        state,
    )
    done = _convert(
        {
            "event": "response.message.done",
            "x_headers": {"x-request-id": "request-1"},
            "finish_reason": "stop",
        },
        state,
    )

    assert delta.message.id is None or delta.message.id.startswith("lc_")
    assert done.message.id == "request-1"
    assert (delta + done).message.id == "request-1"


def test_provider_message_id_yields_to_later_request_id() -> None:
    state = primary.StreamState()
    delta = _convert(
        {
            "event": "response.message.delta",
            "message_id": "provider-message-1",
            "messages": [{"content": [{"text": "answer"}]}],
        },
        state,
    )
    done = _convert(
        {
            "event": "response.message.done",
            "x_headers": {"x-request-id": "request-1"},
            "finish_reason": "stop",
        },
        state,
    )

    assert delta.message.id is None or delta.message.id.startswith("lc_")
    assert done.message.id == "request-1"
    assert (delta + done).message.id == "request-1"


def test_explicit_client_call_id_keeps_provider_state_mapping() -> None:
    output = _aggregate(
        [
            {
                "event": "response.message.delta",
                "messages": [
                    {
                        "function_call": {
                            "id": "call-1",
                            "name": "weather",
                            "arguments": '{"city":"Moscow"}',
                        }
                    }
                ],
            },
            {
                "event": "response.message.done",
                "tools_state_id": "tools-state-1",
                "finish_reason": "function_call",
            },
        ]
    )

    assert output.tool_calls[0]["id"] == "call-1"
    assert output.additional_kwargs["provider_tool_state_by_call_id"] == {
        "call-1": "tools-state-1"
    }


def test_completed_client_tool_call_without_provider_state_fails() -> None:
    state = primary.StreamState()
    _convert(
        {
            "event": "response.message.delta",
            "messages": [
                {
                    "function_call": {
                        "name": "weather",
                        "arguments": '{"city":"Moscow"}',
                    }
                }
            ],
        },
        state,
    )

    with pytest.raises(
        ValueError,
        match="completed without tools_state_id; the call cannot be replayed",
    ):
        _convert(
            {
                "event": "response.message.done",
                "finish_reason": "function_call",
            },
            state,
        )


def test_only_message_done_is_completion_terminal() -> None:
    state = primary.StreamState()
    tool_failed = _convert(
        {
            "event": "response.tool.failed",
            "finish_reason": "tool_error",
            "tool_execution": {
                "call_id": "tool-1",
                "name": "web_search",
                "status": "failed",
            },
        },
        state,
    )
    done = _convert(
        {
            "event": "response.message.done",
            "finish_reason": "stop",
        },
        state,
    )

    assert _message(tool_failed).chunk_position is None
    assert _message(done).chunk_position == "last"


def test_non_authoritative_finish_reasons_are_ordered_diagnostics() -> None:
    state = primary.StreamState()
    chunks = [
        _convert(
            {
                "event": "response.tool.failed",
                "finish_reason": "tool_error",
            },
            state,
        ),
        _convert(
            {
                "event": "response.message.delta",
                "finish_reason": "provisional",
            },
            state,
        ),
        _convert(
            {
                "event": "response.message.done",
                "finish_reason": "stop",
            },
            state,
        ),
    ]

    aggregate = reduce(add, chunks)

    assert chunks[0].generation_info is None
    assert chunks[1].generation_info is None
    assert aggregate.generation_info == {"finish_reason": "stop"}
    assert aggregate.message.response_metadata["finish_reason"] == "stop"
    assert aggregate.message.response_metadata["finish_reason_events"] == [
        {
            "event": "response.tool.failed",
            "finish_reason": "tool_error",
        },
        {
            "event": "response.message.delta",
            "finish_reason": "provisional",
        },
    ]


def test_tool_completed_reason_does_not_corrupt_final_finish_reason() -> None:
    state = primary.StreamState()
    tool_completed = _convert(
        {
            "event": "response.tool.completed",
            "finish_reason": "tool_complete",
        },
        state,
    )
    done = _convert(
        {
            "event": "response.message.done",
            "finish_reason": "stop",
        },
        state,
    )

    aggregate = tool_completed + done

    assert aggregate.generation_info == {"finish_reason": "stop"}
    assert aggregate.message.response_metadata["finish_reason"] == "stop"
    assert aggregate.message.response_metadata["finish_reason_events"] == [
        {
            "event": "response.tool.completed",
            "finish_reason": "tool_complete",
        }
    ]


def test_repeated_done_reason_is_diagnostic_after_authoritative_done() -> None:
    state = primary.StreamState()
    done = _convert(
        {
            "event": "response.message.done",
            "finish_reason": "stop",
        },
        state,
    )
    continuation = _convert(
        {
            "event": "response.message.done",
            "finish_reason": "stop",
            "future_field": {"trace": "trace-1"},
        },
        state,
    )

    aggregate = done + continuation

    assert continuation.generation_info is None
    assert aggregate.generation_info == {"finish_reason": "stop"}
    assert aggregate.message.response_metadata["finish_reason"] == "stop"
    assert aggregate.message.response_metadata["finish_reason_events"] == [
        {
            "event": "response.message.done",
            "finish_reason": "stop",
        }
    ]


def test_generate_from_stream_keeps_authoritative_finish_reason() -> None:
    state = primary.StreamState()
    chunks = [
        _convert(
            {
                "event": "response.tool.failed",
                "finish_reason": "tool_error",
            },
            state,
        ),
        _convert(
            {
                "event": "response.message.done",
                "finish_reason": "stop",
            },
            state,
        ),
    ]

    result = generate_from_stream(iter(chunks))
    generation = result.generations[0]

    assert generation.generation_info == {"finish_reason": "stop"}
    assert generation.message.response_metadata["finish_reason"] == "stop"


def test_identical_message_done_is_deduplicated() -> None:
    state = primary.StreamState()
    event = {
        "event": "response.message.done",
        "message_id": "message-1",
        "finish_reason": "stop",
    }

    assert primary.convert_stream_event(event, state=state) is not None
    assert primary.convert_stream_event(event, state=state) is None


def test_conflicting_message_done_fails() -> None:
    state = primary.StreamState()
    _convert(
        {
            "event": "response.message.done",
            "finish_reason": "stop",
        },
        state,
    )

    with pytest.raises(
        ValueError,
        match="Conflicting primary completion terminal events",
    ):
        _convert(
            {
                "event": "response.message.done",
                "finish_reason": "length",
            },
            state,
        )


def test_content_after_message_done_fails() -> None:
    state = primary.StreamState()
    _convert({"event": "response.message.done", "finish_reason": "stop"}, state)

    with pytest.raises(
        ValueError,
        match="Primary stream content arrived after response.message.done",
    ):
        _convert(
            {
                "event": "response.message.delta",
                "messages": [{"content": [{"text": "too late"}]}],
            },
            state,
        )


def test_metadata_after_message_done_remains_mergeable_but_not_terminal() -> None:
    state = primary.StreamState()
    done = _convert({"event": "response.message.done", "finish_reason": "stop"}, state)
    metadata = _convert(
        {
            "event": "response.metadata",
            "future_field": {"trace": "trace-1"},
        },
        state,
    )

    assert _message(done).chunk_position == "last"
    assert _message(metadata).chunk_position is None
    assert metadata.message.response_metadata["provider_field_events"] == [
        {"future_field": {"trace": "trace-1"}}
    ]


def test_response_error_raises_typed_error_with_raw_payload() -> None:
    state = primary.StreamState()
    event = {
        "event": "response.error",
        "finish_reason": "error",
        "error": {"code": "provider_error", "message": "boom"},
    }

    with pytest.raises(PrimaryStreamError) as raised:
        primary.convert_stream_event(event, state=state)

    assert raised.value.payload == event


@pytest.mark.parametrize(
    ("field", "message"),
    [
        (
            "message_id",
            "Primary GigaChat completion contains multiple provider message_id "
            "values; their replay semantics are unsupported.",
        ),
        (
            "tools_state_id",
            "Primary GigaChat completion contains multiple tools_state_id values; "
            "their replay semantics are unsupported.",
        ),
    ],
)
def test_distinct_provider_ids_fail_with_non_stream_policy(
    field: str,
    message: str,
) -> None:
    state = primary.StreamState()
    _convert({"event": "response.message.delta", field: "id-1"}, state)

    with pytest.raises(ValueError, match=message):
        _convert({"event": "response.message.done", field: "id-2"}, state)


def test_repeated_usage_snapshot_is_emitted_once() -> None:
    state = primary.StreamState()
    usage = {
        "input_tokens": 2,
        "output_tokens": 1,
        "total_tokens": 3,
    }
    first = _convert({"event": "response.message.delta", "usage": usage}, state)
    done = _convert(
        {
            "event": "response.message.done",
            "usage": usage,
            "finish_reason": "stop",
        },
        state,
    )

    assert _message(first).usage_metadata == usage
    assert _message(done).usage_metadata is None
    assert _message(first + done).usage_metadata == usage


def test_changed_usage_snapshot_fails_without_incremental_semantics() -> None:
    state = primary.StreamState()
    _convert(
        {
            "event": "response.message.delta",
            "usage": {"input_tokens": 2, "output_tokens": 1},
        },
        state,
    )

    with pytest.raises(
        ValueError,
        match="incremental usage semantics are unsupported",
    ):
        _convert(
            {
                "event": "response.message.done",
                "usage": {"input_tokens": 2, "output_tokens": 2},
            },
            state,
        )


def test_event_diagnostics_aggregate_as_ordered_lists() -> None:
    state = primary.StreamState()
    chunks = [
        _convert(
            {
                "event": f"response.future.{step}",
                "tool_execution": {
                    "call_id": "provider-tool-1",
                    "status": step,
                },
                "future_field": {"step": step},
            },
            state,
        )
        for step in ("started", "completed")
    ]

    metadata = (chunks[0] + chunks[1]).message.response_metadata
    assert metadata["tool_execution_events"] == [
        {"call_id": "provider-tool-1", "status": "started"},
        {"call_id": "provider-tool-1", "status": "completed"},
    ]
    assert metadata["provider_field_events"] == [
        {"future_field": {"step": "started"}},
        {"future_field": {"step": "completed"}},
    ]
