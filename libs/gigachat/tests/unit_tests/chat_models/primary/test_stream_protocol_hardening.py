"""Adversarial state-machine tests for the primary stream converter."""

from __future__ import annotations

from functools import reduce
from operator import add
from typing import Any, cast

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from langchain_gigachat.chat_models._contracts import primary


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
