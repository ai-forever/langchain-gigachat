"""Cross-layer terminal contracts for the primary converter and wrapper."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from functools import reduce
from operator import add
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGenerationChunk

from langchain_gigachat.chat_models._contracts import primary
from langchain_gigachat.chat_models.gigachat import GigaChat

from .fixtures import CREATED_AT, MESSAGE_ID, MODEL


def _terminal_events() -> Iterator[dict[str, Any]]:
    yield {
        "event": "response.tool.failed",
        "finish_reason": "tool_error",
        "error": {"message": "tool failed"},
    }
    yield {
        "event": "response.message.done",
        "model": MODEL,
        "created_at": CREATED_AT,
        "message_id": MESSAGE_ID,
        "finish_reason": "stop",
    }
    yield {
        "event": "response.provider_metadata",
        "usage": {
            "input_tokens": 2,
            "output_tokens": 1,
            "total_tokens": 3,
        },
        "x_headers": {"x-request-id": "request-late"},
        "provider_extension": {"trace": "trace-late"},
    }
    yield {
        "event": "response.message.done",
        "model": MODEL,
        "created_at": CREATED_AT,
        "message_id": MESSAGE_ID,
        "thread_id": "thread-late",
        "finish_reason": "stop",
    }


def _converted_events() -> list[ChatGenerationChunk]:
    state = primary.StreamState()
    chunks: list[ChatGenerationChunk] = []
    for event in _terminal_events():
        chunk = primary.convert_stream_event(event, state=state)
        if chunk is not None:
            chunks.append(chunk)
    return chunks


async def _async_events() -> AsyncIterator[dict[str, Any]]:
    for event in _terminal_events():
        yield event


def _aggregate(chunks: list[ChatGenerationChunk]) -> AIMessage:
    merged = reduce(add, chunks)
    return cast(AIMessage, merged.message)


def _assert_terminal_contract(message: AIMessage | AIMessageChunk) -> None:
    assert message.response_metadata["finish_reason"] == "stop"
    assert message.response_metadata["events"] == [
        "response.tool.failed",
        "response.message.done",
        "response.provider_metadata",
        "response.message.done",
    ]
    assert message.response_metadata["finish_reason_events"] == [
        {
            "event": "response.tool.failed",
            "finish_reason": "tool_error",
        },
        {
            "event": "response.message.done",
            "finish_reason": "stop",
        },
    ]
    assert message.response_metadata["thread_id"] == "thread-late"
    assert message.response_metadata["x_headers"] == {"x-request-id": "request-late"}
    assert message.response_metadata["provider_field_events"] == [
        {"error": {"message": "tool failed"}},
        {"provider_extension": {"trace": "trace-late"}},
    ]
    assert message.response_metadata["raw_events"] == [
        {
            "event": "response.provider_metadata",
            "usage": {
                "input_tokens": 2,
                "output_tokens": 1,
                "total_tokens": 3,
            },
            "x_headers": {"x-request-id": "request-late"},
            "provider_extension": {"trace": "trace-late"},
        }
    ]
    assert message.usage_metadata == {
        "input_tokens": 2,
        "output_tokens": 1,
        "total_tokens": 3,
    }


def test_converter_and_sync_wrapper_preserve_the_same_terminal_evidence(
    sdk_client: MagicMock,
) -> None:
    converter_message = _aggregate(_converted_events())
    sdk_client.chat.stream.side_effect = lambda payload: _terminal_events()
    llm = GigaChat(model=MODEL, use_api_v2=True, streaming=True)

    internal = list(llm._stream([HumanMessage("Hello")]))
    invoked = llm.invoke("Hello")

    assert len(internal) == 2
    assert isinstance(internal[-1].message, AIMessageChunk)
    assert internal[-1].message.chunk_position == "last"
    assert internal[-1].generation_info == {"finish_reason": "stop"}
    _assert_terminal_contract(converter_message)
    _assert_terminal_contract(_aggregate(internal))
    _assert_terminal_contract(invoked)


async def test_async_wrapper_preserves_terminal_evidence_without_post_final_chunk(
    sdk_client: MagicMock,
) -> None:
    sdk_client.achat.stream.side_effect = lambda payload: _async_events()
    llm = GigaChat(model=MODEL, use_api_v2=True)

    chunks = [chunk async for chunk in llm._astream([HumanMessage("Hello")])]
    public_chunks = [chunk async for chunk in llm.astream("Hello")]

    assert len(chunks) == 2
    assert len(public_chunks) == 2
    assert isinstance(chunks[-1].message, AIMessageChunk)
    assert chunks[-1].message.chunk_position == "last"
    assert public_chunks[-1].chunk_position == "last"
    _assert_terminal_contract(_aggregate(chunks))
    _assert_terminal_contract(cast(AIMessage, reduce(add, public_chunks)))


def test_converter_and_wrapper_both_reject_content_after_done(
    sdk_client: MagicMock,
) -> None:
    def events() -> Iterator[dict[str, Any]]:
        yield {
            "event": "response.message.done",
            "finish_reason": "stop",
        }
        yield {
            "event": "response.message.delta",
            "messages": [{"content": [{"text": "too late"}]}],
        }

    state = primary.StreamState()
    iterator = events()
    first = next(iterator)
    assert primary.convert_stream_event(first, state=state) is not None
    with pytest.raises(ValueError, match="content arrived after"):
        primary.convert_stream_event(next(iterator), state=state)

    sdk_client.chat.stream.side_effect = lambda payload: events()
    with pytest.raises(ValueError, match="content arrived after"):
        list(GigaChat(model=MODEL, use_api_v2=True).stream("Hello"))
