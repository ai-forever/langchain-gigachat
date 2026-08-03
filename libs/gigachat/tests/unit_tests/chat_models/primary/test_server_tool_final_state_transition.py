"""The final done event may repeat the latest known server-tool state."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from functools import reduce
from operator import add
from typing import Any
from unittest.mock import MagicMock

import gigachat.models as gm
from langchain_core.language_models.chat_models import generate_from_stream
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGenerationChunk

from langchain_gigachat.chat_models._contracts import primary
from langchain_gigachat.chat_models.gigachat import GigaChat


def _five_event_sequence() -> list[dict[str, Any]]:
    return [
        {
            "event": "response.tool.started",
            "tools_state_id": "state-1",
            "tool_execution": {
                "name": "web_search",
                "status": "running",
            },
        },
        {
            "event": "response.tool.completed",
            "tool_execution": {
                "name": "web_search",
                "status": "success",
                "output": {"matches": 1},
            },
        },
        {
            "event": "response.tool.started",
            "tool_execution": {
                "name": "image_generate",
                "status": "running",
            },
        },
        {
            "event": "response.tool.completed",
            "tools_state_id": "state-2",
            "tool_execution": {
                "name": "image_generate",
                "status": "success",
                "output": {"image_id": "image-1"},
            },
        },
        {
            "event": "response.message.done",
            "tools_state_id": "state-2",
            "finish_reason": "stop",
        },
    ]


def _events() -> Iterator[gm.PrimaryChatCompletionChunk]:
    for event in _five_event_sequence():
        yield gm.PrimaryChatCompletionChunk.model_validate(event)


async def _async_events() -> AsyncIterator[gm.PrimaryChatCompletionChunk]:
    for event in _five_event_sequence():
        yield gm.PrimaryChatCompletionChunk.model_validate(event)


def _converted_chunks() -> tuple[list[ChatGenerationChunk], primary.StreamState]:
    state = primary.StreamState()
    chunks: list[ChatGenerationChunk] = []
    for event in _five_event_sequence():
        chunk = primary.convert_stream_event(
            gm.PrimaryChatCompletionChunk.model_validate(event),
            state=state,
        )
        assert chunk is not None
        chunks.append(chunk)
    return chunks, state


def _assert_final_state(message: AIMessage | AIMessageChunk) -> None:
    assert message.response_metadata["tools_state_id"] == "state-2"
    assert message.response_metadata["finish_reason"] == "stop"
    assert message.additional_kwargs["provider_server_tool_state_by_call_id"] == {
        "lc_primary-server-tool-0": "state-1",
        "lc_primary-server-tool-1": "state-2",
    }
    results = [
        block
        for block in message.content_blocks
        if block["type"] == "server_tool_result"
    ]
    assert [block["tool_call_id"] for block in results] == [
        "lc_primary-server-tool-0",
        "lc_primary-server-tool-1",
    ]


def test_convert_stream_event_accepts_done_repeating_latest_server_state() -> None:
    chunks, state = _converted_chunks()

    assert len(chunks) == 5
    assert state.tools_state_id == "state-2"
    assert state.provider_tools_state_ids == ["state-1", "state-2"]


def test_generate_from_stream_accepts_done_repeating_latest_server_state() -> None:
    chunks, _ = _converted_chunks()

    message = generate_from_stream(iter(chunks)).generations[0].message

    assert isinstance(message, AIMessage)
    _assert_final_state(message)


def test_internal_sync_stream_accepts_done_repeating_latest_server_state(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.stream.side_effect = lambda payload: _events()
    llm = GigaChat(use_api_v2=True)

    chunks = list(llm._stream([HumanMessage("Use both tools")]))
    message = generate_from_stream(iter(chunks)).generations[0].message

    assert isinstance(message, AIMessage)
    _assert_final_state(message)


async def test_internal_async_stream_accepts_done_repeating_latest_server_state(
    sdk_client: MagicMock,
) -> None:
    sdk_client.achat.stream.side_effect = lambda payload: _async_events()
    llm = GigaChat(use_api_v2=True)

    chunks = [chunk async for chunk in llm._astream([HumanMessage("Use both tools")])]
    message = generate_from_stream(iter(chunks)).generations[0].message

    assert isinstance(message, AIMessage)
    _assert_final_state(message)


def test_public_stream_accepts_done_repeating_latest_server_state(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.stream.side_effect = lambda payload: _events()

    message = reduce(add, GigaChat(use_api_v2=True).stream("Use both tools"))

    _assert_final_state(message)


async def test_public_astream_accepts_done_repeating_latest_server_state(
    sdk_client: MagicMock,
) -> None:
    sdk_client.achat.stream.side_effect = lambda payload: _async_events()

    chunks = [
        chunk async for chunk in GigaChat(use_api_v2=True).astream("Use both tools")
    ]

    _assert_final_state(reduce(add, chunks))


def test_streaming_invoke_accepts_done_repeating_latest_server_state(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.stream.side_effect = lambda payload: _events()
    llm = GigaChat(use_api_v2=True, streaming=True)

    message = llm.invoke("Use both tools")

    _assert_final_state(message)


async def test_streaming_ainvoke_accepts_done_repeating_latest_server_state(
    sdk_client: MagicMock,
) -> None:
    sdk_client.achat.stream.side_effect = lambda payload: _async_events()
    llm = GigaChat(use_api_v2=True, streaming=True)

    message = await llm.ainvoke("Use both tools")

    _assert_final_state(message)


def test_returned_streaming_message_replays_latest_server_state(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.stream.side_effect = lambda payload: _events()
    message = GigaChat(use_api_v2=True, streaming=True).invoke("Use both tools")

    replayed = primary.convert_messages([message], cached_uploads={})[0]

    assert replayed.tools_state_id == "state-2"
