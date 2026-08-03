"""Regressions for logical server tools reported at different source levels."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from functools import reduce
from operator import add
from typing import Any
from unittest.mock import MagicMock

import gigachat.models as gm
import pytest
from langchain_core.language_models.chat_models import generate_from_stream
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from langchain_gigachat.chat_models._contracts import primary
from langchain_gigachat.chat_models.gigachat import GigaChat


def _execution(call_id: str, name: str) -> dict[str, Any]:
    return {
        "call_id": call_id,
        "name": name,
        "status": "success",
    }


def _mixed_source_messages() -> list[dict[str, Any]]:
    return [
        {
            "role": "reasoning",
            "content": [{"tool_execution": _execution("search-1", "web_search")}],
        },
        {
            "role": "reasoning",
            "tool_execution": _execution("image-1", "image_generate"),
        },
    ]


def _response(
    messages: list[dict[str, Any]],
    **overrides: Any,
) -> gm.ChatCompletionResponse:
    values: dict[str, Any] = {"messages": messages, "finish_reason": "stop"}
    values.update(overrides)
    return gm.ChatCompletionResponse.model_validate(values)


def _stream_message(events: list[dict[str, Any]]) -> AIMessage:
    state = primary.StreamState()
    chunks: list[ChatGenerationChunk] = []
    for event in events:
        chunk = primary.convert_stream_event(
            gm.PrimaryChatCompletionChunk.model_validate(event),
            state=state,
        )
        assert chunk is not None
        chunks.append(chunk)
    message = generate_from_stream(iter(chunks)).generations[0].message
    assert isinstance(message, AIMessage)
    return message


def _server_results(message: AIMessage | AIMessageChunk) -> list[dict[str, Any]]:
    return [
        dict(block)
        for block in message.content_blocks
        if block["type"] == "server_tool_result"
    ]


def _assert_two_distinct_results(message: AIMessage | AIMessageChunk) -> None:
    results = _server_results(message)
    assert [result["tool_call_id"] for result in results] == [
        "search-1",
        "image-1",
    ]


def _stream_events() -> Iterator[gm.PrimaryChatCompletionChunk]:
    yield gm.PrimaryChatCompletionChunk.model_validate(
        {
            "event": "response.tool.completed",
            "messages": _mixed_source_messages(),
        }
    )
    yield gm.PrimaryChatCompletionChunk.model_validate(
        {"event": "response.message.done", "finish_reason": "stop"}
    )


async def _async_events() -> AsyncIterator[gm.PrimaryChatCompletionChunk]:
    for event in _stream_events():
        yield event


def test_nonstream_preserves_part_and_message_executions_from_distinct_messages() -> (
    None
):
    result = primary.create_chat_result(_response(_mixed_source_messages()))
    message = result.generations[0].message

    assert isinstance(message, AIMessage)
    _assert_two_distinct_results(message)


def test_stream_event_keeps_distinct_part_and_message_executions() -> None:
    message = _stream_message(
        [
            {
                "event": "response.tool.completed",
                "messages": _mixed_source_messages(),
            }
        ]
    )

    _assert_two_distinct_results(message)


def test_two_message_level_executions_remain_distinct() -> None:
    messages = [
        {
            "role": "reasoning",
            "tool_execution": _execution("search-1", "web_search"),
        },
        {
            "role": "reasoning",
            "tool_execution": _execution("image-1", "image_generate"),
        },
    ]

    result = primary.create_chat_result(_response(messages))
    message = result.generations[0].message

    assert isinstance(message, AIMessage)
    _assert_two_distinct_results(message)


@pytest.mark.parametrize("transport", ["nonstream", "stream"])
def test_nested_and_distinct_response_level_executions_are_both_preserved(
    transport: str,
) -> None:
    nested = _execution("search-1", "web_search")
    top_level = _execution("image-1", "image_generate")
    messages = [
        {
            "role": "reasoning",
            "content": [{"tool_execution": nested}],
        }
    ]

    if transport == "nonstream":
        result = primary.create_chat_result(
            _response(messages, tool_execution=top_level)
        )
        message = result.generations[0].message
        assert isinstance(message, AIMessage)
    else:
        message = _stream_message(
            [
                {
                    "event": "response.tool.completed",
                    "messages": messages,
                    "tool_execution": top_level,
                }
            ]
        )

    _assert_two_distinct_results(message)


@pytest.mark.parametrize("transport", ["nonstream", "stream"])
def test_exact_part_message_response_mirror_is_emitted_once(transport: str) -> None:
    execution = _execution("search-1", "web_search")
    message_values = {
        "role": "reasoning",
        "content": [{"tool_execution": execution}],
        "tool_execution": execution,
    }

    if transport == "nonstream":
        result = primary.create_chat_result(
            _response([message_values], tool_execution=execution)
        )
        message = result.generations[0].message
        assert isinstance(message, AIMessage)
    else:
        message = _stream_message(
            [
                {
                    "event": "response.tool.completed",
                    "messages": [message_values],
                    "tool_execution": execution,
                }
            ]
        )

    assert [block["tool_call_id"] for block in _server_results(message)] == ["search-1"]


def test_running_part_message_response_mirror_is_emitted_once() -> None:
    execution = {
        "call_id": "search-1",
        "name": "web_search",
        "status": "running",
    }
    message = _stream_message(
        [
            {
                "event": "response.tool.in_progress",
                "messages": [
                    {
                        "role": "reasoning",
                        "content": [{"tool_execution": execution}],
                        "tool_execution": execution,
                    }
                ],
                "tool_execution": execution,
            }
        ]
    )

    calls = [
        block
        for block in message.content_blocks
        if block["type"] == "server_tool_call_chunk"
    ]
    assert [block["id"] for block in calls] == ["search-1"]


@pytest.mark.parametrize("transport", ["nonstream", "stream"])
def test_explicit_execution_id_correlates_with_state_only_mirror(
    transport: str,
) -> None:
    explicit_execution = _execution("search-1", "web_search")
    state_only_mirror = {
        "name": "web_search",
        "status": "success",
    }
    message_values = {
        "role": "reasoning",
        "tools_state_id": "provider-state-1",
        "content": [{"tool_execution": explicit_execution}],
        "tool_execution": state_only_mirror,
    }

    if transport == "nonstream":
        result = primary.create_chat_result(_response([message_values]))
        message = result.generations[0].message
        assert isinstance(message, AIMessage)
    else:
        message = _stream_message(
            [
                {
                    "event": "response.tool.completed",
                    "tools_state_id": "provider-state-1",
                    "messages": [message_values],
                },
                {
                    "event": "response.message.done",
                    "tools_state_id": "provider-state-1",
                    "finish_reason": "stop",
                },
            ]
        )

    assert [block["tool_call_id"] for block in _server_results(message)] == ["search-1"]
    assert message.additional_kwargs["provider_server_tool_state_by_call_id"] == {
        "search-1": "provider-state-1"
    }


@pytest.mark.parametrize("transport", ["nonstream", "stream"])
def test_equal_execution_payload_with_distinct_explicit_ids_is_not_deduplicated(
    transport: str,
) -> None:
    first = _execution("search-1", "web_search")
    second = _execution("search-2", "web_search")
    message_values = {
        "role": "reasoning",
        "content": [{"tool_execution": first}],
        "tool_execution": second,
    }

    if transport == "nonstream":
        result = primary.create_chat_result(_response([message_values]))
        message = result.generations[0].message
        assert isinstance(message, AIMessage)
    else:
        message = _stream_message(
            [
                {
                    "event": "response.tool.completed",
                    "messages": [message_values],
                }
            ]
        )

    assert [block["tool_call_id"] for block in _server_results(message)] == [
        "search-1",
        "search-2",
    ]


@pytest.mark.parametrize("transport", ["nonstream", "stream"])
def test_same_execution_id_with_conflicting_payload_fails_closed(
    transport: str,
) -> None:
    message_values = {
        "role": "reasoning",
        "content": [{"tool_execution": _execution("shared-1", "web_search")}],
        "tool_execution": _execution("shared-1", "image_generate"),
    }

    with pytest.raises(ValueError, match=r"(?i)conflict|ambiguous"):
        if transport == "nonstream":
            primary.create_chat_result(_response([message_values]))
        else:
            _stream_message(
                [
                    {
                        "event": "response.tool.completed",
                        "messages": [message_values],
                    }
                ]
            )


def test_public_invoke_preserves_distinct_source_level_executions(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.create.return_value = _response(_mixed_source_messages())

    message = GigaChat(use_api_v2=True).invoke("Use both tools")

    _assert_two_distinct_results(message)


async def test_public_ainvoke_preserves_distinct_source_level_executions(
    sdk_client: MagicMock,
) -> None:
    sdk_client.achat.create.return_value = _response(_mixed_source_messages())

    message = await GigaChat(use_api_v2=True).ainvoke("Use both tools")

    _assert_two_distinct_results(message)


def test_public_stream_preserves_distinct_source_level_executions(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.stream.side_effect = lambda payload: _stream_events()

    message = reduce(add, GigaChat(use_api_v2=True).stream("Use both tools"))

    _assert_two_distinct_results(message)


async def test_public_astream_preserves_distinct_source_level_executions(
    sdk_client: MagicMock,
) -> None:
    sdk_client.achat.stream.side_effect = lambda payload: _async_events()

    chunks = [
        chunk async for chunk in GigaChat(use_api_v2=True).astream("Use both tools")
    ]

    _assert_two_distinct_results(reduce(add, chunks))
