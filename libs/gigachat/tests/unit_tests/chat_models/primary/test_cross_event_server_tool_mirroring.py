"""Cross-event mirrors update one server-tool lifecycle instead of duplicating it."""

from __future__ import annotations

from collections.abc import AsyncIterator
from functools import reduce
from operator import add
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.chat_models import generate_from_stream
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from langchain_gigachat.chat_models._contracts import primary
from langchain_gigachat.chat_models.gigachat import GigaChat


def _execution(
    *,
    call_id: str | None = None,
    name: str = "web_search",
    output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    execution: dict[str, Any] = {
        "name": name,
        "status": "success",
        "output": output or {"matches": 1},
    }
    if call_id is not None:
        execution["call_id"] = call_id
    return execution


def _mirrored_events(*, nested_repeat: bool = False) -> list[dict[str, Any]]:
    execution = _execution()
    repeated: dict[str, Any]
    if nested_repeat:
        repeated = {
            "messages": [
                {
                    "role": "reasoning",
                    "content": [{"tool_execution": execution}],
                }
            ]
        }
    else:
        repeated = {"tool_execution": execution}
    return [
        {
            "event": "response.tool.completed",
            "tool_execution": execution,
        },
        {
            "event": "response.message.done",
            "tools_state_id": "provider-state-1",
            "finish_reason": "stop",
            **repeated,
        },
    ]


def _message(events: list[dict[str, Any]]) -> AIMessage:
    state = primary.StreamState()
    chunks: list[ChatGenerationChunk] = []
    for event in events:
        chunk = primary.convert_stream_event(event, state=state)
        assert chunk is not None
        chunks.append(chunk)
    message = generate_from_stream(iter(chunks)).generations[0].message
    assert isinstance(message, AIMessage)
    return message


def _assert_one_correlated_result(message: AIMessage | AIMessageChunk) -> None:
    results = [
        block
        for block in message.content_blocks
        if block["type"] == "server_tool_result"
    ]
    assert len(results) == 1
    assert results[0]["tool_call_id"] == "lc_primary-server-tool-0"
    assert message.additional_kwargs["provider_server_tool_state_by_call_id"] == {
        "lc_primary-server-tool-0": "provider-state-1"
    }


def test_top_level_repeat_with_late_state_updates_existing_lifecycle() -> None:
    _assert_one_correlated_result(_message(_mirrored_events()))


def test_nested_repeat_with_late_state_updates_existing_lifecycle() -> None:
    _assert_one_correlated_result(_message(_mirrored_events(nested_repeat=True)))


@pytest.mark.parametrize("nested_repeat", [False, True])
def test_conflicting_cross_event_repeat_fails_closed(nested_repeat: bool) -> None:
    first = _execution()
    conflicting = _execution(name="image_generate", output={"image_id": "image-1"})
    repeated: dict[str, Any]
    if nested_repeat:
        repeated = {
            "messages": [
                {
                    "role": "reasoning",
                    "content": [{"tool_execution": conflicting}],
                }
            ]
        }
    else:
        repeated = {"tool_execution": conflicting}

    with pytest.raises(ValueError, match=r"(?i)conflict|ambiguous"):
        _message(
            [
                {
                    "event": "response.tool.completed",
                    "tool_execution": first,
                },
                {
                    "event": "response.message.done",
                    "tools_state_id": "provider-state-1",
                    "finish_reason": "stop",
                    **repeated,
                },
            ]
        )


def test_repeat_with_same_explicit_id_is_still_one_logical_result() -> None:
    execution = _execution(call_id="execution-1")

    message = _message(
        [
            {
                "event": "response.tool.completed",
                "tool_execution": execution,
            },
            {
                "event": "response.message.done",
                "tool_execution": execution,
                "finish_reason": "stop",
            },
        ]
    )

    results = [
        block
        for block in message.content_blocks
        if block["type"] == "server_tool_result"
    ]
    assert len(results) == 1
    assert results[0]["tool_call_id"] == "execution-1"


def test_public_stream_deduplicates_top_level_cross_event_mirror(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.stream.side_effect = lambda payload: iter(_mirrored_events())

    message = reduce(add, GigaChat(use_api_v2=True).stream("Search"))

    _assert_one_correlated_result(message)


async def test_public_astream_deduplicates_nested_cross_event_mirror(
    sdk_client: MagicMock,
) -> None:
    async def events() -> AsyncIterator[dict[str, Any]]:
        for event in _mirrored_events(nested_repeat=True):
            yield event

    sdk_client.achat.stream.side_effect = lambda payload: events()

    chunks = [chunk async for chunk in GigaChat(use_api_v2=True).astream("Search")]

    _assert_one_correlated_result(reduce(add, chunks))


def test_streaming_invoke_deduplicates_cross_event_mirror(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.stream.side_effect = lambda payload: iter(_mirrored_events())

    message = GigaChat(use_api_v2=True, streaming=True).invoke("Search")

    _assert_one_correlated_result(message)
