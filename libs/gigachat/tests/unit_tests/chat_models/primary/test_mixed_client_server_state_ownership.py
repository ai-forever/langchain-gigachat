"""Provider continuation state must have exactly one logical owner."""

from __future__ import annotations

from collections.abc import AsyncIterator
from functools import reduce
from operator import add
from typing import Any
from unittest.mock import MagicMock

import gigachat.models as gm
import pytest
from langchain_core.language_models.chat_models import generate_from_stream
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGenerationChunk

from langchain_gigachat.chat_models._contracts import primary
from langchain_gigachat.chat_models.gigachat import GigaChat


def _convert_all(events: list[dict[str, Any]]) -> AIMessage:
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


def _sdk_event(event: dict[str, Any]) -> gm.PrimaryChatCompletionChunk:
    return gm.PrimaryChatCompletionChunk.model_validate(event)


def _idless_server_completion() -> dict[str, Any]:
    return {
        "event": "response.tool.completed",
        "tool_execution": {
            "name": "web_search",
            "status": "success",
            "output": {"matches": 1},
        },
    }


def _client_call(*, tools_state_id: str | None) -> dict[str, Any]:
    event: dict[str, Any] = {
        "event": "response.message.delta",
        "messages": [
            {
                "function_call": {
                    "name": "lookup_weather",
                    "arguments": {"city": "Moscow"},
                }
            }
        ],
    }
    if tools_state_id is not None:
        event["tools_state_id"] = tools_state_id
    return event


def _mixed_ambiguous_event() -> dict[str, Any]:
    return {
        "event": "response.message.delta",
        "tools_state_id": "shared-state-1",
        "messages": [
            {
                "function_call": {
                    "name": "lookup_weather",
                    "arguments": {"city": "Moscow"},
                },
                "tool_execution": {
                    "name": "web_search",
                    "status": "success",
                },
            }
        ],
    }


def test_client_state_is_not_claimed_by_prior_idless_server_result() -> None:
    message = _convert_all(
        [
            _idless_server_completion(),
            _client_call(tools_state_id="client-state-1"),
            {
                "event": "response.message.done",
                "tools_state_id": "client-state-1",
                "finish_reason": "tool_calls",
            },
        ]
    )

    assert message.tool_calls[0]["id"] == "client-state-1"
    assert message.additional_kwargs.get("provider_server_tool_state_by_call_id") in (
        None,
        {},
    )
    server_result = next(
        block
        for block in message.content_blocks
        if block["type"] == "server_tool_result"
    )
    assert server_result["tool_call_id"] == "lc_primary-server-tool-0"


def test_done_state_with_pending_client_and_server_owners_fails_as_ambiguous() -> None:
    with pytest.raises(ValueError, match=r"(?i)ambiguous|client.*server"):
        _convert_all(
            [
                _idless_server_completion(),
                _client_call(tools_state_id=None),
                {
                    "event": "response.message.done",
                    "tools_state_id": "late-state-1",
                    "finish_reason": "tool_calls",
                },
            ]
        )


def test_prior_server_state_is_not_reused_as_implicit_client_state() -> None:
    with pytest.raises(
        ValueError,
        match=r"(?i)client.*without tools_state_id|client.*cannot be replayed",
    ):
        _convert_all(
            [
                {
                    "event": "response.tool.completed",
                    "tools_state_id": "server-state-1",
                    "tool_execution": {
                        "name": "web_search",
                        "status": "success",
                    },
                },
                _client_call(tools_state_id=None),
                {
                    "event": "response.message.done",
                    "finish_reason": "tool_calls",
                },
            ]
        )


def test_one_state_for_mixed_idless_client_and_server_content_fails_closed() -> None:
    with pytest.raises(ValueError, match=r"(?i)ambiguous|client.*server"):
        _convert_all([_mixed_ambiguous_event()])


def test_mixed_client_and_server_ids_keep_state_owned_by_client() -> None:
    message = _convert_all(
        [
            {
                "event": "response.message.delta",
                "tools_state_id": "client-state-1",
                "messages": [
                    {
                        "function_call": {
                            "name": "lookup_weather",
                            "arguments": {"city": "Moscow"},
                        },
                        "tool_execution": {
                            "call_id": "server-call-1",
                            "name": "web_search",
                            "status": "success",
                        },
                    }
                ],
            },
            {
                "event": "response.message.done",
                "tools_state_id": "client-state-1",
                "finish_reason": "tool_calls",
            },
        ]
    )

    assert message.tool_calls[0]["id"] == "client-state-1"
    assert "provider_tool_state_by_call_id" not in message.additional_kwargs
    assert message.additional_kwargs.get("provider_server_tool_state_by_call_id") in (
        None,
        {},
    )
    server_result = next(
        block
        for block in message.content_blocks
        if block["type"] == "server_tool_result"
    )
    assert server_result["tool_call_id"] == "server-call-1"


@pytest.mark.parametrize("transport", ["nonstream", "stream"])
def test_client_state_cannot_also_identify_explicit_idless_server_mirror(
    transport: str,
) -> None:
    execution = {
        "call_id": "server-call-1",
        "name": "web_search",
        "status": "success",
    }
    message = {
        "role": "assistant",
        "tools_state_id": "shared-state-1",
        "function_call": {
            "name": "lookup_weather",
            "arguments": {"city": "Moscow"},
        },
        "content": [{"tool_execution": execution}],
        "tool_execution": {
            "name": "web_search",
            "status": "success",
        },
    }

    with pytest.raises(ValueError, match=r"(?i)ambiguous|client.*server"):
        if transport == "nonstream":
            primary.create_chat_result(
                gm.ChatCompletionResponse.model_validate({"messages": [message]})
            )
        else:
            primary.convert_stream_event(
                _sdk_event(
                    {
                        "event": "response.tool.completed",
                        "messages": [message],
                    }
                ),
                state=primary.StreamState(),
            )


def test_public_stream_fails_before_returning_ambiguously_owned_state(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.stream.side_effect = lambda payload: iter(
        [_sdk_event(_mixed_ambiguous_event())]
    )

    with pytest.raises(ValueError, match=r"(?i)ambiguous|client.*server"):
        list(GigaChat(use_api_v2=True).stream("Use tools"))


def test_streaming_invoke_fails_before_returning_ambiguously_owned_state(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.stream.side_effect = lambda payload: iter(
        [_sdk_event(_mixed_ambiguous_event())]
    )

    with pytest.raises(ValueError, match=r"(?i)ambiguous|client.*server"):
        GigaChat(use_api_v2=True, streaming=True).invoke("Use tools")


async def test_public_astream_fails_before_returning_ambiguously_owned_state(
    sdk_client: MagicMock,
) -> None:
    async def events() -> AsyncIterator[gm.PrimaryChatCompletionChunk]:
        yield _sdk_event(_mixed_ambiguous_event())

    sdk_client.achat.stream.side_effect = lambda payload: events()

    with pytest.raises(ValueError, match=r"(?i)ambiguous|client.*server"):
        async for _ in GigaChat(use_api_v2=True).astream("Use tools"):
            pass


def test_unambiguous_client_state_survives_public_stream(
    sdk_client: MagicMock,
) -> None:
    events = [
        _idless_server_completion(),
        _client_call(tools_state_id="client-state-1"),
        {
            "event": "response.message.done",
            "tools_state_id": "client-state-1",
            "finish_reason": "tool_calls",
        },
    ]
    sdk_client.chat.stream.side_effect = lambda payload: iter(map(_sdk_event, events))

    message = reduce(add, GigaChat(use_api_v2=True).stream("Use tools"))

    assert message.tool_calls[0]["id"] == "client-state-1"
    assert message.additional_kwargs.get("provider_server_tool_state_by_call_id") in (
        None,
        {},
    )
