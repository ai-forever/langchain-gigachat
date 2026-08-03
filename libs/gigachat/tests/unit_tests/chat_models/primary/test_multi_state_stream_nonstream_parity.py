"""Transport-independent identity policy for sequential server-tool states.

Policy B from the review plan is intentional: a server execution without an
execution-level ID receives a local LangChain ID in both transports, while the
provider state is retained in an explicit call-ID mapping.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from functools import reduce
from operator import add
from typing import Any
from unittest.mock import MagicMock

import gigachat.models as gm
import pytest
from langchain_core.language_models.chat_models import generate_from_stream
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.outputs import ChatGenerationChunk

from langchain_gigachat.chat_models._contracts import primary
from langchain_gigachat.chat_models.gigachat import GigaChat

_STATE_MAPPING = {
    "lc_primary-server-tool-0": "state-1",
    "lc_primary-server-tool-1": "state-2",
}


def _execution(name: str, output: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "status": "success",
        "output": output,
    }


def _explicit_server_messages() -> list[dict[str, Any]]:
    return [
        {
            "role": "reasoning",
            "tools_state_id": "state-1",
            "tool_execution": {
                "call_id": "call-1",
                "name": "search",
                "status": "success",
            },
        },
        {
            "role": "reasoning",
            "tools_state_id": "state-2",
            "tool_execution": {
                "call_id": "call-2",
                "name": "image",
                "status": "success",
            },
        },
    ]


def _response() -> gm.ChatCompletionResponse:
    return gm.ChatCompletionResponse.model_validate(
        {
            "messages": [
                {
                    "role": "reasoning",
                    "tools_state_id": "state-1",
                    "tool_execution": _execution(
                        "web_search",
                        {"matches": 1},
                    ),
                },
                {
                    "role": "reasoning",
                    "tools_state_id": "state-2",
                    "tool_execution": _execution(
                        "image_generate",
                        {"image_id": "image-1"},
                    ),
                },
            ],
            "finish_reason": "stop",
        }
    )


def _events() -> list[dict[str, Any]]:
    return [
        {
            "event": "response.tool.completed",
            "tools_state_id": "state-1",
            "tool_execution": _execution("web_search", {"matches": 1}),
        },
        {
            "event": "response.tool.completed",
            "tools_state_id": "state-2",
            "tool_execution": _execution(
                "image_generate",
                {"image_id": "image-1"},
            ),
        },
        {
            "event": "response.message.done",
            "tools_state_id": "state-2",
            "finish_reason": "stop",
        },
    ]


def _sdk_event(event: dict[str, Any]) -> gm.PrimaryChatCompletionChunk:
    return gm.PrimaryChatCompletionChunk.model_validate(event)


def _unassigned_events() -> list[dict[str, Any]]:
    return [
        {
            "event": "response.message.delta",
            "tools_state_id": "unassigned-state-1",
        },
        {
            "event": "response.message.done",
            "tools_state_id": "unassigned-state-2",
            "finish_reason": "stop",
        },
    ]


def _mixed_explicit_payload() -> dict[str, Any]:
    return {
        "tools_state_id": "server-state",
        "messages": [
            {
                "role": "assistant",
                "tools_state_id": "client-state",
                "function_call": {
                    "name": "lookup_weather",
                    "arguments": {"city": "Moscow"},
                },
            },
            {
                "role": "reasoning",
                "tool_execution": {
                    "call_id": "server-call",
                    "name": "web_search",
                    "status": "success",
                },
            },
        ],
    }


def _mixed_explicit_response() -> gm.ChatCompletionResponse:
    return gm.ChatCompletionResponse.model_validate(_mixed_explicit_payload())


def _mixed_explicit_events() -> list[dict[str, Any]]:
    return [
        {
            "event": "response.tool.completed",
            **_mixed_explicit_payload(),
        },
        {
            "event": "response.message.done",
            "tools_state_id": "server-state",
            "finish_reason": "tool_calls",
        },
    ]


def _nonstream_message() -> AIMessage:
    message = primary.create_chat_result(_response()).generations[0].message
    assert isinstance(message, AIMessage)
    return message


def _stream_message_from_events(events: list[dict[str, Any]]) -> AIMessage:
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


def _stream_message() -> AIMessage:
    return _stream_message_from_events(_events())


def _server_results(message: AIMessage | AIMessageChunk) -> list[dict[str, Any]]:
    return [
        dict(block)
        for block in message.content_blocks
        if block["type"] == "server_tool_result"
    ]


def _assert_policy_b(message: AIMessage | AIMessageChunk) -> None:
    results = _server_results(message)
    assert [result["tool_call_id"] for result in results] == list(_STATE_MAPPING)
    assert message.additional_kwargs["provider_server_tool_state_by_call_id"] == (
        _STATE_MAPPING
    )
    assert message.response_metadata["tools_state_id"] == "state-2"
    assert message.response_metadata["tools_state_ids"] == ["state-1", "state-2"]


def _assert_mixed_explicit_ownership(message: AIMessage | AIMessageChunk) -> None:
    assert message.tool_calls[0]["id"] == "client-state"
    server_result = _server_results(message)[0]
    assert server_result["tool_call_id"] == "server-call"


def _assert_multiple_unassigned_state(message: AIMessage | AIMessageChunk) -> None:
    expected = ["unassigned-state-1", "unassigned-state-2"]
    assert message.additional_kwargs["tools_state_ids"] == expected
    assert "tools_state_id" not in message.additional_kwargs
    assert message.response_metadata["tools_state_ids"] == expected
    assert "tools_state_id" not in message.response_metadata


def test_no_id_server_execution_uses_same_identity_policy_in_both_transports() -> None:
    nonstream = _nonstream_message()
    streamed = _stream_message()

    _assert_policy_b(nonstream)
    _assert_policy_b(streamed)
    assert [
        (block["tool_call_id"], block["status"], block.get("output"))
        for block in _server_results(streamed)
    ] == [
        (block["tool_call_id"], block["status"], block.get("output"))
        for block in _server_results(nonstream)
    ]


def test_nonstream_supports_distinct_server_owned_states() -> None:
    _assert_policy_b(_nonstream_message())


def test_stream_supports_distinct_server_owned_states_through_final_done() -> None:
    _assert_policy_b(_stream_message())


def test_one_stream_event_supports_distinct_message_server_states() -> None:
    state = primary.StreamState()
    event = gm.PrimaryChatCompletionChunk.model_validate(
        {
            "event": "response.tool.completed",
            "messages": [
                {
                    "role": "reasoning",
                    "tools_state_id": "state-1",
                    "tool_execution": _execution(
                        "web_search",
                        {"matches": 1},
                    ),
                },
                {
                    "role": "reasoning",
                    "tools_state_id": "state-2",
                    "tool_execution": _execution(
                        "image_generate",
                        {"image_id": "image-1"},
                    ),
                },
            ],
        }
    )

    completed = primary.convert_stream_event(event, state=state)
    done = primary.convert_stream_event(
        gm.PrimaryChatCompletionChunk.model_validate(
            {
                "event": "response.message.done",
                "tools_state_id": "state-2",
                "finish_reason": "stop",
            }
        ),
        state=state,
    )

    assert completed is not None
    assert done is not None
    message = generate_from_stream(iter([completed, done])).generations[0].message
    assert isinstance(message, AIMessage)
    _assert_policy_b(message)


def test_mixed_explicit_and_idless_executions_preserve_state_ownership() -> None:
    payload = {
        "tools_state_id": "state-B",
        "messages": [
            {
                "role": "reasoning",
                "tool_execution": {
                    "call_id": "explicit-A",
                    "name": "web_search",
                    "status": "success",
                },
            },
            {
                "role": "reasoning",
                "tool_execution": {
                    "name": "image_generate",
                    "status": "success",
                },
            },
        ],
    }
    nonstream = (
        primary.create_chat_result(gm.ChatCompletionResponse.model_validate(payload))
        .generations[0]
        .message
    )
    streamed = _stream_message_from_events(
        [
            {"event": "response.tool.completed", **payload},
            {
                "event": "response.message.done",
                "tools_state_id": "state-B",
                "finish_reason": "stop",
            },
        ]
    )

    assert isinstance(nonstream, AIMessage)
    assert [result["tool_call_id"] for result in _server_results(nonstream)] == [
        "explicit-A",
        "lc_primary-server-tool-0",
    ]
    assert [result["tool_call_id"] for result in _server_results(streamed)] == [
        "explicit-A",
        "lc_primary-server-tool-0",
    ]
    expected_mapping = {"lc_primary-server-tool-0": "state-B"}
    assert (
        nonstream.additional_kwargs["provider_server_tool_state_by_call_id"]
        == expected_mapping
    )
    assert (
        streamed.additional_kwargs["provider_server_tool_state_by_call_id"]
        == expected_mapping
    )


def test_multiple_client_continuation_states_remain_unsupported() -> None:
    response = gm.ChatCompletionResponse.model_validate(
        {
            "messages": [
                {
                    "role": "assistant",
                    "tools_state_id": "client-state-1",
                    "function_call": {
                        "name": "lookup_weather",
                        "arguments": {"city": "Moscow"},
                    },
                },
                {
                    "role": "assistant",
                    "tools_state_id": "client-state-2",
                    "function_call": {
                        "name": "lookup_time",
                        "arguments": {"city": "Moscow"},
                    },
                },
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match=r"(?i)multiple.*(?:client|tools_state_id)|parallel.*client",
    ):
        primary.create_chat_result(response)


@pytest.mark.parametrize("transport", ["nonstream", "stream"])
def test_client_call_rejects_conflicting_top_and_message_states(
    transport: str,
) -> None:
    message = {
        "role": "assistant",
        "tools_state_id": "message-state",
        "function_call": {
            "name": "lookup_weather",
            "arguments": {"city": "Moscow"},
        },
    }
    payload = {
        "tools_state_id": "top-level-state",
        "messages": [message],
    }

    with pytest.raises(ValueError, match=r"(?i)multiple|ambiguous"):
        if transport == "nonstream":
            primary.create_chat_result(
                gm.ChatCompletionResponse.model_validate(payload)
            )
        else:
            primary.convert_stream_event(
                gm.PrimaryChatCompletionChunk.model_validate(
                    {
                        "event": "response.message.delta",
                        **payload,
                    }
                ),
                state=primary.StreamState(),
            )


@pytest.mark.parametrize("transport", ["nonstream", "stream"])
def test_client_message_state_can_differ_from_server_owned_top_level_state(
    transport: str,
) -> None:
    payload = {
        "tools_state_id": "server-state",
        "messages": [
            {
                "role": "assistant",
                "tools_state_id": "client-state",
                "function_call": {
                    "name": "lookup_weather",
                    "arguments": {"city": "Moscow"},
                },
            },
            {
                "role": "reasoning",
                "tools_state_id": "server-state",
                "tool_execution": _execution("web_search", {"matches": 1}),
            },
        ],
    }

    if transport == "nonstream":
        message = (
            primary.create_chat_result(
                gm.ChatCompletionResponse.model_validate(payload)
            )
            .generations[0]
            .message
        )
    else:
        chunk = primary.convert_stream_event(
            gm.PrimaryChatCompletionChunk.model_validate(
                {
                    "event": "response.tool.completed",
                    **payload,
                }
            ),
            state=primary.StreamState(),
        )
        assert chunk is not None
        message = chunk.message

    assert isinstance(message, (AIMessage, AIMessageChunk))
    assert message.tool_calls[0]["id"] == "client-state"
    server_result = _server_results(message)[0]
    assert server_result["tool_call_id"] == "lc_primary-server-tool-0"


@pytest.mark.parametrize("transport", ["nonstream", "stream"])
def test_explicit_server_execution_owns_separate_top_level_state(
    transport: str,
) -> None:
    payload = _mixed_explicit_payload()

    if transport == "nonstream":
        message = (
            primary.create_chat_result(
                gm.ChatCompletionResponse.model_validate(payload)
            )
            .generations[0]
            .message
        )
    else:
        chunk = primary.convert_stream_event(
            gm.PrimaryChatCompletionChunk.model_validate(
                {
                    "event": "response.tool.completed",
                    **payload,
                }
            ),
            state=primary.StreamState(),
        )
        assert chunk is not None
        message = chunk.message

    assert isinstance(message, (AIMessage, AIMessageChunk))
    _assert_mixed_explicit_ownership(message)


@pytest.mark.parametrize("transport", ["nonstream", "stream"])
@pytest.mark.parametrize("message_state", [None, "shared-state"])
def test_explicit_server_context_does_not_steal_client_state(
    transport: str,
    message_state: str | None,
) -> None:
    client_message: dict[str, Any] = {
        "role": "assistant",
        "function_call": {
            "name": "lookup_weather",
            "arguments": {"city": "Moscow"},
        },
    }
    if message_state is not None:
        client_message["tools_state_id"] = message_state
    payload = {
        "tools_state_id": "shared-state",
        "messages": [
            client_message,
            {
                "role": "reasoning",
                "tool_execution": {
                    "call_id": "server-call",
                    "name": "web_search",
                    "status": "success",
                },
            },
        ],
    }

    if transport == "nonstream":
        message = (
            primary.create_chat_result(
                gm.ChatCompletionResponse.model_validate(payload)
            )
            .generations[0]
            .message
        )
    else:
        chunk = primary.convert_stream_event(
            gm.PrimaryChatCompletionChunk.model_validate(
                {"event": "response.tool.completed", **payload}
            ),
            state=primary.StreamState(),
        )
        assert chunk is not None
        message = chunk.message

    assert isinstance(message, (AIMessage, AIMessageChunk))
    assert message.tool_calls[0]["id"] == "shared-state"
    assert _server_results(message)[0]["tool_call_id"] == "server-call"


@pytest.mark.parametrize("transport", ["nonstream", "stream"])
def test_idless_server_execution_conflicts_with_client_fallback_state(
    transport: str,
) -> None:
    payload = {
        "tools_state_id": "shared-state",
        "messages": [
            {
                "role": "assistant",
                "function_call": {
                    "name": "lookup_weather",
                    "arguments": {"city": "Moscow"},
                },
            },
            {
                "role": "reasoning",
                "tool_execution": {
                    "name": "web_search",
                    "status": "success",
                },
            },
        ],
    }

    with pytest.raises(ValueError, match=r"(?i)ambiguous|client.*server"):
        if transport == "nonstream":
            primary.create_chat_result(
                gm.ChatCompletionResponse.model_validate(payload)
            )
        else:
            primary.convert_stream_event(
                gm.PrimaryChatCompletionChunk.model_validate(
                    {"event": "response.tool.completed", **payload}
                ),
                state=primary.StreamState(),
            )


@pytest.mark.parametrize("transport", ["nonstream", "stream"])
def test_two_executions_cannot_claim_one_shared_container_state(
    transport: str,
) -> None:
    executions = [
        _execution("web_search", {"matches": 1}),
        _execution("image_generate", {"image_id": "image-1"}),
    ]
    messages = [
        {
            "role": "reasoning",
            "tools_state_id": "shared-state-1",
            "content": [{"tool_execution": execution} for execution in executions],
        }
    ]

    with pytest.raises(ValueError, match=r"(?i)multiple|shared|ambiguous"):
        if transport == "nonstream":
            primary.create_chat_result(
                gm.ChatCompletionResponse.model_validate({"messages": messages})
            )
        else:
            state = primary.StreamState()
            primary.convert_stream_event(
                gm.PrimaryChatCompletionChunk.model_validate(
                    {
                        "event": "response.tool.completed",
                        "tools_state_id": "shared-state-1",
                        "messages": messages,
                    }
                ),
                state=state,
            )


@pytest.mark.parametrize("transport", ["nonstream", "stream"])
def test_one_execution_cannot_claim_multiple_container_states(
    transport: str,
) -> None:
    messages = [
        {
            "role": "reasoning",
            "tools_state_id": "state-1",
            "tool_execution": {
                "call_id": "same-tool",
                "name": "web_search",
                "status": "completed",
            },
        },
        {
            "role": "reasoning",
            "tools_state_id": "state-2",
            "tool_execution": {
                "call_id": "same-tool",
                "name": "web_search",
                "status": "done",
            },
        },
    ]

    with pytest.raises(
        ValueError,
        match=r"same provider identity.*multiple tools_state_id",
    ):
        if transport == "nonstream":
            primary.create_chat_result(
                gm.ChatCompletionResponse.model_validate({"messages": messages})
            )
        else:
            primary.convert_stream_event(
                gm.PrimaryChatCompletionChunk.model_validate(
                    {
                        "event": "response.tool.completed",
                        "messages": messages,
                    }
                ),
                state=primary.StreamState(),
            )


def test_latest_server_state_is_used_for_history_replay() -> None:
    for message in (_nonstream_message(), _stream_message()):
        replayed = primary.convert_messages([message], cached_uploads={})[0]
        assert replayed.tools_state_id == "state-2"


def test_explicit_server_states_use_latest_state_in_both_transports() -> None:
    messages = _explicit_server_messages()
    nonstream = (
        primary.create_chat_result(
            gm.ChatCompletionResponse.model_validate(
                {"messages": messages, "finish_reason": "stop"}
            )
        )
        .generations[0]
        .message
    )
    streamed = _stream_message_from_events(
        [
            {"event": "response.tool.completed", "messages": messages},
            {"event": "response.message.done", "finish_reason": "stop"},
        ]
    )

    for message in (nonstream, streamed):
        assert isinstance(message, AIMessage)
        assert message.additional_kwargs["tools_state_id"] == "state-2"
        assert message.additional_kwargs["tools_state_ids"] == ["state-1", "state-2"]
        assert message.response_metadata["tools_state_id"] == "state-2"
        replayed = primary.convert_messages([message], cached_uploads={})[0]
        assert replayed.tools_state_id == "state-2"


def test_response_level_server_state_is_final_replay_state() -> None:
    messages = _explicit_server_messages()
    nonstream = (
        primary.create_chat_result(
            gm.ChatCompletionResponse.model_validate(
                {
                    "tools_state_id": "state-2",
                    "messages": messages,
                    "finish_reason": "stop",
                }
            )
        )
        .generations[0]
        .message
    )
    streamed = _stream_message_from_events(
        [
            {"event": "response.tool.completed", "messages": messages},
            {
                "event": "response.message.done",
                "tools_state_id": "state-2",
                "finish_reason": "stop",
            },
        ]
    )

    for message in (nonstream, streamed):
        assert isinstance(message, AIMessage)
        assert message.additional_kwargs["tools_state_id"] == "state-2"
        assert message.response_metadata["tools_state_id"] == "state-2"
        replayed = primary.convert_messages([message], cached_uploads={})[0]
        assert replayed.tools_state_id == "state-2"


def test_single_done_event_response_state_is_final_replay_state() -> None:
    messages = _explicit_server_messages()
    nonstream = (
        primary.create_chat_result(
            gm.ChatCompletionResponse.model_validate(
                {
                    "tools_state_id": "state-2",
                    "messages": messages,
                    "finish_reason": "stop",
                }
            )
        )
        .generations[0]
        .message
    )
    streamed = _stream_message_from_events(
        [
            {
                "event": "response.message.done",
                "tools_state_id": "state-2",
                "messages": messages,
                "finish_reason": "stop",
            }
        ]
    )

    for message in (nonstream, streamed):
        assert isinstance(message, AIMessage)
        assert message.additional_kwargs["tools_state_id"] == "state-2"
        assert message.response_metadata["tools_state_id"] == "state-2"
        replayed = primary.convert_messages([message], cached_uploads={})[0]
        assert replayed.tools_state_id == "state-2"


@pytest.mark.parametrize("transport", ["nonstream", "stream"])
def test_mixed_explicit_response_replays_client_state(transport: str) -> None:
    if transport == "nonstream":
        message = (
            primary.create_chat_result(_mixed_explicit_response())
            .generations[0]
            .message
        )
    else:
        message = _stream_message_from_events(_mixed_explicit_events())
    assert isinstance(message, AIMessage)

    replayed = primary.convert_messages(
        [
            message,
            ToolMessage(
                content='{"temperature": 5}',
                tool_call_id="client-state",
                name="lookup_weather",
            ),
        ],
        cached_uploads={},
    )

    assert replayed[0].tools_state_id == "client-state"
    assert replayed[1].tools_state_id == "client-state"


def test_public_invoke_exposes_multi_state_server_mapping(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.create.return_value = _response()

    _assert_policy_b(GigaChat(use_api_v2=True).invoke("Use both tools"))


async def test_public_ainvoke_exposes_multi_state_server_mapping(
    sdk_client: MagicMock,
) -> None:
    sdk_client.achat.create.return_value = _response()

    message = await GigaChat(use_api_v2=True).ainvoke("Use both tools")

    _assert_policy_b(message)


def test_public_stream_exposes_multi_state_server_mapping(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.stream.side_effect = lambda payload: iter(
        map(_sdk_event, _events())
    )

    message = reduce(add, GigaChat(use_api_v2=True).stream("Use both tools"))

    _assert_policy_b(message)


async def test_public_astream_exposes_multi_state_server_mapping(
    sdk_client: MagicMock,
) -> None:
    async def events() -> AsyncIterator[gm.PrimaryChatCompletionChunk]:
        for event in _events():
            yield _sdk_event(event)

    sdk_client.achat.stream.side_effect = lambda payload: events()
    chunks = [
        chunk async for chunk in GigaChat(use_api_v2=True).astream("Use both tools")
    ]

    _assert_policy_b(reduce(add, chunks))


def test_public_stream_and_streaming_invoke_preserve_unassigned_states(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.stream.side_effect = lambda payload: iter(
        map(_sdk_event, _unassigned_events())
    )
    llm = GigaChat(use_api_v2=True)

    streamed = reduce(add, llm.stream("Keep provider state"))
    invoked = llm.invoke("Keep provider state", stream=True)

    _assert_multiple_unassigned_state(streamed)
    _assert_multiple_unassigned_state(invoked)


async def test_public_astream_and_streaming_ainvoke_preserve_unassigned_states(
    sdk_client: MagicMock,
) -> None:
    async def events() -> AsyncIterator[gm.PrimaryChatCompletionChunk]:
        for event in _unassigned_events():
            yield _sdk_event(event)

    sdk_client.achat.stream.side_effect = lambda payload: events()
    llm = GigaChat(use_api_v2=True)

    chunks = [chunk async for chunk in llm.astream("Keep provider state")]
    invoked = await llm.ainvoke("Keep provider state", stream=True)

    _assert_multiple_unassigned_state(reduce(add, chunks))
    _assert_multiple_unassigned_state(invoked)


def test_public_invoke_preserves_mixed_explicit_ownership(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.create.return_value = _mixed_explicit_response()

    message = GigaChat(use_api_v2=True).invoke("Use client and server tools")

    _assert_mixed_explicit_ownership(message)


async def test_public_ainvoke_preserves_mixed_explicit_ownership(
    sdk_client: MagicMock,
) -> None:
    sdk_client.achat.create.return_value = _mixed_explicit_response()

    message = await GigaChat(use_api_v2=True).ainvoke("Use client and server tools")

    _assert_mixed_explicit_ownership(message)


def test_public_stream_preserves_mixed_explicit_ownership(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.stream.side_effect = lambda payload: iter(
        map(_sdk_event, _mixed_explicit_events())
    )

    message = reduce(
        add,
        GigaChat(use_api_v2=True).stream("Use client and server tools"),
    )

    _assert_mixed_explicit_ownership(message)


async def test_public_astream_preserves_mixed_explicit_ownership(
    sdk_client: MagicMock,
) -> None:
    async def events() -> AsyncIterator[gm.PrimaryChatCompletionChunk]:
        for event in _mixed_explicit_events():
            yield _sdk_event(event)

    sdk_client.achat.stream.side_effect = lambda payload: events()
    chunks = [
        chunk
        async for chunk in GigaChat(use_api_v2=True).astream(
            "Use client and server tools"
        )
    ]

    _assert_mixed_explicit_ownership(reduce(add, chunks))
