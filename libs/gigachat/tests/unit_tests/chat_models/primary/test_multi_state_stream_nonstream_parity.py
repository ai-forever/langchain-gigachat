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
from langchain_core.messages import AIMessage, AIMessageChunk
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


def _nonstream_message() -> AIMessage:
    message = primary.create_chat_result(_response()).generations[0].message
    assert isinstance(message, AIMessage)
    return message


def _stream_message() -> AIMessage:
    state = primary.StreamState()
    chunks: list[ChatGenerationChunk] = []
    for event in _events():
        chunk = primary.convert_stream_event(event, state=state)
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


def _assert_policy_b(message: AIMessage | AIMessageChunk) -> None:
    results = _server_results(message)
    assert [result["tool_call_id"] for result in results] == list(_STATE_MAPPING)
    assert message.additional_kwargs["provider_server_tool_state_by_call_id"] == (
        _STATE_MAPPING
    )
    assert message.response_metadata["tools_state_id"] == "state-2"
    assert message.response_metadata["tools_state_ids"] == ["state-1", "state-2"]


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
                {
                    "event": "response.tool.completed",
                    "tools_state_id": "shared-state-1",
                    "messages": messages,
                },
                state=state,
            )


def test_latest_server_state_is_used_for_history_replay() -> None:
    for message in (_nonstream_message(), _stream_message()):
        replayed = primary.convert_messages([message], cached_uploads={})[0]
        assert replayed.tools_state_id == "state-2"


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
    sdk_client.chat.stream.side_effect = lambda payload: iter(_events())

    message = reduce(add, GigaChat(use_api_v2=True).stream("Use both tools"))

    _assert_policy_b(message)


async def test_public_astream_exposes_multi_state_server_mapping(
    sdk_client: MagicMock,
) -> None:
    async def events() -> AsyncIterator[dict[str, Any]]:
        for event in _events():
            yield event

    sdk_client.achat.stream.side_effect = lambda payload: events()
    chunks = [
        chunk async for chunk in GigaChat(use_api_v2=True).astream("Use both tools")
    ]

    _assert_policy_b(reduce(add, chunks))
