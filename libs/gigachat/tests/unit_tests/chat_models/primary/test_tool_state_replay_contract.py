"""Public replay contracts for provider-issued client-tool continuation state."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock

import gigachat.models as gm
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_gigachat.chat_models.gigachat import GigaChat

from .fixtures import CREATED_AT, MODEL, build_plain_text_response


def _function_response(
    *,
    tools_state_id: str | None,
) -> gm.ChatCompletionResponse:
    return gm.ChatCompletionResponse(
        model=MODEL,
        created_at=CREATED_AT,
        message_id="provider-message",
        messages=[
            gm.ChatMessage(
                role="assistant",
                message_id="provider-message",
                tools_state_id=tools_state_id,
                content=[],
                function_call=gm.PrimaryChatFunctionCall(
                    name="lookup_weather",
                    arguments={"city": "Moscow"},
                ),
            )
        ],
        finish_reason="function_call",
    )


def _late_state_stream() -> Iterator[gm.PrimaryChatCompletionChunk]:
    yield gm.PrimaryChatCompletionChunk.model_validate(
        {
            "event": "response.message.delta",
            "message_id": "provider-message",
            "messages": [
                {
                    "role": "assistant",
                    "function_call": {
                        "name": "lookup_weather",
                        "arguments": {"city": "Moscow"},
                    },
                }
            ],
        }
    )
    yield gm.PrimaryChatCompletionChunk.model_validate(
        {
            "event": "response.message.done",
            "message_id": "provider-message",
            "tools_state_id": "provider-state",
            "finish_reason": "function_call",
        }
    )


def test_nonstream_public_tool_roundtrip_replays_provider_state(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.create.side_effect = [
        _function_response(tools_state_id="provider-state"),
        build_plain_text_response(),
    ]
    llm = GigaChat(model=MODEL, use_api_v2=True)

    function_message = llm.invoke("Check the weather")
    assert isinstance(function_message, AIMessage)
    assert function_message.tool_calls == [
        {
            "name": "lookup_weather",
            "args": {"city": "Moscow"},
            "id": "provider-state",
            "type": "tool_call",
        }
    ]

    llm.invoke(
        [
            HumanMessage("Check the weather"),
            function_message,
            ToolMessage(
                content='{"temperature": 18}',
                tool_call_id="provider-state",
            ),
        ]
    )

    payload = sdk_client.chat.create.call_args_list[1].args[0]
    assistant_message, tool_message = payload.messages[-2:]
    assert assistant_message.message_id == "provider-message"
    assert assistant_message.tools_state_id == "provider-state"
    assert tool_message.tools_state_id == "provider-state"
    assert tool_message.content
    assert tool_message.content[0].function_result
    assert tool_message.content[0].function_result.name == "lookup_weather"


def test_streamed_late_state_becomes_public_tool_call_identity(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.stream.side_effect = lambda payload: _late_state_stream()
    llm = GigaChat(model=MODEL, use_api_v2=True)

    chunks = list(llm.stream("Check the weather"))
    function_message = chunks[0] + chunks[1]
    assert function_message.tool_calls[0]["id"] == "provider-state"
    assert "provider_tool_state_by_call_id" not in function_message.additional_kwargs

    sdk_client.chat.create.return_value = build_plain_text_response()
    llm.invoke(
        [
            function_message,
            ToolMessage(
                content='{"temperature": 18}',
                tool_call_id="provider-state",
            ),
        ]
    )

    payload = sdk_client.chat.create.call_args.args[0]
    assert payload.messages[0].tools_state_id == "provider-state"
    assert payload.messages[1].tools_state_id == "provider-state"


def test_nonstream_message_id_is_not_fabricated_as_provider_state(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.create.return_value = _function_response(tools_state_id=None)

    with pytest.raises(
        ValueError,
        match=r"missing tools_state_id.*message_id.*lookup_weather",
    ):
        GigaChat(model=MODEL, use_api_v2=True).invoke("Check the weather")
