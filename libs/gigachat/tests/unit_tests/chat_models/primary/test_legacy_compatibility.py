"""Small compatibility shield around the unchanged legacy route."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any
from unittest.mock import MagicMock

import gigachat.models as gm
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from langchain_gigachat.chat_models.gigachat import GigaChat, _convert_message_to_dict

from .fixtures import CREATED_AT, MODEL, REQUEST_ID


def _legacy_response() -> gm.ChatCompletion:
    return gm.ChatCompletion(
        choices=[
            gm.Choices(
                message=gm.Messages(
                    role=gm.MessagesRole.ASSISTANT,
                    content="Legacy response",
                ),
                index=0,
                finish_reason="stop",
            )
        ],
        created=CREATED_AT,
        model=MODEL,
        usage=gm.Usage(
            prompt_tokens=4,
            completion_tokens=2,
            total_tokens=6,
        ),
        object="chat.completion",
        x_headers={"x-request-id": REQUEST_ID},
    )


def _legacy_stream() -> Iterator[gm.ChatCompletionChunk]:
    for content, finish_reason in (("Legacy ", None), ("response", "stop")):
        yield gm.ChatCompletionChunk(
            choices=[
                gm.ChoicesChunk(
                    delta=gm.MessagesChunk(
                        role=gm.MessagesRole.ASSISTANT,
                        content=content,
                    ),
                    index=0,
                    finish_reason=finish_reason,
                )
            ],
            created=CREATED_AT,
            model=MODEL,
            object="chat.completion.chunk",
        )


async def _async_stream() -> AsyncIterator[gm.ChatCompletionChunk]:
    for chunk in _legacy_stream():
        yield chunk


def test_legacy_remains_default_for_invoke_and_stream(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.return_value = _legacy_response()
    sdk_client.stream.side_effect = lambda payload: _legacy_stream()

    response = GigaChat(model=MODEL).invoke("Hello")
    chunks = list(GigaChat(model=MODEL).stream("Hello"))

    assert isinstance(response, AIMessage)
    assert response.content == "Legacy response"
    assert all(isinstance(chunk, AIMessageChunk) for chunk in chunks)
    assert "".join(str(chunk.content) for chunk in chunks) == "Legacy response"
    sdk_client.chat.assert_called_once()
    sdk_client.stream.assert_called_once()
    sdk_client.chat.create.assert_not_called()
    sdk_client.chat.stream.assert_not_called()


async def test_legacy_async_stream_remains_default(
    sdk_client: MagicMock,
) -> None:
    sdk_client.astream.side_effect = lambda payload: _async_stream()

    chunks = [chunk async for chunk in GigaChat(model=MODEL).astream("Hello")]

    assert "".join(str(chunk.content) for chunk in chunks) == "Legacy response"
    sdk_client.astream.assert_called_once()
    sdk_client.achat.stream.assert_not_called()


def test_legacy_client_tool_binding_is_unchanged(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.return_value = _legacy_response()
    tool = {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a value.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        },
    }

    GigaChat(model=MODEL).bind_tools([tool], tool_choice="lookup").invoke("Hello")

    payload = sdk_client.chat.call_args.args[0]
    assert isinstance(payload, gm.Chat)
    assert payload.functions is not None
    assert payload.functions[0].name == "lookup"
    assert isinstance(payload.function_call, gm.ChatFunctionCall)
    assert payload.function_call.name == "lookup"


def test_legacy_rejects_primary_builtin_after_route_override(
    sdk_client: MagicMock,
) -> None:
    runnable = (
        GigaChat(model=MODEL, use_api_v2=True)
        .bind_tools([{"type": "web_search"}], tool_choice="web_search")
        .bind(use_api_v2=False)
    )

    with pytest.raises(ValueError, match=r"built-in.*use_api_v2=True"):
        runnable.invoke("Hello")

    sdk_client.chat.assert_not_called()
    sdk_client.chat.create.assert_not_called()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("assistant_id", "assistant-1"),
        ("filter_config", {}),
        ("tool_config", {}),
        ("tools_state_id", "tools-state-1"),
    ],
)
def test_legacy_rejects_primary_only_kwargs(
    sdk_client: MagicMock,
    field: str,
    value: Any,
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"primary-only argument\(s\): {field}.*use_api_v2=True",
    ):
        GigaChat(model=MODEL).invoke("Hello", **{field: value})

    sdk_client.chat.assert_not_called()
    sdk_client.chat.create.assert_not_called()


def test_route_control_flag_is_not_forwarded_to_legacy_payload(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.return_value = _legacy_response()

    GigaChat(model=MODEL, use_api_v2=True).bind(use_api_v2=False).invoke("Hello")

    payload = sdk_client.chat.call_args.args[0]
    assert isinstance(payload, gm.Chat)
    assert "use_api_v2" not in payload.model_dump()
    sdk_client.chat.create.assert_not_called()


def test_legacy_rejects_primary_tool_state_history() -> None:
    message = AIMessage(
        content="stateful primary history",
        additional_kwargs={"tools_state_id": "provider-state"},
    )

    with pytest.raises(ValueError, match="another API contract"):
        _convert_message_to_dict(message)
