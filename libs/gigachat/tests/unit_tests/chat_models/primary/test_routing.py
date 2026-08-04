"""Routing matrix for the opt-in primary chat integration."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import gigachat.models as gm
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from langchain_gigachat.chat_models.gigachat import GigaChat

from .fixtures import (
    MODEL,
    REQUEST_ID,
    build_message_delta_event,
    build_message_done_event,
    build_plain_text_response,
)


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
        created=1_754_000_000,
        model=MODEL,
        usage=gm.Usage(
            prompt_tokens=11,
            completion_tokens=4,
            total_tokens=15,
            precached_prompt_tokens=3,
        ),
        object="chat.completion",
        x_headers={"x-request-id": REQUEST_ID},
    )


def _legacy_stream() -> Iterator[gm.ChatCompletionChunk]:
    yield gm.ChatCompletionChunk(
        choices=[
            gm.ChoicesChunk(
                delta=gm.MessagesChunk(
                    role=gm.MessagesRole.ASSISTANT,
                    content="Legacy ",
                ),
                index=0,
            )
        ],
        created=1_754_000_000,
        model=MODEL,
        object="chat.completion.chunk",
        x_headers={"x-request-id": REQUEST_ID},
    )
    yield gm.ChatCompletionChunk(
        choices=[
            gm.ChoicesChunk(
                delta=gm.MessagesChunk(content="response"),
                index=0,
                finish_reason="stop",
            )
        ],
        created=1_754_000_000,
        model=MODEL,
        object="chat.completion.chunk",
        usage=gm.Usage(
            prompt_tokens=11,
            completion_tokens=4,
            total_tokens=15,
            precached_prompt_tokens=3,
        ),
    )


def _primary_stream() -> Iterator[gm.PrimaryChatCompletionChunk]:
    yield build_message_delta_event()
    yield build_message_done_event()


async def _async_items(items: Iterator[Any]) -> AsyncIterator[Any]:
    for item in items:
        yield item


def _configure_responses(sdk_client: MagicMock) -> None:
    sdk_client.chat.return_value = _legacy_response()
    sdk_client.chat.create.return_value = build_plain_text_response()
    sdk_client.achat.return_value = _legacy_response()
    sdk_client.achat.create.return_value = build_plain_text_response()


def _assert_sync_non_stream_route(sdk_client: MagicMock, route: str) -> None:
    if route == "legacy":
        sdk_client.chat.assert_called_once()
        sdk_client.chat.create.assert_not_called()
    else:
        sdk_client.chat.assert_not_called()
        sdk_client.chat.create.assert_called_once()


def _assert_async_non_stream_route(sdk_client: MagicMock, route: str) -> None:
    legacy_chat = sdk_client.achat
    primary_create = sdk_client.achat.create
    assert isinstance(legacy_chat, AsyncMock)
    assert isinstance(primary_create, AsyncMock)
    if route == "legacy":
        legacy_chat.assert_awaited_once()
        primary_create.assert_not_awaited()
    else:
        legacy_chat.assert_not_awaited()
        primary_create.assert_awaited_once()


@pytest.mark.parametrize(
    ("instance_flag", "invocation_flag", "expected_route", "expected_text"),
    [
        (False, None, "legacy", "Legacy response"),
        (True, None, "primary", "Primary response"),
        (False, True, "primary", "Primary response"),
        (True, False, "legacy", "Legacy response"),
    ],
)
def test_sync_non_stream_routing_matrix(
    sdk_client: MagicMock,
    instance_flag: bool,
    invocation_flag: Optional[bool],
    expected_route: str,
    expected_text: str,
) -> None:
    _configure_responses(sdk_client)
    llm = GigaChat(model=MODEL, use_api_v2=instance_flag)
    runnable = llm if invocation_flag is None else llm.bind(use_api_v2=invocation_flag)

    result = runnable.invoke("Hello")

    assert isinstance(result, AIMessage)
    assert result.content == expected_text
    _assert_sync_non_stream_route(sdk_client, expected_route)


@pytest.mark.parametrize(
    ("instance_flag", "invocation_flag", "expected_route", "expected_text"),
    [
        (False, None, "legacy", "Legacy response"),
        (True, None, "primary", "Primary response"),
        (False, True, "primary", "Primary response"),
        (True, False, "legacy", "Legacy response"),
    ],
)
async def test_async_non_stream_routing_matrix(
    sdk_client: MagicMock,
    instance_flag: bool,
    invocation_flag: Optional[bool],
    expected_route: str,
    expected_text: str,
) -> None:
    _configure_responses(sdk_client)
    llm = GigaChat(model=MODEL, use_api_v2=instance_flag)
    runnable = llm if invocation_flag is None else llm.bind(use_api_v2=invocation_flag)

    result = await runnable.ainvoke("Hello")

    assert isinstance(result, AIMessage)
    assert result.content == expected_text
    _assert_async_non_stream_route(sdk_client, expected_route)


@pytest.mark.parametrize(
    ("use_api_v2", "expected_route", "expected_text"),
    [
        (False, "legacy", "Legacy response"),
        (True, "primary", "Primary "),
    ],
)
def test_sync_streaming_routing_matrix(
    sdk_client: MagicMock,
    use_api_v2: bool,
    expected_route: str,
    expected_text: str,
) -> None:
    sdk_client.stream.side_effect = lambda payload: _legacy_stream()
    sdk_client.chat.stream.side_effect = lambda payload: _primary_stream()
    llm = GigaChat(model=MODEL, use_api_v2=use_api_v2, streaming=True)

    chunks = list(llm.stream("Hello"))

    assert all(isinstance(chunk, AIMessageChunk) for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == expected_text
    if expected_route == "legacy":
        sdk_client.stream.assert_called_once()
        sdk_client.chat.stream.assert_not_called()
    else:
        sdk_client.stream.assert_not_called()
        sdk_client.chat.stream.assert_called_once()


@pytest.mark.parametrize(
    ("use_api_v2", "expected_route", "expected_text"),
    [
        (False, "legacy", "Legacy response"),
        (True, "primary", "Primary "),
    ],
)
async def test_async_streaming_routing_matrix(
    sdk_client: MagicMock,
    use_api_v2: bool,
    expected_route: str,
    expected_text: str,
) -> None:
    sdk_client.astream.side_effect = lambda payload: _async_items(_legacy_stream())
    sdk_client.achat.stream.side_effect = lambda payload: _async_items(
        _primary_stream()
    )
    llm = GigaChat(model=MODEL, use_api_v2=use_api_v2, streaming=True)

    chunks = [chunk async for chunk in llm.astream("Hello")]

    assert all(isinstance(chunk, AIMessageChunk) for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == expected_text
    if expected_route == "legacy":
        sdk_client.astream.assert_called_once()
        sdk_client.achat.stream.assert_not_called()
    else:
        sdk_client.astream.assert_not_called()
        sdk_client.achat.stream.assert_called_once()


def test_control_kwarg_is_not_forwarded_to_primary_payload(
    sdk_client: MagicMock,
) -> None:
    _configure_responses(sdk_client)

    GigaChat(model=MODEL).bind(use_api_v2=True).invoke("Hello")

    payload = sdk_client.chat.create.call_args.args[0]
    assert isinstance(payload, gm.ChatCompletionRequest)
    assert "use_api_v2" not in payload.model_dump()


def test_use_api_v2_is_an_identifying_parameter() -> None:
    assert GigaChat(use_api_v2=True)._identifying_params["use_api_v2"] is True


def test_control_kwarg_is_not_forwarded_to_legacy_payload(
    sdk_client: MagicMock,
) -> None:
    _configure_responses(sdk_client)

    GigaChat(model=MODEL, use_api_v2=True).bind(use_api_v2=False).invoke("Hello")

    payload = sdk_client.chat.call_args.args[0]
    assert isinstance(payload, gm.Chat)
    assert "use_api_v2" not in payload.model_dump()


def test_legacy_rejects_primary_only_arguments_with_actionable_error(
    sdk_client: MagicMock,
) -> None:
    _configure_responses(sdk_client)

    with pytest.raises(ValueError, match=r"use_api_v2=True"):
        GigaChat(model=MODEL).invoke(
            "Hello",
            filter_config={"request_content": {"neuro": False}},
        )

    sdk_client.chat.assert_not_called()
    sdk_client.chat.create.assert_not_called()


def test_default_route_preserves_legacy_result_metadata(
    sdk_client: MagicMock,
) -> None:
    _configure_responses(sdk_client)

    result = GigaChat(model=MODEL).invoke("Hello")

    assert isinstance(result, AIMessage)
    assert result.content == "Legacy response"
    assert result.id == REQUEST_ID
    assert result.usage_metadata == {
        "input_tokens": 11,
        "output_tokens": 4,
        "total_tokens": 15,
        "input_token_details": {"cache_read": 3},
    }
    payload = sdk_client.chat.call_args.args[0]
    assert isinstance(payload, gm.Chat)
    assert payload.messages[0].role == gm.MessagesRole.USER
    assert payload.messages[0].content == "Hello"
