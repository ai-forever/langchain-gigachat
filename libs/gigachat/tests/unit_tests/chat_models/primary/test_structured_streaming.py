"""Direct-stream regressions for native response-format constraints."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any
from unittest.mock import MagicMock

import gigachat.models as gm
import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessageChunk
from pydantic import BaseModel

from langchain_gigachat.chat_models.gigachat import GigaChat

from .fixtures import CREATED_AT, MESSAGE_ID, MODEL


class OutputSchema(BaseModel):
    value: int


def get_weather(location: str) -> None:
    """Get weather at a location."""


def _primary_text_stream(
    text: str,
    *,
    include_done: bool = True,
) -> Iterator[gm.PrimaryChatCompletionChunk]:
    split_at = max(1, len(text) // 2)
    for fragment in (text[:split_at], text[split_at:]):
        if not fragment:
            continue
        yield gm.PrimaryChatCompletionChunk(
            event="response.message.delta",
            model=MODEL,
            created_at=CREATED_AT,
            messages=[
                gm.ChatMessageChunk(
                    role="assistant",
                    message_id=MESSAGE_ID,
                    content=[gm.ChatContentPart(text=fragment)],
                )
            ],
            message_id=MESSAGE_ID,
        )
    if include_done:
        yield gm.PrimaryChatCompletionChunk(
            event="response.message.done",
            model=MODEL,
            created_at=CREATED_AT,
            messages=None,
            message_id=MESSAGE_ID,
            finish_reason="stop",
        )


def _primary_tool_call_stream() -> Iterator[gm.PrimaryChatCompletionChunk]:
    yield gm.PrimaryChatCompletionChunk(
        event="response.message.delta",
        model=MODEL,
        created_at=CREATED_AT,
        tools_state_id="tool-state",
        messages=[
            gm.ChatMessageChunk(
                role="assistant",
                message_id=MESSAGE_ID,
                content=[],
                function_call=gm.PrimaryChatFunctionCall(
                    name="get_weather",
                    arguments={"location": "Moscow"},
                ),
            )
        ],
        message_id=MESSAGE_ID,
    )
    yield gm.PrimaryChatCompletionChunk(
        event="response.message.done",
        model=MODEL,
        created_at=CREATED_AT,
        messages=None,
        message_id=MESSAGE_ID,
        tools_state_id="tool-state",
        finish_reason="tool_calls",
    )


async def _async_items(
    items: Iterator[Any],
) -> AsyncIterator[Any]:
    for item in items:
        yield item


def _bound_model() -> Any:
    return GigaChat(model=MODEL, use_api_v2=True).bind_tools(
        [get_weather],
        response_format=OutputSchema,
        strict=True,
    )


def _assert_one_terminal_chunk(chunks: list[AIMessageChunk]) -> None:
    last_indexes = [
        index for index, chunk in enumerate(chunks) if chunk.chunk_position == "last"
    ]
    assert last_indexes == [len(chunks) - 1]


class _TokenRecorder(BaseCallbackHandler):
    def __init__(self) -> None:
        self.chunks: list[AIMessageChunk] = []

    def on_llm_new_token(
        self,
        token: str | list[str | dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        generation_chunk = kwargs["chunk"]
        self.chunks.append(generation_chunk.message)


def test_valid_sync_direct_stream_keeps_raw_output_without_model_side_parsing(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.stream.side_effect = lambda payload: _primary_text_stream(
        '{"value": 7}'
    )
    recorder = _TokenRecorder()

    chunks = list(
        _bound_model().stream(
            "Hello",
            config={"callbacks": [recorder]},
        )
    )

    assert "".join(chunk.text for chunk in chunks) == '{"value": 7}'
    assert all("parsed" not in chunk.additional_kwargs for chunk in chunks)
    _assert_one_terminal_chunk(chunks)
    assert recorder.chunks == chunks
    _assert_one_terminal_chunk(recorder.chunks)


async def test_valid_async_direct_stream_keeps_raw_output_without_model_side_parsing(
    sdk_client: MagicMock,
) -> None:
    sdk_client.achat.stream.side_effect = lambda payload: _async_items(
        _primary_text_stream('{"value": 7}')
    )

    chunks = [chunk async for chunk in _bound_model().astream("Hello")]

    assert "".join(chunk.text for chunk in chunks) == '{"value": 7}'
    assert all("parsed" not in chunk.additional_kwargs for chunk in chunks)
    _assert_one_terminal_chunk(chunks)


def test_invalid_sync_direct_stream_preserves_raw_text(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.stream.side_effect = lambda payload: _primary_text_stream(
        "not JSON"
    )

    chunks = list(_bound_model().stream("Hello"))

    assert "".join(chunk.text for chunk in chunks) == "not JSON"
    assert all("parsed" not in chunk.additional_kwargs for chunk in chunks)
    _assert_one_terminal_chunk(chunks)


async def test_invalid_async_direct_stream_preserves_raw_text(
    sdk_client: MagicMock,
) -> None:
    sdk_client.achat.stream.side_effect = lambda payload: _async_items(
        _primary_text_stream("not JSON")
    )

    chunks = [chunk async for chunk in _bound_model().astream("Hello")]

    assert "".join(chunk.text for chunk in chunks) == "not JSON"
    assert all("parsed" not in chunk.additional_kwargs for chunk in chunks)
    _assert_one_terminal_chunk(chunks)


def test_streamed_client_tool_call_is_not_parsed(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.stream.side_effect = lambda payload: _primary_tool_call_stream()

    chunks = list(_bound_model().stream("Hello"))

    assert any(chunk.tool_call_chunks for chunk in chunks)
    assert all("parsed" not in chunk.additional_kwargs for chunk in chunks)
    _assert_one_terminal_chunk(chunks)


def test_missing_provider_terminal_event_fails_closed(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.stream.side_effect = lambda payload: _primary_text_stream(
        '{"value": 7}',
        include_done=False,
    )

    with pytest.raises(ValueError, match="ended before response.message.done"):
        list(_bound_model().stream("Hello"))


async def test_async_missing_provider_terminal_event_fails_closed(
    sdk_client: MagicMock,
) -> None:
    sdk_client.achat.stream.side_effect = lambda payload: _async_items(
        _primary_text_stream(
            '{"value": 7}',
            include_done=False,
        )
    )

    with pytest.raises(ValueError, match="ended before response.message.done"):
        _ = [chunk async for chunk in _bound_model().astream("Hello")]
