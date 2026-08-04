"""Primary SDK named-event stream conversion."""

from __future__ import annotations

from functools import reduce
from operator import add
from typing import Any, cast

import gigachat.models as gm
import pytest
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from langchain_gigachat.chat_models._contracts import primary
from langchain_gigachat.chat_models._contracts.primary.types import PrimaryStreamError

from .fixtures import build_official_sdk_server_tool_stream


def _event(values: dict[str, Any]) -> gm.PrimaryChatCompletionChunk:
    return gm.PrimaryChatCompletionChunk.model_validate(values)


def _convert(
    values: dict[str, Any] | gm.PrimaryChatCompletionChunk,
    state: primary.StreamState | None = None,
) -> ChatGenerationChunk:
    event = (
        values if isinstance(values, gm.PrimaryChatCompletionChunk) else _event(values)
    )
    chunk = primary.convert_stream_event(
        event,
        state=state or primary.StreamState(),
    )
    assert chunk is not None
    return chunk


def _blocks(chunk: ChatGenerationChunk) -> list[dict[str, Any]]:
    assert isinstance(chunk.message, AIMessageChunk)
    return [dict(block) for block in chunk.message.content_blocks]


def test_text_reasoning_files_and_citations_keep_ordered_blocks() -> None:
    state = primary.StreamState()
    chunk = _convert(
        {
            "event": "response.message.delta",
            "messages": [
                {"role": "reasoning", "content": [{"text": "Think"}]},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "text": "Answer",
                            "inline_data": {
                                "sources": {
                                    "source-1": {
                                        "url": "https://example.test",
                                        "title": "Example",
                                    }
                                }
                            },
                        },
                        {"files": [{"id": "file-1", "mime": "application/pdf"}]},
                    ],
                },
            ],
        },
        state,
    )

    assert _blocks(chunk) == [
        {"type": "reasoning", "reasoning": "Think", "index": 0},
        {
            "type": "text",
            "text": "Answer",
            "index": 1,
            "annotations": [
                {
                    "type": "citation",
                    "id": "source-1",
                    "url": "https://example.test",
                    "title": "Example",
                }
            ],
        },
        {
            "type": "file",
            "file_id": "file-1",
            "mime_type": "application/pdf",
            "index": 2,
        },
    ]
    assert chunk.message.additional_kwargs["reasoning_content"] == "Think"


def test_fragmented_client_function_uses_late_tools_state_id() -> None:
    state = primary.StreamState()
    chunks = [
        _convert(
            {
                "event": "response.message.delta",
                "messages": [
                    {
                        "role": "assistant",
                        "function_call": {
                            "name": "weather",
                            "arguments": '{"city":',
                        },
                    }
                ],
            },
            state,
        ),
        _convert(
            {
                "event": "response.message.delta",
                "tools_state_id": "tools-state-1",
                "messages": [
                    {
                        "role": "assistant",
                        "function_call": {
                            "name": "weather",
                            "arguments": '"Moscow"}',
                        },
                    }
                ],
            },
            state,
        ),
        _convert(
            {
                "event": "response.message.done",
                "tools_state_id": "tools-state-1",
                "finish_reason": "tool_calls",
            },
            state,
        ),
    ]

    aggregate = reduce(add, chunks)
    assert isinstance(aggregate.message, AIMessageChunk)
    assert aggregate.message.tool_calls == [
        {
            "type": "tool_call",
            "name": "weather",
            "args": {"city": "Moscow"},
            "id": "tools-state-1",
        }
    ]
    assert aggregate.message.additional_kwargs["tools_state_id"] == "tools-state-1"


def test_explicit_server_tool_keeps_provider_call_id() -> None:
    state = primary.StreamState()
    started = _convert(
        {
            "event": "response.tool.started",
            "tool_execution": {
                "call_id": "server-call-1",
                "name": "web_search",
                "status": "running",
                "arguments": {"query": "GigaChat"},
            },
        },
        state,
    )
    completed = _convert(
        {
            "event": "response.tool.completed",
            "tool_execution": {
                "call_id": "server-call-1",
                "name": "web_search",
                "status": "success",
                "output": {"matches": 1},
            },
        },
        state,
    )

    call = _blocks(started)[0]
    result = _blocks(completed)[0]
    assert call["id"] == "server-call-1"
    assert result["tool_call_id"] == "server-call-1"
    assert result["output"] == {"matches": 1}


def test_sequential_idless_server_tools_get_distinct_local_ids() -> None:
    state = primary.StreamState()
    chunks: list[ChatGenerationChunk] = []
    for name in ("web_search", "image_generate"):
        chunks.extend(
            [
                _convert(
                    {
                        "event": "response.tool.started",
                        "tool_execution": {"name": name, "status": "running"},
                    },
                    state,
                ),
                _convert(
                    {
                        "event": "response.tool.completed",
                        "tool_execution": {"name": name, "status": "success"},
                    },
                    state,
                ),
            ]
        )

    blocks = [block for chunk in chunks for block in _blocks(chunk)]
    assert [blocks[0]["id"], blocks[2]["id"]] == [
        "lc_primary-server-tool-0",
        "lc_primary-server-tool-1",
    ]
    assert blocks[1]["tool_call_id"] == blocks[0]["id"]
    assert blocks[3]["tool_call_id"] == blocks[2]["id"]


def test_parallel_idless_server_tools_fail_clearly() -> None:
    event = _event(
        {
            "event": "response.tool.started",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"tool_execution": {"name": "web_search", "status": "running"}},
                        {
                            "tool_execution": {
                                "name": "image_generate",
                                "status": "running",
                            }
                        },
                    ],
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="ambiguous parallel server tools"):
        primary.convert_stream_event(event, state=primary.StreamState())


def test_official_sdk_idless_server_tool_uses_local_identity_only() -> None:
    state = primary.StreamState()
    chunks = [
        _convert(event, state) for event in build_official_sdk_server_tool_stream()
    ]
    aggregate = reduce(add, chunks)
    result = next(
        block
        for block in aggregate.message.content_blocks
        if block["type"] == "server_tool_result"
    )

    assert result["tool_call_id"] == "lc_primary-server-tool-0"
    assert result["tool_call_id"] != "tools-state-1"
    assert aggregate.message.response_metadata["tools_state_ids"] == ["tools-state-1"]


def test_aggregate_does_not_concatenate_repeated_scalar_metadata() -> None:
    state = primary.StreamState()
    events: list[dict[str, Any]] = [
        {
            "event": "response.message.delta",
            "model": "GigaChat-3-Ultra:32.9.23.6",
            "thread_id": "thread-1",
            "messages": [{"role": "assistant", "content": [{"text": "First"}]}],
        },
        {
            "event": "response.message.done",
            "model": "GigaChat-3-Ultra:32.9.23.6",
            "thread_id": "thread-1",
            "finish_reason": "stop",
        },
    ]

    chunks = [_convert(event, state) for event in events]
    aggregate = reduce(add, chunks)

    assert aggregate.message.response_metadata["model"] == (
        "GigaChat-3-Ultra:32.9.23.6"
    )
    assert aggregate.message.response_metadata["model_name"] == (
        "GigaChat-3-Ultra:32.9.23.6"
    )
    assert aggregate.message.response_metadata["thread_id"] == "thread-1"


def test_done_event_preserves_usage_finish_ids_and_headers() -> None:
    state = primary.StreamState()
    chunk = _convert(
        {
            "event": "response.message.done",
            "message_id": "message-1",
            "thread_id": "thread-1",
            "model": "GigaChat-3-Ultra",
            "finish_reason": "stop",
            "x_headers": {"x-request-id": "request-1"},
            "usage": {
                "input_tokens": 8,
                "input_tokens_details": {
                    "prompt_tokens": 8,
                    "cached_tokens": 3,
                },
                "output_tokens": 5,
                "total_tokens": 13,
            },
        },
        state,
    )

    assert isinstance(chunk.message, AIMessageChunk)
    assert chunk.message.id == "request-1"
    assert chunk.message.chunk_position == "last"
    assert chunk.generation_info == {"finish_reason": "stop"}
    assert chunk.message.usage_metadata == {
        "input_tokens": 8,
        "output_tokens": 5,
        "total_tokens": 13,
        "input_token_details": {"cache_read": 3},
    }
    assert chunk.message.response_metadata["message_id"] == "message-1"
    assert chunk.message.response_metadata["thread_id"] == "thread-1"
    assert chunk.message.response_metadata["model"] == "GigaChat-3-Ultra"


def test_unknown_sdk_event_preserves_extension_fields() -> None:
    chunk = _convert(
        {
            "event": "response.provider_extension.delta",
            "future_field": {"value": 42},
        }
    )

    assert chunk.message.response_metadata["provider_fields"] == {
        "future_field": {"value": 42}
    }
    assert chunk.message.response_metadata["provider_events"][0]["event"] == (
        "response.provider_extension.delta"
    )


def test_provider_error_raises_typed_exception() -> None:
    event = _event(
        {
            "event": "response.error",
            "error": {"code": "provider_error", "message": "boom"},
        }
    )

    with pytest.raises(PrimaryStreamError, match="response.error"):
        primary.convert_stream_event(event, state=primary.StreamState())


def test_converter_accepts_only_sdk_models() -> None:
    raw_event = cast(
        gm.PrimaryChatCompletionChunk,
        {"event": "response.message.delta"},
    )
    with pytest.raises(TypeError, match="PrimaryChatCompletionChunk"):
        primary.convert_stream_event(
            raw_event,
            state=primary.StreamState(),
        )
