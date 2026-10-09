"""Semantic parity between primary responses and SDK named stream events."""

from __future__ import annotations

from typing import Any, cast

import gigachat.models as gm
import pytest
from langchain_core.language_models.chat_models import generate_from_stream
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGenerationChunk

from langchain_gigachat.chat_models._contracts import primary


def _non_stream_message(values: dict[str, Any]) -> AIMessage:
    response = gm.ChatCompletionResponse.model_validate(values)
    message = primary.create_chat_result(response).generations[0].message
    assert isinstance(message, AIMessage)
    return message


def _stream_message(values: list[dict[str, Any]]) -> AIMessage:
    state = primary.StreamState()
    chunks: list[ChatGenerationChunk] = []
    for raw in values:
        event = gm.PrimaryChatCompletionChunk.model_validate(raw)
        chunk = primary.convert_stream_event(event, state=state)
        if chunk is not None:
            chunks.append(chunk)
    message = generate_from_stream(iter(chunks)).generations[0].message
    assert isinstance(message, AIMessage)
    return message


def _without_indexes(value: Any) -> Any:
    if isinstance(value, list):
        return [_without_indexes(item) for item in value]
    if not isinstance(value, dict):
        return value
    return {
        key: _without_indexes(item) for key, item in value.items() if key != "index"
    }


def _semantic_content(message: AIMessage) -> list[dict[str, Any]]:
    return [
        cast(dict[str, Any], _without_indexes(dict(block)))
        for block in message.content_blocks
        if block.get("type") not in {"tool_call", "tool_call_chunk"}
    ]


@pytest.mark.parametrize(
    ("messages", "events"),
    [
        (
            [{"role": "assistant", "content": [{"text": "Primary response"}]}],
            [
                {
                    "event": "response.message.delta",
                    "messages": [
                        {"role": "assistant", "content": [{"text": "Primary "}]}
                    ],
                },
                {
                    "event": "response.message.delta",
                    "messages": [
                        {"role": "assistant", "content": [{"text": "response"}]}
                    ],
                },
            ],
        ),
        (
            [
                {"role": "reasoning", "content": [{"text": "Think"}]},
                {"role": "assistant", "content": [{"text": "Answer"}]},
            ],
            [
                {
                    "event": "response.message.delta",
                    "messages": [{"role": "reasoning", "content": [{"text": "Think"}]}],
                },
                {
                    "event": "response.message.delta",
                    "messages": [
                        {"role": "assistant", "content": [{"text": "Answer"}]}
                    ],
                },
            ],
        ),
        (
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "text": "Source",
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
                }
            ],
            [
                {
                    "event": "response.message.delta",
                    "messages": [
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "text": "Source",
                                    "inline_data": {
                                        "sources": {
                                            "source-1": {
                                                "url": "https://example.test",
                                                "title": "Example",
                                            }
                                        }
                                    },
                                },
                                {
                                    "files": [
                                        {"id": "file-1", "mime": "application/pdf"}
                                    ]
                                },
                            ],
                        }
                    ],
                }
            ],
        ),
    ],
    ids=["text", "reasoning", "citation-and-file"],
)
def test_content_contract_parity(
    messages: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> None:
    response_values = {
        "messages": messages,
        "finish_reason": "stop",
    }
    stream_values = [
        *events,
        {"event": "response.message.done", "finish_reason": "stop"},
    ]

    non_stream = _non_stream_message(response_values)
    streamed = _stream_message(stream_values)

    assert streamed.text == non_stream.text
    assert _semantic_content(streamed) == _semantic_content(non_stream)


def test_client_function_contract_parity() -> None:
    function_call = {
        "name": "lookup",
        "arguments": {"key": "value"},
    }
    non_stream = _non_stream_message(
        {
            "messages": [
                {
                    "role": "assistant",
                    "tools_state_id": "tools-state-1",
                    "function_call": function_call,
                }
            ],
            "finish_reason": "tool_calls",
        }
    )
    streamed = _stream_message(
        [
            {
                "event": "response.message.delta",
                "tools_state_id": "tools-state-1",
                "messages": [
                    {
                        "role": "assistant",
                        "tools_state_id": "tools-state-1",
                        "function_call": function_call,
                    }
                ],
            },
            {
                "event": "response.message.done",
                "tools_state_id": "tools-state-1",
                "finish_reason": "tool_calls",
            },
        ]
    )

    assert streamed.tool_calls == non_stream.tool_calls
    assert streamed.tool_calls[0]["id"] == "tools-state-1"


def test_explicit_server_tool_contract_parity() -> None:
    execution = {
        "call_id": "server-call-1",
        "name": "web_search",
        "status": "success",
        "output": {"matches": 1},
    }
    non_stream = _non_stream_message(
        {
            "messages": [
                {
                    "role": "assistant",
                    "tool_execution": execution,
                }
            ],
            "finish_reason": "stop",
        }
    )
    streamed = _stream_message(
        [
            {
                "event": "response.tool.completed",
                "messages": [
                    {
                        "role": "assistant",
                        "tool_execution": execution,
                    }
                ],
            },
            {"event": "response.message.done", "finish_reason": "stop"},
        ]
    )

    assert _semantic_content(streamed) == _semantic_content(non_stream)
    result = next(
        block
        for block in streamed.content_blocks
        if block["type"] == "server_tool_result"
    )
    assert result["tool_call_id"] == "server-call-1"


def test_terminal_usage_and_transport_metadata_contract_parity() -> None:
    usage = {
        "input_tokens": 8,
        "input_tokens_details": {"prompt_tokens": 8, "cached_tokens": 3},
        "output_tokens": 5,
        "total_tokens": 13,
    }
    metadata = {
        "message_id": "message-1",
        "thread_id": "thread-1",
        "model": "GigaChat-3-Ultra",
        "created_at": 1_754_000_000,
        "x_headers": {"x-request-id": "request-1"},
        "usage": usage,
    }
    non_stream = _non_stream_message(
        {
            **metadata,
            "messages": [{"role": "assistant", "content": [{"text": "Answer"}]}],
            "finish_reason": "stop",
        }
    )
    streamed = _stream_message(
        [
            {
                "event": "response.message.delta",
                "messages": [{"role": "assistant", "content": [{"text": "Answer"}]}],
            },
            {
                **metadata,
                "event": "response.message.done",
                "finish_reason": "stop",
            },
        ]
    )

    assert streamed.id == non_stream.id == "request-1"
    assert streamed.usage_metadata == non_stream.usage_metadata
    for field in ("message_id", "thread_id", "model", "x_headers"):
        assert streamed.response_metadata[field] == non_stream.response_metadata[field]
