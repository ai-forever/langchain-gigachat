"""Semantic parity between primary non-stream responses and named stream events."""

from __future__ import annotations

from typing import Any, cast

import gigachat.models as gm
import pytest
from langchain_core.language_models.chat_models import generate_from_stream
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult

from langchain_gigachat.chat_models._contracts import primary

_MODEL = "GigaChat-3-Ultra"
_CREATED_AT = 1_754_000_000
_MESSAGE_ID = "message-parity-1"
_THREAD_ID = "thread-parity-1"
_REQUEST_ID = "request-parity-1"


def _response(
    messages: list[dict[str, Any]],
    **overrides: Any,
) -> gm.ChatCompletionResponse:
    values: dict[str, Any] = {"messages": messages}
    values.update(overrides)
    return gm.ChatCompletionResponse.model_validate(values)


def _non_stream_message(
    response: gm.ChatCompletionResponse,
) -> tuple[AIMessage, ChatResult]:
    result = primary.create_chat_result(response)
    message = result.generations[0].message
    assert isinstance(message, AIMessage)
    return message, result


def _stream_message(events: list[dict[str, Any]]) -> AIMessage:
    state = primary.StreamState()
    chunks: list[ChatGenerationChunk] = []
    for values in events:
        event = gm.PrimaryChatCompletionChunk.model_validate(values)
        chunk = primary.convert_stream_event(event, state=state)
        assert chunk is not None
        chunks.append(chunk)
    message = generate_from_stream(iter(chunks)).generations[0].message
    assert isinstance(message, AIMessage)
    return message


def _canonical_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_canonical_value(item) for item in value]
    if not isinstance(value, dict):
        return value

    normalized = {
        key: _canonical_value(item) for key, item in value.items() if key != "index"
    }
    return normalized


def _semantic_content(message: AIMessage) -> list[dict[str, Any]]:
    return [
        cast(dict[str, Any], _canonical_value(dict(block)))
        for block in message.content_blocks
        if block.get("type") not in {"tool_call", "tool_call_chunk"}
    ]


def _semantic_tool_calls(
    message: AIMessage,
) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in dict(call).items() if key != "error"}
        for call in [*message.tool_calls, *message.invalid_tool_calls]
    ]


def _assert_semantic_parity(
    response: gm.ChatCompletionResponse,
    events: list[dict[str, Any]],
) -> tuple[AIMessage, AIMessage]:
    non_stream, _ = _non_stream_message(response)
    streamed = _stream_message(events)

    assert _semantic_content(streamed) == _semantic_content(non_stream)
    assert _semantic_tool_calls(streamed) == _semantic_tool_calls(non_stream)
    return non_stream, streamed


@pytest.mark.parametrize(
    ("case", "messages", "events"),
    [
        pytest.param(
            "plain_text",
            [{"role": "assistant", "content": [{"text": "Primary response"}]}],
            [
                {
                    "event": "response.message.delta",
                    "messages": [
                        {
                            "role": "assistant",
                            "content": [{"text": "Primary response"}],
                        }
                    ],
                }
            ],
            id="plain-text",
        ),
        pytest.param(
            "multiple_text_fragments",
            [
                {
                    "role": "assistant",
                    "content": [{"text": "Primary "}, {"text": "response"}],
                }
            ],
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
            id="multiple-text-fragments",
        ),
        pytest.param(
            "reasoning_and_answer",
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
            id="reasoning-and-answer",
        ),
        pytest.param(
            "citation_annotations",
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "text": "According to the source",
                            "inline_data": {
                                "sources": {
                                    "source-1": {
                                        "url": "https://example.test/source",
                                        "title": "Example",
                                    }
                                }
                            },
                        }
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
                                    "text": "According to the source",
                                    "inline_data": {
                                        "sources": {
                                            "source-1": {
                                                "url": ("https://example.test/source"),
                                                "title": "Example",
                                            }
                                        }
                                    },
                                }
                            ],
                        }
                    ],
                }
            ],
            id="citation-annotations",
        ),
    ],
)
def test_text_and_annotation_contract_parity(
    case: str,
    messages: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> None:
    del case
    _assert_semantic_parity(_response(messages), events)


@pytest.mark.parametrize(
    ("mime_type", "block_type"),
    [
        ("image/png", "image"),
        ("audio/mpeg", "audio"),
        ("video/mp4", "video"),
        ("application/pdf", "file"),
    ],
)
def test_file_contract_parity(mime_type: str, block_type: str) -> None:
    part = {"files": [{"id": f"{block_type}-1", "mime": mime_type}]}
    response = _response([{"role": "assistant", "content": [part]}])
    events: list[dict[str, Any]] = [
        {
            "event": "response.message.delta",
            "messages": [{"role": "assistant", "content": [part]}],
        }
    ]

    non_stream, streamed = _assert_semantic_parity(response, events)

    assert _semantic_content(non_stream)[0]["type"] == block_type
    assert _semantic_content(streamed)[0]["mime_type"] == mime_type


@pytest.mark.parametrize(
    ("arguments", "fragments", "expected_collection"),
    [
        (
            {"city": "Moscow"},
            ('{"city":', '"Moscow"}'),
            "tool_calls",
        ),
        (
            '{"city":broken}',
            ('{"city":', "broken}"),
            "invalid_tool_calls",
        ),
    ],
)
def test_client_function_contract_parity(
    arguments: Any,
    fragments: tuple[str, str],
    expected_collection: str,
) -> None:
    response = _response(
        [
            {
                "role": "assistant",
                "tools_state_id": "client-tool-1",
                "function_call": {
                    "name": "lookup_weather",
                    "arguments": arguments,
                },
            }
        ],
        message_id=_MESSAGE_ID,
        finish_reason="tool_calls",
    )
    events: list[dict[str, Any]] = [
        {
            "event": "response.message.delta",
            "message_id": _MESSAGE_ID,
            "tools_state_id": "client-tool-1",
            "messages": [
                {
                    "function_call": {
                        "name": "lookup_weather",
                        "arguments": fragments[0],
                    }
                }
            ],
        },
        {
            "event": "response.message.delta",
            "messages": [
                {
                    "function_call": {
                        "name": "lookup_weather",
                        "arguments": fragments[1],
                    }
                }
            ],
        },
        {
            "event": "response.message.done",
            "finish_reason": "tool_calls",
        },
    ]

    non_stream, streamed = _assert_semantic_parity(response, events)

    assert getattr(non_stream, expected_collection)
    assert getattr(streamed, expected_collection)


@pytest.mark.parametrize(
    ("status", "event_name", "output"),
    [
        ("running", "response.tool.started", None),
        ("completed", "response.tool.completed", {"matches": 1}),
        ("failed", "response.tool.failed", {"error": "provider failure"}),
    ],
)
def test_server_tool_contract_parity(
    status: str,
    event_name: str,
    output: dict[str, Any] | None,
) -> None:
    execution: dict[str, Any] = {
        "call_id": "execution-1",
        "name": "web_search",
        "status": status,
        "arguments": {"query": "Moscow"},
    }
    if output is not None:
        execution["output"] = output
    response = _response(
        [
            {
                "role": "assistant",
                "tools_state_id": "state-1",
                "tool_execution": execution,
            }
        ],
        message_id=_MESSAGE_ID,
        finish_reason="tool_calls",
    )
    events: list[dict[str, Any]] = [
        {
            "event": event_name,
            "message_id": _MESSAGE_ID,
            "tools_state_id": "state-1",
            "tool_execution": execution,
        },
        {"event": "response.message.done"},
    ]

    non_stream, streamed = _assert_semantic_parity(response, events)

    server_block = next(
        block
        for block in _semantic_content(non_stream)
        if block["type"].startswith("server_tool")
    )
    identity_field = (
        "tool_call_id" if server_block["type"] == "server_tool_result" else "id"
    )
    assert server_block[identity_field] == "execution-1"
    assert all(
        block["type"] != "server_tool_call_chunk" for block in streamed.content_blocks
    )


def test_mirrored_server_tool_with_inline_data_contract_parity() -> None:
    execution = {
        "call_id": "server-tool-1",
        "name": "web_search",
        "status": "completed",
        "seconds_left": 0,
    }
    inline_data = {
        "sources": {
            "source-1": {
                "url": "https://example.test/source",
                "title": "Example",
            }
        },
        "widgets": [{"kind": "table"}],
        "images": [{"id": "image-1"}],
    }
    part = {
        "tool_execution": execution,
        "inline_data": inline_data,
        "provider_extension": {"trace_id": "trace-1"},
    }
    message = {
        "role": "assistant",
        "tools_state_id": "server-tool-1",
        "content": [part],
        "tool_execution": execution,
    }
    response = _response(
        [message],
        message_id=_MESSAGE_ID,
        tool_execution=execution,
    )
    events: list[dict[str, Any]] = [
        {
            "event": "response.tool.completed",
            "message_id": _MESSAGE_ID,
            "tools_state_id": "server-tool-1",
            "messages": [message],
            "tool_execution": execution,
        }
    ]

    non_stream, streamed = _assert_semantic_parity(response, events)

    assert _semantic_content(non_stream) == [
        {
            "type": "server_tool_result",
            "id": "server-tool-1:result",
            "tool_call_id": "server-tool-1",
            "status": "success",
            "extras": {
                "provider_tool_execution": execution,
                "inline_data": inline_data,
                "provider_data": {"provider_extension": {"trace_id": "trace-1"}},
            },
        }
    ]
    assert _semantic_content(streamed) == _semantic_content(non_stream)


@pytest.mark.parametrize("call_id", [None, "server-tool-1"])
@pytest.mark.parametrize(
    ("event_name", "status"),
    [
        ("response.tool.completed", "success"),
        ("response.tool.failed", "failed"),
    ],
)
def test_message_level_server_tool_keeps_inline_data_contract_parity(
    call_id: str | None,
    event_name: str,
    status: str,
) -> None:
    execution: dict[str, Any] = {
        "name": "web_search",
        "status": status,
    }
    if call_id is not None:
        execution["call_id"] = call_id
    inline_data = {
        "sources": {
            "source-1": {
                "url": "https://example.test/source",
                "title": "Example",
            }
        }
    }
    message = {
        "role": "reasoning",
        "inline_data": inline_data,
        "tool_execution": execution,
    }

    non_stream, streamed = _assert_semantic_parity(
        _response([message]),
        [{"event": event_name, "messages": [message]}],
    )

    for converted in (non_stream, streamed):
        blocks = _semantic_content(converted)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "server_tool_result"
        assert blocks[0]["extras"]["inline_data"] == inline_data


def test_usage_finish_and_late_metadata_contract_parity() -> None:
    usage = {
        "input_tokens": 13,
        "input_tokens_details": {
            "prompt_tokens": 13,
            "cached_tokens": 5,
        },
        "output_tokens": 8,
        "total_tokens": 21,
    }
    response = _response(
        [{"role": "assistant", "content": [{"text": "Primary response"}]}],
        model=_MODEL,
        created_at=_CREATED_AT,
        message_id=_MESSAGE_ID,
        thread_id=_THREAD_ID,
        finish_reason="stop",
        usage=usage,
        x_headers={"x-request-id": _REQUEST_ID},
    )
    events: list[dict[str, Any]] = [
        {
            "event": "response.message.delta",
            "messages": [
                {"role": "assistant", "content": [{"text": "Primary response"}]}
            ],
        },
        {
            "event": "response.message.done",
            "model": _MODEL,
            "created_at": _CREATED_AT,
            "message_id": _MESSAGE_ID,
            "thread_id": _THREAD_ID,
            "finish_reason": "stop",
            "usage": usage,
            "x_headers": {"x-request-id": _REQUEST_ID},
        },
    ]

    non_stream, streamed = _assert_semantic_parity(response, events)

    assert streamed.usage_metadata == non_stream.usage_metadata
    for field in (
        "model",
        "created_at",
        "message_id",
        "thread_id",
        "finish_reason",
        "x_headers",
    ):
        assert streamed.response_metadata[field] == non_stream.response_metadata[field]


def test_unknown_provider_content_and_event_are_preserved() -> None:
    provider_extension = {"kind": "future-content", "value": 42}
    response = _response(
        [
            {
                "role": "assistant",
                "content": [
                    {
                        "text": "Future response",
                        "provider_extension": provider_extension,
                    }
                ],
            }
        ],
        provider_metadata={"preserve": True},
    )
    event = {
        "event": "response.provider_extension.delta",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "text": "Future response",
                        "provider_extension": provider_extension,
                    }
                ],
            }
        ],
        "provider_metadata": {"preserve": True},
    }

    non_stream, result = _non_stream_message(response)
    streamed = _stream_message([event])

    assert _semantic_content(streamed) == _semantic_content(non_stream)
    assert result.llm_output is not None
    assert non_stream.response_metadata["provider_fields"]["provider_metadata"] == {
        "preserve": True
    }
    assert streamed.response_metadata["provider_field_events"] == [
        {"provider_metadata": {"preserve": True}}
    ]
    assert streamed.response_metadata["raw_events"] == [event]
