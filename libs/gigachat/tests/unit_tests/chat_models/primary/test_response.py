"""Tests for primary non-stream response conversion."""

from __future__ import annotations

import pytest
from gigachat import models as gm
from langchain_core.messages import AIMessage

from langchain_gigachat.chat_models._contracts.primary.response import (
    create_chat_result,
)


def _response(**overrides: object) -> gm.ChatCompletionResponse:
    values: dict[str, object] = {
        "model": "GigaChat-3-Ultra",
        "created_at": 1780321868,
        "messages": [{"role": "assistant", "content": [{"text": "Hello"}]}],
        "finish_reason": "stop",
        "usage": {
            "input_tokens": 10,
            "input_tokens_details": {
                "prompt_tokens": 10,
                "cached_tokens": 2,
            },
            "output_tokens": 4,
            "total_tokens": 14,
        },
    }
    values.update(overrides)
    return gm.ChatCompletionResponse.model_validate(values)


def _message(response: gm.ChatCompletionResponse) -> AIMessage:
    result = create_chat_result(response)
    assert len(result.generations) == 1
    message = result.generations[0].message
    assert isinstance(message, AIMessage)
    return message


def test_plain_text_response_keeps_string_content() -> None:
    message = _message(
        _response(
            messages=[
                {"role": "assistant", "content": [{"text": "Hello, "}]},
                {"role": "assistant", "content": [{"text": "world!"}]},
            ]
        )
    )

    assert message.content == "Hello, world!"
    assert message.content_blocks == [{"type": "text", "text": "Hello, world!"}]


def test_response_messages_are_aggregated_into_one_generation() -> None:
    result = create_chat_result(
        _response(
            messages=[
                {"role": "reasoning", "content": [{"text": "Think"}]},
                {"role": "assistant", "content": [{"text": "Answer"}]},
            ]
        )
    )

    assert len(result.generations) == 1
    assert result.generations[0].message.content == [
        {"type": "reasoning", "reasoning": "Think"},
        {"type": "text", "text": "Answer"},
    ]


def test_mixed_text_and_files_use_standard_content_blocks() -> None:
    message = _message(
        _response(
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {"text": "Generated files"},
                        {
                            "files": [
                                {
                                    "id": "image-1",
                                    "mime": "image/png",
                                    "target": "preview",
                                },
                                {"id": "audio-1", "mime": "audio/opus"},
                                {"id": "document-1", "mime": "application/pdf"},
                            ]
                        },
                    ],
                }
            ]
        )
    )

    assert message.content == [
        {"type": "text", "text": "Generated files"},
        {
            "type": "image",
            "file_id": "image-1",
            "mime_type": "image/png",
            "extras": {"target": "preview"},
        },
        {"type": "audio", "file_id": "audio-1", "mime_type": "audio/opus"},
        {
            "type": "file",
            "file_id": "document-1",
            "mime_type": "application/pdf",
        },
    ]


def test_sources_become_citation_annotations() -> None:
    message = _message(
        _response(
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "text": "According to the source",
                            "inline_data": {
                                "sources": {
                                    "1": {
                                        "url": "https://example.com",
                                        "title": "Example",
                                    }
                                }
                            },
                        }
                    ],
                }
            ]
        )
    )

    assert message.content == [
        {
            "type": "text",
            "text": "According to the source",
            "annotations": [
                {
                    "type": "citation",
                    "id": "1",
                    "url": "https://example.com",
                    "title": "Example",
                }
            ],
        }
    ]


def test_client_function_call_uses_tools_state_id() -> None:
    message = _message(
        _response(
            messages=[
                {
                    "role": "assistant",
                    "tools_state_id": "tools-state-1",
                    "content": [
                        {
                            "function_call": {
                                "name": "get_weather",
                                "arguments": {"location": "Moscow"},
                            }
                        }
                    ],
                }
            ],
            finish_reason="function_call",
        )
    )

    assert message.tool_calls == [
        {
            "name": "get_weather",
            "args": {"location": "Moscow"},
            "id": "tools-state-1",
            "type": "tool_call",
        }
    ]
    assert message.additional_kwargs["tools_state_id"] == "tools-state-1"
    assert message.additional_kwargs["function_call"] == {
        "name": "get_weather",
        "arguments": {"location": "Moscow"},
    }


def test_message_level_function_call_is_supported() -> None:
    message = _message(
        _response(
            messages=[
                {
                    "role": "assistant",
                    "message_id": "provider-message-1",
                    "function_call": {
                        "name": "lookup",
                        "arguments": '{"key": "value"}',
                    },
                }
            ]
        )
    )

    assert message.tool_calls[0]["id"] == "provider-message-1"
    assert message.tool_calls[0]["args"] == {"key": "value"}


def test_invalid_function_arguments_raise_clear_error() -> None:
    response = _response(
        messages=[
            {
                "role": "assistant",
                "content": [
                    {
                        "function_call": {
                            "name": "broken",
                            "arguments": "not-json",
                        }
                    }
                ],
            }
        ]
    )

    with pytest.raises(ValueError, match="broken.*invalid JSON arguments"):
        create_chat_result(response)


@pytest.mark.parametrize(
    ("status", "expected_result_status"),
    [("success", "success"), ("failed", "error")],
)
def test_terminal_server_tool_execution_emits_result(
    status: str,
    expected_result_status: str,
) -> None:
    message = _message(
        _response(
            messages=[
                {
                    "role": "reasoning",
                    "tools_state_id": "server-state-1",
                    "content": [
                        {
                            "tool_execution": {
                                "name": "web_search",
                                "status": status,
                                "seconds_left": 0,
                            }
                        }
                    ],
                }
            ]
        )
    )

    assert message.content == [
        {
            "type": "server_tool_result",
            "id": "server-state-1:result",
            "tool_call_id": "server-state-1",
            "status": expected_result_status,
            "extras": {
                "provider_tool_execution": {
                    "name": "web_search",
                    "status": status,
                    "seconds_left": 0,
                }
            },
        }
    ]


def test_running_server_tool_execution_emits_call_only() -> None:
    message = _message(
        _response(
            tool_execution={
                "name": "image_generate",
                "status": "running",
                "seconds_left": 3,
            }
        )
    )

    server_blocks = [
        block
        for block in message.content
        if isinstance(block, dict) and block["type"].startswith("server_tool")
    ]
    assert server_blocks == [
        {
            "type": "server_tool_call",
            "id": "server_tool_response",
            "name": "image_generate",
            "args": {},
            "extras": {
                "provider_tool_execution": {
                    "name": "image_generate",
                    "status": "running",
                    "seconds_left": 3,
                }
            },
        }
    ]


def test_usage_headers_and_ids_are_preserved() -> None:
    response = _response(
        message_id="provider-message-1",
        thread_id="thread-1",
        x_headers={
            "x-request-id": "request-1",
            "x-session-id": "session-1",
        },
        additional_data=[{"kind": "provider-extra"}],
        logprobs=[{"chosen": {"token": "Hello", "token_id": 1, "logprob": -0.1}}],
    )
    result = create_chat_result(response)
    message = result.generations[0].message

    assert isinstance(message, AIMessage)
    assert message.id == "request-1"
    assert message.usage_metadata == {
        "input_tokens": 10,
        "output_tokens": 4,
        "total_tokens": 14,
        "input_token_details": {"cache_read": 2},
    }
    assert message.response_metadata["message_id"] == "provider-message-1"
    assert message.response_metadata["thread_id"] == "thread-1"
    assert message.response_metadata["x_headers"] == {
        "x-request-id": "request-1",
        "x-session-id": "session-1",
    }
    generation_info = result.generations[0].generation_info
    assert generation_info is not None
    assert generation_info["finish_reason"] == "stop"
    assert result.llm_output is not None
    assert result.llm_output["token_usage"]["input_tokens_details"] == {
        "prompt_tokens": 10,
        "cached_tokens": 2,
    }
    assert result.llm_output["provider_response"]["additional_data"] == [
        {"kind": "provider-extra"}
    ]


def test_message_finish_reason_is_used_as_fallback() -> None:
    result = create_chat_result(
        _response(
            finish_reason=None,
            messages=[
                {
                    "role": "assistant",
                    "content": [{"text": "Hello"}],
                    "finish_reason": "length",
                }
            ],
        )
    )

    generation_info = result.generations[0].generation_info
    assert generation_info is not None
    assert generation_info["finish_reason"] == "length"


def test_message_level_metadata_is_promoted_without_losing_raw_message() -> None:
    response = _response(
        message_id=None,
        messages=[
            {
                "role": "assistant",
                "message_id": "message-in-array-1",
                "tools_state_id": "tools-state-1",
                "content": [{"text": "Hello"}],
                "tool_execution": {
                    "name": "web_search",
                    "status": "success",
                },
                "logprobs": [
                    {
                        "chosen": {
                            "token": "Hello",
                            "token_id": 1,
                            "logprob": -0.1,
                        }
                    }
                ],
            }
        ],
    )
    message = _message(response)

    assert message.response_metadata["message_id"] == "message-in-array-1"
    assert message.response_metadata["provider_message_ids"] == ["message-in-array-1"]
    assert message.response_metadata["tools_state_id"] == "tools-state-1"
    assert message.response_metadata["tool_execution"]["name"] == "web_search"
    assert message.response_metadata["logprobs"][0]["chosen"]["token"] == "Hello"
    assert message.additional_kwargs["provider_messages"][0]["message_id"] == (
        "message-in-array-1"
    )


def test_unknown_content_and_provider_fields_are_preserved() -> None:
    response = _response(
        future_response_field={"enabled": True},
        messages=[
            {
                "role": "assistant",
                "future_message_field": "message-extra",
                "content": [{"future_part": {"value": 42}}],
            }
        ],
    )
    result = create_chat_result(response)
    message = result.generations[0].message

    assert message.content == [
        {
            "type": "non_standard",
            "value": {"future_part": {"value": 42}},
        },
        {
            "type": "non_standard",
            "value": {"future_message_field": "message-extra"},
        },
    ]
    assert result.llm_output is not None
    assert result.llm_output["provider_response"]["future_response_field"] == {
        "enabled": True
    }


def test_empty_messages_raise_clear_error() -> None:
    response = _response(messages=[])

    with pytest.raises(ValueError, match="contains no messages"):
        create_chat_result(response)
