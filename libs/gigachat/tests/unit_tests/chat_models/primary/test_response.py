"""Primary non-stream response conversion."""

from __future__ import annotations

from typing import Any

import gigachat.models as gm
import pytest
from langchain_core.messages import AIMessage

from langchain_gigachat.chat_models._contracts import primary


def _response(**overrides: Any) -> gm.ChatCompletionResponse:
    values: dict[str, Any] = {
        "model": "GigaChat-3-Ultra",
        "created_at": 1_754_000_000,
        "messages": [
            {
                "role": "assistant",
                "content": [{"text": "Primary response"}],
            }
        ],
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
    message = primary.create_chat_result(response).generations[0].message
    assert isinstance(message, AIMessage)
    return message


def test_plain_text_messages_form_one_generation() -> None:
    result = primary.create_chat_result(
        _response(
            messages=[
                {"role": "assistant", "content": [{"text": "Primary "}]},
                {"role": "assistant", "content": [{"text": "response"}]},
            ]
        )
    )

    assert len(result.generations) == 1
    assert result.generations[0].message.content == "Primary response"
    assert result.generations[0].generation_info is not None
    assert result.generations[0].generation_info["finish_reason"] == "stop"


def test_reasoning_files_and_citations_use_standard_blocks() -> None:
    message = _message(
        _response(
            messages=[
                {"role": "reasoning", "content": [{"text": "Think"}]},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "text": "Answer",
                            "inline_data": {
                                "sources": {
                                    "source-1": {
                                        "url": "https://example.test/source",
                                        "title": "Example",
                                    }
                                }
                            },
                        },
                        {"files": [{"id": "image-1", "mime": "image/png"}]},
                    ],
                },
            ]
        )
    )

    assert message.content_blocks[0] == {
        "type": "reasoning",
        "reasoning": "Think",
    }
    assert message.content_blocks[1] == {
        "type": "text",
        "text": "Answer",
        "annotations": [
            {
                "type": "citation",
                "id": "source-1",
                "url": "https://example.test/source",
                "title": "Example",
            }
        ],
    }
    assert message.content_blocks[2] == {
        "type": "image",
        "file_id": "image-1",
        "mime_type": "image/png",
    }
    assert message.additional_kwargs["reasoning_content"] == "Think"


def test_client_function_uses_tools_state_id_as_its_only_identity() -> None:
    message = _message(
        _response(
            tools_state_id="tools-state-1",
            messages=[
                {
                    "role": "assistant",
                    "function_call": {
                        "name": "lookup",
                        "arguments": {"key": "value"},
                    },
                }
            ],
            finish_reason="tool_calls",
        )
    )

    assert message.tool_calls == [
        {
            "type": "tool_call",
            "name": "lookup",
            "args": {"key": "value"},
            "id": "tools-state-1",
        }
    ]
    assert message.additional_kwargs["tools_state_id"] == "tools-state-1"


def test_invalid_function_arguments_become_invalid_tool_call() -> None:
    message = _message(
        _response(
            tools_state_id="tools-state-1",
            messages=[
                {
                    "role": "assistant",
                    "function_call": {
                        "name": "broken",
                        "arguments": "not-json",
                    },
                }
            ],
            finish_reason="tool_calls",
        )
    )

    assert message.tool_calls == []
    assert len(message.invalid_tool_calls) == 1
    assert message.invalid_tool_calls[0]["name"] == "broken"
    assert message.invalid_tool_calls[0]["args"] == "not-json"
    assert message.invalid_tool_calls[0]["id"] == "tools-state-1"


@pytest.mark.parametrize(
    ("execution", "expected_id"),
    [
        (
            {
                "call_id": "server-call-1",
                "name": "web_search",
                "status": "success",
            },
            "server-call-1",
        ),
        (
            {"name": "web_search", "status": "success"},
            "lc_primary-server-tool-0",
        ),
    ],
)
def test_server_tool_identity_is_independent_from_tools_state_id(
    execution: dict[str, Any],
    expected_id: str,
) -> None:
    message = _message(
        _response(
            tools_state_id="completion-state",
            messages=[
                {
                    "role": "assistant",
                    "tool_execution": execution,
                }
            ],
        )
    )

    result = next(
        block
        for block in message.content_blocks
        if block["type"] == "server_tool_result"
    )
    assert result["tool_call_id"] == expected_id
    assert result["tool_call_id"] != "completion-state"
    assert message.response_metadata["tools_state_ids"] == ["completion-state"]


def test_usage_headers_and_provider_ids_are_preserved() -> None:
    result = primary.create_chat_result(
        _response(
            message_id="provider-message-1",
            thread_id="thread-1",
            x_headers={
                "x-request-id": "request-1",
                "x-session-id": "session-1",
            },
        )
    )
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


def test_unknown_sdk_extension_fields_are_preserved() -> None:
    message = _message(
        _response(
            future_response_field={"enabled": True},
            messages=[
                {
                    "role": "assistant",
                    "future_message_field": "message-extra",
                    "content": [{"future_part": {"value": 42}}],
                }
            ],
        )
    )

    assert message.content_blocks == [
        {
            "type": "non_standard",
            "value": {"future_part": {"value": 42}},
        },
        {
            "type": "non_standard",
            "value": {"future_message_field": "message-extra"},
        },
    ]
    assert message.response_metadata["provider_fields"] == {
        "future_response_field": {"enabled": True}
    }


def test_multiple_client_function_calls_are_rejected() -> None:
    response = _response(
        tools_state_id="tools-state-1",
        messages=[
            {
                "role": "assistant",
                "function_call": {"name": "one", "arguments": {}},
            },
            {
                "role": "assistant",
                "function_call": {"name": "two", "arguments": {}},
            },
        ],
    )

    with pytest.raises(ValueError, match="multiple client function calls"):
        primary.create_chat_result(response)


def test_empty_messages_raise_clear_error() -> None:
    with pytest.raises(ValueError, match="contains no messages"):
        primary.create_chat_result(_response(messages=[]))
