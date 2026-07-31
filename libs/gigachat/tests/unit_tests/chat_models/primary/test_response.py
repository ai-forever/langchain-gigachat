"""Tests for primary non-stream response conversion."""

from __future__ import annotations

import copy

import pytest
from gigachat import models as gm
from langchain_core.messages import AIMessage

from langchain_gigachat.chat_models._contracts.primary.messages import (
    convert_messages,
)
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
    assert (
        result.generations[0].message.additional_kwargs["reasoning_content"] == "Think"
    )


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
                                {"id": "video-1", "mime": "video/mp4"},
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
        {"type": "video", "file_id": "video-1", "mime_type": "video/mp4"},
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
    assert message.content == []
    assert message.content_blocks == message.tool_calls

    provider_message = convert_messages([message], cached_uploads={})[0]
    assert provider_message.tools_state_id == "tools-state-1"
    assert provider_message.function_call is None
    assert provider_message.content
    assert provider_message.content[0].function_call is not None
    assert provider_message.content[0].function_call.model_dump(
        exclude_none=True,
        by_alias=True,
    ) == {
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
                    "tools_state_id": "tools-state-1",
                    "function_call": {
                        "name": "lookup",
                        "arguments": '{"key": "value"}',
                    },
                }
            ]
        )
    )

    assert message.tool_calls[0]["id"] == "tools-state-1"
    assert message.tool_calls[0]["args"] == {"key": "value"}


@pytest.mark.parametrize(
    "function_call_location",
    ["message", "part"],
)
def test_function_call_without_provider_state_fails_closed(
    function_call_location: str,
) -> None:
    function_call = {
        "name": "lookup",
        "arguments": {"key": "value"},
    }
    message: dict[str, object] = {
        "role": "assistant",
        "message_id": "provider-message-1",
    }
    if function_call_location == "message":
        message["function_call"] = function_call
    else:
        message["content"] = [{"function_call": function_call}]

    with pytest.raises(
        ValueError,
        match=(
            "client function call is missing tools_state_id.*"
            "message_id values are not continuation state.*"
            "'name': 'lookup'"
        ),
    ):
        create_chat_result(_response(messages=[message]))


@pytest.mark.parametrize("state_field", ["tools_state_id", "tool_state_id"])
def test_function_call_uses_response_level_tools_state_id(state_field: str) -> None:
    message = _message(
        _response(
            **{state_field: "response-tools-state"},
            messages=[
                {
                    "role": "assistant",
                    "message_id": "provider-message-1",
                    "function_call": {
                        "name": "lookup",
                        "arguments": {"key": "value"},
                    },
                }
            ],
        )
    )

    assert message.tool_calls == [
        {
            "type": "tool_call",
            "name": "lookup",
            "args": {"key": "value"},
            "id": "response-tools-state",
        }
    ]


def test_invalid_function_arguments_become_invalid_tool_call() -> None:
    message = _message(
        _response(
            message_id="provider-message-1",
            tools_state_id="tools-state-1",
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "text": "I could not prepare the tool arguments.",
                            "function_call": {
                                "name": "broken",
                                "arguments": "not-json",
                            },
                        }
                    ],
                }
            ],
        )
    )

    assert message.content == [
        {"type": "text", "text": "I could not prepare the tool arguments."}
    ]
    assert message.tool_calls == []
    assert message.invalid_tool_calls == [
        {
            "type": "invalid_tool_call",
            "name": "broken",
            "args": "not-json",
            "id": "tools-state-1",
            "error": (
                "Function 'broken' arguments contain invalid JSON: "
                "Expecting value: line 1 column 1 (char 0)"
            ),
        }
    ]
    assert message.additional_kwargs["function_call"]["arguments"] == "not-json"


def test_valid_and_invalid_function_calls_fail_during_response_conversion() -> None:
    response = _response(
        tools_state_id="tools-state-1",
        messages=[
            {
                "role": "assistant",
                "function_call": {
                    "name": "lookup",
                    "arguments": {"key": "value"},
                },
            },
            {
                "role": "assistant",
                "function_call": {
                    "name": "broken",
                    "arguments": "[]",
                },
            },
        ],
    )

    with pytest.raises(
        ValueError,
        match="multiple client function calls.*cannot be replayed",
    ):
        create_chat_result(response)


def test_multiple_valid_function_calls_fail_during_response_conversion() -> None:
    response = _response(
        tools_state_id="tools-state-1",
        messages=[
            {
                "role": "assistant",
                "function_call": {
                    "name": "lookup",
                    "arguments": {"key": "one"},
                },
            },
            {
                "role": "assistant",
                "function_call": {
                    "name": "lookup",
                    "arguments": {"key": "two"},
                },
            },
        ],
    )

    with pytest.raises(
        ValueError,
        match="multiple client function calls.*cannot be replayed",
    ):
        create_chat_result(response)


def test_identical_function_calls_in_different_messages_are_not_deduplicated() -> None:
    function_call = {
        "name": "lookup",
        "arguments": {"key": "value"},
    }
    response = _response(
        tools_state_id="tools-state-1",
        messages=[
            {"role": "assistant", "function_call": function_call},
            {"role": "assistant", "function_call": function_call},
        ],
    )

    with pytest.raises(
        ValueError,
        match="multiple client function calls.*duplicate-looking calls",
    ):
        create_chat_result(response)


def test_identical_part_level_function_calls_are_not_deduplicated() -> None:
    function_call = {
        "name": "lookup",
        "arguments": {"key": "value"},
    }
    response = _response(
        tools_state_id="tools-state-1",
        messages=[
            {
                "role": "assistant",
                "content": [
                    {"function_call": function_call},
                    {"function_call": function_call},
                ],
            }
        ],
    )

    with pytest.raises(
        ValueError,
        match="multiple client function calls.*duplicate-looking calls",
    ):
        create_chat_result(response)


def test_mirrored_function_call_is_emitted_once() -> None:
    function_call = {
        "name": "lookup",
        "arguments": {"key": "value"},
    }
    message = _message(
        _response(
            messages=[
                {
                    "role": "assistant",
                    "tools_state_id": "tools-state-1",
                    "function_call": function_call,
                    "content": [{"function_call": function_call}],
                }
            ],
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
    assert message.additional_kwargs["function_calls"] == [function_call]
    assert message.additional_kwargs["function_call"] == function_call


def test_conflicting_mirrored_function_calls_fail_closed() -> None:
    response = _response(
        messages=[
            {
                "role": "assistant",
                "tools_state_id": "tools-state-1",
                "function_call": {
                    "name": "lookup",
                    "arguments": {"key": "message"},
                },
                "content": [
                    {
                        "function_call": {
                            "name": "lookup",
                            "arguments": {"key": "part"},
                        }
                    }
                ],
            }
        ],
    )

    with pytest.raises(
        ValueError,
        match="conflicting part-level and message-level client function calls",
    ):
        create_chat_result(response)


@pytest.mark.parametrize("name", ["", "   "])
def test_empty_function_name_becomes_invalid_tool_call(name: str) -> None:
    message = _message(
        _response(
            tools_state_id="tools-state-1",
            messages=[
                {
                    "role": "assistant",
                    "function_call": {
                        "name": name,
                        "arguments": {"key": "value"},
                    },
                }
            ],
        )
    )

    assert message.tool_calls == []
    assert message.invalid_tool_calls == [
        {
            "type": "invalid_tool_call",
            "name": name,
            "args": '{"key":"value"}',
            "id": "tools-state-1",
            "error": "Function call name must be a non-empty string.",
        }
    ]


def test_response_conversion_does_not_mutate_provider_model() -> None:
    response = _response(
        tools_state_id="tools-state-1",
        messages=[
            {
                "role": "assistant",
                "content": [
                    {
                        "function_call": {
                            "name": "lookup",
                            "arguments": {"nested": {"value": 1}},
                        }
                    }
                ],
            }
        ],
        additional_data=[{"nested": {"value": 2}}],
    )
    before = copy.deepcopy(response.model_dump(exclude_none=False, by_alias=True))

    create_chat_result(response)

    assert response.model_dump(exclude_none=False, by_alias=True) == before


def test_response_metadata_is_deeply_detached_from_provider_model() -> None:
    response = _response(
        additional_data=[{"nested": {"value": 1}}],
        future_response_field={"nested": {"value": 2}},
    )
    before = copy.deepcopy(response.model_dump(exclude_none=False, by_alias=True))

    message = _message(response)
    message.response_metadata["additional_data"][0]["nested"]["value"] = 10
    message.response_metadata["provider_fields"]["future_response_field"]["nested"][
        "value"
    ] = 20

    assert response.model_dump(exclude_none=False, by_alias=True) == before


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


def test_mirrored_server_tool_execution_uses_part_level_once() -> None:
    execution = {
        "name": "web_search",
        "status": "success",
        "seconds_left": 0,
    }
    message = _message(
        _response(
            messages=[
                {
                    "role": "assistant",
                    "tools_state_id": "server-state-1",
                    "tool_execution": execution,
                    "content": [
                        {
                            "tool_execution": execution,
                            "inline_data": {
                                "sources": {
                                    "source-1": {
                                        "url": "https://example.test/source",
                                        "title": "Example",
                                    }
                                },
                                "widgets": [{"kind": "table"}],
                                "images": [{"id": "image-1"}],
                            },
                            "provider_extension": {"trace_id": "trace-1"},
                        }
                    ],
                }
            ],
            tool_execution=execution,
        )
    )

    assert message.content == [
        {
            "type": "server_tool_result",
            "id": "server-state-1:result",
            "tool_call_id": "server-state-1",
            "status": "success",
            "extras": {
                "provider_tool_execution": execution,
                "inline_data": {
                    "images": [{"id": "image-1"}],
                    "sources": {
                        "source-1": {
                            "url": "https://example.test/source",
                            "title": "Example",
                        }
                    },
                    "widgets": [{"kind": "table"}],
                },
                "provider_data": {"provider_extension": {"trace_id": "trace-1"}},
            },
        }
    ]


def test_message_server_tool_execution_owns_inline_data() -> None:
    message = _message(
        _response(
            messages=[
                {
                    "role": "assistant",
                    "tools_state_id": "server-state-1",
                    "tool_execution": {
                        "name": "web_search",
                        "status": "failed",
                    },
                    "inline_data": {
                        "sources": {
                            "source-1": {
                                "url": "https://example.test/source",
                            }
                        }
                    },
                }
            ]
        )
    )

    assert message.content == [
        {
            "type": "server_tool_result",
            "id": "server-state-1:result",
            "tool_call_id": "server-state-1",
            "status": "error",
            "extras": {
                "provider_tool_execution": {
                    "name": "web_search",
                    "status": "failed",
                },
                "inline_data": {
                    "sources": {
                        "source-1": {
                            "url": "https://example.test/source",
                        }
                    }
                },
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
    assert message.response_metadata["additional_data"] == [{"kind": "provider-extra"}]


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


def test_message_level_metadata_is_promoted_without_raw_message_copy() -> None:
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
    assert "provider_messages" not in message.additional_kwargs


def test_multiple_provider_message_ids_fail_before_returning_message() -> None:
    response = _response(
        message_id="response-message",
        messages=[
            {
                "role": "assistant",
                "message_id": "nested-message",
                "content": [{"text": "Hello"}],
            }
        ],
    )

    with pytest.raises(
        ValueError,
        match="multiple provider message_id values.*replay semantics are unsupported",
    ):
        create_chat_result(response)


def test_multiple_tools_state_ids_fail_before_returning_message() -> None:
    response = _response(
        tools_state_id="response-state",
        messages=[
            {
                "role": "assistant",
                "tools_state_id": "nested-state",
                "content": [{"text": "Hello"}],
            }
        ],
    )

    with pytest.raises(
        ValueError,
        match="multiple tools_state_id values.*replay semantics are unsupported",
    ):
        create_chat_result(response)


def test_repeated_identical_provider_ids_are_deduplicated() -> None:
    message = _message(
        _response(
            message_id="message-1",
            tools_state_id="state-1",
            messages=[
                {
                    "role": "assistant",
                    "message_id": "message-1",
                    "tools_state_id": "state-1",
                    "content": [{"text": "Hello"}],
                },
                {
                    "role": "assistant",
                    "message_id": "message-1",
                    "tools_state_id": "state-1",
                    "content": [{"text": " again"}],
                },
            ],
        )
    )

    assert message.content == "Hello again"
    assert message.response_metadata["message_id"] == "message-1"
    assert message.response_metadata["tools_state_id"] == "state-1"
    assert message.additional_kwargs["tools_state_id"] == "state-1"


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
    assert message.response_metadata["provider_fields"]["future_response_field"] == {
        "enabled": True
    }
    assert "provider_response" not in result.llm_output


def test_empty_messages_raise_clear_error() -> None:
    response = _response(messages=[])

    with pytest.raises(ValueError, match="contains no messages"):
        create_chat_result(response)
