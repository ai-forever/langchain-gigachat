"""Tests for primary non-stream response conversion."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from gigachat import models as gm
from langchain_core.messages import AIMessage, ToolMessage

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


@pytest.mark.parametrize("field_name", ["reasoning", "reasoning_content"])
@pytest.mark.parametrize("level", ["message", "part"])
def test_reasoning_aliases_have_stream_semantics(
    field_name: str,
    level: str,
) -> None:
    if level == "message":
        provider_message = {
            "role": "assistant",
            "content": [{"text": "Answer"}],
            field_name: "Think",
        }
    else:
        provider_message = {
            "role": "assistant",
            "content": [{"text": "Answer", field_name: "Think"}],
        }

    message = _message(_response(messages=[provider_message]))

    assert message.content == [
        {"type": "text", "text": "Answer"},
        {"type": "reasoning", "reasoning": "Think"},
    ]
    assert message.additional_kwargs["reasoning_content"] == "Think"


def test_structured_reasoning_preserves_provider_extras() -> None:
    message = _message(
        _response(
            messages=[
                {
                    "role": "assistant",
                    "reasoning": {
                        "text": "Think",
                        "provider_flag": True,
                    },
                }
            ]
        )
    )

    assert message.content == [
        {
            "type": "reasoning",
            "reasoning": "Think",
            "extras": {"provider_data": {"provider_flag": True}},
        }
    ]
    assert message.additional_kwargs["reasoning_content"] == "Think"


def test_reasoning_aliases_reject_conflicting_values() -> None:
    with pytest.raises(
        ValueError,
        match="conflicting reasoning and reasoning_content",
    ):
        _message(
            _response(
                messages=[
                    {
                        "role": "assistant",
                        "reasoning": "First",
                        "reasoning_content": "Second",
                    }
                ]
            )
        )


@pytest.mark.parametrize("field_name", ["message_id", "tools_state_id"])
@pytest.mark.parametrize("provider_id", ["", "   "])
def test_response_rejects_invalid_provider_identity(
    field_name: str,
    provider_id: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=f"provider {field_name} must be a non-empty string",
    ):
        _message(
            _response(
                messages=[
                    {
                        "role": "assistant",
                        "content": [{"text": "Answer"}],
                        field_name: provider_id,
                    }
                ]
            )
        )


@pytest.mark.parametrize("provider_id", ["", "   ", 7])
def test_response_rejects_invalid_top_level_tools_state_id(
    provider_id: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="provider tools_state_id must be a non-empty string",
    ):
        _message(_response(tools_state_id=provider_id))


def test_response_preserves_valid_provider_identity_without_trimming() -> None:
    message = _message(
        _response(
            message_id=" provider-message ",
            tools_state_id=" provider-state ",
        )
    )

    assert message.id == " provider-message "
    assert message.additional_kwargs["tools_state_id"] == " provider-state "


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


def test_nonstream_function_call_uses_provider_state_as_its_only_identity() -> None:
    message = _message(
        _response(
            messages=[
                {
                    "role": "assistant",
                    "tools_state_id": "provider-state",
                    "function_call": {
                        "id": "call-1",
                        "name": "lookup",
                        "arguments": {"key": "value"},
                    },
                }
            ],
            finish_reason="function_call",
        )
    )

    assert message.tool_calls == [
        {
            "name": "lookup",
            "args": {"key": "value"},
            "id": "provider-state",
            "type": "tool_call",
        }
    ]
    assert "provider_tool_state_by_call_id" not in message.additional_kwargs
    assert message.additional_kwargs["function_call"] == {
        "name": "lookup",
        "arguments": {"key": "value"},
        "id": "call-1",
    }

    converted = convert_messages(
        [
            message,
            ToolMessage(content='{"result": 1}', tool_call_id="provider-state"),
        ],
        cached_uploads={},
    )
    assert converted[0].tools_state_id == "provider-state"
    assert converted[1].tools_state_id == "provider-state"


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
            "id": "lc_primary-server-tool-0:result",
            "tool_call_id": "lc_primary-server-tool-0",
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
    assert message.additional_kwargs["provider_server_tool_state_by_call_id"] == {
        "lc_primary-server-tool-0": "server-state-1"
    }


@pytest.mark.parametrize("execution_level", ["part", "message", "response"])
def test_execution_identity_precedes_container_state_at_every_level(
    execution_level: str,
) -> None:
    execution = {
        "call_id": "execution-1",
        "tool_call_id": "execution-1",
        "id": "execution-1",
        "name": "web_search",
        "status": "success",
    }
    response_values: dict[str, object] = {}
    message_values: dict[str, object] = {
        "role": "assistant",
    }
    if execution_level == "part":
        message_values["tools_state_id"] = "state-1"
        message_values["content"] = [{"tool_execution": execution}]
    elif execution_level == "message":
        message_values["tools_state_id"] = "state-1"
        message_values["tool_execution"] = execution
    else:
        message_values["content"] = [{"text": "Generated"}]
        response_values["tools_state_id"] = "response-state-1"
        response_values["tool_execution"] = execution
    response_values["messages"] = [message_values]
    before = copy.deepcopy(response_values)

    message = _message(_response(**response_values))

    server_result = next(
        block
        for block in message.content_blocks
        if block["type"] == "server_tool_result"
    )
    assert server_result["tool_call_id"] == "execution-1"
    assert server_result["id"] == "execution-1:result"
    assert response_values == before


@pytest.mark.parametrize("execution_level", ["part", "message", "response"])
def test_conflicting_execution_identity_aliases_fail_at_every_level(
    execution_level: str,
) -> None:
    execution = {
        "call_id": "call-A",
        "tool_call_id": "call-B",
        "name": "web_search",
        "status": "success",
    }
    response_values: dict[str, object] = {}
    message_values: dict[str, object] = {"role": "assistant"}
    if execution_level == "part":
        message_values["content"] = [{"tool_execution": execution}]
    elif execution_level == "message":
        message_values["tool_execution"] = execution
    else:
        response_values["tool_execution"] = execution
    response_values["messages"] = [message_values]

    with pytest.raises(ValueError, match="identity aliases contain conflicting"):
        _message(_response(**response_values))


def test_censored_success_uses_completion_error_for_failure_semantics() -> None:
    message = _message(
        _response(
            messages=[
                {
                    "role": "assistant",
                    "tools_state_id": "state-1",
                    "tool_execution": {
                        "name": "image_generate",
                        "status": "success",
                        "censored": True,
                    },
                }
            ],
            finish_reason="error",
        )
    )

    assert message.content == [
        {
            "type": "server_tool_result",
            "id": "lc_primary-server-tool-0:result",
            "tool_call_id": "lc_primary-server-tool-0",
            "status": "success",
            "extras": {
                "provider_tool_execution": {
                    "name": "image_generate",
                    "status": "success",
                    "censored": True,
                }
            },
        }
    ]
    assert message.additional_kwargs["provider_server_tool_state_by_call_id"] == {
        "lc_primary-server-tool-0": "state-1"
    }
    assert message.response_metadata["finish_reason"] == "error"


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
            "id": "lc_primary-server-tool-0",
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


def test_message_id_does_not_conflate_unidentified_server_tools() -> None:
    message = _message(
        _response(
            message_id="provider-message-1",
            messages=[
                {
                    "role": "assistant",
                    "message_id": "provider-message-1",
                    "content": [
                        {
                            "tool_execution": {
                                "name": "web_search",
                                "status": "success",
                            }
                        },
                        {
                            "tool_execution": {
                                "name": "image_generate",
                                "status": "success",
                            }
                        },
                    ],
                }
            ],
        )
    )

    server_results = [
        block
        for block in message.content_blocks
        if block["type"] == "server_tool_result"
    ]
    assert [block["tool_call_id"] for block in server_results] == [
        "lc_primary-server-tool-0",
        "lc_primary-server-tool-1",
    ]
    assert all(
        block["tool_call_id"] != "provider-message-1" for block in server_results
    )


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
            "id": "lc_primary-server-tool-0:result",
            "tool_call_id": "lc_primary-server-tool-0",
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
    assert message.additional_kwargs["provider_server_tool_state_by_call_id"] == {
        "lc_primary-server-tool-0": "server-state-1"
    }


def test_semantically_equivalent_server_tool_mirrors_are_emitted_once() -> None:
    message = _message(
        _response(
            messages=[
                {
                    "role": "assistant",
                    "tool_execution": {
                        "call_id": "search-1",
                        "name": "web_search",
                        "status": "success",
                    },
                    "content": [
                        {
                            "tool_execution": {
                                "call_id": "search-1",
                                "name": "web_search",
                                "status": "completed",
                                "index": 0,
                            }
                        }
                    ],
                }
            ]
        )
    )

    server_results = [
        block
        for block in message.content_blocks
        if block["type"] == "server_tool_result"
    ]
    assert len(server_results) == 1
    assert server_results[0]["tool_call_id"] == "search-1"


def test_distinct_part_and_message_server_tools_are_both_preserved() -> None:
    message = _message(
        _response(
            messages=[
                {
                    "role": "reasoning",
                    "content": [
                        {
                            "tool_execution": {
                                "call_id": "search-1",
                                "name": "web_search",
                                "status": "success",
                            }
                        }
                    ],
                },
                {
                    "role": "reasoning",
                    "tool_execution": {
                        "call_id": "image-1",
                        "name": "image_generate",
                        "status": "success",
                    },
                },
            ]
        )
    )

    assert [
        block["tool_call_id"]
        for block in message.content_blocks
        if block["type"] == "server_tool_result"
    ] == ["search-1", "image-1"]


def test_distinct_message_level_server_tools_are_both_preserved() -> None:
    message = _message(
        _response(
            messages=[
                {
                    "role": "reasoning",
                    "tool_execution": {
                        "call_id": "search-1",
                        "name": "web_search",
                        "status": "success",
                    },
                },
                {
                    "role": "reasoning",
                    "tool_execution": {
                        "call_id": "image-1",
                        "name": "image_generate",
                        "status": "success",
                    },
                },
            ]
        )
    )

    assert [
        block["tool_call_id"]
        for block in message.content_blocks
        if block["type"] == "server_tool_result"
    ] == ["search-1", "image-1"]


def test_distinct_nested_and_response_server_tools_are_both_preserved() -> None:
    message = _message(
        _response(
            messages=[
                {
                    "role": "reasoning",
                    "content": [
                        {
                            "tool_execution": {
                                "call_id": "search-1",
                                "name": "web_search",
                                "status": "success",
                            }
                        }
                    ],
                }
            ],
            tool_execution={
                "call_id": "image-1",
                "name": "image_generate",
                "status": "success",
            },
        )
    )

    assert [
        block["tool_call_id"]
        for block in message.content_blocks
        if block["type"] == "server_tool_result"
    ] == ["search-1", "image-1"]


def test_same_payload_with_distinct_explicit_ids_is_not_deduplicated() -> None:
    common: dict[str, Any] = {"name": "web_search", "status": "success"}
    message = _message(
        _response(
            messages=[
                {
                    "role": "reasoning",
                    "tool_execution": common | {"call_id": "search-1"},
                    "content": [
                        {
                            "tool_execution": common
                            | {
                                "call_id": "search-2",
                            }
                        }
                    ],
                }
            ]
        )
    )

    assert [
        block["tool_call_id"]
        for block in message.content_blocks
        if block["type"] == "server_tool_result"
    ] == ["search-2", "search-1"]


def test_unidentified_mirror_of_distinct_explicit_ids_fails_closed() -> None:
    common: dict[str, Any] = {"name": "web_search", "status": "success"}
    response = _response(
        messages=[
            {
                "role": "reasoning",
                "tool_execution": common | {"call_id": "search-1"},
                "content": [{"tool_execution": common | {"call_id": "search-2"}}],
            }
        ],
        tool_execution=common,
    )

    with pytest.raises(
        ValueError,
        match="unidentified server tool mirror matches multiple distinct",
    ):
        create_chat_result(response)


def test_same_server_tool_id_with_conflicting_payload_fails_closed() -> None:
    response = _response(
        messages=[
            {
                "role": "reasoning",
                "tool_execution": {
                    "call_id": "shared-1",
                    "name": "image_generate",
                    "status": "success",
                },
                "content": [
                    {
                        "tool_execution": {
                            "call_id": "shared-1",
                            "name": "web_search",
                            "status": "success",
                        }
                    }
                ],
            }
        ]
    )

    with pytest.raises(
        ValueError,
        match="same provider identity.*conflicting payloads",
    ):
        create_chat_result(response)


def test_shared_container_state_for_distinct_server_tools_fails_closed() -> None:
    response = _response(
        messages=[
            {
                "role": "reasoning",
                "tools_state_id": "shared-state",
                "content": [
                    {
                        "tool_execution": {
                            "name": "web_search",
                            "status": "success",
                        }
                    },
                    {
                        "tool_execution": {
                            "name": "image_generate",
                            "status": "success",
                        }
                    },
                ],
            }
        ]
    )

    with pytest.raises(
        ValueError,
        match="tools_state_id 'shared-state'.*multiple distinct server tools",
    ):
        create_chat_result(response)


def test_multiple_server_owned_states_are_supported_non_stream() -> None:
    message = _message(
        _response(
            messages=[
                {
                    "role": "reasoning",
                    "tools_state_id": "search-state",
                    "tool_execution": {
                        "name": "web_search",
                        "status": "success",
                    },
                },
                {
                    "role": "reasoning",
                    "tools_state_id": "image-state",
                    "tool_execution": {
                        "name": "image_generate",
                        "status": "success",
                    },
                },
            ]
        )
    )

    assert [
        block["tool_call_id"]
        for block in message.content_blocks
        if block["type"] == "server_tool_result"
    ] == ["lc_primary-server-tool-0", "lc_primary-server-tool-1"]
    assert message.additional_kwargs["provider_server_tool_state_by_call_id"] == {
        "lc_primary-server-tool-0": "search-state",
        "lc_primary-server-tool-1": "image-state",
    }
    assert message.additional_kwargs["tools_state_ids"] == [
        "search-state",
        "image-state",
    ]
    assert message.additional_kwargs["tools_state_id"] == "image-state"
    assert message.response_metadata["tools_state_id"] == "image-state"


def test_explicit_server_id_keeps_container_state_available_to_client_call() -> None:
    message = _message(
        _response(
            messages=[
                {
                    "role": "assistant",
                    "tools_state_id": "client-state",
                    "function_call": {
                        "name": "lookup_weather",
                        "arguments": {"city": "Moscow"},
                    },
                    "tool_execution": {
                        "call_id": "server-call",
                        "name": "web_search",
                        "status": "success",
                    },
                }
            ]
        )
    )

    assert message.tool_calls[0]["id"] == "client-state"
    server_result = next(
        block
        for block in message.content_blocks
        if block["type"] == "server_tool_result"
    )
    assert server_result["tool_call_id"] == "server-call"


def test_shared_state_for_client_and_idless_server_tool_fails_closed() -> None:
    response = _response(
        messages=[
            {
                "role": "assistant",
                "tools_state_id": "ambiguous-state",
                "function_call": {
                    "name": "lookup_weather",
                    "arguments": {"city": "Moscow"},
                },
                "tool_execution": {
                    "name": "web_search",
                    "status": "success",
                },
            }
        ]
    )

    with pytest.raises(
        ValueError,
        match="tools_state_id ownership is ambiguous",
    ):
        create_chat_result(response)


def test_client_call_with_multiple_unowned_states_fails_closed() -> None:
    response = _response(
        tools_state_id="response-state",
        messages=[
            {
                "role": "assistant",
                "tools_state_id": "message-state",
                "function_call": {
                    "name": "lookup_weather",
                    "arguments": {"city": "Moscow"},
                },
            }
        ],
    )

    with pytest.raises(
        ValueError,
        match="client function call has multiple possible tools_state_id values",
    ):
        create_chat_result(response)


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
            "id": "lc_primary-server-tool-0:result",
            "tool_call_id": "lc_primary-server-tool-0",
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
    assert message.additional_kwargs["provider_server_tool_state_by_call_id"] == {
        "lc_primary-server-tool-0": "server-state-1"
    }


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


def test_provider_message_id_is_ai_message_id_fallback() -> None:
    message = _message(_response(message_id="provider-message-1"))

    assert message.id == "provider-message-1"
    assert message.response_metadata["provider_message_ids"] == ["provider-message-1"]


def test_request_id_header_lookup_is_case_insensitive() -> None:
    message = _message(_response(x_headers={"X-Request-Id": "provider-request-1"}))

    assert message.id == "provider-request-1"


@pytest.mark.parametrize("request_id", ["", "   "])
def test_response_rejects_invalid_request_id_header(request_id: str) -> None:
    with pytest.raises(
        ValueError,
        match="provider x-request-id must be a non-empty string",
    ):
        _message(_response(x_headers={"X-Request-Id": request_id}))


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

    assert message.id == "message-in-array-1"
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


def test_multiple_unassigned_tools_state_ids_are_retained_without_singular_state() -> (
    None
):
    message = _message(
        _response(
            tools_state_id="response-state",
            messages=[
                {
                    "role": "assistant",
                    "tools_state_id": "nested-state",
                    "content": [{"text": "Hello"}],
                }
            ],
        )
    )

    assert message.content == "Hello"
    assert message.additional_kwargs["tools_state_ids"] == [
        "response-state",
        "nested-state",
    ]
    assert "tools_state_id" not in message.additional_kwargs
    assert message.response_metadata["tools_state_ids"] == [
        "response-state",
        "nested-state",
    ]
    assert "tools_state_id" not in message.response_metadata


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
