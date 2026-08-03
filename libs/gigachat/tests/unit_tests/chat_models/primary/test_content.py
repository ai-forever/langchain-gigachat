"""Tests for shared primary response and stream content helpers."""

from __future__ import annotations

import pytest
from gigachat import models as gm

from langchain_gigachat.chat_models._contracts.primary.content import (
    citation_annotations,
    convert_function_call,
    convert_provider_file,
    convert_reasoning_value,
    convert_text_content,
    convert_tool_execution,
    create_usage_metadata,
    json_fragment,
    parse_function_arguments,
    provider_dict,
    server_tool_execution_id,
    unknown_provider_fields,
    validate_provider_id,
)


def test_provider_mapping_helpers_detach_known_and_unknown_fields() -> None:
    part = gm.ChatContentPart.model_validate(
        {
            "text": "answer",
            "future_field": {"enabled": True},
        }
    )

    dumped = provider_dict(part)
    dumped["future_field"]["enabled"] = False

    assert part.model_extra == {"future_field": {"enabled": True}}
    assert unknown_provider_fields(part, {"text"}) == {
        "future_field": {"enabled": True}
    }


def test_provider_mapping_outputs_are_deeply_detached() -> None:
    provider_value = {
        "known": {"items": [{"value": 1}]},
        "future_field": {"items": [{"value": 2}]},
    }

    dumped = provider_dict(provider_value)
    unknown = unknown_provider_fields(provider_value, {"known"})
    text_block = convert_text_content(
        "answer",
        role="assistant",
        provider_data=provider_value,
    )

    dumped["known"]["items"][0]["value"] = 10
    unknown["future_field"]["items"][0]["value"] = 20
    text_block["extras"]["provider_data"]["known"]["items"][0]["value"] = 30

    assert provider_value == {
        "known": {"items": [{"value": 1}]},
        "future_field": {"items": [{"value": 2}]},
    }


def test_provider_identity_validator_preserves_valid_values_without_trimming() -> None:
    assert validate_provider_id(None, field="test ID") is None
    assert validate_provider_id(" provider-id ", field="test ID") == " provider-id "


@pytest.mark.parametrize("provider_id", ["", "   ", 7])
def test_provider_identity_validator_rejects_invalid_values(
    provider_id: object,
) -> None:
    with pytest.raises(ValueError, match="provider test ID must be a non-empty string"):
        validate_provider_id(provider_id, field="test ID")


def test_usage_metadata_preserves_cache_accounting() -> None:
    usage = gm.ChatUsage(
        input_tokens=11,
        input_tokens_details=gm.ChatUsageInputTokensDetails(
            prompt_tokens=11,
            cached_tokens=4,
        ),
        output_tokens=3,
        total_tokens=None,
    )

    assert create_usage_metadata(usage) == {
        "input_tokens": 11,
        "output_tokens": 3,
        "total_tokens": 14,
        "input_token_details": {"cache_read": 4},
    }
    assert create_usage_metadata(None) is None


def test_usage_metadata_falls_back_to_prompt_tokens() -> None:
    usage = gm.ChatUsage(
        input_tokens=None,
        input_tokens_details=gm.ChatUsageInputTokensDetails(
            prompt_tokens=7,
            cached_tokens=2,
        ),
        output_tokens=3,
        total_tokens=None,
    )

    assert create_usage_metadata(usage) == {
        "input_tokens": 7,
        "output_tokens": 3,
        "total_tokens": 10,
        "input_token_details": {"cache_read": 2},
    }


def test_usage_metadata_prefers_explicit_zero_input_tokens() -> None:
    usage = gm.ChatUsage(
        input_tokens=0,
        input_tokens_details=gm.ChatUsageInputTokensDetails(prompt_tokens=7),
        output_tokens=3,
        total_tokens=None,
    )

    assert create_usage_metadata(usage) == {
        "input_tokens": 0,
        "output_tokens": 3,
        "total_tokens": 3,
    }


def test_citations_and_text_extras_use_standard_blocks() -> None:
    inline_data = gm.ChatInlineData.model_validate(
        {
            "sources": {
                "source-1": {
                    "url": "https://example.test",
                    "title": "Example",
                    "page": 3,
                }
            },
            "widgets": [{"kind": "table"}],
            "future_inline_field": "preserved",
        }
    )

    assert citation_annotations(inline_data) == [
        {
            "type": "citation",
            "id": "source-1",
            "url": "https://example.test",
            "title": "Example",
            "extras": {"provider_data": {"page": 3}},
        }
    ]
    assert convert_text_content(
        "answer",
        role="assistant",
        inline_data=inline_data,
        provider_data={"future_part_field": 42},
        index=2,
    ) == {
        "type": "text",
        "text": "answer",
        "index": 2,
        "annotations": [
            {
                "type": "citation",
                "id": "source-1",
                "url": "https://example.test",
                "title": "Example",
                "extras": {"provider_data": {"page": 3}},
            }
        ],
        "extras": {
            "inline_data": {
                "widgets": [{"kind": "table"}],
                "future_inline_field": "preserved",
            },
            "provider_data": {"future_part_field": 42},
        },
    }


def test_reasoning_text_uses_reasoning_block() -> None:
    assert convert_text_content("thinking", role="reasoning") == {
        "type": "reasoning",
        "reasoning": "thinking",
    }


def test_structured_reasoning_uses_non_null_text_alias() -> None:
    block = convert_reasoning_value({"reasoning": None, "text": "Think"})

    assert block == {"type": "reasoning", "reasoning": "Think"}


def test_structured_reasoning_rejects_conflicting_text_aliases() -> None:
    with pytest.raises(ValueError, match="conflicting reasoning and text"):
        convert_reasoning_value({"reasoning": "First", "text": "Second"})


def test_provider_file_mime_mapping_and_extras() -> None:
    assert convert_provider_file(
        {
            "id": "video-1",
            "mime": "video/mp4",
            "target": "preview",
            "provider_flag": True,
        },
        index="file-0",
    ) == {
        "type": "video",
        "file_id": "video-1",
        "mime_type": "video/mp4",
        "index": "file-0",
        "extras": {
            "target": "preview",
            "provider_flag": True,
        },
    }


@pytest.mark.parametrize("file_id", [None, "", "   ", 7])
def test_provider_file_rejects_invalid_identity(file_id: object) -> None:
    with pytest.raises(ValueError, match="provider file ID must be a non-empty string"):
        convert_provider_file({"id": file_id, "mime": "image/png"})


def test_server_tool_conversion_supports_response_and_stream_blocks() -> None:
    assert convert_tool_execution(
        {
            "name": "web_search",
            "status": "running",
            "arguments": {"query": "weather"},
        },
        tool_call_id="tool-1",
    ) == [
        {
            "type": "server_tool_call",
            "id": "tool-1",
            "name": "web_search",
            "args": {"query": "weather"},
            "extras": {
                "provider_tool_execution": {
                    "name": "web_search",
                    "status": "running",
                    "arguments": {"query": "weather"},
                }
            },
        }
    ]
    assert (
        convert_tool_execution(
            {
                "name": "web_search",
                "status": "running",
                "arguments": {"query": "weather"},
            },
            tool_call_id="tool-1",
            streaming=True,
            index=3,
        )[0]["args"]
        == '{"query":"weather"}'
    )
    assert convert_tool_execution(
        {
            "name": "web_search",
            "status": "running",
            "output": {"items": []},
        },
        tool_call_id="tool-1",
        event_name="response.tool.completed",
        index=4,
    ) == [
        {
            "type": "server_tool_result",
            "id": "tool-1:result",
            "tool_call_id": "tool-1",
            "status": "success",
            "output": {"items": []},
            "index": 4,
            "extras": {
                "provider_tool_execution": {
                    "name": "web_search",
                    "status": "running",
                    "output": {"items": []},
                }
            },
        }
    ]


@pytest.mark.parametrize(
    ("execution", "expected"),
    [
        (
            {
                "call_id": "call-id",
                "tool_call_id": "call-id",
                "id": "call-id",
            },
            "call-id",
        ),
        (
            {"tool_call_id": "tool-call-id", "id": "tool-call-id"},
            "tool-call-id",
        ),
        ({"id": "id"}, "id"),
        ({}, None),
    ],
)
def test_server_tool_execution_identity_aliases_deduplicate(
    execution: dict[str, object],
    expected: str | None,
) -> None:
    before = execution.copy()

    assert server_tool_execution_id(execution) == expected
    assert execution == before


@pytest.mark.parametrize(
    "execution",
    [
        {"call_id": "call-id", "tool_call_id": "tool-call-id"},
        {"tool_call_id": "tool-call-id", "id": "id"},
        {"call_id": "call-id", "id": "id"},
    ],
)
def test_server_tool_execution_rejects_conflicting_identity_aliases(
    execution: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="identity aliases contain conflicting"):
        server_tool_execution_id(execution)


@pytest.mark.parametrize(
    "execution",
    [
        {"call_id": ""},
        {"call_id": "valid", "tool_call_id": "   "},
        {"call_id": "valid", "tool_call_id": "valid", "id": 7},
    ],
)
def test_server_tool_execution_rejects_invalid_explicit_identity(
    execution: dict[str, object],
) -> None:
    with pytest.raises(
        ValueError,
        match="provider server tool .* must be a non-empty string",
    ):
        server_tool_execution_id(execution)


def test_censored_success_remains_success_with_explicit_provider_evidence() -> None:
    """Match the pinned SDK fixture's status/censorship separation."""
    execution = {
        "name": "image_generate",
        "status": "success",
        "censored": True,
    }

    assert convert_tool_execution(
        execution,
        tool_call_id="image-call-1",
        event_name="response.tool.completed",
    ) == [
        {
            "type": "server_tool_result",
            "id": "image-call-1:result",
            "tool_call_id": "image-call-1",
            "status": "success",
            "extras": {"provider_tool_execution": execution},
        }
    ]


def test_function_argument_parser_preserves_raw_invalid_json() -> None:
    valid = parse_function_arguments(
        '{"city": "Moscow"}',
        function_name="weather",
    )
    invalid = parse_function_arguments(
        '{"city":',
        function_name="weather",
    )
    non_object = parse_function_arguments("[1, 2]", function_name="weather")

    assert valid.value == {"city": "Moscow"}
    assert valid.raw == '{"city": "Moscow"}'
    assert valid.error is None
    assert invalid.value is None
    assert invalid.raw == '{"city":'
    assert invalid.error is not None
    assert "invalid JSON" in invalid.error
    assert non_object.value is None
    assert non_object.raw == "[1, 2]"
    assert non_object.error == (
        "Function 'weather' arguments must be a JSON object; got list"
    )


def test_function_call_conversion_uses_standard_langchain_helpers() -> None:
    valid, invalid = convert_function_call(
        {
            "name": "weather",
            "arguments": {"city": "Moscow"},
        },
        tool_call_id="call-1",
    )
    assert valid == {
        "type": "tool_call",
        "name": "weather",
        "args": {"city": "Moscow"},
        "id": "call-1",
    }
    assert invalid is None

    valid, invalid = convert_function_call(
        {
            "name": "weather",
            "arguments": "broken",
        },
        tool_call_id="call-2",
    )
    assert valid is None
    assert invalid is not None
    assert invalid["type"] == "invalid_tool_call"
    assert invalid["id"] == "call-2"
    assert invalid["name"] == "weather"
    assert invalid["args"] == "broken"
    assert invalid["error"] is not None


@pytest.mark.parametrize(
    "function_call",
    [
        {"arguments": {"city": "Moscow"}},
        {"name": "", "arguments": {"city": "Moscow"}},
        {"name": "   ", "arguments": {"city": "Moscow"}},
    ],
)
def test_function_call_conversion_rejects_missing_or_empty_name(
    function_call: dict[str, object],
) -> None:
    valid, invalid = convert_function_call(
        function_call,
        tool_call_id="call-1",
    )

    assert valid is None
    assert invalid is not None
    assert invalid["type"] == "invalid_tool_call"
    assert invalid["id"] == "call-1"
    assert invalid["args"] == '{"city":"Moscow"}'
    assert invalid["error"] == "Function call name must be a non-empty string."


def test_json_fragment_preserves_string_and_compacts_objects() -> None:
    assert json_fragment('{"partial":') == '{"partial":'
    assert json_fragment({"city": "Москва"}) == '{"city":"Москва"}'
