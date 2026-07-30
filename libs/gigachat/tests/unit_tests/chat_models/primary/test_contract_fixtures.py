"""Contract checks for reusable primary SDK-shaped fixtures."""

from typing import Any

import gigachat.models as gm
import pytest
from pydantic import ValidationError

from .fixtures import (
    MESSAGE_ID,
    REQUEST_ID,
    THREAD_ID,
    TOOLS_STATE_ID,
    build_plain_text_response,
)


def test_plain_text_response_is_sdk_shaped(
    plain_text_response: gm.ChatCompletionResponse,
) -> None:
    assert plain_text_response.messages[0].content
    assert plain_text_response.messages[0].content[0].text == "Primary response"
    assert plain_text_response.usage
    assert plain_text_response.usage.total_tokens == 21
    assert plain_text_response.x_headers == {
        "x-request-id": REQUEST_ID,
        "x-session-id": "session-primary-001",
    }
    assert plain_text_response.message_id == MESSAGE_ID
    assert plain_text_response.thread_id == THREAD_ID


def test_multimodal_response_preserves_part_order_and_provider_data(
    multimodal_response: gm.ChatCompletionResponse,
) -> None:
    content = multimodal_response.messages[0].content
    assert content
    assert content[0].text == "Generated assets"
    assert content[1].files
    assert [file.id_ for file in content[1].files] == [
        "file-image-001",
        "file-audio-001",
    ]
    assert content[2].inline_data
    assert content[2].inline_data.sources
    assert (
        content[2].inline_data.sources["source-001"].url
        == "https://example.test/source"
    )
    assert multimodal_response.additional_data == [
        {"provider_extension": {"preserve": True}}
    ]


def test_function_call_response_has_stable_continuation_state(
    function_call_response: gm.ChatCompletionResponse,
) -> None:
    message = function_call_response.messages[0]
    assert message.function_call
    assert message.function_call.name == "lookup_weather"
    assert message.function_call.arguments == {"city": "Moscow"}
    assert message.tools_state_id == TOOLS_STATE_ID
    assert function_call_response.finish_reason == "tool_calls"


def test_tool_roundtrip_history_reuses_call_identity(
    tool_roundtrip_history: list[gm.ChatMessage],
) -> None:
    call = tool_roundtrip_history[1]
    result = tool_roundtrip_history[2]
    assert call.function_call
    assert call.function_call.name == "lookup_weather"
    assert call.tools_state_id == result.tools_state_id == TOOLS_STATE_ID
    assert result.content
    assert result.content[0].function_result
    assert result.content[0].function_result.name == "lookup_weather"
    assert result.content[0].function_result.result["temperature"] == 18


def test_message_delta_event_is_named_sdk_chunk(
    message_delta_event: gm.PrimaryChatCompletionChunk,
) -> None:
    assert message_delta_event.event == "response.message.delta"
    assert message_delta_event.messages
    assert message_delta_event.messages[0].content
    assert message_delta_event.messages[0].content[0].text == "Primary "
    assert message_delta_event.message_id == MESSAGE_ID


def test_tool_completed_event_carries_execution_and_sources(
    tool_completed_event: gm.PrimaryChatCompletionChunk,
) -> None:
    assert tool_completed_event.event == "response.tool.completed"
    assert tool_completed_event.tool_execution
    assert tool_completed_event.tool_execution.status == "completed"
    assert tool_completed_event.tools_state_id == TOOLS_STATE_ID
    assert tool_completed_event.messages
    assert tool_completed_event.messages[0].content
    inline_data = tool_completed_event.messages[0].content[0].inline_data
    assert inline_data
    assert inline_data.sources
    assert "source-001" in inline_data.sources


def test_message_done_event_is_metadata_only_and_keeps_usage(
    message_done_event: gm.PrimaryChatCompletionChunk,
) -> None:
    assert message_done_event.event == "response.message.done"
    assert message_done_event.messages is None
    assert message_done_event.finish_reason == "stop"
    assert message_done_event.usage
    assert message_done_event.usage.input_tokens_details
    assert message_done_event.usage.input_tokens_details.cached_tokens == 5
    assert message_done_event.thread_id == THREAD_ID


def test_unknown_event_is_accepted_by_current_sdk_without_losing_extras(
    unknown_event: dict[str, Any],
) -> None:
    parsed = gm.PrimaryChatCompletionChunk.model_validate(unknown_event)
    dumped = parsed.model_dump(exclude_none=True)

    assert parsed.event == "response.provider_extension.delta"
    assert dumped["provider_metadata"] == {"preserve": True}
    assert dumped["messages"][0]["content"][0]["provider_extension"] == {
        "kind": "future-content",
        "value": 42,
    }


def test_malformed_event_stays_raw_when_sdk_validation_rejects_it(
    malformed_event: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        gm.PrimaryChatCompletionChunk.model_validate(malformed_event)
    assert malformed_event["provider_error"]["code"] == "malformed_fixture"


def test_fixture_factories_return_independent_models() -> None:
    first = build_plain_text_response()
    second = build_plain_text_response()

    assert first is not second
    assert first.messages[0] is not second.messages[0]
    assert first.messages[0].content
    assert second.messages[0].content
    first.messages[0].content[0].text = "mutated"
    assert second.messages[0].content[0].text == "Primary response"
