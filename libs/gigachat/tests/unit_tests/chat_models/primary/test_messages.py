import copy
import hashlib
from typing import Any

import gigachat.models as gm
import pytest
from langchain_core.messages import (
    AIMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langchain_gigachat.chat_models._contracts import primary


def _dump(messages: list[gm.ChatMessage]) -> list[dict]:
    return [
        message.model_dump(exclude_none=True, by_alias=True) for message in messages
    ]


def test_convert_messages_preserves_text_roles_and_custom_role() -> None:
    messages = [
        SystemMessage(content="rules"),
        HumanMessage(content="question"),
        AIMessage(content="answer"),
        ChatMessage(role="critic", content="review"),
    ]

    assert _dump(primary.convert_messages(messages, cached_uploads={})) == [
        {"content": [{"text": "rules"}], "role": "system"},
        {"content": [{"text": "question"}], "role": "user"},
        {"content": [{"text": "answer"}], "role": "assistant"},
        {"content": [{"text": "review"}], "role": "critic"},
    ]


def test_convert_messages_preserves_mixed_content_order_and_mime() -> None:
    data_url = "data:audio/mpeg;base64,YXVkaW8="
    cached_uploads = {hashlib.sha256(data_url.encode()).hexdigest(): "uploaded-audio"}
    content: list[str | dict[Any, Any]] = [
        {"type": "text", "text": "first"},
        {"type": "image", "file_id": "image-1", "mime_type": "image/png"},
        {"type": "audio_url", "audio_url": {"url": data_url}},
        "last",
    ]
    message = HumanMessage(
        content=content,
        additional_kwargs={"attachments": ["document-1"]},
    )

    converted = primary.convert_messages(
        [message],
        cached_uploads=cached_uploads,
    )[0]

    assert converted.model_dump(exclude_none=True, by_alias=True) == {
        "content": [
            {"text": "first"},
            {"files": [{"id": "image-1", "mime": "image/png"}]},
            {"files": [{"id": "uploaded-audio", "mime": "audio/mpeg"}]},
            {"text": "last"},
            {"files": [{"id": "document-1"}]},
        ],
        "role": "user",
    }


def test_convert_messages_does_not_mutate_content_or_cache() -> None:
    content: list[str | dict[Any, Any]] = [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"giga_id": "image-1"}},
    ]
    cache = {"hash": "file"}
    message = HumanMessage(content=content)
    original_content = copy.deepcopy(content)
    original_cache = copy.deepcopy(cache)

    primary.convert_messages([message], cached_uploads=cache)

    assert message.content == original_content
    assert cache == original_cache


def test_convert_messages_ai_function_call_preserves_provider_ids() -> None:
    message = AIMessage(
        content="calling",
        id="langchain-id",
        additional_kwargs={"message_id": "provider-message"},
        tool_calls=[
            {
                "name": "weather",
                "args": {"city": "Moscow"},
                "id": "tools-state",
                "type": "tool_call",
            }
        ],
    )

    converted = primary.convert_messages([message], cached_uploads={})[0]

    assert converted.model_dump(exclude_none=True, by_alias=True) == {
        "message_id": "provider-message",
        "content": [
            {"text": "calling"},
            {
                "function_call": {
                    "name": "weather",
                    "arguments": {"city": "Moscow"},
                }
            },
        ],
        "tools_state_id": "tools-state",
        "role": "assistant",
    }


def test_convert_messages_prefers_additional_metadata_over_response_metadata() -> None:
    message = AIMessage(
        content="answer",
        additional_kwargs={
            "message_id": "additional-message",
            "functions_state_id": "additional-state",
        },
        response_metadata={
            "message_id": "response-message",
            "tools_state_id": "response-state",
        },
    )

    converted = primary.convert_messages([message], cached_uploads={})[0]

    assert converted.message_id == "additional-message"
    assert converted.tools_state_id == "additional-state"


def test_convert_messages_uses_response_metadata_aliases_as_fallback() -> None:
    message = AIMessage(
        content="answer",
        response_metadata={
            "message_id": "response-message",
            "tool_state_id": "response-state",
        },
    )

    converted = primary.convert_messages([message], cached_uploads={})[0]

    assert converted.message_id == "response-message"
    assert converted.tools_state_id == "response-state"


def test_convert_messages_rejects_parallel_client_tool_calls() -> None:
    message = AIMessage(
        content="",
        tool_calls=[
            {"name": "one", "args": {}, "id": "1", "type": "tool_call"},
            {"name": "two", "args": {}, "id": "2", "type": "tool_call"},
        ],
    )

    with pytest.raises(ValueError, match="multiple client function calls"):
        primary.convert_messages([message], cached_uploads={})


def test_convert_messages_tool_result_resolves_name_from_history() -> None:
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "weather",
                    "args": {"city": "Moscow"},
                    "id": "tools-state",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content='{"temperature": 20}',
            tool_call_id="tools-state",
        ),
    ]

    converted = primary.convert_messages(messages, cached_uploads={})[1]

    assert converted.model_dump(exclude_none=True, by_alias=True) == {
        "content": [
            {
                "function_result": {
                    "name": "weather",
                    "result": {"temperature": 20},
                }
            }
        ],
        "tools_state_id": "tools-state",
        "role": "tool",
    }


def test_convert_messages_tool_result_prefers_explicit_name() -> None:
    message = ToolMessage(
        content="not json",
        tool_call_id="tools-state",
        name="weather",
    )

    converted = primary.convert_messages([message], cached_uploads={})[0]

    assert converted.content
    assert converted.content[0].function_result
    assert converted.content[0].function_result.result == "not json"


def test_convert_messages_tool_result_accepts_nested_json_without_mutation() -> None:
    content = [
        {
            "payload": [
                None,
                False,
                0,
                1.5,
                "001",
                {"nested": ["value", {"ok": True}]},
            ]
        },
        {"type": "text", "text": "plain text"},
    ]
    message = ToolMessage(
        content=content,
        tool_call_id="tools-state",
        name="weather",
    )
    original = copy.deepcopy(message.content)

    converted = primary.convert_messages([message], cached_uploads={})[0]

    assert converted.content
    assert converted.content[0].function_result
    assert converted.content[0].function_result.result == [
        {
            "payload": [
                None,
                False,
                0,
                1.5,
                "001",
                {"nested": ["value", {"ok": True}]},
            ]
        },
        "plain text",
    ]
    assert message.content == original


@pytest.mark.parametrize(
    ("invalid", "match"),
    [
        (object(), r"\$\[0\]\['payload'\]\[0\].*object"),
        ({1: "value"}, r"\$\[0\]\['payload'\].*string keys"),
        (float("nan"), r"\$\[0\]\['payload'\]\[0\].*finite JSON number"),
    ],
)
def test_convert_messages_tool_result_rejects_non_json_values_with_path(
    invalid: Any,
    match: str,
) -> None:
    message = ToolMessage(
        content=[{"payload": [invalid]}]
        if not isinstance(invalid, dict)
        else [{"payload": invalid}],
        tool_call_id="tools-state",
        name="weather",
    )

    with pytest.raises(ValueError, match=match):
        primary.convert_messages([message], cached_uploads={})


def test_convert_messages_rejects_tool_result_without_name() -> None:
    message = ToolMessage(content="result", tool_call_id="tools-state")

    with pytest.raises(ValueError, match="requires a function name"):
        primary.convert_messages([message], cached_uploads={})


def test_convert_messages_function_compatibility_requires_state() -> None:
    message = FunctionMessage(content="result", name="weather")

    with pytest.raises(ValueError, match="prefer ToolMessage"):
        primary.convert_messages([message], cached_uploads={})


def test_convert_messages_function_compatibility_uses_state() -> None:
    message = FunctionMessage(
        content="true",
        name="weather",
        additional_kwargs={"tools_state_id": "tools-state"},
    )

    converted = primary.convert_messages([message], cached_uploads={})[0]

    assert converted.tools_state_id == "tools-state"
    assert converted.content
    assert converted.content[0].function_result
    assert converted.content[0].function_result.result is True


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ([{"type": "video", "file_id": "file-1"}], "Unsupported"),
        (
            [{"type": "image_url", "image_url": {"url": "https://example.test"}}],
            "cached_uploads",
        ),
        ([{"type": "text", "text": 1}], "string 'text'"),
    ],
)
def test_convert_messages_rejects_unsupported_content(
    content: list[str | dict[Any, Any]],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        primary.convert_messages([HumanMessage(content=content)], cached_uploads={})
