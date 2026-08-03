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


def test_convert_messages_deduplicates_attachment_ids_in_first_seen_order() -> None:
    data_url = "data:image/png;base64,aW1hZ2U="
    cached_uploads = {hashlib.sha256(data_url.encode()).hexdigest(): "file-2"}
    message = HumanMessage(
        content=[
            {"type": "image", "file_id": "file-1", "mime_type": "image/png"},
            {"type": "image_url", "image_url": {"giga_id": "file-1"}},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
        additional_kwargs={"attachments": ["file-2", "file-3", "file-1", "file-3"]},
    )
    original = copy.deepcopy(message)

    converted = primary.convert_messages(
        [message],
        cached_uploads=cached_uploads,
    )[0]

    assert converted.model_dump(exclude_none=True, by_alias=True)["content"] == [
        {"files": [{"id": "file-1", "mime": "image/png"}]},
        {"files": [{"id": "file-2", "mime": "image/png"}]},
        {"files": [{"id": "file-3"}]},
    ]
    assert message == original


def test_convert_messages_keeps_distinct_files_with_matching_mime() -> None:
    converted = primary.convert_messages(
        [
            HumanMessage(
                content=[
                    {"type": "image", "file_id": "file-1", "mime": "image/png"},
                    {"type": "image", "file_id": "file-2", "mime": "image/png"},
                ]
            )
        ],
        cached_uploads={},
    )[0]

    assert converted.content
    assert [part.files[0].id_ for part in converted.content if part.files] == [
        "file-1",
        "file-2",
    ]


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


@pytest.mark.parametrize("source", ["tool_calls", "additional_kwargs"])
def test_convert_messages_rejects_function_call_without_provider_state(
    source: str,
) -> None:
    if source == "tool_calls":
        message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "lookup",
                    "args": {"key": "value"},
                    "id": None,
                    "type": "tool_call",
                }
            ],
        )
    else:
        message = AIMessage(
            content="",
            additional_kwargs={
                "function_call": {
                    "name": "lookup",
                    "arguments": {"key": "value"},
                }
            },
        )

    with pytest.raises(
        ValueError,
        match="function call is missing provider tools_state_id",
    ):
        primary.convert_messages([message], cached_uploads={})


def test_convert_messages_rejects_tool_call_id_distinct_from_tools_state_id() -> None:
    message = AIMessage(
        content="",
        additional_kwargs={"tools_state_id": "provider-state"},
        tool_calls=[
            {
                "name": "lookup",
                "args": {"key": "value"},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )

    with pytest.raises(ValueError, match="tool call ID must equal tools_state_id"):
        primary.convert_messages([message], cached_uploads={})


def test_convert_messages_replays_additional_function_call_with_provider_state() -> (
    None
):
    messages = [
        AIMessage(
            content="",
            additional_kwargs={
                "tools_state_id": "provider-state",
                "function_call": {
                    "name": "lookup",
                    "arguments": {"key": "value"},
                },
            },
        ),
        ToolMessage(content='{"result": 1}', tool_call_id="provider-state"),
    ]

    converted = primary.convert_messages(messages, cached_uploads={})

    assert converted[0].tools_state_id == "provider-state"
    assert converted[0].content
    function_call = next(
        part.function_call
        for part in converted[0].content
        if part.function_call is not None
    )
    assert function_call.model_dump(exclude_none=True, by_alias=True) == {
        "name": "lookup",
        "arguments": {"key": "value"},
    }
    assert converted[1].tools_state_id == "provider-state"
    assert converted[1].content
    assert converted[1].content[0].function_result is not None
    assert converted[1].content[0].function_result.name == "lookup"


def test_convert_messages_prefers_additional_metadata_over_response_metadata() -> None:
    message = AIMessage(
        content="answer",
        additional_kwargs={
            "message_id": "additional-message",
            "tools_state_id": "additional-state",
        },
        response_metadata={
            "message_id": "response-message",
            "tools_state_id": "response-state",
        },
    )

    converted = primary.convert_messages([message], cached_uploads={})[0]

    assert converted.message_id == "additional-message"
    assert converted.tools_state_id == "additional-state"


def test_convert_messages_uses_response_metadata_as_fallback() -> None:
    message = AIMessage(
        content="answer",
        response_metadata={
            "message_id": "response-message",
            "tools_state_id": "response-state",
        },
    )

    converted = primary.convert_messages([message], cached_uploads={})[0]

    assert converted.message_id == "response-message"
    assert converted.tools_state_id == "response-state"


@pytest.mark.parametrize("field_name", ["message_id", "tools_state_id"])
@pytest.mark.parametrize("invalid_value", ["", "   "])
def test_convert_messages_rejects_empty_provider_metadata(
    field_name: str,
    invalid_value: str,
) -> None:
    message = AIMessage(
        content="answer",
        additional_kwargs={field_name: invalid_value},
    )

    with pytest.raises(
        ValueError,
        match=f"Primary {field_name} metadata must be a non-empty string",
    ):
        primary.convert_messages([message], cached_uploads={})


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


@pytest.mark.parametrize("invalid_id", ["", "   "])
def test_convert_messages_rejects_empty_ai_tool_call_id(invalid_id: str) -> None:
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "weather",
                "args": {},
                "id": invalid_id,
                "type": "tool_call",
            }
        ],
    )

    with pytest.raises(
        ValueError,
        match="Primary AIMessage tool call ID must be a non-empty string",
    ):
        primary.convert_messages([message], cached_uploads={})


def test_convert_messages_rejects_whitespace_ai_tool_name() -> None:
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "   ",
                "args": {},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )

    with pytest.raises(ValueError, match="requires a function name"):
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


def test_tool_result_rejects_name_that_conflicts_with_history() -> None:
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "weather",
                    "args": {},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="result",
            tool_call_id="call-1",
            name="calendar",
        ),
    ]

    with pytest.raises(
        ValueError,
        match=r"name 'calendar' conflicts with function name 'weather'.*'call-1'",
    ):
        primary.convert_messages(messages, cached_uploads={})


def test_convert_messages_tool_result_accepts_name_matching_history() -> None:
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "weather",
                    "args": {},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="result",
            tool_call_id="call-1",
            name="weather",
        ),
    ]

    converted = primary.convert_messages(messages, cached_uploads={})[1]

    assert converted.content
    assert converted.content[0].function_result
    assert converted.content[0].function_result.name == "weather"


def test_convert_messages_tool_result_accepts_nested_json_without_mutation() -> None:
    content: list[str | dict[Any, Any]] = [
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


def test_convert_messages_preserves_domain_json_that_resembles_text_block() -> None:
    domain_value = {
        "type": "text",
        "text": "customer-authored value",
        "domain": {"kind": "audit-record"},
    }
    message = ToolMessage(
        content=[domain_value],
        tool_call_id="tools-state",
        name="weather",
    )
    original = copy.deepcopy(message.content)

    converted = primary.convert_messages([message], cached_uploads={})[0]

    assert converted.content
    assert converted.content[0].function_result
    assert converted.content[0].function_result.result == [domain_value]
    assert message.content == original


def test_convert_messages_collapses_recognized_text_block_metadata() -> None:
    message = ToolMessage(
        content=[
            {
                "type": "text",
                "text": "plain text",
                "id": "block-1",
                "annotations": [],
                "extras": {"source": "tool"},
            }
        ],
        tool_call_id="tools-state",
        name="weather",
    )

    converted = primary.convert_messages([message], cached_uploads={})[0]

    assert converted.content
    assert converted.content[0].function_result
    assert converted.content[0].function_result.result == ["plain text"]


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
    invalid_content: list[str | dict[Any, Any]] = [
        {
            "payload": [invalid] if not isinstance(invalid, dict) else invalid,
        }
    ]
    message = ToolMessage(
        content=invalid_content,
        tool_call_id="tools-state",
        name="weather",
    )

    with pytest.raises(ValueError, match=match):
        primary.convert_messages([message], cached_uploads={})


def test_convert_messages_rejects_tool_result_without_name() -> None:
    message = ToolMessage(content="result", tool_call_id="tools-state")

    with pytest.raises(ValueError, match="requires a function name"):
        primary.convert_messages([message], cached_uploads={})


@pytest.mark.parametrize("invalid_id", ["", "   "])
def test_convert_messages_rejects_empty_tool_call_id(invalid_id: str) -> None:
    message = ToolMessage(
        content="result",
        tool_call_id=invalid_id,
        name="weather",
    )

    with pytest.raises(ValueError, match="tool_call_id must be a non-empty string"):
        primary.convert_messages([message], cached_uploads={})


def test_convert_messages_rejects_whitespace_tool_name() -> None:
    message = ToolMessage(
        content="result",
        tool_call_id="call-1",
        name="   ",
    )

    with pytest.raises(ValueError, match="name must be a non-empty string"):
        primary.convert_messages([message], cached_uploads={})


@pytest.mark.parametrize(
    "message",
    [
        FunctionMessage(content="result", name="weather"),
        FunctionMessage(
            content="true",
            name="weather",
            additional_kwargs={"tools_state_id": "tools-state"},
        ),
        AIMessage(
            content="",
            additional_kwargs={"functions_state_id": "legacy-state"},
        ),
    ],
)
def test_convert_messages_rejects_legacy_tool_state(message: Any) -> None:
    with pytest.raises(ValueError, match="another API contract"):
        primary.convert_messages([message], cached_uploads={})


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ([{"type": "video", "file_id": "file-1"}], "Unsupported"),
        (
            [{"type": "image_url", "image_url": {"url": "https://example.test"}}],
            "cached_uploads",
        ),
        ([{"type": "image", "file_id": "   "}], "non-empty provider"),
        ([{"type": "text", "text": 1}], "string 'text'"),
    ],
)
def test_convert_messages_rejects_unsupported_content(
    content: list[str | dict[Any, Any]],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        primary.convert_messages([HumanMessage(content=content)], cached_uploads={})


def test_convert_messages_rejects_whitespace_attachment_id() -> None:
    message = HumanMessage(
        content="question",
        additional_kwargs={"attachments": ["   "]},
    )

    with pytest.raises(ValueError, match="non-empty file IDs"):
        primary.convert_messages([message], cached_uploads={})
