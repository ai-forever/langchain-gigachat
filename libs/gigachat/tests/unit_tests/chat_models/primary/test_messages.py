"""LangChain message history to primary SDK message conversion."""

from __future__ import annotations

import copy
import hashlib
import json
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


def _dump(messages: list[gm.ChatMessage]) -> list[dict[str, Any]]:
    return [
        message.model_dump(exclude_none=True, by_alias=True) for message in messages
    ]


def test_preserves_text_roles_and_custom_role() -> None:
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


def test_maps_existing_and_uploaded_file_ids_without_mutation() -> None:
    data_url = "data:audio/mpeg;base64,YXVkaW8="
    cache_key = hashlib.sha256(data_url.encode()).hexdigest()
    message = HumanMessage(
        content=[
            {"type": "text", "text": "describe"},
            {"type": "file", "file_id": "document-1", "mime_type": "application/pdf"},
            {"type": "audio_url", "audio_url": {"url": data_url}},
        ]
    )
    original = copy.deepcopy(message)

    converted = primary.convert_messages(
        [message],
        cached_uploads={cache_key: "uploaded-audio"},
    )[0]

    assert converted.model_dump(exclude_none=True, by_alias=True) == {
        "content": [
            {"text": "describe"},
            {"files": [{"id": "document-1", "mime": "application/pdf"}]},
            {"files": [{"id": "uploaded-audio", "mime": "audio/mpeg"}]},
        ],
        "role": "user",
    }
    assert message == original


def test_replays_assistant_output_as_request_history() -> None:
    message = AIMessage(
        content=[
            {"type": "reasoning", "reasoning": "private work"},
            {
                "type": "server_tool_call",
                "id": "server-1",
                "name": "web_search",
                "args": {},
            },
            {
                "type": "server_tool_result",
                "tool_call_id": "server-1",
                "status": "success",
                "output": {},
            },
            {"type": "text", "text": "final answer"},
            {"type": "image", "file_id": "generated-image", "mime_type": "image/png"},
        ]
    )

    converted = primary.convert_messages([message], cached_uploads={})[0]

    assert converted.model_dump(exclude_none=True, by_alias=True) == {
        "content": [
            {"text": "final answer"},
            {"files": [{"id": "generated-image", "mime": "image/png"}]},
        ],
        "role": "assistant",
    }


def test_client_function_and_tool_result_roundtrip() -> None:
    request = AIMessage(
        content="",
        additional_kwargs={"message_id": "provider-message"},
        tool_calls=[
            {
                "name": "weather",
                "args": {"city": "Moscow"},
                "id": "tools-state-1",
                "type": "tool_call",
            }
        ],
    )
    result_payload = {"temperature": 20, "conditions": ["clear"]}
    result = ToolMessage(
        content=json.dumps(result_payload),
        tool_call_id="tools-state-1",
    )
    original_content = copy.deepcopy(result.content)

    converted = primary.convert_messages([request, result], cached_uploads={})

    assert converted[0].model_dump(exclude_none=True, by_alias=True) == {
        "message_id": "provider-message",
        "content": [
            {
                "function_call": {
                    "name": "weather",
                    "arguments": {"city": "Moscow"},
                }
            }
        ],
        "tools_state_id": "tools-state-1",
        "role": "assistant",
    }
    assert converted[1].model_dump(exclude_none=True, by_alias=True) == {
        "content": [
            {
                "function_result": {
                    "name": "weather",
                    "result": result_payload,
                }
            }
        ],
        "tools_state_id": "tools-state-1",
        "role": "tool",
    }
    assert result.content == original_content


def test_tool_result_can_supply_its_name_without_prior_history() -> None:
    converted = primary.convert_messages(
        [
            ToolMessage(
                content="not json",
                tool_call_id="tools-state-1",
                name="weather",
            )
        ],
        cached_uploads={},
    )[0]

    assert converted.content
    function_result = converted.content[0].function_result
    assert function_result is not None
    assert function_result.name == "weather"
    assert function_result.result == "not json"


def test_client_tool_call_id_is_the_provider_tools_state_id() -> None:
    message = AIMessage(
        content="",
        additional_kwargs={"tools_state_id": "different-state"},
        tool_calls=[
            {
                "name": "weather",
                "args": {},
                "id": "tools-state-1",
                "type": "tool_call",
            }
        ],
    )

    with pytest.raises(ValueError, match="must equal tools_state_id"):
        primary.convert_messages([message], cached_uploads={})


def test_parallel_client_tool_calls_are_rejected() -> None:
    message = AIMessage(
        content="",
        tool_calls=[
            {"name": "one", "args": {}, "id": "state-1", "type": "tool_call"},
            {"name": "two", "args": {}, "id": "state-2", "type": "tool_call"},
        ],
    )

    with pytest.raises(ValueError, match="multiple client function calls"):
        primary.convert_messages([message], cached_uploads={})


@pytest.mark.parametrize(
    "message",
    [
        FunctionMessage(content="result", name="weather"),
        AIMessage(
            content="",
            additional_kwargs={"functions_state_id": "legacy-state"},
        ),
    ],
)
def test_rejects_legacy_tool_state(message: Any) -> None:
    with pytest.raises(ValueError, match="another API contract"):
        primary.convert_messages([message], cached_uploads={})


def test_uncached_url_is_rejected_before_the_sdk_request() -> None:
    message = HumanMessage(
        content=[
            {"type": "image_url", "image_url": {"url": "https://example.test/image"}}
        ]
    )

    with pytest.raises(ValueError, match="cached_uploads"):
        primary.convert_messages([message], cached_uploads={})
