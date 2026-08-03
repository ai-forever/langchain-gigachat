"""Public tool-result JSON and function-name consistency contracts."""

from __future__ import annotations

import copy
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from langchain_gigachat.chat_models.gigachat import GigaChat

from .fixtures import MODEL, build_plain_text_response


def _function_message() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "lookup_weather",
                "args": {"city": "Moscow"},
                "id": "provider-state",
                "type": "tool_call",
            }
        ],
    )


def _tool_result(payload: Any) -> Any:
    message = payload.messages[-1]
    assert message.content
    function_result = message.content[0].function_result
    assert function_result
    return function_result.result


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (
            [
                {
                    "type": "text",
                    "text": "domain value",
                    "confidence": 0.9,
                }
            ],
            [
                {
                    "type": "text",
                    "text": "domain value",
                    "confidence": 0.9,
                }
            ],
        ),
        (
            [
                {
                    "type": "text",
                    "text": "recognized block",
                    "id": "block-1",
                    "extras": {"source": "tool"},
                }
            ],
            ["recognized block"],
        ),
        (
            [
                {
                    "payload": {
                        "type": "text",
                        "text": "nested domain value",
                        "domain": True,
                    }
                }
            ],
            [
                {
                    "payload": {
                        "type": "text",
                        "text": "nested domain value",
                        "domain": True,
                    }
                }
            ],
        ),
    ],
)
def test_public_tool_result_preserves_domain_json_without_caller_mutation(
    sdk_client: MagicMock,
    content: list[str | dict[Any, Any]],
    expected: Any,
) -> None:
    sdk_client.chat.create.return_value = build_plain_text_response()
    function_message = _function_message()
    tool_message = ToolMessage(
        content=content,
        tool_call_id="provider-state",
        name="lookup_weather",
    )
    original_function_message = copy.deepcopy(function_message)
    original_tool_message = copy.deepcopy(tool_message)

    GigaChat(model=MODEL, use_api_v2=True).invoke([function_message, tool_message])

    payload = sdk_client.chat.create.call_args.args[0]
    assert payload.messages[0].tools_state_id == "provider-state"
    assert payload.messages[1].tools_state_id == "provider-state"
    assert _tool_result(payload) == expected
    assert function_message == original_function_message
    assert tool_message == original_tool_message


def test_public_tool_result_cannot_relabel_provider_state(
    sdk_client: MagicMock,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"name 'calendar' conflicts with function name 'lookup_weather'",
    ):
        GigaChat(model=MODEL, use_api_v2=True).invoke(
            [
                _function_message(),
                ToolMessage(
                    content='{"temperature": 18}',
                    tool_call_id="provider-state",
                    name="calendar",
                ),
            ]
        )

    sdk_client.chat.create.assert_not_called()
