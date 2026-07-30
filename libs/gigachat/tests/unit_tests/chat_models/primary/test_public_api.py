"""Public GigaChat integration tests for primary routing and legacy parity."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import gigachat.models as gm
import pytest

from langchain_gigachat.chat_models.gigachat import GigaChat

from .fixtures import CREATED_AT, MESSAGE_ID, MODEL


def _primary_json_response() -> gm.ChatCompletionResponse:
    return gm.ChatCompletionResponse(
        model=MODEL,
        created_at=CREATED_AT,
        messages=[
            gm.ChatMessage(
                role="assistant",
                message_id=MESSAGE_ID,
                content=[gm.ChatContentPart(text='{"value": 7}')],
            )
        ],
        message_id=MESSAGE_ID,
        finish_reason="stop",
    )


def _legacy_json_response() -> gm.ChatCompletion:
    return gm.ChatCompletion(
        choices=[
            gm.Choices(
                message=gm.Messages(
                    role=gm.MessagesRole.ASSISTANT,
                    content='{"value": 7}',
                ),
                index=0,
                finish_reason="stop",
            )
        ],
        created=CREATED_AT,
        model=MODEL,
        usage=gm.Usage(
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        ),
        object="chat.completion",
    )


def _configure_json_responses(sdk_client: MagicMock) -> None:
    sdk_client.chat.return_value = _legacy_json_response()
    sdk_client.chat.create.return_value = _primary_json_response()


@pytest.mark.parametrize(
    ("tool", "tool_name"),
    [
        ({"type": "web_search"}, "web_search"),
        ({"code_interpreter": {}}, "code_interpreter"),
    ],
)
def test_primary_bind_tools_routes_provider_builtins(
    sdk_client: MagicMock,
    tool: dict[str, Any],
    tool_name: str,
) -> None:
    _configure_json_responses(sdk_client)

    result = (
        GigaChat(model=MODEL, use_api_v2=True)
        .bind_tools([tool], tool_choice=tool_name)
        .invoke("Hello")
    )

    assert result.content == '{"value": 7}'
    payload = sdk_client.chat.create.call_args.args[0]
    assert isinstance(payload, gm.ChatCompletionRequest)
    assert payload.tools is not None
    assert payload.tools[0].model_dump(exclude_none=True) == {tool_name: {}}
    assert payload.tool_config == gm.ChatToolConfig(
        mode="forced",
        tool_name=tool_name,
    )
    sdk_client.chat.assert_not_called()


def test_legacy_bind_tools_client_function_is_unchanged(
    sdk_client: MagicMock,
) -> None:
    _configure_json_responses(sdk_client)
    tool = {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a value.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        },
    }

    GigaChat(model=MODEL).bind_tools(
        [tool],
        tool_choice="lookup",
    ).invoke("Hello")

    payload = sdk_client.chat.call_args.args[0]
    assert isinstance(payload, gm.Chat)
    assert payload.functions is not None
    assert payload.functions[0].name == "lookup"
    assert payload.function_call is not None
    assert payload.function_call.name == "lookup"
    sdk_client.chat.create.assert_not_called()


def test_legacy_rejects_builtin_only_after_invocation_route_override(
    sdk_client: MagicMock,
) -> None:
    _configure_json_responses(sdk_client)
    runnable = (
        GigaChat(model=MODEL, use_api_v2=True)
        .bind_tools([{"type": "web_search"}], tool_choice="web_search")
        .bind(use_api_v2=False)
    )

    with pytest.raises(ValueError, match=r"built-in.*use_api_v2=True"):
        runnable.invoke("Hello")

    sdk_client.chat.assert_not_called()
    sdk_client.chat.create.assert_not_called()


def test_invocation_route_override_enables_primary_builtin(
    sdk_client: MagicMock,
) -> None:
    _configure_json_responses(sdk_client)

    (
        GigaChat(model=MODEL)
        .bind_tools([{"type": "web_search"}], tool_choice="web_search")
        .bind(use_api_v2=True)
        .invoke("Hello")
    )

    sdk_client.chat.assert_not_called()
    sdk_client.chat.create.assert_called_once()
