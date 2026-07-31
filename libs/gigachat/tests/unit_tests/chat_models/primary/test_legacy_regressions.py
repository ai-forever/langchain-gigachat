"""Focused regressions proving that the legacy contract remains unchanged."""

from __future__ import annotations

import copy
from collections.abc import AsyncIterator, Iterator
from typing import Any
from unittest.mock import MagicMock

import gigachat.models as gm
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from langchain_gigachat.chat_models.gigachat import GigaChat

from .fixtures import CREATED_AT, MODEL, REQUEST_ID


def _legacy_response(content: str = "Legacy response") -> gm.ChatCompletion:
    return gm.ChatCompletion(
        choices=[
            gm.Choices(
                message=gm.Messages(
                    role=gm.MessagesRole.ASSISTANT,
                    content=content,
                ),
                index=0,
                finish_reason="stop",
            )
        ],
        created=CREATED_AT,
        model=MODEL,
        usage=gm.Usage(
            prompt_tokens=4,
            completion_tokens=2,
            total_tokens=6,
        ),
        object="chat.completion",
        x_headers={"x-request-id": REQUEST_ID},
    )


def _legacy_stream() -> Iterator[gm.ChatCompletionChunk]:
    for content, finish_reason in (("Legacy ", None), ("response", "stop")):
        yield gm.ChatCompletionChunk(
            choices=[
                gm.ChoicesChunk(
                    delta=gm.MessagesChunk(
                        role=gm.MessagesRole.ASSISTANT,
                        content=content,
                    ),
                    index=0,
                    finish_reason=finish_reason,
                )
            ],
            created=CREATED_AT,
            model=MODEL,
            object="chat.completion.chunk",
        )


async def _async_stream() -> AsyncIterator[gm.ChatCompletionChunk]:
    for chunk in _legacy_stream():
        yield chunk


def test_legacy_remains_default_for_plain_invoke(sdk_client: MagicMock) -> None:
    sdk_client.chat.return_value = _legacy_response()

    result = GigaChat(model=MODEL).invoke("Hello")

    assert isinstance(result, AIMessage)
    assert result.content == "Legacy response"
    sdk_client.chat.assert_called_once()
    sdk_client.chat.create.assert_not_called()


def test_legacy_sync_stream_remains_default(sdk_client: MagicMock) -> None:
    sdk_client.stream.side_effect = lambda payload: _legacy_stream()

    chunks = list(GigaChat(model=MODEL).stream("Hello"))

    assert all(isinstance(chunk, AIMessageChunk) for chunk in chunks)
    assert "".join(str(chunk.content) for chunk in chunks) == "Legacy response"
    sdk_client.stream.assert_called_once()
    sdk_client.chat.stream.assert_not_called()


async def test_legacy_async_stream_remains_default(sdk_client: MagicMock) -> None:
    sdk_client.astream.side_effect = lambda payload: _async_stream()

    chunks = [chunk async for chunk in GigaChat(model=MODEL).astream("Hello")]

    assert all(isinstance(chunk, AIMessageChunk) for chunk in chunks)
    assert "".join(str(chunk.content) for chunk in chunks) == "Legacy response"
    sdk_client.astream.assert_called_once()
    sdk_client.achat.stream.assert_not_called()


def test_legacy_client_tool_binding_is_preserved(sdk_client: MagicMock) -> None:
    sdk_client.chat.return_value = _legacy_response()
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

    GigaChat(model=MODEL).bind_tools([tool], tool_choice="lookup").invoke("Hello")

    payload = sdk_client.chat.call_args.args[0]
    assert isinstance(payload, gm.Chat)
    assert payload.functions is not None
    assert payload.functions[0].name == "lookup"
    assert isinstance(payload.function_call, gm.ChatFunctionCall)
    assert payload.function_call.name == "lookup"


def _legacy_function(name: Any) -> dict[str, Any]:
    return {
        "name": name,
        "description": f"Look up {name}.",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    }


def test_legacy_raw_function_and_tool_lists_are_pure_across_invocations(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.return_value = _legacy_response()
    functions = [_legacy_function("first")]
    tools = [{"type": "function", "function": _legacy_function("second")}]
    original_functions = copy.deepcopy(functions)
    original_tools = copy.deepcopy(tools)
    bound = GigaChat(model=MODEL).bind(functions=functions, tools=tools)

    bound.invoke("Hello")
    bound.invoke("Again")

    assert functions == original_functions
    assert tools == original_tools
    for call in sdk_client.chat.call_args_list:
        payload = call.args[0]
        assert [function.name for function in payload.functions] == [
            "first",
            "second",
        ]


def test_legacy_public_function_binding_stays_pure_when_tools_are_added(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.return_value = _legacy_response()
    functions = [_legacy_function("first")]
    tools = [{"type": "function", "function": _legacy_function("second")}]
    bound = GigaChat(model=MODEL).bind_functions(functions).bind(tools=tools)

    bound.invoke("Hello")
    bound.invoke("Again")

    for call in sdk_client.chat.call_args_list:
        payload = call.args[0]
        assert [function.name for function in payload.functions] == [
            "first",
            "second",
        ]


def test_legacy_validation_failure_does_not_mutate_function_inputs(
    sdk_client: MagicMock,
) -> None:
    functions = [_legacy_function(123)]
    tools = [{"type": "function", "function": _legacy_function("second")}]
    original_functions = copy.deepcopy(functions)
    original_tools = copy.deepcopy(tools)

    with pytest.raises(ValueError):
        GigaChat(model=MODEL).bind(functions=functions, tools=tools).invoke("Hello")

    assert functions == original_functions
    assert tools == original_tools
    sdk_client.chat.assert_not_called()


def test_legacy_storage_is_forwarded(sdk_client: MagicMock) -> None:
    sdk_client.chat.return_value = _legacy_response()
    storage = gm.Storage(is_stateful=True, thread_id="legacy-thread")

    GigaChat(model=MODEL).invoke("Hello", storage=storage)

    payload = sdk_client.chat.call_args.args[0]
    assert isinstance(payload, gm.Chat)
    assert payload.storage == storage


def test_legacy_json_schema_structured_output(sdk_client: MagicMock) -> None:
    sdk_client.chat.return_value = _legacy_response('{"value": 7}')
    schema = {
        "title": "Answer",
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
    }

    result = (
        GigaChat(model=MODEL)
        .with_structured_output(schema, method="json_schema")
        .invoke("Hello")
    )

    assert result == {"value": 7}
    payload = sdk_client.chat.call_args.args[0]
    assert isinstance(payload, gm.Chat)
    assert isinstance(payload.response_format, gm.JsonSchemaResponseFormat)
    assert payload.response_format.type == "json_schema"


def test_legacy_file_attachment_path_is_preserved(sdk_client: MagicMock) -> None:
    sdk_client.chat.return_value = _legacy_response()
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Describe the attachment"},
            {
                "type": "file",
                "file_id": "legacy-file-1",
                "mime_type": "application/pdf",
            },
        ]
    )

    GigaChat(model=MODEL).invoke([message])

    payload = sdk_client.chat.call_args.args[0]
    assert isinstance(payload, gm.Chat)
    assert payload.messages[0].content == "Describe the attachment"
    assert payload.messages[0].attachments == ["legacy-file-1"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("assistant_id", "assistant-1"),
        ("disable_filter", True),
        ("filter_config", {}),
        ("model_options", {}),
        ("ranker_options", {}),
        ("reasoning", {"effort": "high"}),
        ("tool_config", {}),
        ("tools_state_id", "tools-state-1"),
        ("user_info", {}),
    ],
)
def test_legacy_rejects_primary_only_kwargs(
    sdk_client: MagicMock,
    field: str,
    value: Any,
) -> None:
    sdk_client.chat.return_value = _legacy_response()

    with pytest.raises(
        ValueError,
        match=rf"primary-only argument\(s\): {field}.*use_api_v2=True",
    ):
        GigaChat(model=MODEL).invoke("Hello", **{field: value})

    sdk_client.chat.assert_not_called()
    sdk_client.chat.create.assert_not_called()


def test_use_api_v2_override_does_not_leak_into_legacy_payload(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.return_value = _legacy_response()

    GigaChat(model=MODEL, use_api_v2=True).bind(use_api_v2=False).invoke("Hello")

    payload = sdk_client.chat.call_args.args[0]
    assert isinstance(payload, gm.Chat)
    assert "use_api_v2" not in payload.model_dump()
    sdk_client.chat.create.assert_not_called()
