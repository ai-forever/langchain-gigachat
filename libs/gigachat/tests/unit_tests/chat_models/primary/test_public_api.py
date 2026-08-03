"""Public GigaChat integration tests for primary routing and legacy parity."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any
from unittest.mock import MagicMock

import gigachat.models as gm
import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from langchain_gigachat.chat_models.gigachat import GigaChat

from .fixtures import CREATED_AT, MESSAGE_ID, MODEL, build_function_call_response


def _primary_json_response(content: str = '{"value": 7}') -> gm.ChatCompletionResponse:
    return gm.ChatCompletionResponse(
        model=MODEL,
        created_at=CREATED_AT,
        messages=[
            gm.ChatMessage(
                role="assistant",
                message_id=MESSAGE_ID,
                content=[gm.ChatContentPart(text=content)],
            )
        ],
        message_id=MESSAGE_ID,
        finish_reason="stop",
    )


def _primary_server_tool_response() -> gm.ChatCompletionResponse:
    return gm.ChatCompletionResponse.model_validate(
        {
            "model": MODEL,
            "created_at": CREATED_AT,
            "messages": [
                {
                    "role": "assistant",
                    "tools_state_id": "state-1",
                    "tool_execution": {
                        "call_id": "execution-1",
                        "name": "web_search",
                        "status": "success",
                    },
                }
            ],
            "finish_reason": "stop",
        }
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


def _assert_schema_less_response_format(payload: Any) -> None:
    assert isinstance(payload, gm.ChatCompletionRequest)
    assert payload.model_options is not None
    response_format = payload.model_options.response_format
    assert isinstance(response_format, gm.ChatResponseFormat)
    assert response_format.model_dump(exclude_none=True, by_alias=True) == {
        "type": "json_schema"
    }


def _primary_json_stream() -> Iterator[gm.PrimaryChatCompletionChunk]:
    for text in ('{"value": ', "7}"):
        yield gm.PrimaryChatCompletionChunk(
            event="response.message.delta",
            model=MODEL,
            created_at=CREATED_AT,
            messages=[
                gm.ChatMessageChunk(
                    role="assistant",
                    message_id=MESSAGE_ID,
                    content=[gm.ChatContentPart(text=text)],
                )
            ],
            message_id=MESSAGE_ID,
        )
    yield gm.PrimaryChatCompletionChunk(
        event="response.message.done",
        model=MODEL,
        created_at=CREATED_AT,
        messages=None,
        message_id=MESSAGE_ID,
        finish_reason="stop",
    )


def _primary_tool_call_stream() -> Iterator[gm.PrimaryChatCompletionChunk]:
    yield gm.PrimaryChatCompletionChunk(
        event="response.message.delta",
        model=MODEL,
        created_at=CREATED_AT,
        tools_state_id="tool-state",
        messages=[
            gm.ChatMessageChunk(
                role="assistant",
                message_id=MESSAGE_ID,
                content=[],
                function_call=gm.PrimaryChatFunctionCall(
                    name="get_weather",
                    arguments={"location": "SF"},
                ),
            )
        ],
        message_id=MESSAGE_ID,
    )
    yield gm.PrimaryChatCompletionChunk(
        event="response.message.done",
        model=MODEL,
        created_at=CREATED_AT,
        messages=None,
        message_id=MESSAGE_ID,
        tools_state_id="tool-state",
        finish_reason="tool_calls",
    )


async def _async_items(
    items: Iterator[Any],
) -> AsyncIterator[Any]:
    for item in items:
        yield item


class OutputSchema(BaseModel):
    value: int


def get_weather(location: str) -> None:
    """Get weather at a location."""


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


def test_primary_invoke_preserves_execution_level_server_tool_identity(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.create.return_value = _primary_server_tool_response()

    result = GigaChat(model=MODEL, use_api_v2=True).invoke("Hello")

    server_result = next(
        block
        for block in result.content_blocks
        if block["type"] == "server_tool_result"
    )
    assert server_result["tool_call_id"] == "execution-1"
    assert result.additional_kwargs["tools_state_id"] == "state-1"
    sdk_client.chat.create.assert_called_once()
    sdk_client.chat.assert_not_called()


async def test_primary_ainvoke_preserves_execution_level_server_tool_identity(
    sdk_client: MagicMock,
) -> None:
    sdk_client.achat.create.return_value = _primary_server_tool_response()

    result = await GigaChat(model=MODEL, use_api_v2=True).ainvoke("Hello")

    server_result = next(
        block
        for block in result.content_blocks
        if block["type"] == "server_tool_result"
    )
    assert server_result["tool_call_id"] == "execution-1"
    assert result.additional_kwargs["tools_state_id"] == "state-1"
    sdk_client.achat.create.assert_awaited_once()
    sdk_client.chat.create.assert_not_called()


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
    assert isinstance(payload.function_call, gm.ChatFunctionCall)
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


def test_legacy_storage_remains_supported(sdk_client: MagicMock) -> None:
    _configure_json_responses(sdk_client)
    storage = gm.Storage(is_stateful=True, thread_id="legacy-thread")

    GigaChat(model=MODEL).invoke("Hello", storage=storage)

    payload = sdk_client.chat.call_args.args[0]
    assert isinstance(payload, gm.Chat)
    assert payload.storage == storage


def test_legacy_rejects_tool_config_instead_of_discarding_it(
    sdk_client: MagicMock,
) -> None:
    _configure_json_responses(sdk_client)

    with pytest.raises(
        ValueError,
        match=r"primary-only argument\(s\): tool_config.*use_api_v2=True",
    ):
        GigaChat(model=MODEL).invoke(
            "Hello",
            tool_config={"mode": "auto"},
        )

    sdk_client.chat.assert_not_called()


@pytest.mark.parametrize(
    "stateful_kwargs",
    [
        {"assistant_id": "assistant"},
        {"storage": {"thread_id": "thread"}},
    ],
)
def test_primary_stateful_public_request_omits_implicit_model(
    sdk_client: MagicMock,
    stateful_kwargs: dict[str, Any],
) -> None:
    _configure_json_responses(sdk_client)

    GigaChat(model=MODEL, use_api_v2=True).invoke("Hello", **stateful_kwargs)

    payload = sdk_client.chat.create.call_args.args[0]
    assert isinstance(payload, gm.ChatCompletionRequest)
    assert payload.model is None


@pytest.mark.parametrize("use_api_v2", [False, True])
def test_json_schema_structured_output_uses_route_specific_format(
    sdk_client: MagicMock,
    use_api_v2: bool,
) -> None:
    _configure_json_responses(sdk_client)
    schema = {
        "title": "Answer",
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
    }

    result = (
        GigaChat(
            model=MODEL,
            use_api_v2=use_api_v2,
        )
        .with_structured_output(
            schema,
            method="json_schema",
        )
        .invoke("Hello")
    )

    assert result == {"value": 7}
    if use_api_v2:
        payload = sdk_client.chat.create.call_args.args[0]
        assert isinstance(payload, gm.ChatCompletionRequest)
        assert payload.model_options is not None
        assert isinstance(
            payload.model_options.response_format,
            gm.ChatResponseFormat,
        )
        assert payload.model_options.response_format.type == "json_schema"
    else:
        payload = sdk_client.chat.call_args.args[0]
        assert isinstance(payload, gm.Chat)
        assert isinstance(payload.response_format, gm.JsonSchemaResponseFormat)
        assert payload.response_format.type == "json_schema"


def test_json_schema_structured_output_executes_pydantic_parser(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.create.return_value = _primary_json_response()

    result = (
        GigaChat(model=MODEL, use_api_v2=True)
        .with_structured_output(OutputSchema, method="json_schema")
        .invoke("Hello")
    )

    assert result == OutputSchema(value=7)


def test_primary_direct_schema_less_json_binding_uses_exact_wire_format(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.create.return_value = _primary_json_response()

    result = (
        GigaChat(model=MODEL, use_api_v2=True)
        .bind(response_format={"type": "json_schema"})
        .invoke("Return JSON")
    )

    assert result.text == '{"value": 7}'
    assert "parsed" not in result.additional_kwargs
    _assert_schema_less_response_format(sdk_client.chat.create.call_args.args[0])
    sdk_client.chat.assert_not_called()


async def test_primary_direct_schema_less_json_binding_supports_ainvoke(
    sdk_client: MagicMock,
) -> None:
    sdk_client.achat.create.return_value = _primary_json_response()

    result = await (
        GigaChat(model=MODEL, use_api_v2=True)
        .bind(response_format={"type": "json_schema"})
        .ainvoke("Return JSON")
    )

    assert result.text == '{"value": 7}'
    assert "parsed" not in result.additional_kwargs
    _assert_schema_less_response_format(sdk_client.achat.create.call_args.args[0])


def test_primary_native_model_options_response_format_is_only_sent_on_invoke(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.create.return_value = _primary_json_response()

    result = (
        GigaChat(model=MODEL, use_api_v2=True)
        .bind(model_options={"response_format": {"type": "json_schema"}})
        .invoke("Return JSON")
    )

    assert result.text == '{"value": 7}'
    assert "parsed" not in result.additional_kwargs
    _assert_schema_less_response_format(sdk_client.chat.create.call_args.args[0])


def test_primary_native_text_format_wins_over_top_level_json_in_invoke(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.create.return_value = _primary_json_response()

    result = (
        GigaChat(model=MODEL, use_api_v2=True)
        .bind(
            model_options={"response_format": {"type": "text"}},
            response_format={"type": "json_schema"},
        )
        .invoke("Return text")
    )

    assert "parsed" not in result.additional_kwargs
    payload = sdk_client.chat.create.call_args.args[0]
    assert payload.model_options is not None
    assert payload.model_options.response_format is not None
    assert payload.model_options.response_format.type == "text"


async def test_primary_native_model_options_response_format_is_only_sent_on_ainvoke(
    sdk_client: MagicMock,
) -> None:
    sdk_client.achat.create.return_value = _primary_json_response()

    result = await (
        GigaChat(model=MODEL, use_api_v2=True)
        .bind(model_options={"response_format": {"type": "json_schema"}})
        .ainvoke("Return JSON")
    )

    assert result.text == '{"value": 7}'
    assert "parsed" not in result.additional_kwargs
    _assert_schema_less_response_format(sdk_client.achat.create.call_args.args[0])


def test_primary_schema_less_json_mode_returns_object(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.create.return_value = _primary_json_response()

    result = (
        GigaChat(model=MODEL, use_api_v2=True)
        .with_structured_output(None, method="json_mode")
        .invoke("Return JSON")
    )

    assert result == {"value": 7}
    _assert_schema_less_response_format(sdk_client.chat.create.call_args.args[0])


def test_schema_less_json_mode_respects_primary_route_override(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.create.return_value = _primary_json_response()

    result = (
        GigaChat(model=MODEL)
        .with_structured_output(None, method="json_mode")
        .bind(use_api_v2=True)
        .invoke("Return JSON")
    )

    assert result == {"value": 7}
    _assert_schema_less_response_format(sdk_client.chat.create.call_args.args[0])
    sdk_client.chat.assert_not_called()


@pytest.mark.parametrize("initial_use_api_v2", [False, True])
def test_schema_less_json_mode_preserves_legacy_transport(
    sdk_client: MagicMock,
    initial_use_api_v2: bool,
) -> None:
    sdk_client.chat.return_value = _legacy_json_response()
    runnable = (
        GigaChat(model=MODEL, use_api_v2=initial_use_api_v2)
        .with_structured_output(None, method="json_mode")
        .bind(use_api_v2=False)
    )

    result = runnable.invoke("Return JSON")

    assert result == {"value": 7}
    payload = sdk_client.chat.call_args.args[0]
    assert isinstance(payload, gm.Chat)
    assert payload.response_format is None
    sdk_client.chat.assert_called_once()
    sdk_client.chat.create.assert_not_called()


async def test_primary_schema_less_json_mode_supports_ainvoke(
    sdk_client: MagicMock,
) -> None:
    sdk_client.achat.create.return_value = _primary_json_response()

    result = await (
        GigaChat(model=MODEL, use_api_v2=True)
        .with_structured_output(None, method="json_mode")
        .ainvoke("Return JSON")
    )

    assert result == {"value": 7}
    _assert_schema_less_response_format(sdk_client.achat.create.call_args.args[0])


def test_primary_schema_less_json_mode_supports_streamed_invoke(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.stream.side_effect = lambda payload: _primary_json_stream()

    result = (
        GigaChat(model=MODEL, use_api_v2=True, streaming=True)
        .with_structured_output(None, method="json_mode")
        .invoke("Return JSON")
    )

    assert result == {"value": 7}
    _assert_schema_less_response_format(sdk_client.chat.stream.call_args.args[0])


def test_primary_direct_schema_less_json_binding_supports_stream(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.stream.side_effect = lambda payload: _primary_json_stream()

    chunks = list(
        GigaChat(model=MODEL, use_api_v2=True)
        .bind(response_format={"type": "json_schema"})
        .stream("Return JSON")
    )

    assert chunks
    assert "".join(chunk.text for chunk in chunks) == '{"value": 7}'
    assert all("parsed" not in chunk.additional_kwargs for chunk in chunks)
    _assert_schema_less_response_format(sdk_client.chat.stream.call_args.args[0])


def test_primary_native_text_format_wins_over_top_level_json_in_stream(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.stream.side_effect = lambda payload: _primary_json_stream()

    chunks = list(
        GigaChat(model=MODEL, use_api_v2=True)
        .bind(
            model_options={"response_format": {"type": "text"}},
            response_format={"type": "json_schema"},
        )
        .stream("Return text")
    )

    assert chunks
    assert "parsed" not in chunks[-1].additional_kwargs
    payload = sdk_client.chat.stream.call_args.args[0]
    assert payload.model_options is not None
    assert payload.model_options.response_format is not None
    assert payload.model_options.response_format.type == "text"


async def test_primary_schema_less_json_mode_supports_streamed_ainvoke(
    sdk_client: MagicMock,
) -> None:
    sdk_client.achat.stream.side_effect = lambda payload: _async_items(
        _primary_json_stream()
    )

    result = await (
        GigaChat(model=MODEL, use_api_v2=True, streaming=True)
        .with_structured_output(None, method="json_mode")
        .ainvoke("Return JSON")
    )

    assert result == {"value": 7}
    _assert_schema_less_response_format(sdk_client.achat.stream.call_args.args[0])


async def test_primary_direct_schema_less_json_binding_supports_astream(
    sdk_client: MagicMock,
) -> None:
    sdk_client.achat.stream.side_effect = lambda payload: _async_items(
        _primary_json_stream()
    )

    chunks = [
        chunk
        async for chunk in GigaChat(model=MODEL, use_api_v2=True)
        .bind(response_format={"type": "json_schema"})
        .astream("Return JSON")
    ]

    assert chunks
    assert "".join(chunk.text for chunk in chunks) == '{"value": 7}'
    assert all("parsed" not in chunk.additional_kwargs for chunk in chunks)
    _assert_schema_less_response_format(sdk_client.achat.stream.call_args.args[0])


async def test_primary_native_json_format_wins_over_top_level_text_in_astream(
    sdk_client: MagicMock,
) -> None:
    sdk_client.achat.stream.side_effect = lambda payload: _async_items(
        _primary_json_stream()
    )

    chunks = [
        chunk
        async for chunk in GigaChat(model=MODEL, use_api_v2=True)
        .bind(
            model_options={"response_format": {"type": "json_schema"}},
            response_format={"type": "text"},
        )
        .astream("Return JSON")
    ]

    assert chunks
    assert "".join(chunk.text for chunk in chunks) == '{"value": 7}'
    assert all("parsed" not in chunk.additional_kwargs for chunk in chunks)
    _assert_schema_less_response_format(sdk_client.achat.stream.call_args.args[0])


@pytest.mark.parametrize("content", ["not JSON", '[{"value": 7}]'])
def test_primary_schema_less_json_mode_include_raw_preserves_invalid_object(
    sdk_client: MagicMock,
    content: str,
) -> None:
    sdk_client.chat.create.return_value = _primary_json_response(content)

    result = (
        GigaChat(model=MODEL, use_api_v2=True)
        .with_structured_output(
            None,
            method="json_mode",
            include_raw=True,
        )
        .invoke("Return JSON")
    )

    assert isinstance(result, dict)
    assert isinstance(result["raw"], AIMessage)
    assert result["raw"].text == content
    assert result["parsed"] is None
    assert isinstance(result["parsing_error"], OutputParserException)


@pytest.mark.parametrize("content", ["not JSON", '[{"value": 7}]'])
def test_primary_schema_less_json_mode_rejects_invalid_object_without_raw(
    sdk_client: MagicMock,
    content: str,
) -> None:
    sdk_client.chat.create.return_value = _primary_json_response(content)
    runnable = GigaChat(
        model=MODEL,
        use_api_v2=True,
    ).with_structured_output(None, method="json_mode")

    with pytest.raises(OutputParserException):
        runnable.invoke("Return JSON")


def test_primary_schema_less_json_binding_leaves_tool_call_unparsed(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.create.return_value = build_function_call_response()

    result = (
        GigaChat(model=MODEL, use_api_v2=True)
        .bind(response_format={"type": "json_schema"})
        .invoke("Call the tool")
    )

    assert result.tool_calls
    assert "parsed" not in result.additional_kwargs


@pytest.mark.parametrize(
    "content",
    [
        "not JSON",
        '{"value": "not an integer"}',
    ],
)
def test_json_schema_include_raw_preserves_invalid_response(
    sdk_client: MagicMock,
    content: str,
) -> None:
    sdk_client.chat.create.return_value = _primary_json_response(content)

    result = (
        GigaChat(model=MODEL, use_api_v2=True)
        .with_structured_output(
            OutputSchema,
            method="json_schema",
            include_raw=True,
        )
        .invoke("Hello")
    )

    assert isinstance(result, dict)
    assert isinstance(result["raw"], AIMessage)
    assert result["raw"].text == content
    assert result["parsed"] is None
    assert isinstance(result["parsing_error"], OutputParserException)


def test_json_schema_invalid_response_without_raw_still_raises_parser_error(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.create.return_value = _primary_json_response("not JSON")

    chain = GigaChat(model=MODEL, use_api_v2=True).with_structured_output(
        OutputSchema,
        method="json_schema",
    )

    with pytest.raises(OutputParserException):
        chain.invoke("Hello")


def test_bind_response_format_preserves_invalid_raw_message(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.create.return_value = _primary_json_response("not JSON")

    result = (
        GigaChat(model=MODEL, use_api_v2=True)
        .bind(response_format=OutputSchema)
        .invoke("Hello")
    )

    assert isinstance(result, AIMessage)
    assert result.text == "not JSON"
    assert "parsed" not in result.additional_kwargs


@pytest.mark.parametrize("use_api_v2", [False, True])
def test_bind_tools_accepts_pydantic_response_format(
    sdk_client: MagicMock,
    use_api_v2: bool,
) -> None:
    _configure_json_responses(sdk_client)

    result = (
        GigaChat(model=MODEL, use_api_v2=use_api_v2)
        .bind_tools(
            [get_weather],
            response_format=OutputSchema,
            strict=True,
        )
        .invoke("What weighs more?")
    )

    assert result.text == '{"value": 7}'
    assert "parsed" not in result.additional_kwargs
    if use_api_v2:
        payload = sdk_client.chat.create.call_args.args[0]
        assert payload.model_options is not None
        response_format = payload.model_options.response_format
        assert isinstance(response_format, gm.ChatResponseFormat)
    else:
        payload = sdk_client.chat.call_args.args[0]
        response_format = payload.response_format
        assert isinstance(response_format, gm.JsonSchemaResponseFormat)
    assert response_format.schema_ == OutputSchema.model_json_schema()
    assert response_format.strict is True


@pytest.mark.parametrize("use_api_v2", [False, True])
def test_bind_tools_rejects_strict_without_response_format(
    use_api_v2: bool,
) -> None:
    llm = GigaChat(model=MODEL, use_api_v2=use_api_v2)

    with pytest.raises(
        ValueError,
        match="strict is supported only together with response_format",
    ):
        llm.bind_tools([get_weather], strict=True)


def test_bind_tools_response_format_leaves_tool_call_unparsed(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.create.return_value = build_function_call_response()

    result = (
        GigaChat(model=MODEL, use_api_v2=True)
        .bind_tools(
            [get_weather],
            response_format=OutputSchema,
            strict=True,
        )
        .invoke("What is the weather in SF?")
    )

    assert result.tool_calls
    assert "parsed" not in result.additional_kwargs


def test_bind_tools_response_format_keeps_raw_message_when_invoke_streams(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.stream.side_effect = lambda payload: _primary_json_stream()

    result = (
        GigaChat(model=MODEL, use_api_v2=True, streaming=True)
        .bind_tools(
            [get_weather],
            response_format=OutputSchema,
            strict=True,
        )
        .invoke("What weighs more?")
    )

    assert result.text == '{"value": 7}'
    assert "parsed" not in result.additional_kwargs


def test_bind_tools_response_format_leaves_streamed_tool_call_unparsed(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.stream.side_effect = lambda payload: _primary_tool_call_stream()

    result = (
        GigaChat(model=MODEL, use_api_v2=True, streaming=True)
        .bind_tools(
            [get_weather],
            response_format=OutputSchema,
            strict=True,
        )
        .invoke("What is the weather in SF?")
    )

    assert result.tool_calls
    assert "parsed" not in result.additional_kwargs


async def test_bind_tools_response_format_keeps_raw_message_when_ainvoke_streams(
    sdk_client: MagicMock,
) -> None:
    sdk_client.achat.stream.side_effect = lambda payload: _async_items(
        _primary_json_stream()
    )

    result = await (
        GigaChat(model=MODEL, use_api_v2=True, streaming=True)
        .bind_tools(
            [get_weather],
            response_format=OutputSchema,
            strict=True,
        )
        .ainvoke("What weighs more?")
    )

    assert result.text == '{"value": 7}'
    assert "parsed" not in result.additional_kwargs
