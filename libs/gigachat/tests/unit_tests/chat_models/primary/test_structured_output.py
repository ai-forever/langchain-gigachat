"""Focused primary structured-output contracts."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import gigachat.models as gm
import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableBinding, RunnableSequence
from pydantic import BaseModel

from langchain_gigachat.chat_models._contracts import primary
from langchain_gigachat.chat_models.gigachat import GigaChat

from .fixtures import CREATED_AT, MESSAGE_ID, MODEL, build_function_call_response


class OutputSchema(BaseModel):
    value: int


class UnionOutputSchema(BaseModel):
    """Return the capacity as a number or a label."""

    capacity: int | str


def test_function_structured_output_preserves_union_schema() -> None:
    model = GigaChat(use_api_v2=True)
    structured = model.with_structured_output(UnionOutputSchema)
    assert isinstance(structured, RunnableSequence)
    assert isinstance(structured.first, RunnableBinding)
    payload = model._build_primary_payload([], structured.first.kwargs)
    function = payload.model_dump(exclude_none=True)["tools"][0]["functions"][
        "specifications"
    ][0]
    assert function["parameters"]["properties"]["capacity"]["anyOf"] == [
        {"type": "integer"},
        {"type": "string"},
    ]


def _json_response(content: str = '{"value": 7}') -> gm.ChatCompletionResponse:
    return gm.ChatCompletionResponse.model_validate(
        {
            "model": MODEL,
            "created_at": CREATED_AT,
            "messages": [
                {
                    "role": "assistant",
                    "message_id": MESSAGE_ID,
                    "content": [{"text": content}],
                }
            ],
            "message_id": MESSAGE_ID,
            "finish_reason": "stop",
        }
    )


def _json_stream() -> Iterator[gm.PrimaryChatCompletionChunk]:
    payloads = [
        {
            "event": "response.message.delta",
            "model": MODEL,
            "created_at": CREATED_AT,
            "messages": [
                {
                    "role": "assistant",
                    "message_id": MESSAGE_ID,
                    "content": [{"text": '{"value": '}],
                }
            ],
            "message_id": MESSAGE_ID,
        },
        {
            "event": "response.message.delta",
            "model": MODEL,
            "created_at": CREATED_AT,
            "messages": [
                {
                    "role": "assistant",
                    "message_id": MESSAGE_ID,
                    "content": [{"text": "7}"}],
                }
            ],
            "message_id": MESSAGE_ID,
        },
        {
            "event": "response.message.done",
            "model": MODEL,
            "created_at": CREATED_AT,
            "message_id": MESSAGE_ID,
            "finish_reason": "stop",
        },
    ]
    yield from (
        gm.PrimaryChatCompletionChunk.model_validate(payload) for payload in payloads
    )


def _response_format(payload: Any) -> dict[str, Any]:
    assert isinstance(payload, gm.ChatCompletionRequest)
    assert payload.model_options is not None
    assert payload.model_options.response_format is not None
    return payload.model_options.response_format.model_dump(
        exclude_none=True,
        by_alias=True,
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            OutputSchema,
            {
                "type": "json_schema",
                "schema": OutputSchema.model_json_schema(),
                "strict": True,
            },
        ),
        (
            OutputSchema.model_json_schema(),
            {
                "type": "json_schema",
                "schema": OutputSchema.model_json_schema(),
                "strict": True,
            },
        ),
        (
            {"type": "json_schema", "schema": {"type": "object"}},
            {"type": "json_schema", "schema": {"type": "object"}},
        ),
        ({"type": "text"}, {"type": "text"}),
        (
            {"type": "regex", "regex": r"[A-Z]{2}-[0-9]{4}"},
            {"type": "regex", "regex": r"[A-Z]{2}-[0-9]{4}"},
        ),
    ],
)
def test_normalize_response_format_uses_sdk_models(
    value: Any,
    expected: dict[str, Any],
) -> None:
    strict = expected.get("strict")

    result = primary.normalize_response_format(value, strict=strict)

    assert isinstance(result, gm.ChatResponseFormat)
    assert result.model_dump(exclude_none=True, by_alias=True) == expected


@pytest.mark.parametrize("strict", [None, False, True])
@pytest.mark.parametrize(
    "value",
    [
        {"type": "json_schema"},
        {"type": "json_schema", "schema": None},
        gm.ChatResponseFormat(type="json_schema"),
        {"type": "json_schema", "json_schema": {}},
    ],
)
def test_json_schema_requires_a_schema(value: Any, strict: bool | None) -> None:
    with pytest.raises(ValueError, match="requires a 'schema' field"):
        primary.normalize_response_format(value, strict=strict)


@pytest.mark.parametrize("location", ["flat", "model_options", "additional_fields"])
def test_low_level_json_schema_without_schema_fails_before_request(
    sdk_client: MagicMock, location: str
) -> None:
    kwargs: dict[str, Any] = {"response_format": {"type": "json_schema"}}
    if location != "flat":
        kwargs = {location: kwargs}

    model = GigaChat(model=MODEL, use_api_v2=True).bind(**kwargs)
    with pytest.raises(ValueError, match="requires a 'schema' field"):
        model.invoke("Return JSON")

    sdk_client.chat.create.assert_not_called()


def test_json_schema_parses_pydantic_output(sdk_client: MagicMock) -> None:
    sdk_client.chat.create.return_value = _json_response()

    result = (
        GigaChat(model=MODEL, use_api_v2=True)
        .with_structured_output(OutputSchema, method="json_schema")
        .invoke("Return JSON")
    )

    assert result == OutputSchema(value=7)
    assert _response_format(sdk_client.chat.create.call_args.args[0]) == {
        "type": "json_schema",
        "schema": OutputSchema.model_json_schema(),
    }


def test_json_mode_supplies_a_minimal_object_schema(sdk_client: MagicMock) -> None:
    sdk_client.chat.create.return_value = _json_response()

    result = (
        GigaChat(model=MODEL, use_api_v2=True)
        .with_structured_output(None, method="json_mode")
        .invoke("Return JSON")
    )

    assert result == {"value": 7}
    assert _response_format(sdk_client.chat.create.call_args.args[0]) == {
        "type": "json_schema",
        "schema": {"type": "object"},
    }


async def test_async_json_mode_supplies_a_minimal_object_schema(
    sdk_client: MagicMock,
) -> None:
    sdk_client.achat.create.return_value = _json_response()

    result = await (
        GigaChat(model=MODEL, use_api_v2=True)
        .with_structured_output(None, method="json_mode")
        .ainvoke("Return JSON")
    )

    assert result == {"value": 7}
    assert _response_format(sdk_client.achat.create.call_args.args[0]) == {
        "type": "json_schema",
        "schema": {"type": "object"},
    }


@pytest.mark.parametrize("via_additional_fields", [False, True])
@pytest.mark.parametrize(
    "model_options",
    [
        {"response_format": {"type": "text"}},
        {"response_format": None},
        gm.ChatModelOptions(response_format=gm.ChatResponseFormat(type="text")),
        gm.ChatModelOptions(response_format=None),
    ],
)
def test_json_mode_rejects_nested_response_format_override(
    sdk_client: MagicMock, model_options: Any, via_additional_fields: bool
) -> None:
    kwargs: dict[str, Any] = {"model_options": model_options}
    if via_additional_fields:
        kwargs = {"additional_fields": kwargs}
    model = GigaChat(model=MODEL, use_api_v2=True).with_structured_output(
        None, method="json_mode"
    )

    with pytest.raises(
        ValueError, match="cannot be combined.*explicit response_format"
    ):
        model.invoke("Return JSON", **kwargs)

    sdk_client.chat.create.assert_not_called()


@pytest.mark.parametrize(
    "model_options",
    [{"temperature": 0.5}, gm.ChatModelOptions(temperature=0.5)],
)
def test_json_mode_allows_model_options_without_response_format(
    sdk_client: MagicMock, model_options: Any
) -> None:
    sdk_client.chat.create.return_value = _json_response()
    model = GigaChat(model=MODEL, use_api_v2=True).with_structured_output(
        None, method="json_mode"
    )

    assert model.invoke("Return JSON", model_options=model_options) == {"value": 7}
    payload = sdk_client.chat.create.call_args.args[0]
    assert payload.model_options.temperature == 0.5
    assert _response_format(payload) == {
        "type": "json_schema",
        "schema": {"type": "object"},
    }


def test_json_mode_without_a_user_schema_requires_primary_route(
    sdk_client: MagicMock,
) -> None:
    model = GigaChat(model=MODEL).with_structured_output(None, method="json_mode")

    with pytest.raises(ValueError, match="requires use_api_v2=True"):
        model.invoke("Return JSON")

    sdk_client.chat.assert_not_called()


def test_legacy_json_mode_with_schema_keeps_parsing_only(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.create.return_value = _json_response()

    with pytest.warns(DeprecationWarning, match="json_mode.*deprecated"):
        model = GigaChat(model=MODEL, use_api_v2=True).with_structured_output(
            OutputSchema, method="json_mode"
        )
    assert model.invoke("Return JSON") == OutputSchema(value=7)

    payload = sdk_client.chat.create.call_args.args[0]
    assert (
        payload.model_options is None or payload.model_options.response_format is None
    )


def test_low_level_response_format_returns_the_raw_message(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.create.return_value = _json_response()

    result = (
        GigaChat(model=MODEL, use_api_v2=True)
        .bind(response_format={"type": "json_schema", "schema": {"type": "object"}})
        .invoke("Return JSON")
    )

    assert isinstance(result, AIMessage)
    assert result.text == '{"value": 7}'
    assert "parsed" not in result.additional_kwargs
    assert _response_format(sdk_client.chat.create.call_args.args[0]) == {
        "type": "json_schema",
        "schema": {"type": "object"},
    }


@pytest.mark.parametrize("content", ["not JSON", "[1, 2]", "7", "true", "null"])
def test_include_raw_reports_json_object_parse_errors(
    sdk_client: MagicMock, content: str
) -> None:
    sdk_client.chat.create.return_value = _json_response(content)

    result = (
        GigaChat(model=MODEL, use_api_v2=True)
        .with_structured_output(None, method="json_mode", include_raw=True)
        .invoke("Return JSON")
    )

    assert isinstance(result, dict)
    assert isinstance(result["raw"], AIMessage)
    assert result["parsed"] is None
    assert isinstance(result["parsing_error"], OutputParserException)


def test_response_format_does_not_parse_client_tool_calls(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.create.return_value = build_function_call_response()

    result = (
        GigaChat(model=MODEL, use_api_v2=True)
        .bind(response_format=OutputSchema)
        .invoke("Call the tool")
    )

    assert result.tool_calls
    assert "parsed" not in result.additional_kwargs


def test_direct_stream_keeps_raw_output_without_model_side_parsing(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.stream.side_effect = lambda payload: _json_stream()

    chunks = list(
        GigaChat(model=MODEL, use_api_v2=True)
        .bind(response_format={"type": "json_schema", "schema": {"type": "object"}})
        .stream("Return JSON")
    )

    assert "".join(chunk.text for chunk in chunks) == '{"value": 7}'
    assert all("parsed" not in chunk.additional_kwargs for chunk in chunks)
    assert _response_format(sdk_client.chat.stream.call_args.args[0]) == {
        "type": "json_schema",
        "schema": {"type": "object"},
    }
