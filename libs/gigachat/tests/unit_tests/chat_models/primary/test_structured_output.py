"""Focused primary structured-output contracts."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import gigachat.models as gm
import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from langchain_gigachat.chat_models._contracts import primary
from langchain_gigachat.chat_models.gigachat import GigaChat

from .fixtures import CREATED_AT, MESSAGE_ID, MODEL, build_function_call_response


class OutputSchema(BaseModel):
    value: int


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
        ({"type": "json_schema"}, {"type": "json_schema"}),
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
    strict = True if "schema" in expected else None

    result = primary.normalize_response_format(value, strict=strict)

    assert isinstance(result, gm.ChatResponseFormat)
    assert result.model_dump(exclude_none=True, by_alias=True) == expected


@pytest.mark.parametrize("strict", [False, True])
def test_schema_less_json_rejects_strict(strict: bool) -> None:
    with pytest.raises(ValueError, match="schema-less response_format"):
        primary.normalize_response_format(
            {"type": "json_schema"},
            strict=strict,
        )


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


def test_schema_less_json_mode_parses_an_object(sdk_client: MagicMock) -> None:
    sdk_client.chat.create.return_value = _json_response()

    result = (
        GigaChat(model=MODEL, use_api_v2=True)
        .with_structured_output(None, method="json_mode")
        .invoke("Return JSON")
    )

    assert result == {"value": 7}
    assert _response_format(sdk_client.chat.create.call_args.args[0]) == {
        "type": "json_schema"
    }


def test_low_level_response_format_returns_the_raw_message(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.create.return_value = _json_response()

    result = (
        GigaChat(model=MODEL, use_api_v2=True)
        .bind(response_format={"type": "json_schema"})
        .invoke("Return JSON")
    )

    assert isinstance(result, AIMessage)
    assert result.text == '{"value": 7}'
    assert "parsed" not in result.additional_kwargs
    assert _response_format(sdk_client.chat.create.call_args.args[0]) == {
        "type": "json_schema"
    }


def test_include_raw_reports_schema_less_parse_errors(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.create.return_value = _json_response("not JSON")

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
        .bind(response_format={"type": "json_schema"})
        .stream("Return JSON")
    )

    assert "".join(chunk.text for chunk in chunks) == '{"value": 7}'
    assert all("parsed" not in chunk.additional_kwargs for chunk in chunks)
    assert _response_format(sdk_client.chat.stream.call_args.args[0]) == {
        "type": "json_schema"
    }
