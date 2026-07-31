"""Public response-format cross-field validation contracts."""

from __future__ import annotations

import copy
from typing import Any
from unittest.mock import MagicMock

import pytest

from langchain_gigachat.chat_models.gigachat import GigaChat

from .fixtures import MODEL, build_plain_text_response


@pytest.mark.parametrize(
    ("response_format", "error"),
    [
        (
            {"type": "text", "schema": {"type": "object"}},
            "type 'text' cannot include field",
        ),
        ({"type": "text", "regex": ".+"}, "type 'text' cannot include field"),
        ({"type": "text", "strict": False}, "type 'text' cannot include field"),
        ({"type": "regex"}, "requires a non-empty string 'regex' field"),
        (
            {"type": "regex", "regex": ".+", "schema": {"type": "string"}},
            "type 'regex' cannot include field",
        ),
        (
            {"type": "regex", "regex": ".+", "strict": True},
            "type 'regex' cannot include field",
        ),
        (
            {"type": "json_schema", "strict": False},
            "schema-less response_format.*cannot include field: strict",
        ),
        (
            {"type": "json_schema", "strict": True},
            "schema-less response_format.*cannot include field: strict",
        ),
        (
            {
                "type": "json_schema",
                "schema": {"type": "object"},
                "regex": ".+",
            },
            "type 'json_schema' cannot include field",
        ),
        (
            {
                "type": "json_schema",
                "schema": {"type": "object"},
                "strict": "yes",
            },
            "field 'strict' must be a boolean",
        ),
    ],
)
def test_invalid_cross_fields_fail_before_public_network_call(
    sdk_client: MagicMock,
    response_format: dict[str, Any],
    error: str,
) -> None:
    original = copy.deepcopy(response_format)

    with pytest.raises(ValueError, match=error):
        GigaChat(model=MODEL, use_api_v2=True).invoke(
            "Hello",
            response_format=response_format,
        )

    assert response_format == original
    sdk_client.chat.create.assert_not_called()


@pytest.mark.parametrize(
    "response_format",
    [
        {"type": "text"},
        {"type": "regex", "regex": r"^[A-Z].+$"},
        {"type": "json_schema"},
        {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
            },
            "strict": True,
        },
    ],
)
def test_valid_cross_field_forms_reach_the_public_sdk_boundary(
    sdk_client: MagicMock,
    response_format: dict[str, Any],
) -> None:
    sdk_client.chat.create.return_value = build_plain_text_response()
    original = copy.deepcopy(response_format)

    GigaChat(model=MODEL, use_api_v2=True).invoke(
        "Hello",
        response_format=response_format,
    )

    assert response_format == original
    payload = sdk_client.chat.create.call_args.args[0]
    assert payload.model_options
    assert payload.model_options.response_format
    assert payload.model_options.response_format.type == response_format["type"]
    if response_format == {"type": "json_schema"}:
        assert payload.model_options.response_format.model_dump(
            exclude_none=True,
            by_alias=True,
        ) == {"type": "json_schema"}
