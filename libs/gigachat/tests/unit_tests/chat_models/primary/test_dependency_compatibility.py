"""Compatibility gates for the supported LangChain Core 1.x range."""

from __future__ import annotations

from typing import Any

import gigachat.models as gm
import pytest
from pydantic import ValidationError
from pydantic.v1 import BaseModel as BaseModelV1

from langchain_gigachat.chat_models._contracts.primary.payload import (
    normalize_response_format,
)
from langchain_gigachat.chat_models.gigachat import GigaChat
from langchain_gigachat.utils.function_calling import (
    _parse_google_docstring,
    convert_pydantic_to_gigachat_function,
)


class LegacyArguments(BaseModelV1):
    """Look up a legacy value."""

    key: str


def test_default_streaming_value_is_not_marked_as_an_explicit_opt_out() -> None:
    model = GigaChat()

    assert model.streaming is False
    assert "streaming" not in model.model_fields_set


def test_explicit_streaming_opt_out_remains_explicit() -> None:
    model = GigaChat(streaming=False)

    assert model.streaming is False
    assert "streaming" in model.model_fields_set


def test_pydantic_v1_schema_remains_accepted_at_compatibility_boundaries() -> None:
    function = convert_pydantic_to_gigachat_function(LegacyArguments)
    response_format = normalize_response_format(LegacyArguments)

    assert function["name"] == "LegacyArguments"
    assert function["description"] == "Look up a legacy value."
    assert response_format is not None
    assert response_format.type == "json_schema"
    assert isinstance(response_format.schema_, dict)
    assert response_format.schema_["title"] == "LegacyArguments"


def test_private_google_docstring_parser_contract_remains_available() -> None:
    description, arguments = _parse_google_docstring(
        "Look up a value.\n\nArgs:\n    key: Lookup key.",
        ["key"],
    )

    assert description == "Look up a value."
    assert arguments == {"key": "Lookup key."}


def test_unknown_sdk_event_preserves_extension_fields(
    unknown_event: dict[str, Any],
) -> None:
    parsed = gm.PrimaryChatCompletionChunk.model_validate(unknown_event)
    dumped = parsed.model_dump(exclude_none=True)

    assert parsed.event == "response.provider_extension.delta"
    assert dumped["provider_metadata"] == {"preserve": True}
    assert dumped["messages"][0]["content"][0]["provider_extension"] == {
        "kind": "future-content",
        "value": 42,
    }


def test_malformed_known_sdk_event_is_rejected(
    malformed_event: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        gm.PrimaryChatCompletionChunk.model_validate(malformed_event)
    assert malformed_event["provider_error"]["code"] == "malformed_fixture"
