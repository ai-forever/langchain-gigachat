"""Tests for primary structured-output format normalization."""

import copy
from typing import Any

import gigachat.models as gm
import pytest
from pydantic import BaseModel

from langchain_gigachat.chat_models._contracts import primary


class Answer(BaseModel):
    value: int


def test_normalizes_legacy_json_schema_response_format() -> None:
    legacy = gm.JsonSchemaResponseFormat(
        schema=Answer.model_json_schema(),
        strict=False,
    )

    normalized = primary.normalize_response_format(legacy)

    assert isinstance(normalized, gm.ChatResponseFormat)
    assert normalized is not legacy
    assert normalized.type == "json_schema"
    assert normalized.schema_ == Answer.model_json_schema()
    assert normalized.strict is False


def test_copies_primary_response_format() -> None:
    response_format = gm.ChatResponseFormat(
        type="json_schema",
        schema={"type": "object"},
        strict=True,
    )

    normalized = primary.normalize_response_format(response_format)

    assert normalized == response_format
    assert normalized is not response_format
    assert normalized is not None
    assert normalized.schema_ is not response_format.schema_


def test_normalizes_mapping_without_mutation() -> None:
    response_format: dict[str, Any] = {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
        },
        "strict": True,
    }
    original = copy.deepcopy(response_format)

    normalized = primary.normalize_response_format(response_format)

    assert response_format == original
    assert normalized is not None
    assert normalized.model_dump(exclude_none=True, by_alias=True) == original


def test_normalizes_pydantic_class_with_strict() -> None:
    normalized = primary.normalize_response_format(Answer, strict=True)

    assert normalized == gm.ChatResponseFormat(
        type="json_schema",
        schema=Answer.model_json_schema(),
        strict=True,
    )


def test_normalizes_raw_json_schema_mapping() -> None:
    schema = Answer.model_json_schema()

    normalized = primary.normalize_response_format(schema)

    assert normalized == gm.ChatResponseFormat(
        type="json_schema",
        schema=schema,
    )


def test_rejects_conflicting_strict_values() -> None:
    with pytest.raises(ValueError, match="already defines strict=False"):
        primary.normalize_response_format(
            gm.JsonSchemaResponseFormat(
                schema=Answer.model_json_schema(),
                strict=False,
            ),
            strict=True,
        )


def test_none_response_format_is_omitted() -> None:
    assert primary.normalize_response_format(None) is None


def test_invalid_response_format_type_raises() -> None:
    with pytest.raises(TypeError, match="response_format must be"):
        primary.normalize_response_format("json_schema")


def test_invalid_response_format_mapping_raises() -> None:
    with pytest.raises(ValueError, match="Invalid primary response_format"):
        primary.normalize_response_format(
            {
                "type": "json_schema",
                "schema": object(),
            }
        )
