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


def test_accepts_schema_less_json_without_mutating_input() -> None:
    response_format = {"type": "json_schema"}
    original = copy.deepcopy(response_format)

    normalized = primary.normalize_response_format(response_format)

    assert response_format == original
    assert normalized is not None
    assert normalized.model_dump(exclude_none=True, by_alias=True) == {
        "type": "json_schema"
    }


def test_explicit_none_schema_normalizes_to_schema_less_json() -> None:
    response_format = {"type": "json_schema", "schema": None, "strict": None}

    normalized = primary.normalize_response_format(response_format)

    assert normalized is not None
    assert normalized.model_dump(exclude_none=True, by_alias=True) == {
        "type": "json_schema"
    }


@pytest.mark.parametrize("strict", [False, True])
def test_rejects_strict_without_schema(strict: bool) -> None:
    with pytest.raises(
        ValueError,
        match="schema-less response_format.*cannot include field: strict",
    ):
        primary.normalize_response_format({"type": "json_schema", "strict": strict})


def test_unwraps_openai_nested_json_schema_format() -> None:
    normalized = primary.normalize_response_format(
        {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": Answer.model_json_schema(),
                "strict": True,
            },
        }
    )

    assert normalized == gm.ChatResponseFormat(
        type="json_schema",
        schema=Answer.model_json_schema(),
        strict=True,
    )


@pytest.mark.parametrize(
    "nested",
    [
        {},
        {"strict": False},
        {"strict": True},
    ],
)
def test_rejects_nested_json_schema_without_schema(
    nested: dict[str, Any],
) -> None:
    with pytest.raises(
        ValueError,
        match="Nested response_format 'json_schema' requires a 'schema' field",
    ):
        primary.normalize_response_format(
            {"type": "json_schema", "json_schema": nested}
        )


@pytest.mark.parametrize(
    ("response_format", "expected"),
    [
        ({"type": "text"}, gm.ChatResponseFormat(type="text")),
        (
            {"type": "regex", "regex": r"[A-Z]{2}-[0-9]{4}"},
            gm.ChatResponseFormat(type="regex", regex=r"[A-Z]{2}-[0-9]{4}"),
        ),
    ],
)
def test_preserves_explicit_sdk_response_formats(
    response_format: dict[str, Any],
    expected: gm.ChatResponseFormat,
) -> None:
    assert primary.normalize_response_format(response_format) == expected


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


def test_normalizes_raw_json_schema_with_type_keyword() -> None:
    schema = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
    }

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


def test_strict_without_response_format_raises() -> None:
    with pytest.raises(
        ValueError,
        match="strict is supported only together with response_format",
    ):
        primary.normalize_response_format(None, strict=True)


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


def test_unknown_explicit_response_format_type_raises() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported primary response_format type 'yaml'",
    ):
        primary.normalize_response_format({"type": "yaml"})


def test_regex_response_format_requires_regex() -> None:
    with pytest.raises(
        ValueError,
        match="response_format type 'regex' requires a non-empty string 'regex' field",
    ):
        primary.normalize_response_format({"type": "regex"})


@pytest.mark.parametrize(
    ("response_format", "match"),
    [
        (
            {"type": "text", "schema": {"type": "object"}},
            r"type 'text' cannot include field\(s\): schema",
        ),
        (
            {"type": "text", "regex": "value"},
            r"type 'text' cannot include field\(s\): regex",
        ),
        (
            {"type": "text", "strict": False},
            r"type 'text' cannot include field\(s\): strict",
        ),
        (
            {"type": "regex", "regex": ""},
            "requires a non-empty string 'regex' field",
        ),
        (
            {"type": "regex", "regex": "value", "schema": {"type": "string"}},
            r"type 'regex' cannot include field\(s\): schema",
        ),
        (
            {"type": "regex", "regex": "value", "strict": False},
            r"type 'regex' cannot include field\(s\): strict",
        ),
        (
            {
                "type": "json_schema",
                "schema": {"type": "object"},
                "regex": "value",
            },
            "type 'json_schema' cannot include field: regex",
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
def test_response_format_rejects_conflicting_cross_fields(
    response_format: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        primary.normalize_response_format(response_format)
