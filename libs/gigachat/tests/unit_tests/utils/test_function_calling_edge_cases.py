"""Edge-case tests for utils/function_calling.py."""

import copy
from typing import Any, Dict, Union

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from langchain_gigachat.utils.function_calling import (
    IncorrectSchemaException,
    _convert_return_schema,
    _model_to_schema,
    _parse_google_docstring,
    convert_to_gigachat_function,
    format_tool_to_gigachat_function,
    gigachat_fix_schema,
)


def _schema_refs(value: Any) -> list[Any]:
    if isinstance(value, dict):
        refs = [value["$ref"]] if "$ref" in value else []
        return refs + [ref for nested in value.values() for ref in _schema_refs(nested)]
    if isinstance(value, list):
        return [ref for nested in value for ref in _schema_refs(nested)]
    return []


# ---------------------------------------------------------------------------
# gigachat_fix_schema
# ---------------------------------------------------------------------------


def test_fix_schema_allof_single() -> None:
    schema: Dict[str, Any] = {
        "properties": {
            "field": {
                "allOf": [{"type": "object", "description": "inner"}],
                "description": "outer",
            }
        }
    }
    result = gigachat_fix_schema(schema)
    assert result["properties"]["field"]["type"] == "object"
    assert result["properties"]["field"]["description"] == "outer"
    assert "allOf" not in result["properties"]["field"]


def test_fix_schema_allof_multiple_raises() -> None:
    schema: Dict[str, Any] = {"allOf": [{"type": "string"}, {"type": "integer"}]}
    with pytest.raises(IncorrectSchemaException):
        gigachat_fix_schema(schema)


def test_fix_schema_anyof_multiple_raises() -> None:
    schema: Dict[str, Any] = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    with pytest.raises(IncorrectSchemaException):
        gigachat_fix_schema(schema)


@pytest.mark.parametrize("keyword", ["allOf", "anyOf"])
def test_fix_schema_single_combinator_is_collapsed(keyword: str) -> None:
    schema = {
        keyword: [
            {
                "type": "object",
                "properties": {"value": {"type": "string"}},
            }
        ]
    }

    assert gigachat_fix_schema(schema) == {
        "type": "object",
        "properties": {"value": {"type": "string"}},
    }


@pytest.mark.parametrize(
    "schema",
    [
        {"allOf": []},
        {"anyOf": []},
        {"allOf": "not-a-list"},
        {"anyOf": [42]},
    ],
)
def test_fix_schema_malformed_combinator_raises_integration_error(
    schema: dict[str, Any],
) -> None:
    with pytest.raises(IncorrectSchemaException, match="allOf|anyOf"):
        gigachat_fix_schema(schema)


def test_fix_schema_preserves_recursive_ref_shape_without_mutation() -> None:
    schema = {
        "type": "object",
        "properties": {"child": {"$ref": "#/$defs/Node"}},
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {
                    "child": {"anyOf": [{"$ref": "#/$defs/Node"}]},
                },
            }
        },
    }
    original = copy.deepcopy(schema)

    result = gigachat_fix_schema(schema)

    assert schema == original
    assert result["$defs"]["Node"]["properties"]["child"] == {"$ref": "#/$defs/Node"}


def test_fix_schema_title_removed_at_top_level() -> None:
    schema: Dict[str, Any] = {"title": "MyModel", "type": "object"}
    result = gigachat_fix_schema(schema)
    assert "title" not in result


def test_fix_schema_list_items() -> None:
    schema = [{"title": "a", "type": "string"}, {"type": "integer"}]
    result = gigachat_fix_schema(schema)
    assert len(result) == 2
    assert "title" not in result[0]


def test_fix_schema_passthrough_scalar() -> None:
    assert gigachat_fix_schema(42) == 42
    assert gigachat_fix_schema("hello") == "hello"


# ---------------------------------------------------------------------------
# _parse_google_docstring
# ---------------------------------------------------------------------------


def test_parse_google_docstring_valid() -> None:
    doc = """Short description.

Args:
    arg1: First argument.
    arg2: Second argument."""
    desc, args = _parse_google_docstring(doc, ["arg1", "arg2"])
    assert desc == "Short description."
    assert args["arg1"] == "First argument."
    assert args["arg2"] == "Second argument."


def test_parse_google_docstring_no_args_block() -> None:
    doc = """Just a description."""
    desc, args = _parse_google_docstring(doc, ["arg1"])
    assert desc == "Just a description."
    assert args == {}


def test_parse_google_docstring_none() -> None:
    desc, args = _parse_google_docstring(None, [])
    assert desc == ""
    assert args == {}


def test_parse_google_docstring_error_on_invalid_with_args() -> None:
    with pytest.raises(ValueError, match="invalid Google-Style docstring"):
        _parse_google_docstring(
            "No args section",
            ["arg1"],
            error_on_invalid_docstring=True,
        )


def test_parse_google_docstring_error_on_invalid_none() -> None:
    with pytest.raises(ValueError, match="invalid Google-Style docstring"):
        _parse_google_docstring(None, [], error_on_invalid_docstring=True)


def test_parse_google_docstring_multiline_arg() -> None:
    doc = """Desc.

Args:
    arg1: First line.
        Continuation.
"""
    _, args = _parse_google_docstring(doc, ["arg1"])
    assert "Continuation." in args["arg1"]


def test_parse_google_docstring_returns_before_args() -> None:
    doc = """Desc.

Returns:
    Something.

Args:
    x: value."""
    desc, args = _parse_google_docstring(doc, ["x"])
    assert desc == "Desc."
    assert args["x"] == "value."


# ---------------------------------------------------------------------------
# _model_to_schema
# ---------------------------------------------------------------------------


def test_model_to_schema_valid() -> None:
    class M(BaseModel):
        x: int

    result = _model_to_schema(M)
    assert "properties" in result
    assert "x" in result["properties"]


def test_model_to_schema_not_pydantic() -> None:
    with pytest.raises(TypeError, match="must be a Pydantic model"):
        _model_to_schema({"not": "a model"})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _convert_return_schema
# ---------------------------------------------------------------------------


def test_convert_return_schema_none() -> None:
    assert _convert_return_schema(None) == {}


def test_convert_return_schema_dict() -> None:
    schema: Dict[str, Any] = {
        "title": "ReturnValue",
        "type": "object",
        "$defs": {"Nested": {"type": "integer"}},
        "properties": {"r": {"allOf": [{"$ref": "#/$defs/Nested"}]}},
    }
    original = copy.deepcopy(schema)

    result = _convert_return_schema(schema)

    assert schema == original
    assert result is not schema
    assert "title" not in result
    assert "$defs" not in result
    assert _schema_refs(result) == []
    assert result["properties"]["r"]["allOf"] == [{"type": "integer"}]
    assert result["properties"]["r"]["description"] == ""


def test_convert_return_schema_recursively_dereferences_raw_schema() -> None:
    schema = {
        "$defs": {
            "Envelope": {
                "type": "object",
                "properties": {"payload": {"$ref": "#/$defs/Payload"}},
            },
            "Payload": {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
            },
        },
        "type": "object",
        "properties": {"result": {"$ref": "#/$defs/Envelope"}},
    }
    original = copy.deepcopy(schema)

    result = _convert_return_schema(schema)

    assert schema == original
    assert "$defs" not in result
    assert _schema_refs(result) == []
    assert result["properties"]["result"]["properties"]["payload"] == {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
    }


def test_convert_return_schema_rejects_missing_local_ref_without_mutation() -> None:
    schema = {
        "type": "object",
        "properties": {"result": {"$ref": "#/$defs/Missing"}},
    }
    original = copy.deepcopy(schema)

    with pytest.raises(
        IncorrectSchemaException,
        match=r"unresolved local \$ref.*#/\$defs/Missing",
    ):
        _convert_return_schema(schema)

    assert schema == original


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        ({"type": "object"}, {"type": "object", "properties": {}}),
        ({"type": "string"}, {"type": "string"}),
        ({}, {}),
    ],
)
def test_convert_return_schema_without_properties_is_explicit(
    schema: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    original = copy.deepcopy(schema)

    assert _convert_return_schema(schema) == expected
    assert schema == original


def test_format_tool_with_raw_schemas_is_immutable_and_description_optional() -> None:
    args_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
    }
    return_schema = {
        "title": "SearchResult",
        "type": "object",
        "properties": {"count": {"type": "integer"}},
    }
    tool = StructuredTool(
        name="search",
        description="Search documents",
        args_schema=args_schema,  # type: ignore[arg-type]
        extras={"return_schema": return_schema},
    )
    original_args = copy.deepcopy(args_schema)
    original_return = copy.deepcopy(return_schema)

    result = format_tool_to_gigachat_function(tool)

    assert args_schema == original_args
    assert return_schema == original_return
    assert result["description"] == "Search documents"
    assert result["parameters"] == args_schema
    assert result["parameters"] is not args_schema
    assert result["return_parameters"] is not return_schema


def test_convert_return_schema_pydantic() -> None:
    class R(BaseModel):
        """Return desc"""

        val: int = Field(description="value")

    result = _convert_return_schema(R)
    assert result["type"] == "object"
    assert "val" in result["properties"]
    assert "title" not in result


# ---------------------------------------------------------------------------
# convert_to_gigachat_function — unsupported type
# ---------------------------------------------------------------------------


def test_convert_to_gigachat_function_unsupported_type() -> None:
    with pytest.raises(ValueError, match="Unsupported function type"):
        convert_to_gigachat_function(12345)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# convert_to_gigachat_function — dict passthrough
# ---------------------------------------------------------------------------


def test_convert_to_gigachat_function_dict_passthrough() -> None:
    d: Dict[str, Any] = {
        "name": "my_fn",
        "description": "desc",
        "parameters": {"type": "object", "properties": {}},
    }
    result = convert_to_gigachat_function(d)
    assert result == d


# ---------------------------------------------------------------------------
# convert_to_gigachat_function — IncorrectSchemaException wraps message
# ---------------------------------------------------------------------------


def test_convert_to_gigachat_function_incorrect_schema() -> None:
    from langchain_core.tools import tool

    @tool
    def bad_fn(x: Union[int, float]) -> str:
        """Bad fn"""
        return str(x)

    with pytest.raises(IncorrectSchemaException, match="do not support"):
        convert_to_gigachat_function(bad_fn)
