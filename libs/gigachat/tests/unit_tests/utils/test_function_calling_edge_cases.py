"""Edge-case tests for utils/function_calling.py."""

from typing import Any, Dict, Union

import pytest
from pydantic import BaseModel, Field

from langchain_gigachat.utils.function_calling import (
    IncorrectSchemaException,
    _convert_return_schema,
    _model_to_schema,
    _parse_google_docstring,
    convert_to_gigachat_function,
    gigachat_fix_schema,
)

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


def test_fix_schema_anyof_nullable_collapses() -> None:
    """Optional[str] — anyOf with null should collapse to the non-null type."""
    schema: Dict[str, Any] = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    result = gigachat_fix_schema(schema)
    assert result == {"type": "string"}


def test_fix_schema_anyof_union_with_null() -> None:
    """str | dict | None — strips null, takes first non-null type."""
    schema: Dict[str, Any] = {
        "anyOf": [
            {"type": "string"},
            {"type": "object", "additionalProperties": True},
            {"type": "null"},
        ]
    }
    result = gigachat_fix_schema(schema)
    assert result["type"] == "string"
    assert "anyOf" not in result


def test_fix_schema_anyof_scalars() -> None:
    """int | str — takes first variant."""
    schema: Dict[str, Any] = {"anyOf": [{"type": "integer"}, {"type": "string"}]}
    result = gigachat_fix_schema(schema)
    assert result == {"type": "integer"}


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
        "type": "object",
        "properties": {"r": {"type": "integer"}},
    }
    result = _convert_return_schema(schema)
    assert result is schema


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


def test_convert_to_gigachat_function_union_param() -> None:
    """Union[int, float] should no longer raise — it collapses to string."""
    from langchain_core.tools import tool

    @tool
    def union_fn(x: Union[int, float]) -> str:
        """Union fn"""
        return str(x)

    result = convert_to_gigachat_function(union_fn)
    assert "function" not in result or isinstance(result, dict)
