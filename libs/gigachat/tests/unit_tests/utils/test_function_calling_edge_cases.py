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
    convert_to_gigachat_tool,
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
    """str | dict | None — merges into object with discriminator."""
    schema: Dict[str, Any] = {
        "anyOf": [
            {"type": "string"},
            {"type": "object", "additionalProperties": True},
            {"type": "null"},
        ]
    }
    result = gigachat_fix_schema(schema)
    assert result["type"] == "object"
    assert "_type" in result["properties"]
    assert result["required"] == ["_type"]


def test_fix_schema_anyof_multiple_scalars_merges() -> None:
    """int | str — merges into object with discriminator."""
    schema: Dict[str, Any] = {"anyOf": [{"type": "integer"}, {"type": "string"}]}
    result = gigachat_fix_schema(schema)
    assert result["type"] == "object"
    assert "_type" in result["properties"]


def test_merge_two_object_variants() -> None:
    """Two object variants with titles merge into flat object."""
    schema: Dict[str, Any] = {
        "anyOf": [
            {
                "type": "object",
                "title": "Cat",
                "properties": {
                    "meow": {"type": "string", "description": "sound"},
                },
            },
            {
                "type": "object",
                "title": "Dog",
                "properties": {
                    "bark": {"type": "string", "description": "woof"},
                },
            },
        ]
    }
    result = gigachat_fix_schema(schema)
    assert result["type"] == "object"
    props = result["properties"]
    assert "_type" in props
    assert props["_type"]["enum"] == ["Cat", "Dog"]
    assert "meow" in props
    assert "bark" in props
    assert props["meow"]["description"].startswith("Cat:")
    assert props["bark"]["description"].startswith("Dog:")
    assert result["required"] == ["_type"]


def test_merge_object_property_collision_same_type() -> None:
    """Shared property with same type merges descriptions."""
    schema: Dict[str, Any] = {
        "anyOf": [
            {
                "type": "object",
                "title": "A",
                "properties": {
                    "name": {"type": "string", "description": "a name"},
                },
            },
            {
                "type": "object",
                "title": "B",
                "properties": {
                    "name": {"type": "string", "description": "b name"},
                },
            },
        ]
    }
    result = gigachat_fix_schema(schema)
    desc = result["properties"]["name"]["description"]
    assert "A:" in desc
    assert "B:" in desc


def test_merge_object_property_collision_different_type_raises() -> None:
    """Shared property with different types raises."""
    schema: Dict[str, Any] = {
        "anyOf": [
            {
                "type": "object",
                "title": "A",
                "properties": {
                    "val": {"type": "integer", "description": "int val"},
                },
            },
            {
                "type": "object",
                "title": "B",
                "properties": {
                    "val": {"type": "string", "description": "str val"},
                },
            },
        ]
    }
    with pytest.raises(IncorrectSchemaException):
        gigachat_fix_schema(schema)


def test_merge_object_no_title_fallback() -> None:
    """Variants without titles use variant_1, variant_2."""
    schema: Dict[str, Any] = {
        "anyOf": [
            {
                "type": "object",
                "properties": {"x": {"type": "integer", "description": "x"}},
            },
            {
                "type": "object",
                "properties": {"y": {"type": "string", "description": "y"}},
            },
        ]
    }
    result = gigachat_fix_schema(schema)
    assert result["properties"]["_type"]["enum"] == [
        "variant_1",
        "variant_2",
    ]


def test_merge_mixed_scalar_and_object() -> None:
    """Scalar + object variants merge into object."""
    schema: Dict[str, Any] = {
        "anyOf": [
            {"type": "string"},
            {
                "type": "object",
                "title": "Obj",
                "properties": {
                    "a": {"type": "integer", "description": "num"},
                },
            },
        ]
    }
    result = gigachat_fix_schema(schema)
    assert result["type"] == "object"
    assert "_type" in result["properties"]


def test_merge_discriminator_name_collision() -> None:
    """If a variant has _type property, discriminator becomes __type."""
    schema: Dict[str, Any] = {
        "anyOf": [
            {
                "type": "object",
                "title": "A",
                "properties": {
                    "_type": {"type": "string", "description": "type"},
                },
            },
            {
                "type": "object",
                "title": "B",
                "properties": {
                    "x": {"type": "integer", "description": "x"},
                },
            },
        ]
    }
    result = gigachat_fix_schema(schema)
    assert "__type" in result["properties"]
    assert result["properties"]["__type"]["enum"] == ["A", "B"]


def test_merge_object_enum_fields() -> None:
    """Enum values on colliding properties are merged."""
    schema: Dict[str, Any] = {
        "anyOf": [
            {
                "type": "object",
                "title": "A",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "status",
                        "enum": ["on", "off"],
                    },
                },
            },
            {
                "type": "object",
                "title": "B",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "status",
                        "enum": ["pending", "done"],
                    },
                },
            },
        ]
    }
    result = gigachat_fix_schema(schema)
    enums = result["properties"]["status"]["enum"]
    assert set(enums) == {"on", "off", "pending", "done"}


def test_merge_end_to_end() -> None:
    """Full integration: Union[ModelA, ModelB] in a tool."""
    from langchain_core.tools import tool

    class FileTarget(BaseModel):
        """Save to file."""

        path: str = Field(description="file path")

    class WebhookTarget(BaseModel):
        """Post to webhook."""

        url: str = Field(description="endpoint URL")

    @tool
    def save(dest: Union[FileTarget, WebhookTarget]) -> str:
        """Save data.

        Args:
            dest: Where to save."""
        return "ok"

    result = convert_to_gigachat_function(save)
    dest_props = result["parameters"]["properties"]["dest"]
    assert dest_props["type"] == "object"
    assert "_type" in dest_props["properties"]
    disc = dest_props["properties"]["_type"]
    assert "FileTarget" in disc["enum"]
    assert "WebhookTarget" in disc["enum"]
    assert "path" in dest_props["properties"]
    assert "url" in dest_props["properties"]


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
    """Union[int, float] merges into object with discriminator."""
    from langchain_core.tools import tool

    @tool
    def union_fn(x: Union[int, float]) -> str:
        """Union fn"""
        return str(x)

    result = convert_to_gigachat_function(union_fn)
    assert isinstance(result, dict)
    x_schema = result["parameters"]["properties"]["x"]
    assert x_schema["type"] == "object"
    assert "_type" in x_schema["properties"]


def test_convert_to_gigachat_tool_preformatted_fixes_schema() -> None:
    """Pre-formatted tool dict with type=function should still be fixed."""
    tool = {
        "type": "function",
        "function": {
            "name": "update_files",
            "description": "Updates a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "fileId": {"type": "string", "description": "File ID"},
                    "properties": {
                        "type": "object",
                        "description": "Key-value pairs",
                    },
                },
                "required": ["fileId"],
            },
        },
    }
    result = convert_to_gigachat_tool(tool)
    # The nested "properties" field (type: object) must get an empty properties dict
    inner = result["function"]["parameters"]["properties"]["properties"]
    assert inner["type"] == "object"
    assert "properties" in inner
    assert inner["properties"] == {}
