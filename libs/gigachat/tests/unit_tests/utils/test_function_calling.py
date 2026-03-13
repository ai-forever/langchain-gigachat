# mypy: disable-error-code="annotation-unchecked"
from typing import Annotated as ExtensionsAnnotated
from typing import Any, Callable, List, Literal, Optional, Union
from typing import TypedDict as TypingTypedDict

import pytest
from typing_extensions import TypedDict as ExtensionsTypedDict
from typing_extensions import is_typeddict

try:
    from typing import Annotated as TypingAnnotated  # type: ignore[attr-defined]
except ImportError:
    TypingAnnotated = ExtensionsAnnotated

from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool, StructuredTool, Tool, tool
from pydantic import BaseModel, Field

from langchain_gigachat.utils.function_calling import (
    IncorrectSchemaException,
    convert_to_gigachat_function,
)


@pytest.fixture()
def pydantic() -> type[BaseModel]:
    class dummy_function(BaseModel):  # noqa: N801
        """dummy function"""

        arg1: Optional[int] = Field(..., description="foo")
        arg2: Literal["bar", "baz"] = Field(..., description="one of 'bar', 'baz'")

    return dummy_function


@pytest.fixture()
def annotated_function() -> Callable:
    def dummy_function(
        arg1: ExtensionsAnnotated[Optional[int], "foo"],
        arg2: ExtensionsAnnotated[Literal["bar", "baz"], "one of 'bar', 'baz'"],
    ) -> None:
        """dummy function"""

    return dummy_function


@pytest.fixture()
def function() -> Callable:
    def dummy_function(arg1: Optional[int], arg2: Literal["bar", "baz"]) -> None:
        """dummy function

        Args:
            arg1: foo
            arg2: one of 'bar', 'baz'
        """

    return dummy_function


@pytest.fixture()
def runnable() -> Runnable:
    class Args(ExtensionsTypedDict):
        arg1: ExtensionsAnnotated[Optional[int], "foo"]
        arg2: ExtensionsAnnotated[Literal["bar", "baz"], "one of 'bar', 'baz'"]

    def dummy_function(input_dict: Args) -> None:
        pass

    return RunnableLambda(dummy_function)


@pytest.fixture()
def dummy_tool() -> BaseTool:
    class Schema(BaseModel):
        arg1: Optional[int] = Field(..., description="foo")
        arg2: Literal["bar", "baz"] = Field(..., description="one of 'bar', 'baz'")

    class DummyFunction(BaseTool):  # type: ignore[override]
        args_schema: type[BaseModel] = Schema
        name: str = "dummy_function"
        description: str = "dummy function"

        def _run(self, *args: Any, **kwargs: Any) -> Any:
            pass

    return DummyFunction()


@pytest.fixture()
def dummy_structured_tool() -> StructuredTool:
    class Schema(BaseModel):
        arg1: Optional[int] = Field(..., description="foo")
        arg2: Literal["bar", "baz"] = Field(..., description="one of 'bar', 'baz'")

    return StructuredTool.from_function(
        lambda x: None,
        name="dummy_function",
        description="dummy function",
        args_schema=Schema,
    )


@pytest.fixture()
def dummy_structured_tool_with_dict_args_schema() -> StructuredTool:
    schema = {
        "properties": {
            "arg1": {"description": "foo", "type": "integer"},
            "arg2": {
                "description": "one of 'bar', 'baz'",
                "enum": ["bar", "baz"],
                "type": "string",
            },
        },
        "required": ["arg2"],
        "title": "dummy_function",
        "type": "object",
    }

    return StructuredTool(
        name="dummy_function",
        description="dummy function",
        args_schema=schema,  # type: ignore[arg-type]
    )


@pytest.fixture()
def dummy_pydantic() -> type[BaseModel]:
    class dummy_function(BaseModel):  # noqa: N801
        """dummy function"""

        arg1: Optional[int] = Field(..., description="foo")
        arg2: Literal["bar", "baz"] = Field(..., description="one of 'bar', 'baz'")

    return dummy_function


@pytest.fixture()
def dummy_pydantic_v2() -> type[BaseModel]:
    class dummy_function(BaseModel):  # noqa: N801
        """dummy function"""

        arg1: Optional[int] = Field(..., description="foo")
        arg2: Literal["bar", "baz"] = Field(..., description="one of 'bar', 'baz'")

    return dummy_function


@pytest.fixture()
def dummy_typing_typed_dict() -> type:
    class dummy_function(TypingTypedDict):  # noqa: N801
        """dummy function"""

        arg1: TypingAnnotated[Optional[int], None, "foo"]  # noqa: F821
        arg2: TypingAnnotated[Literal["bar", "baz"], ..., "one of 'bar', 'baz'"]  # noqa: F722

    return dummy_function


@pytest.fixture()
def dummy_typing_typed_dict_docstring() -> type:
    class dummy_function(TypingTypedDict):  # noqa: N801
        """dummy function

        Args:
            arg1: foo
            arg2: one of 'bar', 'baz'
        """

        arg1: Optional[int]
        arg2: Literal["bar", "baz"]

    return dummy_function


@pytest.fixture()
def dummy_extensions_typed_dict() -> type:
    class dummy_function(ExtensionsTypedDict):  # noqa: N801
        """dummy function"""

        arg1: ExtensionsAnnotated[int, None, "foo"]
        arg2: ExtensionsAnnotated[Literal["bar", "baz"], ..., "one of 'bar', 'baz'"]

    return dummy_function


@pytest.fixture()
def dummy_extensions_typed_dict_docstring() -> type:
    class dummy_function(ExtensionsTypedDict):  # noqa: N801
        """dummy function

        Args:
            arg1: foo
            arg2: one of 'bar', 'baz'
        """

        arg1: Optional[int]
        arg2: Literal["bar", "baz"]

    return dummy_function


@pytest.fixture()
def json_schema() -> dict:
    return {
        "name": "dummy_function",
        "description": "dummy function",
        "return_parameters": None,
        "few_shot_examples": None,
        "parameters": {
            "type": "object",
            "properties": {
                "arg1": {"description": "foo", "type": "integer"},
                "arg2": {
                    "description": "one of 'bar', 'baz'",
                    "enum": ["bar", "baz"],
                    "type": "string",
                },
            },
            "required": ["arg2"],
        },
    }


class Dummy:
    def dummy_function(self, arg1: Optional[int], arg2: Literal["bar", "baz"]) -> None:
        """dummy function

        Args:
            arg1: foo
            arg2: one of 'bar', 'baz'
        """


class DummyWithClassMethod:
    @classmethod
    def dummy_function(cls, arg1: Optional[int], arg2: Literal["bar", "baz"]) -> None:
        """dummy function

        Args:
            arg1: foo
            arg2: one of 'bar', 'baz'
        """


@pytest.fixture()
def function_with_title_parameters() -> type[BaseModel]:
    class Resource(BaseModel):
        """
        Represents a resource. Give it a good title and a short description.
        """

        url: str
        title: str
        description: str

    class ExtractResources(BaseModel):
        """
        Extract the 3-5 most relevant resources from a search result.
        """

        resources: TypingAnnotated[List[Resource], Field(description="массив ресурсов")]

    return ExtractResources


@pytest.mark.parametrize(
    "func",
    [
        "pydantic",
        "function",
        "dummy_structured_tool",
        "dummy_structured_tool_with_dict_args_schema",
        "dummy_tool",
        "dummy_typing_typed_dict",
        "dummy_typing_typed_dict_docstring",
        "dummy_extensions_typed_dict",
        "dummy_extensions_typed_dict_docstring",
        "annotated_function",
        "dummy_pydantic",
        "json_schema",
        Dummy.dummy_function,
        DummyWithClassMethod.dummy_function,
    ],
)
def test_convert_to_gigachat_function(
    func: Any, request: pytest.FixtureRequest
) -> None:
    expected = {
        "name": "dummy_function",
        "description": "dummy function",
        "return_parameters": None,
        "few_shot_examples": None,
        "parameters": {
            "type": "object",
            "properties": {
                "arg1": {"description": "foo", "type": "integer"},
                "arg2": {
                    "description": "one of 'bar', 'baz'",
                    "enum": ["bar", "baz"],
                    "type": "string",
                },
            },
            "required": ["arg2"],
        },
    }
    if isinstance(func, str):
        func = request.getfixturevalue(func)

    if is_typeddict(func):
        expected["parameters"]["properties"]["arg1"]["default"] = None  # type: ignore

    actual = convert_to_gigachat_function(func)  # type: ignore
    assert actual == expected


def test_runnable(runnable: Runnable) -> None:
    expected = {
        "name": "dummy_function",
        "description": "dummy function",
        "return_parameters": None,
        "few_shot_examples": None,
        "parameters": {
            "type": "object",
            "properties": {
                "arg1": {"type": "integer", "description": ""},
                "arg2": {"enum": ["bar", "baz"], "type": "string", "description": ""},
            },
            "required": ["arg2"],
        },
    }
    actual = convert_to_gigachat_function(
        runnable.as_tool(description="dummy function")
    )
    assert actual == expected


def test_simple_tool() -> None:
    def my_function(input_string: str) -> str:  # type: ignore
        pass

    tool = Tool(name="dummy_function", func=my_function, description="test description")
    actual = convert_to_gigachat_function(tool)
    expected = {
        "name": "dummy_function",
        "description": "test description",
        "return_parameters": None,
        "few_shot_examples": None,
        "parameters": {"properties": {}, "type": "object"},
    }
    assert actual == expected


def test_convert_to_openai_function_nested_v2() -> None:
    class NestedV2(BaseModel):
        nested_v2_arg1: int = Field(..., description="foo")
        nested_v2_arg2: Literal["bar", "baz"] = Field(
            ..., description="one of 'bar', 'baz'"
        )

    def my_function(arg1: NestedV2) -> None:
        """dummy function"""

    convert_to_gigachat_function(my_function)


def test_convert_to_gigachat_function_nested() -> None:
    class Nested(BaseModel):
        nested_arg1: int = Field(..., description="foo")
        nested_arg2: Literal["bar", "baz"] = Field(
            ..., description="one of 'bar', 'baz'"
        )

    def my_function(arg1: Nested) -> None:
        """dummy function"""

    expected = {
        "name": "my_function",
        "description": "dummy function",
        "parameters": {
            "type": "object",
            "properties": {
                "arg1": {
                    "type": "object",
                    "description": "",
                    "properties": {
                        "nested_arg1": {"type": "integer", "description": "foo"},
                        "nested_arg2": {
                            "type": "string",
                            "enum": ["bar", "baz"],
                            "description": "one of 'bar', 'baz'",
                        },
                    },
                    "required": ["nested_arg1", "nested_arg2"],
                }
            },
            "required": ["arg1"],
        },
        "return_parameters": None,
        "few_shot_examples": None,
    }

    actual = convert_to_gigachat_function(my_function)
    assert actual == expected


def test_function_optional_param() -> None:
    @tool
    def func5(a: Optional[str], b: str, c: Optional[list[Optional[str]]]) -> None:
        """A test function"""

    func = convert_to_gigachat_function(func5)
    req = func["parameters"]["required"]
    assert set(req) == {"b"}


def test_function_no_params() -> None:
    def nullary_function() -> None:
        """nullary function"""

    func = convert_to_gigachat_function(nullary_function)
    req = func["parameters"].get("required")
    assert not req


def test_convert_union_fail() -> None:
    @tool
    def magic_function(input: Union[int, float]) -> str:  # type: ignore
        """Compute a magic function."""

    with pytest.raises(IncorrectSchemaException):
        convert_to_gigachat_function(magic_function)


def test_function_with_title_parameters(
    function_with_title_parameters: type[BaseModel],
) -> None:
    expected = {
        "name": "ExtractResources",
        "description": "Extract the 3-5 most relevant resources from a search result.",
        "parameters": {
            # noqa
            "properties": {
                "resources": {
                    "description": "массив ресурсов",
                    "items": {
                        "description": "Represents a resource. Give it a good title and a short description.",  # noqa
                        # noqa
                        "properties": {
                            "url": {"type": "string"},
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["url", "title", "description"],
                        "type": "object",
                    },
                    "type": "array",
                }
            },
            "required": ["resources"],
            "type": "object",
        },
        "return_parameters": None,
        "few_shot_examples": None,
    }
    actual = convert_to_gigachat_function(function_with_title_parameters)
    assert actual == expected


def test_convert_to_function_no_args() -> None:
    @tool
    def empty_tool() -> str:
        """No args"""
        return "foo"

    actual = convert_to_gigachat_function(empty_tool)
    assert actual == {
        "name": "empty_tool",
        "description": "No args",
        "few_shot_examples": None,
        "return_parameters": None,
        "parameters": {"properties": {}, "type": "object"},
    }


# Test for return parameters and few shot examples


class ReturnParameters(BaseModel):
    """dummy function"""

    arg1: Optional[int] = Field(..., description="foo")
    arg2: Literal["bar", "baz"] = Field(..., description="one of 'bar', 'baz'")


@pytest.fixture()
def annotated_function_return_parameters() -> Callable:
    def dummy_function(  # type: ignore
        arg1: ExtensionsAnnotated[int, "foo"],
        arg2: ExtensionsAnnotated[Literal["bar", "baz"], "one of 'bar', 'baz'"],
    ) -> ReturnParameters:
        """dummy function"""

    return dummy_function


@pytest.fixture()
def function_return_parameters() -> Callable:
    def dummy_function(  # type: ignore
        arg1: int, arg2: Literal["bar", "baz"]
    ) -> ReturnParameters:
        """dummy function

        Args:
            arg1: foo
            arg2: one of 'bar', 'baz'
        """

    return dummy_function


@pytest.fixture()
def dummy_return_parameters_with_fews_decorator() -> BaseTool:
    @tool(
        extras={
            "few_shot_examples": [
                {"arg1": 1, "arg2": "bar"},
                {"arg1": 2, "arg2": "baz"},
            ]
        }
    )
    def dummy_function(
        arg1: Optional[int] = Field(..., description="foo"),
        arg2: Literal["bar", "baz"] = Field(..., description="one of 'bar', 'baz'"),
    ) -> ReturnParameters:
        """dummy function"""
        return ReturnParameters(arg1=arg1 or 0, arg2=arg2)

    return dummy_function


@pytest.fixture()
def dummy_return_parameters_through_arg_with_fews_decorator() -> BaseTool:
    @tool(
        extras={
            "few_shot_examples": [
                {"arg1": 1, "arg2": "bar"},
                {"arg1": 2, "arg2": "baz"},
            ],
            "return_schema": ReturnParameters,
        }
    )
    def dummy_function(
        arg1: Optional[int] = Field(..., description="foo"),
        arg2: Literal["bar", "baz"] = Field(..., description="one of 'bar', 'baz'"),
    ) -> None:
        """dummy function"""
        return None

    return dummy_function


@pytest.fixture()
def json_schema_return_parameters_with_fews() -> dict:
    return {
        "name": "dummy_function",
        "description": "dummy function",
        "return_parameters": {
            "type": "object",
            "description": "dummy function",
            "properties": {
                "arg1": {"description": "foo", "type": "integer"},
                "arg2": {
                    "description": "one of 'bar', 'baz'",
                    "enum": ["bar", "baz"],
                    "type": "string",
                },
            },
            "required": ["arg2"],
        },
        "few_shot_examples": [{"arg1": 1, "arg2": "bar"}, {"arg1": 2, "arg2": "baz"}],
        "parameters": {
            "type": "object",
            "description": "dummy function",
            "properties": {
                "arg1": {"description": "foo", "type": "integer"},
                "arg2": {
                    "description": "one of 'bar', 'baz'",
                    "enum": ["bar", "baz"],
                    "type": "string",
                },
            },
            "required": ["arg2"],
        },
    }


class DummyReturnParameters:
    def dummy_function(  # type: ignore
        self, arg1: Optional[int], arg2: Literal["bar", "baz"]
    ) -> ReturnParameters:
        """dummy function

        Args:
            arg1: foo
            arg2: one of 'bar', 'baz'
        """


class DummyReturnParametersWithClassMethod:
    @classmethod
    def dummy_function(  # type: ignore
        cls, arg1: Optional[int], arg2: Literal["bar", "baz"]
    ) -> ReturnParameters:
        """dummy function

        Args:
            arg1: foo
            arg2: one of 'bar', 'baz'
        """


@pytest.mark.parametrize(
    "func",
    [
        "annotated_function_return_parameters",
        "function_return_parameters",
        "dummy_return_parameters_through_arg_with_fews_decorator",
        "json_schema_return_parameters_with_fews",
        DummyReturnParameters.dummy_function,
        DummyReturnParametersWithClassMethod.dummy_function,
    ],
)
def test_function_with_return_parameters(
    func: Any, request: pytest.FixtureRequest
) -> None:
    return_params_expected = {
        "type": "object",
        "description": "dummy function",
        "properties": {
            "arg1": {"description": "foo", "type": "integer"},
            "arg2": {
                "description": "one of 'bar', 'baz'",
                "enum": ["bar", "baz"],
                "type": "string",
            },
        },
        "required": ["arg2"],
    }
    if isinstance(func, str):
        func = request.getfixturevalue(func)

    actual_func = convert_to_gigachat_function(func)
    assert actual_func["return_parameters"] == return_params_expected


def test_standard_tool_does_not_auto_infer_return_parameters(
    dummy_return_parameters_with_fews_decorator: BaseTool,
) -> None:
    actual_func = convert_to_gigachat_function(
        dummy_return_parameters_with_fews_decorator
    )
    assert actual_func["return_parameters"] is None


@pytest.mark.parametrize(
    "func",
    [
        "dummy_return_parameters_with_fews_decorator",
        "dummy_return_parameters_through_arg_with_fews_decorator",
        "json_schema_return_parameters_with_fews",
    ],
)
def test_function_with_few_shots(func: Any, request: pytest.FixtureRequest) -> None:
    few_shots_expected = [{"arg1": 1, "arg2": "bar"}, {"arg1": 2, "arg2": "baz"}]
    if isinstance(func, str):
        func = request.getfixturevalue(func)

    actual_func = convert_to_gigachat_function(func)
    assert actual_func["few_shot_examples"] == few_shots_expected


def test_dict_any_field_has_properties_key() -> None:
    """dict[str, Any] param must produce a schema with a 'properties' key.

    Pydantic emits {"type": "object", "additionalProperties": true} for such
    fields; GigaChat requires the 'properties' key to be present.
    """

    class UpdateTestCaseInput(BaseModel):
        """Update fields of an existing test case."""

        test_case_id: str = Field(..., description="Test case identifier")
        patch_json: dict[str, Any] = Field(
            ..., description="Fields to patch as a JSON object"
        )

    actual = convert_to_gigachat_function(UpdateTestCaseInput)

    patch_schema = actual["parameters"]["properties"]["patch_json"]
    assert patch_schema["type"] == "object"
    assert "properties" in patch_schema, (
        "GigaChat requires 'properties' on every object-typed field. "
        "Missing it causes 422: \"Field 'properties.patch_json.properties' is missing\""
    )
    # injecting properties must not drop additionalProperties
    assert patch_schema["additionalProperties"] is True


def test_nested_dict_in_list_has_properties_key() -> None:
    """list[dict[str, Any]] items must also have 'properties' recursively."""

    class ToolInput(BaseModel):
        """Tool with a list of free-form dicts."""

        name: str = Field(..., description="Name")
        items: list[dict[str, Any]] = Field(..., description="List of items")

    actual = convert_to_gigachat_function(ToolInput)

    items_schema = actual["parameters"]["properties"]["items"]
    assert items_schema["type"] == "array"
    item_schema = items_schema["items"]
    assert item_schema.get("type") == "object"
    assert "properties" in item_schema, (
        "Nested object inside array must have 'properties' for GigaChat"
    )


def test_raw_dict_schema_with_array_and_freeform_object() -> None:
    """Raw dict schema: array field is kept intact, object field gets 'properties'."""
    raw_schema = {
        "name": "upload_files",
        "description": "Upload files with metadata",
        "parameters": {
            "type": "object",
            "properties": {
                "file_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "metadata": {
                    "type": "object",
                    "additionalProperties": True,
                },
            },
            "required": ["file_ids", "metadata"],
        },
    }

    actual = convert_to_gigachat_function(raw_schema)
    # file_ids: array schema must pass through unchanged
    file_ids = actual["parameters"]["properties"]["file_ids"]
    assert file_ids["type"] == "array"
    assert file_ids["items"] == {"type": "string"}

    # metadata: object must get 'properties' injected, keep additionalProperties
    metadata = actual["parameters"]["properties"]["metadata"]
    assert metadata["type"] == "object"
    assert metadata["additionalProperties"] is True
    assert "properties" in metadata, (
        "GigaChat requires 'properties' on every object-typed node. "
        "Missing it causes 422: \"Field 'properties.metadata.properties' is missing\""
    )


@pytest.mark.parametrize(
    ("raw_schema", "expected_name", "expected_title"),
    [
        (
            {
                "title": "SomeResult",
                "description": "My desc",
                "properties": {
                    "value": {"type": "integer", "description": "some value"},
                },
                "required": ["value"],
                "type": "object",
            },
            "SomeResult",
            "SomeResult",
        ),
        (
            {
                "name": "my_tool",
                "title": "MyTool",
                "description": "A tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "q": {"type": "string", "description": "query"},
                    },
                },
            },
            "my_tool",
            None,
        ),
    ],
)
def test_raw_dict_schema_title_fallback_behavior(
    raw_schema: dict[str, Any], expected_name: str, expected_title: Optional[str]
) -> None:
    """Preserve title fallback while normalizing name for raw schemas."""
    actual = convert_to_gigachat_function(raw_schema)
    assert actual["name"] == expected_name
    assert actual.get("title") == expected_title
