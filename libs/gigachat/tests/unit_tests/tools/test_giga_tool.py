from typing import Any

import pytest
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import StructuredTool, tool
from pydantic import BaseModel, Field

from langchain_gigachat.tools.giga_tool import (
    GigaBaseTool,
    GigaStructuredTool,
    giga_tool,
)

# ---------------------------------------------------------------------------
# @giga_tool — bare decorator (no args)
# ---------------------------------------------------------------------------


def test_giga_tool_bare_decorator() -> None:
    @giga_tool
    def greet(name: str) -> str:
        """Say hello"""
        return f"Hello {name}"

    assert greet.name == "greet"
    assert greet.description == "Say hello"
    assert isinstance(greet, StructuredTool)


# ---------------------------------------------------------------------------
# @giga_tool("custom_name") — string name
# ---------------------------------------------------------------------------


def test_giga_tool_with_name() -> None:
    @giga_tool("my_greeter")
    def greet(name: str) -> str:
        """Say hello"""
        return f"Hello {name}"

    assert greet.name == "my_greeter"


class GreetResult(BaseModel):
    message: str = Field(description="greeting")


def test_giga_tool_extras_match_standard_tool() -> None:
    fse = [{"request": "greet Bob", "params": {"name": "Bob"}}]

    @giga_tool(extras={"few_shot_examples": fse, "return_schema": GreetResult})
    def greet(name: str) -> str:
        """Say hello"""
        return f"Hello {name}"

    assert greet.extras is not None
    assert greet.extras["return_schema"] is GreetResult
    assert greet.extras["few_shot_examples"] == fse


# ---------------------------------------------------------------------------
# @giga_tool — coroutine function
# ---------------------------------------------------------------------------


def test_giga_tool_coroutine() -> None:
    @giga_tool
    async def agreet(name: str) -> str:
        """Say hello async"""
        return f"Hello {name}"

    assert agreet.name == "agreet"
    assert isinstance(agreet, StructuredTool)
    assert agreet.coroutine is not None


# ---------------------------------------------------------------------------
# giga_tool with Runnable
# ---------------------------------------------------------------------------


def test_giga_tool_with_runnable() -> None:
    def fn(input_dict: dict) -> str:  # type: ignore[type-arg]
        return str(input_dict)

    runnable = RunnableLambda(fn)
    tool = giga_tool("runnable_tool", runnable)
    assert tool.name == "runnable_tool"


def test_giga_tool_with_runnable_kwarg() -> None:
    def fn(input_dict: dict) -> str:  # type: ignore[type-arg]
        return str(input_dict)

    runnable = RunnableLambda(fn)
    tool = giga_tool("runnable_tool", runnable=runnable)
    assert tool.name == "runnable_tool"


def test_giga_tool_with_explicit_description() -> None:
    @giga_tool(description="Custom description")
    def greet(name: str) -> str:
        """Say hello"""
        return f"Hello {name}"

    assert greet.description == "Custom description"


def test_giga_tool_with_name_and_explicit_description() -> None:
    @giga_tool("my_greeter", description="Custom description")
    def greet(name: str) -> str:
        """Say hello"""
        return f"Hello {name}"

    assert greet.name == "my_greeter"
    assert greet.description == "Custom description"


def test_giga_tool_is_standard_tool_alias() -> None:
    assert giga_tool is tool


def test_giga_tool_legacy_few_shot_examples_kwarg_is_rejected() -> None:
    with pytest.raises(TypeError, match="few_shot_examples"):
        giga_tool(few_shot_examples=[])  # type: ignore[call-arg]


def test_giga_tool_legacy_return_schema_kwarg_is_rejected() -> None:
    with pytest.raises(TypeError, match="return_schema"):
        giga_tool(return_schema=GreetResult)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# GigaStructuredTool.from_function — legacy manual subclass path
# ---------------------------------------------------------------------------


def test_from_function_no_description_raises() -> None:
    def no_doc(x: int) -> int:
        return x

    no_doc.__doc__ = None  # type: ignore[assignment]

    with pytest.raises(ValueError, match="Function must have a docstring"):
        GigaStructuredTool.from_function(no_doc, description=None, infer_schema=False)


# ---------------------------------------------------------------------------
# GigaBaseTool fields
# ---------------------------------------------------------------------------


def test_giga_base_tool_defaults() -> None:
    class MyTool(GigaBaseTool):  # type: ignore[override]
        name: str = "test"
        description: str = "test tool"

        def _run(self, *args: Any, **kwargs: Any) -> str:
            return "ok"

    t = MyTool()
    assert t.return_schema is None
    assert t.few_shot_examples is None


def test_giga_tool_partial_decorator_with_extras() -> None:
    decorator = giga_tool(extras={"few_shot_examples": [{"request": "x", "params": {}}]})

    @decorator
    def my_tool(a: int) -> int:
        """Tool with partial"""
        return a

    assert my_tool.name == "my_tool"
    assert my_tool.extras == {"few_shot_examples": [{"request": "x", "params": {}}]}
