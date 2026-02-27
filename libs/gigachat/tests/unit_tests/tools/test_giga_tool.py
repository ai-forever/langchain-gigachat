from typing import Any

import pytest
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

from langchain_gigachat.tools.giga_tool import (
    GigaBaseTool,
    GigaStructuredTool,
    GigaTool,
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
    assert isinstance(greet, GigaStructuredTool)


# ---------------------------------------------------------------------------
# @giga_tool("custom_name") — string name
# ---------------------------------------------------------------------------


def test_giga_tool_with_name() -> None:
    @giga_tool("my_greeter")
    def greet(name: str) -> str:
        """Say hello"""
        return f"Hello {name}"

    assert greet.name == "my_greeter"


# ---------------------------------------------------------------------------
# @giga_tool(few_shot_examples=..., return_schema=...)
# ---------------------------------------------------------------------------


class GreetResult(BaseModel):
    message: str = Field(description="greeting")


def test_giga_tool_with_return_schema_and_few_shots() -> None:
    fse = [{"request": "greet Bob", "params": {"name": "Bob"}}]

    @giga_tool(few_shot_examples=fse, return_schema=GreetResult)
    def greet(name: str) -> GreetResult:
        """Say hello"""
        return GreetResult(message=f"Hello {name}")

    assert greet.return_schema is GreetResult
    assert greet.few_shot_examples == fse


# ---------------------------------------------------------------------------
# @giga_tool — coroutine function
# ---------------------------------------------------------------------------


def test_giga_tool_coroutine() -> None:
    @giga_tool
    async def agreet(name: str) -> str:
        """Say hello async"""
        return f"Hello {name}"

    assert agreet.name == "agreet"
    assert isinstance(agreet, GigaStructuredTool)
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


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_giga_tool_too_many_args() -> None:
    with pytest.raises(ValueError, match="Too many arguments"):
        giga_tool("name", "extra_arg", "more")  # type: ignore[call-overload]


def test_giga_tool_runnable_without_string_name() -> None:
    def fn(input_dict: dict) -> str:  # type: ignore[type-arg]
        return str(input_dict)

    runnable = RunnableLambda(fn)
    with pytest.raises(ValueError, match="name must be a string"):
        giga_tool(runnable=runnable)  # type: ignore[call-overload]


def test_giga_tool_invalid_first_arg() -> None:
    with pytest.raises(ValueError, match="First argument must be a string"):
        giga_tool(123)  # type: ignore[call-overload]


# ---------------------------------------------------------------------------
# infer_schema=False
# ---------------------------------------------------------------------------


def test_giga_tool_no_infer_schema_no_docstring() -> None:
    def no_doc(x: int) -> int:
        return x

    no_doc.__doc__ = None  # type: ignore[assignment]

    with pytest.raises(ValueError, match="Function must have a docstring"):
        giga_tool("no_doc_tool", infer_schema=False)(no_doc)


def test_giga_tool_no_infer_schema_with_docstring() -> None:
    def my_fn(x: int) -> int:
        """Has a docstring"""
        return x

    tool = giga_tool("my_fn_tool", infer_schema=False)(my_fn)
    assert isinstance(tool, GigaTool)
    assert tool.name == "my_fn_tool"


# ---------------------------------------------------------------------------
# GigaStructuredTool.from_function — description inference
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


# ---------------------------------------------------------------------------
# @giga_tool — partial (no name_or_callable, kwargs only)
# ---------------------------------------------------------------------------


def test_giga_tool_partial_decorator() -> None:
    decorator = giga_tool(few_shot_examples=[{"request": "x", "params": {}}])

    @decorator
    def my_tool(a: int) -> int:
        """Tool with partial"""
        return a

    assert my_tool.name == "my_tool"
    assert my_tool.few_shot_examples == [{"request": "x", "params": {}}]
