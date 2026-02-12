"""GigaChat tool: extends LangChain @tool with return_schema and few_shot_examples."""

from __future__ import annotations

import inspect
import textwrap
from typing import Annotated, Any, Awaitable, Callable, Literal, Optional, Union

from langchain_core.callbacks import Callbacks
from langchain_core.runnables import Runnable
from langchain_core.tools import (
    FILTERED_ARGS,
    BaseTool,
    StructuredTool,
    Tool,
    _get_runnable_config_param,
    create_schema_from_function,
)
from langchain_core.utils.pydantic import TypeBaseModel
from pydantic import BaseModel
from pydantic.functional_validators import SkipValidation

from langchain_gigachat.utils.function_calling import create_return_schema_from_function

FewShotExamples = Optional[list[dict[str, Any]]]


class GigaBaseTool(BaseTool):
    """GigaChat tool interface: BaseTool plus return_schema and few_shot_examples."""

    return_schema: Annotated[Optional[TypeBaseModel], SkipValidation()] = None
    """Return schema of JSON that the tool function returns."""
    few_shot_examples: FewShotExamples = None
    """Few-shot examples to help the model understand how to use the tool."""


class GigaTool(GigaBaseTool, Tool):
    """GigaChat simple (str-in/str-out) tool."""


def _filter_schema_args(func: Callable[..., Any]) -> list[str]:
    """Exclude FILTERED_ARGS and RunnableConfig param from inferred args schema."""
    out = list(FILTERED_ARGS)
    if config_param := _get_runnable_config_param(func):
        out.append(config_param)
    return out


class GigaStructuredTool(GigaBaseTool, StructuredTool):
    """GigaChat structured tool with return_schema and few_shot_examples."""

    @classmethod
    def from_function(
        cls,
        func: Optional[Callable[..., Any]] = None,
        coroutine: Optional[Callable[..., Awaitable[Any]]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        return_direct: bool = False,
        args_schema: Union[type[BaseModel], dict[str, Any], None] = None,
        infer_schema: bool = True,
        return_schema: Optional[TypeBaseModel] = None,
        few_shot_examples: FewShotExamples = None,
        *,
        response_format: Literal["content", "content_and_artifact"] = "content",
        parse_docstring: bool = False,
        error_on_invalid_docstring: bool = False,
        **kwargs: Any,
    ) -> GigaStructuredTool:
        """Build a GigaStructuredTool from a function or coroutine.

        Same as StructuredTool.from_function, plus return_schema and few_shot_examples.
        """
        source = func if func is not None else coroutine
        if source is None:
            raise ValueError("Function and/or coroutine must be provided")

        name = name or source.__name__

        if args_schema is None and infer_schema:
            args_schema = create_schema_from_function(
                name,
                source,
                parse_docstring=parse_docstring,
                error_on_invalid_docstring=error_on_invalid_docstring,
                filter_args=_filter_schema_args(source),
            )
        if return_schema is None and infer_schema:
            return_schema = create_return_schema_from_function(source)

        description_ = description
        if description is None and not parse_docstring:
            description_ = source.__doc__ or None
        if description_ is None and args_schema is not None:
            description_ = (
                args_schema.get("description")
                if isinstance(args_schema, dict)
                else getattr(args_schema, "__doc__", None)
            )
        if description_ is None:
            raise ValueError(
                "Function must have a docstring if description not provided."
            )
        if description is None:
            description_ = textwrap.dedent(description_).strip()
        description_ = description_.strip()

        return cls(
            name=name,
            func=func,
            coroutine=coroutine,
            args_schema=args_schema,
            description=description_,
            return_direct=return_direct,
            response_format=response_format,
            return_schema=return_schema,
            few_shot_examples=few_shot_examples,
            **kwargs,
        )


def _create_giga_tool_factory(
    tool_name: str,
    *,
    return_direct: bool = False,
    args_schema: Union[type[BaseModel], dict[str, Any], None] = None,
    infer_schema: bool = True,
    response_format: Literal["content", "content_and_artifact"] = "content",
    parse_docstring: bool = False,
    error_on_invalid_docstring: bool = True,
    return_schema: Optional[type[BaseModel]] = None,
    few_shot_examples: FewShotExamples = None,
    **kwargs: Any,
) -> Callable[[Union[Callable[..., Any], Runnable]], BaseTool]:
    """Build a factory that turns a callable or Runnable into a GigaChat tool."""

    def _make_tool(dec_func: Union[Callable[..., Any], Runnable]) -> BaseTool:
        if isinstance(dec_func, Runnable):
            runnable_input_schema = dec_func.input_schema
            if runnable_input_schema.model_json_schema().get("type") != "object":
                raise ValueError("Runnable must have an object schema.")

            async def _ainvoke(callbacks: Optional[Callbacks] = None, **kw: Any) -> Any:
                return await dec_func.ainvoke(kw, {"callbacks": callbacks})

            def _invoke(callbacks: Optional[Callbacks] = None, **kw: Any) -> Any:
                return dec_func.invoke(kw, {"callbacks": callbacks})

            coroutine = _ainvoke
            func = _invoke
            schema_arg: Union[type[BaseModel], dict[str, Any], None] = (
                runnable_input_schema
            )
            description = repr(dec_func)
        elif inspect.iscoroutinefunction(dec_func):
            coroutine = dec_func
            func = None
            schema_arg = args_schema
            description = None
        else:
            coroutine = None
            func = dec_func
            schema_arg = args_schema
            description = None

        if infer_schema or args_schema is not None:
            return GigaStructuredTool.from_function(
                func,
                coroutine,
                name=tool_name,
                description=description,
                return_direct=return_direct,
                args_schema=schema_arg,
                infer_schema=infer_schema,
                response_format=response_format,
                parse_docstring=parse_docstring,
                error_on_invalid_docstring=error_on_invalid_docstring,
                return_schema=return_schema,
                few_shot_examples=few_shot_examples,
                **kwargs,
            )

        if getattr(dec_func, "__doc__", None) is None:
            raise ValueError(
                "Function must have a docstring if "
                "description not provided and infer_schema is False."
            )
        return GigaTool(
            name=tool_name,
            func=func,
            description=f"{tool_name} tool",
            return_direct=return_direct,
            coroutine=coroutine,
            response_format=response_format,
            return_schema=return_schema,
            few_shot_examples=few_shot_examples,
            **kwargs,
        )

    return _make_tool


def giga_tool(
    name_or_callable: Union[str, Callable[..., Any], None] = None,
    runnable: Optional[Runnable] = None,
    *args: Any,
    return_direct: bool = False,
    args_schema: Union[type[BaseModel], dict[str, Any], None] = None,
    infer_schema: bool = True,
    response_format: Literal["content", "content_and_artifact"] = "content",
    parse_docstring: bool = False,
    error_on_invalid_docstring: bool = True,
    return_schema: Optional[type[BaseModel]] = None,
    few_shot_examples: FewShotExamples = None,
    **kwargs: Any,
) -> Union[BaseTool, Callable[[Union[Callable[..., Any], Runnable]], BaseTool]]:
    """Create a GigaChat tool from a function or Runnable (same as @tool).

    Supports:
    - @giga_tool
    - @giga_tool("name")
    - @giga_tool(return_direct=True, few_shot_examples=[...])
    - giga_tool("name", runnable=my_runnable)

    Extra options vs LangChain's tool: return_schema, few_shot_examples.
    """
    factory_kw: dict[str, Any] = {
        "return_direct": return_direct,
        "args_schema": args_schema,
        "infer_schema": infer_schema,
        "response_format": response_format,
        "parse_docstring": parse_docstring,
        "error_on_invalid_docstring": error_on_invalid_docstring,
        "return_schema": return_schema,
        "few_shot_examples": few_shot_examples,
        **kwargs,
    }
    if (
        len(args) == 1
        and isinstance(args[0], Runnable)
        and isinstance(name_or_callable, str)
    ):
        return _create_giga_tool_factory(name_or_callable, **factory_kw)(args[0])
    if args:
        raise ValueError("Too many arguments for tool decorator.")

    if runnable is not None:
        if not isinstance(name_or_callable, str):
            raise ValueError("When passing runnable, name must be a string.")
        return _create_giga_tool_factory(name_or_callable, **factory_kw)(runnable)

    if name_or_callable is not None:
        if callable(name_or_callable) and hasattr(name_or_callable, "__name__"):
            return _create_giga_tool_factory(name_or_callable.__name__, **factory_kw)(
                name_or_callable
            )
        if isinstance(name_or_callable, str):
            return _create_giga_tool_factory(name_or_callable, **factory_kw)
        raise ValueError(
            f"First argument must be a string or a callable with __name__. "
            f"Got {type(name_or_callable)}."
        )

    def _partial(f: Union[Callable[..., Any], Runnable]) -> BaseTool:
        name = (
            f.get_name() if isinstance(f, Runnable) else getattr(f, "__name__", "tool")
        )
        return _create_giga_tool_factory(name, **factory_kw)(f)

    return _partial
