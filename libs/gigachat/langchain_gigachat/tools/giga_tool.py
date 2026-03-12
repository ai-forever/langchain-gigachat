"""GigaChat tool helpers built on top of LangChain's standard ``@tool``."""

from __future__ import annotations

import inspect
import textwrap
from typing import (
    Annotated,
    Any,
    Awaitable,
    Callable,
    Literal,
    Optional,
    Union,
)

from langchain_core.tools import (
    FILTERED_ARGS,
    BaseTool,
    StructuredTool,
    Tool,
    _get_runnable_config_param,
    create_schema_from_function,
    tool as lc_tool,
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


giga_tool = lc_tool
giga_tool.__doc__ = (
    "Alias of ``langchain_core.tools.tool``.\n\n"
    "GigaChat-specific metadata must be passed via ``extras`` "
    "(for example ``extras={'return_schema': MyModel}``)."
)
