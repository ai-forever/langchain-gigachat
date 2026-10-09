"""Primary client-function and provider-built-in tool mapping."""

from __future__ import annotations

import copy
from typing import Any

import gigachat.models as gm
import pytest
from langchain_core.runnables import RunnableBinding
from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from typing_extensions import TypedDict

from langchain_gigachat.chat_models._contracts import primary
from langchain_gigachat.chat_models.gigachat import GigaChat
from langchain_gigachat.utils.function_calling import (
    convert_to_gigachat_tool,
    normalize_tool_for_binding,
)


def _function(name: str = "weather") -> dict[str, Any]:
    return {
        "name": name,
        "description": f"Call {name}",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
        "few_shot_examples": [{"request": "Moscow", "params": {"city": "Moscow"}}],
        "return_parameters": {
            "type": "object",
            "properties": {"temperature": {"type": "number"}},
        },
    }


def test_groups_client_functions_without_mutating_input() -> None:
    functions = [_function("first")]
    tools = [{"type": "function", "function": _function("second")}]
    original = copy.deepcopy((functions, tools))

    binding = primary.build_tool_binding(
        functions=functions,
        tools=tools,
        function_call=None,
    )

    assert (functions, tools) == original
    assert binding.tools is not None
    grouped = binding.tools[0].functions
    assert grouped is not None
    assert grouped.specifications is not None
    assert [item.name for item in grouped.specifications] == ["first", "second"]
    assert grouped.specifications[0].few_shot_examples is not None
    assert (
        grouped.specifications[0].return_parameters == functions[0]["return_parameters"]
    )


def test_normalizes_provider_builtins_through_sdk_models() -> None:
    tools: list[dict[str, Any]] = [
        {"code_interpreter": {}},
        {"type": "web_search", "indexes": ["news"], "flags": ["fresh"]},
    ]

    binding = primary.build_tool_binding(
        functions=[],
        tools=tools,
        function_call=None,
    )

    assert binding.tools is not None
    assert binding.tools[0] == gm.ChatTool(code_interpreter={})
    assert binding.tools[1] == gm.ChatTool.model_validate(
        {"web_search": {"indexes": ["news"], "flags": ["fresh"]}}
    )


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        ("auto", gm.ChatToolConfig(mode="auto")),
        ("any", gm.ChatToolConfig(mode="any")),
        ("required", gm.ChatToolConfig(mode="any")),
        (
            "weather",
            gm.ChatToolConfig(mode="forced", function_name="weather"),
        ),
        (
            {"name": "web_search"},
            gm.ChatToolConfig(mode="forced", tool_name="web_search"),
        ),
    ],
)
def test_maps_supported_tool_choices(
    choice: Any,
    expected: gm.ChatToolConfig,
) -> None:
    binding = primary.build_tool_binding(
        functions=[_function()],
        tools=[{"web_search": {}}],
        function_call=choice,
    )

    assert binding.tool_config == expected


def test_none_choice_omits_tools() -> None:
    binding = primary.build_tool_binding(
        functions=[_function()],
        tools=[{"web_search": {}}],
        function_call="none",
    )

    assert binding.tools is None
    assert binding.tool_config is None


def test_explicit_sdk_tool_config_is_preserved() -> None:
    config = gm.ChatToolConfig(mode="forced", function_name="weather")

    binding = primary.build_tool_binding(
        functions=[_function()],
        tools=[],
        function_call=None,
        explicit_tool_config=config,
    )

    assert binding.tool_config == config
    assert binding.tool_config is not config


@pytest.mark.parametrize("choice", ["missing"])
def test_unsupported_or_unknown_tool_choice_fails(choice: str) -> None:
    with pytest.raises(ValueError):
        primary.build_tool_binding(
            functions=[_function()],
            tools=[],
            function_call=choice,
        )


@pytest.mark.parametrize(
    "tool",
    [
        {"web_search": {}},
        {"type": "code_interpreter"},
    ],
)
def test_legacy_converter_rejects_primary_builtins(
    tool: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="use_api_v2=True"):
        convert_to_gigachat_tool(tool)


def test_public_normalizer_keeps_route_neutral_client_functions() -> None:
    normalized = normalize_tool_for_binding(_function())

    binding = primary.build_tool_binding(
        functions=[],
        tools=[normalized],
        function_call={"type": "function", "function": {"name": "weather"}},
    )

    assert binding.tools is not None
    assert binding.tool_config == gm.ChatToolConfig(
        mode="forced",
        function_name="weather",
    )


class UnionArguments(BaseModel):
    """Find a room by capacity."""

    capacity: int | str


class UnionTypedArguments(TypedDict):
    """Find a room by capacity."""

    capacity: int | str


def union_function(capacity: int | str) -> str:
    """Find a room by capacity."""
    return str(capacity)


@pytest.mark.parametrize("tool", [UnionArguments, UnionTypedArguments, union_function])
@pytest.mark.parametrize("choice", ["any", "required"])
def test_public_v2_binding_accepts_union_schema_and_any_choice(
    tool: Any, choice: str
) -> None:
    model = GigaChat(use_api_v2=True)
    bound = model.bind_tools([tool], tool_choice=choice)
    assert isinstance(bound, RunnableBinding)
    payload = model._build_primary_payload([], bound.kwargs)
    dumped = payload.model_dump(exclude_none=True, by_alias=True)
    assert dumped["tool_config"] == {"mode": "any"}
    schema = dumped["tools"][0]["functions"]["specifications"][0]["parameters"]
    assert schema["properties"]["capacity"]["anyOf"] == [
        {"type": "integer"},
        {"type": "string"},
    ]
    assert "type" not in schema["properties"]["capacity"]


def test_tool_config_any_checks_eligible_function_names() -> None:
    with pytest.raises(ValueError, match="functions_names_any.*missing"):
        primary.build_tool_binding(
            functions=[_function()],
            tools=[],
            function_call=None,
            explicit_tool_config={"mode": "any", "functions_names_any": ["missing"]},
        )


def test_additional_fields_tools_and_choice_are_normalized() -> None:
    model = GigaChat(use_api_v2=True)
    payload = model._build_primary_payload(
        [],
        {
            "additional_fields": {
                "tools": [{"type": "function", "function": _function()}],
                "function_call": "required",
            },
        },
    )
    assert payload.tool_config == gm.ChatToolConfig(mode="any")
    assert payload.tools is not None
    assert payload.tools[0].functions is not None


def test_additional_fields_preserve_native_sdk_functions() -> None:
    model = GigaChat(use_api_v2=True)
    native_tools = [{"functions": {"specifications": [_function()], "future": True}}]
    payload = model._build_primary_payload(
        [],
        {
            "additional_fields": {
                "tools": native_tools,
                "tool_config": {"mode": "any"},
            },
        },
    )
    assert payload.model_dump(exclude_none=True)["tools"] == native_tools


def test_additional_fields_preserve_provider_tools_outside_public_catalog() -> None:
    from gigachat.api.chat_completions import _build_request_json

    model = GigaChat(use_api_v2=True)
    payload = model._build_primary_payload(
        [],
        {
            "additional_fields": {"tools": [{"memory": {"scope": "assistant"}}]},
            "tool_config": {"mode": "forced", "tool_name": "memory"},
        },
    )
    wire = _build_request_json(payload)
    assert wire["tools"] == [{"memory": {"scope": "assistant"}}]
    assert wire["tool_config"] == {"mode": "forced", "tool_name": "memory"}
    with pytest.raises(ValueError, match="Unsupported tool mapping"):
        model._build_primary_payload([], {"tools": [{"memory": {}}]})


def test_explicit_tools_override_provider_tools_from_extras() -> None:
    payload = GigaChat(use_api_v2=True)._build_primary_payload(
        [],
        {
            "additional_fields": {"tools": [{"memory": {}}]},
            "tools": [{"type": "function", "function": _function()}],
        },
    )
    assert payload.tools is not None and len(payload.tools) == 1
    assert payload.tools[0].functions is not None


def test_none_tools_and_functions_are_treated_as_absent() -> None:
    payload = GigaChat(use_api_v2=True)._build_primary_payload(
        [], {"tools": None, "functions": None}
    )
    assert payload.tools is None


def test_v2_bind_functions_accepts_union_and_any() -> None:
    model = GigaChat(use_api_v2=True)
    bound = model.bind_functions([UnionArguments], function_call="required")
    assert isinstance(bound, RunnableBinding)
    payload = model._build_primary_payload([], bound.kwargs)
    assert payload.tool_config == gm.ChatToolConfig(mode="any")


def test_v2_base_tool_preserves_provider_extensions_and_union() -> None:
    tool = StructuredTool.from_function(
        union_function,
        extras={
            "return_schema": {"type": "string"},
            "few_shot_examples": [{"request": "30 places", "params": {"capacity": 30}}],
        },
    )
    function = normalize_tool_for_binding(tool, use_api_v2=True)["function"]
    assert function["parameters"]["properties"]["capacity"]["anyOf"]
    assert function["return_parameters"] == {"type": "string"}
    assert function["few_shot_examples"][0]["params"] == {"capacity": 30}


def test_legacy_binding_still_rejects_union() -> None:
    from langchain_gigachat.utils.function_calling import IncorrectSchemaException

    with pytest.raises(IncorrectSchemaException):
        GigaChat(use_api_v2=False).bind_tools([UnionArguments])
