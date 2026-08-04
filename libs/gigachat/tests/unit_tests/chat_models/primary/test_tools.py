"""Primary client-function and provider-built-in tool mapping."""

from __future__ import annotations

import copy
from typing import Any

import gigachat.models as gm
import pytest

from langchain_gigachat.chat_models._contracts import primary
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


@pytest.mark.parametrize("choice", ["any", "missing"])
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
