"""Tests for route-neutral public tool normalization."""

import copy
from typing import Any

import pytest

from langchain_gigachat.chat_models._contracts import primary
from langchain_gigachat.utils.function_calling import (
    is_primary_builtin_tool,
    normalize_tool_for_binding,
)


def _function(name: str = "weather") -> dict[str, Any]:
    return {
        "name": name,
        "description": f"Call {name}",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
        },
        "few_shot_examples": [{"request": "Moscow", "params": {"city": "Moscow"}}],
        "return_parameters": {
            "type": "object",
            "properties": {"temperature": {"type": "number"}},
        },
    }


@pytest.mark.parametrize(
    "tool",
    [
        {"type": "web_search"},
        {"web_search": {}},
        {"type": "code_interpreter"},
        {"code_interpreter": {}},
    ],
)
def test_preserves_primary_builtins_without_mutation(
    tool: dict[str, Any],
) -> None:
    original = copy.deepcopy(tool)

    normalized = normalize_tool_for_binding(tool)

    assert normalized == original
    assert normalized is not tool
    assert tool == original
    assert is_primary_builtin_tool(normalized)


def test_contract_for_task_05_route_integration() -> None:
    normalized = normalize_tool_for_binding({"type": "web_search"})

    assert normalized == {"type": "web_search"}

    binding = primary.build_tool_binding(
        functions=[],
        tools=[normalized],
        function_call=None,
    )
    assert binding.tools is not None
    assert binding.tools[0].web_search is not None


def test_deep_copies_nested_builtin_configuration() -> None:
    tool = {"type": "web_search", "indexes": ["news"]}

    normalized = normalize_tool_for_binding(tool)
    normalized["indexes"].append("web")

    assert tool == {"type": "web_search", "indexes": ["news"]}


def test_client_function_uses_existing_conversion_without_mutation() -> None:
    tool = _function()
    original = copy.deepcopy(tool)

    normalized = normalize_tool_for_binding(tool)

    assert tool == original
    assert normalized is not tool
    assert normalized["type"] == "function"
    assert normalized["function"]["name"] == "weather"
    assert normalized["function"]["few_shot_examples"] == original["few_shot_examples"]
    assert normalized["function"]["return_parameters"] == original["return_parameters"]


def test_deep_copies_preformatted_client_function() -> None:
    tool: dict[str, Any] = {"type": "function", "function": _function()}

    normalized = normalize_tool_for_binding(tool)
    normalized["function"]["parameters"]["properties"]["city"]["type"] = "number"

    assert tool["function"]["parameters"]["properties"]["city"]["type"] == "string"


@pytest.mark.parametrize(
    "tool",
    [
        {"type": []},
        {"type": "function", "function": {"name": "web_search"}},
        {
            "name": "web_search",
            "description": "Client function",
            "parameters": {"type": "object", "properties": {}},
        },
        {"web_search": {}, "description": "not canonical"},
    ],
)
def test_builtin_detection_avoids_false_positives(tool: dict[str, Any]) -> None:
    assert not is_primary_builtin_tool(tool)


@pytest.mark.parametrize(
    ("tool", "message"),
    [
        (
            {"web_search": {}, "description": "extra"},
            "must contain only their tool name",
        ),
        (
            {"web_search": []},
            "must be a mapping",
        ),
        (
            {"web_search": {}, "code_interpreter": {}},
            "exactly one built-in",
        ),
        (
            {"type": "web_search", "code_interpreter": {}},
            "exactly one built-in",
        ),
    ],
)
def test_rejects_malformed_builtin_mappings(
    tool: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_tool_for_binding(tool)
