"""Tests for primary-contract tool normalization."""

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


def test_groups_client_functions_and_preserves_metadata() -> None:
    functions = [_function()]
    original = copy.deepcopy(functions)

    binding = primary.build_tool_binding(
        functions=functions,
        tools=[],
        function_call=None,
    )

    assert functions == original
    assert binding.tool_config is None
    assert binding.tools is not None
    assert len(binding.tools) == 1
    grouped = binding.tools[0].functions
    assert grouped is not None
    assert grouped.specifications is not None
    assert len(grouped.specifications) == 1
    specification = grouped.specifications[0]
    assert specification.name == "weather"
    assert specification.few_shot_examples is not None
    assert specification.few_shot_examples[0].request == "Moscow"
    assert specification.return_parameters == functions[0]["return_parameters"]


def test_groups_function_tools_with_functions_argument() -> None:
    binding = primary.build_tool_binding(
        functions=[_function("first")],
        tools=[{"type": "function", "function": _function("second")}],
        function_call=None,
    )

    assert binding.tools is not None
    specifications = binding.tools[0].functions
    assert specifications is not None
    assert specifications.specifications is not None
    assert [item.name for item in specifications.specifications] == [
        "first",
        "second",
    ]


def test_normalizes_canonical_and_type_shorthand_builtins() -> None:
    tools: list[dict[str, Any]] = [
        {"code_interpreter": {}},
        {
            "type": "web_search",
            "indexes": ["news"],
            "flags": ["fresh"],
        },
    ]
    original = copy.deepcopy(tools)

    binding = primary.build_tool_binding(
        functions=[],
        tools=tools,
        function_call=None,
    )

    assert tools == original
    assert binding.tools is not None
    assert binding.tools[0].code_interpreter == {}
    web_search = binding.tools[1].web_search
    assert web_search is not None
    assert web_search.indexes == ["news"]
    assert web_search.flags == ["fresh"]


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        ("auto", {"mode": "auto"}),
        ("weather", {"mode": "forced", "function_name": "weather"}),
        (
            {"name": "web_search"},
            {"mode": "forced", "tool_name": "web_search"},
        ),
    ],
)
def test_maps_tool_choice(choice: Any, expected: dict[str, Any]) -> None:
    binding = primary.build_tool_binding(
        functions=[_function()],
        tools=[{"web_search": {}}],
        function_call=choice,
    )

    assert binding.tool_config is not None
    assert binding.tool_config.model_dump(exclude_none=True) == expected


def test_true_choice_forces_first_available_tool() -> None:
    binding = primary.build_tool_binding(
        functions=[],
        tools=[{"image_generate": {}}, {"web_search": {}}],
        function_call=True,
    )

    assert binding.tool_config == gm.ChatToolConfig(
        mode="forced",
        tool_name="image_generate",
    )


def test_none_choice_omits_tools() -> None:
    binding = primary.build_tool_binding(
        functions=[_function()],
        tools=[{"web_search": {}}],
        function_call="none",
    )

    assert binding.tools is None
    assert binding.tool_config is None


def test_matching_explicit_tool_config_is_preserved_without_mutation() -> None:
    explicit = gm.ChatToolConfig(mode="forced", function_name="weather")

    binding = primary.build_tool_binding(
        functions=[_function()],
        tools=[],
        function_call={"name": "weather"},
        explicit_tool_config=explicit,
    )

    assert binding.tool_config is not explicit
    assert binding.tool_config == explicit
    assert binding.consumed_keys == frozenset(
        {"functions", "tools", "function_call", "tool_config"}
    )


def test_explicit_tool_config_conflict_raises() -> None:
    with pytest.raises(ValueError, match="Conflicting"):
        primary.build_tool_binding(
            functions=[_function()],
            tools=[{"web_search": {}}],
            function_call="weather",
            explicit_tool_config={"mode": "forced", "tool_name": "web_search"},
        )


def test_explicit_tool_config_must_reference_available_tool() -> None:
    with pytest.raises(ValueError, match="function_name 'missing' was not found"):
        primary.build_tool_binding(
            functions=[_function()],
            tools=[],
            function_call=None,
            explicit_tool_config={
                "mode": "forced",
                "function_name": "missing",
            },
        )


@pytest.mark.parametrize(
    ("function_call", "explicit_tool_config"),
    [
        ("auto", None),
        (None, {"mode": "auto"}),
    ],
)
def test_tool_config_requires_an_available_tool(
    function_call: Any,
    explicit_tool_config: Any,
) -> None:
    with pytest.raises(
        ValueError,
        match="tool_config requires at least one available tool",
    ):
        primary.build_tool_binding(
            functions=[],
            tools=[],
            function_call=function_call,
            explicit_tool_config=explicit_tool_config,
        )


@pytest.mark.parametrize(
    ("tool_config", "match"),
    [
        ({"mode": "auto", "function_name": "weather"}, "mode='auto'.*cannot"),
        ({"mode": "forced"}, "mode='forced'.*exactly one"),
        (
            {
                "mode": "forced",
                "function_name": "weather",
                "tool_name": "web_search",
            },
            "mode='forced'.*exactly one",
        ),
        ({"mode": "disabled"}, "Unsupported tool_config mode"),
        ({"mode": "forced", "function_name": ""}, "non-empty string"),
    ],
)
def test_explicit_tool_config_validates_mode_target_semantics(
    tool_config: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        primary.build_tool_binding(
            functions=[_function()],
            tools=[{"web_search": {}}],
            function_call=None,
            explicit_tool_config=tool_config,
        )


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        (False, None),
        ("", "Unrecognized function/tool choice"),
        ({}, "must contain a function name"),
    ],
)
def test_falsey_tool_choices_are_handled_explicitly(
    choice: Any,
    expected: str | None,
) -> None:
    if expected is None:
        binding = primary.build_tool_binding(
            functions=[_function()],
            tools=[],
            function_call=choice,
        )
        assert binding.tool_config is None
        return

    with pytest.raises(ValueError, match=expected):
        primary.build_tool_binding(
            functions=[_function()],
            tools=[],
            function_call=choice,
        )


@pytest.mark.parametrize(
    ("choice", "message"),
    [
        ("any", "does not have a confirmed"),
        ("missing", "was not found"),
        (True, "requires at least one"),
    ],
)
def test_invalid_tool_choices_raise(choice: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        primary.build_tool_binding(
            functions=[],
            tools=[],
            function_call=choice,
        )


def test_duplicate_function_names_raise() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        primary.build_tool_binding(
            functions=[_function(), _function()],
            tools=[],
            function_call=None,
        )


def test_duplicate_builtin_tools_raise() -> None:
    with pytest.raises(ValueError, match="must not be repeated"):
        primary.build_tool_binding(
            functions=[],
            tools=[{"web_search": {}}, {"type": "web_search"}],
            function_call=None,
        )


def test_client_function_name_must_not_collide_with_builtin() -> None:
    with pytest.raises(ValueError, match="must not collide"):
        primary.build_tool_binding(
            functions=[_function("web_search")],
            tools=[{"web_search": {}}],
            function_call=None,
        )


@pytest.mark.parametrize(
    ("tool", "message"),
    [
        (
            {"web_search": {}, "description": "extra"},
            "must contain only their tool name",
        ),
        ({"web_search": None}, "must be a mapping"),
        (
            {"type": "web_search", "code_interpreter": {}},
            "exactly one built-in",
        ),
    ],
)
def test_invalid_builtin_configuration_raises(
    tool: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        primary.build_tool_binding(
            functions=[],
            tools=[tool],
            function_call=None,
        )


@pytest.mark.parametrize("tool", [{"type": "computer_use"}, {"type": []}])
def test_unknown_tool_mapping_raises(tool: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="Unsupported tool mapping"):
        primary.build_tool_binding(
            functions=[],
            tools=[tool],
            function_call=None,
        )


@pytest.mark.parametrize(
    "tool",
    [
        {"web_search": {}},
        {"type": "code_interpreter"},
    ],
)
def test_legacy_tool_conversion_rejects_primary_builtins(
    tool: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="use_api_v2=True"):
        convert_to_gigachat_tool(tool)


def test_build_tool_binding_accepts_route_neutral_client_function() -> None:
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
