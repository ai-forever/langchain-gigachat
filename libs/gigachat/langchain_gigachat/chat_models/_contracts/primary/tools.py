"""Primary tool normalization boundary."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Optional, Sequence

import gigachat.models as gm

from langchain_gigachat.chat_models._contracts.primary.types import ToolBinding
from langchain_gigachat.utils.function_calling import PRIMARY_BUILTIN_TOOL_NAMES

_CONSUMED_KEYS = frozenset({"functions", "tools", "function_call", "tool_config"})


def _function_specification(value: Mapping[str, Any]) -> gm.ChatFunctionSpecification:
    candidate: Any = copy.deepcopy(dict(value))
    if candidate.get("type") == "function":
        candidate = candidate.get("function")
    if not isinstance(candidate, Mapping):
        raise ValueError("Function tools must contain a function mapping.")
    try:
        return gm.ChatFunctionSpecification.model_validate(dict(candidate))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid client function specification: {exc}") from exc


def _builtin_tool(value: Mapping[str, Any]) -> tuple[str, gm.ChatTool]:
    candidate = copy.deepcopy(dict(value))
    tool_type = candidate.get("type")
    if isinstance(tool_type, str) and tool_type in PRIMARY_BUILTIN_TOOL_NAMES:
        tool_name = str(tool_type)
        candidate.pop("type")
        nested_builtin_names = PRIMARY_BUILTIN_TOOL_NAMES.intersection(candidate)
        if nested_builtin_names:
            raise ValueError(
                "Each provider built-in tool mapping must configure exactly "
                "one built-in tool."
            )
        config: Any = candidate
    else:
        present = PRIMARY_BUILTIN_TOOL_NAMES.intersection(candidate)
        if len(present) != 1:
            if present:
                raise ValueError(
                    "Each provider built-in tool mapping must configure exactly "
                    "one built-in tool."
                )
            raise ValueError(
                "Unsupported tool mapping. Expected a function tool or one of: "
                f"{', '.join(sorted(PRIMARY_BUILTIN_TOOL_NAMES))}."
            )
        tool_name = next(iter(present))
        if set(candidate) != {tool_name}:
            raise ValueError(
                "Canonical provider built-in tools must contain only their tool "
                f"name; got extra fields for {tool_name!r}."
            )
        config = candidate[tool_name]

    if not isinstance(config, Mapping):
        raise ValueError(
            f"Configuration for provider built-in tool {tool_name!r} must be a mapping."
        )
    try:
        return tool_name, gm.ChatTool.model_validate({tool_name: dict(config)})
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid configuration for provider built-in tool {tool_name!r}: {exc}"
        ) from exc


def _choice_name(function_call: Any) -> Any:
    if not isinstance(function_call, Mapping):
        return function_call
    candidate = copy.deepcopy(dict(function_call))
    if isinstance(candidate.get("name"), str):
        return candidate["name"]
    nested = candidate.get("function")
    if isinstance(nested, Mapping) and isinstance(nested.get("name"), str):
        return nested["name"]
    raise ValueError(
        "Function/tool choice mappings must contain a function name in "
        "'name' or 'function.name'."
    )


def _forced_config(
    name: str,
    *,
    function_names: Sequence[str],
    builtin_names: Sequence[str],
) -> gm.ChatToolConfig:
    is_function = name in function_names
    is_builtin = name in builtin_names
    if is_function and is_builtin:
        raise ValueError(
            f"Tool choice {name!r} is ambiguous between a client function and "
            "a provider built-in tool."
        )
    if is_function:
        return gm.ChatToolConfig(mode="forced", function_name=name)
    if is_builtin:
        return gm.ChatToolConfig(mode="forced", tool_name=name)
    available = [*function_names, *builtin_names]
    rendered = ", ".join(available) if available else "<none>"
    raise ValueError(
        f"Tool choice {name!r} was not found in the available tools: {rendered}."
    )


def _derived_tool_config(
    function_call: Any,
    *,
    function_names: Sequence[str],
    builtin_names: Sequence[str],
) -> tuple[Optional[gm.ChatToolConfig], bool]:
    choice = _choice_name(function_call)
    if choice is None or choice is False:
        return None, False
    if choice is True:
        if function_names:
            return _forced_config(
                function_names[0],
                function_names=function_names,
                builtin_names=builtin_names,
            ), False
        if builtin_names:
            return _forced_config(
                builtin_names[0],
                function_names=function_names,
                builtin_names=builtin_names,
            ), False
        raise ValueError("A true tool choice requires at least one available tool.")
    if not isinstance(choice, str) or not choice:
        raise ValueError(
            "Unrecognized function/tool choice. Expected None, bool, str, or "
            "a mapping containing a function name."
        )
    if choice == "auto":
        return gm.ChatToolConfig(mode="auto"), False
    if choice == "none":
        return None, True
    if choice == "any":
        raise ValueError(
            "GigaChat API v2 does not have a confirmed tool_choice='any' "
            "semantic. Use 'auto' or select a concrete tool."
        )
    return (
        _forced_config(
            choice,
            function_names=function_names,
            builtin_names=builtin_names,
        ),
        False,
    )


def _explicit_tool_config(value: Any) -> Optional[gm.ChatToolConfig]:
    if value is None:
        return None
    if isinstance(value, gm.ChatToolConfig):
        candidate = value.model_dump(exclude_none=True, by_alias=True)
    elif isinstance(value, Mapping):
        candidate = copy.deepcopy(dict(value))
    else:
        raise TypeError("tool_config must be a ChatToolConfig or mapping.")

    unknown = set(candidate).difference({"mode", "tool_name", "function_name"})
    if unknown:
        raise ValueError(
            f"tool_config contains unsupported fields: {', '.join(sorted(unknown))}."
        )
    mode = candidate.get("mode")
    if mode not in {"auto", "forced"}:
        raise ValueError(
            f"Unsupported tool_config mode {mode!r}; expected 'auto' or 'forced'."
        )

    target_names = [
        field_name
        for field_name in ("tool_name", "function_name")
        if candidate.get(field_name) is not None
    ]
    for field_name in target_names:
        target = candidate[field_name]
        if not isinstance(target, str) or not target:
            raise ValueError(f"tool_config {field_name} must be a non-empty string.")

    if mode == "auto" and target_names:
        raise ValueError(
            "tool_config mode='auto' cannot include tool_name or function_name."
        )
    if mode == "forced" and len(target_names) != 1:
        raise ValueError(
            "tool_config mode='forced' requires exactly one of tool_name or "
            "function_name."
        )
    try:
        return gm.ChatToolConfig.model_validate(candidate)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid tool_config: {exc}") from exc


def _validate_explicit_config_targets(
    config: Optional[gm.ChatToolConfig],
    *,
    function_names: Sequence[str],
    builtin_names: Sequence[str],
) -> None:
    if config is None:
        return
    if config.function_name and config.function_name not in function_names:
        raise ValueError(
            f"tool_config function_name {config.function_name!r} was not found "
            "in the available client functions."
        )
    if config.tool_name and config.tool_name not in builtin_names:
        raise ValueError(
            f"tool_config tool_name {config.tool_name!r} was not found in the "
            "available provider built-in tools."
        )


def _config_dump(value: Optional[gm.ChatToolConfig]) -> Any:
    if value is None:
        return None
    return value.model_dump(exclude_none=True, by_alias=True)


def build_tool_binding(
    *,
    functions: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    function_call: Any,
    explicit_tool_config: Any = None,
) -> ToolBinding:
    """Normalize client and provider tools for the primary contract."""
    function_specs = [_function_specification(function) for function in functions]
    builtin_tools: list[gm.ChatTool] = []
    builtin_names: list[str] = []

    for tool in tools:
        if tool.get("type") == "function":
            function_specs.append(_function_specification(tool))
            continue
        builtin_name, builtin = _builtin_tool(tool)
        builtin_names.append(builtin_name)
        builtin_tools.append(builtin)

    function_names = [spec.name for spec in function_specs]
    if len(set(function_names)) != len(function_names):
        raise ValueError("Client function names must be unique.")
    if len(set(builtin_names)) != len(builtin_names):
        raise ValueError("Provider built-in tools must not be repeated.")
    ambiguous_names = set(function_names).intersection(builtin_names)
    if ambiguous_names:
        rendered = ", ".join(sorted(ambiguous_names))
        raise ValueError(
            "Client function names must not collide with provider built-in "
            f"tool names: {rendered}."
        )

    normalized_tools: list[gm.ChatTool] = []
    if function_specs:
        normalized_tools.append(
            gm.ChatTool(functions=gm.ChatFunctionsTool(specifications=function_specs))
        )
    normalized_tools.extend(builtin_tools)

    derived_config, omit_tools = _derived_tool_config(
        function_call,
        function_names=function_names,
        builtin_names=builtin_names,
    )
    explicit_config = _explicit_tool_config(explicit_tool_config)
    _validate_explicit_config_targets(
        explicit_config,
        function_names=function_names,
        builtin_names=builtin_names,
    )
    if explicit_config is not None and derived_config is not None:
        if _config_dump(explicit_config) != _config_dump(derived_config):
            raise ValueError(
                "Conflicting function/tool choice and explicit tool_config values."
            )
    if explicit_config is not None and omit_tools:
        raise ValueError(
            "tool choice 'none' conflicts with an explicit tool_config value."
        )

    tool_config = explicit_config if explicit_config is not None else derived_config
    if tool_config is not None and not normalized_tools:
        raise ValueError("tool_config requires at least one available tool.")
    return ToolBinding(
        tools=None if omit_tools or not normalized_tools else normalized_tools,
        tool_config=tool_config,
        consumed_keys=_CONSUMED_KEYS,
    )
