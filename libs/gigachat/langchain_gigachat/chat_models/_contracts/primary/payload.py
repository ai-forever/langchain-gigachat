"""Primary request payload construction boundary."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

import gigachat.models as gm
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from langchain_gigachat.chat_models._contracts.primary.messages import (
    convert_messages,
)
from langchain_gigachat.chat_models._contracts.primary.types import (
    RequestDefaults,
    ToolBinding,
)

_MODEL_OPTION_DEFAULTS = (
    "temperature",
    "top_p",
    "max_tokens",
    "repetition_penalty",
    "update_interval",
)
_MODEL_OPTION_KEYS = frozenset(
    {
        *_MODEL_OPTION_DEFAULTS,
        "model_options",
        "reasoning_effort",
        "response_format",
    }
)
_LOCAL_CONTROL_KEYS = frozenset(
    {
        "function_call",
        "functions",
        "messages",
        "tool_choice",
        "use_api_v2",
    }
)
_RANKER_FIELDS = frozenset(gm.ChatRankerOptions.model_fields)


def _copy_mapping_or_model(value: Any, *, field_name: str) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(exclude_none=True, by_alias=True)
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    raise ValueError(f"{field_name} must be a mapping or Pydantic model.")


def _model_options(
    *,
    defaults: RequestDefaults,
    invocation_kwargs: Mapping[str, Any],
) -> gm.ChatModelOptions | None:
    native = invocation_kwargs.get("model_options")
    if native is None:
        options: dict[str, Any] = {}
    else:
        options = _copy_mapping_or_model(native, field_name="model_options")

    for field_name in _MODEL_OPTION_DEFAULTS:
        if options.get(field_name) is not None:
            continue
        invocation_value = invocation_kwargs.get(field_name)
        default_value = getattr(defaults, field_name)
        value = invocation_value if invocation_value is not None else default_value
        if value is not None:
            options[field_name] = copy.deepcopy(value)

    if options.get("reasoning") is None:
        effort = invocation_kwargs.get("reasoning_effort")
        if effort is None:
            effort = defaults.reasoning_effort
        if effort is not None:
            options["reasoning"] = {"effort": effort}

    if options.get("response_format") is None:
        response_format = invocation_kwargs.get("response_format")
        if response_format is not None:
            if isinstance(response_format, BaseModel):
                response_format = response_format.model_dump(
                    exclude_none=True,
                    by_alias=True,
                )
            else:
                response_format = copy.deepcopy(response_format)
            options["response_format"] = response_format

    if not options:
        return None
    return gm.ChatModelOptions.model_validate(options)


def _ranker_options(value: Any) -> gm.ChatRankerOptions | None:
    if value is None:
        return None
    ranker = _copy_mapping_or_model(value, field_name="ranker_options")
    unsupported = set(ranker) - _RANKER_FIELDS
    if unsupported:
        fields = ", ".join(sorted(unsupported))
        raise ValueError(f"Unsupported primary ranker option(s): {fields}.")
    return gm.ChatRankerOptions.model_validate(ranker)


def _copy_tool_binding(binding: ToolBinding) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if binding.tools is not None:
        values["tools"] = [tool.model_copy(deep=True) for tool in binding.tools]
    if binding.tool_config is not None:
        values["tool_config"] = binding.tool_config.model_copy(deep=True)
    return values


def build_payload(
    messages: Sequence[BaseMessage],
    *,
    defaults: RequestDefaults,
    invocation_kwargs: Mapping[str, Any],
    cached_uploads: Mapping[str, str],
    tool_binding: ToolBinding | None = None,
) -> gm.ChatCompletionRequest:
    """Build a primary SDK request without mutating caller-owned inputs."""
    kwargs = copy.deepcopy(dict(invocation_kwargs))
    consumed_keys = set(_LOCAL_CONTROL_KEYS) | set(_MODEL_OPTION_KEYS)
    consumed_keys.add("function_ranker")
    consumed_keys.add("profanity_check")
    if tool_binding is not None:
        consumed_keys.update(tool_binding.consumed_keys)

    payload_values = {
        key: value for key, value in kwargs.items() if key not in consumed_keys
    }
    payload_values["messages"] = convert_messages(
        messages,
        cached_uploads=cached_uploads,
    )

    model = invocation_kwargs.get("model")
    if model is None:
        model = defaults.model
    if model is not None:
        payload_values["model"] = model

    flags = invocation_kwargs.get("flags")
    if flags is None:
        flags = defaults.flags
    if flags is not None:
        payload_values["flags"] = copy.deepcopy(list(flags))

    options = _model_options(
        defaults=defaults,
        invocation_kwargs=invocation_kwargs,
    )
    if options is not None:
        payload_values["model_options"] = options

    disable_filter = invocation_kwargs.get("disable_filter")
    if disable_filter is None:
        profanity_check = invocation_kwargs.get("profanity_check")
        if profanity_check is None:
            profanity_check = defaults.profanity_check
        if profanity_check is not None:
            disable_filter = not profanity_check
    if disable_filter is not None:
        payload_values["disable_filter"] = disable_filter

    explicit_ranker = invocation_kwargs.get("ranker_options")
    if explicit_ranker is None:
        explicit_ranker = invocation_kwargs.get("function_ranker")
    if explicit_ranker is None:
        explicit_ranker = defaults.function_ranker
    ranker_options = _ranker_options(explicit_ranker)
    if ranker_options is not None:
        payload_values["ranker_options"] = ranker_options

    if tool_binding is not None:
        payload_values.update(_copy_tool_binding(tool_binding))

    return gm.ChatCompletionRequest.model_validate(payload_values)
