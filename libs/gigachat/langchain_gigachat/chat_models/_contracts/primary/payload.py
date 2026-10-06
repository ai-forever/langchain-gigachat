"""Primary request payload construction boundary."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence, cast

import gigachat.models as gm
from langchain_core.messages import BaseMessage
from langchain_core.utils.pydantic import is_basemodel_subclass
from pydantic import BaseModel

from langchain_gigachat.chat_models._contracts.primary.messages import (
    convert_messages,
)
from langchain_gigachat.chat_models._contracts.primary.types import (
    RequestDefaults,
    ToolBinding,
)

_MODEL_OPTION_DEFAULTS = (
    "parallel_tool_calls",
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
        "reasoning",
        "reasoning_effort",
        "reasoning_max_tokens",
        "response_format",
    }
)
_LOCAL_CONTROL_KEYS = frozenset(
    {
        "function_call",
        "functions",
        "messages",
        "strict",
        "tool_choice",
        "use_api_v2",
    }
)
_JSON_SCHEMA_TYPES = frozenset(
    {"array", "boolean", "integer", "null", "number", "object", "string"}
)


def _is_json_schema_type(value: Any) -> bool:
    if isinstance(value, str):
        return value in _JSON_SCHEMA_TYPES
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return bool(value) and all(
            isinstance(item, str) and item in _JSON_SCHEMA_TYPES for item in value
        )
    return False


def normalize_response_format(
    response_format: Any,
    *,
    strict: bool | None = None,
) -> gm.ChatResponseFormat | None:
    """Translate LangChain response-format forms to the primary SDK model."""
    if response_format is None:
        if strict is not None:
            raise ValueError("strict is supported only together with response_format.")
        return None
    candidate: dict[str, Any]
    if isinstance(response_format, type) and is_basemodel_subclass(response_format):
        candidate = {
            "type": "json_schema",
            "schema": cast(type[BaseModel], response_format).model_json_schema(),
        }
    elif isinstance(
        response_format,
        (gm.ChatResponseFormat, gm.JsonSchemaResponseFormat),
    ):
        candidate = response_format.model_dump(exclude_none=True, by_alias=True)
    elif isinstance(response_format, Mapping):
        candidate = copy.deepcopy(dict(response_format))
        nested_json_schema = candidate.get("json_schema")
        if (
            candidate.get("type") == "json_schema"
            and isinstance(nested_json_schema, Mapping)
            and "schema" in nested_json_schema
        ):
            candidate = {
                "type": "json_schema",
                "schema": copy.deepcopy(nested_json_schema["schema"]),
                **(
                    {"strict": nested_json_schema["strict"]}
                    if "strict" in nested_json_schema
                    else {}
                ),
            }
        elif candidate.get("type") == "json_schema" and "json_schema" in candidate:
            raise ValueError(
                "Nested response_format 'json_schema' requires a 'schema' field."
            )
        elif "type" not in candidate or _is_json_schema_type(candidate["type"]):
            candidate = {
                "type": "json_schema",
                "schema": candidate,
            }
    else:
        raise TypeError(
            "response_format must be a ChatResponseFormat, "
            "JsonSchemaResponseFormat, Pydantic BaseModel class, mapping, or None."
        )

    if candidate.get("type") == "json_schema" and candidate.get("schema") is None:
        raise ValueError(
            "response_format type 'json_schema' requires a 'schema' field. "
            "Pass schema={'type': 'object'} for an arbitrary JSON object."
        )

    if strict is not None:
        if candidate.get("type") != "json_schema":
            raise ValueError(
                "strict is supported only with a JSON Schema response_format."
            )
        existing_strict = candidate.get("strict")
        if existing_strict is not None and existing_strict != strict:
            raise ValueError(
                "response_format already defines strict="
                f"{existing_strict}, but strict={strict} was also provided."
            )
        candidate["strict"] = strict

    try:
        return gm.ChatResponseFormat.model_validate(candidate)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid primary response_format: {exc}") from exc


def _copy_mapping_or_model(value: Any, *, field_name: str) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(exclude_unset=True, by_alias=True)
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
        if field_name in options:
            continue
        invocation_value = invocation_kwargs.get(field_name)
        default_value = getattr(defaults, field_name)
        value = invocation_value if invocation_value is not None else default_value
        if value is not None:
            options[field_name] = copy.deepcopy(value)

    if "reasoning" not in options:
        if "reasoning" in invocation_kwargs:
            reasoning = invocation_kwargs["reasoning"]
            options["reasoning"] = (
                _copy_mapping_or_model(reasoning, field_name="reasoning")
                if reasoning is not None
                else None
            )
        else:
            reasoning_options = {}
            for field_name, alias in (
                ("effort", "reasoning_effort"),
                ("max_tokens", "reasoning_max_tokens"),
            ):
                value = invocation_kwargs.get(alias)
                if value is None:
                    value = getattr(defaults, alias)
                if value is not None:
                    reasoning_options[field_name] = value
            if reasoning_options:
                options["reasoning"] = reasoning_options

    response_format = options.get(
        "response_format", invocation_kwargs.get("response_format")
    )
    normalized_response_format = normalize_response_format(
        response_format,
        strict=invocation_kwargs.get("strict"),
    )
    if normalized_response_format is not None:
        options["response_format"] = normalized_response_format

    if not options:
        return None
    return gm.ChatModelOptions.model_validate(options)


def _ranker_options(value: Any) -> gm.ChatRankerOptions | None:
    if value is None:
        return None
    ranker = _copy_mapping_or_model(value, field_name="ranker_options")
    return gm.ChatRankerOptions.model_validate(ranker)


def _normalize_primary_storage(
    value: Any,
) -> gm.ChatStorage | bool | None:
    """Normalize only storage forms defined by the primary SDK contract."""
    if value is None or isinstance(value, bool):
        return value

    if isinstance(value, gm.ChatStorage):
        storage = value.model_dump(exclude_none=True, by_alias=True)
    elif isinstance(value, Mapping):
        storage = copy.deepcopy(dict(value))
    else:
        raise ValueError(
            "storage must be None, a bool, gm.ChatStorage, or a compatible mapping."
        )
    return gm.ChatStorage.model_validate(storage)


def _copy_tool_binding(binding: ToolBinding) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if binding.tools is not None:
        values["tools"] = binding.tools
    if binding.tool_config is not None:
        values["tool_config"] = binding.tool_config
    return values


def merge_request_kwargs(invocation_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Merge request extras below explicit invocation fields before normalization."""
    kwargs = copy.deepcopy(dict(invocation_kwargs))
    extra_fields = kwargs.pop("additional_fields", None)
    if extra_fields is None:
        return kwargs
    if not isinstance(extra_fields, Mapping):
        raise ValueError("additional_fields must be a mapping.")
    merged = {
        **copy.deepcopy(dict(extra_fields)),
        **{
            key: value
            for key, value in kwargs.items()
            if value is not None or key not in extra_fields
        },
    }
    # SDK serialization excludes None on regular request fields, but preserves
    # JSON null inside the additional_fields escape hatch.
    null_extras = {
        key: value
        for key, value in extra_fields.items()
        if value is None and merged.get(key) is None
    }
    if null_extras:
        if "additional_fields" not in gm.ChatCompletionRequest.model_fields:
            raise ValueError(
                "JSON null in additional_fields requires an updated GigaChat SDK."
            )
        merged["additional_fields"] = null_extras
    return merged


def build_payload(
    messages: Sequence[BaseMessage],
    *,
    defaults: RequestDefaults,
    invocation_kwargs: Mapping[str, Any],
    cached_uploads: Mapping[str, str],
    tool_binding: ToolBinding | None = None,
) -> gm.ChatCompletionRequest:
    """Build a primary SDK request without mutating caller-owned inputs."""
    invocation_kwargs = merge_request_kwargs(invocation_kwargs)
    kwargs = dict(invocation_kwargs)
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

    storage: gm.ChatStorage | bool | None = None
    if "storage" in kwargs:
        storage = _normalize_primary_storage(kwargs["storage"])
        payload_values["storage"] = storage

    assistant_id = kwargs.get("assistant_id")

    model = invocation_kwargs.get("model")
    has_thread_id = (
        isinstance(storage, gm.ChatStorage) and storage.thread_id is not None
    )
    is_stateful = assistant_id is not None or has_thread_id
    if model is None and not is_stateful:
        model = defaults.model
    if model is not None:
        payload_values["model"] = model

    flags = invocation_kwargs.get("flags")
    if flags is None:
        flags = defaults.flags
    if flags is not None:
        payload_values["flags"] = copy.deepcopy(flags)

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
