"""Shared content conversion helpers for the primary chat contract."""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import UsageMetadata
from langchain_core.messages.tool import (
    InvalidToolCall,
    ToolCall,
    invalid_tool_call,
    tool_call,
)
from pydantic import BaseModel

_TERMINAL_TOOL_STATUSES = {
    "complete",
    "completed",
    "done",
    "error",
    "failed",
    "failure",
    "success",
}
_FAILED_TOOL_STATUSES = {"error", "failed", "failure"}


@dataclass(frozen=True)
class ParsedFunctionArguments:
    """The normalized and raw forms of provider-generated function arguments."""

    value: dict[str, Any] | None
    raw: str
    error: str | None


def provider_dict(value: Any) -> dict[str, Any]:
    """Return a detached provider mapping from an SDK model or mapping."""
    if isinstance(value, BaseModel):
        return value.model_dump(exclude_none=True, by_alias=True)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(
        "Primary provider values must be SDK models or mappings; "
        f"got {type(value).__name__}"
    )


def unknown_provider_fields(
    value: Any,
    known_fields: Collection[str],
) -> dict[str, Any]:
    """Return provider fields that are not part of the known contract."""
    return {
        key: item
        for key, item in provider_dict(value).items()
        if key not in known_fields
    }


def create_usage_metadata(usage: Any | None) -> UsageMetadata | None:
    """Convert provider usage into LangChain's standard token metadata."""
    if usage is None:
        return None

    raw = provider_dict(usage)
    input_tokens = int(raw.get("input_tokens") or 0)
    output_tokens = int(raw.get("output_tokens") or 0)
    total_tokens = raw.get("total_tokens")
    result = UsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=int(
            total_tokens if total_tokens is not None else input_tokens + output_tokens
        ),
    )
    details_value = raw.get("input_tokens_details")
    details = provider_dict(details_value) if details_value is not None else {}
    if details.get("cached_tokens") is not None:
        result["input_token_details"] = {"cache_read": int(details["cached_tokens"])}
    return result


def citation_annotations(inline_data: Any | None) -> list[dict[str, Any]]:
    """Convert provider sources into standard citation annotations."""
    if inline_data is None:
        return []

    sources = provider_dict(inline_data).get("sources")
    if not isinstance(sources, Mapping):
        return []

    annotations: list[dict[str, Any]] = []
    for source_id, source_value in sources.items():
        source = provider_dict(source_value)
        annotation: dict[str, Any] = {
            "type": "citation",
            "id": str(source_id),
        }
        for field in ("url", "title"):
            if source.get(field) is not None:
                annotation[field] = source[field]
        provider_data = {
            key: value for key, value in source.items() if key not in {"url", "title"}
        }
        if provider_data:
            annotation["extras"] = {"provider_data": provider_data}
        annotations.append(annotation)
    return annotations


def inline_extras(inline_data: Any | None) -> dict[str, Any]:
    """Return non-citation inline provider data for a content block."""
    if inline_data is None:
        return {}

    raw = provider_dict(inline_data)
    return {
        key: value
        for key, value in raw.items()
        if key != "sources" and value is not None
    }


def convert_provider_file(
    file_value: Any,
    *,
    index: int | str | None = None,
) -> dict[str, Any]:
    """Convert a provider file reference into a standard content block."""
    raw = provider_dict(file_value)
    mime = raw.get("mime")
    if isinstance(mime, str) and mime.startswith("image/"):
        block_type = "image"
    elif isinstance(mime, str) and mime.startswith("audio/"):
        block_type = "audio"
    elif isinstance(mime, str) and mime.startswith("video/"):
        block_type = "video"
    else:
        block_type = "file"

    block: dict[str, Any] = {
        "type": block_type,
        "file_id": str(raw.get("id", "")),
    }
    if mime is not None:
        block["mime_type"] = mime
    resolved_index = index if index is not None else raw.get("index")
    if isinstance(resolved_index, (int, str)):
        block["index"] = resolved_index

    extras = {
        key: value
        for key, value in raw.items()
        if key not in {"id", "index", "mime"} and value is not None
    }
    if extras:
        block["extras"] = extras
    return block


def convert_text_content(
    text: str,
    *,
    role: str,
    inline_data: Any | None = None,
    provider_data: Mapping[str, Any] | None = None,
    index: int | str | None = None,
) -> dict[str, Any]:
    """Convert provider text or reasoning into a standard content block."""
    if role == "reasoning":
        block: dict[str, Any] = {
            "type": "reasoning",
            "reasoning": text,
        }
    else:
        block = {"type": "text", "text": text}
        annotations = citation_annotations(inline_data)
        if annotations:
            block["annotations"] = annotations

    if index is not None:
        block["index"] = index

    extras: dict[str, Any] = {}
    if inline := inline_extras(inline_data):
        extras["inline_data"] = inline
    if provider_data:
        extras["provider_data"] = dict(provider_data)
    if extras:
        block["extras"] = extras
    return block


def convert_tool_execution(
    execution: Any,
    *,
    tool_call_id: str,
    event_name: str | None = None,
    index: int | str | None = None,
    streaming: bool = False,
) -> list[dict[str, Any]]:
    """Convert provider-managed tool state without owning stream index state."""
    raw = provider_dict(execution)
    status = str(raw.get("status") or "").lower()
    terminal = status in _TERMINAL_TOOL_STATUSES or event_name in {
        "response.tool.completed",
        "response.tool.failed",
    }
    if terminal:
        failed = status in _FAILED_TOOL_STATUSES or event_name == "response.tool.failed"
        block: dict[str, Any] = {
            "type": "server_tool_result",
            "id": f"{tool_call_id}:result",
            "tool_call_id": tool_call_id,
            "status": "error" if failed else "success",
            "extras": {"provider_tool_execution": raw},
        }
        if raw.get("output") is not None:
            block["output"] = raw["output"]
        if index is not None:
            block["index"] = index
        return [block]

    arguments = raw.get("arguments", raw.get("args"))
    if arguments is None:
        arguments = {}
    if streaming:
        arguments = json_fragment(arguments)
    block = {
        "type": "server_tool_call_chunk" if streaming else "server_tool_call",
        "id": tool_call_id,
        "name": raw.get("name") or "unknown",
        "args": arguments,
        "extras": {"provider_tool_execution": raw},
    }
    if index is not None:
        block["index"] = index
    return [block]


def json_fragment(value: Any) -> str:
    """Serialize a provider value as a compact stream argument fragment."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def parse_function_arguments(
    arguments: Any,
    *,
    function_name: str | None = None,
) -> ParsedFunctionArguments:
    """Parse model-generated function arguments without raising on bad JSON."""
    if isinstance(arguments, dict):
        return ParsedFunctionArguments(
            value=dict(arguments),
            raw=json_fragment(arguments),
            error=None,
        )

    raw = arguments if isinstance(arguments, str) else json_fragment(arguments)
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as error:
            return ParsedFunctionArguments(
                value=None,
                raw=raw,
                error=(
                    f"Function {function_name!r} arguments contain invalid JSON: "
                    f"{error}"
                ),
            )
        if isinstance(parsed, dict):
            return ParsedFunctionArguments(value=parsed, raw=raw, error=None)
        parsed_type = type(parsed).__name__
    else:
        parsed_type = type(arguments).__name__

    return ParsedFunctionArguments(
        value=None,
        raw=raw,
        error=(
            f"Function {function_name!r} arguments must be a JSON object; "
            f"got {parsed_type}"
        ),
    )


def convert_function_call(
    function_call_value: Any,
    *,
    tool_call_id: str | None,
) -> tuple[ToolCall | None, InvalidToolCall | None]:
    """Convert one client function call into a valid or invalid standard call."""
    raw = provider_dict(function_call_value)
    name_value = raw.get("name")
    name = str(name_value) if name_value is not None else None
    parsed = parse_function_arguments(
        raw.get("arguments"),
        function_name=name,
    )
    if parsed.value is not None:
        return (
            tool_call(
                name=name or "",
                args=parsed.value,
                id=tool_call_id,
            ),
            None,
        )
    return (
        None,
        invalid_tool_call(
            name=name,
            args=parsed.raw,
            id=tool_call_id,
            error=parsed.error,
        ),
    )
