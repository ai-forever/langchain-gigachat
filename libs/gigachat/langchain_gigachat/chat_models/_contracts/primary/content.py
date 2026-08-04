"""Shared content conversion helpers for the primary chat contract."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages.ai import UsageMetadata
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
_SERVER_TOOL_ID_FIELDS = ("call_id", "tool_call_id", "id")


@dataclass(frozen=True)
class ParsedFunctionArguments:
    """The normalized and raw forms of provider-generated function arguments."""

    value: dict[str, Any] | None
    raw: str
    error: str | None


ToolExecutionSource = Literal["part", "message", "response"]
ToolExecutionCoordinates = tuple[ToolExecutionSource, int | None, int | None]


@dataclass(frozen=True)
class ToolExecutionCandidate:
    """One provider server-tool execution together with its source identity."""

    source: ToolExecutionSource
    order: int
    message_index: int | None
    part_index: int | None
    execution: Any
    normalized_execution: dict[str, Any]
    execution_id: str | None

    @property
    def coordinates(self) -> ToolExecutionCoordinates:
        return self.source, self.message_index, self.part_index


@dataclass(frozen=True)
class ResolvedToolExecution:
    """One logical execution after source mirrors have been correlated."""

    candidate: ToolExecutionCandidate
    tool_call_id: str
    execution_id: str | None
    mirrored_sources: tuple[ToolExecutionCoordinates, ...]


def has_client_function_call(message: Mapping[str, Any]) -> bool:
    """Return whether one provider message contains a client function call."""
    if message.get("function_call") is not None:
        return True
    content = message.get("content")
    if content is None:
        return False
    if isinstance(content, (str, bytes)) or not isinstance(content, Sequence):
        raise TypeError("Primary provider messages.content must be a sequence")
    return any(provider_dict(part).get("function_call") is not None for part in content)


def normalized_tool_execution(execution: Any) -> dict[str, Any]:
    """Return one semantic payload for mirror and repeat correlation."""
    payload = {
        key: copy.deepcopy(value)
        for key, value in provider_dict(execution).items()
        if key not in {*_SERVER_TOOL_ID_FIELDS, "index"}
    }
    status = str(payload.get("status") or "").lower()
    if status in {"complete", "completed", "done", "success"}:
        payload["status"] = "success"
    elif status in _FAILED_TOOL_STATUSES:
        payload["status"] = "failed"
    return payload


def collect_tool_execution_candidates(
    messages: Iterable[Any],
    *,
    response_tool_execution: Any = None,
) -> list[ToolExecutionCandidate]:
    """Collect part, message, and response observations in provider order."""
    candidates: list[ToolExecutionCandidate] = []

    def append_candidate(
        execution: Any,
        *,
        source: ToolExecutionSource,
        message_index: int | None,
        part_index: int | None,
    ) -> None:
        candidates.append(
            ToolExecutionCandidate(
                source=source,
                order=len(candidates),
                message_index=message_index,
                part_index=part_index,
                execution=execution,
                normalized_execution=normalized_tool_execution(execution),
                execution_id=server_tool_execution_id(execution),
            )
        )

    for message_index, message_value in enumerate(messages):
        message = provider_dict(message_value)
        content = message.get("content")
        if content is None:
            parts: Sequence[Any] = ()
        elif isinstance(content, (str, bytes)) or not isinstance(content, Sequence):
            raise TypeError("Primary provider messages.content must be a sequence")
        else:
            parts = content
        for part_index, part_value in enumerate(parts):
            part = provider_dict(part_value)
            if part.get("tool_execution") is not None:
                append_candidate(
                    part["tool_execution"],
                    source="part",
                    message_index=message_index,
                    part_index=part_index,
                )
        if message.get("tool_execution") is not None:
            append_candidate(
                message["tool_execution"],
                source="message",
                message_index=message_index,
                part_index=None,
            )

    if response_tool_execution is not None:
        append_candidate(
            response_tool_execution,
            source="response",
            message_index=None,
            part_index=None,
        )
    return candidates


def _sources_can_mirror(
    left: ToolExecutionCandidate,
    right: ToolExecutionCandidate,
) -> bool:
    if left.execution_id is not None and left.execution_id == right.execution_id:
        return True
    if "response" in {left.source, right.source}:
        return True
    return {left.source, right.source} == {
        "part",
        "message",
    } and left.message_index == right.message_index


def resolve_tool_execution_candidates(
    candidates: Iterable[ToolExecutionCandidate],
    *,
    id_factory: Callable[[], str] | None = None,
) -> list[ResolvedToolExecution]:
    """Correlate source-level mirrors without treating tools_state_id as identity."""
    values = list(candidates)
    payload_by_execution_id: dict[str, dict[str, Any]] = {}
    for candidate in values:
        if candidate.execution_id is None:
            continue
        previous = payload_by_execution_id.setdefault(
            candidate.execution_id,
            candidate.normalized_execution,
        )
        if previous != candidate.normalized_execution:
            raise ValueError(
                "Primary GigaChat server tools use the same provider identity "
                f"{candidate.execution_id!r} with conflicting payloads."
            )

    def execution_ids(group: Iterable[ToolExecutionCandidate]) -> set[str]:
        return {
            candidate.execution_id
            for candidate in group
            if candidate.execution_id is not None
        }

    def can_join(
        group: list[ToolExecutionCandidate],
        candidate: ToolExecutionCandidate,
    ) -> bool:
        combined = [*group, candidate]
        return (
            group[0].normalized_execution == candidate.normalized_execution
            and any(_sources_can_mirror(member, candidate) for member in group)
            and len(execution_ids(combined)) <= 1
        )

    logical_groups: list[list[ToolExecutionCandidate]] = []
    for candidate in values:
        matches = [group for group in logical_groups if can_join(group, candidate)]
        if len(matches) > 1:
            if candidate.execution_id is None:
                raise ValueError(
                    "Primary GigaChat unidentified server tool mirror matches "
                    "multiple distinct logical executions and cannot be "
                    "correlated safely."
                )
            raise ValueError(
                "Primary GigaChat server tool mirror matches multiple logical "
                "executions and cannot be correlated safely."
            )
        if matches:
            matches[0].append(candidate)
        else:
            logical_groups.append([candidate])

    source_priority = {"part": 0, "message": 1, "response": 2}
    resolved: list[ResolvedToolExecution] = []
    next_local_sequence = 0
    for group in logical_groups:
        explicit_ids = execution_ids(group)
        execution_id = next(iter(explicit_ids)) if explicit_ids else None
        if execution_id is None:
            if id_factory is None:
                tool_call_id = f"lc_primary-server-tool-{next_local_sequence}"
                next_local_sequence += 1
            else:
                tool_call_id = id_factory()
        else:
            tool_call_id = execution_id

        representative = min(
            group,
            key=lambda candidate: (
                source_priority[candidate.source],
                candidate.order,
            ),
        )
        resolved.append(
            ResolvedToolExecution(
                candidate=representative,
                tool_call_id=tool_call_id,
                execution_id=execution_id,
                mirrored_sources=tuple(candidate.coordinates for candidate in group),
            )
        )
    return resolved


def provider_dict(value: Any) -> dict[str, Any]:
    """Return a detached provider mapping from an SDK model or mapping."""
    if isinstance(value, BaseModel):
        return value.model_dump(exclude_none=True, by_alias=True)
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    raise TypeError(
        "Primary provider values must be SDK models or mappings; "
        f"got {type(value).__name__}"
    )


def resolve_reasoning_value(value: Any) -> Any | None:
    """Resolve provider reasoning aliases without silently discarding conflicts."""
    raw = provider_dict(value)
    reasoning = raw.get("reasoning")
    reasoning_content = raw.get("reasoning_content")
    if (
        reasoning is not None
        and reasoning_content is not None
        and reasoning != reasoning_content
    ):
        raise ValueError(
            "Primary provider response contains conflicting reasoning and "
            "reasoning_content values."
        )
    return reasoning if reasoning is not None else reasoning_content


def validate_provider_id(value: Any, *, field: str) -> str | None:
    """Validate one optional provider identity without rewriting it."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Primary provider {field} must be a non-empty string.")
    return value


def request_id_from_headers(headers: Mapping[str, Any]) -> str | None:
    """Resolve a case-insensitive provider request ID from response headers."""
    request_id: str | None = None
    for key, value in headers.items():
        if not isinstance(key, str) or key.lower() != "x-request-id":
            continue
        candidate = validate_provider_id(value, field="x-request-id")
        if candidate is None:
            continue
        if request_id is not None and request_id != candidate:
            raise ValueError(
                "Primary provider response contains conflicting x-request-id "
                "header values."
            )
        request_id = candidate
    return request_id


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
    details_value = raw.get("input_tokens_details")
    details = provider_dict(details_value) if details_value is not None else {}
    input_tokens_value = raw.get("input_tokens")
    if input_tokens_value is None:
        input_tokens_value = details.get("prompt_tokens")
    input_tokens = int(input_tokens_value or 0)
    output_tokens = int(raw.get("output_tokens") or 0)
    total_tokens = raw.get("total_tokens")
    result = UsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=int(
            total_tokens if total_tokens is not None else input_tokens + output_tokens
        ),
    )
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

    file_id = validate_provider_id(raw.get("id"), field="file ID")
    if file_id is None:
        raise ValueError("Primary provider file ID must be a non-empty string.")

    block: dict[str, Any] = {
        "type": block_type,
        "file_id": file_id,
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
        extras["provider_data"] = provider_dict(provider_data)
    if extras:
        block["extras"] = extras
    return block


def convert_reasoning_value(
    reasoning_value: Any,
    *,
    index: int | str | None = None,
) -> dict[str, Any]:
    """Convert one provider reasoning value into a standard content block."""
    if isinstance(reasoning_value, (BaseModel, Mapping)):
        reasoning_data = provider_dict(reasoning_value)
        reasoning_text = reasoning_data.get("reasoning")
        fallback_text = reasoning_data.get("text")
        if (
            reasoning_text is not None
            and fallback_text is not None
            and reasoning_text != fallback_text
        ):
            raise ValueError(
                "Primary provider reasoning contains conflicting reasoning and "
                "text values."
            )
        text = reasoning_text if reasoning_text is not None else fallback_text or ""
        extras = {
            key: value
            for key, value in reasoning_data.items()
            if key not in {"index", "reasoning", "text"}
        }
        resolved_index = reasoning_data.get("index", index)
    else:
        text = str(reasoning_value)
        extras = {}
        resolved_index = index

    return convert_text_content(
        str(text),
        role="reasoning",
        provider_data=extras,
        index=resolved_index,
    )


def reasoning_content(
    blocks: Iterable[str | Mapping[str, Any]],
) -> str | None:
    """Collect standard reasoning blocks for the compatibility response field."""
    fragments = [
        value
        for block in blocks
        if isinstance(block, Mapping) and block.get("type") == "reasoning"
        if isinstance(value := block.get("reasoning"), str)
    ]
    return "".join(fragments) if fragments else None


def server_tool_execution_id(execution: Any) -> str | None:
    """Resolve one unambiguous provider-managed tool identity."""
    raw = provider_dict(execution)
    resolved: str | None = None
    for field in _SERVER_TOOL_ID_FIELDS:
        candidate = validate_provider_id(
            raw.get(field),
            field=f"server tool {field}",
        )
        if candidate is None:
            continue
        if resolved is not None and candidate != resolved:
            raise ValueError(
                "Primary provider server tool identity aliases contain "
                "conflicting values."
            )
        resolved = candidate
    return resolved


def convert_tool_execution(
    execution: Any,
    *,
    tool_call_id: str,
    event_name: str | None = None,
    index: int | str | None = None,
    streaming: bool = False,
) -> list[dict[str, Any]]:
    """Convert provider-managed tool state without owning stream index state."""
    validated_tool_call_id = validate_provider_id(
        tool_call_id,
        field="server tool ID",
    )
    if validated_tool_call_id is None:
        raise ValueError("Primary provider server tool ID must be a non-empty string.")
    tool_call_id = validated_tool_call_id
    raw = provider_dict(execution)
    status = str(raw.get("status") or "").lower()
    terminal = status in _TERMINAL_TOOL_STATUSES or event_name in {
        "response.tool.completed",
        "response.tool.failed",
    }
    if terminal:
        # The official SDK stream fixture reports status=success together with
        # censored=true and a completion-level finish_reason=error. Censorship
        # therefore remains provider execution metadata; it does not rewrite a
        # technically successful execution into a failed server tool result.
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
    name = name_value if isinstance(name_value, str) else None
    parsed = parse_function_arguments(
        raw.get("arguments"),
        function_name=name,
    )
    if name is None or not name.strip():
        return (
            None,
            invalid_tool_call(
                name=name,
                args=parsed.raw,
                id=tool_call_id,
                error="Function call name must be a non-empty string.",
            ),
        )
    if parsed.value is not None:
        return (
            tool_call(
                name=name,
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
