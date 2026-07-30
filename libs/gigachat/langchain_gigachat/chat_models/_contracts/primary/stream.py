"""Convert primary named stream events into LangChain generation chunks."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any
from uuid import uuid4

import gigachat.models as gm
from langchain_core.messages import AIMessageChunk, ToolCallChunk
from langchain_core.messages.ai import UsageMetadata
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.outputs import ChatGenerationChunk

from langchain_gigachat.chat_models._contracts.primary.types import StreamState

_KNOWN_EVENTS = frozenset(
    {
        "response.message.delta",
        "response.message.done",
        "response.tool.started",
        "response.tool.delta",
        "response.tool.completed",
        "response.tool.failed",
        "response.error",
    }
)
_EVENT_FIELDS = frozenset(
    {
        "additional_data",
        "created_at",
        "event",
        "finish_reason",
        "logprobs",
        "message_id",
        "messages",
        "model",
        "thread_id",
        "tool_execution",
        "tools_state_id",
        "usage",
        "x_headers",
    }
)
_MESSAGE_FIELDS = frozenset(
    {
        "content",
        "finish_reason",
        "function_call",
        "inline_data",
        "logprobs",
        "message_id",
        "role",
        "tool_execution",
        "tools_state_id",
    }
)
_CONTENT_FIELDS = frozenset(
    {
        "files",
        "function_call",
        "function_result",
        "index",
        "inline_data",
        "reasoning",
        "reasoning_content",
        "text",
        "tool_execution",
    }
)


def _as_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(exclude_none=True, by_alias=True)
        if isinstance(dumped, dict):
            return dumped
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(
        "Primary stream events must be SDK models or mappings; "
        f"got {type(value).__name__}"
    )


def _optional_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    return _as_dict(value)


def _as_dict_list(value: Any, *, field: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"Primary stream field {field!r} must be a sequence")
    return [_as_dict(item) for item in value]


def _take_block_index(state: StreamState, explicit: Any = None) -> int | str:
    if isinstance(explicit, (int, str)):
        if isinstance(explicit, int):
            state.next_block_index = max(state.next_block_index, explicit + 1)
        return explicit
    index = state.next_block_index
    state.next_block_index += 1
    return index


def _request_id(x_headers: Mapping[str, Any]) -> str | None:
    for key, value in x_headers.items():
        if key.lower() == "x-request-id" and value is not None:
            return str(value)
    return None


def _usage_metadata(value: Any) -> UsageMetadata | None:
    usage = _optional_dict(value)
    if not usage:
        return None

    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    total_tokens = usage.get("total_tokens")
    normalized = UsageMetadata(
        input_tokens=int(input_tokens or 0),
        output_tokens=int(output_tokens or 0),
        total_tokens=int(
            total_tokens
            if total_tokens is not None
            else (input_tokens or 0) + (output_tokens or 0)
        ),
    )
    details = _optional_dict(usage.get("input_tokens_details"))
    if details.get("cached_tokens") is not None:
        normalized["input_token_details"] = {
            "cache_read": int(details["cached_tokens"])
        }
    return normalized


def _json_fragment(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _file_block(
    file_value: Any,
    *,
    state: StreamState,
) -> dict[str, Any]:
    file_data = _as_dict(file_value)
    mime = file_data.get("mime")
    block_type = "file"
    if isinstance(mime, str):
        if mime.startswith("image/"):
            block_type = "image"
        elif mime.startswith("audio/"):
            block_type = "audio"

    block: dict[str, Any] = {
        "type": block_type,
        "file_id": str(file_data.get("id", "")),
        "index": _take_block_index(state, file_data.get("index")),
    }
    if mime is not None:
        block["mime_type"] = mime
    extras = {
        key: value
        for key, value in file_data.items()
        if key not in {"id", "index", "mime"}
    }
    if extras:
        block["extras"] = extras
    return block


def _citation_annotations(inline_data: Mapping[str, Any]) -> list[dict[str, Any]]:
    sources = inline_data.get("sources")
    if not isinstance(sources, Mapping):
        return []

    annotations: list[dict[str, Any]] = []
    for source_id, source_value in sources.items():
        source = _optional_dict(source_value)
        annotation: dict[str, Any] = {
            "type": "citation",
            "id": str(source_id),
        }
        for field in ("title", "url"):
            if source.get(field) is not None:
                annotation[field] = source[field]
        extras = {
            key: value for key, value in source.items() if key not in {"title", "url"}
        }
        if extras:
            annotation["extras"] = extras
        annotations.append(annotation)
    return annotations


def _reasoning_block(
    reasoning: Any,
    *,
    state: StreamState,
    explicit_index: Any = None,
) -> dict[str, Any]:
    if isinstance(reasoning, Mapping) or hasattr(reasoning, "model_dump"):
        reasoning_data = _as_dict(reasoning)
        text = reasoning_data.get("reasoning", reasoning_data.get("text", ""))
        extras = {
            key: value
            for key, value in reasoning_data.items()
            if key not in {"index", "reasoning", "text"}
        }
        explicit_index = reasoning_data.get("index", explicit_index)
    else:
        text = str(reasoning)
        extras = {}

    block: dict[str, Any] = {
        "type": "reasoning",
        "reasoning": text,
        "index": _take_block_index(state, explicit_index),
    }
    if extras:
        block["extras"] = extras
    return block


def _tool_call_chunk(
    function_call_value: Any,
    *,
    incoming_tools_state_id: str | None,
    state: StreamState,
) -> ToolCallChunk:
    function_call = _as_dict(function_call_value)
    call_id = incoming_tools_state_id or state.tools_state_id
    if call_id is None:
        call_id = f"{state.message_id}:tool" if state.message_id else f"tool-{uuid4()}"

    if state.tools_state_id is not None and state.tools_state_id != call_id:
        raise ValueError(
            "Primary streaming supports one client tool call per message; "
            f"received tool states {state.tools_state_id!r} and {call_id!r}"
        )

    is_first_fragment = state.tools_state_id is None
    if is_first_fragment:
        state.tools_state_id = call_id
        explicit_index = function_call.get("index")
        if isinstance(explicit_index, int):
            index = explicit_index
            state.next_block_index = max(
                state.next_block_index,
                explicit_index + 1,
            )
        else:
            index = state.next_block_index
            state.next_block_index += 1
    else:
        explicit_index = function_call.get("index")
        if isinstance(explicit_index, int):
            index = explicit_index
        else:
            index = max(0, state.next_block_index - 1)

    arguments = function_call.get("arguments")
    return tool_call_chunk(
        name=function_call.get("name") if is_first_fragment else None,
        args=_json_fragment(arguments) if arguments is not None else "",
        id=call_id,
        index=index,
    )


def _tool_execution_block(
    execution_value: Any,
    *,
    event_name: str | None,
    incoming_tools_state_id: str | None,
    state: StreamState,
) -> dict[str, Any]:
    execution = _as_dict(execution_value)
    call_id = incoming_tools_state_id or state.tools_state_id
    if call_id is None:
        call_id = f"{state.message_id}:tool" if state.message_id else f"tool-{uuid4()}"
    if state.tools_state_id is None:
        state.tools_state_id = call_id

    status = str(execution.get("status", "")).lower()
    is_completed = event_name in {"response.tool.completed", "response.tool.failed"}
    is_completed = is_completed or status in {
        "completed",
        "done",
        "error",
        "failed",
        "failure",
        "success",
    }
    if is_completed:
        failed = event_name == "response.tool.failed" or status in {
            "error",
            "failed",
            "failure",
        }
        block: dict[str, Any] = {
            "type": "server_tool_result",
            "id": f"{call_id}:result",
            "tool_call_id": call_id,
            "status": "error" if failed else "success",
            "index": _take_block_index(state, execution.get("index")),
            "extras": {"provider_tool_execution": execution},
        }
        if execution.get("output") is not None:
            block["output"] = execution["output"]
        return block

    arguments = execution.get("arguments", execution.get("args"))
    block = {
        "type": "server_tool_call_chunk",
        "id": call_id,
        "name": execution.get("name", ""),
        "args": _json_fragment(arguments) if arguments is not None else "",
        "index": _take_block_index(state, execution.get("index")),
        "extras": {"provider_tool_execution": execution},
    }
    return block


def _convert_content_part(
    part_value: Any,
    *,
    event_name: str | None,
    incoming_tools_state_id: str | None,
    state: StreamState,
) -> tuple[list[str | dict[str, Any]], list[ToolCallChunk]]:
    part = _as_dict(part_value)
    content: list[str | dict[str, Any]] = []
    tool_calls: list[ToolCallChunk] = []
    inline_data = _optional_dict(part.get("inline_data"))
    extra = {key: value for key, value in part.items() if key not in _CONTENT_FIELDS}

    if part.get("text") is not None:
        annotations = _citation_annotations(inline_data)
        if (
            annotations
            or extra
            or any(key in inline_data for key in ("images", "widgets"))
        ):
            block: dict[str, Any] = {
                "type": "text",
                "text": str(part["text"]),
                "index": _take_block_index(state, part.get("index")),
            }
            if annotations:
                block["annotations"] = annotations
            provider_extras = dict(extra)
            provider_extras.update(
                {key: value for key, value in inline_data.items() if key != "sources"}
            )
            if provider_extras:
                block["extras"] = provider_extras
            content.append(block)
        else:
            content.append(str(part["text"]))

    for file_value in part.get("files") or []:
        content.append(_file_block(file_value, state=state))

    if part.get("function_call") is not None:
        tool_calls.append(
            _tool_call_chunk(
                part["function_call"],
                incoming_tools_state_id=incoming_tools_state_id,
                state=state,
            )
        )

    if part.get("tool_execution") is not None:
        content.append(
            _tool_execution_block(
                part["tool_execution"],
                event_name=event_name,
                incoming_tools_state_id=incoming_tools_state_id,
                state=state,
            )
        )

    if part.get("function_result") is not None:
        content.append(
            {
                "type": "non_standard",
                "value": {
                    "function_result": _as_dict(part["function_result"]),
                },
            }
        )

    reasoning = part.get("reasoning", part.get("reasoning_content"))
    if reasoning is not None:
        content.append(
            _reasoning_block(
                reasoning,
                state=state,
                explicit_index=part.get("index"),
            )
        )

    if not content and not tool_calls and part:
        content.append({"type": "non_standard", "value": part})
    return content, tool_calls


def _convert_messages(
    messages_value: Any,
    *,
    event_name: str | None,
    incoming_tools_state_id: str | None,
    state: StreamState,
) -> tuple[
    str | list[str | dict[str, Any]],
    list[ToolCallChunk],
    list[dict[str, Any]],
]:
    content: list[str | dict[str, Any]] = []
    tool_calls: list[ToolCallChunk] = []
    message_metadata: list[dict[str, Any]] = []

    for message in _as_dict_list(messages_value, field="messages"):
        message_tool_state = message.get("tools_state_id")
        tool_state_id = (
            str(message_tool_state)
            if message_tool_state is not None
            else incoming_tools_state_id
        )
        for part in _as_dict_list(message.get("content"), field="messages.content"):
            part_content, part_tool_calls = _convert_content_part(
                part,
                event_name=event_name,
                incoming_tools_state_id=tool_state_id,
                state=state,
            )
            content.extend(part_content)
            tool_calls.extend(part_tool_calls)

        if message.get("function_call") is not None:
            tool_calls.append(
                _tool_call_chunk(
                    message["function_call"],
                    incoming_tools_state_id=tool_state_id,
                    state=state,
                )
            )
        if message.get("tool_execution") is not None:
            content.append(
                _tool_execution_block(
                    message["tool_execution"],
                    event_name=event_name,
                    incoming_tools_state_id=tool_state_id,
                    state=state,
                )
            )

        reasoning = message.get("reasoning", message.get("reasoning_content"))
        if reasoning is not None:
            content.append(_reasoning_block(reasoning, state=state))

        metadata = {
            key: value for key, value in message.items() if key not in _MESSAGE_FIELDS
        }
        if message.get("role") not in (None, "assistant"):
            metadata["role"] = message["role"]
        if metadata:
            message_metadata.append(metadata)

    if tool_calls and len({call["id"] for call in tool_calls}) > 1:
        raise ValueError("Primary streaming supports one client tool call per message")
    if content and all(isinstance(item, str) for item in content):
        normalized_content: str | list[str | dict[str, Any]] = "".join(
            item for item in content if isinstance(item, str)
        )
    else:
        normalized_content = content
    return normalized_content, tool_calls, message_metadata


def _response_metadata(
    event: Mapping[str, Any],
    *,
    event_name: str | None,
    provider_message_id: str | None,
    provider_tools_state_id: str | None,
    first_chunk: bool,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if event_name is not None:
        metadata["event"] = event_name

    if first_chunk:
        for field in ("created_at", "model", "thread_id"):
            if event.get(field) is not None:
                metadata[field] = event[field]
        if event.get("x_headers") is not None:
            metadata["x_headers"] = event["x_headers"]
    if provider_message_id is not None and first_chunk:
        metadata["message_id"] = provider_message_id
    if provider_tools_state_id is not None and first_chunk:
        metadata["tools_state_id"] = provider_tools_state_id

    for field in ("additional_data", "finish_reason", "logprobs"):
        if event.get(field) is not None:
            metadata[field] = event[field]
    if event.get("tool_execution") is not None:
        metadata["tool_execution"] = event["tool_execution"]

    provider_fields = {
        key: value for key, value in event.items() if key not in _EVENT_FIELDS
    }
    if provider_fields:
        metadata["provider_fields"] = provider_fields
    if event_name is not None and event_name not in _KNOWN_EVENTS:
        metadata["raw_event"] = dict(event)
    return metadata


def convert_stream_event(
    event: gm.PrimaryChatCompletionChunk | Mapping[str, Any],
    *,
    state: StreamState,
) -> ChatGenerationChunk | None:
    """Convert one SDK named event while preserving identity and block order."""
    event_data = _as_dict(event)
    if not event_data:
        return None

    event_name_value = event_data.get("event")
    event_name = str(event_name_value) if event_name_value is not None else None
    provider_message_id_value = event_data.get("message_id")
    normalized_messages = _as_dict_list(
        event_data.get("messages"),
        field="messages",
    )
    if provider_message_id_value is None:
        provider_message_id_value = next(
            (
                message.get("message_id")
                for message in normalized_messages
                if message.get("message_id") is not None
            ),
            None,
        )
    provider_message_id = (
        str(provider_message_id_value)
        if provider_message_id_value is not None
        else None
    )
    provider_tools_state_value = event_data.get("tools_state_id")
    if provider_tools_state_value is None:
        provider_tools_state_value = next(
            (
                message.get("tools_state_id")
                for message in normalized_messages
                if message.get("tools_state_id") is not None
            ),
            None,
        )
    provider_tools_state_id = (
        str(provider_tools_state_value)
        if provider_tools_state_value is not None
        else None
    )
    x_headers = _optional_dict(event_data.get("x_headers"))

    if state.message_id is None:
        state.message_id = (
            _request_id(x_headers) or provider_message_id or f"primary-stream-{uuid4()}"
        )

    first_chunk = state.first_chunk
    content, tool_calls, message_metadata = _convert_messages(
        normalized_messages,
        event_name=event_name,
        incoming_tools_state_id=provider_tools_state_id,
        state=state,
    )
    if provider_tools_state_id is not None:
        if (
            state.tools_state_id is not None
            and state.tools_state_id != provider_tools_state_id
        ):
            raise ValueError(
                "Primary streaming supports one tool state per message; "
                f"received {state.tools_state_id!r} and {provider_tools_state_id!r}"
            )
        state.tools_state_id = provider_tools_state_id

    top_level_tool_execution = event_data.get("tool_execution")
    if top_level_tool_execution is not None:
        block = _tool_execution_block(
            top_level_tool_execution,
            event_name=event_name,
            incoming_tools_state_id=provider_tools_state_id,
            state=state,
        )
        if isinstance(content, str):
            content = [content, block] if content else [block]
        else:
            content.append(block)

    response_metadata = _response_metadata(
        event_data,
        event_name=event_name,
        provider_message_id=provider_message_id,
        provider_tools_state_id=provider_tools_state_id,
        first_chunk=first_chunk,
    )
    if message_metadata:
        response_metadata["message_metadata"] = message_metadata

    usage_metadata = _usage_metadata(event_data.get("usage"))
    generation_info = None
    if event_data.get("finish_reason") is not None:
        generation_info = {"finish_reason": event_data["finish_reason"]}

    has_payload = bool(
        content or tool_calls or response_metadata or usage_metadata or generation_info
    )
    if not has_payload:
        return None

    state.first_chunk = False
    message = AIMessageChunk(
        content=content,
        id=state.message_id,
        response_metadata=response_metadata,
        tool_call_chunks=tool_calls,
        usage_metadata=usage_metadata,
    )
    return ChatGenerationChunk(message=message, generation_info=generation_info)
