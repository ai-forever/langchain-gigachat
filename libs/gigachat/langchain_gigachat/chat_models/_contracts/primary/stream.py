"""Convert primary named stream events into LangChain generation chunks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from uuid import uuid4

import gigachat.models as gm
from langchain_core.messages import AIMessageChunk, ToolCallChunk
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.outputs import ChatGenerationChunk

from langchain_gigachat.chat_models._contracts.primary.content import (
    convert_provider_file,
    convert_text_content,
    convert_tool_execution,
    create_usage_metadata,
    json_fragment,
    provider_dict,
    reasoning_content,
    unknown_provider_fields,
)
from langchain_gigachat.chat_models._contracts.primary.types import (
    PrimaryStreamError,
    StreamState,
)

_KNOWN_EVENTS = frozenset(
    {
        "response.message.delta",
        "response.message.done",
        "response.tool.started",
        "response.tool.in_progress",
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
        "reasoning",
        "reasoning_content",
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
    try:
        return provider_dict(value)
    except TypeError as error:
        raise TypeError(
            "Primary stream events must be SDK models or mappings; "
            f"got {type(value).__name__}"
        ) from error


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


def _take_numeric_block_index(state: StreamState) -> int:
    index = state.next_block_index
    state.next_block_index += 1
    return index


def _take_text_block_index(
    state: StreamState,
    *,
    role: str,
    explicit: Any = None,
) -> int | str:
    if explicit is None and state.active_text_block_role == role:
        if state.active_text_block_index is not None:
            return state.active_text_block_index

    index = _take_block_index(state, explicit)
    state.active_text_block_index = index
    state.active_text_block_role = role
    return index


def _close_text_block(state: StreamState) -> None:
    state.active_text_block_index = None
    state.active_text_block_role = None


def _request_id(x_headers: Mapping[str, Any]) -> str | None:
    for key, value in x_headers.items():
        if key.lower() == "x-request-id" and value is not None:
            return str(value)
    return None


def _file_block(
    file_value: Any,
    *,
    state: StreamState,
) -> dict[str, Any]:
    file_data = _as_dict(file_value)
    return convert_provider_file(
        file_data,
        index=_take_block_index(state, file_data.get("index")),
    )


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

    return convert_text_content(
        str(text),
        role="reasoning",
        provider_data=extras,
        index=_take_block_index(state, explicit_index),
    )


def _tool_call_chunk(
    function_call_value: Any,
    *,
    incoming_tools_state_id: str | None,
    state: StreamState,
) -> ToolCallChunk:
    function_call = _as_dict(function_call_value)
    explicit_id = function_call.get("id")
    explicit_id = str(explicit_id) if explicit_id is not None else None
    incoming_name = function_call.get("name")
    incoming_name = str(incoming_name) if incoming_name is not None else None
    explicit_index = function_call.get("index")

    is_first_fragment = not state.client_tool_started
    if is_first_fragment:
        if not incoming_name:
            raise ValueError("First primary client tool fragment must include a name")
        call_id = explicit_id or incoming_tools_state_id or state.tools_state_id
        if explicit_index is not None and not isinstance(explicit_index, int):
            raise TypeError("Primary client tool fragment index must be an integer")
        index: int = (
            explicit_index
            if isinstance(explicit_index, int)
            else _take_numeric_block_index(state)
        )
        if isinstance(explicit_index, int):
            state.next_block_index = max(state.next_block_index, explicit_index + 1)

        state.client_tool_started = True
        state.client_tool_name = incoming_name
        state.client_tool_id = call_id
        state.client_tool_index = index
    else:
        stored_call_id = state.client_tool_id
        stored_index = state.client_tool_index
        if stored_index is None:
            raise RuntimeError("Primary client tool stream state is incomplete")
        call_id = (
            stored_call_id
            or explicit_id
            or incoming_tools_state_id
            or state.tools_state_id
        )
        if stored_call_id is None and call_id is not None:
            state.client_tool_id = call_id
        index = stored_index
        if (
            stored_call_id is not None
            and explicit_id is not None
            and explicit_id != stored_call_id
        ):
            raise ValueError(
                "Conflicting primary client tool IDs: "
                f"{stored_call_id!r} and {explicit_id!r}"
            )
        if incoming_name is not None and incoming_name != state.client_tool_name:
            raise ValueError(
                "Conflicting primary client tool names: "
                f"{state.client_tool_name!r} and {incoming_name!r}"
            )
        if explicit_index is not None and explicit_index != index:
            raise ValueError(
                "Conflicting primary client tool indexes: "
                f"{index!r} and {explicit_index!r}"
            )

    arguments = function_call.get("arguments")
    return tool_call_chunk(
        name=incoming_name if is_first_fragment else None,
        args=json_fragment(arguments) if arguments is not None else "",
        id=call_id,
        index=index,
    )


def _client_tool_identity_update(
    state: StreamState,
) -> tuple[list[ToolCallChunk], dict[str, Any]]:
    if (
        not state.client_tool_started
        or state.tools_state_id is None
        or state.client_tool_index is None
    ):
        return [], {}

    if state.client_tool_id is None:
        state.client_tool_id = state.tools_state_id
        return [
            tool_call_chunk(
                name=None,
                args="",
                id=state.tools_state_id,
                index=state.client_tool_index,
            )
        ], {}

    if (
        state.client_tool_id == state.tools_state_id
        or state.client_tool_state_mapping_emitted
    ):
        return [], {}

    state.client_tool_state_mapping_emitted = True
    return [], {
        "provider_tool_state_by_call_id": {
            state.client_tool_id: state.tools_state_id,
        }
    }


def _tool_execution_block(
    execution_value: Any,
    *,
    event_name: str | None,
    incoming_tools_state_id: str | None,
    state: StreamState,
) -> dict[str, Any]:
    execution = _as_dict(execution_value)
    call_id_value = (
        execution.get("call_id")
        or execution.get("tool_call_id")
        or execution.get("id")
        or incoming_tools_state_id
        or state.tools_state_id
    )
    call_id = (
        str(call_id_value)
        if call_id_value is not None
        else (
            f"{state.message_id}:server-tool"
            if state.message_id is not None
            else f"server-tool-{uuid4()}"
        )
    )

    status = str(execution.get("status") or "").lower()
    failed = event_name == "response.tool.failed" or status in {
        "error",
        "failed",
        "failure",
    }
    terminal = event_name in {
        "response.tool.completed",
        "response.tool.failed",
    } or status in {
        "complete",
        "completed",
        "done",
        "error",
        "failed",
        "failure",
        "success",
    }
    incoming_name_value = execution.get("name")
    incoming_name = (
        str(incoming_name_value) if incoming_name_value is not None else None
    )
    known_name = state.server_tool_names.get(call_id)
    if incoming_name:
        if known_name is None:
            state.server_tool_names[call_id] = incoming_name
        elif incoming_name != known_name:
            raise ValueError(
                f"Conflicting names for primary server tool {call_id!r}: "
                f"{known_name!r} and {incoming_name!r}"
            )

    pending_call_id = state.pending_server_tool_result_id
    if pending_call_id is not None and (pending_call_id != call_id or not terminal):
        state.pending_server_tool_result_id = None

    index_map = (
        state.server_tool_result_indexes if terminal else state.server_tool_indexes
    )
    explicit_index = execution.get("index")
    existing_index = index_map.get(call_id)
    if existing_index is None:
        if explicit_index is not None and not isinstance(explicit_index, int):
            raise TypeError("Primary server tool block index must be an integer")
        index = (
            explicit_index
            if isinstance(explicit_index, int)
            else _take_numeric_block_index(state)
        )
        if isinstance(explicit_index, int):
            state.next_block_index = max(state.next_block_index, explicit_index + 1)
        index_map[call_id] = index
    else:
        index = existing_index
        if explicit_index is not None and explicit_index != index:
            raise ValueError(
                f"Conflicting indexes for primary server tool {call_id!r}: "
                f"{index!r} and {explicit_index!r}"
            )

    block = convert_tool_execution(
        execution,
        tool_call_id=call_id,
        event_name=event_name,
        index=index,
        streaming=True,
    )[0]
    if terminal:
        state.pending_server_tool_result_id = None if failed else call_id
        return block

    if existing_index is None:
        block["name"] = incoming_name or ""
    else:
        block["name"] = incoming_name if known_name is None and incoming_name else ""
        block["extras"] = {"provider_tool_execution_updates": [execution]}
    if execution.get("arguments", execution.get("args")) is None:
        block["args"] = ""
    return block


def _pending_server_tool_result_update(
    inline_data_value: Any,
    *,
    provider_data: Mapping[str, Any],
    state: StreamState,
) -> dict[str, Any] | None:
    call_id = state.pending_server_tool_result_id
    if call_id is None:
        return None

    index = state.server_tool_result_indexes.get(call_id)
    if index is None:
        return None

    value: dict[str, Any] = {
        "inline_data": _as_dict(inline_data_value),
    }
    if provider_data:
        value["provider_data"] = dict(provider_data)
    state.pending_server_tool_result_id = None
    return {
        "type": "non_standard",
        "value": value,
        "index": index,
    }


def _convert_content_part(
    part_value: Any,
    *,
    role: str,
    message_inline_data: Any,
    event_name: str | None,
    incoming_tools_state_id: str | None,
    state: StreamState,
) -> tuple[list[dict[str, Any]], list[ToolCallChunk]]:
    part = _as_dict(part_value)
    content: list[dict[str, Any]] = []
    tool_calls: list[ToolCallChunk] = []
    inline_data_value = part.get("inline_data")
    if inline_data_value is None:
        inline_data_value = message_inline_data
    extra = unknown_provider_fields(part, _CONTENT_FIELDS)

    if part.get("text") is not None:
        content.append(
            convert_text_content(
                str(part["text"]),
                role=role,
                inline_data=inline_data_value,
                provider_data=extra,
                index=_take_text_block_index(
                    state,
                    role=role,
                    explicit=part.get("index"),
                ),
            )
        )

    if part.get("files"):
        _close_text_block(state)
    for file_value in part.get("files") or []:
        content.append(_file_block(file_value, state=state))

    if part.get("function_call") is not None:
        _close_text_block(state)
        tool_calls.append(
            _tool_call_chunk(
                part["function_call"],
                incoming_tools_state_id=incoming_tools_state_id,
                state=state,
            )
        )

    if part.get("tool_execution") is not None:
        _close_text_block(state)
        tool_execution_block = _tool_execution_block(
            part["tool_execution"],
            event_name=event_name,
            incoming_tools_state_id=incoming_tools_state_id,
            state=state,
        )
        block_extras = tool_execution_block.setdefault("extras", {})
        if inline_data_value is not None:
            block_extras["inline_data"] = _as_dict(inline_data_value)
            state.pending_server_tool_result_id = None
        if extra:
            block_extras["provider_data"] = extra
        content.append(tool_execution_block)

    if part.get("function_result") is not None:
        _close_text_block(state)
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
        _close_text_block(state)
        content.append(
            _reasoning_block(
                reasoning,
                state=state,
                explicit_index=part.get("index"),
            )
        )

    if not content and not tool_calls and (inline_data_value is not None or extra):
        _close_text_block(state)
        if inline_data_value is not None:
            result_update = _pending_server_tool_result_update(
                inline_data_value,
                provider_data=extra,
                state=state,
            )
            if result_update is not None:
                content.append(result_update)
                return content, tool_calls
        value: dict[str, Any] = dict(extra)
        if inline_data_value is not None:
            value["inline_data"] = _as_dict(inline_data_value)
        content.append({"type": "non_standard", "value": value})
    return content, tool_calls


def _convert_messages(
    messages_value: Any,
    *,
    event_name: str | None,
    incoming_tools_state_id: str | None,
    state: StreamState,
) -> tuple[
    list[str | dict[str, Any]],
    list[ToolCallChunk],
    bool,
]:
    content: list[str | dict[str, Any]] = []
    tool_calls: list[ToolCallChunk] = []
    messages = _as_dict_list(messages_value, field="messages")
    # The SDK can mirror one execution across levels; prefer the deepest source.
    has_part_tool_execution = any(
        part.get("tool_execution") is not None
        for message in messages
        for part in _as_dict_list(message.get("content"), field="messages.content")
    )
    has_message_tool_execution = not has_part_tool_execution and any(
        message.get("tool_execution") is not None for message in messages
    )

    for message in messages:
        role = str(message.get("role") or "assistant")
        message_inline_data = message.get("inline_data")
        message_tool_state = message.get("tools_state_id")
        tool_state_id = (
            str(message_tool_state)
            if message_tool_state is not None
            else incoming_tools_state_id
        )
        for part in _as_dict_list(message.get("content"), field="messages.content"):
            part_content, part_tool_calls = _convert_content_part(
                part,
                role=role,
                message_inline_data=message_inline_data,
                event_name=event_name,
                incoming_tools_state_id=tool_state_id,
                state=state,
            )
            content.extend(part_content)
            tool_calls.extend(part_tool_calls)

        if message.get("function_call") is not None:
            _close_text_block(state)
            tool_calls.append(
                _tool_call_chunk(
                    message["function_call"],
                    incoming_tools_state_id=tool_state_id,
                    state=state,
                )
            )
        if has_message_tool_execution and message.get("tool_execution") is not None:
            _close_text_block(state)
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
            _close_text_block(state)
            content.append(_reasoning_block(reasoning, state=state))

        if not message.get("content") and message_inline_data is not None:
            _close_text_block(state)
            result_update = _pending_server_tool_result_update(
                message_inline_data,
                provider_data={},
                state=state,
            )
            content.append(
                result_update
                or {
                    "type": "non_standard",
                    "value": {"inline_data": _as_dict(message_inline_data)},
                }
            )

        metadata = unknown_provider_fields(message, _MESSAGE_FIELDS)
        if role not in {"assistant", "reasoning", "tool"}:
            metadata["role"] = role
        if metadata:
            _close_text_block(state)
            content.append(
                {
                    "type": "non_standard",
                    "value": metadata,
                    "index": _take_numeric_block_index(state),
                }
            )

    if tool_calls and len({call["id"] for call in tool_calls}) > 1:
        raise ValueError("Primary streaming supports one client tool call per message")
    return (
        content,
        tool_calls,
        has_part_tool_execution or has_message_tool_execution,
    )


def _observe_scalar_metadata(
    state: StreamState,
    *,
    state_field: str,
    metadata_field: str,
    value: Any,
) -> dict[str, Any]:
    if value is None:
        return {}

    current = getattr(state, state_field)
    if current is None:
        setattr(state, state_field, value)
    elif current != value:
        if metadata_field == "message_id":
            raise ValueError(
                "Primary GigaChat completion contains multiple provider message_id "
                "values; their replay semantics are unsupported."
            )
        if metadata_field == "tools_state_id":
            raise ValueError(
                "Primary GigaChat completion contains multiple tools_state_id values; "
                "their replay semantics are unsupported."
            )
        raise ValueError(
            f"Conflicting primary stream {metadata_field}: {current!r} and {value!r}"
        )

    if metadata_field in state.emitted_metadata_fields:
        return {}
    state.emitted_metadata_fields.add(metadata_field)
    return {metadata_field: value}


def _update_stream_metadata(
    state: StreamState,
    *,
    provider_message_id: str | None,
    provider_tools_state_id: str | None,
    thread_id: Any,
    model: Any,
    created_at: Any,
    x_headers: Mapping[str, Any],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    metadata.update(
        _observe_scalar_metadata(
            state,
            state_field="provider_message_id",
            metadata_field="message_id",
            value=provider_message_id,
        )
    )
    metadata.update(
        _observe_scalar_metadata(
            state,
            state_field="tools_state_id",
            metadata_field="tools_state_id",
            value=provider_tools_state_id,
        )
    )
    metadata.update(
        _observe_scalar_metadata(
            state,
            state_field="thread_id",
            metadata_field="thread_id",
            value=str(thread_id) if thread_id is not None else None,
        )
    )
    metadata.update(
        _observe_scalar_metadata(
            state,
            state_field="model",
            metadata_field="model",
            value=str(model) if model is not None else None,
        )
    )
    # SDK stream events can carry distinct timestamps; keep the first for aggregation.
    if created_at is not None:
        if state.created_at is None:
            state.created_at = int(created_at)
        if "created_at" not in state.emitted_metadata_fields:
            state.emitted_metadata_fields.add("created_at")
            metadata["created_at"] = state.created_at

    new_headers: dict[str, Any] = {}
    for key, value in x_headers.items():
        current = state.x_headers.get(key)
        if key not in state.x_headers or current is None:
            state.x_headers[key] = value
            new_headers[key] = value
        elif value is not None and current != value:
            raise ValueError(
                f"Conflicting primary stream header {key!r}: {current!r} and {value!r}"
            )
    if new_headers:
        metadata["x_headers"] = new_headers
        state.emitted_metadata_fields.add("x_headers")
    return metadata


def _response_metadata(
    event: Mapping[str, Any],
    *,
    event_name: str | None,
    observed_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"output_version": "v1"}
    if event_name is not None:
        metadata["events"] = [event_name]
    metadata.update(observed_metadata)

    if event.get("finish_reason") is not None:
        metadata["finish_reason"] = event["finish_reason"]
    for source_field, event_field in (
        ("additional_data", "additional_data_events"),
        ("logprobs", "logprob_events"),
    ):
        if event.get(source_field) is not None:
            metadata[event_field] = [event[source_field]]
    if event.get("tool_execution") is not None:
        metadata["tool_execution_events"] = [event["tool_execution"]]

    provider_fields = {
        key: value for key, value in event.items() if key not in _EVENT_FIELDS
    }
    if provider_fields:
        metadata["provider_field_events"] = [provider_fields]
    if event_name is not None and event_name not in _KNOWN_EVENTS:
        metadata["raw_events"] = [dict(event)]
    return metadata


def _single_provider_id(
    values: Sequence[Any],
    *,
    error_message: str,
) -> str | None:
    ids = list(dict.fromkeys(str(value) for value in values if value is not None))
    if len(ids) > 1:
        raise ValueError(error_message)
    return ids[0] if ids else None


def _usage_update(
    state: StreamState,
    usage_value: Any,
) -> dict[str, Any] | None:
    usage_metadata = create_usage_metadata(usage_value)
    if usage_metadata is None:
        return None
    normalized = dict(usage_metadata)
    if state.usage_metadata is None:
        state.usage_metadata = normalized
        return normalized
    if state.usage_metadata == normalized:
        return None
    raise ValueError(
        "Conflicting primary stream usage snapshots; "
        "incremental usage semantics are unsupported."
    )


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
    if event_name == "response.error":
        state.pending_server_tool_result_id = None
        raise PrimaryStreamError(event_data)

    normalized_messages = _as_dict_list(
        event_data.get("messages"),
        field="messages",
    )
    top_level_tool_execution = event_data.get("tool_execution")
    terminal_continuation = state.completion_event is not None
    if terminal_continuation:
        if event_name == "response.message.done":
            if event_data == state.completion_event:
                return None
            previous_finish = state.completion_event.get("finish_reason")
            incoming_finish = event_data.get("finish_reason")
            if (
                (
                    previous_finish is not None
                    and incoming_finish is not None
                    and previous_finish != incoming_finish
                )
                or normalized_messages
                or top_level_tool_execution is not None
            ):
                raise ValueError("Conflicting primary completion terminal events")
        elif normalized_messages or top_level_tool_execution is not None:
            raise ValueError(
                "Primary stream content arrived after response.message.done"
            )

    provider_message_id = _single_provider_id(
        [
            event_data.get("message_id"),
            *(message.get("message_id") for message in normalized_messages),
        ],
        error_message=(
            "Primary GigaChat completion contains multiple provider message_id "
            "values; their replay semantics are unsupported."
        ),
    )
    provider_tools_state_id = _single_provider_id(
        [
            event_data.get("tools_state_id"),
            *(message.get("tools_state_id") for message in normalized_messages),
        ],
        error_message=(
            "Primary GigaChat completion contains multiple tools_state_id values; "
            "their replay semantics are unsupported."
        ),
    )
    x_headers = _optional_dict(event_data.get("x_headers"))
    observed_metadata = _update_stream_metadata(
        state,
        provider_message_id=provider_message_id,
        provider_tools_state_id=provider_tools_state_id,
        thread_id=event_data.get("thread_id"),
        model=event_data.get("model"),
        created_at=event_data.get("created_at"),
        x_headers=x_headers,
    )

    request_id = _request_id(state.x_headers)
    if request_id is not None:
        state.message_id = request_id
    elif state.message_id is None:
        state.message_id = f"lc_primary-stream-{uuid4()}"
    elif (
        event_name == "response.message.done" and state.provider_message_id is not None
    ):
        state.message_id = state.provider_message_id

    content, tool_calls, has_nested_tool_execution = _convert_messages(
        normalized_messages,
        event_name=event_name,
        incoming_tools_state_id=provider_tools_state_id,
        state=state,
    )

    if top_level_tool_execution is not None and not has_nested_tool_execution:
        _close_text_block(state)
        block = _tool_execution_block(
            top_level_tool_execution,
            event_name=event_name,
            incoming_tools_state_id=provider_tools_state_id,
            state=state,
        )
        content.append(block)

    if (
        event_name == "response.message.done"
        and state.client_tool_started
        and state.tools_state_id is None
    ):
        raise ValueError(
            "Primary client tool call completed without tools_state_id; "
            "the call cannot be replayed."
        )

    identity_chunks, identity_kwargs = _client_tool_identity_update(state)
    tool_calls.extend(identity_chunks)

    if event_name in {
        "response.message.done",
        "response.tool.failed",
        "response.error",
    }:
        state.pending_server_tool_result_id = None

    response_metadata = _response_metadata(
        event_data,
        event_name=event_name,
        observed_metadata=observed_metadata,
    )

    usage_metadata = _usage_update(state, event_data.get("usage"))
    generation_info = None
    if event_data.get("finish_reason") is not None:
        generation_info = {"finish_reason": event_data["finish_reason"]}

    has_payload = bool(
        content or tool_calls or response_metadata or usage_metadata or generation_info
    )
    if not has_payload:
        return None

    if event_name == "response.message.done" and state.completion_event is None:
        state.completion_event = dict(event_data)

    state.first_chunk = False
    additional_kwargs: dict[str, Any] = identity_kwargs
    reasoning = reasoning_content(content)
    if reasoning is not None:
        additional_kwargs["reasoning_content"] = reasoning
    message = AIMessageChunk(
        content=content,
        additional_kwargs=additional_kwargs,
        id=state.message_id,
        response_metadata=response_metadata,
        tool_call_chunks=tool_calls,
        usage_metadata=usage_metadata,
        chunk_position=(
            "last"
            if event_name == "response.message.done" and not terminal_continuation
            else None
        ),
    )
    return ChatGenerationChunk(message=message, generation_info=generation_info)
