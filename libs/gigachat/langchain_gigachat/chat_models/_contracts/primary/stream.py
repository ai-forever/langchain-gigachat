"""Convert primary SDK stream events into LangChain generation chunks."""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast
from uuid import uuid4

import gigachat.models as gm
from langchain_core.messages import AIMessageChunk
from langchain_core.messages.ai import UsageMetadata
from langchain_core.messages.tool import ToolCallChunk, tool_call_chunk
from langchain_core.outputs import ChatGenerationChunk

from langchain_gigachat.chat_models._contracts.primary.content import (
    ResolvedToolExecution,
    collect_tool_execution_candidates,
    convert_provider_file,
    convert_reasoning_value,
    convert_text_content,
    create_usage_metadata,
    json_fragment,
    provider_dict,
    reasoning_content,
    request_id_from_headers,
    resolve_reasoning_value,
    resolve_tool_execution_candidates,
    unknown_provider_fields,
    validate_provider_id,
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
        "inline_data",
        "reasoning",
        "reasoning_content",
        "text",
        "tool_execution",
    }
)
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


def _next_index(state: StreamState) -> int:
    value = state.next_block_index
    state.next_block_index += 1
    return value


def _text_index(state: StreamState, role: str) -> int | str:
    if state.active_text_block_index is None or state.active_text_block_role != role:
        state.active_text_block_index = _next_index(state)
        state.active_text_block_role = role
    return state.active_text_block_index


def _close_text(state: StreamState) -> None:
    state.active_text_block_index = None
    state.active_text_block_role = None


def _reasoning_index(state: StreamState, message_id: str | None) -> int | str:
    if (
        state.active_reasoning_message_id is None
        or state.active_reasoning_message_id != message_id
    ):
        _close_text(state)
        state.active_reasoning_message_id = message_id or "__reasoning__"
        return _next_index(state)
    # The current index is immediately before next_block_index because reasoning
    # stays active until another content kind is observed.
    return state.next_block_index - 1


def _event_messages(event: gm.PrimaryChatCompletionChunk) -> list[gm.ChatMessageChunk]:
    return list(event.messages or [])


def _single_provider_message_id(
    event: gm.PrimaryChatCompletionChunk,
    messages: Sequence[gm.ChatMessageChunk],
) -> str | None:
    values = [event.message_id, *(message.message_id for message in messages)]
    ids = list(
        dict.fromkeys(
            value
            for raw in values
            if (value := validate_provider_id(raw, field="message_id")) is not None
        )
    )
    if len(ids) > 1:
        raise ValueError(
            "Primary GigaChat stream event contains multiple message_id values."
        )
    return ids[0] if ids else None


def _observe_provider_message_id(state: StreamState, value: str | None) -> None:
    if value is None:
        return
    if state.provider_message_id is not None and state.provider_message_id != value:
        raise ValueError("Primary GigaChat stream changed message_id mid-completion.")
    state.provider_message_id = value


def _event_tools_state_ids(
    event: gm.PrimaryChatCompletionChunk,
    messages: Sequence[gm.ChatMessageChunk],
) -> list[str]:
    values = [event.tools_state_id, *(message.tools_state_id for message in messages)]
    return list(
        dict.fromkeys(
            value
            for raw in values
            if (value := validate_provider_id(raw, field="tools_state_id")) is not None
        )
    )


def _remember_tools_state_ids(state: StreamState, values: Iterable[str]) -> None:
    for value in values:
        if value not in state.provider_tools_state_ids:
            state.provider_tools_state_ids.append(value)


def _client_function_calls(
    messages: Sequence[gm.ChatMessageChunk],
) -> list[tuple[gm.PrimaryChatFunctionCall, gm.ChatMessageChunk]]:
    values: list[tuple[gm.PrimaryChatFunctionCall, gm.ChatMessageChunk]] = []
    for message in messages:
        part_calls = [
            part.function_call
            for part in message.content or []
            if part.function_call is not None
        ]
        if len(part_calls) > 1:
            raise ValueError(
                "Primary streaming supports one client tool call per completion."
            )
        selected = part_calls[0] if part_calls else message.function_call
        if part_calls and message.function_call is not None:
            part_value = part_calls[0].model_dump(exclude_none=True, by_alias=True)
            message_value = message.function_call.model_dump(
                exclude_none=True, by_alias=True
            )
            if part_value != message_value:
                raise ValueError(
                    "Primary stream contains conflicting part- and message-level "
                    "client function calls."
                )
        if selected is not None:
            values.append((selected, message))
    if len(values) > 1:
        raise ValueError(
            "Primary streaming supports one client tool call per completion."
        )
    return values


def _set_client_state_id(state: StreamState, value: str) -> None:
    if state.client_tools_state_id is not None and state.client_tools_state_id != value:
        raise ValueError(
            "Primary client tool call received conflicting tools_state_id values."
        )
    state.client_tools_state_id = value
    state.client_tool_id = value
    state.tools_state_id = value


def _client_tool_chunks(
    state: StreamState,
    calls: Sequence[tuple[gm.PrimaryChatFunctionCall, gm.ChatMessageChunk]],
    *,
    event_state_ids: Sequence[str],
) -> list[ToolCallChunk]:
    chunks: list[ToolCallChunk] = []
    if calls:
        function_call, message = calls[0]
        message_state_id = validate_provider_id(
            message.tools_state_id,
            field="tools_state_id",
        )
        candidate_ids = list(
            dict.fromkeys(
                value
                for value in (
                    message_state_id,
                    *(event_state_ids if message_state_id is None else ()),
                )
                if value is not None
            )
        )
        if len(candidate_ids) > 1:
            raise ValueError(
                "Primary client function call has multiple tools_state_id values."
            )
        if candidate_ids:
            _set_client_state_id(state, candidate_ids[0])

        name = function_call.name
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Function call name must be a non-empty string.")
        if state.client_tool_name is not None and state.client_tool_name != name:
            raise ValueError("Primary stream changed client function name mid-call.")
        state.client_tool_name = name
        if state.client_tool_index is None:
            state.client_tool_index = _next_index(state)
        state.client_tool_started = True
        _close_text(state)

        arguments = function_call.arguments
        if isinstance(arguments, str):
            fragment = arguments
            if state.client_tool_argument_mode is None:
                state.client_tool_argument_mode = "fragments"
            elif state.client_tool_argument_mode != "fragments":
                raise ValueError(
                    "Primary stream changed client function argument encoding."
                )
            state.client_tool_arguments_text += fragment
        else:
            fragment = json_fragment(arguments)
            if state.client_tool_argument_mode is None:
                state.client_tool_argument_mode = "complete"
                state.client_tool_arguments_text = fragment
                state.client_tool_arguments_complete = True
            elif (
                state.client_tool_argument_mode == "complete"
                and state.client_tool_arguments_text == fragment
            ):
                fragment = ""
            else:
                raise ValueError(
                    "Primary stream emitted multiple complete client argument values."
                )

        chunks.append(
            tool_call_chunk(
                name=None if state.client_tool_name_emitted else name,
                args=fragment,
                id=(
                    None
                    if state.client_tool_id_emitted
                    else state.client_tools_state_id
                ),
                index=state.client_tool_index,
            )
        )
        state.client_tool_name_emitted = True
        if state.client_tools_state_id is not None:
            state.client_tool_id_emitted = True

    if (
        state.client_tool_started
        and state.client_tools_state_id is None
        and len(event_state_ids) == 1
    ):
        _set_client_state_id(state, event_state_ids[0])

    if (
        state.client_tool_started
        and state.client_tools_state_id is not None
        and not state.client_tool_id_emitted
    ):
        assert state.client_tool_index is not None
        chunks.append(
            tool_call_chunk(
                name=None,
                args=None,
                id=state.client_tools_state_id,
                index=state.client_tool_index,
            )
        )
        state.client_tool_id_emitted = True
    return chunks


def _new_server_tool_id(state: StreamState) -> str:
    value = f"lc_primary-server-tool-{state.next_server_tool_sequence}"
    state.next_server_tool_sequence += 1
    return value


def _is_terminal_tool(execution: Any, event_name: str | None) -> bool:
    status = str(provider_dict(execution).get("status") or "").lower()
    return status in _TERMINAL_TOOL_STATUSES or event_name in {
        "response.tool.completed",
        "response.tool.failed",
    }


def _resolve_server_tools(
    state: StreamState,
    event: gm.PrimaryChatCompletionChunk,
    messages: Sequence[gm.ChatMessageChunk],
) -> list[ResolvedToolExecution]:
    candidates = collect_tool_execution_candidates(
        messages,
        response_tool_execution=event.tool_execution,
    )
    if not candidates:
        return []

    active_available = state.active_server_tool_call_id

    def id_factory() -> str:
        nonlocal active_available
        if active_available is not None:
            value = active_available
            active_available = None
            return value
        return _new_server_tool_id(state)

    resolved = resolve_tool_execution_candidates(candidates, id_factory=id_factory)
    idless = [item for item in resolved if item.execution_id is None]
    if len(idless) > 1:
        raise ValueError(
            "Primary stream contains ambiguous parallel server tools without IDs."
        )
    return resolved


def _argument_delta(state: StreamState, tool_call_id: str, value: Any) -> str:
    current = json_fragment({} if value is None else value)
    previous = state.server_tool_argument_state.get(tool_call_id)
    if previous is None:
        state.server_tool_argument_state[tool_call_id] = ("cumulative", current)
        return current
    _mode, prior = previous
    if current == prior:
        return ""
    if current.startswith(prior):
        state.server_tool_argument_state[tool_call_id] = ("cumulative", current)
        return current[len(prior) :]
    state.server_tool_argument_state[tool_call_id] = (
        "fragments",
        prior + current,
    )
    return current


def _server_tool_block(
    state: StreamState,
    resolved: ResolvedToolExecution,
    *,
    event_name: str | None,
    inline_data: Any = None,
    provider_data: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    raw = provider_dict(resolved.candidate.execution)
    tool_call_id = resolved.tool_call_id
    explicit = resolved.execution_id is not None
    terminal = _is_terminal_tool(raw, event_name)

    if not explicit:
        if state.active_server_tool_call_id not in {None, tool_call_id}:
            raise ValueError(
                "Primary stream contains ambiguous parallel server tools without IDs."
            )
        if not terminal:
            state.active_server_tool_call_id = tool_call_id

    name_value = raw.get("name")
    name = name_value if isinstance(name_value, str) and name_value else None
    previous_name = state.server_tool_names.get(tool_call_id)
    if name is not None and previous_name is not None and previous_name != name:
        raise ValueError("Primary stream changed server tool name mid-execution.")
    if name is not None:
        state.server_tool_names[tool_call_id] = name

    extras: dict[str, Any] = {"provider_tool_execution": raw}
    if inline_data is not None:
        extras["inline_data"] = provider_dict(inline_data)
    if provider_data:
        extras["provider_data"] = copy.deepcopy(dict(provider_data))

    if terminal:
        if tool_call_id in state.completed_server_tool_ids:
            return []
        state.completed_server_tool_ids.add(tool_call_id)
        state.last_completed_server_tool_id = tool_call_id
        if not explicit:
            state.active_server_tool_call_id = None
        status = str(raw.get("status") or "").lower()
        failed = status in _FAILED_TOOL_STATUSES or event_name == "response.tool.failed"
        if tool_call_id not in state.server_tool_result_indexes:
            state.server_tool_result_indexes[tool_call_id] = _next_index(state)
        index = state.server_tool_result_indexes[tool_call_id]
        block: dict[str, Any] = {
            "type": "server_tool_result",
            "id": f"{tool_call_id}:result",
            "tool_call_id": tool_call_id,
            "status": "error" if failed else "success",
            "index": index,
            "extras": extras,
        }
        if raw.get("output") is not None:
            block["output"] = raw["output"]
        return [block]

    first = tool_call_id not in state.server_tool_indexes
    if first:
        state.server_tool_indexes[tool_call_id] = _next_index(state)
    index = state.server_tool_indexes[tool_call_id]
    arguments = raw.get("arguments", raw.get("args"))
    block = {
        "type": "server_tool_call_chunk",
        "args": _argument_delta(state, tool_call_id, arguments),
        "index": index,
        "extras": extras,
    }
    if first:
        block["id"] = tool_call_id
    if name is not None and previous_name is None:
        block["name"] = name
    return [block]


def _resolved_by_coordinates(
    values: Sequence[ResolvedToolExecution],
) -> dict[tuple[str, int | None, int | None], ResolvedToolExecution]:
    result: dict[tuple[str, int | None, int | None], ResolvedToolExecution] = {}
    for resolved in values:
        for coordinates in resolved.mirrored_sources:
            result[coordinates] = resolved
    return result


def _convert_content(
    state: StreamState,
    event: gm.PrimaryChatCompletionChunk,
    messages: Sequence[gm.ChatMessageChunk],
    resolved_tools: Sequence[ResolvedToolExecution],
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    emitted_tools: set[str] = set()
    by_coordinates = _resolved_by_coordinates(resolved_tools)

    for message_index, message in enumerate(messages):
        role = message.role or "assistant"
        message_inline = message.inline_data
        for part_index, part in enumerate(message.content or []):
            unknown = unknown_provider_fields(part, _CONTENT_FIELDS)
            if part.text is not None:
                blocks.append(
                    convert_text_content(
                        part.text,
                        role=role,
                        inline_data=part.inline_data or message_inline,
                        provider_data=unknown,
                        index=_text_index(state, role),
                    )
                )
            for file_ in part.files or []:
                _close_text(state)
                blocks.append(convert_provider_file(file_, index=_next_index(state)))
            if part.function_result is not None:
                _close_text(state)
                blocks.append(
                    {
                        "type": "non_standard",
                        "value": {
                            "function_result": part.function_result.model_dump(
                                exclude_none=True,
                                by_alias=True,
                            )
                        },
                        "index": _next_index(state),
                    }
                )

            resolved = by_coordinates.get(("part", message_index, part_index))
            if resolved is not None and resolved.tool_call_id not in emitted_tools:
                _close_text(state)
                tool_blocks = _server_tool_block(
                    state,
                    resolved,
                    event_name=event.event,
                    inline_data=part.inline_data or message_inline,
                    provider_data=unknown,
                )
                emitted_tools.add(resolved.tool_call_id)
                blocks.extend(tool_blocks)

            reasoning = resolve_reasoning_value(part)
            if reasoning is not None:
                blocks.append(
                    convert_reasoning_value(
                        reasoning,
                        index=_reasoning_index(state, message.message_id),
                    )
                )
            if (
                part.text is None
                and part.tool_execution is None
                and reasoning is None
                and (part.inline_data is not None or unknown)
            ):
                _close_text(state)
                value: dict[str, Any] = {}
                if part.inline_data is not None:
                    value["inline_data"] = provider_dict(part.inline_data)
                value.update(unknown)
                blocks.append(
                    {
                        "type": "non_standard",
                        "value": value,
                        "index": _next_index(state),
                    }
                )

        resolved = by_coordinates.get(("message", message_index, None))
        if resolved is not None and resolved.tool_call_id not in emitted_tools:
            _close_text(state)
            tool_blocks = _server_tool_block(
                state,
                resolved,
                event_name=event.event,
                inline_data=message.inline_data,
            )
            emitted_tools.add(resolved.tool_call_id)
            blocks.extend(tool_blocks)

        reasoning = resolve_reasoning_value(message)
        if reasoning is not None:
            blocks.append(
                convert_reasoning_value(
                    reasoning,
                    index=_reasoning_index(state, message.message_id),
                )
            )
        message_unknown = unknown_provider_fields(message, _MESSAGE_FIELDS)
        if role not in {"assistant", "reasoning", "tool"}:
            message_unknown["role"] = role
        if message_unknown:
            _close_text(state)
            blocks.append(
                {
                    "type": "non_standard",
                    "value": message_unknown,
                    "index": _next_index(state),
                }
            )

    resolved = by_coordinates.get(("response", None, None))
    if resolved is not None and resolved.tool_call_id not in emitted_tools:
        _close_text(state)
        tool_blocks = _server_tool_block(
            state,
            resolved,
            event_name=event.event,
        )
        blocks.extend(tool_blocks)
    return blocks


def _observe_metadata(
    state: StreamState,
    event: gm.PrimaryChatCompletionChunk,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for field_name in ("thread_id", "model"):
        value = getattr(event, field_name)
        if value is None:
            continue
        previous = getattr(state, field_name)
        if previous is not None and previous != value:
            raise ValueError(f"Primary stream changed {field_name} mid-completion.")
        if previous is None:
            setattr(state, field_name, value)
            metadata[field_name] = value

    if event.created_at is not None and state.created_at is None:
        state.created_at = event.created_at
        metadata["created_at"] = event.created_at

    incoming_headers = dict(event.x_headers or {})
    new_headers: dict[str, Any] = {}
    for key, value in incoming_headers.items():
        previous = state.x_headers.get(key)
        if previous is not None and previous != value:
            raise ValueError(f"Primary stream changed x-header {key!r}.")
        if key not in state.x_headers:
            state.x_headers[key] = value
            new_headers[key] = value
    if new_headers:
        metadata["x_headers"] = new_headers

    event_name = event.event
    if event_name is not None:
        metadata["events"] = [event_name]
        if event_name not in _KNOWN_EVENTS:
            metadata["provider_events"] = [provider_dict(event)]

    unknown = unknown_provider_fields(event, _EVENT_FIELDS)
    if unknown:
        metadata["provider_fields"] = unknown
    if event.additional_data is not None:
        metadata["additional_data"] = copy.deepcopy(event.additional_data)
    if event.logprobs is not None:
        metadata["logprobs"] = [
            item.model_dump(exclude_none=True, by_alias=True) for item in event.logprobs
        ]
    return metadata


def _usage_update(state: StreamState, value: Any) -> UsageMetadata | None:
    usage = create_usage_metadata(value)
    if usage is None:
        return None
    normalized = dict(usage)
    if state.usage_metadata is None:
        state.usage_metadata = normalized
        return usage
    if state.usage_metadata == normalized:
        return None
    raise ValueError("Primary stream emitted conflicting usage snapshots.")


def convert_stream_event(
    event: gm.PrimaryChatCompletionChunk,
    *,
    state: StreamState,
) -> ChatGenerationChunk | None:
    """Convert one SDK event; raw SSE mappings are intentionally not accepted."""
    if not isinstance(event, gm.PrimaryChatCompletionChunk):
        raise TypeError(
            "Primary stream conversion requires an SDK "
            "PrimaryChatCompletionChunk instance."
        )
    event_data = provider_dict(event)
    if not event_data:
        return None
    if event.event == "response.error":
        raise PrimaryStreamError(event_data)

    messages = _event_messages(event)
    provider_message_id = _single_provider_message_id(event, messages)
    _observe_provider_message_id(state, provider_message_id)
    event_state_ids = _event_tools_state_ids(event, messages)
    _remember_tools_state_ids(state, event_state_ids)

    if state.completion_event is not None:
        if messages or event.tool_execution is not None:
            raise ValueError(
                "Primary stream emitted content after response.message.done"
            )
        if (
            event.event == "response.message.done"
            and event_data == state.completion_event
        ):
            return None

    calls = _client_function_calls(messages)
    tool_calls = _client_tool_chunks(
        state,
        calls,
        event_state_ids=event_state_ids,
    )
    resolved_tools = _resolve_server_tools(state, event, messages)
    content = _convert_content(state, event, messages, resolved_tools)

    if (
        event.event == "response.message.done"
        and state.client_tool_started
        and state.client_tools_state_id is None
    ):
        raise ValueError(
            "Primary client tool call completed without tools_state_id; "
            "the call cannot be replayed."
        )

    response_metadata = _observe_metadata(state, event)
    if event.finish_reason is not None:
        response_metadata["finish_reason"] = event.finish_reason
    if event.event == "response.message.done":
        if state.provider_message_id is not None:
            response_metadata["message_id"] = state.provider_message_id
        if state.thread_id is not None:
            response_metadata.setdefault("thread_id", state.thread_id)
        if state.model is not None:
            response_metadata.setdefault("model", state.model)
            response_metadata.setdefault("model_name", state.model)
        if state.provider_tools_state_ids:
            response_metadata["tools_state_ids"] = list(state.provider_tools_state_ids)
        if state.client_tools_state_id is not None:
            response_metadata["tools_state_id"] = state.client_tools_state_id

    usage_metadata = _usage_update(state, event.usage)
    generation_info = (
        {"finish_reason": event.finish_reason}
        if event.finish_reason is not None
        else None
    )

    if request_id := request_id_from_headers(state.x_headers):
        state.message_id = request_id
    elif state.provider_message_id is not None:
        state.message_id = state.provider_message_id
    elif state.message_id is None:
        state.message_id = f"lc_primary-stream-{uuid4()}"

    additional_kwargs: dict[str, Any] = {}
    reasoning = reasoning_content(content)
    if reasoning is not None:
        additional_kwargs["reasoning_content"] = reasoning
    if event.event == "response.message.done":
        if state.provider_tools_state_ids:
            additional_kwargs["tools_state_ids"] = list(state.provider_tools_state_ids)
        if state.client_tools_state_id is not None:
            additional_kwargs["tools_state_id"] = state.client_tools_state_id

    has_payload = bool(
        content or tool_calls or response_metadata or usage_metadata or generation_info
    )
    if not has_payload:
        return None

    if event.event == "response.message.done":
        state.completion_event = event_data

    message = AIMessageChunk(
        content=cast(list[str | dict[Any, Any]], content),
        additional_kwargs=additional_kwargs,
        id=state.message_id,
        response_metadata=response_metadata,
        tool_call_chunks=tool_calls,
        usage_metadata=usage_metadata,
        chunk_position="last" if event.event == "response.message.done" else None,
    )
    return ChatGenerationChunk(
        message=message,
        generation_info=generation_info,
    )
