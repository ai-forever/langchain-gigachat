"""Convert primary named stream events into LangChain generation chunks."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, Literal
from uuid import uuid4

import gigachat.models as gm
from langchain_core.messages import AIMessageChunk, ToolCallChunk, UsageMetadata
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.outputs import ChatGenerationChunk

from langchain_gigachat.chat_models._contracts.primary.content import (
    ResolvedToolExecution,
    ToolExecutionCoordinates,
    collect_tool_execution_candidates,
    convert_provider_file,
    convert_text_content,
    convert_tool_execution,
    create_usage_metadata,
    json_fragment,
    provider_dict,
    reasoning_content,
    resolve_tool_execution_candidates,
    server_tool_execution_id,
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
_MULTIPLE_CLIENT_TOOL_CALLS = (
    "Primary streaming supports one client tool call per completion"
)
_SECOND_CLIENT_TOOL_CALL = (
    "Primary streaming received a second client tool call after the first "
    "call's arguments were complete"
)
_AMBIGUOUS_TOOL_STATE_OWNER = (
    "Primary tools_state_id could belong to both a client function call and a "
    "server tool lifecycle; explicit independent identities are required."
)

_ToolStateOwner = Literal["client", "server", "unassigned"]


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
    if role != "reasoning":
        state.active_reasoning_message_id = None
    return index


def _take_reasoning_block_index(
    state: StreamState,
    *,
    explicit: Any = None,
    message_id: str | None = None,
) -> int | str:
    if (
        explicit is None
        and state.active_text_block_role == "reasoning"
        and state.active_text_block_index is not None
        and (
            message_id is None
            or state.active_reasoning_message_id is None
            or message_id == state.active_reasoning_message_id
        )
    ):
        if message_id is not None:
            state.active_reasoning_message_id = message_id
        return state.active_text_block_index

    index = _take_block_index(state, explicit)
    state.active_text_block_index = index
    state.active_text_block_role = "reasoning"
    state.active_reasoning_message_id = message_id
    return index


def _close_text_block(state: StreamState) -> None:
    state.active_text_block_index = None
    state.active_text_block_role = None
    state.active_reasoning_message_id = None


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
    message_id: str | None = None,
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
        index=_take_reasoning_block_index(
            state,
            explicit=explicit_index,
            message_id=message_id,
        ),
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
    if not is_first_fragment and state.client_tool_arguments_complete:
        raise ValueError(_SECOND_CLIENT_TOOL_CALL)
    if is_first_fragment:
        if not incoming_name:
            raise ValueError("First primary client tool fragment must include a name")
        call_id = explicit_id or incoming_tools_state_id or state.client_tools_state_id
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
            or state.client_tools_state_id
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
    if arguments is not None:
        argument_mode = "fragments" if isinstance(arguments, str) else "snapshot"
        if state.client_tool_argument_mode is None:
            state.client_tool_argument_mode = argument_mode
        elif (
            state.client_tool_argument_mode == "snapshot" or argument_mode == "snapshot"
        ):
            raise ValueError(
                "Primary client tool arguments contain multiple or mixed "
                "structured snapshots; cumulative snapshot semantics are unsupported."
            )
        if isinstance(arguments, str):
            state.client_tool_arguments_text += arguments
            try:
                parsed_arguments = json.loads(state.client_tool_arguments_text)
            except (TypeError, ValueError):
                pass
            else:
                state.client_tool_arguments_complete = isinstance(
                    parsed_arguments,
                    Mapping,
                )
        else:
            state.client_tool_arguments_complete = True
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
        or state.client_tools_state_id is None
        or state.client_tool_index is None
    ):
        return [], {}

    if state.client_tool_id is None:
        state.client_tool_id = state.client_tools_state_id
        return [
            tool_call_chunk(
                name=None,
                args="",
                id=state.client_tools_state_id,
                index=state.client_tool_index,
            )
        ], {}

    if (
        state.client_tool_id == state.client_tools_state_id
        or state.client_tool_state_mapping_emitted
    ):
        return [], {}

    state.client_tool_state_mapping_emitted = True
    return [], {
        "provider_tool_state_by_call_id": {
            state.client_tool_id: state.client_tools_state_id,
        }
    }


def _normalized_server_tool_payload(
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        key: deepcopy(value)
        for key, value in execution.items()
        if key not in {"call_id", "tool_call_id", "id", "index"}
    }
    status = str(payload.get("status") or "").lower()
    if status in {"complete", "completed", "done", "success"}:
        payload["status"] = "success"
    elif status in {"error", "failed", "failure"}:
        payload["status"] = "failed"
    return payload


def _is_terminal_server_execution(
    execution: Mapping[str, Any],
    *,
    event_name: str | None,
) -> bool:
    status = str(execution.get("status") or "").lower()
    return event_name in {
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


def _server_tool_repeat_call_id(
    execution: Mapping[str, Any],
    *,
    event_name: str | None,
    incoming_tools_state_id: str | None,
    state: StreamState,
) -> str | None:
    if not _is_terminal_server_execution(execution, event_name=event_name):
        return None

    execution_id = server_tool_execution_id(execution)
    if execution_id is not None:
        call_id = state.server_tool_call_ids_by_execution_id.get(execution_id)
    elif incoming_tools_state_id is not None:
        call_id = state.server_tool_call_ids_by_state_id.get(incoming_tools_state_id)
    else:
        call_id = None
    payload = _normalized_server_tool_payload(execution)

    if call_id is not None and call_id in state.server_tool_terminal_payloads:
        previous = state.server_tool_terminal_payloads[call_id]
        if previous != payload:
            raise ValueError(
                "Conflicting repeated terminal payload or shared provider state "
                f"for primary server tool {call_id!r}."
            )
        return call_id

    if event_name != "response.message.done":
        return None
    matches = [
        known_call_id
        for known_call_id, known_payload in state.server_tool_terminal_payloads.items()
        if known_payload == payload
    ]
    if len(matches) > 1:
        raise ValueError(
            "Primary response.message.done repeated a server tool execution that "
            "matches multiple completed lifecycles; correlation is ambiguous."
        )
    return matches[0] if matches else None


def _server_tool_call_id(
    execution: Mapping[str, Any],
    *,
    incoming_tools_state_id: str | None,
    terminal: bool,
    state: StreamState,
) -> tuple[str, str | None]:
    execution_id = server_tool_execution_id(execution)
    active_call_id = state.active_server_tool_call_id

    if execution_id is not None:
        call_id = state.server_tool_call_ids_by_execution_id.get(execution_id)
        if call_id is None:
            if active_call_id is not None:
                active_execution_ids = {
                    known_execution_id
                    for known_execution_id, known_call_id in (
                        state.server_tool_call_ids_by_execution_id.items()
                    )
                    if known_call_id == active_call_id
                }
                if active_execution_ids and execution_id not in active_execution_ids:
                    raise ValueError(
                        "Primary server tool identity changed while a tool "
                        "lifecycle was active"
                    )
                call_id = active_call_id
            else:
                call_id = execution_id
            state.server_tool_call_ids_by_execution_id[execution_id] = call_id
    elif incoming_tools_state_id is not None:
        call_id = state.server_tool_call_ids_by_state_id.get(incoming_tools_state_id)
        if call_id is None:
            if active_call_id is not None:
                call_id = active_call_id
            else:
                call_id = f"lc_primary-server-tool-{state.next_server_tool_sequence}"
                state.next_server_tool_sequence += 1
    elif active_call_id is not None:
        call_id = active_call_id
    else:
        call_id = f"lc_primary-server-tool-{state.next_server_tool_sequence}"
        state.next_server_tool_sequence += 1

    if incoming_tools_state_id is not None:
        mapped_call_id = state.server_tool_call_ids_by_state_id.get(
            incoming_tools_state_id
        )
        if mapped_call_id is not None and mapped_call_id != call_id:
            raise ValueError(
                "Primary server tool state maps to conflicting tool lifecycles"
            )
        previous_state_id = state.server_tool_state_ids_by_call_id.get(call_id)
        if (
            previous_state_id is not None
            and previous_state_id != incoming_tools_state_id
        ):
            raise ValueError(
                "Primary server tool lifecycle received conflicting tools_state_id "
                "values"
            )
        state.server_tool_call_ids_by_state_id[incoming_tools_state_id] = call_id
        state.server_tool_state_ids_by_call_id[call_id] = incoming_tools_state_id

    if terminal:
        state.active_server_tool_call_id = None
        if (
            execution_id is None
            and call_id not in state.server_tool_state_ids_by_call_id
            and call_id not in state.unresolved_server_tool_call_ids
        ):
            state.unresolved_server_tool_call_ids.append(call_id)
    elif active_call_id is None:
        state.active_server_tool_call_id = call_id
    elif active_call_id != call_id:
        raise ValueError(
            "Primary streaming does not support overlapping server tool lifecycles"
        )

    return call_id, incoming_tools_state_id


def _server_tool_identity_update(
    state: StreamState,
    *,
    incoming_tools_state_id: str | None,
    executions: Sequence[Mapping[str, Any]],
    event_name: str | None,
) -> dict[str, Any]:
    """Associate provider state that arrives after an idless server-tool result."""
    if incoming_tools_state_id is None:
        return {}
    if incoming_tools_state_id in state.server_tool_call_ids_by_state_id:
        return {}
    if state.active_server_tool_call_id is not None:
        # The current lifecycle claims the state while its execution is converted.
        return {}

    unresolved = [
        call_id
        for call_id in state.unresolved_server_tool_call_ids
        if call_id not in state.server_tool_state_ids_by_call_id
    ]
    if not unresolved:
        return {}

    terminal_payloads = [
        _normalized_server_tool_payload(execution)
        for execution in executions
        if _is_terminal_server_execution(execution, event_name=event_name)
    ]
    if terminal_payloads:
        matches = [
            call_id
            for call_id in unresolved
            if state.server_tool_terminal_payloads.get(call_id) in terminal_payloads
        ]
        if len(matches) > 1:
            raise ValueError(
                "Primary repeated server tool execution matches multiple unresolved "
                "tool lifecycles; correlation is ambiguous."
            )
        if not matches and event_name == "response.message.done":
            raise ValueError(
                "Primary response.message.done repeated a conflicting server tool "
                "execution for an unresolved lifecycle."
            )
        if not matches:
            return {}
        unresolved = matches
    if len(unresolved) > 1:
        raise ValueError(
            "Primary server tool state arrived after multiple unresolved "
            "tool lifecycles; correlation is ambiguous."
        )

    call_id = unresolved[0]
    state.server_tool_call_ids_by_state_id[incoming_tools_state_id] = call_id
    state.server_tool_state_ids_by_call_id[call_id] = incoming_tools_state_id
    state.unresolved_server_tool_call_ids.remove(call_id)
    return {}


def _tool_execution_block(
    execution_value: Any,
    *,
    event_name: str | None,
    incoming_tools_state_id: str | None,
    state: StreamState,
) -> dict[str, Any] | None:
    execution = _as_dict(execution_value)

    status = str(execution.get("status") or "").lower()
    failed = event_name == "response.tool.failed" or status in {
        "error",
        "failed",
        "failure",
    }
    terminal = _is_terminal_server_execution(execution, event_name=event_name)
    repeated_call_id = _server_tool_repeat_call_id(
        execution,
        event_name=event_name,
        incoming_tools_state_id=incoming_tools_state_id,
        state=state,
    )
    if repeated_call_id is not None:
        if incoming_tools_state_id is not None:
            mapped_call_id = state.server_tool_call_ids_by_state_id.get(
                incoming_tools_state_id
            )
            if mapped_call_id is not None and mapped_call_id != repeated_call_id:
                raise ValueError(
                    "Primary repeated server tool execution conflicts with its "
                    "provider tools_state_id mapping."
                )
        return None
    call_id, provider_id = _server_tool_call_id(
        execution,
        incoming_tools_state_id=incoming_tools_state_id,
        terminal=terminal,
        state=state,
    )
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
    has_execution_id = any(
        execution.get(identity_field) is not None
        for identity_field in ("call_id", "tool_call_id", "id")
    )
    if not has_execution_id and provider_id is not None and provider_id != call_id:
        block_extras = block.setdefault("extras", {})
        block_extras["provider_server_tool_state_by_call_id"] = {
            call_id: provider_id,
        }
    if terminal:
        payload = _normalized_server_tool_payload(execution)
        previous_payload = state.server_tool_terminal_payloads.get(call_id)
        if previous_payload is not None and previous_payload != payload:
            raise ValueError(
                f"Conflicting repeated terminal payload for primary server tool "
                f"{call_id!r}."
            )
        state.server_tool_terminal_payloads[call_id] = payload
        state.pending_server_tool_result_id = None if failed else call_id
        return block

    if existing_index is None:
        block["name"] = incoming_name or ""
    else:
        block["name"] = incoming_name if known_name is None and incoming_name else ""
        block_extras = block.setdefault("extras", {})
        block_extras["provider_tool_execution_updates"] = [execution]
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
    resolved_tool_execution: ResolvedToolExecution | None,
    role: str,
    message_inline_data: Any,
    event_name: str | None,
    incoming_client_tools_state_id: str | None,
    incoming_server_tools_state_id: str | None,
    message_id: str | None,
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
                incoming_tools_state_id=incoming_client_tools_state_id,
                state=state,
            )
        )

    if resolved_tool_execution is not None:
        _close_text_block(state)
        tool_execution_block = _tool_execution_block(
            _resolved_execution_value(resolved_tool_execution),
            event_name=event_name,
            incoming_tools_state_id=incoming_server_tools_state_id,
            state=state,
        )
        if tool_execution_block is not None:
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
        content.append(
            _reasoning_block(
                reasoning,
                state=state,
                explicit_index=part.get("index"),
                message_id=message_id,
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
    tools_state_owner: _ToolStateOwner | None,
    resolved_tool_executions: Mapping[
        ToolExecutionCoordinates,
        ResolvedToolExecution,
    ],
    state: StreamState,
) -> tuple[list[str | dict[str, Any]], list[ToolCallChunk]]:
    content: list[str | dict[str, Any]] = []
    tool_calls: list[ToolCallChunk] = []
    messages = _as_dict_list(messages_value, field="messages")
    message_parts = [
        _as_dict_list(message.get("content"), field="messages.content")
        for message in messages
    ]
    function_call_message_count = sum(
        bool(
            message.get("function_call") is not None
            or any(part.get("function_call") is not None for part in parts)
        )
        for message, parts in zip(messages, message_parts)
    )
    if function_call_message_count > 1:
        raise ValueError(_MULTIPLE_CLIENT_TOOL_CALLS)

    for message_index, (message, parts) in enumerate(zip(messages, message_parts)):
        role = str(message.get("role") or "assistant")
        message_inline_data = message.get("inline_data")
        message_id_value = message.get("message_id")
        message_id = (
            str(message_id_value)
            if message_id_value is not None
            else state.provider_message_id
        )
        message_tool_state = message.get("tools_state_id")
        tool_state_id = (
            str(message_tool_state)
            if message_tool_state is not None
            else incoming_tools_state_id
        )
        part_function_calls = [
            part["function_call"]
            for part in parts
            if part.get("function_call") is not None
        ]
        if len(part_function_calls) > 1:
            raise ValueError(_MULTIPLE_CLIENT_TOOL_CALLS)
        message_function_call = message.get("function_call")
        mirrored_message_function_call = (
            message_function_call is not None
            and len(part_function_calls) == 1
            and _as_dict(message_function_call) == _as_dict(part_function_calls[0])
        )
        if (
            message_function_call is not None
            and part_function_calls
            and not mirrored_message_function_call
        ):
            raise ValueError(_MULTIPLE_CLIENT_TOOL_CALLS)

        for part_index, part in enumerate(parts):
            part_content, part_tool_calls = _convert_content_part(
                part,
                resolved_tool_execution=resolved_tool_executions.get(
                    ("part", message_index, part_index)
                ),
                role=role,
                message_inline_data=message_inline_data,
                event_name=event_name,
                incoming_client_tools_state_id=(
                    tool_state_id if tools_state_owner == "client" else None
                ),
                incoming_server_tools_state_id=(
                    tool_state_id if tools_state_owner == "server" else None
                ),
                message_id=message_id,
                state=state,
            )
            content.extend(part_content)
            tool_calls.extend(part_tool_calls)

        if message_function_call is not None and not mirrored_message_function_call:
            _close_text_block(state)
            tool_calls.append(
                _tool_call_chunk(
                    message_function_call,
                    incoming_tools_state_id=(
                        tool_state_id if tools_state_owner == "client" else None
                    ),
                    state=state,
                )
            )
        resolved_message_execution = resolved_tool_executions.get(
            ("message", message_index, None)
        )
        if resolved_message_execution is not None:
            _close_text_block(state)
            block = _tool_execution_block(
                _resolved_execution_value(resolved_message_execution),
                event_name=event_name,
                incoming_tools_state_id=(
                    tool_state_id if tools_state_owner == "server" else None
                ),
                state=state,
            )
            if block is not None:
                content.append(block)

        reasoning = message.get("reasoning", message.get("reasoning_content"))
        if reasoning is not None:
            content.append(
                _reasoning_block(
                    reasoning,
                    state=state,
                    message_id=message_id,
                )
            )

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
        raise ValueError(_MULTIPLE_CLIENT_TOOL_CALLS)
    return content, tool_calls


def _event_has_client_function_call(
    messages: Sequence[Mapping[str, Any]],
) -> bool:
    return any(
        message.get("function_call") is not None
        or any(
            part.get("function_call") is not None
            for part in _as_dict_list(message.get("content"), field="messages.content")
        )
        for message in messages
    )


def _resolved_execution_value(
    resolved: ResolvedToolExecution,
) -> dict[str, Any]:
    execution = _as_dict(resolved.candidate.execution)
    if (
        resolved.execution_id is not None
        and server_tool_execution_id(execution) is None
    ):
        execution["call_id"] = resolved.execution_id
    return execution


def _tool_state_owner(
    state: StreamState,
    *,
    incoming_tools_state_id: str | None,
    contains_client_function_call: bool,
    contains_server_tool_execution: bool,
    server_execution_requires_state: bool,
) -> _ToolStateOwner | None:
    if incoming_tools_state_id is None:
        return None

    known_client_owner = incoming_tools_state_id == state.client_tools_state_id
    known_server_owner = (
        incoming_tools_state_id in state.server_tool_call_ids_by_state_id
    )
    unresolved_client_owner = (
        state.client_tool_started and state.client_tools_state_id is None
    )
    unresolved_server_owner = bool(
        state.active_server_tool_call_id
        or any(
            call_id not in state.server_tool_state_ids_by_call_id
            for call_id in state.unresolved_server_tool_call_ids
        )
    )

    if contains_client_function_call:
        if known_server_owner or (
            contains_server_tool_execution and server_execution_requires_state
        ):
            raise ValueError(_AMBIGUOUS_TOOL_STATE_OWNER)
        return "client"
    if contains_server_tool_execution:
        if known_client_owner:
            raise ValueError(_AMBIGUOUS_TOOL_STATE_OWNER)
        return "server"
    if known_client_owner and known_server_owner:
        raise ValueError(_AMBIGUOUS_TOOL_STATE_OWNER)
    if known_client_owner:
        return "client"
    if known_server_owner:
        return "server"

    client_plausible = unresolved_client_owner
    server_plausible = unresolved_server_owner
    if client_plausible and server_plausible:
        raise ValueError(_AMBIGUOUS_TOOL_STATE_OWNER)
    if client_plausible:
        return "client"
    if server_plausible:
        return "server"
    return "unassigned"


def _refresh_replay_tools_state_id(state: StreamState) -> None:
    has_client_state = state.client_tools_state_id is not None
    has_server_state = state.latest_server_tools_state_id is not None
    has_unassigned_state = bool(state.unassigned_tools_state_ids)
    owner_count = sum((has_client_state, has_server_state, has_unassigned_state))
    if owner_count != 1:
        state.tools_state_id = None
    elif has_client_state:
        state.tools_state_id = state.client_tools_state_id
    elif has_server_state:
        state.tools_state_id = state.latest_server_tools_state_id
    elif len(state.unassigned_tools_state_ids) == 1:
        state.tools_state_id = state.unassigned_tools_state_ids[0]
    else:
        state.tools_state_id = None


def _claim_unassigned_client_state(
    state: StreamState,
    *,
    contains_client_function_call: bool,
    incoming_tools_state_id: str | None,
) -> None:
    if (
        not contains_client_function_call
        or incoming_tools_state_id is not None
        or state.client_tools_state_id is not None
        or state.latest_server_tools_state_id is not None
        or state.active_server_tool_call_id is not None
        or state.unresolved_server_tool_call_ids
        or len(state.unassigned_tools_state_ids) != 1
    ):
        return
    state.client_tools_state_id = state.unassigned_tools_state_ids.pop()
    _refresh_replay_tools_state_id(state)


def _observe_tools_state_id(
    state: StreamState,
    *,
    value: str | None,
    owner: _ToolStateOwner | None,
) -> dict[str, Any]:
    if value is None:
        return {}
    if owner is None:
        raise RuntimeError("Primary tool-state ownership was not classified")

    if owner == "client":
        if state.client_tools_state_id not in {None, value}:
            raise ValueError(
                "Primary client function call contains multiple tools_state_id "
                "values; replay semantics are unsupported."
            )
        state.client_tools_state_id = value
        if value in state.unassigned_tools_state_ids:
            state.unassigned_tools_state_ids.remove(value)
    elif owner == "server":
        state.latest_server_tools_state_id = value
        if value in state.unassigned_tools_state_ids:
            state.unassigned_tools_state_ids.remove(value)
    elif value not in state.unassigned_tools_state_ids:
        if state.unassigned_tools_state_ids:
            raise ValueError(
                "Primary GigaChat completion contains multiple tools_state_id values; "
                "their replay semantics are unsupported."
            )
        state.unassigned_tools_state_ids.append(value)

    is_new_observation = value not in state.provider_tools_state_ids
    if is_new_observation:
        state.provider_tools_state_ids.append(value)
    _refresh_replay_tools_state_id(state)

    if is_new_observation and len(state.provider_tools_state_ids) > 1:
        return {"tools_state_id_events": [value]}
    return {}


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
    tools_state_owner: _ToolStateOwner | None,
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
        _observe_tools_state_id(
            state,
            value=provider_tools_state_id,
            owner=tools_state_owner,
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
    finish_reason_authoritative: bool,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"output_version": "v1"}
    if event_name is not None:
        metadata["events"] = [event_name]
    metadata.update(observed_metadata)

    finish_reason = event.get("finish_reason")
    if finish_reason is not None:
        if finish_reason_authoritative:
            metadata["finish_reason"] = finish_reason
        else:
            metadata["finish_reason_events"] = [
                {
                    "event": event_name,
                    "finish_reason": finish_reason,
                }
            ]
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
) -> UsageMetadata | None:
    usage_metadata = create_usage_metadata(usage_value)
    if usage_metadata is None:
        return None
    normalized = dict(usage_metadata)
    if state.usage_metadata is None:
        state.usage_metadata = normalized
        return usage_metadata
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
    completion_event = state.completion_event
    terminal_continuation = completion_event is not None
    completion_authoritative = (
        event_name == "response.message.done" and completion_event is None
    )
    finish_reason_authoritative = completion_authoritative
    if completion_event is not None:
        if event_name == "response.message.done":
            if event_data == completion_event:
                return None
            previous_finish = completion_event.get("finish_reason")
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
            if previous_finish is None and incoming_finish is not None:
                finish_reason_authoritative = True
                completion_event["finish_reason"] = incoming_finish
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
    resolved_tool_executions = resolve_tool_execution_candidates(
        collect_tool_execution_candidates(
            normalized_messages,
            response_tool_execution=top_level_tool_execution,
            response_state_id=provider_tools_state_id,
        )
    )
    resolved_by_coordinates = {
        resolved.candidate.coordinates: resolved
        for resolved in resolved_tool_executions
    }
    tool_executions = [
        _resolved_execution_value(resolved) for resolved in resolved_tool_executions
    ]
    has_server_tool_execution = bool(tool_executions)
    has_client_function_call = _event_has_client_function_call(normalized_messages)
    tools_state_owner = _tool_state_owner(
        state,
        incoming_tools_state_id=provider_tools_state_id,
        contains_client_function_call=has_client_function_call,
        contains_server_tool_execution=has_server_tool_execution,
        server_execution_requires_state=any(
            resolved.execution_id is None for resolved in resolved_tool_executions
        ),
    )
    _claim_unassigned_client_state(
        state,
        contains_client_function_call=has_client_function_call,
        incoming_tools_state_id=provider_tools_state_id,
    )
    server_identity_kwargs = (
        _server_tool_identity_update(
            state,
            incoming_tools_state_id=provider_tools_state_id,
            executions=tool_executions,
            event_name=event_name,
        )
        if tools_state_owner == "server"
        else {}
    )
    observed_metadata = _update_stream_metadata(
        state,
        provider_message_id=provider_message_id,
        provider_tools_state_id=provider_tools_state_id,
        thread_id=event_data.get("thread_id"),
        model=event_data.get("model"),
        created_at=event_data.get("created_at"),
        x_headers=x_headers,
        tools_state_owner=tools_state_owner,
    )

    request_id = _request_id(state.x_headers)
    if request_id is not None:
        state.message_id = request_id
    elif (
        event_name == "response.message.done" and state.provider_message_id is not None
    ):
        state.message_id = state.provider_message_id
    elif state.message_id is None:
        state.message_id = f"lc_primary-stream-{uuid4()}"

    content, tool_calls = _convert_messages(
        normalized_messages,
        event_name=event_name,
        incoming_tools_state_id=provider_tools_state_id,
        tools_state_owner=tools_state_owner,
        resolved_tool_executions=resolved_by_coordinates,
        state=state,
    )

    resolved_response_execution = resolved_by_coordinates.get(("response", None, None))
    if resolved_response_execution is not None:
        _close_text_block(state)
        block = _tool_execution_block(
            _resolved_execution_value(resolved_response_execution),
            event_name=event_name,
            incoming_tools_state_id=(
                provider_tools_state_id if tools_state_owner == "server" else None
            ),
            state=state,
        )
        if block is not None:
            content.append(block)

    if (
        event_name == "response.message.done"
        and state.client_tool_started
        and state.client_tools_state_id is None
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
        finish_reason_authoritative=finish_reason_authoritative,
    )
    if (
        event_name == "response.message.done"
        and state.provider_tools_state_ids
        and "final_tools_state" not in state.emitted_metadata_fields
    ):
        state.emitted_metadata_fields.add("final_tools_state")
        response_metadata["tools_state_ids"] = list(state.provider_tools_state_ids)
        if state.tools_state_id is not None:
            response_metadata["tools_state_id"] = state.tools_state_id

    usage_metadata = _usage_update(state, event_data.get("usage"))
    generation_info = None
    if finish_reason_authoritative and event_data.get("finish_reason") is not None:
        generation_info = {"finish_reason": event_data["finish_reason"]}

    has_payload = bool(
        content or tool_calls or response_metadata or usage_metadata or generation_info
    )
    if not has_payload:
        return None

    if event_name == "response.message.done" and state.completion_event is None:
        state.completion_event = dict(event_data)

    state.first_chunk = False
    additional_kwargs: dict[str, Any] = {
        **identity_kwargs,
        **server_identity_kwargs,
    }
    if event_name == "response.message.done":
        if state.tools_state_id is not None:
            additional_kwargs["tools_state_id"] = state.tools_state_id
        if state.provider_tools_state_ids:
            additional_kwargs["tools_state_ids"] = list(state.provider_tools_state_ids)
        if state.server_tool_state_ids_by_call_id:
            additional_kwargs["provider_server_tool_state_by_call_id"] = dict(
                state.server_tool_state_ids_by_call_id
            )
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
