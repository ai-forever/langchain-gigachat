"""Convert primary named stream events into LangChain generation chunks."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any
from uuid import uuid4

import gigachat.models as gm
from langchain_core.messages import AIMessageChunk, ToolCallChunk, UsageMetadata
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.outputs import ChatGenerationChunk

from langchain_gigachat.chat_models._contracts.primary.content import (
    AMBIGUOUS_TOOL_STATE_OWNER,
    ResolvedToolExecution,
    ToolExecutionCandidate,
    ToolExecutionCoordinates,
    ToolStateOwner,
    classify_tool_state_owners,
    collect_tool_execution_candidates,
    convert_provider_file,
    convert_reasoning_value,
    convert_text_content,
    convert_tool_execution,
    create_usage_metadata,
    has_client_function_call,
    json_fragment,
    normalized_tool_execution,
    provider_dict,
    reasoning_content,
    request_id_from_headers,
    resolve_reasoning_value,
    resolve_tool_execution_candidates,
    server_tool_execution_id,
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
_EMPTY_CLIENT_TOOL_NAME = "Function call name must be a non-empty string."


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
    block = convert_reasoning_value(reasoning, index=explicit_index)
    block["index"] = _take_reasoning_block_index(
        state,
        explicit=block.get("index"),
        message_id=message_id,
    )
    return block


def _tool_call_chunk(
    function_call_value: Any,
    *,
    incoming_tools_state_id: str | None,
    state: StreamState,
) -> ToolCallChunk:
    function_call = _as_dict(function_call_value)
    incoming_name = function_call.get("name")
    if (
        incoming_name is None
        or not isinstance(incoming_name, str)
        or not incoming_name.strip()
    ):
        raise ValueError(_EMPTY_CLIENT_TOOL_NAME)
    explicit_index = function_call.get("index")

    is_first_fragment = not state.client_tool_started
    if is_first_fragment:
        call_id = incoming_tools_state_id or state.client_tools_state_id
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
            stored_call_id or incoming_tools_state_id or state.client_tools_state_id
        )
        if stored_call_id is None and call_id is not None:
            state.client_tool_id = call_id
        index = stored_index
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
    non_argument_continuation = arguments is None or (
        isinstance(arguments, str) and not arguments.strip()
    )
    if (
        not is_first_fragment
        and state.client_tool_arguments_complete
        and not non_argument_continuation
    ):
        raise ValueError(_SECOND_CLIENT_TOOL_CALL)
    if arguments is not None and not (
        state.client_tool_arguments_complete and non_argument_continuation
    ):
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
) -> list[ToolCallChunk]:
    if (
        not state.client_tool_started
        or state.client_tools_state_id is None
        or state.client_tool_index is None
    ):
        return []

    if state.client_tool_id is None:
        state.client_tool_id = state.client_tools_state_id
        return [
            tool_call_chunk(
                name=None,
                args="",
                id=state.client_tools_state_id,
                index=state.client_tool_index,
            )
        ]

    return []


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
    if execution_id is None and call_id is None:
        call_id = state.active_server_tool_call_id
    payload = normalized_tool_execution(execution)

    if call_id is not None and call_id in state.server_tool_terminal_payloads:
        previous = state.server_tool_terminal_payloads[call_id]
        if previous != payload:
            raise ValueError(
                "Conflicting repeated terminal payload or shared provider state "
                f"for primary server tool {call_id!r}."
            )
        return call_id

    if call_id is not None or execution_id is not None:
        # A resolved lifecycle without a terminal payload is receiving its first
        # terminal observation. An unseen explicit ID always starts a new
        # lifecycle; neither case may fall back to payload-only correlation.
        return None

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


def _bind_server_tool_state(
    state: StreamState,
    *,
    call_id: str,
    tools_state_id: str | None,
) -> None:
    """Reconcile every state observation, binding only idless lifecycles."""
    execution_ids = [
        execution_id
        for execution_id, mapped_call_id in (
            state.server_tool_call_ids_by_execution_id.items()
        )
        if mapped_call_id == call_id
    ]
    observed_state_ids = {
        state_id
        for state_id in (
            state.server_tool_state_ids_by_call_id.get(call_id),
            tools_state_id,
            *(
                state.server_tool_observed_state_ids_by_execution_id.get(execution_id)
                for execution_id in execution_ids
            ),
        )
        if state_id is not None
    }
    if len(observed_state_ids) > 1:
        raise ValueError(
            "Primary server tool lifecycle received conflicting "
            f"tools_state_id values: {sorted(observed_state_ids)!r}"
        )
    if not observed_state_ids:
        return

    resolved_state_id = next(iter(observed_state_ids))
    for execution_id in execution_ids:
        state.server_tool_observed_state_ids_by_execution_id[execution_id] = (
            resolved_state_id
        )

    if call_id not in state.server_tool_call_ids_with_idless_observations:
        return

    mapped_call_id = state.server_tool_call_ids_by_state_id.get(resolved_state_id)
    if mapped_call_id is not None and mapped_call_id != call_id:
        raise ValueError("Primary server tool state maps to conflicting lifecycles")

    state.server_tool_call_ids_by_state_id[resolved_state_id] = call_id
    state.server_tool_state_ids_by_call_id[call_id] = resolved_state_id
    if call_id in state.unresolved_server_tool_call_ids:
        state.unresolved_server_tool_call_ids.remove(call_id)


def _server_tool_call_id(
    execution: Mapping[str, Any],
    *,
    incoming_tools_state_id: str | None,
    has_idless_observation: bool,
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

    if has_idless_observation:
        state.server_tool_call_ids_with_idless_observations.add(call_id)

    if not terminal and call_id in state.server_tool_terminal_payloads:
        raise ValueError(
            f"Primary server tool {call_id!r} received a non-terminal update "
            "after its terminal result."
        )

    _bind_server_tool_state(
        state,
        call_id=call_id,
        tools_state_id=incoming_tools_state_id,
    )

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
) -> None:
    """Associate provider state that arrives after an idless server-tool result."""
    if incoming_tools_state_id is None:
        return
    if incoming_tools_state_id in state.server_owned_tools_state_ids:
        # This state was already observed in a server context. If the resolver
        # did not bind it then, a later state-only event must not give it to an
        # unrelated older idless lifecycle.
        return
    if any(server_tool_execution_id(execution) is not None for execution in executions):
        # The resolver retained an explicit identity for this logical execution;
        # conversion will bind a state only when an idless mirror proved it.
        return
    if incoming_tools_state_id in state.server_tool_call_ids_by_state_id:
        return
    if state.active_server_tool_call_id is not None:
        if not executions:
            _bind_server_tool_state(
                state,
                call_id=state.active_server_tool_call_id,
                tools_state_id=incoming_tools_state_id,
            )
        # An execution-bearing event binds while that execution is converted.
        return

    unresolved = [
        call_id
        for call_id in state.unresolved_server_tool_call_ids
        if call_id not in state.server_tool_state_ids_by_call_id
    ]
    if not unresolved:
        return

    terminal_payloads = [
        normalized_tool_execution(execution)
        for execution in executions
        if _is_terminal_server_execution(execution, event_name=event_name)
    ]
    if executions and not terminal_payloads:
        # A non-terminal resolved execution owns the incoming state. It is not
        # a late identity observation for an older completed lifecycle.
        return
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
            return
        unresolved = matches
    if len(unresolved) > 1:
        raise ValueError(
            "Primary server tool state arrived after multiple unresolved "
            "tool lifecycles; correlation is ambiguous."
        )

    call_id = unresolved[0]
    _bind_server_tool_state(
        state,
        call_id=call_id,
        tools_state_id=incoming_tools_state_id,
    )


def _clear_pending_server_tool_result(state: StreamState) -> None:
    state.pending_server_tool_result_ids.clear()
    state.pending_server_tool_result_ids_by_state_id.clear()


def _discard_pending_server_tool_result(
    state: StreamState,
    *,
    call_id: str,
) -> None:
    state.pending_server_tool_result_ids.discard(call_id)
    for tools_state_id, call_ids in list(
        state.pending_server_tool_result_ids_by_state_id.items()
    ):
        call_ids.discard(call_id)
        if not call_ids:
            del state.pending_server_tool_result_ids_by_state_id[tools_state_id]


def _remember_pending_server_tool_result(
    state: StreamState,
    *,
    call_id: str,
    tools_state_id: str | None,
) -> None:
    state.pending_server_tool_result_ids.add(call_id)
    if tools_state_id is not None:
        state.pending_server_tool_result_ids_by_state_id.setdefault(
            tools_state_id,
            set(),
        ).add(call_id)


def _tool_execution_block(
    execution_value: Any,
    *,
    event_name: str | None,
    incoming_tools_state_id: str | None,
    has_idless_observation: bool,
    pending_tools_state_id: str | None,
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
        if has_idless_observation:
            state.server_tool_call_ids_with_idless_observations.add(repeated_call_id)
        _bind_server_tool_state(
            state,
            call_id=repeated_call_id,
            tools_state_id=incoming_tools_state_id,
        )
        return None
    provider_state_was_bound = (
        incoming_tools_state_id is not None
        and incoming_tools_state_id in state.server_tool_call_ids_by_state_id
    )
    call_id, provider_id = _server_tool_call_id(
        execution,
        incoming_tools_state_id=incoming_tools_state_id,
        has_idless_observation=has_idless_observation,
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

    arguments = execution.get("arguments", execution.get("args"))
    emit_arguments = True
    if not terminal and arguments is not None:
        argument_mode = "fragments" if isinstance(arguments, str) else "snapshot"
        serialized_snapshot = (
            ""
            if argument_mode == "fragments"
            else json.dumps(
                arguments,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                default=str,
            )
        )
        previous_argument_state = state.server_tool_argument_state.get(call_id)
        if previous_argument_state is None:
            state.server_tool_argument_state[call_id] = (
                argument_mode,
                serialized_snapshot,
            )
        elif previous_argument_state[0] != argument_mode:
            raise ValueError(
                f"Primary server tool {call_id!r} mixes argument fragments and "
                "structured snapshots."
            )
        elif argument_mode == "snapshot":
            if previous_argument_state[1] != serialized_snapshot:
                raise ValueError(
                    f"Primary server tool {call_id!r} contains conflicting "
                    "structured argument snapshots."
                )
            emit_arguments = False

    if state.pending_server_tool_result_ids and not terminal:
        _clear_pending_server_tool_result(state)

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
    if (
        not has_execution_id
        and provider_id is not None
        and provider_id != call_id
        and not provider_state_was_bound
    ):
        block_extras = block.setdefault("extras", {})
        block_extras["provider_server_tool_state_by_call_id"] = {
            call_id: provider_id,
        }
    if terminal:
        payload = normalized_tool_execution(execution)
        previous_payload = state.server_tool_terminal_payloads.get(call_id)
        if previous_payload is not None and previous_payload != payload:
            raise ValueError(
                f"Conflicting repeated terminal payload for primary server tool "
                f"{call_id!r}."
            )
        state.server_tool_terminal_payloads[call_id] = payload
        if failed:
            _clear_pending_server_tool_result(state)
        else:
            _remember_pending_server_tool_result(
                state,
                call_id=call_id,
                tools_state_id=pending_tools_state_id,
            )
        return block

    if existing_index is None:
        block["name"] = incoming_name or ""
    else:
        block["name"] = incoming_name if known_name is None and incoming_name else ""
        block_extras = block.setdefault("extras", {})
        block_extras.pop("provider_tool_execution", None)
        block_extras["provider_tool_execution_updates"] = [
            {key: value for key, value in execution.items() if key != "index"}
        ]
    if arguments is None or not emit_arguments:
        block["args"] = ""
    return block


def _pending_server_tool_result_update(
    inline_data_value: Any,
    *,
    incoming_tools_state_id: str | None,
    provider_data: Mapping[str, Any],
    state: StreamState,
) -> dict[str, Any] | None:
    if incoming_tools_state_id is not None:
        call_id = state.server_tool_call_ids_by_state_id.get(incoming_tools_state_id)
        if call_id is None:
            hinted_call_ids = (
                state.pending_server_tool_result_ids_by_state_id.get(
                    incoming_tools_state_id,
                    set(),
                )
                & state.pending_server_tool_result_ids
            )
            call_id = next(iter(hinted_call_ids)) if len(hinted_call_ids) == 1 else None
        if call_id is None or call_id not in state.pending_server_tool_result_ids:
            return None
    elif len(state.pending_server_tool_result_ids) == 1:
        call_id = next(iter(state.pending_server_tool_result_ids))
    else:
        return None

    index = state.server_tool_result_indexes.get(call_id)
    if index is None:
        return None

    value: dict[str, Any] = {
        "inline_data": _as_dict(inline_data_value),
    }
    if provider_data:
        value["provider_data"] = dict(provider_data)
    _discard_pending_server_tool_result(state, call_id=call_id)
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
    pending_server_tools_state_id: str | None,
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
            incoming_tools_state_id=resolved_tool_execution.provider_state_id,
            has_idless_observation=(resolved_tool_execution.has_idless_observation),
            pending_tools_state_id=pending_server_tools_state_id,
            state=state,
        )
        if tool_execution_block is not None:
            block_extras = tool_execution_block.setdefault("extras", {})
            if inline_data_value is not None:
                block_extras["inline_data"] = _as_dict(inline_data_value)
                pending_call_id = tool_execution_block.get("tool_call_id")
                if isinstance(pending_call_id, str):
                    _discard_pending_server_tool_result(
                        state,
                        call_id=pending_call_id,
                    )
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

    reasoning = resolve_reasoning_value(part)
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
                incoming_tools_state_id=incoming_server_tools_state_id,
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
    top_level_tools_state_id: str | None,
    tools_state_owners: Mapping[str, ToolStateOwner],
    resolved_tool_executions: Mapping[
        ToolExecutionCoordinates,
        ResolvedToolExecution,
    ],
    state: StreamState,
) -> tuple[list[str | dict[str, Any]], list[ToolCallChunk]]:
    content: list[str | dict[str, Any]] = []
    tool_calls: list[ToolCallChunk] = []
    resolved_provider_state_ids = {
        resolved.provider_state_id
        for resolved in resolved_tool_executions.values()
        if resolved.provider_state_id is not None
    }
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
        message_id = validate_provider_id(
            message.get("message_id"),
            field="message_id",
        )
        if message_id is None:
            message_id = state.provider_message_id
        tool_state_id = validate_provider_id(
            message.get("tools_state_id"),
            field="tools_state_id",
        )
        if tool_state_id is None:
            tool_state_id = top_level_tools_state_id
        tool_state_owner = (
            tools_state_owners.get(tool_state_id) if tool_state_id is not None else None
        )
        pending_server_tools_state_id = (
            tool_state_id
            if tool_state_owner == "server"
            and tool_state_id not in resolved_provider_state_ids
            else None
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
                    tool_state_id if tool_state_owner == "client" else None
                ),
                incoming_server_tools_state_id=(
                    tool_state_id if tool_state_owner == "server" else None
                ),
                pending_server_tools_state_id=pending_server_tools_state_id,
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
                        tool_state_id if tool_state_owner == "client" else None
                    ),
                    state=state,
                )
            )
        resolved_message_execution = resolved_tool_executions.get(
            ("message", message_index, None)
        )
        message_tool_execution_emitted = False
        if resolved_message_execution is not None:
            _close_text_block(state)
            block = _tool_execution_block(
                _resolved_execution_value(resolved_message_execution),
                event_name=event_name,
                incoming_tools_state_id=resolved_message_execution.provider_state_id,
                has_idless_observation=(
                    resolved_message_execution.has_idless_observation
                ),
                pending_tools_state_id=pending_server_tools_state_id,
                state=state,
            )
            if block is not None:
                if message_inline_data is not None:
                    block_extras = block.setdefault("extras", {})
                    block_extras["inline_data"] = _as_dict(message_inline_data)
                    pending_call_id = block.get("tool_call_id")
                    if isinstance(pending_call_id, str):
                        _discard_pending_server_tool_result(
                            state,
                            call_id=pending_call_id,
                        )
                content.append(block)
                message_tool_execution_emitted = True

        reasoning = resolve_reasoning_value(message)
        if reasoning is not None:
            content.append(
                _reasoning_block(
                    reasoning,
                    state=state,
                    message_id=message_id,
                )
            )

        if (
            not message.get("content")
            and message_inline_data is not None
            and not message_tool_execution_emitted
        ):
            _close_text_block(state)
            result_update = _pending_server_tool_result_update(
                message_inline_data,
                incoming_tools_state_id=(
                    tool_state_id if tool_state_owner == "server" else None
                ),
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
    return any(has_client_function_call(message) for message in messages)


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
    incoming_tools_state_id: str,
    local_owner: ToolStateOwner,
) -> ToolStateOwner:
    known_client_owner = incoming_tools_state_id == state.client_tools_state_id
    known_server_owner = (
        incoming_tools_state_id in state.server_tool_call_ids_by_state_id
        or incoming_tools_state_id in state.server_owned_tools_state_ids
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
    if (
        incoming_tools_state_id in state.unassigned_tools_state_ids
        and len(state.unassigned_tools_state_ids) > 1
        and local_owner == "unassigned"
        and (
            known_client_owner
            or known_server_owner
            or unresolved_client_owner
            or unresolved_server_owner
        )
    ):
        raise ValueError(
            "Primary tool lifecycle cannot claim one of multiple unassigned "
            "tools_state_id values."
        )

    if local_owner == "client":
        if known_server_owner:
            raise ValueError(AMBIGUOUS_TOOL_STATE_OWNER)
        return "client"
    if local_owner == "server":
        if known_client_owner:
            raise ValueError(AMBIGUOUS_TOOL_STATE_OWNER)
        return "server"
    if known_client_owner and known_server_owner:
        raise ValueError(AMBIGUOUS_TOOL_STATE_OWNER)
    if known_client_owner:
        return "client"
    if known_server_owner:
        return "server"

    client_plausible = unresolved_client_owner
    server_plausible = unresolved_server_owner
    if client_plausible and server_plausible:
        raise ValueError(AMBIGUOUS_TOOL_STATE_OWNER)
    if client_plausible:
        return "client"
    if server_plausible:
        return "server"
    return "unassigned"


def _event_tool_state_context(
    state: StreamState,
    *,
    messages: Sequence[Mapping[str, Any]],
    candidates: Sequence[ToolExecutionCandidate],
    resolved_executions: Sequence[ResolvedToolExecution],
    top_level_tools_state_id: str | None,
) -> tuple[
    dict[str, ToolStateOwner],
    bool,
    dict[str, list[dict[str, Any]]],
]:
    """Classify each event or message state without collapsing its scope."""
    local_owners = classify_tool_state_owners(
        messages=messages,
        candidates=candidates,
        resolved_executions=resolved_executions,
        top_level_tools_state_id=top_level_tools_state_id,
    )
    client_state_ids = {
        state_id for state_id, owner in local_owners.items() if owner == "client"
    }
    client_without_state = False
    for message in messages:
        if not has_client_function_call(message):
            continue
        message_state = message.get("tools_state_id")
        client_state_id = (
            str(message_state)
            if message_state is not None
            else top_level_tools_state_id
        )
        if client_state_id is None:
            client_without_state = True
        else:
            client_state_ids.add(client_state_id)

    server_state_ids = {
        candidate.container_state_id
        for candidate in candidates
        if candidate.container_state_id is not None
    }
    executions_by_state: dict[str, list[dict[str, Any]]] = {}
    for resolved in resolved_executions:
        state_id = resolved.provider_state_id
        if state_id is not None:
            executions_by_state.setdefault(state_id, []).append(
                _resolved_execution_value(resolved)
            )
    for state_id in server_state_ids:
        executions_by_state.setdefault(state_id, [])

    owners: dict[str, ToolStateOwner] = {}
    for state_id, local_owner in local_owners.items():
        owner = _tool_state_owner(
            state,
            incoming_tools_state_id=state_id,
            local_owner=local_owner,
        )
        owners[state_id] = owner
    if (
        top_level_tools_state_id is not None
        and client_state_ids
        and top_level_tools_state_id not in client_state_ids
        and owners[top_level_tools_state_id] != "server"
    ):
        raise ValueError(
            "Primary GigaChat client function call has multiple possible "
            "tools_state_id values."
        )
    return owners, client_without_state, executions_by_state


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
    client_has_incoming_state: bool,
) -> None:
    if (
        not contains_client_function_call
        or client_has_incoming_state
        or state.client_tools_state_id is not None
        or state.latest_server_tools_state_id is not None
        or state.active_server_tool_call_id is not None
        or state.unresolved_server_tool_call_ids
    ):
        return
    if len(state.unassigned_tools_state_ids) > 1:
        raise ValueError(
            "Primary client function call cannot claim one of multiple "
            "unassigned tools_state_id values."
        )
    if not state.unassigned_tools_state_ids:
        return
    state.client_tools_state_id = state.unassigned_tools_state_ids.pop()
    _refresh_replay_tools_state_id(state)


def _claim_unassigned_server_state(
    state: StreamState,
    *,
    resolved_executions: Sequence[ResolvedToolExecution],
    client_without_state: bool,
) -> list[ResolvedToolExecution]:
    """Bind one earlier unassigned state to one current idless server tool."""
    unbound_executions = [
        resolved
        for resolved in resolved_executions
        if resolved.execution_id is None and resolved.provider_state_id is None
    ]
    if not state.unassigned_tools_state_ids or not unbound_executions:
        return list(resolved_executions)
    if client_without_state:
        raise ValueError(AMBIGUOUS_TOOL_STATE_OWNER)
    if len(state.unassigned_tools_state_ids) > 1:
        raise ValueError(
            "Primary idless server tool cannot claim one of multiple unassigned "
            "tools_state_id values."
        )
    if len(unbound_executions) > 1:
        raise ValueError(
            "Primary multiple idless server tool lifecycles cannot claim one "
            "unassigned tools_state_id value."
        )

    tools_state_id = state.unassigned_tools_state_ids[0]
    target = unbound_executions[0]
    _observe_tools_state_id(
        state,
        value=tools_state_id,
        owner="server",
    )
    return [
        replace(resolved, provider_state_id=tools_state_id)
        if resolved is target
        else resolved
        for resolved in resolved_executions
    ]


def _observe_tools_state_id(
    state: StreamState,
    *,
    value: str | None,
    owner: ToolStateOwner | None,
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
        state.server_owned_tools_state_ids.add(value)
        if value in state.unassigned_tools_state_ids:
            state.unassigned_tools_state_ids.remove(value)
    elif value not in state.unassigned_tools_state_ids:
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
    provider_tool_state_owners: Mapping[str, ToolStateOwner],
    final_server_tools_state_id: str | None,
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
    tools_state_id_events: list[str] = []
    for tools_state_id, owner in provider_tool_state_owners.items():
        observation = _observe_tools_state_id(
            state,
            value=tools_state_id,
            owner=owner,
        )
        tools_state_id_events.extend(observation.get("tools_state_id_events", []))
    if tools_state_id_events:
        metadata["tools_state_id_events"] = tools_state_id_events
    if final_server_tools_state_id is not None:
        state.latest_server_tools_state_id = final_server_tools_state_id
        _refresh_replay_tools_state_id(state)
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
    validated = (validate_provider_id(value, field="message_id") for value in values)
    ids = list(dict.fromkeys(value for value in validated if value is not None))
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
    event: gm.PrimaryChatCompletionChunk,
    *,
    state: StreamState,
) -> ChatGenerationChunk | None:
    """Convert one SDK named event while preserving identity and block order."""
    if not isinstance(event, gm.PrimaryChatCompletionChunk):
        raise TypeError(
            "Primary stream conversion requires an SDK "
            "PrimaryChatCompletionChunk instance."
        )
    event_data = _as_dict(event)
    if not event_data:
        return None

    event_name_value = event_data.get("event")
    event_name = str(event_name_value) if event_name_value is not None else None
    if event_name == "response.error":
        _clear_pending_server_tool_result(state)
        raise PrimaryStreamError(event_data)

    normalized_messages = _as_dict_list(
        event_data.get("messages"),
        field="messages",
    )
    if "messages" in event_data:
        event_data["messages"] = normalized_messages
    x_headers = _optional_dict(event_data.get("x_headers"))
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
        late_message_id = validate_provider_id(
            event_data.get("message_id"),
            field="message_id",
        )
        if late_message_id is not None and late_message_id != state.provider_message_id:
            raise ValueError(
                "Primary stream received a new message_id after response.message.done"
            )
        late_tools_state_id = validate_provider_id(
            event_data.get("tools_state_id"),
            field="tools_state_id",
        )
        if (
            late_tools_state_id is not None
            and late_tools_state_id not in state.provider_tools_state_ids
        ):
            raise ValueError(
                "Primary stream received a new tools_state_id after "
                "response.message.done"
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
    top_level_tools_state_id = validate_provider_id(
        event_data.get("tools_state_id"),
        field="tools_state_id",
    )
    tool_execution_candidates = collect_tool_execution_candidates(
        normalized_messages,
        response_tool_execution=top_level_tool_execution,
        response_state_id=top_level_tools_state_id,
    )
    resolved_tool_executions = resolve_tool_execution_candidates(
        tool_execution_candidates
    )
    has_client_function_call = _event_has_client_function_call(normalized_messages)
    (
        tool_state_owners,
        client_without_state,
        server_executions_by_state,
    ) = _event_tool_state_context(
        state,
        messages=normalized_messages,
        candidates=tool_execution_candidates,
        resolved_executions=resolved_tool_executions,
        top_level_tools_state_id=top_level_tools_state_id,
    )
    for candidate in tool_execution_candidates:
        if (
            candidate.execution_id is None
            or candidate.container_state_id is None
            or tool_state_owners.get(candidate.container_state_id) != "server"
        ):
            continue
        previous_state_id = state.server_tool_observed_state_ids_by_execution_id.get(
            candidate.execution_id
        )
        if (
            previous_state_id is not None
            and previous_state_id != candidate.container_state_id
        ):
            raise ValueError(
                "Primary GigaChat server tools use the same provider identity "
                f"{candidate.execution_id!r} with multiple tools_state_id values: "
                f"{sorted([previous_state_id, candidate.container_state_id])!r}."
            )
        state.server_tool_observed_state_ids_by_execution_id[candidate.execution_id] = (
            candidate.container_state_id
        )
    resolved_tool_executions = _claim_unassigned_server_state(
        state,
        resolved_executions=resolved_tool_executions,
        client_without_state=has_client_function_call and client_without_state,
    )
    _claim_unassigned_client_state(
        state,
        contains_client_function_call=has_client_function_call,
        client_has_incoming_state=not client_without_state,
    )
    state_only_server_candidates = [
        tools_state_id
        for tools_state_id, owner in tool_state_owners.items()
        if owner == "server"
        and server_executions_by_state.get(tools_state_id) is None
        and tools_state_id not in state.server_tool_call_ids_by_state_id
    ]
    has_unbound_server_lifecycle = (
        state.active_server_tool_call_id is not None
        and state.active_server_tool_call_id
        not in state.server_tool_state_ids_by_call_id
    ) or any(
        call_id not in state.server_tool_state_ids_by_call_id
        for call_id in state.unresolved_server_tool_call_ids
    )
    if len(state_only_server_candidates) > 1 and has_unbound_server_lifecycle:
        raise ValueError(
            "Primary multiple state-only tools_state_id values cannot identify "
            "one unresolved server tool lifecycle."
        )
    resolved_by_coordinates = {
        resolved.candidate.coordinates: resolved
        for resolved in resolved_tool_executions
    }
    resolved_provider_state_ids = {
        resolved.provider_state_id
        for resolved in resolved_tool_executions
        if resolved.provider_state_id is not None
    }
    for tools_state_id, owner in tool_state_owners.items():
        if owner != "server":
            continue
        state_executions = server_executions_by_state.get(tools_state_id)
        if state_executions == []:
            # This event contains server executions tagged with the state, but
            # the resolver intentionally assigned it to none of them (for
            # example, an explicit-only execution). Do not give it to an older
            # unresolved lifecycle.
            continue
        _server_tool_identity_update(
            state,
            incoming_tools_state_id=tools_state_id,
            executions=state_executions or [],
            event_name=event_name,
        )
    observed_metadata = _update_stream_metadata(
        state,
        provider_message_id=provider_message_id,
        provider_tool_state_owners=tool_state_owners,
        final_server_tools_state_id=(
            top_level_tools_state_id
            if top_level_tools_state_id is not None
            and tool_state_owners.get(top_level_tools_state_id) == "server"
            else None
        ),
        thread_id=event_data.get("thread_id"),
        model=event_data.get("model"),
        created_at=event_data.get("created_at"),
        x_headers=x_headers,
    )

    request_id = request_id_from_headers(state.x_headers)
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
        top_level_tools_state_id=top_level_tools_state_id,
        tools_state_owners=tool_state_owners,
        resolved_tool_executions=resolved_by_coordinates,
        state=state,
    )

    resolved_response_execution = resolved_by_coordinates.get(("response", None, None))
    if resolved_response_execution is not None:
        _close_text_block(state)
        block = _tool_execution_block(
            _resolved_execution_value(resolved_response_execution),
            event_name=event_name,
            incoming_tools_state_id=resolved_response_execution.provider_state_id,
            has_idless_observation=(resolved_response_execution.has_idless_observation),
            pending_tools_state_id=(
                top_level_tools_state_id
                if top_level_tools_state_id is not None
                and tool_state_owners.get(top_level_tools_state_id) == "server"
                and top_level_tools_state_id not in resolved_provider_state_ids
                else None
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

    identity_chunks = _client_tool_identity_update(state)
    tool_calls.extend(identity_chunks)

    if event_name in {
        "response.message.done",
        "response.tool.failed",
        "response.error",
    }:
        _clear_pending_server_tool_result(state)

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

    additional_kwargs: dict[str, Any] = {}
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
    return ChatGenerationChunk(
        message=message,
        generation_info=generation_info,
    )
