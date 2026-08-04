"""Primary non-stream response conversion."""

from __future__ import annotations

import copy
from typing import Any, Iterable, cast

import gigachat.models as gm
from langchain_core.messages import AIMessage
from langchain_core.messages.tool import InvalidToolCall, ToolCall
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel

from langchain_gigachat.chat_models._contracts.primary.content import (
    collect_tool_execution_candidates,
    convert_function_call,
    convert_provider_file,
    convert_reasoning_value,
    convert_text_content,
    convert_tool_execution,
    create_usage_metadata,
    reasoning_content,
    request_id_from_headers,
    resolve_reasoning_value,
    resolve_tool_execution_candidates,
    unknown_provider_fields,
    validate_provider_id,
)

_MESSAGE_FIELDS = {
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
_PART_FIELDS = {
    "files",
    "function_call",
    "function_result",
    "inline_data",
    "reasoning",
    "reasoning_content",
    "text",
    "tool_execution",
}
_RESPONSE_FIELDS = {
    "additional_data",
    "created_at",
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
_ContentConversion = tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[ToolCall],
    list[InvalidToolCall],
]


def _dump(value: BaseModel | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return value.model_dump(exclude_none=True, by_alias=True)


def _without_none(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _attach_tool_context(
    blocks: list[dict[str, Any]],
    *,
    inline_data: gm.ChatInlineData | None,
    provider_data: dict[str, Any] | None = None,
) -> None:
    """Attach provider result context to its standard server-tool block."""
    if not blocks:
        return

    extras = blocks[-1].setdefault("extras", {})
    if inline_data is not None:
        extras["inline_data"] = inline_data.model_dump(
            exclude_none=True,
            by_alias=True,
        )
    if provider_data:
        extras["provider_data"] = provider_data


def _part_blocks(
    part: gm.ChatContentPart,
    *,
    role: str,
    server_tool_id: str | None,
    emit_tool_execution: bool,
    message_inline_data: gm.ChatInlineData | None,
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    inline_data = part.inline_data or message_inline_data
    unknown = unknown_provider_fields(part, _PART_FIELDS)

    if part.text is not None:
        blocks.append(
            convert_text_content(
                part.text,
                role=role,
                inline_data=inline_data,
                provider_data=unknown,
            )
        )

    for file_ in part.files or []:
        blocks.append(convert_provider_file(file_))

    if part.function_result is not None:
        blocks.append(
            {
                "type": "non_standard",
                "value": {
                    "function_result": part.function_result.model_dump(
                        exclude_none=True, by_alias=True
                    )
                },
            }
        )

    has_tool_execution = emit_tool_execution and part.tool_execution is not None
    if has_tool_execution:
        if server_tool_id is None:
            raise RuntimeError("Primary server tool identity resolution failed")
        tool_blocks = convert_tool_execution(
            part.tool_execution,
            tool_call_id=server_tool_id,
        )
        _attach_tool_context(
            tool_blocks,
            inline_data=inline_data,
            provider_data=unknown,
        )
        blocks.extend(tool_blocks)

    reasoning = resolve_reasoning_value(part)
    if reasoning is not None:
        blocks.append(convert_reasoning_value(reasoning))

    if (
        part.text is None
        and not has_tool_execution
        and reasoning is None
        and (inline_data is not None or unknown)
    ):
        value = {}
        if inline_data is not None:
            value["inline_data"] = inline_data.model_dump(
                exclude_none=True,
                by_alias=True,
            )
        if unknown:
            value.update(unknown)
        blocks.append({"type": "non_standard", "value": value})

    return blocks


def _is_plain_text_message(message: gm.ChatMessage) -> bool:
    if message.role != "assistant":
        return False
    if any(
        value is not None
        for value in (
            message.function_call,
            message.inline_data,
            message.tool_execution,
        )
    ):
        return False
    if unknown_provider_fields(message, _MESSAGE_FIELDS):
        return False
    if resolve_reasoning_value(message) is not None:
        return False

    for part in message.content or []:
        if part.text is None:
            return False
        if any(
            value is not None
            for value in (
                part.files,
                part.function_call,
                part.function_result,
                part.inline_data,
                part.tool_execution,
            )
        ):
            return False
        if unknown_provider_fields(part, _PART_FIELDS):
            return False
        if resolve_reasoning_value(part) is not None:
            return False
    return True


def _text_content(messages: Iterable[gm.ChatMessage]) -> str:
    return "".join(
        part.text or "" for message in messages for part in message.content or []
    )


def _client_tool_state_id(
    message: gm.ChatMessage,
    response: gm.ChatCompletionResponse,
) -> str | None:
    """Return the provider continuation ID used as the LangChain tool-call ID."""
    observed_state_ids = list(
        dict.fromkeys(
            state_id
            for state_id in (
                message.tools_state_id,
                getattr(response, "tools_state_id", None),
            )
            if state_id is not None
        )
    )
    if len(observed_state_ids) > 1:
        raise ValueError(
            "Primary GigaChat client function call has multiple possible "
            f"tools_state_id values: {observed_state_ids!r}."
        )
    return observed_state_ids[0] if observed_state_ids else None


def _content_blocks(
    response: gm.ChatCompletionResponse,
) -> _ContentConversion:
    blocks: list[dict[str, Any]] = []
    raw_function_calls: list[dict[str, Any]] = []
    tool_calls: list[ToolCall] = []
    invalid_tool_calls: list[InvalidToolCall] = []
    function_call_seen = False
    candidates = collect_tool_execution_candidates(
        response.messages,
        response_tool_execution=response.tool_execution,
    )
    resolved_executions = resolve_tool_execution_candidates(candidates)
    resolved_by_coordinates = {
        resolved.candidate.coordinates: resolved for resolved in resolved_executions
    }

    def append_function_call(
        function_call: gm.PrimaryChatFunctionCall,
        *,
        provider_state_id: str | None,
    ) -> None:
        nonlocal function_call_seen
        raw_function_call = function_call.model_dump(
            exclude_none=True,
            by_alias=True,
        )
        if provider_state_id is None:
            raise ValueError(
                "Primary GigaChat client function call is missing tools_state_id "
                "and cannot be replayed. Provider message_id values are not "
                "continuation state. Raw function call: "
                f"{raw_function_call!r}"
            )
        converted = convert_function_call(
            function_call,
            tool_call_id=provider_state_id,
        )
        if function_call_seen:
            raise ValueError(
                "Primary GigaChat completion contains multiple client "
                "function calls and cannot be replayed. Parallel client function "
                "calls are not supported, and duplicate-looking calls are not "
                "assumed to be mirrors."
            )

        function_call_seen = True
        raw_function_calls.append(raw_function_call)
        tool_call, invalid_call = converted
        if tool_call is not None:
            tool_calls.append(tool_call)
        if invalid_call is not None:
            invalid_tool_calls.append(invalid_call)

    for message_index, message in enumerate(response.messages):
        part_function_calls = [
            part.function_call
            for part in message.content or []
            if part.function_call is not None
        ]
        if len(part_function_calls) > 1:
            raise ValueError(
                "Primary GigaChat completion contains multiple client function "
                "calls and cannot be replayed. Parallel client function calls "
                "are not supported, and duplicate-looking calls are not assumed "
                "to be mirrors."
            )
        selected_function_call = (
            part_function_calls[0] if part_function_calls else message.function_call
        )
        if part_function_calls and message.function_call is not None:
            part_raw = part_function_calls[0].model_dump(
                exclude_none=True,
                by_alias=True,
            )
            message_raw = message.function_call.model_dump(
                exclude_none=True,
                by_alias=True,
            )
            if part_raw != message_raw:
                raise ValueError(
                    "Primary GigaChat completion contains conflicting part-level "
                    "and message-level client function calls and cannot be replayed."
                )
        if selected_function_call is not None:
            append_function_call(
                selected_function_call,
                provider_state_id=_client_tool_state_id(
                    message,
                    response,
                ),
            )

        for part_index, part in enumerate(message.content or []):
            resolved_execution = resolved_by_coordinates.get(
                ("part", message_index, part_index)
            )
            blocks.extend(
                _part_blocks(
                    part,
                    role=message.role,
                    server_tool_id=(
                        resolved_execution.tool_call_id
                        if resolved_execution is not None
                        else None
                    ),
                    emit_tool_execution=resolved_execution is not None,
                    message_inline_data=message.inline_data,
                )
            )

        resolved_message_execution = resolved_by_coordinates.get(
            ("message", message_index, None)
        )
        emitted_message_tool_execution = resolved_message_execution is not None
        if resolved_message_execution is not None:
            assert message.tool_execution is not None
            tool_blocks = convert_tool_execution(
                message.tool_execution,
                tool_call_id=resolved_message_execution.tool_call_id,
            )
            _attach_tool_context(
                tool_blocks,
                inline_data=message.inline_data,
            )
            blocks.extend(tool_blocks)

        reasoning = resolve_reasoning_value(message)
        if reasoning is not None:
            blocks.append(convert_reasoning_value(reasoning))

        if (
            not message.content
            and message.inline_data is not None
            and not emitted_message_tool_execution
        ):
            blocks.append(
                {
                    "type": "non_standard",
                    "value": {
                        "inline_data": message.inline_data.model_dump(
                            exclude_none=True, by_alias=True
                        )
                    },
                }
            )

        message_unknown = unknown_provider_fields(message, _MESSAGE_FIELDS)
        if message.role not in {"assistant", "reasoning", "tool"}:
            message_unknown["role"] = message.role
        if message_unknown:
            blocks.append({"type": "non_standard", "value": message_unknown})

    resolved_response_execution = resolved_by_coordinates.get(("response", None, None))
    if resolved_response_execution is not None:
        assert response.tool_execution is not None
        blocks.extend(
            convert_tool_execution(
                response.tool_execution,
                tool_call_id=resolved_response_execution.tool_call_id,
            )
        )

    return (
        blocks,
        raw_function_calls,
        tool_calls,
        invalid_tool_calls,
    )


def _tools_state_ids(response: gm.ChatCompletionResponse) -> list[str]:
    values = [
        getattr(response, "tools_state_id", None),
        *(message.tools_state_id for message in response.messages),
    ]
    validated = (
        validate_provider_id(value, field="tools_state_id") for value in values
    )
    return list(dict.fromkeys(value for value in validated if value is not None))


def _provider_message_ids(response: gm.ChatCompletionResponse) -> list[str]:
    values = [
        response.message_id,
        *(message.message_id for message in response.messages),
    ]
    validated = (validate_provider_id(value, field="message_id") for value in values)
    return list(dict.fromkeys(value for value in validated if value is not None))


def _validate_response_identity(response: gm.ChatCompletionResponse) -> None:
    _tools_state_ids(response)
    if len(_provider_message_ids(response)) > 1:
        raise ValueError(
            "Primary GigaChat completion contains multiple provider message_id "
            "values; their replay semantics are unsupported."
        )


def _response_metadata(
    response: gm.ChatCompletionResponse,
    *,
    finish_reason: str | None,
    x_headers: dict[str, str | None],
    replay_tools_state_id: str | None,
) -> dict[str, Any]:
    provider_message_ids = _provider_message_ids(response)
    message_id = provider_message_ids[0] if provider_message_ids else None

    tool_execution = response.tool_execution
    if tool_execution is None:
        tool_execution = next(
            (
                message.tool_execution
                for message in response.messages
                if message.tool_execution is not None
            ),
            None,
        )

    logprobs = response.logprobs or [
        logprob for message in response.messages for logprob in message.logprobs or []
    ]
    tools_state_ids = _tools_state_ids(response)
    metadata = _without_none(
        {
            "model": response.model,
            "model_name": response.model,
            "finish_reason": finish_reason,
            "created_at": response.created_at,
            "message_id": message_id,
            "thread_id": response.thread_id,
            "tools_state_id": replay_tools_state_id,
            "tools_state_ids": tools_state_ids or None,
            "provider_message_ids": provider_message_ids or None,
            "tool_execution": _dump(tool_execution),
            "logprobs": [
                item.model_dump(exclude_none=True, by_alias=True) for item in logprobs
            ]
            or None,
            "additional_data": copy.deepcopy(response.additional_data),
            "x_headers": x_headers,
        }
    )
    provider_fields = unknown_provider_fields(response, _RESPONSE_FIELDS)
    if provider_fields:
        metadata["provider_fields"] = provider_fields
    return metadata


def create_chat_result(response: gm.ChatCompletionResponse) -> ChatResult:
    """Convert one primary SDK response into one LangChain chat result.

    Primary ``response.messages`` are ordered pieces of a single completion, so
    this function always returns exactly one generation.
    """
    if not response.messages:
        raise ValueError("Primary chat completion response contains no messages")
    _validate_response_identity(response)

    finish_reason = response.finish_reason
    if finish_reason is None:
        finish_reason = next(
            (
                message.finish_reason
                for message in reversed(response.messages)
                if message.finish_reason is not None
            ),
            None,
        )

    plain_text_response = response.tool_execution is None and all(
        _is_plain_text_message(message) for message in response.messages
    )
    converted_content: _ContentConversion | None = None
    if not plain_text_response:
        converted_content = _content_blocks(response)

    tools_state_ids = _tools_state_ids(response)
    replay_tools_state_id = None
    if converted_content is not None:
        tool_calls = converted_content[2]
        invalid_tool_calls = converted_content[3]
        calls = [*tool_calls, *invalid_tool_calls]
        if calls:
            replay_tools_state_id = calls[0].get("id")

    x_headers = dict(response.x_headers or {})
    metadata = _response_metadata(
        response,
        finish_reason=finish_reason,
        x_headers=x_headers,
        replay_tools_state_id=replay_tools_state_id,
    )
    usage_metadata = create_usage_metadata(response.usage)

    additional_kwargs: dict[str, Any] = {}
    if tools_state_ids:
        additional_kwargs["tools_state_ids"] = tools_state_ids
    if replay_tools_state_id is not None:
        additional_kwargs["tools_state_id"] = replay_tools_state_id

    if converted_content is None:
        message = AIMessage(
            content=_text_content(response.messages),
            additional_kwargs=additional_kwargs,
            response_metadata=metadata,
            usage_metadata=usage_metadata,
        )
    else:
        (
            blocks,
            raw_function_calls,
            tool_calls,
            invalid_tool_calls,
        ) = converted_content
        reasoning = reasoning_content(blocks)
        if reasoning is not None:
            additional_kwargs["reasoning_content"] = reasoning
        if raw_function_calls:
            additional_kwargs["function_calls"] = raw_function_calls
            if len(raw_function_calls) == 1:
                additional_kwargs["function_call"] = raw_function_calls[0]
        message = AIMessage(
            content=cast(list[str | dict[Any, Any]], blocks),
            additional_kwargs=additional_kwargs,
            response_metadata=metadata,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
            usage_metadata=usage_metadata,
        )

    request_id = request_id_from_headers(x_headers)
    if request_id is not None:
        message.id = request_id
    elif metadata.get("message_id") is not None:
        message.id = metadata["message_id"]

    generation_info = dict(metadata)
    llm_output = {
        "token_usage": _dump(response.usage) or {},
        "model_name": response.model,
        "x_headers": x_headers,
    }
    return ChatResult(
        generations=[ChatGeneration(message=message, generation_info=generation_info)],
        llm_output=llm_output,
    )
