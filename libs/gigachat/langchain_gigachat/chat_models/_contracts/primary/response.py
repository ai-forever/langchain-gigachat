"""Primary non-stream response conversion."""

from __future__ import annotations

from typing import Any, Iterable, cast

import gigachat.models as gm
from langchain_core.messages import AIMessage
from langchain_core.messages.content import ContentBlock
from langchain_core.messages.tool import InvalidToolCall, ToolCall
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel

from langchain_gigachat.chat_models._contracts.primary.content import (
    convert_function_call,
    convert_provider_file,
    convert_text_content,
    convert_tool_execution,
    create_usage_metadata,
    reasoning_content,
    unknown_provider_fields,
)

_MESSAGE_FIELDS = {
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
_PART_FIELDS = {
    "files",
    "function_call",
    "function_result",
    "inline_data",
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


def _dump(value: BaseModel | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return value.model_dump(exclude_none=True, by_alias=True)


def _without_none(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _part_blocks(
    part: gm.ChatContentPart,
    *,
    role: str,
    server_tool_id: str,
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

    if part.tool_execution is not None:
        blocks.extend(
            convert_tool_execution(
                part.tool_execution,
                tool_call_id=server_tool_id,
            )
        )

    if part.text is None and (inline_data is not None or unknown):
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
    return True


def _text_content(messages: Iterable[gm.ChatMessage]) -> str:
    return "".join(
        part.text or "" for message in messages for part in message.content or []
    )


def _message_tool_id(
    message: gm.ChatMessage,
    response: gm.ChatCompletionResponse,
    *,
    fallback: str,
) -> str:
    response_tools_state_id = getattr(response, "tools_state_id", None)
    return (
        message.tools_state_id
        or response_tools_state_id
        or message.message_id
        or response.message_id
        or fallback
    )


def _content_blocks(
    response: gm.ChatCompletionResponse,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[ToolCall],
    list[InvalidToolCall],
]:
    blocks: list[dict[str, Any]] = []
    raw_function_calls: list[dict[str, Any]] = []
    tool_calls: list[ToolCall] = []
    invalid_tool_calls: list[InvalidToolCall] = []

    for message_index, message in enumerate(response.messages):
        tool_call_id = _message_tool_id(
            message,
            response,
            fallback=f"client_tool_{message_index}",
        )

        for part in message.content or []:
            if part.function_call is not None:
                raw_function_calls.append(
                    part.function_call.model_dump(exclude_none=True, by_alias=True)
                )
                tool_call, invalid_call = convert_function_call(
                    part.function_call,
                    tool_call_id=tool_call_id,
                )
                if tool_call is not None:
                    tool_calls.append(tool_call)
                if invalid_call is not None:
                    invalid_tool_calls.append(invalid_call)
            blocks.extend(
                _part_blocks(
                    part,
                    role=message.role,
                    server_tool_id=tool_call_id,
                    message_inline_data=message.inline_data,
                )
            )

        if message.function_call is not None:
            raw_function_calls.append(
                message.function_call.model_dump(exclude_none=True, by_alias=True)
            )
            tool_call, invalid_call = convert_function_call(
                message.function_call,
                tool_call_id=tool_call_id,
            )
            if tool_call is not None:
                tool_calls.append(tool_call)
            if invalid_call is not None:
                invalid_tool_calls.append(invalid_call)

        if message.tool_execution is not None:
            blocks.extend(
                convert_tool_execution(
                    message.tool_execution,
                    tool_call_id=tool_call_id,
                )
            )

        if not message.content and message.inline_data is not None:
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

    if response.tool_execution is not None:
        blocks.extend(
            convert_tool_execution(
                response.tool_execution,
                tool_call_id=response.message_id or "server_tool_response",
            )
        )

    return blocks, raw_function_calls, tool_calls, invalid_tool_calls


def _tools_state_ids(response: gm.ChatCompletionResponse) -> list[str]:
    values = [
        getattr(response, "tools_state_id", None),
        *(message.tools_state_id for message in response.messages),
    ]
    return list(dict.fromkeys(value for value in values if value is not None))


def _response_metadata(
    response: gm.ChatCompletionResponse,
    *,
    finish_reason: str | None,
    x_headers: dict[str, str | None],
) -> dict[str, Any]:
    provider_message_ids = [
        message.message_id
        for message in response.messages
        if message.message_id is not None
    ]
    message_id = response.message_id
    if message_id is None and provider_message_ids:
        message_id = provider_message_ids[0]

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
            "tools_state_id": tools_state_ids[0] if len(tools_state_ids) == 1 else None,
            "tools_state_ids": tools_state_ids or None,
            "provider_message_ids": provider_message_ids or None,
            "tool_execution": _dump(tool_execution),
            "logprobs": [
                item.model_dump(exclude_none=True, by_alias=True) for item in logprobs
            ]
            or None,
            "additional_data": response.additional_data,
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

    x_headers = dict(response.x_headers or {})
    metadata = _response_metadata(
        response,
        finish_reason=finish_reason,
        x_headers=x_headers,
    )
    usage_metadata = create_usage_metadata(response.usage)

    additional_kwargs: dict[str, Any] = {}
    tools_state_ids = _tools_state_ids(response)
    if tools_state_ids:
        additional_kwargs["tools_state_ids"] = tools_state_ids
        if len(tools_state_ids) == 1:
            additional_kwargs["tools_state_id"] = tools_state_ids[0]

    if response.tool_execution is None and all(
        _is_plain_text_message(message) for message in response.messages
    ):
        message = AIMessage(
            content=_text_content(response.messages),
            additional_kwargs=additional_kwargs,
            response_metadata=metadata,
            usage_metadata=usage_metadata,
        )
    else:
        blocks, raw_function_calls, tool_calls, invalid_tool_calls = _content_blocks(
            response
        )
        reasoning = reasoning_content(blocks)
        if reasoning is not None:
            additional_kwargs["reasoning_content"] = reasoning
        if raw_function_calls:
            additional_kwargs["function_calls"] = raw_function_calls
            if len(raw_function_calls) == 1:
                additional_kwargs["function_call"] = raw_function_calls[0]
        message = AIMessage(
            content_blocks=cast("list[ContentBlock]", blocks),
            additional_kwargs=additional_kwargs,
            response_metadata=metadata,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
            usage_metadata=usage_metadata,
        )

    request_id = x_headers.get("x-request-id")
    if request_id is not None:
        message.id = request_id

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
