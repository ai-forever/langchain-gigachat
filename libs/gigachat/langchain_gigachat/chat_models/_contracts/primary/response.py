"""Primary non-stream response conversion."""

from __future__ import annotations

import json
from typing import Any, Iterable, cast

import gigachat.models as gm
from langchain_core.messages import AIMessage, UsageMetadata
from langchain_core.messages.content import ContentBlock
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel

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
_TERMINAL_TOOL_STATUSES = {
    "complete",
    "completed",
    "done",
    "error",
    "failed",
    "failure",
    "success",
}
_SUCCESS_TOOL_STATUSES = {"complete", "completed", "done", "success"}


def _dump(value: BaseModel | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return value.model_dump(exclude_none=True, by_alias=True)


def _without_none(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _unknown_fields(value: BaseModel, known_fields: set[str]) -> dict[str, Any]:
    dumped = value.model_dump(exclude_none=True, by_alias=True)
    return {key: item for key, item in dumped.items() if key not in known_fields}


def _usage_metadata(usage: gm.ChatUsage | None) -> UsageMetadata | None:
    if usage is None:
        return None

    input_tokens = usage.input_tokens or 0
    output_tokens = usage.output_tokens or 0
    result = UsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=usage.total_tokens
        if usage.total_tokens is not None
        else input_tokens + output_tokens,
    )
    if (
        usage.input_tokens_details is not None
        and usage.input_tokens_details.cached_tokens is not None
    ):
        result["input_token_details"] = {
            "cache_read": usage.input_tokens_details.cached_tokens
        }
    return result


def _source_annotations(inline_data: gm.ChatInlineData | None) -> list[dict[str, Any]]:
    if inline_data is None or not inline_data.sources:
        return []

    annotations: list[dict[str, Any]] = []
    for source_id, source in inline_data.sources.items():
        annotation = _without_none(
            {
                "type": "citation",
                "id": source_id,
                "url": source.url,
                "title": source.title,
            }
        )
        if source.model_extra:
            annotation["extras"] = {"provider_data": dict(source.model_extra)}
        annotations.append(annotation)
    return annotations


def _inline_extras(inline_data: gm.ChatInlineData | None) -> dict[str, Any]:
    if inline_data is None:
        return {}

    extras = _without_none(
        {
            "images": inline_data.images,
            "widgets": inline_data.widgets,
        }
    )
    if inline_data.model_extra:
        extras.update(inline_data.model_extra)
    return extras


def _file_block(file_: gm.ChatContentFile) -> dict[str, Any]:
    mime = file_.mime
    if mime and mime.startswith("image/"):
        block_type = "image"
    elif mime and mime.startswith("audio/"):
        block_type = "audio"
    elif mime and mime.startswith("video/"):
        block_type = "video"
    else:
        block_type = "file"

    block = _without_none(
        {
            "type": block_type,
            "file_id": file_.id_,
            "mime_type": mime,
        }
    )
    extras = _without_none({"target": file_.target})
    if file_.model_extra:
        extras.update(file_.model_extra)
    if extras:
        block["extras"] = extras
    return block


def _arguments(function_call: gm.PrimaryChatFunctionCall) -> dict[str, Any]:
    arguments = function_call.arguments
    if isinstance(arguments, dict):
        return dict(arguments)
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Primary function call {function_call.name!r} has invalid JSON "
                "arguments"
            ) from error
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(
        f"Primary function call {function_call.name!r} arguments must be an object"
    )


def _tool_call_block(
    function_call: gm.PrimaryChatFunctionCall,
    *,
    tool_call_id: str | None,
) -> dict[str, Any]:
    block: dict[str, Any] = {
        "type": "tool_call",
        "id": tool_call_id,
        "name": function_call.name,
        "args": _arguments(function_call),
    }
    if function_call.model_extra:
        block["extras"] = {"provider_data": dict(function_call.model_extra)}
    return block


def _server_tool_blocks(
    execution: gm.ChatToolExecution,
    *,
    tool_call_id: str,
) -> list[dict[str, Any]]:
    raw = execution.model_dump(exclude_none=True, by_alias=True)
    status = (execution.status or "").lower()
    if status in _TERMINAL_TOOL_STATUSES:
        result: dict[str, Any] = {
            "type": "server_tool_result",
            "id": f"{tool_call_id}:result",
            "tool_call_id": tool_call_id,
            "status": "success" if status in _SUCCESS_TOOL_STATUSES else "error",
            "extras": {"provider_tool_execution": raw},
        }
        output = raw.get("output")
        if output is not None:
            result["output"] = output
        return [result]

    arguments = raw.get("arguments", raw.get("args", {}))
    return [
        {
            "type": "server_tool_call",
            "id": tool_call_id,
            "name": execution.name or "unknown",
            "args": arguments,
            "extras": {"provider_tool_execution": raw},
        }
    ]


def _part_blocks(
    part: gm.ChatContentPart,
    *,
    role: str,
    tool_call_id: str | None,
    server_tool_id: str,
    message_inline_data: gm.ChatInlineData | None,
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    inline_data = part.inline_data or message_inline_data
    inline_extras = _inline_extras(inline_data)
    unknown = _unknown_fields(part, _PART_FIELDS)

    if part.text is not None:
        if role == "reasoning":
            text_block: dict[str, Any] = {
                "type": "reasoning",
                "reasoning": part.text,
            }
        else:
            text_block = {"type": "text", "text": part.text}
            annotations = _source_annotations(inline_data)
            if annotations:
                text_block["annotations"] = annotations
        extras = {}
        if inline_extras:
            extras["inline_data"] = inline_extras
        if unknown:
            extras["provider_data"] = unknown
        if extras:
            text_block["extras"] = extras
        blocks.append(text_block)

    for file_ in part.files or []:
        blocks.append(_file_block(file_))

    if part.function_call is not None:
        blocks.append(_tool_call_block(part.function_call, tool_call_id=tool_call_id))

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
            _server_tool_blocks(part.tool_execution, tool_call_id=server_tool_id)
        )

    if not blocks:
        raw = part.model_dump(exclude_none=True, by_alias=True)
        if raw:
            blocks.append({"type": "non_standard", "value": raw})
    elif part.text is None and (inline_extras or unknown):
        value = {}
        if inline_extras:
            value["inline_data"] = inline_extras
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
    if _unknown_fields(message, _MESSAGE_FIELDS):
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
        if _unknown_fields(part, _PART_FIELDS):
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    blocks: list[dict[str, Any]] = []
    raw_function_calls: list[dict[str, Any]] = []

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
            blocks.extend(
                _part_blocks(
                    part,
                    role=message.role,
                    tool_call_id=tool_call_id,
                    server_tool_id=tool_call_id,
                    message_inline_data=message.inline_data,
                )
            )

        if message.function_call is not None:
            raw_function_calls.append(
                message.function_call.model_dump(exclude_none=True, by_alias=True)
            )
            blocks.append(
                _tool_call_block(
                    message.function_call,
                    tool_call_id=tool_call_id,
                )
            )

        if message.tool_execution is not None:
            blocks.extend(
                _server_tool_blocks(
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

        message_unknown = _unknown_fields(message, _MESSAGE_FIELDS)
        if message.role not in {"assistant", "reasoning", "tool"}:
            message_unknown["role"] = message.role
        if message_unknown:
            blocks.append({"type": "non_standard", "value": message_unknown})

    if response.tool_execution is not None:
        blocks.extend(
            _server_tool_blocks(
                response.tool_execution,
                tool_call_id=response.message_id or "server_tool_response",
            )
        )

    return blocks, raw_function_calls


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
    usage_metadata = _usage_metadata(response.usage)

    additional_kwargs: dict[str, Any] = {
        "provider_messages": [
            message.model_dump(exclude_none=True, by_alias=True)
            for message in response.messages
        ]
    }
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
        blocks, raw_function_calls = _content_blocks(response)
        if raw_function_calls:
            additional_kwargs["function_calls"] = raw_function_calls
            if len(raw_function_calls) == 1:
                additional_kwargs["function_call"] = raw_function_calls[0]
        message = AIMessage(
            content_blocks=cast("list[ContentBlock]", blocks),
            additional_kwargs=additional_kwargs,
            response_metadata=metadata,
            usage_metadata=usage_metadata,
        )

    request_id = x_headers.get("x-request-id")
    if request_id is not None:
        message.id = request_id

    generation_info = dict(metadata)
    provider_response = response.model_dump(exclude_none=True, by_alias=True)
    llm_output = {
        "token_usage": _dump(response.usage) or {},
        "model_name": response.model,
        "x_headers": x_headers,
        "provider_response": provider_response,
    }
    return ChatResult(
        generations=[ChatGeneration(message=message, generation_info=generation_info)],
        llm_output=llm_output,
    )
