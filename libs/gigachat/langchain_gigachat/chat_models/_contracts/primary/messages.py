"""LangChain-to-primary message conversion boundary."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import gigachat.models as gm
from gigachat.models.chat_completions import (
    ChatFunctionCall,
    ChatFunctionResult,
)
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import BaseModel

_ATTACHMENT_BLOCK_TYPES = frozenset({"audio", "file", "image"})
_ATTACHMENT_URL_BLOCK_TYPES = frozenset({"audio_url", "document_url", "image_url"})
_ASSISTANT_ATTACHMENT_BLOCK_TYPES = _ATTACHMENT_BLOCK_TYPES | {"video"}
_ASSISTANT_OUTPUT_ONLY_BLOCK_TYPES = frozenset(
    {
        "non_standard",
        "reasoning",
        "server_tool_call",
        "server_tool_call_chunk",
        "server_tool_result",
        "tool_call",
        "tool_call_chunk",
    }
)


def _file_part(
    block: Mapping[str, Any],
    *,
    cached_uploads: Mapping[str, str],
) -> gm.ChatContentPart:
    block_type = block.get("type")
    nested: Mapping[str, Any] = {}
    if block_type in _ATTACHMENT_URL_BLOCK_TYPES:
        candidate = block.get(str(block_type))
        if not isinstance(candidate, Mapping):
            raise ValueError(
                f"Primary {block_type!r} content must contain a mapping under "
                f"{block_type!r}."
            )
        nested = candidate

    file_id = block.get("file_id") or nested.get("giga_id")
    url = block.get("url") or nested.get("url")
    if file_id is None and isinstance(url, str):
        file_id = cached_uploads.get(hashlib.sha256(url.encode()).hexdigest())

    if not isinstance(file_id, str) or not file_id:
        raise ValueError(
            f"Primary {block_type!r} content requires a non-empty provider "
            "file_id/giga_id or a URL already present in cached_uploads."
        )

    mime = (
        block.get("mime_type")
        or block.get("mime")
        or nested.get("mime_type")
        or nested.get("mime")
    )
    if mime is None and isinstance(url, str) and url.startswith("data:"):
        mime = url.removeprefix("data:").partition(";")[0] or None
    if mime is not None and not isinstance(mime, str):
        raise ValueError(f"Primary {block_type!r} content mime_type must be a string.")

    return gm.ChatContentPart(files=[gm.ChatContentFile(id=file_id, mime=mime)])


def _convert_input_content(
    content: str | list[str | dict[str, Any]],
    *,
    cached_uploads: Mapping[str, str],
) -> list[gm.ChatContentPart]:
    if isinstance(content, str):
        return [gm.ChatContentPart(text=content)]

    parts: list[gm.ChatContentPart] = []
    for block in content:
        if isinstance(block, str):
            parts.append(gm.ChatContentPart(text=block))
            continue
        if not isinstance(block, Mapping):
            raise TypeError(
                "Primary message content items must be strings or mappings; "
                f"got {type(block).__name__}."
            )

        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text", "")
            if not isinstance(text, str):
                raise ValueError("Primary text content must contain string 'text'.")
            parts.append(gm.ChatContentPart(text=text))
        elif block_type in _ATTACHMENT_BLOCK_TYPES | _ATTACHMENT_URL_BLOCK_TYPES:
            parts.append(_file_part(block, cached_uploads=cached_uploads))
        else:
            raise ValueError(
                "Unsupported primary message content block type "
                f"{block_type!r}; expected text, image, audio, file, image_url, "
                "audio_url, or document_url."
            )
    return parts


def _convert_assistant_history_content(
    content: str | list[str | dict[str, Any]],
    *,
    cached_uploads: Mapping[str, str],
) -> list[gm.ChatContentPart]:
    """Convert replayable assistant output without inventing request fields."""
    if isinstance(content, str):
        return [gm.ChatContentPart(text=content)]

    parts: list[gm.ChatContentPart] = []
    for block in content:
        if isinstance(block, str):
            parts.append(gm.ChatContentPart(text=block))
            continue
        if not isinstance(block, Mapping):
            raise TypeError(
                "Primary assistant history content items must be strings or "
                f"mappings; got {type(block).__name__}."
            )

        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text", "")
            if not isinstance(text, str):
                raise ValueError("Primary text content must contain string 'text'.")
            parts.append(gm.ChatContentPart(text=text))
        elif block_type in (
            _ASSISTANT_ATTACHMENT_BLOCK_TYPES | _ATTACHMENT_URL_BLOCK_TYPES
        ):
            parts.append(_file_part(block, cached_uploads=cached_uploads))
        elif block_type in _ASSISTANT_OUTPUT_ONLY_BLOCK_TYPES:
            continue
        else:
            # Provider extensions have no request-side SDK representation. The
            # response adapter wraps them as non_standard, but tolerate future
            # standard output blocks here as well so stored history stays usable.
            continue
    return parts


def _additional_file_part(message: BaseMessage) -> gm.ChatContentPart | None:
    attachments = message.additional_kwargs.get("attachments")
    if attachments is None:
        return None
    if not isinstance(attachments, Sequence) or isinstance(attachments, (str, bytes)):
        raise ValueError("additional_kwargs['attachments'] must be a sequence.")

    files: list[gm.ChatContentFile] = []
    for file_id in attachments:
        if not isinstance(file_id, str) or not file_id:
            raise ValueError(
                "additional_kwargs['attachments'] must contain non-empty file IDs."
            )
        files.append(gm.ChatContentFile(id=file_id))
    return gm.ChatContentPart(files=files) if files else None


def _deduplicate_attachment_parts(
    parts: Sequence[gm.ChatContentPart],
) -> list[gm.ChatContentPart]:
    seen_file_ids: set[str] = set()
    deduplicated: list[gm.ChatContentPart] = []
    for part in parts:
        if not part.files:
            deduplicated.append(part)
            continue

        files: list[gm.ChatContentFile] = []
        for file in part.files:
            if file.id_ in seen_file_ids:
                continue
            seen_file_ids.add(file.id_)
            files.append(file.model_copy(deep=True))

        if files:
            deduplicated.append(part.model_copy(deep=True, update={"files": files}))
        elif part.model_dump(exclude={"files"}, exclude_none=True, by_alias=True):
            deduplicated.append(part.model_copy(deep=True, update={"files": None}))
    return deduplicated


def _metadata_value(message: BaseMessage, names: Sequence[str]) -> Any:
    for source in (message.additional_kwargs, message.response_metadata):
        for name in names:
            value = source.get(name)
            if value is not None:
                return value
    return None


def _message_metadata(message: BaseMessage) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    message_id = _metadata_value(message, ("message_id",))
    if message_id is not None:
        if not isinstance(message_id, str) or not message_id:
            raise ValueError("Primary message_id metadata must be a non-empty string.")
        metadata["message_id"] = message_id

    tools_state_id = _metadata_value(
        message,
        ("tools_state_id", "tool_state_id", "functions_state_id"),
    )
    if tools_state_id is not None:
        if not isinstance(tools_state_id, str) or not tools_state_id:
            raise ValueError(
                "Primary tools_state_id metadata must be a non-empty string."
            )
        metadata["tools_state_id"] = tools_state_id
    return metadata


def _tool_call_names(messages: Sequence[BaseMessage]) -> dict[str, str]:
    names: dict[str, str] = {}
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for tool_call in message.tool_calls:
            tool_call_id = tool_call.get("id")
            name = tool_call.get("name")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                continue
            if not isinstance(name, str) or not name:
                continue
            previous = names.get(tool_call_id)
            if previous is not None and previous != name:
                raise ValueError(
                    f"Tool call ID {tool_call_id!r} maps to multiple function names."
                )
            names[tool_call_id] = name
    return names


def _detach_json_tool_result(
    value: Any,
    *,
    path: str,
    active_containers: set[int],
) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                f"Primary tool result at {path} must be a finite JSON number."
            )
        return value

    if not isinstance(value, (Mapping, list)):
        raise ValueError(
            f"Primary tool result at {path} is not JSON-compatible: "
            f"{type(value).__name__}."
        )

    container_id = id(value)
    if container_id in active_containers:
        raise ValueError(f"Primary tool result at {path} contains a cyclic value.")
    active_containers.add(container_id)
    try:
        if isinstance(value, list):
            return [
                _detach_json_tool_result(
                    item,
                    path=f"{path}[{index}]",
                    active_containers=active_containers,
                )
                for index, item in enumerate(value)
            ]

        if value.get("type") == "text":
            text = value.get("text", "")
            if not isinstance(text, str):
                raise ValueError(
                    f"Primary tool result text block at {path} must contain "
                    "string 'text'."
                )
            return text

        detached: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"Primary tool result at {path} mappings must use string keys; "
                    f"got {type(key).__name__}."
                )
            detached[key] = _detach_json_tool_result(
                item,
                path=f"{path}[{key!r}]",
                active_containers=active_containers,
            )
        return detached
    finally:
        active_containers.remove(container_id)


def _normalize_tool_result(content: Any) -> Any:
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except ValueError:
            return content
    return _detach_json_tool_result(
        content,
        path="$",
        active_containers=set(),
    )


def _convert_ai_message(
    message: AIMessage,
    *,
    cached_uploads: Mapping[str, str],
) -> gm.ChatMessage:
    kwargs = _message_metadata(message)
    content = _convert_assistant_history_content(
        message.content,
        cached_uploads=cached_uploads,
    )
    additional_files = _additional_file_part(message)
    if additional_files is not None:
        content.append(additional_files)

    if len(message.tool_calls) > 1:
        raise ValueError(
            "Primary GigaChat does not support multiple client function calls "
            "in one AIMessage."
        )
    if message.tool_calls:
        tool_call = message.tool_calls[0]
        name = tool_call.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Primary AIMessage tool call requires a function name.")
        content.append(
            gm.ChatContentPart(
                function_call=ChatFunctionCall(
                    name=name,
                    arguments=copy.deepcopy(tool_call.get("args", {})),
                )
            )
        )
        tool_call_id = tool_call.get("id")
        if tool_call_id is not None:
            if not isinstance(tool_call_id, str) or not tool_call_id:
                raise ValueError("Primary AIMessage tool call ID must be a string.")
            explicit_state_id = kwargs.get("tools_state_id")
            if explicit_state_id is not None and explicit_state_id != tool_call_id:
                raise ValueError(
                    "AIMessage tool call ID conflicts with tools_state_id."
                )
            kwargs["tools_state_id"] = tool_call_id
    elif function_call := message.additional_kwargs.get("function_call"):
        if isinstance(function_call, BaseModel):
            function_call = function_call.model_dump(
                exclude_none=True,
                by_alias=True,
            )
        elif isinstance(function_call, Mapping):
            function_call = copy.deepcopy(function_call)
        else:
            raise ValueError(
                "additional_kwargs['function_call'] must be a mapping or "
                "Pydantic model."
            )
        content.append(
            gm.ChatContentPart(
                function_call=ChatFunctionCall.model_validate(function_call),
            )
        )

    return gm.ChatMessage(
        role="assistant",
        content=_deduplicate_attachment_parts(content),
        **kwargs,
    )


def _convert_tool_message(
    message: ToolMessage,
    *,
    tool_call_names: Mapping[str, str],
) -> gm.ChatMessage:
    name = message.name or tool_call_names.get(message.tool_call_id)
    if not name:
        raise ValueError(
            "Primary ToolMessage requires a function name. Set ToolMessage.name "
            "or include the preceding AIMessage tool call in the history."
        )
    return gm.ChatMessage(
        role="tool",
        tools_state_id=message.tool_call_id,
        content=[
            gm.ChatContentPart(
                function_result=ChatFunctionResult(
                    name=name,
                    result=_normalize_tool_result(message.content),
                )
            )
        ],
    )


def _convert_function_message(message: FunctionMessage) -> gm.ChatMessage:
    metadata = _message_metadata(message)
    tools_state_id = metadata.get("tools_state_id")
    if tools_state_id is None:
        raise ValueError(
            "Primary FunctionMessage requires additional_kwargs['tools_state_id']; "
            "prefer ToolMessage for unambiguous function-result roundtrips."
        )
    return gm.ChatMessage(
        role="tool",
        tools_state_id=tools_state_id,
        content=[
            gm.ChatContentPart(
                function_result=ChatFunctionResult(
                    name=message.name,
                    result=_normalize_tool_result(message.content),
                )
            )
        ],
    )


def convert_messages(
    messages: Sequence[BaseMessage],
    *,
    cached_uploads: Mapping[str, str],
) -> list[gm.ChatMessage]:
    """Convert a complete LangChain message history to primary SDK messages."""
    tool_call_names = _tool_call_names(messages)
    converted: list[gm.ChatMessage] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            converted.append(
                _convert_ai_message(message, cached_uploads=cached_uploads)
            )
            continue
        elif isinstance(message, ToolMessage):
            converted.append(
                _convert_tool_message(message, tool_call_names=tool_call_names)
            )
            continue
        elif isinstance(message, FunctionMessage):
            converted.append(_convert_function_message(message))
            continue
        elif isinstance(message, ChatMessage):
            if not message.role:
                raise ValueError("Primary ChatMessage role must be non-empty.")
            role = message.role
        else:
            raise TypeError(
                f"Unsupported primary message type {type(message).__name__}."
            )

        content = _convert_input_content(
            message.content,
            cached_uploads=cached_uploads,
        )
        additional_files = _additional_file_part(message)
        if additional_files is not None:
            content.append(additional_files)
        converted.append(
            gm.ChatMessage(
                role=role,
                content=_deduplicate_attachment_parts(content),
                **_message_metadata(message),
            )
        )
    return converted
