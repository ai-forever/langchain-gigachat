"""Tests for _convert_delta_to_message_chunk and _build_stream_chunk."""

import json
from typing import Any, AsyncIterator, Dict, Iterator

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import (
    AIMessageChunk,
    ChatMessageChunk,
    FunctionMessageChunk,
    HumanMessageChunk,
    SystemMessageChunk,
)
from pytest_mock import MockerFixture

from langchain_gigachat.chat_models.gigachat import (
    GigaChat,
    _convert_delta_to_message_chunk,
)

# ---------------------------------------------------------------------------
# _convert_delta_to_message_chunk
# ---------------------------------------------------------------------------


def test_delta_user_role() -> None:
    chunk = _convert_delta_to_message_chunk(
        {"role": "user", "content": "hello"}, AIMessageChunk
    )
    assert isinstance(chunk, HumanMessageChunk)
    assert chunk.content == "hello"


def test_delta_system_role() -> None:
    chunk = _convert_delta_to_message_chunk(
        {"role": "system", "content": "sys"}, SystemMessageChunk
    )
    assert isinstance(chunk, SystemMessageChunk)


def test_delta_function_role() -> None:
    chunk = _convert_delta_to_message_chunk(
        {"role": "function", "content": '{"r":1}', "name": "calc"},
        FunctionMessageChunk,
    )
    assert isinstance(chunk, FunctionMessageChunk)
    assert chunk.name == "calc"


def test_delta_assistant_role() -> None:
    chunk = _convert_delta_to_message_chunk(
        {"role": "assistant", "content": "hi"}, AIMessageChunk
    )
    assert isinstance(chunk, AIMessageChunk)
    assert chunk.content == "hi"


def test_delta_assistant_default_class_takes_priority() -> None:
    """When default_class is AIMessageChunk, assistant branch wins over system role."""
    chunk = _convert_delta_to_message_chunk(
        {"role": "system", "content": "sys"}, AIMessageChunk
    )
    assert isinstance(chunk, AIMessageChunk)


def test_delta_default_class_fallback() -> None:
    chunk = _convert_delta_to_message_chunk({"content": "text"}, HumanMessageChunk)
    assert isinstance(chunk, HumanMessageChunk)


def test_delta_chat_message_chunk_unknown_role() -> None:
    chunk = _convert_delta_to_message_chunk(
        {"role": "custom_role", "content": "c"}, ChatMessageChunk
    )
    assert isinstance(chunk, ChatMessageChunk)
    assert chunk.role == "custom_role"


# ---------------------------------------------------------------------------
# _convert_delta_to_message_chunk — function_call
# ---------------------------------------------------------------------------


def test_delta_with_function_call() -> None:
    delta: Dict[str, Any] = {
        "role": "assistant",
        "content": "",
        "function_call": {"name": "my_tool", "arguments": {"a": 1}},
    }
    chunk = _convert_delta_to_message_chunk(delta, AIMessageChunk)
    assert isinstance(chunk, AIMessageChunk)
    assert chunk.additional_kwargs["function_call"]["name"] == "my_tool"
    assert len(chunk.tool_call_chunks) == 1
    assert chunk.tool_call_chunks[0]["name"] == "my_tool"
    args = chunk.tool_call_chunks[0]["args"]
    assert args is not None
    assert json.loads(args) == {"a": 1}


def test_delta_function_call_name_none() -> None:
    delta: Dict[str, Any] = {
        "role": "assistant",
        "content": "",
        "function_call": {"name": None, "arguments": {}},
    }
    chunk = _convert_delta_to_message_chunk(delta, AIMessageChunk)
    assert chunk.additional_kwargs["function_call"]["name"] == ""


# ---------------------------------------------------------------------------
# _convert_delta_to_message_chunk — functions_state_id
# ---------------------------------------------------------------------------


def test_delta_functions_state_id() -> None:
    delta: Dict[str, Any] = {
        "content": "text",
        "functions_state_id": "sid-1",
    }
    chunk = _convert_delta_to_message_chunk(delta, AIMessageChunk)
    assert isinstance(chunk, AIMessageChunk)
    assert chunk.additional_kwargs["functions_state_id"] == "sid-1"


# ---------------------------------------------------------------------------
# _convert_delta_to_message_chunk — reasoning_content
# ---------------------------------------------------------------------------


def test_delta_reasoning_content() -> None:
    delta: Dict[str, Any] = {
        "role": "assistant",
        "content": "answer",
        "reasoning_content": "thinking...",
    }
    chunk = _convert_delta_to_message_chunk(delta, AIMessageChunk)
    assert chunk.additional_kwargs["reasoning_content"] == "thinking..."


# ---------------------------------------------------------------------------
# _convert_delta_to_message_chunk — image regex in content
# ---------------------------------------------------------------------------


def test_delta_image_regex() -> None:
    content = '<img src="uuid-img" fuse="true"/>text after'
    delta: Dict[str, Any] = {"role": "assistant", "content": content}
    chunk = _convert_delta_to_message_chunk(delta, AIMessageChunk)
    assert chunk.additional_kwargs["image_uuid"] == "uuid-img"
    assert chunk.additional_kwargs["postfix_message"] == "text after"


def test_delta_video_regex() -> None:
    content = '<video cover="cover-1" src="video-1" fuse="true"/>after'
    delta: Dict[str, Any] = {"role": "assistant", "content": content}
    chunk = _convert_delta_to_message_chunk(delta, AIMessageChunk)
    assert chunk.additional_kwargs["cover_uuid"] == "cover-1"
    assert chunk.additional_kwargs["video_uuid"] == "video-1"


# ---------------------------------------------------------------------------
# _build_stream_chunk
# ---------------------------------------------------------------------------


@pytest.fixture()
def gigachat_instance(mocker: MockerFixture) -> GigaChat:
    mocker.patch("gigachat.GigaChat")
    return GigaChat()


def test_build_stream_chunk_first_chunk(gigachat_instance: GigaChat) -> None:
    raw_chunk: Dict[str, Any] = {
        "choices": [{"delta": {"role": "assistant", "content": "hi"}, "index": 0}],
        "created": 123,
        "model": "GigaChat:v1",
        "object": "chat.completion",
        "x_headers": {"x-request-id": "req-1"},
    }
    chunk_m, gen_info, content = gigachat_instance._build_stream_chunk(
        raw_chunk, first_chunk=True
    )
    assert isinstance(chunk_m, AIMessageChunk)
    assert chunk_m.id == "req-1"
    assert gen_info["x_headers"] == {"x-request-id": "req-1"}


def test_build_stream_chunk_not_first_chunk(gigachat_instance: GigaChat) -> None:
    raw_chunk: Dict[str, Any] = {
        "choices": [{"delta": {"role": "assistant", "content": "more"}, "index": 0}],
        "created": 123,
        "model": "GigaChat:v1",
        "object": "chat.completion",
    }
    _, gen_info, _ = gigachat_instance._build_stream_chunk(raw_chunk, first_chunk=False)
    assert "x_headers" not in gen_info


def test_build_stream_chunk_with_finish_reason(gigachat_instance: GigaChat) -> None:
    raw_chunk: Dict[str, Any] = {
        "choices": [
            {
                "delta": {"role": "assistant", "content": "done"},
                "index": 0,
                "finish_reason": "stop",
            }
        ],
        "created": 123,
        "model": "GigaChat:v1",
        "object": "chat.completion",
    }
    _, gen_info, _ = gigachat_instance._build_stream_chunk(raw_chunk, first_chunk=False)
    assert gen_info["finish_reason"] == "stop"
    assert gen_info["model_name"] == "GigaChat:v1"


def test_build_stream_chunk_with_usage(gigachat_instance: GigaChat) -> None:
    raw_chunk: Dict[str, Any] = {
        "choices": [{"delta": {"role": "assistant", "content": ""}, "index": 0}],
        "created": 123,
        "model": "GigaChat:v1",
        "object": "chat.completion",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "precached_prompt_tokens": 5,
        },
    }
    chunk_m, _, _ = gigachat_instance._build_stream_chunk(raw_chunk, first_chunk=False)
    assert isinstance(chunk_m, AIMessageChunk)
    assert chunk_m.usage_metadata is not None
    assert chunk_m.usage_metadata["input_tokens"] == 15
    assert chunk_m.usage_metadata["output_tokens"] == 20
    assert chunk_m.usage_metadata["total_tokens"] == 35


def test_build_stream_chunk_without_content_uses_empty_string(
    gigachat_instance: GigaChat,
) -> None:
    raw_chunk: Dict[str, Any] = {
        "choices": [
            {
                "delta": {
                    "role": "assistant",
                    "function_call": {"name": "my_tool", "arguments": {}},
                },
                "index": 0,
            }
        ],
        "created": 123,
        "model": "GigaChat:v1",
        "object": "chat.completion",
    }
    _, _, content = gigachat_instance._build_stream_chunk(raw_chunk, first_chunk=True)
    assert content == ""


class _TokenRecorder(BaseCallbackHandler):
    def __init__(self) -> None:
        self.tokens: list[str] = []

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.tokens.append(token)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "ending", ["finish", "eof", "metadata_only", "after_finish", "cleared"]
)
async def test_legacy_public_stream_uses_latest_metadata_snapshots(
    mocker: MockerFixture, asynchronous: bool, ending: str
) -> None:
    """Response snapshots must not concatenate during public chunk aggregation."""
    initial = {
        "thread_id": "thread-1",
        "message_id": "message-1",
        "additional_data": {
            "sources": [{"index": 0, "url": "https://example.com", "title": "Draft"}],
            "obsolete": True,
        },
        "error_details": {"http_status": 102, "message": "Pending"},
    }
    final = {
        "thread_id": "thread-1",
        "message_id": "message-1",
        "additional_data": {
            "sources": [{"index": 0, "url": "https://example.com", "title": "Final"}]
        },
        "error_details": {"http_status": 200, "message": "Complete"},
    }
    if ending == "cleared":
        final.update(additional_data={}, error_details={})
    usage = {
        "prompt_tokens": 5,
        "precached_prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 8,
    }
    raw_chunks: list[dict[str, Any]] = [
        {
            "choices": [{"delta": {"content": "Hello "}}],
            "x_headers": {"x-request-id": "request-1"},
            **initial,
        },
        {"choices": [{"delta": {"content": "world"}}], **initial},
    ]
    if ending == "eof":
        raw_chunks[-1].update(final, usage=usage)
    elif ending in ("metadata_only", "after_finish", "cleared"):
        if ending == "after_finish":
            raw_chunks[-1]["choices"][0]["finish_reason"] = "stop"
        raw_chunks.append({"choices": [], "usage": usage, **final})
    else:
        raw_chunks.append(
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": usage,
                **final,
            }
        )
    for raw_chunk in raw_chunks:
        raw_chunk["model"] = "test-model"
    consumed = 0

    def stream(*args: Any) -> Iterator[dict[str, Any]]:
        nonlocal consumed
        for raw_chunk in raw_chunks:
            consumed += 1
            yield raw_chunk

    async def astream(*args: Any) -> AsyncIterator[dict[str, Any]]:
        for raw_chunk in stream():
            yield raw_chunk

    sdk_client = mocker.patch("gigachat.GigaChat").return_value
    sdk_client.stream.side_effect = stream
    sdk_client.astream.side_effect = astream
    recorder = _TokenRecorder()
    llm = GigaChat(model="test-model", callbacks=[recorder])
    if asynchronous:
        async_iterator = llm.astream("hello")
        first = await anext(async_iterator)
        assert consumed == 1, "Text must be yielded before the SDK stream ends"
        chunks = [first, *[chunk async for chunk in async_iterator]]
    else:
        iterator = llm.stream("hello")
        first = next(iterator)
        assert consumed == 1, "Text must be yielded before the SDK stream ends"
        chunks = [first, *iterator]

    assert first.content == "Hello "
    assert all(
        not (chunk.response_metadata.keys() & final.keys()) for chunk in chunks[:-1]
    )
    assert chunks[-1].response_metadata == final
    assert chunks[-1].id == "request-1"
    assert chunks[-1].chunk_position == "last"
    aggregate = chunks[0]
    for chunk in chunks[1:]:
        aggregate += chunk
    assert aggregate.content == "Hello world"
    assert aggregate.id == "request-1"
    assert all(
        aggregate.response_metadata[key] == value for key, value in final.items()
    )
    assert aggregate.response_metadata["x_headers"] == {"x-request-id": "request-1"}
    assert aggregate.usage_metadata == {
        "input_tokens": 7,
        "output_tokens": 3,
        "total_tokens": 10,
        "input_token_details": {"cache_read": 2},
    }
    assert "".join(recorder.tokens) == "Hello world"

    # LangChain's invoke-with-streaming path must preserve the same snapshots.
    recorder.tokens.clear()
    response = (
        await llm.ainvoke("hello", stream=True)
        if asynchronous
        else llm.invoke("hello", stream=True)
    )
    assert response.content == aggregate.content
    assert response.id == aggregate.id
    assert response.usage_metadata == aggregate.usage_metadata
    assert response.response_metadata == aggregate.response_metadata
    assert "".join(recorder.tokens) == "Hello world"
