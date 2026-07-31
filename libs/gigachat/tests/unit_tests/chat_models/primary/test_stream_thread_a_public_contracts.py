"""Public workflow coverage for the final Thread A stream contracts."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from functools import reduce
from operator import add
from typing import Any, cast
from unittest.mock import MagicMock

from langchain_core.language_models.chat_models import generate_from_stream
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGenerationChunk

from langchain_gigachat.chat_models.gigachat import GigaChat

from .fixtures import build_official_sdk_server_tool_stream


def _official_sdk_server_tool_events() -> Iterator[Any]:
    return iter(build_official_sdk_server_tool_stream())


def _assert_official_sdk_server_tool_message(
    message: AIMessage | AIMessageChunk,
) -> None:
    assert isinstance(message.content, list)
    blocks = cast(list[dict[str, Any]], message.content)
    result = next(block for block in blocks if block["type"] == "server_tool_result")
    call_id = result["tool_call_id"]

    assert call_id == "lc_primary-server-tool-0"
    assert result["extras"]["provider_tool_execution"]["censored"] is True
    assert message.additional_kwargs["provider_server_tool_state_by_call_id"] == {
        call_id: "tools-state-1"
    }
    assert message.response_metadata["tools_state_id"] == "tools-state-1"
    assert message.response_metadata["finish_reason"] == "error"
    assert message.usage_metadata == {
        "input_tokens": 1,
        "output_tokens": 2,
        "total_tokens": 3,
        "input_token_details": {"cache_read": 0},
    }


def _count_internal_final_chunks(chunks: list[ChatGenerationChunk]) -> int:
    count = 0
    for chunk in chunks:
        assert isinstance(chunk.message, AIMessageChunk)
        count += chunk.message.chunk_position == "last"
    return count


def _finish_reason_events() -> Iterator[dict[str, Any]]:
    yield {
        "event": "response.tool.failed",
        "finish_reason": "tool_error",
        "error": {"message": "tool failed"},
    }
    yield {
        "event": "response.message.done",
        "finish_reason": "stop",
    }


def _late_finish_reason_events() -> Iterator[dict[str, Any]]:
    yield {
        "event": "response.message.done",
        "message_id": "provider-message-1",
    }
    yield {
        "event": "response.message.done",
        "finish_reason": "stop",
    }


def _mirrored_function_events() -> Iterator[dict[str, Any]]:
    for fragment in ('{"city":', '"Moscow"}'):
        function_call = {
            "name": "weather",
            "arguments": fragment,
        }
        yield {
            "event": "response.message.delta",
            "tools_state_id": "provider-tool-state-1",
            "messages": [
                {
                    "function_call": function_call,
                    "content": [{"function_call": function_call}],
                }
            ],
        }
    yield {
        "event": "response.message.done",
        "finish_reason": "function_call",
    }


def _late_server_identity_events() -> Iterator[dict[str, Any]]:
    yield {
        "event": "response.tool.started",
        "tool_execution": {
            "name": "web_search",
            "status": "running",
        },
    }
    yield {
        "event": "response.tool.completed",
        "tools_state_id": "provider-server-tool-state-1",
        "tool_execution": {
            "name": "web_search",
            "status": "completed",
            "output": {"matches": 1},
        },
    }
    yield {
        "event": "response.message.done",
        "finish_reason": "stop",
    }


def _reasoning_events() -> Iterator[dict[str, Any]]:
    for fragment in ("Think", "ing"):
        yield {
            "event": "response.message.delta",
            "messages": [
                {
                    "message_id": "provider-message-1",
                    "reasoning": {"text": fragment},
                }
            ],
        }
    yield {
        "event": "response.message.delta",
        "messages": [
            {
                "message_id": "provider-message-1",
                "content": [{"text": "Answer"}],
            }
        ],
    }
    yield {
        "event": "response.message.done",
        "finish_reason": "stop",
    }


async def _async_items(
    items: Iterator[dict[str, Any]],
) -> AsyncIterator[dict[str, Any]]:
    for item in items:
        yield item


def _configure_sync_stream(
    sdk_client: MagicMock,
    factory: Any,
) -> None:
    sdk_client.chat.stream.side_effect = lambda payload: factory()


def _configure_async_stream(
    sdk_client: MagicMock,
    factory: Any,
) -> None:
    sdk_client.achat.stream.side_effect = lambda payload: _async_items(factory())


def test_finish_reason_authority_across_sync_public_workflows(
    sdk_client: MagicMock,
) -> None:
    _configure_sync_stream(sdk_client, _finish_reason_events)
    llm = GigaChat(model="GigaChat-3-Ultra", use_api_v2=True, streaming=True)

    internal = list(llm._stream([HumanMessage("Hello")]))
    internal_aggregate = reduce(add, internal)
    public_chunks = list(llm.stream("Hello"))
    public_aggregate = reduce(add, public_chunks)
    invoked = llm.invoke("Hello")

    assert internal_aggregate.generation_info == {"finish_reason": "stop"}
    assert internal_aggregate.message.response_metadata["finish_reason"] == "stop"
    assert public_aggregate.response_metadata["finish_reason"] == "stop"
    assert invoked.response_metadata["finish_reason"] == "stop"
    for message in (internal_aggregate.message, public_aggregate, invoked):
        assert message.response_metadata["finish_reason_events"] == [
            {
                "event": "response.tool.failed",
                "finish_reason": "tool_error",
            }
        ]


async def test_finish_reason_authority_across_async_public_workflows(
    sdk_client: MagicMock,
) -> None:
    _configure_async_stream(sdk_client, _finish_reason_events)
    llm = GigaChat(model="GigaChat-3-Ultra", use_api_v2=True, streaming=True)

    internal = [chunk async for chunk in llm._astream([HumanMessage("Hello")])]
    internal_aggregate = reduce(add, internal)
    public_chunks = [chunk async for chunk in llm.astream("Hello")]
    public_aggregate = reduce(add, public_chunks)
    invoked = await llm.ainvoke("Hello")

    assert internal_aggregate.generation_info == {"finish_reason": "stop"}
    assert internal_aggregate.message.response_metadata["finish_reason"] == "stop"
    assert public_aggregate.response_metadata["finish_reason"] == "stop"
    assert invoked.response_metadata["finish_reason"] == "stop"


def test_late_finish_reason_is_promoted_in_sync_buffered_terminal(
    sdk_client: MagicMock,
) -> None:
    _configure_sync_stream(sdk_client, _late_finish_reason_events)
    llm = GigaChat(model="GigaChat-3-Ultra", use_api_v2=True, streaming=True)

    chunks = list(llm._stream([HumanMessage("Hello")]))
    invoked = llm.invoke("Hello")

    assert len(chunks) == 1
    assert chunks[0].generation_info == {"finish_reason": "stop"}
    assert chunks[0].message.response_metadata["finish_reason"] == "stop"
    assert chunks[0].message.id == "provider-message-1"
    assert invoked.response_metadata["finish_reason"] == "stop"


async def test_late_finish_reason_is_promoted_in_async_buffered_terminal(
    sdk_client: MagicMock,
) -> None:
    _configure_async_stream(sdk_client, _late_finish_reason_events)
    llm = GigaChat(model="GigaChat-3-Ultra", use_api_v2=True, streaming=True)

    chunks = [chunk async for chunk in llm._astream([HumanMessage("Hello")])]
    invoked = await llm.ainvoke("Hello")

    assert len(chunks) == 1
    assert chunks[0].generation_info == {"finish_reason": "stop"}
    assert chunks[0].message.response_metadata["finish_reason"] == "stop"
    assert chunks[0].message.id == "provider-message-1"
    assert invoked.response_metadata["finish_reason"] == "stop"


def test_public_stream_deduplicates_mirrored_function_fragments(
    sdk_client: MagicMock,
) -> None:
    _configure_sync_stream(sdk_client, _mirrored_function_events)
    llm = GigaChat(model="GigaChat-3-Ultra", use_api_v2=True)

    aggregate = reduce(add, llm.stream("Hello"))

    assert isinstance(aggregate, AIMessageChunk)
    assert aggregate.tool_calls == [
        {
            "name": "weather",
            "args": {"city": "Moscow"},
            "id": "provider-tool-state-1",
            "type": "tool_call",
        }
    ]


async def test_public_astream_deduplicates_mirrored_function_fragments(
    sdk_client: MagicMock,
) -> None:
    _configure_async_stream(sdk_client, _mirrored_function_events)
    llm = GigaChat(model="GigaChat-3-Ultra", use_api_v2=True)

    chunks = [chunk async for chunk in llm.astream("Hello")]
    aggregate = reduce(add, chunks)

    assert isinstance(aggregate, AIMessageChunk)
    assert aggregate.tool_calls[0]["args"] == {"city": "Moscow"}


def test_public_stream_reconciles_late_server_tool_identity(
    sdk_client: MagicMock,
) -> None:
    _configure_sync_stream(sdk_client, _late_server_identity_events)
    llm = GigaChat(model="GigaChat-3-Ultra", use_api_v2=True)

    aggregate = reduce(add, llm.stream("Hello"))

    assert isinstance(aggregate.content, list)
    blocks = cast(list[dict[str, Any]], aggregate.content)
    call_block = next(
        block
        for block in blocks
        if block["type"] in {"server_tool_call", "server_tool_call_chunk"}
    )
    result_block = next(
        block for block in blocks if block["type"] == "server_tool_result"
    )
    call_id = call_block["id"]
    assert call_id == "lc_primary-server-tool-0"
    assert result_block["tool_call_id"] == call_id
    assert result_block["extras"]["provider_server_tool_state_by_call_id"] == {
        call_id: "provider-server-tool-state-1"
    }


def test_public_stream_preserves_reasoning_block_lifecycle(
    sdk_client: MagicMock,
) -> None:
    _configure_sync_stream(sdk_client, _reasoning_events)
    llm = GigaChat(model="GigaChat-3-Ultra", use_api_v2=True)

    aggregate = reduce(add, llm.stream("Hello"))

    assert isinstance(aggregate, AIMessageChunk)
    assert isinstance(aggregate.content, list)
    blocks = cast(list[dict[str, Any]], aggregate.content)
    assert blocks == [
        {
            "type": "reasoning",
            "reasoning": "Thinking",
            "index": 0,
        },
        {
            "type": "text",
            "text": "Answer",
            "index": 1,
        },
    ]
    assert aggregate.additional_kwargs["reasoning_content"] == "Thinking"


def test_streaming_invoke_preserves_mirrored_function_call(
    sdk_client: MagicMock,
) -> None:
    _configure_sync_stream(sdk_client, _mirrored_function_events)
    llm = GigaChat(
        model="GigaChat-3-Ultra",
        use_api_v2=True,
        streaming=True,
    )

    result = llm.invoke("Hello")

    assert isinstance(result, AIMessage)
    assert result.tool_calls[0]["args"] == {"city": "Moscow"}


def test_official_sdk_server_tool_stream_across_sync_public_workflows(
    sdk_client: MagicMock,
) -> None:
    _configure_sync_stream(sdk_client, _official_sdk_server_tool_events)
    llm = GigaChat(
        model="GigaChat-3-Ultra",
        use_api_v2=True,
        streaming=True,
    )

    internal = list(llm._stream([HumanMessage("Hello")]))
    internal_aggregate = reduce(add, internal)
    generated = generate_from_stream(iter(internal)).generations[0]
    public_chunks = list(llm.stream("Hello"))
    public_aggregate = reduce(add, public_chunks)
    invoked = llm.invoke("Hello")

    assert _count_internal_final_chunks(internal) == 1
    assert sum(chunk.chunk_position == "last" for chunk in public_chunks) == 1
    assert internal_aggregate.generation_info == {"finish_reason": "error"}
    assert generated.generation_info == {"finish_reason": "error"}
    for message in (
        internal_aggregate.message,
        generated.message,
        public_aggregate,
        invoked,
    ):
        assert isinstance(message, (AIMessage, AIMessageChunk))
        _assert_official_sdk_server_tool_message(message)


async def test_official_sdk_server_tool_stream_across_async_public_workflows(
    sdk_client: MagicMock,
) -> None:
    _configure_async_stream(sdk_client, _official_sdk_server_tool_events)
    llm = GigaChat(
        model="GigaChat-3-Ultra",
        use_api_v2=True,
        streaming=True,
    )

    internal = [chunk async for chunk in llm._astream([HumanMessage("Hello")])]
    internal_aggregate = reduce(add, internal)
    public_chunks = [chunk async for chunk in llm.astream("Hello")]
    public_aggregate = reduce(add, public_chunks)
    invoked = await llm.ainvoke("Hello")

    assert _count_internal_final_chunks(internal) == 1
    assert sum(chunk.chunk_position == "last" for chunk in public_chunks) == 1
    assert internal_aggregate.generation_info == {"finish_reason": "error"}
    for message in (internal_aggregate.message, public_aggregate, invoked):
        assert isinstance(message, (AIMessage, AIMessageChunk))
        _assert_official_sdk_server_tool_message(message)
