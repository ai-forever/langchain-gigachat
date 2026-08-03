"""Reasoning content-block lifecycle tests for primary streams."""

from __future__ import annotations

from functools import reduce
from operator import add
from typing import Any, cast

import gigachat.models as gm
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from langchain_gigachat.chat_models._contracts import primary


def _convert(
    event: dict[str, Any],
    state: primary.StreamState,
) -> ChatGenerationChunk:
    chunk = primary.convert_stream_event(
        gm.PrimaryChatCompletionChunk.model_validate(event),
        state=state,
    )
    assert chunk is not None
    return chunk


def _blocks(chunk: ChatGenerationChunk) -> list[dict[str, Any]]:
    assert isinstance(chunk.message, AIMessageChunk)
    assert isinstance(chunk.message.content, list)
    return cast(list[dict[str, Any]], chunk.message.content)


def _reasoning_event(
    text: str,
    *,
    index: int | None = None,
) -> dict[str, Any]:
    reasoning: dict[str, Any] = {"text": text}
    if index is not None:
        reasoning["index"] = index
    return {
        "event": "response.message.delta",
        "messages": [
            {
                "message_id": "reasoning-message-1",
                "reasoning": reasoning,
            }
        ],
    }


def test_reasoning_fragments_share_one_content_block_index() -> None:
    state = primary.StreamState()
    chunks = [
        _convert(_reasoning_event(fragment), state) for fragment in ("Think", "ing")
    ]

    aggregate = reduce(add, chunks)

    assert [_blocks(chunk)[0]["index"] for chunk in chunks] == [0, 0]
    assert _blocks(aggregate) == [
        {
            "type": "reasoning",
            "reasoning": "Thinking",
            "index": 0,
        }
    ]
    assert aggregate.message.additional_kwargs["reasoning_content"] == "Thinking"


def test_message_level_reasoning_content_keeps_stable_index() -> None:
    state = primary.StreamState()
    chunks = [
        _convert(
            {
                "messages": [
                    {
                        "message_id": "reasoning-message-1",
                        "reasoning_content": fragment,
                    }
                ]
            },
            state,
        )
        for fragment in ("Plan", "ning")
    ]

    aggregate = chunks[0] + chunks[1]

    assert [_blocks(chunk)[0]["index"] for chunk in chunks] == [0, 0]
    assert _blocks(aggregate)[0]["reasoning"] == "Planning"


def test_assistant_transition_closes_reasoning_block() -> None:
    state = primary.StreamState()
    reasoning = _convert(_reasoning_event("Think"), state)
    assistant = _convert(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"text": "Answer"}],
                }
            ]
        },
        state,
    )
    later_reasoning = _convert(_reasoning_event("Check"), state)

    assert _blocks(reasoning)[0]["index"] == 0
    assert _blocks(assistant)[0]["index"] == 1
    assert _blocks(later_reasoning)[0]["index"] == 2


def test_file_and_tool_transitions_close_reasoning_block() -> None:
    state = primary.StreamState()
    reasoning = _convert(_reasoning_event("Think"), state)
    file_chunk = _convert(
        {
            "messages": [
                {
                    "content": [
                        {
                            "files": [
                                {
                                    "id": "file-1",
                                    "mime_type": "application/pdf",
                                }
                            ]
                        }
                    ]
                }
            ]
        },
        state,
    )
    tool_chunk = _convert(
        {
            "tools_state_id": "provider-tool-state-1",
            "messages": [
                {
                    "function_call": {
                        "name": "lookup",
                        "arguments": '{"key":"value"}',
                    }
                }
            ],
        },
        state,
    )
    later_reasoning = _convert(_reasoning_event("Check"), state)

    assert _blocks(reasoning)[0]["index"] == 0
    assert _blocks(file_chunk)[0]["index"] == 1
    tool_message = cast(AIMessageChunk, tool_chunk.message)
    assert tool_message.tool_call_chunks[0]["index"] == 2
    assert _blocks(later_reasoning)[0]["index"] == 3


def test_explicit_new_reasoning_index_starts_independent_block() -> None:
    state = primary.StreamState()
    first = _convert(_reasoning_event("First", index=2), state)
    second = _convert(_reasoning_event("Second", index=5), state)

    aggregate = first + second

    assert [_blocks(first)[0]["index"], _blocks(second)[0]["index"]] == [2, 5]
    assert _blocks(aggregate) == [
        {
            "type": "reasoning",
            "reasoning": "First",
            "index": 2,
        },
        {
            "type": "reasoning",
            "reasoning": "Second",
            "index": 5,
        },
    ]
