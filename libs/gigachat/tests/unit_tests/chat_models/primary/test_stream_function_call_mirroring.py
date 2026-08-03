"""Mirroring and argument-snapshot contracts for primary function-call streams."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import gigachat.models as gm
import pytest
from langchain_core.language_models.chat_models import (
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages import AIMessage, AIMessageChunk
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


def _message(chunk: ChatGenerationChunk) -> AIMessageChunk:
    assert isinstance(chunk.message, AIMessageChunk)
    return cast(AIMessageChunk, chunk.message)


def _function_call(
    arguments: Any,
    *,
    name: str = "weather",
    call_id: str | None = None,
) -> dict[str, Any]:
    call = {"name": name, "arguments": arguments}
    if call_id is not None:
        call["id"] = call_id
    return call


@pytest.mark.parametrize("level", ["part", "message"])
def test_part_only_and_message_only_calls_are_preserved(level: str) -> None:
    function_call = _function_call({"city": "Moscow"})
    message: dict[str, Any] = {"tools_state_id": "tools-state-1"}
    if level == "part":
        message["content"] = [{"function_call": function_call}]
    else:
        message["function_call"] = function_call

    chunk = _convert({"messages": [message]}, primary.StreamState())

    assert _message(chunk).tool_call_chunks == [
        {
            "name": "weather",
            "args": '{"city":"Moscow"}',
            "id": "tools-state-1",
            "index": 0,
            "type": "tool_call_chunk",
        }
    ]


def test_mirrored_part_and_message_call_is_emitted_once() -> None:
    function_call = _function_call({"city": "Moscow"})
    chunk = _convert(
        {
            "messages": [
                {
                    "tools_state_id": "tools-state-1",
                    "function_call": function_call,
                    "content": [{"function_call": function_call}],
                }
            ]
        },
        primary.StreamState(),
    )

    assert len(_message(chunk).tool_call_chunks) == 1
    assert _message(chunk).tool_call_chunks[0]["args"] == '{"city":"Moscow"}'


def test_distinct_part_and_message_calls_are_not_deduplicated() -> None:
    with pytest.raises(
        ValueError,
        match="supports one client tool call per completion",
    ):
        _convert(
            {
                "messages": [
                    {
                        "function_call": _function_call(
                            {"city": "Moscow"},
                            name="weather",
                        ),
                        "content": [
                            {
                                "function_call": _function_call(
                                    {"query": "Moscow"},
                                    name="search",
                                )
                            }
                        ],
                    }
                ]
            },
            primary.StreamState(),
        )


def test_distinct_part_level_calls_are_not_treated_as_fragments() -> None:
    with pytest.raises(
        ValueError,
        match="supports one client tool call per completion",
    ):
        _convert(
            {
                "messages": [
                    {
                        "content": [
                            {
                                "function_call": _function_call(
                                    '{"city":"Moscow"}',
                                    name="weather",
                                )
                            },
                            {
                                "function_call": _function_call(
                                    '{"query":"Moscow"}',
                                    name="search",
                                )
                            },
                        ]
                    }
                ]
            },
            primary.StreamState(),
        )


def test_identical_calls_in_distinct_messages_are_not_treated_as_mirrors() -> None:
    function_call = _function_call({"city": "Moscow"})

    with pytest.raises(
        ValueError,
        match="supports one client tool call per completion",
    ):
        _convert(
            {
                "messages": [
                    {"function_call": function_call},
                    {"function_call": function_call},
                ]
            },
            primary.StreamState(),
        )


def _mirrored_fragment_events() -> list[dict[str, Any]]:
    return [
        {
            "event": "response.message.delta",
            "tools_state_id": "tools-state-1",
            "messages": [
                {
                    "function_call": _function_call(fragment),
                    "content": [{"function_call": _function_call(fragment)}],
                }
            ],
        }
        for fragment in ('{"city":', '"Moscow"}')
    ]


def test_mirrored_string_fragments_aggregate_once() -> None:
    state = primary.StreamState()
    chunks = [_convert(event, state) for event in _mirrored_fragment_events()]

    result = generate_from_stream(iter(chunks))

    message = cast(AIMessage, result.generations[0].message)
    assert message.tool_calls == [
        {
            "name": "weather",
            "args": {"city": "Moscow"},
            "id": "tools-state-1",
            "type": "tool_call",
        }
    ]


async def _async_chunks(
    chunks: list[ChatGenerationChunk],
) -> AsyncIterator[ChatGenerationChunk]:
    for chunk in chunks:
        yield chunk


async def test_mirrored_string_fragments_aggregate_once_async() -> None:
    state = primary.StreamState()
    chunks = [_convert(event, state) for event in _mirrored_fragment_events()]

    result = await agenerate_from_stream(_async_chunks(chunks))

    message = cast(AIMessage, result.generations[0].message)
    assert message.tool_calls[0]["args"] == {"city": "Moscow"}


def test_single_dictionary_argument_snapshot_is_supported() -> None:
    chunk = _convert(
        {
            "messages": [
                {
                    "tools_state_id": "tools-state-1",
                    "function_call": _function_call({"city": "Moscow"}),
                }
            ]
        },
        primary.StreamState(),
    )

    assert _message(chunk).tool_call_chunks[0]["args"] == '{"city":"Moscow"}'


def test_repeated_dictionary_argument_snapshots_fail_closed() -> None:
    state = primary.StreamState()
    _convert(
        {
            "messages": [
                {
                    "tools_state_id": "tools-state-1",
                    "function_call": _function_call({"city": "Mos"}),
                }
            ]
        },
        state,
    )

    with pytest.raises(
        ValueError,
        match="second client tool call",
    ):
        _convert(
            {
                "messages": [
                    {
                        "tools_state_id": "tools-state-1",
                        "function_call": _function_call({"city": "Moscow"}),
                    }
                ]
            },
            state,
        )


def test_second_same_name_call_without_ids_is_rejected_after_complete_json() -> None:
    state = primary.StreamState()
    _convert(
        {
            "messages": [
                {
                    "function_call": _function_call('{"city":"Moscow"}'),
                }
            ]
        },
        state,
    )

    with pytest.raises(
        ValueError,
        match="second client tool call",
    ):
        _convert(
            {
                "messages": [
                    {
                        "function_call": _function_call('{"city":"Kazan"}'),
                    }
                ]
            },
            state,
        )


def test_completed_client_call_accepts_whitespace_argument_continuation() -> None:
    state = primary.StreamState()
    first = _convert(
        {
            "event": "response.message.delta",
            "tools_state_id": "tools-state-1",
            "messages": [
                {
                    "function_call": {
                        "index": 0,
                        "name": "weather",
                        "arguments": '{"city":"Moscow"}',
                    }
                }
            ],
        },
        state,
    )
    continuation_call: dict[str, Any] = {
        "index": 0,
        "name": "weather",
    }
    continuation_call["arguments"] = " \n\t"
    continuation = _convert(
        {
            "event": "response.message.delta",
            "tools_state_id": "tools-state-1",
            "messages": [{"function_call": continuation_call}],
        },
        state,
    )
    done = _convert(
        {
            "event": "response.message.done",
            "tools_state_id": "tools-state-1",
            "finish_reason": "function_call",
        },
        state,
    )

    message = (
        generate_from_stream(iter([first, continuation, done])).generations[0].message
    )
    assert isinstance(message, AIMessage)
    assert message.tool_calls == [
        {
            "name": "weather",
            "args": {"city": "Moscow"},
            "id": "tools-state-1",
            "type": "tool_call",
        }
    ]


def test_whitespace_client_tool_name_fails_closed() -> None:
    with pytest.raises(
        ValueError,
        match="Function call name must be a non-empty string",
    ):
        _convert(
            {
                "event": "response.message.delta",
                "tools_state_id": "tools-state-1",
                "messages": [
                    {
                        "function_call": {
                            "name": "   ",
                            "arguments": "{}",
                        }
                    }
                ],
            },
            primary.StreamState(),
        )


def test_same_name_idless_fragments_continue_until_json_object_is_complete() -> None:
    state = primary.StreamState()
    first = _convert(
        {
            "messages": [
                {
                    "function_call": _function_call('{"city":'),
                }
            ]
        },
        state,
    )
    second = _convert(
        {
            "messages": [
                {
                    "function_call": _function_call('"Moscow"}'),
                }
            ]
        },
        state,
    )
    done = _convert(
        {
            "event": "response.message.done",
            "tools_state_id": "provider-tool-state-1",
            "finish_reason": "function_call",
        },
        state,
    )

    result = generate_from_stream(iter([first, second, done]))
    message = cast(AIMessage, result.generations[0].message)

    assert message.tool_calls == [
        {
            "name": "weather",
            "args": {"city": "Moscow"},
            "id": "provider-tool-state-1",
            "type": "tool_call",
        }
    ]
