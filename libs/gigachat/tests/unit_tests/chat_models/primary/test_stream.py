"""Primary SDK named-event stream conversion."""

from __future__ import annotations

from functools import reduce
from operator import add
from typing import Any, cast

import gigachat.models as gm
import pytest
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from langchain_gigachat.chat_models._contracts import primary
from langchain_gigachat.chat_models._contracts.primary.types import PrimaryStreamError

from .fixtures import build_official_sdk_server_tool_stream


def _event(values: dict[str, Any]) -> gm.PrimaryChatCompletionChunk:
    return gm.PrimaryChatCompletionChunk.model_validate(values)


def _convert(
    values: dict[str, Any] | gm.PrimaryChatCompletionChunk,
    state: primary.StreamState | None = None,
) -> ChatGenerationChunk:
    event = (
        values if isinstance(values, gm.PrimaryChatCompletionChunk) else _event(values)
    )
    chunk = primary.convert_stream_event(
        event,
        state=state or primary.StreamState(),
    )
    assert chunk is not None
    return chunk


def _blocks(chunk: ChatGenerationChunk) -> list[dict[str, Any]]:
    assert isinstance(chunk.message, AIMessageChunk)
    return [dict(block) for block in chunk.message.content_blocks]


@pytest.mark.parametrize("message_id", [None, "message-1"])
def test_reasoning_fragments_merge_without_colliding_with_text(
    message_id: str | None,
) -> None:
    state = primary.StreamState()
    chunks = [
        _convert(
            {
                "event": "response.message.delta",
                "messages": [
                    {"role": "assistant", "message_id": message_id, **fragment}
                ],
            },
            state,
        )
        for fragment in (
            {"reasoning_content": "First "},
            {"reasoning_content": "thought."},
            {"content": [{"text": "Answer."}]},
            {"reasoning_content": "Second "},
            {"reasoning_content": "thought."},
        )
    ]
    assert _blocks(reduce(add, chunks)) == [
        {"type": "reasoning", "reasoning": "First thought.", "index": 0},
        {"type": "text", "text": "Answer.", "index": 1},
        {"type": "reasoning", "reasoning": "Second thought.", "index": 2},
    ]


def test_text_reasoning_files_and_citations_keep_ordered_blocks() -> None:
    state = primary.StreamState()
    chunk = _convert(
        {
            "event": "response.message.delta",
            "messages": [
                {"role": "reasoning", "content": [{"text": "Think"}]},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "text": "Answer",
                            "inline_data": {
                                "sources": {
                                    "source-1": {
                                        "url": "https://example.test",
                                        "title": "Example",
                                    }
                                }
                            },
                        },
                        {"files": [{"id": "file-1", "mime": "application/pdf"}]},
                    ],
                },
            ],
        },
        state,
    )

    assert _blocks(chunk) == [
        {"type": "reasoning", "reasoning": "Think", "index": 0},
        {
            "type": "text",
            "text": "Answer",
            "index": 1,
            "annotations": [
                {
                    "type": "citation",
                    "id": "source-1",
                    "url": "https://example.test",
                    "title": "Example",
                }
            ],
        },
        {
            "type": "file",
            "file_id": "file-1",
            "mime_type": "application/pdf",
            "index": 2,
        },
    ]
    assert chunk.message.additional_kwargs["reasoning_content"] == "Think"


def test_fragmented_client_function_uses_late_tools_state_id() -> None:
    state = primary.StreamState()
    chunks = [
        _convert(
            {
                "event": "response.message.delta",
                "messages": [
                    {
                        "role": "assistant",
                        "function_call": {
                            "name": "weather",
                            "arguments": '{"city":',
                        },
                    }
                ],
            },
            state,
        ),
        _convert(
            {
                "event": "response.message.delta",
                "tools_state_id": "tools-state-1",
                "messages": [
                    {
                        "role": "assistant",
                        "function_call": {
                            "name": "weather",
                            "arguments": '"Moscow"}',
                        },
                    }
                ],
            },
            state,
        ),
        _convert(
            {
                "event": "response.message.done",
                "tools_state_id": "tools-state-1",
                "finish_reason": "tool_calls",
            },
            state,
        ),
    ]

    aggregate = reduce(add, chunks)
    assert isinstance(aggregate.message, AIMessageChunk)
    assert aggregate.message.tool_calls == [
        {
            "type": "tool_call",
            "name": "weather",
            "args": {"city": "Moscow"},
            "id": "tools-state-1",
        }
    ]
    assert aggregate.message.additional_kwargs["tools_state_id"] == "tools-state-1"


def test_explicit_server_tool_keeps_provider_call_id() -> None:
    state = primary.StreamState()
    started = _convert(
        {
            "event": "response.tool.started",
            "tool_execution": {
                "call_id": "server-call-1",
                "name": "web_search",
                "status": "running",
                "arguments": {"query": "GigaChat"},
            },
        },
        state,
    )
    completed = _convert(
        {
            "event": "response.tool.completed",
            "tool_execution": {
                "call_id": "server-call-1",
                "name": "web_search",
                "status": "success",
                "output": {"matches": 1},
            },
        },
        state,
    )

    call = _blocks(started)[0]
    result = _blocks(completed)[0]
    assert call["id"] == "server-call-1"
    assert result["tool_call_id"] == "server-call-1"
    assert result["output"] == {"matches": 1}


def test_sequential_idless_server_tools_get_distinct_local_ids() -> None:
    state = primary.StreamState()
    chunks: list[ChatGenerationChunk] = []
    for name in ("web_search", "image_generate"):
        chunks.extend(
            [
                _convert(
                    {
                        "event": "response.tool.started",
                        "tool_execution": {"name": name, "status": "running"},
                    },
                    state,
                ),
                _convert(
                    {
                        "event": "response.tool.completed",
                        "tool_execution": {"name": name, "status": "success"},
                    },
                    state,
                ),
            ]
        )

    blocks = [block for chunk in chunks for block in _blocks(chunk)]
    assert [blocks[0]["id"], blocks[2]["id"]] == [
        "lc_primary-server-tool-0",
        "lc_primary-server-tool-1",
    ]
    assert blocks[1]["tool_call_id"] == blocks[0]["id"]
    assert blocks[3]["tool_call_id"] == blocks[2]["id"]


def test_parallel_idless_server_tools_fail_clearly() -> None:
    event = _event(
        {
            "event": "response.tool.started",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"tool_execution": {"name": "web_search", "status": "running"}},
                        {
                            "tool_execution": {
                                "name": "image_generate",
                                "status": "running",
                            }
                        },
                    ],
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="ambiguous parallel server tools"):
        primary.convert_stream_event(event, state=primary.StreamState())


def test_official_sdk_idless_server_tool_uses_local_identity_only() -> None:
    state = primary.StreamState()
    chunks = [
        _convert(event, state) for event in build_official_sdk_server_tool_stream()
    ]
    aggregate = reduce(add, chunks)
    result = next(
        block
        for block in aggregate.message.content_blocks
        if block["type"] == "server_tool_result"
    )

    assert result["tool_call_id"] == "lc_primary-server-tool-0"
    assert result["tool_call_id"] != "tools-state-1"
    assert aggregate.message.response_metadata["tools_state_ids"] == ["tools-state-1"]


def test_aggregate_does_not_concatenate_repeated_scalar_metadata() -> None:
    state = primary.StreamState()
    events: list[dict[str, Any]] = [
        {
            "event": "response.message.delta",
            "model": "GigaChat-3-Ultra:32.9.23.6",
            "thread_id": "thread-1",
            "messages": [{"role": "assistant", "content": [{"text": "First"}]}],
        },
        {
            "event": "response.message.done",
            "model": "GigaChat-3-Ultra:32.9.23.6",
            "thread_id": "thread-1",
            "finish_reason": "stop",
        },
    ]

    chunks = [_convert(event, state) for event in events]
    aggregate = reduce(add, chunks)

    assert aggregate.message.response_metadata["model"] == (
        "GigaChat-3-Ultra:32.9.23.6"
    )
    assert aggregate.message.response_metadata["model_name"] == (
        "GigaChat-3-Ultra:32.9.23.6"
    )
    assert aggregate.message.response_metadata["thread_id"] == "thread-1"


def test_done_event_preserves_usage_finish_ids_and_headers() -> None:
    state = primary.StreamState()
    chunk = _convert(
        {
            "event": "response.message.done",
            "message_id": "message-1",
            "thread_id": "thread-1",
            "model": "GigaChat-3-Ultra",
            "finish_reason": "stop",
            "x_headers": {"x-request-id": "request-1"},
            "usage": {
                "input_tokens": 8,
                "input_tokens_details": {
                    "prompt_tokens": 8,
                    "cached_tokens": 3,
                },
                "output_tokens": 5,
                "total_tokens": 13,
            },
        },
        state,
    )

    assert isinstance(chunk.message, AIMessageChunk)
    assert chunk.message.id == "request-1"
    assert chunk.message.chunk_position == "last"
    assert chunk.generation_info == {"finish_reason": "stop"}
    assert chunk.message.usage_metadata == {
        "input_tokens": 11,
        "output_tokens": 5,
        "total_tokens": 16,
        "input_token_details": {"cache_read": 3},
    }
    assert chunk.message.response_metadata["message_id"] == "message-1"
    assert chunk.message.response_metadata["thread_id"] == "thread-1"
    assert chunk.message.response_metadata["model"] == "GigaChat-3-Ultra"


def test_unknown_sdk_event_preserves_extension_fields() -> None:
    chunk = _convert(
        {
            "event": "response.provider_extension.delta",
            "future_field": {"value": 42},
        }
    )

    assert chunk.message.response_metadata["provider_fields"] == {
        "future_field": {"value": 42}
    }
    assert chunk.message.response_metadata["provider_events"][0]["event"] == (
        "response.provider_extension.delta"
    )


def test_provider_error_raises_typed_exception() -> None:
    event = _event(
        {
            "event": "response.error",
            "error": {"code": "provider_error", "message": "boom"},
        }
    )

    with pytest.raises(PrimaryStreamError, match="response.error"):
        primary.convert_stream_event(event, state=primary.StreamState())


def test_converter_accepts_only_sdk_models() -> None:
    raw_event = cast(
        gm.PrimaryChatCompletionChunk,
        {"event": "response.message.delta"},
    )
    with pytest.raises(TypeError, match="PrimaryChatCompletionChunk"):
        primary.convert_stream_event(
            raw_event,
            state=primary.StreamState(),
        )


def test_interleaved_parallel_client_calls_keep_arguments_ids_and_state() -> None:
    state = primary.StreamState()
    chunks = []
    for call_id, args in [
        ("call-1", '{"key":'),
        ("call-2", '{"key":'),
        ("call-2", "2}"),
        ("call-1", "1}"),
    ]:
        chunks.append(
            _convert(
                {
                    "event": "response.message.delta",
                    "messages": [
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "function_call": {
                                        "id": call_id,
                                        "name": "lookup",
                                        "arguments": args,
                                    }
                                }
                            ],
                        }
                    ],
                },
                state,
            )
        )
    chunks.append(
        _convert(
            {
                "event": "response.message.done",
                "tools_state_id": "state",
                "finish_reason": "tool_calls",
            },
            state,
        )
    )
    message = reduce(add, chunks).message
    assert isinstance(message, AIMessageChunk)
    assert message.tool_calls == [
        {"id": "call-1", "name": "lookup", "args": {"key": 1}, "type": "tool_call"},
        {"id": "call-2", "name": "lookup", "args": {"key": 2}, "type": "tool_call"},
    ]
    assert message.additional_kwargs["tools_state_id"] == "state"
    assert message.response_metadata["tools_state_id"] == "state"
    converted = primary.convert_messages([message], cached_uploads={})[0]
    assert converted.tools_state_id == "state"
    assert [
        part.function_call.model_dump(by_alias=True)["id"]
        for part in converted.content or []
        if part.function_call
    ] == ["call-1", "call-2"]


def test_parallel_complete_client_calls_arrive_in_one_event() -> None:
    chunk = _convert(
        {
            "event": "response.message.done",
            "tools_state_id": "state",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "function_call": {
                                "id": f"call-{index}",
                                "name": "lookup",
                                "arguments": {"key": index},
                            }
                        }
                        for index in (1, 2)
                    ],
                }
            ],
        }
    )
    assert isinstance(chunk.message, AIMessageChunk)
    assert [call["id"] for call in chunk.message.tool_calls] == ["call-1", "call-2"]


@pytest.mark.parametrize(
    "additional_data",
    [
        [{"trace": "abc"}],
        pytest.param(
            {"trace": "abc"},
            marks=pytest.mark.skipif(
                "error_details" not in gm.PrimaryChatCompletionChunk.model_fields,
                reason="SDK 0.2.3 does not parse object-shaped additional_data",
            ),
        ),
    ],
)
def test_stream_preserves_object_additional_data_error_and_part_logprobs(
    additional_data: Any,
) -> None:
    logprobs = [{"chosen": {"token": "Hi", "token_id": 1, "logprob": -0.2}}]
    event = _event(
        {
            "event": "response.message.done",
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"text": "Hi", "logprobs": logprobs}],
                }
            ],
            "additional_data": additional_data,
            "error_details": {"reason": "filtered"},
        }
    )
    chunk = _convert(event)
    assert chunk.message.response_metadata["additional_data"] == additional_data
    assert chunk.message.response_metadata["error_details"] == {"reason": "filtered"}
    assert (
        cast(dict[str, Any], chunk.message.content_blocks[0])["extras"][
            "provider_data"
        ]["logprobs"]
        == logprobs
    )


def test_message_level_stream_logprobs_are_preserved() -> None:
    logprobs = [{"chosen": {"token": "Hi", "token_id": 1, "logprob": -0.2}}]
    chunk = _convert(
        {
            "event": "response.message.delta",
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"text": "Hi"}],
                    "logprobs": logprobs,
                }
            ],
        }
    )
    assert chunk.message.response_metadata["logprobs"] == logprobs


@pytest.mark.parametrize("call_id", [None, "real-call"])
@pytest.mark.parametrize("snapshot", [{"key": 1}, '{"key": 1}'])
def test_terminal_argument_snapshot_does_not_repeat_fragments(
    call_id: str | None, snapshot: Any
) -> None:
    state = primary.StreamState()
    chunks = []
    for event, arguments in [
        ("response.message.delta", '{"key":'),
        ("response.message.delta", "1}"),
        ("response.message.done", snapshot),
    ]:
        chunks.append(
            _convert(
                {
                    "event": event,
                    "tools_state_id": "state",
                    "messages": [
                        {
                            "role": "assistant",
                            "function_call": {
                                "id": call_id,
                                "name": "lookup",
                                "arguments": arguments,
                            },
                        }
                    ],
                },
                state,
            )
        )
    message = reduce(add, chunks).message
    assert isinstance(message, AIMessageChunk)
    assert message.tool_calls == [
        {
            "id": call_id or "state",
            "name": "lookup",
            "args": {"key": 1},
            "type": "tool_call",
        }
    ]


@pytest.mark.parametrize(
    "additional_data",
    [
        [{"trace": "abc"}],
        pytest.param(
            {"trace": "abc"},
            marks=pytest.mark.skipif(
                "error_details" not in gm.PrimaryChatCompletionChunk.model_fields,
                reason="SDK 0.2.3 does not parse object-shaped additional_data",
            ),
        ),
    ],
)
def test_repeated_metadata_snapshots_do_not_concatenate_strings(
    additional_data: Any,
) -> None:
    state = primary.StreamState()
    values = {
        "additional_data": additional_data,
        "error_details": {"reason": "filtered"},
    }
    chunks = [
        _convert(
            {
                "event": event,
                "messages": [{"role": "assistant", "content": [{"text": ""}]}],
                **values,
            },
            state,
        )
        for event in ("response.message.delta", "response.message.done")
    ]
    metadata = reduce(add, chunks).message.response_metadata
    assert metadata["additional_data"] == values["additional_data"]
    assert metadata["error_details"] == values["error_details"]


def test_metadata_snapshots_keep_the_latest_value_at_completion() -> None:
    state = primary.StreamState()
    primary.convert_stream_event(
        _event({"event": "response.message.delta", "error_details": {"reason": "one"}}),
        state=state,
    )
    chunk = _convert(
        {"event": "response.message.done", "error_details": {"reason": "two"}},
        state,
    )
    assert chunk.message.response_metadata["error_details"] == {"reason": "two"}


def test_metadata_from_earlier_event_is_kept_when_terminal_omits_it() -> None:
    state = primary.StreamState()
    primary.convert_stream_event(
        _event(
            {"event": "response.message.delta", "additional_data": [{"trace": "abc"}]}
        ),
        state=state,
    )
    chunk = _convert({"event": "response.message.done"}, state)
    assert chunk.message.response_metadata["additional_data"] == [{"trace": "abc"}]


def test_unnamed_terminal_event_preserves_call_id_and_continuation_state() -> None:
    event = _event(
        {
            "finish_reason": "function_call",
            "tools_state_id": "state",
            "messages": [
                {
                    "role": "assistant",
                    "function_call": {
                        "id": "real-call",
                        "name": "lookup",
                        "arguments": {},
                    },
                }
            ],
        }
    )
    assert event.event is None
    chunk = _convert(event)
    assert isinstance(chunk.message, AIMessageChunk)
    assert chunk.message.chunk_position == "last"
    assert chunk.message.additional_kwargs["tools_state_id"] == "state"
    replayed = primary.convert_messages([chunk.message], cached_uploads={})[0]
    assert replayed.tools_state_id == "state"
    assert replayed.content is not None
    assert replayed.content[0].function_call is not None
    assert (
        replayed.content[0].function_call.model_dump(by_alias=True)["id"] == "real-call"
    )
