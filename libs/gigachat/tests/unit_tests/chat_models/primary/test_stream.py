from functools import reduce
from operator import add
from typing import Any, cast

import gigachat.models as gm
import pytest
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from langchain_gigachat.chat_models._contracts import primary


def _convert(
    event: gm.PrimaryChatCompletionChunk | dict,
    state: primary.StreamState | None = None,
) -> ChatGenerationChunk:
    chunk = primary.convert_stream_event(
        event,
        state=state or primary.StreamState(),
    )
    assert chunk is not None
    assert isinstance(chunk.message, AIMessageChunk)
    return chunk


def _message(chunk: ChatGenerationChunk) -> AIMessageChunk:
    return cast(AIMessageChunk, chunk.message)


def _content_blocks(chunk: ChatGenerationChunk) -> list[dict]:
    content = chunk.message.content
    assert isinstance(content, list)
    assert all(isinstance(block, dict) for block in content)
    return cast(list[dict], content)


def test_converts_sdk_message_delta_and_uses_request_id() -> None:
    state = primary.StreamState()
    event = gm.PrimaryChatCompletionChunk(
        event="response.message.delta",
        model="GigaChat-3-Ultra",
        message_id="provider-message",
        thread_id="thread-1",
        x_headers={"x-request-id": "request-1"},
        messages=[
            gm.ChatMessageChunk(
                role="assistant",
                content=[gm.ChatContentPart(text="Привет")],
            )
        ],
    )

    chunk = _convert(event, state)

    assert chunk.text == "Привет"
    assert chunk.message.content == "Привет"
    assert chunk.message.id == "request-1"
    assert chunk.message.response_metadata == {
        "event": "response.message.delta",
        "message_id": "provider-message",
        "model": "GigaChat-3-Ultra",
        "thread_id": "thread-1",
        "x_headers": {"x-request-id": "request-1"},
    }
    assert state.first_chunk is False
    assert state.message_id == "request-1"


def test_sdk_event_timestamps_are_not_stream_identity() -> None:
    state = primary.StreamState()
    delta = _convert(
        gm.PrimaryChatCompletionChunk(
            event="response.message.delta",
            created_at=1760434637,
            messages=[
                gm.ChatMessageChunk(
                    role="assistant",
                    content=[gm.ChatContentPart(text="answer")],
                )
            ],
        ),
        state,
    )
    done = _convert(
        gm.PrimaryChatCompletionChunk(
            event="response.message.done",
            created_at=1760434638,
        ),
        state,
    )

    aggregate = delta + done

    assert delta.message.response_metadata["created_at"] == 1760434637
    assert "created_at" not in done.message.response_metadata
    assert aggregate.message.response_metadata["created_at"] == 1760434637
    assert state.created_at == 1760434637


def test_fragmented_text_aggregates_and_callback_text_is_only_text() -> None:
    state = primary.StreamState()
    chunks = [
        _convert(
            {
                "event": "response.message.delta",
                "message_id": "message-1",
                "messages": [{"content": [{"text": text}]}],
            },
            state,
        )
        for text in ("one", " ", "two")
    ]

    aggregate = reduce(add, chunks)

    assert [chunk.text for chunk in chunks] == ["one", " ", "two"]
    assert aggregate.text == "one two"
    assert aggregate.message.content == "one two"
    assert {chunk.message.id for chunk in chunks} == {"message-1"}


def test_text_and_final_metadata_aggregate_semantically() -> None:
    state = primary.StreamState()
    chunks = [
        _convert(
            {
                "event": "response.message.delta",
                "messages": [
                    {
                        "message_id": "message-1",
                        "content": [{"text": "answer"}],
                    }
                ],
            },
            state,
        ),
        _convert(
            {
                "event": "response.message.done",
                "finish_reason": "stop",
                "usage": {
                    "input_tokens": 2,
                    "output_tokens": 1,
                    "total_tokens": 3,
                },
            },
            state,
        ),
    ]

    aggregate = chunks[0] + chunks[1]

    assert aggregate.text == "answer"
    assert aggregate.message.id == "message-1"
    assert aggregate.generation_info == {"finish_reason": "stop"}
    assert _message(aggregate).usage_metadata == {
        "input_tokens": 2,
        "output_tokens": 1,
        "total_tokens": 3,
    }


def test_fragmented_function_call_keeps_stable_id_and_index() -> None:
    state = primary.StreamState()
    first = _convert(
        {
            "event": "response.message.delta",
            "message_id": "message-1",
            "tools_state_id": "tools-1",
            "messages": [
                {
                    "function_call": {
                        "name": "weather",
                        "arguments": '{"city":',
                    }
                }
            ],
        },
        state,
    )
    second = _convert(
        {
            "event": "response.message.delta",
            "tools_state_id": "tools-1",
            "messages": [
                {
                    "function_call": {
                        "name": "weather",
                        "arguments": '"Moscow"}',
                    }
                }
            ],
        },
        state,
    )

    first_call = _message(first).tool_call_chunks[0]
    second_call = _message(second).tool_call_chunks[0]
    aggregate = first + second

    assert first.text == second.text == ""
    assert first_call == {
        "name": "weather",
        "args": '{"city":',
        "id": "tools-1",
        "index": 0,
        "type": "tool_call_chunk",
    }
    assert second_call == {
        "name": None,
        "args": '"Moscow"}',
        "id": "tools-1",
        "index": 0,
        "type": "tool_call_chunk",
    }
    assert _message(aggregate).tool_calls == [
        {
            "name": "weather",
            "args": {"city": "Moscow"},
            "id": "tools-1",
            "type": "tool_call",
        }
    ]
    assert state.tools_state_id == "tools-1"
    assert state.next_block_index == 1


def test_tools_state_before_function_does_not_hide_first_fragment_name() -> None:
    state = primary.StreamState()
    metadata = _convert(
        {
            "event": "response.message.delta",
            "tools_state_id": "tools-1",
        },
        state,
    )
    first = _convert(
        {
            "event": "response.message.delta",
            "messages": [
                {
                    "function_call": {
                        "name": "weather",
                        "arguments": '{"city":',
                    }
                }
            ],
        },
        state,
    )
    second = _convert(
        {
            "event": "response.message.delta",
            "messages": [
                {
                    "function_call": {
                        "name": "weather",
                        "arguments": '"Moscow"}',
                    }
                }
            ],
        },
        state,
    )

    calls = [
        _message(first).tool_call_chunks[0],
        _message(second).tool_call_chunks[0],
    ]
    aggregate = reduce(add, [metadata, first, second])

    assert metadata.text == first.text == second.text == ""
    assert [call["name"] for call in calls] == ["weather", None]
    assert {call["id"] for call in calls} == {"tools-1"}
    assert {call["index"] for call in calls} == {0}
    assert _message(aggregate).tool_calls[0]["args"] == {"city": "Moscow"}
    assert state.client_tool_started is True
    assert state.client_tool_id == "tools-1"
    assert state.client_tool_index == 0


@pytest.mark.parametrize(
    ("field", "first_value", "second_value", "error"),
    [
        ("name", "weather", "forecast", "Conflicting primary client tool names"),
        ("id", "call-1", "call-2", "Conflicting primary client tool IDs"),
    ],
)
def test_conflicting_client_tool_identity_fails_clearly(
    field: str,
    first_value: str,
    second_value: str,
    error: str,
) -> None:
    state = primary.StreamState()
    first_call = {
        "name": "weather",
        "arguments": "{",
        field: first_value,
    }
    second_call = {
        "name": "weather",
        "arguments": "}",
        field: second_value,
    }
    _convert(
        {
            "messages": [{"function_call": first_call}],
        },
        state,
    )

    with pytest.raises(ValueError, match=error):
        _convert(
            {
                "messages": [{"function_call": second_call}],
            },
            state,
        )


def test_invalid_fragmented_function_arguments_become_invalid_tool_call() -> None:
    state = primary.StreamState()
    chunks = [
        _convert(
            {
                "event": "response.message.delta",
                "tools_state_id": "tools-1",
                "messages": [
                    {
                        "function_call": {
                            "name": "weather",
                            "arguments": fragment,
                        }
                    }
                ],
            },
            state,
        )
        for fragment in ('{"city":', "invalid}")
    ]
    chunks.append(
        _convert(
            {
                "event": "response.message.done",
                "finish_reason": "tool_calls",
            },
            state,
        )
    )

    aggregate = reduce(add, chunks)

    assert _message(aggregate).tool_calls == []
    assert _message(aggregate).invalid_tool_calls == [
        {
            "name": "weather",
            "args": '{"city":invalid}',
            "id": "tools-1",
            "error": None,
            "type": "invalid_tool_call",
        }
    ]


def test_function_call_dict_arguments_are_serialized_without_mutation() -> None:
    event: dict[str, Any] = {
        "event": "response.message.delta",
        "tools_state_id": "tools-1",
        "messages": [
            {
                "function_call": {
                    "name": "weather",
                    "arguments": {"city": "Moscow"},
                }
            }
        ],
    }

    chunk = _convert(event)

    assert _message(chunk).tool_call_chunks[0]["args"] == '{"city":"Moscow"}'
    assert event["messages"][0]["function_call"]["arguments"] == {"city": "Moscow"}


def test_tool_progress_is_a_server_tool_call_chunk() -> None:
    chunk = _convert(
        {
            "event": "response.tool.started",
            "message_id": "message-1",
            "tools_state_id": "tool-1",
            "tool_execution": {
                "name": "web_search",
                "status": "running",
                "seconds_left": 2,
            },
        }
    )

    assert chunk.text == ""
    assert chunk.message.content == [
        {
            "type": "server_tool_call_chunk",
            "id": "tool-1",
            "name": "web_search",
            "args": "",
            "index": 0,
            "extras": {
                "provider_tool_execution": {
                    "name": "web_search",
                    "status": "running",
                    "seconds_left": 2,
                }
            },
        }
    ]


def test_tool_completed_is_a_metadata_only_server_tool_result() -> None:
    chunk = _convert(
        {
            "event": "response.tool.completed",
            "message_id": "message-1",
            "tools_state_id": "tool-1",
            "tool_execution": {
                "name": "code_interpreter",
                "status": "completed",
                "output": {"stdout": "42"},
            },
        }
    )

    assert chunk.text == ""
    assert chunk.message.content == [
        {
            "type": "server_tool_result",
            "id": "tool-1:result",
            "tool_call_id": "tool-1",
            "status": "success",
            "output": {"stdout": "42"},
            "index": 0,
            "extras": {
                "provider_tool_execution": {
                    "name": "code_interpreter",
                    "status": "completed",
                    "output": {"stdout": "42"},
                }
            },
        }
    ]
    assert chunk.message.response_metadata["event"] == "response.tool.completed"


def test_sdk_tool_completed_event_deduplicates_execution_and_keeps_sources(
    tool_completed_event: gm.PrimaryChatCompletionChunk,
) -> None:
    chunk = _convert(tool_completed_event)
    blocks = _content_blocks(chunk)
    result_blocks = [block for block in blocks if block["type"] == "server_tool_result"]
    tool_coordinates = [
        (block["index"], block["tool_call_id"])
        for block in blocks
        if block["type"] == "server_tool_result"
    ]

    assert len(result_blocks) == 1
    assert len(tool_coordinates) == len(set(tool_coordinates)) == 1
    assert result_blocks[0]["extras"]["inline_data"]["sources"] == {
        "source-001": {
            "url": "https://example.test/weather",
            "title": "Weather source",
        }
    }


def test_server_tool_lifecycle_keeps_call_index_and_separate_result_index() -> None:
    state = primary.StreamState()
    started = _convert(
        {
            "event": "response.tool.started",
            "tool_execution": {
                "call_id": "server-1",
                "name": "web_search",
                "status": "running",
                "arguments": '{"query":',
            },
        },
        state,
    )
    delta = _convert(
        {
            "event": "response.tool.delta",
            "tool_execution": {
                "call_id": "server-1",
                "name": "web_search",
                "status": "running",
                "arguments": '"Moscow"}',
            },
        },
        state,
    )
    completed = _convert(
        {
            "event": "response.tool.completed",
            "tool_execution": {
                "call_id": "server-1",
                "name": "web_search",
                "status": "completed",
                "output": {"matches": 1},
            },
        },
        state,
    )

    started_block = _content_blocks(started)[0]
    delta_block = _content_blocks(delta)[0]
    result_block = _content_blocks(completed)[0]
    aggregate = reduce(add, [started, delta, completed])
    aggregate_blocks = _content_blocks(aggregate)

    assert started.text == delta.text == completed.text == ""
    assert started_block["id"] == delta_block["id"] == "server-1"
    assert started_block["index"] == delta_block["index"] == 0
    assert started_block["name"] == "web_search"
    assert delta_block["name"] == ""
    assert result_block["tool_call_id"] == "server-1"
    assert result_block["index"] == 1
    assert aggregate_blocks[0]["name"] == "web_search"
    assert aggregate_blocks[0]["args"] == '{"query":"Moscow"}'
    assert aggregate_blocks[1]["type"] == "server_tool_result"
    assert state.server_tool_indexes == {"server-1": 0}
    assert state.server_tool_result_indexes == {"server-1": 1}


def test_done_without_messages_preserves_finish_usage_and_metadata() -> None:
    state = primary.StreamState()
    chunk = _convert(
        {
            "event": "response.message.done",
            "message_id": "message-1",
            "thread_id": "thread-1",
            "finish_reason": "stop",
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

    assert chunk.text == ""
    assert chunk.message.content == []
    assert chunk.generation_info == {"finish_reason": "stop"}
    assert chunk.message.response_metadata == {
        "event": "response.message.done",
        "finish_reason": "stop",
        "message_id": "message-1",
        "thread_id": "thread-1",
    }
    assert _message(chunk).usage_metadata == {
        "input_tokens": 8,
        "output_tokens": 5,
        "total_tokens": 13,
        "input_token_details": {"cache_read": 3},
    }


def test_usage_only_event_is_not_dropped() -> None:
    chunk = _convert(
        {
            "usage": {
                "input_tokens": 2,
                "output_tokens": 1,
            }
        }
    )

    assert chunk.text == ""
    assert _message(chunk).usage_metadata == {
        "input_tokens": 2,
        "output_tokens": 1,
        "total_tokens": 3,
    }


def test_late_metadata_is_emitted_once_and_survives_aggregation() -> None:
    state = primary.StreamState()
    first = _convert(
        {
            "event": "response.message.delta",
            "messages": [{"content": [{"text": "answer"}]}],
        },
        state,
    )
    final = _convert(
        {
            "event": "response.message.done",
            "message_id": "provider-message",
            "tools_state_id": "tools-state",
            "thread_id": "thread-1",
            "model": "GigaChat-3-Ultra",
            "created_at": 1780321868,
            "x_headers": {
                "x-request-id": "late-request",
                "x-trace-id": "trace-1",
            },
            "finish_reason": "stop",
        },
        state,
    )
    repeated = _convert(
        {
            "event": "response.message.done",
            "message_id": "provider-message",
            "tools_state_id": "tools-state",
            "thread_id": "thread-1",
            "model": "GigaChat-3-Ultra",
            "created_at": 1780321868,
            "x_headers": {
                "x-request-id": "late-request",
                "x-trace-id": "trace-1",
            },
        },
        state,
    )

    aggregate = reduce(add, [first, final, repeated])
    metadata = aggregate.message.response_metadata

    assert first.message.id == final.message.id == repeated.message.id
    assert final.message.response_metadata["message_id"] == "provider-message"
    assert "message_id" not in repeated.message.response_metadata
    assert metadata["message_id"] == "provider-message"
    assert metadata["tools_state_id"] == "tools-state"
    assert metadata["thread_id"] == "thread-1"
    assert metadata["model"] == "GigaChat-3-Ultra"
    assert metadata["created_at"] == 1780321868
    assert metadata["x_headers"] == {
        "x-request-id": "late-request",
        "x-trace-id": "trace-1",
    }
    assert state.provider_message_id == "provider-message"
    assert state.tools_state_id == "tools-state"
    assert state.thread_id == "thread-1"
    assert state.model == "GigaChat-3-Ultra"
    assert state.created_at == 1780321868


def test_late_header_addition_merges_without_repeating_existing_values() -> None:
    state = primary.StreamState()
    first = _convert({"x_headers": {"x-request-id": "request-1"}}, state)
    second = _convert(
        {
            "event": "response.message.done",
            "x_headers": {
                "x-request-id": "request-1",
                "x-trace-id": "trace-1",
            },
        },
        state,
    )

    aggregate = first + second

    assert first.message.response_metadata["x_headers"] == {"x-request-id": "request-1"}
    assert second.message.response_metadata["x_headers"] == {"x-trace-id": "trace-1"}
    assert aggregate.message.response_metadata["x_headers"] == {
        "x-request-id": "request-1",
        "x-trace-id": "trace-1",
    }


def test_unknown_event_preserves_raw_provider_payload() -> None:
    event = {
        "event": "response.future.delta",
        "message_id": "message-1",
        "future_field": {"answer": 42},
    }

    chunk = _convert(event)

    assert chunk.text == ""
    assert chunk.message.response_metadata["event"] == "response.future.delta"
    assert chunk.message.response_metadata["provider_fields"] == {
        "future_field": {"answer": 42}
    }
    assert chunk.message.response_metadata["raw_event"] == event


@pytest.mark.parametrize(
    ("event_name", "finish_reason"),
    [
        ("response.error", "error"),
        ("response.tool.failed", "tool_error"),
    ],
)
def test_error_events_are_metadata_chunks(
    event_name: str,
    finish_reason: str,
) -> None:
    chunk = _convert(
        {
            "event": event_name,
            "finish_reason": finish_reason,
            "error": {"message": "boom"},
        }
    )

    assert chunk.text == ""
    assert chunk.generation_info == {"finish_reason": finish_reason}
    assert chunk.message.response_metadata["provider_fields"] == {
        "error": {"message": "boom"}
    }


def test_files_citations_and_reasoning_keep_monotonic_indexes() -> None:
    state = primary.StreamState()
    chunk = _convert(
        {
            "event": "response.message.delta",
            "messages": [
                {
                    "content": [
                        {
                            "text": "source",
                            "inline_data": {
                                "sources": {
                                    "source-1": {
                                        "url": "https://example.test",
                                        "title": "Example",
                                    }
                                }
                            },
                        },
                        {
                            "files": [
                                {"id": "image-1", "mime": "image/png"},
                                {"id": "audio-1", "mime": "audio/wav"},
                                {
                                    "id": "document-1",
                                    "mime": "application/pdf",
                                    "target": "download",
                                },
                            ]
                        },
                        {"reasoning_content": "checking"},
                    ]
                }
            ],
        },
        state,
    )

    blocks = _content_blocks(chunk)
    assert [block["index"] for block in blocks] == [0, 1, 2, 3, 4]
    assert blocks[0] == {
        "type": "text",
        "text": "source",
        "index": 0,
        "annotations": [
            {
                "type": "citation",
                "id": "source-1",
                "url": "https://example.test",
                "title": "Example",
            }
        ],
    }
    assert blocks[1]["type"] == "image"
    assert blocks[2]["type"] == "audio"
    assert blocks[3] == {
        "type": "file",
        "file_id": "document-1",
        "mime_type": "application/pdf",
        "index": 3,
        "extras": {"target": "download"},
    }
    assert blocks[4] == {
        "type": "reasoning",
        "reasoning": "checking",
        "index": 4,
    }
    assert chunk.text == "source"
    assert state.next_block_index == 5


def test_reasoning_role_and_video_match_non_stream_content_semantics() -> None:
    state = primary.StreamState()
    stream_chunk = _convert(
        {
            "event": "response.message.delta",
            "messages": [
                {"role": "reasoning", "content": [{"text": "Think"}]},
                {
                    "role": "assistant",
                    "content": [
                        {"text": "Answer"},
                        {"files": [{"id": "video-1", "mime": "video/mp4"}]},
                    ],
                },
            ],
        },
        state,
    )
    response = gm.ChatCompletionResponse.model_validate(
        {
            "model": "GigaChat-3-Ultra",
            "created_at": 1780321868,
            "messages": [
                {"role": "reasoning", "content": [{"text": "Think"}]},
                {
                    "role": "assistant",
                    "content": [
                        {"text": "Answer"},
                        {"files": [{"id": "video-1", "mime": "video/mp4"}]},
                    ],
                },
            ],
        }
    )
    non_stream_message = primary.create_chat_result(response).generations[0].message
    stream_blocks = [
        {key: value for key, value in block.items() if key != "index"}
        for block in _content_blocks(stream_chunk)
    ]

    assert stream_chunk.text == "Answer"
    assert [block["index"] for block in _content_blocks(stream_chunk)] == [0, 1, 2]
    assert stream_blocks == non_stream_message.content
    assert stream_blocks[0] == {"type": "reasoning", "reasoning": "Think"}
    assert stream_blocks[2]["type"] == "video"


def test_unknown_message_fields_match_non_stream_content_semantics() -> None:
    messages = [
        {
            "role": "assistant",
            "content": [{"text": "Answer"}],
            "future_message_data": {"trace": "provider-value"},
        }
    ]
    stream_chunk = _convert(
        {
            "event": "response.message.delta",
            "messages": messages,
        }
    )
    response = gm.ChatCompletionResponse.model_validate(
        {
            "model": "GigaChat-3-Ultra",
            "created_at": 1780321868,
            "messages": messages,
        }
    )
    non_stream_message = primary.create_chat_result(response).generations[0].message
    stream_blocks = [
        {key: value for key, value in block.items() if key != "index"}
        for block in _content_blocks(stream_chunk)
    ]

    assert stream_chunk.text == "Answer"
    assert stream_blocks == non_stream_message.content
    assert stream_blocks[1] == {
        "type": "non_standard",
        "value": {"future_message_data": {"trace": "provider-value"}},
    }


def test_empty_mapping_is_ignored() -> None:
    assert primary.convert_stream_event({}, state=primary.StreamState()) is None


def test_invalid_event_type_has_clear_error() -> None:
    with pytest.raises(TypeError, match="SDK models or mappings"):
        primary.convert_stream_event(42, state=primary.StreamState())  # type: ignore[arg-type]
