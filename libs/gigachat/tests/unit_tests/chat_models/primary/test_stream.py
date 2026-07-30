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
    assert [block["index"] for block in blocks] == [0, 1, 2, 3]
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
        "type": "reasoning",
        "reasoning": "checking",
        "index": 3,
    }
    assert chunk.text == "source"
    assert state.next_block_index == 4


def test_empty_mapping_is_ignored() -> None:
    assert primary.convert_stream_event({}, state=primary.StreamState()) is None


def test_invalid_event_type_has_clear_error() -> None:
    with pytest.raises(TypeError, match="SDK models or mappings"):
        primary.convert_stream_event(42, state=primary.StreamState())  # type: ignore[arg-type]
