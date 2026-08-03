from functools import reduce
from operator import add
from typing import Any, cast

import gigachat.models as gm
import pytest
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from langchain_gigachat.chat_models._contracts import primary
from langchain_gigachat.chat_models._contracts.primary.types import PrimaryStreamError


def _convert(
    event: gm.PrimaryChatCompletionChunk | dict,
    state: primary.StreamState | None = None,
) -> ChatGenerationChunk:
    if isinstance(event, dict):
        event = gm.PrimaryChatCompletionChunk.model_validate(event)
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
    assert chunk.message.content == [
        {
            "type": "text",
            "text": "Привет",
            "index": 0,
        }
    ]
    assert chunk.message.id == "request-1"
    assert chunk.message.response_metadata == {
        "events": ["response.message.delta"],
        "message_id": "provider-message",
        "model": "GigaChat-3-Ultra",
        "output_version": "v1",
        "thread_id": "thread-1",
        "x_headers": {"x-request-id": "request-1"},
    }
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


def test_event_names_aggregate_as_an_ordered_list() -> None:
    state = primary.StreamState()
    chunks = [
        _convert({"event": event_name}, state)
        for event_name in (
            "response.message.delta",
            "response.message.delta",
            "response.message.done",
        )
    ]

    aggregate = reduce(add, chunks)

    assert aggregate.message.response_metadata["events"] == [
        "response.message.delta",
        "response.message.delta",
        "response.message.done",
    ]
    assert aggregate.message.response_metadata["output_version"] == "v1"


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
    assert aggregate.message.content == [
        {"type": "text", "text": "one two", "index": 0},
    ]
    assert len({chunk.message.id for chunk in chunks}) == 1
    assert chunks[0].message.id is not None
    assert chunks[0].message.id.startswith("lc_")


def test_fragmented_reasoning_is_available_in_additional_kwargs() -> None:
    state = primary.StreamState()
    chunks = [
        _convert(
            {
                "event": "response.message.delta",
                "messages": [
                    {"role": "reasoning", "content": [{"text": text}]},
                ],
            },
            state,
        )
        for text in ("Think", "ing")
    ]
    chunks.append(
        _convert(
            {
                "event": "response.message.delta",
                "messages": [
                    {"role": "assistant", "content": [{"text": "Answer"}]},
                ],
            },
            state,
        )
    )

    aggregate = reduce(add, chunks)

    assert [
        chunk.message.additional_kwargs.get("reasoning_content") for chunk in chunks
    ] == ["Think", "ing", None]
    assert aggregate.message.additional_kwargs["reasoning_content"] == "Thinking"


@pytest.mark.parametrize("level", ["message", "part"])
def test_conflicting_reasoning_aliases_fail_closed(level: str) -> None:
    message: dict[str, Any] = {"role": "reasoning"}
    aliases = {"reasoning": "First", "reasoning_content": "Second"}
    if level == "message":
        message.update(aliases)
    else:
        message["content"] = [aliases]

    with pytest.raises(ValueError, match="conflicting reasoning"):
        _convert(
            {
                "event": "response.message.delta",
                "messages": [message],
            }
        )


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


def test_conflicting_client_tool_name_fails_clearly() -> None:
    state = primary.StreamState()
    first_call = {
        "name": "weather",
        "arguments": "{",
    }
    second_call = {
        "name": "forecast",
        "arguments": "}",
    }
    _convert(
        {
            "messages": [{"function_call": first_call}],
        },
        state,
    )

    with pytest.raises(ValueError, match="Conflicting primary client tool names"):
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
    state = primary.StreamState()
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
        },
        state,
    )
    done = _convert(
        {
            "event": "response.message.done",
            "tools_state_id": "tool-1",
            "finish_reason": "stop",
        },
        state,
    )

    assert chunk.text == ""
    block = _content_blocks(chunk)[0]
    assert block["type"] == "server_tool_call_chunk"
    assert block["id"] == "lc_primary-server-tool-0"
    assert block["name"] == "web_search"
    assert block["args"] == ""
    assert block["index"] == 0
    assert block["extras"]["provider_tool_execution"] == {
        "name": "web_search",
        "status": "running",
        "seconds_left": 2,
    }
    assert done.message.additional_kwargs["provider_server_tool_state_by_call_id"] == {
        "lc_primary-server-tool-0": "tool-1"
    }


def test_conflicting_server_tool_identity_aliases_fail_closed() -> None:
    with pytest.raises(ValueError, match="identity aliases contain conflicting"):
        _convert(
            {
                "event": "response.tool.in_progress",
                "tool_execution": {
                    "call_id": "call-A",
                    "tool_call_id": "call-B",
                    "name": "web_search",
                    "status": "running",
                },
            }
        )


def test_server_tool_name_can_arrive_after_started() -> None:
    state = primary.StreamState()
    started = _convert(
        {
            "event": "response.tool.started",
            "tool_execution": {
                "call_id": "tool-1",
                "status": "running",
            },
        },
        state,
    )
    delta = _convert(
        {
            "event": "response.tool.delta",
            "tool_execution": {
                "call_id": "tool-1",
                "name": "web_search",
                "status": "running",
            },
        },
        state,
    )

    assert _content_blocks(started)[0]["name"] == ""
    assert _content_blocks(delta)[0]["name"] == "web_search"
    aggregate_block = _content_blocks(started + delta)[0]
    assert aggregate_block["name"] == "web_search"
    assert aggregate_block["extras"]["provider_tool_execution_updates"] == [
        {
            "call_id": "tool-1",
            "name": "web_search",
            "status": "running",
        }
    ]
    assert state.server_tool_names == {"tool-1": "web_search"}


def test_server_tool_name_can_remain_missing_until_completed() -> None:
    state = primary.StreamState()
    started = _convert(
        {
            "event": "response.tool.started",
            "tool_execution": {
                "call_id": "tool-1",
                "status": "running",
            },
        },
        state,
    )
    completed = _convert(
        {
            "event": "response.tool.completed",
            "tool_execution": {
                "call_id": "tool-1",
                "status": "completed",
            },
        },
        state,
    )

    assert _content_blocks(started)[0]["name"] == ""
    assert _content_blocks(completed)[0]["tool_call_id"] == "tool-1"
    assert state.server_tool_names == {}


def test_conflicting_real_server_tool_names_fail_clearly() -> None:
    state = primary.StreamState()
    _convert(
        {
            "event": "response.tool.started",
            "tool_execution": {
                "call_id": "tool-1",
                "name": "web_search",
                "status": "running",
            },
        },
        state,
    )

    with pytest.raises(
        ValueError,
        match="Conflicting names for primary server tool 'tool-1'",
    ):
        _convert(
            {
                "event": "response.tool.completed",
                "tool_execution": {
                    "call_id": "tool-1",
                    "name": "code_interpreter",
                    "status": "completed",
                },
            },
            state,
        )


def test_tool_completed_is_a_metadata_only_server_tool_result() -> None:
    state = primary.StreamState()
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
        },
        state,
    )
    done = _convert(
        {
            "event": "response.message.done",
            "tools_state_id": "tool-1",
            "finish_reason": "stop",
        },
        state,
    )

    assert chunk.text == ""
    block = _content_blocks(chunk)[0]
    assert block["type"] == "server_tool_result"
    assert block["id"] == "lc_primary-server-tool-0:result"
    assert block["tool_call_id"] == "lc_primary-server-tool-0"
    assert block["status"] == "success"
    assert block["output"] == {"stdout": "42"}
    assert block["index"] == 0
    assert block["extras"]["provider_tool_execution"] == {
        "name": "code_interpreter",
        "status": "completed",
        "output": {"stdout": "42"},
    }
    assert done.message.additional_kwargs["provider_server_tool_state_by_call_id"] == {
        "lc_primary-server-tool-0": "tool-1"
    }
    assert chunk.message.response_metadata["events"] == ["response.tool.completed"]


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


def test_web_search_inline_data_updates_server_tool_result() -> None:
    state = primary.StreamState()
    chunks = [
        _convert(
            {
                "event": "response.tool.in_progress",
                "messages": [
                    {
                        "role": "assistant",
                        "content": [{"tool_execution": {"name": "web_search"}}],
                    }
                ],
            },
            state,
        ),
        _convert(
            {
                "event": "response.tool.completed",
                "tools_state_id": "web-search-state-1",
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "tool_execution": {
                                    "name": "web_search",
                                    "status": "success",
                                }
                            }
                        ],
                    }
                ],
            },
            state,
        ),
        _convert(
            {
                "event": "response.message.delta",
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "inline_data": {
                                    "images": [],
                                    "sources": {
                                        "1": {
                                            "url": "https://example.test/news",
                                            "title": "Example news",
                                        }
                                    },
                                }
                            }
                        ],
                    }
                ],
            },
            state,
        ),
        _convert(
            {
                "event": "response.message.delta",
                "messages": [
                    {
                        "role": "assistant",
                        "content": [{"text": "Latest news"}],
                    }
                ],
            },
            state,
        ),
    ]

    inline_data_blocks = _content_blocks(chunks[2])
    aggregate_blocks = _content_blocks(reduce(add, chunks))

    assert inline_data_blocks == [
        {
            "type": "non_standard",
            "value": {
                "inline_data": {
                    "images": [],
                    "sources": {
                        "1": {
                            "url": "https://example.test/news",
                            "title": "Example news",
                        }
                    },
                }
            },
            "index": 1,
        }
    ]
    assert all(block["type"] != "non_standard" for block in aggregate_blocks)
    assert aggregate_blocks[1]["type"] == "server_tool_result"
    assert aggregate_blocks[1]["extras"]["inline_data"]["sources"]["1"] == {
        "url": "https://example.test/news",
        "title": "Example news",
    }
    assert aggregate_blocks[2] == {
        "type": "text",
        "text": "Latest news",
        "index": 2,
    }
    assert state.pending_server_tool_result_ids == set()


def test_late_inline_data_is_not_assigned_after_multiple_terminal_tools() -> None:
    state = primary.StreamState()
    completed = _convert(
        {
            "event": "response.tool.completed",
            "messages": [
                {
                    "tool_execution": {
                        "call_id": "tool-1",
                        "name": "web_search",
                        "status": "completed",
                    }
                },
                {
                    "tool_execution": {
                        "call_id": "tool-2",
                        "name": "image_generate",
                        "status": "completed",
                    }
                },
            ],
        },
        state,
    )
    assert state.pending_server_tool_result_ids == {"tool-1", "tool-2"}
    inline_data = {"sources": {"1": {"url": "https://example.test/ambiguous"}}}
    inline = _convert(
        {
            "event": "response.message.delta",
            "messages": [{"content": [{"inline_data": inline_data}]}],
        },
        state,
    )

    assert _content_blocks(inline) == [
        {
            "type": "non_standard",
            "value": {"inline_data": inline_data},
        }
    ]
    aggregate_blocks = _content_blocks(completed + inline)
    assert [block["tool_call_id"] for block in aggregate_blocks[:2]] == [
        "tool-1",
        "tool-2",
    ]
    assert all(
        "inline_data" not in block.get("extras", {}) for block in aggregate_blocks[:2]
    )
    assert state.pending_server_tool_result_ids == {"tool-1", "tool-2"}


def test_colocated_inline_data_leaves_the_other_result_pending() -> None:
    state = primary.StreamState()
    first = _convert(
        {
            "event": "response.tool.completed",
            "tools_state_id": "state-1",
            "tool_execution": {
                "call_id": "tool-1",
                "name": "web_search",
                "status": "completed",
            },
        },
        state,
    )
    second_inline_data = {"images": [{"id": "image-2"}]}
    second = _convert(
        {
            "event": "response.tool.completed",
            "messages": [
                {
                    "tools_state_id": "state-2",
                    "content": [
                        {
                            "tool_execution": {
                                "call_id": "tool-2",
                                "name": "image_generate",
                                "status": "completed",
                            },
                            "inline_data": second_inline_data,
                        }
                    ],
                }
            ],
        },
        state,
    )
    first_inline_data = {"sources": {"1": {"url": "https://example.test/first"}}}
    late = _convert(
        {
            "event": "response.message.delta",
            "messages": [{"content": [{"inline_data": first_inline_data}]}],
        },
        state,
    )

    assert _content_blocks(late)[0]["index"] == 0
    aggregate_blocks = _content_blocks(first + second + late)
    assert aggregate_blocks[0]["extras"]["inline_data"] == first_inline_data
    assert aggregate_blocks[1]["extras"]["inline_data"] == second_inline_data


def test_state_tagged_inline_data_resolves_one_of_multiple_pending_results() -> None:
    state = primary.StreamState()
    completed = _convert(
        {
            "event": "response.tool.completed",
            "messages": [
                {
                    "tools_state_id": "state-1",
                    "tool_execution": {
                        "call_id": "tool-1",
                        "name": "web_search",
                        "status": "completed",
                    },
                },
                {
                    "tools_state_id": "state-2",
                    "tool_execution": {
                        "call_id": "tool-2",
                        "name": "image_generate",
                        "status": "completed",
                    },
                },
            ],
        },
        state,
    )
    first_inline_data = {"sources": {"1": {"url": "https://example.test/first"}}}
    first_inline = _convert(
        {
            "event": "response.message.delta",
            "messages": [
                {
                    "tools_state_id": "state-1",
                    "content": [{"inline_data": first_inline_data}],
                }
            ],
        },
        state,
    )
    second_inline_data = {"images": [{"id": "image-2"}]}
    second_inline = _convert(
        {
            "event": "response.message.delta",
            "messages": [{"content": [{"inline_data": second_inline_data}]}],
        },
        state,
    )

    assert _content_blocks(first_inline)[0]["index"] == 0
    assert _content_blocks(second_inline)[0]["index"] == 1
    aggregate_blocks = _content_blocks(completed + first_inline + second_inline)
    assert aggregate_blocks[0]["extras"]["inline_data"] == first_inline_data
    assert aggregate_blocks[1]["extras"]["inline_data"] == second_inline_data


def test_pending_result_clears_when_unrelated_tool_begins() -> None:
    state = primary.StreamState()
    completed = _convert(
        {
            "event": "response.tool.completed",
            "tool_execution": {
                "call_id": "tool-1",
                "name": "web_search",
                "status": "completed",
            },
        },
        state,
    )
    assert state.pending_server_tool_result_ids == {"tool-1"}

    started = _convert(
        {
            "event": "response.tool.started",
            "tool_execution": {
                "call_id": "tool-2",
                "name": "code_interpreter",
                "status": "running",
            },
        },
        state,
    )
    inline = _convert(
        {
            "event": "response.message.delta",
            "messages": [
                {
                    "content": [
                        {
                            "inline_data": {
                                "sources": {
                                    "1": {"url": "https://example.test/unrelated"}
                                }
                            }
                        }
                    ]
                }
            ],
        },
        state,
    )

    aggregate_blocks = _content_blocks(completed + started + inline)

    assert state.pending_server_tool_result_ids == set()
    assert aggregate_blocks[0]["type"] == "server_tool_result"
    assert "inline_data" not in aggregate_blocks[0].get("extras", {})
    assert _content_blocks(inline)[0]["type"] == "non_standard"


@pytest.mark.parametrize(
    ("event", "raises_error"),
    [
        ({"event": "response.message.done"}, False),
        ({"event": "response.error"}, True),
        (
            {
                "event": "response.tool.failed",
                "tool_execution": {
                    "call_id": "tool-2",
                    "name": "web_search",
                    "status": "failed",
                },
            },
            False,
        ),
    ],
)
def test_pending_result_clears_on_stream_terminal_or_failure(
    event: dict[str, Any],
    raises_error: bool,
) -> None:
    state = primary.StreamState()
    _convert(
        {
            "event": "response.tool.completed",
            "tool_execution": {
                "call_id": "tool-1",
                "name": "web_search",
                "status": "completed",
            },
        },
        state,
    )
    assert state.pending_server_tool_result_ids == {"tool-1"}

    if raises_error:
        with pytest.raises(PrimaryStreamError, match="response.error"):
            _convert(event, state)
    else:
        _convert(event, state)

    assert state.pending_server_tool_result_ids == set()


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
        "events": ["response.message.done"],
        "finish_reason": "stop",
        "message_id": "message-1",
        "output_version": "v1",
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

    assert first.message.id is not None
    assert first.message.id.startswith("lc_")
    assert final.message.id == repeated.message.id == "late-request"
    assert aggregate.message.id == "late-request"
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


def test_unknown_events_preserve_ordered_raw_provider_payloads() -> None:
    events = [
        {
            "event": "response.future.started",
            "message_id": "message-1",
            "future_field": {"step": 1},
        },
        {
            "event": "response.future.delta",
            "message_id": "message-1",
            "future_field": {"step": 2},
        },
    ]

    state = primary.StreamState()
    chunks = [_convert(event, state) for event in events]
    aggregate = chunks[0] + chunks[1]

    assert [chunk.text for chunk in chunks] == ["", ""]
    assert aggregate.message.response_metadata["events"] == [
        "response.future.started",
        "response.future.delta",
    ]
    assert [
        chunk.message.response_metadata["provider_field_events"] for chunk in chunks
    ] == [
        [{"future_field": {"step": 1}}],
        [{"future_field": {"step": 2}}],
    ]
    assert aggregate.message.response_metadata["raw_events"] == events


def test_tool_in_progress_is_a_known_event() -> None:
    chunk = _convert({"event": "response.tool.in_progress"})

    assert chunk.message.response_metadata["events"] == ["response.tool.in_progress"]
    assert "raw_events" not in chunk.message.response_metadata


def test_tool_failure_is_a_non_terminal_metadata_chunk() -> None:
    chunk = _convert(
        {
            "event": "response.tool.failed",
            "finish_reason": "tool_error",
            "error": {"message": "boom"},
        }
    )

    assert chunk.text == ""
    assert chunk.generation_info is None
    assert _message(chunk).chunk_position is None
    assert chunk.message.response_metadata["finish_reason_events"] == [
        {
            "event": "response.tool.failed",
            "finish_reason": "tool_error",
        }
    ]
    assert chunk.message.response_metadata["provider_field_events"] == [
        {"error": {"message": "boom"}}
    ]


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
    event = gm.PrimaryChatCompletionChunk.model_validate({})
    assert primary.convert_stream_event(event, state=primary.StreamState()) is None


def test_invalid_event_type_has_clear_error() -> None:
    with pytest.raises(TypeError, match="PrimaryChatCompletionChunk"):
        primary.convert_stream_event(42, state=primary.StreamState())  # type: ignore[arg-type]
