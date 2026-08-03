"""Late provider identity contracts for primary server-tool streams."""

from __future__ import annotations

from functools import reduce
from operator import add
from typing import Any, cast

import gigachat.models as gm
import pytest
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from langchain_gigachat.chat_models._contracts import primary

from .fixtures import build_official_sdk_server_tool_stream


def _convert(
    event: gm.PrimaryChatCompletionChunk | dict[str, Any],
    state: primary.StreamState,
) -> ChatGenerationChunk:
    if isinstance(event, dict):
        event = gm.PrimaryChatCompletionChunk.model_validate(event)
    chunk = primary.convert_stream_event(event, state=state)
    assert chunk is not None
    return chunk


def _blocks(chunk: ChatGenerationChunk) -> list[dict[str, Any]]:
    assert isinstance(chunk.message, AIMessageChunk)
    content = chunk.message.content
    assert isinstance(content, list)
    return cast(list[dict[str, Any]], content)


def _started_event(**event_fields: Any) -> dict[str, Any]:
    return {
        "event": "response.tool.started",
        "tool_execution": {
            "name": "web_search",
            "status": "running",
        },
        **event_fields,
    }


def _completed_event(**event_fields: Any) -> dict[str, Any]:
    return {
        "event": "response.tool.completed",
        "tool_execution": {
            "name": "web_search",
            "status": "completed",
            "output": {"matches": 1},
        },
        **event_fields,
    }


def _nested_tool_event(
    event: str,
    name: str,
    **execution_fields: Any,
) -> dict[str, Any]:
    return {
        "event": event,
        "messages": [
            {
                "role": "assistant",
                "content": [{"tool_execution": {"name": name, **execution_fields}}],
            }
        ],
    }


def test_late_tools_state_reconciles_provisional_server_tool_identity() -> None:
    state = primary.StreamState()
    started = _convert(_started_event(), state)
    completed = _convert(
        _completed_event(tools_state_id="provider-tool-state-1"),
        state,
    )

    started_block = _blocks(started)[0]
    result_block = _blocks(completed)[0]
    call_id = started_block["id"]

    assert call_id == "lc_primary-server-tool-0"
    assert result_block["tool_call_id"] == call_id
    assert result_block["extras"]["provider_server_tool_state_by_call_id"] == {
        call_id: "provider-tool-state-1"
    }
    assert state.server_tool_call_ids_by_state_id == {"provider-tool-state-1": call_id}
    assert state.server_tool_state_ids_by_call_id == {call_id: "provider-tool-state-1"}
    assert state.active_server_tool_call_id is None


def test_late_request_id_does_not_replace_provisional_server_tool_id() -> None:
    state = primary.StreamState()
    started = _convert(
        _started_event(message_id="provider-message-1"),
        state,
    )
    completed = _convert(
        _completed_event(
            tools_state_id="provider-tool-state-1",
            x_headers={"x-request-id": "request-1"},
        ),
        state,
    )

    call_id = _blocks(started)[0]["id"]

    assert call_id == "lc_primary-server-tool-0"
    assert call_id not in {"provider-message-1", "request-1"}
    assert _blocks(completed)[0]["tool_call_id"] == call_id


def test_late_name_and_state_reconcile_in_the_same_event() -> None:
    state = primary.StreamState()
    started = _convert(
        {
            "event": "response.tool.started",
            "tool_execution": {"status": "running"},
        },
        state,
    )
    completed = _convert(
        _completed_event(tools_state_id="provider-tool-state-1"),
        state,
    )

    call_id = _blocks(started)[0]["id"]
    aggregate = started + completed

    assert _blocks(started)[0]["name"] == ""
    assert state.server_tool_names == {call_id: "web_search"}
    assert _blocks(aggregate)[0]["name"] == ""
    assert _blocks(aggregate)[1]["tool_call_id"] == call_id


def test_inline_data_after_late_identity_updates_the_same_result() -> None:
    state = primary.StreamState()
    started = _convert(_started_event(), state)
    completed = _convert(
        _completed_event(tools_state_id="provider-tool-state-1"),
        state,
    )
    inline_data = _convert(
        {
            "event": "response.message.delta",
            "messages": [
                {
                    "content": [
                        {
                            "inline_data": {
                                "sources": {
                                    "source-1": {
                                        "url": "https://example.test",
                                    }
                                }
                            }
                        }
                    ]
                }
            ],
        },
        state,
    )

    aggregate = reduce(add, [started, completed, inline_data])
    result_block = _blocks(aggregate)[1]

    assert result_block["tool_call_id"] == _blocks(started)[0]["id"]
    assert result_block["extras"]["inline_data"]["sources"] == {
        "source-1": {"url": "https://example.test"}
    }


def test_two_sequential_server_tools_keep_distinct_identities() -> None:
    state = primary.StreamState()
    chunks = [
        _convert(
            {
                "event": "response.tool.started",
                "tool_execution": {
                    "call_id": "provider-tool-1",
                    "name": "web_search",
                    "status": "running",
                },
            },
            state,
        ),
        _convert(
            {
                "event": "response.tool.completed",
                "tool_execution": {
                    "call_id": "provider-tool-1",
                    "name": "web_search",
                    "status": "completed",
                },
            },
            state,
        ),
        _convert(
            {
                "event": "response.tool.started",
                "tool_execution": {
                    "call_id": "provider-tool-2",
                    "name": "code_interpreter",
                    "status": "running",
                },
            },
            state,
        ),
        _convert(
            {
                "event": "response.tool.failed",
                "tool_execution": {
                    "call_id": "provider-tool-2",
                    "name": "code_interpreter",
                    "status": "failed",
                },
            },
            state,
        ),
    ]

    blocks = _blocks(reduce(add, chunks))

    assert blocks[0]["id"] == "provider-tool-1"
    assert blocks[1]["tool_call_id"] == "provider-tool-1"
    assert blocks[2]["id"] == "provider-tool-2"
    assert blocks[3]["tool_call_id"] == "provider-tool-2"


def test_execution_and_state_ids_use_independent_namespaces() -> None:
    state = primary.StreamState()
    shared_provider_value = "shared-provider-id"
    explicit = _convert(
        {
            "event": "response.tool.completed",
            "tool_execution": {
                "call_id": shared_provider_value,
                "name": "web_search",
                "status": "completed",
            },
        },
        state,
    )
    state_only = _convert(
        {
            "event": "response.tool.completed",
            "tools_state_id": shared_provider_value,
            "tool_execution": {
                "name": "image_generate",
                "status": "completed",
            },
        },
        state,
    )
    done = _convert(
        {
            "event": "response.message.done",
            "tools_state_id": shared_provider_value,
            "finish_reason": "stop",
        },
        state,
    )

    blocks = _blocks(explicit + state_only + done)
    assert [block["tool_call_id"] for block in blocks] == [
        shared_provider_value,
        "lc_primary-server-tool-0",
    ]
    assert done.message.additional_kwargs["provider_server_tool_state_by_call_id"] == {
        "lc_primary-server-tool-0": shared_provider_value
    }


def test_official_sdk_terminal_server_tool_without_identity_uses_local_id() -> None:
    state = primary.StreamState()
    events = build_official_sdk_server_tool_stream()
    snapshots = [event.model_dump(mode="json") for event in events]
    chunks = [_convert(event, state) for event in events]

    aggregate = reduce(add, chunks)
    assert isinstance(aggregate.message, AIMessageChunk)
    message = aggregate.message
    blocks = _blocks(aggregate)
    result = next(block for block in blocks if block["type"] == "server_tool_result")
    call_id = result["tool_call_id"]

    assert call_id == "lc_primary-server-tool-0"
    assert result["extras"]["provider_tool_execution"] == {
        "name": "image_generate",
        "status": "success",
        "censored": True,
    }
    assert message.additional_kwargs["provider_server_tool_state_by_call_id"] == {
        call_id: "tools-state-1"
    }
    assert message.response_metadata["tools_state_id"] == "tools-state-1"
    assert message.response_metadata["finish_reason"] == "error"
    assert aggregate.generation_info == {"finish_reason": "error"}
    assert message.usage_metadata == {
        "input_tokens": 1,
        "output_tokens": 2,
        "total_tokens": 3,
        "input_token_details": {"cache_read": 0},
    }
    assert (
        sum(
            isinstance(chunk.message, AIMessageChunk)
            and chunk.message.chunk_position == "last"
            for chunk in chunks
        )
        == 1
    )
    assert [event.model_dump(mode="json") for event in events] == snapshots


def test_repeated_server_tool_update_merges_all_extras() -> None:
    state = primary.StreamState()
    started = _convert(
        {
            "event": "response.tool.started",
            "tool_execution": {
                "name": "image_generate",
                "status": "running",
                "censored": False,
            },
        },
        state,
    )
    update = _convert(
        {
            "event": "response.tool.in_progress",
            "tools_state_id": "provider-state-1",
            "tool_execution": {
                "name": "image_generate",
                "status": "in_progress",
                "seconds_left": 2,
            },
        },
        state,
    )
    repeated_update = _convert(
        {
            "event": "response.tool.in_progress",
            "tools_state_id": "provider-state-1",
            "tool_execution": {
                "name": "image_generate",
                "status": "in_progress",
                "seconds_left": 1,
            },
        },
        state,
    )

    assert _blocks(started)[0]["extras"]["provider_tool_execution"]["censored"] is False
    assert _blocks(update)[0]["extras"] == {
        "provider_server_tool_state_by_call_id": {
            "lc_primary-server-tool-0": "provider-state-1"
        },
        "provider_tool_execution_updates": [
            {
                "name": "image_generate",
                "status": "in_progress",
                "seconds_left": 2,
            }
        ],
    }
    assert _blocks(repeated_update)[0]["extras"] == {
        "provider_tool_execution_updates": [
            {
                "name": "image_generate",
                "status": "in_progress",
                "seconds_left": 1,
            }
        ]
    }
    aggregate_extras = _blocks(started + update + repeated_update)[0]["extras"]
    assert aggregate_extras["provider_tool_execution"] == {
        "name": "image_generate",
        "status": "running",
        "censored": False,
    }
    assert aggregate_extras["provider_tool_execution_updates"] == [
        {
            "name": "image_generate",
            "status": "in_progress",
            "seconds_left": 2,
        },
        {
            "name": "image_generate",
            "status": "in_progress",
            "seconds_left": 1,
        },
    ]
    assert aggregate_extras["provider_server_tool_state_by_call_id"] == {
        "lc_primary-server-tool-0": "provider-state-1"
    }


def test_progress_snapshots_do_not_merge_by_provider_index() -> None:
    state = primary.StreamState()
    events = [
        {
            "event": "response.tool.in_progress",
            "tool_execution": {
                "call_id": "call-1",
                "name": "image_generate",
                "index": 0,
            },
        },
        {
            "event": "response.tool.in_progress",
            "tool_execution": {
                "call_id": "call-1",
                "name": "image_generate",
                "index": 0,
                "seconds_left": 2,
            },
        },
        {
            "event": "response.tool.in_progress",
            "tool_execution": {
                "call_id": "call-1",
                "name": "image_generate",
                "index": 0,
                "seconds_left": 1,
            },
        },
    ]
    aggregate = reduce(add, (_convert(event, state) for event in events))
    call = _blocks(aggregate)[0]

    assert call["index"] == 0
    assert call["extras"]["provider_tool_execution"] == {
        "call_id": "call-1",
        "name": "image_generate",
        "index": 0,
    }
    assert call["extras"]["provider_tool_execution_updates"] == [
        {
            "call_id": "call-1",
            "name": "image_generate",
            "seconds_left": 2,
        },
        {
            "call_id": "call-1",
            "name": "image_generate",
            "seconds_left": 1,
        },
    ]


def test_repeated_structured_server_arguments_emit_one_snapshot() -> None:
    state = primary.StreamState()
    events = [
        {
            "event": "response.tool.in_progress",
            "tool_execution": {
                "name": "web_search",
                "arguments": {"query": "x", "limit": 1},
            },
        },
        {
            "event": "response.tool.in_progress",
            "tool_execution": {
                "name": "web_search",
                "arguments": {"limit": 1, "query": "x"},
            },
        },
    ]
    aggregate = reduce(add, (_convert(event, state) for event in events))
    call = _blocks(aggregate)[0]

    assert call["args"] == '{"query":"x","limit":1}'
    assert call["extras"]["provider_tool_execution_updates"] == [
        {"name": "web_search", "arguments": {"limit": 1, "query": "x"}}
    ]


@pytest.mark.parametrize(
    ("first", "second", "error"),
    [
        (
            {"query": "first"},
            {"query": "second"},
            "conflicting structured argument snapshots",
        ),
        ('{"query":', {"query": "second"}, "mixes argument fragments"),
        ({"query": "first"}, '"second"}', "mixes argument fragments"),
    ],
)
def test_incompatible_server_argument_updates_fail_closed(
    first: Any,
    second: Any,
    error: str,
) -> None:
    state = primary.StreamState()
    _convert(
        {
            "event": "response.tool.in_progress",
            "tool_execution": {
                "name": "web_search",
                "arguments": first,
            },
        },
        state,
    )

    with pytest.raises(ValueError, match=error):
        _convert(
            {
                "event": "response.tool.in_progress",
                "tool_execution": {
                    "name": "web_search",
                    "arguments": second,
                },
            },
            state,
        )


def test_real_image_tool_stream_preserves_progress_file_and_late_state() -> None:
    events: list[dict[str, Any]] = [
        _nested_tool_event("response.tool.in_progress", "image_generate"),
        _nested_tool_event(
            "response.tool.in_progress",
            "image_generate",
            seconds_left=17,
        ),
        _nested_tool_event(
            "response.tool.in_progress",
            "image_generate",
            seconds_left=8,
        ),
        _nested_tool_event(
            "response.tool.completed",
            "image_generate",
            status="success",
        ),
        {
            "event": "response.message.delta",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "files": [
                                {
                                    "id": "image-1",
                                    "mime": "image/jpeg",
                                    "target": "image",
                                }
                            ]
                        }
                    ],
                }
            ],
        },
        {
            "event": "response.message.delta",
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"text": "вот такая красивая картинка"}],
                }
            ],
        },
        {
            "event": "response.message.done",
            "messages": [{"role": "assistant", "tool_state_id": "provider-state-1"}],
            "finish_reason": "stop",
            "usage": {
                "input_tokens": 913,
                "input_tokens_details": {
                    "prompt_tokens": 913,
                    "cached_tokens": 909,
                },
                "output_tokens": 42,
                "total_tokens": 955,
            },
        },
    ]
    state = primary.StreamState()
    aggregate = reduce(add, (_convert(event, state) for event in events))
    assert isinstance(aggregate.message, AIMessageChunk)
    blocks = _blocks(aggregate)

    call = blocks[0]
    assert call["type"] == "server_tool_call_chunk"
    assert call["name"] == "image_generate"
    assert call["extras"]["provider_tool_execution"] == {"name": "image_generate"}
    assert call["extras"]["provider_tool_execution_updates"] == [
        {"name": "image_generate", "seconds_left": 17},
        {"name": "image_generate", "seconds_left": 8},
    ]
    assert blocks[1]["type"] == "server_tool_result"
    assert blocks[1]["status"] == "success"
    assert blocks[2] == {
        "type": "image",
        "file_id": "image-1",
        "mime_type": "image/jpeg",
        "index": 2,
        "extras": {"target": "image"},
    }
    assert blocks[3]["text"] == "вот такая красивая картинка"
    assert all(block["type"] != "non_standard" for block in blocks)

    call_id = call["id"]
    assert state.server_tool_state_ids_by_call_id == {call_id: "provider-state-1"}
    assert state.unresolved_server_tool_call_ids == []
    assert aggregate.message.additional_kwargs["tools_state_id"] == "provider-state-1"
    assert aggregate.message.additional_kwargs[
        "provider_server_tool_state_by_call_id"
    ] == {call_id: "provider-state-1"}
    assert aggregate.message.usage_metadata == {
        "input_tokens": 913,
        "output_tokens": 42,
        "total_tokens": 955,
        "input_token_details": {"cache_read": 909},
    }


@pytest.mark.parametrize("as_sdk_model", [False, True])
@pytest.mark.parametrize("tool_name", ["code_interpreter", "web_search"])
def test_real_idless_tool_stream_accepts_singular_late_state(
    tool_name: str,
    as_sdk_model: bool,
) -> None:
    state = primary.StreamState()
    events: list[dict[str, Any]] = [
        _nested_tool_event("response.tool.in_progress", tool_name),
        _nested_tool_event(
            "response.tool.completed",
            tool_name,
            status="success",
        ),
        {
            "event": "response.message.done",
            "messages": [{"role": "assistant", "tool_state_id": "provider-state-1"}],
            "finish_reason": "stop",
        },
    ]
    if tool_name == "web_search":
        events.insert(
            -1,
            {
                "event": "response.message.delta",
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "inline_data": {
                                    "sources": {
                                        "4": {
                                            "url": "https://example.test/news",
                                            "title": "News",
                                        }
                                    },
                                    "images": [],
                                }
                            }
                        ],
                    }
                ],
            },
        )
    aggregate = reduce(
        add,
        (
            _convert(
                gm.PrimaryChatCompletionChunk.model_validate(event)
                if as_sdk_model
                else event,
                state,
            )
            for event in events
        ),
    )
    blocks = _blocks(aggregate)
    call_id = blocks[0]["id"]

    assert [block["type"] for block in blocks] == [
        "server_tool_call_chunk",
        "server_tool_result",
    ]
    assert state.server_tool_state_ids_by_call_id == {call_id: "provider-state-1"}
    assert state.unresolved_server_tool_call_ids == []
    assert aggregate.message.response_metadata["tools_state_id"] == "provider-state-1"
    if tool_name == "web_search":
        assert blocks[1]["extras"]["inline_data"] == {
            "images": [],
            "sources": {
                "4": {
                    "url": "https://example.test/news",
                    "title": "News",
                }
            },
        }


def test_equivalent_nested_state_alias_terminal_is_deduplicated() -> None:
    state = primary.StreamState()
    first = primary.convert_stream_event(
        gm.PrimaryChatCompletionChunk.model_validate(
            {
                "event": "response.message.done",
                "messages": [{"role": "assistant", "tool_state_id": "state-1"}],
                "finish_reason": "stop",
            }
        ),
        state=state,
    )
    repeated = primary.convert_stream_event(
        gm.PrimaryChatCompletionChunk.model_validate(
            {
                "event": "response.message.done",
                "messages": [{"role": "assistant", "functions_state_id": "state-1"}],
                "finish_reason": "stop",
            }
        ),
        state=state,
    )

    assert first is not None
    assert repeated is None


def test_state_only_event_binds_the_active_idless_tool() -> None:
    state = primary.StreamState()
    events = [
        _nested_tool_event("response.tool.in_progress", "image_generate"),
        {"event": "response.metadata", "tools_state_id": "state-1"},
        {
            "event": "response.tool.completed",
            "tool_execution": {"status": "success"},
        },
        {
            "event": "response.message.done",
            "tools_state_id": "state-1",
            "finish_reason": "stop",
        },
    ]
    aggregate = reduce(add, (_convert(event, state) for event in events))
    call_id = _blocks(aggregate)[0]["id"]

    assert state.server_tool_state_ids_by_call_id == {call_id: "state-1"}
    assert state.unresolved_server_tool_call_ids == []
    assert aggregate.message.additional_kwargs[
        "provider_server_tool_state_by_call_id"
    ] == {call_id: "state-1"}


def test_multiple_late_states_cannot_claim_one_unresolved_tool() -> None:
    state = primary.StreamState()
    _convert(
        {
            "event": "response.tool.completed",
            "tool_execution": {"name": "web_search", "status": "success"},
        },
        state,
    )

    with pytest.raises(ValueError, match="multiple state-only tools_state_id"):
        _convert(
            {
                "event": "response.message.done",
                "messages": [
                    {"role": "assistant", "tool_state_id": "state-1"},
                    {"role": "assistant", "functions_state_id": "state-2"},
                ],
                "finish_reason": "stop",
            },
            state,
        )


def test_explicit_execution_cannot_switch_state_across_events() -> None:
    state = primary.StreamState()
    _convert(
        {
            "event": "response.tool.completed",
            "functions_state_id": "state-1",
            "tool_execution": {
                "call_id": "same-tool",
                "name": "web_search",
                "status": "completed",
            },
        },
        state,
    )

    with pytest.raises(
        ValueError,
        match=r"same provider identity.*multiple tools_state_id",
    ):
        _convert(
            {
                "event": "response.message.done",
                "messages": [
                    {
                        "role": "assistant",
                        "tool_state_id": "state-2",
                        "tool_execution": {
                            "call_id": "same-tool",
                            "name": "web_search",
                            "status": "done",
                        },
                    }
                ],
                "finish_reason": "stop",
            },
            state,
        )


def test_late_explicit_identity_cannot_switch_an_idless_lifecycle_state() -> None:
    state = primary.StreamState()
    _convert(
        {
            "event": "response.tool.in_progress",
            "tools_state_id": "state-1",
            "tool_execution": {"name": "web_search", "status": "running"},
        },
        state,
    )

    with pytest.raises(ValueError, match="conflicting tools_state_id values"):
        _convert(
            {
                "event": "response.tool.completed",
                "tools_state_id": "state-2",
                "tool_execution": {
                    "call_id": "provider-call",
                    "name": "web_search",
                    "status": "success",
                },
            },
            state,
        )


def test_late_idless_mirror_cannot_switch_an_explicit_lifecycle_state() -> None:
    state = primary.StreamState()
    _convert(
        {
            "event": "response.tool.in_progress",
            "tools_state_id": "state-1",
            "tool_execution": {
                "call_id": "provider-call",
                "name": "web_search",
                "status": "running",
            },
        },
        state,
    )

    with pytest.raises(ValueError, match="conflicting tools_state_id values"):
        _convert(
            {
                "event": "response.tool.completed",
                "tools_state_id": "state-2",
                "tool_execution": {
                    "name": "web_search",
                    "status": "success",
                },
            },
            state,
        )


def test_late_explicit_identity_binds_state_to_the_existing_idless_call() -> None:
    state = primary.StreamState()
    started = _convert(
        {
            "event": "response.tool.in_progress",
            "tool_execution": {"name": "web_search", "status": "running"},
        },
        state,
    )
    completed = _convert(
        {
            "event": "response.tool.completed",
            "tools_state_id": "state-1",
            "tool_execution": {
                "call_id": "provider-call",
                "name": "web_search",
                "status": "success",
            },
        },
        state,
    )
    done = _convert(
        {
            "event": "response.message.done",
            "tools_state_id": "state-1",
            "finish_reason": "stop",
        },
        state,
    )

    call_id = _blocks(started)[0]["id"]
    assert _blocks(completed)[0]["tool_call_id"] == call_id
    assert state.server_tool_call_ids_by_execution_id == {"provider-call": call_id}
    assert state.server_tool_state_ids_by_call_id == {call_id: "state-1"}
    assert done.message.additional_kwargs["provider_server_tool_state_by_call_id"] == {
        call_id: "state-1"
    }


def test_client_owned_container_state_is_not_tracked_as_server_identity() -> None:
    state = primary.StreamState()
    _convert(
        {
            "event": "response.message.delta",
            "tools_state_id": "client-state",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "function_call": {
                                "name": "lookup",
                                "arguments": "{}",
                            }
                        },
                        {
                            "tool_execution": {
                                "call_id": "server-call",
                                "name": "web_search",
                                "status": "running",
                            }
                        },
                    ],
                }
            ],
        },
        state,
    )
    _convert(
        {
            "event": "response.tool.in_progress",
            "tools_state_id": "server-state",
            "tool_execution": {
                "call_id": "server-call",
                "name": "web_search",
                "status": "running",
            },
        },
        state,
    )

    assert state.client_tools_state_id == "client-state"
    assert state.server_tool_observed_state_ids_by_execution_id == {
        "server-call": "server-state"
    }
    assert state.server_tool_state_ids_by_call_id == {}


def test_idless_terminal_repeat_reconciles_the_explicit_container_state() -> None:
    state = primary.StreamState()
    _convert(
        {
            "event": "response.tool.completed",
            "tools_state_id": "state-1",
            "tool_execution": {
                "call_id": "provider-call",
                "name": "web_search",
                "status": "success",
            },
        },
        state,
    )
    done = _convert(
        {
            "event": "response.message.done",
            "tools_state_id": "state-1",
            "messages": [
                {
                    "role": "assistant",
                    "tool_execution": {
                        "name": "web_search",
                        "status": "success",
                    },
                }
            ],
            "finish_reason": "stop",
        },
        state,
    )

    assert state.server_tool_state_ids_by_call_id == {"provider-call": "state-1"}
    assert done.message.additional_kwargs["provider_server_tool_state_by_call_id"] == {
        "provider-call": "state-1"
    }


def test_idless_terminal_repeat_rejects_a_different_container_state() -> None:
    state = primary.StreamState()
    _convert(
        {
            "event": "response.tool.completed",
            "tools_state_id": "state-1",
            "tool_execution": {
                "call_id": "provider-call",
                "name": "web_search",
                "status": "success",
            },
        },
        state,
    )

    with pytest.raises(ValueError, match="conflicting tools_state_id values"):
        _convert(
            {
                "event": "response.message.done",
                "tools_state_id": "state-2",
                "messages": [
                    {
                        "role": "assistant",
                        "tool_execution": {
                            "name": "web_search",
                            "status": "success",
                        },
                    }
                ],
                "finish_reason": "stop",
            },
            state,
        )


def test_state_only_context_does_not_bind_an_explicit_only_lifecycle() -> None:
    state = primary.StreamState()
    _convert(
        {
            "event": "response.tool.in_progress",
            "tools_state_id": "state-1",
            "tool_execution": {
                "call_id": "provider-call",
                "name": "web_search",
                "status": "running",
            },
        },
        state,
    )
    _convert(
        {
            "event": "response.metadata",
            "tools_state_id": "state-1",
        },
        state,
    )

    assert state.server_tool_state_ids_by_call_id == {}
    assert state.server_tool_call_ids_by_state_id == {}


def test_state_only_context_is_remembered_for_explicit_state_consistency() -> None:
    state = primary.StreamState()
    _convert(
        {
            "event": "response.tool.in_progress",
            "tool_execution": {
                "call_id": "provider-call",
                "name": "web_search",
                "status": "running",
            },
        },
        state,
    )
    _convert(
        {
            "event": "response.metadata",
            "tools_state_id": "state-1",
        },
        state,
    )

    assert state.server_tool_observed_state_ids_by_execution_id == {
        "provider-call": "state-1"
    }
    assert state.server_tool_state_ids_by_call_id == {}
    assert state.server_tool_call_ids_by_state_id == {}

    with pytest.raises(ValueError, match="multiple tools_state_id values"):
        _convert(
            {
                "event": "response.tool.in_progress",
                "tools_state_id": "state-2",
                "tool_execution": {
                    "call_id": "provider-call",
                    "name": "web_search",
                    "status": "running",
                },
            },
            state,
        )


def test_state_only_context_cannot_switch_an_explicit_container_state() -> None:
    state = primary.StreamState()
    _convert(
        {
            "event": "response.tool.in_progress",
            "tools_state_id": "state-1",
            "tool_execution": {
                "call_id": "provider-call",
                "name": "web_search",
                "status": "running",
            },
        },
        state,
    )

    with pytest.raises(ValueError, match="conflicting tools_state_id values"):
        _convert(
            {
                "event": "response.metadata",
                "tools_state_id": "state-2",
            },
            state,
        )


def test_sequential_server_tools_do_not_reuse_stale_global_state() -> None:
    state = primary.StreamState()
    chunks = [
        _convert(
            {
                "event": "response.tool.started",
                "tools_state_id": "provider-state-1",
                "tool_execution": {"name": "web_search", "status": "running"},
            },
            state,
        ),
        _convert(
            {
                "event": "response.tool.completed",
                "tool_execution": {"name": "web_search", "status": "success"},
            },
            state,
        ),
        _convert(
            {
                "event": "response.tool.started",
                "tool_execution": {
                    "name": "image_generate",
                    "status": "running",
                },
            },
            state,
        ),
        _convert(
            {
                "event": "response.tool.completed",
                "tools_state_id": "provider-state-2",
                "tool_execution": {
                    "name": "image_generate",
                    "status": "success",
                },
            },
            state,
        ),
        _convert(
            {
                "event": "response.message.done",
                "tools_state_id": "provider-state-2",
                "finish_reason": "stop",
            },
            state,
        ),
    ]

    aggregate = reduce(add, chunks)
    blocks = _blocks(aggregate)
    first_call_id = blocks[0]["id"]
    second_call_id = blocks[2]["id"]

    assert first_call_id == "lc_primary-server-tool-0"
    assert blocks[1]["tool_call_id"] == first_call_id
    assert second_call_id == "lc_primary-server-tool-1"
    assert blocks[3]["tool_call_id"] == second_call_id
    assert second_call_id != first_call_id
    assert aggregate.message.additional_kwargs[
        "provider_server_tool_state_by_call_id"
    ] == {
        first_call_id: "provider-state-1",
        second_call_id: "provider-state-2",
    }
    assert state.provider_tools_state_ids == ["provider-state-1", "provider-state-2"]


def test_late_state_with_two_unresolved_server_tools_fails_ambiguously() -> None:
    state = primary.StreamState()
    for name in ("web_search", "image_generate"):
        _convert(
            {
                "event": "response.tool.completed",
                "tool_execution": {"name": name, "status": "success"},
            },
            state,
        )

    with pytest.raises(
        ValueError,
        match="multiple unresolved tool lifecycles; correlation is ambiguous",
    ):
        _convert(
            {
                "event": "response.message.done",
                "tools_state_id": "provider-state-ambiguous",
            },
            state,
        )
