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
    assert state.server_tool_call_ids_by_provider == {"provider-tool-state-1": call_id}
    assert state.server_tool_provider_ids == {call_id: "provider-tool-state-1"}
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

    assert _blocks(started)[0]["extras"]["provider_tool_execution"]["censored"] is False
    assert _blocks(update)[0]["extras"] == {
        "provider_tool_execution": {
            "name": "image_generate",
            "status": "in_progress",
            "seconds_left": 2,
        },
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
