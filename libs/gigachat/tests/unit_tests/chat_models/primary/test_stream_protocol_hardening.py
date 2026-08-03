"""Adversarial state-machine tests for the primary stream converter."""

from __future__ import annotations

from functools import reduce
from operator import add
from typing import Any, cast

import gigachat.models as gm
import pytest
from langchain_core.language_models.chat_models import generate_from_stream
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from langchain_gigachat.chat_models._contracts import primary
from langchain_gigachat.chat_models._contracts.primary.types import PrimaryStreamError


def _convert(
    event: dict[str, Any],
    state: primary.StreamState,
) -> ChatGenerationChunk:
    sdk_event = gm.PrimaryChatCompletionChunk.model_validate(event)
    chunk = primary.convert_stream_event(sdk_event, state=state)
    assert chunk is not None
    return chunk


def _message(chunk: ChatGenerationChunk) -> AIMessageChunk:
    assert isinstance(chunk.message, AIMessageChunk)
    return cast(AIMessageChunk, chunk.message)


def _aggregate(events: list[dict[str, Any]]) -> AIMessage:
    state = primary.StreamState()
    chunks = [_convert(event, state) for event in events]
    message = reduce(add, chunks).message
    assert isinstance(message, AIMessageChunk)
    return AIMessage(**message.model_dump(exclude={"type"}))


def test_late_tools_state_finalizes_an_idless_client_tool_call() -> None:
    state = primary.StreamState()
    function = _convert(
        {
            "event": "response.message.delta",
            "messages": [
                {
                    "function_call": {
                        "name": "weather",
                        "arguments": '{"city":"Moscow"}',
                    }
                }
            ],
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

    assert _message(function).tool_call_chunks[0]["id"] is None
    assert _message(done).tool_call_chunks == [
        {
            "name": None,
            "args": "",
            "id": "tools-state-1",
            "index": 0,
            "type": "tool_call_chunk",
        }
    ]
    aggregate = function + done
    assert _message(aggregate).tool_calls == [
        {
            "name": "weather",
            "args": {"city": "Moscow"},
            "id": "tools-state-1",
            "type": "tool_call",
        }
    ]


def test_late_request_id_replaces_provisional_stream_message_id() -> None:
    state = primary.StreamState()
    delta = _convert(
        {
            "event": "response.message.delta",
            "messages": [{"content": [{"text": "answer"}]}],
        },
        state,
    )
    done = _convert(
        {
            "event": "response.message.done",
            "x_headers": {"x-request-id": "request-1"},
            "finish_reason": "stop",
        },
        state,
    )

    assert delta.message.id is None or delta.message.id.startswith("lc_")
    assert done.message.id == "request-1"
    assert (delta + done).message.id == "request-1"


def test_provider_message_id_yields_to_later_request_id() -> None:
    state = primary.StreamState()
    delta = _convert(
        {
            "event": "response.message.delta",
            "message_id": "provider-message-1",
            "messages": [{"content": [{"text": "answer"}]}],
        },
        state,
    )
    done = _convert(
        {
            "event": "response.message.done",
            "x_headers": {"x-request-id": "request-1"},
            "finish_reason": "stop",
        },
        state,
    )

    assert delta.message.id is None or delta.message.id.startswith("lc_")
    assert done.message.id == "request-1"
    assert (delta + done).message.id == "request-1"


def test_provider_state_overrides_unsupported_client_call_id_extension() -> None:
    output = _aggregate(
        [
            {
                "event": "response.message.delta",
                "messages": [
                    {
                        "function_call": {
                            "id": "call-1",
                            "name": "weather",
                            "arguments": '{"city":"Moscow"}',
                        }
                    }
                ],
            },
            {
                "event": "response.message.done",
                "tools_state_id": "tools-state-1",
                "finish_reason": "function_call",
            },
        ]
    )

    assert output.tool_calls[0]["id"] == "tools-state-1"
    assert "provider_tool_state_by_call_id" not in output.additional_kwargs


def test_completed_client_tool_call_without_provider_state_fails() -> None:
    state = primary.StreamState()
    _convert(
        {
            "event": "response.message.delta",
            "messages": [
                {
                    "function_call": {
                        "name": "weather",
                        "arguments": '{"city":"Moscow"}',
                    }
                }
            ],
        },
        state,
    )

    with pytest.raises(
        ValueError,
        match="completed without tools_state_id; the call cannot be replayed",
    ):
        _convert(
            {
                "event": "response.message.done",
                "finish_reason": "function_call",
            },
            state,
        )


def test_only_message_done_is_completion_terminal() -> None:
    state = primary.StreamState()
    tool_failed = _convert(
        {
            "event": "response.tool.failed",
            "finish_reason": "tool_error",
            "tool_execution": {
                "call_id": "tool-1",
                "name": "web_search",
                "status": "failed",
            },
        },
        state,
    )
    done = _convert(
        {
            "event": "response.message.done",
            "finish_reason": "stop",
        },
        state,
    )

    assert _message(tool_failed).chunk_position is None
    assert _message(done).chunk_position == "last"


def test_non_authoritative_finish_reasons_are_ordered_diagnostics() -> None:
    state = primary.StreamState()
    chunks = [
        _convert(
            {
                "event": "response.tool.failed",
                "finish_reason": "tool_error",
            },
            state,
        ),
        _convert(
            {
                "event": "response.message.delta",
                "finish_reason": "provisional",
            },
            state,
        ),
        _convert(
            {
                "event": "response.message.done",
                "finish_reason": "stop",
            },
            state,
        ),
    ]

    aggregate = reduce(add, chunks)

    assert chunks[0].generation_info is None
    assert chunks[1].generation_info is None
    assert aggregate.generation_info == {"finish_reason": "stop"}
    assert aggregate.message.response_metadata["finish_reason"] == "stop"
    assert aggregate.message.response_metadata["finish_reason_events"] == [
        {
            "event": "response.tool.failed",
            "finish_reason": "tool_error",
        },
        {
            "event": "response.message.delta",
            "finish_reason": "provisional",
        },
    ]


def test_tool_completed_reason_does_not_corrupt_final_finish_reason() -> None:
    state = primary.StreamState()
    tool_completed = _convert(
        {
            "event": "response.tool.completed",
            "finish_reason": "tool_complete",
        },
        state,
    )
    done = _convert(
        {
            "event": "response.message.done",
            "finish_reason": "stop",
        },
        state,
    )

    aggregate = tool_completed + done

    assert aggregate.generation_info == {"finish_reason": "stop"}
    assert aggregate.message.response_metadata["finish_reason"] == "stop"
    assert aggregate.message.response_metadata["finish_reason_events"] == [
        {
            "event": "response.tool.completed",
            "finish_reason": "tool_complete",
        }
    ]


def test_repeated_done_reason_is_diagnostic_after_authoritative_done() -> None:
    state = primary.StreamState()
    done = _convert(
        {
            "event": "response.message.done",
            "finish_reason": "stop",
        },
        state,
    )
    continuation = _convert(
        {
            "event": "response.message.done",
            "finish_reason": "stop",
            "future_field": {"trace": "trace-1"},
        },
        state,
    )

    aggregate = done + continuation

    assert continuation.generation_info is None
    assert aggregate.generation_info == {"finish_reason": "stop"}
    assert aggregate.message.response_metadata["finish_reason"] == "stop"
    assert aggregate.message.response_metadata["finish_reason_events"] == [
        {
            "event": "response.message.done",
            "finish_reason": "stop",
        }
    ]


def test_late_first_non_null_done_reason_becomes_authoritative() -> None:
    state = primary.StreamState()
    done_without_reason = _convert(
        {"event": "response.message.done", "message_id": "provider-message-1"},
        state,
    )
    late_reason = _convert(
        {"event": "response.message.done", "finish_reason": "stop"},
        state,
    )

    aggregate = done_without_reason + late_reason
    result = generate_from_stream(iter([done_without_reason, late_reason]))

    assert _message(done_without_reason).chunk_position == "last"
    assert _message(late_reason).chunk_position is None
    assert late_reason.generation_info == {"finish_reason": "stop"}
    assert aggregate.generation_info == {"finish_reason": "stop"}
    assert aggregate.message.response_metadata["finish_reason"] == "stop"
    assert "finish_reason_events" not in aggregate.message.response_metadata
    assert result.generations[0].generation_info == {"finish_reason": "stop"}


def test_done_only_stream_uses_provider_message_id_without_provisional_id() -> None:
    state = primary.StreamState()

    done = _convert(
        {
            "event": "response.message.done",
            "message_id": "provider-message-1",
            "finish_reason": "stop",
        },
        state,
    )

    assert done.message.id == "provider-message-1"
    assert state.message_id == "provider-message-1"


def test_generate_from_stream_keeps_authoritative_finish_reason() -> None:
    state = primary.StreamState()
    chunks = [
        _convert(
            {
                "event": "response.tool.failed",
                "finish_reason": "tool_error",
            },
            state,
        ),
        _convert(
            {
                "event": "response.message.done",
                "finish_reason": "stop",
            },
            state,
        ),
    ]

    result = generate_from_stream(iter(chunks))
    generation = result.generations[0]

    assert generation.generation_info == {"finish_reason": "stop"}
    assert generation.message.response_metadata["finish_reason"] == "stop"


def test_identical_message_done_is_deduplicated() -> None:
    state = primary.StreamState()
    event = {
        "event": "response.message.done",
        "message_id": "message-1",
        "finish_reason": "stop",
    }

    sdk_event = gm.PrimaryChatCompletionChunk.model_validate(event)
    assert primary.convert_stream_event(sdk_event, state=state) is not None
    assert primary.convert_stream_event(sdk_event, state=state) is None


def test_conflicting_message_done_fails() -> None:
    state = primary.StreamState()
    _convert(
        {
            "event": "response.message.done",
            "finish_reason": "stop",
        },
        state,
    )

    with pytest.raises(
        ValueError,
        match="Conflicting primary completion terminal events",
    ):
        _convert(
            {
                "event": "response.message.done",
                "finish_reason": "length",
            },
            state,
        )


def test_content_after_message_done_fails() -> None:
    state = primary.StreamState()
    _convert({"event": "response.message.done", "finish_reason": "stop"}, state)

    with pytest.raises(
        ValueError,
        match="Primary stream content arrived after response.message.done",
    ):
        _convert(
            {
                "event": "response.message.delta",
                "messages": [{"content": [{"text": "too late"}]}],
            },
            state,
        )


def test_metadata_after_message_done_remains_mergeable_but_not_terminal() -> None:
    state = primary.StreamState()
    done = _convert({"event": "response.message.done", "finish_reason": "stop"}, state)
    metadata = _convert(
        {
            "event": "response.metadata",
            "future_field": {"trace": "trace-1"},
        },
        state,
    )

    assert _message(done).chunk_position == "last"
    assert _message(metadata).chunk_position is None
    assert metadata.message.response_metadata["provider_field_events"] == [
        {"future_field": {"trace": "trace-1"}}
    ]


def test_response_error_raises_typed_error_with_raw_payload() -> None:
    state = primary.StreamState()
    event = {
        "event": "response.error",
        "finish_reason": "error",
        "error": {"code": "provider_error", "message": "boom"},
    }

    with pytest.raises(PrimaryStreamError) as raised:
        primary.convert_stream_event(
            gm.PrimaryChatCompletionChunk.model_validate(event),
            state=state,
        )

    assert raised.value.payload == event


def test_distinct_provider_message_ids_fail_with_non_stream_policy() -> None:
    state = primary.StreamState()
    _convert({"event": "response.message.delta", "message_id": "id-1"}, state)

    with pytest.raises(
        ValueError,
        match=(
            "Primary GigaChat completion contains multiple provider message_id "
            "values; their replay semantics are unsupported."
        ),
    ):
        _convert(
            {"event": "response.message.done", "message_id": "id-2"},
            state,
        )


@pytest.mark.parametrize("field", ["message_id", "tools_state_id"])
@pytest.mark.parametrize("invalid_value", ["", "   "])
def test_stream_rejects_invalid_provider_identity(
    field: str,
    invalid_value: object,
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"Primary provider {field} must be a non-empty string",
    ):
        _convert(
            {"event": "response.message.delta", field: invalid_value},
            primary.StreamState(),
        )


def test_stream_request_id_header_lookup_is_case_insensitive() -> None:
    chunk = _convert(
        {
            "event": "response.message.delta",
            "x_headers": {"X-Request-Id": "request-1"},
        },
        primary.StreamState(),
    )

    assert chunk.message.id == "request-1"


def test_stream_rejects_conflicting_request_id_headers() -> None:
    with pytest.raises(ValueError, match="conflicting x-request-id"):
        _convert(
            {
                "event": "response.message.delta",
                "x_headers": {
                    "x-request-id": "request-1",
                    "X-Request-ID": "request-2",
                },
            },
            primary.StreamState(),
        )


@pytest.mark.parametrize(
    "events",
    [
        [
            {
                "event": "response.message.done",
                "tools_state_id": "response-state",
                "messages": [{"tools_state_id": "nested-state"}],
                "finish_reason": "stop",
            }
        ],
        [
            {
                "event": "response.message.delta",
                "tools_state_id": "response-state",
            },
            {
                "event": "response.message.done",
                "tools_state_id": "nested-state",
                "finish_reason": "stop",
            },
        ],
    ],
    ids=["one-event", "separate-events"],
)
def test_multiple_unassigned_tools_state_ids_match_non_stream_policy(
    events: list[dict[str, Any]],
) -> None:
    state = primary.StreamState()
    chunks = [_convert(event, state) for event in events]

    message = generate_from_stream(iter(chunks)).generations[0].message
    assert isinstance(message, AIMessage)
    assert message.additional_kwargs["tools_state_ids"] == [
        "response-state",
        "nested-state",
    ]
    assert "tools_state_id" not in message.additional_kwargs
    assert message.response_metadata["tools_state_ids"] == [
        "response-state",
        "nested-state",
    ]
    assert "tools_state_id" not in message.response_metadata
    assert state.tools_state_id is None
    assert state.unassigned_tools_state_ids == [
        "response-state",
        "nested-state",
    ]


def test_multiple_unassigned_states_cannot_be_claimed_by_idless_client() -> None:
    state = primary.StreamState()
    _convert(
        {"event": "response.message.delta", "tools_state_id": "state-1"},
        state,
    )
    _convert(
        {"event": "response.message.delta", "tools_state_id": "state-2"},
        state,
    )

    with pytest.raises(ValueError, match="multiple unassigned tools_state_id"):
        _convert(
            {
                "event": "response.message.delta",
                "messages": [
                    {
                        "function_call": {
                            "name": "weather",
                            "arguments": '{"city":"Moscow"}',
                        }
                    }
                ],
            },
            state,
        )


def test_single_unassigned_state_can_be_claimed_by_idless_client() -> None:
    state = primary.StreamState()
    metadata = _convert(
        {"event": "response.message.delta", "tools_state_id": "state-1"},
        state,
    )
    function = _convert(
        {
            "event": "response.message.delta",
            "messages": [
                {
                    "function_call": {
                        "name": "weather",
                        "arguments": '{"city":"Moscow"}',
                    }
                }
            ],
        },
        state,
    )
    done = _convert(
        {"event": "response.message.done", "finish_reason": "function_call"},
        state,
    )

    message = (
        generate_from_stream(iter([metadata, function, done])).generations[0].message
    )
    assert isinstance(message, AIMessage)
    assert message.tool_calls[0]["id"] == "state-1"
    assert state.client_tools_state_id == "state-1"
    assert state.unassigned_tools_state_ids == []


@pytest.mark.parametrize("owner", ["client", "server"])
def test_explicit_lifecycle_reclassifies_one_of_multiple_unassigned_states(
    owner: str,
) -> None:
    state = primary.StreamState()
    for state_id in ("state-1", "state-2"):
        _convert(
            {"event": "response.message.delta", "tools_state_id": state_id},
            state,
        )

    event: dict[str, Any]
    if owner == "client":
        event = {
            "event": "response.message.delta",
            "messages": [
                {
                    "tools_state_id": "state-1",
                    "function_call": {
                        "name": "weather",
                        "arguments": '{"city":"Moscow"}',
                    },
                }
            ],
        }
    else:
        event = {
            "event": "response.tool.completed",
            "tools_state_id": "state-1",
            "tool_execution": {
                "name": "web_search",
                "status": "completed",
            },
        }

    chunk = _convert(event, state)

    assert state.unassigned_tools_state_ids == ["state-2"]
    if owner == "client":
        assert state.client_tools_state_id == "state-1"
        assert _message(chunk).tool_call_chunks[0]["id"] == "state-1"
    else:
        assert state.server_tool_call_ids_by_state_id == {
            "state-1": "lc_primary-server-tool-0"
        }
        assert state.server_tool_state_ids_by_call_id == {
            "lc_primary-server-tool-0": "state-1"
        }


def test_single_prior_unassigned_state_is_claimed_by_idless_server_tool() -> None:
    state = primary.StreamState()
    metadata = _convert(
        {"event": "response.message.delta", "tools_state_id": "state-1"},
        state,
    )
    completed = _convert(
        {
            "event": "response.tool.completed",
            "tool_execution": {
                "name": "web_search",
                "status": "completed",
                "output": {"matches": 1},
            },
        },
        state,
    )
    done = _convert(
        {"event": "response.message.done", "finish_reason": "stop"},
        state,
    )

    message = (
        generate_from_stream(iter([metadata, completed, done])).generations[0].message
    )
    assert isinstance(message, AIMessage)
    assert state.unassigned_tools_state_ids == []
    assert state.unresolved_server_tool_call_ids == []
    assert state.server_tool_state_ids_by_call_id == {
        "lc_primary-server-tool-0": "state-1"
    }
    assert message.additional_kwargs["provider_server_tool_state_by_call_id"] == {
        "lc_primary-server-tool-0": "state-1"
    }


def test_idless_server_tool_cannot_claim_multiple_prior_unassigned_states() -> None:
    state = primary.StreamState()
    for state_id in ("state-1", "state-2"):
        _convert(
            {"event": "response.message.delta", "tools_state_id": state_id},
            state,
        )

    with pytest.raises(ValueError, match="multiple unassigned tools_state_id"):
        _convert(
            {
                "event": "response.tool.completed",
                "tool_execution": {
                    "name": "web_search",
                    "status": "completed",
                },
            },
            state,
        )


def test_multiple_idless_server_tools_cannot_claim_one_prior_unassigned_state() -> None:
    state = primary.StreamState()
    _convert(
        {"event": "response.message.delta", "tools_state_id": "state-1"},
        state,
    )

    with pytest.raises(ValueError, match="multiple idless server tool lifecycles"):
        _convert(
            {
                "event": "response.tool.completed",
                "messages": [
                    {
                        "role": "reasoning",
                        "tool_execution": {
                            "name": "web_search",
                            "status": "completed",
                        },
                    },
                    {
                        "role": "reasoning",
                        "tool_execution": {
                            "name": "image_generate",
                            "status": "completed",
                        },
                    },
                ],
            },
            state,
        )


@pytest.mark.parametrize("field", ["message_id", "tools_state_id"])
def test_new_identity_metadata_after_message_done_fails_closed(field: str) -> None:
    state = primary.StreamState()
    _convert({"event": "response.message.done", "finish_reason": "stop"}, state)

    with pytest.raises(
        ValueError,
        match=rf"Primary stream received a new {field} after response.message.done",
    ):
        _convert({"event": "response.metadata", field: "late-id"}, state)


@pytest.mark.parametrize("field", ["message_id", "tools_state_id"])
def test_repeated_identity_metadata_after_message_done_is_compatible(
    field: str,
) -> None:
    state = primary.StreamState()
    done = _convert(
        {
            "event": "response.message.done",
            field: "stable-id",
            "finish_reason": "stop",
        },
        state,
    )
    continuation = _convert(
        {"event": "response.metadata", field: "stable-id"},
        state,
    )

    assert _message(done).chunk_position == "last"
    assert _message(continuation).chunk_position is None


def test_repeated_usage_snapshot_is_emitted_once() -> None:
    state = primary.StreamState()
    usage = {
        "input_tokens": 2,
        "output_tokens": 1,
        "total_tokens": 3,
    }
    first = _convert({"event": "response.message.delta", "usage": usage}, state)
    done = _convert(
        {
            "event": "response.message.done",
            "usage": usage,
            "finish_reason": "stop",
        },
        state,
    )

    assert _message(first).usage_metadata == usage
    assert _message(done).usage_metadata is None
    assert _message(first + done).usage_metadata == usage


def test_changed_usage_snapshot_fails_without_incremental_semantics() -> None:
    state = primary.StreamState()
    _convert(
        {
            "event": "response.message.delta",
            "usage": {"input_tokens": 2, "output_tokens": 1},
        },
        state,
    )

    with pytest.raises(
        ValueError,
        match="incremental usage semantics are unsupported",
    ):
        _convert(
            {
                "event": "response.message.done",
                "usage": {"input_tokens": 2, "output_tokens": 2},
            },
            state,
        )


def test_event_diagnostics_aggregate_as_ordered_lists() -> None:
    state = primary.StreamState()
    chunks = [
        _convert(
            {
                "event": f"response.future.{step}",
                "tool_execution": {
                    "call_id": "provider-tool-1",
                    "status": step,
                },
                "future_field": {"step": step},
            },
            state,
        )
        for step in ("started", "completed")
    ]

    metadata = (chunks[0] + chunks[1]).message.response_metadata
    assert metadata["tool_execution_events"] == [
        {"call_id": "provider-tool-1", "status": "started"},
        {"call_id": "provider-tool-1", "status": "completed"},
    ]
    assert metadata["provider_field_events"] == [
        {"future_field": {"step": "started"}},
        {"future_field": {"step": "completed"}},
    ]
