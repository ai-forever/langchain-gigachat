"""Cross-event mirrors update one server-tool lifecycle instead of duplicating it."""

from __future__ import annotations

from collections.abc import AsyncIterator
from functools import reduce
from operator import add
from typing import Any
from unittest.mock import MagicMock

import gigachat.models as gm
import pytest
from langchain_core.language_models.chat_models import generate_from_stream
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from langchain_gigachat.chat_models._contracts import primary
from langchain_gigachat.chat_models.gigachat import GigaChat


def _execution(
    *,
    call_id: str | None = None,
    name: str = "web_search",
    output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    execution: dict[str, Any] = {
        "name": name,
        "status": "success",
        "output": output or {"matches": 1},
    }
    if call_id is not None:
        execution["call_id"] = call_id
    return execution


def _mirrored_events(*, nested_repeat: bool = False) -> list[dict[str, Any]]:
    execution = _execution()
    repeated: dict[str, Any]
    if nested_repeat:
        repeated = {
            "messages": [
                {
                    "role": "reasoning",
                    "content": [{"tool_execution": execution}],
                }
            ]
        }
    else:
        repeated = {"tool_execution": execution}
    return [
        {
            "event": "response.tool.completed",
            "tool_execution": execution,
        },
        {
            "event": "response.message.done",
            "tools_state_id": "provider-state-1",
            "finish_reason": "stop",
            **repeated,
        },
    ]


def _sdk_event(event: dict[str, Any]) -> gm.PrimaryChatCompletionChunk:
    return gm.PrimaryChatCompletionChunk.model_validate(event)


def _message(events: list[dict[str, Any]]) -> AIMessage:
    state = primary.StreamState()
    chunks: list[ChatGenerationChunk] = []
    for event in events:
        chunk = primary.convert_stream_event(
            gm.PrimaryChatCompletionChunk.model_validate(event),
            state=state,
        )
        assert chunk is not None
        chunks.append(chunk)
    message = generate_from_stream(iter(chunks)).generations[0].message
    assert isinstance(message, AIMessage)
    return message


def _assert_one_correlated_result(message: AIMessage | AIMessageChunk) -> None:
    results = [
        block
        for block in message.content_blocks
        if block["type"] == "server_tool_result"
    ]
    assert len(results) == 1
    assert results[0]["tool_call_id"] == "lc_primary-server-tool-0"
    assert message.additional_kwargs["provider_server_tool_state_by_call_id"] == {
        "lc_primary-server-tool-0": "provider-state-1"
    }


def test_top_level_repeat_with_late_state_updates_existing_lifecycle() -> None:
    _assert_one_correlated_result(_message(_mirrored_events()))


def test_nested_repeat_with_late_state_updates_existing_lifecycle() -> None:
    _assert_one_correlated_result(_message(_mirrored_events(nested_repeat=True)))


@pytest.mark.parametrize("nested_repeat", [False, True])
def test_conflicting_cross_event_repeat_fails_closed(nested_repeat: bool) -> None:
    first = _execution()
    conflicting = _execution(name="image_generate", output={"image_id": "image-1"})
    repeated: dict[str, Any]
    if nested_repeat:
        repeated = {
            "messages": [
                {
                    "role": "reasoning",
                    "content": [{"tool_execution": conflicting}],
                }
            ]
        }
    else:
        repeated = {"tool_execution": conflicting}

    with pytest.raises(ValueError, match=r"(?i)conflict|ambiguous"):
        _message(
            [
                {
                    "event": "response.tool.completed",
                    "tool_execution": first,
                },
                {
                    "event": "response.message.done",
                    "tools_state_id": "provider-state-1",
                    "finish_reason": "stop",
                    **repeated,
                },
            ]
        )


def test_repeat_with_same_explicit_id_is_still_one_logical_result() -> None:
    execution = _execution(call_id="execution-1")

    message = _message(
        [
            {
                "event": "response.tool.completed",
                "tool_execution": execution,
            },
            {
                "event": "response.message.done",
                "tool_execution": execution,
                "finish_reason": "stop",
            },
        ]
    )

    results = [
        block
        for block in message.content_blocks
        if block["type"] == "server_tool_result"
    ]
    assert len(results) == 1
    assert results[0]["tool_call_id"] == "execution-1"


def test_repeat_with_same_explicit_id_rejects_conflicting_terminal_payload() -> None:
    with pytest.raises(
        ValueError,
        match=r"(?i)conflicting repeated terminal payload",
    ):
        _message(
            [
                {
                    "event": "response.tool.completed",
                    "tool_execution": _execution(call_id="execution-1"),
                },
                {
                    "event": "response.message.done",
                    "finish_reason": "stop",
                    "tool_execution": _execution(
                        call_id="execution-1",
                        output={"matches": 2},
                    ),
                },
            ]
        )


def test_idless_done_repeat_rejects_ambiguous_equal_payloads() -> None:
    with pytest.raises(ValueError, match=r"matches multiple.*ambiguous"):
        _message(
            [
                {
                    "event": "response.tool.completed",
                    "tool_execution": _execution(),
                },
                {
                    "event": "response.tool.completed",
                    "tool_execution": _execution(),
                },
                {
                    "event": "response.message.done",
                    "finish_reason": "stop",
                    "tool_execution": _execution(),
                },
            ]
        )


@pytest.mark.parametrize("first_call_id", [None, "execution-1"])
@pytest.mark.parametrize("second_output", [{"matches": 1}, {"matches": 2}])
def test_message_done_preserves_new_explicit_execution(
    first_call_id: str | None,
    second_output: dict[str, int],
) -> None:
    message = _message(
        [
            {
                "event": "response.tool.completed",
                "tool_execution": _execution(call_id=first_call_id),
            },
            {
                "event": "response.message.done",
                "finish_reason": "stop",
                "tool_execution": _execution(
                    call_id="execution-2",
                    output=second_output,
                ),
            },
        ]
    )

    results = [
        block
        for block in message.content_blocks
        if block["type"] == "server_tool_result"
    ]
    assert [result["tool_call_id"] for result in results] == [
        first_call_id or "lc_primary-server-tool-0",
        "execution-2",
    ]


def test_explicit_execution_does_not_claim_prior_idless_late_state() -> None:
    message = _message(
        [
            {
                "event": "response.tool.completed",
                "tool_execution": _execution(),
            },
            {
                "event": "response.tool.completed",
                "tools_state_id": "provider-state-2",
                "tool_execution": _execution(call_id="execution-2"),
            },
            {
                "event": "response.message.done",
                "tools_state_id": "provider-state-2",
                "finish_reason": "stop",
            },
        ]
    )

    results = [
        block
        for block in message.content_blocks
        if block["type"] == "server_tool_result"
    ]
    assert [result["tool_call_id"] for result in results] == [
        "lc_primary-server-tool-0",
        "execution-2",
    ]
    assert message.additional_kwargs.get("provider_server_tool_state_by_call_id") in (
        None,
        {},
    )
    assert message.additional_kwargs["tools_state_id"] == "provider-state-2"


def test_idless_late_state_mirror_binds_to_prior_explicit_execution() -> None:
    execution = _execution(call_id="execution-1")
    message = _message(
        [
            {
                "event": "response.tool.completed",
                "tool_execution": execution,
            },
            {
                "event": "response.message.done",
                "tools_state_id": "provider-state-1",
                "finish_reason": "stop",
                "tool_execution": _execution(),
            },
        ]
    )

    results = [
        block
        for block in message.content_blocks
        if block["type"] == "server_tool_result"
    ]
    assert [result["tool_call_id"] for result in results] == ["execution-1"]
    assert message.additional_kwargs["provider_server_tool_state_by_call_id"] == {
        "execution-1": "provider-state-1"
    }


@pytest.mark.parametrize(
    "event_name",
    ["response.tool.completed", "response.message.done"],
)
def test_mixed_event_reconciles_only_resolver_owned_idless_mirror(
    event_name: str,
) -> None:
    mixed_event: dict[str, Any] = {
        "event": event_name,
        "tools_state_id": "provider-state-1",
        "messages": [
            {
                "role": "reasoning",
                "tool_execution": _execution(
                    call_id="execution-2",
                    name="image_generate",
                    output={"image_id": "image-1"},
                ),
            },
            {
                "role": "reasoning",
                "tool_execution": _execution(),
            },
        ],
    }
    if event_name == "response.message.done":
        mixed_event["finish_reason"] = "stop"
    events = [
        {
            "event": "response.tool.completed",
            "tool_execution": _execution(),
        },
        mixed_event,
    ]
    if event_name != "response.message.done":
        events.append(
            {
                "event": "response.message.done",
                "tools_state_id": "provider-state-1",
                "finish_reason": "stop",
            }
        )

    message = _message(events)
    results = [
        block
        for block in message.content_blocks
        if block["type"] == "server_tool_result"
    ]
    assert [result["tool_call_id"] for result in results] == [
        "lc_primary-server-tool-0",
        "execution-2",
    ]
    assert message.additional_kwargs["provider_server_tool_state_by_call_id"] == {
        "lc_primary-server-tool-0": "provider-state-1"
    }


@pytest.mark.parametrize("tools_state_id", [None, "provider-state-2"])
def test_active_idless_lifecycle_wins_over_old_equal_payload(
    tools_state_id: str | None,
) -> None:
    started: dict[str, Any] = {
        "event": "response.tool.started",
        "tool_execution": {
            "name": "web_search",
            "status": "running",
        },
    }
    completed: dict[str, Any] = {
        "event": "response.message.done",
        "finish_reason": "stop",
        "tool_execution": _execution(),
    }
    if tools_state_id is not None:
        started["tools_state_id"] = tools_state_id
        completed["tools_state_id"] = tools_state_id

    message = _message(
        [
            {
                "event": "response.tool.completed",
                "tool_execution": _execution(),
            },
            started,
            completed,
        ]
    )

    results = [
        block
        for block in message.content_blocks
        if block["type"] == "server_tool_result"
    ]
    assert [result["tool_call_id"] for result in results] == [
        "lc_primary-server-tool-0",
        "lc_primary-server-tool-1",
    ]


def test_nonterminal_update_after_terminal_result_fails_closed() -> None:
    with pytest.raises(ValueError, match=r"non-terminal update after.*terminal"):
        _message(
            [
                {
                    "event": "response.tool.completed",
                    "tool_execution": _execution(call_id="execution-1"),
                },
                {
                    "event": "response.tool.in_progress",
                    "tool_execution": {
                        "call_id": "execution-1",
                        "name": "web_search",
                        "status": "running",
                    },
                },
            ]
        )


def test_public_stream_deduplicates_top_level_cross_event_mirror(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.stream.side_effect = lambda payload: iter(
        map(_sdk_event, _mirrored_events())
    )

    message = reduce(add, GigaChat(use_api_v2=True).stream("Search"))

    _assert_one_correlated_result(message)


async def test_public_astream_deduplicates_nested_cross_event_mirror(
    sdk_client: MagicMock,
) -> None:
    async def events() -> AsyncIterator[gm.PrimaryChatCompletionChunk]:
        for event in _mirrored_events(nested_repeat=True):
            yield _sdk_event(event)

    sdk_client.achat.stream.side_effect = lambda payload: events()

    chunks = [chunk async for chunk in GigaChat(use_api_v2=True).astream("Search")]

    _assert_one_correlated_result(reduce(add, chunks))


def test_streaming_invoke_deduplicates_cross_event_mirror(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.stream.side_effect = lambda payload: iter(
        map(_sdk_event, _mirrored_events())
    )

    message = GigaChat(use_api_v2=True, streaming=True).invoke("Search")

    _assert_one_correlated_result(message)
