"""Final public-workflow matrix for the assembled PR #78 contracts."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any
from unittest.mock import MagicMock

import gigachat.models as gm
import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import generate_from_stream
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGenerationChunk
from pydantic import BaseModel

from langchain_gigachat.chat_models._contracts import primary
from langchain_gigachat.chat_models.gigachat import GigaChat

from .fixtures import CREATED_AT, MESSAGE_ID, MODEL, build_function_call_response


class ContractOutput(BaseModel):
    value: int


def lookup_weather(city: str) -> None:
    """Look up weather for a city."""


def _response(
    messages: list[dict[str, Any]],
    **overrides: Any,
) -> gm.ChatCompletionResponse:
    values: dict[str, Any] = {
        "model": MODEL,
        "created_at": CREATED_AT,
        "messages": messages,
        "finish_reason": "stop",
    }
    values.update(overrides)
    return gm.ChatCompletionResponse.model_validate(values)


def _text_response(
    text: str,
    *,
    message_id: str = MESSAGE_ID,
) -> gm.ChatCompletionResponse:
    return _response(
        [
            {
                "role": "assistant",
                "message_id": message_id,
                "content": [{"text": text}],
            }
        ],
        message_id=message_id,
    )


def _primary_text_stream(text: str) -> Iterator[gm.PrimaryChatCompletionChunk]:
    split_at = max(1, len(text) // 2)
    for fragment in (text[:split_at], text[split_at:]):
        if fragment:
            yield gm.PrimaryChatCompletionChunk(
                event="response.message.delta",
                model=MODEL,
                created_at=CREATED_AT,
                messages=[
                    gm.ChatMessageChunk(
                        role="assistant",
                        message_id=MESSAGE_ID,
                        content=[gm.ChatContentPart(text=fragment)],
                    )
                ],
                message_id=MESSAGE_ID,
            )
    yield gm.PrimaryChatCompletionChunk(
        event="response.message.done",
        model=MODEL,
        created_at=CREATED_AT,
        messages=None,
        message_id=MESSAGE_ID,
        finish_reason="stop",
    )


async def _async_items(
    items: Iterator[Any],
) -> AsyncIterator[Any]:
    for item in items:
        yield item


def _assistant_history(payload: gm.ChatCompletionRequest) -> gm.ChatMessage:
    return next(message for message in payload.messages if message.role == "assistant")


@pytest.mark.parametrize(
    ("first_response", "expected_text"),
    [
        pytest.param(
            _text_response("First answer", message_id="plain-message"),
            "First answer",
            id="plain-text",
        ),
        pytest.param(
            _response(
                [
                    {
                        "role": "reasoning",
                        "content": [{"text": "Private reasoning"}],
                    },
                    {
                        "role": "assistant",
                        "content": [{"text": "Reasoned answer"}],
                    },
                ],
                message_id="reasoning-message",
            ),
            "Reasoned answer",
            id="reasoning",
        ),
    ],
)
def test_public_multi_turn_history_matrix(
    sdk_client: MagicMock,
    first_response: gm.ChatCompletionResponse,
    expected_text: str,
) -> None:
    sdk_client.chat.create.side_effect = [
        first_response,
        _text_response("Continued answer", message_id="continued-message"),
    ]
    model = GigaChat(model=MODEL, use_api_v2=True)

    first = model.invoke("Start")
    continued = model.invoke(
        [
            HumanMessage("Start"),
            first,
            HumanMessage("Continue"),
        ]
    )

    assert continued.text == "Continued answer"
    payload = sdk_client.chat.create.call_args_list[1].args[0]
    assistant = _assistant_history(payload)
    assert assistant.content == [gm.ChatContentPart(text=expected_text)]
    assert assistant.message_id == first_response.message_id


def test_public_web_search_output_can_be_continued(
    sdk_client: MagicMock,
) -> None:
    web_response = _response(
        [
            {
                "role": "reasoning",
                "tools_state_id": "web-state",
                "tool_execution": {
                    "call_id": "web-state",
                    "name": "web_search",
                    "status": "completed",
                    "output": {"matches": 1},
                },
            },
            {
                "role": "assistant",
                "content": [{"text": "Verified answer"}],
            },
        ],
        message_id="web-message",
    )
    sdk_client.chat.create.side_effect = [
        web_response,
        _text_response("Follow-up answer"),
    ]
    model = GigaChat(model=MODEL, use_api_v2=True).bind_tools([{"type": "web_search"}])

    first = model.invoke("Search")
    model.invoke(
        [
            HumanMessage("Search"),
            first,
            HumanMessage("Explain the source"),
        ]
    )

    payload = sdk_client.chat.create.call_args_list[1].args[0]
    assistant = _assistant_history(payload)
    assert assistant.content == [gm.ChatContentPart(text="Verified answer")]
    assert assistant.message_id == "web-message"
    assert assistant.tools_state_id == "web-state"


def test_public_generated_file_can_be_continued(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.create.side_effect = [
        _response(
            [
                {
                    "role": "assistant",
                    "content": [
                        {"text": "Generated report"},
                        {
                            "files": [
                                {
                                    "id": "report-file",
                                    "mime": "application/pdf",
                                }
                            ]
                        },
                    ],
                }
            ],
            message_id="file-message",
        ),
        _text_response("File follow-up"),
    ]
    model = GigaChat(model=MODEL, use_api_v2=True)

    first = model.invoke("Generate a report")
    model.invoke(
        [
            HumanMessage("Generate a report"),
            first,
            HumanMessage("Summarize the file"),
        ]
    )

    payload = sdk_client.chat.create.call_args_list[1].args[0]
    assistant = _assistant_history(payload)
    assert assistant.model_dump(exclude_none=True, by_alias=True)["content"] == [
        {"text": "Generated report"},
        {"files": [{"id": "report-file", "mime": "application/pdf"}]},
    ]


def test_public_client_tool_loop_replays_function_and_result(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.create.side_effect = [
        build_function_call_response(),
        _text_response("It is clear and 18 degrees."),
    ]
    model = GigaChat(model=MODEL, use_api_v2=True).bind_tools([lookup_weather])

    first = model.invoke("Weather in Moscow?")
    assert first.tool_calls
    result = model.invoke(
        [
            HumanMessage("Weather in Moscow?"),
            first,
            ToolMessage(
                content='{"temperature": 18, "condition": "clear"}',
                tool_call_id=first.tool_calls[0]["id"],
            ),
        ]
    )

    assert result.text == "It is clear and 18 degrees."
    payload = sdk_client.chat.create.call_args_list[1].args[0]
    assert [message.role for message in payload.messages] == [
        "user",
        "assistant",
        "tool",
    ]
    assert payload.messages[1].function_call == gm.PrimaryChatFunctionCall(
        name="lookup_weather",
        arguments={"city": "Moscow"},
    )
    function_result = payload.messages[2].content[0].function_result
    assert function_result is not None
    assert function_result.name == "lookup_weather"
    assert function_result.result == {
        "temperature": 18,
        "condition": "clear",
    }


def test_public_structured_output_valid_and_invalid_matrix(
    sdk_client: MagicMock,
) -> None:
    model = GigaChat(model=MODEL, use_api_v2=True)
    sdk_client.chat.create.return_value = _text_response('{"value": 7}')

    parsed = model.with_structured_output(
        ContractOutput,
        method="json_schema",
    ).invoke("Return JSON")

    assert parsed == ContractOutput(value=7)

    sdk_client.chat.create.return_value = _text_response("not JSON")
    raw_result = model.with_structured_output(
        ContractOutput,
        method="json_schema",
        include_raw=True,
    ).invoke("Return malformed JSON")

    assert isinstance(raw_result, dict)
    assert isinstance(raw_result["raw"], AIMessage)
    assert raw_result["raw"].text == "not JSON"
    assert raw_result["parsed"] is None
    assert isinstance(raw_result["parsing_error"], OutputParserException)

    with pytest.raises(OutputParserException):
        model.with_structured_output(
            ContractOutput,
            method="json_schema",
        ).invoke("Return malformed JSON")


def test_public_bind_tools_response_format_and_stream_aggregation(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.create.return_value = _text_response('{"value": 7}')
    bound = GigaChat(model=MODEL, use_api_v2=True).bind_tools(
        [lookup_weather],
        response_format=ContractOutput,
        strict=True,
    )

    response = bound.invoke("Return JSON or call the tool")

    assert response.additional_kwargs["parsed"] == ContractOutput(value=7)

    sdk_client.chat.stream.side_effect = lambda payload: _primary_text_stream(
        '{"value": 7}'
    )
    streamed = (
        GigaChat(model=MODEL, use_api_v2=True, streaming=True)
        .bind_tools(
            [lookup_weather],
            response_format=ContractOutput,
            strict=True,
        )
        .invoke("Return streamed JSON")
    )

    assert streamed.text == '{"value": 7}'
    assert streamed.additional_kwargs["parsed"] == ContractOutput(value=7)


def test_public_direct_sync_stream_has_one_actual_terminal_chunk(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.stream.side_effect = lambda payload: _primary_text_stream(
        "Streamed response"
    )

    chunks = list(GigaChat(model=MODEL, use_api_v2=True).stream("Stream"))

    assert "".join(chunk.text for chunk in chunks) == "Streamed response"
    assert [
        index for index, chunk in enumerate(chunks) if chunk.chunk_position == "last"
    ] == [len(chunks) - 1]


async def test_public_async_invoke_and_direct_stream_matrix(
    sdk_client: MagicMock,
) -> None:
    sdk_client.achat.create.return_value = _text_response("Async response")
    model = GigaChat(model=MODEL, use_api_v2=True)

    response = await model.ainvoke("Async invoke")

    assert response.text == "Async response"

    sdk_client.achat.stream.side_effect = lambda payload: _async_items(
        _primary_text_stream("Async stream")
    )
    chunks = [chunk async for chunk in model.astream("Async stream")]

    assert "".join(chunk.text for chunk in chunks) == "Async stream"
    assert [
        index for index, chunk in enumerate(chunks) if chunk.chunk_position == "last"
    ] == [len(chunks) - 1]


def test_late_tool_name_and_unknown_events_survive_real_aggregation() -> None:
    events = [
        {
            "event": "response.tool.started",
            "tool_execution": {
                "call_id": "tool-1",
                "status": "running",
            },
        },
        {
            "event": "response.tool.delta",
            "tool_execution": {
                "call_id": "tool-1",
                "name": "web_search",
                "status": "running",
            },
        },
        {
            "event": "response.future.started",
            "future_field": {"step": 1},
        },
        {
            "event": "response.future.delta",
            "future_field": {"step": 2},
        },
        {
            "event": "response.message.done",
            "finish_reason": "stop",
        },
    ]
    state = primary.StreamState()
    chunks: list[ChatGenerationChunk] = []
    for values in events:
        chunk = primary.convert_stream_event(
            gm.PrimaryChatCompletionChunk.model_validate(values),
            state=state,
        )
        assert chunk is not None
        chunks.append(chunk)

    message = generate_from_stream(iter(chunks)).generations[0].message

    assert isinstance(message, AIMessage)
    tool_update = next(
        block
        for block in message.content_blocks
        if block["type"] == "server_tool_call_chunk"
    )
    assert tool_update["name"] == "web_search"
    assert message.response_metadata["raw_events"] == events[2:4]


def test_mirrored_non_stream_tool_matches_real_streamed_message() -> None:
    execution = {
        "call_id": "server-tool-1",
        "name": "web_search",
        "status": "completed",
        "output": {"matches": 1},
    }
    inline_data = {
        "sources": {
            "source-1": {
                "url": "https://example.test/source",
                "title": "Example",
            }
        }
    }
    part = {
        "tool_execution": execution,
        "inline_data": inline_data,
        "provider_extension": {"trace_id": "trace-1"},
    }
    message = {
        "role": "assistant",
        "tools_state_id": "server-tool-1",
        "content": [part],
        "tool_execution": execution,
    }
    response = _response(
        [message],
        message_id="tool-message",
        tool_execution=execution,
    )
    non_stream = primary.create_chat_result(response).generations[0].message
    assert isinstance(non_stream, AIMessage)

    state = primary.StreamState()
    stream_chunk = primary.convert_stream_event(
        gm.PrimaryChatCompletionChunk.model_validate(
            {
                "event": "response.tool.completed",
                "message_id": "tool-message",
                "tools_state_id": "server-tool-1",
                "messages": [message],
                "tool_execution": execution,
            }
        ),
        state=state,
    )
    assert stream_chunk is not None
    streamed = generate_from_stream(iter([stream_chunk])).generations[0].message

    assert isinstance(streamed, AIMessage)
    expected_result = {
        "type": "server_tool_result",
        "id": "server-tool-1:result",
        "tool_call_id": "server-tool-1",
        "status": "success",
        "extras": {
            "provider_tool_execution": execution,
            "inline_data": inline_data,
            "provider_data": {
                "provider_extension": {
                    "trace_id": "trace-1",
                }
            },
        },
        "output": {"matches": 1},
    }
    assert non_stream.content_blocks == [expected_result]
    assert streamed.content_blocks == [{**expected_result, "index": 0}]
