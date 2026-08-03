"""Output-to-history roundtrips for the primary chat contract."""

from __future__ import annotations

import copy
from typing import Any
from unittest.mock import MagicMock

import gigachat.models as gm
from langchain_core.language_models.chat_models import generate_from_stream
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGenerationChunk

from langchain_gigachat.chat_models._contracts import primary
from langchain_gigachat.chat_models.gigachat import GigaChat


def _response(
    messages: list[dict[str, Any]],
    **overrides: Any,
) -> gm.ChatCompletionResponse:
    values: dict[str, Any] = {"messages": messages}
    values.update(overrides)
    return gm.ChatCompletionResponse.model_validate(values)


def _output_message(response: gm.ChatCompletionResponse) -> AIMessage:
    message = primary.create_chat_result(response).generations[0].message
    assert isinstance(message, AIMessage)
    return message


def _replay(message: AIMessage) -> gm.ChatMessage:
    original = copy.deepcopy(message)
    converted = primary.convert_messages([message], cached_uploads={})[0]
    assert message == original
    return converted


def _stream_output(events: list[dict[str, Any]]) -> AIMessage:
    state = primary.StreamState()
    chunks: list[ChatGenerationChunk] = []
    for values in events:
        event = gm.PrimaryChatCompletionChunk.model_validate(values)
        chunk = primary.convert_stream_event(event, state=state)
        assert chunk is not None
        chunks.append(chunk)
    message = generate_from_stream(iter(chunks)).generations[0].message
    assert isinstance(message, AIMessage)
    return message


def test_reasoning_and_answer_output_replays_only_assistant_text() -> None:
    output = _output_message(
        _response(
            [
                {"role": "reasoning", "content": [{"text": "Think privately"}]},
                {"role": "assistant", "content": [{"text": "Final answer"}]},
            ],
            message_id="provider-message",
        )
    )

    replayed = _replay(output)

    assert replayed.model_dump(exclude_none=True, by_alias=True) == {
        "message_id": "provider-message",
        "content": [{"text": "Final answer"}],
        "role": "assistant",
    }


def test_public_output_can_be_reused_as_multi_turn_history(
    sdk_client: MagicMock,
) -> None:
    first_response = _response(
        [
            {"role": "reasoning", "content": [{"text": "Think privately"}]},
            {"role": "assistant", "content": [{"text": "First answer"}]},
        ],
        message_id="first-message",
    )
    sdk_client.chat.create.side_effect = [
        first_response,
        _response(
            [{"role": "assistant", "content": [{"text": "Continued answer"}]}],
            message_id="continued-message",
        ),
    ]
    model = GigaChat(use_api_v2=True)

    first = model.invoke("Start")
    continued = model.invoke([HumanMessage("Start"), first, HumanMessage("Continue")])

    assert continued.text == "Continued answer"
    payload = sdk_client.chat.create.call_args_list[1].args[0]
    replayed = next(
        message for message in payload.messages if message.role == "assistant"
    )
    assert replayed.message_id == "first-message"
    assert replayed.content == [gm.ChatContentPart(text="First answer")]


def test_citation_annotations_do_not_block_text_history_replay() -> None:
    output = _output_message(
        _response(
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "text": "Source-backed answer",
                            "inline_data": {
                                "sources": {
                                    "source-1": {
                                        "url": "https://example.test/source",
                                        "title": "Example",
                                    }
                                }
                            },
                        }
                    ],
                }
            ]
        )
    )

    replayed = _replay(output)

    assert replayed.content == [gm.ChatContentPart(text="Source-backed answer")]


def test_generated_image_audio_video_and_file_replay_by_provider_id() -> None:
    output = _output_message(
        _response(
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "files": [
                                {"id": "image-1", "mime": "image/png"},
                                {"id": "audio-1", "mime": "audio/mpeg"},
                                {"id": "video-1", "mime": "video/mp4"},
                                {"id": "file-1", "mime": "application/pdf"},
                            ]
                        }
                    ],
                }
            ]
        )
    )

    replayed = _replay(output)

    assert replayed.model_dump(exclude_none=True, by_alias=True)["content"] == [
        {"files": [{"id": "image-1", "mime": "image/png"}]},
        {"files": [{"id": "audio-1", "mime": "audio/mpeg"}]},
        {"files": [{"id": "video-1", "mime": "video/mp4"}]},
        {"files": [{"id": "file-1", "mime": "application/pdf"}]},
    ]


def test_server_tool_result_and_final_text_replay_without_server_state_blocks() -> None:
    output = _output_message(
        _response(
            [
                {
                    "role": "reasoning",
                    "tools_state_id": "server-state",
                    "tool_execution": {
                        "name": "web_search",
                        "status": "completed",
                        "output": {"matches": 1},
                    },
                },
                {"role": "assistant", "content": [{"text": "Verified answer"}]},
            ],
            message_id="provider-message",
        )
    )

    replayed = _replay(output)

    assert replayed.model_dump(exclude_none=True, by_alias=True) == {
        "message_id": "provider-message",
        "content": [{"text": "Verified answer"}],
        "tools_state_id": "server-state",
        "role": "assistant",
    }


def test_unknown_provider_content_does_not_break_history_replay() -> None:
    output = _output_message(
        _response(
            [
                {
                    "role": "assistant",
                    "content": [{"future_part": {"value": 42}}],
                    "future_message_field": "future-value",
                }
            ]
        )
    )

    replayed = _replay(output)

    assert replayed.content == []


def test_non_stream_tools_state_id_survives_output_to_history_roundtrip() -> None:
    output = _output_message(
        _response(
            [
                {
                    "role": "assistant",
                    "content": [{"text": "Continue later"}],
                    "tools_state_id": "non-stream-state",
                }
            ]
        )
    )

    replayed = _replay(output)

    assert replayed.tools_state_id == "non-stream-state"


def test_stream_aggregated_tools_state_id_survives_history_roundtrip() -> None:
    output = _stream_output(
        [
            {
                "event": "response.message.delta",
                "message_id": "stream-message",
                "tools_state_id": "stream-state",
                "messages": [
                    {"role": "assistant", "content": [{"text": "Streamed answer"}]}
                ],
            },
            {
                "event": "response.message.done",
                "message_id": "stream-message",
                "tools_state_id": "stream-state",
                "finish_reason": "stop",
            },
        ]
    )

    replayed = _replay(output)

    assert replayed.message_id == "stream-message"
    assert replayed.tools_state_id == "stream-state"
    assert replayed.content == [gm.ChatContentPart(text="Streamed answer")]


def test_client_function_call_and_tool_result_continue_as_provider_history() -> None:
    output = _output_message(
        _response(
            [
                {
                    "role": "assistant",
                    "message_id": "provider-message",
                    "tools_state_id": "client-state",
                    "function_call": {
                        "name": "lookup_weather",
                        "arguments": {"city": "Moscow"},
                    },
                }
            ],
            message_id="provider-message",
        )
    )
    history = primary.convert_messages(
        [
            output,
            ToolMessage(
                content='{"temperature": 18}',
                tool_call_id="client-state",
            ),
        ],
        cached_uploads={},
    )

    assert history[0].function_call is None
    assert history[0].content
    assert history[0].content[0].function_call == gm.PrimaryChatFunctionCall(
        name="lookup_weather",
        arguments={"city": "Moscow"},
    )
    assert history[0].tools_state_id == "client-state"
    assert history[1].tools_state_id == "client-state"
    assert history[1].content
    assert history[1].content[0].function_result
    assert history[1].content[0].function_result.name == "lookup_weather"


def test_late_stream_tool_state_becomes_tool_call_identity() -> None:
    output = _stream_output(
        [
            {
                "event": "response.message.delta",
                "message_id": "provider-message",
                "messages": [
                    {
                        "role": "assistant",
                        "function_call": {
                            "name": "lookup_weather",
                            "arguments": {"city": "Moscow"},
                        },
                    }
                ],
            },
            {
                "event": "response.message.done",
                "message_id": "provider-message",
                "tools_state_id": "provider-state",
                "finish_reason": "function_call",
            },
        ]
    )

    assert output.tool_calls[0]["id"] == "provider-state"
    assert "provider_tool_state_by_call_id" not in output.additional_kwargs

    history = primary.convert_messages(
        [
            output,
            ToolMessage(
                content='{"temperature": 18}',
                tool_call_id="provider-state",
            ),
        ],
        cached_uploads={},
    )

    assert history[0].tools_state_id == "provider-state"
    assert history[1].tools_state_id == "provider-state"
