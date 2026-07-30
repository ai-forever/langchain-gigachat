"""Wrapper-level stream, structured-output, and orchestration regressions."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import gigachat.models as gm
import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGenerationChunk
from pydantic import BaseModel

from langchain_gigachat.chat_models.gigachat import GigaChat

from .fixtures import CREATED_AT, MESSAGE_ID, MODEL


class OutputSchema(BaseModel):
    value: int


def _message(chunk: ChatGenerationChunk) -> AIMessageChunk:
    assert isinstance(chunk.message, AIMessageChunk)
    return cast(AIMessageChunk, chunk.message)


def _tool_failed_then_done() -> Iterator[gm.PrimaryChatCompletionChunk]:
    yield gm.PrimaryChatCompletionChunk.model_validate(
        {
            "event": "response.tool.failed",
            "model": MODEL,
            "created_at": CREATED_AT,
            "finish_reason": "tool_error",
            "error": {"message": "tool failed"},
        }
    )
    yield gm.PrimaryChatCompletionChunk(
        event="response.message.done",
        model=MODEL,
        created_at=CREATED_AT,
        messages=None,
        message_id=MESSAGE_ID,
        finish_reason="stop",
    )


async def _async_items(items: Iterator[Any]) -> AsyncIterator[Any]:
    for item in items:
        yield item


def test_zero_event_primary_internal_stream_yields_nothing(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.stream.return_value = iter(())
    llm = GigaChat(model=MODEL, use_api_v2=True)

    assert list(llm._stream([HumanMessage("Hello")])) == []


async def test_zero_event_primary_internal_astream_yields_nothing(
    sdk_client: MagicMock,
) -> None:
    sdk_client.achat.stream.return_value = _async_items(iter(()))
    llm = GigaChat(model=MODEL, use_api_v2=True)

    assert [chunk async for chunk in llm._astream([HumanMessage("Hello")])] == []


def test_zero_event_primary_streaming_invoke_uses_langchain_empty_stream_error(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.stream.return_value = iter(())
    llm = GigaChat(model=MODEL, use_api_v2=True, streaming=True)

    with pytest.raises(ValueError, match="No generations found in stream"):
        llm.invoke("Hello")


def test_tool_terminal_does_not_replace_authoritative_message_done(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.stream.side_effect = lambda payload: _tool_failed_then_done()
    llm = GigaChat(model=MODEL, use_api_v2=True)

    chunks = list(llm._stream([HumanMessage("Hello")]))

    assert len(chunks) == 2
    assert chunks[0].generation_info == {"finish_reason": "tool_error"}
    assert _message(chunks[0]).chunk_position is None
    assert chunks[0].message.response_metadata["provider_field_events"] == [
        {"error": {"message": "tool failed"}}
    ]
    assert chunks[1].generation_info == {"finish_reason": "stop"}
    assert _message(chunks[1]).chunk_position == "last"


def _primary_json_response(
    *,
    finish_reason: str,
) -> gm.ChatCompletionResponse:
    return gm.ChatCompletionResponse(
        model=MODEL,
        created_at=CREATED_AT,
        messages=[
            gm.ChatMessage(
                role="assistant",
                message_id=MESSAGE_ID,
                content=[gm.ChatContentPart(text='{"value": 7}')],
            )
        ],
        message_id=MESSAGE_ID,
        finish_reason=finish_reason,
    )


@pytest.mark.parametrize("finish_reason", ["length", "error", "content_filter"])
def test_native_structured_output_rejects_valid_json_from_unsuccessful_finish(
    sdk_client: MagicMock,
    finish_reason: str,
) -> None:
    sdk_client.chat.create.return_value = _primary_json_response(
        finish_reason=finish_reason
    )
    chain = GigaChat(model=MODEL, use_api_v2=True).with_structured_output(
        OutputSchema,
        method="json_schema",
        include_raw=True,
    )

    result = chain.invoke("Hello")

    assert isinstance(result, dict)
    assert result["raw"].content == '{"value": 7}'
    assert result["raw"].response_metadata["finish_reason"] == finish_reason
    assert result["parsed"] is None
    assert isinstance(result["parsing_error"], OutputParserException)


def test_create_agent_raw_schema_uses_explicit_structured_output_profile(
    sdk_client: MagicMock,
) -> None:
    agents = pytest.importorskip("langchain.agents")
    sdk_client.chat.create.return_value = _primary_json_response(finish_reason="stop")
    model = GigaChat(
        model=MODEL,
        use_api_v2=True,
        profile={"structured_output": True},
    )

    agent = agents.create_agent(model, response_format=OutputSchema)
    result = agent.invoke({"messages": [{"role": "user", "content": "Return seven"}]})

    assert result["structured_response"] == OutputSchema(value=7)
    payload = sdk_client.chat.create.call_args.args[0]
    assert payload.response_format.type == "json_schema"


def test_create_agent_raw_schema_without_profile_fails_actionably(
    sdk_client: MagicMock,
) -> None:
    agents = pytest.importorskip("langchain.agents")
    model = GigaChat(model=MODEL, use_api_v2=True)
    agent = agents.create_agent(model, response_format=OutputSchema)

    with pytest.raises(ValueError, match=r"profile=.*structured_output"):
        agent.invoke({"messages": [{"role": "user", "content": "Return seven"}]})

    sdk_client.chat.create.assert_not_called()


def test_create_agent_explicit_provider_strategy_does_not_require_profile(
    sdk_client: MagicMock,
) -> None:
    agents = pytest.importorskip("langchain.agents")
    structured_output = pytest.importorskip("langchain.agents.structured_output")
    sdk_client.chat.create.return_value = _primary_json_response(finish_reason="stop")
    model = GigaChat(model=MODEL, use_api_v2=True)

    agent = agents.create_agent(
        model,
        response_format=structured_output.ProviderStrategy(OutputSchema),
    )
    result = agent.invoke({"messages": [{"role": "user", "content": "Return seven"}]})

    assert result["structured_response"] == OutputSchema(value=7)


def _base64_image_message() -> HumanMessage:
    return HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,aW1hZ2U="},
            }
        ]
    )


def test_legacy_route_validation_happens_before_attachment_upload(
    sdk_client: MagicMock,
) -> None:
    sdk_client.upload_file.return_value = SimpleNamespace(id_="uploaded")
    llm = GigaChat(
        model=MODEL,
        auto_upload_attachments=True,
    )

    with pytest.raises(ValueError, match="primary-only"):
        llm.invoke(
            [_base64_image_message()],
            filter_config={"request_content": {"neuro": False}},
        )

    sdk_client.upload_file.assert_not_called()
    sdk_client.chat.assert_not_called()


def test_response_format_validation_happens_before_attachment_upload(
    sdk_client: MagicMock,
) -> None:
    sdk_client.upload_file.return_value = SimpleNamespace(id_="uploaded")
    llm = GigaChat(
        model=MODEL,
        use_api_v2=True,
        auto_upload_attachments=True,
    )

    with pytest.raises(ValueError, match="Unsupported"):
        llm.invoke(
            [_base64_image_message()],
            response_format={"type": "unsupported"},
        )

    sdk_client.upload_file.assert_not_called()
    sdk_client.chat.create.assert_not_called()
