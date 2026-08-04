"""Live integration coverage for the primary ``/v2/chat/completions`` route."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any, Literal, cast

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
)
from pydantic import BaseModel, Field

from langchain_gigachat.chat_models.gigachat import GigaChat

_AUTH_ENV_NAMES = (
    "GIGACHAT_CREDENTIALS",
    "GIGACHAT_ACCESS_TOKEN",
)
_HAS_AUTH = any(os.getenv(name) for name in _AUTH_ENV_NAMES) or bool(
    os.getenv("GIGACHAT_USER") and os.getenv("GIGACHAT_PASSWORD")
)
_MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat-3-Lightning")
_V1_BASE_URL = os.getenv("GIGACHAT_BASE_URL", "https://api.giga.chat/v1")
_ASSISTANT_ID = os.getenv("GIGACHAT_V2_TEST_ASSISTANT_ID")
_THREAD_ID = os.getenv("GIGACHAT_V2_TEST_THREAD_ID")
_FILE_ID = os.getenv("GIGACHAT_V2_TEST_FILE_ID")

pytestmark = [
    pytest.mark.scheduled,
    pytest.mark.skipif(
        not _HAS_AUTH,
        reason=(
            "live GigaChat tests require GIGACHAT_CREDENTIALS, "
            "GIGACHAT_ACCESS_TOKEN, or GIGACHAT_USER/GIGACHAT_PASSWORD"
        ),
    ),
]


class StructuredAnswer(BaseModel):
    """Deterministic response used to verify native JSON Schema output."""

    status: Literal["PRIMARY_V2_JSON_SCHEMA_OK"] = Field(
        description="Return the exact requested status marker."
    )


class LookupWeather(BaseModel):
    """Look up the weather for a city."""

    city: str = Field(description="City whose weather should be looked up.")


def _auth_kwargs() -> dict[str, Any]:
    mapping = {
        "credentials": "GIGACHAT_CREDENTIALS",
        "access_token": "GIGACHAT_ACCESS_TOKEN",
        "user": "GIGACHAT_USER",
        "password": "GIGACHAT_PASSWORD",
        "scope": "GIGACHAT_SCOPE",
        "auth_url": "GIGACHAT_AUTH_URL",
        "ca_bundle_file": "GIGACHAT_CA_BUNDLE_FILE",
        "cert_file": "GIGACHAT_CERT_FILE",
        "key_file": "GIGACHAT_KEY_FILE",
        "key_file_password": "GIGACHAT_KEY_FILE_PASSWORD",
    }
    return {
        field: value
        for field, env_name in mapping.items()
        if (value := os.getenv(env_name))
    }


@pytest.fixture
def primary_llm() -> GigaChat:
    assert _V1_BASE_URL.rstrip("/").endswith("/v1"), (
        "GIGACHAT_V2_TEST_BASE_URL must be the API /v1 base URL; "
        "the SDK resource must resolve /v2/chat/completions itself"
    )
    return GigaChat(
        model=_MODEL,
        use_api_v2=True,
        base_url=_V1_BASE_URL,
        temperature=0,
        max_tokens=256,
        **_auth_kwargs(),
    )


def _message_text(message: AIMessage | AIMessageChunk) -> str:
    if isinstance(message.content, str):
        return message.content
    return "".join(
        str(block.get("text", ""))
        for block in message.content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _aggregate(chunks: Sequence[AIMessageChunk]) -> AIMessageChunk:
    assert chunks
    result = chunks[0]
    for chunk in chunks[1:]:
        result += chunk
    return result


def _assert_transport_metadata(message: AIMessage | AIMessageChunk) -> None:
    metadata = message.response_metadata
    x_headers = metadata["x_headers"]
    assert isinstance(x_headers, dict)
    assert x_headers["x-request-id"]
    assert message.id == x_headers["x-request-id"]
    assert message.usage_metadata
    assert message.usage_metadata["total_tokens"] > 0


def _assert_stateful_metadata(message: AIMessage | AIMessageChunk) -> None:
    metadata = message.response_metadata
    assert metadata["thread_id"]
    assert metadata["message_id"]


def _assert_server_tool_blocks_if_reported(
    message: AIMessage,
    blocks: Sequence[dict[str, Any]],
) -> None:
    if message.response_metadata.get("tool_execution") is not None:
        assert any(
            block.get("type") in {"server_tool_call", "server_tool_result"}
            for block in blocks
        )


def _assert_streamed_builtin_tool_lifecycle(
    chunks: Sequence[AIMessageChunk],
    aggregate: AIMessageChunk,
) -> None:
    assert sum(chunk.chunk_position == "last" for chunk in chunks) == 1
    assert _message_text(aggregate).strip()

    blocks = [dict(block) for block in aggregate.content_blocks]
    call = next(
        block
        for block in blocks
        if block.get("type") in {"server_tool_call", "server_tool_call_chunk"}
    )
    result = next(
        block for block in blocks if block.get("type") == "server_tool_result"
    )

    assert call["name"] == "web_search"
    assert call["id"]
    assert result["name"] == "web_search"
    assert result["tool_call_id"] == call["id"]
    assert result["status"] in {"success", "error"}
    assert aggregate.response_metadata["finish_reason"]
    _assert_transport_metadata(aggregate)


def test_sync_invoke_uses_primary_route_from_v1_base_url(
    primary_llm: GigaChat,
) -> None:
    result = primary_llm.invoke(
        "Reply with exactly PRIMARY_V2_SYNC_OK and no other text."
    )

    assert isinstance(result, AIMessage)
    assert _message_text(result).strip() == "PRIMARY_V2_SYNC_OK"
    assert primary_llm.base_url == _V1_BASE_URL
    _assert_transport_metadata(result)


async def test_async_invoke(primary_llm: GigaChat) -> None:
    result = await primary_llm.ainvoke(
        "Reply with exactly PRIMARY_V2_ASYNC_OK and no other text."
    )

    assert isinstance(result, AIMessage)
    assert _message_text(result).strip() == "PRIMARY_V2_ASYNC_OK"
    _assert_transport_metadata(result)


def test_sync_stream_aggregation_matches_non_stream(
    primary_llm: GigaChat,
) -> None:
    prompt = "Reply with exactly PRIMARY_V2_STREAM_OK and no other text."

    non_stream = primary_llm.invoke(prompt)
    chunks = list(primary_llm.stream(prompt))
    aggregate = _aggregate(chunks)

    assert all(isinstance(chunk, AIMessageChunk) for chunk in chunks)
    assert _message_text(non_stream).strip() == "PRIMARY_V2_STREAM_OK"
    assert _message_text(aggregate).strip() == _message_text(non_stream).strip()
    assert any(chunk.usage_metadata for chunk in chunks)
    assert any(
        chunk.response_metadata.get("finish_reason") is not None for chunk in chunks
    )
    _assert_transport_metadata(aggregate)


async def test_async_stream(primary_llm: GigaChat) -> None:
    chunks = [
        chunk
        async for chunk in primary_llm.astream(
            "Reply with exactly PRIMARY_V2_ASTREAM_OK and no other text."
        )
    ]
    aggregate = _aggregate(chunks)

    assert all(isinstance(chunk, AIMessageChunk) for chunk in chunks)
    assert _message_text(aggregate).strip() == "PRIMARY_V2_ASTREAM_OK"
    assert any(chunk.usage_metadata for chunk in chunks)
    assert any(
        chunk.response_metadata.get("finish_reason") is not None for chunk in chunks
    )
    _assert_transport_metadata(aggregate)


def test_sync_streamed_web_search_lifecycle(primary_llm: GigaChat) -> None:
    runnable = primary_llm.bind_tools(
        [{"type": "web_search"}],
        tool_choice="web_search",
    )

    chunks = [
        cast(AIMessageChunk, chunk)
        for chunk in runnable.stream(
            "Use web search to find the official Python website, then briefly "
            "identify what Python is."
        )
    ]
    aggregate = _aggregate(chunks)

    _assert_streamed_builtin_tool_lifecycle(chunks, aggregate)


def test_json_schema_structured_output(primary_llm: GigaChat) -> None:
    runnable = primary_llm.with_structured_output(
        StructuredAnswer,
        method="json_schema",
        strict=True,
    )

    result = runnable.invoke("Return the status marker PRIMARY_V2_JSON_SCHEMA_OK.")

    assert result == StructuredAnswer(status="PRIMARY_V2_JSON_SCHEMA_OK")


def test_schema_less_json_mode(primary_llm: GigaChat) -> None:
    result = primary_llm.with_structured_output(
        None,
        method="json_mode",
    ).invoke(
        "Return one JSON object with exactly two fields: "
        '"status" set to "PRIMARY_V2_JSON_MODE_OK" and "confidence" set to 1.'
    )

    assert result == {
        "status": "PRIMARY_V2_JSON_MODE_OK",
        "confidence": 1,
    }


def test_client_function_two_turn_roundtrip(primary_llm: GigaChat) -> None:
    initial_prompt = HumanMessage(
        "Call LookupWeather for Moscow. After the tool result, reply exactly "
        "PRIMARY_V2_TOOL_ROUNDTRIP_OK and no other text."
    )
    forced = primary_llm.bind_tools(
        [LookupWeather],
        tool_choice="LookupWeather",
    )

    tool_request = forced.invoke([initial_prompt])

    assert isinstance(tool_request, AIMessage)
    assert len(tool_request.tool_calls) == 1
    tool_call = tool_request.tool_calls[0]
    assert tool_call["name"] == "LookupWeather"
    assert tool_call["args"]["city"].casefold() == "moscow"
    assert tool_call["id"]

    tool_result = ToolMessage(
        content=json.dumps({"temperature_c": 18, "condition": "clear"}),
        tool_call_id=tool_call["id"],
        name=tool_call["name"],
    )
    follow_up = primary_llm.bind_tools(
        [LookupWeather],
        tool_choice="auto",
    ).invoke([initial_prompt, tool_request, tool_result])

    assert isinstance(follow_up, AIMessage)
    assert _message_text(follow_up).strip() == "PRIMARY_V2_TOOL_ROUNDTRIP_OK"


def test_web_search_returns_text_and_sources(
    primary_llm: GigaChat,
) -> None:
    runnable = primary_llm.bind_tools(
        [{"type": "web_search"}],
        tool_choice="web_search",
    )

    result = runnable.invoke(
        "Find the official Python website and briefly identify what Python is."
    )

    assert isinstance(result, AIMessage)
    blocks: list[dict[str, Any]] = [dict(block) for block in result.content_blocks]
    assert _message_text(result).strip()
    annotations = [
        annotation
        for block in blocks
        if block.get("type") == "text" and isinstance(block.get("annotations"), list)
        for annotation in block["annotations"]
        if isinstance(annotation, dict)
    ]
    assert any(
        annotation.get("type") == "citation" and annotation.get("url")
        for annotation in annotations
    )
    _assert_server_tool_blocks_if_reported(result, blocks)


@pytest.mark.skipif(
    not _ASSISTANT_ID,
    reason=("assistant coverage requires a pre-existing GIGACHAT_V2_TEST_ASSISTANT_ID"),
)
def test_existing_assistant_id(primary_llm: GigaChat) -> None:
    result = primary_llm.invoke(
        "Reply briefly to confirm the assistant request succeeded.",
        assistant_id=_ASSISTANT_ID,
    )

    assert isinstance(result, AIMessage)
    assert _message_text(result).strip()
    _assert_transport_metadata(result)
    _assert_stateful_metadata(result)


@pytest.mark.skipif(
    not _THREAD_ID,
    reason="thread coverage requires a pre-existing GIGACHAT_V2_TEST_THREAD_ID",
)
def test_existing_thread_continuation(primary_llm: GigaChat) -> None:
    result = primary_llm.invoke(
        "Reply briefly to confirm the thread continuation succeeded.",
        storage={"thread_id": _THREAD_ID},
    )

    assert isinstance(result, AIMessage)
    assert _message_text(result).strip()
    assert result.response_metadata["thread_id"] == _THREAD_ID
    _assert_transport_metadata(result)
    _assert_stateful_metadata(result)


@pytest.mark.skipif(
    not _FILE_ID,
    reason="file coverage requires a pre-existing GIGACHAT_V2_TEST_FILE_ID",
)
def test_existing_file_id_roundtrip(primary_llm: GigaChat) -> None:
    result = primary_llm.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": "Briefly describe the attached file."},
                    {"type": "file", "file_id": _FILE_ID},
                ]
            )
        ]
    )

    assert isinstance(result, AIMessage)
    assert _message_text(result).strip()
    _assert_transport_metadata(result)
