"""Regression checks for the SDK contract alignment (without provider requests)."""

import json
from typing import Any

import gigachat.models as gm
import httpx
import pytest
from gigachat.context import session_id_cvar
from gigachat.settings import Settings
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from langchain_gigachat import GigaChat
from langchain_gigachat.chat_models.gigachat import (
    _convert_delta_to_message_chunk,
    _convert_dict_to_message,
    _convert_message_to_dict,
)
from langchain_gigachat.embeddings import GigaChatEmbeddings


def _response() -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": "ok"},
                "index": 0,
                "finish_reason": "stop",
            }
        ],
        "created": 1,
        "model": "test-model",
        "object": "chat.completion",
        "usage": {
            "prompt_tokens": 14,
            "precached_prompt_tokens": 2430,
            "completion_tokens": 2,
            "total_tokens": 16,
        },
    }


@pytest.mark.parametrize("primary", [False, True])
@pytest.mark.parametrize("context_id", [None, "context-session"])
async def test_session_header_reaches_sdk_sync_and_async(
    monkeypatch: pytest.MonkeyPatch,
    primary: bool,
    context_id: str | None,
) -> None:
    if "session_id" not in Settings.model_fields:
        pytest.skip("SDK session_id support requires the aligned SDK")
    seen: list[httpx.Request] = []
    body = (
        {"messages": [{"role": "assistant", "content": "ok"}]}
        if primary
        else _response()
    )

    def send(
        client: httpx.Client, request: httpx.Request, **kwargs: Any
    ) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=body, request=request)

    async def asend(
        client: httpx.AsyncClient, request: httpx.Request, **kwargs: Any
    ) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=body, request=request)

    monkeypatch.setattr(httpx.Client, "send", send)
    monkeypatch.setattr(httpx.AsyncClient, "send", asend)
    llm = GigaChat(
        model="test-model",
        access_token="test-token",
        session_id="client-session",
        use_api_v2=primary,
    )
    token = session_id_cvar.set(context_id)
    try:
        assert llm.invoke("hello").content == "ok"
        assert (await llm.ainvoke("hello")).content == "ok"
        assert session_id_cvar.get() == context_id
    finally:
        session_id_cvar.reset(token)
    assert len(seen) == 2
    assert all(
        request.headers["X-Session-ID"] == (context_id or "client-session")
        for request in seen
    )
    route = "/v2/chat/completions" if primary else "/v1/chat/completions"
    assert all(request.url.path == route for request in seen)


@pytest.mark.parametrize("model_class", [GigaChat, GigaChatEmbeddings])
def test_session_id_not_silently_ignored_by_older_sdk(model_class: Any) -> None:
    model = model_class(session_id="session")
    if "session_id" in Settings.model_fields:
        assert model._get_client_init_kwargs()["session_id"] == "session"
    else:
        with pytest.raises(ValueError, match="session_id requires"):
            model._get_client_init_kwargs()


def test_legacy_reasoning_budget_and_invocation_override() -> None:
    llm = GigaChat(reasoning_effort="low", reasoning_max_tokens=128)
    payload = llm._build_payload(
        [HumanMessage(content="hello")],
        reasoning_effort="high",
        reasoning_max_tokens=64,
    )
    from gigachat.api.chat import _build_request_json

    body = _build_request_json(payload)
    assert body["reasoning_effort"] == "high"
    assert body["reasoning_max_tokens"] == 64
    assert "additional_fields" not in body


def test_legacy_assistant_id_is_supported_by_aligned_sdk() -> None:
    llm = GigaChat()
    llm._validate_legacy_kwargs({"assistant_id": "assistant"})
    if "assistant_id" not in gm.Chat.model_fields:
        with pytest.raises(ValueError, match="updated GigaChat SDK"):
            llm._build_payload(
                [HumanMessage(content="hello")], assistant_id="assistant"
            )
    else:
        payload = llm._build_payload(
            [HumanMessage(content="hello")], assistant_id="assistant"
        )
        assert payload.model_dump()["assistant_id"] == "assistant"


def test_legacy_usage_includes_cached_input_but_keeps_raw_counts() -> None:
    result = GigaChat()._create_chat_result(
        gm.ChatCompletion.model_validate(_response())
    )
    message = result.generations[0].message
    assert isinstance(message, AIMessage)
    assert message.usage_metadata == {
        "input_tokens": 2444,
        "output_tokens": 2,
        "total_tokens": 2446,
        "input_token_details": {"cache_read": 2430},
    }
    assert result.llm_output is not None
    assert result.llm_output["token_usage"]["prompt_tokens"] == 14
    assert result.llm_output["token_usage"]["total_tokens"] == 16


@pytest.mark.skipif(
    "id_" not in gm.FunctionCall.model_fields,
    reason="SDK function-call IDs require aligned SDK",
)
def test_legacy_function_id_roundtrip_and_stream() -> None:
    raw = {
        "role": "assistant",
        "content": "",
        "functions_state_id": "state",
        "function_call": {"id": "call-1", "name": "lookup", "arguments": {"x": 1}},
    }
    message = _convert_dict_to_message(gm.Messages.model_validate(raw))
    assert isinstance(message, AIMessage)
    assert message.tool_calls[0]["id"] == "call-1"
    assert message.additional_kwargs["functions_state_id"] == "state"
    assert (
        _convert_message_to_dict(message).model_dump(by_alias=True)["function_call"][
            "id"
        ]
        == "call-1"
    )
    chunk = _convert_delta_to_message_chunk(raw, AIMessageChunk)
    assert isinstance(chunk, AIMessageChunk)
    assert chunk.tool_calls[0]["id"] == "call-1"


@pytest.mark.skipif(
    "additional_data" not in gm.ChatCompletion.model_fields,
    reason="SDK metadata requires aligned SDK",
)
def test_legacy_response_and_stream_preserve_contract_metadata() -> None:
    payload = _response()
    metadata = {
        "thread_id": "thread",
        "message_id": "message",
        "additional_data": {"sources": [{"title": "Example"}]},
        "error_details": {"http_status": 200},
    }
    payload.update(metadata)
    payload["choices"][0]["message"].update(
        {"inline_data": {"sources": []}, "logprobs": [{"token": "ok", "logprob": -0.1}]}
    )
    result = GigaChat()._create_chat_result(gm.ChatCompletion.model_validate(payload))
    message = result.generations[0].message
    for key, value in metadata.items():
        assert message.response_metadata[key] == value
    assert message.additional_kwargs["inline_data"] == {"sources": []}
    assert message.additional_kwargs["logprobs"][0]["token"] == "ok"
    stream = {**metadata, "choices": [{"delta": {"content": "ok"}}]}
    pending: dict[str, Any] = {}
    llm = GigaChat()
    _, first, _ = llm._build_stream_chunk(stream, True, pending)
    _, second, _ = llm._build_stream_chunk(stream, False, pending)
    assert first == {"x_headers": {}}
    assert second == {}
    assert pending == metadata


@pytest.mark.skipif(
    "additional_data" not in gm.ChatCompletionChunk.model_fields,
    reason="SDK metadata requires aligned SDK",
)
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_legacy_metadata_snapshots_survive_sdk_http_stream(
    monkeypatch: pytest.MonkeyPatch, asynchronous: bool
) -> None:
    common = {
        "created": 1,
        "model": "test-model",
        "object": "chat.completion.chunk",
        "thread_id": "thread",
        "message_id": "message",
    }
    draft = {"sources": [{"index": 0, "url": "https://example.com", "title": "Draft"}]}
    final = {"sources": [{"index": 0, "url": "https://example.com", "title": "Final"}]}
    events = [
        {
            **common,
            "choices": [{"delta": {"content": "ok"}, "index": 0}],
            "additional_data": draft,
            "error_details": {"message": "Pending"},
        },
        {
            **common,
            "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
            "additional_data": draft,
        },
        {
            **common,
            "choices": [],
            "additional_data": final,
            "error_details": {"message": "Complete"},
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        },
    ]
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    body += "data: [DONE]\n\n"
    seen: list[httpx.Request] = []

    def send(
        client: httpx.Client | httpx.AsyncClient, request: httpx.Request, **kwargs: Any
    ) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            text=body,
            headers={"content-type": "text/event-stream", "x-request-id": "request"},
            request=request,
        )

    async def asend(
        client: httpx.AsyncClient, request: httpx.Request, **kwargs: Any
    ) -> httpx.Response:
        return send(client, request, **kwargs)

    monkeypatch.setattr(httpx.Client, "send", send)
    monkeypatch.setattr(httpx.AsyncClient, "send", asend)
    llm = GigaChat(model="test-model", access_token="test-token")
    chunks = (
        [chunk async for chunk in llm.astream("hello")]
        if asynchronous
        else list(llm.stream("hello"))
    )
    aggregate = chunks[0]
    for chunk in chunks[1:]:
        aggregate += chunk
    assert aggregate.content == "ok"
    assert aggregate.id == chunks[-1].id == "request"
    assert aggregate.response_metadata["additional_data"] == final
    assert aggregate.response_metadata["error_details"] == {"message": "Complete"}
    assert aggregate.response_metadata["thread_id"] == "thread"
    assert aggregate.response_metadata["message_id"] == "message"
    assert aggregate.usage_metadata == {
        "input_tokens": 5,
        "output_tokens": 2,
        "total_tokens": 7,
        "input_token_details": {"cache_read": 0},
    }
    assert len(seen) == 1
    assert seen[0].url.path == "/v1/chat/completions"
