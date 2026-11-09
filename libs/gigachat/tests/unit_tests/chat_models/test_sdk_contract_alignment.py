"""Regression checks for the SDK contract alignment (without provider requests)."""

import json
from typing import Any

import gigachat.models as gm
import httpx2 as httpx
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

    def send(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=body, request=request)

    llm = GigaChat(
        model="test-model",
        access_token="test-token",
        session_id="client-session",
        use_api_v2=primary,
        http_client=httpx.Client(transport=httpx.MockTransport(send)),
        async_http_client=httpx.AsyncClient(transport=httpx.MockTransport(send)),
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

    def send(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            text=body,
            headers={"content-type": "text/event-stream", "x-request-id": "request"},
            request=request,
        )

    from gigachat import RequestOptions

    llm = GigaChat(
        model="test-model",
        access_token="test-token",
        default_request_options=RequestOptions(extra_headers={"X-Option": "value"}),
        http_client=httpx.Client(transport=httpx.MockTransport(send)),
        async_http_client=httpx.AsyncClient(transport=httpx.MockTransport(send)),
    )
    chunks = (
        [chunk async for chunk in llm.astream("hello")]
        if asynchronous
        else list(llm.stream("hello"))
    )
    assert seen[0].headers["X-Option"] == "value"
    assert json.loads(seen[0].content)["stream"] is True
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


@pytest.mark.parametrize("primary", [False, True])
async def test_constructor_defaults_reach_chat_wire(primary: bool) -> None:
    from gigachat import RequestOptions

    requests: list[httpx.Request] = []
    options = RequestOptions(
        extra_headers={"X-Test-Option": "visible"},
        extra_query=[("tag", "one"), ("tag", "two")],
        extra_body={"temperature": 0.25},
        timeout=11,
        max_retries=0,
    )

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = (
            {"messages": [{"role": "assistant", "content": "ok"}]}
            if primary
            else _response()
        )
        return httpx.Response(200, json=body, request=request)

    sync = httpx.Client(transport=httpx.MockTransport(respond))
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    model = GigaChat(
        model="test-model",
        access_token="token",
        use_api_v2=primary,
        default_request_options=options,
        http_client=sync,
        async_http_client=async_client,
    )
    assert model.invoke("hello").content == "ok"
    assert (await model.ainvoke("hello")).content == "ok"
    assert model.bind(temperature=0.5).invoke("hello").content == "ok"
    assert len(requests) == 3
    for request in requests:
        assert request.headers["X-Test-Option"] == "visible"
        assert request.url.params.get_list("tag") == ["one", "two"]
        assert json.loads(request.content)["temperature"] == 0.25
        assert request.extensions["timeout"]["read"] == 11
    await model.aclose()
    assert not sync.is_closed
    assert not async_client.is_closed
    sync.close()
    await async_client.aclose()


def test_runtime_options_are_excluded_from_diagnostics() -> None:
    from gigachat import RequestOptions
    from langchain_core.load import dumps

    secret = "synthetic-secret-not-for-tracing"
    options = RequestOptions(extra_headers={"X-Secret": secret})
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    model = GigaChat(default_request_options=options, http_client=client)
    assert model.default_request_options is options
    assert model.http_client is client
    assert secret not in repr(model)
    assert secret not in repr(model.model_dump())
    assert secret not in dumps(model)
    assert "default_request_options" not in model.model_json_schema()["properties"]
    assert "http_client" not in model.model_json_schema()["properties"]
    client.close()


@pytest.mark.parametrize("model_class", [GigaChat, GigaChatEmbeddings])
async def test_unused_close_does_not_construct_owner(
    model_class: Any, mocker: Any
) -> None:
    constructor = mocker.patch("gigachat.GigaChat")
    model = model_class()
    model.close()
    await model.aclose()
    constructor.assert_not_called()
