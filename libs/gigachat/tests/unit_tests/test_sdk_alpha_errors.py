"""Keep the SDK's retry budget, error types and partial-stream safety intact."""

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx2
import pytest
from gigachat import (
    APIConnectionError,
    APIResponseValidationError,
    APITimeoutError,
    RateLimitError,
    RequestOptions,
    ServerError,
    StreamInterruptedError,
)

from langchain_gigachat import GigaChat


@pytest.mark.parametrize("primary", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("status", [429, 503])
async def test_sdk_retry_budget_and_request_id(
    primary: bool, asynchronous: bool, status: int
) -> None:
    requests: list[httpx2.Request] = []

    def send(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            status,
            json={"error": "offline transient failure"},
            headers={"X-Request-ID": "retry-probe", "Retry-After": "0.001"},
        )

    with httpx2.Client(transport=httpx2.MockTransport(send)) as sync_client:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(send)
        ) as async_client:
            model = GigaChat(
                model="test-model",
                access_token="offline-token",
                use_api_v2=primary,
                http_client=sync_client,
                async_http_client=async_client,
                default_request_options=RequestOptions(max_retries=2),
            )
            try:
                with pytest.raises(
                    RateLimitError if status == 429 else ServerError
                ) as e:
                    if asynchronous:
                        await model.ainvoke("hello")
                    else:
                        model.invoke("hello")
                assert e.value.request_id == "retry-probe"
                assert len(requests) == 3
            finally:
                await model.aclose()


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_stateful_primary_request_is_not_retried(asynchronous: bool) -> None:
    requests: list[httpx2.Request] = []

    def send(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        assert json.loads(request.content)["storage"]["thread_id"] == "thread-probe"
        return httpx2.Response(503, json={"error": "offline"})

    with httpx2.Client(transport=httpx2.MockTransport(send)) as sync_client:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(send)
        ) as async_client:
            model = GigaChat(
                model="test-model",
                access_token="offline-token",
                use_api_v2=True,
                http_client=sync_client,
                async_http_client=async_client,
                default_request_options=RequestOptions(max_retries=2),
            )
            try:
                with pytest.raises(ServerError):
                    if asynchronous:
                        await model.ainvoke(
                            "hello", storage={"thread_id": "thread-probe"}
                        )
                    else:
                        model.invoke("hello", storage={"thread_id": "thread-probe"})
                assert len(requests) == 1
            finally:
                await model.aclose()


@pytest.mark.parametrize("primary", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("kind", ["connection", "timeout", "validation"])
async def test_sdk_errors_are_not_masked(
    primary: bool, asynchronous: bool, kind: str
) -> None:
    original = (
        httpx2.ConnectError("offline connection")
        if kind == "connection"
        else httpx2.ReadTimeout("offline timeout")
    )
    error_type = {
        "connection": APIConnectionError,
        "timeout": APITimeoutError,
        "validation": APIResponseValidationError,
    }[kind]

    def send(request: httpx2.Request) -> httpx2.Response:
        if kind != "validation":
            raise original
        return httpx2.Response(
            200, content=b"not JSON", headers={"X-Request-ID": "invalid-probe"}
        )

    with httpx2.Client(transport=httpx2.MockTransport(send)) as sync_client:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(send)
        ) as async_client:
            model = GigaChat(
                model="test-model",
                access_token="offline-token",
                use_api_v2=primary,
                http_client=sync_client,
                async_http_client=async_client,
                default_request_options=RequestOptions(max_retries=0),
            )
            try:
                with pytest.raises(error_type) as e:
                    if asynchronous:
                        await model.ainvoke("hello")
                    else:
                        model.invoke("hello")
                assert type(e.value) is error_type
                if kind == "validation":
                    assert isinstance(e.value, APIResponseValidationError)
                    assert e.value.request_id == "invalid-probe"
                else:
                    assert e.value.__cause__ is original
            finally:
                await model.aclose()


class _InterruptedSyncBody(httpx2.SyncByteStream):
    def __init__(self, event: bytes) -> None:
        self.event = event
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        yield self.event
        raise httpx2.ReadError("offline interrupted stream")

    def close(self) -> None:
        self.closed = True


class _InterruptedAsyncBody(httpx2.AsyncByteStream):
    def __init__(self, event: bytes) -> None:
        self.event = event
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.event
        raise httpx2.ReadError("offline interrupted stream")

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.parametrize("primary", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_partial_stream_is_closed_without_replay(
    primary: bool, asynchronous: bool
) -> None:
    event: dict[str, Any] = (
        {
            "event": "response.message.delta",
            "messages": [{"role": "assistant", "content": "ok"}],
        }
        if primary
        else {
            "choices": [{"delta": {"content": "ok"}, "index": 0}],
            "created": 1,
            "model": "test-model",
            "object": "chat.completion.chunk",
        }
    )
    data = f"data: {json.dumps(event)}\n\n".encode()
    body = _InterruptedAsyncBody(data) if asynchronous else _InterruptedSyncBody(data)
    attempts = 0

    def send(request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        attempts += 1
        return httpx2.Response(
            200,
            stream=body,
            headers={"Content-Type": "text/event-stream", "X-Request-ID": "partial"},
        )

    with httpx2.Client(transport=httpx2.MockTransport(send)) as sync_client:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(send)
        ) as async_client:
            model = GigaChat(
                model="test-model",
                access_token="offline-token",
                use_api_v2=primary,
                http_client=sync_client,
                async_http_client=async_client,
                default_request_options=RequestOptions(max_retries=2),
            )
            chunks = []
            try:
                with pytest.raises(StreamInterruptedError) as e:
                    if asynchronous:
                        async for chunk in model.astream("hello"):
                            chunks.append(chunk)
                    else:
                        chunks.extend(model.stream("hello"))
                assert "".join(chunk.text for chunk in chunks) == "ok"
                assert e.value.partial_output is True
                assert e.value.request_id == "partial"
                assert body.closed
                assert attempts == 1
            finally:
                await model.aclose()


@pytest.mark.parametrize("primary", [False, True])
@pytest.mark.parametrize("method", ["invoke", "ainvoke", "stream", "astream"])
async def test_extra_body_cannot_change_streaming_mode(
    primary: bool, method: str
) -> None:
    def send(request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("conflicting stream mode must fail before HTTP I/O")

    with httpx2.Client(transport=httpx2.MockTransport(send)) as sync_client:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(send)
        ) as async_client:
            model = GigaChat(
                model="test-model",
                access_token="offline-token",
                use_api_v2=primary,
                http_client=sync_client,
                async_http_client=async_client,
                default_request_options=RequestOptions(
                    extra_body={"stream": method in ("invoke", "ainvoke")}
                ),
            )
            try:
                with pytest.raises(ValueError, match="extra_body.stream conflicts"):
                    if method == "invoke":
                        model.invoke("hello")
                    elif method == "ainvoke":
                        await model.ainvoke("hello")
                    elif method == "stream":
                        list(model.stream("hello"))
                    else:
                        _ = [chunk async for chunk in model.astream("hello")]
            finally:
                await model.aclose()


@pytest.mark.parametrize("override", [False, True])
async def test_request_timeout_none_disables_instead_of_inheriting(
    override: bool,
) -> None:
    timeouts = []

    def send(request: httpx2.Request) -> httpx2.Response:
        timeouts.append(request.extensions["timeout"])
        return httpx2.Response(200, json={"data": [], "object": "list"})

    with httpx2.Client(
        transport=httpx2.MockTransport(send), timeout=17.0
    ) as sync_client:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(send), timeout=17.0
        ) as async_client:
            model = GigaChat(
                timeout=42,
                access_token="offline-token",
                http_client=sync_client,
                async_http_client=async_client,
                default_request_options=(
                    RequestOptions(timeout=None) if override else RequestOptions()
                ),
            )
            try:
                model.get_models()
                await model.aget_models()
                expected = None if override else 17.0
                assert (
                    timeouts
                    == [dict.fromkeys(("connect", "read", "write", "pool"), expected)]
                    * 2
                )
            finally:
                await model.aclose()
