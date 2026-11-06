"""Wrapper stream cleanup against actual HTTPX2 response bodies."""

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from typing import Any, cast

import httpx2
import pytest
from langchain_core.messages import HumanMessage

from langchain_gigachat import GigaChat


def _event(primary: bool) -> bytes:
    payload: dict[str, Any] = (
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
    return f"data: {json.dumps(payload)}\n\n".encode()


class SyncBody(httpx2.SyncByteStream):
    def __init__(self, event: bytes) -> None:
        self.event = event
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        yield self.event
        yield self.event

    def close(self) -> None:
        self.closed = True


class AsyncBody(httpx2.AsyncByteStream):
    def __init__(self, event: bytes, *, gated: bool = False) -> None:
        self.event = event
        self.gated = gated
        self.read_waiting = asyncio.Event()
        self.close_started = asyncio.Event()
        self.release_close = asyncio.Event()
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.event
        self.read_waiting.set()
        await asyncio.Event().wait()

    async def aclose(self) -> None:
        self.close_started.set()
        if self.gated:
            await self.release_close.wait()
        self.closed = True


def _response(body: Any) -> httpx2.Response:
    return httpx2.Response(
        200, stream=body, headers={"Content-Type": "text/event-stream"}
    )


@pytest.mark.parametrize("primary", [False, True])
def test_sync_explicit_close_closes_http_body(primary: bool) -> None:
    body = SyncBody(_event(primary))
    with httpx2.Client(
        transport=httpx2.MockTransport(lambda _: _response(body))
    ) as client:
        model = GigaChat(
            model="test-model",
            access_token="token",
            use_api_v2=primary,
            http_client=client,
        )
        try:
            stream: Any = model._stream([HumanMessage(content="hello")])
            assert next(stream).text == "ok"
            assert not body.closed
            stream.close()
            assert body.closed
        finally:
            model.close()


@pytest.mark.parametrize("primary", [False, True])
async def test_async_explicit_close_waits_for_http_body(primary: bool) -> None:
    body = AsyncBody(_event(primary), gated=True)
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda _: _response(body))
    ) as client:
        model = GigaChat(
            model="test-model",
            access_token="token",
            use_api_v2=primary,
            async_http_client=client,
        )
        try:
            stream: Any = model._astream([HumanMessage(content="hello")])
            assert (await stream.__anext__()).text == "ok"
            closing = asyncio.create_task(stream.aclose())
            await asyncio.wait_for(body.close_started.wait(), 2)
            assert not closing.done() and not body.closed
            body.release_close.set()
            await asyncio.wait_for(closing, 2)
            assert body.closed
        finally:
            body.release_close.set()
            await model.aclose()


@pytest.mark.parametrize("primary", [False, True])
async def test_cancel_pending_read_closes_http_body(primary: bool) -> None:
    body = AsyncBody(_event(primary))
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda _: _response(body))
    ) as client:
        model = GigaChat(
            model="test-model",
            access_token="token",
            use_api_v2=primary,
            async_http_client=client,
        )
        try:
            stream: Any = model._astream([HumanMessage(content="hello")])
            assert (await stream.__anext__()).text == "ok"
            pending: asyncio.Task[Any] = asyncio.create_task(stream.__anext__())
            await asyncio.wait_for(body.read_waiting.wait(), 2)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert body.closed
        finally:
            await model.aclose()


class SyncCallback:
    def on_llm_new_token(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("callback failed")


class AsyncCallback:
    async def on_llm_new_token(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("callback failed")


@pytest.mark.parametrize("primary", [False, True])
def test_sync_callback_error_closes_http_body(primary: bool) -> None:
    body = SyncBody(_event(primary))
    with httpx2.Client(
        transport=httpx2.MockTransport(lambda _: _response(body))
    ) as client:
        model = GigaChat(
            model="test-model",
            access_token="token",
            use_api_v2=primary,
            http_client=client,
        )
        try:
            stream: Any = model._stream(
                [HumanMessage(content="hello")], run_manager=cast(Any, SyncCallback())
            )
            with pytest.raises(RuntimeError, match="callback failed"):
                next(stream)
            assert body.closed
        finally:
            model.close()


@pytest.mark.parametrize("primary", [False, True])
async def test_async_callback_error_closes_http_body(primary: bool) -> None:
    body = AsyncBody(_event(primary))
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda _: _response(body))
    ) as client:
        model = GigaChat(
            model="test-model",
            access_token="token",
            use_api_v2=primary,
            async_http_client=client,
        )
        try:
            stream: Any = model._astream(
                [HumanMessage(content="hello")], run_manager=cast(Any, AsyncCallback())
            )
            with pytest.raises(RuntimeError, match="callback failed"):
                await stream.__anext__()
            assert body.closed
        finally:
            await model.aclose()
