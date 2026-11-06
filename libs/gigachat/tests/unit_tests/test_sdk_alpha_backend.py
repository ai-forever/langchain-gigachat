"""Exercise the optional SDK backend against a loopback HTTP server."""

import asyncio
import builtins
import json
from typing import Any

import httpx2
import pytest
from gigachat import DefaultAioHttpClient, RequestOptions
from pydantic import ValidationError

from langchain_gigachat import GigaChat, GigaChatEmbeddings


@pytest.mark.parametrize("primary", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
async def test_aiohttp_backend_on_loopback(
    monkeypatch: pytest.MonkeyPatch, primary: bool, streaming: bool
) -> None:
    pytest.importorskip("aiohttp")
    seen: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    async def forbid_native_transport(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("explicit aiohttp backend used native HTTPX2 transport")

    monkeypatch.setattr(
        httpx2.AsyncHTTPTransport, "handle_async_request", forbid_native_transport
    )

    async def respond(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            head = (await reader.readuntil(b"\r\n\r\n")).decode()
            lines = head.split("\r\n")
            headers = dict(line.lower().split(": ", 1) for line in lines[1:] if line)
            body = json.loads(await reader.readexactly(int(headers["content-length"])))
            seen.append((lines[0], headers, body))
            if body.get("stream", False):
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
                wire = f"data: {json.dumps(event)}\n\n"
                if primary:
                    done = {"event": "response.message.done", "messages": []}
                    wire += f"data: {json.dumps(done)}\n\n"
                data = (wire + "data: [DONE]\n\n").encode()
                content_type = "text/event-stream"
            else:
                payload: dict[str, Any] = (
                    {"messages": [{"role": "assistant", "content": "ok"}]}
                    if primary
                    else {
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
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                            "total_tokens": 2,
                        },
                    }
                )
                data = json.dumps(payload).encode()
                content_type = "application/json"
            writer.write(
                (
                    "HTTP/1.1 200 OK\r\n"
                    f"Content-Type: {content_type}\r\n"
                    f"Content-Length: {len(data)}\r\nConnection: close\r\n\r\n"
                ).encode()
                + data
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(respond, "127.0.0.1", 0)
    async with server:
        port = server.sockets[0].getsockname()[1]
        model = GigaChat(
            base_url=f"http://127.0.0.1:{port}/v1/",
            model="test-model",
            access_token="offline-token",
            use_api_v2=primary,
            async_backend="aiohttp",
            default_request_options=RequestOptions(
                extra_headers={"X-Request-ID": "aiohttp-loopback"},
                extra_query=[("probe", "a"), ("probe", "b")],
                timeout=httpx2.Timeout(5.0, write=None),
                max_retries=0,
            ),
        )
        try:
            if streaming:
                chunks = [chunk async for chunk in model.astream("hello")]
                assert "".join(chunk.text for chunk in chunks) == "ok"
            else:
                assert (await model.ainvoke("hello")).text == "ok"
            owner = model._client
            assert (await model.ainvoke("again")).text == "ok"
            assert model._client is owner
        finally:
            await model.aclose()
    assert len(seen) == 2
    expected_route = "/v2/chat/completions" if primary else "/v1/chat/completions"
    assert seen[0][0].startswith(f"POST {expected_route}?")
    assert "probe=a&probe=b" in seen[0][0]
    assert all(headers["x-request-id"] == "aiohttp-loopback" for _, headers, _ in seen)
    assert seen[0][2].get("stream", False) is streaming


async def test_missing_aiohttp_extra_is_lazy_and_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def no_aiohttp(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "aiohttp":
            raise ImportError("offline simulated base installation")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_aiohttp)
    model = GigaChat(
        async_backend="aiohttp", access_token="offline-token", model="test-model"
    )
    assert "_client" not in model.__dict__
    try:
        with pytest.raises(ImportError, match="aiohttp"):
            await model.ainvoke("hello")
    finally:
        await model.aclose()


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"http2": True}, "HTTP/1.1"),
        ({"timeout": httpx2.Timeout(1.0)}, "write timeout"),
        (
            {"limits": httpx2.Limits(max_connections=2, max_keepalive_connections=1)},
            "max_keepalive_connections",
        ),
        ({"proxy": "socks5://127.0.0.1:9"}, "HTTP and HTTPS"),
    ],
)
def test_aiohttp_unsupported_options_are_explicit(
    kwargs: dict[str, Any], message: str
) -> None:
    pytest.importorskip("aiohttp")
    with pytest.raises(ValueError, match=message):
        DefaultAioHttpClient(**kwargs)


async def test_aiohttp_injected_client_is_borrowed_and_raw_session_is_rejected() -> (
    None
):
    aiohttp = pytest.importorskip("aiohttp")
    async with aiohttp.ClientSession() as session:
        for model_class in (GigaChat, GigaChatEmbeddings):
            with pytest.raises(ValidationError):
                model_class(async_http_client=session)
        async with DefaultAioHttpClient(
            timeout=httpx2.Timeout(5.0, write=None),
            limits=httpx2.Limits(max_connections=2, max_keepalive_connections=None),
        ) as client:
            model = GigaChat(async_backend="aiohttp", async_http_client=client)
            assert model.async_http_client is client
            _ = model._client
            await model.aclose()
            assert not client.is_closed
            assert not session.closed
        assert not session.closed
