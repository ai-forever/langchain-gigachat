"""Request option isolation and wrapper-owned stream cleanup."""

import asyncio
import pickle
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier
from typing import Any

import httpx2 as httpx
import pytest
from gigachat import RequestOptions
from gigachat.context import session_id_cvar
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage

from langchain_gigachat import GigaChat, GigaChatEmbeddings
from langchain_gigachat._client import _GigaChatClientMixin


def test_cache_fingerprint_is_canonical_and_secret_free() -> None:
    a = GigaChat(
        default_request_options=RequestOptions(extra_headers={"a": "secret", "b": "2"})
    )
    b = GigaChat(
        default_request_options=RequestOptions(extra_headers={"b": "2", "a": "secret"})
    )
    c = GigaChat(
        default_request_options=RequestOptions(
            extra_headers={"a": "different", "b": "2"}
        )
    )
    assert a._get_llm_string() == b._get_llm_string()
    assert a._get_llm_string() != c._get_llm_string()
    assert "secret" not in a._get_llm_string()
    assert "request-options-sha256" not in GigaChat()._get_llm_string()
    assert (
        GigaChat(
            default_request_options=RequestOptions(extra_body={"foo": 1})
        )._get_llm_string()
        != GigaChat(
            default_request_options=RequestOptions(extra_body={"foo": 2})
        )._get_llm_string()
    )


@pytest.mark.parametrize("primary", [False, True])
@pytest.mark.parametrize("method", ["invoke", "ainvoke", "stream", "astream"])
async def test_per_call_options_rejected_before_callbacks(
    primary: bool, method: str
) -> None:
    class Recorder(BaseCallbackHandler):
        calls = 0

        def on_chat_model_start(self, *args: Any, **kwargs: Any) -> None:
            self.calls += 1

        def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
            self.calls += 1

    callback = Recorder()
    model = GigaChat(use_api_v2=primary, callbacks=[callback])
    options = RequestOptions(extra_headers={"X-Secret": "secret"})
    with pytest.raises(ValueError, match="Per-call request_options"):
        if method == "invoke":
            model.invoke("hello", request_options=options)
        elif method == "ainvoke":
            await model.ainvoke("hello", request_options=options)
        elif method == "stream":
            list(model.stream("hello", request_options=options))
        else:
            _ = [part async for part in model.astream("hello", request_options=options)]
    assert callback.calls == 0
    assert "_client" not in model.__dict__


async def test_private_astream_close_waits_for_sdk_cleanup(mocker: Any) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    closed = asyncio.Event()

    class GatedStream:
        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *args: Any) -> None:
            entered.set()
            await release.wait()
            closed.set()

        def __aiter__(self) -> Any:
            return self

        async def __anext__(self) -> Any:
            return {
                "model": "test",
                "choices": [{"delta": {"content": "hello"}, "index": 0}],
            }

    sdk = mocker.patch("gigachat.GigaChat").return_value
    sdk.astream.return_value = GatedStream()
    stream = GigaChat()._astream([HumanMessage(content="hello")])
    assert (await stream.__anext__()).text == "hello"
    closing = asyncio.create_task(getattr(stream, "aclose")())
    await asyncio.wait_for(entered.wait(), 1)
    assert not closing.done()
    release.set()
    await asyncio.wait_for(closing, 1)
    assert closed.is_set()


async def test_concurrent_models_share_borrowed_pool_and_keep_context() -> None:
    seen: dict[str, str] = {}

    async def respond(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0)
        seen[request.headers["X-Session-ID"]] = request.headers["X-Model-Option"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "index": 0,
                        "finish_reason": "stop",
                    }
                ],
                "created": 1,
                "model": "test",
                "object": "chat.completion",
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    models = [
        GigaChat(
            model="test-model",
            access_token="token",
            async_http_client=client,
            default_request_options=RequestOptions(
                extra_headers={"X-Model-Option": value}
            ),
        )
        for value in ("a", "b")
    ]

    async def run(index: int) -> None:
        token = session_id_cvar.set(f"session-{index}")
        try:
            assert (await models[index].ainvoke("hello")).content == "ok"
            assert session_id_cvar.get() == f"session-{index}"
        finally:
            session_id_cvar.reset(token)

    await asyncio.gather(run(0), run(1))
    assert seen == {"session-0": "a", "session-1": "b"}
    for model in models:
        await model.aclose()
    assert not client.is_closed
    await client.aclose()


async def test_callback_diagnostics_do_not_contain_default_secret() -> None:
    class Recorder(BaseCallbackHandler):
        records: list[str]

        def __init__(self) -> None:
            self.records = []

        def on_chat_model_start(self, *args: Any, **kwargs: Any) -> None:
            self.records.append(repr((args, kwargs)))

    secret = "synthetic-callback-secret"
    callback = Recorder()

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "index": 0,
                        "finish_reason": "stop",
                    }
                ],
                "created": 1,
                "model": "test",
                "object": "chat.completion",
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    sync = httpx.Client(transport=httpx.MockTransport(respond))
    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    model = GigaChat(
        model="test-model",
        access_token="token",
        callbacks=[callback],
        http_client=sync,
        async_http_client=client,
        default_request_options=RequestOptions(extra_headers={"X-Secret": secret}),
    )
    model.invoke("hello")
    await model.ainvoke("hello")
    assert callback.records
    assert secret not in repr(callback.records)
    sync.close()
    await client.aclose()


@pytest.mark.parametrize("cancel", [False, True])
async def test_context_preserved_after_error_or_cancel(cancel: bool) -> None:
    started = asyncio.Event()

    async def respond(request: httpx.Request) -> httpx.Response:
        started.set()
        if cancel:
            await asyncio.Event().wait()
        raise httpx.ConnectError("offline failure", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    model = GigaChat(
        model="test-model",
        access_token="token",
        async_http_client=client,
        max_retries=0,
    )

    async def run() -> None:
        token = session_id_cvar.set("error-context")
        try:
            try:
                await model.ainvoke("hello")
            except (Exception, asyncio.CancelledError):
                assert session_id_cvar.get() == "error-context"
                raise
        finally:
            session_id_cvar.reset(token)

    task = asyncio.create_task(run())
    await asyncio.wait_for(started.wait(), 1)
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else Exception):
        await task
    await model.aclose()
    await client.aclose()


@pytest.mark.parametrize("use_async_close", [False, True])
async def test_owned_clients_follow_close_lifecycle(use_async_close: bool) -> None:
    model = GigaChat(model="test-model")
    owner = model._client
    sync_api = owner._client
    sync_auth = owner._auth_client
    async_api = owner._aclient
    async_auth = owner._auth_aclient
    if use_async_close:
        await model.aclose()
        assert async_api.is_closed and async_auth.is_closed
    else:
        model.close()
        assert not async_api.is_closed and not async_auth.is_closed
        await model.aclose()
    assert sync_api.is_closed and sync_auth.is_closed
    assert async_api.is_closed and async_auth.is_closed
    assert model._client is owner


async def test_public_astream_close_does_not_guarantee_gated_cleanup(
    mocker: Any,
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    closed = asyncio.Event()

    class GatedStream:
        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *args: Any) -> None:
            entered.set()
            await release.wait()
            closed.set()

        def __aiter__(self) -> Any:
            return self

        async def __anext__(self) -> Any:
            return {
                "model": "test",
                "choices": [{"delta": {"content": "hello"}, "index": 0}],
            }

    sdk = mocker.patch("gigachat.GigaChat").return_value
    sdk.astream.return_value = GatedStream()
    stream = GigaChat().astream("hello")
    assert (await stream.__anext__()).text == "hello"
    closing = asyncio.create_task(getattr(stream, "aclose")())
    await asyncio.sleep(0)
    # Core may return before its nested generator's __aexit__ finishes.
    if closing.done():
        assert not closed.is_set()
    release.set()
    await asyncio.wait_for(closing, 1)
    await asyncio.wait_for(entered.wait(), 1)
    await asyncio.wait_for(closed.wait(), 1)


def test_cache_fingerprint_excludes_transport_and_preserves_query_order() -> None:
    query = [("tag", "a"), ("tag", "b")]
    first = GigaChat(
        default_request_options=RequestOptions(
            extra_query=query, timeout=1, max_retries=0
        )
    )
    second = GigaChat(
        default_request_options=RequestOptions(
            extra_query=query, timeout=20, max_retries=3
        )
    )
    reversed_query = GigaChat(
        default_request_options=RequestOptions(extra_query=list(reversed(query)))
    )
    assert first._get_llm_string() == second._get_llm_string()
    assert first._get_llm_string() != reversed_query._get_llm_string()


def test_real_cache_separates_constructor_defaults() -> None:
    from langchain_core.caches import InMemoryCache

    cache = InMemoryCache()
    seen: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        value = request.headers["X-Variant"]
        seen.append(value)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": value},
                        "index": 0,
                        "finish_reason": "stop",
                    }
                ],
                "created": 1,
                "model": "test",
                "object": "chat.completion",
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(respond))
    models = [
        GigaChat(
            model="test",
            access_token="token",
            cache=cache,
            http_client=client,
            default_request_options=RequestOptions(extra_headers={"X-Variant": value}),
        )
        for value in ("a", "b")
    ]
    assert models[0].invoke("hello").content == "a"
    assert models[1].invoke("hello").content == "b"
    assert models[0].invoke("hello").content == "a"
    assert seen == ["a", "b"]
    for model in models:
        model.close()
    client.close()


async def test_abatch_and_bind_reuse_owner_with_context() -> None:
    seen: list[tuple[str, str]] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0)
        seen.append((request.headers["X-Session-ID"], request.headers["X-Variant"]))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "index": 0,
                        "finish_reason": "stop",
                    }
                ],
                "created": 1,
                "model": "test",
                "object": "chat.completion",
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    models = [
        GigaChat(
            model="test",
            access_token="token",
            async_http_client=client,
            default_request_options=RequestOptions(extra_headers={"X-Variant": value}),
        )
        for value in ("a", "b")
    ]

    async def run(index: int) -> None:
        token = session_id_cvar.set(f"batch-{index}")
        try:
            model = models[index]
            owner = model._client
            bound = model.bind(temperature=0.5)
            assert [
                result.content for result in await bound.abatch(["one", "two"])
            ] == ["ok", "ok"]
            assert model._client is owner
            assert session_id_cvar.get() == f"batch-{index}"
        finally:
            session_id_cvar.reset(token)

    await asyncio.gather(run(0), run(1))
    assert sorted(seen) == [("batch-0", "a")] * 2 + [("batch-1", "b")] * 2
    for model in models:
        await model.aclose()
    assert not client.is_closed
    await client.aclose()


async def test_stream_callbacks_exclude_constructor_options() -> None:
    import json

    class Recorder(BaseCallbackHandler):
        def __init__(self) -> None:
            self.starts: list[str] = []
            self.tokens: list[str] = []

        def on_chat_model_start(self, *args: Any, **kwargs: Any) -> None:
            self.starts.append(repr((args, kwargs)))

        def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
            self.tokens.append(token)
            self.starts.append(repr(kwargs))

    event = {
        "choices": [{"delta": {"content": "hello"}, "index": 0}],
        "created": 1,
        "model": "test",
        "object": "chat.completion.chunk",
    }
    body = f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n"

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, text=body, headers={"content-type": "text/event-stream"}
        )

    sync = httpx.Client(transport=httpx.MockTransport(respond))
    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    callback = Recorder()
    secret = "synthetic-stream-secret"
    model = GigaChat(
        model="test",
        access_token="token",
        callbacks=[callback],
        http_client=sync,
        async_http_client=client,
        default_request_options=RequestOptions(extra_headers={"X-Secret": secret}),
    )
    assert "".join(chunk.text for chunk in model.stream("hello")) == "hello"
    assert "".join([chunk.text async for chunk in model.astream("hello")]) == "hello"
    assert "".join(callback.tokens) == "hellohello"
    assert len(callback.starts) >= 4
    assert secret not in repr(callback.starts)
    await model.aclose()
    sync.close()
    await client.aclose()


@pytest.mark.parametrize("model_class", [GigaChat, GigaChatEmbeddings])
def test_concurrent_cold_client_getter_creates_one_owner(
    model_class: Any, mocker: Any
) -> None:
    model = model_class()
    start = Barrier(8)
    constructor = mocker.patch(
        "gigachat.GigaChat", side_effect=lambda **kw: mocker.Mock()
    )
    # Exercise the getter directly so Python 3.10's cached_property lock cannot
    # hide missing synchronization in the getter on newer interpreters.
    getter = _GigaChatClientMixin.__dict__["_client"].func

    def get_owner() -> Any:
        start.wait(timeout=5)
        return getter(model)

    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(get_owner) for _ in range(8)]
            owners = [future.result(timeout=5) for future in futures]
        assert constructor.call_count == 1
        assert all(owner is owners[0] for owner in owners)
        assert model._client is owners[0]
    finally:
        model.close()


def test_cold_public_batch_creates_one_owner(mocker: Any) -> None:
    import gigachat

    start = Barrier(8)

    class SynchronizedStart(BaseCallbackHandler):
        def on_chat_model_start(self, *args: Any, **kwargs: Any) -> None:
            start.wait(timeout=5)

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "index": 0,
                        "finish_reason": "stop",
                    }
                ],
                "created": 1,
                "model": "test",
                "object": "chat.completion",
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(respond))
    constructor = mocker.patch("gigachat.GigaChat", wraps=gigachat.GigaChat)
    model = GigaChat(
        access_token="token",
        model="test",
        http_client=client,
        callbacks=[SynchronizedStart()],
    )
    try:
        assert "_client" not in model.__dict__
        results = model.batch(["hello"] * 8, config={"max_concurrency": 8})
        assert [result.content for result in results] == ["ok"] * 8
        assert constructor.call_count == 1
    finally:
        model.close()
        client.close()


@pytest.mark.parametrize("model_class", [GigaChat, GigaChatEmbeddings])
def test_models_have_distinct_sdk_owners(model_class: Any) -> None:
    models = [model_class(), model_class()]
    try:
        assert models[0]._client is not models[1]._client
    finally:
        for model in models:
            model.close()


@pytest.mark.parametrize("model_class", [GigaChat, GigaChatEmbeddings])
@pytest.mark.parametrize("copy_method", ["deepcopy", "model_copy", "pickle"])
def test_cold_models_remain_copyable(model_class: Any, copy_method: str) -> None:
    model = model_class(model="test", access_token="token")
    if copy_method == "deepcopy":
        copied = deepcopy(model)
    elif copy_method == "model_copy":
        copied = model.model_copy(deep=True)
    else:
        copied = pickle.loads(pickle.dumps(model))
    assert type(copied) is model_class
    assert copied is not model
    assert copied.model_dump() == model.model_dump()
    assert "_client" not in model.__dict__
    assert "_client" not in copied.__dict__


@pytest.mark.parametrize("model_class", [GigaChat, GigaChatEmbeddings])
def test_failed_client_constructor_can_retry(model_class: Any, mocker: Any) -> None:
    model = model_class()
    owner = mocker.Mock()
    constructor = mocker.patch(
        "gigachat.GigaChat", side_effect=[ValueError("first attempt"), owner]
    )
    try:
        with pytest.raises(ValueError, match="first attempt"):
            _ = model._client
        assert "_client" not in model.__dict__
        # A different thread must be able to acquire the initialization lock.
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(lambda: model._client).result(timeout=5) is owner
        assert model._client is owner
        assert constructor.call_count == 2
    finally:
        model.close()
