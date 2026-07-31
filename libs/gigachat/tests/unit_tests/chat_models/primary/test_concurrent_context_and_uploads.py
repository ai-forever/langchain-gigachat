"""Concurrency regressions for shared clients, contexts, and attachment uploads."""

from __future__ import annotations

import asyncio
import copy
import threading
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import gigachat.models as gm
import httpx
import pytest
from gigachat.api import chat_completions as primary_api
from gigachat.context import (
    authorization_cvar,
    custom_headers_cvar,
    request_id_cvar,
    session_id_cvar,
    trace_id_cvar,
)
from langchain_core.messages import HumanMessage
from pytest_mock import MockerFixture

from langchain_gigachat.chat_models.gigachat import (
    DEFAULT_IMAGE_CACHE_MAX_SIZE,
    GigaChat,
)

from .fixtures import CREATED_AT, MESSAGE_ID, MODEL

_DATA_URL = "data:image/png;base64,aGVsbG8="


def _attachment_message() -> HumanMessage:
    return HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {"url": _DATA_URL},
            }
        ]
    )


def _primary_response(text: str = "ok") -> gm.ChatCompletionResponse:
    return gm.ChatCompletionResponse(
        model=MODEL,
        created_at=CREATED_AT,
        messages=[
            gm.ChatMessage(
                role="assistant",
                message_id=MESSAGE_ID,
                content=[gm.ChatContentPart(text=text)],
            )
        ],
        message_id=MESSAGE_ID,
        finish_reason="stop",
    )


def _primary_stream() -> Iterator[dict[str, Any]]:
    yield {
        "event": "response.message.delta",
        "messages": [{"role": "assistant", "content": [{"text": "ok"}]}],
    }
    yield {
        "event": "response.message.done",
        "message_id": MESSAGE_ID,
        "finish_reason": "stop",
    }


async def _async_primary_stream() -> AsyncIterator[dict[str, Any]]:
    for event in _primary_stream():
        await asyncio.sleep(0)
        yield event


def _sdk_request_headers(payload: Any, *, stream: bool) -> dict[str, str]:
    with httpx.Client(base_url="https://example.test/v2") as client:
        if stream:
            kwargs = primary_api._get_stream_kwargs(client, chat=payload)
        else:
            kwargs = primary_api._get_chat_kwargs(client, chat=payload)
    return dict(kwargs["headers"])


@contextmanager
def _sdk_context(label: str) -> Iterator[None]:
    authorization_token = authorization_cvar.set(f"authorization-{label}")
    custom_headers_token = custom_headers_cvar.set(
        {"X-Custom-Context": f"custom-{label}"}
    )
    request_token = request_id_cvar.set(f"request-{label}")
    session_token = session_id_cvar.set(f"session-{label}")
    trace_token = trace_id_cvar.set(f"trace-{label}")
    try:
        yield
    finally:
        trace_id_cvar.reset(trace_token)
        session_id_cvar.reset(session_token)
        request_id_cvar.reset(request_token)
        custom_headers_cvar.reset(custom_headers_token)
        authorization_cvar.reset(authorization_token)


def _assert_sdk_context_headers(headers: dict[str, str], *, label: str) -> None:
    assert headers["Authorization"] == f"authorization-{label}"
    assert headers["X-Custom-Context"] == f"custom-{label}"
    assert headers["X-Request-ID"] == f"request-{label}"
    assert headers["X-Session-ID"] == f"session-{label}"
    assert headers["X-Trace-ID"] == f"trace-{label}"


def test_sync_uploads_deduplicate_in_flight_content_across_threads(
    mocker: MockerFixture,
) -> None:
    model = GigaChat(auto_upload_attachments=True)
    upload_started = threading.Event()
    second_upload_started = threading.Event()
    release_upload = threading.Event()
    calls = 0
    calls_lock = threading.Lock()

    def upload_file(*args: Any, **kwargs: Any) -> SimpleNamespace:
        nonlocal calls
        with calls_lock:
            calls += 1
            if calls == 2:
                second_upload_started.set()
        upload_started.set()
        if not release_upload.wait(timeout=2):
            raise TimeoutError("test did not release the blocking upload")
        return SimpleNamespace(id_="uploaded-file")

    mocker.patch.object(GigaChat, "upload_file", side_effect=upload_file)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            model._upload_attachments,
            [copy.deepcopy(_attachment_message())],
        )
        assert upload_started.wait(timeout=1)
        second = executor.submit(
            model._upload_attachments,
            [copy.deepcopy(_attachment_message())],
        )
        second_upload_started.wait(timeout=0.2)
        release_upload.set()
        first.result(timeout=1)
        second.result(timeout=1)

    assert calls == 1


async def test_async_uploads_deduplicate_in_flight_content_across_tasks(
    mocker: MockerFixture,
) -> None:
    model = GigaChat(auto_upload_attachments=True)
    upload_started = asyncio.Event()
    release_upload = asyncio.Event()
    calls = 0

    async def upload_file(*args: Any, **kwargs: Any) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        upload_started.set()
        await release_upload.wait()
        return SimpleNamespace(id_="uploaded-file")

    mocker.patch.object(GigaChat, "aupload_file", side_effect=upload_file)

    first = asyncio.create_task(
        model._aupload_attachments([copy.deepcopy(_attachment_message())])
    )
    await asyncio.wait_for(upload_started.wait(), timeout=1)
    second = asyncio.create_task(
        model._aupload_attachments([copy.deepcopy(_attachment_message())])
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert calls == 1
    release_upload.set()
    await asyncio.gather(first, second)

    assert calls == 1


def test_failed_upload_cleans_in_flight_state_and_allows_retry(
    mocker: MockerFixture,
) -> None:
    model = GigaChat(auto_upload_attachments=True)
    upload = mocker.patch.object(
        GigaChat,
        "upload_file",
        side_effect=[
            RuntimeError("upload failed"),
            SimpleNamespace(id_="uploaded-after-retry"),
        ],
    )

    with pytest.raises(RuntimeError, match="upload failed"):
        model._upload_attachments([copy.deepcopy(_attachment_message())])

    assert model._uploads_in_flight == {}

    model._upload_attachments([copy.deepcopy(_attachment_message())])

    assert upload.call_count == 2
    assert list(model._cached_uploads.values()) == ["uploaded-after-retry"]


def test_upload_cache_is_deterministically_bounded() -> None:
    model = GigaChat()

    for index in range(DEFAULT_IMAGE_CACHE_MAX_SIZE + 1):
        model._set_cached_upload(f"hash-{index}", f"file-{index}")

    assert len(model._cached_uploads) == DEFAULT_IMAGE_CACHE_MAX_SIZE
    assert "hash-0" not in model._cached_uploads
    assert model._cached_uploads["hash-1000"] == "file-1000"


def test_first_sdk_client_initialization_is_thread_safe(
    mocker: MockerFixture,
) -> None:
    constructor_started = threading.Event()
    second_constructor_started = threading.Event()
    release_constructor = threading.Event()
    calls = 0
    calls_lock = threading.Lock()
    client = MagicMock()

    def create_client(**kwargs: Any) -> MagicMock:
        nonlocal calls
        with calls_lock:
            calls += 1
            if calls == 2:
                second_constructor_started.set()
        constructor_started.set()
        if not release_constructor.wait(timeout=2):
            raise TimeoutError("test did not release the client constructor")
        return client

    mocker.patch("gigachat.GigaChat", side_effect=create_client)
    model = GigaChat()
    start = threading.Barrier(3)

    def get_client() -> object:
        start.wait(timeout=1)
        return model._client

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(get_client)
        second = executor.submit(get_client)
        start.wait(timeout=1)
        assert constructor_started.wait(timeout=1)
        second_constructor_started.wait(timeout=0.2)
        release_constructor.set()
        assert first.result(timeout=1) is client
        assert second.result(timeout=1) is client

    assert calls == 1


def test_real_sdk_context_headers_reach_sync_request_boundaries(
    sdk_client: MagicMock,
) -> None:
    observed: list[dict[str, str]] = []

    def create_response(payload: Any) -> gm.ChatCompletionResponse:
        observed.append(_sdk_request_headers(payload, stream=False))
        return _primary_response()

    def create_stream(payload: Any) -> Iterator[dict[str, Any]]:
        observed.append(_sdk_request_headers(payload, stream=True))
        return _primary_stream()

    sdk_client.chat.create.side_effect = create_response
    sdk_client.chat.stream.side_effect = create_stream
    model = GigaChat(model=MODEL, use_api_v2=True)

    with _sdk_context("sync"):
        model.invoke("Hello")
        list(model.stream("Hello"))

    assert len(observed) == 2
    for headers in observed:
        _assert_sdk_context_headers(headers, label="sync")


async def test_real_sdk_context_headers_reach_async_request_boundaries(
    sdk_client: MagicMock,
) -> None:
    observed: list[dict[str, str]] = []

    async def create_response(payload: Any) -> gm.ChatCompletionResponse:
        observed.append(_sdk_request_headers(payload, stream=False))
        await asyncio.sleep(0)
        return _primary_response()

    sdk_client.achat.create.side_effect = create_response
    sdk_client.achat.stream.side_effect = lambda payload: _capture_async_stream_headers(
        payload,
        observed,
    )
    model = GigaChat(model=MODEL, use_api_v2=True)

    with _sdk_context("async"):
        await model.ainvoke("Hello")
        [chunk async for chunk in model.astream("Hello")]

    assert len(observed) == 2
    for headers in observed:
        _assert_sdk_context_headers(headers, label="async")


def _capture_async_stream_headers(
    payload: Any,
    observed: list[dict[str, str]],
) -> AsyncIterator[dict[str, Any]]:
    observed.append(_sdk_request_headers(payload, stream=True))
    return _async_primary_stream()


@pytest.mark.parametrize("operation", ["ainvoke", "astream"])
async def test_real_sdk_context_headers_are_isolated_between_async_tasks(
    sdk_client: MagicMock,
    operation: str,
) -> None:
    observed: list[dict[str, str]] = []

    async def create_response(payload: Any) -> gm.ChatCompletionResponse:
        observed.append(_sdk_request_headers(payload, stream=False))
        await asyncio.sleep(0)
        observed.append(_sdk_request_headers(payload, stream=False))
        return _primary_response()

    def create_stream(payload: Any) -> AsyncIterator[dict[str, Any]]:
        observed.append(_sdk_request_headers(payload, stream=True))
        return _async_primary_stream()

    sdk_client.achat.create.side_effect = create_response
    sdk_client.achat.stream.side_effect = create_stream
    model = GigaChat(model=MODEL, use_api_v2=True)

    async def invoke(label: str) -> None:
        with _sdk_context(label):
            if operation == "ainvoke":
                await model.ainvoke("Hello")
            else:
                [chunk async for chunk in model.astream("Hello")]

    await asyncio.gather(invoke("a"), invoke("b"))

    expected_observations = 4 if operation == "ainvoke" else 2
    assert len(observed) == expected_observations
    labels = [headers["X-Request-ID"].removeprefix("request-") for headers in observed]
    assert labels.count("a") == expected_observations // 2
    assert labels.count("b") == expected_observations // 2
    for headers, label in zip(observed, labels):
        _assert_sdk_context_headers(headers, label=label)

    assert request_id_cvar.get() is None
    assert session_id_cvar.get() is None
    assert trace_id_cvar.get() is None
    assert custom_headers_cvar.get() is None
    assert authorization_cvar.get() is None
