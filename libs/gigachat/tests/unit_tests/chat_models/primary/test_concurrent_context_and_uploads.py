"""Concurrency regressions for shared clients, contexts, and attachment uploads."""

from __future__ import annotations

import asyncio
import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import gigachat.models as gm
import pytest
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


async def test_async_request_context_is_isolated_between_tasks(
    sdk_client: MagicMock,
) -> None:
    request_context: ContextVar[str] = ContextVar("request_context")
    observed: list[str] = []

    async def create_response(payload: Any) -> gm.ChatCompletionResponse:
        observed.append(request_context.get())
        await asyncio.sleep(0)
        observed.append(request_context.get())
        return _primary_response(request_context.get())

    sdk_client.achat.create.side_effect = create_response
    model = GigaChat(model=MODEL, use_api_v2=True)

    async def invoke(value: str) -> str:
        token = request_context.set(value)
        try:
            response = await model.ainvoke("Hello")
            return str(response.content)
        finally:
            request_context.reset(token)

    results = await asyncio.gather(invoke("request-a"), invoke("request-b"))

    assert results == ["request-a", "request-b"]
    assert observed.count("request-a") == 2
    assert observed.count("request-b") == 2
