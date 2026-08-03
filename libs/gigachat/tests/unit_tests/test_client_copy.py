from __future__ import annotations

import copy
import ssl
import threading
from types import SimpleNamespace
from typing import Any, Callable

import pytest
from langchain_core.rate_limiters import InMemoryRateLimiter
from pytest_mock import MockerFixture

from langchain_gigachat.chat_models.gigachat import GigaChat
from langchain_gigachat.embeddings.gigachat import GigaChatEmbeddings


def _deepcopy(model: Any) -> Any:
    return copy.deepcopy(model)


def _model_copy(model: Any) -> Any:
    return model.model_copy(deep=True)


@pytest.fixture()
def sdk_clients(mocker: MockerFixture) -> list[Any]:
    clients: list[Any] = []

    def create_client(**kwargs: Any) -> Any:
        client = SimpleNamespace(lock=threading.RLock(), kwargs=kwargs)
        clients.append(client)
        return client

    mocker.patch("gigachat.GigaChat", side_effect=create_client)
    return clients


@pytest.mark.parametrize("copy_model", [_deepcopy, _model_copy])
def test_gigachat_deep_copy_resets_owned_runtime(
    copy_model: Callable[[Any], Any],
    sdk_clients: list[Any],
) -> None:
    ssl_context = ssl.create_default_context()
    rate_limiter = InMemoryRateLimiter()
    model = GigaChat(
        access_token="token",
        flags=["flag"],
        rate_limiter=rate_limiter,
        ssl_context=ssl_context,
    )
    _, completed, _ = model._claim_upload("cached-hash")
    model._complete_upload("cached-hash", completed, "cached-file")
    model._claim_upload("pending-hash")
    original_client = model._client

    copied = copy_model(model)

    assert copied.flags == model.flags
    assert copied.flags is not model.flags
    assert copied.ssl_context is ssl_context
    assert copied.rate_limiter is rate_limiter
    assert copied._cached_uploads == model._cached_uploads
    assert copied._cached_uploads is not model._cached_uploads
    assert copied._upload_cache_lock is not model._upload_cache_lock
    assert copied._uploads_in_flight == {}
    assert copied._uploads_in_flight is not model._uploads_in_flight
    assert copied._client_instance is None

    assert model._client_instance is original_client
    assert "pending-hash" in model._uploads_in_flight
    assert copied._client is not original_client
    assert len(sdk_clients) == 2


@pytest.mark.parametrize("copy_model", [_deepcopy, _model_copy])
def test_embeddings_deep_copy_resets_initialized_client(
    copy_model: Callable[[Any], Any],
    sdk_clients: list[Any],
) -> None:
    ssl_context = ssl.create_default_context()
    model = GigaChatEmbeddings(
        access_token="token",
        ssl_context=ssl_context,
        use_prefix_query=True,
    )
    original_client = model._client

    copied = copy_model(model)

    assert copied.use_prefix_query is True
    assert copied.ssl_context is ssl_context
    assert copied._client_instance is None
    assert model._client_instance is original_client
    assert copied._client is not original_client
    assert len(sdk_clients) == 2
