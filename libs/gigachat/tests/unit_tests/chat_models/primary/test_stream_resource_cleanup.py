"""Resource-cleanup regressions for wrapper-managed provider streams."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import gigachat.models as gm
import pytest
from langchain_core.messages import HumanMessage

from langchain_gigachat.chat_models.gigachat import GigaChat

from .fixtures import CREATED_AT, MODEL


def _primary_delta() -> dict[str, Any]:
    return {
        "event": "response.message.delta",
        "model": MODEL,
        "created_at": CREATED_AT,
        "messages": [
            {
                "role": "assistant",
                "content": [{"text": "partial"}],
            }
        ],
    }


def _legacy_delta() -> dict[str, Any]:
    return {
        "choices": [
            {
                "delta": {"role": "assistant", "content": "partial"},
                "index": 0,
                "finish_reason": None,
            }
        ],
        "created": CREATED_AT,
        "model": MODEL,
        "object": "chat.completion.chunk",
    }


def _event_for_route(
    use_api_v2: bool,
) -> gm.PrimaryChatCompletionChunk | dict[str, Any]:
    if use_api_v2:
        return gm.PrimaryChatCompletionChunk.model_validate(_primary_delta())
    return _legacy_delta()


class TrackingIterator(Iterator[Any]):
    def __init__(
        self,
        items: list[Any],
        *,
        error_after_items: BaseException | None = None,
    ) -> None:
        self._items = iter(items)
        self._error_after_items = error_after_items
        self.closed = False

    def __next__(self) -> Any:
        try:
            return next(self._items)
        except StopIteration:
            if self._error_after_items is not None:
                error = self._error_after_items
                self._error_after_items = None
                raise error
            raise

    def close(self) -> None:
        self.closed = True


class BlockingAsyncIterator(AsyncIterator[dict[str, Any]]):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.closed = False

    async def __anext__(self) -> dict[str, Any]:
        self.started.set()
        await asyncio.Future()
        raise AssertionError("unreachable")

    async def aclose(self) -> None:
        self.closed = True


def _set_sync_stream(
    sdk_client: MagicMock,
    *,
    use_api_v2: bool,
    provider: TrackingIterator,
) -> None:
    if use_api_v2:
        sdk_client.chat.stream.return_value = provider
    else:
        sdk_client.stream.return_value = provider


def _set_async_stream(
    sdk_client: MagicMock,
    *,
    use_api_v2: bool,
    provider: BlockingAsyncIterator,
) -> None:
    if use_api_v2:
        sdk_client.achat.stream.return_value = provider
    else:
        sdk_client.astream.return_value = provider


@pytest.mark.parametrize("use_api_v2", [False, True])
def test_consumer_close_closes_provider_without_synthetic_terminal(
    sdk_client: MagicMock,
    use_api_v2: bool,
) -> None:
    provider = TrackingIterator([_event_for_route(use_api_v2)])
    _set_sync_stream(
        sdk_client,
        use_api_v2=use_api_v2,
        provider=provider,
    )
    stream = GigaChat(model=MODEL, use_api_v2=use_api_v2)._stream(
        [HumanMessage("Hello")],
    )

    first = next(stream)
    close = getattr(stream, "close")
    close()

    assert first.text == "partial"
    assert provider.closed is True


@pytest.mark.parametrize("use_api_v2", [False, True])
def test_callback_exception_closes_provider_and_clears_upload_state(
    sdk_client: MagicMock,
    use_api_v2: bool,
) -> None:
    provider = TrackingIterator([_event_for_route(use_api_v2)])
    _set_sync_stream(
        sdk_client,
        use_api_v2=use_api_v2,
        provider=provider,
    )
    sdk_client.upload_file.return_value = SimpleNamespace(id_="uploaded")
    run_manager = MagicMock()
    run_manager.on_llm_new_token.side_effect = RuntimeError("callback failed")
    llm = GigaChat(
        model=MODEL,
        use_api_v2=use_api_v2,
        auto_upload_attachments=True,
    )
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Describe"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,aW1hZ2U="},
            },
        ]
    )

    with pytest.raises(RuntimeError, match="callback failed"):
        list(
            llm._stream(
                [message],
                run_manager=run_manager,
            )
        )

    assert provider.closed is True
    assert llm._uploads_in_flight == {}


@pytest.mark.parametrize("use_api_v2", [False, True])
def test_provider_exception_closes_stream(
    sdk_client: MagicMock,
    use_api_v2: bool,
) -> None:
    provider = TrackingIterator(
        [_event_for_route(use_api_v2)],
        error_after_items=RuntimeError("provider failed"),
    )
    _set_sync_stream(
        sdk_client,
        use_api_v2=use_api_v2,
        provider=provider,
    )
    llm = GigaChat(model=MODEL, use_api_v2=use_api_v2)

    with pytest.raises(RuntimeError, match="provider failed"):
        list(llm._stream([HumanMessage("Hello")]))

    assert provider.closed is True


@pytest.mark.parametrize("use_api_v2", [False, True])
async def test_async_consumer_cancellation_closes_provider(
    sdk_client: MagicMock,
    use_api_v2: bool,
) -> None:
    provider = BlockingAsyncIterator()
    _set_async_stream(
        sdk_client,
        use_api_v2=use_api_v2,
        provider=provider,
    )
    stream = GigaChat(model=MODEL, use_api_v2=use_api_v2)._astream(
        [HumanMessage("Hello")]
    )
    pending: asyncio.Future[Any] = asyncio.ensure_future(anext(stream))
    await provider.started.wait()

    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending

    assert provider.closed is True
