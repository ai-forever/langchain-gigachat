"""Legacy attachment validation must finish before provider side effects."""

from __future__ import annotations

import copy
import hashlib
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import gigachat.models as gm
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pytest_mock import MockerFixture

from langchain_gigachat.chat_models.gigachat import GigaChat

from .fixtures import CREATED_AT, MODEL

_DATA_URL = "data:image/png;base64,aW1hZ2U="
_DATA_URL_HASH = hashlib.sha256(_DATA_URL.encode()).hexdigest()


def _legacy_response() -> gm.ChatCompletion:
    return gm.ChatCompletion(
        choices=[
            gm.Choices(
                message=gm.Messages(
                    role=gm.MessagesRole.ASSISTANT,
                    content="done",
                ),
                index=0,
                finish_reason="stop",
            )
        ],
        created=CREATED_AT,
        model=MODEL,
        object="chat.completion",
        usage=gm.Usage(
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        ),
    )


def _assert_no_provider_calls(sdk_client: MagicMock) -> None:
    sdk_client.upload_file.assert_not_called()
    sdk_client.aupload_file.assert_not_called()
    sdk_client.chat.assert_not_called()
    sdk_client.chat.create.assert_not_called()
    sdk_client.chat.stream.assert_not_called()
    sdk_client.achat.assert_not_awaited()
    sdk_client.achat.create.assert_not_awaited()
    sdk_client.achat.stream.assert_not_called()


async def _run_public_operation(
    model: GigaChat,
    message: HumanMessage | SystemMessage | AIMessage,
    operation: str,
) -> None:
    if operation == "invoke":
        model.invoke([message])
    elif operation == "ainvoke":
        await model.ainvoke([message])
    elif operation == "stream":
        list(model.stream([message]))
    else:
        assert operation == "astream"
        [chunk async for chunk in model.astream([message])]


@pytest.mark.asyncio
async def test_legacy_additional_attachments_are_deduplicated_without_mutation(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.return_value = _legacy_response()
    sdk_client.achat.return_value = _legacy_response()
    message = HumanMessage(
        content=[
            {"type": "file", "file_id": "file-1"},
            {"type": "image_url", "image_url": {"giga_id": "file-2"}},
        ],
        additional_kwargs={"attachments": ["file-2", "file-3", "file-1", "file-3"]},
    )
    original = copy.deepcopy(message)
    model = GigaChat(model=MODEL)

    model.invoke([message])
    await model.ainvoke([message])

    sync_payload = sdk_client.chat.call_args.args[0]
    async_payload = sdk_client.achat.call_args.args[0]
    assert sync_payload.messages[0].attachments == ["file-1", "file-2", "file-3"]
    assert async_payload.messages[0].attachments == ["file-1", "file-2", "file-3"]
    assert message == original


@pytest.mark.parametrize(
    "attachments",
    ["file-1", b"file-1", ["file-1", ""], ["file-1", "   "], ["file-1", 7]],
)
@pytest.mark.parametrize("operation", ["invoke", "ainvoke"])
@pytest.mark.asyncio
async def test_invalid_legacy_additional_attachments_fail_before_network(
    sdk_client: MagicMock,
    attachments: Any,
    operation: str,
) -> None:
    message = HumanMessage(
        content="describe",
        additional_kwargs={"attachments": attachments},
    )
    original = copy.deepcopy(message)

    with pytest.raises(ValueError, match="additional_kwargs.*attachments"):
        await _run_public_operation(GigaChat(model=MODEL), message, operation)

    assert message == original
    _assert_no_provider_calls(sdk_client)


@pytest.mark.parametrize("use_api_v2", [False, True])
@pytest.mark.parametrize("operation", ["invoke", "ainvoke", "stream", "astream"])
@pytest.mark.asyncio
async def test_malformed_data_url_fails_before_network_on_both_routes(
    sdk_client: MagicMock,
    use_api_v2: bool,
    operation: str,
) -> None:
    message = HumanMessage(
        content=[
            {"type": "image_url", "image_url": {"url": _DATA_URL}},
            {
                "type": "document_url",
                "document_url": {"url": "data:application/pdf;base64,not-valid-%%%"},
            },
        ]
    )
    original = copy.deepcopy(message)
    model = GigaChat(
        model=MODEL,
        use_api_v2=use_api_v2,
        auto_upload_attachments=True,
    )

    with pytest.raises(ValueError, match="Invalid base64 data URL attachment"):
        await _run_public_operation(model, message, operation)

    assert message == original
    _assert_no_provider_calls(sdk_client)


@pytest.mark.parametrize("source", ["tool_calls", "additional_kwargs"])
@pytest.mark.parametrize("operation", ["invoke", "ainvoke", "stream", "astream"])
@pytest.mark.asyncio
async def test_primary_function_call_without_state_fails_before_upload_or_network(
    sdk_client: MagicMock,
    source: str,
    operation: str,
) -> None:
    message_kwargs: dict[str, Any] = {
        "content": [{"type": "image_url", "image_url": {"url": _DATA_URL}}]
    }
    if source == "tool_calls":
        message_kwargs["tool_calls"] = [
            {
                "name": "lookup",
                "args": {"key": "value"},
                "id": None,
                "type": "tool_call",
            }
        ]
    else:
        message_kwargs["additional_kwargs"] = {
            "function_call": {
                "name": "lookup",
                "arguments": {"key": "value"},
            }
        }
    message = AIMessage(**message_kwargs)
    original = copy.deepcopy(message)
    model = GigaChat(
        model=MODEL,
        use_api_v2=True,
        auto_upload_attachments=True,
    )

    with pytest.raises(
        ValueError,
        match="function call is missing provider tools_state_id",
    ):
        await _run_public_operation(model, message, operation)

    assert message == original
    _assert_no_provider_calls(sdk_client)


@pytest.mark.parametrize("message_type", [SystemMessage, AIMessage])
@pytest.mark.parametrize("operation", ["invoke", "ainvoke", "stream", "astream"])
@pytest.mark.asyncio
async def test_legacy_rejects_unsupported_attachment_roles_before_upload(
    sdk_client: MagicMock,
    message_type: type[SystemMessage] | type[AIMessage],
    operation: str,
) -> None:
    message = message_type(
        content=[{"type": "image_url", "image_url": {"url": _DATA_URL}}]
    )
    original = copy.deepcopy(message)
    model = GigaChat(model=MODEL, auto_upload_attachments=True)

    with pytest.raises(ValueError, match="attachments only on HumanMessage"):
        await _run_public_operation(model, message, operation)

    assert message == original
    _assert_no_provider_calls(sdk_client)


def test_legacy_sync_request_keeps_uploaded_id_after_cache_eviction(
    sdk_client: MagicMock,
    mocker: MockerFixture,
) -> None:
    sdk_client.upload_file.return_value = SimpleNamespace(id_="request-file")
    sdk_client.chat.return_value = _legacy_response()
    model = GigaChat(model=MODEL, auto_upload_attachments=True)
    original_upload = GigaChat._upload_attachments

    def upload_then_evict(
        current_model: GigaChat,
        messages: list[Any],
    ) -> dict[str, str]:
        request_uploads = original_upload(current_model, messages)
        with current_model._upload_cache_lock:
            current_model._cached_uploads.pop(_DATA_URL_HASH)
        return request_uploads

    mocker.patch.object(GigaChat, "_upload_attachments", new=upload_then_evict)

    model.invoke(
        [HumanMessage(content=[{"type": "image_url", "image_url": {"url": _DATA_URL}}])]
    )

    payload = sdk_client.chat.call_args.args[0]
    assert payload.messages[0].attachments == ["request-file"]


@pytest.mark.asyncio
async def test_legacy_async_request_keeps_uploaded_id_after_cache_eviction(
    sdk_client: MagicMock,
    mocker: MockerFixture,
) -> None:
    sdk_client.aupload_file = AsyncMock(
        return_value=SimpleNamespace(id_="request-file")
    )
    sdk_client.achat.return_value = _legacy_response()
    model = GigaChat(model=MODEL, auto_upload_attachments=True)
    original_upload = GigaChat._aupload_attachments

    async def upload_then_evict(
        current_model: GigaChat,
        messages: list[Any],
    ) -> dict[str, str]:
        request_uploads = await original_upload(current_model, messages)
        with current_model._upload_cache_lock:
            current_model._cached_uploads.pop(_DATA_URL_HASH)
        return request_uploads

    mocker.patch.object(GigaChat, "_aupload_attachments", new=upload_then_evict)

    await model.ainvoke(
        [HumanMessage(content=[{"type": "image_url", "image_url": {"url": _DATA_URL}}])]
    )

    payload = sdk_client.achat.call_args.args[0]
    assert payload.messages[0].attachments == ["request-file"]
