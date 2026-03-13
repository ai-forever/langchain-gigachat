from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from langchain_gigachat.chat_models.gigachat import GigaChat


@pytest.fixture()
def sdk_mock(mocker: MockerFixture) -> MagicMock:
    mock = mocker.Mock()
    mocker.patch("gigachat.GigaChat", return_value=mock)
    return mock


@pytest.fixture()
def async_sdk_mock(mocker: MockerFixture) -> MagicMock:
    mock = mocker.Mock()
    mocker.patch("gigachat.GigaChat", return_value=mock)
    return mock


# ---------------------------------------------------------------------------
# get_num_tokens
# ---------------------------------------------------------------------------


def test_get_num_tokens_heuristic() -> None:
    llm = GigaChat()
    result = llm.get_num_tokens("hello world")
    assert result == round(len("hello world") / 4.6)


def test_get_num_tokens_api(sdk_mock: MagicMock) -> None:
    token_count = MagicMock()
    token_count.tokens = 42
    sdk_mock.tokens_count.return_value = [token_count]
    llm = GigaChat(use_api_for_tokens=True)
    result = llm.get_num_tokens("hello world")
    assert result == 42
    sdk_mock.tokens_count.assert_called_once_with(["hello world"], None)


# ---------------------------------------------------------------------------
# tokens_count / atokens_count
# ---------------------------------------------------------------------------


def test_tokens_count(sdk_mock: MagicMock) -> None:
    sentinel = MagicMock()
    sdk_mock.tokens_count.return_value = sentinel
    llm = GigaChat()
    result = llm.tokens_count(["a", "b"], model="test")
    assert result is sentinel
    sdk_mock.tokens_count.assert_called_once_with(["a", "b"], "test")


@pytest.mark.asyncio()
async def test_atokens_count(async_sdk_mock: MagicMock) -> None:
    sentinel = MagicMock()
    async_sdk_mock.atokens_count = _make_coro(sentinel)
    llm = GigaChat()
    result = await llm.atokens_count(["a", "b"], model="test")
    assert result is sentinel


# ---------------------------------------------------------------------------
# get_models / get_model
# ---------------------------------------------------------------------------


def test_get_models(sdk_mock: MagicMock) -> None:
    sentinel = MagicMock()
    sdk_mock.get_models.return_value = sentinel
    llm = GigaChat()
    assert llm.get_models() is sentinel


@pytest.mark.asyncio()
async def test_aget_models(async_sdk_mock: MagicMock) -> None:
    sentinel = MagicMock()
    async_sdk_mock.aget_models = _make_coro(sentinel)
    llm = GigaChat()
    assert await llm.aget_models() is sentinel


def test_get_model(sdk_mock: MagicMock) -> None:
    sentinel = MagicMock()
    sdk_mock.get_model.return_value = sentinel
    llm = GigaChat()
    assert llm.get_model("GigaChat") is sentinel
    sdk_mock.get_model.assert_called_once_with("GigaChat")


@pytest.mark.asyncio()
async def test_aget_model(async_sdk_mock: MagicMock) -> None:
    sentinel = MagicMock()
    async_sdk_mock.aget_model = _make_coro(sentinel)
    llm = GigaChat()
    assert await llm.aget_model("GigaChat") is sentinel


# ---------------------------------------------------------------------------
# File operations
# ---------------------------------------------------------------------------


def test_upload_file(sdk_mock: MagicMock) -> None:
    sentinel = MagicMock()
    sdk_mock.upload_file.return_value = sentinel
    llm = GigaChat()
    result = llm.upload_file(("test.txt", b"data"), purpose="general")
    assert result is sentinel


@pytest.mark.asyncio()
async def test_aupload_file(async_sdk_mock: MagicMock) -> None:
    sentinel = MagicMock()
    async_sdk_mock.aupload_file = _make_coro(sentinel)
    llm = GigaChat()
    result = await llm.aupload_file(("test.txt", b"data"))
    assert result is sentinel


def test_get_file(sdk_mock: MagicMock) -> None:
    sentinel = MagicMock()
    sdk_mock.get_file.return_value = sentinel
    llm = GigaChat()
    assert llm.get_file("file-1") is sentinel
    sdk_mock.get_file.assert_called_once_with("file-1")


@pytest.mark.asyncio()
async def test_aget_file(async_sdk_mock: MagicMock) -> None:
    sentinel = MagicMock()
    async_sdk_mock.aget_file = _make_coro(sentinel)
    llm = GigaChat()
    assert await llm.aget_file("file-1") is sentinel


def test_get_file_content(sdk_mock: MagicMock) -> None:
    sentinel = MagicMock()
    sdk_mock.get_image.return_value = sentinel
    llm = GigaChat()
    assert llm.get_file_content("file-1") is sentinel
    sdk_mock.get_image.assert_called_once_with("file-1")


@pytest.mark.asyncio()
async def test_aget_file_content(async_sdk_mock: MagicMock) -> None:
    sentinel = MagicMock()
    async_sdk_mock.aget_image = _make_coro(sentinel)
    llm = GigaChat()
    assert await llm.aget_file_content("file-1") is sentinel


def test_list_files(sdk_mock: MagicMock) -> None:
    sentinel = MagicMock()
    sdk_mock.get_files.return_value = sentinel
    llm = GigaChat()
    assert llm.list_files() is sentinel


@pytest.mark.asyncio()
async def test_alist_files(async_sdk_mock: MagicMock) -> None:
    sentinel = MagicMock()
    async_sdk_mock.aget_files = _make_coro(sentinel)
    llm = GigaChat()
    assert await llm.alist_files() is sentinel


def test_delete_file(sdk_mock: MagicMock) -> None:
    sentinel = MagicMock()
    sdk_mock.delete_file.return_value = sentinel
    llm = GigaChat()
    assert llm.delete_file("file-1") is sentinel
    sdk_mock.delete_file.assert_called_once_with("file-1")


@pytest.mark.asyncio()
async def test_adelete_file(async_sdk_mock: MagicMock) -> None:
    sentinel = MagicMock()
    async_sdk_mock.adelete_file = _make_coro(sentinel)
    llm = GigaChat()
    assert await llm.adelete_file("file-1") is sentinel


# ---------------------------------------------------------------------------
# _identifying_params / _llm_type
# ---------------------------------------------------------------------------


def test_identifying_params() -> None:
    llm = GigaChat(temperature=0.5, model="GigaChat-Pro", max_tokens=100)
    params = llm._identifying_params
    assert params["temperature"] == 0.5
    assert params["model"] == "GigaChat-Pro"
    assert params["max_tokens"] == 100


def test_llm_type() -> None:
    llm = GigaChat()
    assert llm._llm_type == "giga-chat-model"


def test_get_client_init_kwargs_includes_base(sdk_mock: MagicMock) -> None:
    llm = GigaChat(profanity_check=True, flags=["flag1"])
    kwargs = llm._get_client_init_kwargs()
    assert kwargs["profanity_check"] is True
    assert kwargs["flags"] == ["flag1"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_coro(return_value):  # type: ignore[no-untyped-def]
    async def _coro(*args, **kwargs):  # type: ignore[no-untyped-def]
        return return_value

    return _coro
