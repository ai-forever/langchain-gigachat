from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from langchain_gigachat.embeddings.gigachat import GigaChatEmbeddings


@pytest.fixture()
def mock_embeddings_response() -> MagicMock:
    item1 = MagicMock()
    item1.embedding = [0.1, 0.2, 0.3]
    item2 = MagicMock()
    item2.embedding = [0.4, 0.5, 0.6]
    response = MagicMock()
    response.data = [item1, item2]
    return response


@pytest.fixture()
def patch_gigachat_embeddings(
    mocker: MockerFixture, mock_embeddings_response: MagicMock
) -> MagicMock:
    mock = mocker.Mock()
    mock.embeddings.return_value = mock_embeddings_response
    mocker.patch("gigachat.GigaChat", return_value=mock)
    return mock


@pytest.fixture()
def patch_gigachat_aembeddings(
    mocker: MockerFixture, mock_embeddings_response: MagicMock
) -> MagicMock:
    async def _aembeddings(**kwargs):  # type: ignore[no-untyped-def]
        return mock_embeddings_response

    mock = mocker.Mock()
    mock.aembeddings = _aembeddings
    mocker.patch("gigachat.GigaChat", return_value=mock)
    return mock


def test_embed_documents(patch_gigachat_embeddings: MagicMock) -> None:
    emb = GigaChatEmbeddings()
    result = emb.embed_documents(["hello", "world"])
    assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    patch_gigachat_embeddings.embeddings.assert_called_once_with(
        texts=["hello", "world"]
    )


@pytest.mark.asyncio()
async def test_aembed_documents(patch_gigachat_aembeddings: MagicMock) -> None:
    emb = GigaChatEmbeddings()
    result = await emb.aembed_documents(["hello", "world"])
    assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


def test_embed_query_without_prefix(patch_gigachat_embeddings: MagicMock) -> None:
    emb = GigaChatEmbeddings()
    result = emb.embed_query("test query")
    assert result == [0.1, 0.2, 0.3]
    patch_gigachat_embeddings.embeddings.assert_called_once_with(
        texts=["test query"]
    )


def test_embed_query_with_prefix(patch_gigachat_embeddings: MagicMock) -> None:
    emb = GigaChatEmbeddings(use_prefix_query=True)
    result = emb.embed_query("test query")
    assert result == [0.1, 0.2, 0.3]
    expected_text = emb.prefix_query + "test query"
    patch_gigachat_embeddings.embeddings.assert_called_once_with(
        texts=[expected_text]
    )


@pytest.mark.asyncio()
async def test_aembed_query_without_prefix(
    patch_gigachat_aembeddings: MagicMock,
) -> None:
    emb = GigaChatEmbeddings()
    result = await emb.aembed_query("test query")
    assert result == [0.1, 0.2, 0.3]


@pytest.mark.asyncio()
async def test_aembed_query_with_prefix(
    patch_gigachat_aembeddings: MagicMock,
) -> None:
    emb = GigaChatEmbeddings(use_prefix_query=True)
    result = await emb.aembed_query("test query")
    assert result == [0.1, 0.2, 0.3]


def test_model_forwarded_to_sdk(patch_gigachat_embeddings: MagicMock) -> None:
    emb = GigaChatEmbeddings(model="Embeddings")
    emb.embed_documents(["hello"])
    patch_gigachat_embeddings.embeddings.assert_called_once_with(
        texts=["hello"], model="Embeddings"
    )


def test_model_none_not_forwarded(patch_gigachat_embeddings: MagicMock) -> None:
    emb = GigaChatEmbeddings()
    emb.embed_documents(["hello"])
    call_kwargs = patch_gigachat_embeddings.embeddings.call_args[1]
    assert "model" not in call_kwargs
