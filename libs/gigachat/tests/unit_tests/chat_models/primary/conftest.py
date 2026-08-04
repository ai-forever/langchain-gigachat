"""Pytest fixtures shared by primary chat contract tests."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_mock import MockerFixture


@pytest.fixture()
def sdk_client(mocker: MockerFixture) -> MagicMock:
    """Return one client exposing both legacy and primary SDK namespaces."""
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.create = MagicMock()
    client.chat.stream = MagicMock()
    client.achat = AsyncMock()
    client.achat.create = AsyncMock()
    client.achat.stream = MagicMock()
    mocker.patch("gigachat.GigaChat", return_value=client)
    return client
