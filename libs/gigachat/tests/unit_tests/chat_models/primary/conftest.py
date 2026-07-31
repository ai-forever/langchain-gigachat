"""Pytest fixtures shared by primary chat contract tests."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import gigachat.models as gm
import pytest
from pytest_mock import MockerFixture

from .fixtures import (
    build_malformed_event,
    build_tool_completed_event,
    build_unknown_event,
)


@pytest.fixture()
def tool_completed_event() -> gm.PrimaryChatCompletionChunk:
    return build_tool_completed_event()


@pytest.fixture()
def unknown_event() -> dict[str, Any]:
    return build_unknown_event()


@pytest.fixture()
def malformed_event() -> dict[str, Any]:
    return build_malformed_event()


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
