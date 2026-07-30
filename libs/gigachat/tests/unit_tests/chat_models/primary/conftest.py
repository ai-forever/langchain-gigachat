"""Pytest fixtures shared by primary chat contract tests."""

from typing import Any

import gigachat.models as gm
import pytest

from .fixtures import (
    build_function_call_response,
    build_malformed_event,
    build_message_delta_event,
    build_message_done_event,
    build_multimodal_response,
    build_plain_text_response,
    build_tool_completed_event,
    build_tool_roundtrip_history,
    build_unknown_event,
)


@pytest.fixture()
def plain_text_response() -> gm.ChatCompletionResponse:
    return build_plain_text_response()


@pytest.fixture()
def multimodal_response() -> gm.ChatCompletionResponse:
    return build_multimodal_response()


@pytest.fixture()
def function_call_response() -> gm.ChatCompletionResponse:
    return build_function_call_response()


@pytest.fixture()
def tool_roundtrip_history() -> list[gm.ChatMessage]:
    return build_tool_roundtrip_history()


@pytest.fixture()
def message_delta_event() -> gm.PrimaryChatCompletionChunk:
    return build_message_delta_event()


@pytest.fixture()
def tool_completed_event() -> gm.PrimaryChatCompletionChunk:
    return build_tool_completed_event()


@pytest.fixture()
def message_done_event() -> gm.PrimaryChatCompletionChunk:
    return build_message_done_event()


@pytest.fixture()
def unknown_event() -> dict[str, Any]:
    return build_unknown_event()


@pytest.fixture()
def malformed_event() -> dict[str, Any]:
    return build_malformed_event()
