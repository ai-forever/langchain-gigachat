"""Primary named-event stream conversion boundary."""

from __future__ import annotations

from typing import Any, Mapping

import gigachat.models as gm
from langchain_core.outputs import ChatGenerationChunk

from langchain_gigachat.chat_models._contracts.primary.types import StreamState


def convert_stream_event(
    event: gm.PrimaryChatCompletionChunk | Mapping[str, Any],
    *,
    state: StreamState,
) -> ChatGenerationChunk | None:
    """Convert one primary SDK event while preserving cross-event state."""
    raise NotImplementedError("Primary stream conversion is not implemented")
