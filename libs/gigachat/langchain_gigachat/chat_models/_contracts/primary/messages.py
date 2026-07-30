"""LangChain-to-primary message conversion boundary."""

from __future__ import annotations

from typing import Mapping, Sequence

import gigachat.models as gm
from langchain_core.messages import BaseMessage


def convert_messages(
    messages: Sequence[BaseMessage],
    *,
    cached_uploads: Mapping[str, str],
) -> list[gm.ChatMessage]:
    """Convert a complete LangChain message history to primary SDK messages."""
    raise NotImplementedError("Primary message conversion is not implemented")
