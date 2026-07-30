"""Primary non-stream response conversion boundary."""

from __future__ import annotations

import gigachat.models as gm
from langchain_core.outputs import ChatResult


def create_chat_result(response: gm.ChatCompletionResponse) -> ChatResult:
    """Convert one primary SDK response into one LangChain chat result."""
    raise NotImplementedError("Primary response conversion is not implemented")
