"""Primary request payload construction boundary."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import gigachat.models as gm
from langchain_core.messages import BaseMessage

from langchain_gigachat.chat_models._contracts.primary.types import (
    RequestDefaults,
    ToolBinding,
)


def build_payload(
    messages: Sequence[BaseMessage],
    *,
    defaults: RequestDefaults,
    invocation_kwargs: Mapping[str, Any],
    cached_uploads: Mapping[str, str],
    tool_binding: ToolBinding | None = None,
) -> gm.ChatCompletionRequest:
    """Build a primary SDK request without mutating caller-owned inputs."""
    raise NotImplementedError("Primary payload construction is not implemented")
