"""Primary tool normalization boundary."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from langchain_gigachat.chat_models._contracts.primary.types import ToolBinding


def build_tool_binding(
    *,
    functions: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    function_call: Any,
    explicit_tool_config: Any = None,
) -> ToolBinding:
    """Normalize client and provider tools for the primary contract."""
    raise NotImplementedError("Primary tool binding is not implemented")
