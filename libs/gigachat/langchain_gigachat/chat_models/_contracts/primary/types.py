"""Typed boundaries shared by primary chat contract adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import gigachat.models as gm


@dataclass(frozen=True)
class RequestDefaults:
    """Instance defaults admitted by the primary request builder."""

    model: Optional[str]
    profanity_check: Optional[bool]
    temperature: Optional[float]
    top_p: Optional[float]
    max_tokens: Optional[int]
    repetition_penalty: Optional[float]
    update_interval: Optional[float]
    reasoning_effort: Optional[str]
    function_ranker: Optional[Mapping[str, Any]]
    flags: Optional[Sequence[str]]


@dataclass(frozen=True)
class ToolBinding:
    """Normalized tools plus the invocation keys consumed to build them."""

    tools: Optional[list[gm.ChatTool]]
    tool_config: Optional[gm.ChatToolConfig]
    consumed_keys: frozenset[str]


@dataclass
class StreamState:
    """Mutable identity and ordering state shared across stream events."""

    first_chunk: bool = True
    next_block_index: int = 0
    message_id: Optional[str] = None
    tools_state_id: Optional[str] = None
