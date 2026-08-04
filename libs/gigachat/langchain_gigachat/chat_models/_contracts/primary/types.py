"""Typed boundaries shared by primary chat contract adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import gigachat.models as gm
from gigachat.exceptions import GigaChatException


class PrimaryStreamError(GigaChatException):
    """Provider-declared failure from a primary named-event stream."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.payload = dict(payload)
        error = self.payload.get("error")
        super().__init__(f"Primary GigaChat stream returned response.error: {error!r}")


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
    """Small amount of state required to aggregate one primary stream."""

    next_block_index: int = 0
    active_text_block_index: int | str | None = None
    active_text_block_role: Optional[str] = None
    active_reasoning_message_id: Optional[str] = None
    message_id: Optional[str] = None
    provider_message_id: Optional[str] = None
    tools_state_id: Optional[str] = None
    client_tools_state_id: Optional[str] = None
    provider_tools_state_ids: list[str] = field(default_factory=list)
    thread_id: Optional[str] = None
    model: Optional[str] = None
    created_at: Optional[int] = None
    x_headers: dict[str, Any] = field(default_factory=dict)
    emitted_metadata_fields: set[str] = field(default_factory=set)
    completion_event: Optional[dict[str, Any]] = None
    usage_metadata: Optional[dict[str, Any]] = None

    client_tool_started: bool = False
    client_tool_name: Optional[str] = None
    client_tool_id: Optional[str] = None
    client_tool_index: Optional[int] = None
    client_tool_argument_mode: Optional[str] = None
    client_tool_arguments_text: str = ""
    client_tool_arguments_complete: bool = False
    client_tool_name_emitted: bool = False
    client_tool_id_emitted: bool = False

    server_tool_indexes: dict[str, int] = field(default_factory=dict)
    server_tool_result_indexes: dict[str, int] = field(default_factory=dict)
    server_tool_names: dict[str, str] = field(default_factory=dict)
    active_server_tool_call_id: Optional[str] = None
    next_server_tool_sequence: int = 0
    server_tool_argument_state: dict[str, tuple[str, str]] = field(default_factory=dict)
    completed_server_tool_ids: set[str] = field(default_factory=set)
    last_completed_server_tool_id: Optional[str] = None
