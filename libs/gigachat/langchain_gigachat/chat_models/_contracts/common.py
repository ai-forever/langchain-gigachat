"""Route-neutral chat contract definitions."""

from typing import Literal

ChatContract = Literal["legacy", "primary"]
CROSS_CONTRACT_TOOL_STATE_ERROR = (
    "This message history contains provider-specific tool state from another "
    "API contract."
)

__all__ = ["CROSS_CONTRACT_TOOL_STATE_ERROR", "ChatContract"]
