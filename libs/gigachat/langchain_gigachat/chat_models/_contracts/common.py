"""Route-neutral chat contract definitions."""

from typing import Literal

ChatContract = Literal["legacy", "primary"]

__all__ = ["ChatContract"]
