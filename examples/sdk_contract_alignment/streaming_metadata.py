"""Async streaming: aggregate chunks once to inspect final usage and metadata."""

import asyncio
import os

from langchain_core.messages import AIMessageChunk
from langchain_gigachat import GigaChat


async def main() -> None:
    llm = GigaChat(
        model=os.getenv("GIGACHAT_MODEL", "GigaChat"),
        use_api_v2=True,
        max_tokens=128,
    )
    combined: AIMessageChunk | None = None
    async for chunk in llm.astream("В двух предложениях расскажи про озеро Байкал."):
        print(chunk.text, end="", flush=True)
        combined = chunk if combined is None else combined + chunk
    if combined is None:
        raise RuntimeError("The server returned an empty stream.")
    print("\nUsage:", combined.usage_metadata)
    for key in (
        "model",
        "thread_id",
        "finish_reason",
        "additional_data",
        "error_details",
    ):
        print(f"{key}:", combined.response_metadata.get(key))
    # Provider-specific fields may be absent. Additional data/error snapshots
    # are emitted at completion; repeated identifiers are not concatenated.
    print("Content blocks:", combined.content_blocks)


if __name__ == "__main__":
    asyncio.run(main())
