"""Two related calls with one session: full LangChain input versus raw SDK usage."""

import os
from typing import Any
from uuid import uuid4

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.outputs import LLMResult
from langchain_gigachat import GigaChat


class ShowProviderUsage(BaseCallbackHandler):
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        # The callback receives the ChatResult's raw provider accounting.
        # It is different from the normalized AIMessage.usage_metadata.
        print("Raw SDK usage:", (response.llm_output or {}).get("token_usage"))


def main() -> None:
    llm = GigaChat(
        model=os.getenv("GIGACHAT_MODEL", "GigaChat"),
        use_api_v2=True,
        session_id=str(uuid4()),  # Reused for both calls, new for the next run.
        max_tokens=64,
    )
    catalog = "\n".join(
        f"Room {number}: {20 + number % 30} seats, projector available."
        for number in range(101, 161)
    )
    context = SystemMessage("Answer using this room directory:\n" + catalog)
    for question in ("How many seats are in room 101?", "What about room 120?"):
        answer = llm.invoke(
            [context, HumanMessage(question)],
            config={"callbacks": [ShowProviderUsage()]},
        )
        print(answer.text)
        usage: dict[str, Any] = dict(answer.usage_metadata or {})
        cached = (usage.get("input_token_details") or {}).get("cache_read", 0)
        print("Full input:", usage.get("input_tokens"), "including cached:", cached)
        print(
            "Output:", usage.get("output_tokens"), "total:", usage.get("total_tokens")
        )
    # Cache hits depend on the server; a session is not a server-side thread.
    # On v1 the same session_id option works; cached usage has the same meaning.


if __name__ == "__main__":
    main()
