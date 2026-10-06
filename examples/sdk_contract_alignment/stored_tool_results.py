"""Return only tool results to a stored v2 thread, keeping call ID and state."""

import os

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_gigachat import GigaChat


@tool
def get_weather(city: str) -> str:
    """Return illustrative weather data for a city (not a live forecast)."""
    return f"{city}: +20 °C, ясно (демонстрационные данные)."


def main() -> None:
    llm = GigaChat(
        model=os.getenv("GIGACHAT_MODEL", "GigaChat"),
        use_api_v2=True,
        max_tokens=128,
    )
    first = llm.bind_tools([get_weather], tool_choice="get_weather").invoke(
        "Какая погода в Казани?", storage=True
    )
    thread_id = first.response_metadata.get("thread_id")
    state_id = first.additional_kwargs.get("tools_state_id")
    if not thread_id or not state_id or not first.tool_calls:
        raise RuntimeError(
            "Expected a stored thread, continuation state and tool call."
        )
    results = []
    for call in first.tool_calls:
        if call["name"] != get_weather.name:
            raise ValueError(f"Unknown tool: {call['name']}")
        results.append(
            ToolMessage(
                content=get_weather.invoke(call["args"]),
                name=call["name"],
                tool_call_id=call["id"],
                additional_kwargs={"tools_state_id": state_id},
            )
        )
    # Stored history already includes the user message and assistant call.
    # Send only new results. The adapter omits the default model for this thread.
    answer = llm.bind_tools([get_weather], tool_choice="none").invoke(
        results, storage={"thread_id": thread_id}
    )
    print("Thread:", thread_id)
    print(answer.text)


if __name__ == "__main__":
    main()
