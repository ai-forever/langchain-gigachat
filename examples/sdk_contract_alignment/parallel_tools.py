"""V2 anyOf arguments, required first call and parallel results with real IDs."""

import os

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_gigachat import GigaChat


@tool
def get_room(room: int | str) -> str:
    """Return room capacity by number or alphanumeric room code."""
    rooms = {"101": 24, "A-2": 40}
    capacity = rooms.get(str(room))
    return f"Аудитория {room}: {capacity} мест" if capacity else "Аудитория не найдена"


def main() -> None:
    llm = GigaChat(
        model=os.getenv("GIGACHAT_MODEL", "GigaChat"),
        use_api_v2=True,
        parallel_tool_calls=True,
        max_tokens=256,
    )
    # int | str becomes anyOf on v2. The v1 adapter rejects such unions.
    required = llm.bind_tools([get_room], tool_choice="required")
    automatic = llm.bind_tools([get_room], tool_choice="auto")
    history: list[BaseMessage] = [
        HumanMessage("Узнай вместимость аудиторий 101 и A-2 и сравни их.")
    ]
    for turn in range(4):
        # Require a call only at the beginning; let the model finish afterwards.
        answer = (required if turn == 0 else automatic).invoke(history)
        history.append(answer)
        if not answer.tool_calls:
            print(answer.text)
            return
        print("Continuation state:", answer.additional_kwargs.get("tools_state_id"))
        for call in answer.tool_calls:
            if call["name"] != get_room.name:
                raise ValueError(f"Unknown tool: {call['name']}")
            print("Call:", call["id"], call["args"])
            history.append(
                ToolMessage(
                    content=get_room.invoke(call["args"]),
                    name=call["name"],
                    tool_call_id=call["id"],
                )
            )
    raise RuntimeError("The model did not finish within four turns.")


if __name__ == "__main__":
    main()
