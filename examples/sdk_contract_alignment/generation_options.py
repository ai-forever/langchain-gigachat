"""Reasoning budgets on both routes and advanced v2 request options."""

import os

from langchain_gigachat import GigaChat


def main() -> None:
    for use_api_v2 in (False, True):
        llm = GigaChat(
            model=os.getenv("GIGACHAT_MODEL", "GigaChat"),
            use_api_v2=use_api_v2,
            max_tokens=256,
            reasoning_effort="low",
            reasoning_max_tokens=64,
        )
        # v1: reasoning_max_tokens at the root; v2: reasoning.max_tokens
        # inside model_options. The adapter handles the mapping.
        message = llm.invoke("Сколько будет 17 × 23? Коротко объясни ответ.")
        print(f"\nAPI {'v2' if use_api_v2 else 'v1'}: {message.text}")
        print("Finish reason:", message.response_metadata.get("finish_reason"))
        print("Usage:", message.usage_metadata)

    # Explicit nested options override the constructor's max_tokens=256.
    # A nested None would suppress that default instead of sending 256.
    limited = llm.bind(
        model_options={"max_tokens": 32},
        reasoning=None,
        additional_fields={"ranker_options": {"enabled": False}},
    )
    print("\nBound v2 options:", limited.invoke("Назови столицу Татарстана.").text)


if __name__ == "__main__":
    main()
