"""Public legacy payload purity across validation and request construction."""

from __future__ import annotations

import copy
from typing import Any
from unittest.mock import MagicMock

import gigachat.models as gm
import pytest

from langchain_gigachat.chat_models.gigachat import GigaChat

from .fixtures import CREATED_AT, MODEL


def _legacy_response() -> gm.ChatCompletion:
    return gm.ChatCompletion(
        choices=[
            gm.Choices(
                message=gm.Messages(
                    role=gm.MessagesRole.ASSISTANT,
                    content="done",
                ),
                index=0,
                finish_reason="stop",
            )
        ],
        created=CREATED_AT,
        model=MODEL,
        object="chat.completion",
        usage=gm.Usage(
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        ),
    )


def _function(name: Any) -> dict[str, Any]:
    return {
        "name": name,
        "description": f"Use {name}.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                }
            },
        },
    }


def test_public_sync_and_async_legacy_builds_are_deeply_pure(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.return_value = _legacy_response()
    sdk_client.achat.return_value = _legacy_response()
    functions = [_function("first")]
    tools = [{"type": "function", "function": _function("second")}]
    original_functions = copy.deepcopy(functions)
    original_tools = copy.deepcopy(tools)
    bound = GigaChat(model=MODEL).bind(functions=functions, tools=tools)

    bound.invoke("sync")
    assert functions == original_functions
    assert tools == original_tools

    async def invoke_async() -> None:
        await bound.ainvoke("async")

    import asyncio

    asyncio.run(invoke_async())

    assert functions == original_functions
    assert tools == original_tools
    sync_payload = sdk_client.chat.call_args.args[0]
    async_payload = sdk_client.achat.call_args.args[0]
    assert [item.name for item in sync_payload.functions] == ["first", "second"]
    assert [item.name for item in async_payload.functions] == ["first", "second"]


@pytest.mark.parametrize("route", ["sync", "async"])
def test_legacy_validation_failure_never_mutates_nested_function_inputs(
    sdk_client: MagicMock,
    route: str,
) -> None:
    functions = [_function(123)]
    tools = [{"type": "function", "function": _function("second")}]
    original_functions = copy.deepcopy(functions)
    original_tools = copy.deepcopy(tools)
    bound = GigaChat(model=MODEL).bind(functions=functions, tools=tools)

    with pytest.raises(ValueError):
        if route == "sync":
            bound.invoke("sync")
        else:
            import asyncio

            asyncio.run(bound.ainvoke("async"))

    assert functions == original_functions
    assert tools == original_tools
    sdk_client.chat.assert_not_called()
    sdk_client.achat.assert_not_called()
