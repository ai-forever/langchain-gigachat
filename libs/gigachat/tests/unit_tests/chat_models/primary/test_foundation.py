from dataclasses import FrozenInstanceError

import pytest

from langchain_gigachat.chat_models._contracts import primary
from langchain_gigachat.chat_models._contracts.common import ChatContract
from langchain_gigachat.chat_models.gigachat import GigaChat


def test_use_api_v2_defaults_to_false() -> None:
    llm = GigaChat()

    assert llm.use_api_v2 is False
    assert llm._identifying_params["use_api_v2"] is False
    assert llm.model_dump()["use_api_v2"] is False


def test_use_api_v2_is_serialized_and_identifying() -> None:
    llm = GigaChat(use_api_v2=True)

    assert llm.use_api_v2 is True
    assert llm._identifying_params["use_api_v2"] is True
    assert llm.model_dump()["use_api_v2"] is True


def test_contract_names_are_explicit() -> None:
    legacy: ChatContract = "legacy"
    primary_contract: ChatContract = "primary"

    assert (legacy, primary_contract) == ("legacy", "primary")


def test_request_defaults_are_frozen() -> None:
    defaults = primary.RequestDefaults(
        model="GigaChat",
        profanity_check=None,
        temperature=0.1,
        top_p=None,
        max_tokens=128,
        repetition_penalty=None,
        update_interval=None,
        reasoning_effort=None,
        function_ranker=None,
        flags=("flag",),
    )

    with pytest.raises(FrozenInstanceError):
        defaults.model = "other"  # type: ignore[misc]


def test_tool_binding_is_frozen() -> None:
    binding = primary.ToolBinding(
        tools=None,
        tool_config=None,
        consumed_keys=frozenset({"tools"}),
    )

    with pytest.raises(FrozenInstanceError):
        binding.tools = []  # type: ignore[misc]


def test_stream_state_has_deterministic_defaults() -> None:
    state = primary.StreamState()

    assert state.next_block_index == 0
    assert state.message_id is None
    assert state.tools_state_id is None
