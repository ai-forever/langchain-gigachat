"""Primary request mapping owned by the LangChain adapter."""

from __future__ import annotations

import copy
from typing import Any

import gigachat.models as gm
import pytest
from langchain_core.messages import HumanMessage

from langchain_gigachat.chat_models._contracts import primary


def _defaults(**overrides: Any) -> primary.RequestDefaults:
    values: dict[str, Any] = {
        "model": "default-model",
        "profanity_check": True,
        "temperature": 0.1,
        "top_p": 0.2,
        "max_tokens": 100,
        "repetition_penalty": 1.1,
        "update_interval": 0.5,
        "reasoning_effort": "low",
        "function_ranker": {"enabled": True, "top_n": 5},
        "flags": ("default-flag",),
    }
    values.update(overrides)
    return primary.RequestDefaults(**values)


def _dump(payload: gm.ChatCompletionRequest) -> dict[str, Any]:
    return payload.model_dump(exclude_none=True, by_alias=True)


def test_build_payload_maps_defaults_to_primary_fields() -> None:
    payload = primary.build_payload(
        [HumanMessage(content="hello")],
        defaults=_defaults(),
        invocation_kwargs={},
        cached_uploads={},
    )

    assert _dump(payload) == {
        "model": "default-model",
        "messages": [{"content": [{"text": "hello"}], "role": "user"}],
        "model_options": {
            "temperature": 0.1,
            "top_p": 0.2,
            "max_tokens": 100,
            "repetition_penalty": 1.1,
            "update_interval": 0.5,
            "reasoning": {"effort": "low"},
        },
        "ranker_options": {"enabled": True, "top_n": 5},
        "disable_filter": False,
        "flags": ["default-flag"],
    }


def test_native_options_win_over_invocation_and_defaults() -> None:
    payload = primary.build_payload(
        [],
        defaults=_defaults(),
        invocation_kwargs={
            "model": "invocation-model",
            "temperature": 0.4,
            "max_tokens": 200,
            "reasoning_effort": "medium",
            "response_format": {"type": "json_schema"},
            "model_options": {
                "temperature": 0.9,
                "reasoning": {"effort": "high"},
                "response_format": {"type": "text"},
            },
        },
        cached_uploads={},
    )

    assert payload.model == "invocation-model"
    assert payload.model_options is not None
    assert payload.model_options.temperature == 0.9
    assert payload.model_options.max_tokens == 200
    assert payload.model_options.reasoning == gm.ChatReasoning(effort="high")
    assert payload.model_options.response_format == gm.ChatResponseFormat(type="text")


@pytest.mark.parametrize(
    "stateful_kwargs",
    [
        {"assistant_id": "assistant-1"},
        {"storage": {"thread_id": "thread-1"}},
    ],
)
def test_stateful_requests_omit_the_implicit_model(
    stateful_kwargs: dict[str, Any],
) -> None:
    payload = primary.build_payload(
        [HumanMessage(content="continue")],
        defaults=_defaults(),
        invocation_kwargs=stateful_kwargs,
        cached_uploads={},
    )

    assert payload.model is None


@pytest.mark.parametrize(
    "storage",
    [
        False,
        True,
        gm.ChatStorage(thread_id="thread-1"),
        {"thread_id": "thread-1"},
    ],
)
def test_primary_storage_accepts_only_primary_sdk_forms(storage: Any) -> None:
    payload = primary.build_payload(
        [],
        defaults=_defaults(model=None),
        invocation_kwargs={"storage": storage},
        cached_uploads={},
    )

    if storage is False:
        assert payload.storage is None
    elif storage is True:
        assert isinstance(payload.storage, gm.ChatStorage)
    else:
        assert payload.storage == gm.ChatStorage(thread_id="thread-1")


def test_primary_storage_rejects_legacy_storage() -> None:
    with pytest.raises(ValueError, match="gm.ChatStorage"):
        primary.build_payload(
            [],
            defaults=_defaults(),
            invocation_kwargs={
                "storage": gm.Storage(is_stateful=True, thread_id="legacy-thread")
            },
            cached_uploads={},
        )


def test_adapter_specific_filter_and_ranker_mapping() -> None:
    payload = primary.build_payload(
        [],
        defaults=_defaults(),
        invocation_kwargs={
            "profanity_check": False,
            "ranker_options": {"enabled": False},
        },
        cached_uploads={},
    )

    assert payload.disable_filter is True
    assert payload.ranker_options == gm.ChatRankerOptions(enabled=False)


def test_tool_binding_is_applied_and_control_keys_are_consumed() -> None:
    binding = primary.ToolBinding(
        tools=[gm.ChatTool(code_interpreter={})],
        tool_config=gm.ChatToolConfig(mode="forced", tool_name="code_interpreter"),
        consumed_keys=frozenset({"tools", "tool_config"}),
    )

    payload = primary.build_payload(
        [HumanMessage(content="run code")],
        defaults=_defaults(),
        invocation_kwargs={
            "use_api_v2": True,
            "tools": [{"type": "web_search"}],
            "tool_config": {"mode": "auto"},
            "function_call": "auto",
        },
        cached_uploads={},
        tool_binding=binding,
    )

    assert payload.tools == [gm.ChatTool(code_interpreter={})]
    assert payload.tool_config == gm.ChatToolConfig(
        mode="forced",
        tool_name="code_interpreter",
    )
    assert (
        not {
            "use_api_v2",
            "function_call",
        }
        & _dump(payload).keys()
    )


def test_build_payload_does_not_mutate_caller_inputs() -> None:
    messages = [
        HumanMessage(
            content=[
                {"type": "text", "text": "hello"},
                {"type": "file", "file_id": "file-1"},
            ]
        )
    ]
    invocation_kwargs = {
        "model_options": {"temperature": 0.7},
        "storage": {"metadata": {"key": "value"}},
    }
    original_messages = copy.deepcopy(messages)
    original_kwargs = copy.deepcopy(invocation_kwargs)

    primary.build_payload(
        messages,
        defaults=_defaults(),
        invocation_kwargs=invocation_kwargs,
        cached_uploads={},
    )

    assert messages == original_messages
    assert invocation_kwargs == original_kwargs
