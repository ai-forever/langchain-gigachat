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


@pytest.mark.parametrize("as_model", [False, True])
def test_native_none_suppresses_flat_options_and_defaults(as_model: bool) -> None:
    native = {"temperature": None, "reasoning": None, "response_format": None}
    payload = primary.build_payload(
        [],
        defaults=_defaults(),
        invocation_kwargs={
            "temperature": 0.8,
            "reasoning_effort": "high",
            "response_format": {"type": "json_schema"},
            "model_options": gm.ChatModelOptions(**native) if as_model else native,
        },
        cached_uploads={},
    )
    options = _dump(payload)["model_options"]
    assert "temperature" not in options
    assert "reasoning" not in options
    assert "response_format" not in options
    assert options["max_tokens"] == 100


def test_reasoning_budget_and_parallel_calls_map_to_nested_options() -> None:
    payload = primary.build_payload(
        [],
        defaults=_defaults(reasoning_max_tokens=50, parallel_tool_calls=True),
        invocation_kwargs={"reasoning_max_tokens": 25, "parallel_tool_calls": False},
        cached_uploads={},
    )
    dumped = _dump(payload)
    assert dumped["model_options"]["reasoning"] == {"effort": "low", "max_tokens": 25}
    assert dumped["model_options"]["parallel_tool_calls"] is False
    assert "reasoning_max_tokens" not in dumped
    assert "parallel_tool_calls" not in dumped


def test_additional_fields_are_normalized_below_explicit_invocation() -> None:
    kwargs = {
        "model": "explicit-model",
        "temperature": 0.9,
        "additional_fields": {
            "model": "extra-model",
            "temperature": 0.4,
            "reasoning_max_tokens": 10,
            "messages": [{"role": "user", "content": "injected"}],
            "custom_option": {"enabled": True},
        },
    }
    original = copy.deepcopy(kwargs)
    payload = primary.build_payload(
        [HumanMessage(content="real")],
        defaults=_defaults(),
        invocation_kwargs=kwargs,
        cached_uploads={},
    )
    dumped = _dump(payload)
    assert kwargs == original
    assert dumped["model"] == "explicit-model"
    assert dumped["messages"] == [{"role": "user", "content": [{"text": "real"}]}]
    assert dumped["model_options"]["temperature"] == 0.9
    assert dumped["model_options"]["reasoning"]["max_tokens"] == 10
    assert dumped["custom_option"] == {"enabled": True}
    assert "additional_fields" not in dumped


def test_additional_fields_nulls_survive_sdk_serialization() -> None:
    from gigachat.api.chat_completions import _build_request_json

    kwargs = {"additional_fields": {"vendor_option": None}}
    if "additional_fields" not in gm.ChatCompletionRequest.model_fields:
        with pytest.raises(ValueError, match="requires an updated GigaChat SDK"):
            primary.build_payload(
                [], defaults=_defaults(), invocation_kwargs=kwargs, cached_uploads={}
            )
        return
    payload = primary.build_payload(
        [], defaults=_defaults(), invocation_kwargs=kwargs, cached_uploads={}
    )
    wire = _build_request_json(payload)
    assert "vendor_option" in wire and wire["vendor_option"] is None
    assert "additional_fields" not in wire


def test_additional_fields_win_over_explicit_none_like_sdk_serializer() -> None:
    payload = primary.build_payload(
        [],
        defaults=_defaults(),
        cached_uploads={},
        invocation_kwargs={"model": None, "additional_fields": {"model": "extra"}},
    )
    assert payload.model == "extra"


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
    assert _dump(payload)["ranker_options"] == {"enabled": False}


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
