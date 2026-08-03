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


def test_build_payload_uses_native_then_invocation_then_default_precedence() -> None:
    payload = primary.build_payload(
        [],
        defaults=_defaults(temperature=0.1, max_tokens=100, top_p=0.2),
        invocation_kwargs={
            "model": "invocation-model",
            "temperature": 0.4,
            "max_tokens": 200,
            "model_options": gm.ChatModelOptions(
                temperature=0.9,
                repetition_penalty=1.5,
            ),
        },
        cached_uploads={},
    )

    assert payload.model == "invocation-model"
    assert payload.model_options == gm.ChatModelOptions(
        temperature=0.9,
        top_p=0.2,
        max_tokens=200,
        repetition_penalty=1.5,
        update_interval=0.5,
        reasoning=gm.ChatReasoning(effort="low"),
    )


def test_build_payload_native_nested_options_win_over_convenience() -> None:
    payload = primary.build_payload(
        [],
        defaults=_defaults(reasoning_effort="low"),
        invocation_kwargs={
            "reasoning_effort": "medium",
            "response_format": {"type": "json_schema", "schema": {"type": "object"}},
            "model_options": {
                "reasoning": {"effort": "high"},
                "response_format": {"type": "text"},
            },
        },
        cached_uploads={},
    )

    assert payload.model_options
    assert payload.model_options.reasoning == gm.ChatReasoning(effort="high")
    assert payload.model_options.response_format
    assert payload.model_options.response_format.type == "text"


@pytest.mark.parametrize("nested", [False, True])
def test_build_payload_uses_one_response_format_normalizer(nested: bool) -> None:
    response_format = gm.JsonSchemaResponseFormat(
        schema={"type": "object"},
        strict=False,
    )
    invocation_kwargs: dict[str, Any]
    if nested:
        invocation_kwargs = {
            "model_options": {"response_format": response_format},
        }
    else:
        invocation_kwargs = {"response_format": response_format}

    payload = primary.build_payload(
        [],
        defaults=_defaults(),
        invocation_kwargs=invocation_kwargs,
        cached_uploads={},
    )

    assert payload.model_options is not None
    assert payload.model_options.response_format == gm.ChatResponseFormat(
        type="json_schema",
        schema={"type": "object"},
        strict=False,
    )


def test_build_payload_supports_primary_top_level_and_future_fields() -> None:
    payload = primary.build_payload(
        [],
        defaults=_defaults(
            model=None,
            profanity_check=None,
            temperature=None,
            top_p=None,
            max_tokens=None,
            repetition_penalty=None,
            update_interval=None,
            reasoning_effort=None,
            function_ranker=None,
            flags=None,
        ),
        invocation_kwargs={
            "assistant_id": "assistant",
            "tools_state_id": "state",
            "filter_config": {"request_content": {"neuro": False}},
            "storage": {"thread_id": "thread"},
            "user_info": {"timezone": "Europe/Moscow"},
            "stream": True,
            "future_provider_field": {"enabled": True},
        },
        cached_uploads={},
    )

    assert _dump(payload) == {
        "messages": [],
        "assistant_id": "assistant",
        "tools_state_id": "state",
        "filter_config": {"request_content": {"neuro": False}},
        "storage": {"thread_id": "thread"},
        "user_info": {"timezone": "Europe/Moscow"},
        "stream": True,
        "future_provider_field": {"enabled": True},
    }


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        (field_name, invalid_value)
        for field_name in (
            "assistant_id",
            "message_id",
            "thread_id",
            "tool_call_id",
            "tools_state_id",
        )
        for invalid_value in ("", "   ")
    ],
)
def test_build_payload_rejects_empty_request_identifiers(
    field_name: str,
    invalid_value: str,
) -> None:
    with pytest.raises(ValueError, match=f"{field_name} must be a non-empty string"):
        primary.build_payload(
            [],
            defaults=_defaults(),
            invocation_kwargs={field_name: invalid_value},
            cached_uploads={},
        )


def test_build_payload_explicit_disable_filter_wins() -> None:
    payload = primary.build_payload(
        [],
        defaults=_defaults(profanity_check=True),
        invocation_kwargs={
            "profanity_check": True,
            "disable_filter": True,
        },
        cached_uploads={},
    )

    assert payload.disable_filter is True
    assert "profanity_check" not in _dump(payload)


def test_build_payload_explicit_ranker_options_win() -> None:
    payload = primary.build_payload(
        [],
        defaults=_defaults(function_ranker={"enabled": True, "top_n": 5}),
        invocation_kwargs={
            "function_ranker": {"enabled": True, "top_n": 3},
            "ranker_options": gm.ChatRankerOptions(
                enabled=False,
                embeddings_model="Embeddings",
            ),
        },
        cached_uploads={},
    )

    assert payload.ranker_options == gm.ChatRankerOptions(
        enabled=False,
        embeddings_model="Embeddings",
    )


@pytest.mark.parametrize("flags", [["alpha", "beta"], ("alpha", "beta"), []])
def test_build_payload_accepts_flag_sequences_without_mutation(
    flags: list[str] | tuple[str, ...],
) -> None:
    original = copy.deepcopy(flags)

    payload = primary.build_payload(
        [],
        defaults=_defaults(flags=None),
        invocation_kwargs={"flags": flags},
        cached_uploads={},
    )

    assert payload.flags == list(flags)
    assert flags == original
    if isinstance(flags, list):
        assert payload.flags is not flags


@pytest.mark.parametrize(
    ("flags", "match"),
    [
        ("alpha", "must be a sequence"),
        (b"alpha", "must be a sequence"),
        (["alpha", 1], "non-empty strings"),
        ([""], "non-empty strings"),
    ],
)
def test_build_payload_rejects_invalid_flags(flags: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        primary.build_payload(
            [],
            defaults=_defaults(flags=None),
            invocation_kwargs={"flags": flags},
            cached_uploads={},
        )


def test_build_payload_preserves_falsey_public_values() -> None:
    payload = primary.build_payload(
        [],
        defaults=_defaults(
            temperature=0.5,
            top_p=0.5,
            max_tokens=100,
            profanity_check=True,
            function_ranker={"enabled": True},
        ),
        invocation_kwargs={
            "temperature": 0,
            "top_p": 0,
            "max_tokens": 0,
            "storage": False,
            "disable_filter": False,
            "ranker_options": {"enabled": False},
        },
        cached_uploads={},
    )

    assert payload.model_options is not None
    assert payload.model_options.temperature == 0
    assert payload.model_options.top_p == 0
    assert payload.model_options.max_tokens == 0
    # The pinned SDK deliberately serializes storage=False as an omitted
    # storage field, which is the provider's "disable storage" representation.
    assert payload.storage is None
    assert "storage" not in _dump(payload)
    assert payload.disable_filter is False
    assert payload.ranker_options == gm.ChatRankerOptions(enabled=False)


def test_build_payload_rejects_unsupported_legacy_ranker_fields() -> None:
    with pytest.raises(ValueError, match="Unsupported primary ranker option.*unknown"):
        primary.build_payload(
            [],
            defaults=_defaults(function_ranker={"unknown": True}),
            invocation_kwargs={},
            cached_uploads={},
        )


def test_build_payload_applies_tool_binding_and_consumed_keys() -> None:
    binding = primary.ToolBinding(
        tools=[gm.ChatTool(code_interpreter={})],
        tool_config=gm.ChatToolConfig(mode="forced", tool_name="code_interpreter"),
        consumed_keys=frozenset({"tools", "tool_config"}),
    )

    payload = primary.build_payload(
        [],
        defaults=_defaults(),
        invocation_kwargs={
            "tools": [{"code_interpreter": {"unexpected": True}}],
            "tool_config": {"mode": "auto"},
        },
        cached_uploads={},
        tool_binding=binding,
    )

    assert payload.tools == [gm.ChatTool(code_interpreter={})]
    assert payload.tool_config == gm.ChatToolConfig(
        mode="forced",
        tool_name="code_interpreter",
    )


def test_build_payload_does_not_mutate_caller_owned_inputs() -> None:
    messages = [
        HumanMessage(
            content=[
                {"type": "text", "text": "hello"},
                {"type": "image", "file_id": "image-1"},
            ]
        )
    ]
    model_options = {"temperature": 0.7, "custom": {"nested": [1]}}
    invocation_kwargs = {
        "model_options": model_options,
        "storage": {"metadata": {"key": "value"}},
    }
    cached_uploads = {"hash": "file"}
    original_messages = copy.deepcopy(messages)
    original_kwargs = copy.deepcopy(invocation_kwargs)
    original_cache = copy.deepcopy(cached_uploads)

    primary.build_payload(
        messages,
        defaults=_defaults(),
        invocation_kwargs=invocation_kwargs,
        cached_uploads=cached_uploads,
    )

    assert messages == original_messages
    assert invocation_kwargs == original_kwargs
    assert cached_uploads == original_cache


def test_build_payload_drops_langchain_control_keys() -> None:
    payload = primary.build_payload(
        [HumanMessage(content="actual")],
        defaults=_defaults(),
        invocation_kwargs={
            "messages": ["wrong"],
            "use_api_v2": True,
            "functions": [{"name": "legacy"}],
            "function_call": "auto",
            "tool_choice": "auto",
        },
        cached_uploads={},
    )

    dumped = _dump(payload)
    assert dumped["messages"] == [{"content": [{"text": "actual"}], "role": "user"}]
    assert (
        not {
            "use_api_v2",
            "functions",
            "function_call",
            "tool_choice",
        }
        & dumped.keys()
    )
