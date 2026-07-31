"""Reasoning request precedence and public workflow regressions."""

from __future__ import annotations

import copy
from typing import Any
from unittest.mock import MagicMock

import gigachat.models as gm
import pytest
from langchain_core.messages import HumanMessage

from langchain_gigachat.chat_models._contracts import primary
from langchain_gigachat.chat_models.gigachat import GigaChat

from .fixtures import MODEL, build_plain_text_response


def _defaults(*, reasoning_effort: str | None) -> primary.RequestDefaults:
    return primary.RequestDefaults(
        model=None,
        profanity_check=None,
        temperature=None,
        top_p=None,
        max_tokens=None,
        repetition_penalty=None,
        update_interval=None,
        reasoning_effort=reasoning_effort,
        function_ranker=None,
        flags=None,
    )


@pytest.mark.parametrize(
    ("default_effort", "invocation_kwargs", "expected_effort"),
    [
        pytest.param("low", {}, "low", id="instance-default-only"),
        pytest.param(
            "low",
            {"reasoning_effort": "medium"},
            "medium",
            id="invocation-effort-over-instance",
        ),
        pytest.param(
            "low",
            {"reasoning": {"effort": "high"}},
            "high",
            id="top-level-reasoning-over-instance",
        ),
        pytest.param(
            "low",
            {"model_options": {"reasoning": {"effort": "high"}}},
            "high",
            id="native-reasoning-over-instance",
        ),
        pytest.param(
            None,
            {
                "model_options": {"reasoning": {"effort": "high"}},
                "reasoning": {"effort": "medium"},
            },
            "high",
            id="native-over-top-level",
        ),
        pytest.param(
            None,
            {
                "model_options": {"reasoning": {"effort": "high"}},
                "reasoning_effort": "medium",
            },
            "high",
            id="native-over-invocation-effort",
        ),
        pytest.param(
            "low",
            {"model_options": {"reasoning": {"effort": "high"}}},
            "high",
            id="native-over-instance",
        ),
        pytest.param(
            None,
            {
                "reasoning": {"effort": "high"},
                "reasoning_effort": "medium",
            },
            "high",
            id="top-level-over-invocation-effort",
        ),
        pytest.param(
            "low",
            {"reasoning": {"effort": "high"}},
            "high",
            id="top-level-over-instance",
        ),
        pytest.param(
            "low",
            {"reasoning_effort": "high"},
            "high",
            id="invocation-effort-over-instance-pair",
        ),
    ],
)
def test_reasoning_precedence_is_explicit_for_every_entry_point_and_pair(
    default_effort: str | None,
    invocation_kwargs: dict[str, Any],
    expected_effort: str,
) -> None:
    payload = primary.build_payload(
        [],
        defaults=_defaults(reasoning_effort=default_effort),
        invocation_kwargs=invocation_kwargs,
        cached_uploads={},
    )

    assert payload.model_options is not None
    assert payload.model_options.reasoning == gm.ChatReasoning(effort=expected_effort)


def test_top_level_reasoning_is_consumed_locally_without_mutating_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reasoning = {"effort": "high"}
    model_options: dict[str, Any] = {"reasoning": None}
    invocation_kwargs = {
        "reasoning": reasoning,
        "reasoning_effort": "medium",
        "model_options": model_options,
    }
    original_kwargs = copy.deepcopy(invocation_kwargs)
    raw_payload: dict[str, Any] = {}
    original_validate = gm.ChatCompletionRequest.model_validate

    def capture_payload(value: Any) -> gm.ChatCompletionRequest:
        raw_payload.update(value)
        return original_validate(value)

    monkeypatch.setattr(
        gm.ChatCompletionRequest,
        "model_validate",
        staticmethod(capture_payload),
    )

    payload = primary.build_payload(
        [HumanMessage(content="hello")],
        defaults=_defaults(reasoning_effort="low"),
        invocation_kwargs=invocation_kwargs,
        cached_uploads={},
    )

    assert invocation_kwargs == original_kwargs
    assert reasoning == {"effort": "high"}
    assert model_options == {"reasoning": None}
    assert "reasoning" not in raw_payload
    assert raw_payload["model_options"].reasoning == gm.ChatReasoning(effort="high")
    wire_payload = payload.model_dump(exclude_none=True, by_alias=True)
    assert "reasoning" not in wire_payload
    assert wire_payload["model_options"]["reasoning"] == {"effort": "high"}


def test_public_invoke_prefers_explicit_reasoning_without_mutating_it(
    sdk_client: MagicMock,
) -> None:
    sdk_client.chat.create.return_value = build_plain_text_response()
    reasoning = {"effort": "high"}
    original = copy.deepcopy(reasoning)

    result = GigaChat(
        model=MODEL,
        use_api_v2=True,
        reasoning_effort="low",
    ).invoke(
        "Hello",
        reasoning=reasoning,
        reasoning_effort="medium",
    )

    assert result.content == "Primary response"
    payload = sdk_client.chat.create.call_args.args[0]
    assert payload.model_options is not None
    assert payload.model_options.reasoning == gm.ChatReasoning(effort="high")
    assert reasoning == original


@pytest.mark.asyncio
async def test_public_ainvoke_prefers_native_reasoning_without_mutating_it(
    sdk_client: MagicMock,
) -> None:
    sdk_client.achat.create.return_value = build_plain_text_response()
    model_options = {"reasoning": {"effort": "high"}}
    original = copy.deepcopy(model_options)

    result = await GigaChat(
        model=MODEL,
        use_api_v2=True,
        reasoning_effort="low",
    ).ainvoke(
        "Hello",
        model_options=model_options,
        reasoning={"effort": "medium"},
        reasoning_effort="low",
    )

    assert result.content == "Primary response"
    payload = sdk_client.achat.create.call_args.args[0]
    assert payload.model_options is not None
    assert payload.model_options.reasoning == gm.ChatReasoning(effort="high")
    assert model_options == original


def test_invalid_reasoning_fails_before_sync_network_call(
    sdk_client: MagicMock,
) -> None:
    with pytest.raises(ValueError, match="reasoning must be a mapping"):
        GigaChat(model=MODEL, use_api_v2=True).invoke(
            "Hello",
            reasoning="high",
        )

    sdk_client.chat.create.assert_not_called()
    sdk_client.chat.assert_not_called()
    sdk_client.upload.assert_not_called()


@pytest.mark.asyncio
async def test_invalid_reasoning_fails_before_async_network_call(
    sdk_client: MagicMock,
) -> None:
    with pytest.raises(ValueError, match="reasoning must be a mapping"):
        await GigaChat(model=MODEL, use_api_v2=True).ainvoke(
            "Hello",
            reasoning="high",
        )

    sdk_client.achat.create.assert_not_called()
    sdk_client.achat.assert_not_called()
    sdk_client.aupload.assert_not_called()
