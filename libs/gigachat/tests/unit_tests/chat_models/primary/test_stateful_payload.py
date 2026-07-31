import copy
from typing import Any

import gigachat.models as gm
import pytest

from langchain_gigachat.chat_models._contracts import primary


def _defaults(**overrides: Any) -> primary.RequestDefaults:
    values: dict[str, Any] = {
        "model": "default-model",
        "profanity_check": None,
        "temperature": None,
        "top_p": None,
        "max_tokens": None,
        "repetition_penalty": None,
        "update_interval": None,
        "reasoning_effort": None,
        "function_ranker": None,
        "flags": None,
    }
    values.update(overrides)
    return primary.RequestDefaults(**values)


def _build(**invocation_kwargs: Any) -> gm.ChatCompletionRequest:
    return primary.build_payload(
        [],
        defaults=_defaults(),
        invocation_kwargs=invocation_kwargs,
        cached_uploads={},
    )


def test_assistant_request_omits_implicit_default_model() -> None:
    payload = _build(assistant_id="assistant")

    assert payload.assistant_id == "assistant"
    assert payload.model is None


def test_thread_request_omits_implicit_default_model() -> None:
    payload = _build(storage={"thread_id": "thread"})

    assert payload.storage == gm.ChatStorage(thread_id="thread")
    assert payload.model is None


def test_explicit_model_is_preserved_for_assistant_request() -> None:
    payload = _build(assistant_id="assistant", model="invocation-model")

    assert payload.assistant_id == "assistant"
    assert payload.model == "invocation-model"


def test_stateless_request_keeps_default_model() -> None:
    payload = _build()

    assert payload.model == "default-model"


def test_storage_none_is_preserved() -> None:
    payload = _build(storage=None)

    assert payload.storage is None


def test_storage_true_uses_sdk_default_storage() -> None:
    payload = _build(storage=True)

    assert payload.storage == gm.ChatStorage()


def test_storage_false_disables_sdk_storage() -> None:
    payload = _build(storage=False)

    assert payload.storage is None


def test_chat_storage_object_is_copied() -> None:
    storage = gm.ChatStorage(
        limit=4,
        thread_id="thread",
        metadata={"labels": ["one"]},
    )

    payload = _build(storage=storage)

    assert isinstance(payload.storage, gm.ChatStorage)
    assert payload.storage == storage
    assert payload.storage is not storage
    assert payload.storage.metadata is not storage.metadata


def test_mapping_storage_preserves_primary_and_future_fields() -> None:
    storage = {
        "limit": 4,
        "thread_id": "thread",
        "metadata": {"labels": ["one"]},
        "future_storage_field": {"enabled": True},
    }

    payload = _build(storage=storage)

    assert isinstance(payload.storage, gm.ChatStorage)
    assert payload.storage.model_dump(exclude_none=True) == storage


def test_legacy_storage_maps_every_primary_equivalent() -> None:
    storage = gm.Storage(
        is_stateful=True,
        limit=4,
        assistant_id="assistant",
        thread_id="thread",
        metadata={"labels": ["one"]},
    )

    payload = _build(storage=storage)

    assert payload.assistant_id == "assistant"
    assert isinstance(payload.storage, gm.ChatStorage)
    assert payload.storage == gm.ChatStorage(
        limit=4,
        thread_id="thread",
        metadata={"labels": ["one"]},
    )
    assert payload.model is None
    assert "is_stateful" not in payload.storage.model_dump()
    assert "assistant_id" not in payload.storage.model_dump()


def test_legacy_storage_false_maps_to_disabled_storage() -> None:
    payload = _build(storage=gm.Storage(is_stateful=False))

    assert payload.storage is None


def test_legacy_storage_false_rejects_fields_that_would_be_discarded() -> None:
    storage = gm.Storage(
        is_stateful=False,
        assistant_id="assistant",
        thread_id="thread",
        limit=4,
        metadata={"labels": ["one"]},
    )

    with pytest.raises(
        ValueError,
        match=(
            r"Legacy storage with is_stateful=False cannot include additional "
            r"storage field\(s\): assistant_id, limit, metadata, thread_id"
        ),
    ):
        _build(storage=storage)


def test_conflicting_top_level_and_storage_assistant_ids_fail() -> None:
    with pytest.raises(
        ValueError,
        match="Conflicting assistant_id values in the top-level request and storage",
    ):
        _build(
            assistant_id="top-level-assistant",
            storage={"assistant_id": "storage-assistant"},
        )


def test_matching_top_level_and_storage_assistant_ids_are_accepted() -> None:
    payload = _build(
        assistant_id="assistant",
        storage={"assistant_id": "assistant", "limit": 4},
    )

    assert payload.assistant_id == "assistant"
    assert payload.storage == gm.ChatStorage(limit=4)


@pytest.mark.parametrize("field_name", ["assistant_id", "thread_id"])
def test_storage_rejects_empty_identifiers(field_name: str) -> None:
    with pytest.raises(
        ValueError,
        match=f"storage {field_name} must be a non-empty string",
    ):
        _build(storage={field_name: ""})


def test_storage_mapping_is_not_mutated_during_normalization() -> None:
    storage = {
        "assistant_id": "assistant",
        "thread_id": "thread",
        "metadata": {"labels": ["one"]},
    }
    original = copy.deepcopy(storage)

    _build(storage=storage)

    assert storage == original


def test_invalid_storage_type_has_actionable_error() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "storage must be None, a bool, gm.ChatStorage, a compatible mapping, "
            "or legacy gm.Storage"
        ),
    ):
        _build(storage=["thread"])
