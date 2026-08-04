"""Tests for _convert_dict_to_message / _convert_message_to_dict edge cases."""

import pytest
from gigachat.models import FunctionCall, Messages, MessagesRole
from langchain_core.messages import AIMessage, FunctionMessage, HumanMessage

from langchain_gigachat.chat_models.gigachat import (
    _convert_dict_to_message,
    _convert_message_to_dict,
)

# ---------------------------------------------------------------------------
# _convert_dict_to_message — function_call as FunctionCall object
# ---------------------------------------------------------------------------


def test_convert_dict_to_message_function_call_object() -> None:
    msg = Messages(
        id=None,
        role=MessagesRole.ASSISTANT,
        content="",
        function_call=FunctionCall(name="my_tool", arguments={"arg": 1}),
    )
    result = _convert_dict_to_message(msg)
    assert isinstance(result, AIMessage)
    assert result.additional_kwargs["function_call"]["name"] == "my_tool"
    assert result.additional_kwargs["function_call"]["arguments"] == {"arg": 1}
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "my_tool"


def test_convert_dict_to_message_function_call_dict() -> None:
    msg = Messages(
        id=None,
        role=MessagesRole.ASSISTANT,
        content="",
        function_call={"name": "my_tool", "arguments": {"x": 2}},  # type: ignore[arg-type]
    )
    result = _convert_dict_to_message(msg)
    assert isinstance(result, AIMessage)
    assert result.additional_kwargs["function_call"]["name"] == "my_tool"
    assert len(result.tool_calls) == 1


# ---------------------------------------------------------------------------
# _convert_dict_to_message — functions_state_id + image regex
# ---------------------------------------------------------------------------


def test_convert_dict_to_message_functions_state_id_image() -> None:
    img_content = '<img src="uuid-123" fuse="true"/>Some text after'
    msg = Messages(
        id=None,
        role=MessagesRole.ASSISTANT,
        content=img_content,
        functions_state_id="state-1",
    )
    result = _convert_dict_to_message(msg)
    assert isinstance(result, AIMessage)
    assert result.additional_kwargs["functions_state_id"] == "state-1"
    assert result.additional_kwargs["image_uuid"] == "uuid-123"
    assert result.additional_kwargs["postfix_message"] == "Some text after"


# ---------------------------------------------------------------------------
# _convert_dict_to_message — functions_state_id + video regex
# ---------------------------------------------------------------------------


def test_convert_dict_to_message_functions_state_id_video() -> None:
    video_content = '<video cover="cover-uuid" src="video-uuid" fuse="true"/>After'
    msg = Messages(
        id=None,
        role=MessagesRole.ASSISTANT,
        content=video_content,
        functions_state_id="state-2",
    )
    result = _convert_dict_to_message(msg)
    assert isinstance(result, AIMessage)
    assert result.additional_kwargs["cover_uuid"] == "cover-uuid"
    assert result.additional_kwargs["video_uuid"] == "video-uuid"
    assert result.additional_kwargs["postfix_message"] == "After"


# ---------------------------------------------------------------------------
# _convert_dict_to_message — reasoning_content
# ---------------------------------------------------------------------------


def test_convert_dict_to_message_reasoning_content() -> None:
    msg = Messages(
        id=None,
        role=MessagesRole.ASSISTANT,
        content="answer",
    )
    msg.reasoning_content = "thinking..."  # type: ignore[attr-defined]
    result = _convert_dict_to_message(msg)
    assert isinstance(result, AIMessage)
    assert result.additional_kwargs["reasoning_content"] == "thinking..."


# ---------------------------------------------------------------------------
# _convert_dict_to_message — unknown role
# ---------------------------------------------------------------------------


def test_convert_dict_to_message_unknown_role() -> None:
    from unittest.mock import MagicMock

    msg = MagicMock()
    msg.role = "unknown_role"
    msg.content = "foo"
    msg.function_call = None
    msg.functions_state_id = None
    msg.reasoning_content = None
    with pytest.raises(TypeError, match="Got unknown role"):
        _convert_dict_to_message(msg)


# ---------------------------------------------------------------------------
# _convert_dict_to_message — function role
# ---------------------------------------------------------------------------


def test_convert_dict_to_message_function_role() -> None:
    msg = Messages(
        id=None, role=MessagesRole.FUNCTION, content='{"result": 1}', name="calc"
    )
    result = _convert_dict_to_message(msg)
    assert isinstance(result, FunctionMessage)
    assert result.name == "calc"


# ---------------------------------------------------------------------------
# _convert_message_to_dict — multiple tool_calls raises
# ---------------------------------------------------------------------------


def test_convert_message_to_dict_multiple_tool_calls_raises() -> None:
    msg = AIMessage(
        content="",
        tool_calls=[
            {"name": "tool1", "args": {"a": 1}, "id": "1"},
            {"name": "tool2", "args": {"b": 2}, "id": "2"},
        ],
    )
    with pytest.raises(ValueError, match="does not support multiple tool calls"):
        _convert_message_to_dict(msg)


# ---------------------------------------------------------------------------
# _convert_message_to_dict — AIMessage with single tool_call
# ---------------------------------------------------------------------------


def test_convert_message_to_dict_single_tool_call() -> None:
    msg = AIMessage(
        content="calling tool",
        tool_calls=[{"name": "my_tool", "args": {"x": 1}, "id": "tc-1"}],
    )
    result = _convert_message_to_dict(msg)
    assert result.role == MessagesRole.ASSISTANT
    assert result.function_call is not None
    assert result.function_call.name == "my_tool"
    assert result.function_call.arguments == {"x": 1}


# ---------------------------------------------------------------------------
# _convert_message_to_dict — AIMessage with function_call in additional_kwargs
# ---------------------------------------------------------------------------


def test_convert_message_to_dict_ai_with_additional_function_call() -> None:
    msg = AIMessage(
        content="",
        additional_kwargs={"function_call": {"name": "calc", "arguments": {"n": 5}}},
    )
    result = _convert_message_to_dict(msg)
    assert result.function_call is not None
    assert result.function_call.name == "calc"
    assert result.function_call.arguments == {"n": 5}


# ---------------------------------------------------------------------------
# _convert_message_to_dict — functions_state_id forwarded
# ---------------------------------------------------------------------------


def test_convert_message_to_dict_functions_state_id() -> None:
    msg = HumanMessage(
        content="test",
        additional_kwargs={"functions_state_id": "state-abc"},
    )
    result = _convert_message_to_dict(msg)
    assert result.functions_state_id == "state-abc"


# ---------------------------------------------------------------------------
# _convert_message_to_dict — unknown message type raises
# ---------------------------------------------------------------------------


def test_convert_message_to_dict_unknown_type() -> None:
    from langchain_core.messages import BaseMessage

    class CustomMessage(BaseMessage):
        type: str = "custom"

    msg = CustomMessage(content="foo")
    with pytest.raises(TypeError, match="Got unknown type"):
        _convert_message_to_dict(msg)
