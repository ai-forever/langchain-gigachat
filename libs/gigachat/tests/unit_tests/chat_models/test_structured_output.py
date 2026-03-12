"""Tests for with_structured_output and bind_tools edge cases."""

from typing import Any, Dict

import pytest
from pydantic import BaseModel, Field
from pytest_mock import MockerFixture

from langchain_gigachat.chat_models.gigachat import GigaChat


class Answer(BaseModel):
    """Structured answer"""

    value: int = Field(description="the value")


@pytest.fixture()
def llm(mocker: MockerFixture) -> GigaChat:
    mocker.patch("gigachat.GigaChat")
    return GigaChat()


# ---------------------------------------------------------------------------
# with_structured_output — method validation
# ---------------------------------------------------------------------------


def test_structured_output_invalid_method(llm: GigaChat) -> None:
    with pytest.raises(ValueError, match="Unrecognized method"):
        llm.with_structured_output(Answer, method="bad_method")


def test_structured_output_extra_kwargs(llm: GigaChat) -> None:
    with pytest.raises(ValueError, match="unsupported arguments"):
        llm.with_structured_output(Answer, unknown_key=True)


# ---------------------------------------------------------------------------
# with_structured_output — json_mode (Pydantic)
# ---------------------------------------------------------------------------


def test_structured_output_json_mode_pydantic(llm: GigaChat) -> None:
    chain = llm.with_structured_output(Answer, method="json_mode")
    assert chain is not None


def test_structured_output_json_mode_dict(llm: GigaChat) -> None:
    schema: Dict[str, Any] = {
        "title": "Answer",
        "type": "object",
        "properties": {"value": {"type": "integer"}},
    }
    chain = llm.with_structured_output(schema, method="json_mode")
    assert chain is not None


# ---------------------------------------------------------------------------
# with_structured_output — function_calling + include_raw
# ---------------------------------------------------------------------------


def test_structured_output_include_raw(llm: GigaChat) -> None:
    chain = llm.with_structured_output(Answer, include_raw=True)
    assert chain is not None


def test_structured_output_include_raw_dict(llm: GigaChat) -> None:
    schema = Answer.model_json_schema()
    chain = llm.with_structured_output(schema, include_raw=True)
    assert chain is not None


def test_structured_output_function_calling_none_schema(llm: GigaChat) -> None:
    with pytest.raises(ValueError, match="schema must be specified"):
        llm.with_structured_output(None, method="function_calling")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# bind_tools — tool_choice edge cases
# ---------------------------------------------------------------------------


class MyTool(BaseModel):
    """My tool description"""

    x: int = Field(description="param")


def test_bind_tools_bool_true(llm: GigaChat) -> None:
    bound = llm.bind_tools([MyTool], tool_choice=True)
    assert bound.kwargs["function_call"] == {"name": "MyTool"}  # type: ignore[attr-defined]


def test_bind_tools_bool_true_raw_json_schema(llm: GigaChat) -> None:
    schema: Dict[str, Any] = {
        "title": "Answer",
        "description": "Answer schema",
        "type": "object",
        "properties": {"x": {"type": "string"}},
        "required": ["x"],
    }
    bound = llm.bind_tools([schema], tool_choice=True)
    assert bound.kwargs["function_call"] == {"name": "Answer"}  # type: ignore[attr-defined]
    assert bound.kwargs["tools"][0]["function"]["name"] == "Answer"  # type: ignore[attr-defined]
    assert bound.kwargs["tools"][0]["function"]["title"] == "Answer"  # type: ignore[attr-defined]


def test_bind_tools_bool_true_wrapped_title_only_tool(llm: GigaChat) -> None:
    tool: Dict[str, Any] = {
        "type": "function",
        "function": {
            "title": "Answer",
            "description": "Answer schema",
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        },
    }
    bound = llm.bind_tools([tool], tool_choice=True)
    assert bound.kwargs["function_call"] == {"name": "Answer"}  # type: ignore[attr-defined]


def test_bind_tools_bool_true_no_tools(llm: GigaChat) -> None:
    with pytest.raises(ValueError, match="can not be bool if tools are empty"):
        llm.bind_tools([], tool_choice=True)


def test_bind_tools_dict_passthrough(llm: GigaChat) -> None:
    bound = llm.bind_tools([MyTool], tool_choice={"name": "MyTool"})
    assert bound.kwargs["function_call"] == {"name": "MyTool"}  # type: ignore[attr-defined]


def test_bind_tools_none_choice(llm: GigaChat) -> None:
    bound = llm.bind_tools([MyTool], tool_choice="none")
    assert bound.kwargs["function_call"] == "none"  # type: ignore[attr-defined]


def test_bind_tools_unrecognized_type(llm: GigaChat) -> None:
    with pytest.raises(ValueError, match="Unrecognized tool_choice type"):
        llm.bind_tools([MyTool], tool_choice=42)  # type: ignore[arg-type]
