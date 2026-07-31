"""Tests for with_structured_output and bind_tools edge cases."""

from typing import Any, Dict

import gigachat.models as gm
import pytest
from langchain_core.messages import HumanMessage
from langchain_core.prompt_values import ChatPromptValue, StringPromptValue
from langchain_core.runnables import RunnableBinding, RunnableSequence
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
    with pytest.raises(
        ValueError,
        match="'format_instructions'",
    ):
        llm.with_structured_output(Answer, method="bad_method")


def test_structured_output_extra_kwargs(llm: GigaChat) -> None:
    with pytest.raises(ValueError, match="unsupported arguments"):
        llm.with_structured_output(Answer, unknown_key=True)


def test_structured_output_strict_with_wrong_method(llm: GigaChat) -> None:
    with pytest.raises(
        ValueError,
        match="`strict` is only supported with method='json_schema'",
    ):
        llm.with_structured_output(Answer, method="function_calling", strict=True)


def test_structured_output_strict_with_format_instructions(llm: GigaChat) -> None:
    with pytest.raises(
        ValueError,
        match="`strict` is only supported with method='json_schema'",
    ):
        llm.with_structured_output(Answer, method="format_instructions", strict=True)


# ---------------------------------------------------------------------------
# with_structured_output — function_calling (default)
# ---------------------------------------------------------------------------


def test_structured_output_function_calling_pydantic_default(llm: GigaChat) -> None:
    chain = llm.with_structured_output(Answer)
    assert isinstance(chain, RunnableSequence)
    bound = chain.steps[0]
    assert isinstance(bound, RunnableBinding)

    assert bound.kwargs["function_call"] == {"name": "Answer"}
    assert bound.kwargs["tools"][0]["function"]["name"] == "Answer"


# ---------------------------------------------------------------------------
# with_structured_output — json_schema (native API, explicit)
# ---------------------------------------------------------------------------


def test_structured_output_json_schema_explicit(llm: GigaChat) -> None:
    chain = llm.with_structured_output(Answer, method="json_schema", strict=False)
    assert isinstance(chain, RunnableSequence)
    bound = chain.steps[0]
    assert isinstance(bound, RunnableBinding)

    response_format = bound.kwargs["response_format"]
    assert isinstance(response_format, gm.JsonSchemaResponseFormat)
    assert response_format.strict is False


def test_structured_output_json_schema_dict(llm: GigaChat) -> None:
    schema: Dict[str, Any] = {
        "title": "Answer",
        "type": "object",
        "properties": {"value": {"type": "integer"}},
    }
    chain = llm.with_structured_output(schema, method="json_schema")
    assert isinstance(chain, RunnableSequence)
    bound = chain.steps[0]
    assert isinstance(bound, RunnableBinding)

    response_format = bound.kwargs["response_format"]
    assert isinstance(response_format, gm.JsonSchemaResponseFormat)
    assert response_format.schema_["properties"]["value"]["type"] == "integer"


def test_structured_output_json_schema_include_raw(llm: GigaChat) -> None:
    chain = llm.with_structured_output(
        Answer,
        method="json_schema",
        include_raw=True,
    )
    assert chain is not None


def test_structured_output_json_schema_invalid_schema(llm: GigaChat) -> None:
    class NotAModel:
        pass

    with pytest.raises(TypeError, match="dict or a pydantic.BaseModel"):
        llm.with_structured_output(NotAModel, method="json_schema")


# ---------------------------------------------------------------------------
# with_structured_output — json_mode (deprecated legacy)
# ---------------------------------------------------------------------------


def test_structured_output_json_mode_pydantic(llm: GigaChat) -> None:
    with pytest.warns(DeprecationWarning, match="json_mode.*deprecated"):
        chain = llm.with_structured_output(Answer, method="json_mode")
    assert chain is not None


def test_structured_output_json_mode_dict(llm: GigaChat) -> None:
    schema: Dict[str, Any] = {
        "title": "Answer",
        "type": "object",
        "properties": {"value": {"type": "integer"}},
    }
    with pytest.warns(DeprecationWarning, match="json_mode.*deprecated"):
        chain = llm.with_structured_output(schema, method="json_mode")
    assert chain is not None


def test_structured_output_json_mode_without_schema_uses_exact_wire_format(
    llm: GigaChat,
) -> None:
    chain = llm.with_structured_output(None, method="json_mode")

    assert isinstance(chain, RunnableSequence)
    bound = chain.steps[0]
    assert isinstance(bound, RunnableBinding)
    assert bound.kwargs["response_format"] == {"type": "json_schema"}

    payload = llm._build_payload(
        [HumanMessage(content="Return JSON")],
        response_format=bound.kwargs["response_format"],
    )
    assert payload.model_dump(exclude_none=True, by_alias=True)["response_format"] == {
        "type": "json_schema"
    }


def test_structured_output_json_mode_without_schema_rejects_strict(
    llm: GigaChat,
) -> None:
    with pytest.raises(
        ValueError,
        match="`strict` is only supported with method='json_schema'",
    ):
        llm.with_structured_output(None, method="json_mode", strict=False)


@pytest.mark.parametrize(
    "method",
    ["function_calling", "json_schema", "format_instructions"],
)
def test_structured_output_none_requires_json_mode(
    llm: GigaChat,
    method: str,
) -> None:
    with pytest.raises(TypeError, match=rf"method='{method}' requires a schema"):
        llm.with_structured_output(None, method=method)


# ---------------------------------------------------------------------------
# with_structured_output — format_instructions (prompt-based legacy)
# ---------------------------------------------------------------------------


def test_structured_output_format_instructions_pydantic(llm: GigaChat) -> None:
    chain = llm.with_structured_output(Answer, method="format_instructions")
    assert isinstance(chain, RunnableSequence)
    rendered = chain.steps[0].invoke(input="Hello")

    assert rendered.startswith("Hello\n\nSTRICT OUTPUT FORMAT:")
    assert "The output should be formatted as a JSON instance" in rendered
    assert '"value"' in rendered
    assert "Return a JSON object." not in rendered


def test_structured_output_format_instructions_dict_schema(llm: GigaChat) -> None:
    schema = Answer.model_json_schema()
    chain = llm.with_structured_output(schema, method="format_instructions")
    assert isinstance(chain, RunnableSequence)
    rendered = chain.steps[0].invoke(input="Hello")

    assert rendered.startswith("Hello\n\nSTRICT OUTPUT FORMAT:")
    assert "The output should be formatted as a JSON instance" in rendered
    assert '"value"' in rendered
    assert "Return a JSON object." not in rendered


def test_structured_output_format_instructions_prompt_value(llm: GigaChat) -> None:
    chain = llm.with_structured_output(Answer, method="format_instructions")
    assert isinstance(chain, RunnableSequence)
    rendered = chain.steps[0].invoke(input=StringPromptValue(text="Hello"))

    assert isinstance(rendered, ChatPromptValue)
    assert rendered.messages[0].content == "Hello"
    assert "STRICT OUTPUT FORMAT:" in str(rendered.messages[-1].content)


def test_structured_output_format_instructions_unknown_schema_type(
    llm: GigaChat,
) -> None:
    bad_schema: Any = "not-a-schema"
    with pytest.raises(TypeError, match="Pydantic class or a dict"):
        llm.with_structured_output(bad_schema, method="format_instructions")


# ---------------------------------------------------------------------------
# with_structured_output — function_calling + include_raw
# ---------------------------------------------------------------------------


def test_structured_output_function_calling_include_raw(llm: GigaChat) -> None:
    chain = llm.with_structured_output(
        Answer,
        method="function_calling",
        include_raw=True,
    )
    assert chain is not None


def test_structured_output_function_calling_include_raw_dict(llm: GigaChat) -> None:
    schema = Answer.model_json_schema()
    chain = llm.with_structured_output(
        schema,
        method="function_calling",
        include_raw=True,
    )
    assert chain is not None


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
