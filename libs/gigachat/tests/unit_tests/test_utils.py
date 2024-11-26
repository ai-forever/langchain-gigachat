from typing import Generator

import pytest
import requests_mock
from langchain_core.prompts.prompt import PromptTemplate

from langchain_gigachat.tools.load_prompt import load_from_giga_hub


@pytest.fixture
def mock_requests_get() -> Generator:
    with requests_mock.Mocker() as mocker:
        mocker.get(
            "https://raw.githubusercontent.com/ai-forever/gigachain/master/hub/prompts/entertainment/meditation.yaml",
            text=(
                "input_variables: [background, topic]\n"
                "output_parser: null\n"
                "template: 'Create mediation for {topic} with {background}'\n"
                "template_format: f-string\n"
                "_type: prompt"
            ),
        )
        yield mocker


def test__load_from_giga_hub(mock_requests_get: Generator) -> None:
    template = load_from_giga_hub("lc://prompts/entertainment/meditation.yaml")
    assert isinstance(template, PromptTemplate)
    assert template.template == "Create mediation for {topic} with {background}"
    assert "background" in template.input_variables
    assert "topic" in template.input_variables
