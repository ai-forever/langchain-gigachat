import base64
import hashlib
from typing import Any, AsyncGenerator, Iterable, List, Tuple
from unittest.mock import MagicMock

import pytest
from gigachat.models import (
    ChatCompletion,
    ChatCompletionChunk,
    Choices,
    ChoicesChunk,
    Messages,
    MessagesChunk,
    MessagesRole,
    UploadedFile,
    Usage,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolArg
from pydantic import BaseModel, Field
from pytest_mock import MockerFixture
from typing_extensions import Annotated

from langchain_gigachat.chat_models.gigachat import (
    GigaChat,
    _convert_dict_to_message,
    _convert_message_to_dict,
)
from langchain_gigachat.tools.giga_tool import FewShotExamples, giga_tool
from tests.unit_tests.stubs import FakeAsyncCallbackHandler, FakeCallbackHandler


@pytest.fixture
def chat_completion() -> ChatCompletion:
    return ChatCompletion(
        choices=[
            Choices(
                message=Messages(
                    id=None, role=MessagesRole.ASSISTANT, content="Bar Baz"
                ),
                index=0,
                finish_reason="stop",
            )
        ],
        created=1678878333,
        model="GigaChat:v1.2.19.2",
        usage=Usage(prompt_tokens=18, completion_tokens=68, total_tokens=86),
        object="chat.completion",
    )


@pytest.fixture
def chat_completion_stream() -> List[ChatCompletionChunk]:
    return [
        ChatCompletionChunk(
            choices=[ChoicesChunk(delta=MessagesChunk(content="Bar Baz"), index=0)],
            created=1695802242,
            model="GigaChat:v1.2.19.2",
            object="chat.completion",
        ),
        ChatCompletionChunk(
            choices=[
                ChoicesChunk(
                    delta=MessagesChunk(content=" Stream"),
                    index=0,
                    finish_reason="stop",
                )
            ],
            created=1695802242,
            model="GigaChat:v1.2.19.2",
            object="chat.completion",
        ),
    ]


@pytest.fixture
def patch_gigachat(
    mocker: MockerFixture,
    chat_completion: ChatCompletion,
    chat_completion_stream: List[ChatCompletionChunk],
) -> None:
    mock = mocker.Mock()
    mock.chat.return_value = chat_completion
    mock.stream.return_value = chat_completion_stream

    mocker.patch("gigachat.GigaChat", return_value=mock)


@pytest.fixture
def patch_gigachat_achat(
    mocker: MockerFixture, chat_completion: ChatCompletion
) -> None:
    async def return_value_coroutine(value: Any) -> Any:
        return value

    mock = mocker.Mock()
    mock.achat.return_value = return_value_coroutine(chat_completion)

    mocker.patch("gigachat.GigaChat", return_value=mock)


@pytest.fixture
def patch_gigachat_astream(
    mocker: MockerFixture, chat_completion_stream: List[ChatCompletionChunk]
) -> None:
    async def return_value_async_generator(value: Iterable) -> AsyncGenerator:
        for chunk in value:
            yield chunk

    mock = mocker.Mock()
    mock.astream.return_value = return_value_async_generator(chat_completion_stream)

    mocker.patch("gigachat.GigaChat", return_value=mock)


@pytest.fixture
def patch_gigachat_upload_file(
    mocker: MockerFixture,
    chat_completion: ChatCompletion,
    chat_completion_stream: List[ChatCompletionChunk],
) -> MagicMock:
    mocker.patch("gigachat.GigaChat.chat", return_value=chat_completion)
    mocker.patch("gigachat.GigaChat.stream", return_value=chat_completion_stream)
    return mocker.patch(
        "gigachat.GigaChat.upload_file",
        return_value=UploadedFile(
            id="0", object="file", bytes=0, created_at=0, filename="", purpose=""
        ),
    )


@pytest.fixture
def patch_gigachat_aupload_file(
    mocker: MockerFixture,
    chat_completion: ChatCompletion,
    chat_completion_stream: List[ChatCompletionChunk],
) -> MagicMock:
    async_mock = mocker.AsyncMock()
    async_mock.return_value = chat_completion
    mocker.patch("gigachat.GigaChat.achat", side_effect=async_mock)

    async def return_value_async_generator(value: Iterable) -> AsyncGenerator:
        for chunk in value:
            yield chunk

    mocker.patch(
        "gigachat.GigaChat.astream",
        return_value=return_value_async_generator(chat_completion_stream),
    )
    async_mock = mocker.AsyncMock()
    async_mock.return_value = UploadedFile(
        id="0", object="file", bytes=0, created_at=0, filename="", purpose=""
    )
    return mocker.patch("gigachat.GigaChat.aupload_file", side_effect=async_mock)


UploadDialog = Tuple[List[HumanMessage], str, str]


@pytest.fixture
def upload_images_dialog() -> UploadDialog:
    image_1 = f"data:image/jpeg;base64,{base64.b64encode('123'.encode()).decode()}"
    image_2 = f"data:image/jpeg;base64,{base64.b64encode('124'.encode()).decode()}"
    hashed_1 = hashlib.sha256(image_1.encode()).hexdigest()
    hashed_2 = hashlib.sha256(image_2.encode()).hexdigest()
    return (
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": "1"},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_1},
                    },
                ]
            ),
            HumanMessage(
                content=[
                    {"type": "text", "text": "2"},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_2},
                    },
                ]
            ),
            HumanMessage(
                content=[
                    {"type": "text", "text": "3"},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_1},
                    },
                ]
            ),
        ],
        hashed_1,
        hashed_2,
    )


def test__convert_dict_to_message_system() -> None:
    message = Messages(id=None, role=MessagesRole.SYSTEM, content="foo")
    expected = SystemMessage(content="foo")
    actual = _convert_dict_to_message(message)
    assert actual == expected


def test__convert_dict_to_message_human() -> None:
    message = Messages(id=None, role=MessagesRole.USER, content="foo")
    expected = HumanMessage(content="foo")
    actual = _convert_dict_to_message(message)
    assert actual == expected


def test__convert_dict_to_message_ai() -> None:
    message = Messages(id=None, role=MessagesRole.ASSISTANT, content="foo")
    expected = AIMessage(content="foo")
    actual = _convert_dict_to_message(message)
    assert actual == expected


def test__convert_message_to_dict_system() -> None:
    message = SystemMessage(content="foo")
    expected = Messages(id=None, role=MessagesRole.SYSTEM, content="foo")
    actual = _convert_message_to_dict(message)
    assert actual == expected


def test__convert_message_to_dict_human() -> None:
    message = HumanMessage(content="foo")
    expected = Messages(id=None, role=MessagesRole.USER, content="foo")
    actual = _convert_message_to_dict(message)
    assert actual == expected


def test__convert_message_to_dict_ai() -> None:
    message = AIMessage(content="foo")
    expected = Messages(id=None, role=MessagesRole.ASSISTANT, content="foo")
    actual = _convert_message_to_dict(message)
    assert actual == expected


@pytest.mark.parametrize("pairs", (("{}", "{}"), ("abc", '"abc"'), ("[]", "[]")))
def test__convert_message_to_dict_function(pairs: Any) -> None:
    """Checks if string, that was not JSON was converted to JSON"""
    message = FunctionMessage(content=pairs[0], id="1", name="func")
    expected = Messages(id=None, role=MessagesRole.FUNCTION, content=pairs[1])

    actual = _convert_message_to_dict(message)

    assert actual == expected


@pytest.mark.parametrize(
    "role",
    (
        MessagesRole.SYSTEM,
        MessagesRole.USER,
        MessagesRole.ASSISTANT,
        MessagesRole.FUNCTION,
    ),
)
def test__convert_message_to_dict_chat(role: MessagesRole) -> None:
    message = ChatMessage(role=role, content="foo")
    expected = Messages(id=None, role=role, content="foo")
    actual = _convert_message_to_dict(message)
    assert actual == expected


def test_gigachat_predict(patch_gigachat: None) -> None:
    expected = "Bar Baz"

    llm = GigaChat()
    actual = llm.predict("bar")

    assert actual == expected


def test_gigachat_predict_stream(patch_gigachat: None) -> None:
    expected = "Bar Baz Stream"
    llm = GigaChat()
    callback_handler = FakeCallbackHandler()
    actual = llm.predict("bar", stream=True, callbacks=[callback_handler])
    assert actual == expected
    assert callback_handler.llm_streams == 2


@pytest.mark.asyncio()
async def test_gigachat_apredict(patch_gigachat_achat: None) -> None:
    expected = "Bar Baz"

    llm = GigaChat()
    actual = await llm.apredict("bar")

    assert actual == expected


@pytest.mark.asyncio()
async def test_gigachat_apredict_stream(patch_gigachat_astream: None) -> None:
    expected = "Bar Baz Stream"
    llm = GigaChat()
    callback_handler = FakeAsyncCallbackHandler()
    actual = await llm.apredict("bar", stream=True, callbacks=[callback_handler])
    assert actual == expected
    assert callback_handler.llm_streams == 2


def test_gigachat_stream(patch_gigachat: None) -> None:
    expected = [
        AIMessageChunk(content="Bar Baz", response_metadata={"x_headers": {}}, id=""),
        AIMessageChunk(
            content=" Stream", response_metadata={"finish_reason": "stop"}, id=""
        ),
    ]

    llm = GigaChat()
    actual = [chunk for chunk in llm.stream("bar")]
    for chunk in actual:
        chunk.id = ""
    assert actual == expected


@pytest.mark.asyncio()
async def test_gigachat_astream(patch_gigachat_astream: None) -> None:
    expected = [
        AIMessageChunk(content="Bar Baz", response_metadata={"x_headers": {}}, id=""),
        AIMessageChunk(
            content=" Stream", response_metadata={"finish_reason": "stop"}, id=""
        ),
    ]
    llm = GigaChat()
    actual = [chunk async for chunk in llm.astream("bar")]
    for chunk in actual:
        chunk.id = ""
    assert actual == expected


def test_gigachat_build_payload_existing_parameter() -> None:
    llm = GigaChat()
    payload = llm._build_payload([], max_tokens=1)
    assert payload.max_tokens == 1


def test_gigachat_build_payload_non_existing_parameter() -> None:
    llm = GigaChat()
    payload = llm._build_payload([], fake_parameter=1)
    assert getattr(payload, "fake_param", None) is None


async def test_gigachat_bind_without_description() -> None:
    class Person(BaseModel):
        name: str = Field(..., title="Name", description="The person's name")

    llm = GigaChat()
    with pytest.raises(ValueError):
        llm.bind_functions(functions=[Person], function_call="Person")
    with pytest.raises(ValueError):
        llm.bind_tools(tools=[Person], tool_choice="Person")


async def test_gigachat_bind_with_description() -> None:
    class Person(BaseModel):
        """Simple description"""

        name: str = Field(..., title="Name")

    llm = GigaChat()
    llm.bind_functions(functions=[Person], function_call="Person")
    llm.bind_tools(tools=[Person], tool_choice="Person")


@tool
def _test_tool(
    arg: str, config: RunnableConfig, injected: Annotated[str, InjectedToolArg]
) -> None:
    """Some description"""
    return


def test_gigachat_bind_with_injected_vars() -> None:
    llm = GigaChat().bind_tools(tools=[_test_tool])
    assert llm.kwargs["tools"][0]["function"]["parameters"]["required"] == ["arg"]  # type: ignore[attr-defined]


class SendSmsResult(BaseModel):
    status: str = Field(description="status")
    message: str = Field(description="message")


few_shot_examples = [
    {
        "request": "Sms 'hello' to 123",
        "params": {"recipient": "123", "message": "hello"},
    }
]


@giga_tool(few_shot_examples=few_shot_examples)
def _test_send_sms(
    arg: str, config: RunnableConfig, injected: Annotated[str, InjectedToolArg]
) -> SendSmsResult:
    """Sends SMS message"""
    return SendSmsResult(status="success", message="SMS sent")


def test_gigachat_bind_gigatool() -> None:
    llm = GigaChat().bind_tools(tools=[_test_send_sms])
    assert llm.kwargs["tools"][0]["function"]["few_shot_examples"] == few_shot_examples  # type: ignore[attr-defined]
    assert llm.kwargs["tools"][0]["function"]["return_parameters"] == {  # type: ignore[attr-defined]
        "properties": {
            "status": {"description": "status", "type": "string"},
            "message": {"description": "message", "type": "string"},
        },
        "required": ["status", "message"],
        "type": "object",
    }


class SomeResult(BaseModel):
    """My desc"""

    @staticmethod
    def few_shot_examples() -> FewShotExamples:
        return [
            {
                "request": "request example",
                "params": {"is_valid": 1, "description": "correct message"},
            }
        ]

    value: int = Field(description="some value")
    description: str = Field(description="some descriptin")


def test_structured_output() -> None:
    llm = GigaChat().with_structured_output(SomeResult)
    assert llm.steps[0].kwargs["function_call"] == {"name": "SomeResult"}  # type: ignore[attr-defined]
    assert llm.steps[0].kwargs["tools"][0]["function"] == {  # type: ignore[attr-defined]
        "name": "SomeResult",
        "description": "My desc",
        "parameters": {
            "description": "My desc",
            "properties": {
                "value": {"description": "some value", "type": "integer"},
                "description": {"description": "some descriptin", "type": "string"},
            },
            "required": ["value", "description"],
            "type": "object",
        },
        "return_parameters": None,
        "few_shot_examples": [
            {
                "request": "request example",
                "params": {"is_valid": 1, "description": "correct message"},
            }
        ],
    }


def test_ai_message_json_serialization(patch_gigachat: None) -> None:
    llm = GigaChat()
    response = llm.invoke("hello")
    response.model_dump_json()


def test_ai_upload_image(
    patch_gigachat_upload_file: MagicMock, upload_images_dialog: UploadDialog
) -> None:
    llm = GigaChat(auto_upload_images=True)
    dialog, hashed_1, hashed_2 = upload_images_dialog
    llm.invoke(dialog)
    assert len(llm._cached_images.keys()) == 2
    assert patch_gigachat_upload_file.call_count == 2
    assert patch_gigachat_upload_file.call_args_list[0][0][0][1] == b"123"
    assert patch_gigachat_upload_file.call_args_list[1][0][0][1] == b"124"
    assert hashed_1 in llm._cached_images
    assert hashed_2 in llm._cached_images


async def test_ai_aupload_image(
    patch_gigachat_aupload_file: MagicMock, upload_images_dialog: UploadDialog
) -> None:
    llm = GigaChat(auto_upload_images=True)
    dialog, hashed_1, hashed_2 = upload_images_dialog
    await llm.ainvoke(dialog)
    assert len(llm._cached_images.keys()) == 2
    assert patch_gigachat_aupload_file.call_count == 2
    assert patch_gigachat_aupload_file.call_args_list[0][0][0][1] == b"123"
    assert patch_gigachat_aupload_file.call_args_list[1][0][0][1] == b"124"
    assert hashed_1 in llm._cached_images
    assert hashed_2 in llm._cached_images


def test_ai_upload_image_stream(
    patch_gigachat_upload_file: MagicMock, upload_images_dialog: UploadDialog
) -> None:
    llm = GigaChat(auto_upload_images=True)
    dialog, hashed_1, hashed_2 = upload_images_dialog
    list(llm.stream(dialog))
    assert len(llm._cached_images.keys()) == 2
    assert patch_gigachat_upload_file.call_count == 2
    assert patch_gigachat_upload_file.call_args_list[0][0][0][1] == b"123"
    assert patch_gigachat_upload_file.call_args_list[1][0][0][1] == b"124"
    assert hashed_1 in llm._cached_images
    assert hashed_2 in llm._cached_images


async def test_ai_aupload_image_stream(
    patch_gigachat_aupload_file: MagicMock, upload_images_dialog: UploadDialog
) -> None:
    llm = GigaChat(auto_upload_images=True)
    dialog, hashed_1, hashed_2 = upload_images_dialog
    async for _ in llm.astream(dialog):
        pass
    assert len(llm._cached_images.keys()) == 2
    assert patch_gigachat_aupload_file.call_count == 2
    assert patch_gigachat_aupload_file.call_args_list[0][0][0][1] == b"123"
    assert patch_gigachat_aupload_file.call_args_list[1][0][0][1] == b"124"
    assert hashed_1 in llm._cached_images
    assert hashed_2 in llm._cached_images


def test_ai_upload_disabled_image(
    patch_gigachat_upload_file: MagicMock, upload_images_dialog: UploadDialog
) -> None:
    llm = GigaChat()
    dialog, hashed_1, hashed_2 = upload_images_dialog
    llm.invoke(dialog)
    assert len(llm._cached_images.keys()) == 0
    assert patch_gigachat_upload_file.call_count == 0


async def test_ai_aupload_disabled_image(
    patch_gigachat_aupload_file: MagicMock, upload_images_dialog: UploadDialog
) -> None:
    llm = GigaChat()
    dialog, hashed_1, hashed_2 = upload_images_dialog
    await llm.ainvoke(dialog)
    assert len(llm._cached_images.keys()) == 0
    assert patch_gigachat_aupload_file.call_count == 0


def test_ai_upload_image_disabled_stream(
    patch_gigachat_upload_file: MagicMock, upload_images_dialog: UploadDialog
) -> None:
    llm = GigaChat()
    dialog, hashed_1, hashed_2 = upload_images_dialog
    list(llm.stream(dialog))
    assert len(llm._cached_images.keys()) == 0
    assert patch_gigachat_upload_file.call_count == 0


async def test_ai_aupload_image_disabled_stream(
    patch_gigachat_aupload_file: MagicMock, upload_images_dialog: UploadDialog
) -> None:
    llm = GigaChat()
    dialog, _, _ = upload_images_dialog
    async for _ in llm.astream(dialog):
        pass
    assert len(llm._cached_images.keys()) == 0
    assert patch_gigachat_aupload_file.call_count == 0


def test__convert_message_with_attachments_to_dict_system(
    upload_images_dialog: UploadDialog,
) -> None:
    excepted = Messages(id=None, role=MessagesRole.USER, attachments=["1"], content="1")
    dialog, hashed_1, hashed_2 = upload_images_dialog
    actual = _convert_message_to_dict(dialog[0], {hashed_1: "1"})
    assert actual == excepted


def test__convert_message_with_attachments_no_cache_to_dict_system(
    upload_images_dialog: UploadDialog,
) -> None:
    excepted = Messages(id=None, role=MessagesRole.USER, content="1")
    dialog, hashed_1, hashed_2 = upload_images_dialog
    actual = _convert_message_to_dict(dialog[0])
    assert actual == excepted
