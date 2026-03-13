import base64
import hashlib
from typing import Any, AsyncGenerator, Iterable, List, Tuple, cast
from unittest.mock import MagicMock, patch

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
    get_text_and_images_from_content,
)


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
        usage=Usage(
            prompt_tokens=18,
            completion_tokens=68,
            total_tokens=86,
            precached_prompt_tokens=0,
        ),
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
    expected = Messages(
        id=None, role=MessagesRole.FUNCTION, content=pairs[1], name="func"
    )

    actual = _convert_message_to_dict(message)

    assert actual == expected


def test__convert_message_to_dict_tool_message_with_name() -> None:
    """ToolMessage with name forwards it to gm.Messages.name."""
    from langchain_core.messages import ToolMessage

    message = ToolMessage(content="result", tool_call_id="call-1", name="my_tool")
    actual = _convert_message_to_dict(message)

    assert actual.role == MessagesRole.FUNCTION
    assert actual.name == "my_tool"
    assert actual.content == '"result"'  # _validate_content wraps non-JSON strings


def test__convert_message_to_dict_tool_message_without_name() -> None:
    """ToolMessage without name leaves gm.Messages.name as None."""
    from langchain_core.messages import ToolMessage

    message = ToolMessage(content="result", tool_call_id="call-1")
    actual = _convert_message_to_dict(message)

    assert actual.role == MessagesRole.FUNCTION
    assert actual.name is None
    assert actual.content == '"result"'  # _validate_content wraps non-JSON strings


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


def test_gigachat_stream(patch_gigachat: None) -> None:
    expected = [
        AIMessageChunk(content="Bar Baz", response_metadata={"x_headers": {}}, id=""),
        AIMessageChunk(
            content=" Stream",
            response_metadata={
                "model_name": "GigaChat:v1.2.19.2",
                "finish_reason": "stop",
            },
            id="",
        ),
        AIMessageChunk(
            content="",
            additional_kwargs={},
            response_metadata={},
            id="",
            chunk_position="last",
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
            content=" Stream",
            response_metadata={
                "model_name": "GigaChat:v1.2.19.2",
                "finish_reason": "stop",
            },
            id="",
        ),
        AIMessageChunk(
            content="",
            additional_kwargs={},
            response_metadata={},
            id="",
            chunk_position="last",
        ),
    ]
    llm = GigaChat()
    actual = [chunk async for chunk in llm.astream("bar")]
    for chunk in actual:
        chunk.id = ""
    assert actual == expected


def test_gigachat_stream_callbacks(patch_gigachat: None) -> None:
    """Test that streaming triggers callbacks for each token."""
    from typing import Any

    from langchain_core.callbacks import BaseCallbackHandler

    class TokenCounter(BaseCallbackHandler):
        tokens: int = 0

        def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
            self.tokens += 1

    counter = TokenCounter()
    llm = GigaChat()
    list(llm.stream("bar", config={"callbacks": [counter]}))
    # 3 chunks: content, finish, and final empty chunk (chunk_position="last")
    assert counter.tokens == 3


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


async def test_gigachat_bind_functions_auto_and_none() -> None:
    class Person(BaseModel):
        """Simple description"""

        name: str = Field(..., title="Name")

    llm = GigaChat()
    bound_auto = llm.bind_functions(functions=[Person], function_call="auto")
    assert bound_auto.kwargs["function_call"] == "auto"  # type: ignore[attr-defined]

    bound_none = llm.bind_functions(functions=[Person], function_call="none")
    assert bound_none.kwargs["function_call"] == "none"  # type: ignore[attr-defined]


async def test_gigachat_bind_functions_force_name_among_many() -> None:
    class Person(BaseModel):
        """Person description"""

        name: str = Field(..., title="Name")

    class Animal(BaseModel):
        """Animal description"""

        species: str = Field(..., title="Species")

    llm = GigaChat()
    bound = llm.bind_functions(functions=[Person, Animal], function_call="Animal")
    assert bound.kwargs["function_call"] == {"name": "Animal"}  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="not found in provided functions"):
        llm.bind_functions(functions=[Person], function_call="Unknown")


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


@tool(extras={"few_shot_examples": few_shot_examples, "return_schema": SendSmsResult})
def _test_send_sms(
    arg: str, config: RunnableConfig, injected: Annotated[str, InjectedToolArg]
) -> str:
    """Sends SMS message"""
    return "SMS sent"


def test_gigachat_bind_standard_tool_with_extras() -> None:
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
    def few_shot_examples() -> list[dict[str, Any]]:
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


def test_structured_output_json() -> None:
    llm = GigaChat().with_structured_output(SomeResult.model_json_schema())
    assert llm.steps[0].kwargs["function_call"] == {"name": "SomeResult"}  # type: ignore[attr-defined]
    assert llm.steps[0].kwargs["tools"][0]["function"] is not None  # type: ignore[attr-defined]


def test_ai_message_json_serialization(patch_gigachat: None) -> None:
    llm = GigaChat()
    response = llm.invoke("hello")
    response.model_dump_json()


def test_ai_upload_image(
    patch_gigachat_upload_file: MagicMock, upload_images_dialog: UploadDialog
) -> None:
    llm = GigaChat(auto_upload_attachments=True)
    dialog, hashed_1, hashed_2 = upload_images_dialog
    llm.invoke(dialog)
    assert len(llm._cached_uploads.keys()) == 2
    assert patch_gigachat_upload_file.call_count == 2
    assert patch_gigachat_upload_file.call_args_list[0][0][0][1] == b"123"
    assert patch_gigachat_upload_file.call_args_list[1][0][0][1] == b"124"
    assert hashed_1 in llm._cached_uploads
    assert hashed_2 in llm._cached_uploads


async def test_ai_aupload_image(
    patch_gigachat_aupload_file: MagicMock, upload_images_dialog: UploadDialog
) -> None:
    llm = GigaChat(auto_upload_attachments=True)
    dialog, hashed_1, hashed_2 = upload_images_dialog
    await llm.ainvoke(dialog)
    assert len(llm._cached_uploads.keys()) == 2
    assert patch_gigachat_aupload_file.call_count == 2
    assert patch_gigachat_aupload_file.call_args_list[0][0][0][1] == b"123"
    assert patch_gigachat_aupload_file.call_args_list[1][0][0][1] == b"124"
    assert hashed_1 in llm._cached_uploads
    assert hashed_2 in llm._cached_uploads


def test_ai_upload_image_stream(
    patch_gigachat_upload_file: MagicMock, upload_images_dialog: UploadDialog
) -> None:
    llm = GigaChat(auto_upload_attachments=True)
    dialog, hashed_1, hashed_2 = upload_images_dialog
    list(llm.stream(dialog))
    assert len(llm._cached_uploads.keys()) == 2
    assert patch_gigachat_upload_file.call_count == 2
    assert patch_gigachat_upload_file.call_args_list[0][0][0][1] == b"123"
    assert patch_gigachat_upload_file.call_args_list[1][0][0][1] == b"124"
    assert hashed_1 in llm._cached_uploads
    assert hashed_2 in llm._cached_uploads


async def test_ai_aupload_image_stream(
    patch_gigachat_aupload_file: MagicMock, upload_images_dialog: UploadDialog
) -> None:
    llm = GigaChat(auto_upload_attachments=True)
    dialog, hashed_1, hashed_2 = upload_images_dialog
    async for _ in llm.astream(dialog):
        pass
    assert len(llm._cached_uploads.keys()) == 2
    assert patch_gigachat_aupload_file.call_count == 2
    assert patch_gigachat_aupload_file.call_args_list[0][0][0][1] == b"123"
    assert patch_gigachat_aupload_file.call_args_list[1][0][0][1] == b"124"
    assert hashed_1 in llm._cached_uploads
    assert hashed_2 in llm._cached_uploads


def test_ai_upload_disabled_image(
    patch_gigachat_upload_file: MagicMock, upload_images_dialog: UploadDialog
) -> None:
    llm = GigaChat()
    dialog, hashed_1, hashed_2 = upload_images_dialog
    llm.invoke(dialog)
    assert len(llm._cached_uploads.keys()) == 0
    assert patch_gigachat_upload_file.call_count == 0


async def test_ai_aupload_disabled_image(
    patch_gigachat_aupload_file: MagicMock, upload_images_dialog: UploadDialog
) -> None:
    llm = GigaChat()
    dialog, hashed_1, hashed_2 = upload_images_dialog
    await llm.ainvoke(dialog)
    assert len(llm._cached_uploads.keys()) == 0
    assert patch_gigachat_aupload_file.call_count == 0


def test_ai_upload_image_disabled_stream(
    patch_gigachat_upload_file: MagicMock, upload_images_dialog: UploadDialog
) -> None:
    llm = GigaChat()
    dialog, hashed_1, hashed_2 = upload_images_dialog
    list(llm.stream(dialog))
    assert len(llm._cached_uploads.keys()) == 0
    assert patch_gigachat_upload_file.call_count == 0


async def test_ai_aupload_image_disabled_stream(
    patch_gigachat_aupload_file: MagicMock, upload_images_dialog: UploadDialog
) -> None:
    llm = GigaChat()
    dialog, _, _ = upload_images_dialog
    async for _ in llm.astream(dialog):
        pass
    assert len(llm._cached_uploads.keys()) == 0
    assert patch_gigachat_aupload_file.call_count == 0


def test_ai_upload_image_per_instance_cache(
    patch_gigachat_upload_file: MagicMock, upload_images_dialog: UploadDialog
) -> None:
    """Two GigaChat instances have independent image caches (no cross-tenant)."""
    llm1 = GigaChat(auto_upload_attachments=True)
    llm2 = GigaChat(auto_upload_attachments=True)
    dialog, hashed_1, hashed_2 = upload_images_dialog
    llm1.invoke(dialog)
    assert len(llm1._cached_uploads) == 2
    assert hashed_1 in llm1._cached_uploads and hashed_2 in llm1._cached_uploads
    assert len(llm2._cached_uploads) == 0
    llm2.invoke(dialog)
    assert len(llm2._cached_uploads) == 2
    assert llm1._cached_uploads is not llm2._cached_uploads


def test_ai_upload_image_cache_eviction(
    patch_gigachat_upload_file: MagicMock,
) -> None:
    """When cache is at max size, adding a new image evicts oldest entry (FIFO)."""
    with patch(
        "langchain_gigachat.chat_models.gigachat.DEFAULT_IMAGE_CACHE_MAX_SIZE",
        2,
    ):
        llm = GigaChat(auto_upload_attachments=True)
        img_a = f"data:image/png;base64,{base64.b64encode(b'aaa').decode()}"
        img_b = f"data:image/png;base64,{base64.b64encode(b'bbb').decode()}"
        img_c = f"data:image/png;base64,{base64.b64encode(b'ccc').decode()}"
        hash_a = hashlib.sha256(img_a.encode()).hexdigest()
        hash_b = hashlib.sha256(img_b.encode()).hexdigest()
        hash_c = hashlib.sha256(img_c.encode()).hexdigest()
        msg_a = [{"type": "image_url", "image_url": {"url": img_a}}]
        msg_b = [{"type": "image_url", "image_url": {"url": img_b}}]
        msg_c = [{"type": "image_url", "image_url": {"url": img_c}}]
        llm.invoke([HumanMessage(content=cast(Any, msg_a))])
        assert len(llm._cached_uploads) == 1 and hash_a in llm._cached_uploads
        llm.invoke([HumanMessage(content=cast(Any, msg_b))])
        assert len(llm._cached_uploads) == 2
        assert hash_a in llm._cached_uploads and hash_b in llm._cached_uploads
        llm.invoke([HumanMessage(content=cast(Any, msg_c))])
        assert len(llm._cached_uploads) == 2
        assert hash_a not in llm._cached_uploads
        assert hash_b in llm._cached_uploads and hash_c in llm._cached_uploads


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


def test_get_text_and_images_from_content_audio_url_giga_id() -> None:
    """audio_url with giga_id is collected into attachments."""
    content: list[str | dict[str, Any]] = [
        {"type": "text", "text": "Listen"},
        {"type": "audio_url", "audio_url": {"giga_id": "audio-123"}},
    ]
    text, attachments = get_text_and_images_from_content(content, {})
    assert text == "Listen"
    assert attachments == ["audio-123"]


def test_get_text_and_images_from_content_document_url_giga_id() -> None:
    """document_url with giga_id is collected into attachments."""
    content: list[str | dict[str, Any]] = [
        {"type": "text", "text": "Read"},
        {"type": "document_url", "document_url": {"giga_id": "doc-456"}},
    ]
    text, attachments = get_text_and_images_from_content(content, {})
    assert text == "Read"
    assert attachments == ["doc-456"]


def test_get_text_and_images_from_content_audio_url_cached() -> None:
    """audio_url with data URL resolved from cache."""
    url = "data:audio/mp3;base64,YWJj"
    content: list[str | dict[str, Any]] = [
        {"type": "audio_url", "audio_url": {"url": url}}
    ]
    hashed = hashlib.sha256(url.encode()).hexdigest()
    cache = {hashed: "file-id-789"}
    text, attachments = get_text_and_images_from_content(content, cache)
    assert text == ""
    assert attachments == ["file-id-789"]


def test_get_text_and_images_from_content_mixed_attachments() -> None:
    """Mixed text, image/audio/document_url yield correct text and attachments."""
    content: list[str | dict[str, Any]] = [
        {"type": "text", "text": "Summary"},
        {"type": "image_url", "image_url": {"giga_id": "img-1"}},
        {"type": "audio_url", "audio_url": {"giga_id": "aud-1"}},
        {"type": "document_url", "document_url": {"giga_id": "doc-1"}},
    ]
    text, attachments = get_text_and_images_from_content(content, {})
    assert text == "Summary"
    assert attachments == ["img-1", "aud-1", "doc-1"]


def test_get_text_and_images_from_content_standard_blocks() -> None:
    """Standard LangChain content_blocks (image/audio/file + file_id) are supported."""
    content: list[str | dict[str, Any]] = [
        {"type": "text", "text": "Describe"},
        {"type": "image", "file_id": "giga-img-1"},
        {"type": "audio", "file_id": "giga-aud-1"},
        {"type": "file", "file_id": "giga-doc-1"},
    ]
    text, attachments = get_text_and_images_from_content(content, {})
    assert text == "Describe"
    assert attachments == ["giga-img-1", "giga-aud-1", "giga-doc-1"]


def test_get_text_and_images_from_content_deduplicates_attachment_ids() -> None:
    """Attachment IDs are deduplicated when explicit IDs match cached URLs."""
    first_url = "data:image/png;base64,Zmlyc3Q="
    second_url = "data:application/pdf;base64,c2Vjb25k"
    cache = {
        hashlib.sha256(first_url.encode()).hexdigest(): "file-1",
        hashlib.sha256(second_url.encode()).hexdigest(): "file-2",
    }
    content: list[str | dict[str, Any]] = [
        {"type": "image_url", "image_url": {"giga_id": "file-1", "url": first_url}},
        {"type": "file", "file_id": "file-2", "url": second_url},
        {"type": "audio_url", "audio_url": {"url": first_url}},
    ]

    text, attachments = get_text_and_images_from_content(content, cache)

    assert text == ""
    assert attachments == ["file-1", "file-2"]


def test_convert_message_to_dict_with_audio_and_document_attachments() -> None:
    """HumanMessage with audio/document_url (giga_id) gets attachments in payload."""
    msg = HumanMessage(
        content=[
            {"type": "text", "text": "Process this"},
            {"type": "audio_url", "audio_url": {"giga_id": "a1"}},
            {"type": "document_url", "document_url": {"giga_id": "d1"}},
        ]
    )
    actual = _convert_message_to_dict(msg)
    assert actual.role == MessagesRole.USER
    assert actual.content == "Process this"
    assert actual.attachments == ["a1", "d1"]


def test_auto_upload_attachments_audio_url(
    patch_gigachat_upload_file: MagicMock,
) -> None:
    """With auto_upload_attachments=True, audio_url data URL is uploaded and cached."""
    audio_data = "data:audio/mp3;base64," + base64.b64encode(b"audio bytes").decode()
    msg = HumanMessage(
        content=[{"type": "audio_url", "audio_url": {"url": audio_data}}]
    )
    llm = GigaChat(auto_upload_attachments=True)
    llm.invoke([msg])
    assert patch_gigachat_upload_file.call_count == 1
    hashed = hashlib.sha256(audio_data.encode()).hexdigest()
    assert hashed in llm._cached_uploads
    assert llm._cached_uploads[hashed] == "0"


async def test_auto_upload_attachments_document_url(
    patch_gigachat_aupload_file: MagicMock,
) -> None:
    """With auto_upload_attachments=True, document_url data URL is uploaded/cached."""
    doc_data = "data:application/pdf;base64," + base64.b64encode(b"pdf bytes").decode()
    msg = HumanMessage(
        content=[{"type": "document_url", "document_url": {"url": doc_data}}]
    )
    llm = GigaChat(auto_upload_attachments=True)
    await llm.ainvoke([msg])
    assert patch_gigachat_aupload_file.call_count == 1
    hashed = hashlib.sha256(doc_data.encode()).hexdigest()
    assert hashed in llm._cached_uploads


class PersonTool(BaseModel):
    """Get person info"""

    name: str = Field(description="Person name")


def test_bind_tools_any_tool_choice_raises_by_default() -> None:
    """tool_choice='any' should raise ValueError by default."""
    llm = GigaChat()
    with pytest.raises(ValueError, match="does not support tool_choice='any'"):
        llm.bind_tools(tools=[PersonTool], tool_choice="any")


def test_bind_tools_any_tool_choice_with_fallback_enabled() -> None:
    """tool_choice='any' should fallback to 'auto' with warning when enabled."""
    import warnings

    llm = GigaChat(allow_any_tool_choice_fallback=True)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        bound = llm.bind_tools(tools=[PersonTool], tool_choice="any")
        assert len(w) == 1
        assert "does not support tool_choice='any'" in str(w[0].message)
        assert "Using 'auto' instead" in str(w[0].message)
    # Verify fallback to "auto"
    assert bound.kwargs["function_call"] == "auto"  # type: ignore[attr-defined]


def test_bind_tools_auto_tool_choice_works() -> None:
    """tool_choice='auto' should work without issues."""
    llm = GigaChat()
    bound = llm.bind_tools(tools=[PersonTool], tool_choice="auto")
    assert bound.kwargs["function_call"] == "auto"  # type: ignore[attr-defined]


def test_bind_tools_specific_tool_choice_works() -> None:
    """tool_choice with specific tool name should work."""
    llm = GigaChat()
    bound = llm.bind_tools(tools=[PersonTool], tool_choice="PersonTool")
    assert bound.kwargs["function_call"] == {"name": "PersonTool"}  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Connection settings (2.18)
# ---------------------------------------------------------------------------


def test_connection_settings_defaults() -> None:
    """Connection settings default to None (SDK defaults apply)."""
    llm = GigaChat()
    assert llm.max_retries is None
    assert llm.max_connections is None
    assert llm.retry_backoff_factor is None
    assert llm.retry_on_status_codes is None


def test_connection_settings_explicit_values() -> None:
    """Connection settings can be set explicitly."""
    llm = GigaChat(
        max_retries=3,
        max_connections=10,
        retry_backoff_factor=1.0,
        retry_on_status_codes=(429, 503),
    )
    assert llm.max_retries == 3
    assert llm.max_connections == 10
    assert llm.retry_backoff_factor == 1.0
    assert llm.retry_on_status_codes == (429, 503)


def test_connection_settings_forwarded_to_sdk(mocker: MockerFixture) -> None:
    """Connection settings are passed to the SDK client constructor."""
    sdk_mock = mocker.patch("gigachat.GigaChat")

    llm = GigaChat(
        max_retries=2,
        max_connections=5,
        retry_backoff_factor=0.25,
        retry_on_status_codes=(500, 502),
    )
    # Access _client to trigger lazy initialization
    _ = llm._client

    sdk_mock.assert_called_once()
    call_kwargs = sdk_mock.call_args[1]
    assert call_kwargs["max_retries"] == 2
    assert call_kwargs["max_connections"] == 5
    assert call_kwargs["retry_backoff_factor"] == 0.25
    assert call_kwargs["retry_on_status_codes"] == (500, 502)


def test_connection_settings_none_forwarded_to_sdk(mocker: MockerFixture) -> None:
    """When connection settings are None, None is forwarded to the SDK."""
    sdk_mock = mocker.patch("gigachat.GigaChat")

    llm = GigaChat()
    _ = llm._client

    sdk_mock.assert_called_once()
    call_kwargs = sdk_mock.call_args[1]
    assert call_kwargs["max_retries"] is None
    assert call_kwargs["max_connections"] is None
    assert call_kwargs["retry_backoff_factor"] is None
    assert call_kwargs["retry_on_status_codes"] is None
