from __future__ import annotations

import base64
import copy
import hashlib
import json
import logging
import re
from mimetypes import guess_extension
from operator import itemgetter
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
    Union,
    overload,
)
from uuid import uuid4

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolCall,
    ToolCallChunk,
    ToolMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.output_parsers import (
    JsonOutputKeyToolsParser,
    JsonOutputParser,
    PydanticOutputParser,
    PydanticToolsParser,
)
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils.pydantic import is_basemodel_subclass, pre_init
from pydantic import BaseModel

from langchain_gigachat.chat_models.base_gigachat import _BaseGigaChat
from langchain_gigachat.utils.function_calling import (
    convert_to_gigachat_function,
    convert_to_gigachat_tool,
)

if TYPE_CHECKING:
    import gigachat.models as gm

logger = logging.getLogger(__name__)

IMAGE_SEARCH_REGEX = re.compile(
    r'<img\ssrc="(?P<UUID>.+?)"\sfuse=".+?"/>(?P<postfix>.+)?'
)
VIDEO_SEARCH_REGEX = re.compile(
    r'<video\scover="(?P<cover_UUID>.+?)"\ssrc="(?P<UUID>.+?)"\sfuse="true"/>(?P<postfix>.+)?'  # noqa
)


def _validate_content(content: Any) -> Any:
    """If content is string, but not JSON - convert string to json-string"""
    if isinstance(content, str):
        try:
            json.loads(content)
        except ValueError:
            content = json.dumps(content, ensure_ascii=False)
    return content


def _convert_dict_to_message(message: gm.Messages) -> BaseMessage:
    from gigachat.models import FunctionCall, MessagesRole

    additional_kwargs: Dict = {}
    tool_calls = []
    if function_call := message.function_call:
        if isinstance(function_call, FunctionCall):
            additional_kwargs["function_call"] = dict(function_call)
        elif isinstance(function_call, dict):
            additional_kwargs["function_call"] = function_call
        if additional_kwargs.get("function_call") is not None:
            tool_calls = [
                ToolCall(
                    name=additional_kwargs["function_call"]["name"],
                    args=additional_kwargs["function_call"]["arguments"],
                    id=str(uuid4()),
                )
            ]
    if message.functions_state_id:
        additional_kwargs["functions_state_id"] = message.functions_state_id
        match = IMAGE_SEARCH_REGEX.search(message.content)
        if match:
            additional_kwargs["image_uuid"] = match.group("UUID")
            additional_kwargs["postfix_message"] = match.group("postfix")
        match = VIDEO_SEARCH_REGEX.search(message.content)
        if match:
            additional_kwargs["cover_uuid"] = match.group("cover_UUID")
            additional_kwargs["video_uuid"] = match.group("UUID")
            additional_kwargs["postfix_message"] = match.group("postfix")
    if message.role == MessagesRole.SYSTEM:
        return SystemMessage(content=message.content)
    elif message.role == MessagesRole.USER:
        return HumanMessage(content=message.content)
    elif message.role == MessagesRole.ASSISTANT:
        return AIMessage(
            content=message.content,
            additional_kwargs=additional_kwargs,
            tool_calls=tool_calls,
        )
    elif message.role == MessagesRole.FUNCTION:
        return FunctionMessage(
            name=message.name or "", content=_validate_content(message.content)
        )
    else:
        raise TypeError(f"Got unknown role {message.role} {message}")


def get_text_and_images_from_content(
    content: list[Union[str, dict]], cached_images: Dict[str, str]
) -> Tuple[str, List[str]]:
    text_parts = []
    attachments = []
    for content_part in content:
        if isinstance(content_part, str):
            text_parts.append(content_part)
        elif isinstance(content_part, dict):
            if content_part.get("type") == "text":
                text_parts.append(content_part["text"])
            elif content_part.get("type") == "image_url":
                image_data = content_part["image_url"]
                if not isinstance(image_data, dict):
                    continue
                if "giga_id" in content_part["image_url"]:
                    attachments.append(content_part["image_url"].get("giga_id"))
                image_url = content_part["image_url"].get("url")
                hashed = hashlib.sha256(image_url.encode()).hexdigest()
                if hashed in cached_images:
                    attachments.append(cached_images[hashed])
    return " ".join(text_parts), attachments


def _convert_message_to_dict(
    message: BaseMessage, cached_images: Optional[Dict[str, str]] = None
) -> gm.Messages:
    from gigachat.models import Messages, MessagesRole

    kwargs = {}
    if cached_images is None:
        cached_images = {}

    if isinstance(message.content, list):
        content, attachments = get_text_and_images_from_content(
            message.content, cached_images
        )
    else:
        content, attachments = message.content, []

    attachments += message.additional_kwargs.get("attachments", [])
    functions_state_id = message.additional_kwargs.get("functions_state_id", None)
    if functions_state_id:
        kwargs["functions_state_id"] = functions_state_id

    if isinstance(message, SystemMessage):
        kwargs["role"] = MessagesRole.SYSTEM
        kwargs["content"] = content
    elif isinstance(message, HumanMessage):
        kwargs["role"] = MessagesRole.USER
        if attachments:
            kwargs["attachments"] = attachments
        kwargs["content"] = content
    elif isinstance(message, AIMessage):
        if tool_calls := getattr(message, "tool_calls", None):
            function_call = copy.deepcopy(tool_calls[0])

            if "args" in function_call:
                function_call["arguments"] = function_call.pop("args")
        else:
            function_call = message.additional_kwargs.get("function_call", None)
        kwargs["role"] = MessagesRole.ASSISTANT
        kwargs["content"] = content
        kwargs["function_call"] = function_call
    elif isinstance(message, ChatMessage):
        kwargs["role"] = message.role
        kwargs["content"] = content
    elif isinstance(message, FunctionMessage):
        kwargs["role"] = MessagesRole.FUNCTION
        # TODO Switch to using 'result' field in future GigaChat models
        kwargs["content"] = _validate_content(content)
    elif isinstance(message, ToolMessage):
        kwargs["role"] = MessagesRole.FUNCTION
        kwargs["content"] = _validate_content(content)
    else:
        raise TypeError(f"Got unknown type {message}")
    return Messages(**kwargs)


def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    role = _dict.get("role")
    content = _dict.get("content") or ""
    additional_kwargs: Dict = {}
    tool_call_chunks = []
    if _dict.get("function_call"):
        function_call = dict(_dict["function_call"])
        if "name" in function_call and function_call["name"] is None:
            function_call["name"] = ""
        additional_kwargs["function_call"] = function_call
        if additional_kwargs.get("function_call") is not None:
            tool_call_chunks = [
                ToolCallChunk(
                    name=additional_kwargs["function_call"]["name"],
                    args=json.dumps(additional_kwargs["function_call"]["arguments"]),
                    id=str(uuid4()),
                    index=0,
                )
            ]
    if _dict.get("functions_state_id"):
        additional_kwargs["functions_state_id"] = _dict["functions_state_id"]
    match = IMAGE_SEARCH_REGEX.search(content)
    if match:
        additional_kwargs["image_uuid"] = match.group("UUID")
        additional_kwargs["postfix_message"] = match.group("postfix")
    match = VIDEO_SEARCH_REGEX.search(content)
    if match:
        additional_kwargs["cover_uuid"] = match.group("cover_UUID")
        additional_kwargs["video_uuid"] = match.group("UUID")
        additional_kwargs["postfix_message"] = match.group("postfix")

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    elif (
        role == "assistant"
        or default_class == AIMessageChunk
        or "functions_state_id" in _dict
    ):
        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            tool_call_chunks=tool_call_chunks,
        )
    elif role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    elif role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(
            content=_validate_content(content), name=_dict["name"]
        )
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)  # type: ignore[arg-type]
    else:
        return default_class(content=content)  # type: ignore[call-arg]


class _FunctionCall(TypedDict):
    name: str


_BM = TypeVar("_BM", bound=BaseModel)
_DictOrPydanticClass = Union[Dict[str, Any], Type[_BM], Type]
_DictOrPydantic = Union[Dict, _BM]


class _AllReturnType(TypedDict):
    raw: BaseMessage
    parsed: Optional[_DictOrPydantic]
    parsing_error: Optional[BaseException]


def trim_content_to_stop_sequence(
    content: str, stop_sequence: Optional[List[str]]
) -> Union[str, bool]:
    """
    Обрезаем строку к стоп слову.
    Если стоп слово нашлось в строке возвращаем обрезанную строку.
    Если нет, то возвращаем False
    """
    if stop_sequence is None:
        return False
    for stop_w in stop_sequence:
        try:
            index = content.index(stop_w)
            return content[:index]
        except ValueError:
            pass
    return False


class GigaChat(_BaseGigaChat, BaseChatModel):
    """
    LangChain chat model for GigaChat API.

    Args:
        base_url: Address against which requests are executed.
        auth_url: Address for requesting OAuth 2.0 access token.
        credentials: Authorization data.
        scope: API version to which access is provided.
        access_token: JWE token.
        model: Name of the model to receive a response from.
        user: User name for authorization.
        password: Password for authorization.
        timeout: Timeout for requests.
        verify_ssl_certs: Check SSL certificates.
        ca_bundle_file: Path to CA bundle file.
        cert_file: Path to certificate file.
        key_file: Path to key file.
        key_file_password: Password for key file.
        ssl_context: SSL context.
        profanity_check: Check for profanity.
        streaming: Whether to stream the results or not.
        temperature: What sampling temperature to use.
        max_tokens: Maximum number of tokens to generate.
        use_api_for_tokens: Use GigaChat API for tokens count.
        flags: Feature flags.
        top_p: Top_p value to use for nucleus sampling.
            Must be between 0.0 and 1.0.
        repetition_penalty: The penalty applied to repeated tokens.
        update_interval: Minimum interval in seconds that elapses between
            sending tokens.
        auto_upload_images: Auto-upload Base-64 images. Not for production usage.
    """

    """ Auto-upload Base-64 images. Not for production usage! """
    auto_upload_images: bool = False
    """ 
    Dict with cached images, with key as hashed 
    base-64 image to File ID on GigaChat API 
    """
    _cached_images: Dict[str, str] = {}

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        values = super(GigaChat, cls).validate_environment(values)
        if values["auto_upload_images"]:
            logger.warning(
                "`auto_upload_images` is experiment option. "
                "Please, don't use it on production. "
                "Use instead GigaChat.upload_file method for uploading images"
            )
        return values

    async def _aupload_images(self, messages: List[BaseMessage]) -> None:
        for message in messages:
            if isinstance(message.content, list):
                for content_part in message.content:
                    if not isinstance(content_part, dict):
                        continue
                    if content_part.get("type") == "image_url":
                        image_url = content_part["image_url"]["url"]
                        matches = re.search(r"data:(.+);(.+),(.+)", image_url)
                        if matches and not self.auto_upload_images:
                            logger.warning(
                                "You trying to send base-64 images, "
                                "but parameter `auto_upload_images` is not True. "
                                "Set it to True. "
                            )
                        if not matches or not self.auto_upload_images:
                            continue
                        hashed = hashlib.sha256(image_url.encode()).hexdigest()
                        if hashed not in self._cached_images:
                            extension, type_, image_str = matches.groups()
                            if type_ != "base64":
                                continue
                            file = await self.aupload_file(
                                (
                                    f"{uuid4()}{guess_extension(extension)}",
                                    base64.b64decode(image_str),
                                )
                            )
                            self._cached_images[hashed] = file.id_

    def _upload_images(self, messages: List[BaseMessage]) -> None:
        for message in messages:
            if isinstance(message.content, list):
                for content_part in message.content:
                    if not isinstance(content_part, dict):
                        continue
                    if content_part.get("type") == "image_url":
                        image_url = content_part["image_url"]["url"]
                        matches = re.search(r"data:(.+);(.+),(.+)", image_url)
                        if matches and not self.auto_upload_images:
                            logger.warning(
                                "You trying to send base-64 images, "
                                "but parameter `auto_upload_images` is not True. "
                                "Set it to True. "
                            )
                        if not matches or not self.auto_upload_images:
                            continue
                        hashed = hashlib.sha256(image_url.encode()).hexdigest()
                        if hashed not in self._cached_images:
                            extension, type_, image_str = matches.groups()
                            if type_ != "base64":
                                continue
                            file = self.upload_file(
                                (
                                    f"{uuid4()}{guess_extension(extension)}",
                                    base64.b64decode(image_str),
                                )
                            )

                            self._cached_images[hashed] = file.id_

    def _build_payload(self, messages: List[BaseMessage], **kwargs: Any) -> gm.Chat:
        from gigachat.models import Chat

        messages_dicts = [
            _convert_message_to_dict(m, self._cached_images) for m in messages
        ]
        kwargs.pop("messages", None)

        functions = kwargs.pop("functions", [])
        for tool in kwargs.pop("tools", []):
            if tool.get("type", None) == "function" and isinstance(functions, List):
                functions.append(tool["function"])

        function_call = kwargs.pop("function_call", None)

        payload_dict = {
            "messages": messages_dicts,
            "functions": functions,
            "function_call": function_call,
            "profanity_check": self.profanity_check,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "repetition_penalty": self.repetition_penalty,
            "update_interval": self.update_interval,
            **kwargs,
        }

        payload = Chat.model_validate(payload_dict)

        return payload

    def _check_finish_reason(self, finish_reason: str | None) -> None:
        if finish_reason and finish_reason not in {"stop", "function_call"}:
            logger.warning("Giga generation stopped with reason: %s", finish_reason)

    def _create_chat_result(self, response: gm.ChatCompletion) -> ChatResult:
        generations = []
        x_headers = None
        for res in response.choices:
            message = _convert_dict_to_message(res.message)
            x_headers = response.x_headers if response.x_headers else {}
            if x_headers.get("x-request-id") is not None:
                message.id = x_headers["x-request-id"]
            if isinstance(message, AIMessage):
                message.usage_metadata = UsageMetadata(
                    output_tokens=response.usage.completion_tokens,
                    input_tokens=response.usage.prompt_tokens,
                    total_tokens=response.usage.total_tokens,
                    input_token_details={
                        "cache_read": response.usage.precached_prompt_tokens or 0
                    },
                )
            finish_reason = res.finish_reason
            self._check_finish_reason(finish_reason)
            gen = ChatGeneration(
                message=message,
                generation_info={
                    "finish_reason": finish_reason,
                    "model_name": response.model,
                },
            )
            generations.append(gen)
        llm_output = {
            "token_usage": response.usage.model_dump(),
            "model_name": response.model,
            "x_headers": x_headers,
        }
        return ChatResult(generations=generations, llm_output=llm_output)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        self._upload_images(messages)
        payload = self._build_payload(messages, **kwargs)
        response = self._client.chat(payload)
        for choice in response.choices:
            trimmed_content = trim_content_to_stop_sequence(
                choice.message.content, stop
            )
            if isinstance(trimmed_content, str):
                choice.message.content = trimmed_content
                break

        return self._create_chat_result(response)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        await self._aupload_images(messages)
        payload = self._build_payload(messages, **kwargs)
        response = await self._client.achat(payload)
        for choice in response.choices:
            trimmed_content = trim_content_to_stop_sequence(
                choice.message.content, stop
            )
            if isinstance(trimmed_content, str):
                choice.message.content = trimmed_content
                break

        return self._create_chat_result(response)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        self._upload_images(messages)
        payload = self._build_payload(messages, **kwargs)
        message_content = ""

        first_chunk = True
        for chunk_d in self._client.stream(payload):
            chunk = {}
            if not isinstance(chunk_d, dict):
                chunk = chunk_d.model_dump()
            else:
                chunk = chunk_d
            if len(chunk["choices"]) == 0:
                continue

            choice = chunk["choices"][0]
            content = choice.get("delta", {}).get("content", {})
            message_content += content
            if trim_content_to_stop_sequence(message_content, stop):
                return
            chunk_m = _convert_delta_to_message_chunk(choice["delta"], AIMessageChunk)
            usage_metadata = None
            if chunk.get("usage"):
                usage_metadata = UsageMetadata(
                    output_tokens=chunk["usage"]["completion_tokens"],
                    input_tokens=chunk["usage"]["prompt_tokens"],
                    total_tokens=chunk["usage"]["total_tokens"],
                    input_token_details={
                        "cache_read": chunk["usage"].get("precached_prompt_tokens", 0)
                    },
                )
            if isinstance(chunk_m, AIMessageChunk):
                chunk_m.usage_metadata = usage_metadata
            x_headers = chunk.get("x_headers")
            x_headers = x_headers if isinstance(x_headers, dict) else {}
            if "x-request-id" in x_headers:
                chunk_m.id = x_headers["x-request-id"]

            generation_info = {}
            if finish_reason := choice.get("finish_reason"):
                self._check_finish_reason(finish_reason)
                generation_info["model_name"] = chunk.get("model")
                generation_info["finish_reason"] = finish_reason
            if first_chunk:
                generation_info["x_headers"] = x_headers
                first_chunk = False
            if run_manager:
                run_manager.on_llm_new_token(content)

            yield ChatGenerationChunk(message=chunk_m, generation_info=generation_info)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        await self._aupload_images(messages)
        payload = self._build_payload(messages, **kwargs)
        message_content = ""
        first_chunk = True

        async for chunk_d in self._client.astream(payload):
            chunk = {}
            if not isinstance(chunk_d, dict):
                chunk = chunk_d.model_dump()
            else:
                chunk = chunk_d
            if len(chunk["choices"]) == 0:
                continue

            choice = chunk["choices"][0]
            content = choice.get("delta", {}).get("content", {})
            message_content += content
            if trim_content_to_stop_sequence(message_content, stop):
                return
            chunk_m = _convert_delta_to_message_chunk(choice["delta"], AIMessageChunk)
            usage_metadata = None
            if chunk.get("usage"):
                usage_metadata = UsageMetadata(
                    output_tokens=chunk["usage"]["completion_tokens"],
                    input_tokens=chunk["usage"]["prompt_tokens"],
                    total_tokens=chunk["usage"]["total_tokens"],
                    input_token_details={
                        "cache_read": chunk["usage"].get("precached_prompt_tokens", 0)
                    },
                )
            if isinstance(chunk_m, AIMessageChunk):
                chunk_m.usage_metadata = usage_metadata
            x_headers = chunk.get("x_headers")
            x_headers = x_headers if isinstance(x_headers, dict) else {}
            if isinstance(x_headers, dict) and "x-request-id" in x_headers:
                chunk_m.id = x_headers["x-request-id"]

            generation_info = {}
            if finish_reason := choice.get("finish_reason"):
                self._check_finish_reason(finish_reason)
                generation_info["model_name"] = chunk.get("model")
                generation_info["finish_reason"] = finish_reason
            if first_chunk:
                generation_info["x_headers"] = x_headers
                first_chunk = False
            if run_manager:
                await run_manager.on_llm_new_token(content)

            yield ChatGenerationChunk(message=chunk_m, generation_info=generation_info)

    def bind_functions(
        self,
        functions: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, type]],
        function_call: Optional[str] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind functions (and other objects) to this chat model.

        Args:
            functions: A list of function definitions to bind to this chat model.
                Can be  a dictionary, pydantic model, or callable. Pydantic
                models and callables will be automatically converted to
                their schema dictionary representation.
            function_call: Which function to require the model to call.
                Must be the name of the single provided function or
                "auto" to automatically determine which function to call
                (if any).
            kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.
        """
        formatted_functions = [convert_to_gigachat_function(fn) for fn in functions]
        if function_call is not None:
            if len(formatted_functions) != 1:
                raise ValueError(
                    "When specifying `function_call`, you must provide exactly one "
                    "function."
                )
            if formatted_functions[0]["name"] != function_call:
                raise ValueError(
                    f"Function call {function_call} was specified, but the only "
                    f"provided function was {formatted_functions[0]['name']}."
                )
            function_call_ = {"name": function_call}
            kwargs = {**kwargs, "function_call": function_call_}
        return super().bind(functions=formatted_functions, **kwargs)

    # TODO: Fix typing.
    @overload  # type: ignore[override]
    def with_structured_output(
        self,
        schema: Optional[_DictOrPydanticClass] = None,
        *,
        method: Literal[
            "function_calling", "json_mode", "format_instructions"
        ] = "function_calling",
        include_raw: Literal[True] = True,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _AllReturnType]: ...

    @overload
    def with_structured_output(
        self,
        schema: Optional[_DictOrPydanticClass] = None,
        *,
        method: Literal[
            "function_calling", "json_mode", "format_instructions"
        ] = "function_calling",
        include_raw: Literal[False] = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]: ...

    def with_structured_output(
        self,
        schema: Optional[_DictOrPydanticClass] = None,
        *,
        method: Literal[
            "function_calling", "json_mode", "format_instructions"
        ] = "function_calling",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        """Model wrapper that returns outputs formatted to match the given schema."""
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")
        is_pydantic_schema = _is_pydantic_class(schema)
        if method == "function_calling":
            if schema is None:
                raise ValueError(
                    "schema must be specified when method is 'function_calling'. "
                    "Received None."
                )
            func = convert_to_gigachat_tool(schema)["function"]
            key_name = func.get(
                "name", func.get("title")
            )  # In case of pydantic from JSON (For openai capability)
            if is_pydantic_schema:
                output_parser: OutputParserLike = PydanticToolsParser(
                    tools=[schema],  # type: ignore
                    first_tool_only=True,
                )
            else:
                output_parser = JsonOutputKeyToolsParser(
                    key_name=key_name, first_tool_only=True
                )
            llm = self.bind_tools([schema], tool_choice=key_name)
        else:
            llm = self
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)  # type: ignore[arg-type]
                if is_pydantic_schema
                else JsonOutputParser()
            )
            if method == "format_instructions":
                from langchain_core.prompt_values import ChatPromptValue
                from langchain_core.runnables import RunnableLambda

                def add_format_instructions(
                    _input: LanguageModelInput, format_instructions: str
                ) -> LanguageModelInput:
                    if isinstance(_input, ChatPromptValue):
                        messages = _input.messages
                        return type(messages)(
                            list(messages) + [HumanMessage(format_instructions)]  # type: ignore[call-arg]
                        )
                    elif isinstance(_input, str):
                        return _input + f"\n\n{format_instructions}"
                    elif isinstance(_input, Sequence):
                        return type(_input)(
                            list(_input) + [HumanMessage(format_instructions)]  # type: ignore[call-arg]
                        )
                    else:
                        msg = (
                            f"Invalid input type {type(_input)}. "
                            "Must be a PromptValue, str, or list of BaseMessages."
                        )
                        raise ValueError(msg)  # noqa: TRY004

                add_format_instructions_chain = RunnableLambda(
                    lambda _input: add_format_instructions(
                        _input, output_parser.get_format_instructions()
                    )
                )
                llm = add_format_instructions_chain | llm

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        else:
            return llm | output_parser

    def bind_tools(
        self,
        tools: Sequence[
            Union[Dict[str, Any], Type, Type[BaseModel], Callable, BaseTool]
        ],  #  noqa
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "any", "none"], bool]
        ] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind tool-like objects to this chat model.
        Assumes model is compatible with GigaChat tool-calling API."""
        formatted_tools = [convert_to_gigachat_tool(tool) for tool in tools]
        if tool_choice is not None and tool_choice:
            if isinstance(tool_choice, str):
                if tool_choice not in ("auto", "none"):
                    tool_choice = {"name": tool_choice}
            elif isinstance(tool_choice, bool) and tool_choice:
                if not formatted_tools:
                    raise ValueError("tool_choice can not be bool if tools are empty")
                tool_choice = {"name": formatted_tools[0]["name"]}
            elif isinstance(tool_choice, dict):
                pass
            else:
                raise ValueError(
                    f"Unrecognized tool_choice type. Expected str, bool or dict. "
                    f"Received: {tool_choice}"
                )
            kwargs["function_call"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)


def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and is_basemodel_subclass(obj)
