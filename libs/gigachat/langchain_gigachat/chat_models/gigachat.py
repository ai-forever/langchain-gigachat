from __future__ import annotations

import asyncio
import base64
import binascii
import copy
import hashlib
import json
import logging
import re
import threading
import warnings
from concurrent.futures import Future
from mimetypes import guess_extension
from operator import itemgetter
from typing import (
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
    TypeGuard,
    Union,
)
from uuid import uuid4

import gigachat.models as gm
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.exceptions import OutputParserException
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
from langchain_core.output_parsers.format_instructions import (
    JSON_FORMAT_INSTRUCTIONS,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.prompt_values import ChatPromptValue, PromptValue
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough,
)
from langchain_core.tools import BaseTool
from langchain_core.utils.pydantic import is_basemodel_subclass
from pydantic import BaseModel, PrivateAttr, model_validator
from typing_extensions import Self, override

from langchain_gigachat.chat_models._contracts import primary
from langchain_gigachat.chat_models._contracts.common import (
    CROSS_CONTRACT_TOOL_STATE_ERROR,
)
from langchain_gigachat.chat_models.base_gigachat import _BaseGigaChat
from langchain_gigachat.utils.function_calling import (
    convert_to_gigachat_function,
    convert_to_gigachat_tool,
    is_primary_builtin_tool,
    model_to_json_schema,
    normalize_tool_for_binding,
)

logger = logging.getLogger(__name__)

IMAGE_SEARCH_REGEX = re.compile(
    r'<img\ssrc="(?P<UUID>.+?)"\sfuse=".+?"/>(?P<postfix>.+)?'
)
VIDEO_SEARCH_REGEX = re.compile(
    r'<video\scover="(?P<cover_UUID>.+?)"\ssrc="(?P<UUID>.+?)"\sfuse="true"/>(?P<postfix>.+)?'  # noqa
)
BASE64_DATA_REGEX = re.compile(r"data:(.+);(.+),(.+)")

# GigaChat-supported MIME types where mimetypes.guess_extension returns None
# https://developers.sber.ru/docs/ru/gigachat/api/reference/rest/post-file
MIME_EXTENSION_FALLBACK: Dict[str, str] = {
    "audio/mp3": ".mp3",
    "audio/mp4": ".mp4",
    "audio/x-m4a": ".m4a",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/wav": ".wav",
    "audio/x-pn-wav": ".wav",
    "audio/webm": ".weba",
    "audio/x-ogg": ".ogg",
    "audio/opus": ".opus",
    "application/epub": ".epub",
    "application/pptx": ".pptx",
    "application/ppt": ".ppt",
}

DEFAULT_IMAGE_CACHE_MAX_SIZE = 1000

ATTACHMENT_BLOCK_KEYS = ("image_url", "audio_url", "document_url")
_PRIMARY_ONLY_KWARGS = frozenset(
    {
        "assistant_id",
        "disable_filter",
        "filter_config",
        "model_options",
        "ranker_options",
        "reasoning",
        "tool_config",
        "tools_state_id",
        "user_info",
    }
)
_SCHEMA_LESS_JSON_MODE_KEY = "_schema_less_json_mode"


def _extension_for_mime(mime: str) -> str:
    """Return file extension (with dot) for MIME type, falling back to '.bin'."""
    ext = guess_extension(mime.split(";")[0].strip())
    return ext or MIME_EXTENSION_FALLBACK.get(mime.split(";")[0].strip(), ".bin")


def _validate_content(content: Any) -> Any:
    """If content is string, but not JSON - convert string to json-string"""
    if isinstance(content, str):
        try:
            json.loads(content)
        except ValueError:
            content = json.dumps(content, ensure_ascii=False)
    return content


def _convert_dict_to_message(message: gm.Messages) -> BaseMessage:
    additional_kwargs: Dict = {}
    tool_calls = []
    tool_call_id = None
    if function_call := message.function_call:
        if isinstance(function_call, gm.FunctionCall):
            additional_kwargs["function_call"] = dict(function_call)
        elif isinstance(function_call, dict):
            additional_kwargs["function_call"] = function_call
        if additional_kwargs.get("function_call") is not None:
            tool_call_id = str(uuid4())
            tool_calls = [
                ToolCall(
                    name=additional_kwargs["function_call"]["name"],
                    args=additional_kwargs["function_call"]["arguments"],
                    id=tool_call_id,
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
    reasoning_content = getattr(message, "reasoning_content", None)
    if reasoning_content is not None:
        additional_kwargs["reasoning_content"] = reasoning_content
    if message.role == gm.MessagesRole.SYSTEM:
        return SystemMessage(content=message.content)
    elif message.role == gm.MessagesRole.USER:
        return HumanMessage(content=message.content)
    elif message.role == gm.MessagesRole.ASSISTANT:
        return AIMessage(
            content=message.content,
            additional_kwargs=additional_kwargs,
            tool_calls=tool_calls,
        )
    elif message.role == gm.MessagesRole.FUNCTION:
        return FunctionMessage(
            name=message.name or "", content=_validate_content(message.content)
        )
    else:
        raise TypeError(f"Got unknown role {message.role} {message}")


def get_text_and_images_from_content(
    content: list[Union[str, dict]], cached_images: Mapping[str, str]
) -> Tuple[str, List[str]]:
    """Extract text and attachment IDs from LangChain content blocks.

    Supports two formats:

    1) Provider-native (OpenAI-style): type in ("image_url", "audio_url",
       "document_url") with nested block key and "giga_id" or "url" (cache).
    2) Standard LangChain content_blocks: type in ("image", "audio", "file") with
       top-level "file_id" (GigaChat file id) or "url" (resolved via cache).

    Use standard blocks (e.g. content_blocks=[{"type": "image", "file_id": "id"}])
    so that message.content_blocks displays typed blocks; both formats are
    accepted for API payload building.
    """
    text_parts = []
    attachments = []
    seen_attachments = set()

    def append_attachment(attachment_id: str) -> None:
        if attachment_id and attachment_id not in seen_attachments:
            seen_attachments.add(attachment_id)
            attachments.append(attachment_id)

    for content_part in content:
        if isinstance(content_part, str):
            text_parts.append(content_part)
        elif isinstance(content_part, dict):
            block_type = content_part.get("type")
            if block_type == "text":
                text_parts.append(content_part.get("text", ""))
            elif block_type in ("image_url", "audio_url", "document_url"):
                block_key = block_type
                block_data = content_part.get(block_key)
                if not isinstance(block_data, dict):
                    continue
                if block_data.get("giga_id"):
                    append_attachment(block_data["giga_id"])
                url = block_data.get("url")
                if url:
                    hashed = hashlib.sha256(url.encode()).hexdigest()
                    if hashed in cached_images:
                        append_attachment(cached_images[hashed])
            elif block_type in ("image", "audio", "file"):
                if content_part.get("file_id"):
                    append_attachment(content_part["file_id"])
                url = content_part.get("url")
                if url:
                    hashed = hashlib.sha256(url.encode()).hexdigest()
                    if hashed in cached_images:
                        append_attachment(cached_images[hashed])
    return " ".join(text_parts), attachments


def _merge_legacy_additional_attachments(
    attachments: List[str],
    additional_attachments: Any,
) -> List[str]:
    """Validate and merge caller-supplied legacy attachment IDs."""
    if not isinstance(additional_attachments, Sequence) or isinstance(
        additional_attachments, (str, bytes, bytearray)
    ):
        raise ValueError(
            "message.additional_kwargs['attachments'] must be a sequence of "
            "non-empty strings, not str or bytes."
        )

    merged = list(attachments)
    seen = set(merged)
    for attachment_id in additional_attachments:
        if not isinstance(attachment_id, str) or not attachment_id.strip():
            raise ValueError(
                "message.additional_kwargs['attachments'] must contain only "
                "non-empty strings."
            )
        if attachment_id not in seen:
            seen.add(attachment_id)
            merged.append(attachment_id)
    return merged


def _convert_message_to_dict(
    message: BaseMessage, cached_images: Optional[Mapping[str, str]] = None
) -> gm.Messages:
    kwargs = {}
    if cached_images is None:
        cached_images = {}

    primary_state_keys = {
        "provider_server_tool_state_by_call_id",
        "tools_state_id",
        "tools_state_ids",
    }
    if any(
        key in source
        for source in (message.additional_kwargs, message.response_metadata)
        for key in primary_state_keys
    ):
        raise ValueError(CROSS_CONTRACT_TOOL_STATE_ERROR)

    if isinstance(message.content, list):
        content, attachments = get_text_and_images_from_content(
            message.content, cached_images
        )
    else:
        content, attachments = message.content, []

    if "attachments" in message.additional_kwargs:
        attachments = _merge_legacy_additional_attachments(
            attachments,
            message.additional_kwargs["attachments"],
        )
    if attachments and not isinstance(message, HumanMessage):
        raise ValueError(
            "Legacy GigaChat supports attachments only on HumanMessage; "
            f"{type(message).__name__} attachments cannot be transmitted."
        )
    functions_state_id = message.additional_kwargs.get("functions_state_id", None)
    if functions_state_id:
        kwargs["functions_state_id"] = functions_state_id

    if isinstance(message, SystemMessage):
        kwargs["role"] = gm.MessagesRole.SYSTEM
        kwargs["content"] = content
    elif isinstance(message, HumanMessage):
        kwargs["role"] = gm.MessagesRole.USER
        if attachments:
            kwargs["attachments"] = attachments
        kwargs["content"] = content
    elif isinstance(message, AIMessage):
        if tool_calls := getattr(message, "tool_calls", None):
            if len(tool_calls) > 1:
                raise ValueError(
                    "GigaChat API does not support multiple tool calls in a single "
                    "message. Received an AIMessage with "
                    f"{len(tool_calls)} tool_calls. "
                    "Use a single tool call per turn."
                )
            function_call = copy.deepcopy(tool_calls[0])

            if "args" in function_call:
                function_call["arguments"] = function_call.pop("args")
        else:
            function_call = message.additional_kwargs.get("function_call", None)
        kwargs["role"] = gm.MessagesRole.ASSISTANT
        kwargs["content"] = content
        kwargs["function_call"] = function_call
    elif isinstance(message, ChatMessage):
        kwargs["role"] = message.role
        kwargs["content"] = content
    elif isinstance(message, FunctionMessage):
        kwargs["role"] = gm.MessagesRole.FUNCTION
        kwargs["name"] = message.name
        kwargs["content"] = _validate_content(content)
    elif isinstance(message, ToolMessage):
        # LangChain's public surface is tool-oriented, but the provider transport
        # is still function-oriented, so tool results must be serialized back as
        # provider FUNCTION messages for round-trip compatibility.
        kwargs["role"] = gm.MessagesRole.FUNCTION
        if message.name:
            kwargs["name"] = message.name
        kwargs["content"] = _validate_content(content)
    else:
        raise TypeError(f"Got unknown type {message}")
    return gm.Messages(**kwargs)


def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any],
    default_class: Type[BaseMessageChunk],
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
    incoming_functions_state_id = _dict.get("functions_state_id")
    if incoming_functions_state_id:
        additional_kwargs["functions_state_id"] = incoming_functions_state_id
    if _dict.get("reasoning_content") is not None:
        additional_kwargs["reasoning_content"] = _dict["reasoning_content"]
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


def _get_tool_name(tool: Mapping[str, Any]) -> str:
    """Return tool name from normalized or title-only tool payload."""
    if is_primary_builtin_tool(tool):
        tool_type = tool.get("type")
        if isinstance(tool_type, str):
            return tool_type
        return next(iter(tool))

    function = tool.get("function")
    if not isinstance(function, Mapping):
        raise ValueError("Tool payload must contain a function mapping.")

    name = function.get("name") or function.get("title")
    if not isinstance(name, str) or not name:
        raise ValueError("Tool payload must define a non-empty function name or title.")
    return name


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
        max_retries: Maximum number of retries for transient errors
            (SDK default: 0, disabled). Avoid combining with LangChain's
            ``.with_retry()`` to prevent multiplicative retry counts.
        max_connections: Maximum number of simultaneous connections to the
            GigaChat API.
        retry_backoff_factor: Backoff factor for retry delays
            (SDK default: 0.5).
        retry_on_status_codes: HTTP status codes that trigger a retry
            (SDK default: ``(429, 500, 502, 503, 504)``).
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
        auto_upload_attachments: Auto-upload Base-64 content for image_url,
            audio_url, and document_url blocks. Not for production usage.
        allow_any_tool_choice_fallback: Explicitly convert
            ``tool_choice='any'`` to ``'auto'`` with a warning. Disabled by
            default because the conversion weakens forced-tool semantics.
        reasoning_effort: Reasoning effort for reasoning-capable models
            (e.g. GigaChat-2-Reasoning). When set, the API may return
            reasoning_content in the assistant message (see additional_kwargs).
        function_ranker: Function/tool ranking settings. Pass
            ``{"enabled": False}`` to disable function ranking for tool calls.
    """

    auto_upload_attachments: bool = False
    """Auto-upload Base-64 image/audio/document blocks. Not for production usage."""
    allow_any_tool_choice_fallback: bool = False
    """
    Convert ``tool_choice='any'`` to ``'auto'`` with a compatibility warning.
    """

    _cached_uploads: Dict[str, str] = PrivateAttr(default_factory=dict)
    _upload_cache_lock: Any = PrivateAttr(default_factory=threading.Lock)
    _uploads_in_flight: Dict[str, Future[str]] = PrivateAttr(default_factory=dict)

    def __deepcopy__(self, memo: Optional[Dict[int, Any]] = None) -> Self:
        """Deep-copy config/cache with a shared limiter and fresh owned runtime."""
        cached_uploads = self._cached_uploads_snapshot()
        rate_limiter = self.rate_limiter
        staged = self.__copy__()
        staged._cached_uploads = cached_uploads
        staged._upload_cache_lock = None
        staged._uploads_in_flight = {}
        staged.rate_limiter = None
        copied = super(GigaChat, staged).__deepcopy__(memo)
        copied._upload_cache_lock = threading.Lock()
        copied.rate_limiter = rate_limiter
        return copied

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Any) -> Any:
        if isinstance(values, Mapping) and values.get("auto_upload_attachments"):
            logger.warning(
                "`auto_upload_attachments` is experiment option. "
                "Please, don't use it on production. "
                "Use instead GigaChat.upload_file for uploading files."
            )
        return values

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        params = super()._identifying_params
        params["auto_upload_attachments"] = self.auto_upload_attachments
        return params

    def _set_cached_upload_locked(self, hashed: str, file_id: str) -> None:
        if (
            hashed not in self._cached_uploads
            and len(self._cached_uploads) >= DEFAULT_IMAGE_CACHE_MAX_SIZE
        ):
            self._cached_uploads.pop(next(iter(self._cached_uploads)))
        self._cached_uploads[hashed] = file_id

    def _set_cached_upload(self, hashed: str, file_id: str) -> None:
        """Store one cached upload while deterministically bounding the cache."""
        with self._upload_cache_lock:
            self._set_cached_upload_locked(hashed, file_id)

    def _cached_uploads_snapshot(self) -> Dict[str, str]:
        with self._upload_cache_lock:
            return dict(self._cached_uploads)

    def _uploads_for_request(
        self,
        request_uploads: Mapping[str, str],
    ) -> Dict[str, str]:
        """Merge shared cached IDs with request-owned IDs.

        The shared cache is bounded and may evict an attachment between upload
        completion and payload construction. Request-owned IDs therefore win.
        """
        uploads = self._cached_uploads_snapshot()
        uploads.update(request_uploads)
        return uploads

    def _claim_upload(
        self,
        hashed: str,
    ) -> Tuple[Optional[str], Future[str], bool]:
        with self._upload_cache_lock:
            cached = self._cached_uploads.get(hashed)
            if cached is not None:
                completed: Future[str] = Future()
                completed.set_result(cached)
                return cached, completed, False

            pending = self._uploads_in_flight.get(hashed)
            if pending is not None:
                return None, pending, False

            pending = Future()
            self._uploads_in_flight[hashed] = pending
            return None, pending, True

    def _complete_upload(
        self,
        hashed: str,
        pending: Future[str],
        file_id: str,
    ) -> None:
        with self._upload_cache_lock:
            self._set_cached_upload_locked(hashed, file_id)
            current = self._uploads_in_flight.pop(hashed, None)
        if current is pending and not pending.done():
            pending.set_result(file_id)

    def _fail_upload(
        self,
        hashed: str,
        pending: Future[str],
        error: BaseException,
    ) -> None:
        with self._upload_cache_lock:
            current = self._uploads_in_flight.pop(hashed, None)
        if current is pending and not pending.done():
            pending.set_exception(error)

    @staticmethod
    def _uploaded_file_id(file: Any) -> str:
        file_id = getattr(file, "id_", None)
        if not isinstance(file_id, str) or not file_id.strip():
            raise ValueError(
                "GigaChat attachment upload returned an empty file ID; "
                "the attachment cannot be added to the request."
            )
        return file_id

    def _should_upload_block(
        self, block_type: str, url: str
    ) -> Tuple[bool, Optional[re.Match[str]]]:
        """Return (should_upload, data_url_match)."""
        matches = BASE64_DATA_REGEX.search(url)
        if not matches:
            return False, None
        if block_type not in ATTACHMENT_BLOCK_KEYS:
            return False, None
        if not self.auto_upload_attachments:
            if block_type == "image_url":
                logger.warning(
                    "Base-64 image in message but `auto_upload_attachments` is False. "
                    "Set it to True or upload via GigaChat.upload_file."
                )
            return False, None
        return True, matches

    def _attachment_upload_plan(
        self,
        messages: Sequence[BaseMessage],
    ) -> List[Tuple[str, str, bytes]]:
        """Build a complete, side-effect-free plan for attachment uploads."""
        planned: Dict[str, Tuple[str, str, bytes]] = {}
        for message in messages:
            if not isinstance(message.content, list):
                continue
            for content_part in message.content:
                if not isinstance(content_part, dict):
                    continue
                block_type = content_part.get("type")
                if block_type not in ATTACHMENT_BLOCK_KEYS:
                    continue
                block_data = content_part.get(block_type)
                if not isinstance(block_data, dict):
                    continue
                url = block_data.get("url")
                if not isinstance(url, str) or not url:
                    continue
                should_upload, matches = self._should_upload_block(block_type, url)
                if not should_upload or matches is None:
                    continue
                mime, encoding, data_b64 = matches.groups()
                if encoding != "base64":
                    continue
                try:
                    data = base64.b64decode(data_b64, validate=True)
                except binascii.Error as error:
                    raise ValueError(
                        "Invalid base64 data URL attachment; fix the local payload "
                        "before retrying."
                    ) from error
                hashed = hashlib.sha256(url.encode()).hexdigest()
                planned.setdefault(
                    hashed,
                    (hashed, _extension_for_mime(mime), data),
                )
        return list(planned.values())

    async def _aupload_attachments(
        self,
        messages: List[BaseMessage],
    ) -> Dict[str, str]:
        request_uploads: Dict[str, str] = {}
        for hashed, ext, data in self._attachment_upload_plan(messages):
            cached, pending, owns_upload = self._claim_upload(hashed)
            if cached is not None:
                request_uploads[hashed] = cached
                continue
            if not owns_upload:
                request_uploads[hashed] = await asyncio.shield(
                    asyncio.wrap_future(pending)
                )
                continue
            try:
                file = await self.aupload_file((f"{uuid4()}{ext}", data))
                file_id = self._uploaded_file_id(file)
            except BaseException as error:
                self._fail_upload(hashed, pending, error)
                raise
            self._complete_upload(hashed, pending, file_id)
            request_uploads[hashed] = file_id
        return request_uploads

    def _upload_attachments(self, messages: List[BaseMessage]) -> Dict[str, str]:
        request_uploads: Dict[str, str] = {}
        for hashed, ext, data in self._attachment_upload_plan(messages):
            cached, pending, owns_upload = self._claim_upload(hashed)
            if cached is not None:
                request_uploads[hashed] = cached
                continue
            if not owns_upload:
                request_uploads[hashed] = pending.result()
                continue
            try:
                file = self.upload_file((f"{uuid4()}{ext}", data))
                file_id = self._uploaded_file_id(file)
            except BaseException as error:
                self._fail_upload(hashed, pending, error)
                raise
            self._complete_upload(hashed, pending, file_id)
            request_uploads[hashed] = file_id
        return request_uploads

    def _build_payload(self, messages: List[BaseMessage], **kwargs: Any) -> gm.Chat:
        return self._build_legacy_payload(
            messages,
            self._cached_uploads_snapshot(),
            **kwargs,
        )

    def _build_legacy_payload(
        self,
        messages: List[BaseMessage],
        cached_uploads: Mapping[str, str],
        **kwargs: Any,
    ) -> gm.Chat:
        messages_dicts = [_convert_message_to_dict(m, cached_uploads) for m in messages]
        kwargs.pop("messages", None)
        kwargs.pop("use_api_v2", None)
        kwargs.pop(_SCHEMA_LESS_JSON_MODE_KEY, None)
        strict = kwargs.pop("strict", None)
        response_format = kwargs.get("response_format")
        if response_format is not None or strict is not None:
            normalized_response_format = primary.normalize_response_format(
                response_format,
                strict=strict,
            )
            if (
                normalized_response_format is None
                or normalized_response_format.type != "json_schema"
                or not isinstance(normalized_response_format.schema_, dict)
            ):
                raise ValueError(
                    "Legacy GigaChat supports only JSON Schema response_format."
                )
            kwargs["response_format"] = gm.JsonSchemaResponseFormat(
                schema=normalized_response_format.schema_,
                strict=normalized_response_format.strict,
            )

        functions = copy.deepcopy(kwargs.pop("functions", []))
        tools = copy.deepcopy(kwargs.pop("tools", []))
        for tool in tools:
            if tool.get("type", None) == "function" and isinstance(functions, list):
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
            "function_ranker": self.function_ranker,
            **kwargs,
        }
        if self.reasoning_effort is not None:
            payload_dict["reasoning_effort"] = self.reasoning_effort

        payload = gm.Chat.model_validate(payload_dict)

        return payload

    def _resolve_chat_contract(
        self, kwargs: Mapping[str, Any]
    ) -> Literal["legacy", "primary"]:
        return "primary" if kwargs.get("use_api_v2", self.use_api_v2) else "legacy"

    def _validate_legacy_kwargs(self, kwargs: Mapping[str, Any]) -> None:
        unsupported = sorted(_PRIMARY_ONLY_KWARGS.intersection(kwargs))
        if unsupported:
            names = ", ".join(unsupported)
            raise ValueError(
                f"Legacy GigaChat does not support primary-only argument(s): {names}. "
                "Use use_api_v2=True."
            )

        builtin_names = sorted(
            _get_tool_name(tool)
            for tool in kwargs.get("tools", ())
            if is_primary_builtin_tool(tool)
        )
        if builtin_names:
            names = ", ".join(builtin_names)
            raise ValueError(
                "Legacy GigaChat does not support provider built-in tool(s): "
                f"{names}. Use use_api_v2=True."
            )

    def _primary_defaults(self) -> primary.RequestDefaults:
        function_ranker = self.function_ranker
        if isinstance(function_ranker, BaseModel):
            function_ranker = function_ranker.model_dump(
                exclude_none=True, by_alias=True
            )
        return primary.RequestDefaults(
            model=self.model,
            profanity_check=self.profanity_check,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            repetition_penalty=self.repetition_penalty,
            update_interval=self.update_interval,
            reasoning_effort=self.reasoning_effort,
            function_ranker=function_ranker,
            flags=self.flags,
        )

    def _build_primary_payload(
        self,
        messages: List[BaseMessage],
        kwargs: Mapping[str, Any],
        *,
        cached_uploads: Optional[Mapping[str, str]] = None,
    ) -> gm.ChatCompletionRequest:
        invocation_kwargs = dict(kwargs)
        invocation_kwargs.pop("use_api_v2", None)
        schema_less_json_mode = bool(
            invocation_kwargs.pop(_SCHEMA_LESS_JSON_MODE_KEY, False)
        )
        if schema_less_json_mode:
            response_format = invocation_kwargs.get("response_format")
            if response_format is not None:
                raise ValueError(
                    "Schema-less json_mode cannot be combined with an explicit "
                    "response_format."
                )
            invocation_kwargs["response_format"] = {"type": "json_schema"}
        tool_binding = primary.build_tool_binding(
            functions=invocation_kwargs.get("functions", ()),
            tools=invocation_kwargs.get("tools", ()),
            function_call=invocation_kwargs.get("function_call"),
            explicit_tool_config=invocation_kwargs.get("tool_config"),
        )
        return primary.build_payload(
            messages,
            defaults=self._primary_defaults(),
            invocation_kwargs=invocation_kwargs,
            cached_uploads=(
                self._cached_uploads_snapshot()
                if cached_uploads is None
                else cached_uploads
            ),
            tool_binding=tool_binding,
        )

    def _validation_upload_cache(
        self, messages: Sequence[BaseMessage]
    ) -> Dict[str, str]:
        """Return detached placeholder IDs for valid planned data-URL uploads."""
        validation_cache = self._cached_uploads_snapshot()
        for hashed, _ext, _data in self._attachment_upload_plan(messages):
            validation_cache.setdefault(hashed, f"pending-upload-{hashed}")
        return validation_cache

    def _validate_request_before_upload(
        self,
        messages: List[BaseMessage],
        kwargs: Mapping[str, Any],
    ) -> Literal["legacy", "primary"]:
        """Validate the complete request without performing remote side effects."""
        route = self._resolve_chat_contract(kwargs)
        validation_cache = self._validation_upload_cache(messages)
        if route == "primary":
            self._build_primary_payload(
                messages,
                kwargs,
                cached_uploads=validation_cache,
            )
        else:
            self._validate_legacy_kwargs(kwargs)
            self._build_legacy_payload(
                messages,
                validation_cache,
                **dict(kwargs),
            )
        return route

    def _create_chat_result(self, response: gm.ChatCompletion) -> ChatResult:
        """Convert SDK response to ChatResult and preserve tracing metadata.

        The wrapper surfaces provider tracing headers in two places:
        - ``message.id`` carries ``x-request-id`` when present.
        - ``llm_output["x_headers"]`` keeps the full response headers for
          debugging, logging, or support escalation.
        """
        generations = []
        x_headers = None
        for res in response.choices:
            message = _convert_dict_to_message(res.message)
            x_headers = response.x_headers if response.x_headers else {}
            if x_headers.get("x-request-id") is not None:
                # GigaChat request id for tracing and support.
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
            "x_headers": x_headers,  # GigaChat response headers for debugging.
        }
        return ChatResult(generations=generations, llm_output=llm_output)

    def _build_stream_chunk(
        self,
        chunk: Dict[str, Any],
        first_chunk: bool,
    ) -> Tuple[BaseMessageChunk, Dict[str, Any], Any]:
        """Build message chunk and generation_info from a normalized stream chunk dict.

        Usage and x_headers are set here in one place for both _stream and
        _astream. ``x-request-id`` is copied to ``chunk.id`` when present.
        The first streamed chunk also exposes the full ``x_headers`` payload via
        ``generation_info`` so callers can keep tracing metadata in streaming and
        non-streaming paths.

        Caller is responsible for normalizing the raw chunk to a dict and for
        callbacks.
        """
        choice = chunk["choices"][0]
        content = choice.get("delta", {}).get("content", "")
        chunk_m = _convert_delta_to_message_chunk(
            choice["delta"],
            AIMessageChunk,
        )

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

        generation_info: Dict[str, Any] = {}
        if finish_reason := choice.get("finish_reason"):
            generation_info["model_name"] = chunk.get("model")
            generation_info["finish_reason"] = finish_reason
        if first_chunk:
            generation_info["x_headers"] = x_headers

        return (chunk_m, generation_info, content)

    @override
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Kept in the signature for LangChain compatibility, but wrapper-side
        # local stop handling was removed in 0.5.x. See MIGRATION.md.
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        route = self._validate_request_before_upload(messages, kwargs)
        request_uploads = self._upload_attachments(messages)
        uploads = self._uploads_for_request(request_uploads)
        if route == "primary":
            primary_payload = self._build_primary_payload(
                messages,
                kwargs,
                cached_uploads=uploads,
            )
            primary_response = self._client.chat.create(primary_payload)
            result = primary.create_chat_result(primary_response)
        else:
            self._validate_legacy_kwargs(kwargs)
            payload = self._build_legacy_payload(messages, uploads, **kwargs)
            response = self._client.chat(payload)
            result = self._create_chat_result(response)
        return result

    @override
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Kept in the signature for LangChain compatibility, but wrapper-side
        # local stop handling was removed in 0.5.x. See MIGRATION.md.
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        route = self._validate_request_before_upload(messages, kwargs)
        request_uploads = await self._aupload_attachments(messages)
        uploads = self._uploads_for_request(request_uploads)
        if route == "primary":
            primary_payload = self._build_primary_payload(
                messages,
                kwargs,
                cached_uploads=uploads,
            )
            primary_response = await self._client.achat.create(primary_payload)
            result = primary.create_chat_result(primary_response)
        else:
            self._validate_legacy_kwargs(kwargs)
            payload = self._build_legacy_payload(messages, uploads, **kwargs)
            response = await self._client.achat(payload)
            result = self._create_chat_result(response)
        return result

    @override
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        # Kept in the signature for LangChain compatibility, but wrapper-side
        # local stop handling was removed in 0.5.x. See MIGRATION.md.
        route = self._validate_request_before_upload(messages, kwargs)
        request_uploads = self._upload_attachments(messages)
        uploads = self._uploads_for_request(request_uploads)
        terminal_chunk: Optional[ChatGenerationChunk] = None
        saw_converted_chunk = False
        if route == "primary":
            primary_payload = self._build_primary_payload(
                messages,
                kwargs,
                cached_uploads=uploads,
            )
            state = primary.StreamState()
            provider_stream = self._client.chat.stream(primary_payload)
            try:
                for event in provider_stream:
                    primary_chunk = primary.convert_stream_event(event, state=state)
                    if primary_chunk is None:
                        continue
                    saw_converted_chunk = True
                    if primary_chunk.message.content == []:
                        primary_chunk = ChatGenerationChunk(
                            message=primary_chunk.message.model_copy(
                                update={"content": ""}
                            ),
                            generation_info=primary_chunk.generation_info,
                        )
                    if _is_authoritative_primary_terminal(primary_chunk):
                        terminal_chunk = _accept_terminal_chunk(
                            terminal_chunk, primary_chunk, route="Primary"
                        )
                        continue
                    if terminal_chunk is not None:
                        terminal_chunk = _merge_terminal_continuation(
                            terminal_chunk,
                            primary_chunk,
                            route="Primary",
                        )
                        continue
                    if run_manager:
                        run_manager.on_llm_new_token(
                            primary_chunk.text, chunk=primary_chunk
                        )
                    yield primary_chunk
            finally:
                _close_stream_iterator(provider_stream)
            if not saw_converted_chunk:
                return
            if terminal_chunk is None:
                raise ValueError("Primary stream ended before response.message.done")
            terminal_chunk = _finalize_terminal_chunk(terminal_chunk)
            if run_manager:
                run_manager.on_llm_new_token(
                    terminal_chunk.text,
                    chunk=terminal_chunk,
                )
            yield terminal_chunk
            return
        self._validate_legacy_kwargs(kwargs)
        payload = self._build_legacy_payload(messages, uploads, **kwargs)
        first_chunk = True

        legacy_provider_stream = self._client.stream(payload)
        try:
            for chunk_d in legacy_provider_stream:
                chunk = chunk_d if isinstance(chunk_d, dict) else chunk_d.model_dump()
                if len(chunk["choices"]) == 0:
                    continue

                chunk_m, generation_info, content = self._build_stream_chunk(
                    chunk,
                    first_chunk,
                )
                first_chunk = False
                saw_converted_chunk = True
                generation_chunk = ChatGenerationChunk(
                    message=chunk_m,
                    generation_info=generation_info,
                )
                if _is_terminal_stream_chunk(generation_chunk):
                    terminal_chunk = _accept_terminal_chunk(
                        terminal_chunk, generation_chunk, route="Legacy"
                    )
                    continue
                if terminal_chunk is not None:
                    raise ValueError(
                        "Legacy stream emitted content after its terminal chunk"
                    )
                if run_manager:
                    run_manager.on_llm_new_token(content, chunk=generation_chunk)
                yield generation_chunk
        finally:
            _close_stream_iterator(legacy_provider_stream)
        if not saw_converted_chunk:
            return
        if kwargs.get("response_format") is not None:
            terminal_chunk = _finalize_terminal_chunk(terminal_chunk)
        if terminal_chunk is not None:
            if run_manager:
                run_manager.on_llm_new_token(
                    terminal_chunk.text,
                    chunk=terminal_chunk,
                )
            yield terminal_chunk

    @override
    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        # Kept in the signature for LangChain compatibility, but wrapper-side
        # local stop handling was removed in 0.5.x. See MIGRATION.md.
        route = self._validate_request_before_upload(messages, kwargs)
        request_uploads = await self._aupload_attachments(messages)
        uploads = self._uploads_for_request(request_uploads)
        terminal_chunk: Optional[ChatGenerationChunk] = None
        saw_converted_chunk = False
        if route == "primary":
            primary_payload = self._build_primary_payload(
                messages,
                kwargs,
                cached_uploads=uploads,
            )
            state = primary.StreamState()
            provider_stream = self._client.achat.stream(primary_payload)
            try:
                async for event in provider_stream:
                    primary_chunk = primary.convert_stream_event(event, state=state)
                    if primary_chunk is None:
                        continue
                    saw_converted_chunk = True
                    if primary_chunk.message.content == []:
                        primary_chunk = ChatGenerationChunk(
                            message=primary_chunk.message.model_copy(
                                update={"content": ""}
                            ),
                            generation_info=primary_chunk.generation_info,
                        )
                    if _is_authoritative_primary_terminal(primary_chunk):
                        terminal_chunk = _accept_terminal_chunk(
                            terminal_chunk, primary_chunk, route="Primary"
                        )
                        continue
                    if terminal_chunk is not None:
                        terminal_chunk = _merge_terminal_continuation(
                            terminal_chunk,
                            primary_chunk,
                            route="Primary",
                        )
                        continue
                    if run_manager:
                        await run_manager.on_llm_new_token(
                            primary_chunk.text, chunk=primary_chunk
                        )
                    yield primary_chunk
            finally:
                await _aclose_stream_iterator(provider_stream)
            if not saw_converted_chunk:
                return
            if terminal_chunk is None:
                raise ValueError("Primary stream ended before response.message.done")
            terminal_chunk = _finalize_terminal_chunk(terminal_chunk)
            if run_manager:
                await run_manager.on_llm_new_token(
                    terminal_chunk.text,
                    chunk=terminal_chunk,
                )
            yield terminal_chunk
            return
        self._validate_legacy_kwargs(kwargs)
        payload = self._build_legacy_payload(messages, uploads, **kwargs)
        first_chunk = True

        legacy_provider_stream = self._client.astream(payload)
        try:
            async for chunk_d in legacy_provider_stream:
                chunk = chunk_d if isinstance(chunk_d, dict) else chunk_d.model_dump()
                if len(chunk["choices"]) == 0:
                    continue

                chunk_m, generation_info, content = self._build_stream_chunk(
                    chunk,
                    first_chunk,
                )
                first_chunk = False
                saw_converted_chunk = True
                generation_chunk = ChatGenerationChunk(
                    message=chunk_m,
                    generation_info=generation_info,
                )
                if _is_terminal_stream_chunk(generation_chunk):
                    terminal_chunk = _accept_terminal_chunk(
                        terminal_chunk, generation_chunk, route="Legacy"
                    )
                    continue
                if terminal_chunk is not None:
                    raise ValueError(
                        "Legacy stream emitted content after its terminal chunk"
                    )
                if run_manager:
                    await run_manager.on_llm_new_token(content, chunk=generation_chunk)
                yield generation_chunk
        finally:
            await _aclose_stream_iterator(legacy_provider_stream)
        if not saw_converted_chunk:
            return
        if kwargs.get("response_format") is not None:
            terminal_chunk = _finalize_terminal_chunk(terminal_chunk)
        if terminal_chunk is not None:
            if run_manager:
                await run_manager.on_llm_new_token(
                    terminal_chunk.text,
                    chunk=terminal_chunk,
                )
            yield terminal_chunk

    def bind_functions(
        self,
        functions: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, type]],
        function_call: Optional[str] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind functions (legacy) to this chat model.

        Args:
            functions: A list of function definitions to bind to this chat model.
                Can be  a dictionary, pydantic model, or callable. Pydantic
                models and callables will be automatically converted to
                their schema dictionary representation.
            function_call: Which function to require the model to call.
                Supported values:
                - ``None``: Do not force a function call (model decides).
                - ``"auto"``: Let the model decide whether to call a function.
                - ``"none"``: Explicitly disable function calling.
                - ``"<function_name>"``: Force a specific function by name.
            kwargs: Any additional parameters forwarded to the underlying
                runnable binding.
        """
        formatted_functions = [convert_to_gigachat_function(fn) for fn in functions]
        if function_call is not None:
            if function_call in ("auto", "none"):
                kwargs = {**kwargs, "function_call": function_call}
            else:
                available_names = [fn.get("name") for fn in formatted_functions]
                if function_call not in available_names:
                    available = ", ".join(n for n in available_names if n)
                    available = available or "<unknown>"
                    raise ValueError(
                        f"Function call {function_call} was specified, but it was "
                        f"not found in provided functions: {available}."
                    )
                function_call_ = {"name": function_call}
                kwargs = {**kwargs, "function_call": function_call_}
        return super().bind(functions=formatted_functions, **kwargs)

    @override
    def with_structured_output(
        self,
        schema: Dict[str, Any] | type | None,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Dict | BaseModel]:
        """Return a model wrapper that formats outputs to match a schema.

        Args:
            schema: Output schema. Can be a dict-like tool/schema description
                or a Pydantic class. Pass ``None`` with ``method="json_mode"``
                to request a native JSON object without a schema on the primary
                API route.
            include_raw: If ``False``, return only parsed structured output.
                If ``True``, return a dict with ``raw``, ``parsed``, and
                ``parsing_error`` keys.
            **kwargs: Additional options for structured output.
                Supported keys:
                - ``method``: ``"function_calling"`` (default),
                  ``"json_schema"`` (native API-level JSON Schema
                  constraint; requires a model that supports
                  ``response_format``), ``"json_mode"`` (schema-less native
                  JSON on the primary API route), or
                  ``"format_instructions"`` (legacy).
                - ``strict``: best-effort strict schema adherence. Only
                  valid with ``method="json_schema"``. Defaults to ``True``.

        Raises:
            ValueError: If ``method`` is unsupported, ``strict`` is passed
                with a method other than ``"json_schema"``, or unknown
                kwargs are provided.
            TypeError: If ``method`` needs a schema and ``schema`` is neither
                a ``dict`` nor a ``pydantic.BaseModel`` subclass.

        Returns:
            Runnable that keeps the same input type as this chat model and
            returns parsed structured output (or a raw+parsed payload when
            ``include_raw=True``).
        """
        method = kwargs.pop("method", "function_calling")
        if method not in (
            "function_calling",
            "json_schema",
            "json_mode",
            "format_instructions",
        ):
            raise ValueError(
                "Unrecognized method. Expected 'function_calling', 'json_schema', "
                "'json_mode' or 'format_instructions'. "
                f"Received: {method}"
            )
        native_json_mode = method == "json_mode" and schema is None
        if method == "json_mode" and schema is not None:
            warnings.warn(
                "Legacy method='json_mode' behavior is deprecated; use "
                "method='json_schema', or use the primary API route with "
                "schema=None for native schema-less JSON.",
                DeprecationWarning,
                stacklevel=2,
            )
        strict = kwargs.pop("strict", None)
        if strict is not None and method != "json_schema":
            raise ValueError("`strict` is only supported with method='json_schema'.")
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")
        if schema is None and method != "json_mode":
            raise TypeError(f"method={method!r} requires a schema.")
        output_parser: OutputParserLike
        parser_runnable: Runnable[Any, Any]
        if method == "function_calling":
            assert schema is not None
            func = convert_to_gigachat_tool(schema)["function"]
            key_name = func.get(
                "name", func.get("title")
            )  # In case of pydantic from JSON (For openai capability)
            if _is_pydantic_class(schema):
                output_parser = PydanticToolsParser(
                    tools=[schema],
                    first_tool_only=True,
                )
            else:
                output_parser = JsonOutputKeyToolsParser(
                    key_name=key_name, first_tool_only=True
                )
            llm = self.bind_tools([schema], tool_choice=key_name)
        else:
            if method == "json_schema":
                if _is_pydantic_class(schema):
                    response_format_schema = model_to_json_schema(schema)
                elif isinstance(schema, dict):
                    response_format_schema = copy.deepcopy(schema)
                else:
                    raise TypeError(
                        "schema must be a dict or a pydantic.BaseModel "
                        f"subclass; got {type(schema).__name__}"
                    )
                response_format = gm.JsonSchemaResponseFormat(
                    schema=response_format_schema,
                    strict=strict if strict is not None else True,
                )
                llm = self.bind(response_format=response_format)
            elif native_json_mode:
                llm = self.bind(**{_SCHEMA_LESS_JSON_MODE_KEY: True})
            else:
                llm = self
            if _is_pydantic_class(schema):
                output_parser = PydanticOutputParser(pydantic_object=schema)
            else:
                output_parser = JsonOutputParser()
            if method == "format_instructions":
                assert schema is not None
                format_instructions = _format_instructions_for_schema(schema)

                def _inject_fi(
                    _input: LanguageModelInput,
                ) -> LanguageModelInput:
                    return _add_format_instructions(_input, format_instructions)

                llm = RunnableLambda(_inject_fi) | llm
            if method == "json_schema" or native_json_mode:
                parser_runnable = (
                    RunnableLambda(_require_successful_structured_finish)
                    | output_parser
                )
                if native_json_mode:
                    parser_runnable = parser_runnable | RunnableLambda(
                        _require_json_object
                    )
            else:
                parser_runnable = output_parser

        if method == "function_calling":
            parser_runnable = output_parser

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | parser_runnable,
                parsing_error=lambda _: None,
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        else:
            return llm | parser_runnable

    @override
    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "any", "none"], bool]
        ] = None,
        strict: Optional[bool] = None,
        response_format: Optional[Union[Dict[str, Any], Type[BaseModel]]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind tools and an optional structured response schema to this model."""
        if strict is not None and response_format is None:
            raise ValueError("strict is supported only together with response_format.")
        if tool_choice == "any":
            if self.allow_any_tool_choice_fallback:
                warnings.warn(
                    "GigaChat API does not support tool_choice='any'; the "
                    "allow_any_tool_choice_fallback compatibility option maps it "
                    "to 'auto', which does not preserve forced-tool semantics.",
                    UserWarning,
                    stacklevel=2,
                )
                tool_choice = "auto"
            else:
                raise ValueError(
                    "GigaChat API does not support tool_choice='any', and mapping "
                    "it to 'auto' would not preserve forced-tool semantics. For "
                    "create_agent structured output, either pass "
                    "ProviderStrategy(schema) explicitly or provide a verified "
                    "model profile={'structured_output': True}. Otherwise use "
                    "'auto' or a concrete tool name."
                )
        formatted_tools = [normalize_tool_for_binding(tool) for tool in tools]
        if tool_choice is not None and tool_choice is not False:
            if isinstance(tool_choice, str):
                if not tool_choice:
                    raise ValueError("tool_choice must not be an empty string")
                if tool_choice not in ("auto", "none"):
                    tool_choice = {"name": tool_choice}
            elif isinstance(tool_choice, bool):
                if not formatted_tools:
                    raise ValueError("tool_choice can not be bool if tools are empty")
                tool_choice = {"name": _get_tool_name(formatted_tools[0])}
            elif isinstance(tool_choice, dict):
                if not tool_choice:
                    raise ValueError("tool_choice must not be an empty mapping")
            else:
                raise ValueError(
                    f"Unrecognized tool_choice type. Expected str, bool or dict. "
                    f"Received: {tool_choice}"
                )
            kwargs["function_call"] = tool_choice
        if response_format is not None:
            kwargs["response_format"] = response_format
        if strict is not None:
            kwargs["strict"] = strict
        return super().bind(tools=formatted_tools, **kwargs)


def _is_pydantic_class(obj: Any) -> TypeGuard[Type[BaseModel]]:
    return isinstance(obj, type) and is_basemodel_subclass(obj)


def _has_tool_call(message: BaseMessage | BaseMessageChunk) -> bool:
    if not isinstance(message, (AIMessage, AIMessageChunk)):
        return False
    return bool(
        message.tool_calls
        or message.invalid_tool_calls
        or getattr(message, "tool_call_chunks", [])
    )


def _is_terminal_stream_chunk(chunk: ChatGenerationChunk) -> bool:
    message = chunk.message
    if isinstance(message, AIMessageChunk) and message.chunk_position == "last":
        return True
    return bool(
        chunk.generation_info and chunk.generation_info.get("finish_reason") is not None
    )


def _close_stream_iterator(iterator: Iterator[Any]) -> None:
    """Close a synchronous SDK stream when the iterator supports it."""
    close = getattr(iterator, "close", None)
    if callable(close):
        close()


async def _aclose_stream_iterator(iterator: AsyncIterator[Any]) -> None:
    """Close an asynchronous SDK stream when the iterator supports it."""
    aclose = getattr(iterator, "aclose", None)
    if callable(aclose):
        await aclose()


def _is_authoritative_primary_terminal(chunk: ChatGenerationChunk) -> bool:
    """Return whether a primary chunk completes the whole chat response."""
    message = chunk.message
    return isinstance(message, AIMessageChunk) and message.chunk_position == "last"


def _accept_terminal_chunk(
    current: Optional[ChatGenerationChunk],
    incoming: ChatGenerationChunk,
    *,
    route: str,
) -> ChatGenerationChunk:
    """Accept one authoritative terminal, deduplicating exact repeats."""
    if current is None or current == incoming:
        return incoming if current is None else current
    raise ValueError(f"{route} stream emitted conflicting terminal chunks")


def _merge_terminal_continuation(
    terminal: ChatGenerationChunk,
    continuation: ChatGenerationChunk,
    *,
    route: str,
) -> ChatGenerationChunk:
    """Merge metadata emitted after an authoritative terminal into that terminal."""
    if continuation.text or _has_tool_call(continuation.message):
        raise ValueError(f"{route} stream emitted content after its terminal chunk")

    merged = terminal + continuation
    generation_info = dict(merged.generation_info or {})
    terminal_generation_info = terminal.generation_info or {}
    if "finish_reason" in terminal_generation_info:
        generation_info["finish_reason"] = terminal_generation_info["finish_reason"]

    message = merged.message
    if isinstance(message, AIMessageChunk):
        response_metadata = dict(message.response_metadata)
        terminal_response_metadata = terminal.message.response_metadata
        if "finish_reason" in terminal_response_metadata:
            response_metadata["finish_reason"] = terminal_response_metadata[
                "finish_reason"
            ]
        message = message.model_copy(
            update={
                "chunk_position": "last",
                "id": continuation.message.id or message.id,
                "response_metadata": response_metadata,
            }
        )

    return ChatGenerationChunk(
        message=message,
        generation_info=generation_info or None,
    )


def _finalize_terminal_chunk(
    terminal_chunk: Optional[ChatGenerationChunk],
) -> ChatGenerationChunk:
    """Mark the buffered terminal chunk without interpreting its content."""
    chunk = terminal_chunk or ChatGenerationChunk(
        message=AIMessageChunk(content=""),
    )
    message = chunk.message
    if not isinstance(message, AIMessageChunk):
        return chunk
    return ChatGenerationChunk(
        message=message.model_copy(update={"chunk_position": "last"}),
        generation_info=chunk.generation_info,
    )


def _require_successful_structured_finish(message: BaseMessage) -> BaseMessage:
    """Reject syntactically valid structured data from incomplete generations."""
    finish_reason = message.response_metadata.get("finish_reason")
    if finish_reason != "stop":
        raise OutputParserException(
            "GigaChat native structured output was not completed successfully: "
            f"finish_reason={finish_reason!r}.",
            llm_output=message.text,
        )
    return message


def _require_json_object(value: Any) -> dict[str, Any]:
    """Require the object shape promised by schema-less ``json_mode``."""
    if not isinstance(value, dict):
        raise OutputParserException(
            "GigaChat schema-less JSON mode returned valid JSON, but not a JSON "
            "object.",
            llm_output=json.dumps(value, ensure_ascii=False),
        )
    return value


def _format_instructions_for_schema(schema: Dict[str, Any] | type) -> str:
    """Build format instructions for Pydantic or raw JSON-schema input.

    Both branches funnel through the public ``JSON_FORMAT_INSTRUCTIONS``
    template from langchain-core, so the prompt is identical for Pydantic
    classes and raw JSON-schema dicts.
    """
    if _is_pydantic_class(schema):
        json_schema = model_to_json_schema(schema)
    elif isinstance(schema, dict):
        json_schema = schema
    else:
        raise TypeError(
            "schema must be a Pydantic class or a dict (JSON Schema); "
            f"got {type(schema).__name__}."
        )
    # Drop top-level "title" and "type" for brevity, matching PydanticOutputParser.
    reduced = {k: v for k, v in json_schema.items() if k not in ("title", "type")}
    return JSON_FORMAT_INSTRUCTIONS.format(
        schema=json.dumps(reduced, ensure_ascii=False)
    )


def _add_format_instructions(
    _input: LanguageModelInput, format_instructions: str
) -> LanguageModelInput:
    """Append format_instructions as a trailing human message to the LLM input.

    Preserves the container type where meaningful: string in, string out;
    PromptValue in, ChatPromptValue out; otherwise a list of messages.
    """
    fi_message = HumanMessage(content=format_instructions)
    if isinstance(_input, str):
        return f"{_input}\n\n{format_instructions}"
    if isinstance(_input, ChatPromptValue):
        return ChatPromptValue(messages=[*_input.messages, fi_message])
    if isinstance(_input, PromptValue):
        return ChatPromptValue(messages=[*_input.to_messages(), fi_message])
    if isinstance(_input, BaseMessage):
        return [_input, fi_message]
    if isinstance(_input, Sequence):
        return [*_input, fi_message]
    raise TypeError(
        f"Unsupported LanguageModelInput type: {type(_input).__name__}. "
        "Expected str, BaseMessage, Sequence[BaseMessage], or PromptValue."
    )
