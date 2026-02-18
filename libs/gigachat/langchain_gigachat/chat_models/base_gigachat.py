from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

import gigachat.models as gm
from gigachat._types import FileTypes

from langchain_gigachat._client import _GigaChatClientMixin


class _BaseGigaChat(_GigaChatClientMixin):
    profanity_check: Optional[bool] = None
    """Check for profanity."""
    streaming: bool = False
    """Whether to stream the results or not."""
    temperature: Optional[float] = None
    """What sampling temperature to use."""
    max_tokens: Optional[int] = None
    """Maximum number of tokens to generate."""
    use_api_for_tokens: bool = False
    """Use GigaChat API for tokens count."""
    flags: Optional[List[str]] = None
    """Feature flags."""
    top_p: Optional[float] = None
    """top_p value to use for nucleus sampling. Must be between 0.0 and 1.0."""
    repetition_penalty: Optional[float] = None
    """The penalty applied to repeated tokens."""
    update_interval: Optional[float] = None
    """Minimum interval in seconds that elapses between sending tokens."""
    reasoning_effort: Optional[str] = None
    """
    Reasoning effort for reasoning-capable models (e.g. GigaChat-2-Reasoning).
    When set, the API may return reasoning_content in the assistant message.
    """

    @property
    def _llm_type(self) -> str:
        return "giga-chat-model"

    def _get_client_init_kwargs(self) -> Dict[str, Any]:
        kwargs = super()._get_client_init_kwargs()
        kwargs["profanity_check"] = self.profanity_check
        kwargs["flags"] = self.flags
        return kwargs

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "temperature": self.temperature,
            "model": self.model,
            "profanity_check": self.profanity_check,
            "streaming": self.streaming,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "reasoning_effort": self.reasoning_effort,
        }

    def tokens_count(
        self, input_: List[str], model: Optional[str] = None
    ) -> List[gm.TokensCount]:
        """Get tokens of string list."""
        return self._client.tokens_count(input_, model)

    async def atokens_count(
        self, input_: List[str], model: Optional[str] = None
    ) -> List[gm.TokensCount]:
        """Get tokens of strings list (async)."""
        return await self._client.atokens_count(input_, model)

    def get_models(self) -> gm.Models:
        """Get available models of Gigachat."""
        return self._client.get_models()

    async def aget_models(self) -> gm.Models:
        """Get available models of Gigachat (async)."""
        return await self._client.aget_models()

    def get_model(self, model: str) -> gm.Model:
        """Get info about model."""
        return self._client.get_model(model)

    async def aget_model(self, model: str) -> gm.Model:
        """Get info about model (async)."""
        return await self._client.aget_model(model)

    def get_num_tokens(self, text: str) -> int:
        """Count approximate number of tokens."""
        if self.use_api_for_tokens:
            return self.tokens_count([text])[0].tokens
        else:
            return round(len(text) / 4.6)

    def upload_file(
        self, file: FileTypes, purpose: Literal["general", "assistant"] = "general"
    ) -> gm.UploadedFile:
        return self._client.upload_file(file, purpose)

    async def aupload_file(
        self, file: FileTypes, purpose: Literal["general", "assistant"] = "general"
    ) -> gm.UploadedFile:
        return await self._client.aupload_file(file, purpose)

    def get_file(self, file_id: str) -> gm.UploadedFile:
        """Return file metadata by ID (SDK get_file)."""
        return self._client.get_file(file_id)

    async def aget_file(self, file_id: str) -> gm.UploadedFile:
        """Return file metadata by ID (async, SDK aget_file)."""
        return await self._client.aget_file(file_id)

    def get_file_content(self, file_id: str) -> gm.Image:
        """Download file content (base64) by ID. Uses SDK get_image."""
        return self._client.get_image(file_id)

    async def aget_file_content(self, file_id: str) -> gm.Image:
        """Download file content (base64) by ID (async, SDK aget_image)."""
        return await self._client.aget_image(file_id)

    def list_files(self) -> gm.UploadedFiles:
        """Return list of uploaded files (SDK get_files, GET /files)."""
        return self._client.get_files()

    async def alist_files(self) -> gm.UploadedFiles:
        """Return list of uploaded files (async, SDK aget_files)."""
        return await self._client.aget_files()

    def delete_file(self, file_id: str) -> gm.DeletedFile:
        """Delete a file by ID (SDK delete_file, DELETE /files/{{id}})."""
        return self._client.delete_file(file_id)

    async def adelete_file(self, file_id: str) -> gm.DeletedFile:
        """Delete a file by ID (async, SDK adelete_file)."""
        return await self._client.adelete_file(file_id)
