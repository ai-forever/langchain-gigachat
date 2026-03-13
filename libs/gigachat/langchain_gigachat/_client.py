"""Shared GigaChat client mixin for chat models and embeddings."""

from __future__ import annotations

import ssl
from functools import cached_property
from typing import Any, Dict, Optional, Tuple

import gigachat
from langchain_core.load.serializable import Serializable
from pydantic import ConfigDict


class _GigaChatClientMixin(Serializable):
    """Mixin providing GigaChat SDK client initialization, auth, and connection config.

    Subclasses inherit shared connection/authentication fields and a cached
    ``_client`` property that creates a ``gigachat.GigaChat`` instance.
    Keeping this logic in one place avoids chat/embeddings drift for auth,
    retry, and secret-redaction behavior such as ``lc_secrets``.

    Override ``_get_client_init_kwargs`` to inject additional SDK parameters
    (e.g. ``profanity_check`` in the chat model).
    """

    base_url: Optional[str] = None
    """Base API URL."""
    auth_url: Optional[str] = None
    """Auth URL."""
    credentials: Optional[str] = None
    """Auth token."""
    scope: Optional[str] = None
    """Permission scope for access token."""

    access_token: Optional[str] = None
    """Access token for GigaChat."""

    model: Optional[str] = None
    """Model name to use."""
    user: Optional[str] = None
    """Username for authentication."""
    password: Optional[str] = None
    """Password for authentication."""

    timeout: Optional[float] = None
    """Timeout for requests."""
    verify_ssl_certs: Optional[bool] = None
    """Check certificates for all requests."""

    ssl_context: Optional[ssl.SSLContext] = None
    """SSL context."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ca_bundle_file: Optional[str] = None
    """Path to CA bundle file."""
    cert_file: Optional[str] = None
    """Path to certificate file."""
    key_file: Optional[str] = None
    """Path to key file."""
    key_file_password: Optional[str] = None
    """Password for key file."""

    max_retries: Optional[int] = None
    """Maximum number of retries for transient errors.

    SDK default is 0 (disabled). When using LangChain's built-in retry mechanisms
    (e.g. ``.with_retry()``), keep this at ``None``/``0`` to avoid multiplicative
    retry counts.
    """
    max_connections: Optional[int] = None
    """Maximum number of simultaneous connections to the GigaChat API."""
    retry_backoff_factor: Optional[float] = None
    """Backoff factor for retry delays (SDK default: 0.5).

    The delay between retries is calculated as
    ``retry_backoff_factor * (2 ** (retry_number - 1))`` seconds.
    """
    retry_on_status_codes: Optional[Tuple[int, ...]] = None
    """HTTP status codes that trigger a retry.

    SDK default: ``(429, 500, 502, 503, 504)``.
    """

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {
            "credentials": "GIGACHAT_CREDENTIALS",
            "access_token": "GIGACHAT_ACCESS_TOKEN",
            "password": "GIGACHAT_PASSWORD",
            "key_file_password": "GIGACHAT_KEY_FILE_PASSWORD",
        }

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    def _get_client_init_kwargs(self) -> Dict[str, Any]:
        """Return kwargs for ``gigachat.GigaChat`` initialization.

        Subclasses can override to add extra parameters (e.g. ``profanity_check``).
        Always call ``super()._get_client_init_kwargs()`` and update the result.
        """
        return {
            "base_url": self.base_url,
            "auth_url": self.auth_url,
            "credentials": self.credentials,
            "scope": self.scope,
            "access_token": self.access_token,
            "model": self.model,
            "user": self.user,
            "password": self.password,
            "timeout": self.timeout,
            "ssl_context": self.ssl_context,
            "verify_ssl_certs": self.verify_ssl_certs,
            "ca_bundle_file": self.ca_bundle_file,
            "cert_file": self.cert_file,
            "key_file": self.key_file,
            "key_file_password": self.key_file_password,
            "max_retries": self.max_retries,
            "max_connections": self.max_connections,
            "retry_backoff_factor": self.retry_backoff_factor,
            "retry_on_status_codes": self.retry_on_status_codes,
        }

    @cached_property
    def _client(self) -> gigachat.GigaChat:
        """Return GigaChat API client."""
        return gigachat.GigaChat(**self._get_client_init_kwargs())
