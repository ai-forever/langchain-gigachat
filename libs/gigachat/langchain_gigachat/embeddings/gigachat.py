from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.embeddings import Embeddings

from langchain_gigachat._client import _GigaChatClientMixin


class GigaChatEmbeddings(_GigaChatClientMixin, Embeddings):
    """GigaChat Embeddings models."""

    timeout: Optional[float] = 600
    """Timeout for requests. By default it works for long requests."""

    prefix_query: str = (
        "Дано предложение, необходимо найти его парафраз \nпредложение: "
    )

    use_prefix_query: bool = False

    def _get_embed_kwargs(self) -> Dict[str, Any]:
        """Return extra kwargs for the SDK embeddings call."""
        kwargs: Dict[str, Any] = {}
        if self.model is not None:
            kwargs["model"] = self.model
        return kwargs

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using a GigaChat embeddings model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        if not texts:
            return []
        response = self._client.embeddings(texts=texts, **self._get_embed_kwargs())
        return [item.embedding for item in response.data]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using a GigaChat embeddings model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        if not texts:
            return []
        response = await self._client.aembeddings(
            texts=texts, **self._get_embed_kwargs()
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using a GigaChat embeddings model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        if self.use_prefix_query:
            text = self.prefix_query + text
        return self.embed_documents(texts=[text])[0]

    async def aembed_query(self, text: str) -> List[float]:
        """Embed a query using a GigaChat embeddings model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        if self.use_prefix_query:
            text = self.prefix_query + text
        docs = await self.aembed_documents(texts=[text])
        return docs[0]
