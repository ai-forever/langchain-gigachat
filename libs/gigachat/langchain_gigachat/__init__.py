from importlib.metadata import PackageNotFoundError, version

from langchain_gigachat.chat_models import GigaChat
from langchain_gigachat.embeddings import GigaChatEmbeddings

try:
    __version__ = version("langchain-gigachat")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["GigaChat", "GigaChatEmbeddings"]
