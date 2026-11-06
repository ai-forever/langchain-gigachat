"""Offline checks run from outside the source checkout against an installed wheel."""

import importlib.util
import sys
from importlib.metadata import metadata, version
from pathlib import Path

import httpx2

import langchain_gigachat
from langchain_gigachat import GigaChat, GigaChatEmbeddings

assert sys.version_info[:2] == (3, 15)
assert sys.version_info.releaselevel == "final"
assert version("langchain-gigachat") == "0.5.3a1"
assert version("gigachat") == "0.3.0a1"
requirements = metadata("langchain-gigachat").get_all("Requires-Dist") or []
assert "gigachat==0.3.0a1" in requirements
package = Path(langchain_gigachat.__file__).resolve().parent
assert package.is_relative_to(Path(sys.prefix).resolve()), package
assert (package / "py.typed").is_file()
assert bool(importlib.util.find_spec("aiohttp")) == (sys.argv[1] == "aiohttp")
assert GigaChatEmbeddings is not None


def respond(request: httpx2.Request) -> httpx2.Response:
    assert request.url.path.endswith("/chat/completions")
    return httpx2.Response(
        200,
        json={
            "choices": [
                {
                    "message": {"role": "assistant", "content": "wheel smoke"},
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
            "created": 1,
            "model": "test-model",
            "object": "chat.completion",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


with httpx2.Client(transport=httpx2.MockTransport(respond)) as client:
    model = GigaChat(
        access_token="offline-token", model="test-model", http_client=client
    )
    try:
        assert model.invoke("hello").content == "wheel smoke"
    finally:
        model.close()
    assert not client.is_closed
