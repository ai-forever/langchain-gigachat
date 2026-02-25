<div align="center" id="top">

[![GitHub Release](https://img.shields.io/github/v/release/ai-forever/langchain-gigachat?style=flat-square)](https://github.com/ai-forever/langchain-gigachat/releases)
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/ai-forever/langchain-gigachat/check_diffs.yml?style=flat-square)](https://github.com/ai-forever/langchain-gigachat/actions/workflows/check_diffs.yml)
[![PyPI - Version](https://img.shields.io/pypi/v/langchain-gigachat?label=PyPI&style=flat-square)](https://pypi.org/project/langchain-gigachat/#history)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/langchain-gigachat?style=flat-square)](https://pypi.org/project/langchain-gigachat/)
[![GitHub License](https://img.shields.io/github/license/ai-forever/langchain-gigachat?style=flat-square)](https://opensource.org/license/MIT)
[![GitHub Downloads (all assets, all releases)](https://img.shields.io/pypi/dm/langchain-gigachat?style=flat-square)](https://pypistats.org/packages/langchain-gigachat)
[![GitHub Repo stars](https://img.shields.io/github/stars/ai-forever/langchain-gigachat?style=flat-square)](https://star-history.com/#ai-forever/langchain-gigachat)
[![GitHub Open Issues](https://img.shields.io/github/issues-raw/ai-forever/langchain-gigachat)](https://github.com/ai-forever/langchain-gigachat/issues)

[English](README.md) | [Русский](README-ru_RU.md)

</div>

# langchain-gigachat

LangChain integration for [GigaChat](https://giga.chat/) (chat models, embeddings, tool calling, and attachments).

This library is part of [GigaChain](https://github.com/ai-forever/gigachain).

## Quick Install

```bash
pip install -U langchain-gigachat
```


## 🤔 What is this?

This package provides:

- **Chat model**: `langchain_gigachat.GigaChat` (sync/async, streaming, tool calling, structured output)
- **Embeddings**: `langchain_gigachat.GigaChatEmbeddings`
- **Tools helper**: `langchain_gigachat.tools.giga_tool.giga_tool` (extends LangChain `@tool` with GigaChat-specific extras)
- **Attachments**: upload files and send them as message `content_blocks` (images/audio/documents)

## Requirements

- Python **3.10+**
- Access to GigaChat API (credentials, access token, or other supported auth methods)
- TLS root certificate (recommended). If your environment requires it, configure a CA bundle via `GIGACHAT_CA_BUNDLE_FILE` / `ca_bundle_file`.

For details on auth and certificates, see:
- [GigaChat SDK README](https://github.com/ai-forever/gigachat/blob/main/README.md)
- [GigaChat API docs](https://developers.sber.ru/docs/ru/gigachat)

## Quickstart

### Chat

```python
from langchain_gigachat import GigaChat

llm = GigaChat(
    credentials="YOUR_AUTHORIZATION_KEY",
    verify_ssl_certs=False,  # dev-only (recommended: configure CA bundle instead)
)

msg = llm.invoke("Hello, GigaChat!")
print(msg.content)
```

### Streaming

```python
from langchain_gigachat import GigaChat

llm = GigaChat(credentials="YOUR_AUTHORIZATION_KEY", verify_ssl_certs=False)

for chunk in llm.stream("Write a short poem about programming"):
    print(chunk.content, end="", flush=True)
print()
```

### Async

```python
import asyncio

from langchain_gigachat import GigaChat


async def main() -> None:
    llm = GigaChat(credentials="YOUR_AUTHORIZATION_KEY", verify_ssl_certs=False)
    msg = await llm.ainvoke("Explain quantum computing in simple terms.")
    print(msg.content)


asyncio.run(main())
```

### Embeddings

```python
from langchain_gigachat import GigaChatEmbeddings

emb = GigaChatEmbeddings(
    credentials="YOUR_AUTHORIZATION_KEY",
    verify_ssl_certs=False,
    model="Embeddings",
)

vector = emb.embed_query("Привет!")
print(len(vector))
```

## Tool calling

Use `giga_tool` (a drop-in alternative to LangChain `@tool` with extra fields supported by GigaChat).

```python
from langchain_gigachat import GigaChat
from langchain_gigachat.tools.giga_tool import giga_tool


@giga_tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"{city}: sunny"


llm = GigaChat(credentials="YOUR_AUTHORIZATION_KEY", verify_ssl_certs=False)
llm_with_tools = llm.bind_tools([get_weather], tool_choice="auto")

msg = llm_with_tools.invoke("What's the weather in Tokyo?")
print(msg.tool_calls)
```

Notes:

- `tool_choice="any"` is **not supported** by the GigaChat API. Use `"auto"`, `"none"`, or a specific tool name. If you must accept `"any"` from upstream code, set `allow_any_tool_choice_fallback=True` in `GigaChat(...)` to convert it to `"auto"`.

## Structured output

```python
from pydantic import BaseModel, Field

from langchain_gigachat import GigaChat


class Answer(BaseModel):
    """Structured answer."""

    text: str = Field(description="Final answer")
    confidence: float = Field(ge=0, le=1, description="Confidence 0..1")


llm = GigaChat(credentials="YOUR_AUTHORIZATION_KEY", verify_ssl_certs=False)
chain = llm.with_structured_output(Answer)
parsed = chain.invoke("Answer briefly and provide confidence.")
print(parsed)
```

You can also use JSON mode: `llm.with_structured_output(Answer, method="json_mode")`.

## Attachments (images/audio/documents)

Upload a file via the GigaChat Files API and pass it as a standard LangChain `content_blocks` attachment.

```python
from langchain_core.messages import HumanMessage

from langchain_gigachat import GigaChat

llm = GigaChat(credentials="YOUR_AUTHORIZATION_KEY", verify_ssl_certs=False)

with open("image.png", "rb") as f:
    uploaded = llm.upload_file(("image.png", f.read()))

msg = HumanMessage(
    content_blocks=[
        {"type": "text", "text": "Describe the image."},
        {"type": "image", "file_id": uploaded.id_},
    ]
)

reply = llm.invoke([msg])
print(reply.content)
```

## Configuration

All SDK parameters can be passed to `GigaChat(...)` / `GigaChatEmbeddings(...)` directly, or configured via environment variables (prefix `GIGACHAT_`).

Notes:

- If you embed Base64 data URLs into `image_url` / `audio_url` / `document_url` blocks, you can enable `auto_upload_attachments=True` to auto-upload them. This is **not recommended for production**; prefer explicit `upload_file(...)`.
- Retries are handled by the underlying `gigachat` SDK (`max_retries`, `retry_backoff_factor`, `retry_on_status_codes`). Avoid combining SDK retries with LangChain retries (e.g. `.with_retry()`), otherwise the effective attempts multiply.

Common variables:

| Variable | Meaning |
|---|---|
| `GIGACHAT_CREDENTIALS` | OAuth credentials (recommended default) |
| `GIGACHAT_ACCESS_TOKEN` | Pre-obtained access token (JWT) |
| `GIGACHAT_SCOPE` | API scope (`GIGACHAT_API_PERS`, `GIGACHAT_API_B2B`, `GIGACHAT_API_CORP`) |
| `GIGACHAT_BASE_URL` | API base URL |
| `GIGACHAT_VERIFY_SSL_CERTS` | Enable/disable TLS verification |
| `GIGACHAT_CA_BUNDLE_FILE` | Path to CA bundle file |

## 📖 Documentation

- **Source code**: `langchain_gigachat/`
- **GigaChat SDK**: [README](https://github.com/ai-forever/gigachat/blob/main/README.md)

## 💁 Contributing

See [`CONTRIBUTING.md`](../../CONTRIBUTING.md). Development happens under `libs/gigachat` (run `uv sync`, then `make lint_package` / `make test`).