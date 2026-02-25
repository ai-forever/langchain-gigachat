<div align="center" id="top">

[![PyPI](https://img.shields.io/pypi/v/langchain-gigachat?style=flat-square)](https://pypi.org/project/langchain-gigachat/)
[![Python](https://img.shields.io/pypi/pyversions/langchain-gigachat?style=flat-square)](https://pypi.org/project/langchain-gigachat/)
[![CI](https://img.shields.io/github/actions/workflow/status/ai-forever/langchain-gigachat/check_diffs.yml?style=flat-square)](https://github.com/ai-forever/langchain-gigachat/actions/workflows/check_diffs.yml)
[![License](https://img.shields.io/github/license/ai-forever/langchain-gigachat?style=flat-square)](https://opensource.org/license/MIT)
[![Downloads](https://img.shields.io/pypi/dm/langchain-gigachat?style=flat-square)](https://pypistats.org/packages/langchain-gigachat)

[English](README.md) | [Русский](README-ru_RU.md)

</div>

# langchain-gigachat

LangChain integration for [GigaChat](https://giga.chat/) — a large language model.

This library is part of [GigaChain](https://github.com/ai-forever/gigachain) and wraps the [GigaChat Python SDK](https://github.com/ai-forever/gigachat) with LangChain-compatible interfaces.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Authentication](#authentication)
- [Usage Examples](#usage-examples)
  - [Chat](#chat)
  - [Streaming](#streaming)
  - [Async](#async)
  - [Embeddings](#embeddings)
- [Tool Calling](#tool-calling)
- [Structured Output](#structured-output)
- [Attachments](#attachments)
- [Configuration](#configuration)
- [Related Projects](#related-projects)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Chat completions** — synchronous and asynchronous, with streaming
- **Embeddings** — text vectorization via `GigaChatEmbeddings`
- **Tool calling** — `giga_tool` decorator with GigaChat-specific extras
- **Structured output** — Pydantic models and JSON mode
- **Attachments** — images, audio, and documents via the Files API
- **Environment-based configuration** — all parameters configurable via `GIGACHAT_` env vars

## Installation

```bash
pip install -U langchain-gigachat
```

**Requirements:** Python 3.10+

> **Note:** In production, keep TLS verification enabled (default).
> See [Authentication](#authentication) for certificate setup.

## Authentication

Set environment variables and let the SDK pick them up:

```bash
export GIGACHAT_CREDENTIALS="your-authorization-key"
export GIGACHAT_SCOPE="GIGACHAT_API_PERS"  # GIGACHAT_API_B2B or GIGACHAT_API_CORP for enterprise
```

After this, `GigaChat()` works without any arguments in code.

If your environment requires a specific TLS certificate:

```bash
export GIGACHAT_CA_BUNDLE_FILE="/path/to/certs.pem"
```

> **Warning:** Disabling TLS verification (`verify_ssl_certs=False`) is for local development only and is not recommended for production.

For detailed instructions on obtaining credentials and certificates, see the [GigaChat SDK](https://github.com/ai-forever/gigachat) and [API docs](https://developers.sber.ru/docs/ru/gigachat).

## Usage Examples

> The examples below assume authentication is configured via environment variables.
> See [Authentication](#authentication).

### Chat

```python
from langchain_gigachat import GigaChat

llm = GigaChat(credentials="your-authorization-key")

msg = llm.invoke("Hello, GigaChat!")
print(msg.content)
```

### Streaming

Receive tokens as they are generated:

```python
from langchain_gigachat import GigaChat

llm = GigaChat()

for chunk in llm.stream("Write a short poem about programming"):
    print(chunk.content, end="", flush=True)
print()
```

### Async

Use async/await for non-blocking operations:

```python
import asyncio

from langchain_gigachat import GigaChat


async def main():
    llm = GigaChat()
    msg = await llm.ainvoke("Explain quantum computing in simple terms.")
    print(msg.content)


asyncio.run(main())
```

### Embeddings

Generate vector representations of text:

```python
from langchain_gigachat import GigaChatEmbeddings

emb = GigaChatEmbeddings(model="Embeddings")

vector = emb.embed_query("Привет!")
print(len(vector))
```

## Tool Calling

`giga_tool` is a drop-in replacement for LangChain `@tool` with GigaChat-specific extras:

```python
from langchain_gigachat import GigaChat
from langchain_gigachat.tools.giga_tool import giga_tool


@giga_tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"{city}: sunny, 22C"


llm = GigaChat()
llm_with_tools = llm.bind_tools([get_weather], tool_choice="auto")

msg = llm_with_tools.invoke("What's the weather in Tokyo?")
print(msg.tool_calls)
```

> **Note:** `tool_choice="any"` is not supported by the GigaChat API. Use `"auto"`, `"none"`, or a specific tool name. If upstream code passes `"any"`, set `allow_any_tool_choice_fallback=True` to silently convert it to `"auto"`.

## Structured Output

Extract typed data from model responses:

```python
from pydantic import BaseModel, Field

from langchain_gigachat import GigaChat


class Answer(BaseModel):
    text: str = Field(description="Final answer")
    confidence: float = Field(ge=0, le=1, description="Confidence 0..1")


llm = GigaChat()
chain = llm.with_structured_output(Answer)

parsed = chain.invoke("What is the capital of France? Rate your confidence.")
print(parsed)
```

JSON mode is also available: `llm.with_structured_output(Answer, method="json_mode")`.

## Attachments

Upload a file via the Files API, then reference it in `content_blocks`:

```python
from langchain_core.messages import HumanMessage

from langchain_gigachat import GigaChat

llm = GigaChat()

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

> **Note:** Base64 data URLs in `image_url` / `audio_url` / `document_url` blocks can be auto-uploaded with `auto_upload_attachments=True`, but prefer explicit `upload_file()` in production.

## Configuration

All parameters can be passed to `GigaChat(...)` / `GigaChatEmbeddings(...)` directly or via environment variables with the `GIGACHAT_` prefix.

| Variable | Description |
|---|---|
| `GIGACHAT_CREDENTIALS` | OAuth credentials (recommended) |
| `GIGACHAT_ACCESS_TOKEN` | Pre-obtained JWT token |
| `GIGACHAT_SCOPE` | API scope (`GIGACHAT_API_PERS`, `GIGACHAT_API_B2B`, `GIGACHAT_API_CORP`) |
| `GIGACHAT_BASE_URL` | Custom API endpoint |
| `GIGACHAT_VERIFY_SSL_CERTS` | TLS verification on/off |
| `GIGACHAT_CA_BUNDLE_FILE` | Path to CA bundle |

> **Note:** Retries are handled by the underlying `gigachat` SDK (`max_retries`, `retry_backoff_factor`, `retry_on_status_codes`). Don't combine them with LangChain `.with_retry()` — the attempts multiply.

## Related Projects

- **[GigaChain](https://github.com/ai-forever/gigachain)** — a set of solutions for developing LLM applications and multi-agent systems, with support for LangChain, LangGraph, LangChain4j, GigaChat and other LLMs
- **[GigaChat Python SDK](https://github.com/ai-forever/gigachat)** — the underlying Python SDK that powers this integration
- [GigaChat API docs](https://developers.sber.ru/docs/ru/gigachat)

## Contributing

See [`CONTRIBUTING.md`](../../CONTRIBUTING.md). Development happens under `libs/gigachat`:

```bash
uv sync
make lint_package
make test
```

## License

This project is licensed under the MIT License.
