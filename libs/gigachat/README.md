<div align="center" id="top">

[![PyPI](https://img.shields.io/pypi/v/langchain-gigachat?style=flat-square)](https://pypi.org/project/langchain-gigachat/)
[![Python](https://img.shields.io/pypi/pyversions/langchain-gigachat?style=flat-square)](https://pypi.org/project/langchain-gigachat/)
[![CI](https://img.shields.io/github/actions/workflow/status/ai-forever/langchain-gigachat/check_diffs.yml?style=flat-square)](https://github.com/ai-forever/langchain-gigachat/actions/workflows/check_diffs.yml)
[![License](https://img.shields.io/github/license/ai-forever/langchain-gigachat?style=flat-square)](https://opensource.org/license/MIT)
[![Downloads](https://img.shields.io/pypi/dm/langchain-gigachat?style=flat-square)](https://pypistats.org/packages/langchain-gigachat)

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
  - [Reasoning Models](#reasoning-models)
  - [API v2 (`/v2/chat/completions`)](#api-v2-v2chatcompletions)
- [Tool Calling](#tool-calling)
  - [Legacy tool transport](#legacy-tool-transport)
  - [Primary API v2 tool transport](#primary-api-v2-tool-transport)
- [Structured Output](#structured-output)
- [Attachments](#attachments)
  - [File Operations](#file-operations)
- [Configuration](#configuration)
  - [Constructor Parameters](#constructor-parameters)
  - [Environment Variables](#environment-variables)
- [Error Handling](#error-handling)
- [Related Projects](#related-projects)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Chat completions** — synchronous and asynchronous, with streaming
- **Embeddings** — text vectorization via `GigaChatEmbeddings`
- **Tool calling** — standard LangChain `@tool` with GigaChat metadata in `extras`
- **Structured output** — Pydantic models and JSON mode
- **Reasoning models** — `reasoning_effort` for thinking models
- **Attachments** — images, audio, and documents via the Files API
- **File operations** — upload, list, retrieve, and delete files
- **Configurable retry** — exponential backoff via the underlying SDK
- **Environment-based configuration** — all parameters configurable via `GIGACHAT_` env vars
- **Fully typed** — Pydantic V2 models with `py.typed` marker

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

> **Note:** Wrapper-side local `stop` handling was removed in `0.5.x`. The
> public methods still accept `stop` for LangChain signature compatibility, but
> `langchain-gigachat` no longer applies stop-sequence truncation itself. See
> [`MIGRATION.md`](MIGRATION.md) before carrying `stop=...` call sites forward.

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

### Reasoning Models

Use `reasoning_effort` with reasoning-capable models:

```python
from langchain_gigachat import GigaChat

llm = GigaChat(model="GigaChat-2-Reasoning", reasoning_effort="high")

msg = llm.invoke("How many r's are in the word 'strawberry'?")
print(msg.content)
print(msg.additional_kwargs.get("reasoning_content"))  # model's chain-of-thought
```

> **Note:** `reasoning_content` is also available during streaming — each `AIMessageChunk` carries it in `additional_kwargs`.

### API v2 (`/v2/chat/completions`)

The primary v2 contract is opt-in. Legacy requests remain the default.

Enable it on the model for plain sync, async, and streaming calls:

```python
import asyncio

from langchain_gigachat import GigaChat


llm = GigaChat(model="GigaChat-3-Ultra", use_api_v2=True)

response = llm.invoke("What is the capital of Russia?")
print(response.content)


async def main() -> None:
    response = await llm.ainvoke("Name three cities on the Volga.")
    print(response.content)


asyncio.run(main())

for chunk in llm.stream("Write one sentence about Lake Baikal."):
    print(chunk.text, end="", flush=True)
```

Use `bind()` when only one runnable should use v2:

```python
llm = GigaChat(model="GigaChat-3-Ultra")
primary_llm = llm.bind(use_api_v2=True)
response = primary_llm.invoke("Hello!")
```

#### Client tools and `ToolMessage` continuation

Client functions use standard LangChain tools. Pass the returned LangChain
tool-call ID back unchanged through `ToolMessage`:

```python
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from langchain_gigachat import GigaChat


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"{city}: sunny, 22C"


llm = GigaChat(use_api_v2=True)
with_tools = llm.bind_tools([get_weather], tool_choice="auto")

question = HumanMessage("What is the weather in Moscow?")
assistant = with_tools.invoke([question])
call = assistant.tool_calls[0]
tool_result = ToolMessage(
    content=get_weather.invoke(call["args"]),
    tool_call_id=call["id"],
    name=call["name"],
)
answer = with_tools.invoke([question, assistant, tool_result])
print(answer.content)
```

Primary continuation serializes the tool result as provider `role="tool"` with
`function_result` and `tools_state_id`. It does not use the legacy
`role="function"` transport. The returned LangChain tool-call ID is the
provider `tools_state_id`; pass it to `ToolMessage.tool_call_id` unchanged.
Stateful tool history is route-specific and is not translated between the
legacy and primary contracts.

#### Provider built-in tools

Provider built-ins use the public `bind_tools()` API:

```python
from langchain_gigachat import GigaChat

llm = GigaChat(use_api_v2=True)
with_search = llm.bind_tools(
    [{"type": "web_search"}],
    tool_choice="web_search",
)
response = with_search.invoke("Find the latest GigaChat SDK release.")
print(response.content)
```

Built-ins require v2. Binding one and then overriding the runnable with
`use_api_v2=False` raises an actionable `ValueError`.

#### Native structured JSON output

```python
from pydantic import BaseModel

from langchain_gigachat import GigaChat


class City(BaseModel):
    name: str
    population: int


llm = GigaChat(use_api_v2=True)
structured = llm.with_structured_output(City, method="json_schema")
city = structured.invoke("Return information about Kazan.")
```

Primary v2 can also request a native JSON object without a schema and without
`strict`:

```python
json_object_llm = llm.with_structured_output(None, method="json_mode")
payload = json_object_llm.invoke("Return a JSON object with a short answer.")
assert isinstance(payload, dict)
```

The equivalent low-level response format contains only the provider type:

```python
json_object_llm = llm.bind(response_format={"type": "json_schema"})
```

Low-level `bind(response_format=...)` returns an ordinary `AIMessage`.
Use `with_structured_output()` when the runnable should parse and validate the
response, or rely on a LangChain Agent provider strategy.

Primary v2 also preserves other explicit SDK response formats when binding
directly:

```python
ticket_id_llm = llm.bind(
    response_format={
        "type": "regex",
        "regex": r"[A-Z]{2}-[0-9]{4}",
    }
)
```

Supported explicit types are `text`, `json_schema`, and `regex`. A mapping
without an explicit response-format discriminator is treated as a raw JSON
Schema. On the primary route, `{"type": "json_schema"}` is the schema-less
form and is serialized without synthetic `schema` or `strict` fields. Supplying
`strict` without a schema is rejected, including `strict=False`. Unknown
explicit formats are rejected before the provider call. GigaChat has no
confirmed strict tool-schema field, so `bind_tools(..., strict=True)` without
`response_format` raises.

#### Assistant and thread state

Stateful requests accept either a top-level assistant ID or a storage thread:

```python
llm = GigaChat(use_api_v2=True)

assistant_reply = llm.invoke(
    "Continue the assistant conversation.",
    assistant_id="assistant-id",
)
thread_reply = llm.invoke(
    "Continue this thread.",
    storage={"thread_id": "thread-id"},
)
```

For assistant/thread requests, the configured default model is omitted so the
provider can resolve the model from stored state. Pass `model=...` on the
individual invocation only when an explicit override is required.

#### File ID input

An existing provider file ID can be supplied without re-uploading the file:

```python
from langchain_core.messages import HumanMessage

from langchain_gigachat import GigaChat

llm = GigaChat(use_api_v2=True)
message = HumanMessage(
    content=[
        {"type": "text", "text": "Summarize this document."},
        {
            "type": "file",
            "file_id": "provider-file-id",
            "mime_type": "application/pdf",
        },
    ]
)
response = llm.invoke([message])
```

#### Current release status and limitations

`langchain-gigachat==0.5.2a1` requires `langchain-core>=1.2.22,<2` and stable
`gigachat>=0.2.3,<0.3`. CI covers minimum/latest LangChain Core, a focused
LangChain Agent contract, and clean wheel/sdist installation. The
credential-backed live-provider matrix was not run.

Primary v2 currently rejects parallel client tool calls in one assistant
message. `tool_choice="any"` is rejected by default on both routes because
mapping forced-tool semantics to `"auto"` weakens the request. Compatibility
callers may explicitly set `allow_any_tool_choice_fallback=True`; this maps
`"any"` to `"auto"` with a `UserWarning`.

## Tool Calling

### Legacy tool transport

With the default `use_api_v2=False`, use the standard LangChain `@tool`
decorator for client functions. Pass GigaChat-specific metadata via `extras`:

```python
from langchain_gigachat import GigaChat
from langchain_core.tools import tool


@tool(
    extras={
        "few_shot_examples": [{"request": "weather in Tokyo", "params": {"city": "Tokyo"}}]
    }
)
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"{city}: sunny, 22C"


llm = GigaChat()
llm_with_tools = llm.bind_tools([get_weather], tool_choice="auto")

msg = llm_with_tools.invoke("What's the weather in Tokyo?")
print(msg.tool_calls)
```

To configure function/tool ranking, pass `function_ranker` to `GigaChat`. For
example, disable ranking when binding tools:

```python
llm = GigaChat(function_ranker={"enabled": False})
llm_with_tools = llm.bind_tools([get_weather], tool_choice="auto")
```

> **Note:** `tool_choice="any"` is not supported by GigaChat. Use `"auto"`,
> `"none"`, or a specific tool name. If compatibility with upstream code is
> required, `allow_any_tool_choice_fallback=True` explicitly converts `"any"`
> to `"auto"` and emits a `UserWarning` because forced-tool semantics are not
> preserved.

> **Note:** GigaChat API does not support parallel tool calls in a single assistant message. If `AIMessage` contains more than one `tool_calls` entry, a `ValueError` is raised.

#### Legacy `bind_functions()`

For legacy LangChain function-calling flows, `bind_functions()` is still available:

```python
from langchain_gigachat import GigaChat


def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"{city}: sunny, 22C"


llm = GigaChat()
llm_with_functions = llm.bind_functions(
    [get_weather],
    function_call="auto",
)
```

Use `bind_tools()` for new code. `bind_functions()` is kept as a compatibility layer over the provider's `function_call` transport and supports `None`, `"auto"`, `"none"`, or a specific function name.

The legacy provider transport is function-oriented. `ToolMessage` results are
therefore serialized back as provider `role="function"` messages when
continuing a legacy conversation. Provider built-in tools are not supported on
this route.

### Primary API v2 tool transport

With `use_api_v2=True`, `bind_tools()` accepts both client functions and
provider built-ins such as `{"type": "web_search"}`. Client tool results use
provider `role="tool"`, `function_result`, and `tools_state_id`; see the
[complete continuation example](#client-tools-and-toolmessage-continuation).
The primary and legacy transports are selected only after the invocation-level
`use_api_v2` override is resolved.

## Structured Output

Extract typed data from model responses:

```python
from pydantic import BaseModel, Field

from langchain_gigachat import GigaChat


class Answer(BaseModel):
    """An answer with confidence score."""

    text: str = Field(description="Final answer")
    confidence: float = Field(ge=0, le=1, description="Confidence 0..1")


llm = GigaChat()
chain = llm.with_structured_output(Answer)

parsed = chain.invoke("What is the capital of France? Rate your confidence.")
print(parsed)
```

By default, `with_structured_output()` uses GigaChat function calling for
backward-compatible schema extraction. Native API-level JSON Schema constraints
are available explicitly on a compatible model:

```python
primary = GigaChat(use_api_v2=True)
primary.with_structured_output(Answer, method="json_schema")
```

Use `schema=None` with `method="json_mode"` for schema-less native JSON. The
primary request contains `response_format={"type": "json_schema"}` with no
`schema` and no `strict`, and the runnable returns a parsed JSON object:

```python
json_object = primary.with_structured_output(None, method="json_mode")
result = json_object.invoke("Return a JSON object with keys answer and confidence.")
```

`method="json_schema"` requires a schema; its `strict` option applies only to
that schema-bearing form. Model support for native response formats remains
provider-dependent, so retain the default `method="function_calling"` fallback
for mixed model versions. The older schema-bearing
`with_structured_output(schema, method="json_mode")` form remains accepted but
deprecated; the new `schema=None` primary mode is not deprecated.

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

> **Note:** Supported `content_blocks` types: `image`, `audio`, `file`. The pattern is identical for each — only the `type` field differs.

> **Note:** Base64 data URLs in `image_url` / `audio_url` / `document_url` blocks can be auto-uploaded with `auto_upload_attachments=True`, but prefer explicit `upload_file()` in production.

### File Operations

Manage files via the Files API:

```python
from langchain_gigachat import GigaChat

llm = GigaChat()

# Upload
with open("document.pdf", "rb") as f:
    uploaded = llm.upload_file(("document.pdf", f.read()))
print(f"Uploaded: {uploaded.id_}")

# List
files = llm.list_files()
for f in files.data:
    print(f"{f.id_}: {f.filename}")

# Delete
llm.delete_file(uploaded.id_)
```

`get_file()` returns file metadata, while `get_file_content()` downloads file content:

```python
metadata = llm.get_file(uploaded.id_)          # UploadedFile
content = llm.get_file_content(uploaded.id_)   # Image with base64 payload
print(metadata.filename)
print(content.content[:20])
```

All file methods have async variants (`aget_file`, `aget_file_content`, `alist_files`, `adelete_file`, etc.).

## Configuration

All parameters can be passed to `GigaChat(...)` / `GigaChatEmbeddings(...)` directly or via environment variables with the `GIGACHAT_` prefix.

### Constructor Parameters

Most commonly used parameters (all are optional):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `None` | Model name (e.g. `"GigaChat-2-Max"`, `"GigaChat-2-Pro"`) |
| `temperature` | `float` | `None` | Sampling temperature |
| `max_tokens` | `int` | `None` | Maximum number of tokens to generate |
| `top_p` | `float` | `None` | Nucleus sampling threshold (0.0–1.0) |
| `repetition_penalty` | `float` | `None` | Penalty applied to repeated tokens |
| `reasoning_effort` | `str` | `None` | Reasoning effort for reasoning models |
| `function_ranker` | `dict` | `None` | Function/tool ranking settings, e.g. `{"enabled": False}` to disable ranking |
| `credentials` | `str` | `None` | OAuth authorization key |
| `access_token` | `str` | `None` | Pre-obtained JWT token (bypasses OAuth) |
| `scope` | `str` | `None` | API scope (`GIGACHAT_API_PERS` / `_B2B` / `_CORP`) |
| `base_url` | `str` | `None` | Custom API endpoint |
| `verify_ssl_certs` | `bool` | `None` | TLS certificate verification |
| `ca_bundle_file` | `str` | `None` | Path to CA certificate bundle |
| `timeout` | `float` | `None` | Request timeout in seconds |
| `max_retries` | `int` | `None` | Retry attempts for transient errors (SDK default: `0`) |
| `retry_backoff_factor` | `float` | `None` | Exponential backoff multiplier (SDK default: `0.5`) |
| `profanity_check` | `bool` | `None` | Enable profanity filtering |
| `use_api_v2` | `bool` | `False` | Use the `/v2/chat/completions` contract |
| `streaming` | `bool` | `False` | Stream results by default |
| `auto_upload_attachments` | `bool` | `False` | Auto-upload base64 content from `image_url` / `audio_url` / `document_url` blocks |
| `allow_any_tool_choice_fallback` | `bool` | `False` | Explicitly map `tool_choice="any"` to `"auto"` with a warning |

For the full list of parameters (auth, SSL/mTLS, retry, flags, etc.), see the [GigaChat SDK README](https://github.com/ai-forever/gigachat#constructor-parameters) — the LangChain wrapper accepts the same constructor arguments.

### Environment Variables

All parameters can be configured via environment variables with the `GIGACHAT_` prefix (e.g. `GIGACHAT_CREDENTIALS`, `GIGACHAT_MODEL`, `GIGACHAT_BASE_URL`). See the [GigaChat SDK README](https://github.com/ai-forever/gigachat#environment-variables) for the full list.

> **Note:** Retries are handled by the underlying `gigachat` SDK. Don't combine them with LangChain `.with_retry()` — the attempts multiply:
>
> ```python
> llm = GigaChat(max_retries=3, retry_backoff_factor=0.5)  # delays: 0.5s, 1s, 2s
> ```

## Error Handling

SDK exceptions propagate unchanged through the LangChain wrapper (aligned with the `langchain-openai` approach):

```python
from gigachat.exceptions import AuthenticationError, RateLimitError, GigaChatException
from langchain_gigachat import GigaChat

llm = GigaChat()

try:
    llm.invoke("Hello!")
except AuthenticationError as e:
    print(f"Authentication failed: {e}")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after}s")
except GigaChatException as e:
    print(f"GigaChat error: {e}")
```

For the full exception hierarchy and HTTP status code mapping, see the [GigaChat SDK — Error Handling](https://github.com/ai-forever/gigachat#error-handling).

## Tracing Metadata

When the provider returns tracing identifiers or headers, the wrapper preserves
them in both non-streaming and streaming flows:

- `AIMessage.id` / `AIMessageChunk.id` prefers `x-request-id`; primary v2
  responses fall back to the provider `message_id`
- non-streaming responses keep full headers in `ChatResult.llm_output["x_headers"]`
- streaming responses expose full headers on the first chunk via
  `generation_info["x_headers"]`

This makes it possible to correlate LangChain runs with provider-side logs or
support requests without parsing SDK responses directly.

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

Copyright © 2026 [GigaChain](https://github.com/ai-forever/gigachain)
