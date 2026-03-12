# Migration Guide: langchain-gigachat 0.3.x → 0.4.0a1

This guide covers all breaking changes in `langchain-gigachat` 0.4.0a1 and explains how to update your code.

## Requirements

| Dependency | Before (0.3.x) | After (0.4.0a1) |
|------------|-----------------|---------------|
| Python | >= 3.9 | **>= 3.10** |
| `langchain-core` | >= 0.3, < 1 | **>= 1, < 2** |
| `gigachat` (SDK) | >= 0.1.41 | **>= 0.2.0, < 0.3** |

> LangChain Core 1.x dropped Python 3.9 support. GigaChat SDK 0.2.0 migrated to Pydantic V2.
>
> `0.4.0a1` is an alpha pre-release. It will not be installed by default by a plain `pip install -U`; users must opt in to pre-releases explicitly.

---

## Removed APIs

### `verbose` parameter

The `verbose` flag on `GigaChat` was removed. It logged raw requests and responses, duplicating standard Python logging.

```python
# Before
llm = GigaChat(verbose=True)

# After — use Python logging
import logging
logging.getLogger("langchain_gigachat").setLevel(logging.DEBUG)
```

**Why:** The upstream `gigachat` SDK removed the `verbose` parameter. Standard `logging` module is the recommended approach across the LangChain ecosystem.

---

### `predict()` / `apredict()`

These methods were removed in LangChain Core 1.x.

```python
# Before
text = llm.predict("Привет")
text = await llm.apredict("Привет")

# After
text = llm.invoke("Привет").content
text = (await llm.ainvoke("Привет")).content
```

**Why:** LangChain 1.x removed deprecated `predict`/`apredict` methods in favor of `invoke`/`ainvoke`.

---

### `profanity` field

The deprecated `profanity: bool` field on `_BaseGigaChat` was removed.

```python
# Before
llm = GigaChat(profanity=False)

# After
llm = GigaChat(profanity_check=False)
```

**Why:** `profanity` was a deprecated alias for `profanity_check`. The migration shim has been removed.

---

### `one_by_one_mode` and `_debug_delay` (Embeddings)

Both fields were removed from `GigaChatEmbeddings`.

```python
# Before
emb = GigaChatEmbeddings(one_by_one_mode=True, _debug_delay=0.5)

# After — no replacement needed
emb = GigaChatEmbeddings(...)
```

**Why:** The GigaChat Embeddings API handles batching natively on the server side. Client-side batching logic (`MAX_BATCH_SIZE_CHARS`, `MAX_BATCH_SIZE_PARTS`) was unnecessary and has been removed. The SDK passes the full list of texts to the API in a single call.

---

### `auto_upload_images` flag

Replaced by a broader `auto_upload_attachments` flag that covers images, audio, and documents.

```python
# Before
llm = GigaChat(auto_upload_images=True)

# After
llm = GigaChat(auto_upload_attachments=True)
```

**Why:** Multimodal upload support was extended beyond images. A single flag now controls auto-upload for all attachment types.

---

### `with_structured_output(method="format_instructions")`

The `format_instructions` method for structured output has been removed.

```python
# Before
chain = llm.with_structured_output(MyModel, method="format_instructions")

# After — use function_calling (preferred) or json_mode
chain = llm.with_structured_output(MyModel, method="function_calling")
chain = llm.with_structured_output(MyModel, method="json_mode")
```

**Why:** The `format_instructions` method was a legacy prompt-injection approach with weak schema guarantees. The `function_calling` method provides strict schema extraction via the API. See [issue #40](https://github.com/ai-forever/langchain-gigachat/issues/40).

---

### `output_parsers` module

The entire `langchain_gigachat.output_parsers` module has been deleted, including:
- `OutputFunctionsParser`
- `PydanticOutputFunctionsParser`
- `PydanticAttrOutputFunctionsParser`

```python
# Before
from langchain_gigachat.output_parsers.gigachat_functions import (
    PydanticOutputFunctionsParser,
)

# After — use LangChain Core parsers
from langchain_core.output_parsers import PydanticToolsParser, JsonOutputKeyToolsParser
```

**Why:** These parsers were legacy wrappers around `function_call` output. LangChain Core provides equivalent parsers that work with the modern `tool_calls` API.

---

### `load_prompt` module

The `langchain_gigachat.tools.load_prompt` module has been deleted.

**Why:** It was never part of the public API and was not exported from `__init__.py`.

---

## Changed Behaviour

### `stop` support removed

The wrapper no longer implements local stop-sequence handling.

```python
# Before
msg = llm.invoke("Hello STOP world", stop=["STOP"])

# After — remove the argument from call sites
msg = llm.invoke("Hello STOP world")
```

**Why:** The `stop` behavior was wrapper-specific and is no longer maintained.
If you previously relied on it, update call sites to stop passing `stop=...`.

---

### `tool_choice="any"` raises `ValueError`

Previously, `tool_choice="any"` was silently converted to `"auto"`. Now it raises `ValueError` by default.

```python
# Before — silently degraded to "auto"
llm.bind_tools(tools, tool_choice="any")

# After — raises ValueError. Two options:

# Option 1: use "auto" or a specific tool name
llm.bind_tools(tools, tool_choice="auto")
llm.bind_tools(tools, tool_choice="my_tool_name")

# Option 2: opt-in to automatic fallback (with warning)
llm = GigaChat(allow_any_tool_choice_fallback=True, ...)
llm.bind_tools(tools, tool_choice="any")  # converts to "auto" with UserWarning
```

**Why:** GigaChat API does not support `tool_choice="any"` (forced tool calling). Silent conversion to `"auto"` changed semantics unpredictably — the user expected a forced tool call, but the model could return plain text. An explicit error is safer.

---

### Multiple `tool_calls` in `AIMessage` raises `ValueError`

Previously, when an `AIMessage` had multiple `tool_calls`, only `tool_calls[0]` was sent to the API and the rest were silently dropped. Now a `ValueError` is raised.

```python
# If you encounter this error, restructure to use one tool call per turn.
# GigaChat API does not support parallel function calls.
```

**Why:** Silently dropping tool calls corrupted conversation history and led to unpredictable behavior in later turns.

---

### `get_file()` return type changed

`get_file()` / `aget_file()` now return **file metadata** (`gm.UploadedFile`) instead of **file content** (`gm.Image`).

```python
# Before — get_file returned content (base64)
image = llm.get_file(file_id)
data = image.content

# After — get_file returns metadata; use get_file_content for content
metadata = llm.get_file(file_id)          # -> gm.UploadedFile
image = llm.get_file_content(file_id)     # -> gm.Image (base64)
data = image.content
```

New file management methods:
```python
files = llm.list_files()                  # GET /files
llm.delete_file(file_id)                  # DELETE /files/{id}
```

**Why:** The old `get_file` was misleadingly named — it actually downloaded file content via SDK's `get_image`. The new API aligns method names with their actual behavior and the SDK surface.

---

## New Features (non-breaking)

These are additive and require no migration, but are worth knowing about.

### Reasoning models

```python
llm = GigaChat(model="GigaChat-2-Reasoning", reasoning_effort="medium")
msg = llm.invoke([HumanMessage(content="Реши задачу...")])
reasoning = msg.additional_kwargs.get("reasoning_content")
```

### Connection settings

New fields exposed on `GigaChat` and `GigaChatEmbeddings`:
- `max_retries` — maximum retries for transient errors (SDK default: 0)
- `max_connections` — maximum simultaneous connections
- `retry_backoff_factor` — backoff factor for retry delays (SDK default: 0.5)
- `retry_on_status_codes` — HTTP codes that trigger a retry (SDK default: `(429, 500, 502, 503, 504)`)

```python
llm = GigaChat(max_retries=3, retry_backoff_factor=1.0, ...)
```

### Multimodal attachments

Audio and document uploads alongside images:
```python
from langchain_core.messages import HumanMessage

msg = HumanMessage(content_blocks=[
    {"type": "text", "text": "Опиши вложения."},
    {"type": "image", "file_id": "img-id"},
    {"type": "audio", "file_id": "audio-id"},
    {"type": "file", "file_id": "doc-id"},
])
```

### Module exports

Public utilities are exported from the package:
```python
from langchain_gigachat.utils import convert_to_gigachat_function, convert_to_gigachat_tool
```

### Tool decorator

Use the standard `langchain_core.tools.tool` decorator and pass GigaChat-specific metadata via `extras`:

```python
from langchain_core.tools import tool


@tool(
    extras={
        "few_shot_examples": [{"request": "weather in Tokyo", "params": {"city": "Tokyo"}}],
        "return_schema": WeatherResult,
    }
)
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return "sunny"
```

### `__version__`

```python
import langchain_gigachat
print(langchain_gigachat.__version__)  # "0.4.0a1"
```
