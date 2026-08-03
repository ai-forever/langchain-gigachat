# Migration Guide: langchain-gigachat 0.3.x → 0.5.x

This guide covers the breaking changes introduced in `langchain-gigachat`
0.5.0 and the opt-in primary API v2 preview added in `0.5.2a1`.

## Opting into API v2

The legacy contract remains the default. To use `/v2/chat/completions`, enable
it explicitly with `GigaChat(use_api_v2=True)`, or use
`llm.bind(use_api_v2=True)` for one runnable:

```python
from langchain_gigachat import GigaChat

legacy = GigaChat()
primary = legacy.bind(use_api_v2=True)
```

The invocation-level value wins over the constructor default, so
`GigaChat(use_api_v2=True).bind(use_api_v2=False)` deliberately routes back to
legacy. The routing flag is local control state and is never sent to either
provider payload.

### Parameter mapping

| Public input | Legacy contract | Primary API v2 contract |
|--------------|-----------------|-------------------------|
| `messages` | SDK `Chat.messages` | SDK `ChatCompletionRequest.messages` |
| `temperature`, `top_p`, `max_tokens`, `repetition_penalty` | Top-level chat fields | `model_options` |
| `reasoning_effort` | Legacy reasoning field | `model_options.reasoning.effort` |
| `profanity_check` | Legacy field | Inverted to `disable_filter` unless explicitly overridden |
| `function_ranker` | Legacy function ranker | `ranker_options` |
| `response_format` | Legacy response-format field | `model_options.response_format` |
| `functions` / client `bind_tools()` | `functions` plus `function_call` | Functions tool plus `tool_config` |
| Provider built-in `bind_tools()` | Rejected with an instruction to enable v2 | Provider `tools` plus `tool_config` |
| `assistant_id`, `tool_config`, `tools_state_id`, `user_info` | Rejected as primary-only | Forwarded through SDK request models |

Unknown primary request fields accepted by the installed SDK are preserved,
while LangChain-only control keys are consumed locally.

### Tool mapping

Legacy remains function-oriented:

- client tools are serialized through `functions` / `function_call`;
- a result `ToolMessage` becomes provider `role="function"`;
- `functions_state_id` remains legacy-specific continuation state;
- provider built-ins such as `web_search` are rejected.

Primary v2 uses the new tools transport:

- client functions and provider built-ins both use public `bind_tools()`;
- provider built-ins accept canonical mappings such as
  `{"type": "web_search"}`;
- a client result `ToolMessage` becomes provider `role="tool"` with
  `function_result` and `tools_state_id`;
- returned `AIMessage.tool_calls[0]["id"]` is the provider `tools_state_id`, and
  the following `ToolMessage.tool_call_id` must use that value unchanged.

Provider-specific tool state is not translated between the legacy and primary
contracts. Neutral text history can be used with either route; replaying a
stateful tool history through the other route raises an actionable error.

Parallel client tool calls in one assistant message remain unsupported.
`tool_choice="any"` is rejected by default on both routes because converting
forced-tool semantics to `"auto"` weakens the request. Prefer `"auto"`,
`"none"`, or a concrete tool. For compatibility only,
`allow_any_tool_choice_fallback=True` performs that conversion with a visible
`UserWarning`.

### Storage and stateful requests

Legacy storage continues to accept `gigachat.models.Storage` unchanged. Primary
storage accepts `None`, a boolean, `ChatStorage`, a compatible mapping, or a
losslessly convertible legacy `Storage`.

Primary assistant state is top-level:

```python
primary.invoke("Continue", assistant_id="assistant-id")
```

Thread state is nested in storage:

```python
primary.invoke("Continue", storage={"thread_id": "thread-id"})
```

For requests with `assistant_id` or `storage.thread_id`, an implicit model from
the `GigaChat` instance is omitted so the provider can resolve the stored
assistant/thread model. An explicit invocation-level `model=...` is preserved.
Conflicting top-level and storage assistant IDs fail before network I/O.

### Streaming and metadata

Primary sync and async streaming consume the SDK named-event resources
`chat.stream` and `achat.stream`. Supported content has the same LangChain
meaning in stream and non-stream results:

- text, reasoning, files, citations, client calls, and server-tool blocks use
  standard LangChain content blocks;
- usage, finish reason, message/thread IDs, tool state, logprobs, and request
  headers are promoted to standard message/generation metadata;
- unknown provider extensions remain available under
  `response_metadata["provider_fields"]`; raw unknown stream events are kept
  in arrival order under `response_metadata["raw_events"]`.

Public `stream()` / `astream()` buffer the authoritative
`response.message.done` terminal. Compatible metadata-only continuations are
merged into that buffered terminal, so usage, headers, provider extensions,
thread state, and raw events survive in one final chunk. Content or tool calls
after the terminal still fail closed. No public stream emits a chunk after
`chunk_position="last"`.

### Structured output

`with_structured_output()` still defaults to `method="function_calling"` for
backward compatibility. Schema-bearing `method="json_schema"` is normalized
according to the selected route:

- legacy uses the legacy SDK response-format representation;
- primary places `ChatResponseFormat` under `model_options.response_format`.

Caller-owned schemas are copied before normalization. Model support for native
JSON Schema is provider-dependent, so keep the function-calling fallback when
deploying across mixed model versions.

Primary v2 additionally supports schema-less native JSON:

```python
json_llm = GigaChat(use_api_v2=True).with_structured_output(
    None,
    method="json_mode",
)
result = json_llm.invoke("Return a JSON object.")
```

This sends exactly `{"type": "json_schema"}` under
`model_options.response_format`: no placeholder schema and no implicit
`strict=False`. The result parser accepts JSON objects and rejects arrays,
scalars, invalid JSON, tool-call responses, and completions whose authoritative
finish reason is not `stop`. Invocation-level routing is still authoritative;
an override to the legacy route does not forward the primary-only native
response-format marker.

Low-level `bind(response_format=...)` only sends the provider constraint and
returns an ordinary `AIMessage`. Parsing, validation, and `include_raw` behavior
belong to `with_structured_output()` or LangChain Agent provider strategies.

Primary response-format normalization preserves the SDK's explicit `text`,
`json_schema`, and `regex` formats. OpenAI-style nested `json_schema` values are
unwrapped to the SDK shape, while plain JSON Schema mappings remain schema
payloads. When both native `model_options.response_format` and the top-level
convenience argument are present, the native value wins on the provider
payload. Unknown explicit response-format types fail before network I/O.

`strict` applies only to schema-bearing JSON Schema response formats. It is
rejected for schema-less JSON even when explicitly set to `False`. Because
GigaChat does not expose a confirmed strict field for tool schemas,
`bind_tools(..., strict=True)` without `response_format` raises instead of
silently accepting a no-op argument. The older schema-bearing
`method="json_mode"` compatibility path remains deprecated; the new
`schema=None` form is not.

### Dependency and release status

`langchain-gigachat==0.5.2a1` is a prerelease and requires
`langchain-core>=1.2.22,<2` and `gigachat>=0.2.3,<0.3`. Core 1.0/1.1 do not
provide all public contracts used by this integration. CI covers the minimum
and latest supported Core versions, a focused Agent contract, and clean
wheel/sdist installs. The credential-backed live provider matrix was not run.

## Requirements

| Dependency | Before (0.3.x) | Current preview (0.5.2a1) |
|------------|-----------------|----------------------------|
| Python | >= 3.9 | **>= 3.10, < 4** |
| `langchain-core` | >= 0.3, < 1 | **>= 1.2.22, < 2** |
| `gigachat` (SDK) | >= 0.1.41 | **>= 0.2.3, < 0.3** |

> LangChain Core 1.x dropped Python 3.9 support. GigaChat SDK 0.2.0 migrated to Pydantic V2.

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

Previously, `tool_choice="any"` was silently converted to `"auto"`. It now
raises `ValueError` by default.

```python
# Before — silently degraded to "auto"
llm.bind_tools(tools, tool_choice="any")

# After — raises ValueError. Two options:

# Option 1: use "auto" or a specific tool name
llm.bind_tools(tools, tool_choice="auto")
llm.bind_tools(tools, tool_choice="my_tool_name")

# Option 2: explicit compatibility fallback (with warning)
llm = GigaChat(allow_any_tool_choice_fallback=True, ...)
llm.bind_tools(tools, tool_choice="any")  # converts to "auto" with UserWarning
```

**Why:** GigaChat API does not support `tool_choice="any"` (forced tool
calling). Converting it to `"auto"` can allow plain text and weaken the
caller's forced-tool requirement, so the compatibility path is explicit and
warns at the call site.

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

If you used `@giga_tool(...)` to pass GigaChat-specific metadata such as
`few_shot_examples` or `return_schema`, migrate to the standard
`langchain_core.tools.tool` decorator and pass the same metadata via `extras`:

```python
# Before
from langchain_gigachat.tools import giga_tool


@giga_tool(
    few_shot_examples=[{"request": "weather in Tokyo", "params": {"city": "Tokyo"}}],
    return_schema=WeatherResult,
)
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return "sunny"
```

```python
# After
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

For new code, always prefer the standard `@tool` path:

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
print(langchain_gigachat.__version__)  # "0.5.2a1"
```
