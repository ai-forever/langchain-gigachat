# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.5.2a1] — 2026-07-30

Primary API v2 preview.

### Added

- Opt-in `GigaChat(use_api_v2=True)` support for `/v2/chat/completions`,
  including sync/async calls, streaming, tools, attachments, and native JSON
  Schema structured output.
- Schema-less native JSON through
  `with_structured_output(None, method="json_mode")`; the wire payload contains
  neither a placeholder schema nor `strict`.
- Per-runnable route overrides through `llm.bind(use_api_v2=True)`, while
  legacy remains the default.
- Public `bind_tools()` support for primary provider built-ins such as
  `web_search`, alongside standard LangChain client tools.
- Primary assistant/thread state, file-ID content, server-tool content blocks,
  usage metadata, invalid tool calls, and forward-compatible provider fields.

### Changed

- Primary stream and non-stream results now use equivalent LangChain content
  blocks and preserve late finish, usage, request, and tool-state metadata.
- Primary client tool-call IDs use the provider `tools_state_id` required for
  the following `ToolMessage`; provider-specific tool state is not translated
  between the legacy and primary API contracts.
- Primary streaming accepts the SDK's `PrimaryChatCompletionChunk` models; the
  SDK remains responsible for SSE parsing and alias normalization.
- Explicit primary response formats preserve SDK `text`, `json_schema`, and
  `regex` semantics. Unknown response-format types fail before network I/O.
- `bind_tools(..., strict=...)` now requires a JSON Schema `response_format`
  instead of silently discarding strict tool-schema intent.
- Stateful assistant/thread requests no longer receive an implicit default
  model; an explicitly supplied invocation model is still forwarded.
- Primary `ToolMessage` continuation uses provider `role="tool"`,
  `function_result`, and `tools_state_id`. The legacy function transport is
  unchanged.
- Direct `bind(response_format=...)` calls return ordinary `AIMessage` values;
  parsing and `include_raw` behavior belong to `with_structured_output()` and
  LangChain Agent provider strategies.

### Dependencies

- Package version is `0.5.2a1`.
- Requires `langchain-core>=1.2.22,<2`. Core 1.0 and 1.1 do not provide every
  public contract used by this integration.
- Requires stable `gigachat>=0.2.3,<0.3`, verified with sync/async
  `chat.create` and `chat.stream` resource methods.

### CI

- Added minimum/latest LangChain Core checks, a focused LangChain Agent smoke
  check, and clean wheel/sdist installation from the exact PR head.

### Known limitations

- Parallel client tool calls in one primary assistant message are unsupported.
- `tool_choice="any"` is rejected by default on both routes. The explicit
  `allow_any_tool_choice_fallback=True` compatibility path maps it to `"auto"`
  with a `UserWarning` that forced-tool semantics are weakened.
- Live API coverage was not run without provider credentials.

## [0.5.1] — 2026-05-04

### Added

- **Native structured output**: `with_structured_output(method="json_schema")` binds `JsonSchemaResponseFormat` to the chat request. Requires a model with `response_format` support (currently in beta). The default remains `method="function_calling"` for backward compatibility.
- **Function ranker settings**: `GigaChat(function_ranker={"enabled": False})` forwards function/tool ranking settings to the API payload.

### Deprecated

- `with_structured_output(method="json_mode")` now emits a `DeprecationWarning`. The mode still works; prefer `method="json_schema"` for native API-level constraints.

### Dependencies

- Bumped minimum `gigachat` SDK to `>=0.2.1,<0.3` to enable the new `response_format` and `FunctionRanker` payload fields. No source-level breaking change — existing call sites continue to work unchanged.

## [0.5.0] — 2026-03-11

Stable release: LangChain Core 1.x, Pydantic V2, multimodal support, and extensive cleanup.

### Breaking Changes

- **Python ≥ 3.10** required (LangChain Core 1.x minimum).
- **`langchain-core >= 1, < 2`** — upgraded from `>=0.3,<1`.
- **`gigachat >= 0.2.0, < 0.3`** — upgraded from `^0.1.41`.
- **Removed `verbose` parameter** — use Python `logging` at `DEBUG` level instead.
- **Removed `profanity` field** — use `profanity_check` instead.
- **Removed `predict()` / `apredict()`** (dropped by LangChain 1.x) — use `invoke()` / `ainvoke()`.
- **Removed `with_structured_output(method="format_instructions")`** — use `method="function_calling"`.
- **Removed `auto_upload_images`** — use `auto_upload_attachments` (covers images, audio, documents).
- **Removed `GigaChatEmbeddings.one_by_one_mode` and `_debug_delay`** — API handles batching natively.
- **Removed `output_parsers.gigachat_functions` module** — use `PydanticToolsParser` / `JsonOutputKeyToolsParser` from `langchain_core`.
- **Removed wrapper-side `stop` support** — `stop=...` is no longer handled by `langchain-gigachat`.
- **`get_file()` now returns metadata** (`UploadedFile`) instead of content — use `get_file_content()` for binary data.
- **`tool_choice="any"` now raises `ValueError`** — set `allow_any_tool_choice_fallback=True` for auto-fallback.
- **Multiple `tool_calls` in one `AIMessage` now raises `ValueError`** — GigaChat API does not support parallel function calls.
- **`giga_tool` no longer accepts `return_schema` / `few_shot_examples` kwargs** — use standard `@tool(extras={...})`; `giga_tool` is now only an alias of `langchain_core.tools.tool`.

### Added

- **Multimodal file upload**: support for `audio_url`, `document_url` content blocks alongside `image_url`. Standard LangChain blocks (`image`, `audio`, `file`) with `file_id` are also supported.
- **Reasoning model support**: `reasoning_effort` parameter and `reasoning_content` in response `additional_kwargs` for models like GigaChat-2-Reasoning.
- **File API methods**: `list_files()`, `delete_file()`, `get_file_content()` (+ async variants).
- **Connection settings**: `max_retries`, `max_connections`, `retry_backoff_factor`, `retry_on_status_codes` exposed as constructor parameters.
- **`allow_any_tool_choice_fallback`** parameter for explicit opt-in to `tool_choice="any"` → `"auto"` conversion.
- **Module exports**: `tools/__init__.py` and `utils/__init__.py` now export public symbols (`GigaTool`, `giga_tool`, `convert_to_gigachat_function`, etc.).
- **CI**: expanded test matrix (Python 3.10–3.13, experimental 3.14), GitHub issue/PR templates, `CONTRIBUTING.md`.

### Changed

- **Pydantic V2 migration**: all models use native Pydantic V2 APIs (`model_validate`, `model_dump`, `model_config`).
- **Shared client mixin**: `_GigaChatClientMixin` eliminates duplication between `GigaChat` and `GigaChatEmbeddings`, fixes credential leak in embeddings serialization (`lc_secrets`).
- **Per-instance upload cache** (`_cached_uploads`) with FIFO eviction (was class-level unbounded dict).
- **Simplified embeddings**: removed client-side batching logic, single SDK call for all texts.
- **Build tooling**: migrated from Poetry to uv + hatchling (PEP 621).
- **`bind_functions()`** now correctly supports `"auto"` / `"none"` and multiple functions.
- **`FunctionMessage.name` and `ToolMessage.name`** are now correctly forwarded to the API.
- **Streaming refactored**: shared `_build_stream_chunk()` helper centralizes chunk building and `x_headers` propagation.

### Removed

- `trim_content_to_stop_sequence()` — wrapper-side stop sequence handling removed.
- `_check_finish_reason()` — response validation belongs in SDK.
- `_convert_function_to_dict()`, `_get_type_hints()` — dead code.
- `tools/load_prompt.py` — legacy module not part of public API.
- `validate_environment()` validators — redundant with Pydantic V2 and direct imports.
- Poetry lock file and custom CI actions.

### Migration

For detailed before/after code examples, see [MIGRATION.md](MIGRATION.md).
