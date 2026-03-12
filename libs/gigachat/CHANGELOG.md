# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.4.0a1] — 2026-03-11

Alpha pre-release: LangChain Core 1.x, Pydantic V2, multimodal support, and extensive cleanup.

This is a pre-release and will not be installed by default via a plain `pip install -U langchain-gigachat`.

### Breaking Changes

- **Python ≥ 3.10** required (LangChain Core 1.x minimum).
- **`langchain-core >= 1, < 2`** — upgraded from `>=0.3,<1`.
- **`gigachat >= 0.2.0, < 0.3`** — upgraded from `^0.1.41`.
- **Removed `verbose` parameter** — use Python `logging` at `DEBUG` level instead.
- **Removed `profanity` field** — use `profanity_check` instead.
- **Removed `predict()` / `apredict()`** (dropped by LangChain 1.x) — use `invoke()` / `ainvoke()`.
- **Removed `with_structured_output(method="format_instructions")`** — use `method="function_calling"` or `method="json_mode"`.
- **Removed `auto_upload_images`** — use `auto_upload_attachments` (covers images, audio, documents).
- **Removed `GigaChatEmbeddings.one_by_one_mode` and `_debug_delay`** — API handles batching natively.
- **Removed `output_parsers.gigachat_functions` module** — use `PydanticToolsParser` / `JsonOutputKeyToolsParser` from `langchain_core`.
- **Removed wrapper-side `stop` support** — `stop=...` is no longer handled by `langchain-gigachat`.
- **`get_file()` now returns metadata** (`UploadedFile`) instead of content — use `get_file_content()` for binary data.
- **`tool_choice="any"` now raises `ValueError`** — set `allow_any_tool_choice_fallback=True` for auto-fallback.
- **Multiple `tool_calls` in one `AIMessage` now raises `ValueError`** — GigaChat API does not support parallel function calls.

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
