# Refactoring Progress

## General
- [x] Initial setup
  - [x] Create `AGENTS.md` documentation
  - [x] Create `docs/REFACTORING.md` documentation

## Pydantic V2 Migration
- [x] Migrate `function_calling.py` from Pydantic V1 to V2
  - [x] Replace `from pydantic.v1 import BaseModel` with native import
  - [x] Replace `from pydantic.v1 import Field as Field_v1` with `from pydantic import Field`
  - [x] Replace `from pydantic.v1 import create_model as create_model_v1` with `from pydantic import create_model`
  - [x] Update `_convert_typed_dict_to_gigachat_function` to use V2 patterns
  - [x] Update `_convert_any_typed_dicts_to_pydantic` to use V2 patterns
- [x] Migrate `gigachat.py` deprecated methods
  - [x] Replace `Chat.parse_obj(payload_dict)` with `Chat.model_validate(payload_dict)`
  - [x] Replace `response.usage.dict()` with `response.usage.model_dump()`
  - [x] Replace `chunk_d.dict()` with `chunk_d.model_dump()` (2 occurrences in `_stream` and `_astream`)
- [x] Migrate `base_gigachat.py` Config class
  - [x] Replace `class Config: arbitrary_types_allowed = True` with `model_config = ConfigDict(arbitrary_types_allowed=True)`
  - [x] Add `from pydantic import ConfigDict` import
- [x] Migrate `embeddings/gigachat.py` Config class
  - [x] Replace `class Config: arbitrary_types_allowed = True` with `model_config = ConfigDict(arbitrary_types_allowed=True)`
  - [x] Add `from pydantic import ConfigDict` import
- [x] Update dependency version
  - [x] Update `gigachat` dependency (currently using local editable path for development)
  - [x] Run `poetry lock` to update lockfile
- [x] Verification
  - [x] Run `ruff check` to verify no linting errors
  - [x] Run `mypy` to verify type checking passes
  - [x] Run `pytest` to verify no test regressions
  - [x] Fix test expectations for TypedDict default value behavior in Pydantic V2
- [x] Cleanup
  - [x] Remove `BaseModelV2Maybe`/`FieldV2Maybe` aliases in `tests/unit_tests/utils/test_function_calling.py`
  - [x] Remove V1/V2 compatibility code (`hasattr` checks for `model_validate`) in `output_parsers/gigachat_functions.py`
  - [x] Refactor `_model_to_schema()` in `function_calling.py` to use only `model_json_schema()` (remove `.schema()` V1 fallback)

## Remove `verbose` Parameter
- [x] Remove `verbose: bool = False` field from `_BaseGigaChat` in `base_gigachat.py`
- [x] Remove `verbose=self.verbose` from `_client` property in `base_gigachat.py`
- [x] Remove `if self.verbose:` request logging block in `gigachat.py` `_build_payload()`
- [x] Remove `if self.verbose:` response logging in `gigachat.py` `_create_chat_result()`
- [x] Verification
  - [x] Run `ruff check` to verify no linting errors
  - [x] Run `mypy` to verify type checking passes
  - [x] Run `pytest` to verify no test regressions
- [x] Documentation
  - [x] Move content from `docs/V1_MIGRATION.md` to `docs/REFACTORING.md`
  - [x] Delete `docs/V1_MIGRATION.md`

## Poetry to uv Migration
- [x] Convert `pyproject.toml` from Poetry to PEP 621 format
  - [x] Replace `[tool.poetry]` with `[project]` section
  - [x] Convert dependencies to PEP 621 format with version ranges
  - [x] Convert git dependency to uv-compatible format
  - [x] Consolidate Poetry groups (dev/lint/typing/test) into `[dependency-groups]`
  - [x] Change build backend from `poetry-core` to `hatchling`
  - [x] Add hatch build configuration and metadata settings
  - [x] Fix mypy config: change string `"True"` to boolean `true`
- [x] Remove legacy `load_prompt.py` module
  - [x] Delete `langchain_gigachat/tools/load_prompt.py` (unused, not exported)
  - [x] Delete `tests/unit_tests/test_utils.py` (tested only load_prompt)
  - [x] Remove `types-requests` from dependencies
  - [x] Remove `requests_mock` from test dependencies
- [x] Update Makefile
  - [x] Replace all `poetry run` with `uv run`
- [x] Rewrite CI/CD workflows
  - [x] Rewrite `.github/workflows/_lint.yml` to use `astral-sh/setup-uv@v5`
  - [x] Rewrite `.github/workflows/_test.yml` to use `astral-sh/setup-uv@v5`
  - [x] Update `.github/workflows/check_diffs.yml`: remove `POETRY_VERSION` env
  - [x] Update test matrix: Python 3.10-3.13 + 3.14 (experimental)
  - [x] Delete `.github/actions/poetry_setup/` custom action
- [x] Lock file transition
  - [x] Delete `poetry.lock`
  - [x] Generate `uv.lock` with `uv lock`
- [x] Update documentation
  - [x] Update `AGENTS.md`: change setup/run commands to uv
- [x] Verification
  - [x] Run `uv sync` to install dependencies
  - [x] Run `uv run ruff check` — passed
  - [x] Run `uv run mypy` — passed (13 source files)
  - [x] Run `uv run pytest` — 73 passed, 2 xpassed
- [x] Post-migration cleanup
  - [x] Delete empty `.github/actions/` directory
  - [x] Delete unused `.github/scripts/get_min_versions.py` (broken, references Poetry)

## Code Cleanup
- [x] Rewrite `GigaChat` class docstring
  - [x] Replace legacy `langchain_community` example-heavy docstring with upstream `gigachat` SDK format
  - [x] Use `Args:` section to document all parameters
- [x] Remove dead code
  - [x] Delete commented `FunctionInProgressMessageChunk` block in `gigachat.py`
  - [x] Delete unused `_convert_function_to_dict()` function in `gigachat.py`
  - [x] Delete unused `_get_type_hints()` function in `function_calling.py`
  - [x] Remove unused `functools` import in `function_calling.py`
- [x] Verification
  - [x] Run `uv run ruff check` — passed
  - [x] Run `uv run ruff format` — passed
  - [x] Run `uv run pytest` — 73 passed, 2 xpassed

## Phase 2: Refactoring Plan

### 2.1. Mixin for Chat and Embeddings
- [x] Create `_GigaChatClientMixin(Serializable)` in `langchain_gigachat/_client.py`
  - [x] Extract 14 shared connection/auth fields from `_BaseGigaChat` and `GigaChatEmbeddings`
  - [x] Add `lc_secrets` property (fixes credential leak in `GigaChatEmbeddings` serialization)
  - [x] Add `is_lc_serializable()` classmethod
  - [x] Add `_get_client_init_kwargs()` hook method for subclass extension
  - [x] Add `_client` cached property using the hook
- [x] Refactor `_BaseGigaChat` to inherit from `_GigaChatClientMixin`
  - [x] Remove duplicated connection/auth fields (inherited from mixin)
  - [x] Override `_get_client_init_kwargs()` to add `profanity_check` and `flags`
  - [x] Keep chat-specific fields (`temperature`, `streaming`, `max_tokens`, etc.)
- [x] Refactor `GigaChatEmbeddings` to inherit from `_GigaChatClientMixin, Embeddings`
  - [x] Remove duplicated connection/auth fields (inherited from mixin)
  - [x] Replace `BaseModel` base with `_GigaChatClientMixin` (which extends `Serializable`)
  - [x] Keep `timeout=600` override and embeddings-specific fields
- [x] Verification
  - [x] `uv run ruff check` — passed
  - [x] `uv run ruff format --check` — passed
  - [x] `uv run mypy` — passed (12 source files)
  - [x] `uv run pytest` — 82 passed, 79% coverage
  - [x] `_client.py` — 100% coverage

### 2.2. Base64 Image Handling
- [x] Make `_cached_images` per-instance via `PrivateAttr(default_factory=dict)` in `GigaChat` (fixes multi-tenant risk: cache no longer shared across instances)
- [x] Add eviction: cap cache at `DEFAULT_IMAGE_CACHE_MAX_SIZE` (1000), FIFO eviction when full via `_set_cached_image()`
- [x] Add unit tests
  - [x] `test_ai_upload_image_per_instance_cache` — two instances have independent caches
  - [x] `test_ai_upload_image_cache_eviction` — when at max size, oldest entry is evicted (FIFO)
- [x] Verification: `uv run ruff check`, `uv run pytest` (all image-upload tests pass)
- [x] TODO hardening follow-up (non-blocking) — **cancelled**: `auto_upload_attachments` is experimental and explicitly not for production usage (documented in docstring, field help, and runtime warning). Production-grade cache hardening (TTL, key normalization, clear API) is not justified for an experimental feature. Revisit only if the feature is promoted to production-ready.

### 2.3. Multimodal File Upload
- [x] Extend content block handling beyond image-only flow
  - [x] Support provider-native blocks: `image_url`, `audio_url`, `document_url`
  - [x] Support standard LangChain blocks: `image`, `audio`, `file` with `file_id`
- [x] Replace legacy image-only auto-upload flow
  - [x] Remove `auto_upload_images` in favor of `auto_upload_attachments`
  - [x] Reuse upload/cache flow for image/audio/document data URLs
- [x] Add MIME extension fallback mapping for supported GigaChat attachment types
- [x] Verification: `uv run ruff check`, `uv run pytest` (attachment parsing and auto-upload tests pass)

### 2.4. Format Instructions Mode
- [x] Remove `method="format_instructions"` from `GigaChat.with_structured_output()` public API
  - [x] Remove legacy prompt-injection branch in `chat_models/gigachat.py`
  - [x] Remove tests that verify `format_instructions` behavior

### 2.5. LangChain Legacy (LCL) Chains Review
- [x] Review and minimize legacy LangChain patterns
  - [x] Fix `GigaChat.bind_functions()` docstring vs behavior mismatch (`"auto"`/`"none"` support)
  - [x] Allow forcing by name among multiple functions (remove single-function restriction)
  - [x] Add unit tests for `bind_functions()` legacy behaviors

### 2.6 models.dev
- [x] WIP: https://github.com/anomalyco/models.dev/pull/927
### `with_structured_output` Override Typing Compatibility
- [x] Align `GigaChat.with_structured_output()` signature with `BaseChatModel` override contract
  - [x] Keep provider-specific `method` handling via `**kwargs` instead of explicit typed param
  - [x] Remove overload-only typing artifacts that caused mypy override incompatibility
  - [x] Update method docstring to describe `method` option and `include_raw` behavior

### 2.9. Embeddings Batch Settings
- [x] Remove `MAX_BATCH_SIZE_CHARS` and `MAX_BATCH_SIZE_PARTS` constants from `embeddings/gigachat.py`
- [x] Simplify `embed_documents()` to a single SDK call (remove batching loop)
- [x] Simplify `aembed_documents()` to a single SDK call (remove batching loop)
- [x] Extract `_get_embed_kwargs()` helper to DRY model kwarg logic
- [x] Add unit tests for embeddings (`tests/unit_tests/test_embeddings.py`)
  - [x] `test_embed_documents` — all texts passed in single call
  - [x] `test_embed_documents_with_model` — model kwarg forwarded
  - [x] `test_embed_documents_no_model_kwarg` — no extra kwarg when model is None
  - [x] `test_aembed_documents` — async variant
  - [x] `test_embed_query` — delegates to embed_documents
  - [x] `test_embed_query_with_prefix` — prefix prepended when enabled
  - [x] `test_aembed_query` — async query variant
- [x] Verification
  - [x] `uv run ruff check` — passed
  - [x] `uv run ruff format --check` — passed
  - [x] `uv run mypy` — passed
  - [x] `uv run pytest` — 83 passed, 79% coverage

### 2.10. Rewrite README.md
- [ ] First iteration completed and submitted for review; finalization pending review feedback
  - [x] README rewritten to follow `gigachat` package style
  - [x] Known mismatch fixed: removed/updated SDK-only `giga.get_token()` reference from package docs
  - [ ] Final pass after review comments (wording/examples consistency)

### 2.16. CI Refactoring
- [x] Review tests and CI scope
  - [x] Reviewed test suite: removed outdated cases and added missing coverage in active areas
  - [x] Assessed current coverage and identified expansion priorities
  - [x] Kept VCR tests out of scope for this iteration (explicitly deferred)
- [x] Documentation synced in `docs/REFACTORING.md`

### 2.18. Expose SDK Connection Settings
- [x] Add `max_retries`, `max_connections`, `retry_backoff_factor`, `retry_on_status_codes` fields to `_GigaChatClientMixin` in `_client.py`
  - [x] All fields default to `None` (SDK defaults apply via env vars or `Settings`)
  - [x] Added docstrings with SDK defaults and retry-stacking warning
- [x] Forward new fields through `_get_client_init_kwargs()` to SDK constructor
- [x] Update `GigaChat` class docstring in `gigachat.py` to document new parameters
- [x] Add unit tests
  - [x] `test_connection_settings_defaults` — all fields default to None
  - [x] `test_connection_settings_explicit_values` — fields accept explicit values
  - [x] `test_connection_settings_forwarded_to_sdk` — values forwarded to SDK constructor
  - [x] `test_connection_settings_none_forwarded_to_sdk` — None forwarded when unset
  - [x] `test_embeddings_connection_settings_forwarded` — works for `GigaChatEmbeddings` too
- [x] Verification
  - [x] `uv run ruff check` — passed
  - [x] `uv run ruff format --check` — passed
  - [x] `uv run mypy` — passed
  - [x] `uv run pytest` — 51 passed, 70% coverage (100% on `_client.py`)

### 2.11. Remove `trim_content_to_stop_sequence`
- [x] Remove `trim_content_to_stop_sequence()` from `gigachat.py`
- [x] Remove trimming loop from `_generate` and `_agenerate`
- [x] Remove `message_content` and stop-check from `_stream` and `_astream`
- [x] Update `docs/REFACTORING.md` (checkbox + dedicated section)

### 2.13. `TYPE_CHECKING` Block
- [x] Remove `TYPE_CHECKING` and conditional `import gigachat.models as gm` from `gigachat.py`
- [x] Use top-level `import gigachat.models as gm` (no circular import; types used in annotations only)
- [x] Update `docs/REFACTORING.md` (checkbox)

### 2.17. `get_file` Naming and API Surface Cleanup
- [x] Align `get_file`/`aget_file` with SDK: now return file metadata (`gm.UploadedFile`) via SDK `get_file`/`aget_file`
- [x] Add `get_file_content`/`aget_file_content` for file content (base64, `gm.Image`) via SDK `get_image`/`aget_image`
- [x] Add `list_files`/`alist_files` (GET /files), `delete_file`/`adelete_file` (DELETE /files/{id})
- [x] Document in `docs/REFACTORING.md` (checkbox + "File API Cleanup (2.17)" section)

### 2.14. Function/Tool Message Handling Fixes (partial)
- [x] Forward `FunctionMessage.name` to `gm.Messages.name` in `_convert_message_to_dict()`
  - [x] `kwargs["name"] = message.name` always set (field is required in `FunctionMessage`)
  - [x] Fixes silent bug: every `FunctionMessage` was sent to the API without a function name
- [x] Forward `ToolMessage.name` to `gm.Messages.name` in `_convert_message_to_dict()`
  - [x] `kwargs["name"] = message.name` set when `message.name` is truthy (field is optional in `ToolMessage`)
  - [x] When name is absent `gm.Messages.name` stays `None`; API returns an explicit error instead of silently corrupting the payload
- [x] Raise `ValueError` when `AIMessage.tool_calls` contains more than one entry
  - [x] GigaChat API does not support parallel function calls in one turn
  - [x] Previously only `tool_calls[0]` was forwarded; remaining calls were silently dropped, corrupting conversation history
  - [x] Explicit error surfaces the API limitation at conversion time, consistent with `tool_choice='any'` handling
- [x] Update tests
  - [x] Fix `test__convert_message_to_dict_function` — add `name="func"` to expected `Messages` (revealed pre-existing bug)
  - [x] Add `test__convert_message_to_dict_tool_message_with_name` — name is forwarded when set
  - [x] Add `test__convert_message_to_dict_tool_message_without_name` — name stays `None` when absent
- [x] Verification: `uv run ruff check` — passed; `uv run pytest` — 57 passed
- [x] Decision alignment with FunctionCall bridge
  - [x] Keep `ToolMessage -> role="function"` mapping as the accepted compatibility strategy
  - [x] Revisit only after upstream GigaChat API/SDK introduces native `tool_calls` transport
- [x] SDK exception translation decision
  - [x] Audited all SDK call sites — no exceptions silently swallowed or losing context
  - [x] Decided: keep SDK exceptions as-is (aligned with `langchain-openai` practice; `langchain_core.exceptions` too minimal for meaningful translation)
  - [x] Documented rationale in `docs/REFACTORING.md` → "SDK Exception Translation Policy"
- [ ] TODO (remaining scope for 2.14)
  - [ ] Validate compatibility with `create_agent`, middleware, and modern content blocks

### Cleanup `_convert_any_typed_dicts_to_pydantic` in `function_calling.py`
- [x] Replace `_parse_google_docstring` local copy with import from `langchain_core.utils.function_calling`
- [x] Use `get_type_hints(typed_dict, include_extras=True)` with fallback (PEP 649 / Python 3.14+ compat, synced with upstream langchain-core)
- [x] Remove dead `else: pass` branch; change `elif` to `if` for docstring description override
- [x] Add docstring to `_convert_any_typed_dicts_to_pydantic`
- [x] Verification: `uv run ruff check` — passed; `uv run pytest` — 35 passed

### 2.12. `x_headers` Audit
- [x] Document where `x_headers` are set/used (`message.id`, `llm_output`, streaming metadata)
- [x] Add shared stream helper to centralize header propagation behavior
- [x] Keep non-stream behavior explicit (headers in `llm_output`, request id in message id)

### Support reasoning models (`reasoning_effort` / `reasoning_content`)
- [x] Add `reasoning_effort` request parameter to `_BaseGigaChat`
- [x] Include `reasoning_effort` in payload building when configured
- [x] Surface `reasoning_content` in assistant messages (`additional_kwargs`) for sync and stream paths

### Remove `_check_finish_reason`
- [x] Remove `_check_finish_reason()` method from `GigaChat` class
- [x] Remove `_NORMAL_FINISH_REASONS` and `_ERROR_FINISH_REASONS` constants
- [x] Remove call sites in `_create_chat_result()` and `_build_stream_chunk()`
- [x] Remove `test_build_stream_chunk_unusual_finish_reason_warns` test
- [x] Verification: `uv run ruff check` — passed; `uv run pytest` — 198 passed

### Upstream SDK privacy/logging follow-up (external)
- [x] Track upstream hardening for raw SSE parser error logs — **cancelled**: low priority, occurs only on SSE parse errors (rare). If needed, address directly in `gigachat` SDK repo (`api/utils.py:parse_chunk`).
