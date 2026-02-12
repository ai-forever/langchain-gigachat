# Refactoring Notes

**Note**: All information in this file must be grouped by specific issues. Do not separate problems and solutions into different sections; keep them together under the relevant issue heading.

**Context**: The upstream `gigachat` package (v0.2.0) has undergone significant refactoring including Pydantic V2 migration, removal of the `verbose` parameter, improved exception hierarchy, and toolchain migration from Poetry to uv. This refactoring effort aligns `langchain-gigachat` with those changes.

## Dependency Management Strategy

- **Problem**: During simultaneous refactoring of `gigachat` and `langchain-gigachat`, the dependency on `gigachat` must be managed carefully:
  1. **Local path**: `gigachat = { path = "..." }` breaks CI (path doesn't exist on CI runners).
  2. **PyPI**: Can't publish to PyPI until both packages are ready for simultaneous release.
  3. **Circular dependency**: Changes in one package may require changes in the other before either is releasable.
- **Solution**: Install `gigachat` from git branch using PEP 508 direct reference syntax:
  ```toml
  dependencies = [
    "gigachat @ git+https://github.com/ai-forever/gigachat.git@<branch-name>",
  ]
  ```
  See `libs/gigachat/pyproject.toml` for the current branch in use.
  Note: Requires `[tool.hatch.metadata] allow-direct-references = true` in pyproject.toml.
- **Why**:
  - **CI works**: Git URLs resolve on any machine with network access.
  - **Version pinning**: The branch tracks in-progress refactoring changes.
  - **Coordinated release**: Both packages can be developed in sync, then released simultaneously when ready.
- **When to change**: Update to PyPI version (e.g., `gigachat>=X.Y.Z,<X+1`) after coordinated release of both packages.
- **Applies when**: Refactoring involves breaking changes to `gigachat` that require simultaneous development of both packages.

## Workflow

### Progress Tracking
- Refer to `docs/TODO.md` for current status of refactoring tasks.
- Tasks are grouped by issue.
- Only analyzed and approved issues are added to the active plan.
- **Chronological Order**: All sections (issues) must be listed in chronological ascending order (oldest first). New tasks should always be added at the end.

### Implementation Process
1. Before implementing each todo item list, get approval.
2. After implementation, summarize results.
3. After solving each issue:
   - Add detailed information about the solution (why and how) to this file.
   - Update `docs/TODO.md` to reflect detailed implemented steps.

---

## Pydantic V2 Migration

- **Problem**: The codebase uses deprecated Pydantic V1 patterns that are incompatible with the refactored `gigachat` package (which requires `pydantic >= 2`):
  1. **Explicit V1 Imports** in `langchain_gigachat/utils/function_calling.py`:
     - `from pydantic.v1 import BaseModel`
     - `from pydantic.v1 import Field as Field_v1`
     - `from pydantic.v1 import create_model as create_model_v1`
  2. **Deprecated Methods** in `langchain_gigachat/chat_models/gigachat.py`:
     - `Chat.parse_obj(payload_dict)` in `_build_payload()` → should be `Chat.model_validate()`
     - `response.usage.dict()` in `_create_chat_result()` → should be `response.usage.model_dump()`
     - `chunk_d.dict()` in `_stream()` and `_astream()` → should be `chunk_d.model_dump()`
  3. **Old `Config` Class** (V1 style):
     - `langchain_gigachat/chat_models/base_gigachat.py`: `class Config: arbitrary_types_allowed = True`
     - `langchain_gigachat/embeddings/gigachat.py`: Same pattern
     - Should use `model_config = ConfigDict(arbitrary_types_allowed=True)`
  4. **Outdated Dependency Version**:
     - `libs/gigachat/pyproject.toml`: `gigachat = "^0.1.41.post1"` needs update to `^0.2.0`
- **Analysis**:
  - LangChain Core v0.3+ is Pydantic V2-native.
  - The `gigachat` package now requires `pydantic >= 2.0`.
  - Mixing V1 and V2 patterns causes type-checking issues and runtime surprises.
  - The `pydantic.v1` imports in `function_calling.py` are particularly problematic as they create V1 models that may be incompatible with V2 validation.
- **Solution**:
  - **Implementation Details**:
    - Replaced all `pydantic.v1` imports with native `pydantic` imports in `function_calling.py`.
    - Updated `Field_v1` → `Field` and `create_model_v1` → `create_model`.
    - Replaced `.dict()` → `.model_dump()` throughout `gigachat.py`.
    - Replaced `.parse_obj()` → `.model_validate()` in `gigachat.py`.
    - Replaced `class Config:` with `model_config = ConfigDict(...)` in `base_gigachat.py` and `embeddings/gigachat.py`.
    - Updated `gigachat` dependency (currently using local editable path for development).
  - **Test Adjustments**:
    - Pydantic V2 includes `default` values in JSON schema for fields with defaults. Updated test expectations in `test_function_calling.py` to account for `'default': None` being present for optional TypedDict fields.
    - Added `is_typeddict` check in tests to conditionally adjust expected schema for TypedDict inputs.
  - **Why**:
    - **Compatibility**: Required for compatibility with `gigachat >= 0.2.0`.
    - **Performance**: Pydantic V2 is 5-50x faster.
    - **Ecosystem**: Aligns with LangChain Core v0.3+.
- **Verification**:
  - `ruff check` — passed
  - `ruff format --check` — passed
  - `mypy` — passed
  - `pytest` — 74 passed, 2 xpassed (previously xfail tests now pass due to V2 improvements)
- **Cleanup Completed**:
  - Removed `BaseModelV2Maybe`/`FieldV2Maybe` test aliases (legacy V1/V2 distinction).
  - Removed `hasattr` compatibility checks in `output_parsers/gigachat_functions.py`.
  - Refactored `_model_to_schema()` to use only `model_json_schema()` (removed `.schema()` V1 fallback).
- **Status**: Completed.

## Remove `verbose` Parameter

- **Problem**: The `verbose` parameter provided request/response logging functionality, but this is inconsistent with standard LangChain patterns and the upstream `gigachat` SDK (which removed the parameter).
  - Location: `langchain_gigachat/chat_models/base_gigachat.py`: `verbose: bool = False` field in `_BaseGigaChat`
  - Location: `langchain_gigachat/chat_models/base_gigachat.py`: `verbose=self.verbose` passed to `_client` property
  - Location: `langchain_gigachat/chat_models/gigachat.py`: Request logging in `_build_payload()`, response logging in `_create_chat_result()`
- **Solution**:
  - **Implementation Details**:
    - Removed `verbose: bool = False` field from `_BaseGigaChat`.
    - Removed `verbose=self.verbose` from `_client` property.
    - Removed `if self.verbose:` logging blocks from `_build_payload()` and `_create_chat_result()`.
  - **Why**:
    - **Upstream Alignment**: The `gigachat` SDK removed this unused parameter.
    - **Consistency**: Standard logging via Python `logging` module is preferred.
    - **LangChain Patterns**: Aligns with standard LangChain practices.
- **Upstream Coordination**: Required by upstream `gigachat` SDK which removed the parameter.
- **Verification**:
  - `ruff check` — passed
  - `mypy` — passed
  - `pytest` — 74 passed, 2 xpassed
- **Status**: Completed.

## Poetry to uv Migration

- **Problem**: The project used Poetry for dependency management and build, while the upstream `gigachat` package migrated to uv. Alignment with upstream tooling simplifies development and CI/CD.
- **Solution**:
  - **pyproject.toml Conversion**:
    - Replaced `[tool.poetry]` with PEP 621 `[project]` section.
    - Changed build backend from `poetry-core` to `hatchling`.
    - Consolidated 4 Poetry dependency groups (dev/lint/typing/test) into single `[dependency-groups]` `dev` group.
    - Added `[tool.hatch.metadata]` with `allow-direct-references = true` for git dependency support.
    - Fixed mypy config: changed string `"True"` to boolean `true`.
  - **Legacy Code Removal**:
    - Removed `langchain_gigachat/tools/load_prompt.py` — legacy module not exported in public API.
    - Removed `tests/unit_tests/test_utils.py` — only tested the removed module.
    - Removed `types-requests`, `requests`, and `requests_mock` dependencies — only needed by removed module.
  - **Makefile Updates**:
    - Replaced all `poetry run` commands with `uv run`.
  - **CI/CD Rewrite**:
    - Replaced custom `.github/actions/poetry_setup/` with `astral-sh/setup-uv@v5`.
    - Simplified `_lint.yml` and `_test.yml` workflows.
    - Updated cache keys from `poetry.lock` to `uv.lock`.
    - Expanded test matrix: Python 3.9, 3.10, 3.11, 3.12, 3.13 (required) + 3.14 (experimental).
  - **Lock File**:
    - Deleted `poetry.lock`, generated `uv.lock`.
  - **Documentation**:
    - Updated `AGENTS.md` with uv commands.
- **Why**:
  - **Upstream Alignment**: Matches `gigachat` package tooling.
  - **Performance**: uv is significantly faster than Poetry.
  - **Simplicity**: uv's built-in caching and simpler workflow configuration.
  - **Modern Standards**: PEP 621 is the standard for Python project metadata.
- **Verification**:
  - `uv sync` — installed 50 packages successfully
  - `uv run ruff check` — passed
  - `uv run mypy` — passed (13 source files)
  - `uv run pytest` — 73 passed, 2 xpassed, 70% coverage
- **Post-Migration Cleanup**:
  - Deleted empty `.github/actions/` directory (poetry_setup was removed earlier).
  - Deleted `.github/scripts/get_min_versions.py` — unused script that parsed `tool.poetry.dependencies` (no longer exists after PEP 621 migration).
  - Kept `.github/scripts/check_diff.py` — actively used by `check_diffs.yml`.
- **Status**: Completed.

## Code Cleanup

- **Problem**: The codebase contained legacy code and outdated documentation:
  1. `GigaChat` class docstring referenced `langchain_community.chat_models` (wrong import path) and used verbose example-heavy format instead of the upstream `Args:` format.
  2. Commented-out `FunctionInProgressMessageChunk` block in `_convert_delta_to_message_chunk()` — dead code for unimplemented feature.
  3. Unused `_convert_function_to_dict()` function in `gigachat.py` — legacy code superseded by `convert_to_gigachat_function()` in `function_calling.py`.
  4. Unused `_get_type_hints()` wrapper function in `function_calling.py` — the actual code uses `get_type_hints()` directly.
- **Solution**:
  - **Docstring Rewrite**: Replaced example-heavy docstring with upstream `gigachat` SDK format using `Args:` section documenting all parameters.
  - **Dead Code Removal**:
    - Deleted 5-line commented `FunctionInProgressMessageChunk` block.
    - Deleted 27-line `_convert_function_to_dict()` function.
    - Deleted 7-line `_get_type_hints()` function.
    - Removed unused `functools` import.
  - **Why**:
    - **Consistency**: Docstring format matches upstream `gigachat` SDK.
    - **Maintainability**: Removed ~40 lines of dead code.
    - **Correctness**: Import path now references `langchain_gigachat` (implicitly via `Args:` format).
- **Verification**:
  - `uv run ruff check` — passed
  - `uv run ruff format` — passed
  - `uv run pytest` — 73 passed, 2 xpassed
- **Status**: Completed.

## LangChain Core 1.x Support (Branch: `lc1-support`)

- **Problem**: LangChain Core 1.x introduced breaking changes including removal of deprecated methods (`predict`, `apredict`) and stricter typing. The package needed adaptation to support `langchain-core>=1,<2`.
- **Solution**:
  - **Dependency Updates** (`pyproject.toml`):
    - `langchain-core>=0.3,<1` → `langchain-core>=1,<2`
    - `requires-python>=3.9` → `>=3.10` (LangChain 1.x requirement)
    - Temporary package rename: `langchain-gigachat` → `langchain-gigachat-lc1`
  - **Removed Deprecated Fields**:
    - `_BaseGigaChat`: Removed `profanity: bool` field (use `profanity_check` instead)
    - `_BaseGigaChat`: Removed `validate_environment()` pre_init validator. It performed three checks that became redundant:
      1. **Import check** (`import gigachat`) — now `gigachat` is imported directly at module level, so ImportError occurs immediately on module load
      2. **Extra fields warning** — Pydantic V2 handles unknown fields via `model_config`, duplicating this check is unnecessary
      3. **`profanity` → `profanity_check` migration** — the deprecated `profanity` field is now fully removed, no migration needed
    - `GigaChatEmbeddings`: Removed `one_by_one_mode: bool` and `_debug_delay: float` fields
    - `GigaChatEmbeddings`: Removed `validate_environment()` validator (same reasons as above)
  - **Removed Unused Code**:
    - Deleted `_get_python_function_name()` — replaced with direct `function.__name__`
    - Deleted `_FunctionCall` TypedDict — unused
    - Deleted `output_parsers/gigachat_functions.py` module — legacy parsers no longer needed
  - **Import Cleanup**:
    - Direct `gigachat` imports instead of `TYPE_CHECKING` block (simplifies code)
    - Removed unused `logging` module imports where logger was not used
  - **`tool_choice='any'` Handling**:
    - GigaChat API does not support `tool_choice='any'` (forces model to call some tool)
    - **Default behavior**: Raises `ValueError` with clear error message explaining the limitation
    - **Opt-in fallback**: New `allow_any_tool_choice_fallback: bool = False` parameter in `GigaChat` class
    - When `allow_any_tool_choice_fallback=True`: converts `'any'` → `'auto'` with `UserWarning`
    - **Rationale**: Silent conversion to `'auto'` changes semantics unpredictably — user expects forced tool call, but model may return text instead. Explicit error is safer; users must consciously opt-in to fallback behavior.
    - **Tests**: Added 4 tests covering default error, fallback with warning, and normal `'auto'`/specific tool cases
  - **Type Annotation Improvements**:
    - `bind_functions()` and `bind_tools()` return `Runnable[..., AIMessage]` (was `BaseMessage`). Chat models always return `AIMessage` — the previous `BaseMessage` annotation was overly broad. This is a backwards-compatible refinement that improves IDE autocompletion and static analysis.
    - Removed redundant `# type: ignore` comments
    - Removed redundant `isinstance()` checks after type-narrowing assignments
  - **Module Exports**:
    - `tools/__init__.py`: Added exports for `FewShotExamples`, `GigaBaseTool`, `GigaStructuredTool`, `GigaTool`, `giga_tool`
    - `utils/__init__.py`: Added exports for `convert_to_gigachat_function`, `convert_to_gigachat_tool`
  - **Minor Fixes**:
    - Compiled `BASE64_DATA_REGEX` at module level (was inline, repeated)
    - Fixed `_identifying_params` key: `profanity` → `profanity_check`
  - **Test Updates**:
    - Removed tests for `predict()`/`apredict()` methods (removed in LangChain 1.x). Callback coverage preserved: added `test_gigachat_stream_callbacks` using modern API (`config={"callbacks": [...]}`).
    - Deleted `tests/unit_tests/stubs.py` (270+ lines) — contained `FakeCallbackHandler`/`FakeAsyncCallbackHandler` used only by removed tests. New test uses simple 5-line inline `TokenCounter` class instead.
    - Added expected `chunk_position="last"` chunk in stream tests — LangChain Core 1.x `BaseChatModel.stream()` now appends a final marker chunk to signal stream completion. This is upstream behavior, not our code.
- **Why**:
  - **Ecosystem Alignment**: LangChain Core 1.x is the current stable version
  - **Reduced Surface Area**: Deprecated code removal simplifies maintenance
  - **Better DX**: Proper module exports improve discoverability
- **Migration Notes** (for users upgrading from `langchain-core<1`):
  - Replace `llm.predict("text")` → `llm.invoke("text").content`
  - Replace `await llm.apredict("text")` → `(await llm.ainvoke("text")).content`
  - If using `tool_choice='any'`: now raises `ValueError` by default. Either use `'auto'`/specific tool name, or set `GigaChat(allow_any_tool_choice_fallback=True)` for automatic conversion to `'auto'`
- **Status**: In progress (testing).

---

## Embeddings Batch Settings

- **Problem**: `GigaChatEmbeddings` split input lists into sub-batches using two hard-coded constants:
  - `MAX_BATCH_SIZE_CHARS = 1_000_000` (total character count threshold)
  - `MAX_BATCH_SIZE_PARTS = 90` (max texts per request)

  The batching loop accumulated texts, flushed a batch when either limit was exceeded, then processed the remainder — duplicated identically in `embed_documents()` and `aembed_documents()`.
- **Analysis**:
  - The GigaChat Embeddings API accepts `input` as either a `string` or `List[string]` natively — the server handles batching internally.
  - The upstream `gigachat` SDK passes the list directly to the API without any client-side splitting.
  - The client-side batch limits were defensive but unnecessary, adding complexity without benefit.
- **Solution**:
  - **Removed** `MAX_BATCH_SIZE_CHARS` and `MAX_BATCH_SIZE_PARTS` constants.
  - **Simplified** `embed_documents()` and `aembed_documents()` to a single SDK call passing all texts at once.
  - **Extracted** `_get_embed_kwargs()` helper to DRY the `model` kwarg logic shared by both methods.
  - **Added** 7 unit tests covering sync/async `embed_documents`, `embed_query`, prefix query, and model forwarding (97% coverage on `embeddings/gigachat.py`).
- **Why**:
  - **Simplicity**: Removed ~40 lines of duplicated batching logic.
  - **Correctness**: Client-side splitting could reorder or misalign results if the API returned data differently per batch.
  - **API alignment**: Trusts the server to manage batch limits, matching the SDK's behavior.
- **Verification**:
  - `uv run ruff check` — passed
  - `uv run ruff format --check` — passed
  - `uv run mypy` — passed
  - `uv run pytest` — 83 passed (7 new embeddings tests), 79% coverage
- **Status**: Completed.

---

## Phase 2: Refactoring Plan

Agreed upon during the refactoring review meeting. Each item will be expanded with a full Problem/Solution/Verification writeup in a dedicated section as work begins (per the Workflow section above).

- [x] **2.1. Mixin for Chat and Embeddings** — Reduce code duplication between `GigaChat` and `GigaChatEmbeddings` by extracting shared logic (client init, auth, config) into a mixin or shared base class. Also add missing `lc_secrets` to `GigaChatEmbeddings` (currently absent — credentials can leak during serialization).
- [ ] **2.2. Base64 Image Handling** — Check API status for direct base64 submission. If supported — switch to it. If not — implement proper caching with eviction (current `_cached_images` dict has no eviction, risk of overflow). Also fix `_cached_images` from class attribute to per-instance private attr (currently shared across all instances — multi-tenant risk).
- [ ] **2.3. Multimodal File Upload** — Support audio, document (and possibly video) input upload. Extend `get_text_and_images_from_content()` to handle new content block types beyond `text`/`image_url`. Verify compatibility with LangChain content blocks.
- [ ] **2.4. Format Instructions Mode** — Decide: keep or remove `with_structured_output(method="format_instructions")`. If keep — rewrite. *Owner: Крестников.*
- [ ] **2.5. LangChain Legacy (LCL) Chains Review** — Full review of all legacy LangChain chain patterns in the code. Remove where possible. Includes reviewing `bind_functions` (legacy path) — docstring mentions "auto" but implementation only supports force-by-name.
- [ ] **2.6. Register on models.dev** — Add GigaChat models to [models.dev](https://models.dev).
- [ ] **2.7. `profiles.py`** — Research how `profiles.py` works in other LangChain partner packages. Determine necessity, add if needed. Currently absent.
- [ ] **2.8. `giga_tool` Decorator Revision** — Review extra functionality (`return_schema`, `few_shot_examples`) over standard `@tool`. If replaceable by LangChain extras — remove. If removed: rewrite examples, document as **breaking change**.
- [x] **2.9. Embeddings Batch Settings** — API natively handles batches (`input` accepts `List[string]`). Removed custom `MAX_BATCH_SIZE_CHARS` / `MAX_BATCH_SIZE_PARTS` logic. See dedicated section below.
- [ ] **2.10. Rewrite README.md** — Full rewrite following `gigachat` package README style. Fix known mismatch: RU README references `giga.get_token()` which is SDK-only, not wrapped. *Blocked: do after refactoring is complete.*
- [ ] **2.11. Remove `trim_content_to_stop_sequence`** — Fully remove the function and all call sites (`_generate`, `_agenerate`, `_stream`, `_astream`). Stop sequence handling should be API-side.
- [ ] **2.12. `x_headers` Audit** — Map all places where `x_headers` are set/consumed (`response_metadata`, `generation_info`, `message.id`). Decide on refactoring or documentation.
- [ ] **2.13. `TYPE_CHECKING` Block** — Remove conditional `TYPE_CHECKING` import in `gigachat.py` or confirm it is necessary.
- [ ] **2.14. LangChain 1.0 New Mechanisms** — Test compatibility with content blocks, `create_agent`, middleware. Additionally review: multi-tool calling support (currently only `tool_calls[0]` is forwarded), `ToolMessage` role mapping (`role="function"` — check if API supports a proper tool role), and SDK exception translation to LangChain exception types.
- [ ] **2.15. CI/Contribution Documentation** — Create or rewrite CI docs, contribution guide, and other developer docs following LangChain upstream conventions.
- [ ] **2.16. CI Refactoring** — Review tests (remove unnecessary, add missing), assess coverage (~70%), decide on expansion. VCR tests — not now.
- [ ] **2.17. `get_file` Naming and API Surface Cleanup** — `_BaseGigaChat.get_file/aget_file` actually calls SDK `get_image/aget_image` (downloads file content, not metadata). Rename or document clearly. Also consider wrapping additional SDK-only file endpoints (`GET /files`, `DELETE /files/{id}`) if useful.
- [ ] **2.18. Expose SDK Connection Settings** — `max_retries`, `max_connections`, `retry_backoff_factor` are currently SDK-only (configurable only via `GIGACHAT_*` env vars). Evaluate whether to expose them as explicit LangChain wrapper fields for discoverability.
