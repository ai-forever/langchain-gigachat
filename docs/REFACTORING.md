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
- Tasks are grouped by issue.
- Only analyzed and approved issues are added to the active plan.
- **Chronological Order**: All sections (issues) must be listed in chronological ascending order (oldest first). New tasks should always be added at the end.

### Implementation Process
1. Before implementing each todo item list, get approval.
2. After implementation, summarize results.
3. After solving each issue:
   - Add detailed information about the solution (why and how) to this file.
   - Update `docs/REFACTORING_CHANGELOG.md` when any refactoring item is marked as completed.

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
  - **Before release/merge**: rename package in `libs/gigachat/pyproject.toml` from `langchain-gigachat-lc1` back to `langchain-gigachat` to avoid publishing/dependency confusion.
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
- [x] **2.2. Base64 Image Handling** — Implement proper caching with eviction and per-instance cache. See dedicated section below.
- [x] **2.3. Multimodal File Upload** — Support audio, document (and possibly video) input upload. Extend `get_text_and_images_from_content()` to handle new content block types beyond `text`/`image_url`. Verify compatibility with LangChain content blocks. See dedicated section below.
- [x] **2.4. Format Instructions Mode** — **Breaking change approved**: remove `with_structured_output(method="format_instructions")` from public API. Rationale: issue #40 is solved via `function_calling` JSON/Pydantic schema support, while `format_instructions` remains legacy prompt-based behavior with weak schema guarantees and extra maintenance cost. Migration: use `method="function_calling"` (preferred) or `method="json_mode"` where applicable.
- [x] **2.5. LangChain Legacy (LCL) Chains Review** — Full review of all legacy LangChain chain patterns in the code. Remove where possible. Includes reviewing `bind_functions` (legacy path) — docstring mentions "auto" but implementation only supports force-by-name.
- [x] **2.6. Register on models.dev** — Add GigaChat models to [models.dev](https://models.dev).
- [ ] **2.7. `profiles.py`** — On hold: PR submitted, waiting for review result; no actions for now.
- [x] **2.8. `giga_tool` Decorator Revision** — Review extra functionality (`return_schema`, `few_shot_examples`) over standard `@tool`. If replaceable by LangChain extras — remove. If removed: rewrite examples, document as **breaking change**.
- [x] **2.9. Embeddings Batch Settings** — API natively handles batches (`input` accepts `List[string]`). Removed custom `MAX_BATCH_SIZE_CHARS` / `MAX_BATCH_SIZE_PARTS` logic. See dedicated section below.
- [ ] **2.10. Rewrite README.md** — First iteration completed and submitted for review; finalization pending review feedback. The rewrite follows `gigachat` package README style. Known mismatch about SDK-only `giga.get_token()` reference is fixed.
- [x] **2.11. Remove `trim_content_to_stop_sequence`** — Fully remove the function and all call sites (`_generate`, `_agenerate`, `_stream`, `_astream`). Stop sequence handling should be API-side. See dedicated section below.
- [x] **2.12. `x_headers` Audit** — Map all places where `x_headers` are set/consumed (`response_metadata`, `generation_info`, `message.id`). Decide on refactoring or documentation.
- [x] **2.13. `TYPE_CHECKING` Block** — Remove conditional `TYPE_CHECKING` import in `gigachat.py` or confirm it is necessary.
- [ ] **2.14. LangChain 1.0 New Mechanisms** — Test compatibility with content blocks, `create_agent`, middleware. Additionally review: ~~multi-tool calling support (currently only `tool_calls[0]` is forwarded)~~ (done: raises `ValueError`), ~~`ToolMessage`/`FunctionMessage` name forwarding~~ (done), ~~`ToolMessage` role mapping (`role="function"`)~~ (accepted for current `FunctionCall` bridge; revisit after upstream native `tool_calls` support), and SDK exception translation to LangChain exception types.
- [ ] **2.19. SDK `FunctionParametersProperty` Schema Stripping** — Upstream bug: SDK Pydantic model silently drops JSON Schema fields (`additionalProperties`, nested `required`, `format`, etc.), causing 422 errors. Fix required in `gigachat` SDK. See [dedicated section](#sdk-functionparameterssproperty-schema-stripping-219) and issues [#55](https://github.com/ai-forever/langchain-gigachat/issues/55), [#59](https://github.com/ai-forever/langchain-gigachat/issues/59).
- [x] **2.15. CI/Contribution Documentation** — Create or rewrite CI docs, contribution guide, and other developer docs following LangChain upstream conventions.
- [x] **2.16. CI Refactoring** — Completed: tests reviewed (obsolete removed, missing added), coverage assessed, and expansion scope documented. VCR tests remain out of scope for now.
- [x] **2.17. `get_file` Naming and API Surface Cleanup** — `_BaseGigaChat.get_file/aget_file` actually calls SDK `get_image/aget_image` (downloads file content, not metadata). Rename or document clearly. Also consider wrapping additional SDK-only file endpoints (`GET /files`, `DELETE /files/{id}`) if useful.
- [x] **2.18. Expose SDK Connection Settings** — `max_retries`, `max_connections`, `retry_backoff_factor`, `retry_on_status_codes` are now exposed as explicit fields on `_GigaChatClientMixin` (shared by `GigaChat` and `GigaChatEmbeddings`). See dedicated section below.

## CI/Contribution Documentation (2.15)

- **Goal**: Create or rewrite CI docs, contribution guide, and GitHub templates following LangChain upstream conventions. Reference: [gigachat contribution guidelines and templates](https://github.com/ai-forever/gigachat/commit/07756b7923371b2e10014550aeb8d034d08a9443).
- **Plan**:
  1. **CONTRIBUTING.md** (repository root) — Contribution guide adapted from gigachat: types of contributions, development setup (from `libs/gigachat`, uv, make targets), code quality (ruff, mypy, Google docstrings, imperative mood), testing (`make test`, `make lint_package`, `make lint_tests`), commit message guidelines (conventional commits), PR process, issue reporting. Align with AGENTS.md (setup commands, code style). Omit gigachat-specifics (SSL certs, integration test credentials) or keep minimal; document monorepo: run all commands from `libs/gigachat`.
  2. **.github/ISSUE_TEMPLATE/** — (a) `config.yml`: disable blank issues, add contact links (e.g. LangChain docs, GigaChat API docs, related projects). (b) `bug_report.yml`: description, steps to reproduce, expected/actual behaviour, code example, langchain-gigachat and Python versions, OS, optional environment. (c) `feature_request.yml`: problem statement, proposed solution, optional API design, feature area (chat, embeddings, function calling, etc.), priority. Adapt labels and placeholders from gigachat to "langchain-gigachat" / "GigaChat LangChain integration".
  3. **.github/PULL_REQUEST_TEMPLATE.md** — Sections: Description, Motivation (with Closes #), Type of change (bug fix, feature, docs, refactor, etc.), Changes made, Testing (unit/integration, manual), Checklist (code quality, documentation, dependencies, compatibility, commits). Reference local commands: `make lint_package`, `make lint_tests`, `make test` from `libs/gigachat`; optional pre-merge changelog/version.
  4. **CI documentation** — Either a short "CI" subsection inside CONTRIBUTING.md or a single `docs/CI.md`. Describe: workflow runs on push/PR to master and dev; `check_diffs.yml` determines which libs to lint/test; lint workflow runs `make lint_package` and `make lint_tests` (Python 3.10, 3.12); test workflow runs `make test` on Python 3.10–3.13; contributors should run the same from `libs/gigachat` before opening a PR.
- **LangChain conventions**: Keep docstrings (Google style, imperative) and tooling (ruff, mypy) consistent with AGENTS.md and typical LangChain partner packages; contribution flow (fork, branch, PR template, checklist) matches common OSS practice and gigachat’s structure.
- **Status**: Completed. Added CONTRIBUTING.md (root), .github/ISSUE_TEMPLATE/config.yml, bug_report.yml, feature_request.yml, .github/PULL_REQUEST_TEMPLATE.md. CI is documented in CONTRIBUTING.md.

## Multimodal File Upload (2.3)

- **Problem**: Only `text` and `image_url` content blocks were supported. GigaChat API supports text documents, images, and audio (see [POST /file](https://developers.sber.ru/docs/ru/gigachat/api/reference/rest/post-file)); file IDs are passed in `attachments` for chat.
- **Solution**:
  - **Content blocks**: Extended `get_text_and_images_from_content()` to handle `audio_url` and `document_url` with the same pattern as `image_url`: `{"type": "audio_url", "audio_url": {"url": "...", "giga_id": "..."}}` and `document_url` analogously. All resolved IDs are collected into a single `attachments` list (API does not distinguish type). **Standard LangChain blocks** are also supported: `type` in `("image", "audio", "file")` with top-level `file_id` (and optional `url` for cache lookup).
  - **Display in LangChain `content_blocks`**: For messages to show up with proper typed blocks in `message.content_blocks` (e.g. `image`, `audio`, `file` instead of `non_standard`), use the **standard content block format** when constructing messages:
    - **Text**: `{"type": "text", "text": "..."}` (same in both formats).
    - **Image**: `{"type": "image", "url": "https://...", "file_id": "giga-file-id"}` or `{"type": "image", "file_id": "giga-file-id"}`. Use `url` if you rely on cache (e.g. after upload); use `file_id` for an already-known GigaChat file id.
    - **Audio**: `{"type": "audio", "file_id": "giga-file-id"}` or `{"type": "audio", "url": "...", "file_id": "..."}`.
    - **Document**: `{"type": "file", "file_id": "giga-file-id"}` or `{"type": "file", "url": "...", "file_id": "..."}`.
    - Example: `HumanMessage(content_blocks=[{"type": "text", "text": "Опиши вложения."}, {"type": "image", "file_id": "id-1"}, {"type": "audio", "file_id": "id-2"}, {"type": "file", "file_id": "id-3"}])`. The provider-native format (`image_url`/`audio_url`/`document_url` with nested objects) is still accepted; LangChain maps only `image_url` to standard `image` in `content_blocks`, while `audio_url` and `document_url` appear as `non_standard`, so for consistent display use the standard format above.
  - **Cache**: Reuse existing `_cached_images` (hash of URL → file id) for all attachment types so that uploaded audio/document data URLs are also cached.
  - **Upload**: Renamed `_upload_images` / `_aupload_images` to `_upload_attachments` / `_aupload_attachments`. Single flag `auto_upload_attachments` (default False) controls auto-upload for all data-URL blocks: `image_url`, `audio_url`, `document_url`. (Removed redundant `auto_upload_images`; use `auto_upload_attachments` for images too.) Upload uses existing `upload_file()`; MIME → extension mapping added for GigaChat-supported types (e.g. `audio/mp3` → `.mp3`, `application/epub` → `.epub`) where `mimetypes.guess_extension` returns None.
  - **Video**: Not implemented; API doc lists only text, image, and audio (mp4 is under audio MIME).
- **Verification**: `uv run ruff check`, `uv run pytest` (including `test_get_text_and_images_from_content_*`, `test_convert_message_to_dict_with_audio_and_document_attachments`, `test_auto_upload_attachments_*`).
- **Status**: Completed.

## Base64 Image Handling (2.2)

- **Problem**: `_cached_images` was a class-level dict (shared across all `GigaChat` instances — multi-tenant risk) with no eviction (unbounded growth, memory overflow risk).
- **Solution**:
  - **Per-instance cache**: Replaced `_cached_images: Dict[str, str] = {}` with `PrivateAttr(default_factory=dict)` so each instance has its own cache.
  - **Eviction**: Introduced `DEFAULT_IMAGE_CACHE_MAX_SIZE = 1000` and `_set_cached_image()`; when the cache is full, the oldest entry is removed (FIFO) before adding a new one. `_upload_images` and `_aupload_images` call `_set_cached_image()` instead of assigning directly.
- **Verification**: `uv run ruff check`, `uv run pytest` (including `test_ai_upload_image_per_instance_cache`, `test_ai_upload_image_cache_eviction`).
- **Status**: Completed.

## Remove `trim_content_to_stop_sequence`

- **Done**: Function and all call sites removed from `langchain_gigachat/chat_models/gigachat.py`.
  - Deleted `trim_content_to_stop_sequence()` helper.
  - `_generate` / `_agenerate`: no longer trim `choice.message.content` by stop sequences; response is passed through as returned by the API.
  - `_stream` / `_astream`: removed `message_content` accumulation and early return when a stop sequence was found in the stream.
- **Rationale**: Stop sequence handling is delegated to the API (payload already includes `stop` via `_build_payload`); client-side trimming was redundant and duplicated logic.

## Remove `format_instructions` Structured Output Mode

- **Problem**: `with_structured_output(method="format_instructions")` kept legacy prompt-based structured parsing in public API despite overlap with tool/schema-first approaches.
  - Location: `langchain_gigachat/chat_models/gigachat.py` — `with_structured_output()` method accepted `"format_instructions"` and injected parser instructions into user input.
  - Location: `tests/unit_tests/test_gigachat.py` — dedicated test for prompt injection behavior.
- **Solution**:
  - **Implementation Details**:
    - Removed `"format_instructions"` from all `with_structured_output()` method overloads and runtime signature.
    - Deleted legacy branch that built `RunnableLambda` and appended parser format instructions to prompt input.
    - Removed unit test that validated prompt-injection behavior for `"format_instructions"`.
  - **Why**:
    - **API Clarity**: Public structured-output API now exposes only supported modes: `function_calling` and `json_mode`.
    - **Reliability**: Removed prompt-engineering-based fallback with weaker schema guarantees.
    - **Maintenance**: Less branching and less mode-specific test surface.
- **Migration Notes**:
  - Replace `method="format_instructions"` with `method="function_calling"` (preferred for strict schema extraction) or `method="json_mode"` where schema forcing is not required.
- **Verification**:
  - `uv run ruff check libs/gigachat/langchain_gigachat/chat_models/gigachat.py libs/gigachat/tests/unit_tests/test_gigachat.py`
  - `uv run pytest libs/gigachat/tests/unit_tests/test_gigachat.py -k structured_output`
- **Status**: Completed.

## `with_structured_output` Override Typing Compatibility

- **Problem**: `make lint` failed on `mypy` due to an incompatible method override in `GigaChat.with_structured_output()` compared to `langchain_core.language_models.chat_models.BaseChatModel`.
  - Location: `langchain_gigachat/chat_models/gigachat.py` (around method definition for `with_structured_output`).
  - Error: `Signature of "with_structured_output" incompatible with supertype "BaseChatModel"`.
  - Root cause:
    - Provider-specific `method` was part of the typed signature instead of being handled through `**kwargs`.
    - Overload return types narrowed the contract and were not accepted as a compatible override.
    - Signature did not follow the base contract shape used by current LangChain integrations.
- **Solution**:
  - **Implementation Details**:
    - Updated `with_structured_output` signature to match `BaseChatModel` contract:
      - `schema: Dict[str, Any] | type`
      - `include_raw: bool = False`
      - `**kwargs: Any`
    - Kept provider-specific mode selection via `kwargs.pop("method", "function_calling")` with validation for supported values (`function_calling`, `json_mode`).
    - Removed overload-based typing branch and helper TypedDict used only by these overloads.
    - Updated method docstring to document supported `kwargs` options (`method`) and `include_raw` behavior.
  - **Why**:
    - **Type Safety**: Restores `mypy` compatibility with superclass override rules.
    - **LangChain Consistency**: Follows the same pattern as other partner packages: base contract in signature, provider options in `**kwargs`.
    - **Behavior Preservation**: Runtime behavior for supported structured output modes remains unchanged.
- **Status**: Completed.

## LangChain Legacy (LCL) Chains Review

- **Problem**: The codebase still exposed legacy LangChain patterns where possible:
  - `GigaChat.bind_functions()` docstring claimed it supported `"auto"` mode, but the implementation only supported forcing a specific function name and required exactly one function to be provided.
  - `bind_functions()` documentation referenced `langchain` (non-core) module paths, which is inaccurate for `langchain-core>=1`.
- **Note (why LCL was not removed from `with_structured_output`)**:
  - `GigaChat.with_structured_output()` is already implemented using modern tool-calling (`bind_tools()` + tool schema parsing) and does not rely on legacy chain abstractions.
  - The remaining legacy surface (`bind_functions()`) is kept as a compatibility shim for existing user code and for LangChain's historical API, but `with_structured_output()` intentionally stays on the tool-first path.
- **Solution**:
  - Updated `GigaChat.bind_functions()` to support:
    - `function_call="auto"` and `function_call="none"` (forwarded to the API as-is)
    - forcing by name among *multiple* provided functions (`{"name": "<function_name>"}`)
  - Updated the docstring to describe supported `function_call` values and removed the incorrect single-function restriction.
  - Added unit tests for `"auto"`/`"none"` and forcing a name among multiple functions.
- **Why**:
  - **Correctness**: Documentation now matches runtime behavior.
  - **Compatibility**: Keeps `bind_functions()` as a legacy entry point while aligning behavior with modern tool-calling semantics.
  - **Reduced legacy surface**: Avoids special-case constraints that only existed in the legacy path.
- **Verification**:
  - `uv run ruff check` — passed
  - `uv run mypy` — passed
  - `uv run pytest` — passed
- **Status**: Completed.

## Expose SDK Connection Settings

- **Problem**: The `gigachat` SDK supports several connection/retry settings (`max_retries`, `max_connections`, `retry_backoff_factor`, `retry_on_status_codes`) that were configurable only via `GIGACHAT_*` environment variables. Users of the LangChain wrapper had no way to set them programmatically without env vars, making them effectively invisible and undiscoverable.
  - Location: `langchain_gigachat/_client.py` — `_GigaChatClientMixin` did not include these fields.
  - Location: `langchain_gigachat/_client.py` — `_get_client_init_kwargs()` did not forward them to the SDK.
- **Solution**:
  - **Implementation Details**:
    - Added four new fields to `_GigaChatClientMixin`:
      - `max_retries: Optional[int] = None` — maximum number of retries for transient errors (SDK default: 0, disabled).
      - `max_connections: Optional[int] = None` — maximum simultaneous connections to the API.
      - `retry_backoff_factor: Optional[float] = None` — backoff factor for retry delays (SDK default: 0.5).
      - `retry_on_status_codes: Optional[Tuple[int, ...]] = None` — HTTP status codes that trigger a retry (SDK default: `(429, 500, 502, 503, 504)`).
    - All fields default to `None`, meaning the SDK's own defaults (from `Settings` / env vars) apply unchanged.
    - Updated `_get_client_init_kwargs()` to forward the four new fields to the SDK constructor.
    - Updated `GigaChat` class docstring in `chat_models/gigachat.py` to document the new parameters.
    - Added docstring warnings about retry stacking: when using LangChain's `.with_retry()`, keep `max_retries` at `None`/`0` to avoid multiplicative retry counts.
  - **Why**:
    - **Discoverability**: Connection settings are now visible in IDE autocompletion, docstrings, and `help()`.
    - **Programmatic Configuration**: Users can set retry/connection policies without env vars.
    - **Consistency**: All SDK constructor parameters are now exposed through the wrapper.
  - **Design Decision**: Fields default to `None` rather than the SDK defaults (e.g. `max_retries=0`). This ensures that env-var-based configuration (`GIGACHAT_MAX_RETRIES`, etc.) still works as a fallback — the SDK only applies its own defaults when `None` is passed.
- **Verification**:
  - `uv run ruff check` — passed
  - `uv run ruff format --check` — passed
  - `uv run mypy` — passed
  - `uv run pytest` — 51 passed, `_client.py` 100% coverage
- **Status**: Completed.

## x_headers Audit (2.12)

- **Goal**: Map all places where GigaChat API `x_headers` are set or consumed (`response_metadata`, `generation_info`, `message.id`) and decide on refactoring vs documentation.
- **Outcome**: Audit documented; refactor done — shared streaming helper centralizes usage and x_headers for stream path.

### Where x_headers are set

| Location | Source | Used for |
|----------|--------|----------|
| `_create_chat_result()` | `response.x_headers` (GigaChat `ChatCompletion`) | (1) `message.id = x_headers["x-request-id"]` for tracing; (2) `llm_output["x_headers"]` so run output exposes full headers (e.g. `x-request-id`, `x-session-id`, `x-client-id`). |
| `_stream()` / `_astream()` | Via `_build_stream_chunk(chunk, first_chunk)` | Chunk dict → (1) `chunk_m.id = x_headers["x-request-id"]` on each chunk; (2) `generation_info["x_headers"] = x_headers` on the **first** chunk only. Chunk `generation_info` is merged into message `response_metadata` by LangChain. |

### Where x_headers are consumed

- **Inside this package**: Not used for control flow; only passed through to the user.
- **By the user**:  
  - `message.id` / `chunk.id` — GigaChat request id, useful for support and tracing.  
  - `response_metadata["x_headers"]` (stream) or run `llm_output["x_headers"]` (non-stream) — full headers dict for debugging or session/request correlation.
- **Tests**: `test_gigachat_stream` and `test_gigachat_astream` expect the first chunk to have `response_metadata={"x_headers": {}}` (mock returns empty dict).

### Refactor (stream path)

- **`_build_stream_chunk(self, chunk: dict, first_chunk: bool)`** in `gigachat.py`: single place that builds message chunk, usage metadata, x_headers, and generation_info from a normalized stream chunk dict. `_stream()` and `_astream()` now normalize the raw chunk to a dict, call `_build_stream_chunk()`, then run callback and yield. Non-stream path (`_create_chat_result`) unchanged — it still sets `message.id` and `llm_output["x_headers"]` from the response object.

### Design notes

- Non-stream path does **not** put `x_headers` into `ChatGeneration.generation_info`; it only sets `message.id` and `llm_output["x_headers"]`. So non-stream `response_metadata` on the final message comes from `generation_info` (finish_reason, model_name) and does not include `x_headers` unless the framework merges `llm_output` elsewhere.
- Stream path puts `x_headers` in `generation_info` on the first chunk so that the accumulated message's `response_metadata` includes it; that logic now lives in `_build_stream_chunk()`.

### Status

- Completed: audit documented; shared helper `_build_stream_chunk()` added; `_stream` and `_astream` refactored to use it.

## Support reasoning_effort and reasoning_content (GigaChat reasoning models)

- **Goal**: Allow using GigaChat reasoning-capable models (e.g. GigaChat-2-Reasoning) from langchain-gigachat: send `reasoning_effort` in the request and receive `reasoning_content` in the assistant message.
- **Upstream**: The `gigachat` SDK supports `Chat(messages=..., model="GigaChat-2-Reasoning", reasoning_effort="medium")` and returns `reasoning_content` on the assistant message in `ChatCompletion.choices[].message`.
- **Solution**:
  - **Request**: Added optional `reasoning_effort: Optional[str] = None` to `_BaseGigaChat` (`base_gigachat.py`). When set, it is included in the payload in `_build_payload()` (`gigachat.py`) so the GigaChat API receives it.
  - **Response**: In `_convert_dict_to_message()`, for assistant messages, `reasoning_content` is read from the message (via `getattr(message, "reasoning_content", None)` for SDK compatibility) and stored in `AIMessage.additional_kwargs["reasoning_content"]`. The user can read it as `message.additional_kwargs.get("reasoning_content")`.
  - **Streaming**: In `_convert_delta_to_message_chunk()`, if the stream delta contains `reasoning_content`, it is passed into the chunk’s `additional_kwargs` so streamed responses also expose reasoning when the API sends it.
- **Usage**: `llm = GigaChat(model="GigaChat-2-Reasoning", reasoning_effort="medium")` then `msg = llm.invoke([HumanMessage(content="...")])`; reasoning text is in `msg.content` (if reflected in content) or in `msg.additional_kwargs.get("reasoning_content")`.
- **Status**: Implemented.

## File API Cleanup (2.17)

- **Problem**: `_BaseGigaChat.get_file` / `aget_file` delegated to SDK `get_image` / `aget_image` and returned file **content** (base64, `gm.Image`), while the name suggested generic "file" and the SDK already has `get_file` returning **metadata** (`gm.UploadedFile`). The API surface did not expose SDK endpoints for listing files (GET /files) or deleting a file (DELETE /files/{id}).
- **Solution**:
  - **Naming and semantics**: Aligned with the SDK. `get_file` / `aget_file` now return **file metadata** (`gm.UploadedFile`) via SDK `get_file` / `aget_file`. New methods `get_file_content` / `aget_file_content` return **file content** (base64, `gm.Image`) via SDK `get_image` / `aget_image`.
  - **New methods**: Added `list_files` / `alist_files` (SDK `get_files` / `aget_files`, GET /files) returning `gm.UploadedFiles`; `delete_file` / `adelete_file` (SDK `delete_file` / `adelete_file`, DELETE /files/{id}) returning `gm.DeletedFile`.
  - **Breaking change**: Code that used `llm.get_file(file_id)` to obtain content (e.g. `llm.get_file(id).content`) must switch to `llm.get_file_content(file_id).content`. Code that needs only metadata can use `llm.get_file(file_id)` (now returns `UploadedFile`).
- **Verification**: `uv run ruff check`, `uv run mypy`, `uv run pytest`. Updated `libs/gigachat/tmp/content-blocks-testing.ipynb` to use `get_file_content` where content is required.
- **Status**: Completed.

---

## Function/Tool Message Handling Fixes (2.14, partial)

- **Problem**: Three silent bugs in `_convert_message_to_dict()` caused incorrect API payloads without raising any error:
  1. **`FunctionMessage.name` never forwarded** — `gm.Messages.name` was left `None` even though
     `FunctionMessage.name` is a required field in LangChain and the GigaChat API documents `name`
     as *"Required if role is 'function'"*. Every `FunctionMessage` was sent to the API without a
     function name, which could cause the model to misidentify or ignore the function result.
  2. **`ToolMessage.name` never forwarded** — same issue as above for the modern LangChain 1.x
     message type. `ToolMessage.name` is optional (`Optional[str]`), but when set (which happens
     automatically in LangChain agents) it was silently discarded.
  3. **Multiple `tool_calls` silently dropped** — when an `AIMessage` contained more than one
     `tool_calls` entry (parallel tool calling), only `tool_calls[0]` was forwarded to the API and
     the rest were silently lost. The GigaChat API does not support parallel function calls in one
     turn, so the correct behaviour is an explicit error rather than silent data loss.
- **Solution**:
  - `FunctionMessage`: added `kwargs["name"] = message.name`. The field is always present and
    required on the LangChain side, so it is always forwarded.
  - `ToolMessage`: added `kwargs["name"] = message.name` guarded by `if message.name`. When the
    name is absent (the field is optional in LangChain), `gm.Messages.name` stays `None` and the
    API can return its own error — explicit API feedback is better than silent wrong behaviour.
  - `AIMessage` with multiple `tool_calls`: raises `ValueError` with a clear message explaining the
    API limitation and instructing the caller to use one tool call per turn.
- **Why explicit error over silent conversion for multiple tool_calls**:
  - Silently forwarding only `tool_calls[0]` means the model receives a function result for one
    tool but the conversation history contains no record of the other calls — the context becomes
    corrupted and later turns may behave unpredictably.
  - An immediate `ValueError` surfaces the problem at the point of conversion, making it easy to
    diagnose. This follows the same "fail loudly" principle used for `tool_choice='any'`.
- **Tests**:
  - Updated `test__convert_message_to_dict_function` — added `name="func"` to `Messages` expected
    value (was `None`, revealing the pre-existing bug).
  - Added `test__convert_message_to_dict_tool_message_with_name` — verifies `name` is forwarded.
  - Added `test__convert_message_to_dict_tool_message_without_name` — verifies `name` stays `None`
    when `ToolMessage.name` is not set.
- **Verification**:
  - `uv run ruff check` — passed
  - `uv run pytest` — 57 passed
- **Status**: Completed (partial 2.14 — remaining sub-items: content blocks compatibility,
  `create_agent`, middleware, SDK exception translation).

## FunctionCall vs ToolCall Migration Strategy

- **Problem**: During LangChain Core 1.x refactoring, we need to decide whether to replace
  `FunctionCall` with `ToolCall` immediately, or keep compatibility with the current GigaChat API
  contract.
- **Current State**:
  - Public LangChain-facing API is already tool-oriented:
    - `GigaChat.bind_tools(...)` is supported and covered by unit tests.
    - Incoming provider `function_call` payloads are mapped to `AIMessage.tool_calls` and
      `tool_call_chunks` for streaming.
    - README examples use `bind_tools(...)` and `msg.tool_calls`.
  - Provider/SDK-facing API remains function-oriented:
    - Request payload still uses `functions` + `function_call`.
    - `gigachat` SDK models expose `FunctionCall`/`function_call` and do not provide native
      `tool_calls` transport fields.
- **Analysis**:
  - This is an intentional adapter architecture: modern LangChain surface on top of a legacy
    provider protocol.
  - A forced end-to-end migration now would mostly add translation complexity without runtime
    benefit, because downstream API semantics are still `function_call`.
  - Existing guardrails already fail loudly for known provider limitations:
    - multiple `tool_calls` in one assistant message raise `ValueError`;
    - unsupported `tool_choice="any"` raises by default (or falls back to `auto` only with
      explicit opt-in).
- **Risks**:
  - **If we keep the bridge (current approach)**:
    - Moderate maintenance overhead for conversion logic.
    - Requires ongoing parity tests between `tool_calls` (LangChain) and `function_call`
      (provider payload).
  - **If we force immediate migration**:
    - High regression risk in payload compatibility with GigaChat API.
    - Potential breakage for users relying on legacy `bind_functions` behavior.
    - No real feature gain until upstream introduces native `tools/tool_choice/tool_calls`.
- **Decision / Solution**:
  - Keep the current dual-layer compatibility model for now:
    - **Canonical external API**: `bind_tools` + `AIMessage.tool_calls`.
    - **Internal transport**: map to provider `function_call` until upstream changes.
    - **Tool result role mapping**: keep `ToolMessage` mapped to provider `role="function"` while the transport layer remains function-oriented.
  - Treat `bind_functions` as legacy compatibility surface (maintained, but not preferred in docs).
  - Revisit migration only when `gigachat` SDK/API ships first-class `tool_calls` transport.
- **Why this is not urgent**:
  - User-facing LangChain ergonomics are already aligned with modern tool-calling patterns.
  - Immediate full migration would be mostly cosmetic at the transport layer and introduces
    unnecessary risk.
- **Status**: Accepted; no urgent migration required. Keep under periodic review and trigger
  transition when upstream API contract changes.

## SDK `FunctionParametersProperty` Schema Stripping (2.19)

- **Problem**: SDK model `FunctionParametersProperty` defines only `type`, `description`, `items`, `enum`, `properties`. Pydantic V2 silently drops all other JSON Schema fields (`additionalProperties`, nested `required`, `format`, `default`, etc.) during `Chat.model_validate()`, causing 422 API errors.
  - Issues: [#55](https://github.com/ai-forever/langchain-gigachat/issues/55), [#59](https://github.com/ai-forever/langchain-gigachat/issues/59).
- **Solution**: Add `model_config = ConfigDict(extra="allow")` to `FunctionParametersProperty` in the `gigachat` SDK.
- **Status**: Not started (upstream SDK fix required).

---

## Breaking Changes

Consolidated list for release notes. Each entry links to the section where it is fully described.

### Removed APIs

| What | Was | Now | Migration |
|------|-----|-----|-----------|
| `GigaChat(verbose=True)` | Logged requests/responses | Field removed | Use Python `logging` at `DEBUG` level; see [Remove `verbose` Parameter](#remove-verbose-parameter) |
| `llm.predict("text")` | Returned `str` | Method removed (LangChain 1.x) | `llm.invoke("text").content` |
| `await llm.apredict("text")` | Returned `str` | Method removed (LangChain 1.x) | `(await llm.ainvoke("text")).content` |
| `_BaseGigaChat(profanity=True)` | Deprecated alias | Field removed | Use `profanity_check=True` |
| `GigaChatEmbeddings(one_by_one_mode=True)` | Processed texts one-by-one | Field removed | API handles batching natively; no replacement needed |
| `GigaChatEmbeddings(_debug_delay=...)` | Debug delay between requests | Field removed | No replacement |
| `with_structured_output(method="format_instructions")` | Prompt-injection structured output | `ValueError` | Use `method="function_calling"` (preferred) or `method="json_mode"`; see [Remove `format_instructions`](#remove-format_instructions-structured-output-mode) |
| `langchain_gigachat.output_parsers.gigachat_functions` | Legacy output parsers module | Module deleted | Use `PydanticToolsParser` / `JsonOutputKeyToolsParser` from `langchain_core` |
| `langchain_gigachat.tools.load_prompt` | Legacy prompt-loading module | Module deleted | No replacement; was not part of public API |
| `GigaChat(auto_upload_images=True)` | Auto-upload images only | Field removed | Use `auto_upload_attachments=True` (covers images, audio, documents) |

### Changed Behaviour

| What | Before | After | Details |
|------|--------|-------|---------|
| `tool_choice="any"` in `bind_tools()` | Silently converted to `"auto"` | Raises `ValueError` | Set `allow_any_tool_choice_fallback=True` to restore old behaviour with a warning; see [LangChain Core 1.x Support](#langchain-core-1x-support-branch-lc1-support) |
| Multiple `tool_calls` in `AIMessage` | First call forwarded, rest silently dropped | Raises `ValueError` | Use one tool call per turn; GigaChat API does not support parallel function calls |
| `get_file(file_id)` return type | `gm.Image` (file content, base64) | `gm.UploadedFile` (metadata) | Use `get_file_content(file_id)` to download content; see [File API Cleanup](#file-api-cleanup-217) |
| `gigachat` dependency | `^0.1.41.post1` | `>=0.2.0,<0.3` | Upstream breaking change; Pydantic V2 required |
| `langchain-core` dependency | `>=0.3,<1` | `>=1,<2` | LangChain 1.x is now required |
| Python version | `>=3.9` | `>=3.10` | LangChain 1.x minimum |

### Known Regressions (to fix before release)

*None at present.*

---

## Pre-release Checklist

Steps required before merging `lc1-support` branch and publishing to PyPI.

- [ ] **Rename package**: `libs/gigachat/pyproject.toml` — change `name = "langchain-gigachat-lc1"` back to `name = "langchain-gigachat"`. Also update the self-dependency in `[dependency-groups] dev`.
- [ ] **Bump version**: update `version` in `libs/gigachat/pyproject.toml` to the next release (e.g. `0.4.0`). Remove the `b4` pre-release suffix.
- [ ] **Coordinate with `gigachat` release**: ensure `gigachat>=0.2.0,<0.3` is published on PyPI and the git-URL dependency (if any) is replaced with the PyPI constraint.
- [ ] **Run full verification**: `uv run ruff check . && uv run ruff format --check . && uv run mypy langchain_gigachat && uv run pytest`.
- [ ] **Update CHANGELOG / release notes** if maintained.
