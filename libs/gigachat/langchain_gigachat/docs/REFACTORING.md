# LangChain Core v1.0 Migration (lc1-support branch)

This document tracks the refactoring work for LangChain Core v1.0 support.

## Summary

The `lc1-support` branch updates `langchain-gigachat` to support `langchain-core>=1,<2`. This is a significant upgrade since LangChain Core v1.0 contains breaking changes from v0.3.

## What Was Done

### 1. Dependency Updates (`pyproject.toml`)

- Package renamed to `langchain-gigachat-lc1` (temporary name for beta testing)
- Version bumped to `0.4.0b4`
- Minimum Python version raised from `3.9` to `3.10` (required by langchain-core v1)
- `langchain-core` dependency updated from `>=0.3,<1` to `>=1,<2`

### 2. Removed Deprecated `validate_environment` Methods

**Files**: `base_gigachat.py`, `embeddings/gigachat.py`

- Removed `@pre_init` decorated `validate_environment` methods
- These methods performed:
  - Import validation (checking if `gigachat` package is installed)
  - Extra fields warning (logging unknown kwargs)
  - Deprecated `profanity` → `profanity_check` migration
- **Reason**: LangChain Core v1.0 removed support for `@pre_init` decorator. Import validation is unnecessary since missing dependencies will fail at import time anyway.

### 3. Removed Deprecated Fields

**File**: `base_gigachat.py`
- Removed `profanity: bool = True` (deprecated, superseded by `profanity_check`)

**File**: `embeddings/gigachat.py`
- Removed `one_by_one_mode: bool = False` (deprecated, unused)
- Removed `_debug_delay: float = 0` (deprecated, unused)

### 4. Simplified Imports

**Files**: `base_gigachat.py`, `embeddings/gigachat.py`

- Removed `TYPE_CHECKING` conditional imports for `gigachat` module
- Changed to direct imports: `import gigachat` and `import gigachat.models as gm`
- **Reason**: Cleaner code, no lazy-loading needed since gigachat is a required dependency

### 5. Added `tool_choice="any"` Workaround

**File**: `gigachat.py` (method `bind_tools`)

- GigaChat API doesn't support `tool_choice="any"`
- Added automatic conversion of `"any"` → `"auto"` with a warning
- **Reason**: LangChain agents use `tool_choice="any"` to force tool calls. Without this workaround, agents would crash. Using `"auto"` is the closest alternative, though the model may choose not to call any tool.

### 6. Improved Type Annotations

**File**: `gigachat.py`

- Changed return type of `bind_functions()` and `bind_tools()` from `Runnable[..., BaseMessage]` to `Runnable[..., AIMessage]`
- **Reason**: More accurate typing since these methods always return AI messages

### 7. Removed Unused Code

**File**: `gigachat.py`
- Removed unused `_FunctionCall` TypedDict
- Added compiled `BASE64_DATA_REGEX` constant (replaced inline regex)
- Removed redundant `isinstance(x_headers, dict)` check

**File**: `function_calling.py`
- Removed trivial `_get_python_function_name()` wrapper function

**File**: `output_parsers/gigachat_functions.py`
- **Entire file deleted** (112 lines)
- Classes `OutputFunctionsParser`, `PydanticOutputFunctionsParser`, `PydanticAttrOutputFunctionsParser` removed
- **Reason**: Superseded by output parsers in `langchain-core`

### 8. Added Public Module Exports

**File**: `tools/__init__.py`
- Added exports: `FewShotExamples`, `GigaBaseTool`, `GigaStructuredTool`, `GigaTool`, `giga_tool`

**File**: `utils/__init__.py`
- Added exports: `convert_to_gigachat_function`, `convert_to_gigachat_tool`

### 9. Test Updates

**File**: `test_gigachat.py`

- Removed tests for deprecated methods: `test_gigachat_predict`, `test_gigachat_predict_stream`, `test_gigachat_apredict`, `test_gigachat_apredict_stream`
- **Reason**: `predict()` and `apredict()` methods were removed in LangChain Core v1.0 (use `invoke()` and `ainvoke()` instead)
- Updated stream test expectations to include final empty "last" chunk with `chunk_position="last"`
- Removed unused `FakeCallbackHandler`/`FakeAsyncCallbackHandler` imports

### 10. Minor Fixes

- Fixed key name in `_identifying_params`: `"profanity"` → `"profanity_check"`
- Removed unnecessary `type: ignore` comment in `get_num_tokens()`
- Simplified `validate_environment` in `GigaChat` class (removed parent call)

## What Still Needs To Be Done

### High Priority

1. **Rename package back to `langchain-gigachat`** after beta testing is complete
2. **Update README** with LangChain Core v1.0 migration notes
3. **Integration testing** with real GigaChat API
4. **CI/CD updates** for Python 3.10+ only test matrix

### Medium Priority

1. **Review `tool_choice="any"` workaround** — investigate if GigaChat API will add native support
2. **Review output parsers removal** — ensure users have migration path to langchain-core parsers
3. **Add deprecation warnings** for removed parameters in previous version (for smooth upgrade path)

### Low Priority

1. **Performance testing** — ensure no regressions after removing lazy imports
2. **Documentation updates** — update examples and docstrings for LC1 API changes

## Breaking Changes for Users

1. **Python 3.9 no longer supported** — minimum is now Python 3.10
2. **`profanity` parameter removed** — use `profanity_check` instead
3. **`one_by_one_mode` and `_debug_delay` parameters removed** from `GigaChatEmbeddings`
4. **Output parsers removed** — use `langchain_core.output_parsers` instead of `langchain_gigachat.output_parsers`
5. **`predict()` / `apredict()` methods removed** — use `invoke()` / `ainvoke()` instead

## Commits in This Branch

1. `52567cf` — LangChain v1.0 support (main migration)
2. `2d271e3` — 'any' function call type workaround
3. `23cff2d` — chore: remove unused type ignore and meta data
4. `40547d2` — fix: remove unused old checks
5. `4e89499` — chore: remove unused parts of code
