# Refactoring Notes

**Note**: All information in this file must be grouped by specific issues. Do not separate problems and solutions into different sections; keep them together under the relevant issue heading.

**Context**: The upstream `gigachat` package (v0.2.0) has undergone significant refactoring including Pydantic V2 migration, removal of the `verbose` parameter, improved exception hierarchy, and toolchain migration from Poetry to uv. This refactoring effort aligns `langchain-gigachat` with those changes.

## Dependency Management Strategy

- **Problem**: During simultaneous refactoring of `gigachat` and `langchain-gigachat`, the dependency on `gigachat` must be managed carefully:
  1. **Local path**: `gigachat = { path = "..." }` breaks CI (path doesn't exist on CI runners).
  2. **PyPI**: Can't publish to PyPI until both packages are ready for simultaneous release.
  3. **Circular dependency**: Changes in one package may require changes in the other before either is releasable.
- **Solution**: Install `gigachat` from git branch using Poetry's git dependency syntax:
  ```toml
  gigachat = { git = "https://github.com/ai-forever/gigachat.git", branch = "<branch-name>" }
  ```
  See `libs/gigachat/pyproject.toml` for the current branch in use.
- **Why**:
  - **CI works**: Git URLs resolve on any machine with network access.
  - **Version pinning**: The branch tracks in-progress refactoring changes.
  - **Coordinated release**: Both packages can be developed in sync, then released simultaneously when ready.
- **When to change**: Update to PyPI version (e.g., `gigachat = "^X.Y.Z"`) after coordinated release of both packages.
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
