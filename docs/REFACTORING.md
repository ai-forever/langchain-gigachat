# Refactoring Notes

**Note**: All information in this file must be grouped by specific issues. Do not separate problems and solutions into different sections; keep them together under the relevant issue heading.

**Context**: The upstream `gigachat` package (v0.2.0) has undergone significant refactoring including Pydantic V2 migration, removal of the `verbose` parameter, improved exception hierarchy, and toolchain migration from Poetry to uv. This refactoring effort aligns `langchain-gigachat` with those changes.

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
     - `payload.dict(exclude_none=True, by_alias=True)` in `_build_payload()` → should be `payload.model_dump(exclude_none=True, by_alias=True)`
     - `response.usage.dict()` in `_create_chat_result()` → should be `response.usage.model_dump()`
     - `chunk_d.dict()` in `_stream()` and `_astream()` → should be `chunk_d.model_dump()`
  3. **Old `Config` Class** (V1 style):
     - `langchain_gigachat/chat_models/base_gigachat.py`: `class Config: arbitrary_types_allowed = True`
     - `langchain_gigachat/embeddings/gigachat.py`: Same pattern
     - Should use `model_config = ConfigDict(arbitrary_types_allowed=True)`
  4. **Removed `verbose` Parameter**:
     - `langchain_gigachat/chat_models/base_gigachat.py`: `verbose: bool = False` field in `_BaseGigaChat`
     - `langchain_gigachat/chat_models/base_gigachat.py`: `verbose=self.verbose` in `_client` property
     - `langchain_gigachat/chat_models/gigachat.py`: `if self.verbose:` conditional in `_build_payload()`
     - The `verbose` parameter was removed from `gigachat` v0.2.0 as it was unused.
  5. **Outdated Dependency Version**:
     - `libs/gigachat/pyproject.toml`: `gigachat = "^0.1.41.post1"` needs update to `^0.2.0`
- **Analysis**:
  - LangChain Core v0.3+ is Pydantic V2-native.
  - The `gigachat` package now requires `pydantic >= 2.0`.
  - Mixing V1 and V2 patterns causes type-checking issues and runtime surprises.
  - The `pydantic.v1` imports in `function_calling.py` are particularly problematic as they create V1 models that may be incompatible with V2 validation.
  - When upgrading to `gigachat >= 0.2.0`, the `verbose` parameter will cause a `TypeError`.
- **Solution**:
  - **Implementation Details**:
    - Replace all `pydantic.v1` imports with native `pydantic` imports.
    - Update `Field_v1` → `Field` and `create_model_v1` → `create_model`.
    - Replace `.dict()` → `.model_dump()` throughout.
    - Replace `.parse_obj()` → `.model_validate()` throughout.
    - Replace `class Config:` with `model_config = ConfigDict(...)`.
    - Remove `verbose` field and all its usages.
    - Update `gigachat` dependency version to `^0.2.0`.
    - Test thoroughly to ensure schema generation works correctly.
  - **Why**:
    - **Compatibility**: Required for compatibility with `gigachat >= 0.2.0`.
    - **Performance**: Pydantic V2 is 5-50x faster.
    - **Ecosystem**: Aligns with LangChain Core v0.3+.
- **Status**: Pending.
