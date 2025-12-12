# v1.0 Migration Plan

**Purpose**: Track breaking changes and improvements deferred to the v1.0 release when backward compatibility breaks are allowed.

**Note**: All information in this file must be grouped by specific issues. Do not separate problems and solutions into different sections; keep them together under the relevant issue heading.

## Context

The `langchain-gigachat` package follows semantic versioning. Breaking changes to the public API require a major version bump (v1.0). This document tracks:
- Breaking changes that were analyzed but reverted to maintain backward compatibility
- API improvements that require breaking existing user code
- Cleanup tasks that remove deprecated functionality

These changes will be implemented when:
1. The `langchain-gigachat` package is ready for v1.0 release
2. Coordinated with upstream `gigachat` v1.0 release
3. LangChain/LangGraph ecosystem upgrades to v1.0+

**Upstream Dependency**: Changes here must align with `gigachat` package. See `gigachat/docs/V1_MIGRATION.md` for upstream breaking changes.

## Workflow

### Progress Tracking
- Tasks are grouped by issue.
- Only analyzed and approved issues are added to this plan.
- **Chronological Order**: All sections (issues) must be listed in chronological ascending order (oldest first). New tasks should always be added at the end.

### Implementation Process
1. When v1.0 development begins, create corresponding entries in `docs/TODO.md`.
2. Before implementing each breaking change, get approval.
3. After implementation, summarize results.
4. After solving each issue:
   - Update this file with final implementation details.
   - Update `docs/TODO.md` to reflect implemented steps.

---

## Planned Breaking Changes

### Remove `verbose` Parameter
- **Problem**: The `verbose` parameter in `langchain-gigachat` provides request/response logging functionality, but this is inconsistent with standard LangChain patterns and the upstream `gigachat` SDK where the parameter is unused.
  - Location: `langchain_gigachat/chat_models/base_gigachat.py`: `verbose: bool = False` field in `_BaseGigaChat`
  - Location: `langchain_gigachat/chat_models/base_gigachat.py`: `verbose=self.verbose` passed to `_client` property
  - Location: `langchain_gigachat/chat_models/gigachat.py`: Request logging in `_build_payload()`, response logging in `_create_chat_result()`
- **Current Functionality (v0.x)**:
  - When `verbose=True`, logs request payload: `"Giga request: {...}"`
  - When `verbose=True`, logs response content: `"Giga response: ..."`
  - Parameter is passed through to upstream `gigachat.GigaChat()` client (where it is ignored).
- **Analysis**:
  - The logging functionality is useful for debugging but implemented inconsistently.
  - Standard approach is to use Python `logging` module with configurable log levels.
  - Removing the parameter is a breaking change — existing user code with `GigaChat(verbose=True)` will raise `TypeError`.
- **Current State (v0.x)**:
  - Parameter is **restored and marked as deprecated** in docstrings: "Deprecated: will be removed in v1.0."
  - Request/response logging functionality is preserved for backward compatibility.
- **Solution (Remove in v1.0)**:
  - **Approach**: Remove the `verbose` parameter and migrate to standard logging, because:
    1. Users can configure logging via standard Python `logging` module.
    2. Aligns with LangChain patterns and upstream `gigachat` SDK.
  - **Implementation**:
    - Remove `verbose: bool = False` field from `_BaseGigaChat`.
    - Remove `verbose=self.verbose` from `_client` property.
    - Remove `if self.verbose:` logging blocks from `_build_payload()` and `_create_chat_result()`.
    - Optionally: migrate to DEBUG-level logging that users can enable via `logging.getLogger("langchain_gigachat").setLevel(logging.DEBUG)`.
  - **Why**:
    - **Consistency**: Aligns with standard LangChain and Python logging patterns.
    - **Simplicity**: Single logging configuration mechanism instead of custom parameter.
- **Upstream Dependency**: Requires `gigachat` v1.0 to also remove the `verbose` parameter. See `gigachat/docs/V1_MIGRATION.md`.
- **Status**: Deprecated in v0.x. Planned removal in v1.0.

