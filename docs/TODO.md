# Refactoring Progress

## General
- [x] Initial setup
  - [x] Create `AGENTS.md` documentation
  - [x] Create `docs/REFACTORING.md` documentation
  - [x] Create `docs/TODO.md` documentation

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
  - [x] Add checklist to `docs/TODO.md`
  - [x] Delete `docs/V1_MIGRATION.md`
