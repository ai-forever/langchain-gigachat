# Refactoring Progress

## General
- [x] Initial setup
  - [x] Create `AGENTS.md` documentation
  - [x] Create `docs/REFACTORING.md` documentation
  - [x] Create `docs/TODO.md` documentation

## Remove `verbose` Parameter
- [x] Remove `verbose: bool = False` field from `_BaseGigaChat` in `langchain_gigachat/chat_models/base_gigachat.py`
- [x] Remove `verbose=self.verbose` from `_client` property in `langchain_gigachat/chat_models/base_gigachat.py`
- [x] Remove `if self.verbose:` conditional in `_build_payload` method in `langchain_gigachat/chat_models/gigachat.py`

## Pydantic V2 Migration
- [ ] Migrate `function_calling.py` from Pydantic V1 to V2
  - [ ] Replace `from pydantic.v1 import BaseModel` with native import
  - [ ] Replace `from pydantic.v1 import Field as Field_v1` with `from pydantic import Field`
  - [ ] Replace `from pydantic.v1 import create_model as create_model_v1` with `from pydantic import create_model`
  - [ ] Update `_convert_typed_dict_to_gigachat_function` to use V2 patterns
  - [ ] Update `_convert_any_typed_dicts_to_pydantic` to use V2 patterns
- [ ] Migrate `gigachat.py` deprecated methods
  - [ ] Replace `Chat.parse_obj(payload_dict)` with `Chat.model_validate(payload_dict)`
  - [ ] Replace `payload.dict(exclude_none=True, by_alias=True)` with `payload.model_dump(exclude_none=True, by_alias=True)`
  - [ ] Replace `response.usage.dict()` with `response.usage.model_dump()`
  - [ ] Replace `chunk_d.dict()` with `chunk_d.model_dump()` (2 occurrences)
- [ ] Migrate `base_gigachat.py` Config class
  - [ ] Replace `class Config: arbitrary_types_allowed = True` with `model_config = ConfigDict(arbitrary_types_allowed=True)`
  - [ ] Add `from pydantic import ConfigDict` import
- [ ] Migrate `embeddings/gigachat.py` Config class
  - [ ] Replace `class Config: arbitrary_types_allowed = True` with `model_config = ConfigDict(arbitrary_types_allowed=True)`
  - [ ] Add `from pydantic import ConfigDict` import
- [ ] Update dependency version
  - [ ] Change `gigachat = "^0.1.41.post1"` to `gigachat = "^0.2.0"` in `libs/gigachat/pyproject.toml`
  - [ ] Run `poetry lock` to update lockfile
- [ ] Verification
  - [ ] Run `ruff check` to verify no linting errors
  - [ ] Run `mypy` to verify type checking passes
  - [ ] Run `pytest` to verify no test regressions
