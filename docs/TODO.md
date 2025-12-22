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
  - [x] Update test matrix: Python 3.9-3.13 + 3.14 (experimental)
  - [x] Delete `.github/actions/poetry_setup/` custom action
- [x] Lock file transition
  - [x] Delete `poetry.lock`
  - [x] Generate `uv.lock` with `uv lock`
- [x] Update documentation
  - [x] Update `AGENTS.md`: change setup/run commands to uv
  - [x] Update `docs/TODO.md` and `docs/REFACTORING.md`
- [x] Verification
  - [x] Run `uv sync` to install dependencies
  - [x] Run `uv run ruff check` — passed
  - [x] Run `uv run mypy` — passed (13 source files)
  - [x] Run `uv run pytest` — 73 passed, 2 xpassed
- [x] Post-migration cleanup
  - [x] Delete empty `.github/actions/` directory
  - [x] Delete unused `.github/scripts/get_min_versions.py` (broken, references Poetry)
