# AGENTS.md

## Project Overview
LangChain integration package for GigaChat LLM.

### What This Library Does
- Wraps `gigachat` SDK with LangChain-compatible interfaces
- `GigaChat` — Chat model (subclass of `BaseChatModel`) with tool/function calling support
- `GigaChatEmbeddings` — Embeddings model

### Key Modules
- `chat_models/` — `GigaChat` class with `_BaseGigaChat` common logic
- `embeddings/` — `GigaChatEmbeddings` class
- `utils/function_calling.py` — LangChain tool → GigaChat function conversion
- `MIGRATION.md` / `CHANGELOG.md` — public upgrade and release notes

## Dependencies
- **gigachat**: The underlying GigaChat API client library (Pydantic V2, v0.2.0+)
- **langchain-core**: Core LangChain abstractions (v0.3+, Pydantic V2-native)

## Setup Commands
- Install dependencies: `uv sync`
- Run tests: `uv run pytest`
- Lint code: `uv run ruff check .`
- Format code: `uv run ruff format .`
- Type check: `uv run mypy langchain_gigachat`
- Verify all: `uv run ruff check . && uv run ruff format --check . && uv run mypy langchain_gigachat && uv run pytest`

## Code Style
- Documentation: English, Google Python Style Guide, imperative mood ("Return..." not "Returns...")
- Avoid unnecessary comments

## Key Considerations
- **Upstream Dependency**: The `gigachat` package (v0.2.0+) was significantly refactored. Changes here must align with the SDK behavior and schema expectations.
- **LangChain Compatibility**: Must remain compatible with `langchain-core >=1,<2`.
- **Release Context**: This branch contains intentional breaking changes for the `0.5.x` alpha line. Keep `libs/gigachat/MIGRATION.md` and `libs/gigachat/CHANGELOG.md` in sync with behavior changes.
