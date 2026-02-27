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
- `output_parsers/` — Output parsers for function results

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

## Refactoring
See `docs/REFACTORING.md` for detailed analysis and solutions.
Update these docs after solving each issue.

## Key Considerations
- **Upstream Dependency**: The `gigachat` package (v0.2.0+) was significantly refactored. See `gigachat/docs/REFACTORING.md` for details. Changes here must align with upstream.
- **LangChain Compatibility**: Must remain compatible with `langchain-core ^0.3` (Pydantic V2-native).
- **Git Branch Dependency**: During refactoring, `gigachat` is installed from git branch (not local path or PyPI). See `docs/REFACTORING.md` → "Dependency Management Strategy" for rationale.
