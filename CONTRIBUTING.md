# Contributing to langchain-gigachat

Thank you for your interest in contributing to the LangChain integration for GigaChat.

We welcome bug reports, feature requests, documentation improvements, and code contributions. This guide explains how to get started and how we work.

## Table of Contents

- [Types of Contributions](#types-of-contributions)
- [Development Setup](#development-setup)
- [Code Quality and Testing](#code-quality-and-testing)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [CI](#ci)

## Types of Contributions

- **Bug reports** — Help us find and fix issues.
- **Feature requests** — Suggest new features or improvements.
- **Documentation** — Improve docs, docstrings, or examples.
- **Tests** — Add or improve test coverage.
- **Code** — Fix bugs, implement features, or refactor.

## Development Setup

This repository is a monorepo. The LangChain–GigaChat package lives under **`libs/gigachat`**. All commands below must be run from that directory unless stated otherwise.

### Prerequisites

- **Python** 3.10+ (supported: 3.10–3.13; CI also runs 3.14 as experimental).
- **uv** — [installation](https://docs.astral.sh/uv/getting-started/installation/).
- **Git**.

### Setup

1. **Clone the repository** (or your fork).

2. **Go to the package directory and install dependencies:**

   ```bash
   cd libs/gigachat
   uv sync
   ```

3. **Optional: development environment** (pre-commit hooks):

   ```bash
   make dev
   ```

4. **Verify:**

   ```bash
   make format
   make lint_package
   make lint_tests
   make test
   ```

### Quick reference (from `libs/gigachat`)

| Command | Description |
|--------|-------------|
| `make help` | List make targets |
| `make format` | Format code with Ruff |
| `make lint_package` | Lint and type-check `langchain_gigachat` |
| `make lint_tests` | Lint and type-check `tests` |
| `make test` | Run unit tests |
| `make dev` | Install pre-commit hooks (also updates `origin` HEAD via `git remote set-head origin -a`) |

## Code Quality and Testing

- **Style**: PEP 8, Ruff for formatting and linting, line length as in `pyproject.toml`.
- **Types**: mypy with project settings from `libs/gigachat/pyproject.toml` (notably `disallow_untyped_defs = true`).
- **Docstrings**: Google style, **imperative mood** (e.g. “Return the result” not “Returns the result”), English only. See `AGENTS.md` for project conventions.
- **Comments**: Prefer clear names over comments; avoid unnecessary or commented-out code.

Before submitting a PR, run from `libs/gigachat`:

```bash
make format
make lint_package
make lint_tests
make test
```

## Commit Messages

Use **conventional commits** when possible:

- **feat**: New feature  
- **fix**: Bug fix  
- **docs**: Documentation  
- **style**: Formatting, no logic change  
- **refactor**: Refactor, no functional change  
- **test**: Tests  
- **chore**: Maintenance, deps, config  

Subject: imperative, short. Example: `feat: add timeout to chat invocations`.

## Pull Request Process

1. Create a branch from the default branch (e.g. `feature/your-feature` or `fix/your-fix`).
2. Make your changes, add or update tests, and update docs if needed.
3. Run `make format`, `make lint_package`, `make lint_tests`, and `make test` from `libs/gigachat`.
4. Push and open a PR. Fill out the PR template and link related issues (e.g. “Closes #123”).
5. Address review feedback. CI must pass before merge.

## Issue Reporting

- **Bugs**: Use the [Bug report](/.github/ISSUE_TEMPLATE/bug_report.yml) template. Include steps to reproduce, expected vs actual behaviour, versions (langchain-gigachat, Python, OS), and a minimal code example if possible.
- **Features**: Use the [Feature request](/.github/ISSUE_TEMPLATE/feature_request.yml) template. Describe the problem, proposed solution, and optionally API ideas.
- Search existing issues first to avoid duplicates.

## CI

- **When it runs**: On push and pull requests to the configured branches (e.g. `master`, `dev`).
- **What it does**: Changed files are used to decide which libs to run. For each affected lib (e.g. `libs/gigachat`):
  - **Lint**: `make lint_package` and `make lint_tests` (Python 3.10 and 3.12).
  - **Test**: `make test` on Python 3.10, 3.11, 3.12, 3.13. CI also runs Python 3.14 as experimental (allowed to fail).
- **Locally**: Run the same commands from `libs/gigachat` before opening or updating a PR so CI stays green.

For more detail, see `.github/workflows/check_diffs.yml`, `.github/workflows/_lint.yml`, and `.github/workflows/_test.yml`.
