# Contributing to langchain-gigachat

Thank you for your interest in contributing to the LangChain integration for GigaChat.

We welcome bug reports, feature requests, documentation improvements, and code contributions. This guide explains how to get started and how we work.

## Table of Contents

- [Types of Contributions](#types-of-contributions)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Quality Standards](#code-quality-standards)
- [Testing](#testing)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [License](#license)

## Types of Contributions

- **Bug reports** — Help us find and fix issues.
- **Feature requests** — Suggest new features or improvements.
- **Documentation** — Improve docs, docstrings, or examples.
- **Tests** — Add or improve test coverage.
- **Code** — Fix bugs, implement features, or refactor.

Whether you're fixing a typo or implementing a major feature, all contributions are valued.

## Getting Started

### Prerequisites

- **Python**: 3.10+ (supported: 3.10–3.13; CI also runs 3.14 as experimental).
- **uv**: Modern Python package manager ([installation guide](https://docs.astral.sh/uv/getting-started/installation/)).
- **Git**.

### Repository Structure

This repository is a monorepo. The LangChain–GigaChat package lives under **`libs/gigachat`**. All commands below must be run from that directory unless stated otherwise.

### Development Setup

1. **Fork and clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/langchain-gigachat.git
cd langchain-gigachat
```

2. **Go to the package directory and install dependencies**

```bash
cd libs/gigachat
uv sync
```

3. **Set up pre-commit hooks** (optional but recommended)

```bash
make dev
```

4. **Set up environment variables** (for integration testing)

Create a `.env` file in `libs/gigachat`:

```bash
GIGACHAT_CREDENTIALS=your_oauth_credentials_here
GIGACHAT_SCOPE=GIGACHAT_API_PERS
```

> **Note**: Integration tests are optional. See the [Testing](#testing) section for details.

5. **Configure SSL certificates**

The GigaChat API uses certificates issued by the Russian Ministry of Digital Development. For development:

- **Production setup**: Download the "Russian Trusted Root CA" certificate from [Gosuslugi](https://www.gosuslugi.ru/crt) and configure:
  ```bash
  export GIGACHAT_CA_BUNDLE_FILE="/path/to/Russian_Trusted_Root_CA.crt"
  ```

- **Development only** (not recommended): Disable SSL verification:
  ```bash
  export GIGACHAT_VERIFY_SSL_CERTS=false
  ```

6. **Verify installation**

```bash
make format
make lint_package
make lint_tests
make test
```

### Quick Reference (from `libs/gigachat`)

| Command | Description |
|---------|-------------|
| `make help` | List make targets |
| `make format` | Format code with Ruff |
| `make lint_package` | Lint and type-check `langchain_gigachat` |
| `make lint_tests` | Lint and type-check `tests` |
| `make test` | Run unit tests |
| `make dev` | Install pre-commit hooks |

## Development Workflow

### Branching Strategy

We follow a fork-and-pull-request model:

1. **Fork the repository** to your GitHub account
2. **Clone your fork** locally
3. **Create a feature branch** from `master`:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

### Branch Naming Conventions

Use descriptive branch names with prefixes:

- `feature/` — New features (e.g., `feature/add-streaming-timeout`)
- `fix/` — Bug fixes (e.g., `fix/handle-empty-response`)
- `docs/` — Documentation updates (e.g., `docs/update-auth-guide`)
- `refactor/` — Code refactoring (e.g., `refactor/simplify-tool-conversion`)
- `test/` — Test improvements (e.g., `test/add-embeddings-coverage`)

### Making Changes

1. **Make your changes** in your feature branch
2. **Follow code quality standards** (see [Code Quality Standards](#code-quality-standards))
3. **Add or update tests** to cover your changes
4. **Update documentation** if needed (docstrings, README, examples)
5. **Run quality checks** before committing:
   ```bash
   make format
   make lint_package
   make lint_tests
   make test
   ```

6. **Commit your changes** with clear commit messages (see [Commit Message Guidelines](#commit-message-guidelines))

### Pre-commit Hooks

Pre-commit hooks run automatically on `git commit` to ensure code quality:

- **Ruff**: Formatting and linting
- **mypy**: Type checking
- **Standard checks**: Trailing whitespace, YAML validation, etc.

If pre-commit fails, fix the issues and commit again. To manually run pre-commit on all files:

```bash
pre-commit run --all-files
```

### Keeping Your Fork Updated

Regularly sync your fork with the upstream repository:

```bash
# Add upstream remote (one-time setup)
git remote add upstream https://github.com/ai-forever/langchain-gigachat.git

# Fetch and merge upstream changes
git fetch upstream
git checkout master
git merge upstream/master

# Push to your fork
git push origin master
```

### Submitting Your Changes

1. **Push your branch** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open a pull request** on GitHub from your fork to `ai-forever/langchain-gigachat:master`

3. **Fill out the PR template** with all required information

4. **Wait for review** — maintainers will review your PR and may request changes

5. **Address feedback** by making additional commits to your branch

> **Note**: For detailed guidance, see the [Pull Request Process](#pull-request-process) section below.

## Code Quality Standards

### Code Style

- **PEP 8 Compliance**: Follow [PEP 8](https://pep.python.org/pep-0008/) style guidelines
- **Formatting**: Use Ruff for automatic formatting (`make format`)
- **Linting**: Pass all Ruff linting checks (`make lint_package`)
- **Line Length**: As configured in `pyproject.toml`

### Type Hints

- **Required**: All functions and methods must have type hints (`disallow_untyped_defs = true`)
- **Return Types**: Always specify return types, including `None`
- **mypy**: Runs as part of `make lint_package` / `make lint_tests`

### Docstrings

All public modules, classes, functions, and methods must have docstrings.

- **Style**: Google Python Style Guide
- **Mood**: Imperative (e.g., "Return the result" not "Returns the result")
- **Language**: English only
- **Type Information**: In function signatures, not docstrings (we use type hints)

Most functions use minimal, one-line docstrings:

```python
def convert_to_gigachat_function(
    tool: BaseTool,
) -> Function:
    """Convert a LangChain tool to a GigaChat function definition."""
    # Implementation here
```

For complex public APIs, you may optionally add an `Args:` section, but type hints in signatures usually make it unnecessary.

### Comments

- **Minimize comments** — write self-documenting code with clear names
- **Comment "why" not "what"** — the code shows what it does
- **No commented-out code** — delete it instead (Git preserves history)
- **TODO**: Use `# TODO:` for temporary notes, but prefer issues

Example:

```python
# Bad: explains what (obvious from code)
# Loop through messages and process them
for msg in messages:
    process(msg)

# Good: explains why (not obvious)
# Force refresh token before batch operation to avoid mid-batch auth failures
self._refresh_token()
for msg in messages:
    process(msg)
```

### Code Organization

- **Imports**: Organized by Ruff (stdlib, third-party, local)
- **Module Structure**: Follow existing patterns in the codebase
- **Single Responsibility**: Keep functions and classes focused
- **DRY Principle**: Extract common logic, don't repeat yourself

### Dependencies

- **Minimal Dependencies**: Only add dependencies when absolutely necessary
- **Justification Required**: Explain why a new dependency is needed
- **Version Constraints**: Use compatible ranges (e.g., `>=2.0,<3`)
- **Update Lockfile**: Run `uv lock` after changing dependencies

## Testing

### Test Types

**Unit tests** (`tests/unit_tests/`): Verify internal logic using mocked responses.

- Test logic, error handling, and edge cases
- Fast execution (entire suite runs in seconds)
- Aim for high coverage on new code

**Integration tests** (`tests/integration_tests/`): Verify real API interactions.

- Require credentials (set `GIGACHAT_CREDENTIALS` in `.env`)
- Optional for contributors

### Running Tests

```bash
# Run unit tests (default)
make test

# Run integration tests (requires credentials)
make integration_test

# Run specific test file
uv run pytest tests/unit_tests/test_chat_models.py

# Run specific test function
uv run pytest tests/unit_tests/test_chat_models.py::test_bind_tools

# Run tests matching a pattern
uv run pytest -k "test_retry"
```

### Test Coverage

- **Focus on**: Happy path, error cases, edge cases, both sync and async variants
- **Run**: `make test` includes coverage report

### Continuous Integration

Tests run automatically in CI on push and pull requests:

- **Lint**: `make lint_package` and `make lint_tests` (Python 3.10 and 3.12)
- **Test**: `make test` on Python 3.10, 3.11, 3.12, 3.13. CI also runs Python 3.14 as experimental (allowed to fail)

Ensure all checks pass locally before submitting your PR.

## Commit Message Guidelines

### Format

Use the conventional commits format:

```
<type>: <subject>

[optional body]

[optional footer]
```

### Type

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, no logic change)
- **refactor**: Code refactoring (no functional change)
- **test**: Adding or updating tests
- **chore**: Maintenance tasks, dependency updates, build config

### Subject

- Use imperative mood: "Add feature" not "Added feature" or "Adds feature"
- Keep it short (50 characters or less)
- Don't end with a period
- Capitalize the first letter

### Body (Optional)

- Provide context for the change
- Explain "why" not "what" (the diff shows what changed)
- Wrap at 72 characters
- Separate from subject with a blank line

### Footer (Optional)

- Reference issues: `Closes #123` or `Fixes #456`
- Note breaking changes: `BREAKING CHANGE: description`

### Examples

Good:

```
feat: add retry mechanism for rate-limited requests

Implement exponential backoff with configurable max retries.
Handles 429 status codes and Retry-After headers.

Closes #45
```

```
fix: handle tool_choice="any" gracefully

GigaChat API doesn't support "any". Convert to "auto"
when allow_any_tool_choice_fallback is set.
```

```
docs: update SSL certificate setup in README

Add troubleshooting section for common certificate errors
on Windows and macOS.
```

Bad:

```
update stuff          # Too vague
Fixed a bug.          # Not imperative, no context
WIP                   # Not descriptive
```

### Multiple Changes

If your commit includes multiple unrelated changes, split it into separate commits:

```bash
git add langchain_gigachat/chat_models/gigachat.py
git commit -m "feat: add timeout parameter to chat invocation"

git add tests/unit_tests/test_chat_models.py
git commit -m "test: add tests for chat timeout"
```

## Pull Request Process

### Before Submitting

1. **Run all quality checks**
   ```bash
   make format
   make lint_package
   make lint_tests
   make test
   ```

2. **Add tests** for new functionality or bug fixes

3. **Update documentation**
   - Add/update docstrings for new code
   - Update README.md if adding user-facing features

4. **Review your own changes**
   ```bash
   git diff master...your-branch
   ```

5. **Rebase on latest master** (if needed)
   ```bash
   git fetch upstream
   git rebase upstream/master
   ```

6. **Clean up your commits**
   - Use clear commit messages
   - Squash "fix typo" or "WIP" commits if appropriate

### Submitting Your PR

1. **Push your branch** to your fork
2. **Open a pull request** on GitHub
3. **Fill out the PR template** completely:
   - **Description**: Clear explanation of what and why
   - **Type of Change**: Check appropriate boxes
   - **Changes Made**: List key changes
   - **Testing**: Describe how you tested
   - **Checklist**: Verify all items
4. **Link related issues** using keywords: `Closes #123`, `Fixes #456`

### During Review

- Maintainers will review your PR within a few days
- You may receive feedback or requests for changes
- CI checks must pass before merging

Responding to feedback:

1. **Make requested changes** in new commits (don't force-push during active review)
2. **Reply to comments** when changes are made
3. **Re-request review** after addressing all feedback

### CI Checks

Your PR must pass all CI checks:

- **Linting**: Code follows style guidelines
- **Type Checking**: mypy passes
- **Tests**: All tests pass on Python 3.10-3.13

If CI fails, check the error logs in GitHub Actions, fix issues locally, and push fixes.

### Common Rejection Reasons

PRs may be rejected or require significant changes if:

- Missing tests for new functionality
- Breaking changes without discussion
- Doesn't follow code quality standards
- Adds unnecessary dependencies
- Scope is too large (consider breaking into smaller PRs)

### Tips for Faster Review

- **Keep PRs focused** — one feature or fix per PR
- **Keep PRs small** — easier to review, faster to merge
- **Write good descriptions** — help reviewers understand context
- **Respond promptly** to feedback

### Draft PRs

Use draft PRs for work in progress:

1. Open PR as "Draft"
2. Get early feedback on approach
3. Mark "Ready for review" when complete

## Issue Reporting

### Before Creating an Issue

1. **Search existing issues** — check both open and closed
2. **Check the documentation**: [README.md](libs/gigachat/README.md), [GigaChat API docs](https://developers.sber.ru/docs/ru/gigachat/api/main)
3. **Verify it's actually a bug** — test with the latest version: `pip install --upgrade langchain-gigachat`

### Bug Reports

Use the [Bug report](/.github/ISSUE_TEMPLATE/bug_report.yml) template. Include:

- **Clear description** of the bug
- **Steps to reproduce** the issue
- **Expected behavior** vs **actual behavior**
- **Minimal code example** that reproduces the bug
- **langchain-gigachat version**: `pip show langchain-gigachat`
- **Python version**: `python --version`
- **Operating system**
- **Error messages** and stack traces

### Feature Requests

Use the [Feature request](/.github/ISSUE_TEMPLATE/feature_request.yml) template. Include:

- **Problem statement**: What problem does this solve?
- **Proposed solution**: How should it work?
- **API design** (optional): Example code showing proposed usage
- **Use case**: Why is this useful?

### Security Issues

**Do not report security vulnerabilities in public issues.** Report privately through GitHub's security advisory feature or contact the maintainers directly.

### Issue Etiquette

- Be respectful and constructive
- Provide complete information the first time
- Respond to questions from maintainers
- Close the issue if you solve it yourself (and share the solution)
- Don't bump issues with "+1" comments — use reactions instead

## License

By contributing to langchain-gigachat, you agree that your contributions will be licensed under the MIT License.

This means:
- Your code will be freely available to everyone
- Others can use, modify, and distribute your contributions
- You retain copyright to your contributions

When you submit a pull request, you are confirming you have the right to contribute the code and granting the project maintainers and users the right to use your contribution under the MIT License terms.

If you're contributing on behalf of your employer, ensure you have permission to contribute under these terms.

The full license text is available in the [LICENSE](LICENSE) file.
