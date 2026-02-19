## Description

<!-- Provide a clear and concise description of what this PR does -->

## Motivation

<!-- Why is this change needed? Link to related issues if applicable -->

<!--
If this PR fixes an existing issue, uncomment and fill the line below.
Example: Closes #123
-->
<!-- Closes # -->

## Type of Change

<!-- Mark the relevant option with an "x" -->

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Code refactoring (no functional changes)
- [ ] Performance improvement
- [ ] Test coverage improvement
- [ ] CI/CD or tooling change

## Changes Made

<!-- List the main changes made in this PR -->

-
-
-

## Testing

<!-- Describe the tests you ran and how to reproduce them -->

### Test Coverage

- [ ] Unit tests added/updated
- [ ] Integration tests added/updated (if applicable)
- [ ] All existing tests pass locally

### Manual Testing

<!-- Describe any manual testing performed. Run from `libs/gigachat`: -->

```bash
make test
make lint_package
make lint_tests
```

## Checklist

<!-- Mark completed items with an "x" -->

### Code Quality

- [ ] Code follows the project's style guidelines (`make format`, `make lint_package`, `make lint_tests` from `libs/gigachat`)
- [ ] I have performed a self-review of my code
- [ ] I have commented my code only where necessary (hard-to-understand areas)
- [ ] My changes generate no new linter warnings
- [ ] Type checking passes (mypy runs as part of `make lint_package` / `make lint_tests`)
- [ ] All tests pass (`make test`)

### Documentation

- [ ] I have updated the documentation accordingly
- [ ] Docstrings follow Google style with imperative mood
- [ ] I have added examples for new features (if applicable)
- [ ] README.md updated (if applicable)

### Dependencies

- [ ] No new dependencies added (or they are justified and minimal)
- [ ] `uv.lock` updated (if dependencies changed)

### Compatibility

- [ ] Changes are compatible with supported Python versions (3.10–3.13). CI also runs 3.14 as experimental.
- [ ] Async/sync variants both work correctly (if applicable)

### Commits

- [ ] Commit messages are clear and follow conventional commits style where possible
- [ ] Commits are logically organized
- [ ] No debug or commented-out code left in

## Additional Context

<!-- Add any other context, screenshots, or information about the PR here -->

## Pre-merge (for maintainers)

- [ ] Changelog updated (if applicable)
- [ ] Version bump considered (if applicable)
