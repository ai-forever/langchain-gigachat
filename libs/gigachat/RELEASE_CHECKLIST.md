# Release validation checklist

This file is the single source of truth for prerelease and stable-release
validation facts. Update it from the candidate checkout and synchronize any
short summaries in README, MIGRATION, and CHANGELOG in the same commit.

## Candidate snapshot

| Field | Verified value |
|-------|----------------|
| Assembled A–F implementation | `ec0d185cc5c22fdac9d1a0da29f1bdd6a440505d` |
| Package | `langchain-gigachat==0.5.2a1` |
| Python used for local validation | `3.12.12` |
| Python used for clean artifact installs | `3.14.3` |
| Locked LangChain Core | `1.2.5` |
| Supported LangChain Core range | `>=1.2,<2` |
| Pinned GigaChat SDK | `0.2.3a1` |
| Unit suite | PASS — 694 passed, 5 skipped |
| Coverage | PASS — 93.98% (required: 90%) |
| Hosted Agent contract equivalent | PASS — `langchain==1.3.14`, 5 passed |
| Minimum/latest Core matrix | PASS — Core `1.2.0` and `1.5.3`; 694 passed, 5 skipped, 93.98% each |
| Wheel build and clean install | PASS |
| sdist build and clean install | PASS |
| Live provider matrix | BLOCKED — credentials and provider state fixtures were unavailable |
| Stable SDK dependency gate | BLOCKED — stable `gigachat==0.2.1` lacks the required resource API |

The five skipped tests are optional LangChain Agent compatibility tests. The
same five contracts passed in an isolated `langchain==1.3.14` environment;
they are not counted as passed in the default unit-suite total.

Both clean artifact environments resolved `langchain-core==1.5.3` from the
declared compatible range and `gigachat==0.2.3a1` from the exact SDK pin. The
wheel and sdist each imported `langchain_gigachat.GigaChat`, reported package
version `0.5.2a1`, and exposed all four required SDK resource methods.

Built artifact hashes:

```text
692abbe7ef08ae2f5313389430ec97d72c75c675be8e8db50a1a0029fb701dce  langchain_gigachat-0.5.2a1.tar.gz
4aa19eb1b287b33ca831ad1c77b1018e8295ddf0ce3055f2e4d0d39fc1000515  langchain_gigachat-0.5.2a1-py3-none-any.whl
```

The PR body must name the final documentation/validation commit, not only the
assembled A–F implementation above. Re-run this checklist after any code change.

## Deterministic validation

Run from `libs/gigachat`:

```bash
uv sync --frozen
make format
make lint_package
make lint_tests
make test
uv lock --check
uv build
```

For each artifact, create an empty environment, install only that artifact, and
verify imports, versions, dependency metadata, and the required SDK resources:

```bash
uv venv /tmp/langchain-gigachat-wheel
uv pip install \
  --python /tmp/langchain-gigachat-wheel/bin/python \
  dist/langchain_gigachat-0.5.2a1-py3-none-any.whl

uv venv /tmp/langchain-gigachat-sdist
uv pip install \
  --python /tmp/langchain-gigachat-sdist/bin/python \
  dist/langchain_gigachat-0.5.2a1.tar.gz
```

The installed package must report:

```text
langchain-gigachat 0.5.2a1
langchain-core >=1.2,<2
gigachat ==0.2.3a1
chat.create / chat.stream
achat.create / achat.stream
```

## Stable SDK dependency gate

Do not replace the alpha pin merely because a stable SDK exists. Install the
latest stable SDK in an isolated environment and verify all four required
resource methods:

```text
client.chat.create(...)
client.chat.stream(...)
await client.achat.create(...)
client.achat.stream(...)
```

For this candidate, an isolated `gigachat==0.2.1` inspection returned legacy
method objects for `chat` and `achat`; none of `create` or `stream` existed on
those objects. The installed alpha `gigachat==0.2.3a1` exposes
`ChatNamespace.create`, `ChatNamespace.stream`,
`AsyncChatNamespace.create`, and `AsyncChatNamespace.stream`.

Stable release remains blocked until a stable SDK passes this inspection, the
package dependency is changed to a verified stable range, the lockfile is
regenerated, and the complete deterministic and live matrices pass again.

## Live provider gate

Run:

```bash
make integration_tests
```

In the recorded local environment this target collected 11 tests and skipped
all 11 because no GigaChat authentication variables or provider state fixtures
were present. That confirms the gate wiring but is not live evidence.

Record these non-secret facts for every run:

```text
candidate commit SHA
langchain-gigachat / langchain-core / gigachat versions
Python version
model
auth mode (never the credential)
base URL
scope
request IDs for failures
```

Never store credentials, access tokens, authorization headers, private
certificate material, or unsanitized cassettes.

| Scenario | Status |
|----------|--------|
| Plain `invoke` / `ainvoke` | NOT RUN |
| Plain `stream` / `astream` | NOT RUN |
| Streamed client-function roundtrip | NOT RUN |
| Non-stream client-function roundtrip | NOT RUN |
| Web search | NOT RUN |
| Code interpreter | NOT RUN |
| Native JSON Schema | NOT RUN |
| Invalid structured output | NOT RUN |
| Assistant state | NOT RUN |
| Thread state | NOT RUN |
| Existing file ID | NOT RUN |
| Returned `AIMessage` replay | NOT RUN |
| Real SDK ContextVar headers | NOT RUN |

The matrix is incomplete until every scenario has PASS/FAIL evidence. A skipped
test, missing credential, or missing assistant/thread/file fixture is BLOCKED,
not PASS.

## Release decision

Prerelease merge is not approved by this checklist alone. It additionally
requires all P0 findings fixed, P1 findings fixed or explicitly accepted,
hosted Agent tests, full typing/lint CI, matching public documentation, and the
actual final assembly SHA in the PR body.

Stable release is blocked by both the stable SDK dependency gate and the
unexecuted live provider matrix.
