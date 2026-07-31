# Release validation checklist

This file defines the prerelease and stable-release gates. Volatile results are
generated from the candidate checkout and attached to CI or the release run;
they are not copied into package documentation.

## Candidate evidence

Every validation artifact or PR summary must record:

- exact candidate commit SHA and dirty/clean state;
- package, Python, `langchain-core`, and `gigachat` versions;
- commands and exit status for formatting, lint, typing, unit tests, coverage,
  lock validation, build, and clean artifact installs;
- wheel and sdist SHA-256 values built from that candidate;
- stable-SDK resource inspection result;
- live-provider scenario status and sanitized request IDs for failures.

Optional or credential-gated skips remain skips. A separate compatible
environment may demonstrate an optional contract, but it must not inflate the
default suite's pass count.

Re-run every claimed gate after any code change. Evidence for a different SHA
does not validate the current candidate.

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

Stable release remains blocked while the package depends on a prerelease SDK.
Unblock it only after an isolated stable SDK passes this inspection, the
package dependency is changed to the verified stable range, the lockfile is
regenerated, and the complete deterministic, artifact, and live matrices pass
again for the same candidate.

## Live provider gate

Run:

```bash
make integration_tests
```

A collected-but-skipped test confirms only gate wiring; it is not live
provider evidence.

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
| Streamed built-in-tool lifecycle (`stream` / `astream`) | NOT RUN |
| Streamed client-function roundtrip | NOT RUN |
| Non-stream client-function roundtrip | NOT RUN |
| Web search | NOT RUN |
| Code interpreter | NOT RUN |
| Native JSON Schema | NOT RUN |
| Schema-less native JSON (`json_mode`) | NOT RUN |
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
