# GigaChat feature examples

Five standalone scripts for the `langchain-gigachat 0.5.2a1` prerelease.
Each covers the full flow: create a model, send a request, and read the response.
The scripts make live requests only when run; importing them does not send requests.

## Setup

You need Python 3.10+, the `langchain-gigachat 0.5.2a1` prerelease, and the
GigaChat SDK `0.2.4a1` prerelease. Install both packages from source.
From this repository's root, with the SDK source in a sibling directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ../gigachat -e ./libs/gigachat
```

Replace `../gigachat` with the path to your SDK source directory.
SDK `0.2.3` supports the basic flows, but not every feature in these examples.

The SDK reads authentication from `GIGACHAT_CREDENTIALS` or
`GIGACHAT_ACCESS_TOKEN`; its usual scope and TLS settings also apply.
Configure these in your environment. Do not put credentials in the scripts.
Set the model through `GIGACHAT_MODEL`; the default is `GigaChat`.
Reasoning, storage, and forced tool selection require support from your server,
model, and account.

Run an example from the repository root:

```bash
python examples/sdk_contract_alignment/parallel_tools.py
```

## What to try

| Script | Scenario | What to look for |
|---|---|---|
| [generation_options.py](generation_options.py) | Solve the same arithmetic problem through v1 and v2, then make a separate short request with nested options | `reasoning_max_tokens`, the precedence of `model_options.max_tokens`, `additional_fields`, `finish_reason` |
| [parallel_tools.py](parallel_tools.py) | Compare the capacities of two rooms whose identifiers can be `int` or `str` | `anyOf`, `required` on the first turn only, a separate ID for each call and a result for each ID |
| [streaming_metadata.py](streaming_metadata.py) | Stream text with `astream()` and assemble the final message | Text, final usage, `additional_data`/`error_details`, and content blocks without duplicate IDs |
| [session_usage.py](session_usage.py) | Ask two questions about the same room directory with one `X-Session-ID` | Total LangChain input, its cached portion, and raw SDK counters through a callback |
| [stored_tool_results.py](stored_tool_results.py) | Create a stored thread with a function call, then continue it using only the results | The distinct roles of `thread_id`, `tools_state_id`, and `tool_call_id`; no need to resend history or the model |

### How the tool loop works

In `parallel_tools.py`, `get_room(room: int | str)` accepts two argument types.
The v2 schema preserves `anyOf`. The v1 route still has its existing limitation
on these union schemas.

`parallel_tool_calls=True` allows multiple calls in one response. The model may
choose sequential calls instead; the example handles both cases. Each call gets
its own `ToolMessage` with the original `call["id"]`, even when the function name
is the same. The local functions run sequentially here; the flag controls the
shape of the model's response.

`required` applies only to the first turn. After returning the results, the loop
uses `auto` so the model can finish with a text response. The loop is limited to
four turns. A production agent should also handle tool exceptions according to
the application's requirements.

### Sessions, history, and tokens

Reusing `session_id` across related requests lets the server use a shared prefix
cache. It does not store the conversation. Cached usage may remain zero: the
script reports the actual counters and does not assume a guaranteed cache hit.
Conversation storage uses a separate `storage.thread_id`.

For example, if the SDK reports 14 new input tokens, 2430 cached tokens, and 2
output tokens, LangChain returns `input_tokens=2444`, `cache_read=2430`,
`output_tokens=2`, and `total_tokens=2446`. The callback preserves the raw SDK
usage values for billing. `session_usage.py` prints both representations.

When continuing a stored thread without the previous `AIMessage`, pass
`tools_state_id` in `ToolMessage.additional_kwargs`. `tool_call_id` remains the
ID of the individual call. When you pass the full history, as in
`parallel_tools.py`, the adapter extracts the state from the previous `AIMessage`.

### Options and metadata

On v2, generation options go into `model_options`. An explicit nested value takes
precedence over shorthand options and constructor defaults; a nested `None`
suppresses the corresponding default. The example passes `ranker_options`
through `additional_fields`, without a dedicated constructor parameter. This
feature depends on the server: passing extra options does not guarantee that
your account has access to them.

A reasoning budget does not guarantee a final answer. Check `finish_reason` and
the actual response text. `temperature=0` does not guarantee determinism either.

Provider-specific fields may be absent. In v2 streaming, `additional_data` and
`error_details` snapshots arrive at the end. The example adds each chunk exactly
once with `+`, then reads the final metadata.

## Example validation

Offline validation uses the `0.2.4a1` SDK prerelease: the adapter and SDK HTTP
client run with mocked transport responses. This checks request formats,
response handling, and script execution. It does not test live requests or
feature availability for a particular account.

[Changes and compatibility](../../libs/gigachat/MIGRATION.md#sdk-contract-alignment)
· [Package documentation](../../libs/gigachat/README.md)
