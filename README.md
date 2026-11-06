<div align="center">

[![PyPI](https://img.shields.io/pypi/v/langchain-gigachat?style=flat-square)](https://pypi.org/project/langchain-gigachat/)
[![Python](https://img.shields.io/pypi/pyversions/langchain-gigachat?style=flat-square)](https://pypi.org/project/langchain-gigachat/)
[![CI](https://img.shields.io/github/actions/workflow/status/ai-forever/langchain-gigachat/check_diffs.yml?style=flat-square)](https://github.com/ai-forever/langchain-gigachat/actions/workflows/check_diffs.yml)
[![License](https://img.shields.io/github/license/ai-forever/langchain-gigachat?style=flat-square)](https://opensource.org/license/MIT)
[![Downloads](https://img.shields.io/pypi/dm/langchain-gigachat?style=flat-square)](https://pypistats.org/packages/langchain-gigachat)

</div>

# langchain-gigachat

LangChain integration for [GigaChat](https://giga.chat/) — a large language model.

This library is part of [GigaChain](https://github.com/ai-forever/gigachain) and wraps the [GigaChat Python SDK](https://github.com/ai-forever/gigachat) with LangChain-compatible interfaces.

## Quick Start

```bash
python -m pip install "langchain-gigachat==0.5.3a1"
# Optional async aiohttp backend:
python -m pip install "langchain-gigachat[aiohttp]==0.5.3a1"
```

Alpha `0.5.3a1` targets exactly `gigachat==0.3.0a1`, with LangChain Core
`>=1.2.5,<2` and HTTPX2 `>=2.14.0,<3`. API v1 remains the default;
API v2 is opt-in. See [the migration guide](libs/gigachat/MIGRATION.md#sdk-030a1-alpha-053a1) for constructor request defaults,
client ownership, and async cleanup limits.
For runnable examples, see
[example setup](examples/sdk_contract_alignment/README.md#setup).

```python
from langchain_gigachat import GigaChat

llm = GigaChat(credentials="your-authorization-key")
msg = llm.invoke("Hello, GigaChat!")
print(msg.content)
```

## Documentation

Full documentation, usage examples, and configuration reference are in [`libs/gigachat/README.md`](libs/gigachat/README.md).

Runnable [examples](examples/sdk_contract_alignment/README.md)
cover parallel tools, reasoning budgets, streaming metadata, sessions, and stored
thread continuation.

## Related Projects

- **[GigaChain](https://github.com/ai-forever/gigachain)** — a set of solutions for developing LLM applications and multi-agent systems
- **[GigaChat Python SDK](https://github.com/ai-forever/gigachat)** — the underlying Python SDK
- [GigaChat API docs](https://developers.sber.ru/docs/ru/gigachat)

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

This project is licensed under the MIT License.

Copyright © 2026 [GigaChain](https://github.com/ai-forever/gigachain)
