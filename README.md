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
pip install -U langchain-gigachat
```

```python
from langchain_gigachat import GigaChat

llm = GigaChat(credentials="your-authorization-key")
msg = llm.invoke("Hello, GigaChat!")
print(msg.content)
```

## Documentation

Full documentation, usage examples, and configuration reference are in [`libs/gigachat/README.md`](libs/gigachat/README.md).

## Related Projects

- **[GigaChain](https://github.com/ai-forever/gigachain)** — a set of solutions for developing LLM applications and multi-agent systems
- **[GigaChat Python SDK](https://github.com/ai-forever/gigachat)** — the underlying Python SDK
- [GigaChat API docs](https://developers.sber.ru/docs/ru/gigachat)

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

This project is licensed under the MIT License.

Copyright © 2026 [GigaChain](https://github.com/ai-forever/gigachain)
