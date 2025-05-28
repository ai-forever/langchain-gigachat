<div align="center" id="top">

[![GitHub Release](https://img.shields.io/github/v/release/ai-forever/langchain-gigachat?style=flat-square)](https://github.com/ai-forever/langchain-gigachat/releases)
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/ai-forever/langchain-gigachat/check_diffs.yml?style=flat-square)](https://github.com/ai-forever/langchain-gigachat/actions/workflows/check_diffs.yml)
[![GitHub License](https://img.shields.io/github/license/ai-forever/langchain-gigachat?style=flat-square)](https://opensource.org/license/MIT)
[![GitHub Downloads (all assets, all releases)](https://img.shields.io/pypi/dm/langchain-gigachat?style=flat-square?style=flat-square)](https://pypistats.org/packages/langchain-gigachat)
[![GitHub Repo stars](https://img.shields.io/github/stars/ai-forever/langchain-gigachat?style=flat-square)](https://star-history.com/#ai-forever/langchain-gigachat)
[![GitHub Open Issues](https://img.shields.io/github/issues-raw/ai-forever/langchain-gigachat)](https://github.com/ai-forever/langchain-gigachat/issues)

[English](README.md) | [Русский](README-ru_RU.md)

</div>

# langchain-gigachat

This is a library integration with [GigaChat](https://giga.chat/).

## Installation

```bash
pip install -U langchain-gigachat
```

## Quickstart
Follow these simple steps to get up and running quickly.

### Installation

To install the package use following command:

```shell
pip install -U langchain-gigachat
```

### Initialization

To initialize chat model:

```python
from langchain_gigachat.chat_models import GigaChat

giga = GigaChat(credentials="YOUR_AUTHORIZATION_KEY", verify_ssl_certs=False)
```

To initialize embeddings:

```python
from langchain_gigachat.embeddings import GigaChatEmbeddings

embedding = GigaChatEmbeddings(
    credentials="YOUR_AUTHORIZATION_KEY",
    verify_ssl_certs=False
)
```

### Usage

Use the GigaChat object to generate responses:

```python
print(giga.invoke("Hello, world!"))
```

Now you can use the GigaChat object with LangChain's standard primitives to create LLM-applications.