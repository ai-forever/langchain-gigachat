# langchain-gigachat

This is a library integration with [GigaChat](https://giga.chat/).

[![GitHub Release](https://img.shields.io/github/v/release/ai-forever/langchain-gigachat?style=flat-square)](https://github.com/ai-forever/langchain-gigachat/releases)
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/ai-forever/langchain-gigachat/check_diffs.yml?style=flat-square)](https://github.com/ai-forever/langchain-gigachat/actions/workflows/check_diffs.yml)
[![GitHub License](https://img.shields.io/github/license/ai-forever/langchain-gigachat?style=flat-square)](https://opensource.org/license/MIT)
[![GitHub Downloads (all assets, all releases)](https://img.shields.io/github/downloads/ai-forever/langchain-gigachat/total?style=flat-square)](https://pypistats.org/packages/langchain-gigachat)
[![GitHub Repo stars](https://img.shields.io/github/stars/ai-forever/langchain-gigachat?style=flat-square)](https://star-history.com/#ai-forever/langchain-gigachat)
[![GitHub Open Issues](https://img.shields.io/github/issues-raw/ai-forever/langchain-gigachat)](https://github.com/ai-forever/langchain-gigachat/issues)

## Installation

```bash
pip install -U langchain-gigachat
```

# LangChain GigaChat Integration - Quickstart Guide

Welcome to the Quickstart guide for integrating GigaChat with LangChain! Follow these simple steps to get up and running quickly.

## Quickstart Steps

1. **Get an Authorization Key**

   Visit [developers.sber.ru](https://developers.sber.ru) and create an account to obtain your authorization key. This key is required to authenticate with GigaChat.

2. **Initialize the GigaChat Object**

   In your Python code, initialize the GigaChat object using your authorization key:

   ```python
   from langchain_gigachat.chat_models import GigaChat

   giga = GigaChat(credentials="YOUR_AUTHORIZATION_KEY", verify_ssl_certs=False)
   ```
   Note: The `verify_ssl_certs=False` flag is needed if the required certificate from the Russian Ministry of Digital Development is not installed on your computer.

3. **Invoke GigaChat**

   Use the GigaChat object to generate responses:

   ```python
   print(giga.invoke("Hello, world!"))
   ```

Now you can use the GigaChat object with LangChain's standard primitives to create interactive conversational applications.
