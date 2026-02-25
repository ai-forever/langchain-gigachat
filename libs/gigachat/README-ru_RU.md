<div align="center" id="top">

[![GitHub Release](https://img.shields.io/github/v/release/ai-forever/langchain-gigachat?style=flat-square)](https://github.com/ai-forever/langchain-gigachat/releases)
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/ai-forever/langchain-gigachat/check_diffs.yml?style=flat-square)](https://github.com/ai-forever/langchain-gigachat/actions/workflows/check_diffs.yml)
[![PyPI - Version](https://img.shields.io/pypi/v/langchain-gigachat?label=PyPI&style=flat-square)](https://pypi.org/project/langchain-gigachat/#history)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/langchain-gigachat?style=flat-square)](https://pypi.org/project/langchain-gigachat/)
[![GitHub License](https://img.shields.io/github/license/ai-forever/langchain-gigachat?style=flat-square)](https://opensource.org/license/MIT)
[![GitHub Downloads (all assets, all releases)](https://img.shields.io/pypi/dm/langchain-gigachat?style=flat-square)](https://pypistats.org/packages/langchain-gigachat)
[![GitHub Repo stars](https://img.shields.io/github/stars/ai-forever/langchain-gigachat?style=flat-square)](https://star-history.com/#ai-forever/langchain-gigachat)
[![GitHub Open Issues](https://img.shields.io/github/issues-raw/ai-forever/langchain-gigachat)](https://github.com/ai-forever/langchain-gigachat/issues)

[English](README.md) | [Русский](README-ru_RU.md)

</div>

# langchain-gigachat

Библиотека `langchain-gigachat` — интеграция [GigaChat](https://giga.chat/) с LangChain и LangGraph (чат-модель, эмбеддинги, tool/function calling, вложения).

Библиотека входит в набор решений [GigaChain](https://github.com/ai-forever/gigachain).

## Быстрая установка

```bash
pip install -U langchain-gigachat
```

## 🤔 Что это?

Пакет предоставляет:

- **Чат-модель**: `langchain_gigachat.GigaChat` (sync/async, streaming, tool calling, structured output)
- **Эмбеддинги**: `langchain_gigachat.GigaChatEmbeddings`
- **Хелпер для tools**: `langchain_gigachat.tools.giga_tool.giga_tool` (расширение LangChain `@tool` для возможностей GigaChat)
- **Вложения**: загрузка файлов и отправка как `content_blocks` (картинки/аудио/документы)

## Требования

- Python **3.10+**
- Доступ к GigaChat API (ключ авторизации, access token или другой поддерживаемый способ)
- TLS/корневой сертификат (рекомендуется). Если в вашей среде требуется свой bundle, укажите `GIGACHAT_CA_BUNDLE_FILE` / `ca_bundle_file`.

Полезные ссылки:

- [README GigaChat Python SDK](https://github.com/ai-forever/gigachat/blob/main/README.md)
- [Документация GigaChat API](https://developers.sber.ru/docs/ru/gigachat)
- [Сертификаты (НУЦ Минцифры)](https://developers.sber.ru/docs/ru/gigachat/certificates)
- [Как получить ключ авторизации](https://developers.sber.ru/docs/ru/gigachat/quickstart/ind-using-api#poluchenie-avtorizatsionnyh-dannyh)

## Быстрый старт

### Чат

```python
from langchain_gigachat import GigaChat

llm = GigaChat(
    credentials="ВАШ_КЛЮЧ_АВТОРИЗАЦИИ",
    verify_ssl_certs=False,  # только для разработки (лучше настроить CA bundle)
)

msg = llm.invoke("Привет, GigaChat!")
print(msg.content)
```

### Стриминг

```python
from langchain_gigachat import GigaChat

llm = GigaChat(credentials="ВАШ_КЛЮЧ_АВТОРИЗАЦИИ", verify_ssl_certs=False)

for chunk in llm.stream("Напиши короткое четверостишие про программирование"):
    print(chunk.content, end="", flush=True)
print()
```

### Async

```python
import asyncio

from langchain_gigachat import GigaChat


async def main() -> None:
    llm = GigaChat(credentials="ВАШ_КЛЮЧ_АВТОРИЗАЦИИ", verify_ssl_certs=False)
    msg = await llm.ainvoke("Объясни квантовые вычисления простыми словами.")
    print(msg.content)


asyncio.run(main())
```

### Эмбеддинги

```python
from langchain_gigachat import GigaChatEmbeddings

emb = GigaChatEmbeddings(
    credentials="ВАШ_КЛЮЧ_АВТОРИЗАЦИИ",
    verify_ssl_certs=False,
    model="Embeddings",
)

vector = emb.embed_query("Привет!")
print(len(vector))
```

## Tool / Function calling

Используйте `giga_tool` (аналог LangChain `@tool` с дополнительными полями, которые поддерживает GigaChat).

```python
from langchain_gigachat import GigaChat
from langchain_gigachat.tools.giga_tool import giga_tool


@giga_tool
def get_weather(city: str) -> str:
    """Вернуть погоду для города."""
    return f"{city}: солнечно"


llm = GigaChat(credentials="ВАШ_КЛЮЧ_АВТОРИЗАЦИИ", verify_ssl_certs=False)
llm_with_tools = llm.bind_tools([get_weather], tool_choice="auto")

msg = llm_with_tools.invoke("Какая погода в Токио?")
print(msg.tool_calls)
```

Примечания:

- `tool_choice="any"` **не поддерживается** API GigaChat. Используйте `"auto"`, `"none"` или конкретное имя tool. Если вы получаете `"any"` из внешнего кода, можно создать `GigaChat(..., allow_any_tool_choice_fallback=True)`, чтобы автоматически преобразовывать `"any"` → `"auto"`.

## Structured output

```python
from pydantic import BaseModel, Field

from langchain_gigachat import GigaChat


class Answer(BaseModel):
    """Структурированный ответ."""

    text: str = Field(description="Ответ")
    confidence: float = Field(ge=0, le=1, description="Уверенность 0..1")


llm = GigaChat(credentials="ВАШ_КЛЮЧ_АВТОРИЗАЦИИ", verify_ssl_certs=False)
chain = llm.with_structured_output(Answer)
parsed = chain.invoke("Ответь коротко и добавь уверенность.")
print(parsed)
```

Также можно использовать JSON mode: `llm.with_structured_output(Answer, method="json_mode")`.

## Вложения (картинки/аудио/документы)

Загрузите файл через Files API и передайте его как стандартное вложение LangChain (`content_blocks`).

```python
from langchain_core.messages import HumanMessage

from langchain_gigachat import GigaChat

llm = GigaChat(credentials="ВАШ_КЛЮЧ_АВТОРИЗАЦИИ", verify_ssl_certs=False)

with open("image.png", "rb") as f:
    uploaded = llm.upload_file(("image.png", f.read()))

msg = HumanMessage(
    content_blocks=[
        {"type": "text", "text": "Опиши картинку."},
        {"type": "image", "file_id": uploaded.id_},
    ]
)

reply = llm.invoke([msg])
print(reply.content)
```

## Конфигурация

Параметры можно передавать в `GigaChat(...)` / `GigaChatEmbeddings(...)` напрямую или через переменные окружения с префиксом `GIGACHAT_`.

Примечания:

- Если вы передаёте Base64 data URL в блоках `image_url` / `audio_url` / `document_url`, можно включить `auto_upload_attachments=True`, чтобы автоматически загружать такие вложения. Это **не рекомендуется для production**; предпочтительнее явно вызывать `upload_file(...)`.
- Повторы запросов (retry) делает `gigachat` SDK (`max_retries`, `retry_backoff_factor`, `retry_on_status_codes`). Не включайте retries одновременно в SDK и в LangChain (например, `.with_retry()`), иначе число попыток перемножится.

Часто используемые переменные:

| Переменная | Значение |
|---|---|
| `GIGACHAT_CREDENTIALS` | Ключ/credentials (OAuth) |
| `GIGACHAT_ACCESS_TOKEN` | Готовый access token (JWT) |
| `GIGACHAT_SCOPE` | Scope (`GIGACHAT_API_PERS`, `GIGACHAT_API_B2B`, `GIGACHAT_API_CORP`) |
| `GIGACHAT_BASE_URL` | Base URL API |
| `GIGACHAT_VERIFY_SSL_CERTS` | Включить/выключить проверку TLS |
| `GIGACHAT_CA_BUNDLE_FILE` | Путь к CA bundle |

## 📖 Документация

- **Исходники**: `langchain_gigachat/`
- **GigaChat SDK**: [README](https://github.com/ai-forever/gigachat/blob/main/README.md)

## 💁 Участие в разработке

См. [`CONTRIBUTING.md`](../../CONTRIBUTING.md). Разработка ведётся в `libs/gigachat` (команды: `uv sync`, затем `make lint_package` / `make test`).
