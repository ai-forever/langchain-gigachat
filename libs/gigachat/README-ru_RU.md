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

Библиотека `langchain-gigachat` позволяет использовать нейросетевые модели GigaChat при разработке LLM-приложений с помощью фреймворков LangChain и LangGraph.

Библиотека входит в набор решений [GigaChain](https://github.com/ai-forever/gigachain).

## Требования

Для работы с библиотекой и обмена сообщениями с моделями GigaChat понадобятся:

* Python версии 3.9 и выше;
* [сертификат НУЦ Минцифры](https://developers.sber.ru/docs/ru/gigachat/certificates);
* [ключ авторизации](https://developers.sber.ru/docs/ru/gigachat/quickstart/ind-using-api#poluchenie-avtorizatsionnyh-dannyh) GigaChat API.

> [!NOTE]
> Вы также можете использовать другие [способы авторизации](#способы-авторизации).

## Установка

Для установки библиотеки используйте менеджер пакетов pip:

```sh
pip install -U langchain-gigachat
```

## Быстрый старт

### Запрос на генерацию

Пример запроса на генерацию:

```py
from langchain_gigachat.chat_models import GigaChat

giga = GigaChat(
    # Для авторизации запросов используйте ключ, полученный в проекте GigaChat API
    credentials="ваш_ключ_авторизации",
    verify_ssl_certs=False,
)

print(giga.invoke("Hello, world!"))
```

### Создание эмбеддингов

Пример создания векторного представления текста:

```py
from langchain_gigachat.embeddings import GigaChatEmbeddings

embeddings = GigaChatEmbeddings(credentials="ключ_авторизации", verify_ssl_certs=False)
result = embeddings.embed_documents(texts=["Привет!"])
print(result)
```

## Параметры объекта GigaChat

В таблице описаны параметры, которые можно передать при инициализации объекта GigaChat:

| Параметр           | Обязательный | Описание                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ------------------ | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `credentials`      | да           | Ключ авторизации для обмена сообщениями с GigaChat API.<br />Ключ авторизации содержит информацию о версии API, к которой выполняются запросы. Если вы используете версию API для ИП или юрлиц, укажите это явно в параметре `scope`                                                                                                                                                                                                                                                                                                                   |
| `verify_ssl_certs` | нет          | Отключение проверки ssl-сертификатов.<br /><br />Для обращения к GigaChat API нужно [установить корневой сертификат НУЦ Минцифры](#установка-корневого-сертификата-нуц-минцифры).<br /><br />Используйте параметр ответственно, так как отключение проверки сертификатов снижает безопасность обмена данными                                                                                                                                                                                                                                                                                                                                                                                           |
| `scope`            | нет          | Версия API, к которой будет выполнен запрос. По умолчанию запросы передаются в версию для физических лиц. Возможные значения:<ul><li>`GIGACHAT_API_PERS` — версия API для физических лиц;</li><li>`GIGACHAT_API_B2B` — версия API для ИП и юрлиц при работе по предоплате.</li><li>`GIGACHAT_API_CORP` — версия API для ИП и юрлиц при работе по постоплате.</li></ul>                                                                                                                                                                                 |
| `model`            | нет          | необязательный параметр, в котором можно явно задать [модель GigaChat](https://developers.sber.ru/docs/ru/gigachat/models). Вы можете посмотреть список доступных моделей с помощью метода `get_models()`, который выполняет запрос [`GET /models`](https://developers.sber.ru/docs/ru/gigachat/api/reference#get-models).<br /><br />Стоимость запросов к разным моделям отличается. Подробную информацию о тарификации запросов к той или иной модели вы ищите в [официальной документации](https://developers.sber.ru/docs/ru/gigachat/api/tariffs) |
| `base_url`         | нет          | Адрес API. По умолчанию запросы отправляются по адресу `https://gigachat.devices.sberbank.ru/api/v1/`, но если вы хотите использовать [модели в раннем доступе](https://developers.sber.ru/docs/ru/gigachat/models/preview-models), укажите адрес `https://gigachat-preview.devices.sberbank.ru/api/v1`                                                                                                                                                                                                                                                |

> [!TIP]
> Чтобы не указывать параметры при каждой инициализации, задайте их в [переменных окружения](#настройка-переменных-окружения).

## Способы авторизации

Для авторизации запросов, кроме ключа, полученного в личном кабинете, вы можете использовать:

* имя пользователя и пароль для доступа к сервису;
* сертификаты TLS;
* токен доступа (access token), полученный в обмен на ключ авторизации в запросе [`POST /api/v2/oauth`](https://developers.sber.ru/docs/ru/gigachat/api/reference/rest/post-token).

Для этого передайте соответствующие параметры при инициализации.

Пример авторизации с помощью логина и пароля:

```py
giga = GigaChat(
    base_url="https://gigachat.devices.sberbank.ru/api/v1",
    user="имя_пользоваеля",
    password="пароль",
)
```

Авторизация с помощью сертификатов по протоколу TLS (mTLS):

```py
giga = GigaChat(
    base_url="https://gigachat.devices.sberbank.ru/api/v1",
    ca_bundle_file="certs/ca.pem",  # chain_pem.txt
    cert_file="certs/tls.pem",  # published_pem.txt
    key_file="certs/tls.key",
    key_file_password="123456",
    ssl_context=context # optional ssl.SSLContext instance
)
```

Авторизация с помощью токена доступа:

```py
giga = GigaChat(
    access_token="ваш_токен_доступа",
)
```

> [!NOTE]
> Токен действителен в течение 30 минут.
> При использовании такого способа авторизации, в приложении нужно реализовать механизм обновления токена.

### Предварительная авторизация

По умолчанию, библиотека GigaChat получает токен доступа при первом запросе к API.

Если вам нужно получить токен и авторизоваться до выполнения запроса, инициализируйте объект GigaChat и вызовите метод `get_token()`.

```py
giga = GigaChat(
    base_url="https://gigachat.devices.sberbank.ru/api/v1",
    user="имя_пользователя",
    password="пароль",
)
giga.get_token()
```

## Настройка переменных окружения

Чтобы задать параметры с помощью переменных окружения, в названии переменной используйте префикс `GIGACHAT_`.

Пример переменных окружения, которые задают ключ авторизации, версию API и отключают проверку сертификатов.

```sh
export GIGACHAT_CREDENTIALS=...
export GIGACHAT_SCOPE=...
export GIGACHAT_VERIFY_SSL_CERTS=False
```

Пример переменных окружения, которые задают адрес API, имя пользователя и пароль.

```sh
export GIGACHAT_BASE_URL=https://gigachat.devices.sberbank.ru/api/v1
export GIGACHAT_USER=...
export GIGACHAT_PASSWORD=...
```
