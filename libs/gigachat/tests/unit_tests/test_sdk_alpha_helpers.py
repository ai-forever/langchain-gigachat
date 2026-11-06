"""Offline wire checks for the SDK alpha helper methods."""

import base64
import io
import json
from typing import Any

import gigachat.models as gm
import httpx2
import pytest
from gigachat import RequestOptions

from langchain_gigachat import GigaChat

FILE = {
    "id": "f1",
    "object": "file",
    "bytes": 3,
    "created_at": 1,
    "filename": "test.txt",
    "purpose": "general",
}
MODEL = {"id": "test-model", "object": "model", "owned_by": "test"}


def _handler(seen: list[httpx2.Request]) -> Any:
    def send(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        path = request.url.path
        if path.endswith("/tokens/count"):
            return httpx2.Response(
                200, json=[{"tokens": 2, "characters": 3, "object": "tokens"}]
            )
        if path.endswith("/models/test-model"):
            return httpx2.Response(200, json=MODEL)
        if path.endswith("/models"):
            return httpx2.Response(
                200, json={"data": [MODEL], "object": "list", "balance": []}
            )
        if path.endswith("/files/f1/content"):
            return httpx2.Response(200, content=b"\x00image\xff")
        if path.endswith("/files/f1/delete"):
            return httpx2.Response(200, json={"id": "f1", "deleted": True})
        if path.endswith("/files/f1"):
            return httpx2.Response(200, json=FILE)
        if path.endswith("/files"):
            return httpx2.Response(
                200, json=FILE if request.method == "POST" else {"data": [FILE]}
            )
        raise AssertionError(f"unexpected request: {request.url}")

    return send


@pytest.mark.asyncio
async def test_helper_defaults_and_typed_image_wire() -> None:
    seen: list[httpx2.Request] = []
    transport = httpx2.MockTransport(_handler(seen))
    sync = httpx2.Client(transport=transport)
    async_client = httpx2.AsyncClient(transport=transport)
    model = GigaChat(
        model="test-model",
        access_token="token",
        http_client=sync,
        async_http_client=async_client,
        default_request_options=RequestOptions(
            extra_headers={"X-Feature": "helpers"}, extra_query=[("q", "1"), ("q", "2")]
        ),
    )
    try:
        assert model.tokens_count(["abc"])[0].tokens == 2
        assert (await model.atokens_count(["abc"]))[0].tokens == 2
        assert (
            model.get_models().data[0].id_
            == (await model.aget_models()).data[0].id_
            == "test-model"
        )
        assert (
            model.get_model("test-model").id_
            == (await model.aget_model("test-model")).id_
            == "test-model"
        )
        assert model.get_file("f1").id_ == (await model.aget_file("f1")).id_ == "f1"
        assert (
            model.list_files().data[0].id_
            == (await model.alist_files()).data[0].id_
            == "f1"
        )
        assert (
            model.delete_file("f1").deleted and (await model.adelete_file("f1")).deleted
        )
        for image in (
            model.get_file_content("f1"),
            await model.aget_file_content("f1"),
        ):
            assert isinstance(image, gm.Image)
            assert image.content == base64.b64encode(b"\x00image\xff").decode()
        assert len(seen) == 14
        assert all(r.headers["x-feature"] == "helpers" for r in seen)
        assert all(r.url.params.multi_items() == [("q", "1"), ("q", "2")] for r in seen)
    finally:
        await model.aclose()
        sync.close()
        await async_client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "file",
    [
        b"abc",
        "abc",
        ("test.txt", b"abc"),
        ("test.txt", b"abc", "text/plain"),
        ("test.txt", b"abc", "text/plain", {"X-Part": "yes"}),
    ],
)
async def test_upload_forms_pass_through_with_defaults(file: Any) -> None:
    seen: list[httpx2.Request] = []
    transport = httpx2.MockTransport(_handler(seen))
    sync = httpx2.Client(transport=transport)
    async_client = httpx2.AsyncClient(transport=transport)
    model = GigaChat(
        access_token="token",
        http_client=sync,
        async_http_client=async_client,
        default_request_options=RequestOptions(extra_headers={"X-Feature": "upload"}),
    )
    try:
        assert model.upload_file(file).id_ == "f1"
        assert (await model.aupload_file(file)).id_ == "f1"
        assert all(
            r.headers["x-feature"] == "upload" and b"abc" in r.content for r in seen
        )
        assert all("multipart/form-data" in r.headers["content-type"] for r in seen)
    finally:
        await model.aclose()
        sync.close()
        await async_client.aclose()


@pytest.mark.asyncio
async def test_oauth_does_not_receive_api_options() -> None:
    api_seen: list[httpx2.Request] = []
    oauth_seen: list[httpx2.Request] = []

    def api_send(request: httpx2.Request) -> httpx2.Response:
        api_seen.append(request)
        return httpx2.Response(
            200, json=[{"tokens": 2, "characters": 3, "object": "tokens"}]
        )

    def oauth_send(request: httpx2.Request) -> httpx2.Response:
        oauth_seen.append(request)
        return httpx2.Response(
            200, json={"access_token": "token", "expires_at": 9999999999999}
        )

    api = httpx2.Client(transport=httpx2.MockTransport(api_send))
    oauth = httpx2.Client(transport=httpx2.MockTransport(oauth_send))
    api_async = httpx2.AsyncClient(transport=httpx2.MockTransport(api_send))
    oauth_async = httpx2.AsyncClient(transport=httpx2.MockTransport(oauth_send))
    opts = RequestOptions(
        extra_headers={"X-Feature": "private-api"},
        extra_query={"api_only": "1"},
        extra_body={"api_marker": "yes"},
    )
    # Separate instances force both OAuth transports to execute.
    sync_model = GigaChat(
        credentials="dGVzdA==",
        model="test-model",
        http_client=api,
        auth_http_client=oauth,
        default_request_options=opts,
    )
    async_model = GigaChat(
        credentials="dGVzdA==",
        model="test-model",
        async_http_client=api_async,
        auth_async_http_client=oauth_async,
        default_request_options=opts,
    )
    try:
        sync_model.tokens_count(["abc"])
        await async_model.atokens_count(["abc"])
        assert len(api_seen) == len(oauth_seen) == 2
        assert all(
            r.headers["x-feature"] == "private-api" and r.url.params["api_only"] == "1"
            for r in api_seen
        )
        assert all(json.loads(r.content)["api_marker"] == "yes" for r in api_seen)
        assert all(
            "x-feature" not in r.headers
            and "api_only" not in r.url.params
            and b"api_marker" not in r.content
            for r in oauth_seen
        )
    finally:
        await sync_model.aclose()
        await async_model.aclose()
        api.close()
        oauth.close()
        await api_async.aclose()
        await oauth_async.aclose()


def test_embedding_error_reports_sdk_input_index() -> None:
    from langchain_gigachat.embeddings.gigachat import _extract_vectors

    error = gm.EmbeddingError.model_validate(
        {"object": "error", "index": 7, "error": {"message": "bad input"}}
    )
    with pytest.raises(ValueError, match="input index 7"):
        _extract_vectors([error])


@pytest.mark.asyncio
async def test_upload_stream_and_reject_json_extras() -> None:
    seen: list[httpx2.Request] = []
    transport = httpx2.MockTransport(_handler(seen))
    sync = httpx2.Client(transport=transport)
    async_client = httpx2.AsyncClient(transport=transport)
    model = GigaChat(
        access_token="token",
        http_client=sync,
        async_http_client=async_client,
        default_request_options=RequestOptions(extra_headers={"X-Feature": "upload"}),
    )
    invalid = GigaChat(
        access_token="token",
        http_client=sync,
        default_request_options=RequestOptions(extra_body={"marker": 1}),
    )
    try:
        assert model.upload_file(io.BytesIO(b"abc")).id_ == "f1"
        assert (await model.aupload_file(io.BytesIO(b"abc"))).id_ == "f1"
        assert all(b"abc" in request.content for request in seen)
        with pytest.raises(ValueError, match="extra_body.*JSON"):
            invalid.upload_file(b"abc")
        assert len(seen) == 2
    finally:
        await model.aclose()
        await invalid.aclose()
        sync.close()
        await async_client.aclose()
