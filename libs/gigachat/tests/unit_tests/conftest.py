"""Prevent accidental provider traffic during offline unit tests."""

import ipaddress
import socket
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def block_external_network(monkeypatch: pytest.MonkeyPatch) -> None:
    resolve = socket.getaddrinfo
    connect = socket.socket.connect

    def is_local(host: Any) -> bool:
        if host in ("localhost", b"localhost"):
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    def guarded_resolve(host: Any, *args: Any, **kwargs: Any) -> Any:
        if not is_local(host):
            raise AssertionError(f"Unexpected DNS lookup in offline test: {host}")
        return resolve(host, *args, **kwargs)

    def guarded_connect(sock: socket.socket, address: Any) -> Any:
        if sock.family in (socket.AF_INET, socket.AF_INET6) and not is_local(
            address[0]
        ):
            raise AssertionError(f"Unexpected connection in offline test: {address}")
        return connect(sock, address)

    monkeypatch.setattr(socket, "getaddrinfo", guarded_resolve)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
