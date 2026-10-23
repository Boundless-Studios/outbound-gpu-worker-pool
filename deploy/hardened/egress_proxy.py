"""Bounded HTTPS CONNECT relay. Exact hosts, public DNS results, no URL logging.

Run on a separate dual-network container; workload containers have only an
internal isolated network. TLS stays end-to-end with the original client, but
the tunnel's first TLS ClientHello must name the CONNECT host as its SNI, so a
workload cannot reach another tenant of a shared address through an allowed
name. A ClientHello without SNI or with encrypted SNI (ECH) is refused.
"""

import asyncio
import ipaddress
import os
import re
import socket

HEADER_LIMIT = 8192
CONNECT_TIMEOUT = 10
TUNNEL_TIMEOUT = 1800
MAX_CONNECTIONS = 32
HELLO_LIMIT = 65536
_HANDSHAKE, _CLIENT_HELLO = 0x16, 0x01
_SERVER_NAME, _ENCRYPTED_CLIENT_HELLO = 0x0000, 0xFE0D


def allowed_hosts(value: str) -> frozenset[str]:
    hosts = frozenset(value.split(","))
    if not hosts or any(
        not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", host)
        or "." not in host or ".." in host for host in hosts
    ):
        raise ValueError("configure exact lowercase DNS hostnames, without wildcards")
    return hosts


def destination(header: bytes, hosts: frozenset[str]) -> str:
    lines = header.decode("ascii").split("\r\n")
    method, authority, version = lines[0].split(" ")
    if method != "CONNECT" or version not in {"HTTP/1.0", "HTTP/1.1"}:
        raise ValueError("HTTPS CONNECT required")
    host, port = authority.rsplit(":", 1)
    if host not in hosts or port != "443":
        raise ValueError("destination denied")
    return host


async def read_client_hello(reader) -> tuple[bytes, bytes]:
    """The raw TLS records carrying the first ClientHello, and the ClientHello body
    reassembled across them if it was fragmented."""
    raw = bytearray()
    handshake = bytearray()
    while len(handshake) < 4 or len(handshake) < 4 + int.from_bytes(handshake[1:4], "big"):
        header = await reader.readexactly(5)
        length = int.from_bytes(header[3:5], "big")
        if header[0] != _HANDSHAKE or not 0 < length <= 16384 or len(raw) + 5 + length > HELLO_LIMIT:
            raise ValueError("TLS ClientHello required")
        body = await reader.readexactly(length)
        raw += header + body
        handshake += body
    if handshake[0] != _CLIENT_HELLO:
        raise ValueError("TLS ClientHello required")
    return bytes(raw), bytes(handshake[4:4 + int.from_bytes(handshake[1:4], "big")])


def client_hello_server_name(hello: bytes) -> str:
    """The single DNS name a ClientHello's SNI extension names, lowercased."""
    position = 2 + 32  # legacy version, random
    position += 1 + hello[position]  # session id
    position += 2 + int.from_bytes(hello[position:position + 2], "big")  # cipher suites
    position += 1 + hello[position]  # compression methods
    end = position + 2 + int.from_bytes(hello[position:position + 2], "big")
    if end > len(hello):
        raise ValueError("truncated ClientHello")
    position += 2
    names = []
    while position < end:
        kind = int.from_bytes(hello[position:position + 2], "big")
        size = int.from_bytes(hello[position + 2:position + 4], "big")
        data = hello[position + 4:position + 4 + size]
        position += 4 + size
        if kind == _ENCRYPTED_CLIENT_HELLO:
            raise ValueError("encrypted SNI cannot be checked")
        if kind == _SERVER_NAME:
            offset = 2
            while offset < len(data):
                name_type, name_length = data[offset], int.from_bytes(data[offset + 1:offset + 3], "big")
                if name_type == 0:
                    names.append(data[offset + 3:offset + 3 + name_length].decode("ascii").lower())
                offset += 3 + name_length
    if position != end or len(names) != 1:
        raise ValueError("exactly one SNI host name required")
    return names[0]


async def public_addresses(host: str) -> list[str]:
    results = await asyncio.get_running_loop().getaddrinfo(
        host, 443, type=socket.SOCK_STREAM
    )
    addresses = list(dict.fromkeys(result[4][0] for result in results))
    if not addresses or any(not ipaddress.ip_address(ip).is_global
                            or ipaddress.ip_address(ip).is_multicast for ip in addresses):
        raise ValueError("nonpublic DNS result denied")
    return addresses


async def connect_public(host: str):
    # Connect to the validated literal, never resolve the hostname a second time.
    for address in await public_addresses(host):
        try:
            return await asyncio.open_connection(address, 443)
        except OSError:
            continue
    raise OSError("destination unavailable")


async def pump(reader, writer):
    while data := await reader.read(65536):
        writer.write(data)
        await writer.drain()


class Proxy:
    def __init__(self, hosts: frozenset[str]):
        self.hosts = hosts
        self.active = 0

    async def handle(self, reader, writer):
        if self.active >= MAX_CONNECTIONS:
            writer.close()
            return
        self.active += 1
        upstream = None
        established = False
        tasks = []
        try:
            async with asyncio.timeout(CONNECT_TIMEOUT):
                header = await reader.readuntil(b"\r\n\r\n")
                if len(header) > HEADER_LIMIT:
                    raise ValueError("header too large")
                host = destination(header, self.hosts)
                upstream_reader, upstream = await connect_public(host)
            writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            await writer.drain()
            established = True
            async with asyncio.timeout(CONNECT_TIMEOUT):
                raw, hello = await read_client_hello(reader)
            # The TLS destination, not just the CONNECT line, must be the allowed host.
            if client_hello_server_name(hello) != host:
                raise ValueError("TLS server name differs from the CONNECT host")
            upstream.write(raw)
            await upstream.drain()
            async with asyncio.timeout(TUNNEL_TIMEOUT):
                tasks = [asyncio.create_task(pump(reader, upstream)),
                         asyncio.create_task(pump(upstream_reader, writer))]
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        except (ValueError, UnicodeError, IndexError, OSError, TimeoutError,
                asyncio.IncompleteReadError, asyncio.LimitOverrunError):
            if not established:
                writer.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
        finally:
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if upstream is not None:
                upstream.close()
            writer.close()
            self.active -= 1


async def main():
    proxy = Proxy(allowed_hosts(os.environ["OGWP_EGRESS_HOSTS"]))
    server = await asyncio.start_server(proxy.handle, "0.0.0.0", 3128, limit=HEADER_LIMIT)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
