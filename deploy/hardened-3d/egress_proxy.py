"""Bounded HTTPS CONNECT relay. Exact hosts, public DNS results, no URL logging.

Run on a separate dual-network container; workload containers have only an
internal isolated network. TLS stays end-to-end with the original client.
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
            async with asyncio.timeout(TUNNEL_TIMEOUT):
                tasks = [asyncio.create_task(pump(reader, upstream)),
                         asyncio.create_task(pump(upstream_reader, writer))]
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        except (ValueError, UnicodeError, OSError, TimeoutError,
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
