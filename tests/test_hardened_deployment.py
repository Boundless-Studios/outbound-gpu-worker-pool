"""Regression checks for the deployment trust boundary and CONNECT policy."""

import importlib.util
from pathlib import Path

import pytest


def module(name):
    path = Path(__file__).parents[1] / "deploy" / "hardened" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


proxy = module("egress_proxy")
gate = module("verify")


@pytest.mark.parametrize("authority", ["127.0.0.1:443", "metadata.google.internal:443",
    "api.example.com.evil.test:443", "api.example.com:80", "user@api.example.com:443"])
def test_proxy_rejects_unapproved_destinations(authority):
    with pytest.raises(ValueError):
        proxy.destination(f"CONNECT {authority} HTTP/1.1\r\n\r\n".encode(), frozenset({"api.example.com"}))


def test_proxy_accepts_only_exact_https_target():
    assert proxy.destination(b"CONNECT api.example.com:443 HTTP/1.1\r\n\r\n",
                             frozenset({"api.example.com"})) == "api.example.com"
    with pytest.raises(ValueError):
        proxy.destination(b"GET https://api.example.com/ HTTP/1.1\r\n\r\n", frozenset({"api.example.com"}))


@pytest.mark.parametrize("hosts", ["", "*.example.com", "api.example.com,", "localhost", "a..example.com"])
def test_allowlist_fails_closed(hosts):
    with pytest.raises(ValueError):
        proxy.allowed_hosts(hosts)


@pytest.mark.asyncio
async def test_dns_rebinding_and_mixed_private_answers_are_denied(monkeypatch):
    import asyncio
    async def addresses(*args, **kwargs):
        return [(2, 1, 6, "", ("8.8.8.8", 443)), (2, 1, 6, "", ("169.254.169.254", 443))]
    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", addresses)
    with pytest.raises(ValueError, match="nonpublic"):
        await proxy.public_addresses("api.example.com")


def test_missing_services_never_pass_gate():
    assert gate.violations([], {})


def test_host_socket_and_privileges_fail_gate():
    c = {"Config": {"Labels": {"com.docker.compose.service": "agent"}},
         "HostConfig": {"Privileged": True, "CapAdd": ["SYS_ADMIN"]},
         "Mounts": [{"Type": "bind", "Destination": "/var/run/docker.sock", "RW": True}]}
    errors = gate.violations([c], {})
    assert "agent: privileged mode forbidden" in errors
    assert "agent: unreviewed or writable host mount" in errors
    assert "agent: all capabilities must be dropped" in errors


HEALTHCHECK = ["CMD", "/opt/comfy/venv/bin/python", "-c",
               "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8188/system_stats', timeout=3).close()"]


def deployment():
    approved = {"OGWP_AGENT_IMAGE": "sha256:" + "a" * 64,
                "OGWP_COMFY_IMAGE": "sha256:" + "b" * 64,
                "OGWP_EGRESS_HOSTS": "api.example.com",
                "OGWP_MODELS_DIR": "/srv/ogwp/models",
                "OGWP_WORKER_CREDENTIAL": "/srv/ogwp/credentials/worker.json"}
    containers = []
    for name in ["agent", "comfy", "egress"]:
        image = approved["OGWP_COMFY_IMAGE" if name == "comfy" else "OGWP_AGENT_IMAGE"]
        containers.append({"Id": name, "Image": image,
            "Config": {"Labels": {"com.docker.compose.service": name}, "Image": image,
                       "User": "65532:65532", "Env": []},
            "State": {"Running": True, "Health": {"Status": "healthy"}},
            "HostConfig": {"ReadonlyRootfs": True, "CapDrop": ["ALL"],
                           "SecurityOpt": ["no-new-privileges"], "CgroupnsMode": "private",
                           "PidsLimit": 256, "Memory": 1024, "MemorySwap": 1024, "NanoCpus": 1000,
                           "Init": True, "RestartPolicy": {"Name": "no"}, "IpcMode": "private",
                           "Tmpfs": {"/tmp": "size=1g,noexec,nosuid,nodev"}}, "Mounts": [],
            "NetworkSettings": {"Networks": {}}})
    agent, comfy, egress = containers
    agent["HostConfig"]["NetworkMode"] = "container:comfy"
    agent["Mounts"] = [{"Type": "bind", "Destination": "/run/credentials/worker.json",
                        "Source": approved["OGWP_WORKER_CREDENTIAL"], "RW": False}]
    agent["Config"]["Entrypoint"] = ["python", "-m", "outbound_gpu_worker_pool.agent_main"]
    agent["Config"]["Env"] = ["HTTPS_PROXY=http://egress:3128", "NO_PROXY=localhost,127.0.0.1,::1",
                              "OGWP_COMFY_URL=http://127.0.0.1:8188", "OGWP_WORKER_GPU_TELEMETRY=comfy"]
    comfy["HostConfig"]["Dns"] = ["127.0.0.1"]
    comfy["HostConfig"]["Devices"] = [{"PathOnHost": "/dev/dxg", "PathInContainer": "/dev/dxg",
                                       "CgroupPermissions": "rw"}]
    comfy["Config"]["Healthcheck"] = {"Test": list(HEALTHCHECK)}
    comfy["Config"]["Cmd"] = ["--disable-all-custom-nodes", "--disable-api-nodes"]
    comfy["NetworkSettings"]["Networks"] = {"isolated": {}}
    comfy["Mounts"] = [{"Type": "bind", "Destination": "/opt/comfy/models",
                        "Source": approved["OGWP_MODELS_DIR"], "RW": False}]
    egress["Config"]["Entrypoint"] = ["python", "/opt/ogwp/egress_proxy.py"]
    egress["Config"]["Env"] = ["OGWP_EGRESS_HOSTS=api.example.com"]
    egress["NetworkSettings"]["Networks"] = {"isolated": {}, "outbound": {}}
    networks = {"isolated": {"Internal": True, "Driver": "bridge",
                "Options": {"com.docker.network.bridge.gateway_mode_ipv4": "isolated",
                            "com.docker.network.bridge.gateway_mode_ipv6": "isolated"},
                "Containers": {"comfy": {}, "egress": {}}},
                "outbound": {"Driver": "bridge", "Scope": "local", "Internal": False, "Attachable": False}}
    return containers, networks, approved


def test_complete_profile_and_prestart_gate():
    containers, networks, approved = deployment()
    assert gate.violations(containers, networks, approved=approved) == []
    # Compose sets the IPC mode explicitly so the gate never depends on the daemon default.
    compose = (Path(__file__).parents[1] / "deploy/hardened/compose.yaml").read_text()
    assert "  ipc: private\n" in compose
    containers[0]["State"]["Running"] = False
    assert gate.violations(containers, networks, approved=approved)
    assert gate.violations(containers, networks, approved=approved, allow_stopped_agent=True) == []


@pytest.mark.parametrize("change", ["host_mount", "host_gateway", "mutable_image", "extra_peer",
                                    "unconfined", "unbounded_memory", "docker_restart", "shared_pid",
                                    "shared_ipc", "host_ipc", "missing_models", "missing_credential",
                                    "ipv6_gateway", "missing_gpu", "trivial_healthcheck",
                                    "egress_macvlan", "egress_attachable", "egress_swarm",
                                    "agent_entrypoint", "agent_arguments"])
def test_configuration_drift_fails_closed(change):
    containers, networks, approved = deployment()
    if change == "host_mount":
        containers[1]["Mounts"][0]["Source"] = "/"
    elif change == "host_gateway":
        networks["isolated"]["Options"].clear()
    elif change == "mutable_image":
        containers[0]["Config"]["Image"] = "agent:latest"
    elif change == "extra_peer":
        networks["isolated"]["Containers"]["unrelated"] = {}
    elif change == "unconfined":
        containers[1]["HostConfig"]["SecurityOpt"].append("seccomp=unconfined")
    elif change == "unbounded_memory":
        containers[1]["HostConfig"]["Memory"] = 0
    elif change == "docker_restart":
        containers[0]["HostConfig"]["RestartPolicy"]["Name"] = "always"
    elif change == "shared_pid":
        containers[0]["HostConfig"]["PidMode"] = "container:comfy"
    elif change == "shared_ipc":
        containers[0]["HostConfig"]["IpcMode"] = "container:comfy"
    elif change == "host_ipc":
        containers[1]["HostConfig"]["IpcMode"] = "host"
    elif change == "missing_models":
        containers[1]["Mounts"] = []
    elif change == "missing_credential":
        containers[0]["Mounts"] = []
    elif change == "ipv6_gateway":
        networks["isolated"]["Options"]["com.docker.network.bridge.gateway_mode_ipv6"] = "nat"
    elif change == "missing_gpu":
        containers[1]["HostConfig"]["Devices"] = []
    elif change == "trivial_healthcheck":
        containers[1]["Config"]["Healthcheck"] = {"Test": ["CMD", "true"]}
    elif change == "egress_macvlan":
        networks["outbound"]["Driver"] = "macvlan"
    elif change == "egress_attachable":
        networks["outbound"]["Attachable"] = True
    elif change == "egress_swarm":
        networks["outbound"].update(Driver="overlay", Scope="swarm")
    elif change == "agent_entrypoint":
        containers[0]["Config"]["Entrypoint"] = ["sleep", "infinity"]
    elif change == "agent_arguments":
        containers[0]["Config"]["Cmd"] = ["--something-else"]
    assert gate.violations(containers, networks, approved=approved)


def test_legacy_installer_requires_explicit_development_opt_in():
    import os
    import subprocess
    env = dict(os.environ)
    env.pop("OGWP_ALLOW_UNSANDBOXED", None)
    script = Path(__file__).parents[1] / "deploy/agent/install.sh"
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert "deploy/hardened" in result.stderr


@pytest.mark.parametrize("field,value", [("Memory", 128 * 1024**3), ("NanoCpus", 128 * 10**9)])
def test_excessive_resource_limits_rejected(field, value):
    containers, networks, approved = deployment()
    containers[1]["HostConfig"][field] = value
    assert gate.violations(containers, networks, approved=approved)


def test_allowlist_drift_rejected():
    containers, networks, approved = deployment()
    containers[2]["Config"]["Env"] = ["OGWP_EGRESS_HOSTS=evil.example.com"]
    assert gate.violations(containers, networks, approved=approved)


@pytest.mark.parametrize("change", ["none", "content", "extra", "symlink", "traversal", "missing", "fifo", "socket"])
def test_model_content_integrity(tmp_path, change):
    import hashlib
    import json
    root = tmp_path / "models"
    root.mkdir()
    model = root / "test.safetensors"
    model.write_bytes(b"test")
    records = [{"file": model.name, "bytes": 4, "sha256": hashlib.sha256(b"test").hexdigest()}]
    if change == "content":
        model.write_bytes(b"evil")
    elif change == "extra":
        (root / "extra.safetensors").write_bytes(b"test")
    elif change == "symlink":
        (root / "link").symlink_to(model)
    elif change == "traversal":
        records[0]["file"] = "../test.safetensors"
    elif change == "missing":
        model.unlink()
    elif change == "fifo":
        import os
        os.mkfifo(root / "pipe")
    elif change == "socket":
        import os
        import socket
        # Bound by a relative name: macOS caps a socket path at 104 bytes.
        here = os.getcwd()
        os.chdir(root)
        try:
            listener = socket.socket(socket.AF_UNIX)
            listener.bind("s")
            listener.close()
        finally:
            os.chdir(here)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(records))
    assert bool(gate.verify_models(root, manifest)) == (change != "none")


@pytest.mark.asyncio
async def test_proxy_slow_header_times_out_and_releases_slot(monkeypatch):
    import asyncio
    monkeypatch.setattr(proxy, 'CONNECT_TIMEOUT', 0.02)
    relay = proxy.Proxy(frozenset({'api.example.com'}))
    server = await asyncio.start_server(relay.handle, '127.0.0.1', 0)
    try:
        reader, writer = await asyncio.open_connection('127.0.0.1', server.sockets[0].getsockname()[1])
        writer.write(b'CONNECT ')
        await writer.drain()
        response = await asyncio.wait_for(reader.read(), 1)
        assert response.startswith(b'HTTP/1.1 403')
        assert relay.active == 0
        writer.close()
        await writer.wait_closed()
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize('address', ['::1', 'fe80::1', 'fc00::1', '::ffff:127.0.0.1', 'ff02::1'])
async def test_proxy_denies_nonpublic_ipv6(monkeypatch, address):
    import asyncio
    async def lookup(*args, **kwargs):
        return [(10, 1, 6, '', (address, 443, 0, 0))]
    monkeypatch.setattr(asyncio.get_running_loop(), 'getaddrinfo', lookup)
    with pytest.raises(ValueError, match='nonpublic'):
        await proxy.public_addresses('api.example.com')


def test_no_job_can_be_leased_before_both_probes_pass():
    """The agent leases on start, so its probe runs in a one-off container first."""
    script = (Path(__file__).parents[1] / "deploy/hardened/run.sh").read_text()
    lines = [line for line in script.splitlines() if line and not line.startswith("#")]
    start = next(i for i, line in enumerate(lines) if line.endswith(" start agent"))
    agent_probe = next(i for i, line in enumerate(lines) if "OGWP_PROBE_ROLE=agent" in line)
    comfy_probe = next(i for i, line in enumerate(lines) if "OGWP_PROBE_ROLE=comfy" in line)
    prestart_gate = next(i for i, line in enumerate(lines) if "--allow-stopped-agent" in line)
    assert max(agent_probe, comfy_probe, prestart_gate) < start
    assert "run --rm -T --no-deps" in lines[agent_probe] and "--entrypoint python agent" in lines[agent_probe]
    assert not any("exec" in line and "agent_id" in line for line in lines[start:])


def hello(server_name):
    """A real ClientHello from Python's TLS stack, as TLS records."""
    import ssl
    context = ssl.create_default_context()
    context.check_hostname = server_name is not None
    if server_name is None:
        context.verify_mode = ssl.CERT_NONE
    incoming, outgoing = ssl.MemoryBIO(), ssl.MemoryBIO()
    connection = context.wrap_bio(incoming, outgoing, server_hostname=server_name)
    with pytest.raises(ssl.SSLWantReadError):
        connection.do_handshake()
    return outgoing.read()


async def records(data, *, split=None):
    import asyncio
    reader = asyncio.StreamReader()
    if split:
        # Re-frame the handshake across two TLS records, as a client may.
        body = data[5:]
        data = (b"\x16" + data[1:3] + split.to_bytes(2, "big") + body[:split]
                + b"\x16" + data[1:3] + (len(body) - split).to_bytes(2, "big") + body[split:])
    reader.feed_data(data)
    reader.feed_eof()
    return await proxy.read_client_hello(reader)


@pytest.mark.asyncio
@pytest.mark.parametrize("split", [None, 40])
async def test_the_tls_server_name_is_read_from_the_client_hello(split):
    raw, body = await records(hello("api.example.com"), split=split)
    assert proxy.client_hello_server_name(body) == "api.example.com"
    assert raw.startswith(b"\x16")


@pytest.mark.asyncio
async def test_a_client_hello_without_a_server_name_is_refused():
    _, body = await records(hello(None))
    with pytest.raises(ValueError, match="SNI"):
        proxy.client_hello_server_name(body)


def test_an_encrypted_client_hello_is_refused():
    import asyncio
    _, body = asyncio.run(records(hello("api.example.com")))
    # Append an encrypted_client_hello extension (0xfe0d) to the extension list.
    position = 2 + 32
    position += 1 + body[position]
    position += 2 + int.from_bytes(body[position:position + 2], "big")
    position += 1 + body[position]
    length = int.from_bytes(body[position:position + 2], "big")
    extension = b"\xfe\x0d\x00\x01\x00"
    forged = (body[:position] + (length + len(extension)).to_bytes(2, "big")
              + body[position + 2:position + 2 + length] + extension)
    with pytest.raises(ValueError, match="encrypted SNI"):
        proxy.client_hello_server_name(forged)


@pytest.mark.asyncio
@pytest.mark.parametrize("server_name,relayed", [("api.example.com", True), ("other.example.net", False)])
async def test_the_relay_only_tunnels_tls_to_the_connect_host(monkeypatch, server_name, relayed):
    import asyncio
    received = bytearray()

    async def upstream_handler(reader, writer):
        received.extend(await reader.read(65536))
        writer.close()

    upstream = await asyncio.start_server(upstream_handler, "127.0.0.1", 0)
    port = upstream.sockets[0].getsockname()[1]

    async def connect_public(host):
        return await asyncio.open_connection("127.0.0.1", port)

    monkeypatch.setattr(proxy, "connect_public", connect_public)
    relay = proxy.Proxy(frozenset({"api.example.com"}))
    server = await asyncio.start_server(relay.handle, "127.0.0.1", 0)
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", server.sockets[0].getsockname()[1])
        writer.write(b"CONNECT api.example.com:443 HTTP/1.1\r\n\r\n")
        await writer.drain()
        assert (await reader.readuntil(b"\r\n\r\n")).startswith(b"HTTP/1.1 200")
        client_hello = hello(server_name)
        writer.write(client_hello)
        await writer.drain()
        await asyncio.wait_for(reader.read(), 2)
        writer.close()
        await asyncio.sleep(0.05)
        assert bytes(received) == (client_hello if relayed else b"")
        assert relay.active == 0
    finally:
        server.close()
        await server.wait_closed()
        upstream.close()
        await upstream.wait_closed()


def test_the_compose_healthcheck_is_the_one_the_gate_requires():
    compose = (Path(__file__).parents[1] / "deploy/hardened/compose.yaml").read_text()
    assert gate.COMFY_HEALTHCHECK == HEALTHCHECK
    command, executable, flag, code = HEALTHCHECK
    assert f'test: [{command}, {executable}, {flag}, "{code}"]' in compose


def test_systemd_tears_the_workloads_down_however_the_supervisor_ends():
    directory = Path(__file__).parents[1] / "deploy/hardened"
    unit = (directory / "ogwp-hardened.service").read_text()
    assert "ExecStopPost=/bin/bash %h/.local/share/ogwp-hardened/hardened/stop.sh" in unit
    assert 'trap \'/bin/bash "$HERE/stop.sh"\' EXIT' in (directory / "run.sh").read_text()
    stop = [line for line in (directory / "stop.sh").read_text().splitlines() if "stop --timeout" in line]
    assert len(stop) == 2 and stop[0].endswith("agent || true") and "comfy egress" in stop[1]


def test_the_runtime_probe_fails_when_nothing_answers(monkeypatch):
    import urllib.request
    probe = module("probe")

    def refuse(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    assert probe.runtime_answers() is False


def test_the_unit_waits_for_docker_before_counting_a_failed_start():
    directory = Path(__file__).parents[1] / "deploy/hardened"
    unit = (directory / "ogwp-hardened.service").read_text()
    pre = unit.index("ExecStartPre=/bin/bash %h/.local/share/ogwp-hardened/hardened/wait-docker.sh")
    assert pre < unit.index("ExecStart=/bin/bash")
    assert "TimeoutStartSec=11min" in unit  # longer than the 10-minute wait


@pytest.mark.parametrize("failures,timeout,succeeds", [(0, 5, True), (2, 5, True), (1000, 1, False)])
def test_wait_docker_returns_once_docker_answers_and_gives_up_after_its_deadline(tmp_path, failures, timeout, succeeds):
    import os
    import subprocess
    import sys
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "docker").write_text(f"""#!{sys.executable}
import pathlib, sys
calls = pathlib.Path({str(tmp_path / 'calls')!r})
count = int(calls.read_text()) if calls.exists() else 0
calls.write_text(str(count + 1))
sys.exit(1 if count < {failures} else 0)
""")
    (fake / "docker").chmod(0o755)
    env = {**os.environ, "PATH": f"{fake}{os.pathsep}{os.environ['PATH']}",
           "OGWP_DOCKER_WAIT_SECONDS": str(timeout), "OGWP_DOCKER_WAIT_INTERVAL": "0.1"}
    script = Path(__file__).parents[1] / "deploy/hardened/wait-docker.sh"
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=30)
    assert (result.returncode == 0) is succeeds
    if succeeds:
        assert int((tmp_path / "calls").read_text()) == failures + 1
    else:
        assert "did not become ready" in result.stderr
