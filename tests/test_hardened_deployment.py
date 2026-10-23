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
                           "Init": True, "RestartPolicy": {"Name": "no"},
                           "Tmpfs": {"/tmp": "size=1g,noexec,nosuid,nodev"}}, "Mounts": [],
            "NetworkSettings": {"Networks": {}}})
    agent, comfy, egress = containers
    agent["HostConfig"]["NetworkMode"] = "container:comfy"
    agent["Config"]["Env"] = ["HTTPS_PROXY=http://egress:3128", "NO_PROXY=localhost,127.0.0.1,::1",
                              "OGWP_COMFY_URL=http://127.0.0.1:8188", "OGWP_WORKER_GPU_TELEMETRY=comfy"]
    comfy["HostConfig"]["Dns"] = ["127.0.0.1"]
    comfy["Config"]["Cmd"] = ["--disable-all-custom-nodes", "--disable-api-nodes"]
    comfy["NetworkSettings"]["Networks"] = {"isolated": {}}
    comfy["Mounts"] = [{"Type": "bind", "Destination": "/opt/comfy/models",
                        "Source": approved["OGWP_MODELS_DIR"], "RW": False}]
    egress["Config"]["Entrypoint"] = ["python", "/opt/ogwp/egress_proxy.py"]
    egress["Config"]["Env"] = ["OGWP_EGRESS_HOSTS=api.example.com"]
    egress["NetworkSettings"]["Networks"] = {"isolated": {}, "outbound": {}}
    networks = {"isolated": {"Internal": True, "Driver": "bridge",
                "Options": {"com.docker.network.bridge.gateway_mode_ipv4": "isolated"},
                "Containers": {"comfy": {}, "egress": {}}}, "outbound": {}}
    return containers, networks, approved


def test_complete_profile_and_prestart_gate():
    containers, networks, approved = deployment()
    assert gate.violations(containers, networks, approved=approved) == []
    containers[0]["State"]["Running"] = False
    assert gate.violations(containers, networks, approved=approved)
    assert gate.violations(containers, networks, approved=approved, allow_stopped_agent=True) == []


@pytest.mark.parametrize("change", ["host_mount", "host_gateway", "mutable_image", "extra_peer",
                                    "unconfined", "unbounded_memory", "docker_restart"])
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


@pytest.mark.parametrize("change", ["none", "content", "extra", "symlink", "traversal", "missing"])
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
