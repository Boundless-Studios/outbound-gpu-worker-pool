"""Fail-closed Docker configuration gate; prints findings, never environment values.

Run on the Docker host after creation and before enrolling/resuming the agent.
This verifies a container boundary, not the host kernel, model trust, or VPN.
"""

import argparse
import json
import hashlib
from pathlib import Path
import re
import subprocess

SERVICES = {"agent", "comfy", "egress"}
# The ComfyUI healthcheck must be this exact runtime check: any other command can
# report healthy while nothing answers on the port the agent uses.
COMFY_HEALTHCHECK = ["CMD", "/opt/comfy/venv/bin/python", "-c",
                     "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8188/system_stats', timeout=3).close()"]
REQUIRED_MOUNTS = {"agent": {"/run/credentials/worker.json"}, "comfy": {"/opt/comfy/models"}}
MOUNTS = {
    "agent": {"/run/credentials/worker.json"},
    "comfy": {"/opt/comfy/models", "/usr/lib/wsl/lib", "/usr/lib/wsl/drivers"},
    "egress": set(),
}


def violations(containers: list[dict], networks: dict[str, dict], *, allow_stopped_agent=False, approved=None) -> list[str]:
    approved = approved or {}
    errors = []
    indexed = {}
    for c in containers:
        name = c.get("Config", {}).get("Labels", {}).get("com.docker.compose.service")
        if name not in SERVICES or name in indexed:
            errors.append("unexpected or duplicate service")
        indexed[name] = c
    if set(indexed) != SERVICES:
        errors.append("all three isolated services must exist")
    for name, c in indexed.items():
        cfg, host = c.get("Config", {}), c.get("HostConfig", {})
        def require(condition, reason):
            if not condition:
                errors.append(f"{name}: {reason}")
        if name != "agent" or not allow_stopped_agent:
            require(c.get("State", {}).get("Running") is True, "service is not running")
        require(cfg.get("User") == "65532:65532", "dedicated non-root container user required")
        require(bool(re.fullmatch(r"(?:sha256:|.+@sha256:)[0-9a-f]{64}", cfg.get("Image", ""))),
                "image must be an immutable digest")
        require(cfg.get("Image") == approved.get("OGWP_COMFY_IMAGE" if name == "comfy" else "OGWP_AGENT_IMAGE"),
                "image must match the operator-approved profile")
        require(host.get("ReadonlyRootfs") is True, "root filesystem must be read-only")
        require(not host.get("Privileged"), "privileged mode forbidden")
        require(not host.get("CapAdd") and "ALL" in (host.get("CapDrop") or []), "all capabilities must be dropped")
        opts = host.get("SecurityOpt") or []
        require(any(x in opts for x in ["no-new-privileges:true", "no-new-privileges"]), "no-new-privileges required")
        require(set(opts) <= {"no-new-privileges:true", "no-new-privileges"}, "security profile overrides forbidden")
        require(not host.get("PortBindings") and not host.get("PublishAllPorts"), "published ports forbidden")
        # Only the container's own namespaces: "container:<id>" would join another
        # workload's processes or shared memory just as "host" joins the host's.
        require(host.get("PidMode", "") == "" and host.get("IpcMode") == "private",
                "private process and IPC namespaces required")
        require(host.get("NetworkMode") != "host", "host networking forbidden")
        require(host.get("CgroupnsMode") == "private", "private cgroup namespace required")
        require(0 < host.get("PidsLimit", 0) <= 256, "PID bound missing or excessive")
        memory_max, cpu_max = {"comfy": (28 * 1024**3, 8 * 10**9), "agent": (1024**3, 2 * 10**9), "egress": (128 * 1024**2, 500000000)}.get(name, (0, 0))
        require(0 < host.get("Memory", 0) <= memory_max and 0 < host.get("NanoCpus", 0) <= cpu_max, "memory and CPU bounds missing or excessive")
        require(host.get("MemorySwap") == host.get("Memory") and host.get("Memory", 0) > 0,
                "workload swap must be disabled")
        require(host.get("Init") is True, "child reaping required")
        require(host.get("RestartPolicy", {}).get("Name") == "no", "ungated Docker restart policy forbidden")
        require(not host.get("GroupAdd"), "supplementary host groups forbidden")
        require(not host.get("VolumesFrom"), "inherited volumes forbidden")
        require(not host.get("DeviceRequests"), "unreviewed device requests forbidden")
        require(not host.get("DeviceCgroupRules"), "device cgroup overrides forbidden")
        for device in host.get("Devices") or []:
            require(name == "comfy" and device.get("PathOnHost") == "/dev/dxg"
                    and device.get("PathInContainer") == "/dev/dxg"
                    and device.get("CgroupPermissions") == "rw", "unreviewed device mount")
        if name == "comfy":
            # This profile is WSL-only: without the GPU device the runtime would start,
            # advertise its GPU and fail (or run without the GPU) every job it leases.
            require(any(d.get("PathOnHost") == "/dev/dxg" and d.get("PathInContainer") == "/dev/dxg"
                        and d.get("CgroupPermissions") == "rw" for d in host.get("Devices") or []),
                    "WSL GPU device required")
        present = {m.get("Destination") for m in c.get("Mounts", []) if m.get("Type") != "tmpfs"}
        # Verified model bytes mean nothing unless the runtime actually loads them.
        require(REQUIRED_MOUNTS.get(name, set()) <= present, "required host mount missing")
        for mount in c.get("Mounts", []):
            if mount.get("Type") == "tmpfs":
                continue
            require(mount.get("Type") == "bind" and mount.get("Destination") in MOUNTS.get(name, set())
                    and not mount.get("RW"), "unreviewed or writable host mount")
            destination = mount.get("Destination")
            expected_source = {"/opt/comfy/models": approved.get("OGWP_MODELS_DIR"),
                               "/run/credentials/worker.json": approved.get("OGWP_WORKER_CREDENTIAL"),
                               "/usr/lib/wsl/lib": "/usr/lib/wsl/lib",
                               "/usr/lib/wsl/drivers": "/usr/lib/wsl/drivers"}.get(destination)
            require(expected_source is not None and mount.get("Source") == str(Path(expected_source).resolve()),
                    "host mount source differs from approved profile")
        tmpfs = host.get("Tmpfs") or {}
        require(bool(tmpfs) and all("size=" in v for v in tmpfs.values()), "scratch mounts must be bounded")
        allowed_tmpfs = {"agent": {"/scratch", "/tmp"}, "comfy": {"/scratch", "/tmp", "/home", "/jit"},
                         "egress": {"/tmp"}}.get(name, set())
        require(set(tmpfs) <= allowed_tmpfs, "unexpected writable temporary filesystem")
        require(all("nosuid" in v and "nodev" in v and ("noexec" in v or k == "/jit")
                    for k, v in tmpfs.items()), "scratch execution or device policy drift")
        env = dict(x.split("=", 1) for x in cfg.get("Env", []) if "=" in x)
        if name == "agent":
            comfy_id = indexed.get("comfy", {}).get("Id", "missing")
            require(host.get("NetworkMode") == "container:" + comfy_id, "agent must share only the isolated ComfyUI network")
            require(env.get("OGWP_COMFY_START_COMMAND", "") == "", "host runtime start commands forbidden")
            require(env.get("HTTPS_PROXY") == "http://egress:3128", "bounded egress proxy required")
            require(env.get("NO_PROXY") == "localhost,127.0.0.1,::1", "proxy bypass must be loopback only")
            require(env.get("OGWP_COMFY_URL") == "http://127.0.0.1:8188", "loopback runtime required")
            require(env.get("OGWP_WORKER_GPU_TELEMETRY") == "comfy", "GPU telemetry must not require host device access")
        elif name == "comfy":
            attached = c.get("NetworkSettings", {}).get("Networks", {})
            require(len(attached) == 1, "ComfyUI must have exactly one network")
            for network_name in attached:
                network = networks.get(network_name, {})
                options = network.get("Options", {})
                require(network.get("Internal") is True and network.get("Driver") == "bridge"
                        and options.get("com.docker.network.bridge.gateway_mode_ipv4") == "isolated"
                        and options.get("com.docker.network.bridge.gateway_mode_ipv6") == "isolated",
                        "network must deny external routes and host gateway access")
            require(host.get("Dns") == ["127.0.0.1"], "external DNS forwarding forbidden")
            require(c.get("State", {}).get("Health", {}).get("Status") == "healthy", "runtime is not healthy")
            require((cfg.get("Healthcheck") or {}).get("Test") == COMFY_HEALTHCHECK,
                    "the runtime healthcheck must be the reviewed /system_stats check")
            require("--disable-all-custom-nodes" in cfg.get("Cmd", [])
                    and "--disable-api-nodes" in cfg.get("Cmd", []), "unreviewed custom/API nodes must be disabled")
        elif name == "egress":
            attached = c.get("NetworkSettings", {}).get("Networks", {})
            require(len(attached) == 2, "egress proxy requires exactly two networks")
            require(cfg.get("Entrypoint") == ["python", "/opt/ogwp/egress_proxy.py"], "reviewed egress relay required")
            require(c.get("Image") == indexed.get("agent", {}).get("Image"), "proxy must use the approved agent image")
            require(bool(env.get("OGWP_EGRESS_HOSTS")) and "*" not in env.get("OGWP_EGRESS_HOSTS", ""),
                    "explicit egress allowlist required")
            require(env.get("OGWP_EGRESS_HOSTS") == approved.get("OGWP_EGRESS_HOSTS"),
                    "egress allowlist differs from approved profile")
    comfy_networks = set(indexed.get("comfy", {}).get("NetworkSettings", {}).get("Networks", {}))
    egress_networks = set(indexed.get("egress", {}).get("NetworkSettings", {}).get("Networks", {}))
    if not comfy_networks or not comfy_networks < egress_networks:
        errors.append("proxy must bridge the isolated workload network and only one outbound network")
    for network_name in egress_networks - comfy_networks:
        network = networks.get(network_name, {})
        # The relay listens on all its interfaces, so its outbound network must be a
        # host-local bridge nothing else can join or route into.
        if (network.get("Driver") != "bridge" or network.get("Scope") != "local" or network.get("Attachable")
                or network.get("Ingress") or network.get("ConfigFrom", {}).get("Network")
                or network.get("Internal")):
            errors.append("egress outbound network must be a private local bridge")
    for network_name in comfy_networks:
        network = networks.get(network_name, {})
        expected = {indexed.get("comfy", {}).get("Id"), indexed.get("egress", {}).get("Id")}
        if set(network.get("Containers", {})) - expected:
            errors.append("unexpected container on isolated workload network")
    return errors


def verify_models(root: Path, manifest: Path) -> list[str]:
    """Hash an exact curated tree before starting workloads; never follow symlinks."""
    try:
        records = json.loads(manifest.read_text())
        if not isinstance(records, list) or not records or root.is_symlink():
            return ["invalid model manifest or root"]
        expected = {}
        for record in records:
            name = record["file"]
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or name in expected or path.suffix != ".safetensors":
                return ["invalid model manifest path"]
            expected[name] = record
        files = {}
        for path in root.rglob("*"):
            if path.is_symlink():
                return ["model tree contains symlink"]
            if path.is_file():
                files[path.relative_to(root).as_posix()] = path
        if files.keys() != expected.keys():
            return ["model inventory differs from approved manifest"]
        for name, path in files.items():
            if path.stat().st_size != expected[name]["bytes"]:
                return ["model size differs from approved manifest"]
            with path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            if digest != expected[name]["sha256"]:
                return ["model content differs from approved manifest"]
        return []
    except (OSError, ValueError, TypeError, KeyError):
        return ["model integrity verification failed"]


def docker(*args):
    return subprocess.check_output(["docker", *args], text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-models", action="store_true")
    parser.add_argument("--models-only", action="store_true")
    parser.add_argument("--project", default="ogwp-hardened")
    parser.add_argument("--allow-stopped-agent", action="store_true", help="pre-start gate; only the agent may be stopped")
    parser.add_argument("--profile", type=Path, default=Path(__file__).resolve().parents[1] / "profile.env")
    args = parser.parse_args()
    if args.models_only:
        approved = dict(line.split("=", 1) for line in args.profile.read_text().splitlines()
                        if line and not line.startswith("#") and "=" in line)
        errors = verify_models(Path(approved["OGWP_MODELS_DIR"]), args.profile.parent / "model-manifest.json")
        print(json.dumps({"passed": not errors, "violations": errors}))
        return 1 if errors else 0
    ids = docker("ps", "-aq", "--filter", f"label=com.docker.compose.project={args.project}").split()
    containers = json.loads(docker("inspect", *ids)) if ids else []
    names = {name for c in containers for name in c.get("NetworkSettings", {}).get("Networks", {})}
    networks = {n["Name"]: n for n in json.loads(docker("network", "inspect", *sorted(names)))} if names else {}
    # This private file is trusted operator configuration, never workload input.
    approved = dict(line.split("=", 1) for line in args.profile.read_text().splitlines()
                    if line and not line.startswith("#") and "=" in line)
    errors = violations(containers, networks, allow_stopped_agent=args.allow_stopped_agent, approved=approved)
    if args.verify_models:
        errors += verify_models(Path(approved["OGWP_MODELS_DIR"]), args.profile.parent / "model-manifest.json")
    print(json.dumps({"passed": not errors, "violations": errors}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
