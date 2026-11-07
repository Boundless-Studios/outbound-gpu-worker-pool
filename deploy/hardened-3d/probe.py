"""Run inside each 3D workload container; active denial checks emit only booleans.

Supply OGWP_PROBE_ROLE=agent or engine. Run separately from production work.
"""

import json
import os
from pathlib import Path
import socket
import urllib.request

MODELS = "/opt/trellis2-refiner/models"
MODEL_MOUNTS = (f"{MODELS}/visualbruno", f"{MODELS}/microsoft/TRELLIS-image-large",
                f"{MODELS}/facebook", "/opt/rembg/models")
CODE = {"engine": ("/opt/aismith", "/opt/aismith/.venv", "/opt/trellis2-refiner/ComfyUI",
                   "/opt/trellis2-refiner/.venv-cu128", "/opt/blender"),
        "agent": ("/opt/ogwp",)}
CREDENTIAL = "/run/credentials/worker.json"
ENGINE_PORT = 8190


def blocked_connection(host, port):
    try:
        with socket.create_connection((host, port), timeout=2):
            return False
    except OSError:
        return True


def denied_write(directory):
    path = Path(directory) / ".ogwp-isolation-probe"
    try:
        with path.open("x") as file:
            file.write("probe")
        path.unlink()
        return False
    except OSError:
        return True


def writable(directory):
    path = Path(directory) / ".ogwp-scratch-probe"
    try:
        path.write_text("probe")
        path.unlink()
        return True
    except OSError:
        return False


def own_address():
    """This namespace's workload-network address, from Docker's /etc/hosts entry."""
    try:
        address = socket.gethostbyname(socket.gethostname())
    except OSError:
        return None
    return None if address.startswith("127.") else address


def main():
    role = os.environ["OGWP_PROBE_ROLE"]
    if role not in {"agent", "engine"}:
        raise ValueError("unknown role")
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
    checks = {
        "non_root": os.getuid() == 65532,
        "no_supplementary_groups": not (set(os.getgroups()) - {65532}),
        "no_new_privileges": status["NoNewPrivs"].strip() == "1",
        "seccomp_enabled": status["Seccomp"].strip() == "2",
        "no_capabilities": int(status["CapEff"].strip(), 16) == 0,
        "no_host_home": not Path("/home").exists() or not list(Path("/home").iterdir()),
        "no_host_control_sockets": not any(Path(p).exists() for p in
            ["/var/run/docker.sock", "/run/docker.sock", "/run/user/1000/bus", "/mnt/c", "/mnt/k", "/mnt/wsl"]),
        "code_write_denied": all(denied_write(p) for p in CODE[role] if Path(p).exists())
            and all(Path(p).exists() for p in CODE[role]),
        "metadata_route_denied": blocked_connection("169.254.169.254", 80),
        "direct_internet_denied": blocked_connection("1.1.1.1", 443),
        "host_bridge_route_denied": blocked_connection("172.17.0.1", 22),
        "scratch_writable": writable("/scratch"),
    }
    try:
        socket.getaddrinfo("example.com", 443)
        checks["external_dns_denied"] = False
    except OSError:
        checks["external_dns_denied"] = True
    if role == "engine":
        checks["gpu_device_present"] = Path("/dev/dxg").exists()
        checks["models_write_denied"] = all(Path(p).is_dir() and denied_write(p) for p in MODEL_MOUNTS)
        # The engine's hash cache and the wrapper's helper config: tmpfs, not models.
        checks["weight_cache_writable"] = writable(MODELS) and writable(f"{MODELS}/microsoft/TRELLIS.2-4B")
        checks["jit_cache_writable"] = writable("/jit")
        checks["no_worker_credential"] = (not Path(CREDENTIAL).exists() and not any(
            k in os.environ for k in ("GOOGLE_APPLICATION_CREDENTIALS", "OGWP_WORKER_TOKEN", "HF_TOKEN")))
        checks["no_proxy_configured"] = not any(k.lower().endswith("_proxy") for k in os.environ)
        address = own_address()
        # The runner answers on loopback only, never on the workload network.
        checks["runner_loopback_only"] = (address is not None and blocked_connection(address, ENGINE_PORT)
                                          and not blocked_connection("127.0.0.1", ENGINE_PORT))
    else:
        checks["no_gpu_device"] = not Path("/dev/dxg").exists() and not list(Path("/dev").glob("nvidia*"))
        checks["no_gpu_driver_libraries"] = not Path("/usr/lib/wsl").exists()
        checks["no_model_stores"] = not Path(MODELS).exists() and not Path("/opt/rembg").exists()
        checks["credential_readable"] = os.access(CREDENTIAL, os.R_OK)
        checks["credential_write_denied"] = not os.access(CREDENTIAL, os.W_OK)
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(f"http://127.0.0.1:{ENGINE_PORT}/health", timeout=3) as response:
                checks["engine_runner_on_loopback"] = json.load(response).get("ok") is True
        except (OSError, ValueError):
            checks["engine_runner_on_loopback"] = False
        with socket.create_connection(("egress", 3128), timeout=3) as proxy:
            proxy.sendall(b"CONNECT example.com:443 HTTP/1.1\r\n\r\n")
            checks["unapproved_proxy_destination_denied"] = proxy.recv(128).startswith(b"HTTP/1.1 403")
    print(json.dumps({"role": role, "passed": all(checks.values()), "checks": checks}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
