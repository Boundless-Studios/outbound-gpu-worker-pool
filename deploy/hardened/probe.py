"""Run inside each workload container; active denial checks emit only booleans.

Supply OGWP_PROBE_ROLE=agent or comfy. Run separately from production work.
"""

import json
import os
from pathlib import Path
import socket


def blocked_connection(host, port):
    try:
        with socket.create_connection((host, port), timeout=2):
            return False
    except OSError:
        return True


def runtime_answers():
    """The agent's own view of the runtime, independent of the container healthcheck."""
    import urllib.request
    try:
        with urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=5) as response:
            return response.status == 200 and b"devices" in response.read(65536)
    except OSError:
        return False


def denied_write(directory):
    path = Path(directory) / ".ogwp-isolation-probe"
    try:
        with path.open("x") as file:
            file.write("probe")
        path.unlink()
        return False
    except OSError:
        return True


def main():
    role = os.environ["OGWP_PROBE_ROLE"]
    if role not in {"agent", "comfy"}:
        raise ValueError("unknown role")
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
    checks = {
        "non_root": os.getuid() == 65532,
        "no_supplementary_groups": not (set(os.getgroups()) - {65532}),
        "no_new_privileges": status["NoNewPrivs"].strip() == "1",
        "seccomp_enabled": status["Seccomp"].strip() == "2",
        "no_capabilities": int(status["CapEff"].strip(), 16) == 0,
        "no_host_home": not list(Path("/home").iterdir()),
        "no_host_control_sockets": not any(Path(p).exists() for p in
            ["/var/run/docker.sock", "/run/docker.sock", "/run/user/1000/bus", "/mnt/c"]),
        "code_write_denied": denied_write("/opt/comfy" if role == "comfy" else "/opt/ogwp"),
        "metadata_route_denied": blocked_connection("169.254.169.254", 80),
        "direct_internet_denied": blocked_connection("1.1.1.1", 443),
        "direct_internet_ipv6_denied": blocked_connection("2606:4700:4700::1111", 443),
        "host_bridge_route_denied": blocked_connection("172.17.0.1", 22),
    }
    try:
        socket.getaddrinfo("example.com", 443)
        checks["external_dns_denied"] = False
    except OSError:
        checks["external_dns_denied"] = True
    scratch = Path("/scratch/.ogwp-scratch-probe")
    scratch.write_text("probe")
    scratch.unlink()
    checks["scratch_writable"] = True
    if role == "comfy":
        checks["runtime_answers"] = runtime_answers()
        checks["models_write_denied"] = denied_write("/opt/comfy/models")
        checks["no_worker_credential"] = not Path("/run/credentials/worker.json").exists()
    else:
        checks["templates_write_denied"] = denied_write("/opt/ogwp/templates")
        with socket.create_connection(("egress", 3128), timeout=3) as proxy:
            proxy.sendall(b"CONNECT example.com:443 HTTP/1.1\r\n\r\n")
            checks["unapproved_proxy_destination_denied"] = proxy.recv(128).startswith(b"HTTP/1.1 403")
    print(json.dumps({"role": role, "passed": all(checks.values()), "checks": checks}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
