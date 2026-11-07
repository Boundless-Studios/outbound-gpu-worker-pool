"""Fail-closed Docker configuration gate for the 3D engine worker.

Prints findings, never environment values. Run on the Docker host after
creation and before enrolling/resuming the agent. It enforces the same rules
as the H3 gate (`deploy/hardened/verify.py`) with the engine in ComfyUI's role:

- the GPU device and the model stores are mounted only in `engine`, read-only;
- the worker credential is mounted only in `agent`, and the agent has no GPU;
- the agent shares only the engine's network namespace;
- the engine has exactly one internal, gateway-isolated network and no DNS;
- `egress` bridges exactly that network and one outbound network, with the
  approved host allowlist;
- every image is the operator-approved immutable digest for its service;
- every container is non-root, read-only, capability-free, no-new-privileges,
  bounded in PIDs/memory/CPU/scratch, and has no Docker restart policy;
- the engine runs the reviewed `serve` invocation bound to 127.0.0.1 and is
  healthy.

`--models-only` hashes the mounted model stores against the approved manifest
before the engine can load them. The engine's own `inspect --full` runs too (at
enrolment), but the engine is the untrusted party: the gate never relies on its
self-report. `--manifest-from-lock` derives that manifest from the reviewed
`vendor/weights.lock.json` at the engine tag, so both checks agree by
construction.

This verifies a container boundary, not the host kernel, model provenance or
VPN. The host administrator is trusted.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

PROJECT = "ogwp-hardened-3d"
SERVICES = {"agent", "engine", "egress"}
GiB = 1024**3
MiB = 1024**2

REFINER = "/opt/trellis2-refiner"
MODELS = f"{REFINER}/models"
# The weight-lock subtrees the mesh stage loads, each bound read-only. Their
# parent (`models/`) is a small tmpfs: the engine writes its weight-hash cache
# (`.weights-verified.json`) there, and ComfyUI-Trellis2 copies a helper config
# into `microsoft/TRELLIS.2-4B`, its own tmpfs. Neither is a model.
TRELLIS_SUBTREES = ("visualbruno", "microsoft/TRELLIS-image-large", "facebook")
REMBG_HOME = "/opt/rembg"
REMBG_MODELS = f"{REMBG_HOME}/models"
CREDENTIAL = "/run/credentials/worker.json"
WSL_LIB, WSL_DRIVERS = "/usr/lib/wsl/lib", "/usr/lib/wsl/drivers"
MOUNTS = {
    "agent": {CREDENTIAL},
    "engine": {*(f"{MODELS}/{s}" for s in TRELLIS_SUBTREES), REMBG_MODELS, WSL_LIB, WSL_DRIVERS},
    "egress": set(),
}
TMPFS = {
    "agent": {"/scratch", "/tmp"},
    "engine": {"/scratch", "/tmp", "/home", "/jit", MODELS, f"{MODELS}/microsoft/TRELLIS.2-4B"},
    "egress": {"/tmp"},
}
# Only the bounded JIT cache may hold executable files.
EXEC_TMPFS = {"engine": {"/jit"}}
# (memory ceiling, CPU ceiling in NanoCpus, PID ceiling). The engine's ceiling
# fits the qualification measurements: mesh RSS peaks about 9 GB, plus up to
# 4 GiB scratch, 1 GiB JIT and 1 GiB shm, which tmpfs charges to the same cgroup.
BOUNDS = {
    "engine": (28 * GiB, 8 * 10**9, 1024),
    "agent": (2 * GiB, 2 * 10**9, 256),
    "egress": (128 * MiB, 500_000_000, 256),
}
# Below this the measured mesh working set (and its tmpfs) would be OOM-killed.
ENGINE_MEMORY_FLOOR = 16 * GiB
SHM_MAX = 2 * GiB
IMAGE_KEY = {"agent": "OGWP_AGENT_IMAGE", "egress": "OGWP_AGENT_IMAGE", "engine": "OGWP_ENGINE_IMAGE"}

ENGINE_PYTHON = "/opt/aismith/.venv/bin/python"
ENGINE_PORT = 8190
ENGINE_URL = f"http://127.0.0.1:{ENGINE_PORT}"
ENGINE_COMMAND = ["serve", "--host", "127.0.0.1", "--port", str(ENGINE_PORT), "--work-root", "/scratch/engine"]
ENGINE_SCRATCH_DIRS = ("/scratch/engine", "/scratch/refiner/input", "/scratch/refiner/output",
                       "/scratch/refiner/temp", "/scratch/aismith-data", "/scratch/cache", "/jit/tmp")
ENGINE_ENTRYPOINT = [
    ENGINE_PYTHON, "-c",
    "import os,sys; [os.makedirs(p,exist_ok=True) for p in " + repr(ENGINE_SCRATCH_DIRS) + "]; "
    "os.execv(sys.executable,[sys.executable,'-m','backend.engine',*sys.argv[1:]])",
]
ENGINE_HEALTH = (
    "import json,urllib.request; r=json.load(urllib.request.urlopen("
    f"'{ENGINE_URL}/health', timeout=3)); assert r.get('ok') is True"
)
ENGINE_ENV = {
    "AISMITH_OFFLINE": "1",
    "AISMITH_GPU_PREFLIGHT_MAX_WAIT_SECONDS": "0",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TRELLIS_REFINER_DIR": REFINER,
    "BLENDER_EXE": "/opt/blender/blender",
    "U2NET_HOME": REMBG_HOME,
    "AISMITH_DATA_DIR": "/scratch/aismith-data",
    "PYTHONDONTWRITEBYTECODE": "1",
}
AGENT_ENV = {
    "MVS_WORKER_PROFILE": "mesh3d-24gb-standard",
    "OGWP_ENGINE_URL": ENGINE_URL,
    "OGWP_WORKER_PLUGINS": "aismith-engine",
    "OGWP_WORKER_CONCURRENCY": "1",
    "OGWP_WORKER_WORKSPACE": "/scratch/jobs",
    "HTTPS_PROXY": "http://egress:3128",
    "HTTP_PROXY": "http://egress:3128",
    "NO_PROXY": "localhost,127.0.0.1,::1",
    "GOOGLE_APPLICATION_CREDENTIALS": CREDENTIAL,
}
# The runner's own per-input bound (mesh 256 MiB); a larger download is refused.
AGENT_MAX_INPUT_BYTES = 256 * MiB
# No credential-shaped or proxy variable reaches the engine.
SECRET_NAME = re.compile(r"(TOKEN|SECRET|PASSWORD|CREDENTIAL|_KEY$|^OGWP_WORKER_|PROXY$)", re.I)
NO_NEW_PRIVILEGES = {"no-new-privileges:true", "no-new-privileges"}

# The weight sets the mesh3d-24gb-standard stages load (engine_recipes.MESH3D_RECIPE.weight_ids).
REQUIRED_WEIGHT_SETS = ("trellis2-4b-fp8", "trellis-image-large-ss-decoder", "dinov3-vitl16",
                        "birefnet-general", "birefnet-general-lite")
# Non-pickle formats only; executable checkpoints (.ckpt/.pt/.bin/.pth) never mount.
MODEL_SUFFIXES = {".safetensors", ".json", ".onnx"}
STORES = ("trellis", "rembg")


def _env(container: dict) -> dict[str, str]:
    return dict(x.split("=", 1) for x in container.get("Config", {}).get("Env") or [] if "=" in x)


def _tmpfs_size(options: str) -> str | None:
    match = re.search(r"(?:^|,)size=([^,]+)", options)
    return match.group(1) if match else None


def _expected_sources(approved: dict[str, str]) -> dict[str, str | None]:
    trellis, rembg = approved.get("OGWP_TRELLIS_MODELS_DIR"), approved.get("OGWP_REMBG_MODELS_DIR")
    return {
        **{f"{MODELS}/{s}": (str(Path(trellis) / s) if trellis else None) for s in TRELLIS_SUBTREES},
        REMBG_MODELS: rembg,
        CREDENTIAL: approved.get("OGWP_WORKER_CREDENTIAL"),
        WSL_LIB: WSL_LIB,
        WSL_DRIVERS: WSL_DRIVERS,
    }


def violations(containers: list[dict], networks: dict[str, dict], *, allow_stopped_agent=False,
               approved=None) -> list[str]:
    approved = approved or {}
    errors: list[str] = []
    indexed: dict[str, dict] = {}
    for c in containers:
        name = (c.get("Config", {}).get("Labels") or {}).get("com.docker.compose.service")
        if name not in SERVICES or name in indexed:
            errors.append("unexpected or duplicate service")
            continue
        indexed[name] = c
    if set(indexed) != SERVICES:
        errors.append("all three isolated services must exist")
    sources = _expected_sources(approved)
    for name, c in indexed.items():
        cfg, host = c.get("Config", {}), c.get("HostConfig", {})

        def require(condition, reason, name=name):
            if not condition:
                errors.append(f"{name}: {reason}")

        if name != "agent" or not allow_stopped_agent:
            require(c.get("State", {}).get("Running") is True, "service is not running")
        require(cfg.get("User") == "65532:65532", "dedicated non-root container user required")
        require(bool(re.fullmatch(r"(?:sha256:|.+@sha256:)[0-9a-f]{64}", cfg.get("Image", ""))),
                "image must be an immutable digest")
        require(bool(approved.get(IMAGE_KEY[name])) and cfg.get("Image") == approved.get(IMAGE_KEY[name]),
                "image must match the operator-approved profile")
        require(host.get("ReadonlyRootfs") is True, "root filesystem must be read-only")
        require(not host.get("Privileged"), "privileged mode forbidden")
        require(not host.get("CapAdd") and "ALL" in (host.get("CapDrop") or []), "all capabilities must be dropped")
        opts = host.get("SecurityOpt") or []
        require(any(x in opts for x in NO_NEW_PRIVILEGES), "no-new-privileges required")
        require(set(opts) <= NO_NEW_PRIVILEGES, "security profile overrides forbidden")
        require(host.get("Runtime") in (None, "", "runc"), "alternative container runtimes forbidden")
        require(not host.get("PortBindings") and not host.get("PublishAllPorts"), "published ports forbidden")
        for mode in ("PidMode", "IpcMode", "UTSMode"):
            value = host.get(mode) or ""
            require(value != "host" and not value.startswith("container:"), "shared host or container namespaces forbidden")
        require(host.get("NetworkMode") != "host", "host networking forbidden")
        require(host.get("CgroupnsMode") == "private", "private cgroup namespace required")
        memory_max, cpu_max, pids_max = BOUNDS[name]
        require(0 < (host.get("PidsLimit") or 0) <= pids_max, "PID bound missing or excessive")
        require(0 < (host.get("Memory") or 0) <= memory_max and 0 < (host.get("NanoCpus") or 0) <= cpu_max,
                "memory and CPU bounds missing or excessive")
        require(host.get("MemorySwap") == host.get("Memory") and (host.get("Memory") or 0) > 0,
                "workload swap must be disabled")
        require(not host.get("OomKillDisable"), "OOM kill must stay enabled")
        require(0 < (host.get("ShmSize") or 64 * MiB) <= SHM_MAX, "shared memory bound excessive")
        require(host.get("Init") is True, "child reaping required")
        require((host.get("RestartPolicy") or {}).get("Name") == "no", "ungated Docker restart policy forbidden")
        require(not host.get("GroupAdd"), "supplementary host groups forbidden")
        require(not host.get("VolumesFrom"), "inherited volumes forbidden")
        require(not host.get("Binds"), "legacy bind syntax forbidden")
        require(not host.get("ExtraHosts"), "extra host entries forbidden")
        require(not host.get("DeviceRequests"), "unreviewed device requests forbidden")
        require(not host.get("DeviceCgroupRules"), "device cgroup overrides forbidden")
        devices = host.get("Devices") or []
        for device in devices:
            require(name == "engine" and device.get("PathOnHost") == "/dev/dxg"
                    and device.get("PathInContainer") == "/dev/dxg"
                    and device.get("CgroupPermissions") == "rw", "unreviewed device mount")
        binds = set()
        for mount in c.get("Mounts") or []:
            if mount.get("Type") == "tmpfs":
                continue
            destination = mount.get("Destination")
            binds.add(destination)
            require(mount.get("Type") == "bind" and destination in MOUNTS[name] and not mount.get("RW"),
                    "unreviewed or writable host mount")
            require(mount.get("Propagation") in (None, "", "rprivate", "private"), "shared mount propagation forbidden")
            expected = sources.get(destination)
            require(expected is not None and mount.get("Source") == str(Path(expected).resolve()),
                    "host mount source differs from approved profile")
        require(binds == MOUNTS[name], "required read-only mounts missing")
        tmpfs = host.get("Tmpfs") or {}
        require(bool(tmpfs) and all(_tmpfs_size(v) for v in tmpfs.values()), "scratch mounts must be bounded")
        require(set(tmpfs) <= TMPFS[name], "unexpected writable temporary filesystem")
        require(all("nosuid" in v.split(",") and "nodev" in v.split(",")
                    and ("noexec" in v.split(",") or k in EXEC_TMPFS.get(name, set()))
                    for k, v in tmpfs.items()), "scratch execution or device policy drift")
        env = _env(c)
        if name == "agent":
            engine_id = indexed.get("engine", {}).get("Id", "missing")
            require(host.get("NetworkMode") == "container:" + engine_id,
                    "agent must share only the isolated engine network")
            require(not devices and not any(d.startswith("/usr/lib/wsl") for d in binds)
                    and "NVIDIA_VISIBLE_DEVICES" not in env, "agent must have no GPU")
            for key, value in AGENT_ENV.items():
                require(env.get(key) == value, f"{key} must be {value}")
            require(not env.get("OGWP_COMFY_START_COMMAND") and not env.get("MVS_ENGINE_COMMAND"),
                    "host runtime start commands forbidden")
            try:
                bound = 0 < int(env.get("OGWP_WORKER_MAX_INPUT_BYTES", "")) <= AGENT_MAX_INPUT_BYTES
            except ValueError:
                bound = False
            require(bound, "input download bound missing or excessive")
        elif name == "engine":
            attached = c.get("NetworkSettings", {}).get("Networks") or {}
            require(len(attached) == 1, "engine must have exactly one network")
            for network_name in attached:
                network = networks.get(network_name, {})
                options = network.get("Options") or {}
                require(network.get("Internal") is True and network.get("Driver") == "bridge"
                        and options.get("com.docker.network.bridge.gateway_mode_ipv4") == "isolated"
                        and options.get("com.docker.network.bridge.gateway_mode_ipv6") == "isolated",
                        "network must deny external routes and host gateway access")
            require(host.get("Dns") == ["127.0.0.1"], "external DNS forwarding forbidden")
            require(any(d.get("PathInContainer") == "/dev/dxg" for d in devices), "engine GPU device missing")
            require((host.get("Memory") or 0) >= ENGINE_MEMORY_FLOOR, "engine memory below the measured working set")
            require(c.get("State", {}).get("Health", {}).get("Status") == "healthy", "runtime is not healthy")
            test = (cfg.get("Healthcheck") or {}).get("Test") or []
            require(test == ["CMD", ENGINE_PYTHON, "-c", ENGINE_HEALTH], "reviewed /health healthcheck required")
            require(cfg.get("Entrypoint") == ENGINE_ENTRYPOINT and cfg.get("Cmd") == ENGINE_COMMAND,
                    "engine must run the reviewed serve invocation bound to 127.0.0.1")
            require(cfg.get("WorkingDir") == "/opt/aismith", "engine working directory drift")
            for key, value in ENGINE_ENV.items():
                require(env.get(key) == value, f"{key} must be {value}")
            require(bool(approved.get("OGWP_ENGINE_COMMIT"))
                    and env.get("AISMITH_ENGINE_COMMIT") == approved.get("OGWP_ENGINE_COMMIT"),
                    "engine commit differs from approved profile")
            require(not any(SECRET_NAME.search(key) for key in env), "credentials or proxies forbidden in engine")
        elif name == "egress":
            attached = c.get("NetworkSettings", {}).get("Networks") or {}
            require(len(attached) == 2, "egress proxy requires exactly two networks")
            require(cfg.get("Entrypoint") == ["python", "/opt/ogwp/egress_proxy.py"], "reviewed egress relay required")
            require(c.get("Image") == indexed.get("agent", {}).get("Image"), "proxy must use the approved agent image")
            hosts = env.get("OGWP_EGRESS_HOSTS", "")
            require(bool(hosts) and "*" not in hosts, "explicit egress allowlist required")
            require(hosts == approved.get("OGWP_EGRESS_HOSTS"), "egress allowlist differs from approved profile")
    engine_networks = set(indexed.get("engine", {}).get("NetworkSettings", {}).get("Networks") or {})
    egress_networks = set(indexed.get("egress", {}).get("NetworkSettings", {}).get("Networks") or {})
    if not engine_networks or not engine_networks < egress_networks:
        errors.append("proxy must bridge the isolated workload network and only one outbound network")
    for network_name in egress_networks - engine_networks:
        if networks.get(network_name, {}).get("Internal") is True:
            errors.append("proxy outbound network must not be internal")
    for network_name in engine_networks:
        expected = {indexed.get("engine", {}).get("Id"), indexed.get("egress", {}).get("Id")}
        if set(networks.get(network_name, {}).get("Containers") or {}) - expected:
            errors.append("unexpected container on isolated workload network")
    return errors


def manifest_from_lock(lock: dict, weight_ids=REQUIRED_WEIGHT_SETS) -> list[dict]:
    """The approved store manifest for `weight_ids` from a reviewed weights lock."""
    sets = {entry["id"]: entry for entry in lock["weights"]}
    records = []
    for weight_id in weight_ids:
        entry = sets[weight_id]
        for item in entry["files"]:
            if not re.fullmatch(r"[0-9a-f]{64}", item.get("sha256") or ""):
                raise ValueError(f"{weight_id}: every file needs a sha256 pin")
            dest = item["dest"]
            if entry["root"] == "trellis_models":
                store = "trellis"
            elif entry["root"] == "u2net" and dest.startswith("models/"):
                store, dest = "rembg", dest.removeprefix("models/")
            else:
                raise ValueError(f"{weight_id}: root {entry['root']} is not mounted by this profile")
            records.append({"store": store, "file": dest, "bytes": item["size"], "sha256": item["sha256"]})
    return records


def _mounted_roots(stores: dict[str, Path]) -> list[tuple[str, str, Path]]:
    """(store, prefix, directory) for each subtree the engine actually sees."""
    return [*(("trellis", f"{s}/", stores["trellis"] / s) for s in TRELLIS_SUBTREES),
            ("rembg", "", stores["rembg"])]


def verify_models(stores: dict[str, Path], manifest: Path) -> list[str]:
    """Hash the exact mounted model trees before the engine starts; never follow symlinks."""
    try:
        records = json.loads(manifest.read_text())
        if not isinstance(records, list) or not records:
            return ["invalid model manifest or root"]
        expected: dict[tuple[str, str], dict] = {}
        for record in records:
            store, name = record["store"], record["file"]
            path = Path(name)
            if (store not in STORES or not isinstance(name, str) or path.is_absolute() or ".." in path.parts
                    or (store, name) in expected or path.suffix not in MODEL_SUFFIXES
                    or not isinstance(record["bytes"], int)
                    or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])):
                return ["invalid model manifest path"]
            if store == "trellis" and not any(name.startswith(f"{s}/") for s in TRELLIS_SUBTREES):
                return ["model manifest names a file outside the mounted subtrees"]
            expected[(store, name)] = record
        files: dict[tuple[str, str], Path] = {}
        for store, prefix, root in _mounted_roots(stores):
            if stores[store].is_symlink() or root.is_symlink() or not root.is_dir():
                return ["invalid model manifest or root"]
            for path in root.rglob("*"):
                if path.is_symlink():
                    return ["model tree contains symlink"]
                if path.is_file():
                    files[(store, prefix + path.relative_to(root).as_posix())] = path
                elif not path.is_dir():
                    return ["model tree contains a special file"]
        if files.keys() != expected.keys():
            return ["model inventory differs from approved manifest"]
        for key, path in files.items():
            if path.stat().st_size != expected[key]["bytes"]:
                return ["model size differs from approved manifest"]
            with path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            if digest != expected[key]["sha256"]:
                return ["model content differs from approved manifest"]
        return []
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return ["model integrity verification failed"]


def read_profile(path: Path) -> dict[str, str]:
    # This private file is trusted operator configuration, never workload input.
    return dict(line.split("=", 1) for line in path.read_text().splitlines()
                if line and not line.startswith("#") and "=" in line)


def model_stores(approved: dict[str, str]) -> dict[str, Path]:
    return {"trellis": Path(approved["OGWP_TRELLIS_MODELS_DIR"]), "rembg": Path(approved["OGWP_REMBG_MODELS_DIR"])}


def docker(*args):
    return subprocess.check_output(["docker", *args], text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify-models", action="store_true")
    parser.add_argument("--models-only", action="store_true")
    parser.add_argument("--manifest-from-lock", type=Path, metavar="WEIGHTS_LOCK",
                        help="print the approved store manifest for a reviewed weights.lock.json")
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--allow-stopped-agent", action="store_true", help="pre-start gate; only the agent may be stopped")
    parser.add_argument("--profile", type=Path, default=Path(__file__).resolve().parents[1] / "profile-3d.env")
    args = parser.parse_args()
    if args.manifest_from_lock:
        print(json.dumps(manifest_from_lock(json.loads(args.manifest_from_lock.read_text())), indent=1))
        return 0
    approved = read_profile(args.profile)
    manifest = args.profile.parent / "model-manifest-3d.json"
    if args.models_only:
        errors = verify_models(model_stores(approved), manifest)
        print(json.dumps({"passed": not errors, "violations": errors}))
        return 1 if errors else 0
    ids = docker("ps", "-aq", "--filter", f"label=com.docker.compose.project={args.project}").split()
    containers = json.loads(docker("inspect", *ids)) if ids else []
    names = {name for c in containers for name in (c.get("NetworkSettings", {}).get("Networks") or {})}
    networks = {n["Name"]: n for n in json.loads(docker("network", "inspect", *sorted(names)))} if names else {}
    errors = violations(containers, networks, allow_stopped_agent=args.allow_stopped_agent, approved=approved)
    if args.verify_models:
        errors += verify_models(model_stores(approved), manifest)
    print(json.dumps({"passed": not errors, "violations": errors}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
