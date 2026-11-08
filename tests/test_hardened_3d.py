"""The hardened 3D engine deployment (`deploy/hardened-3d`): no Docker, GPU or systemd.

`verify.py` is exercised with synthetic `docker inspect` dicts, its model gate
with temporary stores, the compose files against the gate's own constants, and
the time-share switch with fake host commands.
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from outbound_gpu_worker_pool.engine_recipes import MESH3D, MESH3D_RECIPE

DEPLOY = Path(__file__).resolve().parents[1] / "deploy" / "hardened-3d"
GiB = 1024**3


def _load(name):
    spec = importlib.util.spec_from_file_location(f"hardened_3d_{name}", DEPLOY / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verify = _load("verify")
gate = _load("switch_gate")
egress = _load("egress_proxy")

AGENT_IMAGE = "sha256:" + "a" * 64
ENGINE_IMAGE = "registry.example/aismith-engine@sha256:" + "b" * 64
ENGINE_COMMIT = "840bddd245a7155865eb453aff72b115f3e521ed"
APPROVED = {
    "OGWP_AGENT_IMAGE": AGENT_IMAGE,
    "OGWP_ENGINE_IMAGE": ENGINE_IMAGE,
    "OGWP_ENGINE_COMMIT": ENGINE_COMMIT,
    "OGWP_EGRESS_HOSTS": "pool.example.com,oauth2.googleapis.com,storage.googleapis.com",
    "OGWP_TRELLIS_MODELS_DIR": "/srv/ogwp-models/trellis2-refiner-models",
    "OGWP_REMBG_MODELS_DIR": "/srv/ogwp-models/rembg/models",
    "OGWP_WORKER_CREDENTIAL": "/srv/ogwp-credentials/worker-3d.json",
}
WORKLOAD, INTERNET = "ogwp-hardened-3d_workload", "ogwp-hardened-3d_internet"


def _bind(source, target):
    return {"Type": "bind", "Source": str(Path(source).resolve()), "Destination": target,
            "RW": False, "Propagation": "rprivate"}


def _container(service, image, resolved, *, memory, cpus, pids, tmpfs, env, network_mode):
    return {
        "Id": f"{service}-id",
        "Image": resolved,
        "Config": {"Labels": {"com.docker.compose.service": service, "com.docker.compose.project": "ogwp-hardened-3d"},
                   "User": "65532:65532", "Image": image, "Env": [f"{k}={v}" for k, v in env.items()]},
        "State": {"Running": True},
        "HostConfig": {
            "ReadonlyRootfs": True, "Privileged": False, "CapAdd": None, "CapDrop": ["ALL"],
            "SecurityOpt": ["no-new-privileges:true"], "Runtime": "runc", "PortBindings": {},
            "PublishAllPorts": False, "PidMode": "", "IpcMode": "private", "UTSMode": "",
            "NetworkMode": network_mode, "CgroupnsMode": "private", "PidsLimit": pids,
            "Memory": memory, "MemorySwap": memory, "NanoCpus": cpus, "ShmSize": 64 * 1024**2,
            "Init": True, "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0}, "Tmpfs": tmpfs,
        },
        "Mounts": [],
        "NetworkSettings": {"Networks": {}},
    }


def compliant():
    """A stack as `docker compose up` creates it from compose.yaml + wsl.yaml."""
    engine_env = {**verify.ENGINE_ENV, "AISMITH_ENGINE_COMMIT": ENGINE_COMMIT, "HOME": "/scratch",
                  "TMPDIR": "/jit/tmp", "LD_LIBRARY_PATH": "/usr/lib/wsl/lib", "PATH": "/usr/bin"}
    engine = _container(
        "engine", ENGINE_IMAGE, "sha256:" + "e" * 64, memory=24 * GiB, cpus=8 * 10**9, pids=1024,
        tmpfs={
            "/scratch": "size=4g,uid=65532,gid=65532,mode=0700,noexec,nosuid,nodev",
            "/jit": "size=1g,uid=65532,gid=65532,mode=0700,exec,nosuid,nodev",
            "/tmp": "size=256m,mode=1777,noexec,nosuid,nodev",
            "/home": "size=1m,mode=0755,noexec,nosuid,nodev",
            verify.MODELS: "size=16m,uid=65532,gid=65532,mode=0700,noexec,nosuid,nodev",
            f"{verify.MODELS}/microsoft/TRELLIS.2-4B": "size=1m,uid=65532,gid=65532,mode=0700,noexec,nosuid,nodev",
        },
        env=engine_env, network_mode=WORKLOAD)
    engine["Config"].update({
        "Entrypoint": list(verify.ENGINE_ENTRYPOINT), "Cmd": list(verify.ENGINE_COMMAND),
        "WorkingDir": "/opt/aismith",
        "Healthcheck": {"Test": ["CMD", verify.ENGINE_PYTHON, "-c", verify.ENGINE_HEALTH]},
    })
    engine["State"]["Health"] = {"Status": "healthy"}
    engine["HostConfig"].update({
        "Dns": ["127.0.0.1"], "ShmSize": GiB,
        "Devices": [{"PathOnHost": "/dev/dxg", "PathInContainer": "/dev/dxg", "CgroupPermissions": "rw"}],
    })
    trellis = APPROVED["OGWP_TRELLIS_MODELS_DIR"]
    engine["Mounts"] = [
        *(_bind(f"{trellis}/{s}", f"{verify.MODELS}/{s}") for s in verify.TRELLIS_SUBTREES),
        _bind(APPROVED["OGWP_REMBG_MODELS_DIR"], verify.REMBG_MODELS),
        _bind("/usr/lib/wsl/lib", "/usr/lib/wsl/lib"),
        _bind("/usr/lib/wsl/drivers", "/usr/lib/wsl/drivers"),
    ]
    engine["NetworkSettings"]["Networks"] = {WORKLOAD: {}}

    agent_env = {**verify.AGENT_ENV, "OGWP_WORKER_MAX_INPUT_BYTES": "268435456",
                 "OGWP_WORKER_COORDINATOR_URL": "https://pool.example.com/pool", "OGWP_WORKER_ID": "desk-3d",
                 "OGWP_WORKER_AUTH": "static", "OGWP_WORKER_TOKEN": "not-a-real-token",
                 "OGWP_COMFY_START_COMMAND": "", "MVS_ENGINE_COMMAND": ""}
    agent = _container(
        "agent", AGENT_IMAGE, "sha256:" + "a" * 64, memory=2 * GiB, cpus=2 * 10**9, pids=256,
        tmpfs={"/scratch": "size=1g,uid=65532,gid=65532,mode=0700,noexec,nosuid,nodev",
               "/tmp": "size=64m,mode=1777,noexec,nosuid,nodev"},
        env=agent_env, network_mode="container:engine-id")
    agent["Mounts"] = [_bind(APPROVED["OGWP_WORKER_CREDENTIAL"], verify.CREDENTIAL)]

    egress_box = _container(
        "egress", AGENT_IMAGE, "sha256:" + "a" * 64, memory=128 * 1024**2, cpus=500_000_000, pids=256,
        tmpfs={"/tmp": "size=16m,mode=1777,noexec,nosuid,nodev"},
        env={"OGWP_EGRESS_HOSTS": APPROVED["OGWP_EGRESS_HOSTS"]}, network_mode=WORKLOAD)
    egress_box["Config"]["Entrypoint"] = ["python", "/opt/ogwp/egress_proxy.py"]
    egress_box["NetworkSettings"]["Networks"] = {WORKLOAD: {}, INTERNET: {}}

    isolated = {"com.docker.network.bridge.gateway_mode_ipv4": "isolated",
                "com.docker.network.bridge.gateway_mode_ipv6": "isolated"}
    networks = {
        WORKLOAD: {"Name": WORKLOAD, "Internal": True, "Driver": "bridge", "Options": isolated,
                   "Containers": {"engine-id": {}, "egress-id": {}}},
        INTERNET: {"Name": INTERNET, "Internal": False, "Driver": "bridge", "Options": {},
                   "Containers": {"egress-id": {}}},
    }
    return {"agent": agent, "engine": engine, "egress": egress_box}, networks


def check(mutate=None, *, allow_stopped_agent=False, approved=APPROVED):
    containers, networks = compliant()
    if mutate:
        mutate(containers, networks)
    return verify.violations(list(containers.values()), networks,
                             allow_stopped_agent=allow_stopped_agent, approved=approved)


def _env_set(container, key, value):
    env = [e for e in container["Config"]["Env"] if not e.startswith(f"{key}=")]
    if value is not None:
        env.append(f"{key}={value}")
    container["Config"]["Env"] = env


def test_compliant_stack_passes():
    assert check() == []


def test_stopped_agent_is_allowed_only_before_start():
    def stop(c, n):
        c["agent"]["State"]["Running"] = False
    assert "agent: service is not running" in check(stop)
    assert check(stop, allow_stopped_agent=True) == []

    def stop_engine(c, n):
        c["engine"]["State"]["Running"] = False
    assert "engine: service is not running" in check(stop_engine, allow_stopped_agent=True)


def _gpu(c):
    return {"PathOnHost": "/dev/dxg", "PathInContainer": "/dev/dxg", "CgroupPermissions": "rw"}


VIOLATIONS = {
    # The GPU, models and credential stay on their own side.
    "gpu_in_agent": (lambda c, n: c["agent"]["HostConfig"].update(Devices=[_gpu(c)]),
                     ["agent: unreviewed device mount", "agent: agent must have no GPU"]),
    "wsl_driver_in_agent": (lambda c, n: c["agent"]["Mounts"].append(_bind("/usr/lib/wsl/lib", "/usr/lib/wsl/lib")),
                            ["agent: unreviewed or writable host mount", "agent: agent must have no GPU"]),
    "nvidia_env_in_agent": (lambda c, n: _env_set(c["agent"], "NVIDIA_VISIBLE_DEVICES", "all"),
                            ["agent: agent must have no GPU"]),
    "gpu_request_in_agent": (lambda c, n: c["agent"]["HostConfig"].update(DeviceRequests=[{"Count": -1}]),
                             ["agent: unreviewed device requests forbidden"]),
    "model_in_agent": (lambda c, n: c["agent"]["Mounts"].append(
                           _bind(APPROVED["OGWP_REMBG_MODELS_DIR"], verify.REMBG_MODELS)),
                       ["agent: unreviewed or writable host mount"]),
    "credential_in_engine": (lambda c, n: c["engine"]["Mounts"].append(
                                 _bind(APPROVED["OGWP_WORKER_CREDENTIAL"], verify.CREDENTIAL)),
                             ["engine: unreviewed or writable host mount"]),
    "credential_env_in_engine": (lambda c, n: _env_set(c["engine"], "GOOGLE_APPLICATION_CREDENTIALS", "/x"),
                                 ["engine: credentials or proxies forbidden in engine"]),
    "token_env_in_engine": (lambda c, n: _env_set(c["engine"], "HF_TOKEN", "x"),
                            ["engine: credentials or proxies forbidden in engine"]),
    "proxy_env_in_engine": (lambda c, n: _env_set(c["engine"], "HTTPS_PROXY", "http://egress:3128"),
                            ["engine: credentials or proxies forbidden in engine"]),
    "engine_without_gpu": (lambda c, n: c["engine"]["HostConfig"].update(Devices=[]),
                           ["engine: engine GPU device missing"]),
    "missing_credential_mount": (lambda c, n: c["agent"].update(Mounts=[]),
                                 ["agent: required read-only mounts missing"]),
    # Networking.
    "agent_own_network": (lambda c, n: c["agent"]["HostConfig"].update(NetworkMode=WORKLOAD),
                          ["agent: agent must share only the isolated engine network"]),
    "agent_shares_egress": (lambda c, n: c["agent"]["HostConfig"].update(NetworkMode="container:egress-id"),
                            ["agent: agent must share only the isolated engine network"]),
    "agent_host_network": (lambda c, n: c["agent"]["HostConfig"].update(NetworkMode="host"),
                           ["agent: host networking forbidden"]),
    "engine_outbound_network": (lambda c, n: c["engine"]["NetworkSettings"]["Networks"].update({INTERNET: {}}),
                                ["engine: engine must have exactly one network",
                                 "proxy must bridge the isolated workload network and only one outbound network"]),
    "workload_not_internal": (lambda c, n: n[WORKLOAD].update(Internal=False),
                              ["engine: network must deny external routes and host gateway access"]),
    "workload_gateway_not_isolated": (lambda c, n: n[WORKLOAD].update(Options={}),
                                      ["engine: network must deny external routes and host gateway access"]),
    "engine_external_dns": (lambda c, n: c["engine"]["HostConfig"].update(Dns=None),
                            ["engine: external DNS forwarding forbidden"]),
    "stranger_on_workload": (lambda c, n: n[WORKLOAD]["Containers"].update({"other-id": {}}),
                             ["unexpected container on isolated workload network"]),
    "egress_three_networks": (lambda c, n: c["egress"]["NetworkSettings"]["Networks"].update({"extra": {}}),
                              ["egress: egress proxy requires exactly two networks"]),
    "egress_outbound_internal": (lambda c, n: n[INTERNET].update(Internal=True),
                                 ["proxy outbound network must not be internal"]),
    "egress_wildcard": (lambda c, n: _env_set(c["egress"], "OGWP_EGRESS_HOSTS", "*.example.com"),
                        ["egress: explicit egress allowlist required",
                         "egress: egress allowlist differs from approved profile"]),
    "egress_extra_host": (lambda c, n: _env_set(c["egress"], "OGWP_EGRESS_HOSTS",
                                                APPROVED["OGWP_EGRESS_HOSTS"] + ",evil.example.net"),
                          ["egress: egress allowlist differs from approved profile"]),
    "egress_unreviewed_entrypoint": (lambda c, n: c["egress"]["Config"].update(Entrypoint=["sh"]),
                                     ["egress: reviewed egress relay required"]),
    "published_port": (lambda c, n: c["engine"]["HostConfig"].update(PortBindings={"8190/tcp": [{}]}),
                       ["engine: published ports forbidden"]),
    "extra_hosts": (lambda c, n: c["engine"]["HostConfig"].update(ExtraHosts=["host.docker.internal:host-gateway"]),
                    ["engine: extra host entries forbidden"]),
    # Mounts.
    "writable_model_mount": (lambda c, n: c["engine"]["Mounts"][0].update(RW=True),
                             ["engine: unreviewed or writable host mount"]),
    "model_source_drift": (lambda c, n: c["engine"]["Mounts"][0].update(Source="/home/user/models/visualbruno"),
                           ["engine: host mount source differs from approved profile"]),
    "missing_model_mount": (lambda c, n: c["engine"]["Mounts"].pop(1),
                            ["engine: required read-only mounts missing"]),
    "whole_store_mount": (lambda c, n: c["engine"]["Mounts"].append(
                              _bind(APPROVED["OGWP_TRELLIS_MODELS_DIR"], verify.MODELS)),
                          ["engine: unreviewed or writable host mount"]),
    "volume_mount": (lambda c, n: c["engine"]["Mounts"].append(
                         {"Type": "volume", "Destination": "/data", "RW": True, "Source": "/var/lib/docker/v"}),
                     ["engine: unreviewed or writable host mount"]),
    "shared_propagation": (lambda c, n: c["engine"]["Mounts"][0].update(Propagation="rshared"),
                           ["engine: shared mount propagation forbidden"]),
    "docker_socket": (lambda c, n: c["agent"]["Mounts"].append(_bind("/var/run/docker.sock", "/var/run/docker.sock")),
                      ["agent: unreviewed or writable host mount"]),
    # Images.
    "unapproved_engine_digest": (lambda c, n: c["engine"]["Config"].update(Image="sha256:" + "c" * 64),
                                 ["engine: image must match the operator-approved profile"]),
    "engine_tag_not_digest": (lambda c, n: c["engine"]["Config"].update(Image="aismith-engine:latest"),
                              ["engine: image must be an immutable digest",
                               "engine: image must match the operator-approved profile"]),
    "agent_runs_engine_image": (lambda c, n: c["agent"]["Config"].update(Image=ENGINE_IMAGE),
                                ["agent: image must match the operator-approved profile"]),
    "egress_other_image": (lambda c, n: c["egress"].update(Image="sha256:" + "d" * 64),
                           ["egress: proxy must use the approved agent image"]),
    "engine_commit_drift": (lambda c, n: _env_set(c["engine"], "AISMITH_ENGINE_COMMIT", "f" * 40),
                            ["engine: engine commit differs from approved profile"]),
    # The engine runtime.
    "missing_healthcheck": (lambda c, n: c["engine"]["Config"].update(Healthcheck=None),
                            ["engine: reviewed /health healthcheck required"]),
    "healthcheck_elsewhere": (lambda c, n: c["engine"]["Config"].update(Healthcheck={"Test": ["NONE"]}),
                              ["engine: reviewed /health healthcheck required"]),
    "engine_unhealthy": (lambda c, n: c["engine"]["State"].update(Health={"Status": "starting"}),
                         ["engine: runtime is not healthy"]),
    "serve_on_all_interfaces": (lambda c, n: c["engine"]["Config"].update(
                                    Cmd=["serve", "--host", "0.0.0.0", "--port", "8190", "--work-root", "/scratch/engine"]),
                                ["engine: engine must run the reviewed serve invocation bound to 127.0.0.1"]),
    "serve_allow_private": (lambda c, n: c["engine"]["Config"].update(Cmd=[*verify.ENGINE_COMMAND, "--allow-private"]),
                            ["engine: engine must run the reviewed serve invocation bound to 127.0.0.1"]),
    "engine_shell_entrypoint": (lambda c, n: c["engine"]["Config"].update(Entrypoint=["/bin/sh", "-c"]),
                                ["engine: engine must run the reviewed serve invocation bound to 127.0.0.1"]),
    "engine_online": (lambda c, n: _env_set(c["engine"], "AISMITH_OFFLINE", None),
                      ["engine: AISMITH_OFFLINE must be 1"]),
    "hub_online": (lambda c, n: _env_set(c["engine"], "HF_HUB_OFFLINE", "0"),
                   ["engine: HF_HUB_OFFLINE must be 1"]),
    "refiner_elsewhere": (lambda c, n: _env_set(c["engine"], "TRELLIS_REFINER_DIR", "/scratch/refiner"),
                          ["engine: TRELLIS_REFINER_DIR must be /opt/trellis2-refiner"]),
    # The agent's configuration.
    "agent_concurrency": (lambda c, n: _env_set(c["agent"], "OGWP_WORKER_CONCURRENCY", "2"),
                          ["agent: OGWP_WORKER_CONCURRENCY must be 1"]),
    "agent_remote_engine": (lambda c, n: _env_set(c["agent"], "OGWP_ENGINE_URL", "http://10.0.0.5:8190"),
                            ["agent: OGWP_ENGINE_URL must be http://127.0.0.1:8190"]),
    "agent_local_transport": (lambda c, n: _env_set(c["agent"], "MVS_ENGINE_COMMAND", "/opt/x/aismith-engine"),
                              ["agent: host runtime start commands forbidden"]),
    "agent_extra_plugin": (lambda c, n: _env_set(c["agent"], "OGWP_WORKER_PLUGINS", "aismith-engine,comfy-workflow"),
                           ["agent: OGWP_WORKER_PLUGINS must be aismith-engine"]),
    "agent_other_profile": (lambda c, n: _env_set(c["agent"], "MVS_WORKER_PROFILE", "flux-24gb-standard"),
                            ["agent: MVS_WORKER_PROFILE must be mesh3d-24gb-standard"]),
    "agent_proxy_bypass": (lambda c, n: _env_set(c["agent"], "NO_PROXY", "*"),
                           ["agent: NO_PROXY must be localhost,127.0.0.1,::1"]),
    "agent_unbounded_inputs": (lambda c, n: _env_set(c["agent"], "OGWP_WORKER_MAX_INPUT_BYTES", None),
                               ["agent: input download bound missing or excessive"]),
    "agent_large_inputs": (lambda c, n: _env_set(c["agent"], "OGWP_WORKER_MAX_INPUT_BYTES", str(2 * GiB)),
                           ["agent: input download bound missing or excessive"]),
    # Sandbox flags.
    "root_user": (lambda c, n: c["engine"]["Config"].update(User="0:0"),
                  ["engine: dedicated non-root container user required"]),
    "writable_rootfs": (lambda c, n: c["engine"]["HostConfig"].update(ReadonlyRootfs=False),
                        ["engine: root filesystem must be read-only"]),
    "privileged": (lambda c, n: c["engine"]["HostConfig"].update(Privileged=True),
                   ["engine: privileged mode forbidden"]),
    "cap_add": (lambda c, n: c["engine"]["HostConfig"].update(CapAdd=["SYS_ADMIN"]),
                ["engine: all capabilities must be dropped"]),
    "no_new_privileges_missing": (lambda c, n: c["agent"]["HostConfig"].update(SecurityOpt=[]),
                                  ["agent: no-new-privileges required"]),
    "seccomp_unconfined": (lambda c, n: c["engine"]["HostConfig"]["SecurityOpt"].append("seccomp=unconfined"),
                           ["engine: security profile overrides forbidden"]),
    "nvidia_runtime": (lambda c, n: c["engine"]["HostConfig"].update(Runtime="nvidia"),
                       ["engine: alternative container runtimes forbidden"]),
    "host_pid": (lambda c, n: c["engine"]["HostConfig"].update(PidMode="host"),
                 ["engine: shared host or container namespaces forbidden"]),
    "shared_ipc": (lambda c, n: c["agent"]["HostConfig"].update(IpcMode="container:engine-id"),
                   ["agent: shared host or container namespaces forbidden"]),
    "host_cgroupns": (lambda c, n: c["engine"]["HostConfig"].update(CgroupnsMode="host"),
                      ["engine: private cgroup namespace required"]),
    "restart_policy": (lambda c, n: c["engine"]["HostConfig"].update(RestartPolicy={"Name": "always"}),
                       ["engine: ungated Docker restart policy forbidden"]),
    "no_init": (lambda c, n: c["agent"]["HostConfig"].update(Init=False), ["agent: child reaping required"]),
    "group_add": (lambda c, n: c["engine"]["HostConfig"].update(GroupAdd=["video"]),
                  ["engine: supplementary host groups forbidden"]),
    "oom_kill_disabled": (lambda c, n: c["engine"]["HostConfig"].update(OomKillDisable=True),
                          ["engine: OOM kill must stay enabled"]),
    # Resource bounds.
    "engine_unbounded_memory": (lambda c, n: c["engine"]["HostConfig"].update(Memory=0, MemorySwap=0),
                                ["engine: memory and CPU bounds missing or excessive"]),
    "engine_memory_over_ceiling": (lambda c, n: c["engine"]["HostConfig"].update(Memory=32 * GiB, MemorySwap=32 * GiB),
                                   ["engine: memory and CPU bounds missing or excessive"]),
    "engine_memory_under_floor": (lambda c, n: c["engine"]["HostConfig"].update(Memory=8 * GiB, MemorySwap=8 * GiB),
                                  ["engine: engine memory below the measured working set"]),
    "engine_cpus_over_ceiling": (lambda c, n: c["engine"]["HostConfig"].update(NanoCpus=16 * 10**9),
                                 ["engine: memory and CPU bounds missing or excessive"]),
    "agent_memory_over_ceiling": (lambda c, n: c["agent"]["HostConfig"].update(Memory=8 * GiB, MemorySwap=8 * GiB),
                                  ["agent: memory and CPU bounds missing or excessive"]),
    "engine_swap": (lambda c, n: c["engine"]["HostConfig"].update(MemorySwap=-1),
                    ["engine: workload swap must be disabled"]),
    "engine_unbounded_pids": (lambda c, n: c["engine"]["HostConfig"].update(PidsLimit=None),
                              ["engine: PID bound missing or excessive"]),
    "engine_pids_over_ceiling": (lambda c, n: c["engine"]["HostConfig"].update(PidsLimit=4096),
                                 ["engine: PID bound missing or excessive"]),
    "agent_pids_over_ceiling": (lambda c, n: c["agent"]["HostConfig"].update(PidsLimit=1024),
                                ["agent: PID bound missing or excessive"]),
    "engine_huge_shm": (lambda c, n: c["engine"]["HostConfig"].update(ShmSize=8 * GiB),
                        ["engine: shared memory bound excessive"]),
    # Scratch.
    "unbounded_scratch": (lambda c, n: c["engine"]["HostConfig"]["Tmpfs"].update({"/scratch": "noexec,nosuid,nodev"}),
                          ["engine: scratch mounts must be bounded"]),
    "executable_scratch": (lambda c, n: c["engine"]["HostConfig"]["Tmpfs"].update(
                               {"/scratch": "size=4g,uid=65532,gid=65532,mode=0700,exec,nosuid,nodev"}),
                           ["engine: scratch execution or device policy drift"]),
    "executable_model_cache": (lambda c, n: c["engine"]["HostConfig"]["Tmpfs"].update(
                                   {verify.MODELS: "size=16m,uid=65532,gid=65532,mode=0700,exec,nosuid,nodev"}),
                               ["engine: scratch execution or device policy drift"]),
    "agent_jit": (lambda c, n: c["agent"]["HostConfig"]["Tmpfs"].update(
                      {"/jit": "size=1g,uid=65532,gid=65532,mode=0700,exec,nosuid,nodev"}),
                  ["agent: unexpected writable temporary filesystem",
                   "agent: scratch execution or device policy drift"]),
    "suid_scratch": (lambda c, n: c["agent"]["HostConfig"]["Tmpfs"].update({"/scratch": "size=1g,noexec,nodev"}),
                     ["agent: scratch execution or device policy drift"]),
}


@pytest.mark.parametrize("case", sorted(VIOLATIONS))
def test_each_rule_violation_fails_with_its_message(case):
    mutate, expected = VIOLATIONS[case]
    errors = check(mutate)
    for message in expected:
        assert message in errors, (case, errors)


def test_missing_and_duplicate_services_fail():
    containers, networks = compliant()
    errors = verify.violations([containers["agent"], containers["egress"]], networks, approved=APPROVED)
    assert "all three isolated services must exist" in errors
    errors = verify.violations([*containers.values(), copy.deepcopy(containers["engine"])], networks,
                               approved=APPROVED)
    assert "unexpected or duplicate service" in errors
    comfy = copy.deepcopy(containers["engine"])
    comfy["Config"]["Labels"]["com.docker.compose.service"] = "comfy"
    assert "unexpected or duplicate service" in verify.violations(
        [*containers.values(), comfy], networks, approved=APPROVED)


def test_an_empty_profile_approves_nothing():
    errors = check(approved={})
    assert "engine: image must match the operator-approved profile" in errors
    assert "engine: engine commit differs from approved profile" in errors
    assert "engine: host mount source differs from approved profile" in errors


def test_findings_never_echo_environment_values():
    errors = check(lambda c, n: _env_set(c["engine"], "HF_TOKEN", "hf_secretvalue"))
    assert not any("hf_secretvalue" in e for e in errors)


def test_gate_names_the_recipe_weight_sets():
    assert verify.REQUIRED_WEIGHT_SETS == MESH3D_RECIPE.weight_ids


# The model-store gate.

def _write(path: Path, data: bytes) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


@pytest.fixture
def stores(tmp_path):
    trellis, rembg = tmp_path / "trellis", tmp_path / "rembg" / "models"
    records = []
    for name, data in [("visualbruno/TRELLIS.2-4B-FP8/ckpts_fp8/a.safetensors", b"a" * 10),
                       ("visualbruno/TRELLIS.2-4B-FP8/pipeline_fp8.json", b"{}"),
                       ("microsoft/TRELLIS-image-large/ckpts/ss_dec.safetensors", b"m" * 7),
                       ("facebook/dinov3/model.safetensors", b"f" * 5)]:
        records.append({"store": "trellis", "file": name, **_write(trellis / name, data)})
    for name, data in [("birefnet-general/birefnet-general.onnx", b"o" * 9)]:
        records.append({"store": "rembg", "file": name, **_write(rembg / name, data)})
    # Host content the engine never sees: the store root's own hash cache and the
    # wrapper's helper directory are not mounted, so they are not inventory.
    (trellis / ".weights-verified.json").write_text("{}")
    _write(trellis / "microsoft" / "TRELLIS.2-4B" / "pipeline.json", b"{}")
    manifest = tmp_path / "model-manifest-3d.json"
    manifest.write_text(json.dumps(records))
    return {"trellis": trellis, "rembg": rembg}, manifest, records


def test_model_stores_matching_the_manifest_pass(stores):
    roots, manifest, _ = stores
    assert verify.verify_models(roots, manifest) == []


def test_model_gate_refuses_drift(stores, tmp_path):
    roots, manifest, records = stores
    extra = roots["trellis"] / "facebook" / "dinov3" / "extra.safetensors"
    extra.write_bytes(b"x")
    assert verify.verify_models(roots, manifest) == ["model inventory differs from approved manifest"]
    extra.unlink()
    target = roots["rembg"] / "birefnet-general" / "birefnet-general.onnx"
    target.write_bytes(b"p" * 9)
    assert verify.verify_models(roots, manifest) == ["model content differs from approved manifest"]
    target.write_bytes(b"o" * 8)
    assert verify.verify_models(roots, manifest) == ["model size differs from approved manifest"]
    target.unlink()
    assert verify.verify_models(roots, manifest) == ["model inventory differs from approved manifest"]
    target.write_bytes(b"o" * 9)
    link = roots["trellis"] / "visualbruno" / "link.safetensors"
    link.symlink_to(tmp_path / "elsewhere")
    assert verify.verify_models(roots, manifest) == ["model tree contains symlink"]


def test_model_gate_refuses_a_symlinked_store(stores, tmp_path):
    roots, manifest, _ = stores
    alias = tmp_path / "alias"
    alias.symlink_to(roots["rembg"])
    assert verify.verify_models({**roots, "rembg": alias}, manifest) == ["invalid model manifest or root"]


@pytest.mark.parametrize("record, message", [
    ({"store": "trellis", "file": "visualbruno/x.ckpt"}, "invalid model manifest path"),
    ({"store": "trellis", "file": "visualbruno/../../etc/passwd.json"}, "invalid model manifest path"),
    ({"store": "trellis", "file": "/abs/x.safetensors"}, "invalid model manifest path"),
    ({"store": "home", "file": "x.safetensors"}, "invalid model manifest path"),
    ({"store": "trellis", "file": "experiments/skeleton/model.safetensors"},
     "model manifest names a file outside the mounted subtrees"),
])
def test_model_manifest_entries_are_checked(stores, record, message):
    roots, manifest, records = stores
    manifest.write_text(json.dumps([*records, {**record, "bytes": 1, "sha256": "0" * 64}]))
    assert verify.verify_models(roots, manifest) == [message]


def test_model_gate_fails_closed_on_a_bad_manifest(stores):
    roots, manifest, _ = stores
    manifest.write_text("[]")
    assert verify.verify_models(roots, manifest) == ["invalid model manifest or root"]
    manifest.write_text("not json")
    assert verify.verify_models(roots, manifest) == ["model integrity verification failed"]
    manifest.unlink()
    assert verify.verify_models(roots, manifest) == ["model integrity verification failed"]


def _lock():
    sha = "1" * 64
    return {"schema_version": 1, "weights": [
        {"id": "trellis2-4b-fp8", "root": "trellis_models",
         "files": [{"dest": "visualbruno/TRELLIS.2-4B-FP8/a.safetensors", "size": 3, "sha256": sha}]},
        {"id": "trellis-image-large-ss-decoder", "root": "trellis_models",
         "files": [{"dest": "microsoft/TRELLIS-image-large/ckpts/b.json", "size": 4, "sha256": sha}]},
        {"id": "dinov3-vitl16", "root": "trellis_models",
         "files": [{"dest": "facebook/dinov3-vitl16/model.safetensors", "size": 5, "sha256": sha}]},
        {"id": "unirig-skeleton-ar256", "root": "unirig",
         "files": [{"dest": "experiments/skeleton/model.ckpt", "size": 6, "sha256": sha}]},
        {"id": "birefnet-general", "root": "u2net",
         "files": [{"dest": "models/birefnet-general/birefnet-general.onnx", "size": 7, "sha256": sha}]},
        {"id": "birefnet-general-lite", "root": "u2net",
         "files": [{"dest": "models/birefnet-general-lite/birefnet-general-lite.onnx", "size": 8, "sha256": sha}]},
    ]}


def test_manifest_from_lock_maps_the_mounted_stores():
    records = verify.manifest_from_lock(_lock())
    assert [(r["store"], r["file"], r["bytes"]) for r in records] == [
        ("trellis", "visualbruno/TRELLIS.2-4B-FP8/a.safetensors", 3),
        ("trellis", "microsoft/TRELLIS-image-large/ckpts/b.json", 4),
        ("trellis", "facebook/dinov3-vitl16/model.safetensors", 5),
        ("rembg", "birefnet-general/birefnet-general.onnx", 7),
        ("rembg", "birefnet-general-lite/birefnet-general-lite.onnx", 8),
    ]


def test_manifest_from_lock_refuses_unpinned_or_unmounted_sets():
    lock = _lock()
    lock["weights"][4]["files"][0]["sha256"] = None
    with pytest.raises(ValueError, match="sha256"):
        verify.manifest_from_lock(lock)
    with pytest.raises(ValueError, match="not mounted"):
        verify.manifest_from_lock(_lock(), ("unirig-skeleton-ar256",))


# The compose files agree with the gate.

def _compose():
    return yaml.safe_load((DEPLOY / "compose.yaml").read_text())


def _size(value: str) -> int:
    units = {"k": 1024, "m": 1024**2, "g": GiB}
    return int(value[:-1]) * units[value[-1]] if value[-1] in units else int(value)


def _default(value) -> str:
    """The `${VAR:-default}` default, or the literal value."""
    value = str(value)
    return value.split(":-", 1)[1].rstrip("}") if ":-" in value else value


def test_compose_is_a_separate_project_with_three_sandboxed_services():
    compose = _compose()
    assert compose["name"] == verify.PROJECT == "ogwp-hardened-3d"
    assert set(compose["services"]) == verify.SERVICES
    for name, service in compose["services"].items():
        assert service["user"] == "65532:65532", name
        assert service["read_only"] is True and service["cap_drop"] == ["ALL"], name
        assert service["security_opt"] == ["no-new-privileges:true"], name
        assert service["restart"] == "no" and service["init"] is True, name
        assert "ports" not in service and "privileged" not in service and "devices" not in service, name
        memory_max, cpu_max, pids_max = verify.BOUNDS[name]
        assert 0 < service["pids_limit"] <= pids_max, name
        assert _size(_default(service["mem_limit"])) <= memory_max, name
        assert service["memswap_limit"] == service["mem_limit"], name
        assert float(_default(service["cpus"])) * 10**9 <= cpu_max, name
        tmpfs = {entry.split(":", 1)[0]: entry.split(":", 1)[1] for entry in service["tmpfs"]}
        assert set(tmpfs) <= verify.TMPFS[name], name
        for target, options in tmpfs.items():
            flags = options.split(",")
            assert verify._tmpfs_size(options) and "nosuid" in flags and "nodev" in flags, (name, target)
            assert ("noexec" in flags) != (target in verify.EXEC_TMPFS.get(name, set())), (name, target)
    networks = compose["networks"]
    assert networks["workload"]["internal"] is True
    assert set(networks["workload"]["driver_opts"].values()) == {"isolated"}


def test_compose_engine_matches_the_reviewed_runtime():
    engine = _compose()["services"]["engine"]
    assert engine["image"].startswith("${OGWP_ENGINE_IMAGE:?")
    assert engine["entrypoint"] == verify.ENGINE_ENTRYPOINT
    assert engine["command"] == verify.ENGINE_COMMAND
    assert engine["working_dir"] == "/opt/aismith"
    assert engine["healthcheck"]["test"] == ["CMD", verify.ENGINE_PYTHON, "-c", verify.ENGINE_HEALTH]
    assert engine["networks"] == ["workload"] and engine["dns"] == ["127.0.0.1"]
    assert "env_file" not in engine and "network_mode" not in engine
    environment = engine["environment"]
    assert verify.ENGINE_ENV.items() <= environment.items()
    assert not any(verify.SECRET_NAME.search(key) for key in environment)
    memory = _size(_default(engine["mem_limit"]))
    assert verify.ENGINE_MEMORY_FLOOR <= memory <= verify.BOUNDS["engine"][0]
    assert _size(engine["shm_size"]) <= verify.SHM_MAX
    tmpfs = dict(entry.split(":", 1) for entry in engine["tmpfs"])
    # 256 MiB mesh + 4 x 20 MiB images, the GLB and the stage's intermediates.
    assert _size(verify._tmpfs_size(tmpfs["/scratch"])) >= 2 * GiB
    assert all(path.startswith(("/scratch/", "/jit/")) for path in verify.ENGINE_SCRATCH_DIRS)
    targets = {}
    for volume in engine["volumes"]:
        assert volume["type"] == "bind" and volume["read_only"] is True
        assert volume["bind"] == {"create_host_path": False}
        targets[volume["target"]] = volume["source"]
    assert set(targets) == verify.MOUNTS["engine"] - {verify.WSL_LIB, verify.WSL_DRIVERS}
    for subtree in verify.TRELLIS_SUBTREES:
        assert targets[f"{verify.MODELS}/{subtree}"].endswith("}/" + subtree)
    assert targets[verify.REMBG_MODELS].startswith("${OGWP_REMBG_MODELS_DIR:?")


def test_compose_agent_holds_the_credential_and_no_gpu():
    agent = _compose()["services"]["agent"]
    assert agent["network_mode"] == "service:engine"
    assert agent["depends_on"]["engine"] == {"condition": "service_healthy"}
    assert agent["entrypoint"] == ["python", "-m", "outbound_gpu_worker_pool.agent_main"]
    environment = agent["environment"]
    assert verify.AGENT_ENV.items() <= environment.items()
    assert environment["MVS_WORKER_PROFILE"] == MESH3D
    assert int(environment["OGWP_WORKER_MAX_INPUT_BYTES"]) == verify.AGENT_MAX_INPUT_BYTES
    assert environment["OGWP_COMFY_START_COMMAND"] == "" and environment["MVS_ENGINE_COMMAND"] == ""
    assert [v["target"] for v in agent["volumes"]] == [verify.CREDENTIAL]
    assert agent["volumes"][0]["read_only"] is True
    assert "devices" not in agent and "networks" not in agent
    # The drain outlasts a mesh lease at its full deadline.
    mesh = MESH3D_RECIPE.stage_for("mesh.trellis2.image_to_mesh.v1")
    assert int(agent["stop_grace_period"].rstrip("m")) * 60 > mesh.deadline_seconds


def test_compose_egress_is_the_only_outbound_service():
    services = _compose()["services"]
    assert services["egress"]["networks"] == ["workload", "internet"]
    assert services["egress"]["entrypoint"] == ["python", "/opt/ogwp/egress_proxy.py"]
    assert services["egress"]["image"] == services["agent"]["image"]
    assert services["engine"]["networks"] == ["workload"]


def test_wsl_overlay_gives_the_gpu_to_the_engine_only():
    overlay = yaml.safe_load((DEPLOY / "wsl.yaml").read_text())
    assert set(overlay["services"]) == {"engine"}
    engine = overlay["services"]["engine"]
    assert engine["devices"] == ["/dev/dxg:/dev/dxg:rw"]
    assert {v["target"] for v in engine["volumes"]} == {verify.WSL_LIB, verify.WSL_DRIVERS}
    assert all(v["read_only"] is True and v["source"] == v["target"] for v in engine["volumes"])


def test_launcher_and_unit_mirror_the_h3_supervisor():
    run = (DEPLOY / "run.sh").read_text()
    order = ["other_running; then", "--models-only", "up -d --force-recreate --wait",
             "up --no-start --force-recreate --no-deps agent", "--allow-stopped-agent",
             "OGWP_PROBE_ROLE=engine", "start agent", "OGWP_PROBE_ROLE=agent",
             'echo "GPU isolation verified; 3D worker is running under the gated supervisor."',
             "while sleep 30"]
    positions = [run.index(step) for step in order]
    assert positions == sorted(positions)
    assert "stop --timeout 2100 agent" in run and run.index("stop --timeout 2100 agent") < run.index(
        "stop --timeout 30 engine egress")
    unit = (DEPLOY / "ogwp-hardened-3d.service").read_text()
    assert "ExecStart=/bin/bash %h/.local/share/ogwp-hardened-3d/hardened-3d/run.sh" in unit
    assert "RestartPreventExitStatus=3" in unit and "KillMode=process" in unit
    assert "TimeoutStopSec=37min" in unit  # longer than the agent's 35-minute drain
    switch = (DEPLOY / "switch.sh").read_text()
    assert switch.count("GPU isolation verified;") == 2


@pytest.mark.parametrize("script", ["run.sh", "switch.sh"])
def test_shell_scripts_parse(script):
    subprocess.run(["bash", "-n", str(DEPLOY / script)], check=True)
    assert os.access(DEPLOY / script, os.X_OK)


def test_egress_relay_allows_only_exact_hosts_on_443():
    hosts = egress.allowed_hosts("pool.example.com,storage.googleapis.com")
    assert egress.destination(b"CONNECT pool.example.com:443 HTTP/1.1\r\n\r\n", hosts) == "pool.example.com"
    for header in (b"CONNECT pool.example.com:80 HTTP/1.1\r\n\r\n", b"CONNECT example.com:443 HTTP/1.1\r\n\r\n",
                   b"GET http://pool.example.com/ HTTP/1.1\r\n\r\n"):
        with pytest.raises(ValueError):
            egress.destination(header, hosts)
    for value in ("*.example.com", "Pool.example.com", "localhost", ""):
        with pytest.raises(ValueError):
            egress.allowed_hosts(value)


# The time-share gate.

NOW = 1_800_000_000.0


def _evidence(**overrides):
    evidence = {
        "units": {"h3": "active", "3d": "inactive"},
        "enabled": {"h3": "enabled", "3d": "disabled"},
        "running": {"h3": 3, "3d": 0},
        "h3_queue": {"queue_running": [], "queue_pending": []},
        "h3_history": {"p1": {"status": {"messages": [
            ["execution_start", {"timestamp": (NOW - 3600) * 1000}],
            ["execution_success", {"timestamp": (NOW - 900) * 1000}]]}}},
        "gpu_utilization": [0, 1, 3, 0, 2],
    }
    evidence.update(overrides)
    return evidence


def _to_h3(**overrides):
    return _evidence(units={"h3": "inactive", "3d": "active"}, running={"h3": 0, "3d": 3},
                     h3_queue=None, h3_history=None,
                     engine_health=[{"ok": True, "busy": False, "schema_version": 1}] * 3, **overrides)


def test_gate_passes_an_idle_h3_to_3d_switch():
    assert gate.evaluate("3d", _evidence(), NOW) == []
    assert gate.evaluate("3d", _evidence(h3_history={}), NOW) == []


@pytest.mark.parametrize("overrides, reason", [
    ({"h3_queue": {"queue_running": [[1, "p"]], "queue_pending": []}}, "H3 ComfyUI queue is not empty"),
    ({"h3_queue": {"queue_running": [], "queue_pending": [[2, "q"]]}}, "H3 ComfyUI queue is not empty"),
    ({"h3_history": {"p": {"status": {"messages": [["execution_success", {"timestamp": (NOW - 120) * 1000}]]}}}},
     "H3 history active within 10 minutes"),
    ({"h3_queue": None}, "H3 runtime queue unavailable"),
    ({"h3_history": {"p": {"status": {}}}}, "malformed gate evidence"),
    ({"gpu_utilization": [0, 0, 40, 0, 0]}, "GPU busy: utilization up to 40%"),
    ({"gpu_utilization": [0, 0, 6, 0, 0]}, "GPU busy: utilization up to 6%"),
    ({"gpu_utilization": [0, 0]}, "GPU utilization unavailable"),
    ({"gpu_utilization": [0, None, 0, 0, 0]}, "GPU utilization unavailable"),
    ({"gpu_utilization": []}, "GPU utilization unavailable"),
    ({"units": {"h3": "active", "3d": "active"}}, "3d is already running; the GPU runs one worker at a time"),
    ({"units": {"h3": "active", "3d": "activating"}}, "3d is already running; the GPU runs one worker at a time"),
    ({"running": {"h3": 3, "3d": 1}}, "3d is already running; the GPU runs one worker at a time"),
])
def test_gate_refuses_h3_to_3d(overrides, reason):
    assert reason in gate.evaluate("3d", _evidence(**overrides), NOW)


def test_gate_needs_no_runtime_evidence_from_a_stopped_source():
    stopped = _evidence(units={"h3": "failed", "3d": "inactive"}, running={"h3": 0, "3d": 0},
                        h3_queue=None, h3_history=None)
    assert gate.evaluate("3d", stopped, NOW) == []
    # A stopped unit with live containers is still running.
    leftover = _evidence(units={"h3": "inactive", "3d": "inactive"}, running={"h3": 1, "3d": 0}, h3_queue=None)
    assert "H3 runtime queue unavailable" in gate.evaluate("3d", leftover, NOW)


def test_gate_for_3d_to_h3():
    assert gate.evaluate("h3", _to_h3(), NOW) == []
    busy = [{"ok": True, "busy": False}, {"ok": True, "busy": True}, {"ok": True, "busy": False}]
    assert gate.evaluate("h3", _to_h3() | {"engine_health": busy}, NOW) == ["3D engine runner is busy"]
    assert gate.evaluate("h3", _to_h3() | {"engine_health": None}, NOW) == ["3D engine runner health unavailable"]
    assert gate.evaluate("h3", _to_h3() | {"engine_health": [{"ok": False}] * 3}, NOW) == [
        "3D engine runner health unavailable"]
    assert "GPU busy: utilization up to 90%" in gate.evaluate(
        "h3", _to_h3() | {"gpu_utilization": [90] * 5}, NOW)
    assert "h3 is already running; the GPU runs one worker at a time" in gate.evaluate(
        "h3", _to_h3() | {"running": {"h3": 2, "3d": 3}}, NOW)


def test_gate_refuses_malformed_evidence():
    assert gate.evaluate("3d", {}, NOW) == ["malformed gate evidence"]
    assert gate.evaluate("gpu", _evidence(), NOW) == ["unknown target"]


def test_postcheck_requires_exactly_one_enabled_worker():
    done = {"units": {"h3": "inactive", "3d": "active"}, "enabled": {"h3": "disabled", "3d": "enabled"},
            "running": {"h3": 0, "3d": 3}}
    assert gate.postcheck("3d", done) == []
    assert gate.postcheck("3d", done | {"enabled": {"h3": "enabled", "3d": "enabled"}}) == [
        "h3 unit is still active or enabled"]
    assert gate.postcheck("3d", done | {"running": {"h3": 1, "3d": 3}}) == ["h3 still has running containers"]
    assert gate.postcheck("3d", done | {"running": {"h3": 0, "3d": 2}}) == [
        "3d does not have exactly 3 running containers"]
    assert gate.postcheck("h3", done) != []


def test_parse_utilization_takes_the_busiest_gpu():
    assert gate.parse_utilization("3\n17\n") == 17
    with pytest.raises(ValueError):
        gate.parse_utilization("")
    with pytest.raises(ValueError):
        gate.parse_utilization("[N/A]\n")


def _fake_run(responses, calls):
    def run(command):
        calls.append(command)
        for match, reply in responses:
            if all(part in command for part in match):
                return reply
        return 0, ""
    return run


def test_collect_reads_the_source_runtime_through_docker_exec(monkeypatch):
    monkeypatch.setattr(gate.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    queue = json.dumps({"queue_running": [], "queue_pending": []})
    calls, sleeps = [], []
    run = _fake_run([
        (["is-active", "ogwp-hardened.service"], (0, "active\n")),
        (["is-active", "ogwp-hardened-3d.service"], (3, "inactive\n")),
        (["label=com.docker.compose.service=comfy"], (0, "comfyid\n")),
        (["label=com.docker.compose.project=ogwp-hardened"], (0, "a\nb\nc\n")),
        (["exec", "http://127.0.0.1:8188/queue"], (0, queue)),
        (["exec", "http://127.0.0.1:8188/history?max_items=64"], (0, "{}")),
        (["/usr/bin/nvidia-smi"], (0, "2\n")),
    ], calls)
    evidence = gate.collect("3d", run=run, sleep=sleeps.append)
    assert evidence["units"] == {"h3": "active", "3d": "inactive"}
    assert evidence["running"] == {"h3": 3, "3d": 0}
    assert evidence["h3_queue"] == {"queue_running": [], "queue_pending": []} and evidence["h3_history"] == {}
    assert evidence["gpu_utilization"] == [2] * gate.GPU_SAMPLES
    assert sleeps == [gate.SAMPLE_INTERVAL_SECONDS] * (gate.GPU_SAMPLES - 1)
    assert ["docker", "exec", "comfyid", gate.COMFY_PYTHON, "-c", gate.FETCH,
            "http://127.0.0.1:8188/queue"] in calls
    assert gate.evaluate("3d", evidence, NOW) == []


def test_collect_fails_closed_when_docker_is_unreadable(monkeypatch):
    monkeypatch.setattr(gate.shutil, "which", lambda name: None)
    monkeypatch.setattr(gate, "WSL_NVIDIA_SMI", "/nonexistent/nvidia-smi")
    run = _fake_run([(["docker"], (1, ""))], [])
    evidence = gate.collect("h3", run=run, sleep=lambda s: None)
    assert evidence["running"] == {"h3": 99, "3d": 99}
    reasons = gate.evaluate("h3", evidence, NOW)
    assert "h3 is already running; the GPU runs one worker at a time" in reasons
    assert "GPU utilization unavailable" in reasons


# switch.sh end to end against fake systemctl/docker/journalctl/nvidia-smi.

FAKE_HOST = r'''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
state_path = Path(os.environ["FAKE_STATE"])
state = json.loads(state_path.read_text())
tool, args = Path(sys.argv[0]).name, sys.argv[1:]
with open(os.environ["FAKE_LOG"], "a") as log:
    log.write(" ".join([tool, *args]) + "\n")
units = {"ogwp-hardened.service": "h3", "ogwp-hardened-3d.service": "3d"}
projects = {"ogwp-hardened": "h3", "ogwp-hardened-3d": "3d"}
def save():
    state_path.write_text(json.dumps(state))
if tool == "systemctl":
    args = [a for a in args if a != "--user" and a != "--quiet"]
    verb, name = args[0], args[-1]
    key = units.get(name)
    if verb == "is-active":
        print(state["units"][key]); sys.exit(0 if state["units"][key] == "active" else 3)
    if verb == "is-enabled":
        print(state["enabled"][key]); sys.exit(0 if state["enabled"][key] == "enabled" else 1)
    if verb == "is-failed":
        sys.exit(0 if state["units"][key] == "failed" else 1)
    if verb == "disable":
        state["enabled"][key] = "disabled"; state["units"][key] = "inactive"; state["running"][key] = 0
    if verb == "enable":
        state["enabled"][key] = "enabled"; state["units"][key] = "active"; state["running"][key] = 3
    save(); sys.exit(0)
if tool == "docker":
    if args[0] == "ps":
        filters = [args[i + 1] for i, a in enumerate(args) if a == "--filter"]
        project = next(f.split("compose.project=", 1)[1] for f in filters if "compose.project=" in f)
        count = state["running"][projects[project]]
        service = [f for f in filters if "compose.service=" in f]
        print("\n".join(f"{project}-{i}" for i in range(1 if service and count else count)))
        sys.exit(0)
    if args[0] == "exec":
        url = args[-1]
        if url.endswith("/queue"):
            print(json.dumps({"queue_running": [], "queue_pending": state["pending"]}))
        elif "/history" in url:
            print("{}")
        else:
            print(json.dumps({"ok": True, "busy": False}))
        sys.exit(0)
if tool == "journalctl":
    if state["units"]["3d"] == "active":
        print("GPU isolation verified; 3D worker is running under the gated supervisor.")
    sys.exit(0)
if tool == "nvidia-smi":
    print(state["gpu"]); sys.exit(0)
sys.exit(0)
'''


@pytest.fixture
def fake_host(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = tmp_path / "fake_host.py"
    fake.write_text(FAKE_HOST)
    fake.chmod(0o755)
    for tool in ("systemctl", "docker", "journalctl", "nvidia-smi"):
        (bin_dir / tool).symlink_to(fake)
    verify_dir = tmp_path / "share-3d" / "hardened-3d"
    verify_dir.mkdir(parents=True)
    (verify_dir / "verify.py").write_text("print('{\"passed\": true}')\n")
    state = tmp_path / "state.json"
    state.write_text(json.dumps({
        "units": {"h3": "active", "3d": "inactive"}, "enabled": {"h3": "enabled", "3d": "disabled"},
        "running": {"h3": 3, "3d": 0}, "pending": [], "gpu": 1}))
    log = tmp_path / "calls.log"
    log.write_text("")
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
           "FAKE_STATE": str(state), "FAKE_LOG": str(log), "OGWP_3D_DIR": str(tmp_path / "share-3d"),
           "OGWP_H3_DIR": str(tmp_path / "share-h3")}
    # switch_gate.py sleeps between GPU samples; keep the test fast.
    shim = tmp_path / "shim"
    shim.mkdir()
    (shim / "sitecustomize.py").write_text("import time\ntime.sleep = lambda seconds: None\n")
    env["PYTHONPATH"] = str(shim)

    def switch(*args):
        return subprocess.run(["bash", str(DEPLOY / "switch.sh"), *args], env=env,
                              capture_output=True, text=True, timeout=120)
    return switch, state, log


def _mutating(log):
    return [line for line in log.read_text().splitlines()
            if line.startswith("systemctl") and any(v in line.split() for v in ("disable", "enable", "start", "stop"))]


def test_switch_is_a_dry_run_by_default(fake_host):
    switch, state, log = fake_host
    result = switch("--to", "3d")
    assert result.returncode == 0, result.stderr
    assert "Dry run" in result.stderr
    assert _mutating(log) == []


def test_switch_refuses_without_acting_when_the_gate_fails(fake_host):
    switch, state, log = fake_host
    current = json.loads(state.read_text())
    state.write_text(json.dumps(current | {"pending": [[1, "prompt"]]}))
    result = switch("--to", "3d", "--apply")
    assert result.returncode == 1
    assert "Gate refused; nothing changed." in result.stderr
    assert "H3 ComfyUI queue is not empty" in result.stdout
    assert _mutating(log) == []


def test_switch_apply_drains_then_starts_the_other_worker(fake_host):
    switch, state, log = fake_host
    result = switch("--to", "3d", "--apply")
    assert result.returncode == 0, result.stdout + result.stderr
    final = json.loads(state.read_text())
    assert final["units"] == {"h3": "inactive", "3d": "active"}
    assert final["enabled"] == {"h3": "disabled", "3d": "enabled"}
    mutating = _mutating(log)
    assert mutating[0] == "systemctl --user disable --now ogwp-hardened.service"
    assert mutating[-1] == "systemctl --user enable --now ogwp-hardened-3d.service"


def test_switch_usage_errors(fake_host):
    switch, _, log = fake_host
    assert switch().returncode == 2
    assert switch("--to", "both").returncode == 2
    assert _mutating(log) == []
