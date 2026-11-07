"""The box-local gate for switching the 4090 between H3 and the 3D worker.

The GPU runs EITHER `ogwp-hardened` (H3) OR `ogwp-hardened-3d`, never both.
`switch.sh` asks this gate before it drains one and starts the other:

- `gate --to 3d|h3`: pass only when the target is not running, the GPU is idle
  (every sample at most 5% utilization), and the source is quiet: for H3 an
  empty ComfyUI queue and no history activity for 10 minutes; for the 3D
  worker an engine runner that reports `busy: false` on every sample.
- `postcheck --to 3d|h3`: after the switch, the target unit is active and
  enabled with its three containers running, and the source unit is inactive,
  disabled and has no running container.

Evidence comes from the trusted host only: systemd, `docker ps`, `docker exec`
into the source's own runtime container (its loopback API), and `nvidia-smi`.
Anything missing or malformed refuses. Prints JSON findings, never secrets.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

TARGETS = ("h3", "3d")
UNITS = {"h3": "ogwp-hardened.service", "3d": "ogwp-hardened-3d.service"}
PROJECTS = {"h3": "ogwp-hardened", "3d": "ogwp-hardened-3d"}
SERVICE_COUNT = 3
QUIET_SECONDS = 600
GPU_IDLE_PERCENT = 5
GPU_SAMPLES = 5
HEALTH_SAMPLES = 3
SAMPLE_INTERVAL_SECONDS = 2
STOPPED_STATES = {"inactive", "failed"}
FETCH = "import sys,urllib.request; sys.stdout.write(urllib.request.urlopen(sys.argv[1], timeout=5).read().decode())"
COMFY_PYTHON = "/opt/comfy/venv/bin/python"
ENGINE_PYTHON = "/opt/aismith/.venv/bin/python"
WSL_NVIDIA_SMI = "/usr/lib/wsl/lib/nvidia-smi"


def other(target: str) -> str:
    return "h3" if target == "3d" else "3d"


def latest_history_seconds(history) -> float | None:
    """The newest event time in a ComfyUI `/history` reply, in epoch seconds; None if empty."""
    if not isinstance(history, dict):
        raise ValueError("history is not an object")
    latest = None
    for entry in history.values():
        for message in entry["status"]["messages"]:
            stamp = message[1].get("timestamp")
            if not isinstance(stamp, (int, float)) or isinstance(stamp, bool):
                raise ValueError("history timestamp missing")
            latest = stamp / 1000 if latest is None else max(latest, stamp / 1000)
    return latest


def _running(target: str, evidence: dict) -> bool:
    return (evidence["units"][target] not in STOPPED_STATES) or evidence["running"][target] > 0


def evaluate(target: str, evidence: dict, now: float) -> list[str]:
    """Reasons to refuse switching to `target`; empty means the gate passes."""
    if target not in TARGETS:
        return ["unknown target"]
    source = other(target)
    reasons = []
    try:
        if _running(target, evidence):
            reasons.append(f"{target} is already running; the GPU runs one worker at a time")
        samples = evidence.get("gpu_utilization") or []
        if len(samples) < GPU_SAMPLES or not all(isinstance(s, int) and 0 <= s <= 100 for s in samples):
            reasons.append("GPU utilization unavailable")
        elif max(samples) > GPU_IDLE_PERCENT:
            reasons.append(f"GPU busy: utilization up to {max(samples)}%")
        if _running(source, evidence):
            if source == "h3":
                queue, history = evidence.get("h3_queue"), evidence.get("h3_history")
                if not isinstance(queue, dict) or history is None:
                    reasons.append("H3 runtime queue unavailable")
                else:
                    if queue["queue_running"] or queue["queue_pending"]:
                        reasons.append("H3 ComfyUI queue is not empty")
                    latest = latest_history_seconds(history)
                    if latest is not None and now - latest < QUIET_SECONDS:
                        reasons.append(f"H3 history active within {QUIET_SECONDS // 60} minutes")
            else:
                health = evidence.get("engine_health") or []
                if len(health) < HEALTH_SAMPLES or not all(isinstance(h, dict) and h.get("ok") is True
                                                           for h in health):
                    reasons.append("3D engine runner health unavailable")
                elif any(h.get("busy") is not False for h in health):
                    reasons.append("3D engine runner is busy")
    except (KeyError, IndexError, TypeError, AttributeError, ValueError):
        reasons.append("malformed gate evidence")
    return reasons


def postcheck(target: str, evidence: dict) -> list[str]:
    """Reasons the finished switch is not in the one-worker state."""
    if target not in TARGETS:
        return ["unknown target"]
    source = other(target)
    reasons = []
    try:
        if evidence["units"][target] != "active" or evidence["enabled"][target] != "enabled":
            reasons.append(f"{target} unit is not active and enabled")
        if evidence["running"][target] != SERVICE_COUNT:
            reasons.append(f"{target} does not have exactly {SERVICE_COUNT} running containers")
        if evidence["units"][source] not in STOPPED_STATES or evidence["enabled"][source] == "enabled":
            reasons.append(f"{source} unit is still active or enabled")
        if evidence["running"][source] != 0:
            reasons.append(f"{source} still has running containers")
    except (KeyError, TypeError):
        reasons.append("malformed postcheck evidence")
    return reasons


def parse_utilization(output: str) -> int:
    values = [int(line.strip()) for line in output.splitlines() if line.strip()]
    if not values:
        raise ValueError("no GPU reported")
    return max(values)


def _run(command: list[str]) -> tuple[int, str]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return 1, ""
    return completed.returncode, completed.stdout


def _container(run, project: str, service: str) -> str | None:
    code, out = run(["docker", "ps", "-q", "--filter", f"label=com.docker.compose.project={project}",
                     "--filter", f"label=com.docker.compose.service={service}", "--filter", "status=running"])
    ids = out.split() if code == 0 else []
    return ids[0] if len(ids) == 1 else None


def _fetch_json(run, container: str, python: str, url: str):
    code, out = run(["docker", "exec", container, python, "-c", FETCH, url])
    return json.loads(out) if code == 0 else None


def collect(target: str, *, run=_run, sleep=time.sleep, gate=True) -> dict:
    """Host evidence for `evaluate` (gate) or `postcheck`."""
    evidence: dict = {"units": {}, "enabled": {}, "running": {}}
    for name in TARGETS:
        evidence["units"][name] = run(["systemctl", "--user", "is-active", UNITS[name]])[1].strip() or "unknown"
        evidence["enabled"][name] = run(["systemctl", "--user", "is-enabled", UNITS[name]])[1].strip() or "unknown"
        code, out = run(["docker", "ps", "-q", "--filter", f"label=com.docker.compose.project={PROJECTS[name]}",
                         "--filter", "status=running"])
        # An unreadable Docker state counts as running: refuse rather than guess.
        evidence["running"][name] = len(out.split()) if code == 0 else 99
    if not gate:
        return evidence
    source = other(target)
    try:
        if source == "h3":
            comfy = _container(run, PROJECTS["h3"], "comfy")
            if comfy:
                evidence["h3_queue"] = _fetch_json(run, comfy, COMFY_PYTHON, "http://127.0.0.1:8188/queue")
                evidence["h3_history"] = _fetch_json(run, comfy, COMFY_PYTHON,
                                                     "http://127.0.0.1:8188/history?max_items=64")
        else:
            engine = _container(run, PROJECTS["3d"], "engine")
            if engine:
                evidence["engine_health"] = []
                for index in range(HEALTH_SAMPLES):
                    if index:
                        sleep(SAMPLE_INTERVAL_SECONDS)
                    evidence["engine_health"].append(
                        _fetch_json(run, engine, ENGINE_PYTHON, "http://127.0.0.1:8190/health"))
    except ValueError:
        pass  # malformed runtime replies leave the evidence missing: the gate refuses
    smi = shutil.which("nvidia-smi") or (WSL_NVIDIA_SMI if Path(WSL_NVIDIA_SMI).is_file() else None)
    samples = []
    if smi:
        for index in range(GPU_SAMPLES):
            if index:
                sleep(SAMPLE_INTERVAL_SECONDS)
            code, out = run([smi, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
            try:
                samples.append(parse_utilization(out) if code == 0 else None)
            except ValueError:
                samples.append(None)
    evidence["gpu_utilization"] = samples
    return evidence


def summary(evidence: dict) -> dict:
    """What the gate saw, without runtime payloads."""
    return {
        "units": evidence.get("units"), "enabled": evidence.get("enabled"),
        "running": evidence.get("running"), "gpu_utilization": evidence.get("gpu_utilization"),
        "h3_queue_seen": evidence.get("h3_queue") is not None,
        "engine_health_samples": len(evidence.get("engine_health") or []),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["gate", "postcheck"])
    parser.add_argument("--to", required=True, choices=TARGETS)
    args = parser.parse_args(argv)
    if args.mode == "gate":
        evidence = collect(args.to)
        reasons = evaluate(args.to, evidence, time.time())
    else:
        evidence = collect(args.to, gate=False)
        reasons = postcheck(args.to, evidence)
    print(json.dumps({"mode": args.mode, "target": args.to, "passed": not reasons,
                      "reasons": reasons, "evidence": summary(evidence)}, indent=2))
    return 1 if reasons else 0


if __name__ == "__main__":
    sys.exit(main())
