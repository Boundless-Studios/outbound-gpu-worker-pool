"""Packaging tests: the operator-facing scripts and the systemd user unit.

Everything here runs without root, without systemd, and without network. The
shells are parsed rather than executed where execution would need a machine
(`check-exposure.sh` needs `ss` and a live agent), the unit's required settings
are asserted directly because `systemd-analyze` is not a test dependency, and
`install.sh` runs against a throwaway `OGWP_HOME` with stub `python3` and `pip`
on `PATH`, so no venv is built and nothing is fetched.
"""

import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEPLOY_DIRECTORY = REPOSITORY_ROOT / "deploy" / "agent"
INSTALL_SCRIPT = DEPLOY_DIRECTORY / "install.sh"
EXPOSURE_SCRIPT = DEPLOY_DIRECTORY / "check-exposure.sh"
UNIT_FILE = DEPLOY_DIRECTORY / "outbound-gpu-worker.service"
AGENT_MAIN = REPOSITORY_ROOT / "src" / "outbound_gpu_worker_pool" / "agent_main.py"
PACKAGED_TEMPLATE = "minimax_h3_text_to_video.template.json"

REQUIREMENT = (
    "outbound-gpu-worker-pool[agent,comfy,google-auth] @ "
    "git+https://github.com/Boundless-Studios/outbound-gpu-worker-pool@"
)

# The unit is the whole supervision contract: drain on SIGTERM with room for a
# 20 minute job, restart on its own, and write nowhere but the agent's home.
REQUIRED_UNIT_SETTINGS = {
    "Type": "simple",
    "EnvironmentFile": "%h/.local/share/outbound-gpu-worker/agent.env",
    "ExecStart": (
        "%h/.local/share/outbound-gpu-worker/venv/bin/python "
        "-m outbound_gpu_worker_pool.agent_main"
    ),
    "Restart": "always",
    "RestartSec": "10",
    "KillSignal": "SIGTERM",
    "TimeoutStopSec": "1500",
    "NoNewPrivileges": "yes",
    "PrivateTmp": "yes",
    "ProtectSystem": "strict",
    "ProtectHome": "read-only",
    "ReadWritePaths": "%h/.local/share/outbound-gpu-worker",
    "WantedBy": "default.target",
}


def _agent_environment_variables() -> set[str]:
    """Every `OGWP_WORKER_*` / `OGWP_COMFY_*` name the agent entrypoint reads.

    Read out of the source so the env template cannot drift behind a new knob.
    """
    return set(
        re.findall(
            r"OGWP_(?:WORKER|COMFY)_[A-Z_]+", AGENT_MAIN.read_text(encoding="utf-8")
        )
    )


def _unit_settings(text: str) -> dict[str, str]:
    settings: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";", "[")):
            continue
        key, separator, value = line.partition("=")
        assert separator, f"not a unit setting: {raw}"
        settings[key.strip()] = value.strip()
    return settings


def _stub_bin(directory: Path, pip_log: Path) -> Path:
    """A `python3` that only fakes `-m venv`, and a `pip` that logs its argv."""
    directory.mkdir(parents=True)
    (directory / "python3").write_text(
        "#!/bin/sh\n# install.sh only ever runs `python3 -m venv <dir>`.\n"
        'mkdir -p "$3/bin"\n',
        encoding="utf-8",
    )
    (directory / "pip").write_text(
        f'#!/bin/sh\nprintf "%s\\n" "$*" >> "{pip_log}"\n', encoding="utf-8"
    )
    for name in ("python3", "pip"):
        (directory / name).chmod(0o755)
    return directory


def _run_install(
    *, agent_home: Path, stub_bin: Path, user_home: Path
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ) | {
        "PATH": f"{stub_bin}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(user_home),
        "XDG_CONFIG_HOME": str(user_home / ".config"),
        "OGWP_HOME": str(agent_home),
    }
    return subprocess.run(
        [str(INSTALL_SCRIPT)],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )


@pytest.mark.parametrize(
    "script", [INSTALL_SCRIPT, EXPOSURE_SCRIPT], ids=lambda path: path.name
)
def test_a_the_operator_scripts_parse_and_are_executable(script: Path) -> None:
    assert script.stat().st_mode & stat.S_IXUSR, f"{script.name} is not executable"

    subprocess.run(["bash", "-n", str(script)], check=True)


def test_b_the_unit_declares_the_supervision_contract() -> None:
    text = UNIT_FILE.read_text(encoding="utf-8")

    for section in ("[Unit]", "[Service]", "[Install]"):
        assert section in text
    settings = _unit_settings(text)
    for key, value in REQUIRED_UNIT_SETTINGS.items():
        assert settings.get(key) == value, f"{key} is {settings.get(key)!r}"


def test_c_install_provisions_the_agent_home_without_root(tmp_path: Path) -> None:
    agent_home = tmp_path / "agent"
    user_home = tmp_path / "home"
    pip_log = tmp_path / "pip.log"
    stub_bin = _stub_bin(tmp_path / "bin", pip_log)

    result = _run_install(agent_home=agent_home, stub_bin=stub_bin, user_home=user_home)

    unit = user_home / ".config" / "systemd" / "user" / UNIT_FILE.name
    environment_file = agent_home / "agent.env"
    assert (agent_home / "venv").is_dir()
    assert (agent_home / "workspace").is_dir()
    assert (agent_home / "templates" / PACKAGED_TEMPLATE).is_file()
    assert unit.read_text(encoding="utf-8") == UNIT_FILE.read_text(encoding="utf-8")
    # The env file carries the worker token, so it is never world readable.
    assert stat.S_IMODE(environment_file.stat().st_mode) == 0o600
    template = environment_file.read_text(encoding="utf-8")
    for name in _agent_environment_variables() | {"GOOGLE_APPLICATION_CREDENTIALS"}:
        assert f"#{name}=" in template, f"{name} is not in the env template"
    # A coordinator-only secret must never be suggested to a worker machine.
    assert "OGWP_WORKER_TOKENS=" not in template
    assert f"{REQUIREMENT}main" in pip_log.read_text(encoding="utf-8")
    assert "systemctl --user enable --now outbound-gpu-worker" in result.stdout
    assert "loginctl enable-linger" in result.stdout
    assert "sudo" not in result.stdout


def test_c_install_is_idempotent_and_keeps_operator_curation(tmp_path: Path) -> None:
    agent_home = tmp_path / "agent"
    user_home = tmp_path / "home"
    pip_log = tmp_path / "pip.log"
    stub_bin = _stub_bin(tmp_path / "bin", pip_log)
    _run_install(agent_home=agent_home, stub_bin=stub_bin, user_home=user_home)

    environment_file = agent_home / "agent.env"
    environment_file.write_text(
        environment_file.read_text(encoding="utf-8") + "OGWP_WORKER_ID=gpu-01\n",
        encoding="utf-8",
    )
    curated = agent_home / "templates" / PACKAGED_TEMPLATE
    curated.write_text("{}\n", encoding="utf-8")

    _run_install(agent_home=agent_home, stub_bin=stub_bin, user_home=user_home)

    # A rerun is the upgrade path: it reinstalls, and it edits nothing the
    # operator filled in.
    assert "OGWP_WORKER_ID=gpu-01" in environment_file.read_text(encoding="utf-8")
    assert stat.S_IMODE(environment_file.stat().st_mode) == 0o600
    assert curated.read_text(encoding="utf-8") == "{}\n"
    assert pip_log.read_text(encoding="utf-8").count(REQUIREMENT) == 2


def test_d_a_pinned_ref_is_what_gets_installed(tmp_path: Path) -> None:
    agent_home = tmp_path / "agent"
    user_home = tmp_path / "home"
    pip_log = tmp_path / "pip.log"
    stub_bin = _stub_bin(tmp_path / "bin", pip_log)
    environment = dict(os.environ) | {
        "PATH": f"{stub_bin}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(user_home),
        "XDG_CONFIG_HOME": str(user_home / ".config"),
        "OGWP_HOME": str(agent_home),
        "OGWP_REF": "0123456789abcdef",
    }

    subprocess.run(
        [str(INSTALL_SCRIPT)],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )

    assert f"{REQUIREMENT}0123456789abcdef" in pip_log.read_text(encoding="utf-8")


SS_HEADER = "State Recv-Q Send-Q Local Address:Port Peer Address:Port Process\n"
COMFY_ON_LOOPBACK = (
    'LISTEN 0 4096 127.0.0.1:8188 0.0.0.0:* users:(("python",pid=4242,fd=7))\n'
)
COMFY_ON_EVERY_INTERFACE = (
    'LISTEN 0 4096 0.0.0.0:8188 0.0.0.0:* users:(("python",pid=4242,fd=7))\n'
)
COMFY_ON_IPV6_WILDCARD = (
    'LISTEN 0 4096 [::]:8189 [::]:* users:(("python",pid=4242,fd=7))\n'
)
SSH_ON_EVERY_INTERFACE = "LISTEN 0 128 0.0.0.0:22 0.0.0.0:*\n"
AGENT_LISTENING = (
    'LISTEN 0 4096 127.0.0.1:9999 0.0.0.0:* users:(("python",pid=777,fd=9))\n'
)


def _run_exposure_check(
    tmp_path: Path, *, listing: str, agent_pids: str = ""
) -> subprocess.CompletedProcess[str]:
    """Run the check against a stubbed `ss` and `pgrep`.

    The real ones need a Linux box with the agent running, which is exactly
    what the script is for; the parsing and the verdict are what a test can own.
    """
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir(parents=True)
    (stub_bin / "ss").write_text(
        f"#!/bin/sh\ncat <<'LISTING'\n{listing}LISTING\n", encoding="utf-8"
    )
    (stub_bin / "pgrep").write_text(
        f"#!/bin/sh\nprintf '%s' '{agent_pids}'\n" + ("" if agent_pids else "exit 1\n"),
        encoding="utf-8",
    )
    for name in ("ss", "pgrep"):
        (stub_bin / name).chmod(0o755)
    return subprocess.run(
        ["bash", str(EXPOSURE_SCRIPT)],
        env=dict(os.environ) | {"PATH": f"{stub_bin}{os.pathsep}{os.environ['PATH']}"},
        capture_output=True,
        text=True,
    )


def test_e_a_loopback_only_machine_passes_the_exposure_check(tmp_path: Path) -> None:
    result = _run_exposure_check(
        tmp_path,
        listing=SS_HEADER + COMFY_ON_LOOPBACK + SSH_ON_EVERY_INTERFACE,
        agent_pids="4243",
    )

    assert result.returncode == 0, result.stderr
    # The whole socket table is printed, so the operator sees what a scan would.
    assert "127.0.0.1:8188" in result.stdout
    assert "0.0.0.0:22" in result.stdout
    assert "ComfyUI bind: 127.0.0.1:8188" in result.stdout


@pytest.mark.parametrize(
    "listing",
    [
        pytest.param(COMFY_ON_EVERY_INTERFACE, id="ipv4-wildcard"),
        pytest.param(COMFY_ON_IPV6_WILDCARD, id="ipv6-wildcard"),
    ],
)
def test_e_comfyui_off_loopback_fails_the_exposure_check(
    tmp_path: Path, listing: str
) -> None:
    result = _run_exposure_check(tmp_path, listing=SS_HEADER + listing)

    assert result.returncode == 1
    assert "not loopback" in result.stderr


def test_e_a_listening_agent_fails_the_exposure_check(tmp_path: Path) -> None:
    result = _run_exposure_check(
        tmp_path,
        listing=SS_HEADER + COMFY_ON_LOOPBACK + AGENT_LISTENING,
        agent_pids="777",
    )

    assert result.returncode == 1
    assert "holds a listening socket" in result.stderr


def test_e_no_comfy_listener_is_reported_rather_than_guessed(tmp_path: Path) -> None:
    result = _run_exposure_check(tmp_path, listing=SS_HEADER + SSH_ON_EVERY_INTERFACE)

    assert result.returncode == 0, result.stderr
    assert "ComfyUI bind: none" in result.stdout
