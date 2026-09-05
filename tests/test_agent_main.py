"""Agent entrypoint tests: environment parsing and signal-driven draining."""

import asyncio
import os
import signal

import pytest

from outbound_gpu_worker_pool import agent_main
from outbound_gpu_worker_pool.agent import DEFAULT_MAX_INPUT_BYTES, WorkerAgent

STATIC_ENVIRONMENT = {
    "OGWP_WORKER_COORDINATOR_URL": "https://coordinator.invalid",
    "OGWP_WORKER_ID": "worker-a",
    "OGWP_WORKER_AUTH": "static",
    "OGWP_WORKER_TOKEN": "token-a",
}


def test_a_static_configuration_builds_a_bearer_credentialled_agent() -> None:
    agent = agent_main.build_agent_from_env(
        STATIC_ENVIRONMENT
        | {"OGWP_WORKER_GPU_MODEL": "rtx-4090", "OGWP_WORKER_VRAM_MB": "24576"}
    )

    assert isinstance(agent, WorkerAgent)
    assert agent.worker_id == "worker-a"
    assert agent.credential() == "token-a"
    assert agent.capability_ids == ("test.deterministic.echo.v1",)
    assert agent.max_input_bytes == DEFAULT_MAX_INPUT_BYTES


def test_a_the_scratch_bound_is_configurable() -> None:
    agent = agent_main.build_agent_from_env(
        STATIC_ENVIRONMENT | {"OGWP_WORKER_MAX_INPUT_BYTES": "1048576"}
    )

    assert agent.max_input_bytes == 1024 * 1024


def test_a_the_comfy_plugin_is_built_from_the_packaged_templates() -> None:
    agent = agent_main.build_agent_from_env(
        STATIC_ENVIRONMENT | {"OGWP_WORKER_PLUGINS": "comfy-workflow"}
    )

    assert agent.capability_ids == ("video.minimax_h3.text_to_video.v1",)


def test_the_comfy_start_command_is_split_from_the_environment() -> None:
    plugin = agent_main._comfy_workflow_plugin(
        {"OGWP_COMFY_START_COMMAND": "systemctl --user start comfyui"}
    )

    assert plugin._start_command == ("systemctl", "--user", "start", "comfyui")


def test_the_comfy_start_command_is_none_when_unset_or_empty() -> None:
    unset = agent_main._comfy_workflow_plugin({})
    empty = agent_main._comfy_workflow_plugin({"OGWP_COMFY_START_COMMAND": ""})

    assert unset._start_command is None
    assert empty._start_command is None


def test_the_comfy_startup_timeout_is_configurable_from_the_environment() -> None:
    from outbound_gpu_worker_pool.comfy import DEFAULT_COMFY_STARTUP_TIMEOUT_SECONDS

    default = agent_main._comfy_workflow_plugin({})
    configured = agent_main._comfy_workflow_plugin(
        {"OGWP_COMFY_STARTUP_TIMEOUT_SECONDS": "30"}
    )

    assert default._startup_timeout_seconds == DEFAULT_COMFY_STARTUP_TIMEOUT_SECONDS
    assert configured._startup_timeout_seconds == 30.0


@pytest.mark.parametrize(
    "environment",
    [
        pytest.param(
            {
                key: value
                for key, value in STATIC_ENVIRONMENT.items()
                if key != "OGWP_WORKER_COORDINATOR_URL"
            },
            id="missing-coordinator-url",
        ),
        pytest.param(
            {
                key: value
                for key, value in STATIC_ENVIRONMENT.items()
                if key != "OGWP_WORKER_ID"
            },
            id="missing-worker-id",
        ),
        pytest.param(
            {
                key: value
                for key, value in STATIC_ENVIRONMENT.items()
                if key != "OGWP_WORKER_TOKEN"
            },
            id="static-without-token",
        ),
        pytest.param(
            STATIC_ENVIRONMENT | {"OGWP_WORKER_PLUGINS": "unknown-plugin"},
            id="unknown-plugin",
        ),
        pytest.param(
            STATIC_ENVIRONMENT | {"OGWP_WORKER_AUTH": "google_oidc"},
            id="oidc-without-audience",
        ),
        pytest.param(
            STATIC_ENVIRONMENT | {"OGWP_WORKER_AUTH": "mutual_tls"},
            id="unknown-auth-method",
        ),
    ],
)
def test_b_incomplete_configuration_is_rejected(environment: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        agent_main.build_agent_from_env(environment)


async def test_c_sigterm_drains_the_agent_and_exits_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StubAgent:
        def __init__(self) -> None:
            self.drained = False

        async def run_forever(self, stop: asyncio.Event) -> None:
            await stop.wait()
            self.drained = True

    agent = _StubAgent()
    monkeypatch.setattr(agent_main, "build_agent_from_env", lambda environment: agent)
    asyncio.get_running_loop().call_later(0.01, os.kill, os.getpid(), signal.SIGTERM)

    exit_code = await agent_main.main({})

    assert exit_code == 0
    assert agent.drained
