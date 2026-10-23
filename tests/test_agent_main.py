"""Agent entrypoint tests: environment parsing and signal-driven draining."""

import asyncio
import json
import os
import signal

import pytest

from outbound_gpu_worker_pool import agent_main
from outbound_gpu_worker_pool.agent import DEFAULT_MAX_INPUT_BYTES, WorkerAgent
from outbound_gpu_worker_pool.comfy import ComfyWorkflowPlugin

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

    assert agent.capability_ids == (
        "image.flux2_klein.subject.v1",
        "image.flux2_klein.subject.v2",
        "video.minimax_h3.dialogue.v1",
        "video.minimax_h3.text_to_video.v1",
    )


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


GENERIC_CAPABILITIES = (
    "image.flux2_klein.subject.v1",
    "image.flux2_klein.subject.v2",
    "video.minimax_h3.dialogue.v1",
)


def _standard_settings() -> dict[str, str]:
    from outbound_gpu_worker_pool.machine_profiles import (
        CONFIG_KEY, LAST_FRAME_FEATURE_KEY, RECIPE_KEY, STANDARD, describe_profile,
    )

    versions = {
        RECIPE_KEY: describe_profile(STANDARD)["recipe_version"],
        CONFIG_KEY: "c" * 64,
        LAST_FRAME_FEATURE_KEY: "1",
    }
    return {
        "MVS_WORKER_PROFILE": STANDARD,
        "OGWP_COMFY_URL": "http://127.0.0.1:8188",
        "OGWP_WORKER_VRAM_MB": "24564",
        "OGWP_WORKER_GPU_MODEL": "NVIDIA GeForce RTX 4090",
        "OGWP_WORKER_RUNTIME_VERSIONS": json.dumps(versions),
        "OGWP_WORKER_CONCURRENCY": "1",
    }


@pytest.fixture
def verified_standard(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, str]]:
    """The profile check, as if the running ComfyUI matched the standard recipe."""
    from outbound_gpu_worker_pool import machine_profiles

    calls: list[dict[str, str]] = []

    async def prepare(environment: dict[str, str]) -> dict[str, str]:
        calls.append(environment)
        return _standard_settings()

    monkeypatch.setattr(machine_profiles, "prepare_environment", prepare)
    return calls


async def test_without_a_profile_the_entrypoint_changes_nothing(verified_standard) -> None:
    environment = STATIC_ENVIRONMENT | {"OGWP_WORKER_PLUGINS": "comfy-workflow"}
    prepared, plugin = await agent_main.prepare_mixed_profile(environment)

    assert (prepared, plugin) == (environment, None)
    assert verified_standard == []
    agent = agent_main.build_agent_from_env(prepared)
    assert agent.capability_ids == (*GENERIC_CAPABILITIES, "video.minimax_h3.text_to_video.v1")
    assert all(type(p) is ComfyWorkflowPlugin for p in agent._plugins.values())
    assert agent._runtime_versions == {}


async def test_mixed_mode_keeps_flux_and_dialogue_and_binds_h3_to_the_profile(
    verified_standard,
) -> None:
    from outbound_gpu_worker_pool.machine_profiles import STANDARD_CAPABILITY, ProfileComfyPlugin

    environment = STATIC_ENVIRONMENT | {
        "OGWP_WORKER_PLUGINS": "comfy-workflow",
        "MVS_WORKER_PROFILE": "h3-24gb-standard",
    }
    prepared, plugin = await agent_main.prepare_mixed_profile(environment)
    agent = agent_main.build_agent_from_env(prepared, profile_plugin=plugin)

    assert len(verified_standard) == 1
    assert set(agent.capability_ids) == {*GENERIC_CAPABILITIES, STANDARD_CAPABILITY}
    assert agent._plugins[STANDARD_CAPABILITY] is plugin
    assert isinstance(plugin, ProfileComfyPlugin)
    for capability in GENERIC_CAPABILITIES:
        assert type(agent._plugins[capability]) is ComfyWorkflowPlugin
    # The generic plugin no longer serves the recipe-bound capability at all.
    generic = agent._plugins[GENERIC_CAPABILITIES[0]]
    assert STANDARD_CAPABILITY not in {
        c.capability_id for c in generic.capabilities().capabilities
    }
    # It registers what the profile check found and the plugin re-verifies per job.
    assert agent._runtime_versions == json.loads(_standard_settings()["OGWP_WORKER_RUNTIME_VERSIONS"])
    assert plugin.versions == agent._runtime_versions
    assert agent._vram_mb == 24564


async def test_mixed_mode_refuses_the_bounded_profile(verified_standard) -> None:
    with pytest.raises(ValueError, match="only h3-24gb-standard"):
        await agent_main.prepare_mixed_profile(
            STATIC_ENVIRONMENT | {"MVS_WORKER_PROFILE": "h3-16gb-lightx8"}
        )
    assert verified_standard == []


async def test_a_failed_profile_check_starts_no_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    from outbound_gpu_worker_pool import machine_profiles

    async def refuse(environment: dict[str, str]) -> dict[str, str]:
        raise ValueError("Required model missing: h3.safetensors")

    monkeypatch.setattr(machine_profiles, "prepare_environment", refuse)
    monkeypatch.setattr(agent_main, "build_agent_from_env", lambda *a, **k: pytest.fail("built"))
    with pytest.raises(ValueError, match="Required model missing"):
        await agent_main.main(STATIC_ENVIRONMENT | {"MVS_WORKER_PROFILE": "h3-24gb-standard"})


async def test_mixed_mode_refuses_another_plugin_claiming_the_profile_capability(
    verified_standard, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from outbound_gpu_worker_pool.machine_profiles import STANDARD, ProfileComfyPlugin

    prepared, plugin = await agent_main.prepare_mixed_profile(
        STATIC_ENVIRONMENT | {"MVS_WORKER_PROFILE": STANDARD, "OGWP_WORKER_PLUGINS": "shadow"}
    )
    monkeypatch.setitem(
        agent_main.KNOWN_PLUGINS, "shadow",
        lambda environment: ProfileComfyPlugin(STANDARD, {}, client=None, base_url="http://127.0.0.1:8188"),
    )
    with pytest.raises(ValueError, match="also claimed elsewhere"):
        agent_main.build_agent_from_env(prepared, profile_plugin=plugin)


async def test_main_runs_the_mixed_agent(verified_standard, monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[object] = []

    class _Agent:
        async def run_forever(self, stop: asyncio.Event) -> None:
            return None

    def build(environment, profile_plugin=None):
        built.append((environment["OGWP_WORKER_RUNTIME_VERSIONS"], profile_plugin))
        return _Agent()

    monkeypatch.setattr(agent_main, "build_agent_from_env", build)
    assert await agent_main.main(STATIC_ENVIRONMENT | {"MVS_WORKER_PROFILE": "h3-24gb-standard"}) == 0
    assert built[0][0] == _standard_settings()["OGWP_WORKER_RUNTIME_VERSIONS"]
    assert built[0][1] is not None
