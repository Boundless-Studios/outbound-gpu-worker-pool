"""Run one outbound worker agent from the environment (`OGWP_WORKER_*`).

`python -m outbound_gpu_worker_pool.agent_main` is the standalone entrypoint. The
process needs a coordinator URL, a worker id, one credential source, and the set
of plugins the machine is approved to run; it never needs bucket or database
access. SIGTERM and SIGINT drain: the in-flight job finishes, a final draining
heartbeat is sent, and the process exits 0.
"""

import asyncio
import os
import signal
import tempfile
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path

import httpx

from outbound_gpu_worker_pool.agent import (
    DEFAULT_MAX_INPUT_BYTES,
    HttpAssetTransfer,
    WorkerAgent,
)
from outbound_gpu_worker_pool.plugins import (
    DETERMINISTIC_ECHO_PLUGIN_ID,
    DeterministicEchoPlugin,
    GpuExecutorPlugin,
)

STATIC_AUTH = "static"
GOOGLE_OIDC_AUTH = "google_oidc"
DEFAULT_WORKSPACE_NAME = "outbound-gpu-worker"
COORDINATOR_TIMEOUT_SECONDS = 60.0
TRANSFER_TIMEOUT_SECONDS = 900.0
COMFY_TIMEOUT_SECONDS = 900.0
COMFY_WORKFLOW_PLUGIN_ID = "comfy-workflow"


def _comfy_workflow_plugin(environment: Mapping[str, str]) -> GpuExecutorPlugin:
    """The approved-workflow plugin over the machine's own ComfyUI.

    Imported here rather than at module scope so a machine that runs only the
    reference plugin never needs the `comfy` extra installed.
    """
    from outbound_gpu_worker_pool.comfy import (
        DEFAULT_COMFY_BASE_URL,
        PACKAGED_TEMPLATES_DIRECTORY,
        ComfyWorkflowPlugin,
        TemplateRegistry,
    )

    directory = environment.get("OGWP_COMFY_TEMPLATES_DIR")
    return ComfyWorkflowPlugin(
        TemplateRegistry.from_directory(
            Path(directory) if directory else PACKAGED_TEMPLATES_DIRECTORY
        ),
        client=httpx.AsyncClient(timeout=COMFY_TIMEOUT_SECONDS),
        base_url=environment.get("OGWP_COMFY_URL", DEFAULT_COMFY_BASE_URL),
    )


KNOWN_PLUGINS: dict[str, Callable[[Mapping[str, str]], GpuExecutorPlugin]] = {
    DETERMINISTIC_ECHO_PLUGIN_ID: lambda environment: DeterministicEchoPlugin(),
    COMFY_WORKFLOW_PLUGIN_ID: _comfy_workflow_plugin,
}


def build_agent_from_env(environment: Mapping[str, str]) -> WorkerAgent:
    coordinator_url = _required(environment, "OGWP_WORKER_COORDINATOR_URL")
    worker_id = _required(environment, "OGWP_WORKER_ID")
    auth = environment.get("OGWP_WORKER_AUTH", STATIC_AUTH)
    if auth == STATIC_AUTH:
        token = _required(environment, "OGWP_WORKER_TOKEN")
        credential: Callable[[], str] = lambda: token
    elif auth == GOOGLE_OIDC_AUTH:
        credential = partial(
            _google_id_token, _required(environment, "OGWP_WORKER_AUDIENCE")
        )
    else:
        raise ValueError(f"unsupported OGWP_WORKER_AUTH: {auth}")
    plugins = tuple(
        _plugin(name, environment)
        for name in environment.get(
            "OGWP_WORKER_PLUGINS", DETERMINISTIC_ECHO_PLUGIN_ID
        ).split(",")
    )
    vram_mb = environment.get("OGWP_WORKER_VRAM_MB")
    workspace_root = Path(
        environment.get(
            "OGWP_WORKER_WORKSPACE",
            str(Path(tempfile.gettempdir()) / DEFAULT_WORKSPACE_NAME),
        )
    )
    return WorkerAgent(
        coordinator_url=coordinator_url,
        worker_id=worker_id,
        credential=credential,
        plugins=plugins,
        # The transfer client is separate so the worker credential is never
        # attached to a signed asset URL.
        transfer=HttpAssetTransfer(httpx.AsyncClient(timeout=TRANSFER_TIMEOUT_SECONDS)),
        http=httpx.AsyncClient(timeout=COORDINATOR_TIMEOUT_SECONDS),
        workspace_root=workspace_root,
        concurrency=int(environment.get("OGWP_WORKER_CONCURRENCY", "1")),
        max_input_bytes=int(
            environment.get("OGWP_WORKER_MAX_INPUT_BYTES", DEFAULT_MAX_INPUT_BYTES)
        ),
        gpu_model=environment.get("OGWP_WORKER_GPU_MODEL"),
        vram_mb=int(vram_mb) if vram_mb else None,
    )


async def main(environment: Mapping[str, str] = os.environ) -> int:
    agent = build_agent_from_env(environment)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for number in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(number, stop.set)
    try:
        await agent.run_forever(stop)
    finally:
        for number in (signal.SIGTERM, signal.SIGINT):
            loop.remove_signal_handler(number)
    return 0


def _required(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name)
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _plugin(name: str, environment: Mapping[str, str]) -> GpuExecutorPlugin:
    factory = KNOWN_PLUGINS.get(name.strip())
    if factory is None:
        raise ValueError(f"unknown worker plugin: {name}")
    return factory(environment)


def _google_id_token(audience: str) -> str:
    """Mint a fresh identity token for this machine (needs the `google-auth` extra)."""
    from google.auth.transport.requests import Request
    from google.oauth2 import id_token

    return id_token.fetch_id_token(Request(), audience)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
