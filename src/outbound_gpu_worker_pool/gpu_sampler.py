"""Local GPU telemetry sampling for the worker agent (`nvidia-smi`, no driver bindings).

`GpuSampler.sample()` is synchronous and meant to be run off the event loop (the
agent calls it with `asyncio.to_thread`). It never raises: any failure to find or
run `nvidia-smi`, or to parse its output, yields an empty tuple, and is logged at
warning level exactly once so a GPU-less or misconfigured machine does not spam
logs on every heartbeat.
"""

import logging
import os
import shutil
import subprocess
from collections.abc import Callable, Sequence

from outbound_gpu_worker_pool.contracts import GpuTelemetry

logger = logging.getLogger(__name__)

NVIDIA_SMI_ENV_VAR = "OGWP_NVIDIA_SMI"
DEFAULT_NVIDIA_SMI_BINARY = "nvidia-smi"
WSL_NVIDIA_SMI_PATH = "/usr/lib/wsl/lib/nvidia-smi"
DEFAULT_SAMPLE_TIMEOUT_SECONDS = 3.0
QUERY_FIELDS = "index,name,utilization.gpu,memory.used,memory.total"
NOT_AVAILABLE = "[N/A]"

SubprocessRunner = Callable[..., subprocess.CompletedProcess]


class GpuSampler:
    """Samples every GPU on the local machine via `nvidia-smi`."""

    def __init__(
        self,
        *,
        binary: str | None = None,
        timeout_seconds: float = DEFAULT_SAMPLE_TIMEOUT_SECONDS,
        env: dict[str, str] | None = None,
        runner: SubprocessRunner | None = None,
    ) -> None:
        self._binary_override = binary
        self._timeout_seconds = timeout_seconds
        self._env = os.environ if env is None else env
        self._run: SubprocessRunner = runner if runner is not None else subprocess.run
        self._warned = False

    def sample(self) -> tuple[GpuTelemetry, ...]:
        binary = self._resolve_binary()
        if binary is None:
            self._warn_once("nvidia-smi was not found on this machine")
            return ()
        try:
            result = self._run(
                [
                    binary,
                    f"--query-gpu={QUERY_FIELDS}",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
                check=True,
            )
        except subprocess.TimeoutExpired:
            self._warn_once(f"nvidia-smi timed out after {self._timeout_seconds}s")
            return ()
        except (OSError, subprocess.CalledProcessError) as exc:
            self._warn_once(f"nvidia-smi sampling failed: {exc}")
            return ()
        try:
            return _parse_csv(result.stdout)
        except (ValueError, IndexError) as exc:
            self._warn_once(f"nvidia-smi output could not be parsed: {exc}")
            return ()

    def _resolve_binary(self) -> str | None:
        if self._binary_override is not None:
            return self._binary_override
        env_value = self._env.get(NVIDIA_SMI_ENV_VAR)
        if env_value:
            return env_value
        on_path = shutil.which(DEFAULT_NVIDIA_SMI_BINARY)
        if on_path is not None:
            return on_path
        if os.path.exists(WSL_NVIDIA_SMI_PATH):
            return WSL_NVIDIA_SMI_PATH
        return None

    def _warn_once(self, message: str) -> None:
        if self._warned:
            return
        self._warned = True
        logger.warning("gpu telemetry sampling disabled: %s", message)


def _parse_csv(stdout: str) -> tuple[GpuTelemetry, ...]:
    telemetry: list[GpuTelemetry] = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        fields: Sequence[str] = [field.strip() for field in stripped.split(",")]
        if len(fields) != 5:
            raise ValueError(f"expected 5 fields, got {len(fields)}")
        index, name, utilization_pct, memory_used_mb, memory_total_mb = fields
        telemetry.append(
            GpuTelemetry(
                index=int(index),
                name=name,
                utilization_pct=_optional_int(utilization_pct),
                memory_used_mb=_optional_int(memory_used_mb),
                memory_total_mb=_optional_int(memory_total_mb),
            )
        )
    return tuple(telemetry)


def _optional_int(value: str) -> int | None:
    if value == NOT_AVAILABLE or not value:
        return None
    return int(value)
