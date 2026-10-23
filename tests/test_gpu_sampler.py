"""GpuSampler tests: parsing, missing binary, timeout, and single-warning behavior.

Every scenario injects the `nvidia-smi` binary or its runner so nothing here
actually shells out to a real GPU driver.
"""

import logging
import subprocess

import pytest

from outbound_gpu_worker_pool.contracts import GpuTelemetry
from outbound_gpu_worker_pool.gpu_sampler import GpuSampler
from outbound_gpu_worker_pool.gpu_sampler import ComfyGpuSampler, _parse_comfy_stats


def test_comfy_telemetry_reports_memory_without_fabricating_utilization():
    result = _parse_comfy_stats({"devices": [{"type": "cuda", "index": 0,
        "name": "Test GPU", "vram_total": 8 * 2**30, "vram_free": 6 * 2**30}]})
    assert result == (GpuTelemetry(0, "Test GPU", None, 2048, 8192),)


@pytest.mark.parametrize("origin", ["http://169.254.169.254", "https://example.com",
                                    "http://localhost@evil.test", "http://localhost/path"])
def test_comfy_telemetry_rejects_nonloopback_origins(origin):
    with pytest.raises(ValueError):
        ComfyGpuSampler(origin)


def test_comfy_telemetry_rejects_malformed_memory():
    with pytest.raises(ValueError):
        _parse_comfy_stats({"devices": [{"type": "cuda", "index": 0, "name": "GPU",
                                        "vram_total": 1, "vram_free": 2}]})


def test_comfy_telemetry_failure_is_nonfatal(monkeypatch):
    sampler = ComfyGpuSampler("http://127.0.0.1:8188")
    async def fail():
        raise TimeoutError
    monkeypatch.setattr(sampler, "_sample", fail)
    assert sampler.sample() == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["oversize", "slow", "redirect"])
async def test_comfy_network_budget_and_redirect_policy(monkeypatch, mode):
    import asyncio
    import httpx
    original = httpx.AsyncClient
    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            if mode == "slow":
                await asyncio.sleep(1)
            yield b"x" * 65537
    def handler(request):
        if mode == "redirect":
            return httpx.Response(302, headers={"Location": "http://169.254.169.254"})
        return httpx.Response(200, stream=Stream())
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    sampler = ComfyGpuSampler("http://127.0.0.1:8188", timeout_seconds=0.05)
    with pytest.raises((ValueError, TimeoutError, httpx.HTTPStatusError)):
        await sampler._sample()

SAMPLE_CSV = (
    "0, NVIDIA GeForce RTX 4090, 45, 2048, 24576\n"
    "1, NVIDIA GeForce RTX 4090, 0, 512, 24576\n"
)
SAMPLE_CSV_WITH_NA = "0, NVIDIA GeForce RTX 4090, [N/A], [N/A], 24576\n"


def _completed(stdout: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["nvidia-smi"], returncode=0, stdout=stdout)


def test_sample_parses_the_csv_fixture_into_telemetry() -> None:
    sampler = GpuSampler(
        binary="nvidia-smi", runner=lambda *a, **k: _completed(SAMPLE_CSV)
    )

    telemetry = sampler.sample()

    assert telemetry == (
        GpuTelemetry(
            index=0,
            name="NVIDIA GeForce RTX 4090",
            utilization_pct=45,
            memory_used_mb=2048,
            memory_total_mb=24576,
        ),
        GpuTelemetry(
            index=1,
            name="NVIDIA GeForce RTX 4090",
            utilization_pct=0,
            memory_used_mb=512,
            memory_total_mb=24576,
        ),
    )


def test_sample_treats_n_a_fields_as_none() -> None:
    sampler = GpuSampler(
        binary="nvidia-smi", runner=lambda *a, **k: _completed(SAMPLE_CSV_WITH_NA)
    )

    telemetry = sampler.sample()

    assert telemetry == (
        GpuTelemetry(
            index=0,
            name="NVIDIA GeForce RTX 4090",
            utilization_pct=None,
            memory_used_mb=None,
            memory_total_mb=24576,
        ),
    )


def test_a_missing_binary_yields_empty_and_warns_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sampler = GpuSampler(binary=None, env={})

    with caplog.at_level(logging.WARNING):
        first = sampler.sample()
        second = sampler.sample()
        third = sampler.sample()

    assert first == ()
    assert second == ()
    assert third == ()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1


def test_a_timeout_yields_empty(caplog: pytest.LogCaptureFixture) -> None:
    def _raise_timeout(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=3.0)

    sampler = GpuSampler(binary="nvidia-smi", runner=_raise_timeout)

    with caplog.at_level(logging.WARNING):
        telemetry = sampler.sample()

    assert telemetry == ()
    assert any("timed out" in record.message for record in caplog.records)


def test_a_nonzero_exit_yields_empty() -> None:
    def _raise_called_process_error(*args: object, **kwargs: object) -> None:
        raise subprocess.CalledProcessError(returncode=1, cmd="nvidia-smi")

    sampler = GpuSampler(binary="nvidia-smi", runner=_raise_called_process_error)

    assert sampler.sample() == ()


def test_malformed_output_yields_empty() -> None:
    sampler = GpuSampler(
        binary="nvidia-smi", runner=lambda *a, **k: _completed("not,enough,fields\n")
    )

    assert sampler.sample() == ()


def test_env_var_is_preferred_over_path_and_wsl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampler = GpuSampler(env={"OGWP_NVIDIA_SMI": "/custom/nvidia-smi"})

    assert sampler._resolve_binary() == "/custom/nvidia-smi"


@pytest.mark.parametrize("length,accepted", [(128, True), (129, False)])
def test_comfy_gpu_names_respect_the_heartbeat_limit(length, accepted):
    from outbound_gpu_worker_pool.contracts import MAX_GPU_NAME_LENGTH
    assert MAX_GPU_NAME_LENGTH == 128
    document = {"devices": [{"type": "cuda", "index": 0, "name": "G" * length,
                             "vram_total": 8 * 2**30, "vram_free": 6 * 2**30}]}
    if accepted:
        assert _parse_comfy_stats(document)[0].name == "G" * length
    else:
        with pytest.raises(ValueError, match="GPU name"):
            _parse_comfy_stats(document)
